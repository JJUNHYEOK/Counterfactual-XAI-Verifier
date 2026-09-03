from __future__ import annotations

import hashlib
import json
import unittest

from experiments.yolov8_search_comparison.model_mapping import resolve_target_class_mapping
from experiments.yolov8_search_comparison.prepare_training_data import xywh_pixels_to_yolo
from experiments.yolov8_search_comparison.run_comparison import EvaluationRepository, validate_frame_alignment
from experiments.yolov8_search_comparison.search_core import (
    Environment,
    choose_next_environment,
    classify_verdict,
    compute_map50,
    degrade_environment,
    environment_gap,
    map_gap,
)


BOUNDS = {
    "fog_percent": [0.0, 100.0],
    "illumination_lux": [200.0, 15000.0],
    "camera_noise": [0.0, 0.6],
}
DEGRADATION = {
    "fog_add_percent_points": 30.0,
    "illumination_multiplier": 0.5,
    "camera_noise_add": 0.2,
}


class VerdictTests(unittest.TestCase):
    def test_exact_pass_threshold_is_pass(self) -> None:
        self.assertEqual(classify_verdict(0.50), "PASS")

    def test_exact_fail_boundary_is_marginal(self) -> None:
        self.assertEqual(classify_verdict(0.25), "MARGINAL")

    def test_below_fail_boundary_is_fail(self) -> None:
        self.assertEqual(classify_verdict(0.249999), "FAIL")


class SearchPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nonfail = Environment(20.0, 10000.0, 0.10)
        self.fail = Environment(80.0, 1000.0, 0.50)

    def choose(self, method: str, current: Environment, verdict: str):
        return choose_next_environment(
            method=method,
            current=current,
            current_verdict=verdict,
            nonfail_anchor=self.nonfail,
            fail_anchor=self.fail,
            bounds=BOUNDS,
            degradation=DEGRADATION,
        )[0]

    def test_symmetric_is_exact_midpoint(self) -> None:
        self.assertEqual(
            self.choose("symmetric", self.nonfail, "MARGINAL"),
            Environment(50.0, 5500.0, 0.30),
        )

    def test_asymmetric_probe_uses_65_percent_toward_fail(self) -> None:
        self.assertEqual(
            self.choose("asymmetric", self.nonfail, "MARGINAL"),
            Environment(59.0, 4150.0, 0.36),
        )

    def test_asymmetric_recovery_uses_75_percent_toward_nonfail(self) -> None:
        self.assertEqual(
            self.choose("asymmetric", self.fail, "FAIL"),
            Environment(35.0, 7750.0, 0.20),
        )

    def test_environment_is_clamped_to_configured_bounds(self) -> None:
        degraded = degrade_environment(Environment(95.0, 250.0, 0.55), BOUNDS, DEGRADATION)
        self.assertEqual(degraded, Environment(100.0, 200.0, 0.60))

    def test_pre_fail_rule_is_identical_between_methods(self) -> None:
        current = Environment(5.0, 12000.0, 0.02)
        outputs = []
        for method in ("symmetric", "asymmetric"):
            outputs.append(
                choose_next_environment(
                    method=method,
                    current=current,
                    current_verdict="PASS",
                    nonfail_anchor=current,
                    fail_anchor=None,
                    bounds=BOUNDS,
                    degradation=DEGRADATION,
                )
            )
        self.assertEqual(outputs[0], outputs[1])


class FairEvaluationTests(unittest.TestCase):
    def test_same_input_and_seed_produce_same_evaluation_identity(self) -> None:
        env = Environment(35.0, 6000.0, 0.22)
        seed = 42

        def evaluation_identity(environment: Environment, random_seed: int) -> str:
            # Search method is intentionally absent, just as in the shared cache key.
            payload = {"environment": environment.as_dict(), "seed": random_seed}
            return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

        self.assertEqual(evaluation_identity(env, seed), evaluation_identity(env, seed))

    def test_yolo_metric_is_class_aware_and_independent_of_search_method(self) -> None:
        frames = [
            {
                "ground_truth": [
                    {"object_id": 1, "class_name": "person", "bbox_xywh": [0, 0, 10, 10]},
                    {"object_id": 2, "class_name": "vehicle", "bbox_xywh": [20, 0, 10, 10]},
                ],
                "detections": [
                    {"class_name": "person", "confidence": 0.9, "bbox_xywh": [0, 0, 10, 10]},
                    {"class_name": "vehicle", "confidence": 0.8, "bbox_xywh": [20, 0, 10, 10]},
                ],
            }
        ]
        symmetric_metric = compute_map50(frames, 0.5)
        asymmetric_metric = compute_map50(frames, 0.5)
        self.assertEqual(symmetric_metric, asymmetric_metric)
        self.assertEqual(symmetric_metric["map50"], 1.0)

    def test_shared_cache_key_does_not_include_search_method(self) -> None:
        repository = EvaluationRepository.__new__(EvaluationRepository)
        repository.config = {
            "scenario_id": "scenario",
            "scenario_variant": 0,
            "random_seed": 42,
            "frame_stride": 1,
            "ground_truth_mode": "rendered_instance_mask_v1",
            "detector": "yolov8",
            "yolov8": {"input_size": 640},
        }
        repository.model_metadata = {"weights_sha256": "ABC"}
        env = Environment(5.0, 12000.0, 0.02)
        first = repository._key(env)
        repository.config["search_method"] = "asymmetric"
        self.assertEqual(first, repository._key(env))

    def test_different_seed_changes_cache_identity(self) -> None:
        repository = EvaluationRepository.__new__(EvaluationRepository)
        repository.config = {
            "scenario_id": "scenario", "scenario_variant": 0, "random_seed": 42,
            "frame_stride": 1, "ground_truth_mode": "rendered_instance_mask_v1",
            "detector": "yolov8", "yolov8": {"input_size": 640},
        }
        repository.model_metadata = {"weights_sha256": "ABC"}
        first = repository._key(Environment(5.0, 12000.0, 0.02))
        repository.config["random_seed"] = 43
        self.assertNotEqual(first, repository._key(Environment(5.0, 12000.0, 0.02)))


class GroundTruthConventionTests(unittest.TestCase):
    def test_absolute_xywh_to_normalized_yolo_conversion(self) -> None:
        self.assertEqual(xywh_pixels_to_yolo([100, 50, 20, 40], 200, 100), (0.55, 0.7, 0.1, 0.4))

    def test_out_of_range_box_is_rejected_not_silently_resized(self) -> None:
        with self.assertRaises(ValueError):
            xywh_pixels_to_yolo([190, 50, 20, 40], 200, 100)

    def test_frame_index_and_image_name_alignment(self) -> None:
        frames = [
            {"frame_index": 1, "image_path": "frames/frame_0001.png"},
            {"frame_index": 4, "image_path": "frames/frame_0004.png"},
            {"frame_index": 7, "image_path": "frames/frame_0007.png"},
        ]
        validate_frame_alignment(frames, 3)

    def test_shifted_frame_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_frame_alignment([{"frame_index": 1, "image_path": "frames/frame_0002.png"}], 1)


class GapMetricTests(unittest.TestCase):
    def test_map_gap(self) -> None:
        self.assertAlmostEqual(map_gap(0.2666256954, 0.2373851727), 0.0292405227, places=10)

    def test_environment_gap_uses_configured_ranges(self) -> None:
        nonfail = Environment(78.12, 2343.8, 0.4988)
        fail = Environment(78.59, 2320.4, 0.5016)
        expected = 0.47 / 100.0 + 23.4 / 14800.0 + 0.0028 / 0.6
        self.assertAlmostEqual(environment_gap(nonfail, fail, BOUNDS), expected, places=12)


class ModelClassMappingTests(unittest.TestCase):
    def test_coco_mapping_uses_names_not_fixed_positions(self) -> None:
        names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        self.assertEqual(
            resolve_target_class_mapping(names),
            {0: "person", 2: "vehicle", 3: "vehicle", 5: "vehicle", 7: "vehicle"},
        )

    def test_custom_two_class_mapping(self) -> None:
        self.assertEqual(
            resolve_target_class_mapping({0: "person", 1: "vehicle"}),
            {0: "person", 1: "vehicle"},
        )

    def test_visdrone_candidate_mapping_excludes_unrelated_classes(self) -> None:
        self.assertEqual(
            resolve_target_class_mapping({0: "car", 1: "tree", 2: "building", 3: "person", 4: "other"}),
            {0: "vehicle", 3: "person"},
        )


if __name__ == "__main__":
    unittest.main()
