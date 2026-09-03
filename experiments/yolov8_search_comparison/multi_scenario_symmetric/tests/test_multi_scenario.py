from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ...run_comparison import file_sha256, validate_frame_alignment
from ...search_core import Environment, choose_next_environment, classify_verdict, environment_gap, map_gap
from ..prepare_scenarios import canonical_layout_hash, canonical_trajectory_hash, validate_manifest
from ..run_experiment import DEFAULT_CONFIG, DEFAULT_PLAN, ScenarioEvaluationRepository, read_json


HERE = Path(__file__).resolve().parents[1]
SESSION = "kci_multi_scenario_symmetric_v1__20260902_201715_587801"


class MultiScenarioUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = read_json(DEFAULT_CONFIG)
        cls.plan = read_json(DEFAULT_PLAN)

    def test_symmetric_midpoint_is_exactly_50_50(self) -> None:
        nonfail = Environment(65, 3000, 0.42)
        fail = Environment(95, 1500, 0.60)
        value, reason = choose_next_environment(
            method="symmetric", current=fail, current_verdict="FAIL", nonfail_anchor=nonfail,
            fail_anchor=fail, bounds=self.config["environment_bounds"], degradation=self.config["degradation"],
        )
        self.assertEqual(value, Environment(80, 2250, 0.51))
        self.assertIn("0.50", reason)

    def test_threshold_boundaries(self) -> None:
        self.assertEqual(classify_verdict(0.50), "PASS")
        self.assertEqual(classify_verdict(0.25), "MARGINAL")
        self.assertEqual(classify_verdict(0.249999), "FAIL")

    def test_environment_is_clamped(self) -> None:
        value, _ = choose_next_environment(
            method="symmetric", current=Environment(95, 200, 0.6), current_verdict="PASS",
            nonfail_anchor=Environment(95, 200, 0.6), fail_anchor=None,
            bounds=self.config["environment_bounds"], degradation=self.config["degradation"],
        )
        self.assertEqual(value, Environment(100, 200, 0.6))

    def _repository(self, scenario: dict, root: Path) -> ScenarioEvaluationRepository:
        return ScenarioEvaluationRepository(
            config=self.config, scenario=scenario, cache_root=root, exporter=None, detector=None,
            model_metadata={"weights_sha256": self.config["expected_weights_sha256"]},
            weights_path=Path(self.config["weights_path"]),
        )

    def test_cache_key_contains_scenario_identity(self) -> None:
        env = Environment.from_dict(self.config["initial_environment"])
        with tempfile.TemporaryDirectory() as temp:
            repo = self._repository(self.plan["scenarios"][0], Path(temp))
            identity = repo.identity(env)
            self.assertEqual(identity["scenario_id"], "S0")
            self.assertEqual(identity["trajectory_config_sha256"], self.plan["scenarios"][0]["scenario_config_sha256"])

    def test_scenarios_cannot_mix_evaluation_keys(self) -> None:
        env = Environment.from_dict(self.config["initial_environment"])
        with tempfile.TemporaryDirectory() as temp:
            keys = {self._repository(scenario, Path(temp) / scenario["scenario_id"]).key(env) for scenario in self.plan["scenarios"]}
        self.assertEqual(len(keys), 5)

    def test_fixed_weights_sha256(self) -> None:
        weights = HERE.parent / "training" / "yolov8s_sim_20260902" / "runs" / "full_yolov8s_seed42" / "weights" / "best.pt"
        self.assertEqual(file_sha256(weights), self.config["expected_weights_sha256"])

    def _manifest(self, root: Path) -> tuple[dict, dict]:
        frames = []
        for index in range(1, 182):
            image_path = root / f"frame_{index:04d}.png"
            image_path.write_bytes(b"test")
            frames.append(
                {
                    "frame_index": index,
                    "image_path": str(image_path),
                    "ground_truth": [
                        {"class_name": "person", "bbox_xywh": [1, 1, 10, 10]},
                        {"class_name": "vehicle", "bbox_xywh": [20, 20, 20, 20]},
                    ],
                }
            )
        scenario = self.plan["scenarios"][0]
        manifest = {"image_width": 640, "image_height": 360, "evaluated_frame_count": 181, "frames": frames}
        return manifest, scenario

    def test_gt_coordinates_and_frame_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, scenario = self._manifest(Path(temp))
            summary, _ = validate_manifest(manifest, scenario)
            self.assertTrue(summary["valid"])
            validate_frame_alignment(manifest["frames"], 1)

    def test_invalid_gt_coordinate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest, scenario = self._manifest(Path(temp))
            manifest["frames"][0]["ground_truth"][0]["bbox_xywh"] = [639, 1, 2, 2]
            summary, _ = validate_manifest(manifest, scenario)
            self.assertFalse(summary["valid"])
            self.assertEqual(summary["out_of_bounds_box_count"], 1)

    def test_training_overlap_audit_passed(self) -> None:
        audit = read_json(HERE / "scenarios" / self.plan["plan_id"] / "scenario_leakage_audit.json")
        self.assertTrue(audit["new_scenarios_passed"])
        for item in audit["results"][1:]:
            self.assertEqual(item["duplicate_image_hash_count"], 0)
            self.assertEqual(item["duplicate_frame_hash_count"], 0)
            self.assertEqual(item["duplicate_nonempty_label_hash_count"], 0)

    def test_new_scenario_seeds_layouts_and_trajectories_are_unique(self) -> None:
        new = self.plan["scenarios"][1:]
        self.assertEqual(len({x["seed"] for x in new}), 4)
        self.assertEqual(len({canonical_layout_hash(x["object_layout"]) for x in new}), 4)
        self.assertEqual(len({canonical_trajectory_hash(x) for x in new}), 4)

    def test_gap_formulas(self) -> None:
        nonfail, fail = Environment(78.12, 2343.8, 0.4988), Environment(78.59, 2320.4, 0.5016)
        self.assertAlmostEqual(map_gap(0.2666256954327856, 0.23738517272609652), 0.02924052270668906)
        expected = 0.47 / 100 + 23.4 / 14800 + 0.0028 / 0.6
        self.assertAlmostEqual(environment_gap(nonfail, fail, self.config["environment_bounds"]), expected)

    def test_search_success_rate(self) -> None:
        result = read_json(HERE / "aggregated" / SESSION / "multi_scenario_results.json")
        expected = sum(item["search_success"] for item in result["scenarios"]) / len(result["scenarios"])
        self.assertEqual(expected, result["overall"]["search_success_rate"])
        self.assertEqual(expected, 1.0)

    def test_revalidation_verdicts_match(self) -> None:
        result = read_json(HERE / "test_suites" / SESSION / "suite_summary.json")
        self.assertEqual(result["case_count"], 15)
        self.assertEqual(result["verdict_match_count"], 15)
        self.assertTrue(result["all_verdicts_match"])


if __name__ == "__main__":
    unittest.main()
