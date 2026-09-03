from __future__ import annotations

import unittest

from experiments.yolov8_search_comparison.diverse_training_scenario_split.plan import (
    SELECTED_FRAMES,
    definitions,
    layout_hash,
    trajectory_hash,
)
from experiments.yolov8_search_comparison.search_core import (
    Environment,
    choose_next_environment,
    classify_verdict,
    environment_gap,
    map_gap,
)


class DiversePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenarios = definitions()

    def test_split_counts(self) -> None:
        self.assertEqual(18, sum(x["split"] == "train" for x in self.scenarios))
        self.assertEqual(5, sum(x["split"] == "val" for x in self.scenarios))

    def test_exact_image_targets(self) -> None:
        self.assertEqual(360, sum(x["split"] == "train" for x in self.scenarios) * len(SELECTED_FRAMES))
        self.assertEqual(100, sum(x["split"] == "val" for x in self.scenarios) * len(SELECTED_FRAMES))

    def test_frame_selection_spans_scenario(self) -> None:
        self.assertEqual(20, len(SELECTED_FRAMES))
        self.assertEqual((1, 181), (SELECTED_FRAMES[0], SELECTED_FRAMES[-1]))
        self.assertEqual({9, 10}, {b - a for a, b in zip(SELECTED_FRAMES, SELECTED_FRAMES[1:])})

    def test_identifiers_and_seeds_unique(self) -> None:
        for key in ("scenario_id", "seed"):
            values = [x[key] for x in self.scenarios]
            self.assertEqual(len(values), len(set(values)))

    def test_trajectory_and_layout_unique(self) -> None:
        for function in (trajectory_hash, layout_hash):
            values = [function(x) for x in self.scenarios]
            self.assertEqual(len(values), len(set(values)))

    def test_each_scenario_has_both_classes(self) -> None:
        for scenario in self.scenarios:
            classes = {item["class_id"] for item in scenario["objects"]}
            self.assertEqual({1, 2}, classes)

    def test_environment_is_benign_training_range(self) -> None:
        for scenario in self.scenarios:
            environment = scenario["environment"]
            self.assertGreaterEqual(environment["fog_percent"], 0)
            self.assertLessEqual(environment["fog_percent"], 20)
            self.assertGreaterEqual(environment["illumination_lux"], 8000)
            self.assertLessEqual(environment["illumination_lux"], 15000)
            self.assertGreaterEqual(environment["camera_noise"], 0)
            self.assertLessEqual(environment["camera_noise"], 0.05)

    def test_camera_geometry_fixed_and_explicit(self) -> None:
        for scenario in self.scenarios:
            self.assertEqual([600.0, 600.0, 320.0, 180.0, 60.0], scenario["camera"]["intrinsics"])
            self.assertEqual([640, 360], scenario["camera"]["image_size"])


class BoundaryPolicyTests(unittest.TestCase):
    def test_verdict_boundaries(self) -> None:
        self.assertEqual("PASS", classify_verdict(0.5))
        self.assertEqual("MARGINAL", classify_verdict(0.25))
        self.assertEqual("FAIL", classify_verdict(0.249999))

    def test_symmetric_midpoint(self) -> None:
        bounds = {"fog_percent": [0, 100], "illumination_lux": [200, 15000], "camera_noise": [0, 0.6]}
        nonfail = Environment(65, 3000, 0.42)
        fail = Environment(95, 1500, 0.60)
        nxt, description = choose_next_environment(
            method="symmetric",
            current=fail,
            current_verdict="FAIL",
            nonfail_anchor=nonfail,
            fail_anchor=fail,
            bounds=bounds,
            degradation={"fog_add_percent_points": 30, "illumination_multiplier": 0.5, "camera_noise_add": 0.2},
        )
        self.assertEqual(Environment(80, 2250, 0.51), nxt)
        self.assertIn("symmetric_midpoint", description)

    def test_gap_formulas(self) -> None:
        bounds = {"fog_percent": [0, 100], "illumination_lux": [200, 15000], "camera_noise": [0, 0.6]}
        left, right = Environment(70, 2500, 0.45), Environment(71, 2450, 0.456)
        expected = 1 / 100 + 50 / 14800 + 0.006 / 0.6
        self.assertAlmostEqual(expected, environment_gap(left, right, bounds))
        self.assertAlmostEqual(0.03, map_gap(0.27, 0.24))


if __name__ == "__main__":
    unittest.main()

