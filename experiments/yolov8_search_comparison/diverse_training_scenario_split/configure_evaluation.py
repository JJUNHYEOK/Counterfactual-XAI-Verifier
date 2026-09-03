"""Freeze a new-weight S0--S4 evaluation config after validation-only model selection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .plan import CONFIG_ROOT, FIXED_PLAN, FIXED_SCENARIOS, HERE, PLAN_ID, file_sha256


EVALUATION_ROOT = HERE / "evaluation"


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", type=Path, required=True)
    args = parser.parse_args()
    training_summary_path = args.training_summary.resolve()
    training = json.loads(training_summary_path.read_text(encoding="utf-8"))
    if training.get("S0_S4_accessed_for_training_or_selection") is not False:
        raise RuntimeError("Training summary does not prove the S0-S4 firewall")
    weights = Path(training["best_weights"])
    if file_sha256(weights) != training["best_weights_sha256"]:
        raise RuntimeError("Selected best.pt hash mismatch")

    prior_audit = json.loads((CONFIG_ROOT / "existing_state_audit.json").read_text(encoding="utf-8"))
    fixed = prior_audit["fixed_S0_S4"]
    reuse_checks = []
    for item in fixed["scenario_inputs"]:
        config_path = Path(item["scenario_config_path"])
        manifest_path = Path(item["initial_manifest_path"])
        frame_dir = manifest_path.parent / "frames"
        reuse_checks.append(
            {
                "scenario_id": item["scenario_id"],
                "scenario_config_unchanged": file_sha256(config_path) == item["scenario_config_sha256"],
                "manifest_unchanged": file_sha256(manifest_path) == item["initial_manifest_sha256"],
                "rendered_frame_count": len(list(frame_dir.glob("frame_*.png"))),
                "expected_rendered_frame_count": item["rendered_image_count"],
            }
        )
    reuse_passed = all(
        row["scenario_config_unchanged"] and row["manifest_unchanged"] and row["rendered_frame_count"] == row["expected_rendered_frame_count"] == 181
        for row in reuse_checks
    )
    if not reuse_passed:
        raise RuntimeError("Fixed S0-S4 initial render/GT hashes do not match the pre-work snapshot")

    config = {
        "experiment_id": "kci_diverse_training_s0_s4_v1",
        "training_plan_id": PLAN_ID,
        "training_run_id": training["run_id"],
        "detector": "yolov8",
        "weights_path": str(weights),
        "expected_weights_sha256": training["best_weights_sha256"],
        "expected_model_class_names": {"0": "person", "1": "vehicle"},
        "search_method": "symmetric",
        "global_initial_pass_gate": True,
        "initial_environment": {"fog_percent": 5.0, "illumination_lux": 12000.0, "camera_noise": 0.02},
        "environment_bounds": {"fog_percent": [0.0, 100.0], "illumination_lux": [200.0, 15000.0], "camera_noise": [0.0, 0.6]},
        "degradation": {"fog_add_percent_points": 30.0, "illumination_multiplier": 0.5, "camera_noise_add": 0.2},
        "thresholds": {"pass_map50": 0.5, "fail_map50": 0.25},
        "search_boundary_definition": "PASS and MARGINAL are non-failure; only FAIL is the failure-side anchor.",
        "max_evaluations": 10,
        "frame_stride": 1,
        "expected_frame_count": 181,
        "ground_truth_mode": "rendered_instance_mask_v1",
        "gt_version": "rendered_instance_mask_v1",
        "retain_rendered_frames": True,
        "yolov8": {"input_size": 640, "confidence_threshold": 0.25, "nms_iou_threshold": 0.7, "ap_iou_threshold": 0.5, "device": "auto", "batch_size": 16},
        "fixed_evaluation_statement": "S0-S4 are previously pre-registered fixed evaluation scenarios, not completely unseen external test data.",
        "cache_policy": "A new evaluation root and weight-hash-keyed repository are required; no prior detector outputs or AP/mAP caches are reused.",
    }
    config_path = EVALUATION_ROOT / "config" / f"evaluation_config__{training['run_id']}.json"
    write_json_new(config_path, config)
    preflight = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_summary_path": str(training_summary_path),
        "training_summary_sha256": file_sha256(training_summary_path),
        "weights_path": str(weights),
        "weights_sha256": file_sha256(weights),
        "fixed_plan_path": str(FIXED_PLAN.resolve()),
        "fixed_plan_sha256": file_sha256(FIXED_PLAN),
        "fixed_scenario_root": str(FIXED_SCENARIOS.parent.resolve()),
        "render_and_gt_reuse_allowed": reuse_passed,
        "reuse_checks": reuse_checks,
        "old_detection_or_metric_cache_reuse_allowed": False,
        "new_runs_root": str((EVALUATION_ROOT / "runs").resolve()),
        "new_aggregated_root": str((EVALUATION_ROOT / "aggregated").resolve()),
    }
    preflight_path = EVALUATION_ROOT / "config" / f"evaluation_preflight__{training['run_id']}.json"
    write_json_new(preflight_path, preflight)
    print(
        json.dumps(
            {
                "config": str(config_path.resolve()),
                "config_sha256": file_sha256(config_path),
                "preflight": str(preflight_path.resolve()),
                "fixed_plan": str(FIXED_PLAN.resolve()),
                "scenario_root": str(FIXED_SCENARIOS.parent.resolve()),
                "runs_root": str((EVALUATION_ROOT / "runs").resolve()),
                "aggregated_root": str((EVALUATION_ROOT / "aggregated").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

