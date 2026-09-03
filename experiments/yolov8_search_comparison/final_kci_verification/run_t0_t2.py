"""Run the frozen T0--T2 v2 initial tests and conditional 10-step searches."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..multi_scenario_symmetric.run_experiment import (
    ScenarioEvaluationRepository,
    ScenarioMatlabExporter,
    run_scenario,
    stat_summary,
)
from ..run_comparison import YoloFrameDetector, file_sha256, seed_everything


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PLAN_ID = "new_fixed_tests_t0_t2_v2"
CONFIG_ROOT = HERE / "config" / PLAN_ID
SCENARIO_ROOT = HERE / "scenarios" / PLAN_ID
OUTPUT_ROOT = HERE / "outputs" / PLAN_ID
DEFAULT_PLAN = CONFIG_ROOT / "scenario_plan.json"
DEFAULT_CONFIG = CONFIG_ROOT / "evaluation_config.json"
DEFAULT_GATE = OUTPUT_ROOT / "preinference_gate.json"
RUNS_ROOT = OUTPUT_ROOT / "runs"
AGGREGATED_ROOT = OUTPUT_ROOT / "aggregated"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def flatten(value: Any) -> Any:
    if isinstance(value, (dict, list)) or value is None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: flatten(row.get(key)) for key in fields})


def validate_gate(plan_path: Path, config_path: Path, gate_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = read_json(plan_path)
    config = read_json(config_path)
    gate = read_json(gate_path)
    required = [
        gate["all_required_preinference_checks_passed"],
        not gate["yolo_inference_performed_before_gate"],
        gate["frozen_plan_sha_matches_preregistration"],
        gate["conditions"]["GT_all_valid"],
        gate["conditions"]["exact_duplicate_count"] == 0,
        gate["conditions"]["all_dhash_warnings_directly_reviewed"],
        gate["conditions"]["object_scene_material_duplicate_count"] == 0,
        gate["conditions"]["superseded_v1_preserved"],
        file_sha256(plan_path) == gate["plan_sha256"],
        file_sha256(config_path) == gate["evaluation_config_sha256"],
        file_sha256(OUTPUT_ROOT / "gt_validation.json") == gate["gt_validation_sha256"],
        file_sha256(OUTPUT_ROOT / "similarity_audit.json") == gate["similarity_audit_sha256"],
        file_sha256(OUTPUT_ROOT / "similarity_visual_review.json") == gate["visual_review_sha256"],
    ]
    if not all(required):
        raise RuntimeError(f"Frozen pre-inference gate no longer passes: {required}")
    if plan["plan_id"] != PLAN_ID or gate["plan_id"] != PLAN_ID:
        raise RuntimeError("Plan ID mismatch")
    for scenario in plan["scenarios"]:
        if file_sha256(Path(scenario["scenario_config_path"])) != scenario["scenario_config_sha256"]:
            raise RuntimeError(f"Scenario config changed after pre-registration: {scenario['scenario_id']}")
    return plan, config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--resume-session")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    config_path = args.config.resolve()
    gate_path = args.gate.resolve()
    plan, config = validate_gate(plan_path, config_path, gate_path)

    weights_path = Path(config["weights_path"]).resolve()
    actual_weights_hash = file_sha256(weights_path)
    if actual_weights_hash != config["expected_weights_sha256"]:
        raise RuntimeError(
            f"Weights hash mismatch: expected={config['expected_weights_sha256']} actual={actual_weights_hash}"
        )
    seed_everything(42)
    detector = YoloFrameDetector(weights_path, config["yolov8"])
    model_metadata = detector.metadata(weights_path, actual_weights_hash)
    actual_names = {str(key): value for key, value in model_metadata["model_class_names"].items()}
    if actual_names != config["expected_model_class_names"]:
        raise RuntimeError(f"Checkpoint class mapping mismatch: {actual_names}")
    checkpoint_model = str(model_metadata.get("checkpoint_model_definition") or "").lower()
    if "yolov8s" not in checkpoint_model:
        raise RuntimeError(f"Checkpoint is not YOLOv8s: {checkpoint_model}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    session_id = args.resume_session or f"kci_new_fixed_tests_t0_t2_v2__{timestamp}"
    session_root = RUNS_ROOT / session_id
    aggregate_root = AGGREGATED_ROOT / session_id
    session_root.mkdir(parents=True, exist_ok=bool(args.resume_session))
    aggregate_root.mkdir(parents=True, exist_ok=bool(args.resume_session))
    snapshot = {
        "session_id": session_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "plan_sha256": file_sha256(plan_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "gate_path": str(gate_path),
        "gate_sha256": file_sha256(gate_path),
        "weights_sha256": actual_weights_hash,
        "model": model_metadata,
        "host": {"platform": platform.platform(), "python": sys.version, "executable": sys.executable},
        "execution_rule": "Evaluate the initial normal condition. Run at most 10 symmetric evaluations only when that scenario's initial verdict is PASS; otherwise record the failure without scenario modification.",
        "frozen_config_label_note": "The v2 config retains experiment_id ending in v1 from the base copy; plan_id and session_id identify v2. This label does not alter settings or calculations.",
    }
    snapshot_path = aggregate_root / "config_snapshot.json"
    if snapshot_path.exists():
        previous = read_json(snapshot_path)
        stable = ("plan_sha256", "config_sha256", "gate_sha256", "weights_sha256")
        if not args.resume_session or any(previous[key] != snapshot[key] for key in stable):
            raise RuntimeError("Resume snapshot does not match frozen inputs")
    else:
        write_json_new(snapshot_path, snapshot)

    exporter = ScenarioMatlabExporter()
    results: dict[str, Any] = {}
    try:
        for scenario in plan["scenarios"]:
            scenario_id = scenario["scenario_id"]
            summary_path = session_root / scenario_id / "search" / "scenario_summary.json"
            if args.resume_session and summary_path.exists():
                results[scenario_id] = {
                    "initial": read_json(session_root / scenario_id / "search" / "initial_evaluation.json"),
                    "records": read_json(session_root / scenario_id / "search" / "iterations.json"),
                    "summary": read_json(summary_path),
                }
                continue
            repository = ScenarioEvaluationRepository(
                config=config,
                scenario=scenario,
                cache_root=session_root / scenario_id / "evaluations",
                exporter=exporter,
                detector=detector,
                model_metadata=model_metadata,
                weights_path=weights_path,
            )
            initial_manifest = (
                SCENARIO_ROOT / scenario_id / "initial_condition" / "frame_manifest.json"
            )
            results[scenario_id] = run_scenario(
                session_id=session_id,
                scenario=scenario,
                config=config,
                repository=repository,
                initial_manifest=initial_manifest,
                scenario_run_dir=session_root / scenario_id / "search",
                resume=bool(args.resume_session),
            )
    finally:
        exporter.close()

    ordered = [results[scenario["scenario_id"]] for scenario in plan["scenarios"]]
    summaries = [item["summary"] for item in ordered]
    initial_rows = [item["initial"] for item in ordered]
    records = [record for item in ordered for record in item["records"]]
    successes = [item for item in summaries if item["search_success"]]
    initial_pass_count = sum(item["initial_pass"] for item in summaries)
    overall = {
        "session_id": session_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": PLAN_ID,
        "scenario_count": len(summaries),
        "initial_pass_count": initial_pass_count,
        "initial_pass_rate": initial_pass_count / len(summaries),
        "search_attempt_count": sum(item["initial_pass"] for item in summaries),
        "search_success_count": len(successes),
        "search_success_rate_all_scenarios": len(successes) / len(summaries),
        "search_success_rate_attempted": len(successes) / max(1, initial_pass_count),
        "first_fail_iteration_statistics": stat_summary(
            [item["first_fail_iteration"] for item in successes]
        ),
        "gap_map50_statistics": stat_summary([item["gap_map50"] for item in successes]),
        "gap_environment_statistics": stat_summary(
            [item["gap_environment_normalized"] for item in successes]
        ),
        "search_evaluation_count": len(records),
        "initial_evaluation_count": len(initial_rows),
        "nonmonotonic_increase_count": sum(
            item["nonmonotonic_map_increase_count"] for item in summaries
        ),
        "scenario_with_nonmonotonic_increase_count": sum(
            item["nonmonotonic_map_increase_count"] > 0 for item in summaries
        ),
        "search_execution_seconds": sum(item["total_execution_seconds"] for item in summaries),
        "mean_search_execution_seconds_per_scenario": statistics.fmean(
            item["total_execution_seconds"] for item in summaries
        ),
        "environment_bounds": config["environment_bounds"],
        "scope_note": "Three deterministic scenarios additionally configured in the existing MATLAB simulation environment; no statistical significance or real-flight generalization claim is made.",
    }
    combined = {
        "overall": overall,
        "scenarios": summaries,
        "initial_evaluations": initial_rows,
    }
    write_json_new(aggregate_root / "t0_t2_results.json", combined)
    write_csv_new(aggregate_root / "initial_condition_results.csv", initial_rows)
    write_json_new(aggregate_root / "all_iteration_results.json", records)
    write_csv_new(aggregate_root / "all_iteration_results.csv", records)
    write_csv_new(aggregate_root / "scenario_summary.csv", summaries)
    print(json.dumps({"session_id": session_id, "aggregate_root": str(aggregate_root.resolve()), "overall": overall}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
