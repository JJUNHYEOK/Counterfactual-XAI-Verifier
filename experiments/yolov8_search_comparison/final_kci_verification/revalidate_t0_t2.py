"""Freshly re-render and re-infer T0--T2 anchors without the search cache."""

from __future__ import annotations

import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..multi_scenario_symmetric.run_experiment import (
    ScenarioMatlabExporter,
    matched_iou_summary,
    precision_recall,
)
from ..run_comparison import YoloFrameDetector, file_sha256, seed_everything, validate_frame_alignment
from ..search_core import Environment, classify_verdict, compute_map50
from .run_t0_t2 import CONFIG_ROOT, OUTPUT_ROOT, PLAN_ID, read_json, validate_gate


HERE = Path(__file__).resolve().parent
ORIGINAL_SESSION = "kci_new_fixed_tests_t0_t2_v2__20260903_160812_371388"
ORIGINAL_AGGREGATE = OUTPUT_ROOT / "aggregated" / ORIGINAL_SESSION / "t0_t2_results.json"
REVALIDATION_ROOT = OUTPUT_ROOT / "revalidation"


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def original_conditions(summary: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = [
        {
            "condition": "initial_normal",
            "environment": config["initial_environment"],
            "original_map50": summary["initial_map50"],
            "original_verdict": summary["initial_verdict"],
            "original_cache_key": None,
        }
    ]
    for label, key in (
        ("final_nonfailure", "final_nonfail_anchor"),
        ("final_FAIL", "final_fail_anchor"),
    ):
        anchor = summary[key]
        if anchor is None:
            continue
        conditions.append(
            {
                "condition": label,
                "environment": {
                    name: anchor[name]
                    for name in ("fog_percent", "illumination_lux", "camera_noise")
                },
                "original_map50": anchor["map50"],
                "original_verdict": anchor["verdict"],
                "original_cache_key": anchor["cache_key"],
            }
        )
    return conditions


def main() -> int:
    plan_path = CONFIG_ROOT / "scenario_plan.json"
    config_path = CONFIG_ROOT / "evaluation_config.json"
    gate_path = OUTPUT_ROOT / "preinference_gate.json"
    plan, config = validate_gate(plan_path, config_path, gate_path)
    original = read_json(ORIGINAL_AGGREGATE)
    original_by_id = {item["scenario_id"]: item for item in original["scenarios"]}
    weights = Path(config["weights_path"]).resolve()
    actual_weights_hash = file_sha256(weights)
    if actual_weights_hash != config["expected_weights_sha256"]:
        raise RuntimeError("Weights hash mismatch before independent revalidation")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    session_id = f"{ORIGINAL_SESSION}__fresh_revalidation__{timestamp}"
    session_root = REVALIDATION_ROOT / session_id
    session_root.mkdir(parents=True, exist_ok=False)
    seed_everything(42)
    detector = YoloFrameDetector(weights, config["yolov8"])
    model_metadata = detector.metadata(weights, actual_weights_hash)
    exporter = ScenarioMatlabExporter()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for scenario in plan["scenarios"]:
            if file_sha256(Path(scenario["scenario_config_path"])) != scenario["scenario_config_sha256"]:
                raise RuntimeError(f"Scenario changed before revalidation: {scenario['scenario_id']}")
            for source in original_conditions(original_by_id[scenario["scenario_id"]], config):
                condition_dir = session_root / scenario["scenario_id"] / source["condition"]
                environment = Environment.from_dict(source["environment"])
                condition_started = time.perf_counter()
                manifest = exporter.export(environment, condition_dir, scenario, config)
                validate_frame_alignment(manifest["frames"], int(config["frame_stride"]))
                paths = [str(frame["image_path"]) for frame in manifest["frames"]]
                detections, inference_seconds, speed_rows = detector.detect_paths(paths)
                frames = [
                    {
                        "frame_index": int(frame["frame_index"]),
                        "time_seconds": float(frame["time_seconds"]),
                        "uav_xyz": frame.get("uav_xyz"),
                        "image_path": frame["image_path"],
                        "ground_truth": frame.get("ground_truth", []),
                        "detections": detected,
                    }
                    for frame, detected in zip(manifest["frames"], detections)
                ]
                metrics = compute_map50(frames, float(config["yolov8"]["ap_iou_threshold"]))
                metrics.update(precision_recall(metrics))
                metrics["matched_iou"] = matched_iou_summary(
                    frames, float(config["yolov8"]["ap_iou_threshold"])
                )
                new_score = float(metrics["map50"])
                new_verdict = classify_verdict(
                    new_score,
                    float(config["thresholds"]["pass_map50"]),
                    float(config["thresholds"]["fail_map50"]),
                )
                difference = new_score - float(source["original_map50"])
                evaluation = {
                    "schema_version": "1.0",
                    "session_id": session_id,
                    "scenario_id": scenario["scenario_id"],
                    "condition": source["condition"],
                    "environment": environment.as_dict(),
                    "fresh_render_and_inference": True,
                    "original_search_cache_used": False,
                    "original_search_cache_key_for_comparison_only": source["original_cache_key"],
                    "weights_sha256": actual_weights_hash,
                    "scenario_config_sha256": scenario["scenario_config_sha256"],
                    "model": model_metadata,
                    "matlab": {
                        "version": manifest["matlab_version"],
                        "release": manifest["matlab_release"],
                        "renderer": manifest["renderer"],
                        "evaluated_frame_count": manifest["evaluated_frame_count"],
                    },
                    "metrics": metrics,
                    "new_verdict": new_verdict,
                    "original_map50": source["original_map50"],
                    "original_verdict": source["original_verdict"],
                    "signed_map50_difference": difference,
                    "absolute_map50_difference": abs(difference),
                    "verdict_match": new_verdict == source["original_verdict"],
                    "rendering_seconds": float(manifest["frame_rendering_seconds"]),
                    "geometry_seconds": float(manifest["geometry_simulation_seconds"]),
                    "inference_seconds": inference_seconds,
                    "condition_wall_seconds": time.perf_counter() - condition_started,
                    "ultralytics_speed_ms_per_frame": speed_rows,
                    "frames": frames,
                }
                write_json_new(condition_dir / "revalidation_evaluation.json", evaluation)
                rows.append(
                    {
                        "session_id": session_id,
                        "scenario_id": scenario["scenario_id"],
                        "condition": source["condition"],
                        **environment.as_dict(),
                        "original_map50": source["original_map50"],
                        "revalidated_map50": new_score,
                        "signed_map50_difference": difference,
                        "absolute_map50_difference": abs(difference),
                        "original_verdict": source["original_verdict"],
                        "revalidated_verdict": new_verdict,
                        "verdict_match": new_verdict == source["original_verdict"],
                        "fresh_render_and_inference": True,
                        "original_search_cache_used": False,
                        "weights_sha256": actual_weights_hash,
                        "scenario_config_sha256": scenario["scenario_config_sha256"],
                        "condition_wall_seconds": evaluation["condition_wall_seconds"],
                    }
                )
    finally:
        exporter.close()

    summary = {
        "session_id": session_id,
        "original_session_id": ORIGINAL_SESSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": PLAN_ID,
        "condition_count": len(rows),
        "scenario_count": len({row["scenario_id"] for row in rows}),
        "all_fresh_render_and_inference": all(row["fresh_render_and_inference"] for row in rows),
        "any_original_search_cache_used": any(row["original_search_cache_used"] for row in rows),
        "verdict_match_count": sum(row["verdict_match"] for row in rows),
        "all_verdicts_match": all(row["verdict_match"] for row in rows),
        "maximum_absolute_map50_difference": max(row["absolute_map50_difference"] for row in rows),
        "mean_absolute_map50_difference": statistics.fmean(
            row["absolute_map50_difference"] for row in rows
        ),
        "total_wall_seconds": time.perf_counter() - started,
        "weights_sha256": actual_weights_hash,
        "plan_sha256": file_sha256(plan_path),
        "config_sha256": file_sha256(config_path),
        "rows": rows,
    }
    write_json_new(session_root / "independent_revalidation_summary.json", summary)
    write_csv_new(session_root / "independent_revalidation_results.csv", rows)
    print(json.dumps({key: summary[key] for key in ("session_id", "condition_count", "all_verdicts_match", "maximum_absolute_map50_difference", "total_wall_seconds")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
