"""Independently evaluate one selected weight on the held-out 181-frame baseline."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from .diagnose_baseline import prediction_match_rows, run_candidate
from .run_comparison import write_json


def write_csv(path: Path, row: dict[str, Any]) -> None:
    flat = {
        key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    started = time.perf_counter()
    source = json.loads(args.evaluation.resolve().read_text(encoding="utf-8"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    metadata, frames = run_candidate(
        args.weights.resolve(),
        source["frames"],
        conf=args.confidence,
        nms_iou=args.nms_iou,
        imgsz=args.imgsz,
        device=device,
        batch=args.batch,
    )
    metrics = metadata["metrics"]
    matches = prediction_match_rows(frames)
    ious = [float(row["maximum_same_frame_iou"]) for row in matches]
    class_rows: dict[str, dict[str, Any]] = {}
    for class_name, values in metrics["per_class"].items():
        tp, fp, fn = int(values["tp"]), int(values["fp"]), int(values["fn"])
        class_rows[class_name] = {
            **values,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }
    total_tp = sum(row["tp"] for row in class_rows.values())
    total_fp = sum(row["fp"] for row in class_rows.values())
    total_fn = sum(row["fn"] for row in class_rows.values())
    total_seconds = time.perf_counter() - started
    metric_scoring_seconds = float(source["timings"].get("metric_scoring_seconds", 0.0))
    accounted_full_seconds = (
        float(source["timings"]["geometry_simulation_seconds"])
        + float(source["timings"]["frame_rendering_seconds"])
        + float(metadata["inference_seconds"])
        + metric_scoring_seconds
    )
    result = {
        "schema_version": "1.0",
        "source_evaluation": str(args.evaluation.resolve()),
        "held_out": True,
        "scenario_id": source["scenario_id"],
        "scenario_variant": int(source["matlab"].get("scenario_variant", 0)),
        "random_seed": int(source["random_seed"]),
        "environment": source["environment"],
        "evaluated_frame_count": int(source["matlab"]["evaluated_frame_count"]),
        "ground_truth_mode": source["matlab"]["ground_truth_mode"],
        "model": metadata,
        "evaluation_settings": {
            "image_size": args.imgsz,
            "confidence_threshold": args.confidence,
            "nms_iou_threshold": args.nms_iou,
            "ap_iou_threshold": 0.5,
            "batch_size": args.batch,
            "device": device,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "python_version": platform.python_version(),
        },
        "metrics": {
            "map50": metrics["map50"],
            "person_ap50": metrics["person_ap50"],
            "vehicle_ap50": metrics["vehicle_ap50"],
            "visible_intruder_count": metrics["visible_intruder_count"],
            "detected_intruder_count": metrics["detected_intruder_count"],
            "per_class": class_rows,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "micro_precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
            "micro_recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
            "maximum_detection_to_same_frame_gt_iou": max(ious) if ious else None,
            "mean_detection_to_same_frame_gt_iou": mean(ious) if ious else None,
        },
        "timings": {
            "geometry_simulation_seconds": source["timings"]["geometry_simulation_seconds"],
            "frame_rendering_seconds": source["timings"]["frame_rendering_seconds"],
            "yolo_inference_seconds": metadata["inference_seconds"],
            "metric_scoring_seconds": metric_scoring_seconds,
            "accounted_full_evaluation_seconds": accounted_full_seconds,
            "validation_reuse_wall_seconds": total_seconds,
        },
        "thresholds": {"pass_map50": 0.5, "fail_map50": 0.25},
        "verdict": metadata["verdict"],
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "baseline_validation.json", result)
    csv_row = {
        "verdict": result["verdict"],
        "map50": metrics["map50"],
        "person_ap50": metrics["person_ap50"],
        "vehicle_ap50": metrics["vehicle_ap50"],
        "person_gt": class_rows["person"]["total_gt"],
        "person_detections": class_rows["person"]["n_detections"],
        "person_tp": class_rows["person"]["tp"],
        "person_fp": class_rows["person"]["fp"],
        "person_fn": class_rows["person"]["fn"],
        "person_precision": class_rows["person"]["precision"],
        "person_recall": class_rows["person"]["recall"],
        "vehicle_gt": class_rows["vehicle"]["total_gt"],
        "vehicle_detections": class_rows["vehicle"]["n_detections"],
        "vehicle_tp": class_rows["vehicle"]["tp"],
        "vehicle_fp": class_rows["vehicle"]["fp"],
        "vehicle_fn": class_rows["vehicle"]["fn"],
        "vehicle_precision": class_rows["vehicle"]["precision"],
        "vehicle_recall": class_rows["vehicle"]["recall"],
        "maximum_iou": result["metrics"]["maximum_detection_to_same_frame_gt_iou"],
        "mean_iou": result["metrics"]["mean_detection_to_same_frame_gt_iou"],
        "simulation_seconds": result["timings"]["geometry_simulation_seconds"],
        "rendering_seconds": result["timings"]["frame_rendering_seconds"],
        "inference_seconds": result["timings"]["yolo_inference_seconds"],
        "metric_scoring_seconds": metric_scoring_seconds,
        "accounted_full_evaluation_seconds": accounted_full_seconds,
        "validation_reuse_wall_seconds": total_seconds,
        "weights_path": metadata["absolute_path"],
        "weights_sha256": metadata["sha256"],
        "evaluation_settings": result["evaluation_settings"],
    }
    write_csv(output_dir / "baseline_validation.csv", csv_row)
    write_json(output_dir / "baseline_predictions.json", frames)
    print(json.dumps({"output_dir": str(output_dir), "verdict": result["verdict"], "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
