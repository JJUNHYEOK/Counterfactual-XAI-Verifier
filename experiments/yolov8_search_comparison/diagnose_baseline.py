"""Audit GT/evaluation geometry and compare candidate weights on one baseline.

This script never changes GT, IoU, confidence, or PASS thresholds. It writes
diagnostic alternatives (for example, clipped-to-image GT) separately so an
actual coordinate error can be distinguished from a model-domain failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from .model_mapping import resolve_target_class_mapping
from .search_core import bbox_iou_xywh, classify_verdict, compute_map50


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_EVALUATION = (
    HERE / "raw" / "_shared_evaluations" / "cache"
    / "40e6dc6ae0acf5e2b40d" / "evaluation.json"
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def candidate_metadata(model: Any, path: Path) -> dict[str, Any]:
    checkpoint = model.ckpt or {}
    args = checkpoint.get("train_args") or {}
    stat = path.stat()
    mapping = resolve_target_class_mapping(model.names)
    training_data = args.get("data")
    simulation_evidence = "none; checkpoint metadata does not reference this MATLAB simulation"
    if training_data and "yolov8s_sim_20260902" in str(training_data):
        simulation_evidence = (
            "checkpoint train_args.data references the scenario-separated MATLAB simulation dataset"
        )
    return {
        "absolute_path": str(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
        "task": model.task,
        "model_names": {int(k): str(v) for k, v in model.names.items()},
        "class_count": len(model.names),
        "resolved_experiment_mapping": mapping,
        "checkpoint_date": checkpoint.get("date"),
        "checkpoint_ultralytics_version": checkpoint.get("version"),
        "model_definition": args.get("model"),
        "training_data": training_data,
        "training_epochs": args.get("epochs"),
        "training_imgsz": args.get("imgsz"),
        "training_batch": args.get("batch"),
        "training_seed": args.get("seed"),
        "training_project": args.get("project"),
        "training_name": args.get("name"),
        "simulation_training_evidence": simulation_evidence,
    }


def run_candidate(
    weights: Path,
    source_frames: list[dict[str, Any]],
    *,
    conf: float,
    nms_iou: float,
    imgsz: int,
    device: str,
    batch: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metadata = candidate_metadata(model, weights)
    mapping = {int(k): str(v) for k, v in metadata["resolved_experiment_mapping"].items()}
    paths = [str(frame["image_path"]) for frame in source_frames]
    started = time.perf_counter()
    predicted_frames: list[dict[str, Any]] = []
    original_shapes: set[tuple[int, int]] = set()
    results = model.predict(
        source=paths,
        conf=conf,
        iou=nms_iou,
        imgsz=imgsz,
        device=device,
        batch=batch,
        stream=True,
        verbose=False,
    )
    for source, result in zip(source_frames, results):
        original_shapes.add(tuple(int(v) for v in result.orig_shape[:2]))
        detections: list[dict[str, Any]] = []
        if result.boxes is not None and result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            for box, class_id, confidence in zip(boxes, classes, confidences):
                class_id = int(class_id)
                if class_id not in mapping:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box)
                detections.append(
                    {
                        "source_class_id": class_id,
                        "source_class_name": str(model.names[class_id]),
                        "class_name": mapping[class_id],
                        "confidence": float(confidence),
                        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )
        predicted_frames.append(
            {
                "frame_index": int(source["frame_index"]),
                "time_seconds": float(source["time_seconds"]),
                "image_path": str(source["image_path"]),
                "ground_truth": deepcopy(source["ground_truth"]),
                "detections": detections,
            }
        )
    inference_seconds = time.perf_counter() - started
    metrics = compute_map50(predicted_frames, 0.5)
    metadata.update(
        {
            "runtime_ultralytics_version": ultralytics.__version__,
            "runtime_torch_version": torch.__version__,
            "runtime_python_version": platform.python_version(),
            "runtime_device": device,
            "evaluation_imgsz": imgsz,
            "confidence_threshold": conf,
            "nms_iou_threshold": nms_iou,
            "ap_iou_threshold": 0.5,
            "original_result_shapes_hw": [list(shape) for shape in sorted(original_shapes)],
            "inference_seconds": inference_seconds,
            "metrics": metrics,
            "verdict": classify_verdict(metrics["map50"]),
        }
    )
    return metadata, predicted_frames


def clip_xywh(box: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = (float(v) for v in box)
    x1, y1 = max(0.0, min(float(width), x)), max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def gt_audit_rows(
    frames: list[dict[str, Any]], width: int, height: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    boxes: list[dict[str, Any]] = []
    frame_counts: list[dict[str, Any]] = []
    for frame_position, frame in enumerate(frames):
        people = sum(gt["class_name"] == "person" for gt in frame["ground_truth"])
        vehicles = sum(gt["class_name"] == "vehicle" for gt in frame["ground_truth"])
        frame_counts.append(
            {
                "frame_position_zero_based": frame_position,
                "frame_index_matlab_one_based": frame["frame_index"],
                "image_filename": Path(frame["image_path"]).name,
                "person_count": people,
                "vehicle_count": vehicles,
                "total_count": people + vehicles,
            }
        )
        for gt in frame["ground_truth"]:
            x, y, w, h = (float(v) for v in gt["bbox_xywh"])
            clipped = clip_xywh(gt["bbox_xywh"], width, height)
            boxes.append(
                {
                    "frame_position_zero_based": frame_position,
                    "frame_index_matlab_one_based": frame["frame_index"],
                    "object_id": gt["object_id"],
                    "class_name": gt["class_name"],
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "area": w * h,
                    "center_x": x + w / 2,
                    "center_y": y + h / 2,
                    "nonpositive": w <= 0 or h <= 0,
                    "outside_left": x < 0,
                    "outside_top": y < 0,
                    "outside_right": x + w > width,
                    "outside_bottom": y + h > height,
                    "any_out_of_bounds": x < 0 or y < 0 or x + w > width or y + h > height,
                    "clipped_x": clipped[0],
                    "clipped_y": clipped[1],
                    "clipped_width": clipped[2],
                    "clipped_height": clipped[3],
                    "clipped_area": clipped[2] * clipped[3],
                }
            )
    return boxes, frame_counts


def summarise_gt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for class_name in ("person", "vehicle", "ALL"):
        selected = rows if class_name == "ALL" else [r for r in rows if r["class_name"] == class_name]
        summary: dict[str, Any] = {
            "class_name": class_name,
            "box_count": len(selected),
            "unique_object_count": len({r["object_id"] for r in selected}),
            "nonpositive_count": sum(bool(r["nonpositive"]) for r in selected),
            "out_of_bounds_count": sum(bool(r["any_out_of_bounds"]) for r in selected),
        }
        for key in ("width", "height", "area"):
            values = np.asarray([float(r[key]) for r in selected], dtype=float)
            summary.update(
                {
                    f"{key}_min": float(values.min()) if len(values) else None,
                    f"{key}_q25": float(np.quantile(values, 0.25)) if len(values) else None,
                    f"{key}_median": float(np.median(values)) if len(values) else None,
                    f"{key}_mean": float(values.mean()) if len(values) else None,
                    f"{key}_q75": float(np.quantile(values, 0.75)) if len(values) else None,
                    f"{key}_max": float(values.max()) if len(values) else None,
                }
            )
        summaries.append(summary)
    return summaries


def best_gt_match(det: dict[str, Any], gts: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    candidates = [gt for gt in gts if gt["class_name"] == det["class_name"]]
    if not candidates:
        return None, 0.0
    pairs = [(gt, bbox_iou_xywh(det["bbox_xywh"], gt["bbox_xywh"])) for gt in candidates]
    return max(pairs, key=lambda pair: pair[1])


def prediction_match_rows(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_position, frame in enumerate(frames):
        for detection_index, det in enumerate(frame["detections"]):
            gt, iou = best_gt_match(det, frame["ground_truth"])
            dx = dy = wr = hr = None
            object_id = None
            if gt is not None:
                gx, gy, gw, gh = (float(v) for v in gt["bbox_xywh"])
                px, py, pw, ph = (float(v) for v in det["bbox_xywh"])
                dx = (px + pw / 2) - (gx + gw / 2)
                dy = (py + ph / 2) - (gy + gh / 2)
                wr = pw / gw if gw > 0 else None
                hr = ph / gh if gh > 0 else None
                object_id = gt["object_id"]
            rows.append(
                {
                    "frame_position_zero_based": frame_position,
                    "frame_index_matlab_one_based": frame["frame_index"],
                    "detection_index": detection_index,
                    "class_name": det["class_name"],
                    "source_class_id": det["source_class_id"],
                    "source_class_name": det["source_class_name"],
                    "confidence": det["confidence"],
                    "matched_object_id": object_id,
                    "maximum_same_frame_iou": iou,
                    "center_dx_pixels": dx,
                    "center_dy_pixels": dy,
                    "prediction_to_gt_width_ratio": wr,
                    "prediction_to_gt_height_ratio": hr,
                    "is_tp_at_iou_0_5": iou >= 0.5,
                }
            )
    return rows


def alignment_rows(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in (-2, -1, 0, 1, 2):
        values: list[float] = []
        for frame_pos, frame in enumerate(frames):
            gt_pos = frame_pos + offset
            if not 0 <= gt_pos < len(frames):
                continue
            shifted_gts = frames[gt_pos]["ground_truth"]
            for det in frame["detections"]:
                _, iou = best_gt_match(det, shifted_gts)
                values.append(iou)
        rows.append(
            {
                "gt_frame_offset_relative_to_prediction": offset,
                "pair_count": len(values),
                "mean_max_iou": mean(values) if values else None,
                "median_max_iou": median(values) if values else None,
                "maximum_iou": max(values) if values else None,
                "count_iou_ge_0_5": sum(value >= 0.5 for value in values),
            }
        )
    return rows


def clipped_gt_frames(frames: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for frame in frames:
        copied = deepcopy(frame)
        clipped_gts = []
        for gt in copied["ground_truth"]:
            gt["bbox_xywh"] = clip_xywh(gt["bbox_xywh"], width, height)
            if gt["bbox_xywh"][2] > 0 and gt["bbox_xywh"][3] > 0:
                clipped_gts.append(gt)
        copied["ground_truth"] = clipped_gts
        output.append(copied)
    return output


def draw_overlay(
    frame: dict[str, Any], path: Path, *, show_gt: bool, show_predictions: bool, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = mpimg.imread(frame["image_path"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    if show_gt:
        for gt in frame["ground_truth"]:
            x, y, w, h = gt["bbox_xywh"]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#28c840", linewidth=2))
            ax.text(x, max(3, y), f"GT {gt['class_name']} #{gt['object_id']}", color="white", fontsize=8,
                    bbox={"facecolor": "#16852a", "alpha": 0.8, "pad": 1})
    if show_predictions:
        for det in frame["detections"]:
            x, y, w, h = det["bbox_xywh"]
            _, iou = best_gt_match(det, frame["ground_truth"])
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#ef3038", linewidth=2))
            ax.text(x, min(image.shape[0] - 3, y + h),
                    f"YOLO {det['class_name']} {det['confidence']:.2f} IoU={iou:.3f}",
                    color="white", fontsize=8,
                    bbox={"facecolor": "#a91f26", "alpha": 0.8, "pad": 1})
    ax.set_title(title)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_visualizations(frames: list[dict[str, Any]], output_dir: Path) -> None:
    positions = sorted(set(int(v) for v in np.linspace(0, len(frames) - 1, 20).round()))
    for position in positions:
        frame = frames[position]
        stem = f"frame_{int(frame['frame_index']):04d}"
        draw_overlay(frame, output_dir / "gt" / f"{stem}_gt.png", show_gt=True,
                     show_predictions=False, title=f"Frame {frame['frame_index']} — ground truth")
        draw_overlay(frame, output_dir / "predictions" / f"{stem}_pred.png", show_gt=False,
                     show_predictions=True, title=f"Frame {frame['frame_index']} — YOLO predictions")
        draw_overlay(frame, output_dir / "combined" / f"{stem}_combined.png", show_gt=True,
                     show_predictions=True, title=f"Frame {frame['frame_index']} — GT + YOLO")

    # Reproduce class-wise confidence ordering and one-to-one GT assignment so
    # duplicate detections are correctly included among non-TP cases even when
    # their raw maximum IoU exceeds 0.5.
    misses: list[tuple[float, int, int, float]] = []
    for class_name in ("person", "vehicle"):
        ranked = sorted(
            (
                (float(det["confidence"]), frame_pos, det_pos, det)
                for frame_pos, frame in enumerate(frames)
                for det_pos, det in enumerate(frame["detections"])
                if det["class_name"] == class_name
            ),
            reverse=True,
        )
        used: set[tuple[int, int]] = set()
        for _, frame_pos, det_pos, det in ranked:
            candidates = [
                (gt_pos, gt, bbox_iou_xywh(det["bbox_xywh"], gt["bbox_xywh"]))
                for gt_pos, gt in enumerate(frames[frame_pos]["ground_truth"])
                if gt["class_name"] == class_name
            ]
            if candidates:
                gt_pos, _, iou = max(candidates, key=lambda item: item[2])
            else:
                gt_pos, iou = -1, 0.0
            is_tp = iou >= 0.5 and (frame_pos, gt_pos) not in used
            if is_tp:
                used.add((frame_pos, gt_pos))
            else:
                misses.append((abs(0.5 - iou), frame_pos, det_pos, iou))
    for rank, (_, frame_pos, det_pos, iou) in enumerate(sorted(misses)[:10], start=1):
        frame = frames[frame_pos]
        det = frame["detections"][det_pos]
        draw_overlay(
            frame,
            output_dir / "nearest_misses" / f"rank_{rank:02d}_frame_{frame['frame_index']:04d}.png",
            show_gt=True,
            show_predictions=True,
            title=(f"Nearest miss #{rank}: frame {frame['frame_index']} / "
                   f"{det['class_name']} conf={det['confidence']:.3f} IoU={iou:.5f}"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation = json.loads(args.evaluation.resolve().read_text(encoding="utf-8"))
    source_frames = evaluation["frames"]
    width = int(evaluation["matlab"]["image_width"])
    height = int(evaluation["matlab"]["image_height"])
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (args.output_dir or HERE / "diagnostics" / f"baseline_recovery_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    gt_rows, frame_counts = gt_audit_rows(source_frames, width, height)
    write_csv(output_dir / "gt_box_distribution.csv", gt_rows)
    write_csv(output_dir / "gt_summary.csv", summarise_gt(gt_rows))
    write_csv(output_dir / "frame_object_counts.csv", frame_counts)

    candidate_summaries: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []
    for candidate in args.candidate:
        candidate = candidate.resolve()
        metadata, predicted_frames = run_candidate(
            candidate,
            source_frames,
            conf=args.confidence,
            nms_iou=args.nms_iou,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
        )
        slug = f"{candidate.stem}_{metadata['sha256'][:10]}"
        model_dir = output_dir / "candidates" / slug
        match_rows = prediction_match_rows(predicted_frames)
        alignment = alignment_rows(predicted_frames)
        clipped_metrics = compute_map50(clipped_gt_frames(predicted_frames, width, height), 0.5)
        write_csv(model_dir / "prediction_gt_matches.csv", match_rows)
        write_csv(model_dir / "frame_alignment_offsets.csv", alignment)
        write_json(model_dir / "candidate_metadata.json", metadata)
        write_json(
            model_dir / "candidate_evaluation.json",
            {
                "metadata": metadata,
                "metrics_unmodified_gt": metadata["metrics"],
                "metrics_diagnostic_clipped_gt_not_used_for_selection": clipped_metrics,
                "frames": predicted_frames,
            },
        )
        make_visualizations(predicted_frames, model_dir / "visualizations")
        values = [float(row["maximum_same_frame_iou"]) for row in match_rows]
        summary = {
            "candidate": str(candidate),
            "sha256": metadata["sha256"],
            "model_definition": metadata["model_definition"],
            "training_data": metadata["training_data"],
            "class_count": metadata["class_count"],
            "resolved_mapping": json.dumps(metadata["resolved_experiment_mapping"], sort_keys=True),
            "person_ap50": metadata["metrics"]["person_ap50"],
            "vehicle_ap50": metadata["metrics"]["vehicle_ap50"],
            "map50": metadata["metrics"]["map50"],
            "verdict": metadata["verdict"],
            "person_gt": metadata["metrics"]["per_class"]["person"]["total_gt"],
            "vehicle_gt": metadata["metrics"]["per_class"]["vehicle"]["total_gt"],
            "person_detections": metadata["metrics"]["per_class"]["person"]["n_detections"],
            "vehicle_detections": metadata["metrics"]["per_class"]["vehicle"]["n_detections"],
            "person_tp": metadata["metrics"]["per_class"]["person"]["tp"],
            "vehicle_tp": metadata["metrics"]["per_class"]["vehicle"]["tp"],
            "maximum_same_class_iou": max(values) if values else None,
            "mean_detection_max_iou": mean(values) if values else None,
            "clipped_gt_map50_diagnostic_only": clipped_metrics["map50"],
            "inference_seconds": metadata["inference_seconds"],
        }
        candidate_summaries.append(summary)
        all_metadata.append(metadata)

    write_csv(output_dir / "candidate_comparison.csv", candidate_summaries)
    write_json(
        output_dir / "audit_report.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_evaluation": str(args.evaluation.resolve()),
            "coordinate_contract": {
                "gt_format": "absolute pixel xywh",
                "gt_origin": "continuous MATLAB pinhole coordinates; no normalization",
                "frame_index": "MATLAB one-based index stored explicitly and paired by manifest order",
                "image_size_wh": [width, height],
                "prediction_format_from_ultralytics": "xyxy scaled back to original image shape",
                "scoring_conversion": "prediction xyxy -> absolute pixel xywh",
                "letterbox_note": "Ultralytics result.boxes.xyxy is already scaled to result.orig_shape",
                "ap_iou_threshold": 0.5,
            },
            "gt_summary": summarise_gt(gt_rows),
            "candidate_comparison": candidate_summaries,
            "candidate_metadata": all_metadata,
            "selection_rule": "A candidate is eligible only if unmodified-GT baseline mAP@0.5 >= 0.50.",
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "candidates": candidate_summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
