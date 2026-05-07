"""detection_metrics.py — IoU, matching, mAP50 for the UAV detection pipeline.

Mirrors the MATLAB requirements_eval.m logic in Python so the same
mAP50 / continuity calculations apply once the detector is real
(YOLOv8s) instead of the geometric oracle. Used by:

    * Standalone YOLO test (computes mAP50 against synthetic GT)
    * Phase B Unreal pipeline (replaces F_detector + part of requirements_eval)

Schema:
    detections_per_frame : list[ list[Detection] ]    # one list per timestep
    gt_per_frame         : list[ list[GroundTruth] ]  # parallel to above

GroundTruth carries the true class so per-class mAP can be computed if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dspy_pipeline.yolo_detector import Detection


@dataclass
class GroundTruth:
    cls_name: str                     # 'person' | 'vehicle'
    xyxy:     tuple[float, float, float, float]


# ─────────────────────────────────────────────────────────────────────────────
# Geometric primitives
# ─────────────────────────────────────────────────────────────────────────────

def iou_xyxy(a: tuple[float, float, float, float],
             b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) \
       + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame greedy matching (same convention as requirements_eval.m)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameMatch:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # (det_idx, gt_idx, iou, conf) — for downstream PR-curve construction
    matches: list[tuple[int, int, float, float]] = None
    unmatched_dets: list[int] = None
    unmatched_gts:  list[int] = None


def match_frame(
    detections:    list[Detection],
    gts:           list[GroundTruth],
    iou_threshold: float = 0.5,
    require_class_match: bool = True,
) -> FrameMatch:
    """Greedy IoU matching: highest-confidence det grabs best-IoU GT first."""
    fm = FrameMatch(matches=[], unmatched_dets=[], unmatched_gts=list(range(len(gts))))
    if not detections:
        fm.fn = len(gts)
        return fm
    if not gts:
        fm.fp = len(detections)
        fm.unmatched_dets = list(range(len(detections)))
        return fm

    # Sort detections by descending confidence
    det_order = sorted(range(len(detections)), key=lambda i: -detections[i].conf)
    gt_used   = [False] * len(gts)

    for di in det_order:
        det = detections[di]
        best_iou = 0.0
        best_gi  = -1
        for gi, gt in enumerate(gts):
            if gt_used[gi]:                            continue
            if require_class_match and gt.cls_name != det.cls_name: continue
            iou = iou_xyxy(det.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_gi  = gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            gt_used[best_gi] = True
            fm.matches.append((di, best_gi, best_iou, det.conf))
            fm.tp += 1
        else:
            fm.fp += 1
            fm.unmatched_dets.append(di)
    fm.unmatched_gts = [gi for gi, used in enumerate(gt_used) if not used]
    fm.fn = len(fm.unmatched_gts)
    return fm


# ─────────────────────────────────────────────────────────────────────────────
# mAP50 across an entire run (same algorithm as requirements_eval.m)
# ─────────────────────────────────────────────────────────────────────────────

def compute_map50(
    detections_per_frame: list[list[Detection]],
    gt_per_frame:         list[list[GroundTruth]],
    iou_threshold:        float = 0.5,
) -> dict:
    """Compute aggregate mAP50 over all frames.

    Implements the 11-point + interpolated PR-curve method (matches MATLAB
    compute_map50). Returns dict with map50, tp, fp, fn, total_gt, precision,
    recall, and the per-frame match list for downstream continuity analysis.
    """
    assert len(detections_per_frame) == len(gt_per_frame), \
        "detections and gt frame counts must match"

    total_gt = 0
    all_dets: list[tuple[float, int]] = []   # (conf, hit_flag)
    per_frame_matches: list[FrameMatch] = []

    for dets, gts in zip(detections_per_frame, gt_per_frame):
        fm = match_frame(dets, gts, iou_threshold=iou_threshold)
        per_frame_matches.append(fm)
        total_gt += len(gts)
        # Record each detection with its hit/miss flag (for PR curve)
        det_hits = [0] * len(dets)
        for (di, _, _, _) in fm.matches:
            det_hits[di] = 1
        for di, det in enumerate(dets):
            all_dets.append((det.conf, det_hits[di]))

    # Aggregate counts
    tp = sum(fm.tp for fm in per_frame_matches)
    fp = sum(fm.fp for fm in per_frame_matches)
    fn = sum(fm.fn for fm in per_frame_matches)

    # PR curve from confidence-sorted detections
    map50 = 0.0
    precision = 0.0
    recall    = 0.0
    if total_gt > 0 and all_dets:
        all_dets.sort(key=lambda x: -x[0])
        hits = np.array([h for _, h in all_dets], dtype=float)
        cum_tp = np.cumsum(hits)
        cum_fp = np.cumsum(1 - hits)
        prec_curve   = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        recall_curve = cum_tp / total_gt
        precision = float(prec_curve[-1])
        recall    = float(recall_curve[-1])

        # Interpolated AP (VOC convention)
        mrec = np.concatenate(([0.0], recall_curve, [1.0]))
        mpre = np.concatenate(([0.0], prec_curve,   [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        map50 = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return {
        "map50":      map50,
        "tp":         int(tp),
        "fp":         int(fp),
        "fn":         int(fn),
        "total_gt":   int(total_gt),
        "precision":  precision,
        "recall":     recall,
        "per_frame":  per_frame_matches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Continuity: worst run of consecutive missed-frame events (REQ-3 mirror)
# ─────────────────────────────────────────────────────────────────────────────

def compute_worst_miss_run(
    detections_per_frame: list[list[Detection]],
    gt_per_frame:         list[list[GroundTruth]],
    score_threshold:      float = 0.30,
) -> dict:
    """A frame is 'missed' iff GT exists in it AND no detection above
    score_threshold matches any GT (any-class). Returns longest such run.
    """
    Nt = len(gt_per_frame)
    miss = [False] * Nt
    for ii, (dets, gts) in enumerate(zip(detections_per_frame, gt_per_frame)):
        any_gt   = len(gts) > 0
        any_det  = any(d.conf > score_threshold for d in dets)
        miss[ii] = any_gt and not any_det

    worst_run = 0
    run_len = 0
    worst_start = -1
    cur_start   = -1
    for ii, m in enumerate(miss):
        if m:
            if run_len == 0: cur_start = ii
            run_len += 1
            if run_len > worst_run:
                worst_run   = run_len
                worst_start = cur_start
        else:
            run_len = 0
    return {
        "worst_run":   worst_run,
        "worst_start": worst_start,
        "miss_flags":  miss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REQ-1 + REQ-3 wrapper (matches MATLAB requirements_eval.m semantics)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mission(
    detections_per_frame: list[list[Detection]],
    gt_per_frame:         list[list[GroundTruth]],
    map_threshold:        float = 0.50,
    continuity_threshold: int   = 3,
) -> dict:
    map_result        = compute_map50(detections_per_frame, gt_per_frame)
    continuity_result = compute_worst_miss_run(detections_per_frame, gt_per_frame)

    req1_passed = map_result["map50"] >= map_threshold
    req3_passed = continuity_result["worst_run"] <= continuity_threshold

    return {
        "all_passed":     bool(req1_passed and req3_passed),
        "violated_count": int(not req1_passed) + int(not req3_passed),
        "req1": {
            "id":         "REQ-1",
            "passed":     bool(req1_passed),
            "value":      map_result["map50"],
            "threshold":  map_threshold,
        },
        "req3": {
            "id":         "REQ-3",
            "passed":     bool(req3_passed),
            "value":      continuity_result["worst_run"],
            "threshold":  continuity_threshold,
        },
        "stats": {
            "tp":        map_result["tp"],
            "fp":        map_result["fp"],
            "fn":        map_result["fn"],
            "total_gt":  map_result["total_gt"],
            "precision": map_result["precision"],
            "recall":    map_result["recall"],
        },
    }
