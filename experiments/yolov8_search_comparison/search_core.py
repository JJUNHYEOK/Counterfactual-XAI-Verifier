"""Pure boundary-search and AP@0.5 logic used by the KCI experiment.

This module intentionally imports neither MATLAB, Ultralytics, SHAP, nor an
LLM.  Keeping the policy and scoring code pure makes it possible to prove that
the two experiment arms differ only in their post-FAIL environment selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


ENV_KEYS = ("fog_percent", "illumination_lux", "camera_noise")
TARGET_CLASSES = ("person", "vehicle")


@dataclass(frozen=True)
class Environment:
    fog_percent: float
    illumination_lux: float
    camera_noise: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Environment":
        return cls(*(float(value[key]) for key in ENV_KEYS))

    def as_dict(self) -> dict[str, float]:
        return {
            "fog_percent": self.fog_percent,
            "illumination_lux": self.illumination_lux,
            "camera_noise": self.camera_noise,
        }

    def rounded(self) -> "Environment":
        return Environment(
            round(self.fog_percent, 2),
            round(self.illumination_lux, 1),
            round(self.camera_noise, 4),
        )


def clamp_environment(
    env: Environment, bounds: dict[str, list[float]]
) -> Environment:
    values = env.as_dict()
    return Environment.from_dict(
        {
            key: max(float(bounds[key][0]), min(float(bounds[key][1]), values[key]))
            for key in ENV_KEYS
        }
    ).rounded()


def degrade_environment(
    env: Environment,
    bounds: dict[str, list[float]],
    degradation: dict[str, float],
) -> Environment:
    """Common pre-FAIL rule: fog +30 %p, light x0.5, noise +0.20."""
    return clamp_environment(
        Environment(
            env.fog_percent + float(degradation["fog_add_percent_points"]),
            env.illumination_lux
            * float(degradation["illumination_multiplier"]),
            env.camera_noise + float(degradation["camera_noise_add"]),
        ),
        bounds,
    )


def interpolate_environment(
    current: Environment,
    target: Environment,
    target_weight: float,
    bounds: dict[str, list[float]],
) -> Environment:
    if not 0.0 <= target_weight <= 1.0:
        raise ValueError("target_weight must be in [0, 1]")
    a = current.as_dict()
    b = target.as_dict()
    return clamp_environment(
        Environment.from_dict(
            {
                key: (1.0 - target_weight) * a[key] + target_weight * b[key]
                for key in ENV_KEYS
            }
        ),
        bounds,
    )


def classify_verdict(
    map50: float, pass_threshold: float = 0.50, fail_threshold: float = 0.25
) -> str:
    """Three-tier verdict; MARGINAL is on the search's non-failure side."""
    if map50 >= pass_threshold:
        return "PASS"
    if map50 < fail_threshold:
        return "FAIL"
    return "MARGINAL"


def choose_next_environment(
    *,
    method: str,
    current: Environment,
    current_verdict: str,
    nonfail_anchor: Environment | None,
    fail_anchor: Environment | None,
    bounds: dict[str, list[float]],
    degradation: dict[str, float],
) -> tuple[Environment | None, str]:
    """Return the next environment and an auditable calculation label.

    Before the first FAIL both methods use exactly the same degradation rule.
    Once a FAIL exists, symmetric search uses the two-anchor midpoint.  The
    asymmetric arm uses 65% toward FAIL after a non-FAIL and 75% toward the
    non-FAIL anchor after a FAIL.
    """
    if method not in {"symmetric", "asymmetric"}:
        raise ValueError(f"Unsupported search method: {method}")
    if current_verdict not in {"PASS", "MARGINAL", "FAIL"}:
        raise ValueError(f"Unsupported verdict: {current_verdict}")

    if fail_anchor is None:
        return (
            degrade_environment(current, bounds, degradation),
            "common_pre_fail_degradation: fog+30pp, illumination*0.50, noise+0.20",
        )

    if nonfail_anchor is None:
        return None, "stop_unbracketed: FAIL exists but no PASS/MARGINAL anchor"

    if method == "symmetric":
        return (
            interpolate_environment(nonfail_anchor, fail_anchor, 0.50, bounds),
            "symmetric_midpoint: 0.50*nonfail + 0.50*fail",
        )

    if current_verdict == "FAIL":
        return (
            interpolate_environment(current, nonfail_anchor, 0.75, bounds),
            "asymmetric_recovery: 0.25*current_fail + 0.75*nonfail",
        )
    return (
        interpolate_environment(current, fail_anchor, 0.65, bounds),
        "asymmetric_probe: 0.35*current_nonfail + 0.65*fail",
    )


def environment_gap(
    nonfail: Environment, fail: Environment, bounds: dict[str, list[float]]
) -> float:
    a, b = nonfail.as_dict(), fail.as_dict()
    return float(
        sum(
            abs(a[key] - b[key])
            / (float(bounds[key][1]) - float(bounds[key][0]))
            for key in ENV_KEYS
        )
    )


def map_gap(nonfail_map50: float, fail_map50: float) -> float:
    """Absolute mAP gap between the final non-failure and failure anchors."""
    return abs(float(nonfail_map50) - float(fail_map50))


def bbox_iou_xywh(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    ax2, ay2, bx2, by2 = ax + max(0.0, aw), ay + max(0.0, ah), bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1, ix2, iy2 = max(ax, bx), max(ay, by), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return 0.0 if union <= 0.0 else inter / union


def _class_ap(
    frames: list[dict[str, Any]], class_name: str, iou_threshold: float
) -> dict[str, Any]:
    gt_by_frame: dict[int, list[dict[str, Any]]] = {}
    detections: list[tuple[float, int, dict[str, Any]]] = []
    gt_object_ids: set[int] = set()

    for frame_pos, frame in enumerate(frames):
        gts = [g for g in frame.get("ground_truth", []) if g["class_name"] == class_name]
        gt_by_frame[frame_pos] = gts
        gt_object_ids.update(int(g["object_id"]) for g in gts)
        for det in frame.get("detections", []):
            if det["class_name"] == class_name:
                detections.append((float(det["confidence"]), frame_pos, det))

    total_gt = sum(len(v) for v in gt_by_frame.values())
    detections.sort(key=lambda item: -item[0])
    used = {frame_pos: set() for frame_pos in gt_by_frame}
    hits: list[float] = []
    detected_object_ids: set[int] = set()

    for _, frame_pos, det in detections:
        best_idx, best_iou = -1, 0.0
        for gt_idx, gt in enumerate(gt_by_frame[frame_pos]):
            if gt_idx in used[frame_pos]:
                continue
            iou = bbox_iou_xywh(det["bbox_xywh"], gt["bbox_xywh"])
            if iou > best_iou:
                best_idx, best_iou = gt_idx, iou
        is_hit = best_idx >= 0 and best_iou >= iou_threshold
        hits.append(float(is_hit))
        if is_hit:
            used[frame_pos].add(best_idx)
            detected_object_ids.add(int(gt_by_frame[frame_pos][best_idx]["object_id"]))

    tp = int(sum(hits))
    fp = len(hits) - tp
    fn = total_gt - tp
    ap = 0.0
    if total_gt > 0 and hits:
        hit_array = np.asarray(hits, dtype=float)
        cum_tp = np.cumsum(hit_array)
        cum_fp = np.cumsum(1.0 - hit_array)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        recall = cum_tp / total_gt
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for idx in range(len(mpre) - 2, -1, -1):
            mpre[idx] = max(mpre[idx], mpre[idx + 1])
        changes = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))

    return {
        "class_name": class_name,
        "ap50": ap,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt": total_gt,
        "n_detections": len(detections),
        "visible_object_ids": sorted(gt_object_ids),
        "detected_object_ids": sorted(detected_object_ids),
    }


def compute_map50(frames: list[dict[str, Any]], iou_threshold: float = 0.5) -> dict[str, Any]:
    """Compute class-aware person/vehicle AP and their unweighted mean."""
    per_class = {
        name: _class_ap(frames, name, iou_threshold) for name in TARGET_CLASSES
    }
    active = [entry["ap50"] for entry in per_class.values() if entry["total_gt"] > 0]
    detected_ids = {
        (name, obj_id)
        for name, entry in per_class.items()
        for obj_id in entry["detected_object_ids"]
    }
    visible_ids = {
        (name, obj_id)
        for name, entry in per_class.items()
        for obj_id in entry["visible_object_ids"]
    }
    return {
        "map50": float(np.mean(active)) if active else 0.0,
        "person_ap50": per_class["person"]["ap50"],
        "vehicle_ap50": per_class["vehicle"]["ap50"],
        "detected_intruder_count": len(detected_ids),
        "visible_intruder_count": len(visible_ids),
        "per_class": per_class,
    }
