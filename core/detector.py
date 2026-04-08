from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_llm_input(
    frame_id: str,
    detector_name: str,
    image_size: tuple[int, int] | list[int],
    detections: list[dict[str, Any]],
    xai_summary: dict[str, Any],
    perf_summary: dict[str, Any],
    scenario_constraints: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """
    실제 검출/XAI 결과를 LLM 입력 JSON 형식으로 묶는다.
    """
    if scenario_constraints is None:
        scenario_constraints = {
            "allow_weather_change": True,
            "allow_lighting_change": True,
            "allow_obstacle_density_change": True,
        }

    return {
        "scene_id": frame_id,
        "task": "uav_collision_avoidance",
        "perception": {
            "detector": detector_name,
            "input_resolution": list(image_size),
            "detections": detections,
        },
        "performance_signals": perf_summary,
        "xai_signals": xai_summary,
        "scenario_constraints": scenario_constraints,
    }


def _make_dummy_perf_summary(
    corruption_type: str,
    severity: int,
    baseline_confidence: float,
    current_confidence: float,
    missed_detection: bool,
) -> dict[str, Any]:
    confidence_drop = max(0.0, baseline_confidence - current_confidence)

    if confidence_drop >= 0.12:
        confidence_trend = "decreasing"
    elif confidence_drop >= 0.05:
        confidence_trend = "slightly_decreasing"
    else:
        confidence_trend = "stable"

    if missed_detection or severity >= 4:
        miss_rate_trend = "increasing"
    elif severity >= 2:
        miss_rate_trend = "slightly_increasing"
    else:
        miss_rate_trend = "stable"

    if missed_detection:
        failure_type = "missed_detection"
    elif confidence_drop >= 0.10:
        failure_type = "late_avoidance"
    else:
        failure_type = "nominal"

    risk_score = min(
        1.0,
        0.15 * severity
        + confidence_drop * 2.0
        + (0.20 if missed_detection else 0.0),
    )

    return {
        "confidence_trend": confidence_trend,
        "miss_rate_trend": miss_rate_trend,
        "risk_score": round(risk_score, 3),
        "failure_type": failure_type,
        "corruption_type": corruption_type,
        "severity": severity,
        "baseline_confidence": round(baseline_confidence, 3),
        "current_confidence": round(current_confidence, 3),
    }


def _make_dummy_xai_summary(
    corruption_type: str,
    severity: int,
    current_confidence: float,
    missed_detection: bool,
) -> dict[str, Any]:
    factor_map = {
        "fog": [
            {"name": "fog_density", "importance": 0.74},
            {"name": "visibility_reduction", "importance": 0.61},
            {"name": "edge_blurring", "importance": 0.42},
        ],
        "rain": [
            {"name": "rain_intensity", "importance": 0.71},
            {"name": "motion_blur", "importance": 0.57},
            {"name": "specular_glare", "importance": 0.39},
        ],
        "glare": [
            {"name": "sun_glare", "importance": 0.76},
            {"name": "reflection", "importance": 0.58},
            {"name": "contrast_collapse", "importance": 0.41},
        ],
        "night": [
            {"name": "low_illumination", "importance": 0.79},
            {"name": "contrast_loss", "importance": 0.55},
            {"name": "sensor_noise", "importance": 0.43},
        ],
    }

    dominant_factors = factor_map.get(
        corruption_type,
        [
            {"name": corruption_type, "importance": 0.70},
            {"name": "environment_shift", "importance": 0.52},
            {"name": "background_confusion", "importance": 0.37},
        ],
    )

    if missed_detection:
        attention_summary = (
            "critical object evidence is lost and attention shifts away from the target region"
        )
    elif current_confidence < 0.5:
        attention_summary = (
            "attention becomes unstable and spreads toward background edges under degraded conditions"
        )
    elif severity >= 3:
        attention_summary = (
            "attention remains near the object but becomes less concentrated as degradation increases"
        )
    else:
        attention_summary = (
            "attention is mostly stable but begins to weaken under environmental degradation"
        )

    return {
        "method": "stub-xai",
        "dominant_factors": dominant_factors,
        "attention_summary": attention_summary,
    }


def build_dummy_llm_input(
    frame_id: str = "run_001_frame_0231",
    detector_name: str = "yolox-small",
    image_size: tuple[int, int] | list[int] = (1920, 1080),
    corruption_type: str = "fog",
    severity: int = 3,
    baseline_confidence: float = 0.91,
    current_confidence: float = 0.63,
    missed_detection: bool = False,
    detections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    XAI가 아직 없을 때 사용할 더미 입력 생성기.
    """

    if detections is None:
        # severity가 높을수록 score를 낮게 잡아 더미 상황을 만든다.
        base_score = max(0.18, current_confidence - 0.08)
        detections = [
            {
                "label": "pole",
                "score": round(base_score, 2),
                "bbox_xywh": [812, 316, 42, 155],
            },
            {
                "label": "tree",
                "score": round(min(0.95, base_score + 0.11), 2),
                "bbox_xywh": [421, 290, 96, 214],
            },
        ]

    perf_summary = _make_dummy_perf_summary(
        corruption_type=corruption_type,
        severity=severity,
        baseline_confidence=baseline_confidence,
        current_confidence=current_confidence,
        missed_detection=missed_detection,
    )

    xai_summary = _make_dummy_xai_summary(
        corruption_type=corruption_type,
        severity=severity,
        current_confidence=current_confidence,
        missed_detection=missed_detection,
    )

    return build_llm_input(
        frame_id=frame_id,
        detector_name=detector_name,
        image_size=image_size,
        detections=detections,
        xai_summary=xai_summary,
        perf_summary=perf_summary,
        scenario_constraints={
            "allow_weather_change": True,
            "allow_lighting_change": True,
            "allow_obstacle_density_change": False,
        },
    )


def save_llm_input(obj: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

    return out_path


def load_llm_input(in_path: str | Path) -> dict[str, Any]:
    in_path = Path(in_path)
    with in_path.open("r", encoding="utf-8") as f:
        return json.load(f)