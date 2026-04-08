from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    pass


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_llm_input(data: dict[str, Any]) -> bool:
    errors: list[str] = []

    if not isinstance(data, dict):
        raise SchemaValidationError("입력 JSON 최상위는 dict 형태여야 합니다.")

    required_top = [
        "scene_id",
        "task",
        "perception",
        "performance_signals",
        "xai_signals",
        "scenario_constraints",
    ]
    for key in required_top:
        if key not in data:
            errors.append(f"최상위 필드 누락: {key}")

    perception = data.get("perception", {})
    if not isinstance(perception, dict):
        errors.append("perception은 dict여야 합니다.")
    else:
        for key in ["detector", "input_resolution", "detections"]:
            if key not in perception:
                errors.append(f"perception 필드 누락: {key}")

        input_resolution = perception.get("input_resolution")
        if not (
            isinstance(input_resolution, list)
            and len(input_resolution) == 2
            and all(_is_number(v) for v in input_resolution)
        ):
            errors.append("perception.input_resolution은 길이 2의 숫자 리스트여야 합니다.")

        detections = perception.get("detections")
        if not isinstance(detections, list):
            errors.append("perception.detections는 리스트여야 합니다.")
        else:
            for idx, det in enumerate(detections):
                if not isinstance(det, dict):
                    errors.append(f"detections[{idx}]는 dict여야 합니다.")
                    continue
                for key in ["label", "score", "bbox_xywh"]:
                    if key not in det:
                        errors.append(f"detections[{idx}] 필드 누락: {key}")
                bbox = det.get("bbox_xywh")
                if not (
                    isinstance(bbox, list)
                    and len(bbox) == 4
                    and all(_is_number(v) for v in bbox)
                ):
                    errors.append(f"detections[{idx}].bbox_xywh는 길이 4의 숫자 리스트여야 합니다.")
                score = det.get("score")
                if not _is_number(score):
                    errors.append(f"detections[{idx}].score는 숫자여야 합니다.")

    perf = data.get("performance_signals", {})
    if not isinstance(perf, dict):
        errors.append("performance_signals는 dict여야 합니다.")
    else:
        for key in ["confidence_trend", "miss_rate_trend", "risk_score", "failure_type"]:
            if key not in perf:
                errors.append(f"performance_signals 필드 누락: {key}")

    xai = data.get("xai_signals", {})
    if not isinstance(xai, dict):
        errors.append("xai_signals는 dict여야 합니다.")
    else:
        for key in ["method", "dominant_factors", "attention_summary"]:
            if key not in xai:
                errors.append(f"xai_signals 필드 누락: {key}")

        dominant_factors = xai.get("dominant_factors")
        if not isinstance(dominant_factors, list):
            errors.append("xai_signals.dominant_factors는 리스트여야 합니다.")
        else:
            for idx, factor in enumerate(dominant_factors):
                if not isinstance(factor, dict):
                    errors.append(f"dominant_factors[{idx}]는 dict여야 합니다.")
                    continue
                for key in ["name", "importance"]:
                    if key not in factor:
                        errors.append(f"dominant_factors[{idx}] 필드 누락: {key}")

    constraints = data.get("scenario_constraints", {})
    if not isinstance(constraints, dict):
        errors.append("scenario_constraints는 dict여야 합니다.")

    if errors:
        raise SchemaValidationError("\n".join(errors))

    return True