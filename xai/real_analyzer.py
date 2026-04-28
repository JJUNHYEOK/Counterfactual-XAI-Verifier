import os
from typing import Any

import numpy as np
from ultralytics import YOLO


def absolute_cleaner(val: Any) -> float:
    try:
        s = str(val).replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
        return float(s)
    except Exception:
        return 0.0


def _is_harmful_when_increasing(feature_name: str) -> bool:
    lowered = feature_name.lower()
    tokens = (
        "fog",
        "noise",
        "blur",
        "wind",
        "delay",
        "obstacle",
        "density",
        "rain",
        "snow",
    )
    return any(token in lowered for token in tokens)


def _feature_scale(feature_name: str, value: float) -> float:
    lowered = feature_name.lower()
    if "percent" in lowered:
        return 100.0
    if "lux" in lowered:
        return 10000.0
    if "noise" in lowered or "density" in lowered:
        return 1.0
    if "blur" in lowered:
        return 20.0
    if "wind" in lowered:
        return 20.0
    if "delay" in lowered:
        return 10.0
    return max(abs(value), 1.0)


class RealXAIAnalyzer:
    """Simulation-grounded analyzer using observed run history only.

    This class does not train surrogate models and does not compute SHAP values.
    """

    def __init__(self):
        self.yolo_model = YOLO("yolo11x.pt")
        self.history: list[dict[str, Any]] = []

    def _extract_environment(self, scenario_data: dict[str, Any]) -> dict[str, float]:
        if not isinstance(scenario_data, dict):
            return {}

        env = scenario_data.get("environment_parameters")
        if not isinstance(env, dict):
            env = scenario_data.get("current_environment")
        if not isinstance(env, dict):
            env = {}

        out: dict[str, float] = {}
        for key, value in env.items():
            out[str(key)] = absolute_cleaner(value)
        return out

    def _save_annotated_image(self, image_path: str, results) -> None:
        try:
            base_name = os.path.basename(image_path)
            annotated_name = base_name.replace("current", "annotated") if "current" in base_name else f"annotated_{base_name}"
            save_dir = os.path.dirname(image_path)
            annotated_path = os.path.join(save_dir, annotated_name)
            results[0].save(filename=annotated_path)
        except Exception as exc:
            print(f"[XAI] Annotated image save failed: {exc}")

    def _build_feature_importance(self, current_env: dict[str, float], current_map50: float) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []

        if self.history:
            prev = self.history[-1]
            prev_env = prev.get("env", {}) if isinstance(prev.get("env"), dict) else {}
            prev_map50 = float(prev.get("map50", current_map50))
            map_drop = max(0.0, prev_map50 - current_map50)

            feature_names = sorted(set(prev_env) | set(current_env))
            for feature in feature_names:
                prev_val = float(prev_env.get(feature, 0.0))
                curr_val = float(current_env.get(feature, prev_val))
                delta = curr_val - prev_val
                if abs(delta) <= 1e-12:
                    continue

                scale = _feature_scale(feature, curr_val)
                delta_norm = abs(delta) / max(scale, 1e-9)

                if _is_harmful_when_increasing(feature):
                    alignment = 1.5 if delta > 0.0 else 0.7
                else:
                    alignment = 1.0

                score = delta_norm * (1.0 + 5.0 * map_drop) * alignment
                rows.append({"name": feature, "score": score})

        if not rows:
            for feature, value in current_env.items():
                scale = _feature_scale(feature, value)
                severity = abs(value) / max(scale, 1e-9)
                rows.append({"name": feature, "score": severity})

        if not rows:
            return [{"name": "no_environment_signal", "importance": 1.0}]

        total = sum(max(row["score"], 0.0) for row in rows)
        if total <= 1e-12:
            importance = 1.0 / len(rows)
            normalized = [{"name": row["name"], "importance": importance} for row in rows]
        else:
            normalized = [
                {
                    "name": row["name"],
                    "importance": max(row["score"], 0.0) / total,
                }
                for row in rows
            ]

        normalized.sort(key=lambda row: row["importance"], reverse=True)
        return [{"name": row["name"], "importance": float(round(row["importance"], 6))} for row in normalized[:8]]

    def analyze(self, image_path: str, scenario_data: dict[str, Any]):
        results = self.yolo_model(image_path, verbose=False)
        self._save_annotated_image(image_path, results)

        boxes = results[0].boxes
        current_map50 = float(np.mean(boxes.conf.cpu().numpy())) if len(boxes) > 0 else 0.10

        current_env = self._extract_environment(scenario_data)
        feature_importance = self._build_feature_importance(current_env=current_env, current_map50=current_map50)

        self.history.append({"env": current_env, "map50": current_map50})

        return current_map50, feature_importance
