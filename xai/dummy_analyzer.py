from typing import Any


def _normalize_importance(raw_scores: dict[str, float]) -> dict[str, float]:
    total = sum(raw_scores.values())
    if total <= 0:
        return {"wind_speed": 0.34, "delay": 0.33, "obstacle_density": 0.33}

    normalized = {k: v / total for k, v in raw_scores.items()}
    rounded = {k: round(v, 2) for k, v in normalized.items()}
    diff = round(1.0 - sum(rounded.values()), 2)
    if diff != 0:
        max_key = max(rounded, key=rounded.get)
        rounded[max_key] = round(rounded[max_key] + diff, 2)
    return rounded


def analyze_xai_dummy(sim_log: dict[str, Any]) -> dict[str, Any]:
    """Return a lightweight, deterministic XAI summary based on simulator output."""
    if not isinstance(sim_log, dict):
        sim_log = {}

    scenario_id = str(sim_log.get("scenario_id", "sim_run_unknown"))
    status = str(sim_log.get("status", "UNKNOWN")).upper()
    min_distance = float(sim_log.get("min_distance", 0.0))
    wind_speed = float(sim_log.get("wind_speed", 0.0))
    delay = float(sim_log.get("delay", 0.0))
    obstacle_density = float(sim_log.get("obstacle_density", 0.0))
    collision = bool(sim_log.get("collision", False))
    mission_completed = bool(sim_log.get("mission_completed", False))

    wind_score = 0.50 * min(max(wind_speed / 10.0, 0.0), 1.0)
    delay_score = 0.30 * min(max(delay / 5.0, 0.0), 1.0)
    obstacle_score = 0.20 * min(max(obstacle_density, 0.0), 1.0)

    feature_importance = _normalize_importance(
        {
            "wind_speed": wind_score,
            "delay": delay_score,
            "obstacle_density": obstacle_score,
        }
    )

    if status in {"FAIL", "FAILURE"} or collision or not mission_completed or min_distance < 1.0:
        failure_risk = "HIGH"
    elif min_distance < 1.8 or (wind_score + delay_score + obstacle_score) >= 0.45:
        failure_risk = "MEDIUM"
    else:
        failure_risk = "LOW"

    top_feature = max(feature_importance, key=feature_importance.get)
    insight_map = {
        "wind_speed": "High wind speed contributed most to the drop in minimum separation distance.",
        "delay": "Operational delay contributed most to tighter safety margins in this run.",
        "obstacle_density": "Dense obstacles contributed most to reduced navigation safety margin.",
    }

    return {
        "scenario_id": scenario_id,
        "status": status,
        "xai_analysis": {
            "failure_risk": failure_risk,
            "feature_importance": feature_importance,
            "insight": insight_map[top_feature],
        },
    }


def get_example_sim_log() -> dict[str, float | str]:
    return {
        "scenario_id": "sim_run_example",
        "status": "SUCCESS",
        "min_distance": 1.72,
        "wind_speed": 6.1,
        "delay": 1.8,
        "obstacle_density": 0.45,
        "collision": False,
        "mission_completed": True,
        "message": "Mission completed with narrow safety margin.",
    }
