from typing import Any


def analyze_xai_dummy(sim_log: dict[str, Any]) -> dict[str, float]:
    """Return fixed dummy XAI importance for Week 1 integration tests."""
    if not isinstance(sim_log, dict):
        sim_log = {}

    # Read agreed keys so upstream/downstream schema handoff is exercised.
    sim_log.get("status", "unknown")
    sim_log.get("min_distance")
    sim_log.get("wind_speed")

    return {"wind_speed_importance": 0.68}


def get_example_sim_log() -> dict[str, float | str]:
    return {
        "status": "success",
        "min_distance": 2.4,
        "wind_speed": 4.0,
    }
