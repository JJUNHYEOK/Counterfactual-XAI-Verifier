from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return target


def build_xai_input_packet(
    scene_id: str,
    scenario: dict[str, Any],
    sim_result: dict[str, Any],
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "scenario": scenario,
        "sim_result": sim_result,
        "eval_result": eval_result,
    }
