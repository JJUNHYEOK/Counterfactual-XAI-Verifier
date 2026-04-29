"""Build DSPy training dataset from existing Simulink iteration artifacts.

Reads the data/ directory produced by run_counterfactual_loop.m and converts
each xai_input_iter_NNN.json into a dspy.Example with three input fields:

  iteration_history   — JSON array of the last 5 iterations' env + metrics
  xai_analysis        — JSON of xai_signals (dominant_factors, attention_summary)
  current_performance — JSON of performance_signals (map50, clearance, runs, …)

These examples are used by BootstrapFewShot / MIPROv2 to discover which
reasoning patterns produce the most adversarial scenarios.

Public API:
  load_training_examples(data_dir)  → list[dspy.Example]
  build_inputs_from_xai(xai_data, data_dir) → dict[str, str]   (for inference)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import dspy


# ─────────────────────────────────────────────────────────────────────────────
# Training set loader
# ─────────────────────────────────────────────────────────────────────────────

def load_training_examples(
    data_dir: str | Path,
    max_examples: int = 50,
) -> list[dspy.Example]:
    """Load all available xai_input_iter_*.json files as DSPy training examples.

    Args:
        data_dir:     Path to the data/ directory.
        max_examples: Hard cap on the number of examples returned.

    Returns:
        List of dspy.Example objects with input fields declared via
        `.with_inputs(...)`.  No gold-label output fields are set —
        BootstrapFewShot will derive them by running the student module
        and scoring with the Simulink metric.
    """
    data_path = Path(data_dir)
    examples: list[dspy.Example] = []

    for i in range(1, max_examples + 1):
        xai_path = data_path / f"xai_input_iter_{i:03d}.json"
        if not xai_path.exists():
            break

        try:
            xai_data = json.loads(xai_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Dataset] Warning: skipping {xai_path.name}: {exc}")
            continue

        inputs = _extract_inputs(xai_data, data_path, current_iter=i)
        example = dspy.Example(**inputs).with_inputs(
            "iteration_history", "xai_analysis", "current_performance"
        )
        examples.append(example)

    print(f"[Dataset] Loaded {len(examples)} training examples from {data_path}")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Single-example builder (used at inference time by dspy_for_simulink.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs_from_xai(
    xai_data: dict,
    data_dir: str | Path,
) -> dict[str, str]:
    """Convert a live xai_input dict into DSPy input fields.

    Called per-iteration by dspy_for_simulink.py (the MATLAB CLI adapter).

    Args:
        xai_data: Parsed content of xai_input.json produced by build_xai_input.m.
        data_dir: Path to data/ directory for reading historical eval/scenario files.

    Returns:
        dict with keys iteration_history, xai_analysis, current_performance.
    """
    current_iter = _extract_iter_number(str(xai_data.get("scene_id", "")))
    return _extract_inputs(xai_data, Path(data_dir), current_iter=current_iter)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_inputs(
    xai_data: dict,
    data_path: Path,
    current_iter: int,
) -> dict[str, str]:
    """Build the three DSPy input strings from an xai_input dict."""
    iteration_history = _build_iteration_history(data_path, current_iter, window=5)
    xai_analysis      = json.dumps(xai_data.get("xai_signals", {}), ensure_ascii=False)
    current_perf      = json.dumps(xai_data.get("performance_signals", {}), ensure_ascii=False)
    return {
        "iteration_history":  iteration_history,
        "xai_analysis":       xai_analysis,
        "current_performance": current_perf,
    }


def _extract_iter_number(scene_id: str) -> int:
    """Parse iteration number from strings like 'iter_003' or 'scenario_003'. """
    m = re.search(r"(\d+)", scene_id)
    return int(m.group(1)) if m else 1


def _build_iteration_history(
    data_path: Path,
    current_iter: int,
    window: int = 5,
) -> str:
    """Build a compact JSON array of the most recent simulation iterations.

    Looks back up to `window` steps before current_iter.
    Each entry contains the env params and the key metrics so the LLM can
    reason about trend direction (converging toward / diverging from failure).
    """
    history: list[dict] = []
    start = max(1, current_iter - window)

    for j in range(start, current_iter):
        eval_path = data_path / f"eval_iter_{j:03d}.json"
        scen_path = data_path / f"scenario_iter_{j:03d}.json"

        if not eval_path.exists() or not scen_path.exists():
            # Also try dspy_* variants written by run_dspy_adversarial.py
            eval_path = data_path / f"dspy_eval_iter_{j:03d}.json"
            scen_path = data_path / f"dspy_scenario_iter_{j:03d}.json"
            if not eval_path.exists() or not scen_path.exists():
                continue

        try:
            eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
            scen_data = json.loads(scen_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Prefer image-based eval results (more realistic); fall back to geometric
        img = eval_data.get("image_based", eval_data.get("geometric", eval_data))
        env = scen_data.get("environment_parameters", {})

        history.append({
            "iter":                   j,
            "fog_density_percent":    round(float(env.get("fog_density_percent", 0)),   1),
            "illumination_lux":       round(float(env.get("illumination_lux",    8000)), 0),
            "camera_noise_level":     round(float(env.get("camera_noise_level",  0)),   3),
            "map50":                  round(float(img.get("req1", img).get("value", -1) if isinstance(img.get("req1"), dict) else img.get("map50", -1)), 4),
            "min_clearance_m":        round(float(img.get("req2", img).get("value", -1) if isinstance(img.get("req2"), dict) else img.get("min_clearance", -1)), 3),
            "max_consecutive_misses": int(img.get("req3", img).get("value", -1) if isinstance(img.get("req3"), dict) else img.get("worst_run", -1)),
            "all_passed":             bool(img.get("all_passed", True)),
            "violated_count":         int(img.get("violated_count", 0)),
        })

    return json.dumps(history, ensure_ascii=False)
