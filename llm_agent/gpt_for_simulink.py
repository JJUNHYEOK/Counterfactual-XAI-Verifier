"""CLI adapter so MATLAB / Simulink can call the LLM scenario generator.

Usage (from project root):
    python -m llm_agent.gpt_for_simulink \
        --input  data/xai_input.json \
        --output data/scenario_iter_002.json

The script:
    1) Reads xai_input.json (multi-requirement format produced by build_xai_input.m).
    2) Calls llm_agent.gpt_generator.GPTGenerator.generate_counterfactual.
    3) Normalizes the LLM response to a flat scenario JSON that
       build_mountain_uav_model.m / setup_base_workspace can load.
    4) On any failure (missing API key, network error, parse error) it falls
       back to a deterministic rule-based scenario mutation, so the loop
       in run_counterfactual_loop.m never hangs.

Exit codes:
    0  scenario produced successfully (LLM or fallback)
    2  unrecoverable I/O failure (input not readable / output not writable)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


# -- safe imports (LLM may be unavailable in offline / CI runs) ---------------
try:
    from llm_agent.gpt_generator import GPTGenerator  # type: ignore
    _LLM_IMPORT_OK = True
    _LLM_IMPORT_ERR: str | None = None
except Exception as exc:  # noqa: BLE001
    GPTGenerator = None  # type: ignore[assignment]
    _LLM_IMPORT_OK = False
    _LLM_IMPORT_ERR = f"{type(exc).__name__}: {exc}"


# -- parameter bounds (must mirror build_xai_input.m search_space) ------------
_BOUNDS: dict[str, tuple[float, float]] = {
    "fog_density_percent": (0.0, 100.0),
    "illumination_lux":    (200.0, 20000.0),
    "camera_noise_level":  (0.0, 0.6),
}


def _clip(name: str, value: float) -> float:
    lo, hi = _BOUNDS.get(name, (-1e9, 1e9))
    return max(lo, min(hi, float(value)))


def _read_xai_input(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _current_env(xai_input: dict[str, Any]) -> dict[str, float]:
    env = (
        xai_input.get("scenario", {})
        .get("environment_parameters", {})
        if isinstance(xai_input, dict) else {}
    )
    return {
        "fog_density_percent": float(env.get("fog_density_percent", 0.0)),
        "illumination_lux":    float(env.get("illumination_lux", 8000.0)),
        "camera_noise_level":  float(env.get("camera_noise_level", 0.0)),
    }


def _normalize_llm_output(
    raw: dict[str, Any] | None,
    fallback_env: dict[str, float],
    iter_tag: str,
    xai_input: dict[str, Any],
) -> dict[str, Any]:
    """Coerce any of several LLM response shapes into the flat scenario file."""

    def _flat_env_from_raw() -> dict[str, float] | None:
        if not isinstance(raw, dict):
            return None
        env = (
            raw.get("scenario", {}).get("environment_parameters")
            if isinstance(raw.get("scenario"), dict) else None
        )
        if isinstance(env, dict) and any(k in env for k in _BOUNDS):
            return {k: _clip(k, env.get(k, fallback_env[k])) for k in _BOUNDS}
        env = raw.get("environment_parameters")
        if isinstance(env, dict) and any(k in env for k in _BOUNDS):
            return {k: _clip(k, env.get(k, fallback_env[k])) for k in _BOUNDS}
        return None

    # If LLM output is usable, run it through the bisection state machine too,
    # so the LLM mostly suggests a target neighborhood and we honor it but
    # still update _loop_state for traceability.
    llm_env = _flat_env_from_raw()

    next_env, next_state, decision = _bisection_step(fallback_env, xai_input, llm_env)

    target_hypothesis = ""
    llm_reasoning = ""
    if isinstance(raw, dict):
        target_hypothesis = str(raw.get("target_hypothesis", "")).strip()
        llm_reasoning     = str(raw.get("llm_reasoning", "")).strip()

    return {
        "scenario_id":            f"scenario_{iter_tag}",
        "target_hypothesis":      target_hypothesis or decision["hypothesis"],
        "environment_parameters": next_env,
        "llm_reasoning":          llm_reasoning or decision["reasoning"],
        "_loop_state":            next_state,
    }


# -----------------------------------------------------------------------------
# Bisection-style boundary search
# -----------------------------------------------------------------------------
def _bisection_step(
    cur_env: dict[str, float],
    xai_input: dict[str, Any],
    llm_env: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, Any], dict[str, str]]:
    """Decide the next environment based on the bisection state machine.

    Inputs:
        cur_env   : current scenario's env params
        xai_input : full xai_input dict (contains last requirement results
                    plus prior _loop_state in scenario block)
        llm_env   : env suggested by LLM (may be None)

    State machine (direction = +1 means worsen, -1 means recover):
        last_passed=None  -> direction=+1, step=1.0  (first iter)
        PASS  -> PASS     -> keep direction=+1, step unchanged
        PASS  -> FAIL     -> direction=-1, step *= 0.5  (just crossed)
        FAIL  -> FAIL     -> keep direction=-1, step unchanged
        FAIL  -> PASS     -> direction=+1, step *= 0.5  (re-crossed)
    """
    sc = xai_input.get("scenario") if isinstance(xai_input, dict) else None
    sc = sc if isinstance(sc, dict) else {}
    # MATLAB renames "_loop_state" -> "x_loop_state" when jsondecoding, so
    # accept either spelling on input.
    prev_state = sc.get("_loop_state") or sc.get("x_loop_state")
    if not isinstance(prev_state, dict):
        prev_state = {"direction": 1, "step_factor": 1.0,
                      "last_passed": None, "history": []}

    history = list(prev_state.get("history") or [])
    direction = int(prev_state.get("direction", 1))
    step = float(prev_state.get("step_factor", 1.0))
    last_passed = prev_state.get("last_passed")

    perf = xai_input.get("performance_signals") or {}
    violated = int(perf.get("violated_count", 0))
    cur_passed = (violated == 0)

    history.append({
        "env": dict(cur_env),
        "passed": bool(cur_passed),
        "violated_count": violated,
        "map50":          float(perf.get("map50",                  -1.0)),
        "min_clearance":  float(perf.get("min_clearance_m",        -1.0)),
        "worst_run":      float(perf.get("max_consecutive_misses", -1.0)),
    })

    # State transition
    if last_passed is None:
        direction = +1   # always begin by worsening from the seed PASS state
    elif last_passed and cur_passed:
        direction = +1
    elif last_passed and not cur_passed:
        direction = -1
        step *= 0.5
    elif (not last_passed) and (not cur_passed):
        direction = -1
    else:  # not last_passed and cur_passed
        direction = +1
        step *= 0.5

    # Compute next env
    next_env = _apply_direction(cur_env, direction, step)

    # If LLM proposed something, blend LLM's suggestion with bisection
    # direction (so LLM can fine-tune within the bisection envelope, but the
    # search keeps progressing).
    if llm_env is not None:
        next_env = _blend(next_env, llm_env, weight_llm=0.4)
        next_env = {k: _clip(k, v) for k, v in next_env.items()}

    next_state = {
        "direction": direction,
        "step_factor": max(step, 1e-3),    # floor so we never get stuck
        "last_passed": bool(cur_passed),
        "history": history[-20:],          # cap to 20 most recent
    }

    decision = {
        "hypothesis": (
            f"Bisection step: direction={'WORSEN' if direction>0 else 'RECOVER'}, "
            f"step_factor={step:.3f}, prev={'PASS' if last_passed else ('FAIL' if last_passed is not None else 'INIT')}, "
            f"now={'PASS' if cur_passed else 'FAIL'}"
        ),
        "reasoning": (
            "Boundary search: keep moving in the same direction while the verdict "
            "is unchanged; halve the step when the boundary is crossed."
        ),
    }
    return next_env, next_state, decision


def _apply_direction(env: dict[str, float], direction: int, step: float) -> dict[str, float]:
    # direction: +1 worsen (more fog, less illum, more noise), -1 recover.
    fog_step   = 18.0  * step * direction
    illum_mul  = 1.0   - 0.30 * step * direction
    noise_step = 0.10  * step * direction

    return {
        "fog_density_percent": _clip(
            "fog_density_percent",
            env["fog_density_percent"] + fog_step),
        "illumination_lux":    _clip(
            "illumination_lux",
            env["illumination_lux"] * max(0.05, illum_mul)),
        "camera_noise_level":  _clip(
            "camera_noise_level",
            env["camera_noise_level"] + noise_step),
    }


def _blend(a: dict[str, float], b: dict[str, float], weight_llm: float) -> dict[str, float]:
    w = max(0.0, min(1.0, weight_llm))
    return {k: a[k] * (1 - w) + b.get(k, a[k]) * w for k in a}


def _call_llm(xai_input: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Returns (raw_response, source_tag).
    source_tag is 'llm' or 'fallback:<reason>'."""
    if not _LLM_IMPORT_OK or GPTGenerator is None:
        return None, f"fallback:import_error:{_LLM_IMPORT_ERR}"
    try:
        gen = GPTGenerator()
    except Exception as exc:  # noqa: BLE001
        return None, f"fallback:client_init:{exc}"
    try:
        raw = gen.generate_counterfactual(xai_input)
        if raw is None:
            return None, "fallback:llm_returned_none"
        return raw, "llm"
    except Exception as exc:  # noqa: BLE001
        return None, f"fallback:llm_call_error:{exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-driven counterfactual scenario generator for Simulink loop")
    parser.add_argument("--input",  required=True, help="Path to xai_input.json from MATLAB")
    parser.add_argument("--output", required=True, help="Path where the next scenario JSON is written")
    parser.add_argument("--iter_tag", default="iter_002", help="Scenario id suffix")
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM call, always use rule-based fallback (for offline tests)")
    args = parser.parse_args(argv)

    in_path  = Path(args.input)
    out_path = Path(args.output)

    try:
        xai_input = _read_xai_input(in_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] cannot read input {in_path}: {exc}", file=sys.stderr)
        return 2

    cur_env = _current_env(xai_input)

    if args.no_llm:
        raw = None
        source = "fallback:user_requested"
    else:
        raw, source = _call_llm(xai_input)

    next_scenario = _normalize_llm_output(raw, cur_env, args.iter_tag, xai_input)
    next_scenario["_source"] = source

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(next_scenario, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] cannot write output {out_path}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    e = next_scenario["environment_parameters"]
    print(f"[OK] wrote {out_path} (source={source}) "
          f"fog={e['fog_density_percent']:.1f} "
          f"illum={e['illumination_lux']:.0f} "
          f"noise={e['camera_noise_level']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
