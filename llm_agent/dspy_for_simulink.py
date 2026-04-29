"""dspy_for_simulink.py — DSPy-based CLI adapter for the MATLAB/Simulink loop.

Drop-in replacement for llm_agent.gpt_for_simulink that uses a DSPy
ChainOfThought module (optionally pre-compiled with BootstrapFewShot /
MIPROv2) instead of direct OpenAI API calls.

To switch the MATLAB loop to DSPy, change run_counterfactual_loop.m line:

    OLD:  python -m llm_agent.gpt_for_simulink  ...
    NEW:  python -m llm_agent.dspy_for_simulink ...

All other CLI flags (--input, --output, --iter_tag, --no_llm) are identical.

Exit codes:
    0  scenario written successfully (DSPy or rule-based fallback)
    2  unrecoverable I/O error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent          # project root
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

# Parameter bounds — must mirror _BOUNDS in gpt_for_simulink.py
_BOUNDS: dict[str, tuple[float, float]] = {
    "fog_density_percent": (0.0,   100.0),
    "illumination_lux":    (200.0, 20000.0),
    "camera_noise_level":  (0.0,   0.6),
}


def _clip(name: str, value: float) -> float:
    lo, hi = _BOUNDS.get(name, (-1e9, 1e9))
    return max(lo, min(hi, float(value)))


# ─────────────────────────────────────────────────────────────────────────────
# DSPy inference
# ─────────────────────────────────────────────────────────────────────────────

def _run_dspy(
    xai_input: dict,
    model_dir: Path,
) -> tuple[dict[str, float], str, str]:
    """Run the DSPy module and return (env_params, reasoning, source_tag).

    Tries to load a pre-compiled program from
    dspy_pipeline/compiled_program.json; falls back to the raw
    (un-optimised) module if the file is not found.
    """
    import dspy
    from dspy_pipeline.modules  import AdversarialScenarioGenerator
    from dspy_pipeline.dataset  import build_inputs_from_xai

    # Configure LM
    openai_key    = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if openai_key:
        lm = dspy.LM(
            "openai/gpt-4o-mini",
            api_key=openai_key,
            temperature=0.4,
            max_tokens=1024,
        )
    elif anthropic_key:
        lm = dspy.LM(
            "anthropic/claude-haiku-4-5-20251001",
            api_key=anthropic_key,
            temperature=0.4,
            max_tokens=1024,
        )
    else:
        raise EnvironmentError(
            "No LLM API key. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )

    dspy.configure(lm=lm)

    # Load or create generator
    compiled_path = model_dir / "dspy_pipeline" / "compiled_program.json"
    generator     = AdversarialScenarioGenerator()
    source: str

    if compiled_path.exists():
        try:
            generator.load(str(compiled_path))
            source = "dspy:compiled"
        except Exception:
            source = "dspy:raw"
    else:
        source = "dspy:raw"

    # Build DSPy input fields from the xai_input dict
    data_dir = model_dir / "data"
    inputs   = build_inputs_from_xai(xai_input, data_dir)

    prediction = generator(**inputs)

    env_params = {k: _clip(k, v)
                  for k, v in prediction.environment_parameters.items()}
    reasoning  = (
        getattr(prediction, "analysis",   "") or
        getattr(prediction, "reasoning",  "")
    )
    return env_params, reasoning, source


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback (mirrors gpt_for_simulink.py)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_mutation(xai_input: dict) -> tuple[dict[str, float], str]:
    """Deterministic boundary-search mutation when DSPy is unavailable."""
    env = {}
    if isinstance(xai_input, dict):
        env = xai_input.get("scenario", {}).get("environment_parameters", {})

    fog   = float(env.get("fog_density_percent", 30.0))
    illum = float(env.get("illumination_lux",    4000.0))
    noise = float(env.get("camera_noise_level",  0.1))

    perf        = xai_input.get("performance_signals", {}) if isinstance(xai_input, dict) else {}
    last_passed = int(perf.get("violated_count", 1)) == 0

    if last_passed:                     # worsen → try to cross the failure boundary
        fog   = min(100.0, fog   + 18.0)
        illum = max(200.0, illum *  0.70)
        noise = min(0.6,   noise +  0.10)
    else:                               # recover slightly → locate exact boundary
        fog   = max(0.0,     fog   -  9.0)
        illum = min(20000.0, illum *  1.25)
        noise = max(0.0,     noise -  0.05)

    result = {
        "fog_density_percent": _clip("fog_density_percent", fog),
        "illumination_lux":    _clip("illumination_lux",    illum),
        "camera_noise_level":  _clip("camera_noise_level",  noise),
    }
    reasoning = (
        "Rule-based mutation applied (DSPy unavailable). "
        f"Direction: {'WORSEN' if last_passed else 'RECOVER'}."
    )
    return result, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Bisection state pass-through
# ─────────────────────────────────────────────────────────────────────────────

def _carry_loop_state(
    xai_input: dict,
    env_params: dict[str, float],
) -> dict:
    """Preserve the bisection _loop_state from the incoming xai_input.

    This keeps run_counterfactual_loop.m's convergence check working even
    when the DSPy adapter is driving scenario generation.
    """
    sc = xai_input.get("scenario", {}) if isinstance(xai_input, dict) else {}
    prev_state = sc.get("_loop_state") or sc.get("x_loop_state")
    if not isinstance(prev_state, dict):
        prev_state = {
            "direction":   1,
            "step_factor": 1.0,
            "last_passed": None,
            "history":     [],
        }

    perf          = xai_input.get("performance_signals", {}) if isinstance(xai_input, dict) else {}
    violated      = int(perf.get("violated_count", 0))
    cur_passed    = violated == 0
    last_passed   = prev_state.get("last_passed")
    step          = float(prev_state.get("step_factor", 1.0))
    direction     = int(prev_state.get("direction", 1))

    if last_passed is None:
        direction = 1
    elif last_passed and cur_passed:
        direction = 1
    elif last_passed and not cur_passed:
        direction = -1; step *= 0.5
    elif not last_passed and not cur_passed:
        direction = -1
    else:
        direction = 1; step *= 0.5

    history = list(prev_state.get("history") or [])
    history.append({
        "env":             env_params,
        "passed":          cur_passed,
        "violated_count":  violated,
        "map50":           float(perf.get("map50", -1.0)),
    })

    return {
        "direction":   direction,
        "step_factor": max(step, 1e-3),
        "last_passed": cur_passed,
        "history":     history[-20:],
    }


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DSPy-based scenario generator CLI for the MATLAB Simulink loop"
    )
    parser.add_argument("--input",    required=True,
                        help="Path to xai_input.json (written by build_xai_input.m)")
    parser.add_argument("--output",   required=True,
                        help="Path where the next scenario JSON will be written")
    parser.add_argument("--iter_tag", default="iter_002",
                        help="Scenario ID suffix, e.g. iter_003")
    parser.add_argument("--no_llm",   action="store_true",
                        help="Skip DSPy/LLM; always use rule-based fallback")
    args = parser.parse_args(argv)

    in_path  = Path(args.input)
    out_path = Path(args.output)

    # ── Read xai_input.json ──────────────────────────────────────────────
    try:
        xai_input = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERROR] Cannot read {in_path}: {exc}", file=sys.stderr)
        return 2

    # ── Generate next environment params ─────────────────────────────────
    reasoning: str
    source:    str

    if args.no_llm:
        env_params, reasoning = _rule_mutation(xai_input)
        source = "fallback:no_llm"
    else:
        try:
            env_params, reasoning, source = _run_dspy(xai_input, ROOT)
        except Exception as exc:
            print(f"[WARN] DSPy failed ({type(exc).__name__}: {exc}). "
                  "Falling back to rule-based mutation.", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            env_params, reasoning = _rule_mutation(xai_input)
            source = f"fallback:{type(exc).__name__}"

    # ── Build scenario JSON (MATLAB-compatible) ──────────────────────────
    loop_state = _carry_loop_state(xai_input, env_params)

    scenario = {
        "scenario_id":            f"scenario_{args.iter_tag}",
        "target_hypothesis":      (
            "DSPy-optimised adversarial scenario targeting UAV detection failure"
        ),
        "environment_parameters": env_params,
        "llm_reasoning":          reasoning,
        "_loop_state":            loop_state,   # bisection state for run_counterfactual_loop.m
        "_source":                source,
        "_generator":             "dspy_for_simulink",
    }

    # ── Write output ─────────────────────────────────────────────────────
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(scenario, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[ERROR] Cannot write {out_path}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 2

    e = env_params
    print(
        f"[OK] wrote {out_path} (source={source})  "
        f"fog={e['fog_density_percent']:.1f}  "
        f"illum={e['illumination_lux']:.0f}  "
        f"noise={e['camera_noise_level']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
