"""Single-shot decision helper exposed to the MATLAB dashboard.

Self-contained on purpose: top-level imports must not depend on the
``dspy_pipeline`` package because matlab_bridge / shap_analyzer / dspy
itself can fail to import in a fresh MATLAB-spawned Python interpreter
(missing optional deps, no MATLAB engine, no LLM key, etc). The boundary
policy used in 90 % of decisions is pure arithmetic and runs without any
of those dependencies. Only the LLM-explore branch (used when no FAIL
anchor has been found yet) lazy-imports DSPy, and falls back cleanly if
that import fails.

Returns a plain dict with keys:
  fog_density_percent, illumination_lux, camera_noise_level, mode, analysis

The MATLAB dashboard converts the result with ``struct(pyResult)``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────
# Pure-Python boundary policy (mirrors dspy_pipeline.orchestrator's
# _bisect_between / _rule_push / _rule_relax — kept inline so this module
# imports successfully even when the dspy_pipeline tree has broken deps)
# ──────────────────────────────────────────────────────────────────────────

def _bisect(a: dict, b: dict, weight: float = 0.65) -> dict:
    keys = set(a) | set(b)
    return {
        k: round((1.0 - weight) * float(a.get(k, 0)) + weight * float(b.get(k, 0)), 4)
        for k in keys
    }


def _rule_push(env: dict) -> dict:
    fog   = float(env.get("fog_density_percent", 30))
    illum = float(env.get("illumination_lux",    4000))
    noise = float(env.get("camera_noise_level",  0.1))
    return {
        "fog_density_percent": round(min(100,   fog   + 30),   2),
        "illumination_lux":    round(max(200,   illum * 0.50), 1),
        "camera_noise_level":  round(min(0.60,  noise + 0.20), 4),
    }


def _rule_relax(env: dict) -> dict:
    fog   = float(env.get("fog_density_percent", 30))
    illum = float(env.get("illumination_lux",    4000))
    noise = float(env.get("camera_noise_level",  0.1))
    return {
        "fog_density_percent": round(max(0,     fog   - 25),   2),
        "illumination_lux":    round(min(20000, illum * 1.80), 1),
        "camera_noise_level":  round(max(0,     noise - 0.18), 4),
    }


def _envelope(env: dict, mode: str, analysis: str) -> dict:
    return {
        "fog_density_percent": float(env["fog_density_percent"]),
        "illumination_lux":    float(env["illumination_lux"]),
        "camera_noise_level":  float(env["camera_noise_level"]),
        "mode":                mode,
        "analysis":            analysis,
    }


def _seed() -> dict:
    return _envelope(
        {"fog_density_percent": 30.0, "illumination_lux": 4000.0, "camera_noise_level": 0.1},
        "seed",
        "Initial seed (no history yet)",
    )


def _records_to_anchors(records):
    last_pass = None
    last_fail = None
    for r in records:
        env = {
            "fog_density_percent": float(r["fog"]),
            "illumination_lux":    float(r["ill"]),
            "camera_noise_level":  float(r["noi"]),
        }
        if r.get("passed"):
            last_pass = env
        else:
            last_fail = env
    return last_pass, last_fail


# ──────────────────────────────────────────────────────────────────────────
# Optional DSPy LLM-explore branch (lazy — never imports at module load)
# ──────────────────────────────────────────────────────────────────────────

_DSPY_TRIED = False
_DSPY_LM    = None


def _try_dspy_explore(records, last_env, last_record):
    """Try to use dspy_pipeline's LLM-explore. Returns env dict on success,
    None if anything goes wrong (missing deps, no API key, runtime error)."""
    global _DSPY_TRIED, _DSPY_LM
    try:
        if not _DSPY_TRIED:
            _DSPY_TRIED = True
            from dspy_pipeline.orchestrator import _build_lm
            _DSPY_LM = _build_lm(None)

        if _DSPY_LM is None:
            return None

        import dspy
        from dspy_pipeline.modules       import AdversarialScenarioGenerator
        from dspy_pipeline.orchestrator  import _build_xai_signals, _build_perf_signals

        # Build orchestrator-shaped record from our last record
        last_orc = {
            "step":                len(records),
            "fog_density_percent": last_env["fog_density_percent"],
            "illumination_lux":    last_env["illumination_lux"],
            "camera_noise_level":  last_env["camera_noise_level"],
            "map50":               float(last_record.get("f1", 0.0)),
            "all_passed":          bool(last_record.get("passed", False)),
            "violated_count":      0 if last_record.get("passed", False) else 1,
            "min_clearance":       float("nan"),
            "worst_run":           0,
        }
        xai  = _build_xai_signals(last_orc, None)
        perf = _build_perf_signals(last_orc)
        hist = []
        for i, r in enumerate(records[-5:]):
            hist.append({
                "iter":                i + 1,
                "fog_density_percent": float(r["fog"]),
                "illumination_lux":    float(r["ill"]),
                "camera_noise_level":  float(r["noi"]),
                "map50":               round(float(r.get("f1", 0.0)), 4),
                "all_passed":          bool(r.get("passed", False)),
                "violated_count":      0 if r.get("passed", False) else 1,
            })

        gen = AdversarialScenarioGenerator()
        with dspy.context(lm=_DSPY_LM):
            pred = gen(
                iteration_history   = json.dumps(hist),
                xai_analysis        = json.dumps(xai),
                current_performance = json.dumps(perf),
            )
        return pred.environment_parameters
    except Exception as exc:                          # noqa: BLE001
        print(f"[dashboard_step] DSPy explore unavailable: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def next_case_from_history(history_json: str, model_dir: str = ".") -> dict:
    """Decide the next counterfactual case from the MATLAB dashboard's
    run history. Returns a dict (see module docstring)."""
    try:
        records = json.loads(history_json) if history_json else []
        if not isinstance(records, list):
            records = []
    except Exception as exc:                          # noqa: BLE001
        print(f"[dashboard_step] bad history JSON: {exc}")
        return _seed()

    if not records:
        return _seed()

    last_pass, last_fail = _records_to_anchors(records)
    last      = records[-1]
    last_env  = {
        "fog_density_percent": float(last["fog"]),
        "illumination_lux":    float(last["ill"]),
        "camera_noise_level":  float(last["noi"]),
    }
    last_passed = bool(last.get("passed", False))

    # PASS + FAIL anchor → push toward boundary
    if last_passed and last_fail is not None:
        return _envelope(_bisect(last_env, last_fail),
                         "boundary_push",
                         "PASS → bisect 65% toward FAIL anchor")

    # FAIL + PASS anchor → recover toward boundary
    if (not last_passed) and last_pass is not None:
        return _envelope(_bisect(last_env, last_pass),
                         "boundary_recover",
                         "FAIL → bisect 65% toward PASS anchor")

    # PASS but no FAIL anchor → DSPy LLM-explore (lazy import + fallback)
    if last_passed:
        explored = _try_dspy_explore(records, last_env, last)
        if explored is not None:
            return _envelope(explored, "llm_explore",
                             "DSPy LLM explore (no FAIL anchor yet)")
        return _envelope(_rule_push(last_env), "rule_push",
                         "Rule-based push (LLM unavailable / not configured)")

    # FAIL with no PASS anchor → relax toward baseline
    return _envelope(_rule_relax(last_env), "rule_relax",
                     "Rule-based relax (no PASS anchor)")


# Smoke test
if __name__ == "__main__":
    demo = [
        {"fog": 10, "ill": 8000, "noi": 0.05, "f1": 0.92, "passed": True},
        {"fog": 40, "ill": 4000, "noi": 0.20, "f1": 0.88, "passed": True},
        {"fog": 70, "ill": 2000, "noi": 0.40, "f1": 0.31, "passed": False},
    ]
    print(json.dumps(next_case_from_history(json.dumps(demo)),
                     indent=2, ensure_ascii=False))
