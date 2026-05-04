"""Single-step orchestrator for the boundary-search loop.

Exposes a small API that both the CLI script (run_dspy_adversarial.py)
and the Streamlit demo (app_demo.py) can use, so the loop logic is
implemented once and shared:

    sess = init_session(model_dir='.', sim_mode='engine')
    step1 = run_one_step(sess, current_scenario={'environment_parameters': seed_env},
                         viz_png_path='assets/iter_001.png')
    next_scenario = decide_next_scenario(sess, viz_png_path='assets/iter_002.png')
    step2 = run_one_step(sess, next_scenario, ...)
    ...

The session object holds: bridge, generator, history, last_pass_env,
last_fail_env, boundary_log, shap_signals (latest), and counters.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import dspy

from dspy_pipeline.matlab_bridge import MatlabSimulinkBridge, SimulationResult
from dspy_pipeline.metric        import init_metric_bridge
from dspy_pipeline.modules       import AdversarialScenarioGenerator
from dspy_pipeline.shap_analyzer import compute_shap_signals, is_available as shap_available


# ─────────────────────────────────────────────────────────────────────────────
# Session container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    bridge:        MatlabSimulinkBridge
    generator:     AdversarialScenarioGenerator
    data_dir:      Path
    history:       list[dict]            = field(default_factory=list)
    last_pass_env: dict | None           = None
    last_fail_env: dict | None           = None
    boundary_log:  list[dict]            = field(default_factory=list)
    last_shap:    dict | None            = None
    sim_mode:     str                    = "engine"


# ─────────────────────────────────────────────────────────────────────────────
# Session lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def init_session(
    model_dir:       str,
    sim_mode:        str = "engine",
    model_override:  str | None = None,
    compiled_path:   str | None = None,
) -> Session:
    """Boot the LM, MATLAB bridge, and DSPy module. Returns a Session handle."""
    _configure_lm(model_override)

    bridge = init_metric_bridge(model_dir, mode=sim_mode)

    generator = AdversarialScenarioGenerator()
    if compiled_path and Path(compiled_path).exists():
        generator.load(str(compiled_path))
        print(f"[Orchestrator] Loaded compiled program ← {compiled_path}")

    return Session(
        bridge=bridge,
        generator=generator,
        data_dir=Path(model_dir) / "data",
        sim_mode=sim_mode,
    )


def _configure_lm(model_override: str | None) -> None:
    openai_key    = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if model_override:
        api_key = anthropic_key if model_override.startswith("anthropic/") else openai_key
        model   = model_override
    elif openai_key:
        model, api_key = "openai/gpt-4o-mini", openai_key
    elif anthropic_key:
        model, api_key = "anthropic/claude-haiku-4-5-20251001", anthropic_key
    else:
        raise EnvironmentError("No LLM API key found in env. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
    dspy.configure(lm=dspy.LM(model, api_key=api_key, temperature=0.4, max_tokens=1024))
    print(f"[Orchestrator] LM configured: {model}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-step API
# ─────────────────────────────────────────────────────────────────────────────

def run_one_step(
    session:          Session,
    current_scenario: dict,
    viz_png_path:     str | None = None,
) -> dict:
    """Run one simulation step and append to history. Returns a step record."""
    env_params = current_scenario.get("environment_parameters", {})
    step       = len(session.history) + 1

    t0 = time.time()
    if session.sim_mode == "engine" and viz_png_path:
        result = session.bridge.run_simulation(env_params, viz_png_path=viz_png_path)
    else:
        result = session.bridge.run_simulation(env_params)
    elapsed = time.time() - t0

    # Update boundary anchors
    if result.all_passed:
        session.last_pass_env = dict(env_params)
    else:
        session.last_fail_env = dict(env_params)

    # Append to history (flat keys help SHAP)
    record = {
        "step":                step,
        "env":                 env_params,
        "fog_density_percent": env_params.get("fog_density_percent", 0),
        "illumination_lux":    env_params.get("illumination_lux", 8000),
        "camera_noise_level":  env_params.get("camera_noise_level", 0),
        "map50":               result.map50,
        "min_clearance":       result.min_clearance,
        "worst_run":           result.worst_run,
        "all_passed":          result.all_passed,
        "violated_count":      result.violated_count,
        "elapsed_s":           round(elapsed, 2),
        "viz_png":             viz_png_path,
        "scenario_id":         current_scenario.get("scenario_id", f"step_{step:03d}"),
        "decision_mode":       current_scenario.get("decision_mode", "seed"),
        "target_hypothesis":   current_scenario.get("target_hypothesis", ""),
        "dspy_analysis":       current_scenario.get("dspy_analysis", ""),
    }
    session.history.append(record)

    # Persist artifact
    eval_artifact = {
        "step":       step,
        "env_params": env_params,
        **result.to_dict(),
        "sim_mode":   session.sim_mode,
        "viz_png":    viz_png_path,
    }
    session.data_dir.mkdir(exist_ok=True)
    (session.data_dir / f"dspy_eval_iter_{step:03d}.json").write_text(
        json.dumps(eval_artifact, indent=2), encoding="utf-8",
    )
    return record


def decide_next_scenario(session: Session) -> dict:
    """Pick the next scenario using an explicit push-on-PASS / recover-on-FAIL policy.

    Goal: locate the failure boundary by oscillating around it, NOT by
    monotonically degrading. The policy is:

        last result = PASS → degrade ("push" — make conditions harsher)
                              · if no FAIL yet     → LLM/SHAP exploration
                              · if FAIL anchor seen → bisect toward FAIL anchor

        last result = FAIL → recover ("recover" — make conditions easier)
                              · if PASS anchor seen → bisect toward PASS anchor
                              · if no PASS yet     → rule-based relaxation

    Each PASS-FAIL transition halves the boundary segment, so the swing
    amplitude shrinks geometrically — this finds the boundary precisely
    while never just "degrading further" after a FAIL.
    """
    if not session.history:
        raise RuntimeError("decide_next_scenario called with empty history")

    last       = session.history[-1]
    env_params = last["env"]
    step_next  = last["step"] + 1
    last_passed = bool(last["all_passed"])

    # SHAP from accumulated samples (>= 3 needed)
    shap_signals = compute_shap_signals(session.history, env_params)
    shap_payload = shap_signals.to_payload() if shap_signals else None
    session.last_shap = shap_payload

    has_pass_anchor = session.last_pass_env is not None
    has_fail_anchor = session.last_fail_env is not None

    # ── Decision tree (explicit push / recover) ─────────────────────────
    if last_passed and has_fail_anchor:
        # PUSH: PASS observed and we know where failure lies → step toward it.
        next_env   = _bisect_between(env_params, session.last_fail_env)
        width      = _segment_width(session.last_pass_env, session.last_fail_env)
        mode       = "boundary_push"
        hypothesis = (
            f"직전 시뮬은 PASS. 알려진 FAIL 지점({_brief_env(session.last_fail_env)}) "
            f"방향으로 한 단계 push 하여 경계를 좁힙니다."
        )
        analysis = (
            "📊 PUSH 모드 — PASS이므로 더 가혹하게.\n"
            f"  현재 PASS:  {_brief_env(env_params)}\n"
            f"  목표 FAIL:  {_brief_env(session.last_fail_env)}\n"
            f"  다음 후보:  {_brief_env(next_env)}\n"
            f"  segment width 추적: {width}"
        )
        _log_boundary(session, step_next, mode, next_env, width)
        print(f"  [Decision] PUSH (PASS→가혹화) toward known FAIL anchor")

    elif (not last_passed) and has_pass_anchor:
        # RECOVER: FAIL observed and we know where pass lies → step back.
        next_env   = _bisect_between(env_params, session.last_pass_env)
        width      = _segment_width(session.last_pass_env, session.last_fail_env)
        mode       = "boundary_recover"
        hypothesis = (
            f"직전 시뮬은 FAIL. 알려진 PASS 지점({_brief_env(session.last_pass_env)}) "
            f"방향으로 회복하여 경계를 좁힙니다 (악화가 아닌 완화)."
        )
        analysis = (
            "📊 RECOVER 모드 — FAIL이므로 다시 완화.\n"
            f"  현재 FAIL:  {_brief_env(env_params)}\n"
            f"  목표 PASS:  {_brief_env(session.last_pass_env)}\n"
            f"  다음 후보:  {_brief_env(next_env)}\n"
            f"  segment width 추적: {width}"
        )
        _log_boundary(session, step_next, mode, next_env, width)
        print(f"  [Decision] RECOVER (FAIL→회복) toward known PASS anchor")

    elif last_passed and not has_fail_anchor:
        # EXPLORE: still all PASS. Use LLM/SHAP to push intelligently.
        next_env, hypothesis, analysis, mode = _llm_explore(session, last, shap_payload)
        print(f"  [Decision] EXPLORE (PASS만 있음, FAIL 미발견) → LLM/SHAP push")

    else:
        # last_passed == False and no PASS anchor: never passed yet.
        # Recover by rule-based relaxation toward baseline.
        next_env   = _rule_relax(env_params)
        hypothesis = "PASS 시드를 찾지 못함. 규칙 기반으로 환경을 baseline 쪽으로 완화."
        analysis   = (
            "📊 RULE_RELAX — PASS 앵커 부재.\n"
            f"  현재 FAIL:  {_brief_env(env_params)}\n"
            f"  완화 후보:  {_brief_env(next_env)}"
        )
        mode = "rule_relax"
        print(f"  [Decision] RULE_RELAX (PASS 미발견, 규칙으로 환경 완화)")

    next_scenario = {
        "scenario_id":            f"scenario_step_{step_next:03d}",
        "environment_parameters": next_env,
        "target_hypothesis":      hypothesis,
        "dspy_analysis":          analysis,
        "decision_mode":          mode,
        "shap_signals":           shap_payload,
    }
    scen_path = session.data_dir / f"dspy_scenario_iter_{step_next:03d}.json"
    scen_path.write_text(json.dumps(next_scenario, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return next_scenario


def _log_boundary(session, step_next, mode, midpoint, width):
    session.boundary_log.append({
        "step":     step_next,
        "mode":     mode,
        "pass_env": session.last_pass_env,
        "fail_env": session.last_fail_env,
        "midpoint": midpoint,
        "width":    width,
    })


def _brief_env(env: dict) -> str:
    return (f"fog={env.get('fog_density_percent', 0):.1f}%, "
            f"illum={env.get('illumination_lux', 0):.0f}lx, "
            f"noise={env.get('camera_noise_level', 0):.2f}")


def _llm_explore(session: Session, last: dict, shap_payload: dict | None):
    """Phase-1 exploration: LLM/SHAP-guided push when no FAIL exists yet."""
    xai_signals  = _build_xai_signals(last, shap_payload)
    perf_signals = _build_perf_signals(last)
    iter_history = [
        {
            "iter":                h["step"],
            "fog_density_percent": h["fog_density_percent"],
            "illumination_lux":    h["illumination_lux"],
            "camera_noise_level":  h["camera_noise_level"],
            "map50":               round(h["map50"], 4),
            "all_passed":          h["all_passed"],
            "violated_count":      h["violated_count"],
        }
        for h in session.history[-5:]
    ]
    try:
        prediction = session.generator(
            iteration_history   = json.dumps(iter_history, ensure_ascii=False),
            xai_analysis        = json.dumps(xai_signals,  ensure_ascii=False),
            current_performance = json.dumps(perf_signals, ensure_ascii=False),
        )
        return (prediction.environment_parameters,
                prediction.target_hypothesis,
                prediction.analysis or prediction.reasoning,
                "llm_explore")
    except Exception as exc:
        print(f"[Orchestrator] DSPy error ({exc}); rule fallback.")
        return (_rule_push(last["env"]),
                "Rule-based fallback push (LLM 호출 실패)",
                "",
                "rule_push_fallback")


def _rule_push(env: dict) -> dict:
    """Deterministic 'push harder' step (used when LLM fails)."""
    fog   = float(env.get("fog_density_percent", 30))
    illum = float(env.get("illumination_lux",    4000))
    noise = float(env.get("camera_noise_level",  0.1))
    return {
        "fog_density_percent": round(min(100, fog + 18), 2),
        "illumination_lux":    round(max(200, illum * 0.70), 1),
        "camera_noise_level":  round(min(0.60, noise + 0.10), 4),
    }


def _rule_relax(env: dict) -> dict:
    """Deterministic 'recover toward baseline' step."""
    fog   = float(env.get("fog_density_percent", 30))
    illum = float(env.get("illumination_lux",    4000))
    noise = float(env.get("camera_noise_level",  0.1))
    return {
        "fog_density_percent": round(max(0, fog - 12), 2),
        "illumination_lux":    round(min(20000, illum * 1.30), 1),
        "camera_noise_level":  round(max(0, noise - 0.08), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (mirrors run_dspy_adversarial.py — kept inline so this module is
# self-contained and the CLI script remains independently runnable)
# ─────────────────────────────────────────────────────────────────────────────

def _bisect_between(a: dict, b: dict) -> dict:
    keys = set(a) | set(b)
    return {k: round((float(a.get(k, 0)) + float(b.get(k, 0))) / 2.0, 4) for k in keys}


def _segment_width(a: dict, b: dict) -> dict:
    keys = set(a) | set(b)
    return {k: round(abs(float(a.get(k, 0)) - float(b.get(k, 0))), 4) for k in keys}


def _build_perf_signals(last: dict) -> dict:
    vc = last["violated_count"]
    return {
        "map50":                  last["map50"],
        "min_clearance_m":        last["min_clearance"],
        "max_consecutive_misses": last["worst_run"],
        "violated_count":         vc,
        "worst_requirement":      ("REQ-1" if last["map50"] < 0.50
                                   else "REQ-3" if last["worst_run"] > 3
                                   else "none"),
        "failure_type":           ("multi_failure"  if vc > 1
                                   else "single_failure" if vc == 1 else "nominal"),
    }


def _build_xai_signals(last: dict, shap_payload: dict | None) -> dict:
    if shap_payload and shap_payload.get("global_feature_importance"):
        dominant = [
            {"name": d["name"], "importance": d["importance"], "direction": d.get("direction")}
            for d in shap_payload["global_feature_importance"]
        ]
        method = "xgboost_shap"
    else:
        dominant = [
            {"name": "fog_density_percent", "importance": 0.5},
            {"name": "camera_noise_level",  "importance": 0.3},
            {"name": "illumination_lux",    "importance": 0.2},
        ]
        method = "heuristic_prior"
    out = {
        "method":           method,
        "dominant_factors": dominant,
        "attention_summary": (
            f"REQ-1={last['map50']:.3f}(th=0.50)  "
            f"REQ-3={last['worst_run']}fr(th=3)  "
            f"fog={last['fog_density_percent']:.1f}%  "
            f"illum={last['illumination_lux']:.0f}lx  "
            f"noise={last['camera_noise_level']:.2f}"
        ),
    }
    if shap_payload:
        out["shap_signals"] = shap_payload
    return out


def _rule_mutation(env: dict, last_passed: bool) -> dict:
    """Backward-compat wrapper: dispatch to push or relax based on last result."""
    return _rule_push(env) if last_passed else _rule_relax(env)
