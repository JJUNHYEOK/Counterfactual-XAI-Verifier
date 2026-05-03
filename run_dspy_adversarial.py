#!/usr/bin/env python3
"""run_dspy_adversarial.py — DSPy-based adversarial UAV scenario generation.

Implements the full LLM → Simulink → DSPy optimization feedback loop:

  1. Configure DSPy LM  (OpenAI / Anthropic from .env)
  2. Build training set from existing data/xai_input_iter_*.json files
  3. Initialise Simulink bridge  (MATLAB Engine API or fast mock proxy)
  4. Run DSPy BootstrapFewShot / MIPROv2 optimisation
       → LLM prompts are automatically improved based on real Simulink scores
  5. Execute adversarial iteration loop
       For each step:
         a) Run Simulink with current env params
         b) Score result (all_passed? violated_count?)
         c) Build XAI / performance feedback
         d) Call compiled DSPy module → next adversarial scenario
         e) Save artifacts to data/
  6. Print results table + save data/dspy_loop_summary.json

Usage:
    # Fast demo (mock simulator, no MATLAB required)
    python run_dspy_adversarial.py --iterations 10 --sim-mode mock

    # Full run with real MATLAB Engine
    python run_dspy_adversarial.py --iterations 15 --sim-mode engine

    # Skip optimisation, load pre-compiled program
    python run_dspy_adversarial.py --iterations 10 --no-optimize

    # Use MIPROv2 instead of BootstrapFewShot
    python run_dspy_adversarial.py --optimizer mipro --iterations 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import dspy
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Language model setup
# ─────────────────────────────────────────────────────────────────────────────

def configure_lm(model_override: str | None = None) -> dspy.LM:
    """Configure DSPy LM from environment variables.

    Priority:
      1. --model CLI argument (explicit override)
      2. OPENAI_API_KEY   → openai/gpt-4o-mini
      3. ANTHROPIC_API_KEY → anthropic/claude-haiku-4-5-20251001
    """
    openai_key    = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if model_override:
        # Infer the right API key from the model name prefix
        if model_override.startswith("anthropic/"):
            api_key = anthropic_key
        else:
            api_key = openai_key
        model = model_override
    elif openai_key:
        model   = "openai/gpt-4o-mini"
        api_key = openai_key
    elif anthropic_key:
        model   = "anthropic/claude-haiku-4-5-20251001"
        api_key = anthropic_key
    else:
        raise EnvironmentError(
            "No LLM API key found.\n"
            "Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file."
        )

    lm = dspy.LM(model, api_key=api_key, temperature=0.4, max_tokens=1024)
    dspy.configure(lm=lm)
    print(f"[LM] Configured: {model}")
    return lm


# ─────────────────────────────────────────────────────────────────────────────
# XAI / performance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_dominant_factors(
    sim_result,
    env_params: dict[str, float],
) -> list[dict]:
    """Estimate feature importance from current env params.

    Mirrors the heuristic in build_xai_input.m:
    fog and low illumination are typically the strongest drivers of REQ-1.
    """
    fog   = float(env_params.get("fog_density_percent", 0))
    illum = float(env_params.get("illumination_lux", 8000))
    noise = float(env_params.get("camera_noise_level", 0))

    fog_score   = fog / 100.0
    illum_score = max(0.0, 1.0 - illum / 8000.0)   # 0 at 8000 lux, 1 at 0 lux
    noise_score = noise / 0.6

    total = fog_score + illum_score + noise_score + 1e-9
    return [
        {"name": "fog_density_percent", "importance": round(fog_score / total,   3)},
        {"name": "illumination_lux",    "importance": round(illum_score / total, 3)},
        {"name": "camera_noise_level",  "importance": round(noise_score / total, 3)},
    ]


def _build_xai_signals(
    sim_result,
    env_params: dict[str, float],
    shap_payload: dict | None = None,
) -> dict:
    """Bundle XAI signals for the LLM.

    If a SHAP payload is supplied (XGBoost+SHAP from accumulated runs),
    its global importance ranking *replaces* the heuristic dominant_factors —
    SHAP values reflect the actual measured response surface, while the
    heuristic is just a hand-coded prior. The full SHAP payload is also
    nested under `shap_signals` so the prompt can show signed local values.
    """
    if shap_payload and shap_payload.get("global_feature_importance"):
        dominant = [
            {"name": d["name"], "importance": d["importance"], "direction": d.get("direction")}
            for d in shap_payload["global_feature_importance"]
        ]
        method = "xgboost_shap"
    else:
        dominant = _infer_dominant_factors(sim_result, env_params)
        method = "heuristic_prior"

    out = {
        "method":           method,
        "dominant_factors": dominant,
        "attention_summary": (
            f"REQ-1={sim_result.map50:.3f}(th=0.50)  "
            f"REQ-2={sim_result.min_clearance:.2f}m(th=2.0m)  "
            f"REQ-3={sim_result.worst_run}fr(th=3)  "
            f"fog={env_params.get('fog_density_percent',0):.1f}%  "
            f"illum={env_params.get('illumination_lux',0):.0f}lx  "
            f"noise={env_params.get('camera_noise_level',0):.2f}"
        ),
    }
    if shap_payload:
        out["shap_signals"] = shap_payload
    return out


def _bisect_between(env_a: dict[str, float], env_b: dict[str, float]) -> dict[str, float]:
    """Midpoint of two environments — used to narrow PASS/FAIL boundary."""
    keys = set(env_a) | set(env_b)
    return {
        k: round((float(env_a.get(k, 0.0)) + float(env_b.get(k, 0.0))) / 2.0, 4)
        for k in keys
    }


def _segment_width(env_a: dict[str, float], env_b: dict[str, float]) -> dict[str, float]:
    """Per-parameter |env_a - env_b| absolute width (for boundary tightness logging)."""
    keys = set(env_a) | set(env_b)
    return {k: round(abs(float(env_a.get(k, 0.0)) - float(env_b.get(k, 0.0))), 4) for k in keys}


def _build_perf_signals(sim_result) -> dict:
    vc = sim_result.violated_count
    return {
        "map50":                  sim_result.map50,
        "min_clearance_m":        sim_result.min_clearance,
        "max_consecutive_misses": sim_result.worst_run,
        "violated_count":         vc,
        "worst_requirement":      (
            "REQ-1" if sim_result.map50 < 0.85 else
            "REQ-3" if sim_result.worst_run > 3 else
            "REQ-2" if sim_result.min_clearance < 2.0 else
            "none"
        ),
        "failure_type": (
            "multi_failure"  if vc > 1 else
            "single_failure" if vc == 1 else
            "nominal"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based fallback mutation (mirrors gpt_for_simulink.py)
# ─────────────────────────────────────────────────────────────────────────────

def _rule_mutation(env: dict[str, float], last_passed: bool) -> dict[str, float]:
    fog   = float(env.get("fog_density_percent", 30))
    illum = float(env.get("illumination_lux",    4000))
    noise = float(env.get("camera_noise_level",  0.1))

    if last_passed:                  # worsen conditions
        fog   = min(100.0, fog   + 18.0)
        illum = max(200.0, illum * 0.70)
        noise = min(0.60,  noise + 0.10)
    else:                            # recover slightly (boundary search)
        fog   = max(0.0,     fog   -  9.0)
        illum = min(20000.0, illum *  1.25)
        noise = max(0.0,     noise -  0.05)

    return {
        "fog_density_percent": round(fog,   2),
        "illumination_lux":    round(illum, 1),
        "camera_noise_level":  round(noise, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(history: list[dict]) -> None:
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"{'Step':>4}  {'Fog%':>6}  {'Illum(lx)':>10}  {'Noise':>6}  "
          f"{'mAP50':>7}  {'Result':>6}  {'Violated':>8}")
    print(sep)
    for h in history:
        e = h["env"]
        status = "FAIL" if not h["all_passed"] else "PASS"
        print(
            f"  {h['step']:>3}  {e.get('fog_density_percent',0):>6.1f}  "
            f"{e.get('illumination_lux',0):>10.0f}  "
            f"{e.get('camera_noise_level',0):>6.3f}  "
            f"{h['map50']:>7.4f}  {status:>6}  "
            f"{h['violated_count']:>8}"
        )
    n      = len(history)
    fails  = sum(1 for h in history if not h["all_passed"])
    print(sep)
    print(f"Total {n} steps  |  Failures {fails}  |  "
          f"Failure rate {fails/max(n,1)*100:.1f}%")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> list[dict]:
    from dspy_pipeline.modules  import AdversarialScenarioGenerator
    from dspy_pipeline.metric   import init_metric_bridge, simulink_adversarial_metric
    from dspy_pipeline.dataset  import load_training_examples
    from dspy_pipeline.optimizer import build_optimizer

    data_dir      = ROOT / "data"
    compiled_path = ROOT / "dspy_pipeline" / "compiled_program.json"
    data_dir.mkdir(exist_ok=True)

    # ── 1. Language model ────────────────────────────────────────────────
    configure_lm(args.model)

    # ── 2. Simulink bridge ───────────────────────────────────────────────
    print(f"\n[Bridge] Initialising Simulink bridge  mode={args.sim_mode}")
    init_metric_bridge(str(ROOT), mode=args.sim_mode)

    # ── 3. Training set ──────────────────────────────────────────────────
    trainset = load_training_examples(str(data_dir))
    if not trainset:
        print("[Warning] No training examples found in data/.  "
              "The module will run without few-shot demos.")

    # ── 4. Build / load compiled module ─────────────────────────────────
    generator = AdversarialScenarioGenerator()

    if not args.no_optimize and trainset:
        print(f"\n[Optimizer] Running {args.optimizer} optimisation "
              f"on {len(trainset)} examples …")
        optimizer = build_optimizer(
            simulink_adversarial_metric,
            optimizer_type=args.optimizer,
            max_bootstrapped_demos=args.max_demos,
            max_labeled_demos=args.max_demos,
        )
        t0        = time.time()
        generator = optimizer.compile(generator, trainset=trainset)
        print(f"[Optimizer] Done in {time.time()-t0:.1f}s")

        compiled_path.parent.mkdir(exist_ok=True)
        generator.save(str(compiled_path))
        print(f"[Optimizer] Compiled program saved → {compiled_path}")

    elif compiled_path.exists():
        generator.load(str(compiled_path))
        print(f"[Optimizer] Loaded pre-compiled program ← {compiled_path}")
    else:
        print("[Optimizer] Skipped (--no-optimize, no compiled program found). "
              "Using raw module.")

    # ── 5. Seed scenario ─────────────────────────────────────────────────
    seed_path = data_dir / "scenario_iter_001.json"
    if not seed_path.exists():
        seed = {
            "scenario_id": "scenario_001_seed",
            "environment_parameters": {
                "fog_density_percent": 0.0,
                "illumination_lux":    8000.0,
                "camera_noise_level":  0.0,
            },
        }
        seed_path.write_text(json.dumps(seed, indent=2), encoding="utf-8")

    with open(seed_path, encoding="utf-8") as f:
        current_scenario = json.load(f)

    # ── 6. Boundary-search loop ──────────────────────────────────────────
    # Phase 1 (no FAIL on record): LLM with SHAP-guided counterfactual to
    #         push the env from a PASS toward FAIL.
    # Phase 2 (have both PASS and FAIL): bisect between the most-recent PASS
    #         and most-recent FAIL to localise the boundary surface.
    print(f"\n[Loop] Starting boundary-search loop  iterations={args.iterations}\n")
    history: list[dict] = []
    last_pass_env: dict[str, float] | None = None
    last_fail_env: dict[str, float] | None = None
    boundary_log: list[dict] = []

    from dspy_pipeline.metric    import get_bridge
    from dspy_pipeline.shap_analyzer import compute_shap_signals, is_available as shap_available

    if not shap_available():
        print("[SHAP] xgboost/shap not installed — falling back to heuristic XAI.")

    for step in range(1, args.iterations + 1):
        print(f"{'='*60}")
        print(f"[Step {step}/{args.iterations}]")

        env_params = current_scenario.get("environment_parameters", {})

        # Simulate
        t0     = time.time()
        result = get_bridge().run_simulation(env_params)
        elapsed = time.time() - t0

        status = "FAIL" if not result.all_passed else "PASS"
        print(
            f"  Env : fog={env_params.get('fog_density_percent',0):.1f}%  "
            f"illum={env_params.get('illumination_lux',0):.0f}lx  "
            f"noise={env_params.get('camera_noise_level',0):.3f}"
        )
        print(
            f"  Sim : mAP50={result.map50:.4f}  "
            f"clearance={result.min_clearance:.2f}m  "
            f"runs={result.worst_run}  "
            f"violated={result.violated_count}  "
            f"[{status}]  ({elapsed:.2f}s)"
        )

        # Update boundary anchors
        if result.all_passed:
            last_pass_env = dict(env_params)
        else:
            last_fail_env = dict(env_params)

        # Save simulation artifact
        eval_artifact = {
            "step": step, "env_params": env_params,
            **result.to_dict(), "sim_mode": args.sim_mode,
        }
        (data_dir / f"dspy_eval_iter_{step:03d}.json").write_text(
            json.dumps(eval_artifact, indent=2), encoding="utf-8"
        )

        # Append to history (used by SHAP and prompt)
        history.append({
            "step":                step,
            "env":                 env_params,
            "fog_density_percent": env_params.get("fog_density_percent", 0),
            "illumination_lux":    env_params.get("illumination_lux", 8000),
            "camera_noise_level":  env_params.get("camera_noise_level", 0),
            "map50":               result.map50,
            "all_passed":          result.all_passed,
            "violated_count":      result.violated_count,
        })

        if step == args.iterations:
            break                    # no need to generate a "next" scenario

        # ── SHAP from accumulated samples ────────────────────────────────
        shap_signals = compute_shap_signals(history, env_params)
        shap_payload = shap_signals.to_payload() if shap_signals else None
        if shap_payload:
            top = shap_payload["global_feature_importance"][0]
            print(f"  [SHAP] n={shap_payload['n_samples']}  "
                  f"top={top['name']} (imp={top['importance']:.2f}, {top['direction']})  "
                  f"R²={shap_payload.get('model_r2')}")

        # ── Decide next scenario ─────────────────────────────────────────
        have_boundary = (last_pass_env is not None) and (last_fail_env is not None)

        if have_boundary:
            # Phase 2 — bisect between last PASS and last FAIL
            next_env   = _bisect_between(last_pass_env, last_fail_env)
            width      = _segment_width(last_pass_env, last_fail_env)
            mode       = "bisect"
            hypothesis = (
                f"마지막 PASS와 FAIL의 중간점에서 경계 위치를 좁힙니다 "
                f"(segment width: {width})."
            )
            analysis = (
                "📊 현재 상황: PASS↔FAIL 경계 구간을 확보. bisection으로 폭을 절반씩 좁힙니다.\n"
                f"  PASS anchor: {last_pass_env}\n"
                f"  FAIL anchor: {last_fail_env}\n"
                f"  midpoint:    {next_env}\n"
                "🎯 경계 탐색 전략: 매 iteration마다 segment width가 1/2로 줄어듭니다."
            )
            boundary_log.append({
                "step":       step + 1,
                "pass_env":   last_pass_env,
                "fail_env":   last_fail_env,
                "midpoint":   next_env,
                "width":      width,
            })
            print(f"  [Bisect] PASS↔FAIL width={width} → midpoint {next_env}")
        else:
            # Phase 1 — LLM exploration with SHAP guidance
            xai_signals  = _build_xai_signals(result, env_params, shap_payload)
            perf_signals = _build_perf_signals(result)
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
                for h in history[-5:]
            ]
            print(f"  [DSPy] Generating counterfactual (SHAP-guided)…")
            try:
                prediction = generator(
                    iteration_history  = json.dumps(iter_history, ensure_ascii=False),
                    xai_analysis       = json.dumps(xai_signals,  ensure_ascii=False),
                    current_performance= json.dumps(perf_signals, ensure_ascii=False),
                )
                next_env   = prediction.environment_parameters
                hypothesis = prediction.target_hypothesis
                analysis   = prediction.analysis or prediction.reasoning
                mode       = "llm_explore"
                print(
                    f"  [DSPy] Next: fog={next_env.get('fog_density_percent',0):.1f}%  "
                    f"illum={next_env.get('illumination_lux',0):.0f}lx  "
                    f"noise={next_env.get('camera_noise_level',0):.3f}"
                )
                print(f"  [DSPy] Hypothesis: {hypothesis}")
            except Exception as exc:
                print(f"  [DSPy] Error ({exc}). Using rule-based fallback.")
                next_env   = _rule_mutation(env_params, result.all_passed)
                hypothesis = "Rule-based fallback mutation"
                analysis   = ""
                mode       = "rule_fallback"

        current_scenario = {
            "scenario_id":            f"scenario_dspy_step_{step+1:03d}",
            "environment_parameters": next_env,
            "target_hypothesis":      hypothesis,
            "dspy_analysis":          analysis,
            "decision_mode":          mode,
            "shap_signals":           shap_payload,
        }
        scen_path = data_dir / f"dspy_scenario_iter_{step+1:03d}.json"
        scen_path.write_text(
            json.dumps(current_scenario, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── 7. Summary ───────────────────────────────────────────────────────
    _print_summary(history)

    if boundary_log:
        last = boundary_log[-1]
        print(
            f"\n[Boundary] Final segment width = {last['width']}\n"
            f"           PASS anchor: {last['pass_env']}\n"
            f"           FAIL anchor: {last['fail_env']}"
        )
    else:
        print("\n[Boundary] No PASS↔FAIL transition observed — try more iterations "
              "or relax the seed.")

    summary = {
        "total_steps":   len(history),
        "passes":        sum(1 for h in history if h["all_passed"]),
        "failures":      sum(1 for h in history if not h["all_passed"]),
        "optimizer":     args.optimizer,
        "sim_mode":      args.sim_mode,
        "history":       history,
        "boundary_log":  boundary_log,
        "final_pass_env": last_pass_env,
        "final_fail_env": last_fail_env,
    }
    summary_path = data_dir / "dspy_loop_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[Done] Summary saved → {summary_path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DSPy-based adversarial UAV scenario generation for Simulink testbed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_dspy_adversarial.py --sim-mode mock --iterations 10\n"
            "  python run_dspy_adversarial.py --sim-mode engine --iterations 20\n"
            "  python run_dspy_adversarial.py --optimizer mipro --iterations 15\n"
            "  python run_dspy_adversarial.py --no-optimize --iterations 5\n"
        ),
    )
    parser.add_argument(
        "--iterations", type=int, default=10,
        help="Number of adversarial iterations (default: 10)",
    )
    parser.add_argument(
        "--optimizer", type=str, default="bootstrap",
        choices=["bootstrap", "mipro"],
        help="DSPy optimizer: 'bootstrap' (BootstrapFewShot) or 'mipro' (MIPROv2) [default: bootstrap]",
    )
    parser.add_argument(
        "--sim-mode", dest="sim_mode", type=str, default="auto",
        choices=["auto", "engine", "mock", "subprocess"],
        help=(
            "Simulink backend: 'engine' (MATLAB Engine API), "
            "'mock' (fast proxy, no MATLAB), 'subprocess', 'auto' [default: auto]"
        ),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Override DSPy LM model ID, e.g. "
            "'openai/gpt-4o' or 'anthropic/claude-opus-4-7'"
        ),
    )
    parser.add_argument(
        "--max-demos", dest="max_demos", type=int, default=4,
        help="Max bootstrapped demos for BootstrapFewShot (default: 4)",
    )
    parser.add_argument(
        "--no-optimize", action="store_true",
        help="Skip optimisation step; use pre-compiled program if available, else raw module",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
