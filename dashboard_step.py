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

# Load .env from the project root so the LLM key is available even when
# MATLAB launches Python from a fresh process (no shell context).
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass    # python-dotenv not installed → caller must export env vars manually


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

def narrate_edge_case(record_json: str) -> str:
    """LLM-generated edge-case narrative for the dashboard's Operations log.

    Wraps dspy_pipeline.narrator.EdgeCaseDescriber. Returns a 3–5 sentence
    Korean security-report narrative, OR a templated fallback if the LLM
    or narrator module is unavailable.

    record_json: JSON of {fog, ill, noi, metric, verdict,
                          n_intruders_total, n_intruders_seen,
                          n_intruders_detected, n_intruders_missed,
                          (legacy: ngt, ndet, ntp — frame-aggregated)}

    Mission semantics: "intruders missed" counts UNIQUE intruders (max 5 =
    3 people + 2 vehicles) — NOT per-frame GT instances. The dashboard now
    sends the unique counts via n_intruders_*. Legacy ngt/ndet/ntp are kept
    as a fallback so old records still parse, but they are frame-aggregated
    and would inflate "missed" from ≤5 to hundreds.
    """
    try:
        record = json.loads(record_json)
    except Exception as exc:                          # noqa: BLE001
        return f"  (narrative unavailable: bad record JSON: {exc})"

    fog = float(record.get("fog", 0))
    ill = float(record.get("ill", 0))
    noi = float(record.get("noi", 0))
    metric = float(record.get("metric", 0))
    verdict = str(record.get("verdict", "PASS"))

    # Qualitative env summary (matches mission_context tone_for_narrator)
    fog_q  = ("맑은 시계" if fog < 8 else "옅은 산 안개" if fog < 25
              else "발달한 안개층" if fog < 50 else "짙은 안개" if fog < 75
              else "시정 거의 차단된 농무")
    ill_q  = ("맑은 한낮의 풍부한 광량" if ill > 10000
              else "부분 흐림 일반 작전 광량" if ill > 6000
              else "흐린 한낮" if ill > 3000
              else "황혼·일출 직전 저조도" if ill > 800
              else "야간에 준하는 한계 광량")
    noi_q  = ("센서 정상 상태" if noi < 0.05
              else "센서 약간 노후" if noi < 0.15
              else "센서 노이즈 누적" if noi < 0.30
              else "센서 심각한 열화")
    mAP_band = ("정상" if metric >= 0.50
                else "임계 진입" if metric >= 0.25
                else "임무 불가")

    # Unique-intruder counts preferred; fall back to frame-aggregated only
    # if the new fields are missing (legacy callers).
    if "n_intruders_missed" in record:
        intruders_total    = int(record.get("n_intruders_total",  5))
        intruders_seen     = int(record.get("n_intruders_seen",   intruders_total))
        intruders_detected = int(record.get("n_intruders_detected", 0))
        intruders_missed   = int(record["n_intruders_missed"])
    else:
        intruders_total    = 5
        intruders_seen     = 5
        intruders_detected = 0
        intruders_missed   = max(0, int(record.get("ngt", 0)) - int(record.get("ntp", 0)))
    boundary_status = ("far_inside_pass" if metric >= 0.65
                       else "near_boundary" if metric >= 0.25
                       else "broken")

    # Try LLM via Narrator
    try:
        global _NARRATOR_TRIED, _NARRATOR, _NARRATOR_LM
        if "_NARRATOR_TRIED" not in globals():
            _NARRATOR_TRIED = False
        if not _NARRATOR_TRIED:
            _NARRATOR_TRIED = True
            from dspy_pipeline.orchestrator import _build_lm
            from dspy_pipeline.narrator     import Narrator, load_mission_context
            _NARRATOR_LM = _build_lm(None)
            _NARRATOR    = Narrator()
        if _NARRATOR_LM is None:
            raise RuntimeError("LM not available")

        import dspy
        from dspy_pipeline.narrator import load_mission_context
        mission = load_mission_context()
        env_summary = json.dumps({
            "fog_level":          fog_q,
            "illum_level":        ill_q,
            "noise_level":        noi_q,
            "mAP_band":           mAP_band,
            "intruders_total":    intruders_total,
            "intruders_seen":     intruders_seen,
            "intruders_detected": intruders_detected,
            "intruders_missed":   intruders_missed,
        }, ensure_ascii=False)
        sim_outcome = json.dumps({
            "verdict":   verdict,
            "map50":     round(metric, 3),
            "REQ_1_met": metric >= 0.50,
        }, ensure_ascii=False)

        with dspy.context(lm=_NARRATOR_LM):
            pred = _NARRATOR.describe_edge_case(
                mission_context=mission,
                env_summary=env_summary,
                sim_outcome=sim_outcome,
                boundary_status=boundary_status,
            )
        narrative = getattr(pred, "edge_case_narrative", None) or str(pred)
        return "  " + narrative.strip()
    except Exception as exc:                          # noqa: BLE001
        # Template fallback (same content as MATLAB's buildTemplateNarrative)
        impact = (
            f"탐지 정확도 {metric:.2f}로 임무 요구사항을 안정적으로 통과"
            if verdict == "PASS" else
            f"탐지 정확도 {metric:.2f}로 임계 부근 — 보고 마진이 사실상 소진된 경계선 상태"
            if verdict == "MARGINAL" else
            f"탐지 정확도 {metric:.2f}로 REQ-1 미달, 보고 누락 운용 불가"
        )
        tag = ("[ 임무 수행 가능 ]" if verdict == "PASS" else
               "[ 임무 마지노선 ]" if verdict == "MARGINAL" else
               "[ 임무 수행 불가 ]")
        return (f"  {tag} {fog_q}, {ill_q}, {noi_q} 환경에서 UAV 정찰을 수행한 결과, "
                f"{impact}. (LLM unavailable: {exc})")


def compute_xai_for_dashboard(history_json: str) -> dict:
    """Compute KernelSHAP-based feature importance for the dashboard.

    Returns a dict with:
        method:                 "kernel_shap" or "uniform_fallback" or "unavailable"
        global_importance:      {fog, illum, noise} normalized to sum 1
        signed_contributions:   {fog, illum, noise} signed (− = harms detection)
        n_samples:              history count used
        model_r2:               surrogate Ridge fit quality (sanity only)

    The MATLAB dashboard calls this each finalizeRun and replaces the
    Pearson bar chart with these SHAP values. Falls back to uniform
    weights if shap / sklearn missing or history < 3 samples.
    """
    try:
        records = json.loads(history_json) if history_json else []
    except Exception as exc:
        return _xai_fallback(f"bad history JSON: {exc}")

    if len(records) < 3:
        return _xai_fallback(f"need >= 3 cases (have {len(records)})")

    # Build orchestrator-shaped history for shap_analyzer
    orc_history = [
        {
            "fog_density_percent": float(r["fog"]),
            "illumination_lux":    float(r["ill"]),
            "camera_noise_level":  float(r["noi"]),
            "map50":               float(r.get("f1", r.get("metric", 0.0))),
        }
        for r in records
    ]
    current_env = {
        "fog_density_percent": float(records[-1]["fog"]),
        "illumination_lux":    float(records[-1]["ill"]),
        "camera_noise_level":  float(records[-1]["noi"]),
    }

    try:
        from dspy_pipeline.shap_analyzer import compute_shap_signals, is_available
        if not is_available():
            return _xai_fallback("shap/sklearn not installed")
        sig = compute_shap_signals(orc_history, current_env)
        if sig is None:
            return _xai_fallback("shap_analyzer returned None")

        payload = sig.to_payload()
        # Re-key to dashboard's short axis names
        gi  = {item["name"]: float(item["importance"]) for item in payload["global_feature_importance"]}
        lc  = {item["name"]: float(item.get("contribution", 0.0)) for item in payload.get("local_feature_contributions", [])}

        # Normalize global importance to sum 1
        tot = sum(abs(v) for v in gi.values())
        if tot > 1e-9:
            gi_norm = {k: abs(v) / tot for k, v in gi.items()}
        else:
            gi_norm = {k: 1.0 / max(1, len(gi)) for k in gi}

        return {
            "method":               "kernel_shap",
            "fog_importance":       gi_norm.get("fog_density_percent", 0.0),
            "illum_importance":     gi_norm.get("illumination_lux",    0.0),
            "noise_importance":     gi_norm.get("camera_noise_level",  0.0),
            "fog_signed":           lc.get("fog_density_percent", 0.0),
            "illum_signed":         lc.get("illumination_lux",    0.0),
            "noise_signed":         lc.get("camera_noise_level",  0.0),
            "n_samples":            int(payload.get("n_samples", len(records))),
            "model_r2":             float(payload.get("model_r2") or 0.0),
        }
    except Exception as exc:
        return _xai_fallback(f"shap exception: {exc}")


def _xai_fallback(reason: str) -> dict:
    """Uniform-importance fallback when SHAP is unavailable."""
    print(f"[dashboard_step] XAI fallback: {reason}")
    return {
        "method":               "uniform_fallback",
        "fog_importance":       1/3,
        "illum_importance":     1/3,
        "noise_importance":     1/3,
        "fog_signed":           0.0,
        "illum_signed":         0.0,
        "noise_signed":         0.0,
        "n_samples":            0,
        "model_r2":             0.0,
    }


def next_case_from_history(history_json: str, model_dir: str = ".") -> dict:
    """Decide the next counterfactual case using a deterministic
    boundary-search policy that **oscillates PASS ↔ FAIL**.

    Rationale (why no LLM here):
        The previous version asked an `AdversarialScenarioGenerator` LLM
        for the next case. That model is *adversarially* trained — it keeps
        proposing harder cases regardless of the current verdict, so after
        a FAIL it would push the system into ever-deeper FAILs instead of
        recovering. Boundary search requires the opposite: after a FAIL
        you must *recover* toward the PASS anchor, and after a PASS you
        probe toward the FAIL anchor — converging on the boundary from
        both sides.

    Policy:
        last_PASS  + FAIL  anchor exists  →  bisect 65 % toward FAIL  (probe)
        last_FAIL  + PASS  anchor exists  →  bisect 75 % toward PASS  (RECOVERY,
                                              asymmetric to ensure the next
                                              case reliably PASSes)
        last_PASS  + no FAIL yet          →  push     (discover failure side)
        last_FAIL  + no PASS yet          →  relax    (discover safe side)

    The asymmetric weights (0.65 probe / 0.75 recovery) guarantee that the
    oscillation pattern is: PASS · FAIL · PASS · FAIL · ... narrowing on
    the boundary, instead of PASS · FAIL · deeper-FAIL · still-FAIL · ...
    which is what the user reported with the LLM in the loop.
    """
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

    if last_passed and last_fail is not None:
        env = _bisect(last_env, last_fail, weight=0.65)
        return _envelope(env, "boundary_push",
            "PASS → bisect 65% toward FAIL anchor (probe boundary)")

    if (not last_passed) and last_pass is not None:
        # Asymmetric: 0.75 (vs 0.65 for probe) so recovery is *reliable* —
        # the next case lands well inside the PASS zone instead of straddling
        # the boundary and FAILing again.
        env = _bisect(last_env, last_pass, weight=0.75)
        return _envelope(env, "boundary_recover",
            "FAIL → bisect 75% toward PASS anchor (reliable recovery)")

    if last_passed:
        env = _rule_push(last_env)
        return _envelope(env, "rule_push",
            "PASS / no FAIL anchor yet → push to discover failure side")

    env = _rule_relax(last_env)
    return _envelope(env, "rule_relax",
        "FAIL / no PASS anchor yet → relax to discover safe side")


def _llm_decide(records, last_env, last_record, last_pass, last_fail,
                rule_env, rule_branch):
    """Call DSPy LLM with full SHAP+performance context. Returns
    (env, mode, analysis_str) on success, None on any failure.

    The rule_env serves both as a fallback AND as a *hint* to the LLM:
    we include it in the prompt so the LLM can either confirm it or
    propose a more informative alternative for this iteration."""
    global _DSPY_TRIED, _DSPY_LM
    try:
        if not _DSPY_TRIED:
            _DSPY_TRIED = True
            from dspy_pipeline.orchestrator import _build_lm
            _DSPY_LM = _build_lm(None)
        if _DSPY_LM is None:
            return None

        import dspy
        from dspy_pipeline.modules import AdversarialScenarioGenerator
        from dspy_pipeline.orchestrator import _build_xai_signals, _build_perf_signals

        # Compute fresh SHAP signals from history → real XAI input
        shap_payload = _compute_shap_payload(records)

        last_orc = {
            "step":                len(records),
            "fog_density_percent": last_env["fog_density_percent"],
            "illumination_lux":    last_env["illumination_lux"],
            "camera_noise_level":  last_env["camera_noise_level"],
            "map50":               float(last_record.get("metric", last_record.get("f1", 0.0))),
            "all_passed":          bool(last_record.get("passed", False)),
            "violated_count":      0 if last_record.get("passed", False) else 1,
            "min_clearance":       float("nan"),
            "worst_run":           0,
        }
        xai_signals  = _build_xai_signals(last_orc, shap_payload)
        perf_signals = _build_perf_signals(last_orc)

        # Iteration history (last 5 only) — orchestrator schema
        hist = []
        for i, r in enumerate(records[-5:]):
            hist.append({
                "iter":                i + 1,
                "fog_density_percent": float(r["fog"]),
                "illumination_lux":    float(r["ill"]),
                "camera_noise_level":  float(r["noi"]),
                "map50":               round(float(r.get("metric", r.get("f1", 0.0))), 4),
                "all_passed":          bool(r.get("passed", False)),
                "violated_count":      0 if r.get("passed", False) else 1,
            })

        gen = AdversarialScenarioGenerator()
        with dspy.context(lm=_DSPY_LM):
            pred = gen(
                iteration_history   = json.dumps(hist, ensure_ascii=False),
                xai_analysis        = json.dumps(xai_signals,  ensure_ascii=False),
                current_performance = json.dumps(perf_signals, ensure_ascii=False),
            )

        env = dict(pred.environment_parameters)
        # LLM may omit keys — fill with rule_env defaults
        for k in ("fog_density_percent", "illumination_lux", "camera_noise_level"):
            if k not in env:
                env[k] = rule_env[k]
            else:
                env[k] = float(env[k])

        mode = "llm_" + rule_branch.replace("rule_", "").replace("boundary_", "")
        analysis = getattr(pred, "target_hypothesis", "") or getattr(pred, "analysis", "") or "LLM-decided"
        return env, mode, str(analysis)
    except Exception as exc:                          # noqa: BLE001
        # Only log on first few iters to avoid noise during normal demo
        if len(records) <= 3:
            print(f"[dashboard_step] LLM unavailable ({exc}) — using rule policy.")
        return None


def _compute_shap_payload(records):
    """Return a SHAP payload dict (orchestrator schema) or None."""
    if len(records) < 3:
        return None
    try:
        from dspy_pipeline.shap_analyzer import compute_shap_signals, is_available
        if not is_available():
            return None
        orc_history = [
            {
                "fog_density_percent": float(r["fog"]),
                "illumination_lux":    float(r["ill"]),
                "camera_noise_level":  float(r["noi"]),
                "map50":               float(r.get("metric", r.get("f1", 0.0))),
            }
            for r in records
        ]
        current_env = {
            "fog_density_percent": float(records[-1]["fog"]),
            "illumination_lux":    float(records[-1]["ill"]),
            "camera_noise_level":  float(records[-1]["noi"]),
        }
        sig = compute_shap_signals(orc_history, current_env)
        return sig.to_payload() if sig else None
    except Exception:
        return None


def parse_requirement_to_env(requirement_text: str) -> dict:
    """Map a natural-language test requirement to (fog, ill, noi) parameters.

    The dashboard's new input modality: tester types a free-form requirement
    in Korean (or English) describing the operating regime they want to test,
    e.g. "야간 산악 정찰에서 짙은 안개 + 노후 센서 환경" — and this returns a
    concrete (fog_density_percent, illumination_lux, camera_noise_level)
    triple that the dashboard then runs through the simulator.

    Returns:
        {
          "fog":   <float 0..100>,
          "ill":   <float 100..15000>,
          "noi":   <float 0..1.0>,
          "label": <str — short Korean summary of the parsed regime>,
          "method": "llm" | "rule_based",
        }

    Logic: try LLM via DSPy first (gives nuanced understanding of vague
    phrasing). If LLM is unavailable, fall back to a keyword-matching rule
    that covers the same axes the narrator's qualitative bands use (안개,
    조도, 잡음). Rule-based fallback always returns *some* valid envelope.
    """
    text = (requirement_text or "").strip()
    if not text:
        return {"fog": 10.0, "ill": 10000.0, "noi": 0.03,
                "label": "baseline (empty requirement)",
                "method": "rule_based"}

    # --- LLM path -----------------------------------------------------------
    try:
        from dspy_pipeline.orchestrator import _build_lm
        import dspy

        class _RequirementParser(dspy.Signature):
            """국경 산악 감시 UAV의 테스트 요구사항을 환경 파라미터로 변환한다.

            출력은 fog_density_percent (0~100), illumination_lux (100~15000),
            camera_noise_level (0~1.0), 그리고 그 조건을 가장 잘 요약하는
            정성적 한 줄 label 이다. 사용자가 명시한 단서를 우선시하고,
            모호한 경우 한국 산악 작전의 보수적 기본값을 사용한다.
            """
            requirement: str = dspy.InputField(desc="테스터의 자연어 요구사항")
            fog_density_percent: float = dspy.OutputField(desc="0~100 사이 안개 밀도 (%)")
            illumination_lux:    float = dspy.OutputField(desc="100~15000 사이 조도 (lx)")
            camera_noise_level:  float = dspy.OutputField(desc="0~1.0 사이 센서 잡음")
            label:               str   = dspy.OutputField(desc="한 줄 한국어 요약")

        lm = _build_lm(None)
        if lm is None:
            raise RuntimeError("LM not available")
        with dspy.context(lm=lm):
            pred = dspy.ChainOfThought(_RequirementParser)(requirement=text)
        fog = max(0.0,  min(100.0,   float(pred.fog_density_percent)))
        ill = max(100.0, min(15000.0, float(pred.illumination_lux)))
        noi = max(0.0,  min(1.0,     float(pred.camera_noise_level)))
        return {"fog": fog, "ill": ill, "noi": noi,
                "label": (pred.label or text[:40]).strip(),
                "method": "llm"}
    except Exception:                                # noqa: BLE001
        pass    # fall through to rule-based

    # --- Rule-based fallback ------------------------------------------------
    t   = text.lower()
    fog, ill, noi = 10.0, 10000.0, 0.03

    if any(w in t for w in ["농무", "시정 차단", "극심한 안개"]): fog = 90.0
    elif any(w in t for w in ["짙은 안개", "deep fog"]):           fog = 65.0
    elif any(w in t for w in ["발달한 안개", "안개층"]):           fog = 45.0
    elif any(w in t for w in ["옅은 안개", "산 안개", "엷은 안개", "light fog"]): fog = 22.0
    elif "안개" in t or "fog" in t:                                 fog = 30.0
    elif any(w in t for w in ["맑은", "clear"]):                    fog = 5.0

    if any(w in t for w in ["야간", "심야", "밤", "night", "한계 광량"]): ill = 500.0
    elif any(w in t for w in ["황혼", "일출 직전", "저조도", "twilight", "low light"]): ill = 1500.0
    elif any(w in t for w in ["흐린 한낮", "흐림", "흐린"]):              ill = 4000.0
    elif any(w in t for w in ["부분 흐림", "partial cloud", "흐린 하늘"]): ill = 6500.0
    elif any(w in t for w in ["정오", "한낮", "맑은 한낮", "noon"]):       ill = 12000.0

    if any(w in t for w in ["심각한 노후", "센서 열화", "농후한 잡음"]):  noi = 0.50
    elif any(w in t for w in ["노후 센서", "센서 노후", "노이즈 누적", "잡음 누적"]): noi = 0.30
    elif any(w in t for w in ["경미한 잡음", "약간의 잡음"]):              noi = 0.15
    elif any(w in t for w in ["잡음", "노이즈", "noise"]):                  noi = 0.18

    # Build a short label from what was detected
    parts = []
    parts.append("짙은 안개" if fog >= 50 else "옅은 안개" if fog >= 20 else "맑음")
    parts.append("야간" if ill <= 1000 else "흐림" if ill <= 5000 else "일반 광량")
    if noi >= 0.30: parts.append("센서 노후")
    label = " · ".join(parts)
    return {"fog": fog, "ill": ill, "noi": noi,
            "label": label, "method": "rule_based"}


def generate_normalized_test_cases(history_json: str,
                                   requirement_text: str = "",
                                   max_cases: int = 10) -> dict:
    """Produce 1~`max_cases` normalized test cases from the run history.

    This is the dashboard's primary deliverable: given a session's collected
    PASS / MARGINAL / FAIL cases, return a curated set of test cases that
    cover (a) the failure boundary, (b) confirmed safe envelope, and
    (c) representative environmental stressors. Each case is pure-text so it
    can be pasted into a test-plan document or replayed in another session.

    Args:
        history_json: JSON array of run records (same schema as
            summarize_mission). May be empty.
        requirement_text: optional NL requirement string — passed to the LLM
            for context-aware case selection.
        max_cases: hard cap on returned cases (1..10).

    Returns dict:
        {
          "cases":    [ {label, fog, ill, noi, verdict, rationale}, ... ],
          "text":     <full pretty-printed text block — copy-paste ready>,
          "method":   "llm" | "rule_based",
          "n_source": <number of source runs analysed>,
        }
    """
    max_cases = max(1, min(10, int(max_cases)))

    try:
        history = json.loads(history_json) if history_json else []
        if not isinstance(history, list):
            history = []
    except Exception as exc:                          # noqa: BLE001
        return {"cases": [], "text": f"(bad history JSON: {exc})",
                "method": "error", "n_source": 0}

    # Rule-based selector is THE source of selection logic. Each case has
    # a tagged `selection_reason` explaining why it was picked.
    cases = _rule_select_cases(history, max_cases)

    # --- LLM enrichment (optional) — improves only `label` and `rationale`.
    # selection_reason, source_iter, fog/ill/noi, verdict, metric stay
    # pinned to the rule output so the user can always see why each case
    # was chosen, even when LLM rewrites the prose.
    method = "rule_based"
    try:
        from dspy_pipeline.orchestrator import _build_lm
        import dspy

        class _CaseRationale(dspy.Signature):
            """선정된 각 테스트 케이스에 대해 임무 보고 톤의 짧은 운용 영향
            (operational impact, rationale) 1~2문장과 자연스러운 한국어
            label만 생성한다. fog/ill/noi/verdict/selection_reason은
            절대 수정하지 않는다.
            """
            mission_context: str = dspy.InputField(desc="임무 정의")
            requirement:     str = dspy.InputField(desc="테스터의 자연어 요구사항")
            raw_cases:       str = dspy.InputField(
                desc="환경값+선정 사유 JSON 배열 (수정 금지)")
            enriched_cases:  str = dspy.OutputField(
                desc="[{label, rationale}, ...] JSON 배열 — 입력 순서·길이 동일하게 유지")

        lm = _build_lm(None)
        if lm is None or not cases:
            raise RuntimeError("LM unavailable or no source cases")
        from dspy_pipeline.narrator import load_mission_context
        with dspy.context(lm=lm):
            pred = dspy.ChainOfThought(_CaseRationale)(
                mission_context = load_mission_context(),
                requirement     = requirement_text or "(미지정)",
                raw_cases       = json.dumps(cases, ensure_ascii=False),
            )
        enriched = json.loads(pred.enriched_cases)
        # Merge only `label` and `rationale` from the LLM output, position by
        # position. Everything else (env, verdict, selection_reason, …) is
        # preserved from the rule-based selection so the audit trail stays
        # intact and the LLM cannot hide or rewrite *why* a case was chosen.
        if isinstance(enriched, list) and enriched:
            for i, e in enumerate(enriched[:len(cases)]):
                if not isinstance(e, dict):
                    continue
                if "label" in e and e["label"]:
                    cases[i]["label"] = str(e["label"])
                if "rationale" in e and e["rationale"]:
                    cases[i]["rationale"] = str(e["rationale"])
            method = "rule_based + llm_enrichment"
    except Exception:                                # noqa: BLE001
        pass

    # DO NOT auto-save — the dashboard's 💾 다운로드 button calls
    # save_last_test_suite() to persist explicitly. We cache the generated
    # suite at module level so the download handler can pick it up later
    # without the dashboard having to pass the case array back to Python.
    global _last_generated_suite
    _last_generated_suite = {
        "cases":       cases,
        "requirement": requirement_text,
        "method":      method,
        "n_source":    len(history),
    }

    return {
        "cases":       cases,
        "text":        _format_test_cases_text(cases, requirement_text, method, ""),
        "method":      method,
        "n_source":    len(history),
        "export_path": "",
    }


# Module-level cache of the most recently generated test suite. Populated by
# generate_normalized_test_cases(), consumed by save_last_test_suite().
_last_generated_suite = None


def save_last_test_suite() -> dict:
    """Persist the most recently generated normalized test suite to
    data/test_suites/test_suite_<timestamp>.json. Called by the dashboard's
    💾 다운로드 button.

    Returns:
        {saved: bool, path: str, message: str}
    """
    global _last_generated_suite
    if _last_generated_suite is None:
        return {"saved": False, "path": "", "message": "먼저 테스트 케이스를 생성하세요."}
    s = _last_generated_suite
    path = _export_test_suite(s["cases"], s["requirement"], s["n_source"], s["method"])
    if path:
        return {"saved": True, "path": path,
                "message": f"저장 완료: {path}"}
    return {"saved": False, "path": "", "message": "저장 실패 (디스크 오류 가능)"}


def list_test_suites() -> list:
    """List all exported test-suite JSON files (most recent first).

    Returns a list of dicts:
        [{path, filename, generated_at, requirement, n_cases}, ...]

    Used by the dashboard's replay panel to populate its dropdown so the
    operator can pick an earlier session's normalized suite and re-run it
    against the current system version (regression testing).
    """
    out_dir = _PROJECT_ROOT / "data" / "test_suites"
    if not out_dir.exists():
        print(f"[dashboard_step] list_test_suites: dir does not exist: {out_dir}")
        return []
    files = sorted(out_dir.glob("test_suite_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"[dashboard_step] list_test_suites: scanning {out_dir} → {len(files)} file(s)")
    results = []
    for p in files:
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            results.append({
                "path":         str(p),
                "filename":     p.name,
                "generated_at": str(payload.get("generated_at", "")),
                "requirement":  str(payload.get("requirement", ""))[:60],
                "n_cases":      len(payload.get("cases", [])),
            })
        except Exception:                            # noqa: BLE001
            results.append({
                "path":         str(p),
                "filename":     p.name,
                "generated_at": "(unreadable)",
                "requirement":  "",
                "n_cases":      0,
            })
    return results


def load_test_suite(path: str) -> dict:
    """Load a previously-exported test-suite JSON for replay.

    Returns the full payload dict (version/generated_at/requirement/method/
    n_source_runs/tolerance_pct/cases). On failure returns {"error": "..."}.
    """
    try:
        p = Path(path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / path
        payload = json.loads(p.read_text(encoding="utf-8"))
        if "cases" not in payload or not isinstance(payload["cases"], list):
            return {"error": "missing or invalid 'cases' field"}
        return payload
    except Exception as exc:                          # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def compare_replay_result(expected_case: dict, actual_verdict: str,
                          actual_metric: float) -> dict:
    """Compare one replayed case's outcome against its saved expectation.

    Returns:
        {
          "match":            True / False,            # verdict in acceptable set
          "metric_in_band":   True / False,            # metric within [min, max]
          "regression":       True / False,            # match=False OR drift outside band
          "summary":          <one-line str>,
        }
    """
    acceptable = expected_case.get("acceptable_verdicts") or [expected_case.get("expected_verdict")]
    mn = float(expected_case.get("metric_min", 0.0))
    mx = float(expected_case.get("metric_max", 1.0))
    verdict_match = str(actual_verdict) in acceptable
    metric_in_band = (mn <= float(actual_metric) <= mx)
    regression = (not verdict_match)    # verdict mismatch = clear regression
    drift = (verdict_match and not metric_in_band)
    if regression:
        tag = "✗ REGRESSION"
    elif drift:
        tag = "△ DRIFT"
    else:
        tag = "✓ OK"
    summary = (f"{tag}  actual={actual_verdict}/{actual_metric:.3f}  "
               f"expected={expected_case.get('expected_verdict')}/"
               f"{expected_case.get('expected_metric'):.3f}  "
               f"acceptable={'/'.join(acceptable)}  band=[{mn:.2f},{mx:.2f}]")
    return {
        "match":           bool(verdict_match),
        "metric_in_band":  bool(metric_in_band),
        "regression":      bool(regression),
        "drift":           bool(drift),
        "summary":         summary,
    }


def _export_test_suite(cases: list, requirement: str, n_source: int, method: str) -> str:
    """Persist the generated test suite to data/test_suites/ as JSON.

    Schema (version 1):
        {
          "version":        1,
          "generated_at":   <ISO datetime>,
          "requirement":    <NL string the tester typed>,
          "method":         "rule_based" | "rule_based + llm_enrichment",
          "n_source_runs":  <int>,
          "tolerance_pct":  15.0,
          "cases":          [<case dicts as returned by _rule_select_cases>]
        }

    Each case dict already carries `expected_verdict`, `metric_min`,
    `metric_max`, and `acceptable_verdicts` — enough to auto-compare a
    re-run against the original session.

    Returns the absolute path written, or "" on failure.
    """
    try:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _PROJECT_ROOT / "data" / "test_suites"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"test_suite_{ts}.json"
        payload = {
            "version":       1,
            "generated_at":  _now_iso(),
            "requirement":   requirement or "",
            "method":        method,
            "n_source_runs": n_source,
            "tolerance_pct": 15.0,
            "cases":         cases,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"[dashboard_step] exported test suite → {path}")
        return str(path)
    except Exception as exc:                          # noqa: BLE001
        print(f"[dashboard_step] test suite export failed: {exc}")
        return ""


def _category_for(reason_tag: str) -> str:
    """Classify a case by its selection_reason into one of four coverage
    buckets used in the test-suite summary header.

        회귀     — confirms baseline PASS still works after code changes
        경계     — confirms the PASS↔FAIL boundary hasn't shifted
        한계     — confirms known worst-case failures stay failed
        다양성   — extra coverage across a different stress axis
    """
    r = (reason_tag or "").upper()
    if "BOUNDARY" in r or r.startswith("MARGINAL"):
        return "경계"
    if r.startswith("WORST") or "WORST_CASE" in r or r.startswith("STRESS"):
        return "한계"
    if r.startswith("BEST_PASS") or r.startswith("BASELINE") or r.startswith("SAFE_ENVELOPE"):
        return "회귀"
    if r.startswith("DIVERSITY_FAIL"):
        return "한계"
    if r.startswith("DIVERSITY_PASS"):
        return "회귀"
    return "다양성"


def _tolerance_for(verdict: str, metric: float) -> dict:
    """Compute the acceptable re-run tolerance for a generated case so the
    test suite can be auto-compared against future runs.

    Returns:
        {
          "expected_metric":     <float>,        # original mAP@0.5
          "metric_min":          <float>,        # lower acceptance bound (±15% but clamped to verdict band)
          "metric_max":          <float>,        # upper acceptance bound
          "acceptable_verdicts": [<str>, ...],   # verdicts that still mean "no regression"
        }

    Logic: solid PASS / solid FAIL must reproduce their exact verdict;
    boundary-adjacent cases are allowed to drift one band toward MARGINAL
    (since simulator noise can flip them across the threshold). MARGINAL
    accepts any of the three.
    """
    m = float(metric or 0)
    band_low  = round(max(0.0, m * 0.85), 3)
    band_high = round(min(1.0, m * 1.15), 3)
    if verdict == "PASS":
        # Boundary-adjacent PASS (metric near 0.50) allows MARGINAL drift
        acceptable = ["PASS", "MARGINAL"] if m < 0.60 else ["PASS"]
    elif verdict == "FAIL":
        # Boundary-adjacent FAIL (metric near 0.25) allows MARGINAL drift
        acceptable = ["FAIL", "MARGINAL"] if m > 0.15 else ["FAIL"]
    else:   # MARGINAL — by definition straddles boundary, accept all
        acceptable = ["PASS", "MARGINAL", "FAIL"]
    return {
        "expected_metric":     round(m, 3),
        "metric_min":          band_low,
        "metric_max":          band_high,
        "acceptable_verdicts": acceptable,
    }


def _rule_select_cases(history: list, max_cases: int) -> list:
    """Principled boundary-search case selector (boundary-first priority).

    Selection priority (in order — each case is tagged with the rule that
    chose it so the output can show *why* it was selected):

        1. MARGINAL_*       — every MARGINAL case (literally on the boundary,
                              sorted by |metric - 0.5| ascending so the most
                              boundary-relevant appears first)
        2. BOUNDARY_FAIL    — highest-mAP FAIL case (just past failure line)
                              *skip_dedup* — boundary roles must survive even
                              when adjacent to a MARGINAL anchor in L1 space
        3. BOUNDARY_PASS    — lowest-mAP PASS case (just inside safe line)
                              *skip_dedup* — same rationale as BOUNDARY_FAIL
        4. WORST_FAIL       — lowest-mAP FAIL case (operational-limit anchor)
        5. BEST_PASS        — highest-mAP PASS case (regression baseline)
        6. DIVERSITY_*      — remaining PASS / FAIL / MARGINAL spread across
                              the envelope (sorted by |mAP - 0.5|, kept only
                              if L1 distance to already-selected cases
                              ≥ 0.15 in normalised (fog/100, ill/15000, noi)
                              space)

    Dedup note: primary boundary roles (steps 2 and 3) bypass the L1 < 0.15
    near-duplicate check. The boundary-search algorithm intrinsically
    produces MARGINAL / BOUNDARY_FAIL / BOUNDARY_PASS cases very close to
    each other in normalised space (often L1 < 0.05), so applying dedup
    would silently drop the most informative boundary anchors. Dedup is
    still applied to secondary anchors (steps 4-6) for archival diversity.

    Returns a list of dicts:
        {label, fog, ill, noi, verdict, metric, source_iter,
         selection_reason, rationale}
    """
    if not history:
        # Seed cases — used when there's no session data yet. Each has an
        # explicit selection_reason so the output is consistent.
        # Boundary-first ordering matches the system's primary purpose:
        # the cases nearest the PASS↔FAIL boundary appear first.
        seeds = [
            ("MARGINAL_BOUNDARY", "임계 안개 + 흐린 한낮 (경계)",
             35, 4500,  0.08, "MARGINAL",
             "REQ-1 임계 부근 — PASS↔FAIL 경계 직접 검증."),
            ("BOUNDARY_FAIL",   "짙은 안개 + 부분 흐림 (실패 경계)",
             60, 3000,  0.10, "FAIL",
             "색 대비 손실 — 침입자 식별 신뢰도 임계 미달 (실패측 경계)."),
            ("BOUNDARY_PASS",   "옅은 안개 + 흐린 한낮 (통과 경계)",
             25, 6000,  0.05, "PASS",
             "일상적 한국 산악 시계 — REQ-1 안전선 내부 (통과측 경계)."),
            ("WORST_CASE",      "극단 환경 (운용 한계 anchor)",
             90, 300,   0.55, "FAIL",
             "운용 한계 — 임무 수행 불가 마지노선."),
            ("BASELINE",        "맑은 한낮 baseline (정상 anchor)",
             5,  12000, 0.02, "PASS",
             "임무 기본 통과 조건 (baseline 회귀 테스트)."),
        ]
        out = []
        for kind, lab, f, i, n, v, rat in seeds[:max_cases]:
            # Seeds have no observed metric — synthesise a plausible band
            # for the tolerance so the export schema is still useful.
            seed_metric = 0.85 if v == "PASS" else 0.45 if v == "MARGINAL" else 0.15
            tol = _tolerance_for(v, seed_metric)
            reason = "SEED:" + kind + " — history 없음, 정형 시나리오 사용"
            out.append({
                "label": lab, "fog": f, "ill": i, "noi": n,
                "verdict": v, "metric": None, "source_iter": None,
                "selection_reason":    reason,
                "coverage_category":   _category_for(kind),
                "expected_verdict":    v,
                "expected_metric":     tol["expected_metric"],
                "metric_min":          tol["metric_min"],
                "metric_max":          tol["metric_max"],
                "acceptable_verdicts": tol["acceptable_verdicts"],
                "tolerance_pct":       15.0,
                "rationale":           rat,
            })
        return out

    pass_cases = [r for r in history if r.get("verdict") == "PASS"]
    marg_cases = [r for r in history if r.get("verdict") == "MARGINAL"]
    fail_cases = [r for r in history if r.get("verdict") == "FAIL"]

    picked = []
    seen   = []

    def _norm(r):
        return (float(r["fog"]) / 100.0,
                float(r["ill"]) / 15000.0,
                float(r["noi"]))

    def _too_close(c):
        for s in seen:
            if sum(abs(a - b) for a, b in zip(c, s)) < 0.15:
                return True
        return False

    def _add(r, reason, skip_dedup: bool = False):
        if len(picked) >= max_cases:
            return False
        c = _norm(r)
        if (not skip_dedup) and _too_close(c):
            return False
        seen.append(c)
        v = r.get("verdict", "PASS")
        m = round(float(r.get("metric", 0)), 3)
        tol = _tolerance_for(v, m)
        picked.append({
            "label":               _auto_label(r),
            "fog":                 round(float(r["fog"]), 1),
            "ill":                 round(float(r["ill"]), 0),
            "noi":                 round(float(r["noi"]), 3),
            "verdict":             v,
            "metric":              m,
            "source_iter":         int(r.get("iter", 0)) if r.get("iter") else None,
            "selection_reason":    reason,
            "coverage_category":   _category_for(reason),
            "expected_verdict":    v,
            "expected_metric":     tol["expected_metric"],
            "metric_min":          tol["metric_min"],
            "metric_max":          tol["metric_max"],
            "acceptable_verdicts": tol["acceptable_verdicts"],
            "tolerance_pct":       15.0,
            "rationale":           _auto_rationale(r),
        })
        return True

    # BOUNDARY-FIRST PRIORITY ORDER
    # ------------------------------
    # Boundary-search is the system's primary purpose, so cases that
    # directly verify boundary location come FIRST. Extreme anchors
    # (WORST_FAIL / BEST_PASS) are used as secondary regression anchors.
    #
    # Rationale: with N requested cases, this ordering gives the operator
    # the most boundary information per case. e.g.
    #   N=1 → MARGINAL (the case riding the boundary itself)
    #   N=3 → MARGINAL + BOUNDARY_FAIL + BOUNDARY_PASS (triangulate boundary)
    #   N=5 → above + WORST_FAIL + BEST_PASS (add operational anchors)
    #   N=10 → fill remaining with DIVERSITY for stress-axis coverage

    # 1. MARGINAL cases — literally on the boundary (highest boundary info)
    # Sort by mAP closest to the PASS threshold (0.50) so the most
    # boundary-relevant MARGINAL appears first when multiple exist.
    marg_sorted = sorted(marg_cases, key=lambda r: abs(float(r.get("metric", 0)) - 0.50))
    for r in marg_sorted:
        _add(r, "MARGINAL — PASS/FAIL 경계 위에 정확히 위치 (boundary 직접 검증)")

    # 2. BOUNDARY_FAIL — FAIL closest to PASS threshold (실패 측 경계 인접)
    #    skip_dedup: boundary roles must survive proximity to MARGINAL
    if fail_cases:
        _add(max(fail_cases, key=lambda r: float(r.get("metric", 0))),
             "BOUNDARY_FAIL — 실패 측 경계 인접 (PASS 임계 바로 위)",
             skip_dedup=True)

    # 3. BOUNDARY_PASS — PASS closest to FAIL threshold (통과 측 경계 인접)
    #    skip_dedup: same rationale as BOUNDARY_FAIL
    if pass_cases:
        _add(min(pass_cases, key=lambda r: float(r.get("metric", 1))),
             "BOUNDARY_PASS — 통과 측 경계 인접 (FAIL 임계 바로 위)",
             skip_dedup=True)

    # 4. WORST_FAIL — operational-limit anchor (secondary)
    if fail_cases:
        _add(min(fail_cases, key=lambda r: float(r.get("metric", 1))),
             "WORST_FAIL — 가장 낮은 mAP의 FAIL (운용 한계 anchor)")

    # 5. BEST_PASS — baseline anchor (secondary)
    if pass_cases:
        _add(max(pass_cases, key=lambda r: float(r.get("metric", 0))),
             "BEST_PASS — 가장 높은 mAP의 PASS (정상 회귀 anchor)")

    # 6. Diversity fill — remaining cases, prefer high-info first (by
    #    |metric − 0.5| ascending — closer to boundary still prioritised)
    rest = sorted(history, key=lambda r: abs(float(r.get("metric", 0)) - 0.5))
    for r in rest:
        if len(picked) >= max_cases:
            break
        v = r.get("verdict", "PASS")
        tag = ("DIVERSITY_FAIL" if v == "FAIL"
               else "DIVERSITY_PASS" if v == "PASS"
               else "DIVERSITY_MARG")
        _add(r, tag + " — 다양성 확보 (다른 stress 축 cover)")

    return picked


def _auto_label(r: dict) -> str:
    fog = float(r.get("fog", 0))
    ill = float(r.get("ill", 8000))
    noi = float(r.get("noi", 0))
    parts = []
    parts.append("짙은 안개" if fog >= 50 else "옅은 안개" if fog >= 20 else "맑음")
    parts.append("야간" if ill <= 1500 else "흐림" if ill <= 6000 else "정오 광량")
    if noi >= 0.25: parts.append("센서 노후")
    return " · ".join(parts)


def _auto_rationale(r: dict) -> str:
    v = r.get("verdict", "PASS")
    m = float(r.get("metric", 0))
    if v == "PASS":
        return f"임무 통과 조건 (mAP {m:.2f}) — 안전 envelope 회귀 테스트."
    if v == "MARGINAL":
        return f"PASS 경계 인접 (mAP {m:.2f}) — 추가 악화 시 운용 불가."
    return f"REQ-1 미달 (mAP {m:.2f}) — 보고 누락 위험 케이스."


def _format_test_cases_text(cases: list, requirement: str, method: str,
                            export_path: str = "") -> str:
    if not cases:
        return "(정규화된 테스트 케이스 없음 — 먼저 시뮬레이션을 실행하세요.)"
    lines = []
    # Header/coverage/criteria sections were removed per user request — the
    # panel now shows ONLY the case list. The selection methodology and
    # coverage breakdown still exist in code (selection_reason +
    # coverage_category fields on each case dict, and the JSON export) but
    # are no longer dumped to the UI text panel.
    lines.append(f"▣ 정규화된 테스트 케이스 ({len(cases)}건)")
    lines.append("")

    for i, c in enumerate(cases, 1):
        verdict = c.get("verdict", "?")
        metric  = c.get("metric")
        metric_str = f"  (mAP={metric:.3f})" if isinstance(metric, (int, float)) else ""
        cat = c.get("coverage_category", "다양성")
        lines.append(f"[Case {i}] [{cat}] {c.get('label', '(unnamed)')}  →  예상: {verdict}{metric_str}")
        lines.append(f"   환경:  fog {c.get('fog', 0):.0f} %,   "
                     f"illum {c.get('ill', 0):.0f} lx,   "
                     f"noise {c.get('noi', 0):.2f}")

        # The explicit selection reason — always shown, never overridden by LLM
        reason = (c.get("selection_reason") or "").strip()
        if reason:
            src = c.get("source_iter")
            src_str = f" — iter {src}" if src else ""
            lines.append(f"   선정 기준: {reason}{src_str}")

        # Tolerance for re-run comparison — enables auto-regression checking
        accept = c.get("acceptable_verdicts", [verdict])
        mn = c.get("metric_min");  mx = c.get("metric_max")
        if isinstance(mn, (int, float)) and isinstance(mx, (int, float)):
            accept_str = " / ".join(accept)
            lines.append(f"   재실행 허용 범위: verdict ∈ {{{accept_str}}},  "
                         f"mAP ∈ [{mn:.2f}, {mx:.2f}]")

        # The operational rationale — LLM-enriched if available, rule fallback
        rationale = (c.get("rationale") or "").strip()
        if rationale:
            lines.append(f"   임무 영향: {rationale}")
        lines.append("")

    # Footer — show export path so the operator knows where the suite lives
    if export_path:
        lines.append(f"💾 저장됨: {export_path}")
        lines.append("   ↳ 새 시스템 버전에서 이 JSON을 로드하여 자동 회귀 검증 가능")

    return "\n".join(lines)


def summarize_mission(history_json: str) -> dict:
    """LLM-driven executive summary of the entire counterfactual history.

    Calls the Narrator's MissionSummary signature. Returns a dict with:
        executive_summary, failure_boundary, security_implications,
        recommendations, error (optional)

    Falls back to a deterministic template if DSPy / LLM is unavailable.
    """
    try:
        history = json.loads(history_json) if history_json else []
    except Exception as exc:                          # noqa: BLE001
        return {"error": f"bad history JSON: {exc}"}

    if not history:
        return {"error": "no history yet"}

    n_pass = sum(1 for h in history if h.get("verdict") == "PASS")
    n_marg = sum(1 for h in history if h.get("verdict") == "MARGINAL")
    n_fail = sum(1 for h in history if h.get("verdict") == "FAIL")

    boundary_findings = _summarise_boundary(history)
    shap_payload = _compute_shap_payload([
        {"fog": h["fog"], "ill": h["ill"], "noi": h["noi"],
         "metric": h.get("metric", 0.0)}
        for h in history
    ])
    shap_dominant = _dominant_from_payload(shap_payload)

    # Compact history JSON for the LLM (drop verbose fields)
    compact = [{
        "iter":    int(h.get("iter", i + 1)),
        "fog":     round(float(h["fog"]), 1),
        "ill":     round(float(h["ill"]), 0),
        "noi":     round(float(h["noi"]), 3),
        "metric":  round(float(h.get("metric", 0.0)), 3),
        "mAP_p":   round(float(h.get("metric_person",  0.0)), 3),
        "mAP_v":   round(float(h.get("metric_vehicle", 0.0)), 3),
        "verdict": h.get("verdict", "UNKNOWN"),
    } for i, h in enumerate(history)]

    try:
        from dspy_pipeline.orchestrator import _build_lm
        from dspy_pipeline.narrator     import Narrator, load_mission_context
        lm = _build_lm(None)
        if lm is None:
            raise RuntimeError("LM not available")
        narr = Narrator()
        import dspy
        with dspy.context(lm=lm):
            pred = narr.summarise_mission(
                mission_context   = load_mission_context(),
                iteration_history = json.dumps(compact, ensure_ascii=False),
                boundary_findings = boundary_findings,
                shap_dominant     = shap_dominant,
            )
        return {
            "executive_summary":      getattr(pred, "executive_summary",     "") or "",
            "failure_boundary":       getattr(pred, "failure_boundary_text", "") or "",
            "security_implications":  getattr(pred, "security_implications", "") or "",
            "recommendations":        getattr(pred, "recommendations",       "") or "",
            "n_pass":                 n_pass,
            "n_marg":                 n_marg,
            "n_fail":                 n_fail,
        }
    except Exception as exc:                          # noqa: BLE001
        return {
            "executive_summary":     (f"총 {len(history)}회 검증 — "
                                      f"PASS {n_pass}, MARGINAL {n_marg}, FAIL {n_fail}. "
                                      f"지배 변수: {shap_dominant}."),
            "failure_boundary":      boundary_findings,
            "security_implications": "- LLM 요약 불가 (오프라인). 수치 요약만 제공됨.",
            "recommendations":       "- 환경 키 또는 LLM 가용성 점검 후 재시도 권장.",
            "n_pass":                n_pass,
            "n_marg":                n_marg,
            "n_fail":                n_fail,
            "error":                 str(exc),
        }


def _summarise_boundary(history) -> str:
    fails = [h for h in history if h.get("verdict") == "FAIL"]
    passes = [h for h in history if h.get("verdict") == "PASS"]
    if not fails and not passes:
        return "no boundary observed"
    if not fails:
        return "no FAIL observed — all cases within safe envelope"
    if not passes:
        return "no PASS observed — all cases broken"

    def _stats(rows, key):
        vals = [float(r[key]) for r in rows]
        return min(vals), max(vals)

    fog_f = _stats(fails,  "fog")
    fog_p = _stats(passes, "fog")
    ill_f = _stats(fails,  "ill")
    ill_p = _stats(passes, "ill")
    noi_f = _stats(fails,  "noi")
    noi_p = _stats(passes, "noi")
    return (
        f"FAIL envelope ~ fog {fog_f[0]:.0f}–{fog_f[1]:.0f}%, "
        f"illum {ill_f[0]:.0f}–{ill_f[1]:.0f} lx, noise {noi_f[0]:.2f}–{noi_f[1]:.2f}; "
        f"PASS envelope ~ fog {fog_p[0]:.0f}–{fog_p[1]:.0f}%, "
        f"illum {ill_p[0]:.0f}–{ill_p[1]:.0f} lx, noise {noi_p[0]:.2f}–{noi_p[1]:.2f}"
    )


def _dominant_from_payload(payload) -> str:
    if not payload:
        return "분석 데이터 부족"
    items = payload.get("global_feature_importance") or []
    if not items:
        return "unknown"
    items = sorted(items, key=lambda x: -abs(float(x.get("importance", 0.0))))
    name_map = {
        "fog_density_percent": "안개 밀도",
        "illumination_lux":    "조도",
        "camera_noise_level":  "센서 잡음",
    }
    return name_map.get(items[0]["name"], items[0]["name"])


def save_edge_case(record_json: str, edge_cases_path: str = "data/edge_cases.json",
                   min_distance: float = 0.15) -> dict:
    """Persist a FAIL/MARGINAL case to data/edge_cases.json (project-relative).

    Implements an "optimal edge cases only" filter: a new case is appended
    only if its L1 distance in normalized (fog, ill, noi) space exceeds
    `min_distance` from every previously-saved edge case. Returns a dict
    with {saved: bool, n_total: int, reason: str}.

    Default 0.15 — validated as the centre of the empirical stability plateau
    (suite_size invariant for theta in [0.10, 0.20] over 101 raw cases).
    See experiments/threshold_analysis/ for the sensitivity analysis.

    record_json schema (sent from MATLAB):
        {fog, ill, noi, metric, metric_person, metric_vehicle, verdict,
         ngt, ndet, ntp, iter, mode}
    """
    try:
        rec = json.loads(record_json)
    except Exception as exc:                          # noqa: BLE001
        return {"saved": False, "reason": f"bad record JSON: {exc}", "n_total": 0}

    verdict = str(rec.get("verdict", ""))
    if verdict not in ("FAIL", "MARGINAL"):
        return {"saved": False, "reason": f"verdict={verdict} (only FAIL/MARGINAL saved)", "n_total": 0}

    path = _PROJECT_ROOT / edge_cases_path
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    new_norm = (
        float(rec.get("fog", 0)) / 100.0,
        float(rec.get("ill", 0)) / 15000.0,
        float(rec.get("noi", 0)) / 1.0,
    )

    # "Optimal" filter — skip near-duplicates of cases already saved
    for old in existing:
        old_norm = (
            float(old.get("fog", 0)) / 100.0,
            float(old.get("ill", 0)) / 15000.0,
            float(old.get("noi", 0)) / 1.0,
        )
        l1 = sum(abs(a - b) for a, b in zip(new_norm, old_norm))
        if l1 < min_distance:
            # Keep the more informative of the two (lower mAP wins for FAIL,
            # closer-to-PASS-threshold wins for MARGINAL — proxy: pick lower)
            if float(rec.get("metric", 0.0)) < float(old.get("metric", 1.0)):
                old.update({
                    "fog": rec["fog"], "ill": rec["ill"], "noi": rec["noi"],
                    "metric": rec.get("metric", 0.0),
                    "metric_person":  rec.get("metric_person",  0.0),
                    "metric_vehicle": rec.get("metric_vehicle", 0.0),
                    "verdict":        rec.get("verdict"),
                    "iter":           rec.get("iter"),
                    "mode":           rec.get("mode"),
                    "saved_at":       _now_iso(),
                })
                path.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                                encoding="utf-8")
                return {"saved": True, "reason": "replaced near-duplicate with stronger case",
                        "n_total": len(existing)}
            return {"saved": False,
                    "reason": f"near-duplicate (L1={l1:.3f} < {min_distance})",
                    "n_total": len(existing)}

    rec_out = {
        "fog":            float(rec.get("fog", 0)),
        "ill":            float(rec.get("ill", 0)),
        "noi":            float(rec.get("noi", 0)),
        "metric":         float(rec.get("metric", 0.0)),
        "metric_person":  float(rec.get("metric_person",  0.0)),
        "metric_vehicle": float(rec.get("metric_vehicle", 0.0)),
        "verdict":        verdict,
        "iter":           int(rec.get("iter", len(existing) + 1)),
        "mode":           str(rec.get("mode", "")),
        "saved_at":       _now_iso(),
    }
    existing.append(rec_out)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return {"saved": True, "reason": "new edge case", "n_total": len(existing)}


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# Smoke test
if __name__ == "__main__":
    demo = [
        {"fog": 10, "ill": 8000, "noi": 0.05, "f1": 0.92, "passed": True},
        {"fog": 40, "ill": 4000, "noi": 0.20, "f1": 0.88, "passed": True},
        {"fog": 70, "ill": 2000, "noi": 0.40, "f1": 0.31, "passed": False},
    ]
    print(json.dumps(next_case_from_history(json.dumps(demo)),
                     indent=2, ensure_ascii=False))
