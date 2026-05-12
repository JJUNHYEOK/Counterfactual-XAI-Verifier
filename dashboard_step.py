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
    """Decide the next counterfactual case from the MATLAB dashboard's
    run history. Returns a dict (see module docstring).

    Phase 5: every branch consults the DSPy LLM with SHAP signals as XAI
    context. The deterministic bisect/rule policies are the fallback if
    the LLM call fails."""
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

    # ─── Determine the deterministic fallback for this branch first ────
    if last_passed and last_fail is not None:
        rule_env = _bisect(last_env, last_fail)
        rule_mode, rule_analysis = "boundary_push", "PASS → bisect 65% toward FAIL anchor (rule)"
    elif (not last_passed) and last_pass is not None:
        rule_env = _bisect(last_env, last_pass)
        rule_mode, rule_analysis = "boundary_recover", "FAIL → bisect 65% toward PASS anchor (rule)"
    elif last_passed:
        rule_env = _rule_push(last_env)
        rule_mode, rule_analysis = "rule_push", "PASS / no FAIL anchor → push (rule)"
    else:
        rule_env = _rule_relax(last_env)
        rule_mode, rule_analysis = "rule_relax", "FAIL / no PASS anchor → relax (rule)"

    # ─── LLM consultation across all branches with SHAP context ────────
    llm_result = _llm_decide(records, last_env, last, last_pass, last_fail,
                              rule_env, rule_mode)
    if llm_result is not None:
        env, mode, analysis = llm_result
        return _envelope(env, mode, analysis)

    # LLM unavailable → deterministic fallback
    return _envelope(rule_env, rule_mode, rule_analysis)


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
                   min_distance: float = 0.12) -> dict:
    """Persist a FAIL/MARGINAL case to data/edge_cases.json (project-relative).

    Implements an "optimal edge cases only" filter: a new case is appended
    only if its L1 distance in normalized (fog, ill, noi) space exceeds
    `min_distance` from every previously-saved edge case. Returns a dict
    with {saved: bool, n_total: int, reason: str}.

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
