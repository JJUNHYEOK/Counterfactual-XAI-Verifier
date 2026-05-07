#!/usr/bin/env python3
"""app_demo.py — Border Mountain Surveillance UAV verification demo.

Streamlit UI with three sections (per user spec):
  1. Simulation viewer  — MATLAB visualizer PNG (left: 3D mountain + UAV,
                          right: top-down EO camera with intruder bboxes)
  2. Scenario comparison — previous ↔ current with natural-language
                            edge-case narratives (LLM, security-report tone)
  3. Mission summary    — full LLM-generated executive report at the bottom

Run:
    streamlit run app_demo.py

Sidebar:
  - [▶ Next Step]   — execute one boundary-search iteration
  - [↻ Reset]       — clear session state
  - sim mode (engine / mock)
  - threshold knobs (mAP, continuity)

Note: run with the project venv active so DSPy / matlab.engine / streamlit
      / xgboost / shap are all available.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "국경 산악 감시 UAV 검증",
    page_icon   = "🛰️",
    layout      = "wide",
)

ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Imports that need streamlit's lazy loading
# ─────────────────────────────────────────────────────────────────────────────

from dspy_pipeline.orchestrator import (
    init_session, run_one_step, decide_next_scenario,
)
from dspy_pipeline.narrator import (
    Narrator, load_mission_context, qualitative_env, boundary_status,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state bootstrapping
# ─────────────────────────────────────────────────────────────────────────────

def _seed_scenario() -> dict:
    return {
        "scenario_id": "scenario_seed_001",
        "environment_parameters": {
            "fog_density_percent": 0.0,
            "illumination_lux":    8000.0,
            "camera_noise_level":  0.0,
        },
        "decision_mode":     "seed",
        "target_hypothesis": "정상 정찰 조건 — baseline 측정.",
        "dspy_analysis":     "맑은 정오, 잡음 없음. 침입자 5명(사람3+차량2) 모두 탐지 기대.",
    }


@st.cache_resource(show_spinner="MATLAB 엔진과 LLM 초기화 중 (15~30초)…")
def _boot_session(sim_mode: str):
    sess = init_session(model_dir=str(ROOT), sim_mode=sim_mode)
    return sess


@st.cache_resource(show_spinner="LLM Narrator 초기화…")
def _boot_narrator():
    return Narrator()


def _ensure_state():
    if "next_scenario" not in st.session_state:
        st.session_state.next_scenario = _seed_scenario()
    if "edge_narratives" not in st.session_state:
        st.session_state.edge_narratives = []           # parallel to history
    if "comparison" not in st.session_state:
        st.session_state.comparison = ""
    if "summary" not in st.session_state:
        st.session_state.summary = None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🛰️ SENTINEL")
st.sidebar.caption("국경 산악 감시 UAV — 자동 시나리오 검증 시스템")
sim_mode = st.sidebar.selectbox("시뮬 백엔드", ["engine", "mock"], index=0,
    help="engine = MATLAB Simulink (실제 모델), mock = 빠른 분석 proxy")

session = _boot_session(sim_mode)
narrator = _boot_narrator()
mission_ctx = load_mission_context()
_ensure_state()

st.sidebar.markdown("---")
col_a, col_b = st.sidebar.columns(2)
run_clicked  = col_a.button("▶ 다음 정찰", type="primary", use_container_width=True,
    help="UAV가 한 단계 앞으로 정찰 시뮬레이션을 수행합니다")
reset_clicked = col_b.button("↻ 초기화", use_container_width=True,
    help="모든 시나리오와 임무 이력 초기화")

if reset_clicked:
    _boot_session.clear()
    st.session_state.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 임무 진행")
st.sidebar.caption(f"검증된 시나리오: **{len(session.history)} 회**")
if session.history:
    last = session.history[-1]
    badge = "🟢 임무 성공" if last['all_passed'] else "🔴 임무 실패"
    st.sidebar.markdown(f"**최근 판정**: {badge}")
    st.sidebar.metric("탐지 정확도 (mAP50)", f"{last['map50']:.3f}",
        delta=f"{last['map50'] - 0.50:+.3f}",
        delta_color="normal" if last['all_passed'] else "inverse")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 다음 시나리오")
ne = st.session_state.next_scenario["environment_parameters"]
mode = st.session_state.next_scenario.get('decision_mode', 'seed')
mode_kr = {
    "seed":             "🟦 정상 정찰 (시드)",
    "boundary_push":    "🔥 환경 가혹화 (PUSH)",
    "boundary_recover": "💨 조건 완화 (RECOVER)",
    "llm_explore":      "🧠 LLM 적대적 생성",
    "rule_push_fallback":"⚙ 규칙 기반 push",
    "rule_relax":       "⚙ 규칙 기반 relax",
    "bisect":           "🎯 경계 좁히기",
}.get(mode, mode)
st.sidebar.markdown(f"**모드**: {mode_kr}")
st.sidebar.write(f"- 안개: `{ne.get('fog_density_percent', 0):.1f}%`")
st.sidebar.write(f"- 조도: `{ne.get('illumination_lux', 0):.0f} lx`")
st.sidebar.write(f"- 잡음: `{ne.get('camera_noise_level', 0):.3f}`")


# ─────────────────────────────────────────────────────────────────────────────
# Header — mission context (always visible)
# ─────────────────────────────────────────────────────────────────────────────

st.title("🛰️ SENTINEL — UAV 임무 검증 시스템")
st.caption("Counterfactual XAI Verifier · DSPy + SHAP + LLM 기반 자동 시나리오 생성")
mc = json.loads(mission_ctx)
with st.container(border=True):
    h_col1, h_col2, h_col3 = st.columns([2, 2, 1])
    with h_col1:
        st.markdown(f"### 🎯 작전 임무")
        st.markdown(f"**{mc['mission_title']}**")
        st.caption(mc["objective"])
    with h_col2:
        st.markdown(f"### 🚨 탐지 대상")
        st.markdown(f"**{' · '.join(mc['detection_targets'])}** (비인가 침입자)")
        st.caption(f"임무 실패: 환경 변조로 인한 침입자 식별 누락")
    with h_col3:
        st.markdown("### ⚖ 검증 기준")
        st.caption(f"REQ-1 탐지 정확도 ≥ {mc['requirements']['REQ_1']['threshold']}")
        st.caption(f"REQ-3 연속 미탐 ≤ {mc['requirements']['REQ_3']['threshold']} 프레임")


# ─────────────────────────────────────────────────────────────────────────────
# Run-step handler
# ─────────────────────────────────────────────────────────────────────────────

if run_clicked:
    step_no = len(session.history) + 1
    viz_png = str(ASSETS_DIR / f"sim_step_{step_no:03d}.png")

    with st.spinner(f"📡 정찰 시나리오 #{step_no} 시뮬레이션 수행 중…"):
        rec = run_one_step(session, st.session_state.next_scenario, viz_png_path=viz_png)

    # Generate edge-case narrative for this new step
    with st.spinner("🛰️ 임무 분석관 LLM 보고서 작성 중…"):
        try:
            import dspy
            from dspy_pipeline.orchestrator import get_lm
            qenv = qualitative_env(rec)
            bstatus = boundary_status(rec, session.last_pass_env, session.last_fail_env)
            with dspy.context(lm=get_lm()):
                ec_pred = narrator.describe_edge_case(
                    mission_context = mission_ctx,
                    env_summary     = json.dumps(qenv, ensure_ascii=False),
                    sim_outcome     = ("PASS" if rec["all_passed"] else "FAIL")
                                      + f", violated_count={rec['violated_count']}",
                    boundary_status = bstatus,
                )
            narrative = ec_pred.edge_case_narrative
        except Exception as exc:
            narrative = f"(자연어 생성 실패: {exc})"
        st.session_state.edge_narratives.append(narrative)

    # Generate scenario-comparison narrative if we have a prior step
    if len(session.history) >= 2:
        prev = session.history[-2]
        curr = session.history[-1]
        with st.spinner("📊 이전 ↔ 현재 시나리오 비교 분석…"):
            try:
                import dspy
                from dspy_pipeline.orchestrator import get_lm
                with dspy.context(lm=get_lm()):
                    cmp_pred = narrator.compare_scenarios(
                        mission_context  = mission_ctx,
                        previous_summary = json.dumps({
                            "env":        qualitative_env(prev),
                            "result":     "PASS" if prev["all_passed"] else "FAIL",
                            "narrative":  st.session_state.edge_narratives[-2],
                        }, ensure_ascii=False),
                        current_summary  = json.dumps({
                            "env":        qualitative_env(curr),
                            "result":     "PASS" if curr["all_passed"] else "FAIL",
                            "narrative":  st.session_state.edge_narratives[-1],
                        }, ensure_ascii=False),
                    )
                st.session_state.comparison = cmp_pred.comparison_narrative
            except Exception as exc:
                st.session_state.comparison = f"(비교 생성 실패: {exc})"

    # Decide next scenario
    with st.spinner("🧠 적대적 환경 시나리오 자동 생성 중 (DSPy + SHAP)…"):
        try:
            st.session_state.next_scenario = decide_next_scenario(session)
        except Exception as exc:
            st.error(f"다음 시나리오 결정 실패: {exc}")

    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Simulation viewer (largest)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 📷 SECTION 1 — 시뮬레이션 화면 (MATLAB/Simulink)")

if not session.history:
    st.info("👈 사이드바의 **▶ Next Step**을 눌러 첫 시뮬레이션을 실행하세요. "
            "정상 조건(시드 시나리오)에서 시작합니다.")
else:
    last = session.history[-1]
    viz_path = ASSETS_DIR / f"sim_step_{last['step']:03d}.png"
    if viz_path.exists():
        st.image(str(viz_path),
                 caption=f"Step {last['step']} — fog={last['fog_density_percent']:.0f}%, "
                         f"illum={last['illumination_lux']:.0f} lx, "
                         f"noise={last['camera_noise_level']:.2f}  ({sim_mode} mode)",
                 use_container_width=True)
    else:
        st.warning(f"PNG 캡처 없음: {viz_path}. (mock 모드에서는 캡처 미생성)")
    if sim_mode == "engine":
        st.caption("💡 MATLAB 창에서 실시간 애니메이션이 함께 재생됩니다.")

    # Live metrics row (under the image)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("mAP50",            f"{last['map50']:.3f}",
              delta=f"{last['map50'] - 0.50:+.3f} vs th",
              delta_color="normal" if last['all_passed'] else "inverse")
    m2.metric("연속 미탐 (frame)", f"{last['worst_run']}",
              delta=f"≤ 3 임계", delta_color="off")
    m3.metric("위반 REQ 수",        f"{last['violated_count']} / 2",
              delta=("PASS" if last['all_passed'] else "FAIL"),
              delta_color="normal" if last['all_passed'] else "inverse")
    m4.metric("실행 시간",          f"{last.get('elapsed_s', 0):.1f}s")
    m5.metric("결정 모드",          last.get('decision_mode', 'seed'))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Scenario comparison
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 📊 SECTION 2 — 시나리오 비교 (이전 ↔ 현재)")

if len(session.history) == 0:
    st.caption("실행 후 표시됩니다.")
else:
    cols = st.columns(2)
    prev = session.history[-2] if len(session.history) >= 2 else None
    curr = session.history[-1]
    prev_narr = st.session_state.edge_narratives[-2] if len(st.session_state.edge_narratives) >= 2 else None
    curr_narr = st.session_state.edge_narratives[-1] if st.session_state.edge_narratives else None

    def _render_card(col, rec, narrative, label):
        with col:
            with st.container(border=True):
                if rec is None:
                    st.markdown(f"### {label}")
                    st.caption("(이전 시나리오 없음)")
                    return
                status = "PASS ✅" if rec["all_passed"] else "FAIL ❌"
                st.markdown(f"### {label}  ·  Step {rec['step']}  ·  {status}")
                qe = qualitative_env(rec)
                st.markdown(
                    f"**환경 요약**: 안개 *{qe['fog_level']}* · 조도 *{qe['illum_level']}* · "
                    f"센서잡음 *{qe['noise_level']}*"
                )
                st.markdown(
                    f"**탐지 성능**: *{qe['mAP_band']}*  ·  연속 미탐 *{rec['worst_run']} frame*"
                )
                st.markdown("**🛡️ 엣지 케이스 분석 (자연어)**")
                if narrative:
                    st.markdown(f"> {narrative}")
                else:
                    st.caption("(생성 대기)")

    _render_card(cols[0], prev, prev_narr, "🔹 이전 시나리오")
    _render_card(cols[1], curr, curr_narr, "🔸 현재 시나리오")

    if st.session_state.comparison:
        with st.container(border=True):
            st.markdown("**📍 비교 분석**")
            st.markdown(st.session_state.comparison)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Mission summary (LLM)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 📋 SECTION 3 — 종합 결과 요약 (LLM 자연어 보고서)")

if len(session.history) < 2:
    st.caption("최소 2 step 이상 진행되면 종합 보고서가 자동 생성됩니다.")
else:
    if st.button("📝 종합 보고서 생성/갱신", use_container_width=False):
        with st.spinner("LLM이 임무 보고서 작성 중…"):
            try:
                compressed = [
                    {
                        "iter":    h["step"],
                        "env":     qualitative_env(h),
                        "result":  "PASS" if h["all_passed"] else "FAIL",
                        "mAP":     round(h["map50"], 3),
                    }
                    for h in session.history
                ]
                if session.boundary_log:
                    last_b = session.boundary_log[-1]
                    bf = (f"PASS anchor: {qualitative_env({**last_b['pass_env'], 'map50':0.6})}; "
                          f"FAIL anchor: {qualitative_env({**last_b['fail_env'], 'map50':0.4})}; "
                          f"L1 width keys={list(last_b['width'].keys())}")
                else:
                    bf = "아직 PASS↔FAIL 전환 미발견"
                if session.last_shap and session.last_shap.get("global_feature_importance"):
                    top = session.last_shap["global_feature_importance"][0]
                    sd = f"{top['name']} (importance={top['importance']:.2f})"
                else:
                    sd = "SHAP 미사용"
                import dspy
                from dspy_pipeline.orchestrator import get_lm
                with dspy.context(lm=get_lm()):
                    pred = narrator.summarise_mission(
                        mission_context  = mission_ctx,
                        iteration_history = json.dumps(compressed, ensure_ascii=False),
                        boundary_findings = bf,
                        shap_dominant    = sd,
                    )
                st.session_state.summary = {
                    "executive":      pred.executive_summary,
                    "boundary":       pred.failure_boundary_text,
                    "implications":   pred.security_implications,
                    "recommendations": pred.recommendations,
                }
            except Exception as exc:
                st.error(f"요약 생성 실패: {exc}")

    if st.session_state.summary:
        with st.container(border=True):
            st.markdown("### 📌 Executive Summary")
            st.markdown(st.session_state.summary["executive"])
            st.markdown("### 🎯 발견된 임무 실패 경계 (Failure Boundary)")
            st.markdown(st.session_state.summary["boundary"])
            st.markdown("### 🛡️ 안보적 시사점")
            st.markdown(st.session_state.summary["implications"])
            st.markdown("### ✅ 운용 권고사항")
            st.markdown(st.session_state.summary["recommendations"])

st.markdown("---")
st.caption("© Counterfactual-XAI-Verifier · DSPy + MATLAB Simulink + SHAP 기반 boundary search 데모")
