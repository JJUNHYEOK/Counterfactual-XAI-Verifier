#!/usr/bin/env python3
"""app_demo_unreal.py — Border-surveillance UAV demo with photoreal Unreal frames.

Phase B-4 of the Unreal integration. Layout:

    SECTION 1 (top)    — two camera panels:
                          left:  3rd-person chase view (Unreal)
                          right: 1st-person nadir view (Unreal) with YOLO bboxes
    SECTION 2 (mid)    — detection table (class, conf, bbox) + scenario metrics
    SECTION 3 (bottom) — LLM narration (security-report tone) of what was found

Differences vs the legacy app_demo.py (synthetic-only):
    * Real photoreal images via mountain_uav_unreal.slx + Unreal Engine
    * Real YOLOv8s object detection (no oracle bboxes)
    * Each "Run Step" takes ~30-90 s due to Unreal sim cost — be patient

Run:
    streamlit run app_demo_unreal.py
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

st.set_page_config(
    page_title = "국경 산악 감시 — Unreal Photoreal Demo",
    page_icon  = "🛰️",
    layout     = "wide",
)

ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (avoid double-loading huge libs in Streamlit reruns)
# ─────────────────────────────────────────────────────────────────────────────

from dspy_pipeline.unreal_bridge import UnrealSimulinkBridge
from dspy_pipeline.yolo_detector import UAVDetector
from dspy_pipeline.narrator      import (
    Narrator, load_mission_context, qualitative_env, boundary_status,
)


# ─────────────────────────────────────────────────────────────────────────────
# Resource bootstrap (cached so we don't restart MATLAB / reload YOLO)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="MATLAB Engine + Unreal 시작 (30~60초)…")
def boot_unreal_bridge():
    b = UnrealSimulinkBridge(model_dir=str(ROOT))
    b.start()
    return b


@st.cache_resource(show_spinner="YOLOv8s 로드 중…")
def boot_yolo(weights: str = "yolov8s.pt"):
    return UAVDetector(weights=weights)


@st.cache_resource(show_spinner="LLM Narrator 초기화…")
def boot_narrator():
    return Narrator()


# Build DSPy LM once and cache it — Streamlit creates a new thread per rerun,
# and DSPy 3.x's settings.configure() is thread-bound (raises RuntimeError if
# a different thread tries to use the configured LM). The fix: build the LM
# outside any thread-bound configure() call, and wrap each LLM invocation in
# `with dspy.context(lm=lm):` so it works from any rerun thread.
@st.cache_resource(show_spinner=False)
def get_dspy_lm():
    import os, dspy
    openai_key    = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if openai_key:
        model, api_key = "openai/gpt-4o-mini", openai_key
    elif anthropic_key:
        model, api_key = "anthropic/claude-haiku-4-5-20251001", anthropic_key
    else:
        return None
    return dspy.LM(model, api_key=api_key, temperature=0.4, max_tokens=1024)


def _ensure_lm_configured():
    if "_lm_ready" in st.session_state:
        return
    if get_dspy_lm() is None:
        st.error("OPENAI_API_KEY / ANTHROPIC_API_KEY 가 .env에 없음 — LLM narration 비활성")
        st.session_state._lm_ready = False
    else:
        st.session_state._lm_ready = True


def _ensure_state():
    defaults = {
        "step":          0,
        "last_fp_png":   None,
        "last_tp_png":   None,
        "last_overlay":  None,
        "last_dets":     [],
        "last_summary":  "",
        "last_narrative": None,
        "history":        [],          # list of step records
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🛰️ Unreal Demo 컨트롤")
st.sidebar.caption("photoreal Unreal Engine + YOLOv8s + LLM 자연어 보고")

conf_threshold = st.sidebar.slider(
    "YOLO 신뢰도 임계", 0.05, 0.90, 0.15, step=0.05,
    help="낮출수록 더 많은 후보 검출 (FP 증가)",
)
narrate = st.sidebar.checkbox("LLM 자연어 보고", value=True)

run_btn   = st.sidebar.button("▶ Run Unreal Step", type="primary", width="stretch")
reset_btn = st.sidebar.button("↻ Reset",            width="stretch")

if reset_btn:
    boot_unreal_bridge.clear()
    st.session_state.clear()
    st.rerun()

_ensure_state()
_ensure_lm_configured()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**진행**: {st.session_state.step} step 완료")
if st.session_state.last_dets:
    n_p = sum(1 for d in st.session_state.last_dets if d.cls_name == "person")
    n_v = sum(1 for d in st.session_state.last_dets if d.cls_name == "vehicle")
    st.sidebar.metric("최근 검출", f"{n_p} 사람 + {n_v} 차량")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🛰️ 국경 산악 감시 UAV — Unreal 통합 데모")
mc_raw = load_mission_context()
mc     = json.loads(mc_raw)
with st.container(border=True):
    cols = st.columns([2, 2, 1])
    with cols[0]:
        st.markdown(f"**임무**: {mc['mission_title']}")
        st.caption(mc["objective"])
    with cols[1]:
        st.markdown(f"**탐지 대상**: {' · '.join(mc['detection_targets'])}")
        st.caption(f"검출기: YOLOv8s (real ML, no oracle)")
    with cols[2]:
        st.markdown("**렌더러**")
        st.caption("Unreal Engine 5.5 photoreal")


# ─────────────────────────────────────────────────────────────────────────────
# Run-step handler
# ─────────────────────────────────────────────────────────────────────────────

bridge   = boot_unreal_bridge()
detector = boot_yolo()
narrator = boot_narrator() if narrate else None

if run_btn:
    st.session_state.step += 1
    n  = st.session_state.step
    fp = ASSETS_DIR / f"unreal_fp_{n:03d}.png"
    tp = ASSETS_DIR / f"unreal_tp_{n:03d}.png"

    with st.spinner(f"Step {n} — Unreal sim 실행 중 (~30~60초)…"):
        result = bridge.run_step(
            env_params  = {"fog_density_percent": 0,
                           "illumination_lux":   8000,
                           "camera_noise_level": 0},
            save_paths  = {"first_person": str(fp), "third_person": str(tp)},
            detector    = detector,
            detect_conf = conf_threshold,
        )

    st.session_state.last_fp_png  = result.fp_png
    st.session_state.last_tp_png  = result.tp_png
    st.session_state.last_dets    = result.detections
    st.session_state.last_summary = result.summary()

    # Build YOLO overlay for the 1st-person frame
    overlay_path = None
    if result.fp_png:
        try:
            import cv2
            img = cv2.imread(result.fp_png)
            if img is not None:
                overlay = UAVDetector.overlay_detections(img, result.detections)
                overlay_path = result.fp_png.replace(".png", "_yolo.png")
                cv2.imwrite(overlay_path, overlay)
        except Exception as e:
            st.warning(f"Overlay 생성 실패: {e}")
    st.session_state.last_overlay = overlay_path

    # LLM narration of what was detected
    if narrator and st.session_state.get("_lm_ready"):
        with st.spinner("LLM 자연어 보고 생성…"):
            try:
                import dspy
                lm = get_dspy_lm()

                # Build a synthetic env summary for narration
                qenv = {
                    "fog_level":      "맑음",
                    "illum_level":    "주간 일조",
                    "noise_level":    "잡음 거의 없음",
                    "mAP_band":       "n/a (real YOLO)",
                    "all_passed":     result.n_persons + result.n_vehicles >= 4,
                    "violated_count": max(0, 5 - result.n_persons - result.n_vehicles),
                    "worst_run":      0,
                }
                # dspy.context() is thread-safe across Streamlit reruns
                with dspy.context(lm=lm):
                    ec_pred = narrator.describe_edge_case(
                        mission_context = mc_raw,
                        env_summary     = json.dumps(qenv, ensure_ascii=False),
                        sim_outcome     = (
                            f"YOLOv8s가 사람 {result.n_persons}명, "
                            f"차량 {result.n_vehicles}대 식별. "
                            f"임무 기대치(사람3·차량2 모두 식별) 대비 "
                            f"{'달성' if result.n_persons + result.n_vehicles >= 5 else '부분 달성/실패'}."
                        ),
                        boundary_status = "real_detection_phase_b",
                    )
                st.session_state.last_narrative = ec_pred.edge_case_narrative
            except Exception as e:
                st.session_state.last_narrative = f"(LLM 생성 실패: {e})"

    st.session_state.history.append({
        "step":      n,
        "n_persons": result.n_persons,
        "n_vehicles":result.n_vehicles,
        "elapsed_s": result.elapsed_s,
    })
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Two camera panels
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 📷 SECTION 1 — UAV 1인칭 nadir 카메라 + YOLO 검출")

if st.session_state.last_fp_png is None:
    st.info("👈 사이드바의 **▶ Run Unreal Step** 클릭 — Unreal 창이 뜨면서 시뮬레이션이 실행됩니다.")
else:
    # Single nadir panel — narrow centered, fixed width to keep aspect natural
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.markdown("**드론 1인칭 nadir view** — UAV 본체 시점에서 거의 수직 아래 + YOLO bbox")
        display_img = st.session_state.last_overlay or st.session_state.last_fp_png
        if display_img:
            st.image(display_img, width=640)   # explicit native PNG width (640×360)
        else:
            st.warning("1st-person PNG 없음")
        st.caption(
            "💡 **3인칭 chase view 는 MATLAB 의 Unreal 창** (별도 데스크톱 창) 에서 직접 보세요. "
            "SceneCfg.FreeCamera 가 드론 뒤에서 내려다보도록 설정되어 있습니다."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Detection table + metrics
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 📊 SECTION 2 — 검출 결과 + 임무 지표")

if st.session_state.last_dets is None or st.session_state.step == 0:
    st.caption("실행 후 표시됩니다.")
else:
    n_p = sum(1 for d in st.session_state.last_dets if d.cls_name == "person")
    n_v = sum(1 for d in st.session_state.last_dets if d.cls_name == "vehicle")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("탐지 사람",   f"{n_p}",  delta=f"기대 3")
    m2.metric("탐지 차량",   f"{n_v}",  delta=f"기대 2")
    m3.metric("총 식별",     f"{n_p + n_v} / 5",
              delta=("PASS" if n_p + n_v >= 4 else "MISS"),
              delta_color="normal" if n_p + n_v >= 4 else "inverse")
    m4.metric("YOLO 임계",  f"{conf_threshold:.2f}")

    if st.session_state.last_dets:
        import pandas as pd
        df = pd.DataFrame([{
            "class": d.cls_name,
            "conf":  round(d.conf, 3),
            "x1":    round(d.xyxy[0]),
            "y1":    round(d.xyxy[1]),
            "x2":    round(d.xyxy[2]),
            "y2":    round(d.xyxy[3]),
            "width": round(d.xyxy[2] - d.xyxy[0]),
            "height":round(d.xyxy[3] - d.xyxy[1]),
        } for d in st.session_state.last_dets])
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.warning("YOLO가 어떤 객체도 검출하지 못함. 사이드바에서 conf threshold를 더 낮춰보세요 (0.05~0.10).")

    st.caption(f"마지막 sim: {st.session_state.last_summary}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LLM narration
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 📋 SECTION 3 — 자연어 보고 (안보 톤)")

if st.session_state.last_narrative is None:
    if st.session_state.step > 0:
        st.caption("(narration 비활성화되었거나 LLM 호출 실패)")
    else:
        st.caption("실행 후 표시됩니다.")
else:
    with st.container(border=True):
        st.markdown(st.session_state.last_narrative)

st.markdown("---")
st.caption(
    f"© Counterfactual-XAI-Verifier · Phase B-4 Unreal demo  ·  "
    f"steps run: {st.session_state.step}  ·  "
    f"💡 영상 캡처에 ~30~60초 소요"
)
