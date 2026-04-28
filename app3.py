import streamlit as st
import json
import pandas as pd
from pathlib import Path
import re
import importlib
from PIL import Image
import plotly.express as px
import threading 
import main2     
from ultralytics import YOLO
import numpy as np

# 1. 페이지 설정 및 초기화
st.set_page_config(page_title="UAV Safety Verifier", layout="wide", initial_sidebar_state="expanded")

DATA_DIR, IMAGE_DIR = Path("./data"), Path("./assets")
for d in [DATA_DIR, IMAGE_DIR]: d.mkdir(exist_ok=True)

# UI용 경량 YOLO 모델 로드 (캐싱)
@st.cache_resource
def get_ui_detector():
    return YOLO("yolo11n.pt") 

ui_detector = get_ui_detector()

def get_latest_live_image():
    parsed_img = []
    for path in IMAGE_DIR.glob("current_iter_*.jpg"):
        step_val = extract_step(path.stem, "current_iter")
        if step_val is not None:
            try:
                mtime_ns = path.stat().st_mtime_ns
            except FileNotFoundError:
                continue
            parsed_img.append((mtime_ns, step_val, path))

    if not parsed_img:
        return None, None

    parsed_img.sort(key=lambda item: (item[0], item[1]))
    _, latest_step, latest_path = parsed_img[-1]
    return latest_step, latest_path

@st.cache_data(show_spinner=False)
def run_ui_inference_cached(image_path: str, image_mtime_ns: int):
    _ = image_mtime_ns 
    results = ui_detector(image_path, verbose=False)
    return results[0].plot()

def initialize_live_state():
    st.session_state.setdefault("live_active", False)
    st.session_state.setdefault("live_last_frame_rgb", None)
    st.session_state.setdefault("live_last_step", None)
    st.session_state.setdefault("live_last_data", None)
    st.session_state.setdefault("live_last_image_key", None)
    st.session_state.setdefault("live_rendered_image_key", None)
    st.session_state.setdefault("live_rendered_status_key", None)
    st.session_state.setdefault("live_rendered_info_key", None)

# 커스텀 CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 100% !important; padding-top: 0.6rem !important;
        padding-right: 0.9rem !important; padding-left: 0.9rem !important; padding-bottom: 0.75rem !important;
    }
    section[data-testid="stSidebar"] { width: 248px !important; min-width: 248px !important; }
    section[data-testid="stSidebar"] > div { width: 248px !important; min-width: 248px !important; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stSubheader,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span { color: #111111 !important; }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] { color: #ffffff !important; }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] span { color: #ffffff !important; }
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) { color: #111111 !important; }
    [data-testid="stSidebar"] .stButton>button { margin-top: 0.1rem !important; margin-bottom: 0.1rem !important; }
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"], .status-card, .metric-card {
        white-space: normal !important; overflow: visible !important;
        text-overflow: clip !important; overflow-wrap: anywhere !important; word-break: break-word !important;
    }
    h1, h2, h3, p { margin-top: 0.2rem !important; margin-bottom: 0.42rem !important; }
    .status-card { padding: 14px; border-radius: 12px; background-color: #1f2937; border: 1px solid #374151; margin-bottom: 10px; }
    .metric-card { padding: 14px; border-radius: 12px; background-color: #111827; border: 1px solid #374151; margin-bottom: 10px; }
    .metric-title { font-size: 1.1rem; opacity: 0.95; margin-bottom: 6px; color: #ffffff !important; }
    .metric-score { font-size: 2.4rem; font-weight: 700; line-height: 1.1; margin-bottom: 6px; }
    .metric-status { font-size: 1rem; color: #ffffff !important; }
    .status-card h2, .status-card p { color: #ffffff !important; }
    .stMetric { background-color: #111827; padding: 12px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 2.55em; }
    @media (max-width: 1200px) {
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
            width: 222px !important; min-width: 222px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def load_json(step):
    try:
        primary = DATA_DIR / f"dashboard_step_{step}.json"
        candidates = []
        if primary.exists(): candidates.append(primary)
        candidates.extend(sorted(DATA_DIR.glob(f"dashboard_step_{step}*.json"), key=lambda p: p.stat().st_mtime, reverse=True))
        seen = set()
        ordered_candidates = []
        for path in candidates:
            if str(path) in seen: continue
            seen.add(str(path))
            ordered_candidates.append(path)
        for path in ordered_candidates:
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        return None
    except: return None

def _to_float(value, default=0.0):
    try: return float(value)
    except Exception:
        try:
            text = str(value).replace("[", "").replace("]", "").replace("'", "").replace('"', "").strip()
            return float(text)
        except Exception: return float(default)

def normalize_panel_2_xai(panel_2_xai):
    if not isinstance(panel_2_xai, list): return []
    rows = []
    for item in panel_2_xai:
        if not isinstance(item, dict): continue
        name = str(item.get("name", "")).strip()
        if not name: continue
        rows.append({"feature": name, "importance": _to_float(item.get("importance", 0.0), default=0.0)})
    rows.sort(key=lambda x: x["importance"], reverse=True)
    return rows

def compact_text(text, max_chars=320, max_lines=6):
    text = str(text or "").strip()
    if not text: return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = "\n".join(lines) if lines else text
    return compact


def escape_markdown_tilde(text: str) -> str:
    """Avoid accidental strikethrough rendering from range text like 20~30%."""
    return str(text or "").replace("~", r"\~")

def extract_step(stem: str, prefix: str) -> int | None:
    match = re.search(rf"{re.escape(prefix)}_(\d+)", stem)
    if not match: return None
    try: return int(match.group(1))
    except ValueError: return None

# 2. 사이드바 시스템 제어
st.sidebar.title("시스템 제어 센터")
st.sidebar.subheader("엔진 제어")
if 'backend_thread' not in st.session_state:
    st.session_state.backend_thread = None

if st.sidebar.button("검증 파이프라인 가동", type="primary"):
    if st.session_state.backend_thread is None or not st.session_state.backend_thread.is_alive():
        importlib.reload(main2)
        thread = threading.Thread(target=main2.run_dynamic_pipeline, daemon=True)
        thread.start()
        st.session_state.backend_thread = thread
        st.sidebar.success("엔진이 백그라운드에서 가동되었습니다!")
    else:
        st.sidebar.warning("엔진이 이미 가동 중입니다.")

st.sidebar.markdown("---")
mode = st.sidebar.radio("작동 모드 선택", ["실시간 모니터링 (Live)", "히스토리 분석"])

# 3. 메인 화면 - 실시간 모니터링 (Live)
if mode == "실시간 모니터링 (Live)":
    st.header("실시간 자율 검증 스트리밍")
    
    initialize_live_state()

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ 시작"): st.session_state.live_active = True
    if c2.button("⏹️ 중지"): st.session_state.live_active = False

    live_status_slot = st.empty()
    live_image_slot = st.empty()
    live_info_slot = st.empty()

    @st.fragment(run_every="1s")
    def render_live_monitor():
        if st.session_state.live_active:
            latest_step, latest_img_path = get_latest_live_image()
            if latest_img_path is not None:
                mtime_ns = latest_img_path.stat().st_mtime_ns
                image_key = f"{latest_img_path}|{mtime_ns}"

                if st.session_state.live_last_image_key != image_key:
                    annotated_frame = run_ui_inference_cached(str(latest_img_path), mtime_ns)
                    st.session_state.live_last_frame_rgb = annotated_frame[..., ::-1]
                    st.session_state.live_last_step = latest_step
                    st.session_state.live_last_data = load_json(latest_step)
                    st.session_state.live_last_image_key = image_key

        last_frame = st.session_state.live_last_frame_rgb
        last_step = st.session_state.live_last_step
        last_data = st.session_state.live_last_data

        if last_frame is not None and last_step is not None:
            current_render_image_key = st.session_state.live_last_image_key
        else:
            current_render_image_key = "__no_frame__"

        if current_render_image_key != st.session_state.live_rendered_image_key:
            if last_frame is not None and last_step is not None:
                live_image_slot.image(last_frame, channels="RGB", use_container_width=True, caption=f"Iteration {last_step}: [Real-time Target Detection Active]")
            else:
                live_image_slot.info("표시 가능한 실시간 프레임이 아직 없습니다.")
            st.session_state.live_rendered_image_key = current_render_image_key

        if last_frame is not None and last_step is not None:
            pass
        elif st.session_state.live_last_step is None:
            if st.session_state.live_rendered_status_key != ("waiting", None):
                live_status_slot.info("새 프레임 대기 중입니다.")
                st.session_state.live_rendered_status_key = ("waiting", None)
            if st.session_state.live_rendered_info_key != ("waiting", st.session_state.live_active):
                with live_info_slot.container():
                    st.caption("검증 파이프라인에서 첫 결과를 생성하는 중입니다.")
                st.session_state.live_rendered_info_key = ("waiting", st.session_state.live_active)
            return

        if isinstance(last_data, dict) and last_step is not None:
            score = float(last_data["panel_1_visual"]["map50_score"])
            # 💡 [핵심 수정] 하드코딩 0.85 제거, JSON의 동적 Safety Line 적용
            safety_line = float(last_data.get("safety_line", 0.70)) 
            color = "#ff4b4b" if score < safety_line else "#28a745"
            status_text = "탐지 위험 (Safety Line 붕괴)" if score < safety_line else "정상 (PASSED)"
            status_key = (last_step, round(score, 6), status_text)
            
            if status_key != st.session_state.live_rendered_status_key:
                live_status_slot.markdown(
                    f"""
                    <div class='status-card' style='border-left: 10px solid {color};'>
                        <h2 style='margin:0; color:{color};'>최종 성능값: {score:.4f}</h2>
                        <p style='margin:0; font-size:1.1em;'>상태: <b>{status_text}</b>
                        | Step: {last_step}/5 | <span style='font-size:0.9em; color:#9ca3af;'>기준선: {safety_line:.4f}</span></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.session_state.live_rendered_status_key = status_key

            hypothesis = last_data.get("panel_3_llm", {}).get("hypothesis", "")
            cf_panel = last_data.get("panel_4_counterfactual", {})
            cf_summary = cf_panel.get("summary") if isinstance(cf_panel, dict) else None
            info_key = (last_step, hypothesis, cf_summary, st.session_state.live_active)
            if info_key != st.session_state.live_rendered_info_key:
                with live_info_slot.container():
                    st.info(f"**반사실 시나리오 생성 가설** {hypothesis}")
                    if cf_summary:
                        st.caption(f"XAI Counterfactual: {cf_summary}")
                    if not st.session_state.live_active:
                        st.caption("모니터링 일시 정지 상태입니다. 마지막 프레임을 유지합니다.")
                st.session_state.live_rendered_info_key = info_key
        elif last_step is not None:
            status_key = ("pending", last_step)
            if status_key != st.session_state.live_rendered_status_key:
                live_status_slot.warning(f"Iteration {last_step}: 변조 이미지 생성 완료. 분석 대기 중...")
                st.session_state.live_rendered_status_key = status_key

            info_key = ("pending", last_step, st.session_state.live_active)
            if info_key != st.session_state.live_rendered_info_key:
                with live_info_slot.container():
                    if not st.session_state.live_active:
                        st.caption("모니터링 일시 정지 상태입니다. 마지막 프레임을 유지합니다.")
                st.session_state.live_rendered_info_key = info_key
        else:
            status_key = ("idle", st.session_state.live_active)
            if status_key != st.session_state.live_rendered_status_key:
                live_status_slot.info("실시간 모니터링을 시작하면 프레임과 추론 결과가 표시됩니다.")
                st.session_state.live_rendered_status_key = status_key
            info_key = ("idle", st.session_state.live_active)
            if info_key != st.session_state.live_rendered_info_key:
                with live_info_slot.container():
                    if not st.session_state.live_active:
                        st.caption("모니터링 일시 정지 상태입니다. 마지막 프레임을 유지합니다.")
                st.session_state.live_rendered_info_key = info_key

    render_live_monitor()

# 4. 메인 화면 - 히스토리 분석
else:
    st.header("검증 히스토리 정밀 분석")
    
    json_files = list(DATA_DIR.glob("dashboard_step_*.json"))
    steps = sorted(
        {
            step_val
            for f in json_files
            for step_val in [extract_step(f.stem, "dashboard_step")]
            if step_val is not None
        }
    )
    
    if not steps:
        st.error("🚨 분석할 데이터가 존재하지 않습니다.")
        st.info("사이드바의 '검증 파이프라인 가동' 버튼을 눌러 먼저 데이터를 생성해주세요.")
        
        if (IMAGE_DIR / "step_1.jpg").exists():
            st.image(Image.open(IMAGE_DIR / "step_1.jpg"), caption="원본 베이스라인 이미지", use_container_width=True)
            
    else:
        if len(steps) == 1:
            sel = steps[0]
            st.info(f"현재 선택 가능한 검증 단계가 1개(Step {sel})라 자동 선택했습니다.")
        else:
            sel = st.select_slider("검증 단계 선택", options=steps)
        d = load_json(sel)
        
        if d:
            score = float(d["panel_1_visual"]["map50_score"])
            # 💡 [핵심 수정] 하드코딩 0.85 제거, JSON의 동적 Safety Line 적용
            threshold = float(d.get("safety_line", 0.70)) 
            delta = score - threshold
            is_pass = score >= threshold
            color = "#22c55e" if is_pass else "#ef4444"
            status_text = "✅ 정상 (PASSED)" if is_pass else "⚠️ 위험 (FAILED)"

            llm_reasoning = str(d.get("panel_3_llm", {}).get("reasoning", "")).strip()
            llm_reasoning_short = compact_text(llm_reasoning, max_chars=360, max_lines=7)

            cf_panel = d.get("panel_4_counterfactual", {})
            cf_summary_full = cf_panel.get("summary", "") if isinstance(cf_panel, dict) else ""
            cf_summary_short = compact_text(cf_summary_full, max_chars=280, max_lines=4)

            xai_rows = normalize_panel_2_xai(d.get("panel_2_xai"))

            history_data = []
            for s in steps:
                hist_d = load_json(s)
                if hist_d:
                    history_data.append({"Step": s, "mAP": hist_d["panel_1_visual"]["map50_score"]})

            col_l, col_r = st.columns([1.75, 0.9], gap="small")
            with col_l:
                ann_path = IMAGE_DIR / f"annotated_iter_{sel}.jpg"
                orig_path = IMAGE_DIR / d["panel_1_visual"]["rendered_image"]
                final_img_path = ann_path if ann_path.exists() else orig_path
                
                if final_img_path.exists():
                    st.image(Image.open(final_img_path), use_container_width=True, 
                             caption=f"Step {sel} {'분석 완료' if ann_path.exists() else '분석 중'}")
                else:
                    st.error("이미지 파일을 찾을 수 없습니다.")

                if history_data:
                    df_hist = pd.DataFrame(history_data)
                    fig = px.line(df_hist, x="Step", y="mAP", markers=True, title="성능 하락 타임라인 (mAP50)")
                    
                    # 💡 [핵심 수정] 빨간 점선 동적 설정
                    fig.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Safety Line ({threshold:.4f})",
                        annotation_font_color="#ffffff",
                    )
                    fig.update_layout(
                        yaxis_range=[0, 1],
                        height=260,
                        margin=dict(l=20, r=10, t=40, b=10),
                        font=dict(color="#ffffff"),
                        paper_bgcolor="#0f172a",
                        plot_bgcolor="#0f172a",
                        title_font=dict(color="#ffffff"),
                        xaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
                        yaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            with col_r:
                st.markdown(
                    f"""
                    <div class='metric-card' style='border-left: 10px solid {color};'>
                        <div class='metric-title'>mAP50 Score</div>
                        <div class='metric-score' style='color:{color};'>{score:.4f}</div>
                        <div class='metric-status' style='color:{color};'>{status_text} (기준선 {threshold:.4f} 대비 {delta:+.4f})</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                tab_summary, tab_detail = st.tabs(["핵심 요약", "상세 근거"])

                with tab_summary:
                    st.markdown("**🧠 LLM 추론 요약**")
                    if llm_reasoning_short:
                        st.info(escape_markdown_tilde(llm_reasoning_short))
                    else:
                        st.caption("LLM 추론 문장이 없습니다.")

                    st.markdown("**🧭 Counterfactual 요약**")
                    if cf_summary_short:
                        st.info(escape_markdown_tilde(cf_summary_short))
                    else:
                        st.caption("Counterfactual 요약이 없습니다.")

                    st.markdown("**📊 XAI (SHAP) 변수 기여도**")
                    if xai_rows:
                        df_xai = pd.DataFrame(xai_rows)
                        df_xai_plot = df_xai.iloc[::-1]
                        max_importance = max(float(df_xai["importance"].max()), 0.0)
                        xaxis_max = max_importance * 1.15 if max_importance > 0 else 1.0
                        fig_xai = px.bar(
                            df_xai_plot,
                            x="importance",
                            y="feature",
                            orientation="h",
                            text="importance",
                            color="importance",
                            color_continuous_scale=["#1d4ed8", "#16a34a", "#f59e0b", "#ef4444"],
                            title=f"Step {sel} SHAP 중요도",
                        )
                        fig_xai.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                        fig_xai.update_layout(
                            height=250,
                            margin=dict(l=10, r=10, t=36, b=10),
                            xaxis_title="importance",
                            yaxis_title="feature",
                            coloraxis_showscale=False,
                            font=dict(color="#ffffff"),
                            paper_bgcolor="#0f172a",
                            plot_bgcolor="#0f172a",
                        )
                        fig_xai.update_xaxes(range=[0, xaxis_max], autorange=False)
                        st.plotly_chart(fig_xai, use_container_width=True)
                    else:
                        st.caption("현재 step에는 SHAP 중요도 데이터가 없습니다.")

                with tab_detail:
                    st.markdown("**LLM 추론 원문**")
                    if llm_reasoning:
                        st.success(escape_markdown_tilde(llm_reasoning))
                    else:
                        st.caption("LLM 추론 원문이 없습니다.")

                    st.markdown("**Counterfactual 경계 탐색 원문**")
                    if cf_summary_full:
                        st.info(escape_markdown_tilde(cf_summary_full))
                    else:
                        st.caption("Counterfactual 원문 요약이 없습니다.")

                    if xai_rows:
                        st.markdown("**XAI (SHAP) 원본 테이블**")
                        st.dataframe(pd.DataFrame(xai_rows), use_container_width=True, hide_index=True)
