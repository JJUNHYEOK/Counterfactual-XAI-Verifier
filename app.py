import streamlit as st
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import plotly.express as px
import time
import threading 
import main     
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

# 커스텀 CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .status-card { 
        padding: 20px; border-radius: 12px; background-color: #1f2937; 
        border: 1px solid #374151; margin-bottom: 15px;
    }
    .stMetric { background-color: #111827; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

def load_json(step):
    try:
        with open(DATA_DIR / f"dashboard_step_{step}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except: return None

# 2. 사이드바 시스템 제어
st.sidebar.title("시스템 제어 센터")

st.sidebar.subheader("엔진 제어")
if 'backend_thread' not in st.session_state:
    st.session_state.backend_thread = None

if st.sidebar.button("검증 파이프라인 가동", type="primary"):
    if st.session_state.backend_thread is None or not st.session_state.backend_thread.is_alive():
        # main.py의 run_dynamic_pipeline 호출
        thread = threading.Thread(target=main.run_dynamic_pipeline, daemon=True)
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
    
    if 'live_active' not in st.session_state:
        st.session_state.live_active = False

    c1, c2 = st.sidebar.columns(2)
    if c1.button("▶️ 시작"): st.session_state.live_active = True
    if c2.button("⏹️ 중지"): st.session_state.live_active = False

    if st.session_state.live_active:
        p_status = st.empty()
        p_img = st.empty()
        p_info = st.empty()
        
        while st.session_state.live_active:
            img_files = sorted(list(IMAGE_DIR.glob("current_iter_*.jpg")), 
                               key=lambda x: int(x.stem.split('_')[-1]))
            
            if img_files:
                latest_img_path = img_files[-1]
                latest_step = int(latest_img_path.stem.split('_')[-1])
                
                results = ui_detector(str(latest_img_path), verbose=False)
                annotated_frame = results[0].plot()
                display_img = Image.fromarray(annotated_frame[..., ::-1])
                
                p_img.image(display_img, use_container_width=True, 
                           caption=f"Iteration {latest_step}: [Real-time Target Detection Active]")
                
                data = load_json(latest_step)
                if data:
                    score = data["panel_1_visual"]["map50_score"]
                    color = "#ff4b4b" if score < 0.85 else "#28a745"
                    p_status.markdown(f"""
                        <div class='status-card' style='border-left: 10px solid {color};'>
                            <h2 style='margin:0; color:{color};'>최종 신뢰도: {score:.4f}</h2>
                            <p style='margin:0; font-size:1.1em;'>상태: <b>{'⚠️ 위험 (Safety Line 붕괴)' if score < 0.85 else '✅ 정상 (PASSED)'}</b> 
                            | Step: {latest_step}/5</p>
                        </div>
                    """, unsafe_allow_html=True)
                    with p_info.container():
                        st.info(f"**반사실 시나리오 생성 가설:** {data['panel_3_llm']['hypothesis']}")
                else:
                    p_status.warning(f"Iteration {latest_step}: 변조 이미지 생성 완료. SHAP 분석 대기 중...")
            
            time.sleep(1)
            st.rerun()
    else:
        st.info("왼쪽 사이드바에서 '▶️ 시작'을 눌러 모니터링을 개시하세요.")

# 4. 메인 화면 - 히스토리 분석 (여기에 방어 로직이 집중됨)
else:
    st.header("검증 히스토리 정밀 분석")
    
    # 🔍 방어 로직: 파일 목록을 먼저 가져옴
    json_files = list(DATA_DIR.glob("dashboard_step_*.json"))
    steps = sorted([int(f.stem.split('_')[-1]) for f in json_files])
    
    # [방어 로직 시작] steps 리스트가 비어있으면 슬라이더를 아예 안 그림
    if not steps:
        st.error("🚨 분석할 데이터가 존재하지 않습니다.")
        st.info("사이드바의 '검증 파이프라인 가동' 버튼을 눌러 먼저 데이터를 생성해주세요.")
        
        # 샘플 이미지가 있다면 보여주기 (없으면 생략 가능)
        if (IMAGE_DIR / "step_1.jpg").exists():
            st.image(Image.open(IMAGE_DIR / "step_1.jpg"), caption="원본 베이스라인 이미지", use_container_width=True)
            
    else:
        # 데이터가 1개 이상 있을 때만 슬라이더 렌더링 (RangeError 방지)
        sel = st.select_slider("검증 단계 선택", options=steps)
        d = load_json(sel)
        
        if d:
            col_l, col_r = st.columns([1.5, 1])
            with col_l:
                ann_path = IMAGE_DIR / f"annotated_iter_{sel}.jpg"
                orig_path = IMAGE_DIR / d["panel_1_visual"]["rendered_image"]
                final_img_path = ann_path if ann_path.exists() else orig_path
                
                if final_img_path.exists():
                    st.image(Image.open(final_img_path), use_container_width=True, 
                             caption=f"Step {sel} {'분석 완료' if ann_path.exists() else '분석 중'}")
                else:
                    st.error("이미지 파일을 찾을 수 없습니다.")
                    
            with col_r:
                st.metric("mAP50 Score", f"{d['panel_1_visual']['map50_score']:.4f}", 
                          delta=f"{d['panel_1_visual']['map50_score']-0.85:.4f}", delta_color="inverse")
                st.markdown("---")
                st.markdown("**🧠 LLM 추론 근거**")
                st.success(d["panel_3_llm"]["reasoning"])

            # 하단 추이 그래프
            st.divider()
            history_data = []
            for s in steps:
                hist_d = load_json(s)
                if hist_d:
                    history_data.append({"Step": s, "mAP": hist_d["panel_1_visual"]["map50_score"]})
            
            if history_data:
                df_hist = pd.DataFrame(history_data)
                fig = px.line(df_hist, x="Step", y="mAP", markers=True, title="성능 하락 타임라인 (mAP50)")
                fig.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="Safety Line (85%)")
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)