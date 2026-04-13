import streamlit as st
import json
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="UAV XAI Dashboard", layout="wide")
st.title("🚁 UAV 자율 검증 능동 탐색 대시보드")

# 경로 설정
DATA_DIR = Path("./data")
IMAGE_DIR = Path("./assets")

# 사이드바: Iteration 선택
step = st.sidebar.slider("검증 단계(Iteration) 선택", 1, 3, 1)

# 데이터 로드
json_path = DATA_DIR / f"dashboard_step_{step}.json"
if json_path.exists():
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # UI 레이아웃
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader(f"🖼️ 변조된 UAV 시야 (Step {step})")
        img_name = data["panel_1_visual"]["rendered_image"]
        img = Image.open(IMAGE_DIR / img_name)
        st.image(img, use_container_width=True)
        
        # 메트릭 표시
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("mAP50 Score", f"{data['panel_1_visual']['map50_score']:.4f}")
        m_col2.info(f"적용 파라미터: {data['panel_1_visual']['params']}")

    with col2:
        st.subheader("💡 LLM 공격 가설")
        st.success(f"**가설:** {data['panel_3_llm']['hypothesis']}")
        st.write(f"**추론 로직:** {data['panel_3_llm']['reasoning']}")

        st.divider()

        st.subheader("📊 XAI 원인 분석 (SHAP)")
        shap_data = data["panel_2_xai"]
        df = pd.DataFrame(shap_data)
        st.bar_chart(data=df, x="name", y="importance", color="#ff4b4b")
        st.caption("어떤 환경 변수가 탐지율 하락에 가장 크게 기여했는지 보여줍니다.")

else:
    st.warning(f"데이터가 없습니다. {json_path.name} 확인 필요!")