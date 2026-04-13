# 🚁 Counterfactual XAI-based UAV Safety Verifier

본 프로젝트는 UAV(무인 항공기)의 객체 탐지 모델에 대해 LLM(Large Language Model) 기반의 반사실적 시나리오를 생성하고, XAI(Explainable AI)를 통해 모델의 안전 임계치(Safety Line)를 자율적으로 검증하는 프레임워크입니다.

## 🌟 주요 기능
- **LLM-driven Scenario Generation**: GPT-4o-mini를 이용한 지능형 환경 결함(안개, 노이즈, 블러 등) 설계.
- **Real-time XAI Analysis**: YOLO11x 탐지 결과에 대해 XGBoost와 SHAP을 활용한 결함 기여도 분석.
- **Asynchronous Monitoring**: 멀티스레딩 기반의 실시간 탐지 모니터링 및 성능 추이 시각화 대시보드.
- **Interactive Control**: Streamlit UI를 통한 백엔드 엔진 제어 및 히스토리 분석.

## 📁 프로젝트 구조
```text
.
├── app.py                  # Streamlit 기반 통합 제어 대시보드
├── main.py                 # 백엔드 자율 검증 파이프라인 엔진
├── xai/
│   └── real_analyzer.py    # YOLO 탐지 및 SHAP 분석 모듈
├── llm_agent/
│   └── gpt_generator.py    # LLM 시나리오 생성 모듈
├── data/                   # 분석 결과(JSON) 저장 폴더
├── assets/                 # 이미지(원본, 변조, 결과) 저장 폴더
└── requirements.txt        # 의존성 패키지 목록
```

## 설치 및 실행

```bash
pip install -r requirements.txt
(가상환경 추천)
```

`.env` 파일 생성 후 API 키 입력:

```text
OPENAI_API_KEY="sk-your-openai-api-key"
```

### 대시보드 실행

```bash
streamlit run app.py
```

