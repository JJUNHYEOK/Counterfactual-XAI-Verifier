# 🛩️ 지능형 요구사항 검증 시스템 - LLM 파이프라인 (Zone 4)

본 브랜치(`LLM`)는 UAV 자율비행 충돌 회피 및 경로 탐색 알고리즘 검증을 위한 지능형 요구사항 검증 시스템의 핵심 의사결정 엔진을 구현합니다. 
XAI 모듈이 분석한 실패 원인(JSON)을 바탕으로, LLM이 시스템 실패를 유도하는 가혹한 Counterfactual 시나리오 파라미터(YAML)를 자동 생성합니다.

## 📂 디렉토리 구조 (File Tree)

\`\`\`text
llm_pipeline/
├── data/                      # 모듈 간 데이터 연동 테스트를 위한 Mock 데이터
│   ├── mock_xai_input.json    # [Input] XAI 모듈의 Causal Feature Importance 분석 결과
│   └── mock_simulink_output.yaml # [Output] Simulink 실행을 위한 최종 Counterfactual 파라미터
│
├── prompts/                   # Multi-LLM 프롬프트 엔지니어링
│   └── system_prompts.py      # GPT-5.4(일반 생성) 및 Claude 4.6(교차 검증) 시스템 프롬프트
│
├── core/                      # LLM API 통신 및 생성 코어 로직
│   └── gpt_generator.py       # OpenAI API 연동 및 시나리오 파라미터 생성 엔진
│
├── parsers/                   # 데이터 입출력 인터프리터
│   └── yaml_exporter.py       # LLM 자연어 출력을 안전한 YAML 포맷으로 파싱 및 저장
│
├── .env                       # API Key 환경변수 (Git 업로드 제외)
├── .gitignore                 # 보안 및 캐시 파일 무시 설정
├── requirements.txt           # 패키지 의존성 목록
└── main.py                    # LLM 파이프라인 실행 진입점 (Entry Point)
\`\`\`

## ⚙️ 설치 및 환경 설정 (Prerequisites)

**1. 패키지 설치**
프로젝트 실행에 필요한 파이썬 라이브러리를 설치합니다.
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**2. 환경 변수 세팅 (.env)**
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 발급받은 API 키를 입력합니다. (이 파일은 `.gitignore`에 의해 Github에 올라가지 않습니다.)
\`\`\`text
OPENAI_API_KEY="sk-당신의_오픈AI_API_키"
\`\`\`

## 🚀 사용법 (Usage)

`main.py`를 실행하여 전체 파이프라인을 테스트할 수 있습니다.
\`\`\`bash
python main.py
\`\`\`

**실행 흐름:**
1. `data/mock_xai_input.json` 파일에서 가상의 XAI 분석 결과를 로드합니다.
2. `core/gpt_generator.py`가 XAI 데이터를 분석하여 가장 취약한 환경 변수를 악화시키는 시나리오를 추론합니다.
3. `parsers/yaml_exporter.py`가 생성된 결과를 문법 오류 없이 파싱하여 `data/mock_simulink_output.yaml`로 저장합니다. 시스템 담당자는 이 파일을 Simulink에 바로 주입할 수 있습니다.

## 📌 향후 개발 계획 (Next Steps)
- [ ] Multi-LLM 라우팅 구현: Claude Opus 4.6을 활용한 생성 결과의 현실성 및 엣지 케이스 교차 검증 로직 추가.
- [ ] API 연동 안정화: 타 모듈(XAI, Simulink) 개발 완료 시 실제 입출력 데이터 스트림으로 파이프라인 전환.