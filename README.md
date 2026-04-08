# 🛩️ 지능형 요구사항 검증 시스템 - LLM 파이프라인 (Zone 4)

본 브랜치(`LLM`)는 UAV 구조 임무 중 **비전 기반 객체 탐지 알고리즘**의 신뢰성을 검증하기 위한 지능형 의사결정 엔진을 구현합니다. XAI 모듈이 분석한 탐지 실패 원인(JSON)을 바탕으로, 시스템의 탐지 방어선($mAP_{50}$ 85%)을 무너뜨리는 가혹한 **Counterfactual 기상 시나리오 파라미터**를 자동 생성합니다.

## 🎯 검증 요구사항 (Requirements)
* **정상 운용(Baseline):** 주간 맑은 날씨(시정 10km↑, 10,000 Lux↑)에서 조난자 탐지율 $mAP_{50}$ 90% 이상 유지.
* **강건성 방어(Defense):** LLM이 생성한 가혹 조건(안개, 저조도 등)에서도 **최소 $mAP_{50}$ 85% 이상**을 방어해야 함.
* **최종 목적:** 시스템이 실패하는 최악의 조건(Worst-case Boundary)을 지능적으로 탐색하여 소프트웨어의 신뢰성 한계를 식별.

## 📂 디렉토리 구조 (File Tree)

```text
Counterfactual-XAI-Verifier/ (Branch: LLM)
├── llm_agent/
│   ├── __init__.py           # 패키지 인식용 빈 파일
│   └── gpt_generator.py      # OpenAI API 연동 및 JSON 기반 시나리오 생성 엔진
├── prompts/                   # 프롬프트 엔지니어링 레이어
│   ├── __init__.py            # 패키지 인식용 빈 파일
│   └── system_prompts.py      # 시야 방해(안개, 조도) 에이전트 페르소나 정의
├── .env                       # API Key 환경변수 (Git 업로드 제외)
├── requirements.txt           # 패키지 의존성 (openai, python-dotenv)
├── test_llm_standalone.py     # [Entry Point] LLM 파이프라인 단독 검증 스크립트
└── README.md                  # 프로젝트 문서
```

## ⚙️ 설치 및 환경 설정 (Prerequisites)

**1. 패키지 설치**
프로젝트 실행에 필요한 파이썬 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

**2. 환경 변수 세팅 (.env)**
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 발급받은 API 키를 입력합니다. (이 파일은 `.gitignore`에 의해 Github에 올라가지 않습니다.)
```text
OPENAI_API_KEY="sk-당신의_오픈AI_API_키"
```

## 🚀 사용법 (Usage)

`test_llm_standalone.py`를 실행하여 전체 파이프라인을 테스트할 수 있습니다.
```bash
python test_llm_standalone.py
```

**실행 흐름:**
1. XAI 데이터 로드: 탐지 결과 및 feature importance 데이터 로드
2. `llm_agent/gpt_generator.py`가 XAI 데이터를 분석하여 가장 취약한 환경 변수(안개 농도, 조도 등)를 식별
3. 결과 생성: `json_object` 모드를 통해 Simulink에 주입 가능한 환경 파라미터를 JSON 포맷으로 출력

## 📌 향후 개발 계획 (Next Steps)
- [ ] Multi-LLM 라우팅 구현: Claude Opus 4.6을 활용한 생성 결과의 현실성 및 엣지 케이스 교차 검증 로직 추가.
- [ ] Simulink 통합: JSON 파라미터를 가상 환경에 실시간으로 주입
- [ ] XAI 피드백 루프: 실제 객체 탐지 결과로부터 산출된 XAI 기여도를 입력 스트림으로 연동
