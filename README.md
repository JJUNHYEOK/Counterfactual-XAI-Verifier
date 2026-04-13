# Counterfactual-XAI-Verifier

UAV 구조 임무 중 비전 기반 객체 탐지 알고리즘의 신뢰성을 검증하는 **지능형 요구사항 검증 시스템**입니다.
XAI 모듈이 분석한 탐지 실패 원인을 바탕으로 Counterfactual 기상 시나리오를 자동 생성하고, Simulink 시뮬레이션으로 검증합니다.

> 본 브랜치(`test_v2`)는 `MATLAB/Simulink/SI`, `XAI`, `LLM` 브랜치를 통합한 브랜치입니다.

## 브랜치 구성

| 브랜치 | 역할 |
|--------|------|
| `MATLAB/Simulink/SI` | UAV Simulink 시뮬레이션 + SI 파이프라인 |
| `XAI` | 환경 변수 기여도 분석 모듈 |
| `LLM` | GPT 기반 Counterfactual 시나리오 생성 |
| `test_v2` | 전체 브랜치 통합 (현재 브랜치) |

## 전체 파이프라인

1. SI (Simulink)에서 시나리오 실행 및 결과 산출
2. XAI에서 결과 원인 분석 및 환경 변수 기여도 정량화
3. LLM에서 XAI 결과를 읽고 Counterfactual 시나리오 생성
4. 생성된 시나리오를 다시 SI에 주입하여 반복 검증

## 검증 요구사항

- **정상 운용(Baseline):** 주간 맑은 날씨(시정 10km+, 10,000 Lux+)에서 조난자 탐지율 mAP50 90% 이상 유지
- **강건성 방어(Defense):** LLM이 생성한 가혹 조건(안개, 저조도 등)에서도 최소 mAP50 85% 이상 방어
- **최종 목적:** 시스템이 실패하는 최악의 조건(Worst-case Boundary)을 지능적으로 탐색하여 소프트웨어의 신뢰성 한계 식별

## XAI 역할

- 입력: SI에서 생성된 `scenario`, `sim_result`, `eval_result`
- 처리: 실패/성능저하 원인에 대한 환경 변수 기여도 계산
- 출력: LLM이 바로 읽을 수 있는 `xai_input.json`

## 입력/출력 스키마(요약)

### XAI 입력(예시)

```json
{
  "scene_id": "iter_001",
  "scenario": {
    "environment_parameters": {
      "fog_density_percent": 30.0,
      "illumination_lux": 4000.0,
      "camera_noise_level": 0.1
    }
  },
  "sim_result": {
    "avg_confidence": 0.817,
    "risk_score": 0.553
  },
  "eval_result": {
    "map50": 0.4096,
    "requirement_threshold": 0.85,
    "requirement_violated": true
  }
}
```

### XAI 출력(LLM 입력, 예시)

```json
{
  "scene_id": "iter_001",
  "task": "uav_object_detection",
  "performance_signals": {
    "confidence_trend": "decreasing",
    "miss_rate_trend": "increasing",
    "risk_score": 0.59,
    "failure_type": "detection_performance_drop",
    "map50": 0.4096,
    "threshold": 0.85
  },
  "xai_signals": {
    "method": "direct-posthoc",
    "dominant_factors": [
      {"name": "fog_density", "importance": 0.409},
      {"name": "illumination_lux", "importance": 0.455},
      {"name": "camera_noise", "importance": 0.136}
    ],
    "attention_summary": "mAP50 is 0.4096 under fog=30.0, illum=4000.0, noise=0.10"
  }
}
```

## 폴더 구조

```text
Counterfactual-XAI-Verifier/
├── core/
│   ├── detector.py           # LLM 입력 빌더 및 스키마 검증
│   └── schema.py             # JSON 스키마 검증 로직
├── llm_agent/
│   ├── __init__.py
│   └── gpt_generator.py      # OpenAI API 연동 및 시나리오 생성 엔진
├── matlab/
│   ├── build_sim_input.m     # Simulink 입력 생성
│   ├── run.m                 # 시뮬레이션 실행
│   └── analyze_results.m     # 결과 분석
├── prompts/
│   ├── __init__.py
│   └── system_prompts.py     # 에이전트 페르소나 정의
├── xai/
│   ├── __init__.py
│   ├── dummy_analyzer.py     # XAI 분석 더미 구현
│   └── io_adapter.py         # SI <-> XAI <-> LLM 인터페이스 어댑터
├── data/                     # 시나리오 및 시뮬레이션 결과 데이터
├── schemas/
│   └── xai_input.schema.json # XAI 입력 JSON 스키마
├── main.py                   # 전체 파이프라인 진입점 (LLM + core)
├── simulator.py              # Python 시뮬레이터 인터페이스
├── requirements.txt          # 패키지 의존성
├── uav.slx                   # UAV Simulink 모델
└── uav_cf_viewer.slx         # Counterfactual 뷰어 Simulink 모델
```

## 설치 및 실행

```bash
pip install -r requirements.txt
```

`.env` 파일 생성 후 API 키 입력:

```text
OPENAI_API_KEY="sk-your-openai-api-key"
```

### 전체 파이프라인 실행

```bash
python main.py
```

### XAI 모듈 단독 실행

```bash
python xai/main.py
```

### LLM 단독 검증

```bash
python test_llm_standalone.py
```

## 네이밍/연동 원칙

- 함수명/파일명/JSON key는 SI/LLM 브랜치 스타일을 우선 재사용
- 새 네이밍은 최소화하고 snake_case 유지
- 브랜치 간 호환성을 최우선으로 유지
