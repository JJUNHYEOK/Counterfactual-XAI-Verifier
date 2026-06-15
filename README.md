# Counterfactual-XAI-Verifier — `yeah` 브랜치

> **XAI-LLM 기반 Mission-Critical 자율 에이전트 지능형 검증 프레임워크**  
> 반사실적(Counterfactual) 시나리오 생성을 통한 엣지 케이스 탐색 고도화

---

## 개요

본 브랜치(`yeah`)는 **XAI-LLM 기반 폐루프(Closed-Loop) 검증 파이프라인**의 통합 진입점입니다.

방산·항공·자율주행 등 Mission-Critical 환경에서는 미세한 설계 오차가 심각한 자산 손실 및 인명 사고로 직결됩니다. 기존의 수동 설계 시나리오·랜덤 테스팅 방식으로는 기하급수적으로 증가하는 엣지 케이스를 탐색하기 어렵고, 실패가 발생해도 변수 간 인과관계를 규명하기 어렵습니다.

이 프레임워크는 세 가지 핵심 기술의 결합으로 이 문제를 해결합니다.

- **반사실적 탐색**: 성공 케이스에서 환경 변수를 점진적으로 악화시켜 취약점의 임계 영역을 능동적으로 도출
- **XAI 기반 원인 분석**: 환경 변수별 기여도를 수치화하여 실패 원인을 정량적으로 규명
- **LLM 자동 시나리오 생성**: XAI 분석 결과를 기반으로 다음 테스트 케이스를 자동 설계·주입

---

## 전체 파이프라인

```
[SI 브랜치] 시나리오 실행 및 시뮬레이션 결과 산출
     ↓
[yeah 브랜치 / XAI] 실패 원인 분석 및 환경 변수 기여도 정량화
     ↓
[LLM 브랜치] XAI 결과 기반 반사실적 시나리오 자동 생성
     ↓
[SI 브랜치] 생성된 시나리오 재주입 → 반복
```

| 브랜치 | 역할 |
|--------|------|
| [SI 브랜치](https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/MATLAB/Simulink/SI) | MATLAB/Simulink 3D 시뮬레이션 실행 및 탐지 결과 산출 |
| [yeah 브랜치](https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/yeah) (현재) | XAI 기반 원인 분석 및 반사실적 경계 탐색 |
| [LLM 브랜치](https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/LLM) | LLM 기반 시나리오 자동 설계 및 자연어 보고서 생성 |

---

## 주요 기능

### 1. XAI 기반 실패 원인 분석
안개 밀도(`fog_density`), 조도(`illumination_lux`), 센서 노이즈(`camera_noise_level`) 등 환경 변수가 탐지 성능(mAP)에 미친 영향을 수치화합니다. 단순 PASS/FAIL 판정을 넘어, 어떤 변수가 실패에 얼마나 기여했는지 정량적으로 추정합니다.

### 2. 반사실적 경계 탐색 (Counterfactual Boundary Search)
비대칭 이분 탐색 알고리즘을 통해 탐지 모델이 붕괴되는 **임계 환경 조건(failure boundary)**을 자동으로 식별합니다. 성공 케이스를 기점으로 환경을 점진적으로 악화시켜 최소 실패 조건을 도출합니다.

### 3. LLM 연동 JSON 출력
분석 결과를 구조화된 JSON 포맷(`xai_input.json`)으로 출력하여 LLM 브랜치가 즉시 읽을 수 있는 형태로 전달합니다.

### 4. 테스트 케이스 자산화 및 Replay
생성된 테스트 케이스를 JSON으로 저장하고 재실행(Replay)하여 검증 결과의 재현성을 보장합니다.

---

## 폴더 구조

```
Counterfactual-XAI-Verifier/  (yeah 브랜치)
│
├── xai/
│   ├── __init__.py                    # 모듈 진입점 (analyze_xai_dummy, generate_counterfactual_and_boundary, I/O 어댑터 export)
│   ├── dummy_analyzer.py              # 환경 변수 기여도 정규화 및 XAI 요약 산출
│   ├── counterfactual_boundary.py     # 반사실적 경계 탐색 알고리즘 (이분 탐색 기반)
│   └── io_adapter.py                  # JSON 입출력 및 XAI 입력 패킷 빌더
│
├── schemas/
│   └── xai_input.schema.json          # LLM 브랜치 연동용 JSON 스키마 정의
│
├── data/
│   ├── scenario_iter_001.json         # 시나리오 입력 예시
│   ├── sim_result_iter_001.json       # 시뮬레이션 결과 예시
│   ├── eval_iter_001.json             # 평가 결과 예시
│   └── xai_input.json                 # XAI → LLM 전달용 최종 출력
│
├── simulator.py                       # 더미 시뮬레이터 (환경 파라미터 기반 리스크 산출)
└── README.md
```

---

## 입출력 스키마

### XAI 입력 (SI 브랜치로부터)

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

### XAI 출력 (LLM 브랜치 입력용)

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
      {"name": "fog_density",      "importance": 0.409},
      {"name": "illumination_lux", "importance": 0.455},
      {"name": "camera_noise",     "importance": 0.136}
    ],
    "attention_summary": "mAP50 is 0.4096 under fog=30.0, illum=4000.0, noise=0.10"
  }
}
```

---

## 실행 방법

### 요구 사항

- Python 3.10 이상

### 설치

```bash
git clone https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier.git
cd Counterfactual-XAI-Verifier
git checkout yeah
```

### XAI 분석 실행

```bash
python -c "
from xai import analyze_xai_dummy, get_example_sim_log
result = analyze_xai_dummy(get_example_sim_log())
import json; print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

### 반사실적 경계 탐색 실행

```bash
python -c "
from xai import generate_counterfactual_and_boundary
# LLM 브랜치로부터 전달받은 xai_input 패킷을 인자로 전달
result = generate_counterfactual_and_boundary(xai_input_packet)
import json; print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

---

## 브랜치 간 연동 원칙

- 함수명·파일명·JSON 키는 SI/LLM 브랜치 스타일을 우선 재사용
- 새 네이밍은 최소화하고 `snake_case` 유지
- 브랜치 간 호환성을 최우선으로 유지; 병합 비용이 큰 rename은 지양

---

## 한계 및 향후 과제

| 한계 | 현황 | 향후 계획 |
|------|------|-----------|
| 단일 임무 도메인 | 국경 산악 감시(UAV 객체 탐지)에서만 검증 완료 | 방산·항공·자율주행 등 다중 도메인으로 확장 |
| 실제 하드웨어 미연동 | 가상 시뮬레이션 환경에 국한 | 실제 드론 센서와 HIL(Hardware-in-the-Loop) 검증 구현 |
| 기여도 분석의 근사성 | 상관 기반 통계적 근사치 | 인과추론(Causal Inference) 기법 적용으로 설명력 강화 |

---

## 팀

**팀 봄동두쫀쿠** — 김진우, 신지아, 정준혁
