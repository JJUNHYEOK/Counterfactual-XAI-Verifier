# Counterfactual-XAI-Verifier (XAI Branch)

본 브랜치는 UAV 검증 파이프라인에서 **XAI(설명) 파트**를 담당합니다.
XAI는 별도 예측 모델을 만드는 모듈이 아니라, **SI 시뮬레이션/평가 결과를 기반으로 환경 변수 기여도를 설명하는 직접형(post-hoc) 모듈**입니다.

## 브랜치 링크

- SI 브랜치: [https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/MATLAB/Simulink/SI](https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/MATLAB/Simulink/SI)
- XAI 브랜치: [https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/XAI](https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/XAI)
- LLM 브랜치: [https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/LLM](https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/LLM)

## 전체 파이프라인

1. SI 브랜치에서 시나리오 실행 및 결과 산출
2. XAI 브랜치에서 결과 원인 분석 및 변수 기여도 정량화
3. LLM 브랜치에서 XAI 결과를 읽고 counterfactual 시나리오 생성
4. 생성된 시나리오를 다시 SI 브랜치에 주입

## XAI의 현재 역할

- 입력: SI에서 생성된 `scenario`, `sim_result`, `eval_result`
- 처리: 실패/성능저하 원인에 대한 환경 변수 기여도 계산
- 출력: LLM이 바로 읽을 수 있는 `xai_input.json`

즉, 현재 프로젝트에서 XAI 구현은 **환경 변수 기여도 분석 자체**입니다.

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
├── data/
│   ├── scenario_iter_001.json
│   ├── sim_result_iter_001.json
│   ├── eval_iter_001.json
│   └── xai_input.json
├── schemas/
│   └── xai_input.schema.json
├── xai/
│   ├── dummy_analyzer.py
│   ├── io_adapter.py
│   └── __init__.py
├── main.py
└── simulator.py
```

## 실행

```bash
python main.py
```

## 네이밍/연동 원칙

- 함수명/파일명/JSON key는 SI/LLM 브랜치 스타일을 우선 재사용
- 새 네이밍은 최소화하고 snake_case 유지
- 병합 비용이 큰 rename은 피하고, 브랜치 간 호환성을 최우선으로 유지
