# 지능형 요구사항 검증 시스템 - XAI 파이프라인 (Zone 3)

본 브랜치(Zone 3)는 UAV 자율비행 충돌 회피 검증 파이프라인에서  
**XAI 분석 모듈의 Week 1 Dummy 루프**를 구현합니다.

현재 목적은 실제 SHAP/모델 정확도가 아니라,  
Simulator 출력(dict)을 받아 XAI 출력(dict)으로 **에러 없이 전달**되는지 검증하는 것입니다.

## 📂 디렉토리 구조 (File Tree)

```text
Counterfactual-XAI-Verifier/
├── main.py                       # LLM -> Simulator -> XAI 더미 루프 실행
├── simulator.py                  # 시뮬레이터 더미 로그 생성
└── xai/
    ├── __init__.py
    └── dummy_analyzer.py         # analyze_xai_dummy(sim_log) 구현
```

## 🎯 Week 1 Dummy 구현 범위

- 함수명: `analyze_xai_dummy(sim_log)`
- 입력: Simulator가 넘겨주는 `dict` 형태 로그
- 출력: 고정 더미 결과 `{"wind_speed_importance": 0.68}`
- 제외: SHAP, feature importance 학습/추론 로직 (Week 2 이후)

## 🧾 팀 공통 스키마 (초안)

입력 `sim_log`

```python
{
  "status": "success",
  "min_distance": 2.4,
  "wind_speed": 4.0
}
```

출력

```python
{
  "wind_speed_importance": 0.68
}
```

## ⚙️ 실행 방법 (Usage)

프로젝트 루트에서 실행:

```bash
python main.py
```

실행 흐름:

1. `generate_params_dummy()`가 테스트 파라미터 생성
2. `Simulator.run_sim_dummy()`가 더미 시뮬레이션 로그 반환
3. `analyze_xai_dummy(sim_log)`가 고정 XAI 더미 결과 반환

## ✅ 완료 기준 체크

- `from xai.dummy_analyzer import analyze_xai_dummy` 정상 import
- 예시 `sim_log` 입력 시 dict 반환
- Week 1 범위만 구현 (확장 로직 미포함)

## 📌 향후 개발 계획 (Next Steps)

- Week 2: 실제 XAI 연산 로직(예: 경량 feature importance) 연결
- Week 3: 예외 처리/재시도/타입 검증 강화
- Week 4: Counterfactual 생성 모듈(Zone 4)과 통합 테스트
