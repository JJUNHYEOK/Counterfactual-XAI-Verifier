# Counterfactual-XAI-Verifier (XAI)

## XAI 역할

XAI는 XGBoost와 같은 예측 모델을 사용하지 않고, 실제 Simulink 실행 결과 함수에 KernelSHAP을 직접 적용하여 환경 변수별 mAP50 기여도를 산출한다. 산출된 SHAP 기반 기여도는 LLM의 후속 counterfactual 시나리오 생성을 위한 guidance signal로 사용된다.

## 기존 대비 변경점

- 기존: Simulink 결과 데이터를 XGBoost surrogate model로 학습하고, SHAP으로 XGBoost 예측을 설명
- 수정 후: XGBoost 없이 KernelSHAP이 Simulink 실행 결과 함수 자체를 직접 설명
- 목적: 실제 시뮬레이션 결과 기반의 SHAP-guided Counterfactual 시나리오 생성을 지원

## 핵심 원칙

- XAI는 예측 모델을 학습하지 않습니다.
- XAI는 XGBoost/RandomForest surrogate를 사용하지 않습니다.
- XAI는 LLM을 호출하지 않습니다.
- XAI는 시뮬레이션 실행 방식을 변경하지 않습니다.
- XAI는 입력을 읽고 분석 결과 JSON만 생성합니다.

## 입력

권장 입력 필드:

- `current_scenario`
- `previous_scenario`
- `scenario_history`
- `result_status`
- `map50`
- `safety_line`
- `feature_ranges`
- Simulink 실행 결과 callable (in-process 전달) 또는 실행 이력 기반 결과

참고:

- JSON 자체에는 callable 객체를 직접 저장할 수 없으므로, 런타임에서 `simulink_callable` 인자로 전달하는 방식을 지원합니다.
- callable이 없으면 제공된 실행 이력/재실행 결과를 사용한 fallback 평가를 사용합니다.

## PASS / FAIL 동작

### PASS

- PASS 상태에서도 `mAP50` 하락 또는 `safety_margin` 감소가 있으면 KernelSHAP으로 저하 기여 변수를 분석합니다.
- 상위 변수는 다음 시나리오를 더 가혹하게 만들기 위한 후보로 출력됩니다.

예:

- `fog` 상위 기여: `increase`
- `illumination` 상위 기여: `decrease`
- `noise` 상위 기여: `increase`

### FAIL

- 현재 FAIL 실행에 대해 KernelSHAP 기여도를 계산합니다.
- SHAP 상위 변수를 `boundary_focus_features`에 저장합니다.
- 경계 탐색/반사실 조정은 해당 변수 중심 guidance를 따릅니다.

## 출력

기존 출력 파일은 유지합니다.

- `counterfactual_explanations.json`
- `boundary_candidates.json`

추가 출력 파일:

- `xai_signals_step_n.json`

핵심 출력 형식:

```json
{
  "method": "simulation_grounded_kernelshap",
  "result_status": "PASS or FAIL",
  "risk_level": "safe / degrading / near_boundary / failed",
  "top_features": [
    {
      "feature": "환경 변수명",
      "shap_value": 0.0,
      "shap_importance": 0.0,
      "direction": "increase / decrease / adjust_near_boundary / keep",
      "reason": "KernelSHAP 분석에 기반한 LLM용 설명"
    }
  ],
  "boundary_focus_features": [],
  "llm_guidance": "다음 counterfactual 시나리오 생성을 위한 짧은 지침"
}
```

호환성 유지 필드:

- `schema_version`
- `base_case`
- `minimal_change_candidates`
- `nearest_boundary_candidates`
- `guidance`
- `attribution_score` (top_features 내 호환용)

## 실행

```bash
python main_counterfactual.py \
  --input_path data/counterfactual_case_input.json \
  --output_dir outputs_counterfactual
```

산출물:

- `counterfactual_explanations.json`
- `boundary_candidates.json`
- `xai_signals_step_n.json`
