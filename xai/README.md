# Counterfactual-XAI-Verifier (XAI Branch)

본 브랜치는 `SI -> XAI -> LLM -> SI` 순환 파이프라인에서 **XAI 모듈**을 담당합니다.
현재 XAI의 1순위 목적은 feature importance 시각화가 아니라, **counterfactual explanation + pass/fail 경계(boundary) 탐색**입니다.

## 브랜치 링크

- SI 브랜치: [https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/MATLAB/Simulink/SI](https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/MATLAB/Simulink/SI)
- XAI 브랜치: [https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/XAI](https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/XAI)
- LLM 브랜치: [https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/LLM](https://github.com/JJUNHYEOK/Counterfactual-XAI-Verifier/tree/LLM)

## 현재 XAI 모듈 목적

1. 실패(또는 특정) 시나리오를 입력으로 받아
2. 결과를 뒤집는 최소 변화 조건(pass<->fail) 후보를 찾고
3. decision boundary에 가까운 후보를 함께 산출해
4. LLM이 후속 counterfactual 시나리오 생성에 재사용 가능한 JSON을 제공

## 중요: SHAP + Counterfactual 결합 방식

- 현재 메인 흐름은 **counterfactual/boundary 탐색**입니다.
- 여기에 SHAP을 결합하려면 `--shap_json_path`로 `tabular_xai` 결과(`xai_llm_output.json`)를 넣어
  **SHAP-guided counterfactual**로 실행합니다.
- `tabular_xai/`는 SHAP 신호 생성용 보조 모듈이며, 탐색 실행 엔트리는 `main_counterfactual.py`입니다.

## 전체 파이프라인 연결 위치

1. SI 브랜치: 시나리오 실행 + 결과 로그 생성
2. XAI 브랜치(현재 구현): counterfactual/boundary 후보 생성
3. LLM 브랜치: XAI 결과(JSON) 기반으로 다음 가혹 시나리오 생성
4. 다시 SI에 주입해 반복 검증

## 폴더 구조 (현재 중심)

```text
Counterfactual-XAI-Verifier/
├── data/
│   ├── counterfactual_case_input.json
│   ├── scenario_iter_001.json
│   ├── sim_result_iter_001.json
│   ├── eval_iter_001.json
│   └── xai_input.json
├── schemas/
│   └── xai_input.schema.json
├── xai/
│   ├── counterfactual_boundary.py   # 핵심 탐색 모듈
│   ├── dummy_analyzer.py
│   ├── io_adapter.py
│   └── __init__.py
├── main_counterfactual.py           # 현재 권장 엔트리 포인트
├── main.py                          # 1주차 통합 더미 파이프라인
├── main_dummy.py
└── simulator.py
```

## 입력 파일 형식 (counterfactual)

권장 입력 예시는 `data/counterfactual_case_input.json`입니다.
simulator 더미 기반 입력 예시는 `data/counterfactual_case_sim_dummy.json`입니다.
핵심 필드:

- `scene_id`
- `scenario.environment_parameters`
- `performance_signals.map50`, `performance_signals.threshold`
- `xai_signals.dominant_factors`
- `search_space.bounds`, `search_space.mutable_parameters` (선택)
- `counterfactual_request.target_status`, `counterfactual_request.evaluator` (선택)

## 출력 파일 형식

실행 결과는 기본적으로 아래 2개 JSON으로 저장됩니다.

1. `counterfactual_explanations.json`
- `source_status`, `target_status`
- `base_case.decision_margin`
- `minimal_change_candidates[]`
- 후보별 `parameter_changes` (어떤 파라미터를 얼마나 변경해야 하는지)

2. `boundary_candidates.json`
- `base_case.distance_to_boundary`
- `nearest_boundary_candidates[]`
- 후보별 `decision_margin` 및 `distance_l1_normalized`

## 실행 방법

```bash
cd XAI
python main_counterfactual.py \
  --input_path data/counterfactual_case_input.json \
  --output_dir outputs_counterfactual \
  --mode auto \
  --random_seed 42
```

simulator dummy 기반 샘플:

```bash
cd XAI
python main_counterfactual.py \
  --input_path data/counterfactual_case_sim_dummy.json \
  --output_dir outputs_counterfactual_sim \
  --mode auto \
  --random_seed 42
```

SHAP 결합 실행 예시:

```bash
cd XAI
python main_counterfactual.py \
  --input_path data/counterfactual_case_input.json \
  --shap_json_path tabular_xai/outputs_llm_ready/xai_llm_output.json \
  --shap_case_id case_0007 \
  --output_dir outputs_counterfactual_shap \
  --mode map50_proxy \
  --random_seed 42
```

선택 인자:
- `--target_status PASS|FAIL` (미지정 시 자동 반전)
- `--num_counterfactuals 3`
- `--num_boundaries 5`

## 1주차 더미 파이프라인(유지)

```bash
cd XAI
python main.py
```

## 네이밍/연동 원칙

- 기존 SI/LLM 스타일의 snake_case 키를 우선 재사용
- 큰 rename 없이 모듈 추가 방식으로 확장
- 입출력 변경 시 README와 스키마를 함께 갱신
