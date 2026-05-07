"""DSPy Signature for UAV adversarial scenario generation.

Defines the LLM's I/O contract: given XAI feedback from the previous
Simulink run, generate environment parameters that cause mission failure.
"""

import dspy


class UAVAdversarialScenario(dspy.Signature):
    """UAV 국경 산악 감시 임무의 PASS↔FAIL 경계를 찾기 위한
    EXPLORE-단계 시나리오 생성기.

    호출 컨텍스트 — 본 시그니처는 "PUSH (탐색)" 모드에서만 호출됩니다:
      · 호출 조건: 지금까지의 모든 시뮬이 PASS이고, 아직 FAIL anchor가
        발견되지 않은 상태 → boundary가 어디 있는지 모름
      · 역할:     SHAP signals를 활용해 점진적으로 가혹화하여
                  첫 FAIL을 빠르게 찾도록 다음 환경을 제안
      · 일단 첫 FAIL이 발견되면 본 시그니처는 더 이상 호출되지 않으며,
        결정론적 bisection이 PASS↔FAIL 사이를 좁히며 boundary를 정밀화함

    검증 목표 — 두 요구사항 중 하나 이상 위반 시 임무 실패:
      REQ-1: mAP50 >= 0.50  (탐지 정확도 — Safety Line, baseline 대비 30%↓)
      REQ-3: 연속 미탐지 프레임 <= 3  (탐지 연속성 — 보고 누락 위험)
    (REQ-2 이격거리는 본 연구에서는 비변동이라 평가 대상에서 제외)

    Push 전략 (공격적 변이 — visible counterfactual):
      1) XAI dominant_factors / SHAP global_importance가 지목한
         취약 파라미터를 우선 공략 (가장 영향력 큰 변수부터)
      2) 단일 변수가 아닌 복합 결함 (Composite Fault) 조합 우선:
           DIF (Weather × Lighting): fog_density_percent ↑↑ + illumination_lux ↓↓
           TIS (Sensor × Weather):   camera_noise_level ↑↑ + fog_density_percent ↑
      3) 직전 대비 각 파라미터를 25~50 % 폭으로 큰 step 조정 — 반사실 시나리오는
         미미한 변화가 아니라 "확실히 더 가혹한 조건" 으로 명확히 식별되어야 한다.
         예: fog 20 % → 50 %, illum 6000 → 3000, noise 0.10 → 0.30.
      4) 단, mAP50 이 임계값(0.50) 보다 크게 미달(예: 0.20 이하)될 정도로
         과도하게 가혹하면 boundary 위치 정보가 부정확해지므로,
         이전 단계의 mAP50 트렌드를 보고 "한 단계 더 가혹"한 정도로 조정.
      5) 변경 폭이 작아 보이면 큰 의미가 없다 — 사용자에게 시각적·서사적으로
         "환경이 분명히 악화되었다"고 보일 만한 폭의 변화를 권장.
    """

    # ── Inputs ──────────────────────────────────────────────────────────────
    iteration_history: str = dspy.InputField(
        desc=(
            "이전 시뮬레이션 반복들의 환경 파라미터와 결과 이력 (JSON 배열). "
            "각 항목: {iter, fog_density_percent, illumination_lux, "
            "camera_noise_level, map50, all_passed, violated_count}"
        )
    )
    xai_analysis: str = dspy.InputField(
        desc=(
            "XAI 분석 결과 (JSON). 탐지 실패에 가장 크게 기여하는 파라미터와 "
            "중요도(0~1) 포함. "
            "형식: {method, dominant_factors:[{name,importance}], attention_summary}"
        )
    )
    current_performance: str = dspy.InputField(
        desc=(
            "현재 시뮬레이션의 성능 지표 (JSON). "
            "형식: {map50, min_clearance_m, max_consecutive_misses, "
            "violated_count, worst_requirement, failure_type}"
        )
    )

    # ── Outputs ─────────────────────────────────────────────────────────────
    analysis: str = dspy.OutputField(
        desc=(
            "3단계 구조의 분석 보고서:\n"
            "  📊 현재 상황 분석: 어떤 요구사항이 왜 실패/성공하고 있는가\n"
            "  🎯 공격 전략: 어떤 파라미터를 어떻게 조작할 것인가 (DIF/TIS 복합 결함)\n"
            "  📉 예상 효과: 이 변화가 어떤 요구사항을 어떤 메커니즘으로 위반하는가"
        )
    )
    environment_parameters_json: str = dspy.OutputField(
        desc=(
            "다음 시뮬레이션을 위한 환경 파라미터 — 반드시 유효한 JSON 객체만 출력:\n"
            '{"fog_density_percent": <float 0.0~100.0>, '
            '"illumination_lux": <float 200.0~20000.0>, '
            '"camera_noise_level": <float 0.0~0.6>}'
        )
    )
    target_hypothesis: str = dspy.OutputField(
        desc=(
            "이 시나리오가 UAV 임무 실패를 유발할 것이라는 구체적 가설 (한 문장). "
            "예: 'fog 65%와 illumination 800lux의 DIF 복합 결함으로 "
            "REQ-1 mAP50이 0.85 미만으로 저하될 것'"
        )
    )
