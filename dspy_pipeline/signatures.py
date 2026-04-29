"""DSPy Signature for UAV adversarial scenario generation.

Defines the LLM's I/O contract: given XAI feedback from the previous
Simulink run, generate environment parameters that cause mission failure.
"""

import dspy


class UAVAdversarialScenario(dspy.Signature):
    """UAV 산악 정찰 임무에서 객체 탐지 실패(Mission_Success=0)를 유발하는
    적대적 환경 시나리오를 생성합니다.

    검증 목표 — 세 요구사항 중 하나 이상 위반 시 임무 실패:
      REQ-1: mAP50 >= 0.85  (탐지 정확도 — 낮을수록 실패)
      REQ-2: 최소이격거리 >= 2.0m  (안전 간격 — 낮을수록 실패)
      REQ-3: 연속 미탐지 프레임 <= 3  (탐지 연속성 — 높을수록 실패)

    반사실적(Counterfactual) 탐색 원칙:
      1) 최소 파라미터 변화로 실패 경계(decision boundary)를 정밀 탐색.
      2) XAI dominant_factors가 지목한 취약 파라미터를 집중 공략.
      3) 단일 변수가 아닌 복합 결함(Composite Fault) 조합 우선:
           DIF (Weather + Blur): fog_density_percent + low illumination_lux
           TIS (Sensor + Blur):  camera_noise_level + fog_density_percent
      4) 개별 변수를 이전 대비 5~15% 내외로 점진적으로 조정.
         mAP50이 0.5 미만이면 과도한 공격 → 파라미터를 20~30% 완화.
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
