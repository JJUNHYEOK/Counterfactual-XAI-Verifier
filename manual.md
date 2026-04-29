┌─────────────────────────────────────────────────────────────┐
│           경로 A: Python 독립 실행 (MATLAB Engine)            │
│                                                             │
│  run_dspy_adversarial.py                                    │
│    ├─ configure_lm()           ← .env 의 API 키 읽기         │
│    ├─ init_metric_bridge('engine')                          │
│    │    └─ matlab.engine.start_matlab()                     │
│    │         └─ mountain_uav_model.slx  ←───── 기존 모델     │
│    │              └─ requirements_eval.m  ←── 기존 평가      │
│    ├─ load_training_examples()  ← data/xai_input_iter_*.json│
│    ├─ BootstrapFewShot.compile()                            │
│    └─ 반복 루프: simulate → DSPy generate → simulate …       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│         경로 B: MATLAB 루프 유지 (subprocess 방식)            │
│                                                             │
│  run_counterfactual_loop.m  (기존 그대로)                    │
│    ├─ sim('mountain_uav_model')  ←────────── 기존 모델        │
│    ├─ requirements_eval(simOut)  ←────────── 기존 평가        │
│    ├─ build_xai_input()          ←────────── 기존 XAI 빌더   │
│    └─ system("python -m llm_agent.dspy_for_simulink ...")   │
│              │                                              │
│              └─ dspy_for_simulink.py  ← NEW                 │
│                   ├─ DSPy ChainOfThought 모듈 실행           │
│                   ├─ compiled_program.json 로드 (있으면)      │
│                   └─ scenario_iter_NNN.json 생성            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│         경로 C: Mock 모드 (MATLAB 없이 즉시 테스트)           │
│                                                             │
│  python run_dspy_adversarial.py --sim-mode mock             │
│    └─ MatlabSimulinkBridge._run_mock()                      │
│         F_detector 수식 Python 재현:                         │
│         visibility = 1 - 0.6*fog - 0.20*low_light - ...    │
│         (build_mountain_uav_model.m 와 동일한 수식)           │
└─────────────────────────────────────────────────────────────┘

1단계: 패키지 설치


cd Counterfactual-XAI-Verifier
pip install dspy>=2.5.0
2단계-A: MATLAB 없이 즉시 데모 (mock 모드)


python run_dspy_adversarial.py --sim-mode mock --iterations 10
2단계-B: 실제 Simulink 연동 (Engine 모드)


pip install matlabengine          # MATLAB R2023b+ 설치 필요
python run_dspy_adversarial.py --sim-mode engine --iterations 15
2단계-C: 기존 MATLAB 루프에 DSPy 붙이기 (run_counterfactual_loop.m 이미 수정 완료)


% MATLAB 커맨드 창에서
run_counterfactual_loop(10)       % DSPy가 시나리오 생성
run_counterfactual_loop(10, struct('no_llm', true))  % 규칙 기반 fallback

