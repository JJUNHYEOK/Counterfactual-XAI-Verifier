function result = run_experiment()

modelName = "my_loop_model";
load_system(modelName);

% Simulink용 기본 변수 (초기화용)
t0 = [0 5 10 15]';
u0 = [0 1 2 0]';

assignin('base', 'scenario', timeseries(u0, t0));
assignin('base', 'expected', timeseries(u0, t0));

% ✅ 단일 Fail 시나리오
sc = make_fail_scenario();

scenario = timeseries(sc.u, sc.t);
expected = make_expected_output(sc.t, sc.u);

assignin('base', 'scenario', scenario);
assignin('base', 'expected', expected);

simIn = Simulink.SimulationInput(modelName);
simIn = simIn.setVariable("scenario", scenario);
simIn = simIn.setVariable("expected", expected);
simIn = simIn.setModelParameter("StopTime", num2str(sc.t(end)));

simOut = sim(simIn);

% 결과 분석
result = analyze_results(sc, simOut);

disp("===== RESULT SUMMARY =====");
disp(result);

% LLM 해석
prompt = get_interpreter_system_prompt();
interpretation = call_llm_interpret(prompt, result);

disp("===== LLM INTERPRETATION =====");
disp(interpretation);

end
