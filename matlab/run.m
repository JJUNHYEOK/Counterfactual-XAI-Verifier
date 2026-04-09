function history = run(maxIter)
if nargin < 1
    maxIter = 5;
end

mdl = "uav_cf_viewer";
load_system(mdl);

history = struct([]);

for k = 1:maxIter
    % 1) 현재 시나리오 JSON 읽기
    scenario = jsondecode(fileread(fullfile(pwd,"data","rescenario_01.json")));

    % 2) Simulink 입력 설정
    simIn = Simulink.SimulationInput(mdl);
    simIn = setVariable(simIn,"FOG_DENSITY_PERCENT",scenario.environment_parameters.fog_density_percent);
    simIn = setVariable(simIn,"ILLUMINATION_LUX",scenario.environment_parameters.illumination_lux);
    simIn = setVariable(simIn,"CAMERA_NOISE_LEVEL",scenario.environment_parameters.camera_noise_level);
    simIn = setModelParameter(simIn,"StopTime","20");

    % 3) 시뮬레이션 실행
    simOut = sim(simIn);

    % 4) 결과 수집
    result = collect_results(simOut);

    % 5) 분석
    analysis = analyze_results(result);

    % 6) 더미 XAI JSON 생성
    xaiInput = build_xai_stub(scenario,analysis,k);
    fid = fopen(fullfile(pwd,"data","xai_input.json"),"w");
    fwrite(fid,jsonencode(xaiInput,PrettyPrint=true),"char");
    fclose(fid);

    % 7) Python LLM 호출
    system("python main.py --input_json data/xai_input.json --output_yaml data/rescenario_01.json");

    history(k).scenario = scenario;
    history(k).result = result;
    history(k).analysis = analysis;
end
end