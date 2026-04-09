function history = run_counterfactual_detection_pipeline_v2(maxIter, useStubDetection, edgeBudget)
% run_counterfactual_detection_pipeline_v2
% -------------------------------------------------------------------------
% 목적
%   LLM 기반 반사실적(Counterfactual) 기상 시나리오를 자동 생성하여,
%   드론 객체 탐지 모델이 최소 요구 성능(mAP50 85%)을 위반하는
%   엣지 케이스를 발굴한다.
%
% 개선 사항
%   1) LLM 출력이 JSON이든 YAML 유사 텍스트든 읽을 수 있도록 보강
%   2) edge case를 하나 찾았다고 바로 종료하지 않고 edgeBudget까지 탐색
%   3) iteration별 "실제로 사용한 시나리오"와 "다음 시나리오"를 분리 저장
%   4) 실행 전 3인칭 드론 뷰어 패치를 자동 적용 시도
%   5) stub 평가의 재현성을 위해 RNG 고정
%
% 사용
%   history = run_counterfactual_detection_pipeline_v2
%   history = run_counterfactual_detection_pipeline_v2(5, true, 3)
% -------------------------------------------------------------------------

if nargin < 1 || isempty(maxIter)
    maxIter = 5;
end
if nargin < 2 || isempty(useStubDetection)
    useStubDetection = true;
end
if nargin < 3 || isempty(edgeBudget)
    edgeBudget = 3;
end

rng(0, "twister");

mdl = "uav_cf_viewer";
dataDir = fullfile(pwd, "data");
if ~exist(dataDir, "dir")
    mkdir(dataDir);
end

history = struct([]);
edgeCaseCount = 0;

% -------------------------------------------------------------------------
% 0) 모델 준비
% -------------------------------------------------------------------------
if ~bdIsLoaded(mdl) && ~isfile(mdl + ".slx")
    if exist("build_and_run_uav_cf_viewer", "file") == 2
        fprintf("[INIT] uav_cf_viewer.slx가 없어 자동 생성 시도\n");
        build_and_run_uav_cf_viewer(false);
        bdclose all;
    else
        error("uav_cf_viewer.slx도 없고 build_and_run_uav_cf_viewer.m도 없습니다.");
    end
end

if ~isfile(mdl + ".slx")
    error("uav_cf_viewer.slx 생성에 실패했습니다.");
end

if ~bdIsLoaded(mdl)
    load_system(mdl);
end

% 3인칭 드론 뷰어 패치 자동 적용
if exist("patch_uav_cf_viewer_visibility", "file") == 2
    try
        patch_uav_cf_viewer_visibility();
    catch ME
        warning("[VIEW] 3인칭 뷰 패치 적용 실패: %s", ME.message);
    end
end

try
    set_param(mdl, "SimulationCommand", "Update");
catch
end

% -------------------------------------------------------------------------
% 1) 시작 시나리오 준비
% -------------------------------------------------------------------------
scenarioPath = fullfile(dataDir, "cf_case_01.json");
if ~isfile(scenarioPath)
    scenario = seed_initial_scenario();
    write_json(scenarioPath, scenario);
else
    scenario = read_llm_scenario_any(scenarioPath, seed_initial_scenario());
end

fprintf("============================================================\n");
fprintf("Counterfactual detection pipeline start (v2)\n");
fprintf("Goal            : discover weather-driven edge cases with mAP50 < 0.85\n");
fprintf("Model           : %s\n", mdl);
fprintf("Iterations      : %d\n", maxIter);
fprintf("Stub detection  : %d\n", useStubDetection);
fprintf("Edge budget     : %d\n", edgeBudget);
fprintf("============================================================\n");

% -------------------------------------------------------------------------
% 2) 반복 루프
% -------------------------------------------------------------------------
for k = 1:maxIter
    fprintf("\n================ Iteration %d / %d ================\n", k, maxIter);

    if k > 1 && isfile(scenarioPath)
        scenario = read_llm_scenario_any(scenarioPath, scenario);
    end

    currentScenario = scenario;
    iterTag = sprintf("iter_%03d", k);

    scenarioIterPath = fullfile(dataDir, "scenario_" + iterTag + ".json");
    write_json(scenarioIterPath, currentScenario);

    % 2-1) 시나리오 -> Simulink 변수 매핑
    simIn = build_sim_input_from_scenario(mdl, currentScenario);

    % 2-2) 시뮬레이션 실행
    fprintf("[SIM] Running Simulink scenario...\n");
    simOut = sim(simIn);

    % 2-3) 결과 수집
    simResult = collect_sim_results(simOut);
    write_json(fullfile(dataDir, "sim_result_" + iterTag + ".json"), simResult);

    % 2-4) 성능 평가
    if useStubDetection
        evalResult = stub_detection_eval(currentScenario, simResult);
    else
        evalResult = real_detector_hook(currentScenario, simResult);
    end
    write_json(fullfile(dataDir, "eval_" + iterTag + ".json"), evalResult);

    fprintf("[EVAL] mAP50 = %.4f | violated = %d\n", ...
        evalResult.map50, evalResult.requirement_violated);

    % 2-5) XAI 입력 생성
    xaiInput = build_xai_input_for_llm(currentScenario, simResult, evalResult, iterTag);
    xaiPath = fullfile(dataDir, "xai_input_" + iterTag + ".json");
    write_json(xaiPath, xaiInput);
    write_json(fullfile(dataDir, "xai_input.json"), xaiInput);

    % 2-6) edge case 저장
    if evalResult.requirement_violated
        edgeCaseCount = edgeCaseCount + 1;

        edgeCase.scenario = currentScenario;
        edgeCase.sim_result = simResult;
        edgeCase.eval = evalResult;
        edgeCase.xai_input = xaiInput;

        edgePath = fullfile(dataDir, "edge_case_" + iterTag + ".json");
        write_json(edgePath, edgeCase);
        fprintf("[EDGE] Requirement violated. Edge case #%d saved: %s\n", edgeCaseCount, edgePath);
    end

    % 2-7) 다음 시나리오 생성
    nextScenarioPath = fullfile(dataDir, "cf_case_01.json");
    [nextScenario, meta] = generate_next_scenario_with_llm_or_fallback_v2( ...
        xaiPath, nextScenarioPath, currentScenario, evalResult);

    fprintf("[NEXT] source=%s\n", meta.source);
    if strlength(meta.note) > 0
        fprintf("[NEXT] note=%s\n", meta.note);
    end

    scenario = nextScenario;

    % 2-8) 기록 저장
    history(k).scenario = currentScenario;
    history(k).next_scenario = nextScenario;
    history(k).sim_result = simResult;
    history(k).eval = evalResult;
    history(k).xai_input = xaiInput;
    history(k).next_source = meta.source;
    history(k).next_note = meta.note;

    % 2-9) 종료 조건
    if edgeCaseCount >= edgeBudget
        fprintf("[STOP] edge budget reached (%d/%d)\n", edgeCaseCount, edgeBudget);
        break;
    end
end

fprintf("\nPipeline finished.\n");
end

% =========================================================================
% Scenario normalization / seed
% =========================================================================

function scenario = seed_initial_scenario()
scenario = struct();
scenario.scenario_id = "scenario_001";
scenario.target_hypothesis = ...
    "Weather-driven counterfactual scenario to discover a UAV detection edge case with mAP50 below 0.85";
scenario.environment_parameters = struct( ...
    "fog_density_percent", 30.0, ...
    "illumination_lux", 4000.0, ...
    "camera_noise_level", 0.10);
scenario.llm_reasoning = ...
    "Initial seed scenario for weather degradation search.";
end

function scenario = normalize_llm_scenario(raw, fallbackScenario)
if nargin < 2 || isempty(fallbackScenario)
    fallbackScenario = seed_initial_scenario();
end

scenario = fallbackScenario;

if ~isstruct(raw)
    return;
end

if isfield(raw, "scenario_id")
    scenario.scenario_id = string(raw.scenario_id);
elseif isfield(raw, "scene_id")
    scenario.scenario_id = string(raw.scene_id);
end

if isfield(raw, "target_hypothesis")
    scenario.target_hypothesis = string(raw.target_hypothesis);
elseif isfield(raw, "llm_reasoning")
    scenario.target_hypothesis = string(raw.llm_reasoning);
end

if isfield(raw, "llm_reasoning")
    scenario.llm_reasoning = string(raw.llm_reasoning);
elseif isfield(raw, "target_hypothesis")
    scenario.llm_reasoning = string(raw.target_hypothesis);
end

if isfield(raw, "environment_parameters") && isstruct(raw.environment_parameters)
    env = raw.environment_parameters;
else
    env = struct();
end

if ~isfield(scenario, "environment_parameters") || ~isstruct(scenario.environment_parameters)
    scenario.environment_parameters = struct();
end

scenario.environment_parameters.fog_density_percent = ...
    get_or_default(env, "fog_density_percent", ...
    get_or_default(scenario.environment_parameters, "fog_density_percent", 30));

scenario.environment_parameters.illumination_lux = ...
    get_or_default(env, "illumination_lux", ...
    get_or_default(scenario.environment_parameters, "illumination_lux", 4000));

scenario.environment_parameters.camera_noise_level = ...
    get_or_default(env, "camera_noise_level", ...
    get_or_default(scenario.environment_parameters, "camera_noise_level", 0.1));
end

function scenario = read_llm_scenario_any(pathStr, fallbackScenario)
txt = strtrim(fileread(pathStr));
if strlength(txt) == 0
    error("빈 파일입니다: %s", pathStr);
end

txt = strip_code_fence(txt);

% 1) JSON 우선 시도
try
    raw = jsondecode(txt);
    scenario = normalize_llm_scenario(raw, fallbackScenario);
    return;
catch
end

% 2) YAML/자유 텍스트 유사 형식에서 필요한 핵심 필드만 추출
scenario = fallbackScenario;
changed = false;

[val, ok] = extract_string_field(txt, "scenario_id");
if ok
    scenario.scenario_id = string(val);
    changed = true;
end

[val, ok] = extract_string_field(txt, "target_hypothesis");
if ok
    scenario.target_hypothesis = string(val);
    changed = true;
end

[val, ok] = extract_string_field(txt, "llm_reasoning");
if ok
    scenario.llm_reasoning = string(val);
    changed = true;
end

[num, ok] = extract_numeric_field(txt, "fog_density_percent");
if ok
    scenario.environment_parameters.fog_density_percent = num;
    changed = true;
end

[num, ok] = extract_numeric_field(txt, "illumination_lux");
if ok
    scenario.environment_parameters.illumination_lux = num;
    changed = true;
end

[num, ok] = extract_numeric_field(txt, "camera_noise_level");
if ok
    scenario.environment_parameters.camera_noise_level = num;
    changed = true;
end

if ~changed
    error("JSON/YAML 유사 출력에서 시나리오 필드를 추출하지 못했습니다: %s", pathStr);
end
end

function txt = strip_code_fence(txt)
txt = regexprep(txt, '^\s*```(?:json|yaml|yml)?\s*', '', 'once');
txt = regexprep(txt, '\s*```\s*$', '', 'once');
txt = string(txt);
end

function [val, ok] = extract_string_field(txt, fieldName)
ok = false;
val = "";

pat = "(?m)[""'']?" + regexptranslate("escape", fieldName) + "[""'']?\s*:\s*(.+?)\s*$";
tok = regexp(txt, pat, "tokens", "once");

if isempty(tok)
    return;
end

val = string(strtrim(tok{1}));
val = regexprep(val, '^[\"\'']', '');
val = regexprep(val, '[\"\''\,]\s*$', '');
val = strtrim(val);

% 중첩 객체 시작 같은 잘못된 매칭 방지
if startsWith(val, "{") || startsWith(val, "[")
    ok = false;
    val = "";
    return;
end

ok = strlength(val) > 0;
end

function [num, ok] = extract_numeric_field(txt, fieldName)
ok = false;
num = NaN;

pat = "(?m)[""'']?" + regexptranslate("escape", fieldName) + "[""'']?\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)";
tok = regexp(txt, pat, "tokens", "once");

if isempty(tok)
    return;
end

num = str2double(tok{1});
ok = ~isnan(num);
end

% =========================================================================
% Simulink input / result
% =========================================================================

function simIn = build_sim_input_from_scenario(mdl, scenario)
env = scenario.environment_parameters;

fog = get_or_default(env, "fog_density_percent", 30);
illum = get_or_default(env, "illumination_lux", 4000);
noise = get_or_default(env, "camera_noise_level", 0.1);

simIn = Simulink.SimulationInput(mdl);

simIn = setVariable(simIn, "FOG_DENSITY_PERCENT", fog);
simIn = setVariable(simIn, "ILLUMINATION_LUX", illum);
simIn = setVariable(simIn, "CAMERA_NOISE_LEVEL", noise);

% 3인칭 시점에서 잘 보이도록 초기 위치를 정리
simIn = setVariable(simIn, "UAV_X0", 0);
simIn = setVariable(simIn, "UAV_Y0", 0);
simIn = setVariable(simIn, "UAV_Z0", 2.5);

simIn = setVariable(simIn, "OBS_X0", 12);
simIn = setVariable(simIn, "OBS_Y0", 0);
simIn = setVariable(simIn, "OBS_Z0", 1.0);

simIn = setVariable(simIn, "REQ_MIN_CLEARANCE", 2.0);
simIn = setVariable(simIn, "REQ_MIN_CONFIDENCE", 0.55);

simIn = setModelParameter(simIn, "StopTime", "20");
end

function simResult = collect_sim_results(simOut)
simResult = struct();

simResult.collision_flag = read_last_value(simOut, "collision_flag_log", NaN);
simResult.running_min_dist = read_last_value(simOut, "running_min_dist_log", NaN);
simResult.avg_confidence = read_last_value(simOut, "avg_confidence_log", NaN);
simResult.miss_count = read_last_value(simOut, "miss_count_log", NaN);
simResult.requirement_satisfied = read_last_value(simOut, "requirement_satisfied_log", NaN);
simResult.risk_score = read_last_value(simOut, "risk_score_log", NaN);

simResult.uav_xyz_final = read_last_vector(simOut, "uav_xyz_log", [NaN NaN NaN]);
simResult.uav_rpy_final = read_last_vector(simOut, "uav_rpy_log", [NaN NaN NaN]);
simResult.obs_xyz_final = read_last_vector(simOut, "obs_xyz_log", [NaN NaN NaN]);

if isprop(simOut, "tout")
    simResult.tout_end = simOut.tout(end);
else
    simResult.tout_end = NaN;
end
end

% =========================================================================
% Detection evaluation
% =========================================================================

function evalResult = stub_detection_eval(scenario, simResult)
env = scenario.environment_parameters;
fog = get_or_default(env, "fog_density_percent", 0);
illum = get_or_default(env, "illumination_lux", 5000);
noise = get_or_default(env, "camera_noise_level", 0);

nFrames = 50;
gtBoxes = cell(nFrames,1);
predBoxes = cell(nFrames,1);
predScores = cell(nFrames,1);
predLabels = cell(nFrames,1);
gtLabels = cell(nFrames,1);

baseBox = [320 180 120 90];
deterioration = ...
      0.55 * min(1, fog / 100) ...
    + 0.20 * max(0, (3000 - illum) / 3000) ...
    + 0.15 * max(0, (illum - 10000) / 10000) ...
    + 0.25 * min(1, noise);

simPenalty = 0;
if ~isnan(simResult.avg_confidence)
    simPenalty = max(0, 0.8 - simResult.avg_confidence);
end

missProb = min(0.95, 0.05 + 0.75 * deterioration + 0.30 * simPenalty);
scoreMean = max(0.05, 0.95 - 0.75 * deterioration - 0.25 * simPenalty);
jitter = 8 + 45 * deterioration;

for i = 1:nFrames
    gtBoxes{i} = baseBox;
    gtLabels{i} = "obstacle";

    if rand() < missProb
        predBoxes{i} = zeros(0,4);
        predScores{i} = zeros(0,1);
        predLabels{i} = strings(0,1);
    else
        dx = round(randn() * jitter);
        dy = round(randn() * jitter);
        dw = round(randn() * jitter * 0.4);
        dh = round(randn() * jitter * 0.4);

        box = [
            baseBox(1) + dx, ...
            baseBox(2) + dy, ...
            max(20, baseBox(3) + dw), ...
            max(20, baseBox(4) + dh)
        ];

        score = min(1.0, max(0.01, scoreMean + 0.08 * randn()));

        predBoxes{i} = box;
        predScores{i} = score;
        predLabels{i} = "obstacle";
    end
end

[ap50, precisionCurve, recallCurve, tpCount, fpCount, fnCount] = ...
    compute_ap50_single_class(gtBoxes, gtLabels, predBoxes, predScores, predLabels, "obstacle");

evalResult = struct();
evalResult.class_name = "obstacle";
evalResult.ap50 = ap50;
evalResult.map50 = ap50;
evalResult.tp_count = tpCount;
evalResult.fp_count = fpCount;
evalResult.fn_count = fnCount;
evalResult.precision_curve = precisionCurve;
evalResult.recall_curve = recallCurve;
evalResult.requirement_threshold = 0.85;
evalResult.requirement_violated = ap50 < 0.85;
evalResult.summary = sprintf( ...
    "fog=%.1f, illum=%.1f, noise=%.2f -> mAP50=%.4f", ...
    fog, illum, noise, ap50);
end

function evalResult = real_detector_hook(~, ~)
error([ ...
    "real_detector_hook()는 아직 비어 있습니다.\n" ...
    "실제 카메라 프레임, GT bbox, detector 예측을 연결한 뒤 이 함수만 교체하세요." ...
]);
end

% =========================================================================
% XAI input for LLM
% =========================================================================

function xaiInput = build_xai_input_for_llm(scenario, simResult, evalResult, iterTag)
env = scenario.environment_parameters;

fog = get_or_default(env, "fog_density_percent", 0);
illum = get_or_default(env, "illumination_lux", 5000);
noise = get_or_default(env, "camera_noise_level", 0);

fogScore = min(1, fog / 100);
illumScore = min(1, abs(illum - 6000) / 6000);
noiseScore = min(1, noise);

total = fogScore + illumScore + noiseScore;
if total <= 0
    fogImp = 0.34; illumImp = 0.33; noiseImp = 0.33;
else
    fogImp = fogScore / total;
    illumImp = illumScore / total;
    noiseImp = noiseScore / total;
end

if evalResult.map50 < 0.85
    confTrend = "decreasing";
    missTrend = "increasing";
    failType = "detection_performance_drop";
else
    confTrend = "stable";
    missTrend = "stable";
    failType = "nominal";
end

xaiInput = struct();
xaiInput.scene_id = iterTag;
xaiInput.task = "uav_object_detection";
xaiInput.goal = "Find a weather-driven counterfactual edge case where mAP50 falls below 0.85";
xaiInput.perception = struct( ...
    "detector", "stub-or-external-detector", ...
    "input_resolution", [640 360], ...
    "detections", struct([]));

xaiInput.performance_signals = struct( ...
    "confidence_trend", confTrend, ...
    "miss_rate_trend", missTrend, ...
    "risk_score", min(1.0, 1.0 - evalResult.map50), ...
    "failure_type", failType, ...
    "map50", evalResult.map50, ...
    "threshold", 0.85);

xaiInput.xai_signals = struct( ...
    "method", "stub-xai", ...
    "dominant_factors", [ ...
        struct("name","fog_density","importance",round(fogImp,3)), ...
        struct("name","illumination_lux","importance",round(illumImp,3)), ...
        struct("name","camera_noise","importance",round(noiseImp,3)) ...
    ], ...
    "attention_summary", sprintf( ...
        "mAP50 is %.4f under fog=%.1f, illum=%.1f, noise=%.2f", ...
        evalResult.map50, fog, illum, noise));

xaiInput.scenario_constraints = struct( ...
    "allow_weather_change", true, ...
    "allow_lighting_change", true, ...
    "allow_obstacle_density_change", false);

xaiInput.sim_result = simResult;
end

% =========================================================================
% LLM next scenario generation
% =========================================================================

function [nextScenario, meta] = generate_next_scenario_with_llm_or_fallback_v2(xaiPath, outScenarioPath, prevScenario, evalResult)
nextScenario = [];
meta = struct("source","fallback","note","");

status = -999;

if exist("main.py", "file") == 2 || exist(fullfile(pwd,"main.py"), "file") == 2
    % 기존 Python 인터페이스와의 호환을 위해 --output_yaml 유지
    cmd = sprintf('python main.py --input_json "%s" --output_yaml "%s"', xaiPath, outScenarioPath);
    status = system(cmd);

    if isfile(outScenarioPath)
        try
            nextScenario = read_llm_scenario_any(outScenarioPath, prevScenario);
            write_json(outScenarioPath, nextScenario);  % MATLAB 쪽에서는 이후 항상 JSON으로 재저장
            meta.source = "llm";
            meta.note = sprintf("python status=%d, llm output accepted", status);
            return;
        catch ME
            meta.note = "LLM output file exists but parse/normalize failed: " + string(ME.message);
        end
    else
        meta.note = sprintf("python status=%d, no output file", status);
    end
end

fprintf("[LLM-FALLBACK] Using rule-based next scenario. %s\n", meta.note);

nextScenario = prevScenario;
env = nextScenario.environment_parameters;

fog = get_or_default(env, "fog_density_percent", 20);
illum = get_or_default(env, "illumination_lux", 5000);
noise = get_or_default(env, "camera_noise_level", 0.1);

if evalResult.map50 >= 0.85
    fog = min(100, fog + 15);
    illum = max(500, illum - 700);
    noise = min(1.0, noise + 0.08);
else
    fog = min(100, fog + 5);
    illum = max(500, illum - 250);
    noise = min(1.0, noise + 0.03);
end

nextScenario.scenario_id = string("fallback_next_" + string(randi(100000)));
nextScenario.target_hypothesis = ...
    "Rule-based weather mutation to continue searching for mAP50 < 0.85 edge cases";
nextScenario.environment_parameters.fog_density_percent = fog;
nextScenario.environment_parameters.illumination_lux = illum;
nextScenario.environment_parameters.camera_noise_level = noise;
nextScenario.llm_reasoning = "Fallback rule-based scenario update.";

write_json(outScenarioPath, nextScenario);
end

% =========================================================================
% AP50 computation
% =========================================================================

function [ap50, precisionCurve, recallCurve, tpCount, fpCount, fnCount] = ...
    compute_ap50_single_class(gtBoxes, gtLabels, predBoxes, predScores, predLabels, className)

detections = [];
nFrames = numel(gtBoxes);
totalGt = 0;

for i = 1:nFrames
    gtMask = gtLabels{i} == className;
    if isempty(gtMask)
        gtMask = false(0,1);
    end
    gt_i = gtBoxes{i};
    if ~isempty(gt_i)
        totalGt = totalGt + sum(gtMask);
    end

    pBox = predBoxes{i};
    pScore = predScores{i};
    pLabel = predLabels{i};

    if isempty(pBox)
        continue;
    end

    for j = 1:size(pBox,1)
        if string(pLabel(j)) == className
            d.frame = i;
            d.box = pBox(j,:);
            d.score = pScore(j);
            detections = [detections; d]; %#ok<AGROW>
        end
    end
end

if totalGt == 0
    ap50 = 0;
    precisionCurve = [];
    recallCurve = [];
    tpCount = 0;
    fpCount = 0;
    fnCount = 0;
    return;
end

if isempty(detections)
    ap50 = 0;
    precisionCurve = 0;
    recallCurve = 0;
    tpCount = 0;
    fpCount = 0;
    fnCount = totalGt;
    return;
end

scores = [detections.score]';
[~, order] = sort(scores, "descend");
detections = detections(order);

matched = cell(nFrames,1);
for i = 1:nFrames
    gt_i = gtBoxes{i};
    if isempty(gt_i)
        matched{i} = false(0,1);
    else
        matched{i} = false(size(gt_i,1),1);
    end
end

tp = zeros(numel(detections),1);
fp = zeros(numel(detections),1);

for k = 1:numel(detections)
    fr = detections(k).frame;
    gt_i = gtBoxes{fr};
    gt_l = gtLabels{fr};

    if isempty(gt_i)
        fp(k) = 1;
        continue;
    end

    gtMask = gt_l == className;
    gt_c = gt_i(gtMask,:);
    if isempty(gt_c)
        fp(k) = 1;
        continue;
    end

    bestIou = -inf;
    bestIdx = -1;

    for g = 1:size(gt_c,1)
        iou = bbox_iou_xywh(detections(k).box, gt_c(g,:));
        if iou > bestIou
            bestIou = iou;
            bestIdx = g;
        end
    end

    if bestIou >= 0.5
        mappedIdx = find(gtMask);
        realIdx = mappedIdx(bestIdx);

        if ~matched{fr}(realIdx)
            tp(k) = 1;
            matched{fr}(realIdx) = true;
        else
            fp(k) = 1;
        end
    else
        fp(k) = 1;
    end
end

cumTp = cumsum(tp);
cumFp = cumsum(fp);

precisionCurve = cumTp ./ max(cumTp + cumFp, eps);
recallCurve = cumTp / totalGt;

mrec = [0; recallCurve; 1];
mpre = [0; precisionCurve; 0];

for i = numel(mpre)-1:-1:1
    mpre(i) = max(mpre(i), mpre(i+1));
end

idx = find(mrec(2:end) ~= mrec(1:end-1));
ap50 = sum((mrec(idx+1) - mrec(idx)) .* mpre(idx+1));

tpCount = sum(tp);
fpCount = sum(fp);
fnCount = totalGt - tpCount;
end

function iou = bbox_iou_xywh(a, b)
ax1 = a(1); ay1 = a(2); ax2 = a(1) + a(3); ay2 = a(2) + a(4);
bx1 = b(1); by1 = b(2); bx2 = b(1) + b(3); by2 = b(2) + b(4);

ix1 = max(ax1, bx1);
iy1 = max(ay1, by1);
ix2 = min(ax2, bx2);
iy2 = min(ay2, by2);

iw = max(0, ix2 - ix1);
ih = max(0, iy2 - iy1);
interArea = iw * ih;

aArea = max(0, a(3)) * max(0, a(4));
bArea = max(0, b(3)) * max(0, b(4));
unionArea = aArea + bArea - interArea;

if unionArea <= 0
    iou = 0;
else
    iou = interArea / unionArea;
end
end

% =========================================================================
% Common utilities
% =========================================================================

function value = read_last_value(simOut, fieldName, defaultValue)
value = defaultValue;
try
    s = simOut.(fieldName);
    if isstruct(s) && isfield(s, "signals") && isfield(s.signals, "values")
        vals = s.signals.values;
        if isnumeric(vals)
            value = vals(end);
        end
    end
catch
end
end

function value = read_last_vector(simOut, fieldName, defaultValue)
value = defaultValue;
try
    s = simOut.(fieldName);
    if isstruct(s) && isfield(s, "signals") && isfield(s.signals, "values")
        vals = s.signals.values;
        if isnumeric(vals)
            if ndims(vals) == 2
                value = vals(end,:);
            elseif ndims(vals) == 3
                value = squeeze(vals(end,:,:));
            end
        end
    end
catch
end
end

function val = get_or_default(s, fieldName, defaultVal)
if isstruct(s) && isfield(s, fieldName)
    val = s.(fieldName);
else
    val = defaultVal;
end
end

function write_json(pathStr, obj)
txt = jsonencode(obj, PrettyPrint=true);
fid = fopen(pathStr, "w");
if fid < 0
    error("파일 저장 실패: %s", pathStr);
end
fwrite(fid, txt, "char");
fclose(fid);
end
