function patch_uav_cf_viewer_visibility()
% patch_uav_cf_viewer_visibility
% -------------------------------------------------------------------------
% 목적
%   뷰어에서 드론이 하늘만 보이지 않도록, 버전 의존성이 낮은
%   Custom 3인칭 시점을 강제로 설정
%
% 전략
%   1) Scene view를 무조건 Custom으로 둔다.
%   2) Initial viewer translation / rotation을 prompt 기준으로 찾는다.
%   3) 어떤 파라미터에 실제로 값이 들어갔는지 출력한다.
% -------------------------------------------------------------------------

mdl = "uav_cf_viewer";

if ~bdIsLoaded(mdl) && isfile(mdl + ".slx")
    load_system(mdl);
end
if ~bdIsLoaded(mdl)
    error("uav_cf_viewer.slx가 없습니다.");
end

visSys   = mdl + "/Visualization_3D";
uavBlk   = visSys + "/Actor_UAV";
obsBlk   = visSys + "/Actor_Obstacle";
sceneBlk = visSys + "/Scene_Config";

assert_block(uavBlk);
assert_block(obsBlk);
assert_block(sceneBlk);

% -------------------------------------------------------------------------
% 1) 드론 / 장애물 위치를 화면 중앙 근처로 정리
% -------------------------------------------------------------------------
assignin("base","UAV_X0", 0);
assignin("base","UAV_Y0", 0);
assignin("base","UAV_Z0", 2.5);

assignin("base","OBS_X0", 12);
assignin("base","OBS_Y0", 0);
assignin("base","OBS_Z0", 1.0);

% 뷰어 점검용으로는 날씨를 일단 맑게 둠
assignin("base","FOG_DENSITY_PERCENT", 0);
assignin("base","ILLUMINATION_LUX", 8000);
assignin("base","CAMERA_NOISE_LEVEL", 0);

% -------------------------------------------------------------------------
% 2) Actor 이름과 형상 정리
% -------------------------------------------------------------------------
set_first_matching_param(uavBlk, ["ActorName","Actor Name","Name"], "UAV");
set_first_matching_param(uavBlk, ["ParentName","Parent Name"], "Scene Origin");
set_first_matching_param(uavBlk, ["Operation"], "Create at setup");
set_first_matching_param(uavBlk, ["InputPorts","Input ports"], sprintf("Translation\nRotation"));
set_first_matching_param(uavBlk, ["InitializationScript","Initialization script"], ...
    [ ...
    "createShape(actor,""box"",[1.8 1.2 0.35]);" newline ...
    "actor.Color = [0.10 0.45 0.95];" newline ...
    ]);

set_first_matching_param(obsBlk, ["ActorName","Actor Name","Name"], "Obstacle1");
set_first_matching_param(obsBlk, ["ParentName","Parent Name"], "Scene Origin");
set_first_matching_param(obsBlk, ["Operation"], "Create at setup");
set_first_matching_param(obsBlk, ["InputPorts","Input ports"], "Translation");
set_first_matching_param(obsBlk, ["InitializationScript","Initialization script"], ...
    [ ...
    "createShape(actor,""cylinder"",[0.8 4.0]);" newline ...
    "actor.Color = [0.95 0.20 0.20];" newline ...
    ]);

% -------------------------------------------------------------------------
% 3) Scene을 Custom 시점으로 강제
% -------------------------------------------------------------------------
set_by_prompt_words(sceneBlk, ["scene","view"], "Custom");

% 장면은 바닥이 잘 보이는 쪽 우선 시도
set_first_matching_param(sceneBlk, ["SceneName","Scene name"], "Large Parking Lot");
set_first_matching_param(sceneBlk, ["SceneName","Scene name"], "Empty Grass");
set_first_matching_param(sceneBlk, ["SceneName","Scene name"], "Urban Road");

% 카메라를 드론 뒤/위에 둔다.
% x=-12, y=0, z=6 에서 +x 방향을 대략 바라보게 설정
okT = set_by_prompt_words(sceneBlk, ["initial","viewer","translation"], "[-12 0 6]");
okR = set_by_prompt_words(sceneBlk, ["initial","viewer","rotation"], "[0 -15 0]");

% 일부 버전은 initial 대신 translation/rotation만 있을 수 있음
if ~okT
    okT = set_by_prompt_words(sceneBlk, ["translation"], "[-12 0 6]");
end
if ~okR
    okR = set_by_prompt_words(sceneBlk, ["rotation"], "[0 15 0]");
end

save_system(mdl);

try
    set_param(mdl, "SimulationCommand", "Update");
catch ME
    warning("모델 업데이트 실패: %s", ME.message);
end

disp("------------------------------------------------------------");
disp("Custom 3rd-person view patch applied");
disp("- Weather cleared for viewer debugging");
disp("- Camera target idea: behind and above UAV");
disp("- If you still see sky, flip pitch sign: [0 -15 0]");
disp("------------------------------------------------------------");

disp_scene_params(sceneBlk);
end

% -------------------------------------------------------------------------
% helpers
% -------------------------------------------------------------------------

function assert_block(pathStr)
try
    get_param(pathStr, "Handle");
catch
    error("블록을 찾지 못했습니다: %s", pathStr);
end
end

function ok = set_by_prompt_words(blockPath, words, value)
ok = false;
try
    dp = get_param(blockPath, "DialogParameters");
    if isempty(dp), return; end

    f = fieldnames(dp);
    for i = 1:numel(f)
        promptText = "";
        try
            promptText = string(dp.(f{i}).Prompt);
        catch
            promptText = string(f{i});
        end

        s = lower(promptText + " " + string(f{i}));
        hit = true;
        for k = 1:numel(words)
            if ~contains(s, lower(string(words(k))))
                hit = false;
                break;
            end
        end

        if hit
            try
                set_param(blockPath, f{i}, value);
                fprintf("[SET] %s | %s = %s\n", f{i}, promptText, value);
                ok = true;
                return;
            catch
            end
        end
    end
catch
end
end

function ok = set_first_matching_param(blockPath, candidateNames, value)
ok = false;
try
    dp = get_param(blockPath, "DialogParameters");
    if isempty(dp), return; end

    f = fieldnames(dp);
    fLow = lower(string(f));
    cand = lower(string(candidateNames));

    for i = 1:numel(cand)
        idx = find(fLow == cand(i), 1, "first");
        if ~isempty(idx)
            try
                set_param(blockPath, f{idx}, value);
                ok = true;
                return;
            catch
            end
        end
    end

    for i = 1:numel(cand)
        idx = find(contains(fLow, cand(i)), 1, "first");
        if ~isempty(idx)
            try
                set_param(blockPath, f{idx}, value);
                ok = true;
                return;
            catch
            end
        end
    end
catch
end
end

function disp_scene_params(sceneBlk)
try
    dp = get_param(sceneBlk, "DialogParameters");
    f = fieldnames(dp);
    disp("---- Scene_Config parameters related to view ----");
    for i = 1:numel(f)
        promptText = "";
        try
            promptText = string(dp.(f{i}).Prompt);
        catch
            promptText = string(f{i});
        end
        s = lower(promptText + " " + string(f{i}));
        if contains(s,"view") || contains(s,"viewer") || contains(s,"translation") || contains(s,"rotation")
            try
                val = string(get_param(sceneBlk, f{i}));
            catch
                val = "(unreadable)";
            end
            fprintf("%-30s | %-35s | %s\n", f{i}, promptText, val);
        end
    end
    disp("------------------------------------------------");
catch
end
end