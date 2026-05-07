function test_vehtag_inject()
% test_vehtag_inject — vehTagList를 수동으로 갱신해서 카메라가 UAV에
%   진짜 attach 되는지 검증한다.
%   verify_camera_params에서 발견한 vehTagList 메커니즘 활용.

mdl = "vehtag_test";
if bdIsLoaded(mdl), close_system(mdl, 0); end
if isfile(mdl + ".slx"), delete(mdl + ".slx"); end
new_system(mdl); open_system(mdl);
set_param(mdl, "SolverType", "Variable-step", "Solver", "VariableStepAuto", ...
    "StopTime", "3");

% Scene
add_block("uavsim3dlib/Simulation 3D Scene Configuration", char(mdl + "/SceneCfg"));
set_param(char(mdl + "/SceneCfg"), "SceneDesc", "Empty Grass");
set_param(char(mdl + "/SceneCfg"), "EnableWindow", "on");
set_param(char(mdl + "/SceneCfg"), "Position", [40 40 220 110]);

% UAV with explicit ActorName
add_block("uavsim3dlib/Simulation 3D UAV Vehicle", char(mdl + "/UAV"));
set_param(char(mdl + "/UAV"), "ActorName",          "MyDrone");
set_param(char(mdl + "/UAV"), "Mesh",               "Quadrotor");
set_param(char(mdl + "/UAV"), "Color",              "Red");
set_param(char(mdl + "/UAV"), "InitialTranslation", "[0 0 30]");
set_param(char(mdl + "/UAV"), "InitialRotation",    "[0 0 0]");
set_param(char(mdl + "/UAV"), "Position", [280 40 440 130]);

% UAV needs 2 input signals
add_block("simulink/Sources/Constant", char(mdl + "/UAV_Pos"));
set_param(char(mdl + "/UAV_Pos"), "Value", "[0 0 30]");
set_param(char(mdl + "/UAV_Pos"), "Position", [40 160 130 200]);
add_block("simulink/Sources/Constant", char(mdl + "/UAV_Rot"));
set_param(char(mdl + "/UAV_Rot"), "Value", "[0 0 0]");
set_param(char(mdl + "/UAV_Rot"), "Position", [40 220 130 260]);
add_line(mdl, "UAV_Pos/1", "UAV/1", "autorouting", "smart");
add_line(mdl, "UAV_Rot/1", "UAV/2", "autorouting", "smart");

% Force model to register actor
try, set_param(mdl, 'SimulationCommand', 'update'); catch, end

% --- Camera with vehTagList INJECTION
add_block("uavsim3dlib/Simulation 3D Camera", char(mdl + "/CamA"));
blkA = char(mdl + "/CamA");

% Step 1: get current vehTagList
fprintf("\n=== 카메라 추가 직후 ===\n");
fprintf("CamA.vehTagList = '%s'\n", get_param(blkA, "vehTagList"));
fprintf("CamA.vehTag     = '%s'\n", get_param(blkA, "vehTag"));

% Step 2: try MULTIPLE strategies to inject MyDrone into the list
fprintf("\n--- 전략 1: vehTagList 직접 set (cell-as-string) ---\n");
try_and_log(blkA, "vehTagList", "{'Scene Origin', 'MyDrone'}");
try_and_log(blkA, "vehTag", "MyDrone");

fprintf("\n--- 전략 2: vehTagList = single string 'MyDrone' ---\n");
try_and_log(blkA, "vehTagList", "MyDrone");
try_and_log(blkA, "vehTag", "MyDrone");

fprintf("\n--- 전략 3: vehTagList = quoted single ---\n");
try_and_log(blkA, "vehTagList", "{'MyDrone'}");
try_and_log(blkA, "vehTag", "MyDrone");

% Set basic camera params
set_param(blkA, "ImageSize",   "[640 360]");
set_param(blkA, "mountLoc",    "Origin");
set_param(blkA, "offsetFlag",  "on");
set_param(blkA, "tmountOffset", "[-12 0 6]");
set_param(blkA, "rmountOffset", "[0 -0.35 0]");
set_param(blkA, "Position", [560 60 720 130]);

% Logger
add_block("simulink/Sinks/To Workspace", char(mdl + "/TW_A"));
set_param(char(mdl + "/TW_A"), "VariableName", "camA_log");
set_param(char(mdl + "/TW_A"), "SaveFormat",   "Structure With Time");
set_param(char(mdl + "/TW_A"), "Position", [800 60 940 110]);
add_line(mdl, "CamA/1", "TW_A/1", "autorouting", "smart");

fprintf("\n=== 최종 카메라 상태 ===\n");
fprintf("CamA.vehTagList = '%s'\n", get_param(blkA, "vehTagList"));
fprintf("CamA.vehTag     = '%s'\n", get_param(blkA, "vehTag"));
fprintf("CamA.tmountOffset = '%s'\n", get_param(blkA, "tmountOffset"));
fprintf("CamA.rmountOffset = '%s'\n", get_param(blkA, "rmountOffset"));

save_system(mdl);
fprintf("\n[Test] Running sim…\n");
out = sim(mdl);

% Save last frame
try
    log = out.get("camA_log");
    if ~isempty(log) && isstruct(log) && isfield(log, "signals")
        v = log.signals.values;
        if numel(size(v)) == 4
            last = v(:,:,:,end);
        else
            last = v;
        end
        if ~isfolder("assets"), mkdir("assets"); end
        png = "assets/test_vehtag_camA.png";
        if isa(last, "uint8")
            imwrite(last, png);
        else
            imwrite(uint8(last*255), png);
        end
        fprintf("[Test] CamA → %s  (size=%s)\n", png, mat2str(size(v)));
    end
catch ME
    fprintf("[Test] extract FAILED: %s\n", ME.message);
end

close_system(mdl, 0);
fprintf("\n[Test] 끝. assets/test_vehtag_camA.png 확인.\n");
fprintf("[Test] 위 출력에서 CamA.vehTag가 'MyDrone'으로 바뀌었으면 성공.\n");
fprintf("       카메라가 드론을 따라가서 chase view가 보여야 함.\n");
end


function try_and_log(blk, pname, pval)
fprintf("  set %s = '%s' → ", pname, pval);
try
    set_param(blk, pname, pval);
    actual = get_param(blk, pname);
    if ~ischar(actual) && ~isstring(actual), actual = mat2str(actual); end
    fprintf("OK (got '%s')\n", actual);
catch ME
    fprintf("FAIL (%s)\n", ME.message);
end
end
