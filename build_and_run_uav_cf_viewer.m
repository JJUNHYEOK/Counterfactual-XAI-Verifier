function simOut = build_and_run_uav_cf_viewer(doRun)
% build_and_run_uav_cf_viewer
% v6 patch: break algebraic loop with top-level Unit Delay on control commands
% v5 patch: match global fixed-step to Simulation 3D blocks (0.02 s)
% v4 patch: fix 3D actor initialization quote escaping and suppress variant warnings in library search
% v3 patch: coerce MATLAB Function script text to scalar char
% v2 patch: Compare To Constant programmatic parameter uses relop
% ------------------------------------------------------------
% 목적:
%   - uav_cf_viewer.slx 모델을 자동 생성
%   - 6개 서브시스템(Scenario_Params, World_Dynamics,
%     Perception_Stub, Decision_Logic, Requirement_Monitor,
%     Visualization_3D)을 전부 자동 배치
%   - 가능한 경우 Simulation 3D Scene Configuration /
%     Simulation 3D Actor 블록까지 자동 생성
%   - 기본 변수값을 넣고 바로 시뮬레이션 실행
%
% 사용:
%   simOut = build_and_run_uav_cf_viewer;       % 생성 + 실행
%   simOut = build_and_run_uav_cf_viewer(true); % 생성 + 실행
%   build_and_run_uav_cf_viewer(false);         % 생성만
%
% 주의:
%   1) Simulink 3D Animation이 설치되어 있어야 3D 블록이 붙습니다.
%   2) 릴리즈마다 3D 블록 마스크 파라미터 이름이 약간 다를 수 있어,
%      스크립트가 대화형으로 파라미터를 찾아 설정하도록 구성했습니다.
%   3) 3D 블록이 없으면 모델은 3D 없이도 생성되며, 로그와 로직은 동작합니다.
% ------------------------------------------------------------

if nargin < 1
    doRun = true;
end

mdl = "uav_cf_viewer";

%% ------------------------------------------------------------
% 0. 기본 변수값
%% ------------------------------------------------------------
assignin("base","FOG_DENSITY_PERCENT", 65);
assignin("base","ILLUMINATION_LUX", 1800);
assignin("base","CAMERA_NOISE_LEVEL", 0.25);

assignin("base","UAV_X0", 0);
assignin("base","UAV_Y0", 0);
assignin("base","UAV_Z0", 1);

assignin("base","OBS_X0", 10);
assignin("base","OBS_Y0", 1.5);
assignin("base","OBS_Z0", 1);

assignin("base","REQ_MIN_CLEARANCE", 2.0);
assignin("base","REQ_MIN_CONFIDENCE", 0.55);

%% ------------------------------------------------------------
% 1. 모델 생성
%% ------------------------------------------------------------
if bdIsLoaded(mdl)
    close_system(mdl, 0);
end

if isfile(mdl + ".slx")
    delete(mdl + ".slx");
end

new_system(mdl);
open_system(mdl);

set_param(mdl, ...
    "SolverType", "Fixed-step", ...
    "Solver", "FixedStepDiscrete", ...
    "FixedStep", "0.02", ...
    "StopTime", "20", ...
    "SaveOutput", "on", ...
    "SignalLogging", "on");

%% ------------------------------------------------------------
% 2. Top-level Subsystems 생성
%% ------------------------------------------------------------
add_subsystem(mdl, "Scenario_Params",     [40   80  220 420]);
add_subsystem(mdl, "World_Dynamics",      [310  80  570 300]);
add_subsystem(mdl, "Perception_Stub",     [660  70  900 210]);
add_subsystem(mdl, "Decision_Logic",      [660 240 900 380]);
add_subsystem(mdl, "Requirement_Monitor", [1000 70 1260 260]);
add_subsystem(mdl, "Visualization_3D",    [1000 300 1260 500]);

%% ------------------------------------------------------------
% 2-1. Top-level command delays to break algebraic loop
%% ------------------------------------------------------------
add_block("simulink/Discrete/Unit Delay", mdl + "/UD_v_cmd_top");
set_param(mdl + "/UD_v_cmd_top", ...
    "InitialCondition", "1.0", ...
    "SampleTime", "-1", ...
    "Position", [910 285 950 315]);

add_block("simulink/Discrete/Unit Delay", mdl + "/UD_yaw_rate_cmd_top");
set_param(mdl + "/UD_yaw_rate_cmd_top", ...
    "InitialCondition", "0.0", ...
    "SampleTime", "-1", ...
    "Position", [910 335 950 365]);


%% ------------------------------------------------------------
% 3. 각 서브시스템 자동 구성
%% ------------------------------------------------------------
build_scenario_params(mdl + "/Scenario_Params");
build_world_dynamics(mdl + "/World_Dynamics");
build_perception_stub(mdl + "/Perception_Stub");
build_decision_logic(mdl + "/Decision_Logic");
build_requirement_monitor(mdl + "/Requirement_Monitor");
has3D = build_visualization_3d(mdl + "/Visualization_3D");

%% ------------------------------------------------------------
% 4. Top-level 로그 블록 추가
%% ------------------------------------------------------------
add_toworkspace(mdl, "TW_collision_flag",         "collision_flag_log",         [1320  70 1410  95]);
add_toworkspace(mdl, "TW_running_min_dist",       "running_min_dist_log",       [1320 110 1410 135]);
add_toworkspace(mdl, "TW_avg_confidence",         "avg_confidence_log",         [1320 150 1410 175]);
add_toworkspace(mdl, "TW_miss_count",             "miss_count_log",             [1320 190 1410 215]);
add_toworkspace(mdl, "TW_requirement_satisfied",  "requirement_satisfied_log",  [1320 230 1410 255]);
add_toworkspace(mdl, "TW_risk_score",             "risk_score_log",             [1320 270 1410 295]);
add_toworkspace(mdl, "TW_uav_xyz",                "uav_xyz_log",                [1320 320 1410 345]);
add_toworkspace(mdl, "TW_uav_rpy",                "uav_rpy_log",                [1320 360 1410 385]);
add_toworkspace(mdl, "TW_obs_xyz",                "obs_xyz_log",                [1320 400 1410 425]);

%% ------------------------------------------------------------
% 5. Top-level 배선
%% ------------------------------------------------------------

% Scenario_Params -> World_Dynamics
safe_add_line(mdl, "Scenario_Params/4",  "World_Dynamics/1"); % uav_x0
safe_add_line(mdl, "Scenario_Params/5",  "World_Dynamics/2"); % uav_y0
safe_add_line(mdl, "Scenario_Params/6",  "World_Dynamics/3"); % uav_z0
safe_add_line(mdl, "Scenario_Params/7",  "World_Dynamics/4"); % obs_x0
safe_add_line(mdl, "Scenario_Params/8",  "World_Dynamics/5"); % obs_y0
safe_add_line(mdl, "Scenario_Params/9",  "World_Dynamics/6"); % obs_z0

% Scenario_Params -> Perception_Stub
safe_add_line(mdl, "Scenario_Params/1",  "Perception_Stub/2"); % fog_density
safe_add_line(mdl, "Scenario_Params/2",  "Perception_Stub/3"); % illumination
safe_add_line(mdl, "Scenario_Params/3",  "Perception_Stub/4"); % camera_noise

% Scenario_Params -> Requirement_Monitor
safe_add_line(mdl, "Scenario_Params/10", "Requirement_Monitor/5"); % req_min_clearance
safe_add_line(mdl, "Scenario_Params/11", "Requirement_Monitor/6"); % req_min_confidence

% World_Dynamics -> Perception_Stub / Decision_Logic / Visualization / Requirement
safe_add_line(mdl, "World_Dynamics/4", "Perception_Stub/1");     % rel_dist
safe_add_line(mdl, "World_Dynamics/4", "Decision_Logic/1");      % rel_dist
safe_add_line(mdl, "World_Dynamics/5", "Decision_Logic/2");      % rel_bearing
safe_add_line(mdl, "World_Dynamics/4", "Requirement_Monitor/1"); % rel_dist

safe_add_line(mdl, "World_Dynamics/1", "Visualization_3D/1");    % uav_xyz
safe_add_line(mdl, "World_Dynamics/2", "Visualization_3D/2");    % uav_rpy
safe_add_line(mdl, "World_Dynamics/3", "Visualization_3D/3");    % obs_xyz

% Perception_Stub -> Decision_Logic / Requirement_Monitor
safe_add_line(mdl, "Perception_Stub/1", "Decision_Logic/3");      % confidence
safe_add_line(mdl, "Perception_Stub/2", "Decision_Logic/4");      % miss_flag
safe_add_line(mdl, "Perception_Stub/1", "Requirement_Monitor/2"); % confidence
safe_add_line(mdl, "Perception_Stub/2", "Requirement_Monitor/3"); % miss_flag
safe_add_line(mdl, "Perception_Stub/3", "Requirement_Monitor/4"); % risk_partial

% Decision_Logic -> top-level delays -> World_Dynamics
safe_add_line(mdl, "Decision_Logic/1", "UD_v_cmd_top/1");         % v_cmd delayed
safe_add_line(mdl, "Decision_Logic/2", "UD_yaw_rate_cmd_top/1");  % yaw_rate delayed
safe_add_line(mdl, "UD_v_cmd_top/1", "World_Dynamics/7");         % v_cmd
safe_add_line(mdl, "UD_yaw_rate_cmd_top/1", "World_Dynamics/8");  % yaw_rate_cmd

% Requirement_Monitor -> To Workspace
safe_add_line(mdl, "Requirement_Monitor/1", "TW_collision_flag/1");
safe_add_line(mdl, "Requirement_Monitor/2", "TW_running_min_dist/1");
safe_add_line(mdl, "Requirement_Monitor/3", "TW_avg_confidence/1");
safe_add_line(mdl, "Requirement_Monitor/4", "TW_miss_count/1");
safe_add_line(mdl, "Requirement_Monitor/5", "TW_requirement_satisfied/1");
safe_add_line(mdl, "Requirement_Monitor/6", "TW_risk_score/1");

% World state logs
safe_add_line(mdl, "World_Dynamics/1", "TW_uav_xyz/1");
safe_add_line(mdl, "World_Dynamics/2", "TW_uav_rpy/1");
safe_add_line(mdl, "World_Dynamics/3", "TW_obs_xyz/1");

%% ------------------------------------------------------------
% 6. 보기 좋게 정리
%% ------------------------------------------------------------
try
    Simulink.BlockDiagram.arrangeSystem(mdl);
catch
end

save_system(mdl);
open_system(mdl);

disp("------------------------------------------------------------");
disp("모델 생성 완료: " + mdl + ".slx");
if has3D
    disp("3D 블록도 자동 생성됨");
else
    disp("3D 블록은 찾지 못해 placeholder만 생성됨");
end
disp("기본 변수값은 base workspace에 주입됨");
disp("------------------------------------------------------------");

%% ------------------------------------------------------------
% 7. 실행
%% ------------------------------------------------------------
if doRun
    set_param(mdl, "SimulationCommand", "Update");
    simOut = sim(mdl);
else
    simOut = [];
end

end

%% ========================================================================
% Subsystem Builders
%% ========================================================================

function build_scenario_params(sys)
safe_delete(sys + "/In1");
safe_delete(sys + "/Out1");

params = {
    "C_FOG_DENSITY",         "FOG_DENSITY_PERCENT",  "fog_density";
    "C_ILLUMINATION",        "ILLUMINATION_LUX",     "illumination_lux";
    "C_CAMERA_NOISE",        "CAMERA_NOISE_LEVEL",   "camera_noise";
    "C_UAV_X0",              "UAV_X0",               "uav_x0";
    "C_UAV_Y0",              "UAV_Y0",               "uav_y0";
    "C_UAV_Z0",              "UAV_Z0",               "uav_z0";
    "C_OBS_X0",              "OBS_X0",               "obs_x0";
    "C_OBS_Y0",              "OBS_Y0",               "obs_y0";
    "C_OBS_Z0",              "OBS_Z0",               "obs_z0";
    "C_REQ_MIN_CLEARANCE",   "REQ_MIN_CLEARANCE",    "req_min_clearance";
    "C_REQ_MIN_CONFIDENCE",  "REQ_MIN_CONFIDENCE",   "req_min_confidence";
};

for i = 1:size(params,1)
    y = 30 + (i-1)*45;

    add_block("simulink/Commonly Used Blocks/Constant", sys + "/" + params{i,1});
    set_param(sys + "/" + params{i,1}, ...
        "Value", params{i,2}, ...
        "Position", [30 y 90 y+25]);

    add_block("simulink/Ports & Subsystems/Out1", sys + "/" + params{i,3});
    set_param(sys + "/" + params{i,3}, ...
        "Position", [180 y 210 y+20]);

    safe_add_line(sys, params{i,1} + "/1", params{i,3} + "/1");
end
end

function build_world_dynamics(sys)
safe_delete(sys + "/In1");
safe_delete(sys + "/Out1");

inNames = [
    "uav_x0"
    "uav_y0"
    "uav_z0"
    "obs_x0"
    "obs_y0"
    "obs_z0"
    "v_cmd"
    "yaw_rate_cmd"
];

outNames = [
    "uav_xyz"
    "uav_rpy"
    "obs_xyz"
    "rel_dist"
    "rel_bearing"
];

for i = 1:numel(inNames)
    y = 40 + (i-1)*40;
    add_block("simulink/Ports & Subsystems/In1", sys + "/" + inNames(i));
    set_param(sys + "/" + inNames(i), "Position", [20 y 50 y+18]);
end

for i = 1:numel(outNames)
    y = 60 + (i-1)*55;
    add_block("simulink/Ports & Subsystems/Out1", sys + "/" + outNames(i));
    set_param(sys + "/" + outNames(i), "Position", [980 y 1010 y+18]);
end

add_block("simulink/Discrete/Unit Delay", sys + "/UD_uav_x");
set_param(sys + "/UD_uav_x", "InitialCondition", "0", "Position", [170 90 210 120]);

add_block("simulink/Discrete/Unit Delay", sys + "/UD_uav_y");
set_param(sys + "/UD_uav_y", "InitialCondition", "0", "Position", [170 150 210 180]);

add_block("simulink/Discrete/Unit Delay", sys + "/UD_uav_yaw");
set_param(sys + "/UD_uav_yaw", "InitialCondition", "0", "Position", [170 210 210 240]);

add_block("simulink/Commonly Used Blocks/Constant", sys + "/C_Ts");
set_param(sys + "/C_Ts", "Value", "0.02", "Position", [340 60 380 85]);

add_block("simulink/Sources/Clock", sys + "/Clock_t");
set_param(sys + "/Clock_t", "Position", [270 300 310 325]);

add_block("simulink/Logic and Bit Operations/Compare To Constant", sys + "/CMP_init");
set_param(sys + "/CMP_init", ...
    "const", "0.001", ...
    "relop", "<", ...
    "Position", [350 295 435 325]);

add_block("simulink/Signal Routing/Switch", sys + "/SW_x_init");
set_param(sys + "/SW_x_init", "Threshold", "0.5", "Criteria", "u2 >= Threshold", ...
    "Position", [300 90 345 120]);

add_block("simulink/Signal Routing/Switch", sys + "/SW_y_init");
set_param(sys + "/SW_y_init", "Threshold", "0.5", "Criteria", "u2 >= Threshold", ...
    "Position", [300 150 345 180]);

add_block("simulink/Signal Routing/Switch", sys + "/SW_yaw_init");
set_param(sys + "/SW_yaw_init", "Threshold", "0.5", "Criteria", "u2 >= Threshold", ...
    "Position", [300 210 345 240]);

add_block("simulink/Commonly Used Blocks/Constant", sys + "/C_YAW0");
set_param(sys + "/C_YAW0", "Value", "0", "Position", [240 220 270 240]);

add_block("simulink/User-Defined Functions/MATLAB Function", sys + "/F_world_update");
set_param(sys + "/F_world_update", "Position", [500 70 760 260]);
set_matlab_function_script(sys + "/F_world_update", ...
    [ ...
    "function [x_next, y_next, yaw_next, rel_dist, rel_bearing] = F_world_update(x, y, yaw, v_cmd, yaw_rate_cmd, Ts, obs_x, obs_y)" newline ...
    "x_next = x + Ts * v_cmd * cos(yaw);" newline ...
    "y_next = y + Ts * v_cmd * sin(yaw);" newline ...
    "yaw_next = yaw + Ts * yaw_rate_cmd;" newline ...
    "dx = obs_x - x_next;" newline ...
    "dy = obs_y - y_next;" newline ...
    "rel_dist = sqrt(dx^2 + dy^2);" newline ...
    "rel_bearing = atan2(dy, dx) - yaw_next;" newline ...
    "end" ]);

add_block("simulink/Signal Routing/Vector Concatenate", sys + "/VC_uav_xyz");
set_param(sys + "/VC_uav_xyz", "NumInputs", "3", "Position", [820 50 860 95]);

add_block("simulink/Signal Routing/Vector Concatenate", sys + "/VC_uav_rpy");
set_param(sys + "/VC_uav_rpy", "NumInputs", "3", "Position", [820 120 860 165]);

add_block("simulink/Signal Routing/Vector Concatenate", sys + "/VC_obs_xyz");
set_param(sys + "/VC_obs_xyz", "NumInputs", "3", "Position", [820 190 860 235]);

add_block("simulink/Commonly Used Blocks/Constant", sys + "/C_ROLL0");
set_param(sys + "/C_ROLL0", "Value", "0", "Position", [760 120 790 140]);

add_block("simulink/Commonly Used Blocks/Constant", sys + "/C_PITCH0");
set_param(sys + "/C_PITCH0", "Value", "0", "Position", [760 150 790 170]);

safe_add_line(sys, "Clock_t/1",       "CMP_init/1");

safe_add_line(sys, "uav_x0/1",        "SW_x_init/1");
safe_add_line(sys, "UD_uav_x/1",      "SW_x_init/3");
safe_add_line(sys, "CMP_init/1",      "SW_x_init/2");

safe_add_line(sys, "uav_y0/1",        "SW_y_init/1");
safe_add_line(sys, "UD_uav_y/1",      "SW_y_init/3");
safe_add_line(sys, "CMP_init/1",      "SW_y_init/2");

safe_add_line(sys, "C_YAW0/1",        "SW_yaw_init/1");
safe_add_line(sys, "UD_uav_yaw/1",    "SW_yaw_init/3");
safe_add_line(sys, "CMP_init/1",      "SW_yaw_init/2");

safe_add_line(sys, "SW_x_init/1",     "F_world_update/1");
safe_add_line(sys, "SW_y_init/1",     "F_world_update/2");
safe_add_line(sys, "SW_yaw_init/1",   "F_world_update/3");
safe_add_line(sys, "v_cmd/1",         "F_world_update/4");
safe_add_line(sys, "yaw_rate_cmd/1",  "F_world_update/5");
safe_add_line(sys, "C_Ts/1",          "F_world_update/6");
safe_add_line(sys, "obs_x0/1",        "F_world_update/7");
safe_add_line(sys, "obs_y0/1",        "F_world_update/8");

safe_add_line(sys, "F_world_update/1","UD_uav_x/1");
safe_add_line(sys, "F_world_update/2","UD_uav_y/1");
safe_add_line(sys, "F_world_update/3","UD_uav_yaw/1");

safe_add_line(sys, "F_world_update/1","VC_uav_xyz/1");
safe_add_line(sys, "F_world_update/2","VC_uav_xyz/2");
safe_add_line(sys, "uav_z0/1",        "VC_uav_xyz/3");

safe_add_line(sys, "C_ROLL0/1",       "VC_uav_rpy/1");
safe_add_line(sys, "C_PITCH0/1",      "VC_uav_rpy/2");
safe_add_line(sys, "F_world_update/3","VC_uav_rpy/3");

safe_add_line(sys, "obs_x0/1",        "VC_obs_xyz/1");
safe_add_line(sys, "obs_y0/1",        "VC_obs_xyz/2");
safe_add_line(sys, "obs_z0/1",        "VC_obs_xyz/3");

safe_add_line(sys, "VC_uav_xyz/1",    "uav_xyz/1");
safe_add_line(sys, "VC_uav_rpy/1",    "uav_rpy/1");
safe_add_line(sys, "VC_obs_xyz/1",    "obs_xyz/1");
safe_add_line(sys, "F_world_update/4","rel_dist/1");
safe_add_line(sys, "F_world_update/5","rel_bearing/1");
end

function build_perception_stub(sys)
safe_delete(sys + "/In1");
safe_delete(sys + "/Out1");

inNames = ["rel_dist","fog_density","illumination_lux","camera_noise"];
outNames = ["confidence","miss_flag","risk_partial"];

for i = 1:numel(inNames)
    y = 40 + (i-1)*40;
    add_block("simulink/Ports & Subsystems/In1", sys + "/" + inNames(i));
    set_param(sys + "/" + inNames(i), "Position", [20 y 50 y+18]);
end

for i = 1:numel(outNames)
    y = 80 + (i-1)*60;
    add_block("simulink/Ports & Subsystems/Out1", sys + "/" + outNames(i));
    set_param(sys + "/" + outNames(i), "Position", [430 y 460 y+18]);
end

add_block("simulink/User-Defined Functions/MATLAB Function", sys + "/F_perception_stub");
set_param(sys + "/F_perception_stub", "Position", [130 50 350 210]);
set_matlab_function_script(sys + "/F_perception_stub", ...
    [ ...
    "function [confidence, miss_flag, risk_partial] = F_perception_stub(rel_dist, fog_pct, illum_lux, noise_level)" newline ...
    "fog = fog_pct / 100.0;" newline ...
    "low_light_penalty = max(0, (3000 - illum_lux) / 3000);" newline ...
    "high_light_penalty = max(0, (illum_lux - 10000) / 10000);" newline ...
    "visibility = 1.0 - 0.65*fog - 0.20*low_light_penalty - 0.12*high_light_penalty - 0.18*noise_level;" newline ...
    "visibility = max(0.0, min(1.0, visibility));" newline ...
    "distance_factor = max(0.0, min(1.0, 1 - rel_dist / 20));" newline ...
    "confidence = 0.15 + 0.55*visibility + 0.30*distance_factor;" newline ...
    "confidence = max(0.0, min(1.0, confidence));" newline ...
    "miss_flag = confidence < 0.35;" newline ...
    "risk_partial = min(1.0, 0.4*(1-confidence) + 0.4*double(miss_flag) + 0.2*max(0,(8-rel_dist)/8));" newline ...
    "end" ]);

safe_add_line(sys, "rel_dist/1",          "F_perception_stub/1");
safe_add_line(sys, "fog_density/1",       "F_perception_stub/2");
safe_add_line(sys, "illumination_lux/1",  "F_perception_stub/3");
safe_add_line(sys, "camera_noise/1",      "F_perception_stub/4");

safe_add_line(sys, "F_perception_stub/1", "confidence/1");
safe_add_line(sys, "F_perception_stub/2", "miss_flag/1");
safe_add_line(sys, "F_perception_stub/3", "risk_partial/1");
end

function build_decision_logic(sys)
safe_delete(sys + "/In1");
safe_delete(sys + "/Out1");

inNames = ["rel_dist","rel_bearing","confidence","miss_flag"];
outNames = ["v_cmd","yaw_rate_cmd","mode"];

for i = 1:numel(inNames)
    y = 40 + (i-1)*40;
    add_block("simulink/Ports & Subsystems/In1", sys + "/" + inNames(i));
    set_param(sys + "/" + inNames(i), "Position", [20 y 50 y+18]);
end

for i = 1:numel(outNames)
    y = 80 + (i-1)*60;
    add_block("simulink/Ports & Subsystems/Out1", sys + "/" + outNames(i));
    set_param(sys + "/" + outNames(i), "Position", [430 y 460 y+18]);
end

add_block("simulink/User-Defined Functions/MATLAB Function", sys + "/F_decision_logic");
set_param(sys + "/F_decision_logic", "Position", [130 50 350 220]);
set_matlab_function_script(sys + "/F_decision_logic", ...
    [ ...
    "function [v_cmd, yaw_rate_cmd, mode] = F_decision_logic(rel_dist, rel_bearing, confidence, miss_flag)" newline ...
    "% mode: 0=cruise, 1=avoid, 2=brake" newline ...
    "if miss_flag && rel_dist < 8" newline ...
    "    v_cmd = 0.5;" newline ...
    "    yaw_rate_cmd = 0.0;" newline ...
    "    mode = 2;" newline ...
    "elseif rel_dist < 10 && confidence < 0.5" newline ...
    "    v_cmd = 1.0;" newline ...
    "    yaw_rate_cmd = sign(rel_bearing) * 0.4;" newline ...
    "    mode = 1;" newline ...
    "elseif rel_dist < 6" newline ...
    "    v_cmd = 0.8;" newline ...
    "    yaw_rate_cmd = sign(rel_bearing) * 0.6;" newline ...
    "    mode = 1;" newline ...
    "else" newline ...
    "    v_cmd = 2.0;" newline ...
    "    yaw_rate_cmd = 0.0;" newline ...
    "    mode = 0;" newline ...
    "end" newline ...
    "end" ]);

safe_add_line(sys, "rel_dist/1",          "F_decision_logic/1");
safe_add_line(sys, "rel_bearing/1",       "F_decision_logic/2");
safe_add_line(sys, "confidence/1",        "F_decision_logic/3");
safe_add_line(sys, "miss_flag/1",         "F_decision_logic/4");

safe_add_line(sys, "F_decision_logic/1",  "v_cmd/1");
safe_add_line(sys, "F_decision_logic/2",  "yaw_rate_cmd/1");
safe_add_line(sys, "F_decision_logic/3",  "mode/1");
end

function build_requirement_monitor(sys)
safe_delete(sys + "/In1");
safe_delete(sys + "/Out1");

inNames = ["rel_dist","confidence","miss_flag","risk_partial","req_min_clearance","req_min_confidence"];
outNames = ["collision_flag","running_min_dist","avg_confidence","miss_count","requirement_satisfied","risk_score"];

for i = 1:numel(inNames)
    y = 35 + (i-1)*35;
    add_block("simulink/Ports & Subsystems/In1", sys + "/" + inNames(i));
    set_param(sys + "/" + inNames(i), "Position", [20 y 50 y+18]);
end

for i = 1:numel(outNames)
    y = 50 + (i-1)*45;
    add_block("simulink/Ports & Subsystems/Out1", sys + "/" + outNames(i));
    set_param(sys + "/" + outNames(i), "Position", [620 y 650 y+18]);
end

add_block("simulink/Discrete/Unit Delay", sys + "/UD_min_dist");
set_param(sys + "/UD_min_dist", "InitialCondition", "999", "Position", [150 40 190 65]);

add_block("simulink/Discrete/Unit Delay", sys + "/UD_miss_count");
set_param(sys + "/UD_miss_count", "InitialCondition", "0", "Position", [150 90 190 115]);

add_block("simulink/Discrete/Unit Delay", sys + "/UD_conf_sum");
set_param(sys + "/UD_conf_sum", "InitialCondition", "0", "Position", [150 140 190 165]);

add_block("simulink/Discrete/Unit Delay", sys + "/UD_sample_count");
set_param(sys + "/UD_sample_count", "InitialCondition", "0", "Position", [150 190 190 215]);

add_block("simulink/User-Defined Functions/MATLAB Function", sys + "/F_req_monitor");
set_param(sys + "/F_req_monitor", "Position", [250 30 520 260]);
set_matlab_function_script(sys + "/F_req_monitor", ...
    [ ...
    "function [collision_flag, running_min_dist, miss_count, avg_confidence, requirement_satisfied, risk_score, conf_sum_next, sample_count_next] = ..." newline ...
    "F_req_monitor(rel_dist, confidence, miss_flag, risk_partial, prev_min_dist, prev_miss_count, prev_conf_sum, prev_sample_count, req_min_clearance, req_min_confidence)" newline ...
    "collision_flag = rel_dist <= 0.5;" newline ...
    "running_min_dist = min(prev_min_dist, rel_dist);" newline ...
    "miss_count = prev_miss_count + double(miss_flag);" newline ...
    "conf_sum_next = prev_conf_sum + confidence;" newline ...
    "sample_count_next = prev_sample_count + 1;" newline ...
    "avg_confidence = conf_sum_next / max(sample_count_next, 1);" newline ...
    "requirement_satisfied = (~collision_flag) && (running_min_dist >= req_min_clearance) && (avg_confidence >= req_min_confidence);" newline ...
    "risk_score = min(1.0, 0.5*risk_partial + 0.5*double(~requirement_satisfied));" newline ...
    "end" ]);

safe_add_line(sys, "rel_dist/1",             "F_req_monitor/1");
safe_add_line(sys, "confidence/1",           "F_req_monitor/2");
safe_add_line(sys, "miss_flag/1",            "F_req_monitor/3");
safe_add_line(sys, "risk_partial/1",         "F_req_monitor/4");
safe_add_line(sys, "UD_min_dist/1",          "F_req_monitor/5");
safe_add_line(sys, "UD_miss_count/1",        "F_req_monitor/6");
safe_add_line(sys, "UD_conf_sum/1",          "F_req_monitor/7");
safe_add_line(sys, "UD_sample_count/1",      "F_req_monitor/8");
safe_add_line(sys, "req_min_clearance/1",    "F_req_monitor/9");
safe_add_line(sys, "req_min_confidence/1",   "F_req_monitor/10");

safe_add_line(sys, "F_req_monitor/2",        "UD_min_dist/1");
safe_add_line(sys, "F_req_monitor/3",        "UD_miss_count/1");
safe_add_line(sys, "F_req_monitor/7",        "UD_conf_sum/1");
safe_add_line(sys, "F_req_monitor/8",        "UD_sample_count/1");

safe_add_line(sys, "F_req_monitor/1",        "collision_flag/1");
safe_add_line(sys, "F_req_monitor/2",        "running_min_dist/1");
safe_add_line(sys, "F_req_monitor/4",        "avg_confidence/1");
safe_add_line(sys, "F_req_monitor/3",        "miss_count/1");
safe_add_line(sys, "F_req_monitor/5",        "requirement_satisfied/1");
safe_add_line(sys, "F_req_monitor/6",        "risk_score/1");
end

function has3D = build_visualization_3d(sys)
safe_delete(sys + "/In1");
safe_delete(sys + "/Out1");

inNames = ["uav_xyz","uav_rpy","obs_xyz"];
for i = 1:numel(inNames)
    y = 50 + (i-1)*60;
    add_block("simulink/Ports & Subsystems/In1", sys + "/" + inNames(i));
    set_param(sys + "/" + inNames(i), "Position", [20 y 50 y+18]);
end

has3D = false;

% 후보 라이브러리들을 불러오기
candidates = ["sim3dlib","sl3dlib","uavlib","robotlib"];
for i = 1:numel(candidates)
    try
        load_system(candidates(i));
    catch
    end
end

scenePath = find_block_by_name_any_library("Simulation 3D Scene Configuration");
actorPath = find_block_by_name_any_library("Simulation 3D Actor");

if strlength(scenePath) == 0 || strlength(actorPath) == 0
    % 3D 블록을 찾지 못하면 placeholder 생성
    add_block("simulink/Ports & Subsystems/Subsystem", sys + "/NO_3D_BLOCKS_FOUND");
    set_param(sys + "/NO_3D_BLOCKS_FOUND", "Position", [180 60 520 220]);
    add_block("simulink/Commonly Used Blocks/Note", sys + "/README_3D");
    set_param(sys + "/README_3D", ...
        "Position", [180 250 540 340], ...
        "Text", sprintf("Simulation 3D 블록을 찾지 못했습니다.\\nSimulink 3D Animation 설치 후 다시 실행하세요."));
    return;
end

has3D = true;

% Scene Config
add_block(scenePath, sys + "/Scene_Config");
set_param(sys + "/Scene_Config", "Position", [560 90 760 180]);

% Actor UAV
add_block(actorPath, sys + "/Actor_UAV");
set_param(sys + "/Actor_UAV", "Position", [220 40 470 140]);

% Actor Obstacle
add_block(actorPath, sys + "/Actor_Obstacle");
set_param(sys + "/Actor_Obstacle", "Position", [220 180 470 280]);

% 파라미터를 최대한 자동 탐색해서 설정
set_first_matching_param(sys + "/Actor_UAV", ["ActorName","Actor Name"], "UAV");
set_first_matching_param(sys + "/Actor_UAV", ["ParentName","Parent Name"], "Scene Origin");
set_first_matching_param(sys + "/Actor_UAV", ["Operation"], "Create at setup");
set_first_matching_param(sys + "/Actor_UAV", ["InputPorts","Input ports"], sprintf("Translation\nRotation"));
set_first_matching_param(sys + "/Actor_UAV", ["InitializationScript","Initialization script"], ...
    sprintf("createShape(actor,""box"",[0.6 0.2 0.08]);\nactor.Color = [0.1 0.4 0.9];"));

set_first_matching_param(sys + "/Actor_Obstacle", ["ActorName","Actor Name"], "Obstacle1");
set_first_matching_param(sys + "/Actor_Obstacle", ["ParentName","Parent Name"], "Scene Origin");
set_first_matching_param(sys + "/Actor_Obstacle", ["Operation"], "Create at setup");
set_first_matching_param(sys + "/Actor_Obstacle", ["InputPorts","Input ports"], "Translation");
set_first_matching_param(sys + "/Actor_Obstacle", ["InitializationScript","Initialization script"], ...
    sprintf("createShape(actor,""cylinder"",[0.25 2.0]);\nactor.Color = [0.8 0.2 0.2];"));

% Scene 설정은 기본값 유지, 가능하면 장면 이름 지정
set_first_matching_param(sys + "/Scene_Config", ["SceneName","Scene name"], "Empty Grass");
set_first_matching_param(sys + "/Scene_Config", ["SceneView","Scene view"], "Scene Origin");

% 가능하면 actor 입력 포트 수 갱신
try
    set_param(sys + "/Actor_UAV", "SimulationCommand", "Update");
catch
end
try
    set_param(sys + "/Actor_Obstacle", "SimulationCommand", "Update");
catch
end

% 배선 시도
% UAV actor: Translation, Rotation
safe_add_line(sys, "uav_xyz/1", "Actor_UAV/1");
safe_add_line(sys, "uav_rpy/1", "Actor_UAV/2");

% Obstacle actor: Translation
safe_add_line(sys, "obs_xyz/1", "Actor_Obstacle/1");

% 보기 좋게 정리
try
    Simulink.BlockDiagram.arrangeSystem(sys);
catch
end
end

%% ========================================================================
% Helpers
%% ========================================================================

function add_subsystem(parent, name, position)
add_block("simulink/Ports & Subsystems/Subsystem", parent + "/" + name);
set_param(parent + "/" + name, "Position", position);
end

function add_toworkspace(parent, blockName, varName, position)
add_block("simulink/Sinks/To Workspace", parent + "/" + blockName);
set_param(parent + "/" + blockName, ...
    "VariableName", varName, ...
    "SaveFormat", "Structure With Time", ...
    "Position", position);
end

function set_matlab_function_script(blockPath, scriptText)
cfg = get_param(blockPath, "MATLABFunctionConfiguration");

% MATLAB Function 블록은 스칼라 char/string만 허용
if isstring(scriptText)
    scriptText = join(scriptText, "");
    scriptText = char(scriptText);
elseif iscell(scriptText)
    scriptText = char(strjoin(string(scriptText), ""));
elseif ~ischar(scriptText)
    scriptText = char(string(scriptText));
end

cfg.FunctionScript = scriptText;
end

function safe_delete(pathStr)
try
    if ~isempty(get_param(pathStr, "Handle"))
        delete_block(pathStr);
    end
catch
end
end

function safe_add_line(sys, src, dst)
try
    add_line(sys, src, dst, "autorouting", "smart");
catch
end
end

function blkPath = find_block_by_name_any_library(blockName)
blkPath = "";
loaded = find_system("type", "block_diagram");
for i = 1:numel(loaded)
    try
        hits = find_system(loaded{i}, ...
            "LookUnderMasks", "all", ...
            "FollowLinks", "on", ...
            "MatchFilter", @Simulink.match.allVariants, ...
            "Type", "block", ...
            "Name", blockName);
        if ~isempty(hits)
            blkPath = string(hits{1});
            return;
        end
    catch
    end
end
end

function ok = set_first_matching_param(blockPath, candidateNames, value)
ok = false;
try
    dp = get_param(blockPath, "DialogParameters");
    if isempty(dp)
        return;
    end
    f = fieldnames(dp);
    fLow = lower(string(f));
    cand = lower(string(candidateNames));

    for i = 1:numel(cand)
        idx = find(fLow == cand(i), 1, "first");
        if ~isempty(idx)
            set_param(blockPath, f{idx}, value);
            ok = true;
            return;
        end
    end

    % 완전 일치가 없으면 contains로 재시도
    for i = 1:numel(cand)
        idx = find(contains(fLow, cand(i)), 1, "first");
        if ~isempty(idx)
            set_param(blockPath, f{idx}, value);
            ok = true;
            return;
        end
    end
catch
end
end
