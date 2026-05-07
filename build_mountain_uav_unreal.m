function build_mountain_uav_unreal(doRun)
% build_mountain_uav_unreal — Phase B 단순화 최종.
%
%   설계 결정: R2025b uavsim3dlib 의 카메라 mount offset/dual-camera 기능이
%   믿을 수 없게 작동하므로, 검증된 메커니즘 (드론 InitialRotation + 카메라
%   부착) 만 사용한다.
%
%     * ONE 드론  (MainDrone, 빨간 quadrotor) — pitch +1.4 rad (~80° down)
%       → 드론이 거의 수직 아래를 바라봄. nose-down 자세.
%     * ONE 카메라  (Cam_Nadir) — 드론에 부착, 오프셋 없음.
%       → 드론이 보는 nadir view 그대로 캡처.
%     * SceneCfg.FreeCamera — 드론을 따라가는 chase 시점으로 설정
%       → MATLAB Unreal native 창이 chase view 역할 (별도 캡처 없음).
%
%   Streamlit 은 nadir view 1장만 표시. 3인칭은 MATLAB Unreal 창 직접 보면 됨.

if nargin < 1 || isempty(doRun), doRun = false; end

mdl = "mountain_uav_unreal";
setup_unreal_workspace();

if bdIsLoaded(mdl), close_system(mdl, 0); end
if isfile(mdl + ".slx"), delete(mdl + ".slx"); end

new_system(mdl); open_system(mdl);
set_param(mdl, ...
    "SolverType",      "Variable-step", ...
    "Solver",          "VariableStepAuto", ...
    "StopTime",        "8", ...
    "SaveOutput",      "on", ...
    "SignalLogging",   "on");

% =========================================================================
% 1) Scene
% =========================================================================
add_block("uavsim3dlib/Simulation 3D Scene Configuration", char(mdl + "/SceneCfg"));
try_set(char(mdl + "/SceneCfg"), "Position",      [40 40 220 110]);
try_set(char(mdl + "/SceneCfg"), "SceneDesc",     "Empty Grass");
try_set(char(mdl + "/SceneCfg"), "Ts",            "0.05");
try_set(char(mdl + "/SceneCfg"), "EnableWindow",  "on");
try_set(char(mdl + "/SceneCfg"), "EnableWeather", "on");
try_set(char(mdl + "/SceneCfg"), "fog",           "0");
try_set(char(mdl + "/SceneCfg"), "rain",          "0");
try_set(char(mdl + "/SceneCfg"), "SunAltitude",   "60");
try_set(char(mdl + "/SceneCfg"), "SunAzimuth",    "180");

% Free camera (Unreal native window) — chase view of drone area
try_set(char(mdl + "/SceneCfg"), "FreeCameraInitialTranslation", "[-15 0 12]");
try_set(char(mdl + "/SceneCfg"), "FreeCameraInitialRotation",    "[0 0.30 0]");
fprintf("[Build] Scene = Empty Grass\n");
fprintf("[Build] Unreal native window: chase view from [-15 0 12], pitch ~17°\n");

% =========================================================================
% 2) MainDrone — visible Red quadrotor, tilted nose-down 80°
% =========================================================================
DRONE_NAME = "MainDrone";
% Compromise: moderate altitude + 35° forward-down tilt.
% At z=8 with pitch 0.61 rad (35°), camera centerline hits ground at
% horizontal forward 8/tan(35°) = 11.4m. Pedestrians placed in that area.
% Wider view than near-nadir, easier to see people in the frame.
DRONE_POS  = "[0 0 8]";
DRONE_ROT  = "[0 0.61 0]";   % 35° nose down

blk = char(mdl + "/" + DRONE_NAME);
add_block("uavsim3dlib/Simulation 3D UAV Vehicle", blk);
try_set(blk, "Position",            [280 40 440 130]);
try_set(blk, "Mesh",                "Quadrotor");
try_set(blk, "Color",               "Red");
try_set(blk, "ActorName",           char(DRONE_NAME));
try_set(blk, "UseGlobalFrame",      "off");
try_set(blk, "InitialTranslation",  DRONE_POS);
try_set(blk, "InitialRotation",     DRONE_ROT);
try_set(blk, "SampleTime",          "0.05");

% UAV needs 2 input ports (translation, rotation)
add_block("simulink/Sources/Constant", char(mdl + "/UAV_Pos"));
try_set(char(mdl + "/UAV_Pos"), "Value",    DRONE_POS);
try_set(char(mdl + "/UAV_Pos"), "Position", [40 160 130 200]);
add_block("simulink/Sources/Constant", char(mdl + "/UAV_Rot"));
try_set(char(mdl + "/UAV_Rot"), "Value",    DRONE_ROT);
try_set(char(mdl + "/UAV_Rot"), "Position", [40 220 130 260]);

safe_add_line(mdl, "UAV_Pos/1", "MainDrone/1");
safe_add_line(mdl, "UAV_Rot/1", "MainDrone/2");
fprintf("[Build] MainDrone (Red Quadrotor) at %s, pitch %s rad (~80° down)\n", DRONE_POS, DRONE_ROT);

% =========================================================================
% 3) Pedestrians — placed in camera focal area (~11m forward of drone)
% =========================================================================
% Drone at [0 0 8] looking 35° down → centerline hits ground at x≈11.4m.
% Place pedestrians in 8-14m forward range so they fill the frame.
ped_specs = {
    "Ped1", "Male 1",   [10, -1, 0],   0;
    "Ped2", "Female 2", [11,  1, 0],  90;
    "Ped3", "Male 2",   [13, -2, 0], 180;
};
for k = 1:size(ped_specs, 1)
    name = ped_specs{k,1}; ptype = ped_specs{k,2};
    pos  = ped_specs{k,3}; yaw   = ped_specs{k,4};
    pblk = char(mdl + "/" + name);
    add_block("drivingsim3d/Simulation 3D Pedestrian", pblk);
    try_set(pblk, "PedestrianType", ptype);
    try_set(pblk, "InitialPos",     sprintf("[%g %g %g]", pos(1), pos(2), pos(3)));
    try_set(pblk, "InitialYaw",     sprintf("%g", yaw));
    try_set(pblk, "Position",       [280 (170 + (k-1)*70) 440 (220 + (k-1)*70)]);
    fprintf("[Build] %s (%s) at [%g %g %g]\n", name, ptype, pos(1), pos(2), pos(3));
end

% Force update so vehTag enum picks up MainDrone
try, set_param(mdl, 'SimulationCommand', 'update'); catch, end

% =========================================================================
% 4) Single nadir camera attached to MainDrone
% =========================================================================
cblk = char(mdl + "/Cam_Nadir");
add_block("uavsim3dlib/Simulation 3D Camera", cblk);
try_set(cblk, "ImageSize",    "[640 360]");
try_set(cblk, "vehTagList",   sprintf("{'Scene Origin', '%s'}", DRONE_NAME));
ok = try_set(cblk, "vehTag",  char(DRONE_NAME));
try_set(cblk, "mountLoc",     "Origin");
try_set(cblk, "offsetFlag",   "on");
try_set(cblk, "tmountOffset", "[0 0 0]");
try_set(cblk, "rmountOffset", "[0 0 0]");
try_set(cblk, "extTmount",    "off");
try_set(cblk, "extRmount",    "off");
try_set(cblk, "SampleTime",   "0.1");
try_set(cblk, "Position",     [560 60 720 130]);

actual = "(?)"; try, actual = get_param(cblk, "vehTag"); catch, end
fprintf("[Build] Cam_Nadir attached to '%s' (status: %s)\n", actual, ...
    string(actual) == string(DRONE_NAME));

% =========================================================================
% 5) Logger — single nadir output
% =========================================================================
add_block("simulink/Sinks/To Workspace", char(mdl + "/TW_Cam1p"));
try_set(char(mdl + "/TW_Cam1p"), "VariableName", "cam1p_log");
try_set(char(mdl + "/TW_Cam1p"), "SaveFormat",   "Structure With Time");
try_set(char(mdl + "/TW_Cam1p"), "SampleTime",   "0.1");
try_set(char(mdl + "/TW_Cam1p"), "Position",     [800 60 940 110]);
safe_add_line(mdl, "Cam_Nadir/1", "TW_Cam1p/1");

% =========================================================================
% 6) Save
% =========================================================================
try, Simulink.BlockDiagram.arrangeSystem(mdl); catch, end
save_system(mdl);
fprintf("[Build] %s.slx saved.\n", mdl);

if doRun
    fprintf("[Build] Running 8 s simulation…\n");
    try
        out = sim(mdl);
        try
            val = out.get("cam1p_log");
            if ~isempty(val), assignin('base', 'cam1p_log', val); end
        catch
        end
        fprintf("[Build] Done.\n");
    catch ME
        fprintf("\n[Build] sim() FAILED: %s\n", getReport(ME, 'extended'));
    end
end
end


% =========================================================================
function setup_unreal_workspace()
assignin("base", "IMG_SIZE", [640, 360]);
fprintf("[Build] IMG_SIZE = 640x360\n");
end


function ok = try_set(blk, pname, pval)
ok = false;
try
    if isnumeric(pval), pval = mat2str(pval); end
    set_param(blk, pname, pval);
    ok = true;
catch
end
end


function safe_add_line(sys, src, dst)
try
    add_line(sys, src, dst, "autorouting", "smart");
catch
end
end
