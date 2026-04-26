function build_mountain_uav_model(doRun)
% build_mountain_uav_model
% -------------------------------------------------------------------------
% Builds mountain_uav_model.slx, a Simulink model that:
%   - Loads UAV scenario from data/scenario_iter_001.json (fog/illum/noise)
%   - Defines a synthetic mountain terrain (DEM) in the base workspace
%   - Places 5 tree-shaped obstacles on the mountainside
%   - Flies a UAV with constant velocity over the terrain
%   - Performs geometric object detection from a forward-facing camera,
%     degraded by fog / illumination / camera noise
%   - Logs UAV pose, obstacle states, GT/detected bboxes, scores, distances
%
% Usage:
%   build_mountain_uav_model;          % build only
%   build_mountain_uav_model(true);    % build + run + visualize
%
% After running, call mountain_visualizer(simOut) to animate results.
% -------------------------------------------------------------------------

if nargin < 1 || isempty(doRun)
    doRun = false;
end

mdl = "mountain_uav_model";

setup_base_workspace();

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
    "FixedStep", "0.1", ...
    "StopTime", "30", ...
    "SaveOutput", "on", ...
    "SignalLogging", "on");

% --- Top level subsystems ---
add_subsystem(mdl, "Scenario_Params",  [40   60  240 460]);
add_subsystem(mdl, "UAV_Dynamics",     [310  60  520 200]);
add_subsystem(mdl, "Object_Detector",  [600  60  900 360]);

% Top-level Clock (drives UAV dynamics)
add_block("simulink/Sources/Clock", mdl + "/Clock");
set_param(mdl + "/Clock", "Position", [310 230 350 260]);

% --- To Workspace logging blocks ---
add_toworkspace(mdl, "TW_uav_xyz",     "uav_xyz_log",      [970  70 1080  95]);
add_toworkspace(mdl, "TW_scores",      "det_scores_log",   [970 110 1080 135]);
add_toworkspace(mdl, "TW_det_bboxes",  "det_bboxes_log",   [970 150 1080 175]);
add_toworkspace(mdl, "TW_gt_bboxes",   "gt_bboxes_log",    [970 190 1080 215]);
add_toworkspace(mdl, "TW_dists",       "rel_dists_log",    [970 230 1080 255]);

% --- Build subsystems ---
build_scenario_params(mdl + "/Scenario_Params");
build_uav_dynamics(mdl + "/UAV_Dynamics");
build_object_detector(mdl + "/Object_Detector");

% --- Top-level wiring ---

% Scenario_Params output port indices (in order added below):
%   1: fog
%   2: illum
%   3: noise
%   4: obs_xyz (5x3)
%   5: obs_rh  (5x2)
%   6: uav_x0  (1x3)
%   7: uav_v   (1x3)
%   8: cam_intrin (1x4)
%   9: img_size (1x2)

% Scenario_Params -> UAV_Dynamics
safe_add_line(mdl, "Scenario_Params/6", "UAV_Dynamics/2");  % x0
safe_add_line(mdl, "Scenario_Params/7", "UAV_Dynamics/3");  % v
safe_add_line(mdl, "Clock/1",           "UAV_Dynamics/1");  % t

% UAV_Dynamics -> Object_Detector
safe_add_line(mdl, "UAV_Dynamics/1", "Object_Detector/1");  % uav_xyz

% Scenario_Params -> Object_Detector
safe_add_line(mdl, "Scenario_Params/4", "Object_Detector/2");  % obs_xyz
safe_add_line(mdl, "Scenario_Params/5", "Object_Detector/3");  % obs_rh
safe_add_line(mdl, "Scenario_Params/1", "Object_Detector/4");  % fog
safe_add_line(mdl, "Scenario_Params/2", "Object_Detector/5");  % illum
safe_add_line(mdl, "Scenario_Params/3", "Object_Detector/6");  % noise
safe_add_line(mdl, "Scenario_Params/8", "Object_Detector/7");  % cam_intrin
safe_add_line(mdl, "Scenario_Params/9", "Object_Detector/8");  % img_size

% UAV_Dynamics -> Logger
safe_add_line(mdl, "UAV_Dynamics/1", "TW_uav_xyz/1");

% Object_Detector -> Loggers
safe_add_line(mdl, "Object_Detector/1", "TW_scores/1");
safe_add_line(mdl, "Object_Detector/2", "TW_det_bboxes/1");
safe_add_line(mdl, "Object_Detector/3", "TW_gt_bboxes/1");
safe_add_line(mdl, "Object_Detector/4", "TW_dists/1");

try
    Simulink.BlockDiagram.arrangeSystem(mdl);
catch
end

save_system(mdl);

fprintf("[BUILD] %s.slx saved.\n", mdl);

if doRun
    set_param(mdl, "SimulationCommand", "Update");
    fprintf("[RUN] Simulating %s ...\n", mdl);
    simOut = sim(mdl);
    assignin("base", "lastSimOut", simOut);
    fprintf("[RUN] Done. Logged to base workspace as 'lastSimOut'.\n");
    try
        mountain_visualizer(simOut);
    catch ME
        warning("Visualization failed: %s", ME.message);
    end
end

end

% =========================================================================
% Base workspace setup: scenario, terrain, obstacles, UAV
% =========================================================================
function setup_base_workspace()

% --- Load scenario from data/ if available ---
fog = 30.0; illum = 4000.0; noise = 0.1;
scenarioPath = fullfile(pwd, "data", "scenario_iter_001.json");
if isfile(scenarioPath)
    try
        sc = jsondecode(fileread(scenarioPath));
        env = sc.environment_parameters;
        if isfield(env, "fog_density_percent"),  fog   = double(env.fog_density_percent);  end
        if isfield(env, "illumination_lux"),     illum = double(env.illumination_lux);     end
        if isfield(env, "camera_noise_level"),   noise = double(env.camera_noise_level);   end
        fprintf("[SCENARIO] Loaded from %s (fog=%.1f, illum=%.1f, noise=%.2f)\n", ...
            scenarioPath, fog, illum, noise);
    catch ME
        warning("Could not parse scenario: %s. Using defaults.", ME.message);
    end
else
    fprintf("[SCENARIO] No scenario file at %s. Using defaults.\n", scenarioPath);
end

assignin("base", "FOG_DENSITY_PERCENT", fog);
assignin("base", "ILLUMINATION_LUX",    illum);
assignin("base", "CAMERA_NOISE_LEVEL",  noise);

% --- Generate mountain terrain ---
[Xg, Yg, Zg] = mountain_terrain_grid(200, 2.0);
assignin("base", "TERRAIN_X", Xg);
assignin("base", "TERRAIN_Y", Yg);
assignin("base", "TERRAIN_Z", Zg);

% --- Tree obstacles (5 trees on mountainside) ---
treeXY = [
    -30,   5;
    -10,  -8;
     10,   3;
     30, -10;
     50,   8
];
treeRadius = 1.2;
treeHeight = 7.0;

OBSTACLES_XYZ = zeros(5, 3);
OBSTACLES_RH  = zeros(5, 2);
for k = 1:size(treeXY,1)
    tx = treeXY(k,1); ty = treeXY(k,2);
    tz = interp2(Xg, Yg, Zg, tx, ty, "linear", 0);
    OBSTACLES_XYZ(k,:) = [tx, ty, tz];
    OBSTACLES_RH(k,:)  = [treeRadius, treeHeight];
end
assignin("base", "OBSTACLES_XYZ", OBSTACLES_XYZ);
assignin("base", "OBSTACLES_RH",  OBSTACLES_RH);

% --- UAV initial state and constant velocity ---
uavX0 = -80; uavY0 = 0;
uavZ0 = max(Zg(:)) + 8;    % just above the highest peak
assignin("base", "UAV_X0_VEC", [uavX0, uavY0, uavZ0]);
assignin("base", "UAV_V_VEC",  [3.0, 0.0, 0.0]);   % m/s along +X

% --- Camera intrinsics: [fx, fy, cx, cy, pitch_down_deg] ---
% Pitch tilts the camera down so trees on the mountainside fall in the FOV.
assignin("base", "CAM_INTRIN", [600, 600, 320, 180, 15]);
assignin("base", "IMG_SIZE",   [640, 360]);

end

% =========================================================================
% Mountain terrain (Gaussian mixture)
% =========================================================================
function [Xg, Yg, Zg] = mountain_terrain_grid(extent, step)
xs = -extent/2 : step : extent/2;
ys = -extent/2 : step : extent/2;
[Xg, Yg] = meshgrid(xs, ys);

peaks = [
       0,    0,  30,  35;
      55,   30,  22,  28;
     -50,   25,  18,  30;
      35,  -45,  20,  25;
     -30,  -40,  15,  22
];

Zg = zeros(size(Xg));
for k = 1:size(peaks,1)
    cx = peaks(k,1); cy = peaks(k,2); h = peaks(k,3); s = peaks(k,4);
    Zg = Zg + h * exp(-((Xg-cx).^2 + (Yg-cy).^2) / (2*s^2));
end
Zg = Zg + 0.6 * sin(0.10*Xg) .* cos(0.12*Yg);
Zg = max(Zg, 0);
end

% =========================================================================
% Subsystem: Scenario_Params
% =========================================================================
function build_scenario_params(sys)
safe_delete(sys + "/In1");
safe_delete(sys + "/Out1");

specs = {
    "C_FOG",        "FOG_DENSITY_PERCENT", "fog";
    "C_ILLUM",      "ILLUMINATION_LUX",    "illum";
    "C_NOISE",      "CAMERA_NOISE_LEVEL",  "noise";
    "C_OBS_XYZ",    "OBSTACLES_XYZ",       "obs_xyz";
    "C_OBS_RH",     "OBSTACLES_RH",        "obs_rh";
    "C_UAV_X0",     "UAV_X0_VEC",          "uav_x0";
    "C_UAV_V",      "UAV_V_VEC",           "uav_v";
    "C_CAM",        "CAM_INTRIN",          "cam_intrin";
    "C_IMG",        "IMG_SIZE",            "img_size";
};

for i = 1:size(specs,1)
    y = 30 + (i-1)*45;
    add_block("simulink/Commonly Used Blocks/Constant", sys + "/" + specs{i,1});
    set_param(sys + "/" + specs{i,1}, ...
        "Value", specs{i,2}, ...
        "Position", [30 y 100 y+25]);

    add_block("simulink/Ports & Subsystems/Out1", sys + "/" + specs{i,3});
    set_param(sys + "/" + specs{i,3}, "Position", [220 y 250 y+20]);

    safe_add_line(sys, specs{i,1} + "/1", specs{i,3} + "/1");
end
end

% =========================================================================
% Subsystem: UAV_Dynamics
% =========================================================================
function build_uav_dynamics(sys)
safe_delete(sys + "/In1");
safe_delete(sys + "/Out1");

% Inputs
add_block("simulink/Ports & Subsystems/In1", sys + "/t");
set_param(sys + "/t", "Position", [20 40 50 58]);

add_block("simulink/Ports & Subsystems/In1", sys + "/x0");
set_param(sys + "/x0", "Position", [20 90 50 108]);

add_block("simulink/Ports & Subsystems/In1", sys + "/v");
set_param(sys + "/v", "Position", [20 140 50 158]);

% Output
add_block("simulink/Ports & Subsystems/Out1", sys + "/uav_xyz");
set_param(sys + "/uav_xyz", "Position", [430 90 460 108]);

% MATLAB Function block: uav_xyz = x0 + t*v
add_block("simulink/User-Defined Functions/MATLAB Function", sys + "/F_uav_pos");
set_param(sys + "/F_uav_pos", "Position", [180 30 360 170]);
set_matlab_function_script(sys + "/F_uav_pos", char( ...
    "function uav_xyz = F_uav_pos(t, x0, v)" + newline + ...
    "%#codegen" + newline + ...
    "uav_xyz = x0 + t .* v;" + newline + ...
    "end"));

safe_add_line(sys, "t/1",         "F_uav_pos/1");
safe_add_line(sys, "x0/1",        "F_uav_pos/2");
safe_add_line(sys, "v/1",         "F_uav_pos/3");
safe_add_line(sys, "F_uav_pos/1", "uav_xyz/1");
end

% =========================================================================
% Subsystem: Object_Detector
% =========================================================================
function build_object_detector(sys)
safe_delete(sys + "/In1");
safe_delete(sys + "/Out1");

% Inputs
inNames = ["uav_xyz","obs_xyz","obs_rh","fog","illum","noise","cam_intrin","img_size"];
for i = 1:numel(inNames)
    y = 25 + (i-1)*38;
    add_block("simulink/Ports & Subsystems/In1", sys + "/" + inNames(i));
    set_param(sys + "/" + inNames(i), "Position", [20 y 50 y+18]);
end

% Outputs
outNames = ["det_scores","det_bboxes","gt_bboxes","rel_dists"];
for i = 1:numel(outNames)
    y = 50 + (i-1)*60;
    add_block("simulink/Ports & Subsystems/Out1", sys + "/" + outNames(i));
    set_param(sys + "/" + outNames(i), "Position", [560 y 590 y+18]);
end

% MATLAB Function block
add_block("simulink/User-Defined Functions/MATLAB Function", sys + "/F_detector");
set_param(sys + "/F_detector", "Position", [200 30 460 360]);

scriptText = build_detector_script_text();
set_matlab_function_script(sys + "/F_detector", scriptText);

% Wire inputs
for i = 1:numel(inNames)
    safe_add_line(sys, inNames(i) + "/1", "F_detector/" + i);
end

% Wire outputs
for i = 1:numel(outNames)
    safe_add_line(sys, "F_detector/" + i, outNames(i) + "/1");
end
end

% =========================================================================
% Detector script (MATLAB Function block body)
% =========================================================================
function txt = build_detector_script_text()
lines = [
"function [det_scores, det_bboxes, gt_bboxes, rel_dists] = F_detector(uav_xyz, obs_xyz, obs_rh, fog_pct, illum_lux, noise_level, cam_intrin, img_size)"
"%#codegen"
"% obs_xyz: 5x3, obs_rh: 5x2 (radius, height)"
"% Returns geometry-projected bboxes (gt) plus weather-degraded detector outputs."
""
"N = size(obs_xyz, 1);"
"det_scores = zeros(N, 1);"
"det_bboxes = zeros(N, 4);"
"gt_bboxes  = zeros(N, 4);"
"rel_dists  = zeros(N, 1);"
""
"fx = cam_intrin(1); fy = cam_intrin(2); cx0 = cam_intrin(3); cy0 = cam_intrin(4);"
"pitch = cam_intrin(5) * pi/180;"
"sp = sin(pitch); cp = cos(pitch);"
"img_w = img_size(1); img_h = img_size(2);"
""
"% Weather visibility (shared across obstacles)"
"fog_norm   = max(0, min(1, fog_pct / 100));"
"low_light  = max(0, (3000 - illum_lux) / 3000);"
"high_light = max(0, (illum_lux - 12000) / 12000);"
"visibility = max(0, 1 - 0.6*fog_norm - 0.20*low_light - 0.10*high_light - 0.20*noise_level);"
""
"for k = 1:N"
"    base_x = obs_xyz(k,1); base_y = obs_xyz(k,2); base_z = obs_xyz(k,3);"
"    r = obs_rh(k,1); h = obs_rh(k,2);"
""
"    dx = base_x - uav_xyz(1);"
"    dy = base_y - uav_xyz(2);"
"    dz_base = base_z - uav_xyz(3);"
"    dz_top  = (base_z + h) - uav_xyz(3);"
""
"    rel_dists(k) = sqrt(dx*dx + dy*dy + (dz_base + h/2)^2);"
""
"    % Camera looks +X then pitched down by 'pitch' rad (around Y axis)."
"    % Camera frame in world:"
"    %   z_c (forward) = ( cos(p), 0, -sin(p) )"
"    %   x_c (right)   = ( 0,      1,  0      )"
"    %   y_c (down)    = (-sin(p), 0, -cos(p) )"
"    % cam_x =  dy ; cam_y = -dx*sp - dz*cp ; cam_z = dx*cp - dz*sp"
"    cz_base = dx*cp - dz_base*sp;"
"    cz_top  = dx*cp - dz_top *sp;"
"    if cz_base < 0.5 || cz_top < 0.5"
"        continue;"  % obstacle behind camera plane
"    end"
"    cz_avg = 0.5*(cz_base + cz_top);"
""
"    cam_x_left  = dy - r;"
"    cam_x_right = dy + r;"
"    cam_y_top   = -dx*sp - dz_top *cp;"
"    cam_y_bot   = -dx*sp - dz_base*cp;"
""
"    u_left  = fx * cam_x_left  / cz_avg + cx0;"
"    u_right = fx * cam_x_right / cz_avg + cx0;"
"    v_top   = fy * cam_y_top   / cz_top  + cy0;"
"    v_bot   = fy * cam_y_bot   / cz_base + cy0;"
""
"    bbox_u = u_left;"
"    bbox_v = v_top;"
"    bbox_w = u_right - u_left;"
"    bbox_h = v_bot - v_top;"
""
"    if bbox_w <= 0 || bbox_h <= 0"
"        continue;"
"    end"
""
"    % Reject if entirely outside image"
"    if u_right < 0 || u_left > img_w || v_bot < 0 || v_top > img_h"
"        continue;"
"    end"
""
"    % Reject if too far"
"    if cz_avg > 90"
"        continue;"
"    end"
""
"    gt_bboxes(k,:) = [bbox_u, bbox_v, bbox_w, bbox_h];"
""
"    dist_attn = max(0.05, 1 - cz_avg/90);"
"    sc = visibility * dist_attn;"
""
"    % Deterministic jitter (no randn in codegen)"
"    seed = mod(uav_xyz(1)*0.7 + base_x*1.3 + base_y*0.9, 6.2831853);"
"    jx = sin(seed)        * (3 + 18*(1 - visibility));"
"    jy = cos(seed*1.3)    * (3 + 18*(1 - visibility));"
"    jw = sin(seed*1.7)    * (1 + 6*(1  - visibility));"
"    jh = cos(seed*2.1)    * (1 + 6*(1  - visibility));"
""
"    det_bboxes(k,:) = [bbox_u + jx, bbox_v + jy, max(2, bbox_w + jw), max(2, bbox_h + jh)];"
"    det_scores(k)   = sc;"
"end"
"end"
];
txt = char(strjoin(lines, newline));
end

% =========================================================================
% Helpers
% =========================================================================
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
if isstring(scriptText)
    scriptText = char(strjoin(scriptText, ""));
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
