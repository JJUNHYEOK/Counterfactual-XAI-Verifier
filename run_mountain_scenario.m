function [simOut, summary] = run_mountain_scenario(varargin)
% run_mountain_scenario
% -------------------------------------------------------------------------
% Entry point for the mountain UAV scenario test:
%   1) Loads the current scenario from data/scenario_iter_001.json
%   2) Builds mountain_uav_model.slx (if missing)
%   3) Runs the Simulink simulation
%   4) Renders a 3D figure with the mountain, UAV trajectory and camera view
%   5) Reports mAP50 / TP / FP / FN
%
% Usage:
%   run_mountain_scenario;                       % default GUI run + viz
%   run_mountain_scenario("Visible", false, ...
%                         "SavePng", "out/scene.png");  % batch / headless
%   [simOut, summary] = run_mountain_scenario(...);
%
% Optional name/value arguments:
%   Visible       (logical, true)   - show figure on screen
%   Animate       (logical, true)   - animate over simulation time
%   SavePng       (string,  "")     - if non-empty, write final figure here
%   Rebuild       (logical, false)  - force rebuild of the .slx model
% -------------------------------------------------------------------------

p = inputParser;
addParameter(p, "Visible", true,  @(x) islogical(x) || isnumeric(x));
addParameter(p, "Animate", true,  @(x) islogical(x) || isnumeric(x));
addParameter(p, "SavePng", "",    @(x) ischar(x) || isstring(x));
addParameter(p, "Rebuild", false, @(x) islogical(x) || isnumeric(x));
parse(p, varargin{:});

opts = p.Results;
opts.Visible = logical(opts.Visible);
opts.Animate = logical(opts.Animate);
opts.SavePng = string(opts.SavePng);
opts.Rebuild = logical(opts.Rebuild);

mdl = "mountain_uav_model";

% --- Build model if missing or rebuild requested ---
if opts.Rebuild || ~isfile(mdl + ".slx")
    fprintf("[STEP 1] Building Simulink model...\n");
    build_mountain_uav_model(false);
else
    fprintf("[STEP 1] Reusing existing %s.slx (set Rebuild=true to force rebuild)\n", mdl);
    % Re-seed base workspace from current scenario JSON
    seed_base_workspace_from_scenario();
end

if ~bdIsLoaded(mdl)
    load_system(mdl);
end

% Ensure model has access to base workspace
try
    set_param(mdl, "SimulationCommand", "Update");
catch ME
    warning("Model update warning: %s", ME.message);
end

% --- Run simulation ---
fprintf("[STEP 2] Simulating %s ...\n", mdl);
tStart = tic;
simOut = sim(mdl);
fprintf("[STEP 2] Done in %.2fs.\n", toc(tStart));

% --- Visualize ---
fprintf("[STEP 3] Rendering visualization...\n");
vizOpts = struct( ...
    "visible",   opts.Visible, ...
    "animate",   opts.Animate, ...
    "savePath",  opts.SavePng);

[~, summary] = mountain_visualizer(simOut, vizOpts);

% --- Final report ---
fog   = evalin("base", "FOG_DENSITY_PERCENT");
illum = evalin("base", "ILLUMINATION_LUX");
noise = evalin("base", "CAMERA_NOISE_LEVEL");

fprintf("\n============================================================\n");
fprintf("  Mountain UAV Scenario Summary\n");
fprintf("============================================================\n");
fprintf("  Scenario: fog=%.1f%%  illum=%.0f lux  noise=%.2f\n", fog, illum, noise);
fprintf("  Detections: TP=%d  FP=%d  FN=%d  (total GT=%d)\n", ...
    summary.tp, summary.fp, summary.fn, summary.total_gt);
fprintf("  mAP50 = %.4f  (threshold = 0.85, %s)\n", ...
    summary.map50, ...
    iif(summary.map50 < 0.85, "REQUIREMENT VIOLATED", "requirement satisfied"));
fprintf("============================================================\n\n");

end

% =========================================================================
% Helpers
% =========================================================================
function seed_base_workspace_from_scenario()
% Reload scenario JSON and refresh the base workspace constants Simulink
% reads (does NOT regenerate terrain / obstacle layout, that is fixed by
% build_mountain_uav_model).
fog = 30.0; illum = 4000.0; noise = 0.1;
scenarioPath = fullfile(pwd, "data", "scenario_iter_001.json");
if isfile(scenarioPath)
    try
        sc = jsondecode(fileread(scenarioPath));
        env = sc.environment_parameters;
        if isfield(env, "fog_density_percent"), fog   = double(env.fog_density_percent); end
        if isfield(env, "illumination_lux"),    illum = double(env.illumination_lux);    end
        if isfield(env, "camera_noise_level"),  noise = double(env.camera_noise_level);  end
    catch
    end
end
assignin("base", "FOG_DENSITY_PERCENT", fog);
assignin("base", "ILLUMINATION_LUX",    illum);
assignin("base", "CAMERA_NOISE_LEVEL",  noise);

% Make sure the static scene constants exist (rebuild if missing)
needRebuild = ~all_present_in_base({"TERRAIN_X","TERRAIN_Y","TERRAIN_Z", ...
    "OBSTACLES_XYZ","OBSTACLES_RH","UAV_X0_VEC","UAV_V_VEC","CAM_INTRIN","IMG_SIZE"});
if needRebuild
    fprintf("[INFO] Static scene vars missing in base workspace; rebuilding model.\n");
    build_mountain_uav_model(false);
end
end

function tf = all_present_in_base(names)
tf = true;
for i = 1:numel(names)
    try
        evalin("base", char(names{i}));
    catch
        tf = false; return;
    end
end
end

function out = iif(cond, a, b)
if cond, out = a; else, out = b; end
end
