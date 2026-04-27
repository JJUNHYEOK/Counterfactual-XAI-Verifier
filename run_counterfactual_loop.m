function history = run_counterfactual_loop(maxIter, opts)
% run_counterfactual_loop — Iterative counterfactual search over UAV
% mountain detection. Each iteration:
%   1)  Load current scenario JSON (initial: data/scenario_iter_001.json)
%   2)  Push fog/illum/noise into base workspace, build/sim the model
%   3)  Evaluate REQ-1, REQ-2, REQ-3 with two detectors:
%         - geometric  (current Object_Detector inside Simulink)
%         - image-feature (image_detector.m on rendered frames)
%   4)  Save scenario / eval / xai_input JSONs into data/
%   5)  Render a figure to assets/iter_NNN.png
%   6)  Call Python LLM (llm_agent/gpt_for_simulink) for next scenario
%   7)  Stop early if all three requirements pass (target_status='PASS')
%       or after maxIter iterations.
%
% Usage:
%   run_counterfactual_loop                      % 3 iters, defaults
%   run_counterfactual_loop(5)
%   run_counterfactual_loop(3, struct( ...
%       'no_llm', true, ...                      % skip OpenAI, use fallback
%       'eval_stride', 10, ...                   % render every 10th frame
%       'save_pngs', true, ...
%       'rebuild_model', false))

% ---------- defaults ----------
if nargin < 1 || isempty(maxIter), maxIter = 3; end
if nargin < 2 || isempty(opts), opts = struct(); end
if ~isfield(opts, "no_llm"),         opts.no_llm = false; end
if ~isfield(opts, "eval_stride"),    opts.eval_stride = 8; end
if ~isfield(opts, "save_pngs"),      opts.save_pngs = true; end
if ~isfield(opts, "rebuild_model"),  opts.rebuild_model = false; end
if ~isfield(opts, "python_exe"),     opts.python_exe = "python"; end
if ~isfield(opts, "show_figure"),    opts.show_figure = true; end
if ~isfield(opts, "animate_per_iter"), opts.animate_per_iter = false; end
if ~isfield(opts, "iter_pause_seconds"), opts.iter_pause_seconds = 1.5; end
if ~isfield(opts, "early_stop_on_pass"), opts.early_stop_on_pass = false; end
if ~isfield(opts, "convergence_step"), opts.convergence_step = 0.04; end

mdl = "mountain_uav_model";

dataDir   = fullfile(pwd, "data");
assetsDir = fullfile(pwd, "assets");
if ~exist(dataDir, "dir"),   mkdir(dataDir);   end
if ~exist(assetsDir, "dir"), mkdir(assetsDir); end

% Initial scenario must exist
scenarioPath = fullfile(dataDir, "scenario_iter_001.json");
if ~isfile(scenarioPath)
    seed = struct( ...
        "scenario_id", "scenario_001", ...
        "target_hypothesis", "Initial seed scenario", ...
        "environment_parameters", struct( ...
            "fog_density_percent", 30.0, ...
            "illumination_lux",    4000.0, ...
            "camera_noise_level",  0.1), ...
        "llm_reasoning", "Seed.");
    write_json(scenarioPath, seed);
end

history = struct([]);

% Always build at loop start: ensures base workspace contains TERRAIN_*,
% OBSTACLES_*, CAM_INTRIN, IMG_SIZE, UAV_*_VEC etc. Build is cheap (~2 s).
fprintf("[BUILD] Generating %s.slx (seeds base workspace)\n", mdl);
build_mountain_uav_model(false);
if ~bdIsLoaded(mdl), load_system(mdl); end

print_banner(maxIter, opts);

for iter = 1:maxIter
    iterTag = sprintf("iter_%03d", iter);
    fprintf("\n================ Iteration %d / %d (%s) ================\n", ...
        iter, maxIter, iterTag);

    % --- 1. Load scenario, push to base workspace
    scenario = read_json(scenarioPath);
    env = scenario.environment_parameters;
    assignin("base", "FOG_DENSITY_PERCENT", double(env.fog_density_percent));
    assignin("base", "ILLUMINATION_LUX",    double(env.illumination_lux));
    assignin("base", "CAMERA_NOISE_LEVEL",  double(env.camera_noise_level));
    fprintf("[SCEN] fog=%.1f%%  illum=%.0flx  noise=%.2f  (file=%s)\n", ...
        env.fog_density_percent, env.illumination_lux, env.camera_noise_level, scenarioPath);

    try
        set_param(mdl, "SimulationCommand", "Update");
    catch
    end

    % --- 2. Run simulation
    fprintf("[SIM]  running...\n");
    tStart = tic;
    simOut = sim(mdl);
    fprintf("[SIM]  done in %.2fs\n", toc(tStart));

    % --- 3a. Geometric eval (cheap, all frames)
    fprintf("[EVAL-GEOM] computing on geometric detector...\n");
    eval_geom = requirements_eval(simOut);

    % --- 3b. Image-based eval (slower, strided)
    fprintf("[EVAL-IMG]  rendering + image_detector (stride=%d)...\n", opts.eval_stride);
    eval_img = image_based_eval(simOut, opts.eval_stride);

    % --- 4. Save iteration artifacts
    write_json(fullfile(dataDir, "scenario_" + iterTag + ".json"), scenario);
    write_json(fullfile(dataDir, "eval_" + iterTag + ".json"), struct( ...
        "iter", iter, ...
        "geometric", eval_geom, ...
        "image_based", eval_img));

    % Forward _loop_state (renamed to x_loop_state by jsondecode) so the
    % adapter's bisection state machine sees prior history.
    prev_state = [];
    if isfield(scenario, "x_loop_state"), prev_state = scenario.x_loop_state;
    elseif isfield(scenario, "loop_state"), prev_state = scenario.loop_state;
    end

    xai_input = build_xai_input(eval_img, scenario.scenario_id, iterTag, ...
        fullfile(dataDir, "xai_input.json"), prev_state);
    write_json(fullfile(dataDir, "xai_input_" + iterTag + ".json"), xai_input);

    % --- 5. Visualization (optionally on-screen) + PNG save
    if opts.save_pngs || opts.show_figure
        try
            pngPath = "";
            if opts.save_pngs
                pngPath = fullfile(assetsDir, iterTag + ".png");
            end
            vizOpts = struct( ...
                "visible", opts.show_figure, ...
                "animate", opts.animate_per_iter, ...
                "savePath", pngPath);
            mountain_visualizer(simOut, vizOpts);
            if opts.show_figure
                drawnow;
                pause(opts.iter_pause_seconds);
            end
        catch ME
            warning("[VIZ] Figure failed: %s", ME.message);
        end
    end

    % --- 6. Build history entry
    rec = struct();
    rec.iter = iter;
    rec.iterTag = iterTag;
    rec.scenario = scenario;
    rec.eval_geom = eval_geom;
    rec.eval_img  = eval_img;
    rec.xai_input = xai_input;
    history = vertcat_struct(history, rec); %#ok<NASGU>
    history(iter) = rec; %#ok<AGROW>

    % --- 7. Termination check
    %   - early_stop_on_pass=true  : stop the moment all REQs pass (legacy)
    %   - early_stop_on_pass=false : keep bisecting toward the boundary
    %     (stop only at maxIter or when step_factor has converged below
    %     convergence_step, meaning the boundary is well localized).
    if opts.early_stop_on_pass && eval_img.all_passed
        fprintf("[STOP] All three requirements satisfied at iter %d (early_stop_on_pass).\n", iter);
        break;
    end
    if iter == maxIter
        fprintf("[STOP] Reached maxIter=%d.\n", maxIter);
        break;
    end

    % --- 8. Call LLM for next scenario
    nextTag    = sprintf("iter_%03d", iter+1);
    nextPath   = fullfile(dataDir, "scenario_" + nextTag + ".json");
    flagNoLlm  = ""; if opts.no_llm, flagNoLlm = " --no_llm"; end
    cmd = sprintf('%s -m llm_agent.gpt_for_simulink --input "%s" --output "%s" --iter_tag "%s"%s', ...
        char(opts.python_exe), ...
        char(fullfile(dataDir, "xai_input.json")), ...
        char(nextPath), char(nextTag), flagNoLlm);
    fprintf("[LLM]  %s\n", cmd);
    [status, cmdout] = system(cmd);
    fprintf("%s", cmdout);
    if status ~= 0 || ~isfile(nextPath)
        fprintf("[LLM]  FAILED (status=%d). Falling back to deterministic mutation in-process.\n", status);
        nextScenario = matlab_rule_mutation(scenario, eval_img);
        write_json(nextPath, nextScenario);
    end

    scenarioPath = nextPath;

    % --- 9. Convergence check (boundary localized)
    try
        nextScObj = read_json(nextPath);
        if isfield(nextScObj, "x_loop_state") && isfield(nextScObj.x_loop_state, "step_factor")
            stepNow = double(nextScObj.x_loop_state.step_factor);
            if stepNow < opts.convergence_step
                fprintf("[STOP] Bisection step_factor=%.4f < %.4f, boundary localized.\n", ...
                    stepNow, opts.convergence_step);
                break;
            end
        end
    catch
    end
end

% --- Final summary ---
print_history(history);

summaryPath = fullfile(dataDir, "loop_summary.json");
write_json(summaryPath, summarize(history));
fprintf("\n[DONE] Summary saved: %s\n", summaryPath);
end


% =========================================================================
% Image-based requirement eval
% =========================================================================
function eval_img = image_based_eval(simOut, stride)
% Strided re-evaluation: render the camera image at every 'stride'-th frame
% and compute mAP50 + continuity using image_detector. REQ-2 (clearance)
% is the same as geometric since it depends only on UAV pose, not detection.

[t_vec, uav_xyz] = read_log_vec(simOut, "uav_xyz_log");
[~,     gtBB]    = read_log_3d (simOut, "gt_bboxes_log");

Nt   = numel(t_vec);
Nobs = size(gtBB, 2);

obs_xyz = read_base("OBSTACLES_XYZ", zeros(0,3));
obs_rh  = read_base("OBSTACLES_RH",  zeros(0,2));
camIntr = read_base("CAM_INTRIN", [600 600 320 180 15]);
imgSize = read_base("IMG_SIZE",   [640 360]);
fog   = read_base("FOG_DENSITY_PERCENT", 0);
illum = read_base("ILLUMINATION_LUX",    8000);
noise = read_base("CAMERA_NOISE_LEVEL",  0);

frameIdx = 1:max(1, stride):Nt;
M = numel(frameIdx);

img_scores = zeros(M, Nobs);
img_dets   = zeros(M, Nobs, 4);
img_gts    = zeros(M, Nobs, 4);
sample_t   = zeros(M, 1);

for jj = 1:M
    ii = frameIdx(jj);
    sample_t(jj) = t_vec(ii);

    img = render_camera_image(uav_xyz(ii,:), obs_xyz, obs_rh, ...
        fog, illum, noise, camIntr, imgSize);

    gt_frame = squeeze(gtBB(ii, :, :));     % Nobs x 4
    if size(gt_frame, 1) ~= Nobs
        gt_frame = gt_frame';
    end
    img_gts(jj, :, :) = reshape(gt_frame, [1, Nobs, 4]);

    [scK, bbK] = image_detector(img, gt_frame);
    img_scores(jj, :) = scK(:)';
    img_dets(jj, :, :) = reshape(bbK, [1, Nobs, 4]);
end

% Synthesize a sim-like input for requirements_eval logic
synth = struct();
synth.uav_xyz_log    = make_struct_with_time(sample_t, uav_xyz(frameIdx, :));
synth.det_scores_log = make_struct_with_time(sample_t, permute(img_scores, [2 3 1]));   % Nobs x 1 x M
synth.det_bboxes_log = make_struct_with_time(sample_t, permute(img_dets,  [2 3 1]));    % Nobs x 4 x M
synth.gt_bboxes_log  = make_struct_with_time(sample_t, permute(img_gts,   [2 3 1]));
synth.rel_dists_log  = make_struct_with_time(sample_t, zeros(M, Nobs));

eval_img = requirements_eval(synth);
eval_img.note = sprintf("image-based, stride=%d (M=%d / %d frames)", stride, M, Nt);
end

function s = make_struct_with_time(t, vals)
% Mimic Simulink "Structure With Time" so requirements_eval can read it.
s = struct();
s.time = t(:);
s.signals = struct();
s.signals.values = vals;
end


% =========================================================================
% In-process fallback if Python adapter fails entirely
% =========================================================================
function next = matlab_rule_mutation(prev, eval_img)
env = prev.environment_parameters;
fog   = double(env.fog_density_percent);
illum = double(env.illumination_lux);
noise = double(env.camera_noise_level);

if eval_img.all_passed
    fog   = min(100,    fog + 10);
    illum = max(200,    illum * 0.75);
    noise = min(0.6,    noise + 0.05);
else
    fog   = max(0,      fog - 8);
    illum = min(20000,  illum * 1.25);
    noise = max(0,      noise - 0.04);
end

next = struct( ...
    "scenario_id",      sprintf("scenario_fallback_%d", randi(99999)), ...
    "target_hypothesis","matlab_rule_mutation (Python adapter unavailable)", ...
    "environment_parameters", struct( ...
        "fog_density_percent", fog, ...
        "illumination_lux",    illum, ...
        "camera_noise_level",  noise), ...
    "llm_reasoning",    "Fallback mutation when Python adapter failed.");
end


% =========================================================================
% Helpers
% =========================================================================
function [t, vals] = read_log_vec(simOut, name)
s = read_signal(simOut, name);
t = s.time;
v = s.values;
sz = size(v);
if numel(sz) == 2 && (sz(1) == 1 || sz(2) == 1)
    vals = v(:);
elseif numel(sz) == 2
    vals = v;
elseif numel(sz) == 3
    Nt = sz(3);
    vals = reshape(permute(v, [3 1 2]), Nt, sz(1)*sz(2));
else
    vals = v;
end
end

function [t, vals] = read_log_3d(simOut, name)
s = read_signal(simOut, name);
t = s.time;
v = s.values;
sz = size(v);
if numel(sz) == 3
    vals = permute(v, [3 1 2]);
elseif numel(sz) == 2
    Nt = numel(t);
    vals = reshape(v, [Nt, sz(2), 1]);
else
    vals = v;
end
end

function s = read_signal(simOut, name)
try, raw = simOut.get(name); catch, raw = []; end
if isempty(raw)
    if isstruct(simOut) && isfield(simOut, char(name))
        raw = simOut.(char(name));
    end
end
if isempty(raw)
    try, raw = evalin("base", char(name)); catch, raw = []; end
end
if isempty(raw), error("Could not find logged signal: %s", name); end
if isstruct(raw) && isfield(raw, "time") && isfield(raw, "signals")
    s.time   = raw.time;
    s.values = raw.signals.values;
else
    error("Unexpected signal format for %s", name);
end
end

function val = read_base(name, default)
try, val = evalin("base", char(name)); catch, val = default; end
end

function write_json(pathStr, obj)
txt = jsonencode(obj, PrettyPrint=true);
fid = fopen(pathStr, "w");
if fid < 0, error("Cannot write %s", pathStr); end
fwrite(fid, txt, "char"); fclose(fid);
end

function obj = read_json(pathStr)
txt = fileread(pathStr);
obj = jsondecode(txt);
end

function arr = vertcat_struct(arr, rec) %#ok<DEFNU>
if isempty(arr), arr = rec; return; end
arr(end+1) = rec;
end

function print_banner(maxIter, opts)
fprintf("\n============================================================\n");
fprintf("  Counterfactual loop\n");
fprintf("    maxIter         = %d\n", maxIter);
fprintf("    eval_stride     = %d\n", opts.eval_stride);
fprintf("    no_llm          = %d\n", opts.no_llm);
fprintf("    save_pngs       = %d\n", opts.save_pngs);
fprintf("    rebuild_model   = %d\n", opts.rebuild_model);
fprintf("============================================================\n");
end

function print_history(h)
fprintf("\n+-------+-------+--------+----------+----------+--------------+\n");
fprintf("| iter  | fog%% | illum  | noise    |  mAP50   |  REQs PASS?  |\n");
fprintf("+-------+-------+--------+----------+----------+--------------+\n");
for k = 1:numel(h)
    e = h(k).scenario.environment_parameters;
    map = h(k).eval_img.req1.value;
    pat = sprintf("%s%s%s", ...
        ternary(h(k).eval_img.req1.passed, "1", "0"), ...
        ternary(h(k).eval_img.req2.passed, "1", "0"), ...
        ternary(h(k).eval_img.req3.passed, "1", "0"));
    fprintf("| %4d  | %5.1f | %6.0f | %8.2f | %8.4f |   %s [r1 r2 r3]\n", ...
        h(k).iter, e.fog_density_percent, e.illumination_lux, e.camera_noise_level, ...
        map, pat);
end
fprintf("+-------+-------+--------+----------+----------+--------------+\n");
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end

function s = summarize(h)
items = cell(numel(h), 1);
for k = 1:numel(h)
    e = h(k).scenario.environment_parameters;
    items{k} = struct( ...
        "iter", h(k).iter, ...
        "fog", e.fog_density_percent, ...
        "illum", e.illumination_lux, ...
        "noise", e.camera_noise_level, ...
        "map50_geom",   h(k).eval_geom.req1.value, ...
        "map50_image",  h(k).eval_img.req1.value, ...
        "min_clear",    h(k).eval_img.req2.value, ...
        "worst_run",    h(k).eval_img.req3.value, ...
        "req1_pass",    h(k).eval_img.req1.passed, ...
        "req2_pass",    h(k).eval_img.req2.passed, ...
        "req3_pass",    h(k).eval_img.req3.passed, ...
        "all_passed",   h(k).eval_img.all_passed);
end
s = struct("iterations", {items}, "count", numel(h));
end
