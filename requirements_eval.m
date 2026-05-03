function eval = requirements_eval(simOut, opts)
% requirements_eval — Evaluate three independent requirements from a
% mountain_uav_model simulation output.
%
%   REQ-1  Detection performance       :  mAP50 >= map_threshold
%   REQ-2  Safety clearance            :  min(UAV->terrain/obstacle) >= clearance_threshold
%   REQ-3  Detection continuity        :  worst run of consecutive misses <= continuity_threshold
%
% Returns a struct with per-requirement pass/fail/value plus an overall
% verdict (AND of all three).
%
% Usage:
%   eval = requirements_eval(simOut)
%   eval = requirements_eval(simOut, struct('map_threshold', 0.85, ...
%                                           'clearance_threshold', 2.0, ...
%                                           'continuity_threshold', 3, ...
%                                           'det_score_threshold', 0.30))

if nargin < 2 || isempty(opts), opts = struct(); end
if ~isfield(opts, "map_threshold"),        opts.map_threshold        = 0.50; end
if ~isfield(opts, "clearance_threshold"),  opts.clearance_threshold  = 2.0;  end
if ~isfield(opts, "continuity_threshold"), opts.continuity_threshold = 3;    end
if ~isfield(opts, "det_score_threshold"),  opts.det_score_threshold  = 0.30; end

[t_vec, uav_xyz] = read_log_vec(simOut, "uav_xyz_log");
[~,     scores]  = read_log_vec(simOut, "det_scores_log");
[~,     detBB]   = read_log_3d (simOut, "det_bboxes_log");
[~,     gtBB]    = read_log_3d (simOut, "gt_bboxes_log");

Nt   = numel(t_vec);
Nobs = size(gtBB, 2);

% -------------------------------------------------------------------------
% REQ-1: mAP50
% -------------------------------------------------------------------------
[map50, tp, fp, fn, total_gt] = compute_map50(gtBB, detBB, scores, ...
    Nobs, opts.det_score_threshold);

req1.id        = "REQ-1";
req1.name      = "Detection performance (mAP50)";
req1.threshold = opts.map_threshold;
req1.value     = map50;
req1.passed    = map50 >= opts.map_threshold;
req1.tp        = tp;
req1.fp        = fp;
req1.fn        = fn;
req1.total_gt  = total_gt;

% -------------------------------------------------------------------------
% REQ-2: minimum clearance UAV <-> terrain/obstacles
% -------------------------------------------------------------------------
[min_clear, t_min, where] = compute_min_clearance(uav_xyz, t_vec);

req2.id        = "REQ-2";
req2.name      = "Safety clearance (min distance UAV->terrain/obstacles)";
req2.threshold = opts.clearance_threshold;
req2.value     = min_clear;
req2.passed    = min_clear >= opts.clearance_threshold;
req2.violated_at_time = t_min;
req2.violated_against = where;

% -------------------------------------------------------------------------
% REQ-3: continuity (worst run of consecutive whole-frame misses)
% -------------------------------------------------------------------------
[worst_run, worst_run_start_t] = compute_worst_miss_run( ...
    gtBB, scores, opts.det_score_threshold, t_vec);

req3.id        = "REQ-3";
req3.name      = "Detection continuity (max consecutive missed frames)";
req3.threshold = opts.continuity_threshold;
req3.value     = worst_run;
req3.passed    = worst_run <= opts.continuity_threshold;
req3.worst_run_start_time = worst_run_start_t;

% -------------------------------------------------------------------------
% Overall verdict
% -------------------------------------------------------------------------
eval = struct();
eval.req1 = req1;
eval.req2 = req2;
eval.req3 = req3;
eval.all_passed = req1.passed && req2.passed && req3.passed;
eval.violated_count = double(~req1.passed) + double(~req2.passed) + double(~req3.passed);
eval.frames    = Nt;
eval.obstacles = Nobs;

% -------------------------------------------------------------------------
% Per-requirement contributing-factor hint (used by xai_input)
% -------------------------------------------------------------------------
fog   = read_base("FOG_DENSITY_PERCENT", 0);
illum = read_base("ILLUMINATION_LUX",    8000);
noise = read_base("CAMERA_NOISE_LEVEL",  0);

eval.scenario_factors = struct( ...
    "fog_density_percent", fog, ...
    "illumination_lux",    illum, ...
    "camera_noise_level",  noise);

print_summary(eval);
end

% =========================================================================
% mAP50
% =========================================================================
function [ap50, tp, fp, fn, totalGt] = compute_map50(gtBB, detBB, scores, Nobs, scoreTh)
Nt = size(gtBB, 1);
tp = 0; fp = 0; fn = 0; ap50 = 0;
allDet = zeros(0, 2);
totalGt = 0;

for ii = 1:Nt
    for k = 1:Nobs
        gt = reshape(gtBB(ii, k, :), 1, []);
        dt = reshape(detBB(ii, k, :), 1, []);
        sc = scores(ii, k);
        gtPresent  = any(gt ~= 0);
        detPresent = (sc > scoreTh) && any(dt ~= 0);

        if gtPresent, totalGt = totalGt + 1; end

        if gtPresent && detPresent
            iou = bbox_iou(dt, gt);
            if iou >= 0.5
                tp = tp + 1; allDet(end+1,:) = [sc, 1]; %#ok<AGROW>
            else
                fp = fp + 1; fn = fn + 1; allDet(end+1,:) = [sc, 0]; %#ok<AGROW>
            end
        elseif gtPresent && ~detPresent
            fn = fn + 1;
        elseif ~gtPresent && detPresent
            fp = fp + 1; allDet(end+1,:) = [sc, 0]; %#ok<AGROW>
        end
    end
end

if totalGt > 0 && ~isempty(allDet)
    [~, ord] = sort(allDet(:,1), "descend");
    allDet = allDet(ord, :);
    cumTp = cumsum(allDet(:,2));
    cumFp = cumsum(1 - allDet(:,2));
    precision = cumTp ./ max(cumTp + cumFp, eps);
    recall = cumTp / totalGt;
    mrec = [0; recall; 1];
    mpre = [0; precision; 0];
    for i = numel(mpre)-1:-1:1
        mpre(i) = max(mpre(i), mpre(i+1));
    end
    idx = find(mrec(2:end) ~= mrec(1:end-1));
    ap50 = sum((mrec(idx+1) - mrec(idx)) .* mpre(idx+1));
end
end

function iou = bbox_iou(a, b)
ax1=a(1); ay1=a(2); ax2=a(1)+a(3); ay2=a(2)+a(4);
bx1=b(1); by1=b(2); bx2=b(1)+b(3); by2=b(2)+b(4);
ix1=max(ax1,bx1); iy1=max(ay1,by1);
ix2=min(ax2,bx2); iy2=min(ay2,by2);
iw=max(0,ix2-ix1); ih=max(0,iy2-iy1);
inter = iw*ih;
ua = max(0,a(3))*max(0,a(4)) + max(0,b(3))*max(0,b(4)) - inter;
if ua <= 0, iou = 0; else, iou = inter/ua; end
end

% =========================================================================
% REQ-2: minimum clearance distance
% =========================================================================
function [minClear, tMin, where] = compute_min_clearance(uav_xyz, t_vec)
% Distance from UAV to (a) terrain surface beneath it, (b) every tree.
Xg = read_base("TERRAIN_X", []);
Yg = read_base("TERRAIN_Y", []);
Zg = read_base("TERRAIN_Z", []);
obs_xyz = read_base("OBSTACLES_XYZ", zeros(0,3));
obs_rh  = read_base("OBSTACLES_RH",  zeros(0,2));

Nt = size(uav_xyz, 1);
minClear = inf;
tMin     = NaN;
where    = "";

for ii = 1:Nt
    ux = uav_xyz(ii, 1); uy = uav_xyz(ii, 2); uz = uav_xyz(ii, 3);

    % Terrain clearance: vertical distance UAV is above ground beneath it
    if ~isempty(Xg)
        zGround = interp2(Xg, Yg, Zg, ux, uy, "linear", 0);
        dTerr = uz - zGround;
    else
        dTerr = inf;
    end

    % Obstacle clearance: distance from UAV to each tree's nearest surface
    dObs = inf;
    obsHit = "";
    for k = 1:size(obs_xyz, 1)
        tx = obs_xyz(k,1); ty = obs_xyz(k,2); tz_base = obs_xyz(k,3);
        r  = obs_rh(k,1);  h  = obs_rh(k,2);

        % Nearest point on cylinder (radius r, height h, vertical axis)
        dxy = sqrt((ux-tx)^2 + (uy-ty)^2);
        dxy_to_surf = max(0, dxy - r);

        % Vertical dist to cylinder span [tz_base, tz_base+h]
        if uz < tz_base
            dz = tz_base - uz;
        elseif uz > tz_base + h
            dz = uz - (tz_base + h);
        else
            dz = 0;
        end

        d = sqrt(dxy_to_surf^2 + dz^2);
        if d < dObs
            dObs = d;
            obsHit = sprintf("tree_%d", k);
        end
    end

    if dTerr < dObs
        d = dTerr; src = "terrain";
    else
        d = dObs;  src = obsHit;
    end

    if d < minClear
        minClear = d;
        tMin = t_vec(ii);
        where = src;
    end
end
end

% =========================================================================
% REQ-3: worst run of consecutive whole-frame misses
% =========================================================================
function [worstRun, worstStartT] = compute_worst_miss_run(gtBB, scores, scoreTh, t_vec)
Nt = size(gtBB, 1);
miss = false(Nt, 1);

for ii = 1:Nt
    anyGt  = any(any(gtBB(ii,:,:) ~= 0, 3), 2);
    anyDet = any(scores(ii, :) > scoreTh);
    miss(ii) = anyGt && ~anyDet;   % all GT present but no detection passed
end

worstRun = 0;
worstStartT = NaN;
runLen = 0;
runStart = 0;
for ii = 1:Nt
    if miss(ii)
        if runLen == 0, runStart = ii; end
        runLen = runLen + 1;
        if runLen > worstRun
            worstRun = runLen;
            worstStartT = t_vec(runStart);
        end
    else
        runLen = 0;
    end
end
end

% =========================================================================
% Common log readers (mirror mountain_visualizer)
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
    try, raw = evalin("base", name); catch, raw = []; end
end
if isempty(raw), error("Could not find logged signal: %s", name); end
if isstruct(raw) && isfield(raw, "time") && isfield(raw, "signals")
    s.time = raw.time;
    s.values = raw.signals.values;
else
    error("Unexpected signal format for %s", name);
end
end

function val = read_base(name, default)
try
    val = evalin("base", char(name));
catch
    val = default;
end
end

function print_summary(eval)
mark = @(b) ternary(b, "PASS", "FAIL");
fprintf("\n+-- Requirements Evaluation --------------------------------------+\n");
fprintf("| %s [%s]  threshold=%.2f  value=%.4f\n", eval.req1.id, mark(eval.req1.passed), eval.req1.threshold, eval.req1.value);
fprintf("| %s [%s]  threshold=%.2f m value=%.2f m  (closest=%s)\n", eval.req2.id, mark(eval.req2.passed), eval.req2.threshold, eval.req2.value, eval.req2.violated_against);
fprintf("| %s [%s]  threshold=%d frames value=%d frames\n", eval.req3.id, mark(eval.req3.passed), eval.req3.threshold, eval.req3.value);
fprintf("| OVERALL: %s  (violated %d of 3)\n", mark(eval.all_passed), eval.violated_count);
fprintf("+-----------------------------------------------------------------+\n\n");
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
