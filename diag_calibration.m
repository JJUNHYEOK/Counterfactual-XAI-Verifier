function diag_calibration()
% diag_calibration — Headless calibration runner for the dashboard pipeline.
%
% Replicates exactly what mountain_uav_dashboard.m does each frame:
%   render_camera_image → image_detector → cumulative mAP@0.5
% For each of the 4 calibration cases, prints:
%   - mean detector score (people and vehicles separately)
%   - per-frame TP/FP/FN counts
%   - cumulative mAP@0.5 (VOC envelope)
%
% This is the SAME mAP that the dashboard computes — so we can diagnose
% why mAP@0.5 = 0.051 was observed and what to adjust.

fprintf('\n========================================================\n');
fprintf('  Headless calibration of dashboard detection pipeline\n');
fprintf('========================================================\n\n');

% Seed base workspace once with the dashboard's expected env
init_uav_workspace(10, 10000, 0.03);

mdl = "mountain_uav_model";
if ~bdIsLoaded(mdl)
    if ~isfile(mdl + ".slx")
        build_mountain_uav_model(false);
    else
        load_system(mdl);
    end
end

fprintf('[diag] Running Simulink once to get trajectory + GT bboxes...\n');
simOut = sim(mdl);

% Read trajectory + GT
[t_vec, uav_xyz] = read_log_vec(simOut, "uav_xyz_log");
[~,     gtBB]    = read_log_3d (simOut, "gt_bboxes_log");

obs_xyz   = evalin("base", "OBSTACLES_XYZ");
obs_rh    = evalin("base", "OBSTACLES_RH");
try, obs_class = evalin("base", "OBSTACLES_CLASS"); catch, obs_class = ones(size(obs_xyz,1),1); end
imgSize   = evalin("base", "IMG_SIZE");
camIntrin = evalin("base", "CAM_INTRIN");
camW = imgSize(1); camH = imgSize(2);

Nt   = numel(t_vec);
Nobs = size(obs_xyz, 1);

% 6 calibration cases — added 2 bisection midpoints to verify MARGINAL
% band emerges naturally between Baseline (PASS) and Mid stress (FAIL).
cases = struct(...
    "name", {"Sterile", "Baseline", "Bisect 50%", "Bisect 65%", "Mid stress", "Heavy stress"}, ...
    "fog",  {0,         10,         25,           29.5,         40,           70}, ...
    "ill",  {12000,     10000,      7500,         6750,         5000,         1500}, ...
    "noi",  {0.0,       0.03,       0.115,        0.140,        0.20,         0.40});

fprintf('\n[diag] Nt = %d frames, Nobs = %d intruders (classes: %s)\n\n', ...
    Nt, Nobs, mat2str(obs_class(:)'));

for c = 1:numel(cases)
    fog = cases(c).fog; ill = cases(c).ill; noi = cases(c).noi;
    fprintf('---- Case %d: %s (fog=%.0f%%, illum=%.0flx, noise=%.2f) ----\n', ...
        c, cases(c).name, fog, ill, noi);

    detEvents = zeros(0, 2);
    totalGt = 0;
    nFrames = 0;
    score_log = nan(Nt, Nobs);   % all scores for stats

    for ii = 1:Nt
        uav = uav_xyz(ii, :);
        gtFrame = squeeze(gtBB(ii, :, :));
        if Nobs == 1, gtFrame = reshape(gtFrame, 1, 4); end
        if size(gtFrame, 2) ~= 4
            gtFrame = reshape(gtFrame, [], 4);
        end

        img = render_camera_image(uav, obs_xyz, obs_rh, fog, ill, noi, camIntrin, [camW camH]);
        [scores, detBB] = image_detector(img, gtFrame);
        score_log(ii, :) = scores(:)';

        ngt = 0; ndet = 0; ntp = 0;
        for k = 1:Nobs
            gt = gtFrame(k, :);
            dt = detBB(k, :);
            sc = scores(k);
            gtPresent  = any(gt ~= 0);
            detPresent = (sc > 0.20) && any(dt ~= 0);
            if gtPresent, ngt = ngt + 1; end
            if detPresent
                ndet = ndet + 1;
                tpFlag = 0;
                if gtPresent && bbox_iou(dt, gt) >= 0.5
                    tpFlag = 1;
                    ntp = ntp + 1;
                end
                detEvents(end+1, :) = [sc, tpFlag]; %#ok<AGROW>
            end
        end
        totalGt = totalGt + ngt;
        nFrames = nFrames + 1;
    end

    % Compute mAP@0.5 (VOC envelope, mirrors dashboard's computeRunMetric)
    if totalGt > 0 && ~isempty(detEvents)
        [~, ord] = sort(detEvents(:,1), "descend");
        ev = detEvents(ord, :);
        cumTp = cumsum(ev(:,2));
        cumFp = cumsum(1 - ev(:,2));
        precision = cumTp ./ max(cumTp + cumFp, eps);
        recall    = cumTp / totalGt;
        mrec = [0; recall; 1];
        mpre = [0; precision; 0];
        for i = numel(mpre)-1:-1:1
            mpre(i) = max(mpre(i), mpre(i+1));
        end
        idx = find(mrec(2:end) ~= mrec(1:end-1));
        mAP = sum((mrec(idx+1) - mrec(idx)) .* mpre(idx+1));
    else
        mAP = 0;
    end

    % Per-class score stats
    person_idx  = obs_class == 1;
    vehicle_idx = obs_class == 2;
    psc = score_log(:, person_idx);   psc = psc(:);
    vsc = score_log(:, vehicle_idx);  vsc = vsc(:);

    nDetEvents = size(detEvents, 1);
    nTP = sum(detEvents(:,2));
    nFP = nDetEvents - nTP;

    % Only count scores where intruder is actually visible (GT bbox nonzero)
    gt_vis = gtBB(:, :, :);
    vis_mask = any(gt_vis ~= 0, 3);
    psc_vis = score_log(:, person_idx);   psc_vis = psc_vis(vis_mask(:, person_idx));
    vsc_vis = score_log(:, vehicle_idx);  vsc_vis = vsc_vis(vis_mask(:, vehicle_idx));
    fprintf('  Score stats — person (visible) : n=%d, mean=%.3f, p50=%.3f, p90=%.3f, frac>0.20=%.2f\n', ...
        numel(psc_vis), mean(psc_vis), median(psc_vis), prctile(psc_vis, 90), ...
        mean(psc_vis > 0.20));
    fprintf('  Score stats — vehicle (visible): n=%d, mean=%.3f, p50=%.3f, p90=%.3f, frac>0.20=%.2f\n', ...
        numel(vsc_vis), mean(vsc_vis), median(vsc_vis), prctile(vsc_vis, 90), ...
        mean(vsc_vis > 0.20));
    fprintf('  Events: totalGt=%d, det_events=%d, TP=%d, FP=%d\n', ...
        totalGt, nDetEvents, nTP, nFP);
    fprintf('  >>> mAP@0.5 = %.4f <<<\n\n', mAP);
end

fprintf('========================================================\n');
fprintf('  Diagnosis complete.\n');
fprintf('========================================================\n');
end

% ─────────────────────────────────────────────────────────────────────────
% Helpers (copied from mountain_uav_dashboard.m)
% ─────────────────────────────────────────────────────────────────────────

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
try
    raw = simOut.get(name);
catch
    raw = [];
end
if isempty(raw)
    try, raw = evalin("base", name); catch, raw = []; end
end
if isempty(raw)
    error("Could not find logged signal: %s", name);
end
if isstruct(raw) && isfield(raw, "time") && isfield(raw, "signals")
    s.time = raw.time;
    s.values = raw.signals.values;
else
    error("Unexpected signal format for %s", name);
end
end

function iou = bbox_iou(a, b)
ax1 = a(1); ay1 = a(2); ax2 = a(1)+a(3); ay2 = a(2)+a(4);
bx1 = b(1); by1 = b(2); bx2 = b(1)+b(3); by2 = b(2)+b(4);
ix1 = max(ax1, bx1); iy1 = max(ay1, by1);
ix2 = min(ax2, bx2); iy2 = min(ay2, by2);
iw = max(0, ix2-ix1); ih = max(0, iy2-iy1);
inter = iw*ih;
ua = max(0,a(3))*max(0,a(4)) + max(0,b(3))*max(0,b(4)) - inter;
if ua <= 0, iou = 0; else, iou = inter/ua; end
end
