function mountain_uav_dashboard(simOut)
% mountain_uav_dashboard
% -------------------------------------------------------------------------
% Interactive App Designer-style dashboard for the mountain UAV scenario.
%
% Plays back the pre-simulated UAV trajectory while LIVE-rendering the
% synthetic camera image and re-running the detector each frame. Fog /
% illumination / noise sliders are counterfactual interventions: dragging
% them re-renders the current frame instantly and the detection scores
% update on the fly — a visceral "what-if" for the XAI verifier story.
%
% Usage:
%   mountain_uav_dashboard()           % runs sim once, then opens dashboard
%   mountain_uav_dashboard(simOut)     % reuse an existing simOut
%
% The pre-simulated trajectory + GT bboxes are reused across all weather
% perturbations because they are geometric (UAV pose × target pose); the
% rendered image and detector output are recomputed each frame from the
% current slider values.
% -------------------------------------------------------------------------

if nargin < 1 || isempty(simOut)
    fprintf("[DASHBOARD] No simOut provided — running scenario simulation...\n");
    simOut = run_sim_for_dashboard();
end

% --- Pre-simulated time-varying signals (geometry only) ---
[t_vec, uav_xyz] = read_log_vec(simOut, "uav_xyz_log");
[~,     gtBB]    = read_log_3d (simOut, "gt_bboxes_log");

% --- Static scene from base workspace ---
Xg  = evalin("base", "TERRAIN_X");
Yg  = evalin("base", "TERRAIN_Y");
Zg  = evalin("base", "TERRAIN_Z");
obs_xyz = evalin("base", "OBSTACLES_XYZ");
obs_rh  = evalin("base", "OBSTACLES_RH");
try
    obs_class = evalin("base", "OBSTACLES_CLASS");
catch
    obs_class = ones(size(obs_xyz, 1), 1);
end
try
    SCENERY = evalin("base", "SCENERY_OBJECTS");
catch
    SCENERY = zeros(0, 5);
end
imgSize   = evalin("base", "IMG_SIZE");
camIntrin = evalin("base", "CAM_INTRIN");
camW = imgSize(1); camH = imgSize(2);

fog0   = evalin("base", "FOG_DENSITY_PERCENT");
illum0 = evalin("base", "ILLUMINATION_LUX");
noise0 = evalin("base", "CAMERA_NOISE_LEVEL");

Nt   = numel(t_vec);
Nobs = size(obs_xyz, 1);

INTRUDER_LABEL = ["Person", "Vehicle"];
INTRUDER_COLOR = [0.20 0.50 0.95;
                  0.95 0.55 0.10];

% =========================================================================
% Build UI
% =========================================================================
fig = uifigure("Name", "Mountain UAV — Counterfactual XAI Dashboard", ...
    "Position", [60 60 1500 880], "Color", [0.97 0.97 0.99]);

main = uigridlayout(fig, [4, 2], ...
    "RowHeight",   {38, '1x', 110, 100}, ...
    "ColumnWidth", {'1x', '1x'}, ...
    "RowSpacing", 6, "ColumnSpacing", 8, ...
    "Padding", [10 10 10 10]);

% Header banner
hdr = uilabel(main, ...
    "Text", "  Initialising...", ...
    "FontSize", 16, "FontWeight", "bold", ...
    "BackgroundColor", [0.10 0.20 0.40], "FontColor", "w", ...
    "VerticalAlignment", "center");
hdr.Layout.Row = 1; hdr.Layout.Column = [1 2];

% Left axes — 3D scene
ax3 = uiaxes(main);
ax3.Layout.Row = 2; ax3.Layout.Column = 1;
title(ax3, "3D Scene  —  UAV surveillance flight");

% Right axes — camera image
ax2 = uiaxes(main);
ax2.Layout.Row = 2; ax2.Layout.Column = 2;
title(ax2, "EO Camera  +  detection (live re-rendered)");

% Counterfactual control panel
ctrlPanel = uipanel(main, ...
    "Title", "Counterfactual case  (set parameters, then press ▶ Run)", ...
    "BackgroundColor", [0.95 0.95 0.97], "FontWeight", "bold");
ctrlPanel.Layout.Row = 3; ctrlPanel.Layout.Column = [1 2];
ctrl = uigridlayout(ctrlPanel, [1, 3], ...
    "ColumnWidth", {'1x', '1x', '1x'}, ...
    "Padding", [10 6 10 6], "ColumnSpacing", 14);

[fogSld, fogLbl] = mkSlider(ctrl, "Fog density",     0,    100,  fog0,   "%%");
[illSld, illLbl] = mkSlider(ctrl, "Illumination",  100,  15000, illum0, "lx");
[noiSld, noiLbl] = mkSlider(ctrl, "Camera noise",    0,    1.0,  noise0, "");

% Playback control panel
pbPanel = uipanel(main, "BackgroundColor", [0.96 0.96 0.98]);
pbPanel.Layout.Row = 4; pbPanel.Layout.Column = [1 2];
pb = uigridlayout(pbPanel, [2, 5], ...
    "RowHeight", {32, 50}, ...
    "ColumnWidth", {180, 90, 90, 110, '1x'}, ...
    "Padding", [10 4 10 4], "RowSpacing", 4, "ColumnSpacing", 10);

btnRun   = uibutton(pb, "Text", "▶ Run Counterfactual", ...
    "BackgroundColor", [0.20 0.65 0.30], "FontColor", "w", "FontWeight", "bold");
btnRun.Layout.Row = 1; btnRun.Layout.Column = 1;

btnStop  = uibutton(pb, "Text", "⏹ Stop", ...
    "BackgroundColor", [0.85 0.30 0.20], "FontColor", "w", "Enable", "off");
btnStop.Layout.Row = 1; btnStop.Layout.Column = 2;

btnReset = uibutton(pb, "Text", "⟲ Reset", ...
    "BackgroundColor", [0.60 0.60 0.65], "FontColor", "w");
btnReset.Layout.Row = 1; btnReset.Layout.Column = 3;

autoToggle = uicheckbox(pb, "Text", "Auto-loop (DSPy)", ...
    "Value", false, "FontWeight", "bold", ...
    "Tooltip", "After each run, automatically iterate toward the failure boundary using DSPy / boundary-search policy");
autoToggle.Layout.Row = 1; autoToggle.Layout.Column = 4;

iterLbl = uilabel(pb, ...
    "Text", "Iter: 0 / 10  —  manual mode", ...
    "FontWeight", "bold", "HorizontalAlignment", "left");
iterLbl.Layout.Row = 1; iterLbl.Layout.Column = 5;

frameSld = uislider(pb, "Limits", [1 max(2,Nt)], "Value", 1, ...
    "MajorTicks", round(linspace(1, max(2,Nt), 5)));
frameSld.Layout.Row = 2; frameSld.Layout.Column = [1 4];

frameLbl = uilabel(pb, ...
    "Text", sprintf("Frame: 1 / %d", Nt), ...
    "HorizontalAlignment", "right", "FontWeight", "bold");
frameLbl.Layout.Row = 2; frameLbl.Layout.Column = 5;

% =========================================================================
% Static 3D content
% =========================================================================
surf(ax3, Xg, Yg, Zg, "EdgeColor", "none", "FaceAlpha", 0.85);
shading(ax3, "interp");
colormap(ax3, terrain_colormap());
hold(ax3, "on");

for k = 1:size(SCENERY, 1)
    sx = SCENERY(k,1); sy = SCENERY(k,2); sz = SCENERY(k,3);
    sr = SCENERY(k,4); st = SCENERY(k,5);
    if st == 1
        draw_scenery_tree(ax3, [sx sy sz], sr);
    else
        draw_scenery_rock(ax3, [sx sy sz], sr);
    end
end
for k = 1:Nobs
    cls = obs_class(k);
    col = INTRUDER_COLOR(cls, :);
    draw_intruder(ax3, obs_xyz(k,:), obs_rh(k,1), obs_rh(k,2), cls, col);
    text(ax3, obs_xyz(k,1), obs_xyz(k,2), obs_xyz(k,3) + obs_rh(k,2) + 1.0, ...
        sprintf("%s %d", INTRUDER_LABEL(cls), k), ...
        "Color", col, "FontWeight", "bold", "FontSize", 9, ...
        "HorizontalAlignment", "center");
end
plot3(ax3, uav_xyz(:,1), uav_xyz(:,2), uav_xyz(:,3), "b-", "LineWidth", 1.4);
uavMarker = plot3(ax3, uav_xyz(1,1), uav_xyz(1,2), uav_xyz(1,3), ...
    "Marker", "diamond", "MarkerSize", 14, ...
    "MarkerFaceColor", [0.10 0.45 0.95], ...
    "MarkerEdgeColor", "k", "LineStyle", "none");
camLines = gobjects(4, 1);
for ii = 1:4
    camLines(ii) = plot3(ax3, [0 0],[0 0],[0 0], ...
        "Color", [0.9 0.2 0.2], "LineWidth", 1.2);
end
xlabel(ax3, "X (m)"); ylabel(ax3, "Y (m)"); zlabel(ax3, "Z (m)");
grid(ax3, "on"); axis(ax3, "equal"); view(ax3, 35, 30);
xlim(ax3, [min(Xg(:)) max(Xg(:))]);
ylim(ax3, [min(Yg(:)) max(Yg(:))]);
zlim(ax3, [0 max(Zg(:))+30]);

% =========================================================================
% Camera image init
% =========================================================================
img0 = render_camera_image(uav_xyz(1,:), obs_xyz, obs_rh, ...
    fog0, illum0, noise0, camIntrin, [camW camH]);
imHandle = imagesc(ax2, img0);
set(ax2, "YDir", "reverse");
hold(ax2, "on");
xlim(ax2, [0 camW]); ylim(ax2, [0 camH]);
xlabel(ax2, "u (px)"); ylabel(ax2, "v (px)");
axis(ax2, "image");

% =========================================================================
% State + callbacks
% =========================================================================
state = struct();
state.frameIdx      = 1;
state.mode          = "idle";          % "idle" | "running" | "cooldown"
state.runStats      = [];
state.history       = struct( ...      % counterfactual run history
    "iter", {}, "fog", {}, "ill", {}, "noi", {}, ...
    "f1", {}, "passed", {}, "ngt", {}, "ndet", {}, "ntp", {}, ...
    "mode", {}, "analysis", {});
state.iterCount     = 0;
state.maxIter       = 10;
state.cooldownTimer = [];
state.passF1Thresh  = 0.85;            % requirement threshold (matches requirements_eval)

fogSld.ValueChangedFcn   = @(~,~) renderFrame();
fogSld.ValueChangingFcn  = @(~,e) onSlideLive("fog", e);
illSld.ValueChangedFcn   = @(~,~) renderFrame();
illSld.ValueChangingFcn  = @(~,e) onSlideLive("ill", e);
noiSld.ValueChangedFcn   = @(~,~) renderFrame();
noiSld.ValueChangingFcn  = @(~,e) onSlideLive("noi", e);

btnRun.ButtonPushedFcn   = @(~,~) startRun();
btnStop.ButtonPushedFcn  = @(~,~) stopRun();
btnReset.ButtonPushedFcn = @(~,~) doReset();
autoToggle.ValueChangedFcn = @(s,~) onAutoToggle(s);

frameSld.ValueChangedFcn  = @(s,~) onFrameSliderRelease(s);
frameSld.ValueChangingFcn = @(~,e) onFrameSliderLive(e);

% Timer drives playback during a counterfactual run
tmr = timer("ExecutionMode", "fixedRate", "Period", 0.08, ...
    "BusyMode", "drop", ...
    "TimerFcn", @(~,~) onTick());
fig.CloseRequestFcn = @(~,~) cleanup();

% First paint of the initial frame so the user can see what the camera
% currently sees with the default weather; do NOT start auto-playback —
% the run is initiated by the ▶ Run Counterfactual button.
hdr.BackgroundColor = [0.10 0.20 0.40];
hdr.Text = "  Idle  —  set counterfactual parameters, then press  ▶ Run Counterfactual";
renderFrame();
start(tmr);

% =========================================================================
% Nested callbacks (share workspace with the parent function)
% =========================================================================
    function onTick()
        if ~isvalid(fig), return; end
        if state.mode ~= "running", return; end
        state.frameIdx = state.frameIdx + 1;
        if state.frameIdx > Nt
            finalizeRun();
            return;
        end
        frameSld.Value = state.frameIdx;
        frameLbl.Text  = sprintf("Frame: %d / %d", state.frameIdx, Nt);
        renderFrame();
    end

    function startRun()
        if state.mode == "running", return; end
        cancelCooldown();
        state.mode     = "running";
        state.frameIdx = 1;
        state.runStats = struct( ...
            "ngt", 0, "ndet", 0, "ntp", 0, "nFrames", 0, ...
            "fog", fogSld.Value, "ill", illSld.Value, "noi", noiSld.Value);
        setCaseControlsEnabled(false);
        btnRun.Enable  = "off";
        btnStop.Enable = "on";
        frameSld.Value = 1;
        frameLbl.Text  = sprintf("Frame: 1 / %d", Nt);
        if autoToggle.Value
            iterLbl.Text = sprintf("Auto: running iter %d / %d ...", ...
                numel(state.history) + 1, state.maxIter);
        end
        renderFrame();
    end

    function stopRun()
        if state.mode ~= "running", return; end
        finalizeRun();
    end

    function finalizeRun()
        % Append the just-completed run to history (for boundary search)
        rs = state.runStats;
        if ~isempty(rs) && rs.nFrames > 0
            recall = rs.ntp / max(1, rs.ngt);
            prec   = rs.ntp / max(1, rs.ndet);
            f1     = 2 * prec * recall / max(1e-6, prec + recall);
            passed = f1 >= state.passF1Thresh;

            rec = struct( ...
                "iter",     numel(state.history) + 1, ...
                "fog",      rs.fog, "ill", rs.ill, "noi", rs.noi, ...
                "f1",       f1, "passed", passed, ...
                "ngt",      rs.ngt, "ndet", rs.ndet, "ntp", rs.ntp, ...
                "mode",     "manual", "analysis", "");
            state.history(end + 1) = rec;
        end

        state.mode = "idle";
        setCaseControlsEnabled(true);
        btnRun.Enable  = "on";
        btnStop.Enable = "off";
        showRunSummary();

        % Auto-loop: schedule next iteration after a brief pause so the user
        % can read the verdict before sliders animate to the next case.
        if autoToggle.Value && numel(state.history) < state.maxIter
            scheduleNextAutoRun();
        elseif autoToggle.Value
            autoToggle.Value = false;
            iterLbl.Text = sprintf("Auto: done (%d iters)", numel(state.history));
        end
    end

    function onAutoToggle(src)
        if src.Value
            iterLbl.Text = sprintf("Auto: %d / %d iters", numel(state.history), state.maxIter);
            % If currently idle, kick off the loop now using current sliders
            % as the seed case (user-defined starting point).
            if state.mode == "idle"
                startRun();
            end
        else
            iterLbl.Text = sprintf("Iter: %d / %d  —  manual mode", numel(state.history), state.maxIter);
            % Cancel any pending cooldown so a queued auto-step doesn't fire
            cancelCooldown();
            if state.mode == "cooldown"
                state.mode = "idle";
                btnRun.Enable = "on";
                setCaseControlsEnabled(true);
            end
        end
    end

    function scheduleNextAutoRun()
        % Compute next case from history using boundary-search policy
        [nextEnv, modeStr, analysisStr] = decideNextCase(state.history);

        % Animate sliders to new values
        applyEnvToSliders(nextEnv);

        % Re-render preview at frame 1 with new params so user sees the case
        state.frameIdx = 1;
        frameSld.Value = 1;
        frameLbl.Text  = sprintf("Frame: 1 / %d", Nt);
        renderFrame();

        % Header + iter label show the suggested next case
        hdr.BackgroundColor = [0.20 0.30 0.65];
        hdr.Text = sprintf( ...
            "  → next case (DSPy / %s):  fog=%.0f%%  illum=%.0flx  noise=%.2f   |   %s", ...
            modeStr, nextEnv.fog, nextEnv.ill, nextEnv.noi, analysisStr);
        iterLbl.Text = sprintf("Auto: %d / %d  —  starting next iter...", ...
            numel(state.history), state.maxIter);

        % Lock controls during cooldown so user can't interrupt mid-decision
        state.mode = "cooldown";
        btnRun.Enable = "off";
        setCaseControlsEnabled(false);

        % One-shot timer fires after pause and starts the next run
        cancelCooldown();
        state.cooldownTimer = timer( ...
            "StartDelay", 1.8, ...
            "TimerFcn", @(t,~) onCooldownFire(t), ...
            "ExecutionMode", "singleShot");
        start(state.cooldownTimer);
    end

    function onCooldownFire(t)
        try, delete(t); catch, end
        state.cooldownTimer = [];
        if ~isvalid(fig), return; end
        if ~autoToggle.Value, return; end           % toggled off mid-cooldown
        state.mode = "idle";                         % unlock briefly
        startRun();
    end

    function cancelCooldown()
        if ~isempty(state.cooldownTimer) && isvalid(state.cooldownTimer)
            try, stop(state.cooldownTimer); catch, end
            try, delete(state.cooldownTimer); catch, end
        end
        state.cooldownTimer = [];
    end

    function applyEnvToSliders(env)
        fogV = max(fogSld.Limits(1), min(fogSld.Limits(2), env.fog));
        illV = max(illSld.Limits(1), min(illSld.Limits(2), env.ill));
        noiV = max(noiSld.Limits(1), min(noiSld.Limits(2), env.noi));
        fogSld.Value = fogV;  fogLbl.Text = sprintf("Fog density: %.0f %%", fogV);
        illSld.Value = illV;  illLbl.Text = sprintf("Illumination: %.0f lx", illV);
        noiSld.Value = noiV;  noiLbl.Text = sprintf("Camera noise: %.2f", noiV);
    end

    function [nextEnv, modeStr, analysisStr] = decideNextCase(history)
        % Try DSPy / Python orchestrator first; fall back to the same
        % deterministic boundary policy in MATLAB if Python isn't available.
        try
            histJson = jsonencode(history);
            pyResult = py.dashboard_step.next_case_from_history(histJson, char(pwd));
            d = struct(pyResult);
            nextEnv = struct( ...
                "fog", double(d.fog_density_percent), ...
                "ill", double(d.illumination_lux), ...
                "noi", double(d.camera_noise_level));
            % py.str → char → string: explicit double conversion is the
            % safe path across MATLAB releases (string(py.str) works in
            % R2020a+ but breaks silently in some older builds).
            modeStr     = string(char(d.mode));
            analysisStr = string(char(d.analysis));
        catch ME
            % Python/DSPy unavailable — use MATLAB-native boundary search
            [nextEnv, modeStr, analysisStr] = decideNextCaseRule(history);
            if numel(state.history) <= 1
                fprintf("[DASHBOARD] DSPy not available (%s) — using MATLAB rule policy.\n", ME.message);
            end
        end
    end

    function [nextEnv, modeStr, analysisStr] = decideNextCaseRule(history)
        if isempty(history)
            nextEnv = struct("fog", 30, "ill", 4000, "noi", 0.1);
            modeStr = "seed"; analysisStr = "Initial seed";
            return;
        end
        passEnv = []; failEnv = [];
        for k = 1:numel(history)
            h = history(k);
            e = struct("fog", h.fog, "ill", h.ill, "noi", h.noi);
            if h.passed, passEnv = e; else, failEnv = e; end
        end
        last = history(end);
        lastEnv = struct("fog", last.fog, "ill", last.ill, "noi", last.noi);
        if last.passed && ~isempty(failEnv)
            nextEnv = bisectEnv(lastEnv, failEnv, 0.65);
            modeStr = "boundary_push";
            analysisStr = sprintf("PASS → bisect 65%% toward FAIL anchor (fog=%.0f, illum=%.0f, noise=%.2f)", ...
                failEnv.fog, failEnv.ill, failEnv.noi);
        elseif ~last.passed && ~isempty(passEnv)
            nextEnv = bisectEnv(lastEnv, passEnv, 0.65);
            modeStr = "boundary_recover";
            analysisStr = sprintf("FAIL → bisect 65%% toward PASS anchor (fog=%.0f, illum=%.0f, noise=%.2f)", ...
                passEnv.fog, passEnv.ill, passEnv.noi);
        elseif last.passed
            nextEnv = pushEnv(lastEnv);
            modeStr = "rule_push";
            analysisStr = "No FAIL anchor yet — push harder (rule)";
        else
            nextEnv = relaxEnv(lastEnv);
            modeStr = "rule_relax";
            analysisStr = "No PASS anchor yet — relax toward baseline (rule)";
        end
    end

    function doReset()
        if state.mode == "running"
            finalizeRun();
        end
        state.frameIdx = 1;
        frameSld.Value = 1;
        frameLbl.Text  = sprintf("Frame: 1 / %d", Nt);
        renderFrame();
        hdr.BackgroundColor = [0.10 0.20 0.40];
        hdr.Text = "  Idle  —  set counterfactual parameters, then press  ▶ Run Counterfactual";
    end

    function setCaseControlsEnabled(en)
        s = "off"; if en, s = "on"; end
        fogSld.Enable = s;
        illSld.Enable = s;
        noiSld.Enable = s;
    end

    function showRunSummary()
        rs = state.runStats;
        if isempty(rs) || rs.nFrames == 0
            hdr.BackgroundColor = [0.40 0.40 0.45];
            hdr.Text = "  Run stopped before any frame ran.";
            return;
        end
        recall = rs.ntp / max(1, rs.ngt);
        prec   = rs.ntp / max(1, rs.ndet);
        f1     = 2 * prec * recall / max(1e-6, prec + recall);
        verdict = "PASS"; color = [0.10 0.45 0.20];
        if f1 < 0.5
            verdict = "FAIL";     color = [0.55 0.10 0.15];
        elseif f1 < 0.85
            verdict = "MARGINAL"; color = [0.55 0.40 0.05];
        end
        hdr.BackgroundColor = color;
        hdr.Text = sprintf( ...
            "  [ %s ]   case: fog=%.0f%%  illum=%.0flx  noise=%.2f   |   GT %d  Det %d  TP %d   |   Recall %.2f  Prec %.2f   F1 %.2f", ...
            verdict, rs.fog, rs.ill, rs.noi, ...
            rs.ngt, rs.ndet, rs.ntp, recall, prec, f1);
    end

    function onFrameSliderRelease(src)
        if state.mode == "running"
            frameSld.Value = state.frameIdx;       % snap back — locked during run
            return;
        end
        state.frameIdx = max(1, min(Nt, round(src.Value)));
        frameLbl.Text  = sprintf("Frame: %d / %d", state.frameIdx, Nt);
        renderFrame();
    end

    function onFrameSliderLive(evt)
        if state.mode == "running", return; end
        state.frameIdx = max(1, min(Nt, round(evt.Value)));
        frameLbl.Text  = sprintf("Frame: %d / %d", state.frameIdx, Nt);
        renderFrame();
    end

    function onSlideLive(which, evt)
        if state.mode == "running", return; end    % case is locked during a run
        v = evt.Value;
        switch which
            case "fog", fogLbl.Text = sprintf("Fog density: %.0f %%", v);
            case "ill", illLbl.Text = sprintf("Illumination: %.0f lx", v);
            case "noi", noiLbl.Text = sprintf("Camera noise: %.2f", v);
        end
        renderFrame();
    end

    function renderFrame()
        if ~isvalid(fig), return; end
        ii = state.frameIdx;
        uav = uav_xyz(ii, :);

        fog = fogSld.Value;
        ill = illSld.Value;
        noi = noiSld.Value;

        % Live-update the slider value labels (covers ValueChanged path too)
        fogLbl.Text = sprintf("Fog density: %.0f %%", fog);
        illLbl.Text = sprintf("Illumination: %.0f lx", ill);
        noiLbl.Text = sprintf("Camera noise: %.2f", noi);

        % --- Update 3D marker + frustum ---
        set(uavMarker, "XData", uav(1), "YData", uav(2), "ZData", uav(3));
        pitch_rad = camIntrin(5) * pi / 180;
        fcorn = camera_frustum_corners(uav, pitch_rad, ...
            camIntrin(1), camIntrin(2), camIntrin(3), camIntrin(4), ...
            camW, camH, 35);
        for ll = 1:4
            set(camLines(ll), ...
                "XData", [uav(1) fcorn(ll,1)], ...
                "YData", [uav(2) fcorn(ll,2)], ...
                "ZData", [uav(3) fcorn(ll,3)]);
        end

        % --- Re-render synthetic camera image with CURRENT weather ---
        img = render_camera_image(uav, obs_xyz, obs_rh, ...
            fog, ill, noi, camIntrin, [camW camH]);
        set(imHandle, "CData", img);

        % --- Re-run pixel-level detector on the new image ---
        gtFrame = squeeze(gtBB(ii, :, :));
        if Nobs == 1, gtFrame = reshape(gtFrame, 1, 4); end
        if size(gtFrame, 2) ~= 4
            gtFrame = reshape(gtFrame, [], 4);
        end
        [scores, detBB_frame] = image_detector(img, gtFrame);

        % --- Overlay GT (green dashed) + detected (red solid) bboxes ---
        delete(findobj(ax2, "Tag", "bbox_overlay"));
        ngt = 0; ndet = 0; ntp = 0;
        for k = 1:Nobs
            gt = gtFrame(k, :);
            dt = detBB_frame(k, :);
            sc = scores(k);
            cls_lbl = INTRUDER_LABEL(obs_class(k));
            if any(gt ~= 0)
                ngt = ngt + 1;
                rectangle("Parent", ax2, "Position", clamp_box(gt, camW, camH), ...
                    "EdgeColor", [0.10 0.85 0.20], "LineStyle", "--", ...
                    "LineWidth", 1.5, "Tag", "bbox_overlay");
                text(ax2, gt(1), max(8, gt(2) - 6), sprintf("%s GT", cls_lbl), ...
                    "Color", [0.10 0.85 0.20], "FontWeight", "bold", "FontSize", 8, ...
                    "Tag", "bbox_overlay");
            end
            if sc > 0.30 && any(dt ~= 0)
                ndet = ndet + 1;
                if any(gt ~= 0) && bbox_iou(dt, gt) >= 0.5
                    ntp = ntp + 1;
                end
                rectangle("Parent", ax2, "Position", clamp_box(dt, camW, camH), ...
                    "EdgeColor", [0.95 0.20 0.20], "LineWidth", 1.7, ...
                    "Tag", "bbox_overlay");
                text(ax2, dt(1), min(camH - 8, dt(2) + dt(4) + 12), ...
                    sprintf("%s %.2f", cls_lbl, sc), ...
                    "Color", [0.95 0.20 0.20], "FontWeight", "bold", "FontSize", 8, ...
                    "Tag", "bbox_overlay");
            end
        end

        % --- Header & stats — only updated DURING a counterfactual run ---
        % In idle mode the header keeps the previous summary (or the initial
        % "Idle..." prompt) so the user can scrub frames / drag sliders to
        % preview without the header flickering.
        if state.mode == "running"
            state.runStats.ngt     = state.runStats.ngt     + ngt;
            state.runStats.ndet    = state.runStats.ndet    + ndet;
            state.runStats.ntp     = state.runStats.ntp     + ntp;
            state.runStats.nFrames = state.runStats.nFrames + 1;

            rs = state.runStats;
            cumRecall = rs.ntp / max(1, rs.ngt);
            cumPrec   = rs.ntp / max(1, rs.ndet);
            cumF1     = 2 * cumPrec * cumRecall / max(1e-6, cumPrec + cumRecall);

            hdr.BackgroundColor = [0.10 0.30 0.55];
            hdr.Text = sprintf( ...
                "  ▶ RUNNING   case: fog=%.0f%%  illum=%.0flx  noise=%.2f   |   Frame %d/%d  t=%.2fs   |   cum F1 %.2f", ...
                fog, ill, noi, ii, Nt, t_vec(ii), cumF1);
        end

        frameLbl.Text = sprintf("Frame: %d / %d", ii, Nt);
        title(ax2, sprintf("EO Camera   t=%.2fs   fog=%.0f%%  illum=%.0flx  noise=%.2f", ...
            t_vec(ii), fog, ill, noi));
    end

    function cleanup()
        cancelCooldown();
        try
            stop(tmr); delete(tmr);
        catch
        end
        delete(fig);
    end
end


% =========================================================================
% Helpers (file-local — duplicated from mountain_visualizer for self-containment)
% =========================================================================

function simOut = run_sim_for_dashboard()
mdl = "mountain_uav_model";

% Seed base workspace so all nine Constant blocks
% (C_FOG, C_ILLUM, C_NOISE, C_OBS_XYZ, C_OBS_RH, C_UAV_X0, C_UAV_V,
% C_CAM, C_IMG) have valid variables. Defaults match scenario_iter_001.
needsSeed = ~all_base_vars_present({"TERRAIN_X","TERRAIN_Y","TERRAIN_Z", ...
    "OBSTACLES_XYZ","OBSTACLES_RH","UAV_X0_VEC","UAV_V_VEC", ...
    "CAM_INTRIN","IMG_SIZE","FOG_DENSITY_PERCENT", ...
    "ILLUMINATION_LUX","CAMERA_NOISE_LEVEL"});
if needsSeed
    fprintf("[DASHBOARD] Seeding base workspace via init_uav_workspace()...\n");
    init_uav_workspace();
end

if ~bdIsLoaded(mdl)
    if ~isfile(mdl + ".slx")
        build_mountain_uav_model(false);
    else
        load_system(mdl);
    end
end
try
    set_param(mdl, "SimulationCommand", "Update");
catch
end
simOut = sim(mdl);
end


function tf = all_base_vars_present(names)
tf = true;
for i = 1:numel(names)
    try
        evalin("base", char(names{i}));
    catch
        tf = false; return;
    end
end
end


function [sld, lbl] = mkSlider(parent, name, lo, hi, val, unit)
% Composite: vertical [label-with-value, slider]
panel = uipanel(parent, "BorderType", "none", "BackgroundColor", [0.95 0.95 0.97]);
g = uigridlayout(panel, [2, 1], "RowHeight", {22, '1x'}, ...
    "Padding", [4 4 4 4], "RowSpacing", 2);
if strcmp(unit, "%%")
    lbl = uilabel(g, "Text", sprintf("%s: %.0f %%", name, val), ...
        "FontWeight", "bold", "FontSize", 12);
elseif unit == ""
    lbl = uilabel(g, "Text", sprintf("%s: %.2f", name, val), ...
        "FontWeight", "bold", "FontSize", 12);
else
    lbl = uilabel(g, "Text", sprintf("%s: %.0f %s", name, val, unit), ...
        "FontWeight", "bold", "FontSize", 12);
end
sld = uislider(g, "Limits", [lo hi], "Value", max(lo, min(hi, val)), ...
    "MajorTicks", linspace(lo, hi, 5));
end


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
    vals = permute(v, [3 1 2]);     % Nt x N x M
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


function box = clamp_box(b, w, h)
x  = max(0, b(1));
y  = max(0, b(2));
ww = max(2, min(b(3), w - x));
hh = max(2, min(b(4), h - y));
box = [x, y, ww, hh];
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


function draw_scenery_tree(ax, base, r)
trunkR = max(0.10, r * 0.25);
trunkH = max(1.5, r * 1.8);
canopyR = r;
canopyH = max(2.5, r * 3.2);
[xc, yc, zc] = cylinder(trunkR, 8);
zc = zc * trunkH; xc = xc + base(1); yc = yc + base(2); zc = zc + base(3);
surf(ax, xc, yc, zc, "EdgeColor", "none", "FaceColor", [0.40 0.25 0.12]);
[sx, sy, sz] = sphere(10);
sx = sx * canopyR + base(1);
sy = sy * canopyR + base(2);
sz = sz * (canopyH * 0.5) + base(3) + trunkH;
surf(ax, sx, sy, sz, "EdgeColor", "none", "FaceColor", [0.10 0.40 0.18], "FaceAlpha", 0.95);
end


function draw_scenery_rock(ax, base, r)
[sx, sy, sz] = sphere(8);
sx = sx * r + base(1);
sy = sy * r + base(2);
sz = sz * (r * 0.5) + base(3) + r * 0.3;
surf(ax, sx, sy, sz, "EdgeColor", "none", "FaceColor", [0.50 0.45 0.40], "FaceAlpha", 0.95);
end


function draw_intruder(ax, base, r, h, cls, color)
if cls == 2
    [xc, yc, zc] = cylinder(r, 16);
    zc = zc * h; xc = xc + base(1); yc = yc + base(2); zc = zc + base(3);
    surf(ax, xc, yc, zc, "EdgeColor", [0.30 0.30 0.30], ...
        "FaceColor", color, "FaceAlpha", 0.95);
else
    [xc, yc, zc] = cylinder(r, 12);
    zc = zc * h; xc = xc + base(1); yc = yc + base(2); zc = zc + base(3);
    surf(ax, xc, yc, zc, "EdgeColor", "none", ...
        "FaceColor", color, "FaceAlpha", 0.95);
    [sx, sy, sz] = sphere(8);
    sx = sx * (r * 1.2) + base(1);
    sy = sy * (r * 1.2) + base(2);
    sz = sz * (r * 1.0) + base(3) + h;
    surf(ax, sx, sy, sz, "EdgeColor", "none", ...
        "FaceColor", color * 0.85, "FaceAlpha", 0.95);
end
end


function corners = camera_frustum_corners(uav, pitch, fx, fy, cx, cy, w, h, depth)
sp = sin(pitch); cp = cos(pitch);
pix = [0 0; w 0; w h; 0 h];
corners = zeros(4, 3);
for k = 1:4
    u = pix(k,1); v = pix(k,2);
    cam_x = (u - cx) / fx;
    cam_y = (v - cy) / fy;
    cam_z = 1.0;
    s = depth / cam_z;
    cam_x = cam_x * s; cam_y = cam_y * s; cam_z = cam_z * s;
    world_off = cam_x * [0 1 0] + cam_y * [-sp 0 -cp] + cam_z * [cp 0 -sp];
    corners(k, :) = uav + world_off;
end
end


% =========================================================================
% Boundary-search policy helpers (mirrors dspy_pipeline/orchestrator.py
% _bisect_between / _rule_push / _rule_relax). Used as MATLAB-native fallback
% when Python DSPy isn't reachable from this MATLAB session.
% =========================================================================
function out = bisectEnv(a, b, w)
out = struct( ...
    "fog", round((1 - w) * a.fog + w * b.fog, 2), ...
    "ill", round((1 - w) * a.ill + w * b.ill, 1), ...
    "noi", round((1 - w) * a.noi + w * b.noi, 4));
end


function out = pushEnv(e)
out = struct( ...
    "fog", round(min(100,    e.fog + 30),   2), ...
    "ill", round(max(200,    e.ill * 0.50), 1), ...
    "noi", round(min(0.60,   e.noi + 0.20), 4));
end


function out = relaxEnv(e)
out = struct( ...
    "fog", round(max(0,      e.fog - 25),   2), ...
    "ill", round(min(20000,  e.ill * 1.80), 1), ...
    "noi", round(max(0,      e.noi - 0.18), 4));
end


function cmap = terrain_colormap()
cmap = [
    0.20 0.40 0.10
    0.30 0.55 0.15
    0.50 0.65 0.30
    0.65 0.55 0.35
    0.70 0.55 0.40
    0.80 0.70 0.55
    0.92 0.92 0.92
];
cmap = interp1(linspace(0,1,size(cmap,1)), cmap, linspace(0,1,128));
end
