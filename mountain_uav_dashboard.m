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

% Ensure MATLAB's Python points at the project's .venv (where dspy, shap,
% python-dotenv are installed). Idempotent; safe to call every entry.
setup_pyenv();

% --- Python module bootstrap --------------------------------------------
% Two failure modes have historically broken `py.dashboard_step.<fn>` calls:
%   (1) Python's sys.path doesn't include the project root, so
%       `import dashboard_step` raises ModuleNotFoundError.
%   (2) MATLAB caches the module across runs, so post-edit functions
%       (summarize_mission / save_edge_case) aren't visible on the next call.
% Fix both: insert project root explicitly, then import + reload, then run
% a no-op smoke call so a hard failure surfaces here (not 30 lines later
% inside a try/catch that just logs "LLM unavailable").
projRoot = string(fileparts(mfilename("fullpath")));
try
    if count(py.sys.path, projRoot) == 0
        insert(py.sys.path, int64(0), projRoot);
    end
    pyMod = py.importlib.import_module("dashboard_step");
    pyMod = py.importlib.reload(pyMod);
    % Smoke-test all three entry points the dashboard relies on. The
    % no-op JSON ("[]" or "{}") triggers each function's early-exit path
    % without doing real work, just confirming the symbol resolves.
    pyMod.narrate_edge_case(jsonencode(struct( ...
        "fog", 0, "ill", 8000, "noi", 0, "metric", 1.0, "verdict", "PASS", ...
        "ngt", 0, "ndet", 0, "ntp", 0)));
    pyMod.summarize_mission("[]");
    pyMod.save_edge_case(jsonencode(struct("verdict", "PASS")));
    fprintf("[DASHBOARD] dashboard_step module loaded OK " + ...
            "(narrate_edge_case / summarize_mission / save_edge_case ready)\n");
catch ME
    fprintf("[DASHBOARD] **Python bootstrap FAILED** — LLM features disabled.\n");
    fprintf("    Reason: %s\n", ME.message);
    fprintf("    CWD: %s\n", pwd);
    try, fprintf("    Python exe: %s\n", string(pyenv().Executable)); catch, end
    fprintf("    Fix: cd to the project root (where dashboard_step.py lives) and re-run.\n");
end

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

% Dashboard seed = Korean average operating environment (NOT scenario_iter_001
% defaults). Backed by:
%   fog 10 %    : KMA + Foggy Cityscapes β ≈ 0.001–0.003 (light atmospheric haze
%                 typical for Korean mountain ops year-round)
%   illum 10000 : NASA POWER 37.5°N annual mean noon (mix of clear/partial cloud)
%   noise 0.03  : FLIR/Sony EO sensor floor, ImageNet-C severity 0–1
% Simulink base workspace may hold different values (init_uav_workspace default
% 30/4000/0.1) — those still drive Simulink's geometric simulation; the dashboard
% overrides the *rendered weather + slider initial* with our calibrated demo seed.
fog0   = 10.0;
illum0 = 10000.0;
noise0 = 0.03;
% Keep the base workspace in sync so any downstream tools see the same values
assignin("base", "FOG_DENSITY_PERCENT", fog0);
assignin("base", "ILLUMINATION_LUX",    illum0);
assignin("base", "CAMERA_NOISE_LEVEL",  noise0);

Nt   = numel(t_vec);
Nobs = size(obs_xyz, 1);

INTRUDER_LABEL = ["Person", "Vehicle"];
INTRUDER_COLOR = [0.20 0.50 0.95;
                  0.95 0.55 0.10];

% =========================================================================
% Build UI
% =========================================================================
fig = uifigure("Name", "Mountain UAV — Counterfactual XAI Dashboard", ...
    "Position", [60 20 1500 1160], "Color", [0.97 0.97 0.99]);

% Top mission/business-goal header removed per user request — mission
% metadata is still loaded internally for LLM narration, just not shown.
main = uigridlayout(fig, [5, 2], ...
    "RowHeight",   {40, '1x', 360, 130, 100}, ...
    "ColumnWidth", {'1x', '1x'}, ...
    "RowSpacing", 6, "ColumnSpacing", 8, ...
    "Padding", [10 10 10 10]);

% ---- Row 1: Verdict / status header (the dynamic one) ----
hdr = uilabel(main, ...
    "Text", "  Initialising...", ...
    "FontSize", 15, "FontWeight", "bold", ...
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

%{
% --- DISABLED per user request: counterfactual boundary 3-D scatter ---
ax_boundary = uiaxes(main);
ax_boundary.Layout.Row = 4; ax_boundary.Layout.Column = 1;
title(ax_boundary, "Counterfactual boundary discovery  —  PASS (green) / FAIL (red)");
%}

% Operations-log panel — left half of row 3.
xaiPanel = uipanel(main, ...
    "Title", "Operations log  —  edge-case narrative + case history", ...
    "BackgroundColor", [0.96 0.96 0.99], "FontWeight", "bold");
xaiPanel.Layout.Row = 3; xaiPanel.Layout.Column = 1;

% LLM 임무 종합 요약 — right half of row 3, sibling of the log section.
% Per user request the summary lives next to the log instead of inside it,
% so the operator sees both panels at a glance without tab-switching or
% scrolling. Updated by onSummaryClicked (button lives in playback panel).
mainSummaryPanel = uipanel(main, ...
    "Title", "LLM 임무 종합 요약 (전체 case 누적 분석)", ...
    "BackgroundColor", [0.94 0.96 1.00], "FontWeight", "bold");
mainSummaryPanel.Layout.Row = 3; mainSummaryPanel.Layout.Column = 2;
mainSummaryInner = uigridlayout(mainSummaryPanel, [1, 1], "Padding", [8 6 8 6]);
lblSummaryLLM = uitextarea(mainSummaryInner, ...
    "Value",      "  📋 LLM 임무 요약 버튼을 눌러 전체 case의 종합 보고서를 생성하세요.", ...
    "Editable",   "off", ...
    "FontSize",   12, ...
    "FontName",   "Malgun Gothic", ...
    "BackgroundColor", [0.94 0.96 1.00]);

xaiOuter = uigridlayout(xaiPanel, [2, 1], ...
    "RowHeight", {'1x', 56}, ...
    "Padding", [4 4 4 4], "RowSpacing", 2);

xaiTabs = uitabgroup(xaiOuter);
xaiTabs.Layout.Row = 1;

%{
% --- DISABLED per user request: mAP@0.5 trend tab ---
tab_trend  = uitab(xaiTabs, "Title", "mAP@0.5 trend");
trendInner = uigridlayout(tab_trend, [1, 1], "Padding", [4 4 4 4]);
ax_trend   = uiaxes(trendInner);
title(ax_trend, "Run cases to populate mAP@0.5 trend");

% --- DISABLED per user request: feature importance tab ---
tab_xai  = uitab(xaiTabs, "Title", "Feature importance");
xaiInner = uigridlayout(tab_xai, [1, 1], "Padding", [4 4 4 4]);
ax_xai   = uiaxes(xaiInner);
title(ax_xai, "Run cases to populate feature importance");
%}

% Operations log: latest narrative on top, case history below. LLM summary
% lives in its own top-level panel next to this one (see mainSummaryPanel
% in main grid row 3, col 2 — adjacent to the log section).
tab_ops  = uitab(xaiTabs, "Title", "Operations log");
opsInner = uigridlayout(tab_ops, [2, 1], ...
    "RowHeight", {'1x', 165}, ...
    "Padding", [6 6 6 6], "RowSpacing", 6);

% --- Latest edge-case narrative (per-run report) -------------------------
narrPanel = uipanel(opsInner, ...
    "Title", "Latest edge-case narrative (case별 운용 보고)", ...
    "BackgroundColor", [1.00 0.99 0.95], "FontWeight", "bold", "FontSize", 12);
narrPanel.Layout.Row = 1;
narrInner = uigridlayout(narrPanel, [1, 1], "Padding", [8 4 8 4]);
lblNarrative = uilabel(narrInner, ...
    "Text", "  Run a counterfactual case to generate a narrative.", ...
    "FontSize", 13, "WordWrap", "on", ...
    "VerticalAlignment", "top");

% --- Case history (last 5 iters as fixed-width table) --------------------
histPanel = uipanel(opsInner, ...
    "Title", "Case history (latest 5)", ...
    "BackgroundColor", [0.97 0.97 0.99], "FontWeight", "bold", "FontSize", 12);
histPanel.Layout.Row = 2;
histInner = uigridlayout(histPanel, [1, 1], "Padding", [8 4 8 4]);
lblHistory = uilabel(histInner, ...
    "Text", "  (empty)", ...
    "FontSize", 12, "WordWrap", "off", ...
    "FontName", "Consolas", ...
    "VerticalAlignment", "top");

% Shared bottom label bar (dominant cause + verdict counts)
lblBox = uigridlayout(xaiOuter, [2, 1], ...
    "RowHeight", {28, 24}, "RowSpacing", 2, "Padding", [6 0 6 0]);
lblBox.Layout.Row = 2;

lblDominant = uilabel(lblBox, ...
    "Text", "  Run a case to start boundary discovery.", ...
    "FontWeight", "bold", "FontSize", 12);
lblDominant.Layout.Row = 1;

lblSummary = uilabel(lblBox, ...
    "Text", "  Tries: 0   PASS: 0   MARGINAL: 0   FAIL: 0", ...
    "FontSize", 11, "FontColor", [0.30 0.30 0.35]);
lblSummary.Layout.Row = 2;

% Counterfactual control panel — predefined scenario dropdown only.
% Slider bars and the REQ threshold dropdown were removed per user request;
% PASS threshold is locked to 0.50 (REQ-1 default) in state.passThresh below.
ctrlPanel = uipanel(main, ...
    "Title", "Counterfactual case  (시나리오 선택 후 ▶ Run Counterfactual)", ...
    "BackgroundColor", [0.95 0.95 0.97], "FontWeight", "bold");
ctrlPanel.Layout.Row = 4; ctrlPanel.Layout.Column = [1 2];
ctrl = uigridlayout(ctrlPanel, [1, 1], ...
    "Padding", [10 6 10 6]);

% --- Pre-defined scenarios -----------------------------------------------
% 5 PASS-yielding operating regimes covering nominal Korean mountain
% surveillance conditions (clear-to-light-haze, daytime). All values stay
% well inside the REQ-1 = 0.50 safety envelope so every scenario verifies
% the system passes — they vary in *character* (haze / cloud / sensor age)
% rather than in stress level.
SCENARIO_NAMES = [ ...
    "① 맑은 한낮 baseline (정상 작전)", ...
    "② 봄·가을 옅은 시계 (일반 시계)", ...
    "③ 옅은 산 안개 + 부분 흐림", ...
    "④ 부분 흐림 정오 (광량 양호)", ...
    "⑤ 이른 오후 옅은 안개 (센서 약간 노후)"];
SCENARIO_FOG = [  5,   10,   18,   12,   15];
SCENARIO_ILL = [12000, 10000, 9000, 7500, 8500];
SCENARIO_NOI = [ 0.02, 0.03, 0.04, 0.05, 0.06];

scnPanel = uipanel(ctrl, "BorderType", "none", "BackgroundColor", [0.95 0.95 0.97]);
scnGrid  = uigridlayout(scnPanel, [3, 1], "RowHeight", {22, 32, '1x'}, ...
    "Padding", [4 4 4 4], "RowSpacing", 4);
uilabel(scnGrid, "Text", "사전 정의 시나리오  (모두 PASS 예상 · REQ-1 mAP ≥ 0.50)", ...
    "FontWeight", "bold", "FontSize", 12);
scenarioDropdown = uidropdown(scnGrid, ...
    "Items",     SCENARIO_NAMES, ...
    "ItemsData", 1:numel(SCENARIO_NAMES), ...
    "Value",     1, ...
    "FontSize",  12, ...
    "Tooltip",   "선택하면 fog · illumination · noise 값이 자동 반영됩니다.");
lblScenarioVals = uilabel(scnGrid, ...
    "Text", sprintf("  선택된 환경값:  fog %.0f %%,  illum %.0f lx,  noise %.2f", ...
        SCENARIO_FOG(1), SCENARIO_ILL(1), SCENARIO_NOI(1)), ...
    "FontSize", 11, "FontColor", [0.30 0.30 0.40]);

% --- Hidden state holder: fog/ill/noi sliders ----------------------------
% Sliders are kept (off-screen) so all downstream code paths (renderFrame,
% startRun, applyEnvToSliders, DSPy auto-loop) continue to read .Value
% transparently. The visible UI is the scenario dropdown above; selecting a
% scenario writes through to these hidden sliders via onScenarioChanged.
hiddenCtrlPanel = uipanel(fig, ...
    "Visible", "off", "Position", [1 1 200 60]);
hiddenCtrl = uigridlayout(hiddenCtrlPanel, [1, 3], ...
    "ColumnWidth", {'1x', '1x', '1x'});
[fogSld, fogLbl] = mkSlider(hiddenCtrl, "Fog density",     0,    100,  SCENARIO_FOG(1), "%%");
[illSld, illLbl] = mkSlider(hiddenCtrl, "Illumination",  100,  15000, SCENARIO_ILL(1), "lx");
[noiSld, noiLbl] = mkSlider(hiddenCtrl, "Camera noise",    0,    1.0,  SCENARIO_NOI(1), "");

% Playback control panel
pbPanel = uipanel(main, "BackgroundColor", [0.96 0.96 0.98]);
pbPanel.Layout.Row = 5; pbPanel.Layout.Column = [1 2];
pb = uigridlayout(pbPanel, [2, 6], ...
    "RowHeight", {32, 50}, ...
    "ColumnWidth", {180, 90, 90, 110, 150, '1x'}, ...
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

btnSummary = uibutton(pb, "Text", "📋 LLM 임무 요약", ...
    "BackgroundColor", [0.30 0.40 0.65], "FontColor", "w", "FontWeight", "bold", ...
    "Tooltip", "전체 case history를 LLM에 보내 임무 종합 요약·실패 경계·시사점·권고를 생성");
btnSummary.Layout.Row = 1; btnSummary.Layout.Column = 5;

iterLbl = uilabel(pb, ...
    "Text", "Iter: 0 / 10  —  manual mode", ...
    "FontWeight", "bold", "HorizontalAlignment", "left");
iterLbl.Layout.Row = 1; iterLbl.Layout.Column = 6;

frameSld = uislider(pb, "Limits", [1 max(2,Nt)], "Value", 1, ...
    "MajorTicks", round(linspace(1, max(2,Nt), 5)));
frameSld.Layout.Row = 2; frameSld.Layout.Column = [1 5];

frameLbl = uilabel(pb, ...
    "Text", sprintf("Frame: 1 / %d", Nt), ...
    "HorizontalAlignment", "right", "FontWeight", "bold");
frameLbl.Layout.Row = 2; frameLbl.Layout.Column = 6;

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
    "metric", {}, "verdict", {}, ...   % verdict ∈ "PASS" | "MARGINAL" | "FAIL"
    "metric_person", {}, "metric_vehicle", {}, ...   % per-class mAP@0.5
    "ngt", {}, "ndet", {}, "ntp", {}, ...
    "n_intruders_total", {}, "n_intruders_seen", {}, ...
    "n_intruders_detected", {}, "n_intruders_missed", {}, ...
    "mode", {}, "analysis", {});
state.iterCount     = 0;
state.maxIter       = 10;
state.cooldownTimer = [];
% Provenance fields — what produced the NEXT case to be recorded.
% Default "seed" for the very first run; reset to "manual" after each
% finalizeRun so user-driven Run clicks are tagged correctly; replaced
% by DSPy mode (boundary_push/recover/refine, llm_*, rule_*) whenever
% scheduleNextAutoRun decides the upcoming case.
state.nextMode      = "seed";
state.nextAnalysis  = "Initial seed — user-defined operating baseline";
% 3-tier thresholds calibrated to *our* system (heuristic detector on
% rendered synthetic images, not a real CNN). Aligned with:
%   PASS = 0.50 → mission_context.json REQ-1 (mAP@0.5 ≥ 0.50, the project
%                 spec; baseline 대비 30 % 상대 하락 안전선)
%   FAIL = 0.25 → Hendrycks ICLR'19 robustness consensus ("50 % relative
%                 drop from spec = system broken")
%   MARGINAL band 0.25–0.50 = boundary search's natural oscillation zone
state.passThresh    = 0.50;            % spec compliance (mission_context REQ-1)
state.failThresh    = 0.25;            % "system broken" — robustness literature consensus

fogSld.ValueChangedFcn   = @(~,~) renderFrame();
fogSld.ValueChangingFcn  = @(~,e) onSlideLive("fog", e);
illSld.ValueChangedFcn   = @(~,~) renderFrame();
illSld.ValueChangingFcn  = @(~,e) onSlideLive("ill", e);
noiSld.ValueChangedFcn   = @(~,~) renderFrame();
noiSld.ValueChangingFcn  = @(~,e) onSlideLive("noi", e);

btnRun.ButtonPushedFcn   = @(~,~) startRun();
btnStop.ButtonPushedFcn  = @(~,~) stopRun();
btnReset.ButtonPushedFcn = @(~,~) doReset();
btnSummary.ButtonPushedFcn = @(~,~) onSummaryClicked();
autoToggle.ValueChangedFcn = @(s,~) onAutoToggle(s);
scenarioDropdown.ValueChangedFcn = @(s,~) onScenarioChanged(s);

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
updateAnalysisPanels();
updateOpsLog();
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
            "fog", fogSld.Value, "ill", illSld.Value, "noi", noiSld.Value, ...
            "detEvents", zeros(0, 3), ...    % rows: [score, tp_flag, cls] for per-class mAP@0.5
            "totalGt",         0, ...        % frame-aggregated GT (used by mAP only — not "intruders missed")
            "totalGt_person",  0, ...
            "totalGt_vehicle", 0, ...
            "visibleFlags",    false(1, Nobs), ...   % per-intruder: was GT visible at least once?
            "tpFlags",         false(1, Nobs));      % per-intruder: was it ever detected (TP) at least once?
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
            metric    = computeRunMetric(rs);
            metric_p  = computeRunMetricClass(rs, 1);   % person
            metric_v  = computeRunMetricClass(rs, 2);   % vehicle
            verdict = classifyVerdict(metric);

            % Unique-intruder counts (mission semantics: max 5 = 3 people + 2 vehicles).
            % rs.ngt / rs.ntp are frame-aggregated and used only for mAP — they are
            % NOT the right thing to report as "intruders missed" in operational text.
            n_intruders_total    = Nobs;
            n_intruders_seen     = sum(rs.visibleFlags);
            n_intruders_detected = sum(rs.tpFlags & rs.visibleFlags);
            n_intruders_missed   = n_intruders_seen - n_intruders_detected;

            rec = struct( ...
                "iter",     numel(state.history) + 1, ...
                "fog",      rs.fog, "ill", rs.ill, "noi", rs.noi, ...
                "metric",   metric, "verdict", verdict, ...
                "metric_person",  metric_p, ...
                "metric_vehicle", metric_v, ...
                "ngt",      rs.ngt, "ndet", rs.ndet, "ntp", rs.ntp, ...
                "n_intruders_total",    n_intruders_total, ...
                "n_intruders_seen",     n_intruders_seen, ...
                "n_intruders_detected", n_intruders_detected, ...
                "n_intruders_missed",   n_intruders_missed, ...
                "mode",     char(state.nextMode), ...
                "analysis", char(state.nextAnalysis));
            state.history(end + 1) = rec;

            % Persist FAIL / MARGINAL cases as edge cases (optimal filter
            % done in Python). PASS cases are not edge cases.
            if verdict ~= "PASS"
                trySaveEdgeCase(rec);
            end

            % Consumed — reset to "manual" so next user-driven click is
            % tagged correctly. Auto-loop's scheduleNextAutoRun will
            % overwrite this to the DSPy mode before the next run starts.
            state.nextMode     = "manual";
            state.nextAnalysis = "User-driven slider values";
            updateAnalysisPanels();
            updateOpsLog();
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
        % Show "DSPy thinking" status BEFORE the (potentially slow) LLM call
        hdr.BackgroundColor = [0.30 0.30 0.55];
        hdr.Text = sprintf( ...
            "  ⚙ DSPy LLM analyzing iter %d → %d  (SHAP + history → next counterfactual case)...", ...
            numel(state.history), numel(state.history) + 1);
        iterLbl.Text = sprintf("Auto: %d / %d  —  LLM consulting...", ...
            numel(state.history), state.maxIter);
        drawnow;

        % Compute next case from history using boundary-search policy
        [nextEnv, modeStr, analysisStr] = decideNextCase(state.history);

        % Record provenance for the upcoming auto-iter's history entry
        state.nextMode     = modeStr;
        state.nextAnalysis = analysisStr;

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
            histJson = pyHistoryJson(history);
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
            % Python/DSPy unavailable — use MATLAB-native 3-tier policy
            [nextEnv, modeStr, analysisStr] = decideNextCaseRule(history);
            if numel(state.history) <= 1
                fprintf("[DASHBOARD] DSPy not available (%s) — using MATLAB rule policy.\n", ME.message);
            end
        end
    end

    function payload = pyHistoryJson(history)
        % Translate the dashboard's verdict-aware history into the dict shape
        % dashboard_step.py expects (passed = verdict != "FAIL", so MARGINAL
        % is treated as "not yet broken" and Python keeps pushing).
        n = numel(history);
        if n == 0
            payload = "[]"; return;
        end
        items = cell(1, n);
        for k = 1:n
            h = history(k);
            items{k} = struct( ...
                "fog",     h.fog, ...
                "ill",     h.ill, ...
                "noi",     h.noi, ...
                "f1",      h.metric, ...
                "passed",  h.verdict ~= "FAIL", ...
                "verdict", char(h.verdict));
        end
        payload = jsonencode(items);
    end

    function [nextEnv, modeStr, analysisStr] = decideNextCaseRule(history)
        % MATLAB-native 3-tier boundary policy. MARGINAL cases serve as
        % half-anchors: they help refine the boundary even when only one
        % side (PASS or FAIL) has been observed.
        if isempty(history)
            nextEnv = struct("fog", 30, "ill", 4000, "noi", 0.1);
            modeStr = "seed"; analysisStr = "Initial seed";
            return;
        end

        passEnv = []; failEnv = []; marginalEnv = [];
        for k = 1:numel(history)
            h = history(k);
            e = struct("fog", h.fog, "ill", h.ill, "noi", h.noi);
            switch h.verdict
                case "PASS",     passEnv     = e;
                case "MARGINAL", marginalEnv = e;
                case "FAIL",     failEnv     = e;
            end
        end

        last    = history(end);
        lastEnv = struct("fog", last.fog, "ill", last.ill, "noi", last.noi);

        switch last.verdict
            case "PASS"
                if ~isempty(failEnv)
                    nextEnv = bisectEnv(lastEnv, failEnv, 0.65);
                    modeStr = "boundary_push";
                    analysisStr = "PASS → bisect 65% toward FAIL anchor";
                elseif ~isempty(marginalEnv)
                    nextEnv = bisectEnv(lastEnv, marginalEnv, 0.65);
                    modeStr = "boundary_push_to_margin";
                    analysisStr = "PASS → bisect 65% toward MARGINAL (no FAIL anchor yet)";
                else
                    nextEnv = pushEnv(lastEnv);
                    modeStr = "rule_push";
                    analysisStr = "All PASS so far — push harder (rule)";
                end

            case "FAIL"
                if ~isempty(passEnv)
                    nextEnv = bisectEnv(lastEnv, passEnv, 0.65);
                    modeStr = "boundary_recover";
                    analysisStr = "FAIL → bisect 65% toward PASS anchor";
                elseif ~isempty(marginalEnv)
                    nextEnv = bisectEnv(lastEnv, marginalEnv, 0.65);
                    modeStr = "boundary_recover_to_margin";
                    analysisStr = "FAIL → bisect 65% toward MARGINAL (no PASS anchor yet)";
                else
                    nextEnv = relaxEnv(lastEnv);
                    modeStr = "rule_relax";
                    analysisStr = "All FAIL so far — relax toward baseline (rule)";
                end

            otherwise   % MARGINAL — already near the boundary
                if ~isempty(passEnv) && ~isempty(failEnv)
                    nextEnv = bisectEnv(passEnv, failEnv, 0.50);
                    modeStr = "boundary_refine";
                    analysisStr = "MARGINAL → midpoint of PASS↔FAIL (narrow boundary)";
                elseif ~isempty(passEnv)
                    nextEnv = pushEnv(lastEnv);
                    modeStr = "rule_push";
                    analysisStr = "MARGINAL with no FAIL yet — push to find broken side";
                elseif ~isempty(failEnv)
                    nextEnv = relaxEnv(lastEnv);
                    modeStr = "rule_relax";
                    analysisStr = "MARGINAL with no PASS yet — relax to find safe side";
                else
                    nextEnv = pushEnv(lastEnv);
                    modeStr = "rule_push";
                    analysisStr = "MARGINAL only — push to probe boundary";
                end
        end
    end

    function doReset()
        if state.mode == "running"
            finalizeRun();
        end
        state.frameIdx     = 1;
        state.nextMode     = "seed";
        state.nextAnalysis = "Initial seed — user-defined operating baseline";
        frameSld.Value = 1;
        frameLbl.Text  = sprintf("Frame: 1 / %d", Nt);
        renderFrame();
        updateAnalysisPanels();
        updateOpsLog();
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
        metric   = computeRunMetric(rs);
        metric_p = computeRunMetricClass(rs, 1);
        metric_v = computeRunMetricClass(rs, 2);
        verdict  = classifyVerdict(metric);
        switch verdict
            case "PASS",     color = [0.10 0.45 0.20];   % green
            case "MARGINAL", color = [0.55 0.40 0.05];   % amber
            otherwise,       color = [0.55 0.10 0.15];   % red (FAIL)
        end
        % Unique-intruder counts (max Nobs = 5 = 3 people + 2 vehicles).
        % rs.ngt/ntp are frame-aggregated and would inflate "missed" to hundreds.
        n_seen     = sum(rs.visibleFlags);
        n_detected = sum(rs.tpFlags & rs.visibleFlags);
        n_missed   = n_seen - n_detected;
        hdr.BackgroundColor = color;
        hdr.Text = sprintf( ...
            "  [ %s ]   fog=%.0f%%  illum=%.0flx  noise=%.2f   |   침입자 %d/%d 식별 (누락 %d)   |   %s = %.3f   (사람 %.2f · 차량 %.2f)   (PASS≥%.2f)", ...
            verdict, rs.fog, rs.ill, rs.noi, ...
            n_detected, n_seen, n_missed, ...
            metricLabel(), metric, metric_p, metric_v, ...
            state.passThresh);
    end

    function trySaveEdgeCase(rec)
        % Persist FAIL / MARGINAL parameter sets via dashboard_step.save_edge_case.
        % "Optimal edge cases only" — Python side deduplicates near-neighbours
        % in normalized (fog,ill,noi) space, so repeated FAILs in the same
        % corner of the envelope don't accumulate.
        try
            payload = jsonencode(struct( ...
                "fog",            rec.fog, ...
                "ill",            rec.ill, ...
                "noi",            rec.noi, ...
                "metric",         rec.metric, ...
                "metric_person",  rec.metric_person, ...
                "metric_vehicle", rec.metric_vehicle, ...
                "verdict",        char(rec.verdict), ...
                "iter",           rec.iter, ...
                "mode",           rec.mode, ...
                "n_intruders_total",    rec.n_intruders_total, ...
                "n_intruders_seen",     rec.n_intruders_seen, ...
                "n_intruders_detected", rec.n_intruders_detected, ...
                "n_intruders_missed",   rec.n_intruders_missed, ...
                "ngt",            rec.n_intruders_seen, ...
                "ndet",           rec.n_intruders_detected, ...
                "ntp",            rec.n_intruders_detected));
            resPy = py.dashboard_step.save_edge_case(payload);
            d = struct(resPy);
            if logical(d.saved)
                fprintf("[EdgeCase] saved (%s) — total %d on disk\n", ...
                    string(char(d.reason)), double(d.n_total));
            end
        catch ME
            fprintf("[EdgeCase] save failed: %s\n", ME.message);
        end
    end

    function v = classifyVerdict(metric)
        if metric >= state.passThresh
            v = "PASS";
        elseif metric < state.failThresh
            v = "FAIL";
        else
            v = "MARGINAL";
        end
    end

    function s = metricLabel()
        s = "mAP@0.5";
    end

    function metric = computeRunMetric(rs)
        % Overall mAP@0.5 = unweighted mean of per-class mAP. Mirrors COCO's
        % "mean of class-AP" definition rather than micro-averaging FPs across
        % classes, so a 0-recall on one class can't be hidden by the other.
        m_p = computeRunMetricClass(rs, 1);
        m_v = computeRunMetricClass(rs, 2);
        % Only average over classes that have at least one GT this run —
        % otherwise a class with no intruders would bias the score down to 0.
        active = [];
        if isfield(rs, "totalGt_person")  && rs.totalGt_person  > 0, active(end+1) = m_p; end
        if isfield(rs, "totalGt_vehicle") && rs.totalGt_vehicle > 0, active(end+1) = m_v; end
        if isempty(active)
            metric = 0;
        else
            metric = mean(active);
        end
    end

    function metric = computeRunMetricClass(rs, cls)
        % Per-class AP@0.5 = VOC11-style envelope over (score-sorted) detection
        % events filtered to this class. cls: 1=person, 2=vehicle.
        if ~isfield(rs, "detEvents") || isempty(rs.detEvents)
            metric = 0; return;
        end
        if cls == 1
            gtCount = rs.totalGt_person;
        else
            gtCount = rs.totalGt_vehicle;
        end
        if gtCount == 0
            metric = 0; return;
        end
        ev = rs.detEvents(rs.detEvents(:, 3) == cls, :);
        if isempty(ev)
            metric = 0; return;
        end
        [~, ord] = sort(ev(:, 1), "descend");
        ev = ev(ord, :);
        cumTp = cumsum(ev(:, 2));
        cumFp = cumsum(1 - ev(:, 2));
        precision = cumTp ./ max(cumTp + cumFp, eps);
        recall    = cumTp / gtCount;
        mrec = [0; recall; 1];
        mpre = [0; precision; 0];
        for i = numel(mpre) - 1 : -1 : 1
            mpre(i) = max(mpre(i), mpre(i + 1));
        end
        idx    = find(mrec(2:end) ~= mrec(1:end-1));
        metric = sum((mrec(idx + 1) - mrec(idx)) .* mpre(idx + 1));
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
        ngt_p = 0; ngt_v = 0;     % per-class GT counts this frame
        % Per-frame detection events for mAP@0.5 — rows: [score, tp_flag, cls]
        evtFrame = zeros(0, 3);
        for k = 1:Nobs
            gt = gtFrame(k, :);
            dt = detBB_frame(k, :);
            sc = scores(k);
            cls = obs_class(k);
            cls_lbl = INTRUDER_LABEL(cls);
            gtPresent  = any(gt ~= 0);
            detPresent = (sc > 0.20) && any(dt ~= 0);

            if gtPresent
                ngt = ngt + 1;
                if cls == 1, ngt_p = ngt_p + 1; else, ngt_v = ngt_v + 1; end
                % Per-intruder visibility flag (for unique-intruder counting)
                if state.mode == "running"
                    state.runStats.visibleFlags(k) = true;
                end
                rectangle("Parent", ax2, "Position", clamp_box(gt, camW, camH), ...
                    "EdgeColor", [0.10 0.85 0.20], "LineStyle", "--", ...
                    "LineWidth", 1.5, "Tag", "bbox_overlay");
                text(ax2, gt(1), max(8, gt(2) - 6), sprintf("%s GT", cls_lbl), ...
                    "Color", [0.10 0.85 0.20], "FontWeight", "bold", "FontSize", 8, ...
                    "Tag", "bbox_overlay");
            end
            if detPresent
                ndet = ndet + 1;
                tpFlag = 0;
                if gtPresent && bbox_iou(dt, gt) >= 0.5
                    tpFlag = 1;
                    ntp = ntp + 1;
                    % Per-intruder TP flag (sticky — once detected, ever detected)
                    if state.mode == "running"
                        state.runStats.tpFlags(k) = true;
                    end
                end
                evtFrame(end + 1, :) = [sc, tpFlag, cls]; %#ok<AGROW>
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
            state.runStats.ngt              = state.runStats.ngt              + ngt;
            state.runStats.ndet             = state.runStats.ndet             + ndet;
            state.runStats.ntp              = state.runStats.ntp              + ntp;
            state.runStats.nFrames          = state.runStats.nFrames          + 1;
            state.runStats.totalGt          = state.runStats.totalGt          + ngt;
            state.runStats.totalGt_person   = state.runStats.totalGt_person   + ngt_p;
            state.runStats.totalGt_vehicle  = state.runStats.totalGt_vehicle  + ngt_v;
            if ~isempty(evtFrame)
                state.runStats.detEvents = [state.runStats.detEvents; evtFrame];
            end

            cumMetric = computeRunMetric(state.runStats);
            cumP = computeRunMetricClass(state.runStats, 1);
            cumV = computeRunMetricClass(state.runStats, 2);

            hdr.BackgroundColor = [0.10 0.30 0.55];
            hdr.Text = sprintf( ...
                "  ▶ RUNNING   case: fog=%.0f%%  illum=%.0flx  noise=%.2f   |   Frame %d/%d  t=%.2fs   |   cum %s = %.3f   (사람 %.2f · 차량 %.2f)", ...
                fog, ill, noi, ii, Nt, t_vec(ii), metricLabel(), cumMetric, cumP, cumV);
        end

        frameLbl.Text = sprintf("Frame: %d / %d", ii, Nt);
        title(ax2, sprintf("EO Camera   t=%.2fs   fog=%.0f%%  illum=%.0flx  noise=%.2f", ...
            t_vec(ii), fog, ill, noi));
    end

    function updateAnalysisPanels()
        % Update both the boundary scatter (C) and the XAI feature
        % importance / dominant cause panel (B). Called after each
        % run finalises, and once at startup.
        if ~isvalid(fig), return; end
        N = numel(state.history);

        %{
        % ====== DISABLED per user request: 3 visualizations below ========
        %   (1) Counterfactual boundary 3-D scatter   (ax_boundary)
        %   (2) mAP@0.5 trend chart                   (ax_trend)
        %   (3) XAI feature-importance bar chart      (ax_xai)
        % The handles ax_boundary / ax_trend / ax_xai no longer exist, so
        % this whole block is wrapped in %{...%} to keep the code in place
        % but inert. Restore by deleting the surrounding %{ %} markers AND
        % uncommenting the matching creation block earlier in the file.
        % -----------------------------------------------------------------

        % --- Boundary scatter ---
        cla(ax_boundary);
        hold(ax_boundary, "on");
        grid(ax_boundary, "on");
        if N == 0
            title(ax_boundary, "Counterfactual boundary discovery  —  (no cases yet)");
        else
            fogs     = arrayfun(@(r) r.fog, state.history);
            ills     = arrayfun(@(r) r.ill, state.history);
            nois     = arrayfun(@(r) r.noi, state.history);
            verdicts = arrayfun(@(r) r.verdict, state.history);

            isPass = verdicts == "PASS";
            isMarg = verdicts == "MARGINAL";
            isFail = verdicts == "FAIL";

            if N >= 2
                plot3(ax_boundary, fogs, ills, nois, "-", ...
                    "Color", [0.55 0.55 0.60], "LineWidth", 0.8);
            end
            if any(isPass)
                scatter3(ax_boundary, fogs(isPass), ills(isPass), nois(isPass), ...
                    90, [0.10 0.65 0.20], "filled", "MarkerEdgeColor", "k");
            end
            if any(isMarg)
                scatter3(ax_boundary, fogs(isMarg), ills(isMarg), nois(isMarg), ...
                    90, [0.95 0.65 0.10], "filled", "MarkerEdgeColor", "k");
            end
            if any(isFail)
                scatter3(ax_boundary, fogs(isFail), ills(isFail), nois(isFail), ...
                    90, [0.85 0.20 0.15], "filled", "MarkerEdgeColor", "k");
            end
            scatter3(ax_boundary, fogs(end), ills(end), nois(end), ...
                200, [0.95 0.85 0.10], "d", "LineWidth", 2.0);

            title(ax_boundary, sprintf( ...
                "Counterfactual boundary  —  %d case%s   (PASS green / MARGINAL amber / FAIL red)", ...
                N, repmat("s", 1, double(N ~= 1))));
        end
        xlabel(ax_boundary, "Fog (%)");
        ylabel(ax_boundary, "Illumination (lx)");
        zlabel(ax_boundary, "Noise");
        xlim(ax_boundary, [0 100]);
        ylim(ax_boundary, [0 15000]);
        zlim(ax_boundary, [0 1]);
        view(ax_boundary, 35, 25);

        % --- mAP@0.5 trend ---
        cla(ax_trend);
        hold(ax_trend, "on");
        grid(ax_trend, "on");
        if N == 0
            title(ax_trend, sprintf("%s trend  —  (no cases yet)", metricLabel()));
            ax_trend.XLim = [0.5 10.5];
        else
            xs = 1:N;
            metricsT = arrayfun(@(r) r.metric, state.history);
            plot(ax_trend, xs, metricsT, "-", ...
                "Color", [0.55 0.55 0.60], "LineWidth", 0.9);
            yline(ax_trend, state.passThresh, "--", ...
                sprintf("PASS ≥ %.2f", state.passThresh), ...
                "Color", [0.10 0.55 0.20], "LineWidth", 1.1, ...
                "LabelHorizontalAlignment", "left");
            yline(ax_trend, state.failThresh, "--", ...
                sprintf("FAIL < %.2f", state.failThresh), ...
                "Color", [0.75 0.15 0.15], "LineWidth", 1.1, ...
                "LabelHorizontalAlignment", "left");
            verdictsT = arrayfun(@(r) r.verdict, state.history);
            isPassT = verdictsT == "PASS";
            isMargT = verdictsT == "MARGINAL";
            isFailT = verdictsT == "FAIL";
            if any(isPassT)
                scatter(ax_trend, xs(isPassT), metricsT(isPassT), 80, ...
                    [0.10 0.65 0.20], "filled", "MarkerEdgeColor", "k");
            end
            if any(isMargT)
                scatter(ax_trend, xs(isMargT), metricsT(isMargT), 80, ...
                    [0.95 0.65 0.10], "filled", "MarkerEdgeColor", "k");
            end
            if any(isFailT)
                scatter(ax_trend, xs(isFailT), metricsT(isFailT), 80, ...
                    [0.85 0.20 0.15], "filled", "MarkerEdgeColor", "k");
            end
            scatter(ax_trend, xs(end), metricsT(end), 160, ...
                [0.95 0.85 0.10], "d", "LineWidth", 2.0);
            title(ax_trend, sprintf("%s trend  —  iter 1..%d", metricLabel(), N));
            ax_trend.XLim = [0.5, max(N + 0.5, 10.5)];
        end
        xlabel(ax_trend, "Iteration");
        ylabel(ax_trend, metricLabel());
        ax_trend.YLim = [0 1];

        % --- XAI feature importance ---
        cla(ax_xai);
        if N < 3
            title(ax_xai, sprintf( ...
                "Feature importance  —  need ≥ 3 cases  (currently %d)", N));
            ax_xai.XTick = [];  ax_xai.YTick = [];
            ax_xai.XLim = [0 1]; ax_xai.YLim = [0 1];
        else
            fogs    = arrayfun(@(r) r.fog,    state.history);
            ills    = arrayfun(@(r) r.ill,    state.history);
            nois    = arrayfun(@(r) r.noi,    state.history);
            metrics = arrayfun(@(r) r.metric, state.history);

            xaiMethod = "pearson_fallback";
            xaiR2 = NaN;
            xaiN  = N;
            corrs = [safeCorr(fogs, metrics), safeCorr(ills, metrics), safeCorr(nois, metrics)];
            try
                histJson = pyHistoryJson(state.history);
                xaiResult = py.dashboard_step.compute_xai_for_dashboard(histJson);
                d = struct(xaiResult);
                methodStr = string(char(d.method));
                if methodStr == "kernel_shap"
                    imp = [double(d.fog_importance), ...
                           double(d.illum_importance), ...
                           double(d.noise_importance)];
                    signed = [double(d.fog_signed), double(d.illum_signed), double(d.noise_signed)];
                    for kk = 1:3
                        if abs(signed(kk)) < 1e-9
                            signed(kk) = -corrs(kk);
                        end
                    end
                    corrs = -signed;
                    xaiMethod = "KernelSHAP";
                    xaiR2 = double(d.model_r2);
                    xaiN  = double(d.n_samples);
                else
                    error("SHAP fallback: %s", methodStr);
                end
            catch ME
                imp = abs(corrs);
                tot = sum(imp);
                if tot < 1e-6, imp = [0 0 0]; else, imp = imp / tot; end
                xaiMethod = "Pearson r";
                if numel(state.history) <= 3
                    fprintf("[XAI] SHAP unavailable (%s) — Pearson fallback.\n", ME.message);
                end
            end

            cdata = zeros(3, 3);
            for k = 1:3
                if corrs(k) < 0
                    cdata(k,:) = [0.85 0.25 0.20];
                else
                    cdata(k,:) = [0.20 0.50 0.85];
                end
            end
            bar(ax_xai, imp, "FaceColor", "flat", "CData", cdata, ...
                "EdgeColor", [0.20 0.20 0.25]);
            ax_xai.XTick = 1:3;
            ax_xai.XTickLabel = {"Fog", "Illum", "Noise"};
            ax_xai.YLim = [0 max(0.05, max(imp) * 1.20)];
            ylabel(ax_xai, "Normalized importance");
            grid(ax_xai, "on");

            for k = 1:3
                arrow = "↓"; if corrs(k) >= 0, arrow = "↑"; end
                text(ax_xai, k, imp(k) + 0.02, sprintf("%s%.2f", arrow, corrs(k)), ...
                    "HorizontalAlignment", "center", ...
                    "FontWeight", "bold", "FontSize", 9);
            end
            if isnan(xaiR2)
                title(ax_xai, sprintf( ...
                    "Feature importance  (%s)  red=harms · blue=helps  N=%d", ...
                    xaiMethod, xaiN));
            else
                title(ax_xai, sprintf( ...
                    "Feature importance  (%s, surrogate R²=%.2f)  red=harms · blue=helps  N=%d", ...
                    xaiMethod, xaiR2, xaiN));
            end
        end
        %}

        % --- Dominant cause line (B) -------------------------------------
        if N == 0
            lblDominant.Text = "  Run a case to start boundary discovery.";
            lblDominant.FontColor = [0.30 0.30 0.35];
        else
            last = state.history(end);
            mLab = metricLabel();
            switch last.verdict
                case "PASS"
                    lblDominant.Text = sprintf("  ✓ Latest: PASS  (%s = %.3f)", mLab, last.metric);
                    lblDominant.FontColor = [0.10 0.50 0.20];
                case "MARGINAL"
                    lblDominant.Text = sprintf( ...
                        "  ◐ Latest: MARGINAL  (%s = %.3f)  —  on the boundary (%.2f ≤ x < %.2f)", ...
                        mLab, last.metric, state.failThresh, state.passThresh);
                    lblDominant.FontColor = [0.55 0.40 0.05];
                otherwise   % FAIL
                    refIdx = find(arrayfun(@(r) r.verdict == "PASS", state.history), 1, "last");
                    if isempty(refIdx)
                        lblDominant.Text = sprintf( ...
                            "  ✗ Latest: FAIL  (%s = %.3f)  —  no PASS reference yet", ...
                            mLab, last.metric);
                    else
                        refEnv = state.history(refIdx);
                        devs   = [abs(last.fog - refEnv.fog) / 100, ...
                                  abs(last.ill - refEnv.ill) / 15000, ...
                                  abs(last.noi - refEnv.noi) / 1.0];
                        [~, di] = max(devs);
                        names = ["fog", "illumination", "noise"];
                        lblDominant.Text = sprintf( ...
                            "  ✗ Latest: FAIL (%s = %.3f)  —  dominant cause: %s  (Δ = %.0f%% from last PASS)", ...
                            mLab, last.metric, names(di), devs(di) * 100);
                    end
                    lblDominant.FontColor = [0.55 0.10 0.15];
            end
        end

        nPass = sum(arrayfun(@(r) r.verdict == "PASS",     state.history));
        nMarg = sum(arrayfun(@(r) r.verdict == "MARGINAL", state.history));
        nFail = sum(arrayfun(@(r) r.verdict == "FAIL",     state.history));
        lblSummary.Text = sprintf("  Tries: %d   PASS: %d   MARGINAL: %d   FAIL: %d", ...
            N, nPass, nMarg, nFail);
    end


    function updateOpsLog()
        % Update Operations log tab: latest narrative + history list.
        if ~isvalid(fig), return; end
        N = numel(state.history);
        if N == 0
            lblNarrative.Text = "  Run a counterfactual case to generate a narrative.";
            lblHistory.Text   = "  (empty)";
            return;
        end

        last = state.history(end);
        % Try LLM-driven narrative via DSPy Narrator; fall back to template
        % if Python/LLM call fails. Template is identical schema (security
        % report tone) so demo flow is uninterrupted on failure.
        try
            % Send UNIQUE-intruder counts (max 5 = 3 people + 2 vehicles) as
            % "ngt" / "ntp" so the narrator reports "X missed of 5 intruders"
            % instead of frame-aggregated "40 missed". Mission semantics =
            % per-unique-intruder, not per-frame-instance.
            recPy = struct( ...
                "fog", last.fog, "ill", last.ill, "noi", last.noi, ...
                "metric", last.metric, "verdict", char(last.verdict), ...
                "ngt", last.n_intruders_seen, ...
                "ndet", last.n_intruders_detected, ...
                "ntp",  last.n_intruders_detected, ...
                "n_intruders_total",    last.n_intruders_total, ...
                "n_intruders_seen",     last.n_intruders_seen, ...
                "n_intruders_detected", last.n_intruders_detected, ...
                "n_intruders_missed",   last.n_intruders_missed);
            payload = jsonencode(recPy);
            narrPy = py.dashboard_step.narrate_edge_case(payload);
            lblNarrative.Text = string(char(narrPy));
        catch ME
            lblNarrative.Text = buildTemplateNarrative(last);
            if numel(state.history) <= 2
                fprintf("[Narrator] LLM unavailable (%s) — template fallback.\n", ME.message);
            end
        end

        % History list — latest 5 entries, formatted as fixed-width columns.
        % Per-class mAP (사람·차량) columns surface class-level failures that
        % the overall mAP could otherwise mask.
        startIdx = max(1, N - 4);
        lines = strings(0, 1);
        lines(end+1) = "  iter  verdict   fog %  illum lx  noise   mAP    사람    차량   mode";
        lines(end+1) = "  ----  --------  -----  --------  -----   -----  -----  -----  -----------";
        for k = N : -1 : startIdx
            h = state.history(k);
            verdictTag = sprintf("[%s]", h.verdict);
            verdictPad = pad(verdictTag, 8, "right");
            mp = 0; mv = 0;
            if isfield(h, "metric_person"),  mp = h.metric_person;  end
            if isfield(h, "metric_vehicle"), mv = h.metric_vehicle; end
            lines(end+1) = sprintf("  %3d   %s  %5.1f  %7.0f   %5.2f   %.3f  %.3f  %.3f  %s", ...
                h.iter, verdictPad, h.fog, h.ill, h.noi, h.metric, mp, mv, h.mode);
        end
        lblHistory.Text = strjoin(lines, newline);
    end

    function txt = buildTemplateNarrative(rec)
        % Auto-generated edge-case narrative in security-report tone.
        % Phase 6 will replace this with LLM-driven Narrator output.

        % Qualitative environment descriptors
        if rec.fog < 8,        fogQ = "맑은 시계";
        elseif rec.fog < 25,   fogQ = "옅은 산 안개";
        elseif rec.fog < 50,   fogQ = "발달한 안개층";
        elseif rec.fog < 75,   fogQ = "짙은 안개";
        else,                  fogQ = "시정 거의 차단된 농무";
        end

        if rec.ill > 10000,    illQ = "맑은 한낮의 풍부한 광량";
        elseif rec.ill > 6000, illQ = "부분 흐림 일반 작전 광량";
        elseif rec.ill > 3000, illQ = "흐린 한낮";
        elseif rec.ill > 800,  illQ = "황혼·일출 직전 저조도";
        else,                  illQ = "야간에 준하는 한계 광량";
        end

        if rec.noi < 0.05,     noiQ = "센서 정상 상태";
        elseif rec.noi < 0.15, noiQ = "센서 약간 노후";
        elseif rec.noi < 0.30, noiQ = "센서 노이즈 누적";
        else,                  noiQ = "센서 심각한 열화";
        end

        switch rec.verdict
            case "PASS"
                impact = sprintf("탐지 정확도 %.2f로 임무 요구사항(REQ-1 ≥ 0.50)을 안정적으로 통과하여 침입자 식별·보고에 신뢰성이 확보됨", rec.metric);
                tag = "[ 임무 수행 가능 ]";
            case "MARGINAL"
                impact = sprintf("탐지 정확도 %.2f로 REQ-1 임계 부근에 위치, 보고 마진이 사실상 소진된 경계선 상태이며 추가 환경 악화 시 임무 신뢰성 상실 우려", rec.metric);
                tag = "[ 임무 마지노선 ]";
            otherwise
                impact = sprintf("탐지 정확도 %.2f로 REQ-1 미달, 침입자 보고가 누락되는 운용 불가 상태", rec.metric);
                tag = "[ 임무 수행 불가 ]";
        end

        txt = sprintf("  %s  %s, %s, %s 환경에서 UAV 정찰을 수행한 결과, %s.", ...
            tag, fogQ, illQ, noiQ, impact);
    end


    function onScenarioChanged(src)
        % Operator picked a pre-defined scenario. Look up the corresponding
        % (fog, illumination, noise) triple and write through to the hidden
        % sliders so downstream code paths (renderFrame, startRun, DSPy
        % auto-loop) keep reading the chosen environment unchanged.
        if state.mode == "running"
            return;     % case is locked during a run — ignore changes
        end
        idx = double(src.Value);
        idx = max(1, min(numel(SCENARIO_FOG), idx));
        env = struct("fog", SCENARIO_FOG(idx), ...
                     "ill", SCENARIO_ILL(idx), ...
                     "noi", SCENARIO_NOI(idx));
        applyEnvToSliders(env);
        lblScenarioVals.Text = sprintf( ...
            "  선택된 환경값:  fog %.0f %%,  illum %.0f lx,  noise %.2f", ...
            env.fog, env.ill, env.noi);
        renderFrame();
    end

    function onSummaryClicked()
        % LLM 요약본 — collect the full counterfactual history and ask the
        % DSPy MissionSummary signature for an executive report. Result is
        % rendered inline in the right-side Operations-log panel
        % (lblSummaryLLM) — no modal dialog.
        if isempty(state.history)
            lblSummaryLLM.Value = ...
                "  임무 요약을 생성하려면 최소 1회 이상의 case 실행이 필요합니다.";
            return;
        end
        prevText = btnSummary.Text;
        btnSummary.Enable = "off";
        btnSummary.Text   = "⏳ LLM 요약 생성 중...";
        lblSummaryLLM.Value = "  ⏳ LLM이 전체 case 이력을 종합 분석 중 (10~30초)...";
        drawnow;
        try
            histJson = jsonencode(arrayfun(@(h) struct( ...
                "iter",            h.iter, ...
                "fog",             h.fog, ...
                "ill",             h.ill, ...
                "noi",             h.noi, ...
                "metric",          h.metric, ...
                "metric_person",   h.metric_person, ...
                "metric_vehicle",  h.metric_vehicle, ...
                "verdict",         char(h.verdict)), ...
                state.history, "UniformOutput", false));
            sumPy = py.dashboard_step.summarize_mission(histJson);
            d = struct(sumPy);
            execTxt  = string(char(d.executive_summary));
            boundTxt = string(char(d.failure_boundary));
            implTxt  = string(char(d.security_implications));
            recTxt   = string(char(d.recommendations));
            lines = strings(0, 1);
            lines(end+1) = "▣ 종합 요약";
            lines(end+1) = execTxt;
            lines(end+1) = "";
            lines(end+1) = "▣ 실패 경계";
            lines(end+1) = boundTxt;
            lines(end+1) = "";
            lines(end+1) = "▣ 안보 시사점";
            lines(end+1) = implTxt;
            lines(end+1) = "";
            lines(end+1) = "▣ 운용 권고";
            lines(end+1) = recTxt;
            lblSummaryLLM.Value = cellstr(lines);
        catch ME
            lblSummaryLLM.Value = sprintf("LLM 요약 생성 실패: %s", ME.message);
        end
        btnSummary.Enable = "on";
        btnSummary.Text   = prevText;
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
% Sim length is tuned so the UAV's forward-tilted camera (pitch 60° from
% horizontal → optical axis hits ground ~26 m ahead) captures all 5 targets
% by t=StopTime, without an excessively long playback. UAV starts at x=-80
% with vx=3 m/s and the furthest intruder is at x=48. StopTime=38 puts the
% UAV at x = -80 + 3*38 = 34 m, with the camera looking ~60 m ahead — well
% past intruder 5. 380 frames (= 38 s / 0.1 s FixedStep) keeps the dashboard
% playback under ~30 s of wall time.
try
    set_param(mdl, "StopTime", "38");
catch ME
    fprintf("[DASHBOARD] StopTime override skipped: %s\n", ME.message);
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


function txt = loadMissionCard()
% Reads mission_context.json and formats into a 3-line dashboard banner.
% Line 3 surfaces the *business goal* explicitly — what the operator is
% trying to achieve with these counterfactual runs (not just the technical
% mission). Falls back to a generic placeholder if the JSON is missing.
try
    raw = fileread("mission_context.json");
    mc  = jsondecode(raw);
    title  = mc.mission_title;
    object = mc.objective;
    targets = strjoin(string(mc.detection_targets), " · ");
    req1 = sprintf("mAP@0.5 ≥ %.2f", mc.requirements.REQ_1.threshold);
    req3 = sprintf("연속 미탐 ≤ %d frame", mc.requirements.REQ_3.threshold);
    line1 = sprintf("  🛰  %s  —  %s", title, object);
    line2 = sprintf("  탐지 대상: %s   |   REQ-1 %s   |   REQ-3 %s   |   플랫폼: %s", ...
        targets, req1, req3, mc.operational_context.platform);
    line3 = "  🎯 비즈니스 목표: 환경 변동(안개·조도·잡음)에 대한 임무 실패 경계를 정량 식별하고, 운용 한계 + 위험 시나리오를 사전 도출";
    txt = sprintf("%s\n%s\n%s", line1, line2, line3);
catch ME
    txt = sprintf("  Mission context unavailable (%s)  —  국경 산악 감시 임무 (기본)", ...
        ME.message);
end
end


function r = safeCorr(x, y)
% Pearson correlation with NaN/zero-variance guards. Returns 0 when
% either vector has zero variance (all identical samples) or fewer
% than 2 points — keeps the XAI bar plot well-defined for small N.
x = double(x(:)); y = double(y(:));
if numel(x) < 2 || std(x) < 1e-9 || std(y) < 1e-9
    r = 0; return;
end
mx = mean(x); my = mean(y);
num = sum((x - mx) .* (y - my));
den = sqrt(sum((x - mx).^2) * sum((y - my).^2));
if den < 1e-12, r = 0; else, r = num / den; end
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
