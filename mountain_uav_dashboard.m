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
    pyMod.parse_requirement_to_env("baseline");
    % NOTE: generate_normalized_test_cases is NOT smoke-tested here because
    % its side-effect (writing test_suite_<ts>.json to disk) would pollute
    % data/test_suites/ on every dashboard launch. The function is tested
    % implicitly when the user clicks 🧪 Generate.
    pyMod.list_test_suites();
    pyMod.compare_replay_result( ...
        py.dict(struct("expected_verdict", "PASS", "expected_metric", 0.8, ...
                       "metric_min", 0.6, "metric_max", 1.0, ...
                       "acceptable_verdicts", {{ "PASS" }})), ...
        "PASS", 0.8);
    fprintf("[DASHBOARD] dashboard_step module loaded OK " + ...
            "(narrate / summarize / save_edge / parse_NL / gen_tests / " + ...
            "list_suites / compare_replay ready)\n");
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
    "Position", [60 60 1500 950], "Color", [0.97 0.97 0.99]);

% =========================================================================
% NEW LAYOUT — 4-tab workflow architecture
%
% Row 1: shared status header (visible across all tabs)
% Row 2: uitabgroup with 4 tabs corresponding to the user workflow:
%        ① 요구사항 (verification requirement input)
%        ② 시뮬레이션 (boundary-search execution + history)
%        ③ Replay   (normalized test cases + replay controls)
%        ④ LLM 요약 (mission-level summary report)
%
% Tab switches happen automatically at key handoff points (Run → ②,
% Generate tests → ③, LLM summary → ④) so the user is steered through
% the natural workflow without losing free navigation.
% =========================================================================
main = uigridlayout(fig, [2, 1], ...
    "RowHeight",   {40, '1x'}, ...
    "ColumnWidth", {'1x'}, ...
    "RowSpacing", 6, ...
    "Padding", [10 10 10 10]);

% ---- Shared status header (always visible above tabs) ----
hdr = uilabel(main, ...
    "Text", "  Initialising...", ...
    "FontSize", 15, "FontWeight", "bold", ...
    "BackgroundColor", [0.10 0.20 0.40], "FontColor", "w", ...
    "VerticalAlignment", "center");
hdr.Layout.Row = 1; hdr.Layout.Column = 1;

% ---- Tab group ----
tabGroup = uitabgroup(main);
tabGroup.Layout.Row = 2; tabGroup.Layout.Column = 1;

tab1 = uitab(tabGroup, "Title", "① 요구사항 입력");
tab2 = uitab(tabGroup, "Title", "② 시뮬레이션");
tab3 = uitab(tabGroup, "Title", "③ Replay");
tab4 = uitab(tabGroup, "Title", "④ LLM 요약");

% =========================================================================
% Preset scenario constants — used by Tab 1 dropdown + auto-loop env init.
% =========================================================================
SCENARIO_NAMES = [ ...
    "① 맑은 한낮 baseline (정상 작전) — fog 5%·ill 12000lx·noi 0.02", ...
    "② 봄·가을 옅은 시계 (일상 작전) — fog 10%·ill 10000lx·noi 0.03", ...
    "③ 옅은 산 안개 (시계 제한 작전) — fog 18%·ill 9000lx·noi 0.04", ...
    "④ 정오 부분 흐림 (광량 양호 작전) — fog 12%·ill 7500lx·noi 0.05", ...
    "⑤ 이른 오후 옅은 안개 (센서 약간 노후) — fog 15%·ill 8500lx·noi 0.06"];
SCENARIO_FOG = [  5,   10,   18,   12,   15];
SCENARIO_ILL = [12000, 10000, 9000, 7500, 8500];
SCENARIO_NOI = [ 0.02, 0.03, 0.04, 0.05, 0.06];
SCENARIO_NL = [ ...
    "맑은 한낮 baseline — 정상 산악 작전 환경에서 침입자 식별",
    "봄·가을의 옅은 시계 — 일반적 한국 산악 일상 작전",
    "옅은 산 안개와 부분 흐림 — 시계가 약간 제한된 상태",
    "부분 흐림 정오 — 광량은 충분하나 약간의 안개",
    "이른 오후 옅은 안개 — 노후 센서로 약간의 잡음 누적"];

% =========================================================================
% TAB 1 — ① 요구사항 입력
% =========================================================================
tab1Grid = uigridlayout(tab1, [4, 1], ...
    "RowHeight", {80, 30, 130, '1x'}, ...
    "Padding", [40 30 40 30], "RowSpacing", 16);

% Intro / instructions (single string, no array/sprintf to avoid parse edge-cases)
uilabel(tab1Grid, ...
    "Text", "검증 요구사항 입력  -  본 시스템은 PASS/FAIL 경계를 자율 탐색하여 1~10건의 정규화된 회귀 테스트 스위트를 산출합니다. 아래에서 시작 시나리오를 선택한 뒤 [시뮬레이션 시작] 버튼을 누르세요.", ...
    "FontSize", 13, "WordWrap", "on", "VerticalAlignment", "top");

uilabel(tab1Grid, "Text", "▣ 사전 정의 시나리오 (5건 · 모두 PASS 예상)", ...
    "FontWeight", "bold", "FontSize", 13);

% Preset selector — dropdown + parsed env display
presetPanel = uipanel(tab1Grid, ...
    "BackgroundColor", [0.96 0.97 0.99], "BorderType", "line");
presetGrid = uigridlayout(presetPanel, [2, 1], ...
    "RowHeight", {40, 30}, ...
    "Padding", [20 20 20 20], "RowSpacing", 10);
scenarioDropdown = uidropdown(presetGrid, ...
    "Items",     SCENARIO_NAMES, ...
    "ItemsData", 1:numel(SCENARIO_NAMES), ...
    "Value",     1, ...
    "FontSize",  13, ...
    "Tooltip",   "선택 후 ▶ 시뮬레이션 시작 을 누르면 해당 환경으로 boundary search 시작.");
lblParsedEnvVisible = uilabel(presetGrid, ...
    "Text", sprintf("  → fog %.0f %%,  illum %.0f lx,  noise %.2f", ...
        SCENARIO_FOG(1), SCENARIO_ILL(1), SCENARIO_NOI(1)), ...
    "FontSize", 12, "FontColor", [0.30 0.30 0.40]);

% Run button (large, centered) — switches to Tab 2 + starts boundary search
runRow = uigridlayout(tab1Grid, [1, 3], ...
    "ColumnWidth", {'1x', 280, '1x'}, "Padding", [0 0 0 0]);
btnRun = uibutton(runRow, "Text", "▶ 시뮬레이션 시작 (boundary search)", ...
    "BackgroundColor", [0.20 0.65 0.30], "FontColor", "w", ...
    "FontWeight", "bold", "FontSize", 14, ...
    "Tooltip", "선택된 프리셋에서 출발해 PASS↔FAIL 경계를 자동 탐색 (최대 10 iter).");
btnRun.Layout.Column = 2;

% =========================================================================
% TAB 2 — ② 시뮬레이션 (3D scene + EO camera + history + playback controls)
% =========================================================================
tab2Grid = uigridlayout(tab2, [3, 2], ...
    "RowHeight",   {'1x', 140, 100}, ...
    "ColumnWidth", {'1x', '1x'}, ...
    "RowSpacing", 6, "ColumnSpacing", 8, ...
    "Padding", [10 10 10 10]);

% Top — 3D scene (col 1) + EO camera (col 2)
ax3 = uiaxes(tab2Grid);
ax3.Layout.Row = 1; ax3.Layout.Column = 1;
title(ax3, "3D Scene  —  UAV surveillance flight");

ax2 = uiaxes(tab2Grid);
ax2.Layout.Row = 1; ax2.Layout.Column = 2;
title(ax2, "EO Camera  +  detection (live re-rendered)");

% Middle — Case history (full width, scrollable)
histPanel = uipanel(tab2Grid, ...
    "Title", "📑 Case history (전체 · 스크롤)", ...
    "BackgroundColor", [0.97 0.97 0.99], "FontWeight", "bold");
histPanel.Layout.Row = 2; histPanel.Layout.Column = [1 2];
histInner = uigridlayout(histPanel, [1, 1], "Padding", [8 4 8 4]);
lblHistory = uitextarea(histInner, ...
    "Value",      "  (empty — 시뮬레이션을 실행하면 누적됩니다)", ...
    "Editable",   "off", ...
    "FontSize",   12, "FontName", "Consolas", ...
    "BackgroundColor", [0.97 0.97 0.99]);

% Bottom — playback / control row
pbPanel = uipanel(tab2Grid, "BackgroundColor", [0.96 0.96 0.98]);
pbPanel.Layout.Row = 3; pbPanel.Layout.Column = [1 2];
pb = uigridlayout(pbPanel, [2, 5], ...
    "RowHeight", {32, 50}, ...
    "ColumnWidth", {90, 90, 170, 200, '1x'}, ...
    "Padding", [10 4 10 4], "RowSpacing", 4, "ColumnSpacing", 10);

btnStop = uibutton(pb, "Text", "⏹ Stop", ...
    "BackgroundColor", [0.85 0.30 0.20], "FontColor", "w", "Enable", "off");
btnStop.Layout.Row = 1; btnStop.Layout.Column = 1;

btnReset = uibutton(pb, "Text", "⟲ Reset", ...
    "BackgroundColor", [0.60 0.60 0.65], "FontColor", "w");
btnReset.Layout.Row = 1; btnReset.Layout.Column = 2;

autoToggle = uicheckbox(pb, "Text", "Auto-loop (boundary)", ...
    "Value", false, "FontWeight", "bold", ...
    "Tooltip", "PASS↔FAIL 경계 자동 탐색. ▶ Run 누르면 자동 ON.");
autoToggle.Layout.Row = 1; autoToggle.Layout.Column = 3;

iterLbl = uilabel(pb, ...
    "Text", "Iter: 0 / 10  —  manual mode", ...
    "FontWeight", "bold", "HorizontalAlignment", "left");
iterLbl.Layout.Row = 1; iterLbl.Layout.Column = 4;

% Filler / status spacer (avoid chained .Layout assignment to be safe across MATLAB versions)
pbSpacer = uilabel(pb, "Text", "", "HorizontalAlignment", "right");
pbSpacer.Layout.Row = 1; pbSpacer.Layout.Column = 5;

frameSld = uislider(pb, "Limits", [1 max(2,Nt)], "Value", 1, ...
    "MajorTicks", round(linspace(1, max(2,Nt), 5)));
frameSld.Layout.Row = 2; frameSld.Layout.Column = [1 4];

frameLbl = uilabel(pb, ...
    "Text", sprintf("Frame: 1 / %d", Nt), ...
    "HorizontalAlignment", "right", "FontWeight", "bold");
frameLbl.Layout.Row = 2; frameLbl.Layout.Column = 5;

% =========================================================================
% TAB 3 — ③ Replay (정규화 테스트 케이스 + per-case replay controls)
% =========================================================================
tab3Grid = uigridlayout(tab3, [3, 1], ...
    "RowHeight", {'1x', 50, 170}, ...
    "Padding", [10 10 10 10], "RowSpacing", 8);

% Top — normalized test cases panel (the deliverable)
testCasesPanel = uipanel(tab3Grid, ...
    "Title", "🧪 정규화된 테스트 케이스 (옵티마이저 결과)", ...
    "BackgroundColor", [0.96 1.00 0.96], "FontWeight", "bold");
testCasesInner = uigridlayout(testCasesPanel, [1, 1], "Padding", [8 6 8 6]);
lblTestCases = uitextarea(testCasesInner, ...
    "Value",      "  ② 시뮬레이션 탭에서 boundary search를 마친 뒤 아래 '🧪 정규화된 테스트 케이스 생성' 버튼을 눌러 케이스를 산출하세요.", ...
    "Editable",   "off", ...
    "FontSize",   12, "FontName", "Malgun Gothic", ...
    "BackgroundColor", [0.96 1.00 0.96]);

% Middle — case count input + Generate button
genRow = uigridlayout(tab3Grid, [1, 4], ...
    "ColumnWidth", {180, 90, '1x', 280}, ...
    "Padding", [0 0 0 0], "ColumnSpacing", 8);

uilabel(genRow, "Text", "산출할 케이스 개수:", ...
    "FontWeight", "bold", "FontSize", 12, ...
    "HorizontalAlignment", "right");

caseCountSpinner = uispinner(genRow, ...
    "Limits", [0 10], "Value", 0, "Step", 1, ...
    "ValueDisplayFormat", "%d", ...
    "Tooltip", "0 = 기본 스위트(최대 10건, 우선순위순) / 1~10 = 정확히 N건만 산출");

uilabel(genRow, "Text", "  (0 = 기본 스위트 · 1~10 = 우선순위 상위 N건만)", ...
    "FontSize", 11, "FontColor", [0.35 0.35 0.45]);

btnGenerateTests = uibutton(genRow, "Text", "🧪 정규화된 테스트 케이스 생성", ...
    "BackgroundColor", [0.20 0.55 0.30], "FontColor", "w", ...
    "FontWeight", "bold", "FontSize", 13, ...
    "Tooltip", "지금까지 수집된 case에서 정규화된 회귀 테스트 케이스를 산출");

% Bottom — Replay controls (suite + per-case selection + replay)
replayInnerPanel = uipanel(tab3Grid, ...
    "Title", "📂 이전 테스트 재현 (Replay) — 스위트에서 개별 케이스 선택 후 재실행", ...
    "BackgroundColor", [1.00 0.97 0.93], "FontWeight", "bold");
replayInnerGrid = uigridlayout(replayInnerPanel, [4, 3], ...
    "RowHeight",   {28, 32, 32, 22}, ...
    "ColumnWidth", {90, '1x', 130}, ...
    "Padding", [10 6 10 6], "RowSpacing", 6, "ColumnSpacing", 8);

% Row 1 of replay panel — header labels for columns
uilabel(replayInnerGrid, "Text", "", "FontWeight", "bold");
uilabel(replayInnerGrid, "Text", "선택", "FontWeight", "bold", "FontSize", 12);
uilabel(replayInnerGrid, "Text", "동작", "FontWeight", "bold", "FontSize", 12, ...
    "HorizontalAlignment", "center");

% Row 2 — suite selector
lblSuiteLabel = uilabel(replayInnerGrid, "Text", "스위트:", ...
    "FontWeight", "bold", "FontSize", 12, "HorizontalAlignment", "right");
lblSuiteLabel.Layout.Row = 2; lblSuiteLabel.Layout.Column = 1;

replayDropdown = uidropdown(replayInnerGrid, ...
    "Items",     {'  (테스트 스위트 없음 — 먼저 🧪 생성 버튼을 사용하세요)'}, ...
    "ItemsData", {''}, ...
    "Value",     '', ...
    "FontSize",  12, ...
    "Tooltip",   "data/test_suites/ 에 저장된 이전 세션의 정규화 케이스 파일 목록");
replayDropdown.Layout.Row = 2; replayDropdown.Layout.Column = 2;

btnRefreshSuites = uibutton(replayInnerGrid, "Text", "🔄 새로고침", ...
    "BackgroundColor", [0.70 0.70 0.75], "FontColor", "w");
btnRefreshSuites.Layout.Row = 2; btnRefreshSuites.Layout.Column = 3;

% Row 3 — individual case selector + Replay button
lblCaseLabel = uilabel(replayInnerGrid, "Text", "케이스:", ...
    "FontWeight", "bold", "FontSize", 12, "HorizontalAlignment", "right");
lblCaseLabel.Layout.Row = 3; lblCaseLabel.Layout.Column = 1;

caseDropdown = uidropdown(replayInnerGrid, ...
    "Items",     {'  (스위트 선택 후 표시됨)'}, ...
    "ItemsData", {0}, ...
    "Value",     0, ...
    "FontSize",  12, ...
    "Tooltip",   "선택한 스위트의 개별 케이스 — 우선순위 순으로 정렬");
caseDropdown.Layout.Row = 3; caseDropdown.Layout.Column = 2;

btnReplay = uibutton(replayInnerGrid, "Text", "▶ 이 케이스 Replay", ...
    "BackgroundColor", [0.50 0.30 0.70], "FontColor", "w", "FontWeight", "bold");
btnReplay.Layout.Row = 3; btnReplay.Layout.Column = 3;

% Hidden — kept for backward compatibility with existing callbacks/state.
% replay-flow now runs ONE case at a time; the stop button is not needed
% (single-case replays auto-finish in ~38s) but we leave the handle alive
% so onReplayStopClicked / state.replayActive logic can still reference it.
btnReplayStop = uibutton(replayInnerGrid, "Text", "⏹ 중단", ...
    "BackgroundColor", [0.70 0.30 0.30], "FontColor", "w", "Enable", "off", ...
    "Visible", "off");

% Row 4 — status spanning columns
lblReplayStatus = uilabel(replayInnerGrid, ...
    "Text", "  대기 — 스위트 + 케이스 선택 후 ▶ Replay 를 누르세요.", ...
    "FontSize", 11, "FontColor", [0.30 0.25 0.15]);
lblReplayStatus.Layout.Row = 4; lblReplayStatus.Layout.Column = [1 3];

% =========================================================================
% TAB 4 — ④ LLM 임무 종합 요약
% =========================================================================
tab4Grid = uigridlayout(tab4, [2, 1], ...
    "RowHeight", {'1x', 50}, ...
    "Padding", [10 10 10 10], "RowSpacing", 8);

mainSummaryPanel = uipanel(tab4Grid, ...
    "Title", "📋 LLM 임무 종합 요약 (전체 case 누적 분석)", ...
    "BackgroundColor", [0.94 0.96 1.00], "FontWeight", "bold");
mainSummaryInner = uigridlayout(mainSummaryPanel, [1, 1], "Padding", [10 8 10 8]);
lblSummaryLLM = uitextarea(mainSummaryInner, ...
    "Value",      "  📋 아래 'LLM 임무 요약 생성' 버튼을 눌러 지금까지의 전체 case에 대한 종합 보고서를 생성하세요.", ...
    "Editable",   "off", ...
    "FontSize",   12, "FontName", "Malgun Gothic", ...
    "BackgroundColor", [0.94 0.96 1.00]);

btnSummary = uibutton(tab4Grid, "Text", "📋 LLM 임무 요약 생성", ...
    "BackgroundColor", [0.30 0.40 0.65], "FontColor", "w", ...
    "FontWeight", "bold", "FontSize", 13, ...
    "Tooltip", "전체 case history를 LLM에 보내 임무 종합 요약·실패 경계·시사점·권고를 생성");

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

% Hidden state holders for the (now-disabled) natural-language widgets.
% The callbacks below still reference `lblRequirement.Value` and
% `lblParsedEnv.Text`; rather than rewriting them, we keep these widgets
% alive but invisible. The preset dropdown writes the chosen scenario's
% Korean description into lblRequirement so downstream code (LLM mission
% summary, normalized test-case requirement field) still has a meaningful
% requirement string to consume.
hiddenNlPanel = uipanel(fig, "Visible", "off", "Position", [1 1 200 60]);
hiddenNlGrid  = uigridlayout(hiddenNlPanel, [2, 1]);
lblRequirement = uitextarea(hiddenNlGrid, ...
    "Value", char(SCENARIO_NL(1)), "Editable", "off");
lblParsedEnv = uilabel(hiddenNlGrid, "Text", "");

% (The old playback panel (pbPanel) was split and redistributed across the
%  new tabs above: Run→Tab1, Stop/Reset/Auto-loop/frameSld→Tab2,
%  Generate→Tab3, LLM-summary→Tab4. No standalone pbPanel here anymore.)

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
% --- Replay-mode state (drives Row 5's "이전 테스트 재현" workflow) -----
%   replayActive : true while a saved suite is being re-run sequentially.
%   replaySuite  : full payload dict loaded from data/test_suites/*.json.
%   replayIdx    : 1-based index of the case currently in flight.
%   replayResults: cell of compare_replay_result outputs (one per case).
state.replayActive  = false;
state.replaySuite   = [];
state.replayIdx     = 0;
state.replayResults = {};
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

btnRun.ButtonPushedFcn   = @(~,~) onRunRequested();
btnStop.ButtonPushedFcn  = @(~,~) stopRun();
btnReset.ButtonPushedFcn = @(~,~) doReset();
btnSummary.ButtonPushedFcn = @(~,~) onSummaryClicked();
btnGenerateTests.ButtonPushedFcn = @(~,~) onGenerateTestsClicked();
btnRefreshSuites.ButtonPushedFcn = @(~,~) onRefreshSuites();
btnReplay.ButtonPushedFcn        = @(~,~) onReplayClicked();
btnReplayStop.ButtonPushedFcn    = @(~,~) onReplayStopClicked();
replayDropdown.ValueChangedFcn   = @(s,~) onSuiteSelected(s);
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
onRefreshSuites();    % populate replay dropdown from disk at startup
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

        % --- Replay-mode hook: compare result to saved expectation, then
        % schedule the next case (or finish). Mutually exclusive with
        % Auto-loop — replay deliberately turns Auto-loop off before start.
        if state.replayActive
            recordReplayResult();
            % Brief pause so the operator can read the verdict before the
            % sliders animate to the next replay case.
            cancelCooldown();
            state.cooldownTimer = timer( ...
                "StartDelay", 1.2, ...
                "TimerFcn", @(t,~) onReplayCooldownFire(t), ...
                "ExecutionMode", "singleShot");
            start(state.cooldownTimer);
            return;
        end

        % Auto-loop: schedule next iteration after a brief pause so the user
        % can read the verdict before sliders animate to the next case.
        fprintf("[Loop] finalizeRun done — autoToggle.Value=%d, history=%d/%d\n", ...
            logical(autoToggle.Value), numel(state.history), state.maxIter);
        if autoToggle.Value && numel(state.history) < state.maxIter
            fprintf("[Loop] scheduling next auto run...\n");
            scheduleNextAutoRun();
        elseif autoToggle.Value
            autoToggle.Value = false;
            iterLbl.Text = sprintf("Auto: done (%d iters)", numel(state.history));
            fprintf("[Loop] maxIter reached — auto-loop disabled.\n");
        else
            fprintf("[Loop] autoToggle off — stopping at iter %d.\n", ...
                numel(state.history));
        end
    end

    function onReplayCooldownFire(t)
        try, delete(t); catch, end
        state.cooldownTimer = [];
        if ~isvalid(fig), return; end
        if ~state.replayActive, return; end
        advanceReplay();
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
        fprintf("[Loop] scheduleNextAutoRun: iter %d -> %d\n", ...
            numel(state.history), numel(state.history) + 1);
        try
            fprintf("[Loop]   .. step 1: enter try block\n");
            % Show "thinking" status BEFORE the next-case decision. NOTE: no
            % drawnow here — drawnow used to be the suspected hang point. UI
            % text will refresh on the next normal MATLAB event-loop tick.
            hdr.BackgroundColor = [0.30 0.30 0.55];
            fprintf("[Loop]   .. step 2: hdr.BackgroundColor set\n");
            hdr.Text = sprintf( ...
                "  Boundary policy analyzing iter %d -> %d ...", ...
                numel(state.history), numel(state.history) + 1);
            fprintf("[Loop]   .. step 3: hdr.Text set\n");
            iterLbl.Text = sprintf("Auto: %d / %d  -  deciding next case...", ...
                numel(state.history), state.maxIter);
            fprintf("[Loop]   .. step 4: iterLbl.Text set\n");

            fprintf("[Loop]   .. step 5: calling decideNextCase\n");
            [nextEnv, modeStr, analysisStr] = decideNextCase(state.history);
            fprintf("[Loop]   .. step 6: decided fog=%.1f ill=%.0f noi=%.2f mode=%s\n", ...
                nextEnv.fog, nextEnv.ill, nextEnv.noi, modeStr);
            state.nextMode     = modeStr;
            state.nextAnalysis = analysisStr;

            applyEnvToSliders(nextEnv);
            fprintf("[Loop]   .. step 7: applyEnvToSliders done\n");

            state.frameIdx = 1;
            frameSld.Value = 1;
            frameLbl.Text  = sprintf("Frame: 1 / %d", Nt);
            renderFrame();
            fprintf("[Loop]   .. step 8: renderFrame done\n");

            hdr.BackgroundColor = [0.20 0.30 0.65];
            hdr.Text = sprintf( ...
                "  -> next case (%s):  fog=%.0f%%  illum=%.0flx  noise=%.2f", ...
                modeStr, nextEnv.fog, nextEnv.ill, nextEnv.noi);
            iterLbl.Text = sprintf("Auto: %d / %d  -  starting next iter...", ...
                numel(state.history), state.maxIter);

            state.mode = "cooldown";
            btnRun.Enable = "off";
            setCaseControlsEnabled(false);

            cancelCooldown();
            fprintf("[Loop]   .. step 9: creating cooldown timer\n");
            state.cooldownTimer = timer( ...
                "StartDelay", 1.8, ...
                "TimerFcn", @(t,~) onCooldownFire(t), ...
                "ExecutionMode", "singleShot");
            fprintf("[Loop]   .. step 10: timer created, calling start()\n");
            start(state.cooldownTimer);
            fprintf("[Loop]   .. step 11: timer started (will fire in 1.8s)\n");
        catch ME
            fprintf("[Loop] !!! scheduleNextAutoRun ERROR:\n");
            fprintf("        identifier: %s\n", ME.identifier);
            fprintf("        message: %s\n", ME.message);
            for s = 1:numel(ME.stack)
                fprintf("        at %s (line %d)\n", ME.stack(s).name, ME.stack(s).line);
            end
            % Unlock controls so user can recover
            state.mode = "idle";
            btnRun.Enable  = "on";
            setCaseControlsEnabled(true);
            autoToggle.Value = false;
            iterLbl.Text = "Auto: stopped on error (check console)";
        end
    end

    function onCooldownFire(t)
        try
            try, delete(t); catch, end
            state.cooldownTimer = [];
            fprintf("[Loop] onCooldownFire — autoToggle.Value=%d, mode=%s\n", ...
                logical(autoToggle.Value), state.mode);
            if ~isvalid(fig), return; end
            if ~autoToggle.Value
                fprintf("[Loop] cooldown fired but autoToggle is OFF — abort next iter.\n");
                return;
            end
            state.mode = "idle";                         % unlock briefly
            startRun();
        catch ME
            fprintf("[Loop] !!! onCooldownFire ERROR: %s\n        %s\n", ...
                ME.identifier, ME.message);
            for s = 1:numel(ME.stack)
                fprintf("        at %s (line %d)\n", ME.stack(s).name, ME.stack(s).line);
            end
        end
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
                % Asymmetric weight 0.75 (vs 0.65 for the PASS→FAIL probe).
                % Recovery must be RELIABLE — otherwise the next case lands
                % on / past the boundary and FAILs again, never recovering.
                % System purpose = oscillate PASS↔FAIL to find the boundary.
                if ~isempty(passEnv)
                    nextEnv = bisectEnv(lastEnv, passEnv, 0.75);
                    modeStr = "boundary_recover";
                    analysisStr = "FAIL → bisect 75% toward PASS anchor (reliable recovery)";
                elseif ~isempty(marginalEnv)
                    nextEnv = bisectEnv(lastEnv, marginalEnv, 0.75);
                    modeStr = "boundary_recover_to_margin";
                    analysisStr = "FAIL → bisect 75% toward MARGINAL (no PASS anchor yet)";
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

        %{
        % --- Dominant cause line + verdict-count summary --- DISABLED -----
        % These wrote into lblDominant / lblSummary, both of which are now
        % commented out in the UI build. Leaving the original code visible
        % so it can be reinstated if the labels come back.
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
                otherwise
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
        %}
    end


    function updateOpsLog()
        % Per-run LLM narrative stays disabled (per earlier user request),
        % but case history was re-added to the UI as a top-level row-4
        % panel — populate `lblHistory` here. Detailed per-run record is
        % still persisted via autoSaveSession.
        if ~isvalid(fig), return; end
        autoSaveSession();

        % --- Case history list (ALL runs, newest first, scrollable) -----
        N = numel(state.history);
        if N == 0
            lblHistory.Value = "  (empty — 시뮬레이션을 실행하면 누적됩니다)";
            return;
        end
        lines = strings(0, 1);
        lines(end+1) = "  iter  verdict    fog %  illum lx  noise   mAP    사람   차량   mode";
        lines(end+1) = "  ----  ---------  -----  --------  -----   -----  -----  -----  -----------";
        % All entries (no 5-case cap) — uitextarea provides scroll for us.
        for k = N : -1 : 1
            h = state.history(k);
            verdictTag = sprintf("[%s]", h.verdict);
            verdictPad = pad(verdictTag, 9, "right");
            mp = 0; mv = 0;
            if isfield(h, "metric_person"),  mp = h.metric_person;  end
            if isfield(h, "metric_vehicle"), mv = h.metric_vehicle; end
            lines(end+1) = sprintf("  %3d   %s  %5.1f  %7.0f   %5.2f   %.3f  %.3f  %.3f  %s", ...
                h.iter, verdictPad, h.fog, h.ill, h.noi, h.metric, mp, mv, h.mode); %#ok<AGROW>
        end
        lblHistory.Value = cellstr(lines);
        %{
        N = numel(state.history);
        if N == 0
            lblNarrative.Text = "  Run a counterfactual case to generate a narrative.";
            lblHistory.Text   = "  (empty)";
            return;
        end
        last = state.history(end);
        try
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
            narrPy = py.dashboard_step.narrate_edge_case(jsonencode(recPy));
            lblNarrative.Text = string(char(narrPy));
        catch ME
            lblNarrative.Text = buildTemplateNarrative(last);
        end
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
        %}
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
        % Preset picker now directly applies the chosen scenario's env to
        % the hidden sliders (no NL round-trip). Visible label echoes the
        % current values; hidden lblRequirement keeps the Korean
        % description in sync so downstream features (LLM summary, test-
        % case requirement field) still see a meaningful requirement.
        if state.mode == "running"
            return;
        end
        idx = double(src.Value);
        idx = max(1, min(numel(SCENARIO_FOG), idx));
        env = struct("fog", SCENARIO_FOG(idx), ...
                     "ill", SCENARIO_ILL(idx), ...
                     "noi", SCENARIO_NOI(idx));
        applyEnvToSliders(env);
        lblParsedEnvVisible.Text = sprintf( ...
            "  → fog %.0f %%,  illum %.0f lx,  noise %.2f", ...
            env.fog, env.ill, env.noi);
        lblRequirement.Value = char(SCENARIO_NL(idx));
        renderFrame();
    end

    function onRunRequested()
        % ▶ Run starts a *boundary-search session*, not a single sim.
        % Auto-loop is force-enabled here so the system iterates PASS↔FAIL
        % to characterise the failure envelope (up to state.maxIter cases).
        % Also switch to the simulation tab so the user sees the sim view.
        if ~autoToggle.Value
            autoToggle.Value = true;
            iterLbl.Text = sprintf("Auto: %d / %d iters", ...
                numel(state.history), state.maxIter);
            fprintf("[Run] Auto-loop 자동 활성화 — boundary search 시작 (최대 %d iter).\n", ...
                state.maxIter);
        end
        % Auto-switch to ② 시뮬레이션 tab so the operator watches the run
        try, tabGroup.SelectedTab = tab2; catch, end
        startRun();
    end

    function onGenerateTestsClicked()
        % Call Python optimizer to produce 1~10 normalized test cases from
        % the collected PASS/MARGINAL/FAIL history. Output is rendered in
        % the top-level testCasesPanel (lblTestCases uitextarea).
        prevText = btnGenerateTests.Text;
        btnGenerateTests.Enable = "off";
        btnGenerateTests.Text   = "⏳ 옵티마이저 실행 중...";
        lblTestCases.Value = "  ⏳ 정규화된 테스트 케이스 생성 중...";
        drawnow;
        try
            nl = lblRequirement.Value;
            if iscell(nl), nl = strjoin(nl, newline); end
            histJson = jsonencode(arrayfun(@(h) struct( ...
                "iter",            h.iter, ...
                "fog",             h.fog, ...
                "ill",             h.ill, ...
                "noi",             h.noi, ...
                "metric",          h.metric, ...
                "verdict",         char(h.verdict)), ...
                state.history, "UniformOutput", false));
            % Read desired count: 0 = 기본(10) / 1~10 = 정확히 N건
            userCount = round(caseCountSpinner.Value);
            if userCount <= 0
                maxCases = 10;   % default suite
                fprintf("[Gen] user count = 0 → 기본 스위트 (최대 10건)\n");
            else
                maxCases = min(10, max(1, userCount));
                fprintf("[Gen] user count = %d → 우선순위 상위 %d건 산출\n", ...
                    userCount, maxCases);
            end
            res = py.dashboard_step.generate_normalized_test_cases( ...
                histJson, string(nl), int32(maxCases));
            d = struct(res);
            txt = string(char(d.text));
            lines = splitlines(txt);
            lblTestCases.Value = cellstr(lines);
            % Auto-refresh replay dropdown so the freshly-exported suite
            % shows up without the user having to click 🔄 새로고침.
            onRefreshSuites();
        catch ME
            lblTestCases.Value = sprintf("정규화된 테스트 케이스 생성 실패: %s", ME.message);
        end
        btnGenerateTests.Enable = "on";
        btnGenerateTests.Text   = prevText;
        % We're already on Tab 3 (since the Generate button is here). No
        % auto-switch needed, but ensure the panel is visible.
        try, tabGroup.SelectedTab = tab3; catch, end
    end

    function autoSaveSession()
        % Per-session disk persistence. Every finalizeRun appends to
        % data/sessions/session_<launch-time>.json so the full record (env,
        % verdict, per-class mAP, unique intruders) survives MATLAB restart
        % and can be reloaded for offline analysis or test-case replay.
        try
            if ~isfield(state, "sessionFile") || isempty(state.sessionFile)
                ts = datestr(now, "yyyymmdd_HHMMSS");
                sessDir = fullfile(pwd, "data", "sessions");
                if ~isfolder(sessDir), mkdir(sessDir); end
                state.sessionFile = fullfile(sessDir, "session_" + ts + ".json");
            end
            recs = cell(1, numel(state.history));
            for k = 1:numel(state.history)
                h = state.history(k);
                recs{k} = struct( ...
                    "iter",                h.iter, ...
                    "fog",                 h.fog, ...
                    "ill",                 h.ill, ...
                    "noi",                 h.noi, ...
                    "metric",              h.metric, ...
                    "metric_person",       h.metric_person, ...
                    "metric_vehicle",      h.metric_vehicle, ...
                    "verdict",             char(h.verdict), ...
                    "n_intruders_total",   h.n_intruders_total, ...
                    "n_intruders_seen",    h.n_intruders_seen, ...
                    "n_intruders_detected",h.n_intruders_detected, ...
                    "n_intruders_missed",  h.n_intruders_missed, ...
                    "mode",                h.mode, ...
                    "analysis",            h.analysis);
            end
            fid = fopen(state.sessionFile, "w", "n", "UTF-8");
            if fid > 0
                fwrite(fid, jsonencode(recs, "PrettyPrint", true));
                fclose(fid);
            end
        catch ME
            fprintf("[Session] save failed: %s\n", ME.message);
        end
    end

    function onRefreshSuites()
        % Rescan data/test_suites/ and update the replay dropdown items.
        % Uses cell-of-char-vectors throughout — the previous version mixed
        % string scalars and cellstr which some MATLAB releases reject on
        % uidropdown (silent failure mode: dropdown stays empty).
        try
            pyList = py.dashboard_step.list_test_suites();
            cellList = cell(pyList);
            nFound = numel(cellList);
            if nFound == 0
                replayDropdown.Items     = {'  (테스트 스위트 없음 — 먼저 🧪 생성 버튼을 사용하세요)'};
                replayDropdown.ItemsData = {''};
                replayDropdown.Value     = '';
                lblReplayStatus.Text = "  대기 — 저장된 스위트가 없습니다. 시뮬레이션 후 🧪 정규화된 테스트 케이스 생성을 누르세요.";
                fprintf("[Refresh] data/test_suites/ 스캔 → 0건 (디스크에 파일 없거나 디렉터리 미존재).\n");
                return;
            end
            namesCell = cell(1, nFound);
            pathsCell = cell(1, nFound);
            for k = 1:nFound
                d = struct(cellList{k});
                fname  = char(d.filename);
                req    = char(d.requirement);
                ncases = double(d.n_cases);
                if length(req) > 30
                    req = [req(1:30) char(8230)];   % … ellipsis
                end
                namesCell{k} = sprintf('%s  ·  %d cases  ·  %s', fname, ncases, req);
                pathsCell{k} = char(d.path);
            end
            % Order matters: Items + ItemsData first, THEN Value pinned to a
            % member of ItemsData. Setting Value before the new ItemsData
            % is in place causes MATLAB to keep the old (stale) value.
            replayDropdown.Items     = namesCell;
            replayDropdown.ItemsData = pathsCell;
            replayDropdown.Value     = pathsCell{1};
            lblReplayStatus.Text = sprintf("  %d개의 스위트 발견 — 케이스 선택 후 ▶ Replay 를 누르세요.", nFound);
            fprintf("[Refresh] data/test_suites/ 스캔 → %d건 발견.\n", nFound);
            % Auto-populate case dropdown for the first suite so user can
            % replay immediately without having to re-click the suite dropdown.
            onSuiteSelected([]);
        catch ME
            lblReplayStatus.Text = sprintf("  새로고침 실패: %s", ME.message);
            fprintf("[Refresh] 실패: %s\n", ME.message);
        end
    end

    function onSuiteSelected(~)
        % Suite dropdown changed → load that suite's cases into caseDropdown.
        % Cases are listed in PRIORITY order as stored in the JSON (already
        % sorted by the optimizer: WORST_FAIL → BEST_PASS → BOUNDARY_* →
        % MARGINAL → DIVERSITY).
        try
            selPath = string(replayDropdown.Value);
            if strlength(selPath) == 0
                caseDropdown.Items     = {'  (스위트 선택 후 표시됨)'};
                caseDropdown.ItemsData = {0};
                caseDropdown.Value     = 0;
                return;
            end
            pyPayload = py.dashboard_step.load_test_suite(selPath);
            payload   = struct(pyPayload);
            if isfield(payload, "error")
                caseDropdown.Items     = {sprintf('  (로드 실패: %s)', string(char(payload.error)))};
                caseDropdown.ItemsData = {0};
                caseDropdown.Value     = 0;
                return;
            end
            cases = cell(payload.cases);
            if isempty(cases)
                caseDropdown.Items     = {'  (스위트에 case 없음)'};
                caseDropdown.ItemsData = {0};
                caseDropdown.Value     = 0;
                return;
            end
            % Build display labels — show case # + verdict + label
            namesCell = cell(1, numel(cases));
            idxData   = cell(1, numel(cases));
            for k = 1:numel(cases)
                c = struct(cases{k});
                lbl = "(unnamed)";
                if isfield(c, "label"), lbl = string(char(c.label)); end
                vd  = "?";
                if isfield(c, "verdict"), vd = string(char(c.verdict)); end
                cat = "";
                if isfield(c, "coverage_category")
                    cat = " [" + string(char(c.coverage_category)) + "]";
                end
                namesCell{k} = char(sprintf('Case %d%s — %s → 예상: %s', ...
                    k, cat, lbl, vd));
                idxData{k} = k;
            end
            caseDropdown.Items     = namesCell;
            caseDropdown.ItemsData = idxData;
            caseDropdown.Value     = idxData{1};
            fprintf("[Suite] '%s' 로드 → %d 케이스 목록 채움\n", ...
                replayDropdown.Items{find(cellfun(@(p) strcmp(p, char(selPath)), replayDropdown.ItemsData), 1)}, ...
                numel(cases));
        catch ME
            caseDropdown.Items     = {sprintf('  (로드 실패: %s)', ME.message)};
            caseDropdown.ItemsData = {0};
            caseDropdown.Value     = 0;
            fprintf("[Suite] 로드 실패: %s\n", ME.message);
        end
    end

    function onReplayClicked()
        fprintf("[Replay] onReplayClicked: button pressed\n");
        % Pre-flight checks. Recovery: if state.replayActive is true but no
        % suite is loaded (e.g. a previous click errored mid-setup), we
        % reset to allow retry instead of permanently blocking.
        if state.replayActive && ~isempty(state.replaySuite)
            lblReplayStatus.Text = "  이미 replay 진행 중. ⏹ Replay 중단 후 다시 시도하세요.";
            fprintf("[Replay]   already active — abort\n");
            return;
        end
        if state.replayActive
            fprintf("[Replay]   replayActive stuck on True but suite empty — auto-reset\n");
            state.replayActive = false;
        end
        if state.mode == "running"
            lblReplayStatus.Text = "  진행 중인 case 완료 후 replay를 시도하세요.";
            fprintf("[Replay]   mode=running — abort\n");
            return;
        end
        if state.mode == "cooldown"
            % Auto-loop is mid-cooldown; cancel it so replay can take over
            fprintf("[Replay]   mode=cooldown — cancelling auto cooldown\n");
            cancelCooldown();
            state.mode = "idle";
        end
        selPath = string(replayDropdown.Value);
        fprintf("[Replay]   selPath='%s' (len=%d)\n", selPath, strlength(selPath));
        if strlength(selPath) == 0
            lblReplayStatus.Text = "  스위트가 선택되지 않았습니다. 🔄 새로고침 후 드롭다운에서 선택하세요.";
            return;
        end
        % Read which individual case the user picked
        caseIdx = double(caseDropdown.Value);
        if isempty(caseIdx) || caseIdx < 1
            lblReplayStatus.Text = "  케이스가 선택되지 않았습니다. 위 드롭다운에서 케이스 하나를 선택하세요.";
            fprintf("[Replay]   no case selected (caseIdx=%g)\n", caseIdx);
            return;
        end
        try
            fprintf("[Replay]   .. step 1: calling py.dashboard_step.load_test_suite\n");
            pyPayload = py.dashboard_step.load_test_suite(selPath);
            fprintf("[Replay]   .. step 2: converting py payload to MATLAB struct\n");
            payload   = struct(pyPayload);
            if isfield(payload, "error")
                msg = string(char(payload.error));
                lblReplayStatus.Text = sprintf("  로드 실패: %s", msg);
                fprintf("[Replay]   .. payload error: %s\n", msg);
                return;
            end
            fprintf("[Replay]   .. step 3: extracting cases list\n");
            cases = cell(payload.cases);
            fprintf("[Replay]   .. step 4: %d case(s) in suite, user picked #%d\n", ...
                numel(cases), caseIdx);
            if isempty(cases) || caseIdx > numel(cases)
                lblReplayStatus.Text = sprintf("  선택된 케이스 인덱스 %d 가 스위트 범위(%d)를 벗어남.", ...
                    caseIdx, numel(cases));
                return;
            end
            fprintf("[Replay]   .. step 5: extracting only case #%d (single-case replay mode)\n", caseIdx);
            % Single-case replay — state.replaySuite contains ONLY the
            % chosen case. advanceReplay will run it once, finalizeRun will
            % call recordReplayResult → cooldown → advanceReplay sees idx
            % == length and calls finishReplay.
            state.replaySuite   = { struct(cases{caseIdx}) };
            state.replayIdx     = 0;
            state.replayResults = {};
            state.replayActive  = true;
            fprintf("[Replay]   .. step 6: state.replayActive=true, single case cached\n");
            % Disable auto-loop during replay (they conflict)
            if autoToggle.Value
                fprintf("[Replay]   .. step 7: disabling auto-loop\n");
                autoToggle.Value = false;
            end
            % UI lock
            btnReplay.Enable     = "off";
            btnReplayStop.Enable = "on";
            btnRefreshSuites.Enable = "off";
            replayDropdown.Enable = "off";
            caseDropdown.Enable   = "off";
            % Show banner in the test-cases panel
            lblTestCases.Value = sprintf("[Replay] Case %d 단독 재실행 중 ...", caseIdx);
            fprintf("[Replay]   .. step 8: calling advanceReplay() for the single picked case\n");
            advanceReplay();
        catch ME
            lblReplayStatus.Text = sprintf("  로드 실패: %s", ME.message);
            fprintf("[Replay] !!! ERROR:\n");
            fprintf("        identifier: %s\n", ME.identifier);
            fprintf("        message: %s\n", ME.message);
            for s = 1:numel(ME.stack)
                fprintf("        at %s (line %d)\n", ME.stack(s).name, ME.stack(s).line);
            end
            % Reset so user can retry
            state.replayActive = false;
            state.replaySuite  = [];
            btnReplay.Enable        = "on";
            btnReplayStop.Enable    = "off";
            btnRefreshSuites.Enable = "on";
            replayDropdown.Enable   = "on";
            caseDropdown.Enable     = "on";
        end
    end

    function onReplayStopClicked()
        if ~state.replayActive, return; end
        state.replayActive = false;
        cancelCooldown();
        btnReplay.Enable        = "on";
        btnReplayStop.Enable    = "off";
        btnRefreshSuites.Enable = "on";
        replayDropdown.Enable   = "on";
        caseDropdown.Enable     = "on";
        lblReplayStatus.Text = sprintf("  중단됨 — %d/%d case 완료.", ...
            state.replayIdx, numel(state.replaySuite));
        finishReplay();
    end

    function advanceReplay()
        fprintf("[Replay] advanceReplay called: idx=%d, total=%d, active=%d\n", ...
            state.replayIdx, numel(state.replaySuite), logical(state.replayActive));
        try
            if ~state.replayActive
                fprintf("[Replay]   replayActive=false, abort advance\n");
                return;
            end
            if state.replayIdx >= numel(state.replaySuite)
                fprintf("[Replay]   all cases done, calling finishReplay\n");
                finishReplay();
                return;
            end
            state.replayIdx = state.replayIdx + 1;
            c = state.replaySuite{state.replayIdx};
            env = struct( ...
                "fog", double(c.fog), ...
                "ill", double(c.ill), ...
                "noi", double(c.noi));
            fprintf("[Replay]   case %d: fog=%.1f ill=%.0f noi=%.2f\n", ...
                state.replayIdx, env.fog, env.ill, env.noi);
            applyEnvToSliders(env);
            % Reset playback head BEFORE renderFrame — otherwise frameIdx is
            % still at Nt+1 from the previous case's finalizeRun, and
            % renderFrame indexes uav_xyz / gtBB out of bounds. (Matches
            % what scheduleNextAutoRun does for auto-loop.)
            state.frameIdx = 1;
            frameSld.Value = 1;
            frameLbl.Text  = sprintf("Frame: 1 / %d", Nt);
            % Provenance for the upcoming history record
            labelStr = "";
            if isfield(c, "label"), labelStr = string(char(c.label)); end
            state.nextMode     = sprintf("replay_%d_of_%d", ...
                state.replayIdx, numel(state.replaySuite));
            state.nextAnalysis = char(sprintf("Replay: %s", labelStr));
            lblReplayStatus.Text = sprintf( ...
                "  Replay %d/%d - %s  (fog=%.0f%%, illum=%.0flx, noise=%.2f)", ...
                state.replayIdx, numel(state.replaySuite), labelStr, ...
                env.fog, env.ill, env.noi);
            renderFrame();
            fprintf("[Replay]   .. calling startRun()\n");
            startRun();
            fprintf("[Replay]   .. startRun() returned (sim now running)\n");
        catch ME
            fprintf("[Replay] !!! advanceReplay ERROR:\n");
            fprintf("        identifier: %s\n", ME.identifier);
            fprintf("        message: %s\n", ME.message);
            for s = 1:numel(ME.stack)
                fprintf("        at %s (line %d)\n", ME.stack(s).name, ME.stack(s).line);
            end
            state.replayActive = false;
            btnReplay.Enable        = "on";
            btnReplayStop.Enable    = "off";
            btnRefreshSuites.Enable = "on";
            replayDropdown.Enable   = "on";
            lblReplayStatus.Text = sprintf("  Replay 오류로 중단 (콘솔 확인): %s", ME.message);
        end
    end

    function recordReplayResult()
        % Called from finalizeRun after a replay case completes. Compares
        % the actual outcome to the saved expectation via Python helper.
        if ~state.replayActive, return; end
        if isempty(state.history), return; end
        if state.replayIdx < 1 || state.replayIdx > numel(state.replaySuite), return; end
        last = state.history(end);
        c    = state.replaySuite{state.replayIdx};
        try
            cmpPy = py.dashboard_step.compare_replay_result( ...
                py.dict(c), ...
                char(last.verdict), ...
                last.metric);
            d = struct(cmpPy);
            state.replayResults{end+1} = struct( ...
                "idx",      state.replayIdx, ...
                "expected", string(char(c.expected_verdict)), ...
                "actual",   string(last.verdict), ...
                "metric",   last.metric, ...
                "match",    logical(d.match), ...
                "drift",    logical(d.drift), ...
                "regression", logical(d.regression), ...
                "summary",  string(char(d.summary)));
        catch ME
            % Python compare unavailable — do a rule-based comparison in
            % MATLAB so replay still produces a report even offline.
            accept = string(c.acceptable_verdicts);
            if iscell(c.acceptable_verdicts)
                accept = string(c.acceptable_verdicts);
            end
            isMatch = any(string(last.verdict) == accept);
            mn = double(c.metric_min);
            mx = double(c.metric_max);
            inBand = (last.metric >= mn) && (last.metric <= mx);
            tag = "✗ REGRESSION";
            if isMatch && inBand,      tag = "✓ OK";
            elseif isMatch,            tag = "△ DRIFT";
            end
            state.replayResults{end+1} = struct( ...
                "idx",        state.replayIdx, ...
                "expected",   string(char(c.expected_verdict)), ...
                "actual",     string(last.verdict), ...
                "metric",     last.metric, ...
                "match",      isMatch, ...
                "drift",      isMatch && ~inBand, ...
                "regression", ~isMatch, ...
                "summary",    sprintf("%s (fallback): %s", tag, ME.message));
        end
    end

    function finishReplay()
        % Build a one-shot summary of all replay results and dump it into
        % the test-cases panel. Re-enables the UI controls.
        results = state.replayResults;
        state.replayActive = false;
        btnReplay.Enable        = "on";
        btnReplayStop.Enable    = "off";
        btnRefreshSuites.Enable = "on";
        replayDropdown.Enable   = "on";
        caseDropdown.Enable     = "on";

        if isempty(results)
            lblTestCases.Value = "  (Replay 결과 없음)";
            return;
        end
        nOK = 0; nDrift = 0; nReg = 0;
        for k = 1:numel(results)
            r = results{k};
            if r.regression,    nReg   = nReg + 1;
            elseif r.drift,     nDrift = nDrift + 1;
            else,               nOK    = nOK + 1;
            end
        end
        lines = strings(0, 1);
        lines(end+1) = sprintf("▣ Replay 결과 — 총 %d cases", numel(results));
        lines(end+1) = sprintf("  ✓ OK %d  ·  △ DRIFT %d  ·  ✗ REGRESSION %d", nOK, nDrift, nReg);
        if nReg > 0
            lines(end+1) = "  ⚠ Regression 발생 — 새 시스템에서 기대된 verdict 가 재현되지 않았습니다.";
        elseif nDrift > 0
            lines(end+1) = "  ⓘ Drift 감지 — verdict 는 유지되었으나 mAP 가 허용 범위를 벗어났습니다.";
        else
            lines(end+1) = "  ✓ 전체 회귀 통과 — 시스템 envelope 가 원본 세션과 일치합니다.";
        end
        lines(end+1) = "";
        for k = 1:numel(results)
            r = results{k};
            lines(end+1) = sprintf("[Case %d] %s", r.idx, r.summary);  %#ok<AGROW>
        end
        lblTestCases.Value = cellstr(lines);
        lblReplayStatus.Text = sprintf( ...
            "  Replay 완료 — OK %d / DRIFT %d / REGRESSION %d (결과는 위 패널에 표시)", ...
            nOK, nDrift, nReg);
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
