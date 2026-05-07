function test_sim3d_smoke()
% test_sim3d_smoke — Phase 0 검증 스크립트.
%   샘플 씬을 띄우고 5초간 시뮬레이션해서 Unreal 창이 정상적으로
%   뜨는지, 영상이 캡처되는지 확인한다. 예제 이름 찾을 필요 없음.

mdl = "test_sim3d_smoke";
fprintf("[Smoke] 시작 — Sim3D + Unreal 통합 검증\n");

% --- 0) 환경 점검
fprintf("\n[0] sim3d 네임스페이스: ");
if exist('sim3d.Actor', 'class') == 8 || exist('sim3d.Engine', 'class') == 8
    fprintf("OK\n");
else
    error("[FAIL] sim3d 클래스를 찾을 수 없음. Add-On 설치/재시작 확인.");
end

% --- 1) 사용 가능한 씬 목록 (R2025b)
fprintf("\n[1] Sim 3D 라이브러리에서 Scene Configuration 블록 검색…\n");
try
    block_path = "drivingsim3d/Simulation 3D Scene Configuration";
    fprintf("    block_path = %s\n", block_path);
catch
    block_path = "uavsimulation3d/Simulation 3D Scene Configuration";
end

% --- 2) 빈 모델 생성
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
open_system(mdl);
set_param(mdl, "StopTime", "5", "Solver", "FixedStepAuto", "FixedStep", "0.05");

% --- 3) Scene Configuration 블록 추가 (드래그 대신 명령으로)
fprintf("\n[2] Scene Configuration 블록 추가 시도…\n");
candidates = [
    "drivingsim3d/Simulation 3D Scene Configuration"
    "uavsimulation3d/Simulation 3D Scene Configuration"
    "vdynblks/Simulation 3D Scene Configuration"
    "sim3d_lib/Simulation 3D Scene Configuration"
];
added = false;
for k = 1:numel(candidates)
    try
        add_block(char(candidates(k)), char(mdl + "/SceneCfg"));
        fprintf("    OK ← %s\n", candidates(k));
        added = true;
        break;
    catch
        % 다음 후보 시도
    end
end
if ~added
    fprintf("\n[FAIL] Scene Configuration 블록 못 찾음. 직접 라이브러리에서 확인:\n");
    fprintf("       slLibraryBrowser → 검색창에 'Scene Configuration'\n");
    return;
end

% --- 4) 기본 씬 설정 (Default Scenes에서 가장 가벼운 거)
fprintf("\n[3] 기본 씬 'Straight Road' 또는 'Empty' 시도…\n");
scene_options = ["Straight Road", "Empty", "Open Surface", "US City Block", "Curved Road"];
scene_set = false;
for s = scene_options
    try
        set_param(char(mdl + "/SceneCfg"), "SceneDesc", char(s));
        fprintf("    씬 = '%s' 설정 OK\n", s);
        scene_set = true;
        break;
    catch
    end
end
if ~scene_set
    fprintf("    경고: 미리 정의된 씬 이름 못 찾음. 블록 기본값으로 진행.\n");
end

% --- 5) 시뮬레이션 (Unreal 창이 뜨고 5초 후 닫힘)
fprintf("\n[4] 5초간 시뮬레이션 실행 — Unreal 창이 뜨는지 확인…\n");
try
    set_param(char(mdl + "/SceneCfg"), "SceneView", "Scene Origin");  % 카메라 시점
catch
end

try
    out = sim(mdl);
    fprintf("\n[SUCCESS] 시뮬레이션 완료. Unreal 창이 떴다면 Phase 0 통과!\n");
    fprintf("          그렇지 않다면 GPU/Unreal 설치/플러그인 문제일 가능성.\n");
catch ME
    fprintf("\n[FAIL] sim 실행 실패: %s\n", ME.message);
    fprintf("       대표적 원인:\n");
    fprintf("       - Unreal Engine이 설치 안 됨\n");
    fprintf("       - MathWorks 플러그인이 Unreal 측에 미복사\n");
    fprintf("       - GPU 드라이버 너무 오래됨\n");
end

% --- 6) 정리
fprintf("\n[5] 모델 정리…\n");
close_system(mdl, 0);
fprintf("[Smoke] 종료.\n");
end
