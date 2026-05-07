function find_uav_blocks_deep()
% find_uav_blocks_deep — R2025b에서 UAV/Drone 블록 + 사용 가능한 씬 전체 탐색
%
%   세 가지를 한 번에 찾는다:
%     1) UAV Toolbox Interface for Unreal Engine이 설치한 블록의 실제 라이브러리 경로
%     2) Simulation 3D Scene Configuration 블록이 지원하는 모든 씬 이름
%     3) Quadcopter / Drone / UAV 키워드를 포함한 모든 블록

fprintf("\n=== Part 1: UAV Toolbox Interface 설치 위치 추적 ===\n\n");

% 설치된 add-on 정보에서 UAV 패키지 폴더 찾기
addons = matlab.addons.installedAddons;
uav_rows = addons(contains(addons.Name, "UAV", "IgnoreCase", true), :);
disp(uav_rows);

% R2025b에서 UAV Interface for Unreal이 설치하는 블록 라이브러리 후보 더 넓게
candidate_libs = [
    "uavsim3d"; "uav3dsim"; "uavInterface"; "uavtoolbox"; "uav_toolbox"
    "uavsimulation3d"; "uav3dsimlib"; "uavsim3dlib"; "uavinterfacelib"
    "uavlib3d"; "uavlib"; "drivingsim3d"; "vdynblks"
    "shared3d"; "sim3d_shared"; "spkg_uav3d"; "spkg_uav"
    "matlab_unreal"; "matlabwalker"; "uavphotoreal"
];

available = strings(0);
for k = 1:numel(candidate_libs)
    libname = candidate_libs(k);
    try
        load_system(char(libname));
        fprintf("[OK]   %s\n", libname);
        available(end+1) = libname; %#ok<AGROW>
    catch
        fprintf("[skip] %s\n", libname);
    end
end

fprintf("\n=== Part 2: Quadcopter / Drone / UAV 키워드 블록 검색 ===\n");

drone_keywords = ["UAV", "Drone", "Quad", "Multi", "Helic", "Aerial", "Aircraft"];
for libname = available(:).'
    fprintf("\n--- %s ---\n", libname);
    try
        blocks = find_system(char(libname), "FollowLinks", "on", ...
            "LookUnderMasks", "all", "Type", "block");
        any_match = false;
        for kw = drone_keywords
            matches = blocks(contains(blocks, kw, "IgnoreCase", true));
            % subblock(슬래시 깊이 > 1)는 제외해서 최상위 블록만 표시
            top_matches = matches(arrayfun(@(s) count(s{1}, '/') == 1, matches));
            if ~isempty(top_matches)
                fprintf("  [%s]\n", kw);
                for m = top_matches(:).'
                    fprintf("    %s\n", m{1});
                end
                any_match = true;
            end
        end
        if ~any_match
            fprintf("  (no drone-related top-level block)\n");
        end
    catch ME
        fprintf("  search failed: %s\n", ME.message);
    end
end

fprintf("\n=== Part 3: Sim 3D Scene Configuration이 지원하는 씬 목록 ===\n\n");

% 임시 모델에 Scene Cfg 블록 추가 후 SceneDesc 파라미터의 enum 옵션 추출
mdl = "scene_enum_probe";
try
    if bdIsLoaded(mdl), close_system(mdl, 0); end
    new_system(mdl);
    add_block("drivingsim3d/Simulation 3D Scene Configuration", char(mdl + "/cfg"));
    blk = char(mdl + "/cfg");

    % Mask object에서 SceneDesc/SceneName 파라미터 옵션 list 가져오기
    pinfo = get_param(blk, "DialogParameters");
    pnames = fieldnames(pinfo);
    fprintf("블록 파라미터 전체:\n");
    for k = 1:numel(pnames)
        p = pinfo.(pnames{k});
        fprintf("  %-30s  type=%s\n", pnames{k}, p.Type);
        if strcmp(p.Type, "enum") && isfield(p, 'Enum')
            fprintf("    옵션: ");
            fprintf("%s | ", p.Enum{:});
            fprintf("\n");
        end
    end

    close_system(mdl, 0);
catch ME
    fprintf("Scene 옵션 탐색 실패: %s\n", ME.message);
end

fprintf("\n=== Part 4: 드론/항공 관련 예제 모델 검색 ===\n\n");

try
    % UAV Toolbox 데모 폴더 위치
    uav_root = fullfile(matlabroot, "toolbox", "uav");
    if ~isfolder(uav_root)
        uav_root = fullfile(matlabroot, "toolbox", "shared", "uav");
    end
    if isfolder(uav_root)
        fprintf("UAV Toolbox 루트: %s\n", uav_root);
        % 항공 관련 .slx 또는 .mlx 예제 찾기
        slx_files = dir(fullfile(uav_root, "**", "*Quadrotor*.slx"));
        slx_files = [slx_files; dir(fullfile(uav_root, "**", "*Drone*.slx"))];
        slx_files = [slx_files; dir(fullfile(uav_root, "**", "*Photoreal*.slx"))];
        for k = 1:numel(slx_files)
            fprintf("  %s\n", fullfile(slx_files(k).folder, slx_files(k).name));
        end
    end

    % UAV Interface for Unreal 설치 폴더
    intf_root = fullfile(matlabroot, "toolbox", "shared", "sim3dprojects");
    if isfolder(intf_root)
        fprintf("\nSim3D project root: %s\n", intf_root);
        sub = dir(intf_root);
        for k = 1:numel(sub)
            if sub(k).isdir && ~startsWith(sub(k).name, '.')
                fprintf("  - %s\n", sub(k).name);
            end
        end
    end

catch ME
    fprintf("탐색 실패: %s\n", ME.message);
end

fprintf("\n=== Part 5: 라이브러리 브라우저에서 UAV Toolbox 트리 ===\n\n");
fprintf("아래 명령으로 GUI 라이브러리 브라우저를 열고 'UAV Toolbox' 트리 펼쳐보세요:\n");
fprintf("  slLibraryBrowser\n");
fprintf("→ 'UAV Toolbox' 또는 'Aerospace Blockset'에서 'Simulation 3D Quadrotor' 같은 게 보이면\n");
fprintf("   그 블록을 우클릭 → 'Block Path to Clipboard' 로 정확한 경로 알려주세요.\n");

fprintf("\n=== 끝 ===\n");
end
