function find_sim3d_blocks()
% find_sim3d_blocks — R2025b에서 사용 가능한 Sim3D 블록 전체 목록 출력.
%   build_mountain_uav_unreal.m이 어떤 라이브러리/블록 이름을 써야 할지
%   여기 결과를 보고 결정.

fprintf("\n=== Sim3D 관련 라이브러리 자동 발견 ===\n\n");

% 1) 알려진 Sim3D 라이브러리 후보 모두 로드 시도
candidate_libs = [
    "drivingsim3d"
    "uavsimulation3d"
    "uav3dsim"
    "uavlib3d"
    "vdynblks"
    "sim3d_lib"
    "sim3d"
    "drivingsim3dvr"
    "uavsim3d"
    "shared_3d"
    "uav3dlib"
];

available_libs = strings(0);
for k = 1:numel(candidate_libs)
    libname = candidate_libs(k);
    try
        load_system(char(libname));
        fprintf("[OK]   %s   (loaded)\n", libname);
        available_libs(end+1) = libname; %#ok<AGROW>
    catch
        fprintf("[skip] %s   (not found)\n", libname);
    end
end

% 2) 사용 가능한 라이브러리 안의 블록 모두 나열 (UAV/Camera/Vehicle/Pedestrian 키워드)
fprintf("\n=== 키워드 매칭 블록 ===\n");
keywords = ["UAV", "Camera", "Vehicle", "Character", "Pedestrian", "Scene"];

for libname = available_libs
    fprintf("\n--- 라이브러리: %s ---\n", libname);
    try
        blocks = find_system(char(libname), "FollowLinks", "on", ...
            "LookUnderMasks", "all", "Type", "block");
        for kw = keywords
            matches = blocks(contains(blocks, kw, "IgnoreCase", true));
            if ~isempty(matches)
                fprintf("  [%s]\n", kw);
                for m = matches(:).'
                    fprintf("    %s\n", m{1});
                end
            end
        end
    catch ME
        fprintf("  (블록 목록 가져오기 실패: %s)\n", ME.message);
    end
end

% 3) Add-On이 어떻게 등록되어 있는지
fprintf("\n=== 설치된 Sim3D / Unreal 관련 Add-On ===\n");
try
    addons = matlab.addons.installedAddons;
    rows = addons(contains(addons.Name, "Unreal", "IgnoreCase", true) | ...
                  contains(addons.Name, "3D",     "IgnoreCase", true) | ...
                  contains(addons.Name, "Sim3D",  "IgnoreCase", true), :);
    disp(rows);
catch
    fprintf("(Add-On 정보 조회 실패)\n");
end

fprintf("\n=== 끝 ===\n");
fprintf("위 결과에서 'UAV' 또는 'Drone' 또는 'Quad'를 포함한 블록 경로를 알려주세요.\n");
end
