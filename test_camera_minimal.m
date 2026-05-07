function test_camera_minimal()
% test_camera_minimal — Minimum viable test: Empty Grass scene + 1 camera
%   high above origin looking down. Tests 4 rotation candidates to find
%   which actually points the camera at the ground.
%
%   This eliminates ZalaZONE/UAV/Pedestrian variables — if we can get
%   ANY view of the ground, we'll know the right convention.
%
%   Outputs PNG to assets/test_cam_*.png for each rotation tried.

mdl = "test_cam_min";
if bdIsLoaded(mdl), close_system(mdl, 0); end
if isfile(mdl + ".slx"), delete(mdl + ".slx"); end

new_system(mdl); open_system(mdl);
set_param(mdl, "SolverType", "Variable-step", "Solver", "VariableStepAuto", ...
    "StopTime", "3");

% --- Scene: Empty Grass (known flat ground at z=0)
add_block("uavsim3dlib/Simulation 3D Scene Configuration", char(mdl + "/SceneCfg"));
set_param(char(mdl + "/SceneCfg"), "SceneDesc", "Empty Grass");
set_param(char(mdl + "/SceneCfg"), "EnableWindow", "on");
set_param(char(mdl + "/SceneCfg"), "Position", [40 40 220 110]);

% --- 1 Pedestrian at origin (visible target)
add_block("drivingsim3d/Simulation 3D Pedestrian", char(mdl + "/Ped1"));
set_param(char(mdl + "/Ped1"), "PedestrianType", "Male 1");
set_param(char(mdl + "/Ped1"), "InitialPos", "[0 0 0]");
set_param(char(mdl + "/Ped1"), "InitialYaw", "0");
set_param(char(mdl + "/Ped1"), "Position", [40 130 220 200]);

% --- 4 cameras at world position [0 0 30], each with different rotation
% Goal: find which rotation makes the camera look DOWN at the ground
rotations = {
    "RotA_neg_pitch", "[0 -1.5708 0]"   % -90° (radians, what we tried)
    "RotB_pos_pitch", "[0  1.5708 0]"   % +90° (opposite sign)
    "RotC_neg_yaw",   "[0  0 -1.5708]"  % -90° yaw (in case axes swapped)
    "RotD_zero",      "[0  0  0]"       % identity (default forward)
};
yoff = 250;
for k = 1:size(rotations, 1)
    cam_name = rotations{k, 1};
    rot      = rotations{k, 2};
    add_block("uavsim3dlib/Simulation 3D Camera", char(mdl + "/" + cam_name));
    set_param(char(mdl + "/" + cam_name), "ImageSize",     "[640 360]");
    set_param(char(mdl + "/" + cam_name), "vehTag",        "Scene Origin");
    set_param(char(mdl + "/" + cam_name), "mountLoc",      "Origin");
    set_param(char(mdl + "/" + cam_name), "offsetFlag",    "on");
    set_param(char(mdl + "/" + cam_name), "tmountOffset",  "[0 0 30]");   % world [0 0 30]
    set_param(char(mdl + "/" + cam_name), "rmountOffset",  rot);
    set_param(char(mdl + "/" + cam_name), "extTmount",     "off");
    set_param(char(mdl + "/" + cam_name), "extRmount",     "off");
    set_param(char(mdl + "/" + cam_name), "SampleTime",    "0.5");
    set_param(char(mdl + "/" + cam_name), "Position",      [400 yoff 560 yoff+50]);

    % To Workspace logger
    tw_name = sprintf("TW_%d", k);
    add_block("simulink/Sinks/To Workspace", char(mdl + "/" + tw_name));
    set_param(char(mdl + "/" + tw_name), "VariableName", sprintf("cam_log_%d", k));
    set_param(char(mdl + "/" + tw_name), "SaveFormat",   "Structure With Time");
    set_param(char(mdl + "/" + tw_name), "Position",     [620 yoff 760 yoff+50]);
    add_line(mdl, char(cam_name + "/1"), char(tw_name + "/1"), "autorouting", "smart");
    yoff = yoff + 80;
end

save_system(mdl);
fprintf("[Test] Built %s.slx with 4 test cameras\n", mdl);
fprintf("[Test] Running sim (3 s)…\n");
out = sim(mdl);

% Save last frame of each camera as PNG for visual inspection
mkdir_safe("assets");
for k = 1:size(rotations, 1)
    var = sprintf("cam_log_%d", k);
    try
        log = out.get(var);
        if ~isempty(log) && isstruct(log) && isfield(log, "signals")
            v = log.signals.values;
            if numel(size(v)) == 4
                last = v(:,:,:,end);
            else
                last = v;
            end
            png_path = sprintf("assets/test_cam_%s.png", rotations{k,1});
            if isa(last, "uint8")
                imwrite(last, png_path);
            else
                imwrite(uint8(last*255), png_path);
            end
            fprintf("[Test] %s  →  %s  (size=%s)\n", rotations{k,1}, png_path, mat2str(size(v)));
        else
            fprintf("[Test] %s  →  empty log\n", rotations{k,1});
        end
    catch ME
        fprintf("[Test] %s  →  extract FAILED: %s\n", rotations{k,1}, ME.message);
    end
end

close_system(mdl, 0);
fprintf("\n[Test] 끝. assets/test_cam_*.png 4개 파일 확인.\n");
fprintf("[Test] 그중 'ground + pedestrian' 보이는 게 옳은 회전 규약입니다.\n");
end


function mkdir_safe(p)
if ~isfolder(p), mkdir(p); end
end
