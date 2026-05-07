function test_low_altitude()
% test_low_altitude — Empty Grass + 매우 낮은 고도 + 가까운 사람.
%   ZalaZONE의 좌표계 의심을 배제하기 위한 최소 셋업.

mdl = "low_alt_test";
if bdIsLoaded(mdl), close_system(mdl, 0); end
if isfile(mdl + ".slx"), delete(mdl + ".slx"); end
new_system(mdl); open_system(mdl);
set_param(mdl, "SolverType","Variable-step", "Solver","VariableStepAuto", "StopTime","3");

% --- Scene: Empty Grass (평지, 원점 = 지면)
add_block("uavsim3dlib/Simulation 3D Scene Configuration", char(mdl + "/SceneCfg"));
set_param(char(mdl + "/SceneCfg"), "SceneDesc",     "Empty Grass");
set_param(char(mdl + "/SceneCfg"), "EnableWindow",  "on");
set_param(char(mdl + "/SceneCfg"), "Position",      [40 40 220 110]);

% --- Pedestrian at world origin (definitely on ground)
add_block("drivingsim3d/Simulation 3D Pedestrian", char(mdl + "/Ped"));
set_param(char(mdl + "/Ped"), "PedestrianType", "Male 1");
set_param(char(mdl + "/Ped"), "InitialPos",     "[0 0 0]");
set_param(char(mdl + "/Ped"), "InitialYaw",     "0");
set_param(char(mdl + "/Ped"), "Position",       [40 130 220 200]);

% --- Drone at low altitude 3m, 5m back from pedestrian
add_block("uavsim3dlib/Simulation 3D UAV Vehicle", char(mdl + "/UAV"));
set_param(char(mdl + "/UAV"), "ActorName",          "TestDrone");
set_param(char(mdl + "/UAV"), "Mesh",               "Quadrotor");
set_param(char(mdl + "/UAV"), "Color",              "Red");
set_param(char(mdl + "/UAV"), "InitialTranslation", "[-5 0 3]");
set_param(char(mdl + "/UAV"), "InitialRotation",    "[0 0 0]");
set_param(char(mdl + "/UAV"), "Position",           [280 40 440 130]);

add_block("simulink/Sources/Constant", char(mdl + "/UAV_Pos"));
set_param(char(mdl + "/UAV_Pos"), "Value", "[-5 0 3]");
set_param(char(mdl + "/UAV_Pos"), "Position", [40 230 130 270]);
add_block("simulink/Sources/Constant", char(mdl + "/UAV_Rot"));
set_param(char(mdl + "/UAV_Rot"), "Value", "[0 0 0]");
set_param(char(mdl + "/UAV_Rot"), "Position", [40 290 130 330]);
add_line(mdl, "UAV_Pos/1", "UAV/1", "autorouting", "smart");
add_line(mdl, "UAV_Rot/1", "UAV/2", "autorouting", "smart");

% Force update so vehTag enum picks up TestDrone
try, set_param(mdl, 'SimulationCommand', 'update'); catch, end

% --- 4 cameras testing different rotation conventions, all attached to drone
% Drone at [-5 0 3]. Pedestrian at [0 0 0] = 5m forward of drone, 3m below.
% Camera attached to drone with no offset, looking at pedestrian needs:
%   - pitch DOWN to point at ground 5m forward
%   - angle = atan(3/5) = 31° down
% Try multiple rotation values to find which works.
rotations = {
    "Rot_neg31",  "[0 -0.54 0]"   % -31° (negative pitch convention)
    "Rot_pos31",  "[0  0.54 0]"   % +31°
    "Rot_neg90",  "[0 -1.5708 0]" % straight down (negative)
    "Rot_pos90",  "[0  1.5708 0]" % straight down (positive)
};

yoff = 80;
for k = 1:size(rotations, 1)
    cam_name = rotations{k, 1};
    rot      = rotations{k, 2};
    add_block("uavsim3dlib/Simulation 3D Camera", char(mdl + "/" + cam_name));
    blk = char(mdl + "/" + cam_name);
    set_param(blk, "ImageSize",     "[640 360]");
    set_param(blk, "vehTagList",    "{'Scene Origin', 'TestDrone'}");
    set_param(blk, "vehTag",        "TestDrone");
    set_param(blk, "mountLoc",      "Origin");
    set_param(blk, "offsetFlag",    "on");
    set_param(blk, "tmountOffset",  "[0 0 0]");      % no offset, at drone center
    set_param(blk, "rmountOffset",  rot);
    set_param(blk, "extTmount",     "off");
    set_param(blk, "extRmount",     "off");
    set_param(blk, "SampleTime",    "0.5");
    set_param(blk, "Position",      [560 yoff 720 yoff+50]);

    % Verify
    actual_tag = get_param(blk, "vehTag");
    fprintf("[Test] %s  vehTag='%s'  rot=%s\n", cam_name, actual_tag, rot);

    % Logger
    tw_name = sprintf("TW_%d", k);
    add_block("simulink/Sinks/To Workspace", char(mdl + "/" + tw_name));
    set_param(char(mdl + "/" + tw_name), "VariableName", sprintf("cam_%d", k));
    set_param(char(mdl + "/" + tw_name), "SaveFormat",   "Structure With Time");
    set_param(char(mdl + "/" + tw_name), "Position",     [800 yoff 940 yoff+50]);
    add_line(mdl, char(cam_name + "/1"), char(tw_name + "/1"), "autorouting", "smart");
    yoff = yoff + 70;
end

save_system(mdl);
fprintf("\n[Test] Running sim…\n");
out = sim(mdl);

if ~isfolder("assets"), mkdir("assets"); end
for k = 1:size(rotations, 1)
    var = sprintf("cam_%d", k);
    try
        log = out.get(var);
        if ~isempty(log) && isstruct(log) && isfield(log, "signals")
            v = log.signals.values;
            if numel(size(v)) == 4, last = v(:,:,:,end); else, last = v; end
            png = sprintf("assets/test_lowalt_%s.png", rotations{k,1});
            if isa(last, "uint8"), imwrite(last, png);
            else, imwrite(uint8(last*255), png); end
            fprintf("[Test] %s → %s\n", rotations{k,1}, png);
        end
    catch
    end
end

close_system(mdl, 0);
fprintf("\n[Test] 끝.\n");
fprintf("[Test] assets/test_lowalt_*.png 4개 확인.\n");
fprintf("       사람이 보이는 게 옳은 회전 규약입니다.\n");
end
