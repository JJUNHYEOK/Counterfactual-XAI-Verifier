function test_drone_pitch()
% test_drone_pitch — 카메라의 rmountOffset이 무시되는 걸 우회.
%   드론 자체를 pitch 시켜서 부착된 카메라가 따라 회전하도록 한다.
%   여러 회전값을 동시에 테스트해서 어떤 게 실제로 down view를 만드는지 확인.

mdl = "drone_pitch_test";
if bdIsLoaded(mdl), close_system(mdl, 0); end
if isfile(mdl + ".slx"), delete(mdl + ".slx"); end
new_system(mdl); open_system(mdl);
set_param(mdl, "SolverType","Variable-step", "Solver","VariableStepAuto", "StopTime","3");

% Scene
add_block("uavsim3dlib/Simulation 3D Scene Configuration", char(mdl + "/SceneCfg"));
set_param(char(mdl + "/SceneCfg"), "SceneDesc","Empty Grass");
set_param(char(mdl + "/SceneCfg"), "EnableWindow","on");
set_param(char(mdl + "/SceneCfg"), "Position", [40 40 220 110]);

% Pedestrian at origin
add_block("drivingsim3d/Simulation 3D Pedestrian", char(mdl + "/Ped"));
set_param(char(mdl + "/Ped"), "PedestrianType","Male 1");
set_param(char(mdl + "/Ped"), "InitialPos","[0 0 0]");
set_param(char(mdl + "/Ped"), "InitialYaw","0");
set_param(char(mdl + "/Ped"), "Position", [40 130 220 200]);

% Test 4 different drone rotations
% Drone at world [-5, 0, 3], pedestrian at [0, 0, 0]
% Need to look 5m forward + 3m down → pitch atan(3/5) = 31° down
rotations = {
    "Drone_p31_neg",  "[0 -0.54 0]"   % pitch -31° (radians)
    "Drone_p31_pos",  "[0  0.54 0]"   % pitch +31°
    "Drone_p31d_neg", "[0 -31 0]"     % pitch -31° (degrees)
    "Drone_p31d_pos", "[0  31 0]"     % pitch +31° (degrees)
};

yoff = 80;
for k = 1:size(rotations, 1)
    name = rotations{k, 1};
    rot  = rotations{k, 2};

    % Drone with this rotation
    drone_blk = char(mdl + "/" + name + "_UAV");
    add_block("uavsim3dlib/Simulation 3D UAV Vehicle", drone_blk);
    set_param(drone_blk, "ActorName",          char(name));
    set_param(drone_blk, "Mesh",               "Quadrotor");
    set_param(drone_blk, "Color",              "Red");
    set_param(drone_blk, "InitialTranslation", "[-5 0 3]");
    set_param(drone_blk, "InitialRotation",    rot);
    set_param(drone_blk, "Position",           [280 yoff 440 yoff+50]);

    % Constants for inputs
    pos_blk = char(mdl + "/" + name + "_Pos");
    add_block("simulink/Sources/Constant", pos_blk);
    set_param(pos_blk, "Value", "[-5 0 3]");
    set_param(pos_blk, "Position", [40 yoff+220 130 yoff+250]);
    rot_blk = char(mdl + "/" + name + "_Rot");
    add_block("simulink/Sources/Constant", rot_blk);
    set_param(rot_blk, "Value", rot);
    set_param(rot_blk, "Position", [40 yoff+260 130 yoff+290]);
    add_line(mdl, char(name + "_Pos/1"), char(name + "_UAV/1"), "autorouting","smart");
    add_line(mdl, char(name + "_Rot/1"), char(name + "_UAV/2"), "autorouting","smart");

    yoff = yoff + 80;
end

% Force update so all UAVs registered
try, set_param(mdl, 'SimulationCommand', 'update'); catch, end

% Add one camera per drone, attached, NO mount offset (just follow drone)
yoff = 80;
for k = 1:size(rotations, 1)
    name = rotations{k, 1};

    cam_blk = char(mdl + "/" + name + "_Cam");
    add_block("uavsim3dlib/Simulation 3D Camera", cam_blk);
    set_param(cam_blk, "ImageSize",     "[640 360]");
    set_param(cam_blk, "vehTagList",    sprintf("{'Scene Origin', '%s'}", name));
    set_param(cam_blk, "vehTag",        char(name));
    set_param(cam_blk, "mountLoc",      "Origin");
    set_param(cam_blk, "offsetFlag",    "off");          % NO offset, use drone's pose directly
    set_param(cam_blk, "tmountOffset",  "[0 0 0]");
    set_param(cam_blk, "rmountOffset",  "[0 0 0]");
    set_param(cam_blk, "extTmount",     "off");
    set_param(cam_blk, "extRmount",     "off");
    set_param(cam_blk, "SampleTime",    "0.5");
    set_param(cam_blk, "Position",      [560 yoff 720 yoff+50]);

    actual = get_param(cam_blk, "vehTag");
    fprintf("[Test] %s  vehTag='%s'  drone_rot=%s\n", name, actual, rotations{k,2});

    % Logger
    tw_blk = char(mdl + "/" + name + "_TW");
    add_block("simulink/Sinks/To Workspace", tw_blk);
    set_param(tw_blk, "VariableName", sprintf("cam_%d", k));
    set_param(tw_blk, "SaveFormat",   "Structure With Time");
    set_param(tw_blk, "Position",     [800 yoff 940 yoff+50]);
    add_line(mdl, char(name + "_Cam/1"), char(name + "_TW/1"), "autorouting","smart");

    yoff = yoff + 80;
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
            png = sprintf("assets/test_dronepitch_%s.png", rotations{k,1});
            if isa(last, "uint8"), imwrite(last, png);
            else, imwrite(uint8(last*255), png); end
            fprintf("[Test] %s → %s\n", rotations{k,1}, png);
        end
    catch
    end
end

close_system(mdl, 0);
fprintf("\n[Test] 끝.\n");
fprintf("[Test] 4개 PNG 중 사람이 보이는 게 있다면 → 그 회전 규약(부호+단위)이 답.\n");
end
