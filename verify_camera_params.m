function verify_camera_params()
% verify_camera_params — set every camera parameter and immediately read it
%   back so we know which set_param calls are actually being honored.
%   Also tries multiple boolean / numeric formats to find the right one.

mdl = "verify_cam";
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
add_block("uavsim3dlib/Simulation 3D Camera", char(mdl + "/Cam"));
blk = char(mdl + "/Cam");

fprintf("\n=== 카메라 파라미터 set→get 검증 ===\n\n");

% Test 1: ImageSize
test_set(blk, "ImageSize", "[640 360]");

% Test 2: vehTag — try each enum value
fprintf("\n--- vehTag enum 시도 ---\n");
for val = ["Scene Origin", "SimulinkVehicle1", "Custom"]
    test_set(blk, "vehTag", char(val));
end

% Test 3: mountLoc
test_set(blk, "mountLoc", "Origin");

% Test 4: offsetFlag — try multiple boolean formats
fprintf("\n--- offsetFlag boolean format 시도 ---\n");
for val = ["on", "off", "true", "false", "1", "0"]
    test_set(blk, "offsetFlag", char(val));
end

% Test 5: tmountOffset — string
test_set(blk, "tmountOffset", "[0 0 30]");

% Test 6: rmountOffset
test_set(blk, "rmountOffset", "[0 -1.5708 0]");

% Test 7: extTmount, extRmount
test_set(blk, "extTmount", "off");
test_set(blk, "extRmount", "off");

% Test 8: SampleTime
test_set(blk, "SampleTime", "0.1");

% Test 9: FocalLength, OpticalCenter
test_set(blk, "FocalLength",   "[600 600]");
test_set(blk, "OpticalCenter", "[320 180]");

% Test 10: mountPoint, mountOrientation (alternative way?)
fprintf("\n--- mountPoint / mountOrientation (alternative path?) ---\n");
test_set(blk, "mountPoint",       "[0 0 30]");
test_set(blk, "mountOrientation", "[0 -1.5708 0]");

% Final state of every param
fprintf("\n=== 최종 파라미터 dump ===\n");
pinfo = get_param(blk, "DialogParameters");
fnames = fieldnames(pinfo);
for k = 1:numel(fnames)
    name = fnames{k};
    try
        val = get_param(blk, name);
        if ~ischar(val) && ~isstring(val), val = mat2str(val); end
        fprintf("  %-32s = '%s'\n", name, val);
    catch ME
        fprintf("  %-32s = ERROR: %s\n", name, ME.message);
    end
end

close_system(mdl, 0);
fprintf("\n=== 끝 ===\n");
end


function test_set(blk, pname, pval)
try
    set_param(blk, pname, pval);
    actual = get_param(blk, pname);
    if ~ischar(actual) && ~isstring(actual), actual = mat2str(actual); end
    if string(actual) == string(pval)
        fprintf("  [OK]    %-25s set='%-15s' got='%s'\n", pname, pval, actual);
    else
        fprintf("  [DIFF]  %-25s set='%-15s' got='%s'\n", pname, pval, actual);
    end
catch ME
    fprintf("  [FAIL]  %-25s set='%-15s' err='%s'\n", pname, pval, ME.message);
end
end
