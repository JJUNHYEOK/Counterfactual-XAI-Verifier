function output_path = build_geometry_reference(output_path)
%BUILD_GEOMETRY_REFERENCE Persist validated fixed Simulink geometry once.
arguments
    output_path (1,:) char = fullfile(fileparts(mfilename('fullpath')), 'geometry_reference_v1.mat')
end

init_uav_workspace(5, 12000, 0.02);
mdl = 'mountain_uav_model';
if ~isfile(mdl + ".slx")
    build_mountain_uav_model(false);
elseif ~bdIsLoaded(mdl)
    load_system(mdl);
end
set_param(mdl, 'StopTime', '18');
sim_out = sim(mdl);
names = string(sim_out.who);
required = ["uav_xyz_log", "gt_bboxes_log"];
if ~all(ismember(required, names))
    error('Required logs missing. Available: %s', strjoin(names, ', '));
end

uav_raw = sim_out.get('uav_xyz_log');
gt_raw = sim_out.get('gt_bboxes_log');
if ~isstruct(uav_raw) || ~isfield(uav_raw, 'time') || ~isfield(uav_raw, 'signals')
    error('Unexpected uav_xyz_log format: %s', class(uav_raw));
end
if ~isstruct(gt_raw) || ~isfield(gt_raw, 'signals')
    error('Unexpected gt_bboxes_log format: %s', class(gt_raw));
end

uav_values = uav_raw.signals.values;
uav_size = size(uav_values);
if numel(uav_size) == 3
    uav_xyz = reshape(permute(uav_values, [3 1 2]), uav_size(3), uav_size(1) * uav_size(2));
else
    uav_xyz = uav_values;
end
gt_values = gt_raw.signals.values;
gt_size = size(gt_values);
if numel(gt_size) == 3
    gt_bboxes = permute(gt_values, [3 1 2]);
else
    gt_bboxes = gt_values;
end

geometry = struct();
geometry.time_vector = uav_raw.time;
geometry.uav_xyz = uav_xyz;
geometry.gt_bboxes = gt_bboxes;
geometry.terrain_x = evalin('base', 'TERRAIN_X');
geometry.terrain_y = evalin('base', 'TERRAIN_Y');
geometry.terrain_z = evalin('base', 'TERRAIN_Z');
geometry.obs_xyz = evalin('base', 'OBSTACLES_XYZ');
geometry.obs_rh = evalin('base', 'OBSTACLES_RH');
geometry.obs_class = evalin('base', 'OBSTACLES_CLASS');
geometry.img_size = evalin('base', 'IMG_SIZE');
geometry.cam_intrin = evalin('base', 'CAM_INTRIN');
geometry.stop_time_seconds = 18;
geometry.source_model = which(mdl + ".slx");
geometry.matlab_release = version('-release');
geometry.created_at = char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
save(output_path, 'geometry', '-v7');
fprintf('Saved geometry reference: %s\n', output_path);
end
