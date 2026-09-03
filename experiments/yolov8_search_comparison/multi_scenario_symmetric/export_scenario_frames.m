function manifest_json = export_scenario_frames( ...
    fog, illumination, camera_noise, output_dir, random_seed, frame_stride, ...
    detector_mode, ground_truth_mode, scenario_config_path, scenario_config_sha256)
%EXPORT_SCENARIO_FRAMES Render one pre-registered trajectory with pixel GT.
% The scenario JSON is immutable experiment input. Weather changes image
% formation only; trajectory, object placement, camera, and GT geometry stay
% fixed across boundary-search iterations.

arguments
    fog (1,1) double
    illumination (1,1) double
    camera_noise (1,1) double
    output_dir (1,:) char
    random_seed (1,1) double
    frame_stride (1,1) double
    detector_mode (1,:) char
    ground_truth_mode (1,:) char
    scenario_config_path (1,:) char
    scenario_config_sha256 (1,:) char
end

fog = max(0, min(100, fog));
illumination = max(200, min(15000, illumination));
camera_noise = max(0, min(0.60, camera_noise));
frame_stride = max(1, round(frame_stride));
if lower(string(detector_mode)) ~= "yolov8"
    error('The multi-scenario paper experiment only supports yolov8.');
end
if lower(string(ground_truth_mode)) ~= "rendered_instance_mask_v1"
    error('Legacy projected GT is forbidden in this experiment.');
end
if ~isfile(scenario_config_path)
    error('Scenario config not found: %s', scenario_config_path);
end

scenario = jsondecode(fileread(scenario_config_path));
if scenario.seed ~= random_seed
    error('Scenario seed mismatch: JSON=%d call=%d', scenario.seed, random_seed);
end
if scenario.total_frames ~= 181
    error('Every registered scenario must contain exactly 181 frames.');
end

rng(random_seed, 'twister');
if ~isfolder(output_dir), mkdir(output_dir); end
frames_dir = fullfile(output_dir, 'frames');
if ~isfolder(frames_dir), mkdir(frames_dir); end

geometry_tic = tic;
reference_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'geometry_reference_v1.mat');
if ~isfile(reference_path)
    error('Validated geometry reference not found: %s', reference_path);
end
loaded = load(reference_path, 'geometry');
geometry = loaded.geometry;

% Initialise the deterministic background scenery using the repository's
% established workspace builder, then replace only registered scenario data.
init_uav_workspace(fog, illumination, camera_noise);

assignin('base', 'TERRAIN_X', geometry.terrain_x);
assignin('base', 'TERRAIN_Y', geometry.terrain_y);
assignin('base', 'TERRAIN_Z', geometry.terrain_z);
assignin('base', 'IMG_SIZE', double(scenario.camera.image_size));
assignin('base', 'CAM_INTRIN', double(scenario.camera.intrinsics));
assignin('base', 'FOG_DENSITY_PERCENT', fog);
assignin('base', 'ILLUMINATION_LUX', illumination);
assignin('base', 'CAMERA_NOISE_LEVEL', camera_noise);

objects = scenario.objects;
n_obs = numel(objects);
obs_xyz_initial = zeros(n_obs, 3);
obs_rh = zeros(n_obs, 2);
obs_class = zeros(n_obs, 1);
for obj_idx = 1:n_obs
    xy = double(objects(obj_idx).xy(:).');
    z = interp2(geometry.terrain_x, geometry.terrain_y, geometry.terrain_z, ...
        xy(1), xy(2), 'linear', 0);
    obs_xyz_initial(obj_idx, :) = [xy, z];
    obs_rh(obj_idx, :) = double(objects(obj_idx).radius_height(:).');
    obs_class(obj_idx) = double(objects(obj_idx).class_id);
end
assignin('base', 'OBSTACLES_XYZ', obs_xyz_initial);
assignin('base', 'OBSTACLES_RH', obs_rh);
assignin('base', 'OBSTACLES_CLASS', obs_class);
% SCENERY_OBJECTS was populated by init_uav_workspace above.

alpha = linspace(0, 1, scenario.total_frames).';
start_xyz = double(scenario.trajectory.start_xyz(:).');
end_xyz = double(scenario.trajectory.end_xyz(:).');
uav_xyz = start_xyz + alpha .* (end_xyz - start_xyz);
time_vector = linspace(0, double(scenario.duration_seconds), scenario.total_frames).';
geometry_seconds = toc(geometry_tic);

% Rebuild the persistent 3-D scene only when the registered scenario changes.
persistent active_scenario_id;
if isempty(active_scenario_id) || ~strcmp(active_scenario_id, scenario.scenario_id)
    clear render_eo_image eo_camera_3d_render;
    setappdata(0, 'EO_FORCE_INTRUDER_REDRAW', true);
    active_scenario_id = scenario.scenario_id;
end

frame_indices = 1:frame_stride:scenario.total_frames;
frame_records = cell(1, numel(frame_indices));
walk_active = obs_class == 1;
walk_radius = double(scenario.person_motion.radius_m);
walk_omega = double(scenario.person_motion.omega_rad_s);
walk_phase = (1:n_obs).' * 1.7 + double(scenario.person_motion.phase_offset_rad);
base_obs_xy = obs_xyz_initial(:, 1:2);
cam_intrin = double(scenario.camera.intrinsics(:).');
img_size = double(scenario.camera.image_size(:).');

render_tic = tic;
for out_idx = 1:numel(frame_indices)
    frame_idx = frame_indices(out_idx);
    uav = uav_xyz(frame_idx, :);
    obs_xyz = obs_xyz_initial;
    t_now = time_vector(frame_idx);
    for obj_idx = 1:n_obs
        if walk_active(obj_idx)
            nx = base_obs_xy(obj_idx, 1) + walk_radius * sin(walk_omega * t_now + walk_phase(obj_idx));
            ny = base_obs_xy(obj_idx, 2) + walk_radius * cos(walk_omega * t_now + walk_phase(obj_idx));
            nz = interp2(geometry.terrain_x, geometry.terrain_y, geometry.terrain_z, nx, ny, 'linear', 0);
            obs_xyz(obj_idx, :) = [nx, ny, nz];
        end
    end

    [image, rendered_bboxes] = render_eo_image(uav, obs_xyz, obs_rh, fog, illumination, ...
        camera_noise, cam_intrin, img_size);
    image_path = fullfile(frames_dir, sprintf('frame_%04d.png', frame_idx));
    imwrite(max(0, min(1, image)), image_path);

    gt_items = {};
    for obj_idx = 1:n_obs
        bbox = double(rendered_bboxes(obj_idx, :));
        if any(bbox ~= 0) && bbox(3) > 0 && bbox(4) > 0
            gt_items{end + 1} = struct( ... %#ok<AGROW>
                'object_id', obj_idx, ...
                'simulation_class_id', obs_class(obj_idx), ...
                'class_name', class_name_local(obs_class(obj_idx)), ...
                'bbox_xywh', bbox);
        end
    end
    frame_records{out_idx} = struct( ...
        'frame_index', frame_idx, ...
        'time_seconds', time_vector(frame_idx), ...
        'uav_xyz', uav, ...
        'image_path', image_path, ...
        'ground_truth', {gt_items});
end
render_seconds = toc(render_tic);

manifest = struct( ...
    'schema_version', '2.0', ...
    'scenario_id', scenario.scenario_id, ...
    'scenario_config_path', scenario_config_path, ...
    'scenario_config_sha256', upper(scenario_config_sha256), ...
    'trajectory_name', scenario.trajectory.name, ...
    'trajectory_xyz', uav_xyz, ...
    'object_initial_xyz', obs_xyz_initial, ...
    'object_radius_height', obs_rh, ...
    'object_classes', obs_class, ...
    'environment', struct('fog_percent', fog, 'illumination_lux', illumination, 'camera_noise', camera_noise), ...
    'random_seed', random_seed, ...
    'detector_mode', 'yolov8', ...
    'ground_truth_mode', 'rendered_instance_mask_v1', ...
    'renderer', 'render_eo_image + eo_camera_3d_render visible-instance colour pass', ...
    'image_width', img_size(1), ...
    'image_height', img_size(2), ...
    'frame_stride', frame_stride, ...
    'total_simulation_frames', scenario.total_frames, ...
    'evaluated_frame_count', numel(frame_indices), ...
    'object_count', n_obs, ...
    'person_count', sum(obs_class == 1), ...
    'vehicle_count', sum(obs_class == 2), ...
    'geometry_simulation_seconds', geometry_seconds, ...
    'geometry_source', ['registered_linear_trajectory_from: ' reference_path], ...
    'frame_rendering_seconds', render_seconds, ...
    'matlab_version', version, ...
    'matlab_release', version('-release'), ...
    'frames', {frame_records});

manifest_json = jsonencode(manifest);
fid = fopen(fullfile(output_dir, 'frame_manifest.json'), 'w', 'n', 'UTF-8');
if fid < 0, error('Could not open frame_manifest.json for writing.'); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, manifest_json, 'char');
end


function name = class_name_local(class_id)
if class_id == 1
    name = 'person';
elseif class_id == 2
    name = 'vehicle';
else
    error('Unsupported simulation class ID: %d', class_id);
end
end
