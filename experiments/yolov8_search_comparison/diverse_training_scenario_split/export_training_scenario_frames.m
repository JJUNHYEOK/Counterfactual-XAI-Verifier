function manifest_json = export_training_scenario_frames(output_dir, scenario_config_path, scenario_config_sha256)
%EXPORT_TRAINING_SCENARIO_FRAMES Render 20 pre-registered frames with visible-pixel GT.

arguments
    output_dir (1,:) char
    scenario_config_path (1,:) char
    scenario_config_sha256 (1,:) char
end

if ~isfile(scenario_config_path)
    error('Scenario config not found: %s', scenario_config_path);
end
scenario = jsondecode(fileread(scenario_config_path));
if scenario.total_frames ~= 181 || numel(scenario.selected_frame_indices) ~= 20
    error('Registered training scenarios require 181 source frames and exactly 20 selected frames.');
end
if string(scenario.ground_truth_version) ~= "rendered_instance_mask_v1"
    error('Legacy projected GT is forbidden.');
end
frame_indices = double(scenario.selected_frame_indices(:).');
if any(frame_indices < 1) || any(frame_indices > scenario.total_frames) || ...
        any(diff(frame_indices) <= 0) || frame_indices(1) ~= 1 || frame_indices(end) ~= 181
    error('Selected frame indices must be unique, ordered, and span frames 1 through 181.');
end

rng(double(scenario.seed), 'twister');
if ~isfolder(output_dir), mkdir(output_dir); end
frames_dir = fullfile(output_dir, 'frames');
if ~isfolder(frames_dir), mkdir(frames_dir); end

reference_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'geometry_reference_v1.mat');
if ~isfile(reference_path)
    error('Validated geometry reference not found: %s', reference_path);
end
loaded = load(reference_path, 'geometry');
geometry = loaded.geometry;
env = scenario.environment;

geometry_tic = tic;
init_uav_workspace(double(env.fog_percent), double(env.illumination_lux), double(env.camera_noise));
assignin('base', 'TERRAIN_X', geometry.terrain_x);
assignin('base', 'TERRAIN_Y', geometry.terrain_y);
assignin('base', 'TERRAIN_Z', geometry.terrain_z);
assignin('base', 'IMG_SIZE', double(scenario.camera.image_size));
assignin('base', 'CAM_INTRIN', double(scenario.camera.intrinsics));
assignin('base', 'FOG_DENSITY_PERCENT', double(env.fog_percent));
assignin('base', 'ILLUMINATION_LUX', double(env.illumination_lux));
assignin('base', 'CAMERA_NOISE_LEVEL', double(env.camera_noise));

objects = scenario.objects;
n_obs = numel(objects);
obs_xyz_initial = zeros(n_obs, 3);
obs_rh = zeros(n_obs, 2);
obs_class = zeros(n_obs, 1);
for obj_idx = 1:n_obs
    xy = double(objects(obj_idx).xy(:).');
    z = interp2(geometry.terrain_x, geometry.terrain_y, geometry.terrain_z, xy(1), xy(2), 'linear', 0);
    obs_xyz_initial(obj_idx, :) = [xy, z];
    obs_rh(obj_idx, :) = double(objects(obj_idx).radius_height(:).');
    obs_class(obj_idx) = double(objects(obj_idx).class_id);
end
assignin('base', 'OBSTACLES_XYZ', obs_xyz_initial);
assignin('base', 'OBSTACLES_RH', obs_rh);
assignin('base', 'OBSTACLES_CLASS', obs_class);

alpha = linspace(0, 1, scenario.total_frames).';
start_xyz = double(scenario.trajectory.start_xyz(:).');
end_xyz = double(scenario.trajectory.end_xyz(:).');
uav_xyz = start_xyz + alpha .* (end_xyz - start_xyz);
time_vector = linspace(0, double(scenario.duration_seconds), scenario.total_frames).';
geometry_seconds = toc(geometry_tic);

% Force one clean scene build per registered scenario so object placement
% cannot leak through the renderer's persistent graphics state.
clear render_eo_image eo_camera_3d_render;
setappdata(0, 'EO_FORCE_INTRUDER_REDRAW', true);

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

    [image, rendered_bboxes] = render_eo_image(uav, obs_xyz, obs_rh, ...
        double(env.fog_percent), double(env.illumination_lux), double(env.camera_noise), cam_intrin, img_size);
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
        'source_frame_number', frame_idx, ...
        'time_seconds', time_vector(frame_idx), ...
        'uav_xyz', uav, ...
        'image_path', image_path, ...
        'ground_truth', {gt_items});
end
render_seconds = toc(render_tic);

manifest = struct( ...
    'schema_version', '1.0', ...
    'scenario_id', scenario.scenario_id, ...
    'split', scenario.split, ...
    'scenario_config_path', scenario_config_path, ...
    'scenario_config_sha256', upper(scenario_config_sha256), ...
    'trajectory_name', scenario.trajectory.name, ...
    'trajectory_xyz', uav_xyz, ...
    'object_initial_xyz', obs_xyz_initial, ...
    'object_radius_height', obs_rh, ...
    'object_classes', obs_class, ...
    'environment', env, ...
    'random_seed', scenario.seed, ...
    'ground_truth_mode', 'rendered_instance_mask_v1', ...
    'renderer', 'render_eo_image + eo_camera_3d_render visible-instance colour pass', ...
    'image_width', img_size(1), ...
    'image_height', img_size(2), ...
    'frame_selection_mode', '20 temporally uniform pre-registered indices spanning the 18-second scenario', ...
    'selected_frame_indices', frame_indices, ...
    'frame_interval_pattern', diff(frame_indices), ...
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

