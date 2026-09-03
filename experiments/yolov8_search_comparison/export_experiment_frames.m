function manifest_json = export_experiment_frames( ...
    fog, illumination, camera_noise, output_dir, random_seed, frame_stride, detector_mode, ground_truth_mode, scenario_variant)
%EXPORT_EXPERIMENT_FRAMES Render the dashboard's EO frames and explicit GT.
%
% The rendered image is the only input to YOLO. Ground-truth boxes are
% exported separately and are never passed to the YOLO inference call. The
% optional "heuristic" mode is retained solely to reproduce the legacy
% image_detector(img, gt_boxes) execution path when explicitly selected.

arguments
    fog (1,1) double
    illumination (1,1) double
    camera_noise (1,1) double
    output_dir (1,:) char
    random_seed (1,1) double = 42
    frame_stride (1,1) double = 1
    detector_mode (1,:) char = 'yolov8'
    ground_truth_mode (1,:) char = 'rendered_instance_mask_v1'
    scenario_variant (1,1) double = 0
end

fog = max(0, min(100, fog));
illumination = max(200, min(15000, illumination));
camera_noise = max(0, min(0.60, camera_noise));
frame_stride = max(1, round(frame_stride));
detector_mode = lower(string(detector_mode));
ground_truth_mode = lower(string(ground_truth_mode));
scenario_variant = round(scenario_variant);
if ~any(detector_mode == ["yolov8", "heuristic"])
    error("detector_mode must be 'yolov8' or 'heuristic'.");
end
if ~any(ground_truth_mode == ["rendered_instance_mask_v1", "legacy_pinhole_projection"])
    error("Unsupported ground_truth_mode: %s", ground_truth_mode);
end

rng(random_seed, 'twister');
if ~isfolder(output_dir), mkdir(output_dir); end
frames_dir = fullfile(output_dir, 'frames');
if ~isfolder(frames_dir), mkdir(frames_dir); end

persistent geometry geometry_source;
geometry_seconds = 0;
if isempty(geometry) || ~isfield(geometry, 'uav_xyz')
    geometry_tic = tic;
    % The three environment values affect image formation, not the fixed
    % flight trajectory or object placement. Initialise all base variables
    % once, then reuse the exact same Simulink geometry for every method.
    init_uav_workspace(fog, illumination, camera_noise);
    geometry_reference_path = fullfile(fileparts(mfilename('fullpath')), 'geometry_reference_v1.mat');
    if isfile(geometry_reference_path)
        loaded = load(geometry_reference_path, 'geometry');
        geometry = loaded.geometry;
        geometry_source = ['validated_reference_mat: ' geometry_reference_path];
    else
        mdl = 'mountain_uav_model';
        if ~isfile(mdl + ".slx")
            build_mountain_uav_model(false);
        elseif ~bdIsLoaded(mdl)
            load_system(mdl);
        end
        set_param(mdl, 'StopTime', '18');
        sim_out = sim(mdl);
        [time_vector, uav_xyz] = read_log_vec_local(sim_out, 'uav_xyz_log');
        [~, gt_bboxes] = read_log_3d_local(sim_out, 'gt_bboxes_log');

        geometry = struct();
        geometry.time_vector = time_vector;
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
        geometry_source = 'live_simulink_simulation';
    end
    geometry_seconds = toc(geometry_tic);

    % Force one clean persistent 3-D scene for this experiment session.
    clear render_eo_image eo_camera_3d_render;
    setappdata(0, 'EO_FORCE_INTRUDER_REDRAW', true);
end

% Restore the fixed geometry in case any caller changed the base workspace.
assignin('base', 'TERRAIN_X', geometry.terrain_x);
assignin('base', 'TERRAIN_Y', geometry.terrain_y);
assignin('base', 'TERRAIN_Z', geometry.terrain_z);
assignin('base', 'OBSTACLES_XYZ', geometry.obs_xyz);
assignin('base', 'OBSTACLES_RH', geometry.obs_rh);
assignin('base', 'OBSTACLES_CLASS', geometry.obs_class);
assignin('base', 'IMG_SIZE', geometry.img_size);
assignin('base', 'CAM_INTRIN', geometry.cam_intrin);
assignin('base', 'FOG_DENSITY_PERCENT', fog);
assignin('base', 'ILLUMINATION_LUX', illumination);
assignin('base', 'CAMERA_NOISE_LEVEL', camera_noise);

uav_xyz = geometry.uav_xyz;
gt_bboxes = geometry.gt_bboxes;
time_vector = geometry.time_vector;
obs_xyz_initial = geometry.obs_xyz;
obs_rh = geometry.obs_rh;
obs_class = geometry.obs_class(:);
cam_intrin = geometry.cam_intrin;
cam_w = geometry.img_size(1);
cam_h = geometry.img_size(2);
n_obs = size(obs_xyz_initial, 1);

walk_omega = 0.6;
walk_radius = 1.5;
walk_active = obs_class == 1;
walk_phase = (1:n_obs)' * 1.7;
base_obs_xy = obs_xyz_initial(:, 1:2);

% Deterministic training-only scenario variation. Variant zero is the final
% held-out baseline and remains bit-for-bit unchanged. Nonzero variants alter
% the whole 181-frame trajectory/object layout as one indivisible scenario;
% target geometry and the pixel-exact GT extraction are unchanged.
uav_offset = [0, 0, 0];
object_xy_offsets = zeros(n_obs, 2);
walk_phase_offset = 0;
if scenario_variant ~= 0
    uav_offset = [ ...
        2.5 * sin(0.73 * scenario_variant), ...
        3.0 * sin(1.13 * scenario_variant), ...
        3.0 * cos(0.91 * scenario_variant)];
    uav_xyz = uav_xyz + uav_offset;
    for obj_idx = 1:n_obs
        object_xy_offsets(obj_idx, :) = [ ...
            2.2 * sin(0.47 * scenario_variant + 1.31 * obj_idx), ...
            2.5 * cos(0.63 * scenario_variant + 0.89 * obj_idx)];
    end
    base_obs_xy = base_obs_xy + object_xy_offsets;
    for obj_idx = 1:n_obs
        obs_xyz_initial(obj_idx, 1:2) = base_obs_xy(obj_idx, :);
        obs_xyz_initial(obj_idx, 3) = interp2( ...
            geometry.terrain_x, geometry.terrain_y, geometry.terrain_z, ...
            base_obs_xy(obj_idx, 1), base_obs_xy(obj_idx, 2), 'linear', 0);
    end
    walk_phase_offset = 0.43 * scenario_variant;
    walk_phase = walk_phase + walk_phase_offset;
end

frame_indices = 1:frame_stride:numel(time_vector);
frame_records = cell(1, numel(frame_indices));
render_tic = tic;

for out_idx = 1:numel(frame_indices)
    frame_idx = frame_indices(out_idx);
    uav = uav_xyz(frame_idx, :);
    obs_xyz = obs_xyz_initial;
    t_now = (frame_idx - 1) * 0.1; % identical to mountain_uav_dashboard.m
    for obj_idx = 1:n_obs
        if walk_active(obj_idx)
            nx = base_obs_xy(obj_idx, 1) + ...
                walk_radius * sin(walk_omega * t_now + walk_phase(obj_idx));
            ny = base_obs_xy(obj_idx, 2) + ...
                walk_radius * cos(walk_omega * t_now + walk_phase(obj_idx));
            nz = interp2(geometry.terrain_x, geometry.terrain_y, geometry.terrain_z, ...
                nx, ny, 'linear', 0);
            obs_xyz(obj_idx, :) = [nx, ny, nz];
        end
    end

    gt_frame = squeeze(gt_bboxes(frame_idx, :, :));
    if n_obs == 1, gt_frame = reshape(gt_frame, 1, 4); end
    if size(gt_frame, 2) ~= 4, gt_frame = reshape(gt_frame, [], 4); end

    pitch_rad = cam_intrin(5) * pi / 180;
    sp = sin(pitch_rad); cp = cos(pitch_rad);
    for obj_idx = 1:n_obs
        if walk_active(obj_idx)
            gt_frame(obj_idx, :) = project_walker_bbox_local( ...
                uav, obs_xyz(obj_idx, :), obs_rh(obj_idx, 1), obs_rh(obj_idx, 2), ...
                cam_intrin(1), cam_intrin(2), cam_intrin(3), cam_intrin(4), ...
                sp, cp, cam_w, cam_h);
        end
    end

    [image, rendered_bboxes] = render_eo_image(uav, obs_xyz, obs_rh, fog, illumination, ...
        camera_noise, cam_intrin, [cam_w cam_h]);
    if ground_truth_mode == "rendered_instance_mask_v1"
        evaluation_bboxes = rendered_bboxes;
    else
        evaluation_bboxes = gt_frame;
    end
    image_path = fullfile(frames_dir, sprintf('frame_%04d.png', frame_idx));
    imwrite(max(0, min(1, image)), image_path);

    gt_items = {};
    for obj_idx = 1:n_obs
        bbox = double(evaluation_bboxes(obj_idx, :));
        if any(bbox ~= 0) && bbox(3) > 0 && bbox(4) > 0
            gt_items{end + 1} = struct( ... %#ok<AGROW>
                'object_id', obj_idx, ...
                'simulation_class_id', obs_class(obj_idx), ...
                'class_name', class_name_local(obs_class(obj_idx)), ...
                'bbox_xywh', bbox, ...
                'legacy_projected_bbox_xywh', double(gt_frame(obj_idx, :)));
        end
    end

    heuristic_items = {};
    if detector_mode == "heuristic"
        [scores, det_bboxes] = image_detector(image, gt_frame);
        for obj_idx = 1:n_obs
            bbox = double(det_bboxes(obj_idx, :));
            score = double(scores(obj_idx));
            if score > 0.05 && any(bbox ~= 0)
                heuristic_items{end + 1} = struct( ... %#ok<AGROW>
                    'source_class_id', obs_class(obj_idx), ...
                    'class_name', class_name_local(obs_class(obj_idx)), ...
                    'confidence', score, ...
                    'bbox_xywh', bbox);
            end
        end
    end

    frame_records{out_idx} = struct( ...
        'frame_index', frame_idx, ...
        'time_seconds', double(time_vector(frame_idx)), ...
        'image_path', image_path, ...
        'ground_truth', {gt_items}, ...
        'heuristic_detections', {heuristic_items});
end

render_seconds = toc(render_tic);
manifest = struct( ...
    'schema_version', '1.0', ...
    'environment', struct('fog_percent', fog, ...
        'illumination_lux', illumination, 'camera_noise', camera_noise), ...
    'random_seed', random_seed, ...
    'scenario_variant', scenario_variant, ...
    'scenario_transform', struct( ...
        'uav_xyz_offset', uav_offset, ...
        'object_xy_offsets', object_xy_offsets, ...
        'walk_phase_offset_radians', walk_phase_offset), ...
    'detector_mode', char(detector_mode), ...
    'ground_truth_mode', char(ground_truth_mode), ...
    'renderer', 'render_eo_image (3-D with deterministic 2-D fallback)', ...
    'image_width', cam_w, ...
    'image_height', cam_h, ...
    'frame_stride', frame_stride, ...
    'total_simulation_frames', numel(time_vector), ...
    'evaluated_frame_count', numel(frame_indices), ...
    'object_count', n_obs, ...
    'person_count', sum(obs_class == 1), ...
    'vehicle_count', sum(obs_class == 2), ...
    'geometry_simulation_seconds', geometry_seconds, ...
    'geometry_source', geometry_source, ...
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
else
    name = 'vehicle';
end
end


function [t, vals] = read_log_vec_local(sim_out, name)
s = read_signal_local(sim_out, name);
t = s.time;
v = s.values;
sz = size(v);
if numel(sz) == 2 && (sz(1) == 1 || sz(2) == 1)
    vals = v(:);
elseif numel(sz) == 2
    vals = v;
elseif numel(sz) == 3
    vals = reshape(permute(v, [3 1 2]), sz(3), sz(1) * sz(2));
else
    vals = v;
end
end


function [t, vals] = read_log_3d_local(sim_out, name)
s = read_signal_local(sim_out, name);
t = s.time;
v = s.values;
sz = size(v);
if numel(sz) == 3
    vals = permute(v, [3 1 2]);
elseif numel(sz) == 2
    vals = reshape(v, [numel(t), sz(2), 1]);
else
    vals = v;
end
end


function s = read_signal_local(sim_out, name)
try
    raw = sim_out.get(name);
catch
    raw = [];
end
if isempty(raw)
    try, raw = evalin('base', name); catch, raw = []; end
end
if isempty(raw)
    error('Could not find logged signal: %s', name);
end
if isstruct(raw) && isfield(raw, 'time') && isfield(raw, 'signals')
    s.time = raw.time;
    s.values = raw.signals.values;
else
    error('Unexpected signal format for %s', name);
end
end


function bbox = project_walker_bbox_local(uav, base, r, h, fx, fy, cx, cy, sp, cp, w, h_img)
bbox = zeros(1, 4);
dx = base(1) - uav(1);
dy = base(2) - uav(2);
dz_base = base(3) - uav(3);
dz_top = (base(3) + h) - uav(3);
cz_base = dx * cp - dz_base * sp;
cz_top = dx * cp - dz_top * sp;
if cz_base < 0.5 || cz_top < 0.5, return; end
cz_avg = 0.5 * (cz_base + cz_top);
cam_x_left = dy - r;
cam_x_right = dy + r;
cam_y_top = -dx * sp - r * sp - dz_top * cp;
cam_y_bottom = -dx * sp + r * sp - dz_base * cp;
u_left = fx * cam_x_left / cz_avg + cx;
u_right = fx * cam_x_right / cz_avg + cx;
v_top = fy * cam_y_top / cz_top + cy;
v_bottom = fy * cam_y_bottom / cz_base + cy;
bw = u_right - u_left;
bh = v_bottom - v_top;
if bw <= 0 || bh <= 0, return; end
if u_right < 0 || u_left > w || v_bottom < 0 || v_top > h_img, return; end
if cz_avg > 150, return; end
bbox = [u_left, v_top, bw, bh];
end
