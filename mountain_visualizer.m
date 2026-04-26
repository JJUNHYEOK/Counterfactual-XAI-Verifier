function [figHandle, summary] = mountain_visualizer(simOut, opts)
% mountain_visualizer
% -------------------------------------------------------------------------
% Renders the mountain UAV scene from a simulation output:
%   - Left panel : 3D mountain terrain + UAV trajectory + tree obstacles
%   - Right panel: synthetic camera view with GT (green dashed) and detected
%                  (red solid) bounding boxes, overlaid on a sky/ground gradient.
%
% Inputs:
%   simOut : Simulink.SimulationOutput from build_mountain_uav_model
%   opts (optional struct):
%       .visible      logical (default true)   - show figure on screen
%       .savePath     string  (default "")     - if non-empty, save PNG here
%       .animate      logical (default true)   - animate over time (vs final frame only)
%       .frameStride  integer (default 2)      - step size for animation
%       .pauseSeconds double  (default 0.05)
%
% Outputs:
%   figHandle : MATLAB figure handle
%   summary   : struct with mAP50 and per-frame stats
% -------------------------------------------------------------------------

if nargin < 2 || isempty(opts), opts = struct(); end
if ~isfield(opts, "visible"),     opts.visible     = true;   end
if ~isfield(opts, "savePath"),    opts.savePath    = "";     end
if ~isfield(opts, "animate"),     opts.animate     = true;   end
if ~isfield(opts, "frameStride"), opts.frameStride = 2;      end
if ~isfield(opts, "pauseSeconds"),opts.pauseSeconds= 0.05;   end

% --- Read logged signals ---
[t_vec, uav_xyz] = read_log_vec(simOut, "uav_xyz_log");
[~,     scores]   = read_log_vec(simOut, "det_scores_log");
[~,     detBB]    = read_log_3d (simOut, "det_bboxes_log");   % Nt x N x 4
[~,     gtBB]     = read_log_3d (simOut, "gt_bboxes_log");
[~,     dists]    = read_log_vec(simOut, "rel_dists_log");

% Read static scene from base workspace
Xg  = evalin("base", "TERRAIN_X");
Yg  = evalin("base", "TERRAIN_Y");
Zg  = evalin("base", "TERRAIN_Z");
obs_xyz = evalin("base", "OBSTACLES_XYZ");
obs_rh  = evalin("base", "OBSTACLES_RH");
imgSize = evalin("base", "IMG_SIZE");
camW = imgSize(1); camH = imgSize(2);

Nt = numel(t_vec);
Nobs = size(obs_xyz, 1);

% --- mAP50 (over per-frame, per-tree detections) ---
summary = compute_summary(gtBB, detBB, scores, Nobs);
fprintf("[VIZ] frames=%d, obstacles=%d, TP=%d, FP=%d, FN=%d, mAP50=%.4f\n", ...
    Nt, Nobs, summary.tp, summary.fp, summary.fn, summary.map50);

% --- Figure ---
visStr = "on"; if ~opts.visible, visStr = "off"; end
figHandle = figure("Name", "Mountain UAV Test", "Color", "w", ...
    "Position", [80 80 1300 600], "Visible", visStr);

% Left: 3D scene
ax3 = subplot(1, 2, 1, "Parent", figHandle);
surf(ax3, Xg, Yg, Zg, "EdgeColor", "none", "FaceAlpha", 0.85);
shading(ax3, "interp");
colormap(ax3, terrain_colormap());
hold(ax3, "on");

% Trees: cylinder mesh
for k = 1:Nobs
    draw_tree(ax3, obs_xyz(k,:), obs_rh(k,1), obs_rh(k,2));
end

% UAV trajectory
plot3(ax3, uav_xyz(:,1), uav_xyz(:,2), uav_xyz(:,3), ...
    "b-", "LineWidth", 1.5);

% UAV marker (animated)
uavMarker = plot3(ax3, uav_xyz(1,1), uav_xyz(1,2), uav_xyz(1,3), ...
    "Marker", "diamond", "MarkerSize", 14, ...
    "MarkerFaceColor", [0.10 0.45 0.95], ...
    "MarkerEdgeColor", "k", "LineStyle", "none", ...
    "LineWidth", 1.2);

% Camera FOV frustum lines (animated). Length 35 m.
camIntrin = evalin("base", "CAM_INTRIN");
pitch_rad = camIntrin(5) * pi / 180;
fovLen    = 35;
camLines = gobjects(4, 1);
for ii_l = 1:4
    camLines(ii_l) = plot3(ax3, [0 0], [0 0], [0 0], ...
        "Color", [0.9 0.2 0.2], "LineWidth", 1.2);
end

xlabel(ax3, "X (m)"); ylabel(ax3, "Y (m)"); zlabel(ax3, "Z (m)");
title(ax3, "Mountain Scene + UAV");
grid(ax3, "on"); axis(ax3, "equal"); view(ax3, 35, 30);
xlim(ax3, [min(Xg(:)) max(Xg(:))]);
ylim(ax3, [min(Yg(:)) max(Yg(:))]);
zlim(ax3, [0 max(Zg(:))+30]);

% Right: camera image plane
ax2 = subplot(1, 2, 2, "Parent", figHandle);

fog = evalin("base", "FOG_DENSITY_PERCENT");
illum = evalin("base", "ILLUMINATION_LUX");
noise = evalin("base", "CAMERA_NOISE_LEVEL");

img0 = render_camera_image(uav_xyz(1,:), obs_xyz, obs_rh, ...
    fog, illum, noise, camIntrin, [camW camH]);
imHandle = imagesc(ax2, img0);
set(ax2, "YDir", "reverse");
hold(ax2, "on");
xlim(ax2, [0 camW]); ylim(ax2, [0 camH]);
xlabel(ax2, "u (px)"); ylabel(ax2, "v (px)");
title(ax2, "Camera View (rendered scene + bboxes)");
axis(ax2, "image");

if opts.animate && opts.visible
    frameIdx = 1:max(1, opts.frameStride):Nt;
else
    % Static render: pick the frame with the most active GT bboxes
    gtPresentPerFrame = sum(any(gtBB ~= 0, 3), 2);   % Nt x 1
    [~, bestFrame] = max(gtPresentPerFrame);
    if isempty(bestFrame) || gtPresentPerFrame(bestFrame) == 0
        bestFrame = round(Nt / 2);
    end
    frameIdx = bestFrame;
end

for ii = frameIdx
    set(uavMarker, "XData", uav_xyz(ii,1), "YData", uav_xyz(ii,2), "ZData", uav_xyz(ii,3));

    % Update camera frustum lines
    fov_corners = camera_frustum_corners(uav_xyz(ii,:), pitch_rad, ...
        camIntrin(1), camIntrin(2), camIntrin(3), camIntrin(4), camW, camH, fovLen);
    for ll = 1:4
        set(camLines(ll), ...
            "XData", [uav_xyz(ii,1) fov_corners(ll,1)], ...
            "YData", [uav_xyz(ii,2) fov_corners(ll,2)], ...
            "ZData", [uav_xyz(ii,3) fov_corners(ll,3)]);
    end

    delete(findobj(ax2, "Tag", "bbox_overlay"));

    % Re-render the synthetic camera image for this frame
    frameImg = render_camera_image(uav_xyz(ii,:), obs_xyz, obs_rh, ...
        fog, illum, noise, camIntrin, [camW camH]);
    set(imHandle, "CData", frameImg);

    for k = 1:Nobs
        gt = squeeze(gtBB(ii, k, :))';
        dt = squeeze(detBB(ii, k, :))';
        sc = scores(ii, k);

        if any(gt ~= 0)
            rectangle("Parent", ax2, "Position", clamp_box(gt, camW, camH), ...
                "EdgeColor", [0.10 0.75 0.20], "LineStyle", "--", ...
                "LineWidth", 1.4, "Tag", "bbox_overlay");
        end
        if sc > 0.30 && any(dt ~= 0)
            rectangle("Parent", ax2, "Position", clamp_box(dt, camW, camH), ...
                "EdgeColor", [0.95 0.20 0.20], "LineWidth", 1.6, ...
                "Tag", "bbox_overlay");
            text(ax2, dt(1), max(8, dt(2) - 6), sprintf("%.2f", sc), ...
                "Color", [0.95 0.20 0.20], "FontWeight", "bold", ...
                "Tag", "bbox_overlay");
        end
    end

    set(figHandle, "Name", sprintf( ...
        "Mountain UAV  t=%.2fs  fog=%.0f%%  illum=%.0flx  noise=%.2f  mAP50=%.3f", ...
        t_vec(ii), fog, illum, noise, summary.map50));

    drawnow;
    if opts.visible && opts.animate, pause(opts.pauseSeconds); end
end

% Save PNG if requested
if strlength(opts.savePath) > 0
    try
        exportgraphics(figHandle, opts.savePath, "Resolution", 200);
        fprintf("[VIZ] Figure saved to %s\n", opts.savePath);
    catch ME
        warning("Figure save failed: %s", ME.message);
    end
end

end

% =========================================================================
% Helpers
% =========================================================================

function [t, vals] = read_log_vec(simOut, name)
% Returns time vector and Nt x M (or Nt) values for scalar / 1D / 2D signals.
s = read_signal(simOut, name);
t = s.time;

v = s.values;
sz = size(v);

if numel(sz) == 2 && (sz(1) == 1 || sz(2) == 1)
    vals = v(:);
elseif numel(sz) == 2
    vals = v;
elseif numel(sz) == 3
    % Format: m x n x Nt -> reshape to Nt x (m*n) for scalar series, but
    % typically used here for column vectors of length Nt.
    Nt = sz(3);
    vals = reshape(permute(v, [3 1 2]), Nt, sz(1)*sz(2));
else
    vals = v;
end
end

function [t, vals] = read_log_3d(simOut, name)
% For NxM signals over time. Returns vals as Nt x N x M.
s = read_signal(simOut, name);
t = s.time;
v = s.values;
sz = size(v);

if numel(sz) == 3
    vals = permute(v, [3 1 2]);  % Nt x N x M
elseif numel(sz) == 2
    Nt = numel(t);
    vals = reshape(v, [Nt, sz(2), 1]);
else
    vals = v;
end
end

function s = read_signal(simOut, name)
% Best-effort read of "Structure With Time" signal from simOut.
try
    raw = simOut.get(name);
catch
    raw = [];
end
if isempty(raw)
    try, raw = evalin("base", name); catch, raw = []; end
end
if isempty(raw)
    error("Could not find logged signal: %s", name);
end
if isstruct(raw) && isfield(raw, "time") && isfield(raw, "signals")
    s.time = raw.time;
    s.values = raw.signals.values;
else
    error("Unexpected signal format for %s", name);
end
end

function summary = compute_summary(gtBB, detBB, scores, Nobs)
Nt = size(gtBB, 1);
tp = 0; fp = 0; fn = 0;
ap50 = 0;

allDet = zeros(0, 2);   % rows: [score, tp_flag]
totalGt = 0;

for ii = 1:Nt
    for k = 1:Nobs
        gt = reshape(gtBB(ii, k, :), 1, []);
        dt = reshape(detBB(ii, k, :), 1, []);
        sc = scores(ii, k);

        gtPresent  = any(gt ~= 0);
        detPresent = (sc > 0.30) && any(dt ~= 0);

        if gtPresent
            totalGt = totalGt + 1;
        end

        if gtPresent && detPresent
            iou = bbox_iou(dt, gt);
            if iou >= 0.5
                tp = tp + 1;
                allDet(end+1, :) = [sc, 1]; %#ok<AGROW>
            else
                fp = fp + 1;
                fn = fn + 1;
                allDet(end+1, :) = [sc, 0]; %#ok<AGROW>
            end
        elseif gtPresent && ~detPresent
            fn = fn + 1;
        elseif ~gtPresent && detPresent
            fp = fp + 1;
            allDet(end+1, :) = [sc, 0]; %#ok<AGROW>
        end
    end
end

if totalGt > 0 && ~isempty(allDet)
    [~, ord] = sort(allDet(:,1), "descend");
    allDet = allDet(ord, :);
    cumTp = cumsum(allDet(:,2));
    cumFp = cumsum(1 - allDet(:,2));
    precision = cumTp ./ max(cumTp + cumFp, eps);
    recall = cumTp / totalGt;

    mrec = [0; recall; 1];
    mpre = [0; precision; 0];
    for i = numel(mpre)-1:-1:1
        mpre(i) = max(mpre(i), mpre(i+1));
    end
    idx = find(mrec(2:end) ~= mrec(1:end-1));
    ap50 = sum((mrec(idx+1) - mrec(idx)) .* mpre(idx+1));
end

summary = struct();
summary.tp       = double(tp);
summary.fp       = double(fp);
summary.fn       = double(fn);
summary.total_gt = double(totalGt);
summary.map50    = double(ap50);
end

function iou = bbox_iou(a, b)
ax1 = a(1); ay1 = a(2); ax2 = a(1)+a(3); ay2 = a(2)+a(4);
bx1 = b(1); by1 = b(2); bx2 = b(1)+b(3); by2 = b(2)+b(4);
ix1 = max(ax1, bx1); iy1 = max(ay1, by1);
ix2 = min(ax2, bx2); iy2 = min(ay2, by2);
iw = max(0, ix2-ix1); ih = max(0, iy2-iy1);
inter = iw*ih;
ua = max(0,a(3))*max(0,a(4)) + max(0,b(3))*max(0,b(4)) - inter;
if ua <= 0, iou = 0; else, iou = inter/ua; end
end

function box = clamp_box(b, w, h)
x = max(0, b(1));
y = max(0, b(2));
ww = max(2, min(b(3), w - x));
hh = max(2, min(b(4), h - y));
box = [x, y, ww, hh];
end

function draw_tree(ax, base, r, h)
[xc, yc, zc] = cylinder(r, 16);
zc = zc * h;
xc = xc + base(1);
yc = yc + base(2);
zc = zc + base(3);
surf(ax, xc, yc, zc, ...
    "EdgeColor", "none", ...
    "FaceColor", [0.10 0.55 0.20], ...
    "FaceAlpha", 0.95);
% Trunk: small brown box at base
[bx, by, bz] = cylinder(r*0.35, 8);
bz = bz * (h*0.30);
bx = bx + base(1); by = by + base(2); bz = bz + base(3);
surf(ax, bx, by, bz, ...
    "EdgeColor", "none", ...
    "FaceColor", [0.45 0.25 0.10]);
end

function img = render_camera_image(uav, obs_xyz, obs_rh, fog, illum, noise, cam_intrin, img_size)
% Renders a synthetic camera image: sky, mountain skyline (from base
% workspace TERRAIN_*), projected tree silhouettes, then applies
% fog/illumination/noise. Returns HxWx3 in [0,1].

W = img_size(1); H = img_size(2);
fx = cam_intrin(1); fy = cam_intrin(2);
cx = cam_intrin(3); cy = cam_intrin(4);
pitch = cam_intrin(5) * pi/180;
sp = sin(pitch); cp = cos(pitch);

% --- Sky / ground gradient (horizon depends on pitch) ---
% Camera pitched DOWN by 'pitch' rad: horizon moves UP in image
% (smaller v). Far horizontal points project to v = cy - fy*tan(pitch).
v_horizon = cy - fy * tan(pitch);
v_horizon = max(8, min(H-8, v_horizon));

img = zeros(H, W, 3);
vv = (1:H).';

sky_idx = 1:floor(v_horizon);
if ~isempty(sky_idx)
    a = vv(sky_idx) ./ max(1, v_horizon);
    img(sky_idx, :, 1) = repmat(0.55 + 0.30*a, 1, W);
    img(sky_idx, :, 2) = repmat(0.70 + 0.20*a, 1, W);
    img(sky_idx, :, 3) = repmat(0.88 + 0.08*a, 1, W);
end

gnd_idx = ceil(v_horizon):H;
if ~isempty(gnd_idx)
    a = (vv(gnd_idx) - v_horizon) ./ max(1, H - v_horizon);
    img(gnd_idx, :, 1) = repmat(0.32 - 0.18*a, 1, W);
    img(gnd_idx, :, 2) = repmat(0.46 - 0.22*a, 1, W);
    img(gnd_idx, :, 3) = repmat(0.20 - 0.10*a, 1, W);
end

% --- Mountain skyline (project terrain points, splat with depth fade) ---
img = paint_terrain(img, uav, fx, fy, cx, cy, sp, cp, W, H);

% --- Project + render trees, painter's algorithm (back to front) ---
N = size(obs_xyz, 1);
depth = nan(N, 1);
proj  = zeros(N, 4);

for k = 1:N
    [bbox, dval] = project_cylinder(uav, obs_xyz(k,:), obs_rh(k,1), obs_rh(k,2), ...
        cx, cy, fx, fy, sp, cp);
    if isempty(bbox), continue; end
    proj(k, :) = bbox;
    depth(k) = dval;
end

[~, order] = sort(depth, "descend", "MissingPlacement", "first");
for k = order(:).'
    if isnan(depth(k)), continue; end
    bbox = proj(k, :);
    if all(bbox == 0), continue; end
    img = paint_tree(img, bbox, depth(k), k);
end

% --- Weather effects ---
img = apply_weather(img, fog, illum, noise);

img = max(0, min(1, img));
end

function img = paint_terrain(img, uav, fx, fy, cx, cy, sp, cp, W, H)
% Projects subsampled terrain grid points into the camera image and splats
% each point with size matching its world cell footprint. Painter's
% algorithm (back-to-front) gives a smooth mountain skyline with
% atmospheric fade.

persistent Xs Ys Zs gridSpacing
if isempty(Xs) || isempty(gridSpacing)
    try
        Xg = evalin("base", "TERRAIN_X");
        Yg = evalin("base", "TERRAIN_Y");
        Zg = evalin("base", "TERRAIN_Z");
    catch
        return;
    end
    ds = 1;   % use full resolution (terrain grid already coarse)
    Xs = Xg(1:ds:end, 1:ds:end); Xs = Xs(:);
    Ys = Yg(1:ds:end, 1:ds:end); Ys = Ys(:);
    Zs = Zg(1:ds:end, 1:ds:end); Zs = Zs(:);
    % World spacing between adjacent samples (assumes uniform grid).
    if size(Xg,2) >= 2
        gridSpacing = abs(Xg(1,2) - Xg(1,1)) * ds;
    else
        gridSpacing = 2.0 * ds;
    end
end

dx = Xs - uav(1);
dy = Ys - uav(2);
dz = Zs - uav(3);

cz = dx*cp - dz*sp;
valid = cz > 1.0;
if ~any(valid), return; end

dx_v = dx(valid); dy_v = dy(valid); dz_v = dz(valid);
cz_v = cz(valid);
cam_x = dy_v;
cam_y = -dx_v*sp - dz_v*cp;

u = fx * cam_x ./ cz_v + cx;
v = fy * cam_y ./ cz_v + cy;

inFOV = u >= -50 & u <= W+50 & v >= -50 & v <= H+50;
u = u(inFOV); v = v(inFOV); cz_v = cz_v(inFOV);
Zs_v = Zs(valid); Zs_v = Zs_v(inFOV);
if isempty(u), return; end

[~, ord] = sort(cz_v, "descend");
u = u(ord); v = v(ord); cz_v = cz_v(ord); Zs_v = Zs_v(ord);

sky_col = [0.78 0.84 0.92];
ui = round(u); vi = round(v);

% Splat half-size matches projected world cell, with floor of 1 px
splatRadius = max(1, ceil(0.55 * gridSpacing * fx ./ cz_v));
splatRadius = min(splatRadius, 35);

% Pre-compute terrain max for color elevation banding
zMax = max(Zs);

for k = 1:length(ui)
    cz_k = cz_v(k);
    fade = max(0.20, min(1, 1 - cz_k/140));

    % Elevation banding: low = green, mid = brown, high = grey/white
    h_norm = max(0, min(1, Zs_v(k) / max(1, zMax)));
    if h_norm < 0.45
        baseCol = [0.22 0.45 0.18];               % grass
    elseif h_norm < 0.75
        a = (h_norm - 0.45) / 0.30;
        baseCol = [0.22 0.45 0.18] * (1-a) + [0.55 0.45 0.30] * a;   % grass -> rocky
    else
        a = (h_norm - 0.75) / 0.25;
        baseCol = [0.55 0.45 0.30] * (1-a) + [0.85 0.85 0.88] * a;   % rocky -> snow
    end
    earth = baseCol * fade + sky_col * (1 - fade);

    rad = splatRadius(k);
    u0 = max(1, ui(k) - rad); u1_b = min(W, ui(k) + rad);
    v0 = max(1, vi(k) - rad); v1_b = min(H, vi(k) + rad);
    if u1_b < u0 || v1_b < v0, continue; end

    img(v0:v1_b, u0:u1_b, 1) = earth(1);
    img(v0:v1_b, u0:u1_b, 2) = earth(2);
    img(v0:v1_b, u0:u1_b, 3) = earth(3);
end
end

function [bbox, depth] = project_cylinder(uav, base, r, h, cx, cy, fx, fy, sp, cp)
dx = base(1) - uav(1);
dy = base(2) - uav(2);
dz_b = base(3) - uav(3);
dz_t = (base(3) + h) - uav(3);

cz_b = dx*cp - dz_b*sp;
cz_t = dx*cp - dz_t*sp;
if cz_b < 0.5 || cz_t < 0.5
    bbox = []; depth = NaN; return;
end

cz_avg = 0.5*(cz_b + cz_t);

cam_x_l = dy - r;
cam_x_r = dy + r;
cam_y_t = -dx*sp - dz_t*cp;
cam_y_b = -dx*sp - dz_b*cp;

u_l = fx * cam_x_l / cz_avg + cx;
u_r = fx * cam_x_r / cz_avg + cx;
v_t = fy * cam_y_t / cz_t + cy;
v_b = fy * cam_y_b / cz_b + cy;

bbox = [u_l, v_t, u_r - u_l, v_b - v_t];
depth = cz_avg;
end

function img = paint_tree(img, bbox, depth, treeId)
% Renders a more realistic tree: ellipse crown with directional shading,
% per-tree color variation, atmospheric fade, tapered trunk.
[H, W, ~] = size(img);

u1 = max(1, floor(bbox(1)));
u2 = min(W, ceil(bbox(1) + bbox(3)));
v1 = max(1, floor(bbox(2)));
v2 = min(H, ceil(bbox(2) + bbox(4)));
if u2 <= u1 || v2 <= v1, return; end

bw = u2 - u1 + 1;
bh = v2 - v1 + 1;
if bw < 2 || bh < 4, return; end

% --- Per-tree color variation (deterministic from id) ---
hueShift = 0.85 + 0.30 * mod(treeId * 0.6180339, 1);   % 0.85..1.15
crownBase = [0.12, 0.42, 0.16] .* hueShift;
trunkBase = [0.38, 0.22, 0.10] .* (0.95 + 0.10*mod(treeId*0.37, 1));

% --- Atmospheric perspective: blend toward sky with depth ---
sky = [0.78, 0.84, 0.92];
fade = max(0.30, min(1, 1 - depth/110));
crownColor = crownBase * fade + sky * (1 - fade);
trunkColor = trunkBase * fade + sky * (1 - fade);

% --- Layout: crown 75% top, trunk 25% bottom ---
crownH = max(2, round(bh * 0.75));
crownEnd = v1 + crownH - 1;
trunkV1 = crownEnd + 1;

% --- Crown: ellipse with directional shading ---
[uu, vv] = meshgrid(u1:u2, v1:crownEnd);
cx_b = (u1 + u2) / 2;
cy_b = (v1 + crownEnd) / 2;
rx = max(1, bw / 2 - 0.3);
ry = max(1, crownH / 2 - 0.3);

nx = (uu - cx_b) / rx;
ny = (vv - cy_b) / ry;
mask = (nx.^2 + ny.^2) <= 1;

% Shading: lighter top-left (sun from upper-left), darker bottom-right
shade = 0.78 + 0.32 * (-nx*0.55 - ny*0.65);
shade = max(0.55, min(1.20, shade));

% Subtle leaf-cluster texture: low-amplitude noise tied to position
texture = 0.92 + 0.16 * sin(3.1*uu + treeId).*cos(2.7*vv + treeId*1.3);
shade = shade .* texture;

R_slice = squeeze(img(v1:crownEnd, u1:u2, 1));
G_slice = squeeze(img(v1:crownEnd, u1:u2, 2));
B_slice = squeeze(img(v1:crownEnd, u1:u2, 3));

R_slice(mask) = crownColor(1) * shade(mask);
G_slice(mask) = crownColor(2) * shade(mask);
B_slice(mask) = crownColor(3) * shade(mask);

img(v1:crownEnd, u1:u2, 1) = R_slice;
img(v1:crownEnd, u1:u2, 2) = G_slice;
img(v1:crownEnd, u1:u2, 3) = B_slice;

% --- Trunk: tapered, with vertical light gradient ---
if trunkV1 <= v2
    trunkW_top    = max(1, round(bw * 0.20));
    trunkW_bottom = max(1, round(bw * 0.28));

    for vv_t = trunkV1:v2
        a = (vv_t - trunkV1) / max(1, v2 - trunkV1);
        wThis = round(trunkW_top * (1 - a) + trunkW_bottom * a);
        ucenter = round(cx_b);
        uL = max(u1, ucenter - floor(wThis/2));
        uR = min(u2, uL + wThis - 1);
        if uR < uL, continue; end

        % Vertical shading: top brighter, bottom darker (ground shadow)
        vShade = 1.05 - 0.40 * a;
        img(vv_t, uL:uR, 1) = trunkColor(1) * vShade;
        img(vv_t, uL:uR, 2) = trunkColor(2) * vShade;
        img(vv_t, uL:uR, 3) = trunkColor(3) * vShade;
    end
end
end

function img = apply_weather(img, fog, illum, noise)
[H, W, C] = size(img);

% Fog: blend toward whitish
fog_norm = max(0, min(1, fog/100));
fogBlend = 0.75 * fog_norm;
fogColor = cat(3, 0.86, 0.89, 0.92);
img = img * (1 - fogBlend) + repmat(fogColor, [H, W, 1]) * fogBlend;

% Illumination: brightness gain + low-light blue tint
illum_norm = max(0, min(1.5, illum / 8000));
brightness = 0.40 + 0.65 * min(1, illum_norm);
img = img * brightness;

if illum_norm < 0.6
    blueShift = (0.6 - illum_norm) * 0.18;
    img(:,:,1) = img(:,:,1) - 0.5*blueShift;
    img(:,:,3) = img(:,:,3) + 0.4*blueShift;
end

% Camera noise
if noise > 0
    img = img + (noise * 0.12) * randn(H, W, C);
end
end

function img = make_camera_background(w, h, pitch_rad, fy, cy)
% Sky/ground gradient with horizon position consistent with the camera pitch.
% Camera pitched down: horizon at v = cy - fy * tan(pitch_rad).
if nargin < 3, pitch_rad = 0; end
if nargin < 4, fy = h*1.0; end
if nargin < 5, cy = h*0.5; end

v_horizon = cy - fy * tan(pitch_rad);
v_horizon = max(8, min(h-8, v_horizon));

img = zeros(h, w, 3);
for v = 1:h
    if v < v_horizon
        a = v / max(1, v_horizon);
        img(v,:,1) = 0.55 + 0.30*a;
        img(v,:,2) = 0.70 + 0.20*a;
        img(v,:,3) = 0.85 + 0.10*a;
    else
        a = (v - v_horizon) / max(1, h - v_horizon);
        img(v,:,1) = 0.50 - 0.25*a;
        img(v,:,2) = 0.60 - 0.20*a;
        img(v,:,3) = 0.35 - 0.15*a;
    end
end
img = max(0, min(1, img));
end

function rgb = weather_tint(fog, illum)
fog_norm = max(0, min(1, fog/100));
illum_norm = max(0, min(1, illum/15000));
gray = 0.55 + 0.4*fog_norm;
brightness = 0.4 + 0.6*illum_norm;
rgb = (gray * brightness) * [1 1 1];
rgb = max(0, min(1, rgb));
end

function corners = camera_frustum_corners(uav, pitch, fx, fy, cx, cy, w, h, depth)
% Returns 4x3 world-coordinate positions of the 4 corners of the FOV at
% the given depth (along camera z axis), assuming camera looks +X then
% pitches down by 'pitch' rad (around world Y axis).
sp = sin(pitch); cp = cos(pitch);

% Pixel corners
pix = [0 0; w 0; w h; 0 h];
corners = zeros(4, 3);

for k = 1:4
    u = pix(k, 1); v = pix(k, 2);
    % Ray in camera frame at depth=1
    cam_x = (u - cx) / fx;
    cam_y = (v - cy) / fy;
    cam_z = 1.0;
    % Scale to requested depth
    s = depth / cam_z;
    cam_x = cam_x * s; cam_y = cam_y * s; cam_z = cam_z * s;

    % Camera frame -> world (inverse of pitched rotation)
    %   z_c_world = ( cp, 0, -sp )
    %   x_c_world = ( 0,  1,  0  )
    %   y_c_world = (-sp, 0, -cp )
    world_off = cam_x * [0 1 0] + cam_y * [-sp 0 -cp] + cam_z * [cp 0 -sp];
    corners(k, :) = uav + world_off;
end
end

function cmap = terrain_colormap()
cmap = [
    0.20 0.40 0.10
    0.30 0.55 0.15
    0.50 0.65 0.30
    0.65 0.55 0.35
    0.70 0.55 0.40
    0.80 0.70 0.55
    0.92 0.92 0.92
];
cmap = interp1(linspace(0,1,size(cmap,1)), cmap, linspace(0,1,128));
end
