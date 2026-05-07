function img = render_camera_image(uav, obs_xyz, obs_rh, fog, illum, noise, cam_intrin, img_size)
% render_camera_image — Synthetic surveillance camera renderer.
%
% Returns HxWx3 image in [0,1]. Auto-switches between two render modes:
%   * Forward-perspective (pitch < 30°): sky + horizon + mountain skyline
%   * Top-down surveillance (pitch >= 30°): per-pixel ground ray-casting
%     onto TERRAIN_Z, no horizon, plus procedural forest texture
%
% Also reads OBSTACLES_CLASS (Nx1, 1=person, 2=vehicle) and SCENERY_OBJECTS
% (Mx5: x, y, z, radius, type) from the base workspace if available, so
% intruders are rendered with the correct silhouette and non-target trees /
% rocks appear as background clutter for visual realism.
%
% cam_intrin: [fx, fy, cx, cy, pitch_down_deg]
% img_size:   [W, H]

W = img_size(1); H = img_size(2);
fx = cam_intrin(1); fy = cam_intrin(2);
cx = cam_intrin(3); cy = cam_intrin(4);
pitch_deg = cam_intrin(5);
pitch = pitch_deg * pi/180;
sp = sin(pitch); cp = cos(pitch);

img = zeros(H, W, 3);

% Read class labels (1=person, 2=vehicle); default to all-person if absent
try
    obs_class = evalin("base", "OBSTACLES_CLASS");
catch
    obs_class = ones(size(obs_xyz, 1), 1);
end

% Read scenery objects (non-target background) if available
try
    scenery = evalin("base", "SCENERY_OBJECTS");   % Mx5: [x, y, z, r, type]
catch
    scenery = zeros(0, 5);
end

% =========================================================================
% Step 1: Ground / sky base
% =========================================================================
if pitch_deg >= 30
    % Top-down surveillance mode — ray-cast every pixel onto terrain
    img = paint_topdown_ground(img, uav, fx, fy, cx, cy, sp, cp, W, H);
else
    % Forward perspective mode — sky/horizon + mountain skyline
    img = paint_forward_sky(img, fy, cy, pitch, W, H);
    img = paint_terrain_skyline(img, uav, fx, fy, cx, cy, sp, cp, W, H);
end

% =========================================================================
% Step 2: Scenery (procedural background — non-target trees, rocks)
% =========================================================================
%   Drawn BEFORE intruders so intruders appear in front. Painter's
%   algorithm: back-to-front by depth.
all_back = build_scene_objects(scenery, [], []);
img = paint_objects_painter(img, uav, all_back, ...
    fx, fy, cx, cy, sp, cp, W, H, "scenery");

% =========================================================================
% Step 3: Intruders (people + vehicles — the actual detection targets)
% =========================================================================
intruder_list = build_scene_objects([], obs_xyz, obs_rh, obs_class);
img = paint_objects_painter(img, uav, intruder_list, ...
    fx, fy, cx, cy, sp, cp, W, H, "intruder");

% =========================================================================
% Step 4: Weather effects (fog, illumination, noise) — last
% =========================================================================
img = apply_weather(img, fog, illum, noise);

img = max(0, min(1, img));
end


% =========================================================================
% Top-down ground rendering — ray-cast every pixel onto the terrain mesh
% =========================================================================
function img = paint_topdown_ground(img, uav, fx, fy, cx, cy, sp, cp, W, H)
try
    Xg = evalin("base", "TERRAIN_X");
    Yg = evalin("base", "TERRAIN_Y");
    Zg = evalin("base", "TERRAIN_Z");
catch
    img(:) = 0.4;       % grey fallback
    return;
end

% Pixel rays in camera frame
[U, V] = meshgrid(1:W, 1:H);
xc = (U - cx) / fx;
yc = (V - cy) / fy;

% Camera frame in world: pitched DOWN by 'pitch'.
%   Camera +Z (forward) = ( cp, 0, -sp )
%   Camera +X (right)   = ( 0,  1,  0  )
%   Camera +Y (down)    = (-sp, 0, -cp )
% World ray = xc * cam_X + yc * cam_Y + 1 * cam_Z
r_x = -sp .* yc + cp;
r_y = xc;
r_z = -cp .* yc - sp;

% Intersect with z = 0 (mean ground plane); clamp ray to look-down only
valid = r_z < -1e-3;
t = -uav(3) ./ r_z;
t(~valid | t <= 0) = NaN;

Xw = uav(1) + t .* r_x;
Yw = uav(2) + t .* r_y;

% Sample terrain elevation
Xmin = min(Xg(:)); Xmax = max(Xg(:));
Ymin = min(Yg(:)); Ymax = max(Yg(:));
in_range = ~isnan(Xw) & Xw >= Xmin & Xw <= Xmax & Yw >= Ymin & Yw <= Ymax;
Zw = zeros(H, W);
Zw(in_range) = interp2(Xg, Yg, Zg, Xw(in_range), Yw(in_range), "linear", 0);

% Elevation-based shading (low=forest green, mid=dry grass, high=rocky)
zmax = max(Zg(:));
if zmax < 1, zmax = 1; end
e = max(0, min(1, Zw / zmax));            % 0..1

% Three colour bands blended by elevation
forest = [0.18, 0.42, 0.22];     % dark green
grass  = [0.46, 0.52, 0.28];     % dry yellow-green
rock   = [0.55, 0.50, 0.42];     % brown-grey

w_forest = max(0, 1 - 2.5 * e);
w_rock   = max(0, 2.5 * e - 1.5);
w_grass  = max(0, 1 - w_forest - w_rock);
norm = w_forest + w_grass + w_rock + 1e-9;
w_forest = w_forest ./ norm;
w_grass  = w_grass  ./ norm;
w_rock   = w_rock   ./ norm;

R = w_forest .* forest(1) + w_grass .* grass(1) + w_rock .* rock(1);
G = w_forest .* forest(2) + w_grass .* grass(2) + w_rock .* rock(2);
B = w_forest .* forest(3) + w_grass .* grass(3) + w_rock .* rock(3);

% Procedural texture — high-freq sin noise simulates canopy/grass texture
tex = 0.06 * sin(0.55 * Xw) .* cos(0.65 * Yw) ...
    + 0.04 * sin(0.20 * Xw + 0.30 * Yw);
R = R + tex;  G = G + tex * 1.2;  B = B + tex * 0.6;

% Hill-shading: brighten if neighbour pixel has lower terrain
shade = zeros(H, W);
shade(2:end, :) = (Zw(2:end, :) - Zw(1:end-1, :)) * 0.04;
R = R + shade;  G = G + shade;  B = B + shade;

% Out-of-terrain = pale grey haze (we're high up, edge of mapped area)
oof = ~in_range;
R(oof) = 0.50;  G(oof) = 0.55;  B(oof) = 0.60;

img(:, :, 1) = max(0, min(1, R));
img(:, :, 2) = max(0, min(1, G));
img(:, :, 3) = max(0, min(1, B));
end


% =========================================================================
% Forward-perspective sky/horizon — kept for low-pitch (legacy) modes
% =========================================================================
function img = paint_forward_sky(img, fy, cy, pitch, W, H)
v_horizon = cy - fy * tan(pitch);
v_horizon = max(8, min(H-8, v_horizon));
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
end


function img = paint_terrain_skyline(img, uav, fx, fy, cx, cy, sp, cp, W, H)
% Forward-perspective terrain skyline — only used for pitch < 30°.
try
    Xg = evalin("base", "TERRAIN_X");
    Yg = evalin("base", "TERRAIN_Y");
    Zg = evalin("base", "TERRAIN_Z");
catch
    return;
end
% Sample the mountain ridge sparsely and project columns
[Hmesh, Wmesh] = size(Zg);
step = max(1, round(min(Hmesh, Wmesh) / 90));
for j = 1:step:Wmesh
    for i = 1:step:Hmesh
        wx = Xg(i, j); wy = Yg(i, j); wz = Zg(i, j);
        dx = wx - uav(1); dy = wy - uav(2); dz = wz - uav(3);
        cz = dx * cp - dz * sp;
        if cz < 1, continue; end
        cy_pix = -dx * sp - dz * cp;
        u = round(fx * dy   / cz + cx);
        v = round(fy * cy_pix / cz + cy);
        if u >= 1 && u <= W && v >= 1 && v <= H
            shade = max(0, 1 - cz / 200);
            img(v, u, 1) = 0.30 * shade;
            img(v, u, 2) = 0.40 * shade;
            img(v, u, 3) = 0.20 * shade;
        end
    end
end
end


% =========================================================================
% Object list builder — unifies intruders and scenery into one schema
% =========================================================================
function objs = build_scene_objects(scenery, obs_xyz, obs_rh, varargin)
% Returns Nx struct-like cell with fields needed by the painter:
%   pos[3], r, h, kind ("person", "vehicle", "tree", "rock"), color
objs = struct("pos", {}, "r", {}, "h", {}, "kind", {}, "color", {});
% Scenery (background)
for k = 1:size(scenery, 1)
    obj.pos   = scenery(k, 1:3);
    obj.r     = scenery(k, 4);
    obj.h     = scenery(k, 4) * 1.6;       % scenery height ~ 1.6 * r
    type      = scenery(k, 5);
    if type == 1
        obj.kind  = "tree";
        obj.color = [0.10, 0.35, 0.15];
        obj.h     = max(2.0, scenery(k, 4) * 3.5);    % trees tall
    else
        obj.kind  = "rock";
        obj.color = [0.45, 0.42, 0.38];
        obj.h     = scenery(k, 4) * 0.7;              % rocks flat
    end
    objs(end + 1) = obj; %#ok<AGROW>
end
% Intruders (foreground, real targets)
if ~isempty(obs_xyz) && ~isempty(varargin)
    obs_class = varargin{1};
    for k = 1:size(obs_xyz, 1)
        obj.pos = obs_xyz(k, :);
        obj.r   = obs_rh(k, 1);
        obj.h   = obs_rh(k, 2);
        if k <= numel(obs_class) && obs_class(k) == 2
            obj.kind  = "vehicle";
            obj.color = [0.95, 0.55, 0.10];   % orange
        else
            obj.kind  = "person";
            obj.color = [0.20, 0.50, 0.95];   % blue
        end
        objs(end + 1) = obj; %#ok<AGROW>
    end
end
end


% =========================================================================
% Painter's algorithm — back-to-front rendering of all objects
% =========================================================================
function img = paint_objects_painter(img, uav, objs, fx, fy, cx, cy, sp, cp, W, H, role)
N = numel(objs);
if N == 0, return; end
depth = nan(N, 1);
bb    = zeros(N, 4);
for k = 1:N
    [bbox, dval] = project_cylinder(uav, objs(k).pos, objs(k).r, objs(k).h, ...
        cx, cy, fx, fy, sp, cp);
    if isempty(bbox), continue; end
    bb(k, :)   = bbox;
    depth(k)   = dval;
end
[~, ord] = sort(depth, "descend", "MissingPlacement", "first");
for k = ord(:).'
    if isnan(depth(k)), continue; end
    bbox = bb(k, :);
    if all(bbox == 0), continue; end
    img = paint_one_object(img, bbox, depth(k), objs(k), W, H);
end
end


function img = paint_one_object(img, bbox, depth, obj, W, H)
u0 = max(1, round(bbox(1)));
v0 = max(1, round(bbox(2)));
u1 = min(W, round(bbox(1) + bbox(3)));
v1 = min(H, round(bbox(2) + bbox(4)));
if u1 < u0 || v1 < v0, return; end

% Distance attenuation (closer = darker / clearer; far = washed out)
atten = max(0.4, 1 - depth / 180);
c     = obj.color * atten;

if obj.kind == "tree"
    % Procedural canopy: fill ellipse-ish region with green + edge dither
    [vv, uu] = ndgrid(v0:v1, u0:u1);
    cu = (u0 + u1) / 2; cv = (v0 + v1) / 2;
    rad_u = max(1, (u1 - u0) / 2);
    rad_v = max(1, (v1 - v0) / 2);
    mask = ((uu - cu) / rad_u) .^ 2 + ((vv - cv) / rad_v) .^ 2 <= 1;
    for ch = 1:3
        sl = img(v0:v1, u0:u1, ch);
        sl(mask) = c(ch) + 0.04 * sin(uu(mask) + vv(mask));
        img(v0:v1, u0:u1, ch) = sl;
    end
elseif obj.kind == "rock"
    img(v0:v1, u0:u1, 1) = c(1);
    img(v0:v1, u0:u1, 2) = c(2);
    img(v0:v1, u0:u1, 3) = c(3);
elseif obj.kind == "vehicle"
    % Vehicle: filled rectangle with darker hood band on top
    img(v0:v1, u0:u1, 1) = c(1);
    img(v0:v1, u0:u1, 2) = c(2);
    img(v0:v1, u0:u1, 3) = c(3);
    band = min(v1, v0 + max(1, round((v1 - v0) * 0.35)));
    img(v0:band, u0:u1, :) = img(v0:band, u0:u1, :) * 0.6;
elseif obj.kind == "person"
    % Person: filled torso (lower 70%) + head (upper 30%, slightly narrower)
    split = v0 + round((v1 - v0) * 0.30);
    img(split:v1, u0:u1, 1) = c(1);
    img(split:v1, u0:u1, 2) = c(2);
    img(split:v1, u0:u1, 3) = c(3);
    head_u0 = max(u0, u0 + round((u1 - u0) * 0.20));
    head_u1 = min(u1, u1 - round((u1 - u0) * 0.20));
    if head_u1 >= head_u0
        head_color = [0.85, 0.70, 0.55];     % skin tone
        img(v0:split, head_u0:head_u1, 1) = head_color(1);
        img(v0:split, head_u0:head_u1, 2) = head_color(2);
        img(v0:split, head_u0:head_u1, 3) = head_color(3);
    end
end
end


% =========================================================================
% Project a cylinder (axis-aligned) into the image; returns bbox + depth
% =========================================================================
function [bbox, depth_val] = project_cylinder(uav, base, r, h, cx, cy, fx, fy, sp, cp)
bbox = [];  depth_val = NaN;

dx = base(1) - uav(1);
dy = base(2) - uav(2);
dz_base = base(3) - uav(3);
dz_top  = (base(3) + h) - uav(3);

cz_base = dx * cp - dz_base * sp;
cz_top  = dx * cp - dz_top  * sp;
if cz_base < 0.5 || cz_top < 0.5, return; end
cz_avg = 0.5 * (cz_base + cz_top);
if cz_avg > 200, return; end

cam_x_left  = dy - r;
cam_x_right = dy + r;
cam_y_top   = -dx * sp - dz_top  * cp;
cam_y_bot   = -dx * sp - dz_base * cp;

u_left  = fx * cam_x_left  / cz_avg  + cx;
u_right = fx * cam_x_right / cz_avg  + cx;
v_top   = fy * cam_y_top   / cz_top  + cy;
v_bot   = fy * cam_y_bot   / cz_base + cy;

bbox_w = u_right - u_left;
bbox_h = v_bot   - v_top;
if bbox_w <= 0 || bbox_h <= 0, return; end

bbox      = [u_left, v_top, bbox_w, bbox_h];
depth_val = cz_avg;
end


% =========================================================================
% Weather post-processing
% =========================================================================
function img = apply_weather(img, fog, illum, noise)
fog_norm   = max(0, min(1, fog / 100));
low_light  = max(0, (3000  - illum) / 3000);
high_light = max(0, (illum - 12000) / 12000);

% Fog: blend toward grey-white
fog_color = reshape([0.85, 0.85, 0.88], 1, 1, 3);
img = (1 - 0.85 * fog_norm) .* img + (0.85 * fog_norm) .* fog_color;

% Illumination: dim or wash out
if low_light > 0
    img = img * max(0.10, 1 - 0.85 * low_light);
end
if high_light > 0
    img = (1 - 0.4 * high_light) .* img + (0.4 * high_light) .* 1.0;
end

% Noise: deterministic "salt & speckle" pattern (no randn for codegen safety)
if noise > 0
    [H, W, ~] = size(img);
    [UU, VV] = meshgrid(1:W, 1:H);
    pattern = sin(0.83 * UU + 1.7 * VV) .* cos(2.1 * UU - 0.6 * VV);
    speckle = noise * 0.40 * pattern;
    img = img + repmat(speckle, [1, 1, 3]);
end
end
