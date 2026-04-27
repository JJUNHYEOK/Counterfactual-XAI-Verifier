function img = render_camera_image(uav, obs_xyz, obs_rh, fog, illum, noise, cam_intrin, img_size)
% render_camera_image — Top-level synthetic camera renderer.
% Returns HxWx3 image in [0,1] given UAV pose, obstacles, weather and camera
% intrinsics. Also reads TERRAIN_X/Y/Z from base workspace for the mountain
% skyline pass. Used by both mountain_visualizer.m (for the right-hand
% panel) and image_detector.m / image_based_eval (for image-feature
% detection).
%
% cam_intrin: [fx, fy, cx, cy, pitch_down_deg]
% img_size:   [W, H]

W = img_size(1); H = img_size(2);
fx = cam_intrin(1); fy = cam_intrin(2);
cx = cam_intrin(3); cy = cam_intrin(4);
pitch = cam_intrin(5) * pi/180;
sp = sin(pitch); cp = cos(pitch);

% --- Sky / ground gradient (horizon depends on pitch) ---
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

% --- Mountain skyline ---
img = paint_terrain(img, uav, fx, fy, cx, cy, sp, cp, W, H);

% --- Trees, painter's algorithm (back to front) ---
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

% =========================================================================
function img = paint_terrain(img, uav, fx, fy, cx, cy, sp, cp, W, H)
persistent Xs Ys Zs gridSpacing
if isempty(Xs) || isempty(gridSpacing)
    try
        Xg = evalin("base", "TERRAIN_X");
        Yg = evalin("base", "TERRAIN_Y");
        Zg = evalin("base", "TERRAIN_Z");
    catch
        return;
    end
    ds = 1;
    Xs = Xg(1:ds:end, 1:ds:end); Xs = Xs(:);
    Ys = Yg(1:ds:end, 1:ds:end); Ys = Ys(:);
    Zs = Zg(1:ds:end, 1:ds:end); Zs = Zs(:);
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
splatRadius = max(1, ceil(0.55 * gridSpacing * fx ./ cz_v));
splatRadius = min(splatRadius, 35);

zMax = max(Zs);

for k = 1:length(ui)
    cz_k = cz_v(k);
    fade = max(0.20, min(1, 1 - cz_k/140));
    h_norm = max(0, min(1, Zs_v(k) / max(1, zMax)));
    if h_norm < 0.45
        baseCol = [0.22 0.45 0.18];
    elseif h_norm < 0.75
        a = (h_norm - 0.45) / 0.30;
        baseCol = [0.22 0.45 0.18] * (1-a) + [0.55 0.45 0.30] * a;
    else
        a = (h_norm - 0.75) / 0.25;
        baseCol = [0.55 0.45 0.30] * (1-a) + [0.85 0.85 0.88] * a;
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

% =========================================================================
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

bbox  = [u_l, v_t, u_r - u_l, v_b - v_t];
depth = cz_avg;
end

% =========================================================================
function img = paint_tree(img, bbox, depth, treeId)
[H, W, ~] = size(img);

u1 = max(1, floor(bbox(1)));
u2 = min(W, ceil(bbox(1) + bbox(3)));
v1 = max(1, floor(bbox(2)));
v2 = min(H, ceil(bbox(2) + bbox(4)));
if u2 <= u1 || v2 <= v1, return; end

bw = u2 - u1 + 1;
bh = v2 - v1 + 1;
if bw < 2 || bh < 4, return; end

hueShift = 0.85 + 0.30 * mod(treeId * 0.6180339, 1);
crownBase = [0.12, 0.42, 0.16] .* hueShift;
trunkBase = [0.38, 0.22, 0.10] .* (0.95 + 0.10*mod(treeId*0.37, 1));

sky = [0.78, 0.84, 0.92];
fade = max(0.30, min(1, 1 - depth/110));
crownColor = crownBase * fade + sky * (1 - fade);
trunkColor = trunkBase * fade + sky * (1 - fade);

crownH = max(2, round(bh * 0.75));
crownEnd = v1 + crownH - 1;
trunkV1 = crownEnd + 1;

[uu, vv] = meshgrid(u1:u2, v1:crownEnd);
cx_b = (u1 + u2) / 2;
cy_b = (v1 + crownEnd) / 2;
rx = max(1, bw / 2 - 0.3);
ry = max(1, crownH / 2 - 0.3);

nx = (uu - cx_b) / rx;
ny = (vv - cy_b) / ry;
mask = (nx.^2 + ny.^2) <= 1;

shade = 0.78 + 0.32 * (-nx*0.55 - ny*0.65);
shade = max(0.55, min(1.20, shade));

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
        vShade = 1.05 - 0.40 * a;
        img(vv_t, uL:uR, 1) = trunkColor(1) * vShade;
        img(vv_t, uL:uR, 2) = trunkColor(2) * vShade;
        img(vv_t, uL:uR, 3) = trunkColor(3) * vShade;
    end
end
end

% =========================================================================
function img = apply_weather(img, fog, illum, noise)
[H, W, C] = size(img);

fog_norm = max(0, min(1, fog/100));

% --- Fog blur: smaller particles obscure fine detail (edges fade) ---
% Box-blur radius scales with fog density. Cheap, no Image Processing Toolbox.
blurR = round(fog_norm * 2.5);   % 0..2 pixels at fog=100
if blurR > 0
    k = ones(2*blurR+1, 2*blurR+1) / ((2*blurR+1)^2);
    for c = 1:C
        img(:,:,c) = conv2(img(:,:,c), k, "same");
    end
end

% --- Fog haze: blend toward whitish ---
fogBlend = 0.75 * fog_norm;
fogColor = cat(3, 0.86, 0.89, 0.92);
img = img * (1 - fogBlend) + repmat(fogColor, [H, W, 1]) * fogBlend;

% --- Illumination: brightness gain + low-light blue tint ---
illum_norm = max(0, min(1.5, illum / 8000));
brightness = 0.40 + 0.65 * min(1, illum_norm);
img = img * brightness;

if illum_norm < 0.6
    blueShift = (0.6 - illum_norm) * 0.18;
    img(:,:,1) = img(:,:,1) - 0.5*blueShift;
    img(:,:,3) = img(:,:,3) + 0.4*blueShift;
end

% --- Camera noise ---
if noise > 0
    img = img + (noise * 0.12) * randn(H, W, C);
end
end
