function img = eo_camera_3d_render(uav, obs_xyz, obs_rh, fog, illum, noise, cam_intrin, img_size)
% eo_camera_3d_render — UAV 1인칭 시점의 진짜 3D EO 카메라 이미지 렌더러.
%
% render_camera_image (2D synthetic) 의 드롭인 대체 (drop-in replacement).
%
% 첫 호출에서 hidden offscreen figure 에 3D 씬을 구축하고, 이후 호출에서는
% 카메라 위치 + 침입자 위치만 갱신 후 getframe 으로 캡처. fog/illum/noise
% post-processing 까지 동일하게 적용해서 결과 이미지를 반환.
%
% 인자/반환은 render_camera_image 와 동일하므로 호출부 변경 최소.

W = img_size(1); H = img_size(2);
fx = cam_intrin(1); fy = cam_intrin(2);
cx = cam_intrin(3); cy = cam_intrin(4); %#ok<NASGU>
pitch_deg = cam_intrin(5);

% Persistent hidden figure + handles so we set up the scene only once.
persistent eo3d;
if isempty(eo3d) || ~isgraphics(eo3d.fig)
    eo3d = setup_scene(W, H);
end

% Refresh intruders only when positions differ from the previously drawn
% set (handles + position cache stored in eo3d.last_obs_xyz). Most of the
% time the intruders are stationary so this is a no-op after first frame.
if ~isfield(eo3d, "last_obs_xyz") || ~isequal(eo3d.last_obs_xyz, obs_xyz) ...
        || ~isequal(eo3d.last_obs_rh, obs_rh)
    eo3d = redraw_intruders(eo3d, obs_xyz, obs_rh);
end

% Position camera at UAV with given pitch_down
configure_camera(eo3d.ax, uav, pitch_deg, fy, H);

drawnow limitrate;
frame = getframe(eo3d.ax);
img = double(frame.cdata) / 255;

% Resize captured frame to expected camera image size (W, H)
if size(img, 1) ~= H || size(img, 2) ~= W
    img = imresize(img, [H W]);
end

% MATLAB's 3D camera handedness puts world +y to image LEFT, but the
% pinhole projector that computes GT bboxes (project_cylinder in
% render_camera_image.m) puts +y to image RIGHT. Flip horizontally so
% the 3D capture geometrically aligns with the GT bbox overlay.
img = img(:, end:-1:1, :);

% Weather post-processing — same as render_camera_image
img = apply_weather(img, fog, illum, noise);
end


% =========================================================================
% Build the hidden 3D scene once: terrain + intruders + scenery + sky
% =========================================================================
function eo3d = setup_scene(W, H)
% Hidden offscreen figure sized to camera resolution so getframe returns
% an image of roughly W x H. We force the figure invisible and put it
% off-screen so it doesn't flicker on the user's display.
eo3d.fig = figure("Visible", "off", "Position", [-5000 -5000 W H], ...
    "Color", [0.55 0.70 0.85], "MenuBar", "none", "ToolBar", "none", ...
    "Renderer", "opengl");
eo3d.ax = axes("Parent", eo3d.fig, "Units", "normalized", ...
    "Position", [0 0 1 1]);
hold(eo3d.ax, "on");
axis(eo3d.ax, "off");

% Sky-tinted background by clearing to a gradient via patch
set(eo3d.ax, "Color", "none");

% --- Terrain (mountain-style coloring + strong shading + far-field peaks) ---
eo3d.terrain = [];
try
    Xg = evalin("base", "TERRAIN_X");
    Yg = evalin("base", "TERRAIN_Y");
    Zg = evalin("base", "TERRAIN_Z");
    % Far-field mountain boost — only away from the UAV flight corridor
    % (|y| > 25 m) so intruders/UAV near the path stay on the original
    % terrain elevation and project to the same GT bbox as the 2D model.
    dist_from_path = max(0, abs(Yg) - 25);   % 0 within corridor, grows outside
    far_x = max(0, abs(Xg) - 70);            % extra peaks beyond x = ±70
    mtn_boost = 0.45 * dist_from_path.^1.25 + 0.30 * far_x.^1.10 ...
              + 1.8 * sin(0.07 * Xg) .* cos(0.05 * Yg) .* (dist_from_path > 10);
    Zr = Zg + mtn_boost;

    eo3d.terrain = surf(eo3d.ax, Xg, Yg, Zr, Zr, ...   % CData = Zr → color by height
        "EdgeColor", "none", ...
        "FaceLighting", "gouraud", ...
        "AmbientStrength", 0.35, ...
        "DiffuseStrength", 0.85, ...
        "SpecularStrength", 0.05);
    % Mountain colormap: grass(low) → tan(mid) → rock(high) → snow(peak)
    mtn_cmap = [
        0.18 0.40 0.16;    % deep grass valley
        0.28 0.50 0.20;
        0.38 0.55 0.25;    % alpine meadow
        0.52 0.52 0.30;
        0.62 0.50 0.32;    % tan / dry slope
        0.58 0.45 0.35;
        0.52 0.42 0.36;    % rocky brown
        0.50 0.42 0.40;
        0.55 0.50 0.48;    % grey rocks
        0.70 0.68 0.65;
        0.85 0.85 0.88;    % near-peak rocks
        0.95 0.95 0.97;    % snow cap
    ];
    colormap(eo3d.ax, mtn_cmap);
    caxis(eo3d.ax, [0 max(Zr(:))]);
    xlim(eo3d.ax, [min(Xg(:)) max(Xg(:))]);
    ylim(eo3d.ax, [min(Yg(:)) max(Yg(:))]);
    zlim(eo3d.ax, [0 max(Zr(:)) + 80]);
catch
    xlim(eo3d.ax, [-100 100]); ylim(eo3d.ax, [-100 100]); zlim(eo3d.ax, [0 80]);
end

% --- Scenery (trees, rocks — non-targets, background clutter) ---
eo3d.scenery = [];
try
    scenery = evalin("base", "SCENERY_OBJECTS");   % Mx5: [x, y, z, r, type]
    for k = 1:size(scenery, 1)
        x = scenery(k,1); y = scenery(k,2); z = scenery(k,3);
        r = scenery(k,4); t = scenery(k,5);
        if t == 1
            % Tree: green canopy sphere + brown trunk cylinder
            h = draw_tree(eo3d.ax, [x y z], r);
        else
            % Rock: low grey blob
            h = draw_rock(eo3d.ax, [x y z], r);
        end
        eo3d.scenery = [eo3d.scenery; h];
    end
catch
end

% --- Intruders (people + vehicles — these update per frame) ---
eo3d.intruders = [];          % cell array of patch/surface handles per intruder

% Camera basics — perspective, decent FOV
set(eo3d.ax, "Projection", "perspective");
set(eo3d.ax, "DataAspectRatio", [1 1 1]);
set(eo3d.ax, "Clipping", "off");

% Light source — sun-like
eo3d.light = light(eo3d.ax, "Position", [50 50 200], "Style", "infinite");
lighting(eo3d.ax, "gouraud");
material(eo3d.ax, "dull");
end


% =========================================================================
% Replace/refresh intruder geometry in the scene (returns updated struct)
% =========================================================================
function eo3d = redraw_intruders(eo3d, obs_xyz, obs_rh)
try
    obs_class = evalin("base", "OBSTACLES_CLASS");
catch
    obs_class = ones(size(obs_xyz, 1), 1);
end

% Delete previous intruder handles and redraw.
for k = 1:numel(eo3d.intruders)
    if isgraphics(eo3d.intruders(k))
        delete(eo3d.intruders(k));
    end
end
eo3d.intruders = gobjects(0);

for k = 1:size(obs_xyz, 1)
    base = obs_xyz(k, :);
    r = obs_rh(k, 1);
    h = obs_rh(k, 2);
    cls = 1;
    if k <= numel(obs_class), cls = obs_class(k); end
    if cls == 2
        hh = draw_vehicle_3d(eo3d.ax, base, r, h);
    else
        hh = draw_person_3d(eo3d.ax, base, r, h);
    end
    eo3d.intruders = [eo3d.intruders; hh];
end

% Remember positions so next frame can skip redraw when they're unchanged.
eo3d.last_obs_xyz = obs_xyz;
eo3d.last_obs_rh  = obs_rh;
end


% =========================================================================
% Configure camera = UAV 1st-person view (pitched down by pitch_deg)
% =========================================================================
function configure_camera(ax, uav, pitch_deg, fy, H)
pitch = pitch_deg * pi/180;
% UAV looks along +x in world frame, pitched down by `pitch` from horizontal.
look_dir = [cos(pitch), 0, -sin(pitch)];
target = uav(:).' + 20 * look_dir;
campos(ax, uav(:).');
camtarget(ax, target);
camup(ax, [0 0 1]);
% Vertical field of view from focal length: 2 * atan(H / (2*fy)) [rad]
vfov_deg = 2 * atand(H / (2 * fy));
camva(ax, vfov_deg);
end


% =========================================================================
% Person 3D model: humanoid figure
%   legs (2 cylinders) + torso (tapered cylinder) + arms (2 cylinders) +
%   head (sphere) + hair cap. Proportions chosen so the silhouette reads
%   as "human" even when viewed from above at 60° pitch.
% =========================================================================
function handles = draw_person_3d(ax, base, r, h)
shirt_color = [0.20 0.50 0.95];   % blue shirt (matches old 2D person color)
pants_color = [0.15 0.20 0.50];   % darker blue pants
skin_color  = [0.92 0.78 0.62];   % light skin
hair_color  = [0.28 0.18 0.12];   % dark brown

leg_h   = h * 0.48;
torso_h = h * 0.32;
head_h  = h * 0.20;
leg_top   = base(3) + leg_h;
torso_top = leg_top + torso_h;

handles = gobjects(0);

% --- Legs: two parallel cylinders ---
leg_r   = r * 0.20;
leg_sep = r * 0.30;
[xc, yc, zc] = cylinder(leg_r, 10);
zc = zc * leg_h;
for sgn = [-1, +1]
    h_leg = surf(ax, xc + base(1), yc + base(2) + sgn*leg_sep, zc + base(3), ...
        "EdgeColor", "none", "FaceColor", pants_color);
    handles = [handles; h_leg]; %#ok<AGROW>
end

% --- Torso: tapered cylinder (wide shoulders → narrow waist top→bottom flip)
torso_r_bot = r * 0.32;
torso_r_top = r * 0.45;       % shoulders wider than waist for clear silhouette
[xt, yt, zt] = cylinder([torso_r_bot, torso_r_top], 14);
zt = zt * torso_h + leg_top;
h_torso = surf(ax, xt + base(1), yt + base(2), zt, ...
    "EdgeColor", "none", "FaceColor", shirt_color);
handles = [handles; h_torso];

% --- Arms: 2 cylinders along sides of torso ---
arm_r = r * 0.12;
arm_h = torso_h * 0.95;
[xa, ya, za] = cylinder(arm_r, 8);
za = za * arm_h + leg_top + torso_h * 0.05;
for sgn = [-1, +1]
    h_arm = surf(ax, xa + base(1), ya + base(2) + sgn * torso_r_top, za, ...
        "EdgeColor", "none", "FaceColor", shirt_color * 0.85);
    handles = [handles; h_arm];
end

% --- Head: sphere ---
head_r = r * 0.30;
[sx, sy, sz] = sphere(12);
h_head = surf(ax, sx*head_r + base(1), sy*head_r + base(2), ...
    sz*head_r + torso_top + head_r, ...
    "EdgeColor", "none", "FaceColor", skin_color);
handles = [handles; h_head];

% --- Hair: smaller sphere offset to top of head, only top half visible ---
hr = head_r * 1.05;
[hx, hy, hz] = sphere(10);
% Top-hemisphere by clamping bottom: set CData / use alpha — simpler:
% Draw a slightly raised dark sphere; the body sphere covers the bottom.
h_hair = surf(ax, hx*hr + base(1), hy*hr + base(2), ...
    hz*hr*0.85 + torso_top + head_r + head_r*0.25, ...
    "EdgeColor", "none", "FaceColor", hair_color);
handles = [handles; h_hair];
end


% =========================================================================
% Vehicle 3D model: SUV silhouette
%   lower chassis box + upper cabin box + 4 black wheel cylinders + windshield.
%   Oriented along +x (assumes UAV flies +x; intruder vehicles parked or
%   moving along corridor). Patch faces with vertex colors → clear body /
%   window contrast.
% =========================================================================
function handles = draw_vehicle_3d(ax, base, r, h)
body_color   = [0.95 0.55 0.10];   % orange body
roof_color   = body_color * 0.75;
window_color = [0.18 0.25 0.40];   % dark blue-grey glass
wheel_color  = [0.10 0.10 0.10];   % black tire
hub_color    = [0.55 0.55 0.55];   % grey hub

% Approximate SUV dimensions (longer-than-wide)
L = max(3.6, r * 2.3);    % length (along x)
W = max(2.0, r * 1.25);   % width (along y)
chassis_h = h * 0.55;
cabin_h   = h * 0.45;

handles = gobjects(0);

% --- Chassis (lower box) ---
hC = draw_box(ax, base(1), base(2), base(3), L, W, chassis_h, body_color);
handles = [handles; hC];

% --- Cabin (upper box, narrower & shorter, slightly toward back) ---
cabin_L = L * 0.65;
cabin_W = W * 0.92;
cabin_x = base(1) - L * 0.05;          % cabin slightly toward back (visual hint)
cabin_z0 = base(3) + chassis_h;
hCab = draw_box(ax, cabin_x, base(2), cabin_z0, cabin_L, cabin_W, cabin_h, ...
    window_color);
handles = [handles; hCab];

% --- Roof (thin coloured plate on top of cabin so cabin doesn't look like
%     pure glass when viewed from above) ---
roof_z = cabin_z0 + cabin_h * 0.85;
hRoof = draw_box(ax, cabin_x, base(2), roof_z, cabin_L * 0.95, cabin_W * 0.95, ...
    cabin_h * 0.18, roof_color);
handles = [handles; hRoof];

% --- 4 wheels: cylinders rotated to lie horizontally with axis along y ---
wheel_r = 0.42;
wheel_w = 0.30;
wx_off = L * 0.32;                     % front/rear axle offset
wy_off = W * 0.50 + wheel_w * 0.10;    % outside body
[cx, cy, cz] = cylinder(wheel_r, 14);
% Rotate so axis lies along y: cylinder length cz → y direction
xL = cx;
yL = (cz - 0.5) * wheel_w;             % centred wheel width on y
zL = cy + wheel_r;                     % wheel touches ground at base(3)
for sx = [-wx_off, +wx_off]
    for sy = [-wy_off, +wy_off]
        h_wheel = surf(ax, xL + base(1) + sx, yL + base(2) + sy, zL + base(3), ...
            "EdgeColor", "none", "FaceColor", wheel_color);
        handles = [handles; h_wheel];
        % Small hub disc (centre of wheel)
        [hubx, huby, hubz] = cylinder(wheel_r * 0.35, 12);
        hubx2 = hubx;
        huby2 = (hubz - 0.5) * (wheel_w * 1.05);
        hubz2 = huby + wheel_r;
        h_hub = surf(ax, hubx2 + base(1) + sx, huby2 + base(2) + sy, hubz2 + base(3), ...
            "EdgeColor", "none", "FaceColor", hub_color);
        handles = [handles; h_hub];
    end
end
end


% =========================================================================
% Axis-aligned box helper using patch — accepts (cx, cy, z0, Lx, Ly, Lz, color)
% =========================================================================
function h = draw_box(ax, cx, cy, z0, Lx, Ly, Lz, color)
hx = Lx / 2; hy = Ly / 2;
verts = [
    cx - hx, cy - hy, z0;
    cx + hx, cy - hy, z0;
    cx + hx, cy + hy, z0;
    cx - hx, cy + hy, z0;
    cx - hx, cy - hy, z0 + Lz;
    cx + hx, cy - hy, z0 + Lz;
    cx + hx, cy + hy, z0 + Lz;
    cx - hx, cy + hy, z0 + Lz;
];
faces = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
h = patch("Parent", ax, "Vertices", verts, "Faces", faces, ...
    "FaceColor", color, "EdgeColor", "none", ...
    "FaceLighting", "gouraud", "AmbientStrength", 0.45);
end


% =========================================================================
% Tree / rock helpers (scenery only — no per-frame update needed)
% =========================================================================
function h = draw_tree(ax, base, r)
% Distinct tree silhouette: tall thick brown trunk + bushy green canopy stack.
% Multi-layer canopy makes it readable as "tree" even at distance.
trunk_h = r * 2.2;                      % taller trunk so canopy clears ground
trunk_r = max(0.20, r * 0.28);
[xc, yc, zc] = cylinder(trunk_r, 12);
zc = zc * trunk_h;
xc = xc + base(1);  yc = yc + base(2);  zc = zc + base(3);
t = surf(ax, xc, yc, zc, "EdgeColor", "none", "FaceColor", [0.42 0.27 0.16]);

% Lower canopy ball
[sx, sy, sz] = sphere(10);
lc = surf(ax, sx*r*1.05 + base(1), sy*r*1.05 + base(2), sz*r*0.9 + base(3) + trunk_h*0.85, ...
    "EdgeColor", "none", "FaceColor", [0.15 0.42 0.18]);
% Mid canopy ball (slightly higher and offset)
mc = surf(ax, sx*r*0.85 + base(1), sy*r*0.85 + base(2), sz*r*0.8 + base(3) + trunk_h*1.15, ...
    "EdgeColor", "none", "FaceColor", [0.18 0.50 0.22]);
% Top tip (smaller, brighter green) — gives clear pine-tree silhouette
tc = surf(ax, sx*r*0.55 + base(1), sy*r*0.55 + base(2), sz*r*0.6 + base(3) + trunk_h*1.45, ...
    "EdgeColor", "none", "FaceColor", [0.22 0.58 0.25]);
h = [t; lc; mc; tc];
end

function h = draw_rock(ax, base, r)
% Irregular rocky look: cluster of 2-3 grey spheres with mottled colors
% so it doesn't read as a perfect ball.
[sx, sy, sz] = sphere(10);
% Main rock body
b1 = surf(ax, sx*r + base(1), sy*r + base(2), sz*r*0.55 + base(3) + r*0.30, ...
    "EdgeColor", "none", "FaceColor", [0.55 0.50 0.45]);
% Adjacent smaller bump (offset for irregularity)
b2 = surf(ax, sx*r*0.65 + base(1) + r*0.45, sy*r*0.65 + base(2) - r*0.30, ...
    sz*r*0.45 + base(3) + r*0.20, ...
    "EdgeColor", "none", "FaceColor", [0.45 0.40 0.36]);
% Small chip (darker — moss/shadow shading)
b3 = surf(ax, sx*r*0.35 + base(1) - r*0.40, sy*r*0.35 + base(2) + r*0.20, ...
    sz*r*0.30 + base(3) + r*0.10, ...
    "EdgeColor", "none", "FaceColor", [0.38 0.36 0.32]);
h = [b1; b2; b3];
end


% =========================================================================
% Weather post-processing — identical to render_camera_image's apply_weather
% =========================================================================
function img = apply_weather(img, fog, illum, noise)
fog_norm   = max(0, min(1, fog / 100));
low_light  = max(0, (3000  - illum) / 3000);
high_light = max(0, (illum - 12000) / 12000);

% Fog: blend toward grey-white
fog_color = reshape([0.85, 0.85, 0.88], 1, 1, 3);
img = (1 - 0.85 * fog_norm) .* img + (0.85 * fog_norm) .* fog_color;

% Illumination
if low_light > 0
    img = img * max(0.10, 1 - 0.85 * low_light);
end
if high_light > 0
    img = (1 - 0.4 * high_light) .* img + (0.4 * high_light) .* 1.0;
end

% Noise (deterministic pattern)
if noise > 0
    [Hh, Ww, ~] = size(img);
    [UU, VV] = meshgrid(1:Ww, 1:Hh);
    pattern = sin(0.83 * UU + 1.7 * VV) .* cos(2.1 * UU - 0.6 * VV);
    speckle = noise * 0.40 * pattern;
    img = img + repmat(speckle, [1, 1, 3]);
end
img = max(0, min(1, img));
end
