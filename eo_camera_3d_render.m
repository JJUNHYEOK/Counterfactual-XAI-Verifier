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

% --- Terrain ---
try
    Xg = evalin("base", "TERRAIN_X");
    Yg = evalin("base", "TERRAIN_Y");
    Zg = evalin("base", "TERRAIN_Z");
    eo3d.terrain = surf(eo3d.ax, Xg, Yg, Zg, ...
        "EdgeColor", "none", ...
        "FaceColor", [0.42 0.50 0.30], ...
        "FaceLighting", "gouraud", ...
        "AmbientStrength", 0.55, ...
        "DiffuseStrength", 0.65);
catch
    eo3d.terrain = [];
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
% Person 3D model: body cylinder + head sphere (similar to mountain_visualizer)
% =========================================================================
function handles = draw_person_3d(ax, base, r, h)
color = [0.20 0.50 0.95];
% Body cylinder
[xc, yc, zc] = cylinder(r, 12);
zc = zc * h;  xc = xc + base(1);  yc = yc + base(2);  zc = zc + base(3);
b = surf(ax, xc, yc, zc, "EdgeColor", "none", "FaceColor", color);
% Head sphere
[sx, sy, sz] = sphere(10);
sx = sx * (r*1.1) + base(1);
sy = sy * (r*1.1) + base(2);
sz = sz * (r*1.0) + base(3) + h;
hd = surf(ax, sx, sy, sz, "EdgeColor", "none", "FaceColor", color*0.85);
handles = [b; hd];
end


% =========================================================================
% Vehicle 3D model: wider squat cylinder + dome on top (simplified)
% =========================================================================
function handles = draw_vehicle_3d(ax, base, r, h)
color = [0.95 0.55 0.10];
% Body cylinder (wider, shorter)
[xc, yc, zc] = cylinder(r, 16);
zc = zc * (h * 0.55);  xc = xc + base(1);  yc = yc + base(2);  zc = zc + base(3);
b = surf(ax, xc, yc, zc, "EdgeColor", [0.30 0.30 0.30], "FaceColor", color);
% Roof / cabin — smaller cylinder stacked on top
[xc2, yc2, zc2] = cylinder(r * 0.55, 16);
zc2 = zc2 * (h * 0.45) + base(3) + h * 0.55;
xc2 = xc2 + base(1);  yc2 = yc2 + base(2);
c = surf(ax, xc2, yc2, zc2, "EdgeColor", "none", "FaceColor", color*0.75);
handles = [b; c];
end


% =========================================================================
% Tree / rock helpers (scenery only — no per-frame update needed)
% =========================================================================
function h = draw_tree(ax, base, r)
% Trunk
[xc, yc, zc] = cylinder(r*0.18, 10);
zc = zc * (r*1.4);  xc = xc + base(1);  yc = yc + base(2);  zc = zc + base(3);
t = surf(ax, xc, yc, zc, "EdgeColor", "none", "FaceColor", [0.45 0.30 0.20]);
% Canopy
[sx, sy, sz] = sphere(8);
sx = sx * r + base(1);
sy = sy * r + base(2);
sz = sz * (r * 0.8) + base(3) + r * 1.2;
c = surf(ax, sx, sy, sz, "EdgeColor", "none", "FaceColor", [0.12 0.40 0.18]);
h = [t; c];
end

function h = draw_rock(ax, base, r)
[sx, sy, sz] = sphere(8);
sx = sx * r + base(1);
sy = sy * r + base(2);
sz = sz * (r * 0.5) + base(3) + r * 0.3;
h = surf(ax, sx, sy, sz, "EdgeColor", "none", ...
    "FaceColor", [0.50 0.45 0.40]);
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
