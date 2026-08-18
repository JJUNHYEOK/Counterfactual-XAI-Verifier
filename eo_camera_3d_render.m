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

% Honor dashboard's "force intruder redraw" flag (set on each new
% boundary-search session) so persistent handles from the previous
% session can't ghost-render alongside the newly-walking persons.
force_redraw = false;
try
    if getappdata(0, 'EO_FORCE_INTRUDER_REDRAW')
        force_redraw = true;
        setappdata(0, 'EO_FORCE_INTRUDER_REDRAW', false);
    end
catch
end

% Refresh intruders when forced, on first frame, on layout change, or
% when position delta exceeds 0.3 m (so walking persons get visible
% motion without paying the redraw cost every frame).
needs_redraw = false;
if force_redraw || ...
        ~isfield(eo3d, "last_obs_xyz") || ...
        size(eo3d.last_obs_xyz, 1) ~= size(obs_xyz, 1) || ...
        ~isequal(eo3d.last_obs_rh, obs_rh)
    needs_redraw = true;
elseif max(abs(eo3d.last_obs_xyz(:) - obs_xyz(:))) > 0.3
    needs_redraw = true;
end
if needs_redraw
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
eo3d.ax = axes("Parent", eo3d.fig, "Units", "pixels", ...
    "Position", [1 1 W H]);
% Pixel-exact viewport (W×H) with zero inset so the captured frame matches
% the pinhole projection 1:1 — `LooseInset` zeroed prevents MATLAB from
% reserving any decoration margin even after `axis off`.
set(eo3d.ax, "LooseInset", [0 0 0 0]);
set(eo3d.ax, "ActivePositionProperty", "position");
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

    % Flatten terrain in a smooth disc around each intruder so vehicles /
    % people sit cleanly on the ground instead of being buried by the
    % surrounding mountain undulations. Inside R1 the terrain is hard-set
    % to slightly below the intruder base z; from R1 to R2 it smoothstep-
    % blends back to the natural undulating terrain.
    try
        obs_xyz_local = evalin("base", "OBSTACLES_XYZ");
        obs_rh_local  = evalin("base", "OBSTACLES_RH");
        for k = 1:size(obs_xyz_local, 1)
            cxi = obs_xyz_local(k, 1);
            cyi = obs_xyz_local(k, 2);
            czi = obs_xyz_local(k, 3);
            r_obj = obs_rh_local(k, 1);
            R1 = max(2.0, r_obj * 2.0);    % full flatten radius
            R2 = R1 + 3.0;                 % blend out to here
            d = sqrt((Xg - cxi).^2 + (Yg - cyi).^2);
            flat_z = czi - 0.10;           % 10 cm below intruder base
            in_flat  = d < R1;
            in_blend = (d >= R1) & (d < R2);
            Zr(in_flat) = flat_z;
            if any(in_blend(:))
                t = (d(in_blend) - R1) / (R2 - R1);
                s = 3*t.^2 - 2*t.^3;        % smoothstep
                Zr(in_blend) = (1 - s) .* flat_z + s .* Zr(in_blend);
            end
        end
    catch
    end

    % Upsample grid 2x for smoother shading (less polygon-edge banding).
    Xf = linspace(min(Xg(:)), max(Xg(:)), size(Xg, 2)*2 - 1);
    Yf = linspace(min(Yg(:)), max(Yg(:)), size(Yg, 1)*2 - 1);
    [Xff, Yff] = meshgrid(Xf, Yf);
    Zff = interp2(Xg, Yg, Zr, Xff, Yff, "spline");
    eo3d.terrain = surf(eo3d.ax, Xff, Yff, Zff, Zff, ...   % CData = Zff → color by height
        "EdgeColor", "none", ...
        "FaceColor", "interp", ...   % smooth color across faces (no blocky bands)
        "FaceLighting", "gouraud", ...
        "AmbientStrength", 0.40, ...
        "DiffuseStrength", 0.80, ...
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

% --- Scenery (trees, rocks, bushes, logs, snow patches — non-targets) ---
% Variety within each type is chosen via a deterministic position hash, so
% the same scenery item always renders as the same sub-variant across runs.
eo3d.scenery = [];
try
    scenery = evalin("base", "SCENERY_OBJECTS");   % Mx5: [x, y, z, r, type]
    for k = 1:size(scenery, 1)
        x = scenery(k,1); y = scenery(k,2); z = scenery(k,3);
        r = scenery(k,4); t = scenery(k,5);
        % Deterministic [0,1) hash from position
        seed = mod(abs(x * 7.31 + y * 5.13 + r * 11.7), 1.0);
        if t == 1
            % Vegetation: pine tree (60%) or bush (40%)
            if seed < 0.60
                h = draw_tree(eo3d.ax, [x y z], r);
            else
                h = draw_bush(eo3d.ax, [x y z], r);
            end
        else
            % Ground feature: rock (65%) or fallen log (35%)
            if seed < 0.65
                h = draw_rock(eo3d.ax, [x y z], r);
            else
                h = draw_log(eo3d.ax, [x y z], r);
            end
        end
        eo3d.scenery = [eo3d.scenery; h];
    end
catch
end

% --- Intruders (people + vehicles — these update per frame) ---
eo3d.intruders = gobjects(0); % graphics-handle column for redraw_intruders

% Camera basics — perspective, decent FOV
set(eo3d.ax, "Projection", "perspective");
set(eo3d.ax, "DataAspectRatio", [1 1 1]);
% Lock the plot box aspect to W:H:1 so the camva-determined vertical FOV
% maps cleanly to image rows without any axes stretching (a mismatched
% PlotBoxAspectRatio is the root cause of the GT bbox vs object offset).
set(eo3d.ax, "PlotBoxAspectRatioMode", "manual");
set(eo3d.ax, "PlotBoxAspectRatio", [W H 1]);
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

% Delete previous intruder handles (tracked array + any tagged axes
% children that may have leaked across sessions).
for k = 1:numel(eo3d.intruders)
    if isgraphics(eo3d.intruders(k))
        delete(eo3d.intruders(k));
    end
end
delete(findobj(eo3d.ax, "Tag", "eo_intruder"));   % catch untracked stragglers
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
    for ih = 1:numel(hh)
        try, set(hh(ih), "Tag", "eo_intruder"); catch, end
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
% No pitch bias — MATLAB perspective and pinhole projections cannot be
% exactly matched (MATLAB uses data-extent based frustum, not focal
% length). Residual offset is accepted; image_detector now samples a
% wider interior to remain robust to loose bbox alignment.
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
% Small z lift to keep feet clearly above the (flattened) terrain.
base(3) = base(3) + 0.05;
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
% Small z lift to keep the chassis bottom clearly above the (flattened)
% terrain and avoid z-fighting / visual burial on slight slopes.
base(3) = base(3) + 0.05;
body_color   = [0.95 0.55 0.10];   % orange body
roof_color   = body_color * 0.75;
window_color = [0.18 0.25 0.40];   % dark blue-grey glass
wheel_color  = [0.10 0.10 0.10];   % black tire
hub_color    = [0.55 0.55 0.55];   % grey hub

% Visual SUV dimensions matched to the F_detector box-projected bbox:
% L = 2r (length along x, exact match with bbox vertical extent),
% W < 2r (narrower for SUV silhouette, fits inside bbox width).
L = 2 * r;                          % full bbox length match
W = r * 1.30;                       % narrower (typical SUV W/L ≈ 0.65)
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
% Angular rocky boulder — low-poly polyhedron via patch (sharp edges read
% clearly as "rock" from camera distance, unlike a smooth sphere).
% Build a 10-vertex random polyhedron seeded by position for determinism.
seed1 = mod(abs(base(1)*0.731 + base(2)*0.519), 1.0);
seed2 = mod(abs(base(1)*1.137 + base(2)*0.913), 1.0);

% 7 top vertices on an irregular dome + 1 base ring of 6
n_top = 6;
theta_top = linspace(0, 2*pi, n_top + 1); theta_top(end) = [];
% slight angular jitter so faces aren't symmetric
theta_top = theta_top + 0.25 * sin(seed1 * 6.28 + (1:n_top));
% radius jitter per vertex
rj = r * (0.75 + 0.45 * (mod(seed2 * 100 + (1:n_top), 1)));
top_x = rj .* cos(theta_top);
top_y = rj .* sin(theta_top);
top_z = r * (0.55 + 0.25 * sin(seed1 * 4 + (1:n_top)));   % bumpy top

n_bot = 6;
theta_bot = linspace(0, 2*pi, n_bot + 1); theta_bot(end) = [];
rb = r * (0.95 + 0.10 * cos(seed2 * 5 + (1:n_bot)));
bot_x = rb .* cos(theta_bot);
bot_y = rb .* sin(theta_bot);
bot_z = zeros(1, n_bot) + 0.05 * r;

apex_z = r * 0.85;     % single top point

verts = [top_x' top_y' top_z';                  % 1..6 (top ring)
         bot_x' bot_y' bot_z';                  % 7..12 (bottom ring)
         0 0 apex_z];                           % 13 (apex)
verts = verts + base;

% Faces — top ring to apex (triangle fan), then side quads ring-to-ring
faces_top = [(1:n_top)'  circshift((1:n_top)', -1)  repmat(13, n_top, 1)];
faces_side = zeros(n_top, 4);
for k = 1:n_top
    k_next = mod(k, n_top) + 1;
    faces_side(k,:) = [k, k_next, k_next + n_top, k + n_top];
end

% Mottled rocky colors
c_top  = [0.60 0.56 0.50];
c_side = [0.48 0.44 0.40];

h1 = patch("Parent", ax, "Vertices", verts, "Faces", faces_top, ...
    "FaceColor", c_top, "EdgeColor", "none", ...
    "FaceLighting", "gouraud", "AmbientStrength", 0.45);
h2 = patch("Parent", ax, "Vertices", verts, "Faces", faces_side, ...
    "FaceColor", c_side, "EdgeColor", "none", ...
    "FaceLighting", "gouraud", "AmbientStrength", 0.40);

% A smaller satellite chip beside the main boulder for irregularity
[sx, sy, sz] = sphere(8);
chip_r = r * 0.40;
ox = 0.55 * r * cos(seed1 * 6.28);
oy = 0.55 * r * sin(seed2 * 6.28);
h3 = surf(ax, sx*chip_r + base(1) + ox, sy*chip_r + base(2) + oy, ...
    sz*chip_r*0.50 + base(3) + chip_r*0.25, ...
    "EdgeColor", "none", "FaceColor", [0.40 0.36 0.32]);

h = [h1; h2; h3];
end


% =========================================================================
% Bush — dense low shrub: stem + cluster of green puffs with brown twig hints
% =========================================================================
function h = draw_bush(ax, base, r)
[sx, sy, sz] = sphere(10);
green_main  = [0.18 0.40 0.18];
green_light = [0.30 0.55 0.28];
green_dark  = [0.13 0.32 0.16];
twig_color  = [0.30 0.20 0.12];

handles = gobjects(0);

% Hidden brown stem at base (only ~30% protrudes from foliage)
stem_h = r * 0.55;
stem_r = r * 0.10;
[cx, cy, cz] = cylinder(stem_r, 8);
cz = cz * stem_h;
h_stem = surf(ax, cx + base(1), cy + base(2), cz + base(3), ...
    "EdgeColor","none", "FaceColor", twig_color);
handles = [handles; h_stem];

% Central dense puff (largest)
b1 = surf(ax, sx*r + base(1), sy*r + base(2), ...
    sz*r*0.65 + base(3) + r*0.45, ...
    "EdgeColor","none", "FaceColor", green_main);
% 4 surrounding puffs at slightly varying heights — leafy clustering
seed = mod(abs(base(1)*0.7 + base(2)*0.5), 1.0);
puffs = [
    +0.55, +0.10, 0.50, 0.35,  1;   % dx_frac, dy_frac, size_frac, h_offset_frac, color_idx
    -0.35, -0.45, 0.45, 0.40,  2;
    +0.10, -0.50, 0.50, 0.30,  1;
    -0.45, +0.40, 0.40, 0.45,  2;
];
for k = 1:size(puffs, 1)
    dx = puffs(k,1) * r * (0.9 + 0.2*sin(seed*5 + k));
    dy = puffs(k,2) * r * (0.9 + 0.2*cos(seed*5 + k));
    sr = puffs(k,3) * r;
    sh = puffs(k,4) * r;
    cidx = puffs(k,5);
    if cidx == 1, col = green_light; else, col = green_dark; end
    bb = surf(ax, sx*sr + base(1) + dx, sy*sr + base(2) + dy, ...
        sz*sr*0.55 + base(3) + r*0.45 + sh, ...
        "EdgeColor","none", "FaceColor", col);
    handles = [handles; bb];
end
handles = [handles; b1];
h = handles;
end


% =========================================================================
% Fallen log — horizontal brown cylinder + bark texture stripe + end grain
% =========================================================================
function h = draw_log(ax, base, r)
log_r = r * 0.40;
log_L = r * 2.8;
seed = mod(abs(base(1)*0.7 + base(2)*0.5), 1.0);
yaw  = (seed - 0.5) * 0.7;       % slight rotation in xy so logs don't all align with y-axis

% Cylinder body — generate axis along z, then re-axis to lie horizontally
[cx, cy, cz] = cylinder(log_r, 16);
xL_local = cx;
yL_local = (cz - 0.5) * log_L;
zL_local = cy + log_r;            % cylinder centre at z = log_r (touches ground)

% Apply yaw rotation about z-axis
cs = cos(yaw); sn = sin(yaw);
xL = cs * xL_local - sn * yL_local + base(1);
yL = sn * xL_local + cs * yL_local + base(2);
zL = zL_local + base(3);

bark_color = [0.40 0.26 0.16];
b = surf(ax, xL, yL, zL, "EdgeColor", "none", "FaceColor", bark_color);

% End caps — concentric brown rings to suggest end-grain
theta = linspace(0, 2*pi, 24);
end_color_outer = [0.32 0.20 0.12];
end_color_inner = [0.55 0.40 0.28];
e_handles = gobjects(0);
for end_sign = [-1 +1]
    cap_y_local = end_sign * 0.5 * log_L;
    % Outer disc
    cap_x_local = log_r * cos(theta);
    cap_z_local = log_r * sin(theta) + log_r;
    cap_x = cs * cap_x_local - sn * cap_y_local + base(1);
    cap_y = sn * cap_x_local + cs * cap_y_local + base(2);
    cap_z = cap_z_local + base(3);
    e = patch("Parent", ax, "XData", cap_x, "YData", cap_y, "ZData", cap_z, ...
        "FaceColor", end_color_outer, "EdgeColor", "none");
    e_handles = [e_handles; e];
    % Inner disc (smaller, lighter — heartwood)
    cap_x_local = log_r * 0.45 * cos(theta);
    cap_z_local = log_r * 0.45 * sin(theta) + log_r;
    cap_x = cs * cap_x_local - sn * cap_y_local + base(1);
    cap_y = sn * cap_x_local + cs * cap_y_local + base(2);
    cap_z = cap_z_local + base(3);
    e2 = patch("Parent", ax, "XData", cap_x, "YData", cap_y, "ZData", cap_z, ...
        "FaceColor", end_color_inner, "EdgeColor", "none");
    e_handles = [e_handles; e2];
end

% Bark texture: a darker stripe along the log length
stripe_color = [0.28 0.18 0.10];
sx_stripe = log_r * cos(theta(1:end-1)) * 0.05 + log_r * 0.95;
% This is getting complex; just draw a thin dark line via a small thin cylinder
[tx, ty, tz] = cylinder(log_r * 0.06, 4);
tx_l = tx + log_r * 0.85;        % offset to side of log so it's visible
ty_l = (tz - 0.5) * log_L * 0.95;
tz_l = ty + log_r;
tx2 = cs * tx_l - sn * ty_l + base(1);
ty2 = sn * tx_l + cs * ty_l + base(2);
tz2 = tz_l + base(3);
e3 = surf(ax, tx2, ty2, tz2, "EdgeColor", "none", "FaceColor", stripe_color);

h = [b; e_handles; e3];
end


% =========================================================================
% Snow patch — low flat white ellipsoid sitting on the ground
% =========================================================================
function h = draw_snowpatch(ax, base, r)
[sx, sy, sz] = sphere(12);
% Wide, flat dome
b1 = surf(ax, sx*r*1.10 + base(1), sy*r*0.85 + base(2), ...
    sz*r*0.18 + base(3) + r*0.05, ...
    "EdgeColor", "none", "FaceColor", [0.95 0.96 0.98], ...
    "AmbientStrength", 0.65, "DiffuseStrength", 0.40);
% Smaller adjacent patch for irregularity
b2 = surf(ax, sx*r*0.55 + base(1) + r*0.50, sy*r*0.40 + base(2) - r*0.30, ...
    sz*r*0.12 + base(3) + r*0.03, ...
    "EdgeColor", "none", "FaceColor", [0.92 0.93 0.95]);
h = [b1; b2];
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
