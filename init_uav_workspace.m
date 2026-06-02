function init_uav_workspace(fog, illum, noise)
% init_uav_workspace — Initialise ALL base-workspace variables required by
%   mountain_uav_model.slx's Scenario_Params Constant blocks.
%
%   Called by matlab_bridge.py before every sim() call so that ALL nine
%   Constant blocks (C_FOG, C_ILLUM, C_NOISE, C_OBS_XYZ, C_OBS_RH,
%   C_UAV_X0, C_UAV_V, C_CAM, C_IMG) have valid workspace variables.
%
%   Usage (from Python subprocess script):
%     init_uav_workspace(fog_density_percent, illumination_lux, camera_noise_level)
%     init_uav_workspace(30, 4000, 0.1)
%
%   Defaults match the seed scenario (scenario_iter_001.json).

if nargin < 1 || isempty(fog),   fog   = 30.0;   end
if nargin < 2 || isempty(illum), illum = 4000.0; end
if nargin < 3 || isempty(noise), noise = 0.1;    end

% ── Environment (changes every iteration) ────────────────────────────────
assignin('base', 'FOG_DENSITY_PERCENT', double(fog));
assignin('base', 'ILLUMINATION_LUX',    double(illum));
assignin('base', 'CAMERA_NOISE_LEVEL',  double(noise));

% ── Terrain (fixed geometry, computed once) ───────────────────────────────
[Xg, Yg, Zg] = terrain_grid_();
assignin('base', 'TERRAIN_X', Xg);
assignin('base', 'TERRAIN_Y', Yg);
assignin('base', 'TERRAIN_Z', Zg);

% ── Intruder targets: 3 people + 2 vehicles on the mountainside ──────────
% Mission: detect unauthorized intruders (people / vehicles) in a border
% mountain area. Trees / shrubs are NOT detection targets — only intruders.
%
% Object dimensions chosen so bboxes are ~10-25 px wide at the surveillance
% altitude (small enough to be a "small-object" challenge, big enough that
% IoU>=0.5 matching is feasible in clean conditions).
%   person  : r=0.50,  h=1.80
%   vehicle : r=1.60,  h=1.80   (≈ 3.2 m diameter cylinder ~ small SUV)

intruderXY = [
      4,  2;     % person 1  (UAV starts at -22 → 26 m ahead, ~2 s empty start)
     16, -1;     % person 2  (~12 m gap)
     28,  3;     % person 3  (~12 m gap)
     40, -2;     % vehicle 1 (~12 m gap)
     52,  2;     % vehicle 2 (~12 m gap) — last intruder, near end of flight
];   % span 48 m, spacing ~12 m → 1 intruder per frame (no overlap)
intruderClass = [1; 1; 1; 2; 2];        % 1=person, 2=vehicle
intruderDims  = [
    0.50, 1.80;   % person dims (r, h)
    0.50, 1.80;
    0.50, 1.80;
    1.20, 1.60;   % vehicle dims — r=1.2 so bbox tightly matches SUV L=2r=2.4m
    1.20, 1.60;
];

N = size(intruderXY, 1);
OBS_XYZ = zeros(N, 3);
OBS_RH  = zeros(N, 2);
for k = 1:N
    tx = intruderXY(k,1);  ty = intruderXY(k,2);
    tz = interp2(Xg, Yg, Zg, tx, ty, 'linear', 0);
    OBS_XYZ(k,:) = [tx, ty, tz];
    OBS_RH(k,:)  = intruderDims(k,:);
end
assignin('base', 'OBSTACLES_XYZ',   OBS_XYZ);
assignin('base', 'OBSTACLES_RH',    OBS_RH);
assignin('base', 'OBSTACLES_CLASS', intruderClass);

% ── Scenery: non-target background (trees + rocks) ───────────────────────
% These objects appear in the 3D scene and the camera image to give the
% UAV operator a realistic mountain environment, but they are NOT in
% OBSTACLES_XYZ so the detector / requirements eval ignores them.
% Deterministic generation so every iteration sees the same landscape.
SCENERY_OBJECTS = make_scenery_(Xg, Yg, Zg);
assignin('base', 'SCENERY_OBJECTS', SCENERY_OBJECTS);

% ── UAV initial state — surveillance overflight ──────────────────────────
% +15 m above the highest peak ⇒ ≈ 45 m AGL over the valley.
% Lower altitude than first attempt; gives larger projected bboxes
% (10-25 px) which are still "small-object" but actually detectable.
uavZ0 = max(Zg(:)) + 15;
assignin('base', 'UAV_X0_VEC', [-22.0, 0.0, uavZ0]);   % ~2 s empty start before first intruder enters FOV
assignin('base', 'UAV_V_VEC',  [3.0,   0.0, 0.0]);

% ── Camera intrinsics: [fx, fy, cx, cy, pitch_down_deg] ──────────────────
% pitch=60° (looking down at 60° from horizontal) keeps a clear "from above"
% surveillance feel while preserving non-degenerate vertical projection.
% pitch=90° would collapse object height to 0 px (all-broken bboxes).
assignin('base', 'CAM_INTRIN', [600, 600, 320, 180, 60]);
assignin('base', 'IMG_SIZE',   [640, 360]);

end


% =========================================================================
% Local helper: mountain terrain (must match build_mountain_uav_model.m)
% =========================================================================
function S = make_scenery_(Xg, Yg, Zg)
% Generates ~80 procedural background objects DENSELY clustered around the
% UAV flight corridor (intruder x∈[-7,9], y∈[-3,4]).
%   type 1 = tree   (radius 0.6~1.5 m, taller)
%   type 2 = rock   (radius 0.7~2.0 m, low/flat)
% Output: Mx5 array [x, y, z, radius, type]
% Avoids placing scenery within 2 m of any intruder.
intruderXY = [4,2; 16,-1; 28,3; 40,-2; 52,2];

% Scenery covers the full flight corridor with comfortable density.
X_RANGE = [-25, 60];     % spans UAV start (-22) through past last intruder (52)
Y_RANGE = [-15, 15];
M       = 130;
EXCL_R  = 2.5;

S = zeros(0, 5);
for k = 1:M
    seed  = mod(k * 12.9898 + 78.233, 1.0);
    seed2 = mod(k * 39.346 + 11.135, 1.0);
    seed3 = mod(k * 67.123 + 53.842, 1.0);
    seed4 = mod(k * 29.478 + 91.231, 1.0);

    x = X_RANGE(1) + (X_RANGE(2) - X_RANGE(1)) * seed;
    y = Y_RANGE(1) + (Y_RANGE(2) - Y_RANGE(1)) * seed2;
    z = interp2(Xg, Yg, Zg, x, y, 'linear', 0);

    % Skip if too close to any intruder
    d = min(sqrt((intruderXY(:,1)-x).^2 + (intruderXY(:,2)-y).^2));
    if d < EXCL_R, continue; end

    if seed3 < 0.7
        type = 1;                 % tree / bush (decided in renderer)
        r    = 0.6 + 0.9 * seed4;
    else
        type = 2;                 % rock / fallen log (decided in renderer)
        r    = 0.7 + 1.3 * seed4;
    end
    S(end+1, :) = [x, y, z, r, type]; %#ok<AGROW>
end
end


function [Xg, Yg, Zg] = terrain_grid_()
% 8-peak Gaussian mixture + ridges + a shallow E-W valley.
% MUST match build_mountain_uav_model.mountain_terrain_grid().
extent = 200;  step = 2.0;
xs = -extent/2 : step : extent/2;
ys = -extent/2 : step : extent/2;
[Xg, Yg] = meshgrid(xs, ys);

peaks = [
       0,    0,  30,  35;
      55,   30,  22,  28;
     -50,   25,  18,  30;
      35,  -45,  20,  25;
     -30,  -40,  15,  22;
      80,    5,  16,  22;
     -75,  -10,  14,  20;
      15,   60,  19,  26
];

Zg = zeros(size(Xg));
for k = 1:size(peaks,1)
    cx = peaks(k,1); cy = peaks(k,2); h = peaks(k,3); s = peaks(k,4);
    Zg = Zg + h * exp(-((Xg-cx).^2 + (Yg-cy).^2) / (2*s^2));
end

Zg = Zg + 0.8 * sin(0.10*Xg) .* cos(0.12*Yg);
Zg = Zg + 0.4 * sin(0.05*Xg + 0.07*Yg);

% Shallow E-W valley around y=0 where intruders move
valley = -3.0 * exp(-((Yg + 1).^2) / 50);
Zg = Zg + valley;

Zg = max(Zg, 0);
end
