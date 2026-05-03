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

% ── Tree obstacles (fixed layout on mountainside) ─────────────────────────
treeXY = [-30,  5;
          -10, -8;
           10,  3;
           30,-10;
           50,  8];
treeRadius = 1.2;
treeHeight = 7.0;

OBS_XYZ = zeros(5, 3);
OBS_RH  = zeros(5, 2);
for k = 1:5
    tx = treeXY(k,1);  ty = treeXY(k,2);
    tz = interp2(Xg, Yg, Zg, tx, ty, 'linear', 0);
    OBS_XYZ(k,:) = [tx, ty, tz];
    OBS_RH(k,:)  = [treeRadius, treeHeight];
end
assignin('base', 'OBSTACLES_XYZ', OBS_XYZ);
assignin('base', 'OBSTACLES_RH',  OBS_RH);

% ── UAV initial state (fixed trajectory) ─────────────────────────────────
uavZ0 = max(Zg(:)) + 8;          % 8 m above the highest peak
assignin('base', 'UAV_X0_VEC', [-80.0, 0.0, uavZ0]);
assignin('base', 'UAV_V_VEC',  [3.0,   0.0, 0.0]);   % m/s along +X

% ── Camera intrinsics: [fx, fy, cx, cy, pitch_down_deg] ──────────────────
assignin('base', 'CAM_INTRIN', [600, 600, 320, 180, 15]);
assignin('base', 'IMG_SIZE',   [640, 360]);

end


% =========================================================================
% Local helper: mountain terrain (must match build_mountain_uav_model.m)
% =========================================================================
function [Xg, Yg, Zg] = terrain_grid_()
extent = 200;  step = 2.0;
xs = -extent/2 : step : extent/2;
ys = -extent/2 : step : extent/2;
[Xg, Yg] = meshgrid(xs, ys);

peaks = [
       0,    0,  30,  35;
      55,   30,  22,  28;
     -50,   25,  18,  30;
      35,  -45,  20,  25;
     -30,  -40,  15,  22
];

Zg = zeros(size(Xg));
for k = 1:size(peaks,1)
    cx = peaks(k,1); cy = peaks(k,2); h = peaks(k,3); s = peaks(k,4);
    Zg = Zg + h * exp(-((Xg-cx).^2 + (Yg-cy).^2) / (2*s^2));
end
Zg = Zg + 0.6 * sin(0.10*Xg) .* cos(0.12*Yg);
Zg = max(Zg, 0);
end
