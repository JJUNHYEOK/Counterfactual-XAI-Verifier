function [det_scores, det_bboxes] = image_detector(img, gt_bboxes)
% image_detector — Pixel-feature based detector that consumes the rendered
% camera image and produces a confidence per ground-truth bounding box.
%
% Unlike the geometric Object_Detector inside the Simulink model, this
% function actually inspects the RENDERED image content. That means weather
% effects (fog blur/haze, low light, noise) on the rendered image
% naturally degrade detection without any explicit weather-to-score map.
%
% Inputs
%   img        : HxWx3 image in [0, 1]
%   gt_bboxes  : Nx4 [u, v, w, h] ground-truth bounding boxes (rows of zeros
%                are skipped).
%
% Outputs
%   det_scores : Nx1 confidence in [0, 1]
%   det_bboxes : Nx4 detected bbox (slight feature-driven jitter on top of GT)
%
% Features per box (vegetation-cued, scaled to [0,1] then mixed):
%   1) Vegetation index   : (G - 0.5*(R+B)) / max(G, 0.05)  high for greens
%   2) Edge density       : Sobel magnitude mean inside bbox
%   3) Vertical structure : column-wise std vs row-wise std (trees are taller than wide)
%   4) Contrast           : luminance std normalized
%
% Score = sigmoid(weighted_sum - bias). Tuned so that a clean rendered tree
% scores ~0.85 and a fogged-out tree scores ~0.2.

[H, W, ~] = size(img);
N = size(gt_bboxes, 1);
det_scores = zeros(N, 1);
det_bboxes = zeros(N, 4);

if N == 0, return; end

% Pre-compute global luminance + edges once (cheaper than per-bbox)
lum = 0.2126*img(:,:,1) + 0.7152*img(:,:,2) + 0.0722*img(:,:,3);
[gx, gy] = imgradientxy_simple(lum);
edgeMag = sqrt(gx.^2 + gy.^2);

veg = vegetation_index(img);

w_veg     = 0.40;
w_edge    = 0.30;
w_vert    = 0.15;
w_contr   = 0.15;
bias      = 0.40;

for k = 1:N
    bb = gt_bboxes(k, :);
    if all(bb == 0) || bb(3) <= 0 || bb(4) <= 0
        continue;
    end

    [u1, u2, v1, v2] = clamp_box_indices(bb, W, H);
    if u2 <= u1 || v2 <= v1
        continue;
    end

    patch_lum  = lum(v1:v2, u1:u2);
    patch_edge = edgeMag(v1:v2, u1:u2);
    patch_veg  = veg(v1:v2, u1:u2);

    % --- Feature 1: vegetation index (clipped/normalized) ---
    veg_feat = clipped_mean(patch_veg, 0, 0.6) / 0.6;

    % --- Feature 2: edge density (mean Sobel magnitude) ---
    edge_feat = clipped_mean(patch_edge, 0, 0.5) / 0.5;

    % --- Feature 3: vertical structure ---
    %   Trees are taller than they are wide -> stronger vertical gradients
    col_std = mean(std(patch_lum, 0, 1));
    row_std = mean(std(patch_lum, 0, 2));
    vert_feat = max(0, min(1, (row_std - col_std) * 8 + 0.5));

    % --- Feature 4: contrast (luminance std) ---
    contr_feat = max(0, min(1, std(patch_lum(:)) * 4));

    raw = w_veg*veg_feat + w_edge*edge_feat + w_vert*vert_feat + w_contr*contr_feat - bias;
    score = 1.0 / (1.0 + exp(-6 * raw));

    det_scores(k) = score;

    % Detected bbox: bbox center, size adjusted by feature confidence
    cx_b = bb(1) + bb(3)/2;
    cy_b = bb(2) + bb(4)/2;
    sz_factor = 0.8 + 0.4 * score;          % low score -> shrunk box (uncertain)
    new_w = bb(3) * sz_factor;
    new_h = bb(4) * sz_factor;

    % Position jitter scaled by (1 - score)
    seed = mod(bb(1)*0.31 + bb(2)*0.17, 6.2831853);
    jitter = (1 - score) * 6;
    jx = sin(seed)     * jitter;
    jy = cos(seed*1.7) * jitter;

    det_bboxes(k, :) = [cx_b + jx - new_w/2, cy_b + jy - new_h/2, new_w, new_h];
end
end

% =========================================================================
% Helpers
% =========================================================================

function [u1, u2, v1, v2] = clamp_box_indices(bb, W, H)
u1 = max(1, floor(bb(1)));
u2 = min(W, ceil (bb(1) + bb(3)));
v1 = max(1, floor(bb(2)));
v2 = min(H, ceil (bb(2) + bb(4)));
end

function v = clipped_mean(arr, lo, hi)
arr = max(lo, min(hi, arr));
if isempty(arr), v = 0; else, v = mean(arr(:)); end
end

function veg = vegetation_index(img)
R = img(:,:,1); G = img(:,:,2); B = img(:,:,3);
veg = (G - 0.5*(R + B)) ./ max(G, 0.05);
veg = max(0, veg);
end

function [gx, gy] = imgradientxy_simple(im)
% Sobel x/y (no Image Processing Toolbox dependency).
kx = [-1 0 1; -2 0 2; -1 0 1];
ky = kx';
gx = conv2(im, kx, "same");
gy = conv2(im, ky, "same");
end
