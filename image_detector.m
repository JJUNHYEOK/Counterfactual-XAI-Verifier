function [det_scores, det_bboxes] = image_detector(img, gt_bboxes)
% image_detector — Object-vs-background detector for the rendered EO image.
%
% Each GT bounding box is scored by how much its interior looks like a
% solid object placed on a different-colored background. The signals are
% chosen so each environmental stress axis (fog / illumination / noise)
% degrades the score through its physical effect on the rendered image:
%
%   1) Boundary contrast (Δ RGB)
%        Mean RGB of bbox interior vs. surrounding ring. High when an
%        intruder's color differs from the terrain underneath it.
%        Fog blends both regions toward grey-white → this signal drops.
%
%   2) Interior uniformity (1 − normalized luminance std)
%        Intruders render as solid color regions (blue people, orange
%        vehicles). High when the interior is clean. Camera noise adds
%        speckle texture inside the bbox → uniformity drops.
%
%   3) Mean luminance
%        Captures sensitivity to low illumination. Low light dims the
%        whole image and pushes intruder colors into a near-black band
%        where colors are hard to separate.
%
% The three signals combine multiplicatively so any single degradation
% can break detection — matching how a real CCD/sensor would fail.
%
% Output bboxes are jittered/resized proportional to (1 − score) so low-
% confidence detections also fail IoU ≥ 0.5 → become FP under stress.

[H, W, ~] = size(img);
N = size(gt_bboxes, 1);
det_scores = zeros(N, 1);
det_bboxes = zeros(N, 4);
if N == 0, return; end

R = img(:, :, 1);
G = img(:, :, 2);
B = img(:, :, 3);
lum = 0.2126 * R + 0.7152 * G + 0.0722 * B;

for k = 1:N
    bb = gt_bboxes(k, :);
    if all(bb == 0) || bb(3) <= 0 || bb(4) <= 0
        continue;
    end

    [u1, u2, v1, v2] = clamp_box_indices(bb, W, H);
    if u2 <= u1 || v2 <= v1
        continue;
    end
    bw = u2 - u1 + 1;
    bh = v2 - v1 + 1;
    if bw < 2 || bh < 2
        continue;
    end

    % --- Interior region statistics: full bbox ---
    u1c = u1; u2c = u2; v1c = v1; v2c = v2;
    r_in = mean(R(v1c:v2c, u1c:u2c), "all");
    g_in = mean(G(v1c:v2c, u1c:u2c), "all");
    b_in = mean(B(v1c:v2c, u1c:u2c), "all");
    lum_in_patch  = lum(v1c:v2c, u1c:u2c);
    lum_in_mean   = mean(lum_in_patch, "all");
    lum_in_std    = std(lum_in_patch(:));

    % --- Surrounding ring (expand bbox ~50 % each side) ---
    mu = max(2, round(bw * 0.50));
    mv = max(2, round(bh * 0.50));
    u1o = max(1, u1 - mu);  u2o = min(W, u2 + mu);
    v1o = max(1, v1 - mv);  v2o = min(H, v2 + mv);
    r_out = mean(R(v1o:v2o, u1o:u2o), "all");
    g_out = mean(G(v1o:v2o, u1o:u2o), "all");
    b_out = mean(B(v1o:v2o, u1o:u2o), "all");

    % --- Signal 1: Boundary contrast (RGB Euclidean) ---
    % No saturation boost — let fog progressively reduce color separation
    % so mid/heavy stress are clearly distinguishable from sterile.
    color_diff = sqrt( (r_in - r_out)^2 + (g_in - g_out)^2 + (b_in - b_out)^2 );
    contrast_feat = min(1, color_diff * 1.0);     % 1.0 RGB distance → 1.0

    % --- Signal 2: Interior uniformity (sharper noise sensitivity) ---
    % Clean intruders have intrinsic luminance std ≈ 0.10 (head/body
    % transition). Subtract this baseline and apply sharper decay so
    % noise σ ≈ 0.16 produces a clear uniformity hit.
    extra_std = max(0, lum_in_std - 0.10);
    uniformity_feat = exp(-extra_std * 10);       % extra 0.05 → 0.61, 0.10 → 0.37

    % --- Signal 3: Mean luminance (raised threshold, low-light hurts) ---
    % Dimmed image (illum << 3000 lx) produces lower mean luminance which
    % the detector treats as reduced sensor SNR.
    lum_feat = min(1, lum_in_mean / 0.30);

    % --- Combine multiplicatively (any axis can independently break score) ---
    score = contrast_feat * uniformity_feat * lum_feat;
    score = max(0, min(1, score));
    det_scores(k) = score;

    % --- Detection bbox: tighter than original, jitter scaled by (1-score) ---
    cx_b = bb(1) + bb(3) / 2;
    cy_b = bb(2) + bb(4) / 2;
    sz_factor = 0.90 + 0.20 * score;                  % 0.90 (low) .. 1.10 (high)
    new_w = bb(3) * sz_factor;
    new_h = bb(4) * sz_factor;

    seed = mod(bb(1) * 0.31 + bb(2) * 0.17, 6.2831853);
    jitter = (1 - score) * 3;                         % was 6 — keeps IoU > 0.5 longer
    jx = sin(seed)       * jitter;
    jy = cos(seed * 1.7) * jitter;
    det_bboxes(k, :) = [cx_b + jx - new_w / 2, cy_b + jy - new_h / 2, new_w, new_h];
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
