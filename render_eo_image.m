function img = render_eo_image(uav, obs_xyz, obs_rh, fog, illum, noise, cam_intrin, img_size)
% render_eo_image — EO 카메라 이미지 dispatcher (2D synthetic vs 3D capture).
%
% USE_3D=true 면 eo_camera_3d_render (진짜 3D 1인칭 시점 캡처),
% false 면 기존 render_camera_image (2D synthetic) 호출.
%
% RENDER_EVERY_N>1 로 설정하면 3D 캡처를 N 프레임당 1회만 수행, 나머지 프레임은
% 캐시된 이미지를 재사용. 자동 boundary search 루프에서 getframe 비용을 낮추는
% 용도. 시각적 연속성이 약간 깎이지만 mAP 계산은 GT bbox 기반이므로 검증
% 결과에는 영향 없음.

USE_3D         = true;   % toggle: true = 3D 1st person, false = 2D synthetic
RENDER_EVERY_N = 1;      % capture every N frames in 3D mode (≥1)

persistent cached_img frame_counter
if isempty(frame_counter), frame_counter = 0; end
frame_counter = frame_counter + 1;

if ~USE_3D
    img = render_camera_image(uav, obs_xyz, obs_rh, fog, illum, noise, cam_intrin, img_size);
    return;
end

% 3D mode
if isempty(cached_img) || mod(frame_counter, RENDER_EVERY_N) == 1 || RENDER_EVERY_N <= 1
    try
        img = eo_camera_3d_render(uav, obs_xyz, obs_rh, fog, illum, noise, cam_intrin, img_size);
        cached_img = img;
    catch ME
        warning("eo_camera_3d_render failed (%s); falling back to 2D", ME.message);
        img = render_camera_image(uav, obs_xyz, obs_rh, fog, illum, noise, cam_intrin, img_size);
        cached_img = img;
    end
else
    img = cached_img;
end
end
