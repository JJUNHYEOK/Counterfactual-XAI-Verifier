# 최종 실험 조건 고정 문서

- 고정 ID: `final_experiment_lock_v1`
- 생성 시각(UTC): `2026-09-03T06:43:17.889143+00:00`
- 목적: KCI final reproducibility lock; documents actual executed settings without retraining or threshold adjustment.

> 이 시점 이후 S0~S4 또는 신규 시험 결과를 보고 가중치, 판정 기준, 탐색 방법 및 환경 범위를 변경하지 않는다.

## 1. 최종 가중치

- 구조: 다양화된 합성자료로 재학습한 YOLOv8s two-class object detector (탐지, 2개 클래스, 11,136,374 parameters)
- 경로: `C:\Users\lab\Counterfactual-XAI-Verifier\experiments\yolov8_search_comparison\diverse_training_scenario_split\training\diverse_training_scenario_split_v2\diverse_yolov8s_seed42__20260903_010327_583537\weights\best.pt`
- 크기: 22,495,523 bytes
- 예상 SHA-256: `E141B3DC0104CBC03AD2EE573FFA451C369C9D93D8690E560E07C146078F5D9E`
- 실제 SHA-256: `E141B3DC0104CBC03AD2EE573FFA451C369C9D93D8690E560E07C146078F5D9E`
- 대조 결과: **MATCH**

## 2. 영상·탐지·평가 설정

- 렌더 영상: 640×360 PNG 프레임 시퀀스, frame stride 1, S0~S4 각 181프레임
- YOLO 입력 크기: 640
- 탐지 confidence 기준: 0.25
- 중복 탐지 제거 NMS IoU: 0.7
- AP 일치 IoU: 0.5
- 추론 batch: 설정 16, 관측 유효 batch 16
- 장치: 설정 `auto`, 실제 `cuda:0`
- 클래스: 체크포인트 0=`person`, 1=`vehicle`. 현재 체크포인트에서는 추가 병합이 없었다.
- 일반 변환 규칙: person/pedestrian/people/human은 person, vehicle/car/motorcycle/motorbike/bus/truck/van은 vehicle, 나머지는 제외한다.

## 3. GT와 mAP

- GT 모드: `rendered_instance_mask_v1`
- 렌더러: `render_eo_image + eo_camera_3d_render visible-instance colour pass`
- 생성 방식: A second no-fog/no-noise visible-instance colour rendering pass identifies visible pixels per object. The tight min/max pixel rectangle becomes the GT bbox; fully hidden/out-of-view objects emit no box.
- AP: Per class, detections over all frames are sorted by descending confidence and greedily matched one-to-one to the highest-IoU unused GT in the same frame; IoU >= 0.5 is TP. Integral under the monotonic precision envelope at recall-change points.
- mAP: Unweighted arithmetic mean of person AP50 and vehicle AP50 for classes with at least one GT.

## 4. 판정식과 우선순위

- PASS: `mAP@0.5 >= 0.5`
- MARGINAL: `0.25 <= mAP@0.5 < 0.5`
- FAIL: `mAP@0.5 < 0.25`
- 비교 우선순위: Evaluate PASS first (>=0.50), then FAIL (<0.25), otherwise MARGINAL. Equality 0.50 is PASS and equality 0.25 is MARGINAL.
- 앵커 우선순위: FAIL updates the failure anchor; PASS or MARGINAL updates the non-failure anchor.

## 5. 환경과 대칭 탐색

- 초기점: `{"camera_noise": 0.02, "fog_percent": 5.0, "illumination_lux": 12000.0}`
- 범위: `{"camera_noise": [0.0, 0.6], "fog_percent": [0.0, 100.0], "illumination_lux": [200.0, 15000.0]}`
- 최초 FAIL 이전: `fog'=clamp(fog+30); illumination'=clamp(illumination*0.5); noise'=clamp(noise+0.2)`
- 최초 FAIL 이후: `next = clamp_and_round(0.50*nonfailure_anchor + 0.50*failure_anchor), component-wise`
- 최대 평가 횟수: 10회
- Gap_mAP: `Gap_mAP = |mAP50_nonfailure - mAP50_FAIL|`
- Gap_env: `Gap_env = |nf_fog-f_fog|/100 + |nf_lux-f_lux|/14800 + |nf_noise-f_noise|/0.6` (세 정규화 항의 합이며 평균이 아님)
- clamp 후 반올림: fog 2자리, illumination 1자리, noise 4자리

## 6. 난수와 자료 계획

- 학습 난수값: 42; 평가 공통 난수값: 42
- 학습: 18개 시나리오 × 20프레임 = 360장
- 검증: 5개 시나리오 × 20프레임 = 100장
- 학습 시나리오: T01_east_level_low, T02_east_level_high, T03_east_ascending, T04_east_descending, T05_west_retreat_low, T06_west_retreat_high, T07_north_lateral, T08_south_lateral, T09_diagonal_swnE, T10_diagonal_nwse, T11_far_to_near, T12_near_to_far, T13_high_short, T14_low_close, T15_lateral_ascending, T16_lateral_descending, T17_partial_occlusion, T18_wide_diagonal
- 검증 시나리오: V01_east_ascending, V02_west_retreat, V03_north_lateral, V04_diagonal_descending, V05_occluded_approach
- 가중치 선택: Ultralytics best.pt selected from the five-scenario validation split only.
- 고정 시험 계획: `multi_scenario_symmetric_plan_v2`, S0~S4 (S0, S1, S2, S3, S4)

## 7. 실행 환경

- CPU: 11th Gen Intel(R) Core(TM) i5-11400 @ 2.60GHz (6 cores/12 threads)
- RAM: 15.84 GiB (17,009,278,976 bytes)
- GPU: NVIDIA GeForce RTX 2060, 6144 MiB, driver 591.59
- Python: 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]
- MATLAB: 25.2.0.3150157 (R2025b) Update 4, release 2025b
- PyTorch: 2.5.1+cu121; CUDA runtime 12.1; cuDNN 90100
- Ultralytics: 8.4.46
- OS: Windows-10-10.0.19045-SP0

## 8. Git 상태

- 브랜치: `yeah`
- HEAD: `c2500915be591fb4308d377bf92d73d45cd68879`
- clean 여부: `False`
```text
## yeah
 M eo_camera_3d_render.m
 M render_eo_image.m
?? experiments/yolov8_search_comparison/
```

## 9. 유사 영상 감사와 주장 제한

- 시각 검토: 16쌍, 정확 SHA 중복 0쌍
- 분류: 배경/빈 화면 유사 6쌍, 객체 구성 충분히 다름 1쌍, 객체·크기·배경 매우 유사 9쌍
- 위험도: 낮음 0쌍, 중간 7쌍, 높음 9쌍
- 결론: S0-S4 are previously pre-registered fixed evaluation scenarios. They must not be described as independent, completely unseen, or external test data because the visual audit found material near-similarity.

## 10. 확인된 불일치

- `evaluation JSON model.weights_provenance`: 기록값 `general Ultralytics COCO pretrained YOLOv8s; not custom-trained`. 실제 판단은 `diversified synthetic-data retrained YOLOv8s best.pt`이다. Use the checkpoint fields and verified hash; do not use the erroneous provenance string in the paper.
- `dataset_summary.near_duplicate_review_passed`: 기록값 `True`. 실제 판단은 `The automated audit passed its configured warning policy, but subsequent mandatory visual review found 9 high-risk category-3 pairs.`이다. Do not describe S0-S4 as independent/unseen/external test data.

## 11. 고정 파일 해시

총 62개 파일의 경로·크기·SHA-256을 `final_experiment_lock_files.csv`에 기록했다.

## 12. 사용할 수 없는 주장

- S0-S4 are independent, completely unseen, or external test data.
- Exact-hash separation proves absence of train-test visual leakage.
- The results generalize to real flight environments.
- Synthetic data alone guarantees real-world detection performance.
