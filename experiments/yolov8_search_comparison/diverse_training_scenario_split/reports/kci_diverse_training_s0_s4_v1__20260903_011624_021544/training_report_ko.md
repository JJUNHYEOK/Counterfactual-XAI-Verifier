# YOLOv8s 학습 보고서

새 모델은 COCO `yolov8s.pt`에서 시작하여 기존 설정(640, batch 4, SGD, 최대 50 epoch, patience 12, seed 42)을 유지했다. 모델 선택에는 5개 사전등록 검증 시나리오만 사용했고 S0~S4는 사용하지 않았다.

| 항목 | 값 |
|---|---:|
| 완료 epoch | 50 |
| 선택 epoch | 48 |
| 사람 AP@0.5 | 0.974423 |
| 차량 AP@0.5 | 0.962491 |
| mAP@0.5 | 0.968457 |
| mAP@0.5:0.95 | 0.865924 |
| 정밀도 | 0.984358 |
| 재현율 | 0.958071 |
| 학습 시간(초) | 726.178 |
| best.pt 크기(bytes) | 22495523 |
| best.pt SHA-256 | `E141B3DC0104CBC03AD2EE573FFA451C369C9D93D8690E560E07C146078F5D9E` |
| CPU | 11th Gen Intel(R) Core(TM) i5-11400 @ 2.60GHz (6코어/12스레드) |
| GPU / VRAM | NVIDIA GeForce RTX 2060 / 6144 MiB |
| 시스템 RAM | 15.84 GiB |
| 최대 프로세스 RSS | 2.06 GiB |
| 최대 시스템 메모리 사용량 | 13.08 GiB (82.6%) |
| PyTorch 최대 GPU 할당/예약 | 1012.9/1112.0 MiB |

이는 새로운 MATLAB 모의환경 시나리오에서 모델 선택에 사용한 내부 검증 결과이며 실제 UAV 일반화 성능이 아니다.
