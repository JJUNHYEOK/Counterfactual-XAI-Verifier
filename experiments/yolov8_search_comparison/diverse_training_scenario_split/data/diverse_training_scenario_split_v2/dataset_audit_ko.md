# 시나리오 다양화 자료 감사

## 판정

- 전체 자료 감사: **PASS**
- 학습/검증: 18/5개 시나리오, 360/100장
- 선택 프레임: 18초 181프레임 중 사전 확정한 20개([1, 10, 20, 29, 39, 48, 58, 67, 77, 86, 96, 105, 115, 124, 134, 143, 153, 162, 172, 181])
- GT: 실제 가시 픽셀 인스턴스 색상 렌더(`rendered_instance_mask_v1`)
- 기존 핵심 파일 보존: PASS

## 시나리오별 GT

| ID | 분할 | 영상 | 사람 GT | 차량 GT | 두 클래스 동시 프레임 | 빈 프레임 | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T01_east_level_low | train | 20 | 13 | 14 | 10 | 3 | PASS |
| T02_east_level_high | train | 20 | 21 | 20 | 19 | 0 | PASS |
| T03_east_ascending | train | 20 | 13 | 15 | 12 | 5 | PASS |
| T04_east_descending | train | 20 | 15 | 17 | 14 | 2 | PASS |
| T05_west_retreat_low | train | 20 | 14 | 14 | 10 | 10 | PASS |
| T06_west_retreat_high | train | 20 | 38 | 32 | 18 | 0 | PASS |
| T07_north_lateral | train | 20 | 25 | 27 | 17 | 2 | PASS |
| T08_south_lateral | train | 20 | 40 | 42 | 20 | 0 | PASS |
| T09_diagonal_swnE | train | 20 | 12 | 14 | 9 | 3 | PASS |
| T10_diagonal_nwse | train | 20 | 13 | 16 | 12 | 4 | PASS |
| T11_far_to_near | train | 20 | 16 | 15 | 14 | 3 | PASS |
| T12_near_to_far | train | 20 | 24 | 20 | 20 | 0 | PASS |
| T13_high_short | train | 20 | 31 | 34 | 18 | 0 | PASS |
| T14_low_close | train | 20 | 4 | 7 | 4 | 13 | PASS |
| T15_lateral_ascending | train | 20 | 24 | 26 | 19 | 0 | PASS |
| T16_lateral_descending | train | 20 | 22 | 25 | 13 | 6 | PASS |
| T17_partial_occlusion | train | 20 | 17 | 19 | 16 | 2 | PASS |
| T18_wide_diagonal | train | 20 | 11 | 17 | 10 | 2 | PASS |
| V01_east_ascending | val | 20 | 18 | 17 | 14 | 3 | PASS |
| V02_west_retreat | val | 20 | 35 | 33 | 20 | 0 | PASS |
| V03_north_lateral | val | 20 | 25 | 28 | 17 | 0 | PASS |
| V04_diagonal_descending | val | 20 | 16 | 15 | 14 | 3 | PASS |
| V05_occluded_approach | val | 20 | 12 | 15 | 12 | 5 | PASS |

범위 밖/비양수 상자, 잘못된 클래스, 프레임-라벨 번호 불일치는 모두 0이어야 통과한다. 각 시나리오마다 5장의 오버레이 진단 영상을 생성하였다.

## 시나리오 단위 분리와 정확 중복

- ID/시드/궤적/객체 배치 중복: 0건
- 영상 SHA-256 중복: 0건
- 영상+라벨 쌍 중복: 0건
- 비어 있지 않은 정답 SHA-256 중복: 0건
- 정확 중복 판정: **PASS**

빈 정답 파일은 내용상 같은 SHA-256이 불가피하므로 원시 라벨 중복 판정에서 제외하되, 해당 프레임의 영상+라벨 결합 해시는 계속 검사하였다.

## 지각 해시 근접 중복

64비트 dHash의 해밍 거리를 사용했으며 경고 기준은 ≤ 4이다. 경고는 삭제나 재구성을 자동 요구하지 않고, 정확 SHA 및 사전 확정된 궤적·배치가 서로 다른지 함께 검토한다.

| 비교 | 쌍 수 | 최소 거리 | 중앙값 | 평균 | 경고 쌍 |
|---|---:|---:|---:|---:|---:|
| train-vs-validation | 36000 | 5 | 31.0 | 31.314 | 0 |
| train-vs-S0-S4 | 325800 | 1 | 32.0 | 31.784 | 16 |
| validation-vs-S0-S4 | 90500 | 5 | 32.0 | 31.694 | 0 |

근접 중복 검토 판정: **PASS**. 가장 가까운 영상 쌍은 `near_duplicate_audit.json`에 경로와 함께 기록하였다.

## 해석 범위

자료 다양화는 궤적 축·방향, 상승/하강, 시작·종료 위치, 고도, 객체 배치·간격·부분 가림, 관측 거리, 정상 범위 환경 조건을 대상으로 했다. 카메라 방향·기울기와 다중 지형/배경은 현 렌더러가 고정값만 지원하므로 변경하지 않았다.
