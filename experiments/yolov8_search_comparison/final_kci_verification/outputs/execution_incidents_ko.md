# 최종 검증 실행 중 실패·불확실성 기록

- 신규 시험 계획 v1(`new_fixed_tests_t0_t2_v1`)의 YOLO 전 GT 검사에서 T0는 사람·차량 GT가 모두 0, T1은 사람 GT가 0이었다. v1 계획·렌더·감사 자료는 그대로 보존했다. 검출 결과를 보기 전에 거리·객체 배치만 교정한 v2를 새 계획과 새 해시로 고정했고, v2는 사람·차량 GT 가시성 검사를 통과했다. v1에는 YOLO 추론을 수행하지 않았다.
- 최초 MATLAB Engine 사전검사 시도는 샌드박스 안에서 초기화가 정체돼 프레임·출력 생성 전에 해당 Python/MATLAB 프로세스만 종료했다. 같은 고정 계획을 샌드박스 밖에서 재실행해 정상 완료했다.
- 최종 고정 문서 생성의 첫 시도는 시각 감사 요약 키 이름 불일치로 Markdown 작성 전에 실패했다. 이미 만들어진 부분 JSON·CSV는 `outputs/final_experiment_lock_v1`에 보존했으며, 수정 후 완전한 세 파일은 `outputs/final_experiment_lock_v1_complete`에 새로 생성했다.
- 논문 보고서 생성의 첫 시도는 Python iterator 처리 오류로 빈 보고서 디렉터리만 만든 뒤 실패했다. 빈 디렉터리를 보존하고 수정 후 완전한 결과를 세션명 뒤 `_complete`가 붙은 새 디렉터리에 생성했다.
- 평가 JSON의 `model.weights_provenance`는 경로 문자열 판별 결함으로 v2 재학습 가중치를 일반 COCO 가중치라고 잘못 표기한다. 체크포인트의 `train_args`와 세 곳의 일치하는 SHA-256을 우선 근거로 사용하며, 해당 잘못된 provenance 문자열은 논문에 사용하지 않는다.
- `pytest` 패키지는 현재 가상환경에 설치돼 있지 않아 pytest 명령은 실행되지 않았다. 같은 세 테스트 모듈을 표준 `unittest`로 실행했으며 45개 테스트가 모두 통과했다.
