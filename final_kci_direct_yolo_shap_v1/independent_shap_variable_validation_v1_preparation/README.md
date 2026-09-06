# 직접 Shapley 변수 선택 독립 검증 실행 준비

IVS00~IVS09의 새 장면 10개에서 장면당 11조건, 총 110조건을 평가하도록 잠갔다. 번호가 가장 큰 `static_validation_attempt_*.json`에서 최신 `STATIC_VALIDATION_PASS`를 확인한다. 최초 `static_validation.json`의 실패 기록은 보존했으며, 실제 렌더러 제한이 아닌 과거 관측 메타데이터 범위를 하드 제한으로 취급한 검사 오류였다. `preparation_revision_002.json`에 결과 생성 전 기술 교정과 불변 항목을 기록했다.

## 실행

다음 명령은 `DESKTOP-CMOIPGE\lab` 일반 PowerShell에서만 실행한다. `MATLAB_PREFDIR`을 설정하지 않는다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\lab\Counterfactual-XAI-Verifier\.k\final_kci_direct_yolo_shap_v1\independent_shap_variable_validation_v1_preparation\run_independent_validation_lab.ps1"
```

자동 재시도는 없다. 명시적으로 승인된 재개만 `-Resume`을 사용하며, 결과·완료 manifest·해시가 모두 검증된 조건만 재사용한다. 첫 실행 예상은 약 157분이며 전체 제한은 4시간이다. 중앙값 기반 저장공간 추정은 약 8.2GB이고 보수적으로 11GB를 확보한다.
