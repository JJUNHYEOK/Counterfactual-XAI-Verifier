"""Create the final T0--T2 tables and Korean paper Sections 4 and 5."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EXP_ROOT = REPO_ROOT / "experiments" / "yolov8_search_comparison"
DIV_ROOT = EXP_ROOT / "diverse_training_scenario_split"
PLAN_ID = "new_fixed_tests_t0_t2_v2"
SESSION_ID = "kci_new_fixed_tests_t0_t2_v2__20260903_160812_371388"
ROOT = HERE / "outputs" / PLAN_ID
AGGREGATE_ROOT = ROOT / "aggregated" / SESSION_ID
# Keep the empty directory from the first failed report attempt as an audit
# trail and write the complete report set to a fresh directory.
REPORT_ROOT = ROOT / "reports" / f"{SESSION_ID}_complete"
LOCK_ROOT = HERE / "outputs" / "final_experiment_lock_v1_complete"
TRAIN_PLAN_ROOT = DIV_ROOT / "config" / "diverse_training_scenario_split_v2"
TRAIN_DATA_ROOT = DIV_ROOT / "data" / "diverse_training_scenario_split_v2"
T_PLAN_ROOT = HERE / "config" / PLAN_ID


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def env_text(value: dict[str, Any]) -> str:
    return (
        f"fog={value['fog_percent']:.2f}%, "
        f"illumination={value['illumination_lux']:.1f} lx, "
        f"noise={value['camera_noise']:.4f}"
    )


def result_rows(
    scenarios: list[dict[str, Any]], revalidation: dict[str, Any]
) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in revalidation["rows"]:
        by_scenario.setdefault(row["scenario_id"], []).append(row)
    rows = []
    for item in scenarios:
        rows.append(
            {
                "scenario_id": item["scenario_id"],
                "initial_map50": item["initial_map50"],
                "initial_verdict": item["initial_verdict"],
                "initial_pass": item["initial_pass"],
                "first_fail_iteration": item["first_fail_iteration"],
                "search_success": item["search_success"],
                "evaluation_count": item["evaluation_count"],
                "final_nonfailure_fog_percent": item["final_nonfail_anchor"]["fog_percent"],
                "final_nonfailure_illumination_lux": item["final_nonfail_anchor"]["illumination_lux"],
                "final_nonfailure_camera_noise": item["final_nonfail_anchor"]["camera_noise"],
                "final_nonfailure_map50": item["final_nonfail_anchor"]["map50"],
                "final_nonfailure_verdict": item["final_nonfail_anchor"]["verdict"],
                "final_FAIL_fog_percent": item["final_fail_anchor"]["fog_percent"],
                "final_FAIL_illumination_lux": item["final_fail_anchor"]["illumination_lux"],
                "final_FAIL_camera_noise": item["final_fail_anchor"]["camera_noise"],
                "final_FAIL_map50": item["final_fail_anchor"]["map50"],
                "gap_map50": item["gap_map50"],
                "gap_environment_normalized": item["gap_environment_normalized"],
                "nonmonotonic_map_increase_count": item["nonmonotonic_map_increase_count"],
                "accounted_search_seconds": item["total_execution_seconds"],
                "scenario_driver_wall_seconds": item["scenario_driver_wall_seconds"],
                "revalidated_condition_count": len(by_scenario[item["scenario_id"]]),
                "revalidation_all_verdicts_match": all(
                    row["verdict_match"] for row in by_scenario[item["scenario_id"]]
                ),
                "revalidation_max_abs_map50_difference": max(
                    row["absolute_map50_difference"]
                    for row in by_scenario[item["scenario_id"]]
                ),
            }
        )
    return rows


def iteration_table(rows: list[dict[str, Any]], scenario_id: str) -> list[str]:
    selected = [row for row in rows if row["scenario_id"] == scenario_id]
    lines = [
        "| 반복 | 안개(%) | 조도(lx) | 잡음 | mAP@0.5 | 판정 |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected:
        lines.append(
            f"| {row['iteration']} | {row['fog_percent']:.2f} | "
            f"{row['illumination_lux']:.1f} | {row['camera_noise']:.4f} | "
            f"{row['map50']:.6f} | {row['verdict']} |"
        )
    return lines


def main() -> int:
    if REPORT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {REPORT_ROOT}")
    REPORT_ROOT.mkdir(parents=True, exist_ok=False)
    lock = read_json(LOCK_ROOT / "final_experiment_lock.json")
    training_plan = read_json(TRAIN_PLAN_ROOT / "scenario_plan.json")
    dataset = read_json(TRAIN_DATA_ROOT / "dataset_summary.json")
    t_plan = read_json(T_PLAN_ROOT / "scenario_plan.json")
    gt = read_json(ROOT / "gt_validation.json")
    similarity = read_json(ROOT / "similarity_visual_review.json")
    results = read_json(AGGREGATE_ROOT / "t0_t2_results.json")
    iterations = read_json(AGGREGATE_ROOT / "all_iteration_results.json")
    revalidation_paths = sorted(
        (ROOT / "revalidation").glob("*/independent_revalidation_summary.json")
    )
    if not revalidation_paths:
        raise FileNotFoundError("No completed independent revalidation summary")
    revalidation_path = revalidation_paths[-1]
    revalidation = read_json(revalidation_path)
    if not revalidation["all_verdicts_match"] or revalidation["any_original_search_cache_used"]:
        raise RuntimeError("Independent revalidation did not meet the reporting gate")
    if len(iterations) != 30 or not gt["all_valid"]:
        raise RuntimeError("T0-T2 result/GT gate did not pass")

    summary_rows = result_rows(results["scenarios"], revalidation)
    write_csv_new(REPORT_ROOT / "t0_t2_scenario_results.csv", summary_rows)
    concise_iterations = [
        {
            "scenario_id": row["scenario_id"],
            "iteration": row["iteration"],
            "fog_percent": row["fog_percent"],
            "illumination_lux": row["illumination_lux"],
            "camera_noise": row["camera_noise"],
            "map50": row["map50"],
            "verdict": row["verdict"],
            "next_environment": json.dumps(row["next_environment"], ensure_ascii=False, sort_keys=True),
            "next_condition_calculation": row["next_condition_calculation"],
            "evaluation_store_used": row["evaluation_store_used"],
            "total_execution_seconds": row["total_execution_seconds"],
        }
        for row in iterations
    ]
    write_csv_new(REPORT_ROOT / "t0_t2_iteration_results.csv", concise_iterations)
    final_result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": PLAN_ID,
        "session_id": SESSION_ID,
        "source_hashes": {
            "final_experiment_lock": sha256(LOCK_ROOT / "final_experiment_lock.json"),
            "T0_T2_plan": sha256(T_PLAN_ROOT / "scenario_plan.json"),
            "T0_T2_evaluation_config": sha256(T_PLAN_ROOT / "evaluation_config.json"),
            "preinference_gate": sha256(ROOT / "preinference_gate.json"),
            "search_results": sha256(AGGREGATE_ROOT / "t0_t2_results.json"),
            "iteration_results": sha256(AGGREGATE_ROOT / "all_iteration_results.json"),
            "independent_revalidation": sha256(revalidation_path),
        },
        "overall": results["overall"],
        "scenario_results": summary_rows,
        "independent_revalidation": {
            key: revalidation[key]
            for key in (
                "condition_count",
                "scenario_count",
                "all_fresh_render_and_inference",
                "any_original_search_cache_used",
                "verdict_match_count",
                "all_verdicts_match",
                "maximum_absolute_map50_difference",
                "mean_absolute_map50_difference",
                "total_wall_seconds",
            )
        },
        "limitations": [
            "T0-T2 are deterministic scenarios additionally configured in the existing MATLAB simulation environment, not real-flight or external data.",
            "All scenarios share one deterministic terrain/background and a fixed +x, 60-degree-down camera boresight.",
            "The three environment variables change jointly, so their individual causal effects are not separated.",
            "Three scenarios do not support a claim of statistical superiority.",
            "The S0-S4 visual audit found nine high-risk object/background near-similarity pairs; S0-S4 must not be called independent test data.",
        ],
    }
    write_json_new(REPORT_ROOT / "t0_t2_final_results.json", final_result)

    gt_by_id = {item["scenario_id"]: item for item in gt["validations"]}
    result_md = [
        "# T0~T2 최종 시험 결과",
        "",
        f"- 계획: `{PLAN_ID}` (`{sha256(T_PLAN_ROOT / 'scenario_plan.json')}`)",
        f"- 본 실행 세션: `{SESSION_ID}`",
        f"- 가중치: `{lock['weights_verification']['actual_sha256']}`",
        f"- 초기 PASS: {results['overall']['initial_pass_count']}/{results['overall']['scenario_count']} ({results['overall']['initial_pass_rate'] * 100:.1f}%)",
        f"- 경계 탐색 성공: {results['overall']['search_success_count']}/{results['overall']['scenario_count']} ({results['overall']['search_success_rate_all_scenarios'] * 100:.1f}%)",
        f"- 비단조 증가: {results['overall']['nonmonotonic_increase_count']}건",
        f"- 독립 재검증: {revalidation['verdict_match_count']}/{revalidation['condition_count']} 판정 일치, 최대 |ΔmAP|={revalidation['maximum_absolute_map50_difference']:.6f}",
        "",
        "| 시나리오 | 초기 mAP/판정 | 최초 FAIL | 최종 비실패 조건(mAP) | 최종 FAIL 조건(mAP) | Gap_mAP | Gap_env | 벽시계 시간(s) |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for summary, row in zip(results["scenarios"], summary_rows):
        result_md.append(
            f"| {summary['scenario_id']} | {summary['initial_map50']:.6f} / {summary['initial_verdict']} | "
            f"{summary['first_fail_iteration']} | {env_text(summary['final_nonfail_anchor'])} "
            f"({summary['final_nonfail_anchor']['map50']:.6f}, {summary['final_nonfail_anchor']['verdict']}) | "
            f"{env_text(summary['final_fail_anchor'])} ({summary['final_fail_anchor']['map50']:.6f}, FAIL) | "
            f"{summary['gap_map50']:.6f} | {summary['gap_environment_normalized']:.6f} | "
            f"{row['scenario_driver_wall_seconds']:.2f} |"
        )
    result_md.extend(
        [
            "",
            "실행시간 표의 값은 시나리오 드라이버 벽시계 시간이다. 평가 구성요소 시간을 합산한 기록값은 "
            f"T0={summary_rows[0]['accounted_search_seconds']:.2f}s, T1={summary_rows[1]['accounted_search_seconds']:.2f}s, "
            f"T2={summary_rows[2]['accounted_search_seconds']:.2f}s이며, 반복 1의 초기 평가 캐시 레코드 시간을 포함한다.",
            "",
            "`최종 비실패 조건과 FAIL 조건 사이의 경계 구간`만 보고하며 단일한 정확 경계값으로 해석하지 않는다.",
        ]
    )
    (REPORT_ROOT / "t0_t2_results_ko.md").write_text(
        "\n".join(result_md) + "\n", encoding="utf-8"
    )

    train_envs = [item["environment"] for item in training_plan["scenarios"]]
    train_ranges = {
        key: [min(float(item[key]) for item in train_envs), max(float(item[key]) for item in train_envs)]
        for key in ("fog_percent", "illumination_lux", "camera_noise")
    }
    section4 = [
        "# 4. 실험 계획",
        "",
        "## 4.1 실험 환경과 목적",
        "",
        "본 실험은 새로운 모의환경을 제안하는 것이 아니라, 기존 MATLAB 기반 무인항공기 시뮬레이션 "
        "환경에서 학습·검증 자료와 고정 평가 장면의 시각적 유사성을 감사하고, 고정된 조건에서 추가 "
        "시나리오를 평가하는 것을 목적으로 한다. MATLAB은 선형 UAV 궤적, 지형과 수목·바위·통나무 "
        "등의 배경, 사람·차량 객체 및 EO 카메라 영상을 생성한다. 렌더 영상은 640×360 PNG 프레임이며 "
        "시나리오당 18초, 181프레임이다.",
        "",
        "## 4.2 학습·검증·평가 자료",
        "",
        f"학습 자료는 {len(lock['training_and_validation']['training_scenarios'])}개 시나리오에서 각 20프레임씩 "
        f"선택한 {dataset['train_image_count']}장이고, 검증 자료는 {len(lock['training_and_validation']['validation_scenarios'])}개 "
        f"시나리오의 {dataset['validation_image_count']}장이다. 시나리오 단위로 학습/검증을 분리했다. 학습·검증 "
        f"렌더 조건 범위는 안개 {train_ranges['fog_percent'][0]:g}~{train_ranges['fog_percent'][1]:g}%, 조도 "
        f"{train_ranges['illumination_lux'][0]:g}~{train_ranges['illumination_lux'][1]:g} lx, 카메라 잡음 "
        f"{train_ranges['camera_noise'][0]:g}~{train_ranges['camera_noise'][1]:g}였다.",
        "",
        "기존 S0~S4는 사전 확정된 고정 평가 시나리오이다. 다만 dHash 거리 4 이하 16쌍을 직접 "
        "검토한 결과, 정확 중복은 없었으나 객체 위치·크기와 배경까지 매우 유사한 고위험 쌍이 9개였다. "
        "따라서 S0~S4를 독립·완전 미관측·외부 시험자료로 표현하지 않는다. 추가 시험 T0~T2는 기존 "
        "MATLAB 시뮬레이션 환경에서 추가 구성한 시나리오이다. v1의 YOLO 전 GT 검사에서 T0 전체와 "
        "T1 사람 가시성이 실패해 v1을 보존하고, 경로 거리와 객체 위치·크기만 교정한 v2를 YOLO 전에 "
        f"재고정했다(계획 SHA-256 `{sha256(T_PLAN_ROOT / 'scenario_plan.json')}`). v2 GT는 T0 P/V="
        f"{gt_by_id['T0']['person_gt_count']}/{gt_by_id['T0']['vehicle_gt_count']}, T1="
        f"{gt_by_id['T1']['person_gt_count']}/{gt_by_id['T1']['vehicle_gt_count']}, T2="
        f"{gt_by_id['T2']['person_gt_count']}/{gt_by_id['T2']['vehicle_gt_count']}였으며 유효하지 않은 bbox와 "
        "클래스는 0건이었다. 741,195쌍의 영상 비교에서 정확 중복은 0건, dHash 경고는 17건이었고 "
        "17건 모두 신규 쪽이 빈 GT인 배경 유사로 직접 확인됐다.",
        "",
        "## 4.3 자료 다양화와 고정 요소",
        "",
        "다양화 요소는 UAV 시작·종료 위치, 진행 방향, 상승·하강, 관측 거리, 사람·차량 위치와 간격, "
        "객체 물리 크기, 근접 배치에 의한 부분 가림 및 보행 반경·각속도·위상이다. 반면 카메라 광학계 "
        "(fx=fy=600, cx=320, cy=180), 세계 +x 방향과 아래쪽 60°의 고정 보어사이트, 단일 결정적 지형과 "
        "배경 구성은 고정돼 있다.",
        "",
        "## 4.4 탐지기 학습과 평가 설정",
        "",
        "탐지기는 `다양화된 합성자료로 재학습한 YOLOv8s`이다. 최종 `best.pt`는 "
        f"{lock['weights_verification']['size_bytes']:,} bytes이며 SHA-256은 "
        f"`{lock['weights_verification']['actual_sha256']}`이다. 학습은 640 입력, batch 4, SGD, 최대 50 epoch, "
        "patience 12, 초기 학습률 0.01, momentum 0.937, weight decay 0.0005, seed 42로 수행했고, 5개 "
        "검증 시나리오만으로 best.pt를 선택했다. 시험 추론은 입력 640, confidence 0.25, NMS IoU 0.70, "
        "batch 16을 사용했다.",
        "",
        "## 4.5 GT와 판정 기준",
        "",
        "GT는 `rendered_instance_mask_v1` 방식이다. 안개·잡음을 제거한 별도의 visible-instance 색상 패스에서 "
        "실제로 보이는 객체 픽셀의 최소·최대 좌표로 bbox를 만들며, 완전히 가려졌거나 화면 밖인 객체는 "
        "GT를 만들지 않는다. 시뮬레이터 클래스 1은 person, 클래스 2는 vehicle로 변환한다. AP는 클래스별 "
        "confidence 내림차순 탐지를 같은 프레임의 미사용 GT와 탐욕적으로 일대일 대응하고 IoU≥0.50을 "
        "TP로 처리한 뒤 precision envelope 적분으로 계산한다. mAP@0.5는 GT가 존재하는 person과 vehicle "
        "AP의 비가중 평균이다.",
        "",
        "PASS는 mAP@0.5≥0.50, FAIL은 mAP@0.5<0.25, MARGINAL은 0.25≤mAP@0.5<0.50이다. "
        "비교 순서는 PASS, FAIL, 그 밖의 MARGINAL이며, PASS와 MARGINAL은 비실패 앵커, FAIL만 실패 "
        "앵커로 사용한다.",
        "",
        "## 4.6 대칭 탐색과 지표",
        "",
        "초기 조건은 안개 5%, 조도 12,000 lx, 잡음 0.02이다. 초기 PASS인 시나리오만 10회 탐색했다. "
        "첫 FAIL 전에는 안개를 30%p 증가시키고 조도를 0.5배로 낮추며 잡음을 0.20 증가시켰다. 첫 FAIL "
        "후 다음 점은 각 성분별 `0.50×비실패 앵커 + 0.50×FAIL 앵커`로 계산했다. 범위는 안개 "
        "0~100%, 조도 200~15,000 lx, 잡음 0~0.60이며 clamp 후 각각 소수 2, 1, 4자리로 반올림했다. "
        "Gap_mAP은 두 앵커 mAP의 절댓값 차이이고, Gap_env는 `|Δfog|/100 + |Δlux|/14800 + "
        "|Δnoise|/0.6`이다. 평가 지표는 초기 PASS 비율, 탐색 성공률, 최초 FAIL 반복, 최종 비실패 조건과 "
        "FAIL 조건 사이의 경계 구간, 두 Gap, 악화 방향에서의 비단조 mAP 증가 횟수 및 실행시간이다.",
        "",
        "## 4.7 장비와 소프트웨어",
        "",
        f"CPU는 {lock['hardware']['cpu']['name']} ({lock['hardware']['cpu']['physical_cores']} cores/"
        f"{lock['hardware']['cpu']['logical_processors']} threads), RAM은 {lock['hardware']['system_memory']['total_gib']:.2f} GiB, "
        f"GPU는 {lock['hardware']['gpu']['name']} {lock['hardware']['gpu']['memory_total_mib']} MiB(driver "
        f"{lock['hardware']['gpu']['driver_version']})이다. Windows 10.0.19045, Python 3.11.9, MATLAB "
        f"{lock['software']['matlab']['version']}, PyTorch {lock['software']['pytorch_training_record']}, CUDA "
        f"{lock['software']['cuda_training_record']}, cuDNN {lock['software']['cudnn_training_record']}, Ultralytics "
        f"{lock['software']['ultralytics_training_record']}를 사용했다.",
        "",
        f"> {lock['freeze_statement']}",
    ]
    (REPORT_ROOT / "paper_section_4_experiment_plan_ko.md").write_text(
        "\n".join(section4) + "\n", encoding="utf-8"
    )

    section5 = [
        "# 5. 실험 결과",
        "",
        "## 5.1 초기 정상조건",
        "",
        f"T0, T1, T2의 초기 mAP@0.5는 각각 {results['scenarios'][0]['initial_map50']:.6f}, "
        f"{results['scenarios'][1]['initial_map50']:.6f}, {results['scenarios'][2]['initial_map50']:.6f}였고 모두 "
        "PASS였다. 따라서 초기 PASS 비율은 3/3(100%)였으며, 세 시나리오 모두 사전 고정된 규칙에 "
        "따라 10회 대칭 탐색을 수행했다.",
        "",
        "## 5.2 반복별 탐색 이력",
        "",
    ]
    for scenario_id in ("T0", "T1", "T2"):
        section5.extend([f"### {scenario_id}", "", *iteration_table(iterations, scenario_id), ""])
    section5.extend(
        [
            "## 5.3 최종 경계 구간과 시나리오 종합",
            "",
            "| 시나리오 | 최초 FAIL 반복 | 최종 비실패(mAP/판정) | 최종 FAIL(mAP) | Gap_mAP | Gap_env |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for summary in results["scenarios"]:
        section5.append(
            f"| {summary['scenario_id']} | {summary['first_fail_iteration']} | "
            f"{env_text(summary['final_nonfail_anchor'])}, {summary['final_nonfail_anchor']['map50']:.6f}/"
            f"{summary['final_nonfail_anchor']['verdict']} | {env_text(summary['final_fail_anchor'])}, "
            f"{summary['final_fail_anchor']['map50']:.6f}/FAIL | {summary['gap_map50']:.6f} | "
            f"{summary['gap_environment_normalized']:.6f} |"
        )
    section5.extend(
        [
            "",
            "세 시나리오 모두 4회차의 (안개 95%, 조도 1,500 lx, 잡음 0.60)에서 최초 FAIL이 "
            "관측됐고, 10회 이내에 비실패/FAIL 앵커를 모두 확보하여 경계 탐색 성공률은 3/3(100%)였다. "
            "이 결과는 단일한 정확한 실패 경계값이 아니라 각 시나리오의 `최종 비실패 조건과 FAIL 조건 "
            "사이의 경계 구간`이다.",
            "",
            "시나리오 드라이버 벽시계 시간은 T0 "
            f"{summary_rows[0]['scenario_driver_wall_seconds']:.2f}s, T1 {summary_rows[1]['scenario_driver_wall_seconds']:.2f}s, "
            f"T2 {summary_rows[2]['scenario_driver_wall_seconds']:.2f}s였고 합계는 "
            f"{sum(row['scenario_driver_wall_seconds'] for row in summary_rows):.2f}s였다. 반복 1의 초기 평가 캐시 레코드 시간을 "
            "포함하는 구성요소 합산 시간은 각각 "
            f"{summary_rows[0]['accounted_search_seconds']:.2f}s, {summary_rows[1]['accounted_search_seconds']:.2f}s, "
            f"{summary_rows[2]['accounted_search_seconds']:.2f}s였다.",
            "",
            "## 5.4 비단조 변화",
            "",
            "연속 조건에서 안개와 잡음이 증가하고 조도가 감소했는데도 mAP가 증가한 경우를 비단조 증가로 "
            "정의했다. 본 30회 탐색 이력에서는 해당 사건이 0건이었다. 이는 세 변수를 동시에 변화시킨 "
            "관측 결과이며 개별 변수의 인과 효과를 분리한 결과가 아니다.",
            "",
            "## 5.5 독립 재검증",
            "",
            "각 시나리오의 초기 정상조건, 최종 비실패 조건, 최종 FAIL 조건을 원 탐색 캐시 없이 다시 "
            f"렌더링하고 추론했다. 총 {revalidation['condition_count']}개 조건의 판정이 모두 일치했고, 최대 및 평균 "
            f"절대 mAP 차이는 각각 {revalidation['maximum_absolute_map50_difference']:.6f}, "
            f"{revalidation['mean_absolute_map50_difference']:.6f}이었다. 독립 재검증 벽시계 시간은 "
            f"{revalidation['total_wall_seconds']:.2f}s였다.",
            "",
            "## 5.6 한계",
            "",
            "본 결과는 기존 MATLAB 시뮬레이션 환경에서 추가 구성한 결정적 T0~T2 세 시나리오에 "
            "한정된다. 단일 지형과 고정 카메라 보어사이트를 공유하며 실제 비행환경 자료를 사용하지 않았다. "
            "따라서 실제 비행환경 일반화, 통계적 우수성 또는 합성자료만으로 실제 탐지 성능이 보장된다는 "
            "주장을 하지 않는다. 안개·조도·잡음을 동시에 변화시켰으므로 개별 환경변수의 인과적 영향을 "
            "분리하지도 않는다. 또한 기존 S0~S4에서는 객체와 배경이 매우 유사한 고위험 쌍 9개가 확인돼 "
            "독립 시험자료로 부를 수 없다. YOLOv8s 자체도 본 연구의 새로운 탐지 모델이 아니며, 사용한 것은 "
            "다양화된 합성자료로 재학습한 YOLOv8s이다.",
        ]
    )
    (REPORT_ROOT / "paper_section_5_experiment_results_ko.md").write_text(
        "\n".join(section5) + "\n", encoding="utf-8"
    )

    source_manifest = []
    source_paths = [
        LOCK_ROOT / "final_experiment_lock.json",
        T_PLAN_ROOT / "scenario_plan.json",
        T_PLAN_ROOT / "evaluation_config.json",
        ROOT / "preinference_gate.json",
        ROOT / "gt_validation.json",
        ROOT / "similarity_audit.json",
        ROOT / "similarity_visual_review.json",
        AGGREGATE_ROOT / "t0_t2_results.json",
        AGGREGATE_ROOT / "all_iteration_results.json",
        revalidation_path,
    ]
    for path in source_paths:
        source_manifest.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "role": "verified source used for final aggregation and paper text",
            }
        )
    write_csv_new(REPORT_ROOT / "report_source_hashes.csv", source_manifest)
    print(json.dumps({"report_root": str(REPORT_ROOT.resolve()), "files": sorted(path.name for path in REPORT_ROOT.iterdir())}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
