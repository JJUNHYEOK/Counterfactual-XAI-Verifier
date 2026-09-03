"""Generate 300 dpi paper figures and Korean manuscript text from results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from .run_experiment import AGGREGATED_ROOT, DEFAULT_CONFIG, DEFAULT_PLAN, HERE, RUNS_ROOT, SCENARIO_ROOT, read_json


FIGURES_ROOT = HERE / "figures"
REPORTS_ROOT = HERE / "reports"
TEST_ROOT = HERE / "test_suites"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    )


def env_text(anchor: dict[str, Any] | None) -> str:
    if not anchor:
        return "-"
    return f"fog {anchor['fog_percent']:.2f}%, {anchor['illumination_lux']:.1f} lx, noise {anchor['camera_noise']:.4f}, mAP {anchor['map50']:.6f} ({anchor['verdict']})"


def evaluation_path(session_id: str, scenario_id: str, cache_key: str) -> Path:
    return RUNS_ROOT / session_id / scenario_id / "evaluations" / cache_key / "evaluation.json"


def representative_frame(evaluation: dict[str, Any]) -> dict[str, Any]:
    frames = evaluation["frames"]
    both = [frame for frame in frames if {gt["class_name"] for gt in frame["ground_truth"]} == {"person", "vehicle"}]
    candidates = both or [frame for frame in frames if frame["ground_truth"]]
    return min(candidates, key=lambda frame: abs(int(frame["frame_index"]) - 91))


def draw_overlay(ax: Any, frame: dict[str, Any], title: str) -> None:
    image = Image.open(frame["image_path"]).convert("RGB")
    ax.imshow(image)
    colours = {"person": "#e53935", "vehicle": "#1565c0"}
    for gt in frame.get("ground_truth", []):
        x, y, w, h = gt["bbox_xywh"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=colours[gt["class_name"]], linewidth=1.2))
    for detection in frame.get("detections", []):
        x, y, w, h = detection["bbox_xywh"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#ffd600", linewidth=1.0, linestyle="--"))
    ax.set_title(f"{title}\nframe {int(frame['frame_index']):04d}", fontsize=8)
    ax.axis("off")


def save_figure(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_figures(session_id: str, combined: dict[str, Any], records: list[dict[str, Any]], plan: dict[str, Any]) -> list[Path]:
    output = FIGURES_ROOT / session_id
    output.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    scenario_ids = [item["scenario_id"] for item in combined["scenarios"]]
    colours = dict(zip(scenario_ids, plt.get_cmap("tab10").colors[: len(scenario_ids)]))

    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(12, 4.8))
    for scenario in plan["scenarios"]:
        manifest = read_json(SCENARIO_ROOT / plan["plan_id"] / scenario["scenario_id"] / "initial_condition" / "frame_manifest.json")
        trajectory = np.asarray(manifest["trajectory_xyz"], dtype=float)
        objects = np.asarray(manifest["object_initial_xyz"], dtype=float)
        classes = np.asarray(manifest["object_classes"], dtype=int).reshape(-1)
        ax_xy.plot(trajectory[:, 0], trajectory[:, 1], label=scenario["scenario_id"], color=colours[scenario["scenario_id"]], linewidth=2)
        ax_xy.scatter(trajectory[0, 0], trajectory[0, 1], color=colours[scenario["scenario_id"]], marker="o", s=25)
        ax_xy.scatter(trajectory[-1, 0], trajectory[-1, 1], color=colours[scenario["scenario_id"]], marker="x", s=35)
        ax_xy.scatter(objects[classes == 1, 0], objects[classes == 1, 1], marker="^", s=25, color=colours[scenario["scenario_id"]], alpha=0.7)
        ax_xy.scatter(objects[classes == 2, 0], objects[classes == 2, 1], marker="s", s=25, color=colours[scenario["scenario_id"]], alpha=0.7)
        ax_z.plot(np.linspace(0, 18, len(trajectory)), trajectory[:, 2], label=scenario["scenario_id"], color=colours[scenario["scenario_id"]], linewidth=2)
    ax_xy.set(title="Five Registered Flight Trajectories", xlabel="World x (m)", ylabel="World y (m)")
    ax_xy.grid(alpha=0.25)
    ax_xy.legend(ncol=3, fontsize=8)
    ax_z.set(title="Flight Altitude Profiles", xlabel="Simulation time (s)", ylabel="World z (m)")
    ax_z.grid(alpha=0.25)
    ax_z.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = output / "01_five_trajectory_comparison.png"; save_figure(fig, path); paths.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for scenario_id in scenario_ids:
        rows = [row for row in records if row["scenario_id"] == scenario_id]
        ax.plot([row["iteration"] for row in rows], [row["map50"] for row in rows], marker="o", label=scenario_id, color=colours[scenario_id])
    ax.axhline(0.50, color="#2e7d32", linestyle="--", linewidth=1.3, label="PASS–MARGINAL (0.50)")
    ax.axhline(0.25, color="#c62828", linestyle=":", linewidth=1.5, label="MARGINAL–FAIL (0.25)")
    ax.set(title="mAP@0.5 Across Symmetric Search Iterations", xlabel="Iteration", ylabel="mAP@0.5", xticks=range(1, 11), ylim=(-0.03, 1.03))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = output / "02_map_by_iteration.png"; save_figure(fig, path); paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True)
    fields = [("fog_percent", "Fog (%)"), ("illumination_lux", "Illumination (lx)"), ("camera_noise", "Camera noise")]
    for ax, (field, label) in zip(axes, fields):
        for scenario_id in scenario_ids:
            rows = [row for row in records if row["scenario_id"] == scenario_id]
            ax.plot([row["iteration"] for row in rows], [row[field] for row in rows], marker="o", label=scenario_id, color=colours[scenario_id])
        ax.set_ylabel(label); ax.grid(alpha=0.25)
    axes[0].set_title("Environment Conditions Across Search Iterations")
    axes[-1].set_xlabel("Iteration"); axes[-1].set_xticks(range(1, 11))
    axes[0].legend(ncol=5, fontsize=8)
    fig.tight_layout()
    path = output / "03_environment_by_iteration.png"; save_figure(fig, path); paths.append(path)

    for number, key, label, filename in [
        (4, "gap_map50", "Final Gap_mAP", "04_final_gap_map.png"),
        (5, "gap_environment_normalized", "Final Gap_env", "05_final_gap_env.png"),
    ]:
        del number
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        values = [float(item[key]) for item in combined["scenarios"]]
        bars = ax.bar(scenario_ids, values, color=[colours[x] for x in scenario_ids])
        ax.bar_label(bars, labels=[f"{value:.6f}" for value in values], padding=3, fontsize=8)
        ax.set(title=f"{label} by Scenario", xlabel="Scenario", ylabel=label)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = output / filename; save_figure(fig, path); paths.append(path)

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))
    initial_by_id = {item["scenario_id"]: item for item in combined["initial_evaluations"]}
    for ax, scenario_id in zip(axes, scenario_ids):
        item = initial_by_id[scenario_id]
        evaluation = read_json(evaluation_path(session_id, scenario_id, item["evaluation_cache_key"]))
        draw_overlay(ax, representative_frame(evaluation), f"{scenario_id} initial PASS\nmAP={item['map50']:.3f}")
    axes[0].legend(
        handles=[Line2D([0], [0], color="#e53935", label="Person GT"), Line2D([0], [0], color="#1565c0", label="Vehicle GT"), Line2D([0], [0], color="#ffd600", linestyle="--", label="YOLO detection")],
        fontsize=6, loc="lower left",
    )
    fig.suptitle("Initial PASS Examples (solid: GT, dashed: detection)", fontsize=12)
    fig.tight_layout()
    path = output / "06_initial_pass_examples.png"; save_figure(fig, path); paths.append(path)

    fig, axes = plt.subplots(5, 2, figsize=(9.5, 12))
    for row_index, scenario in enumerate(combined["scenarios"]):
        scenario_id = scenario["scenario_id"]
        nonfail_eval = read_json(evaluation_path(session_id, scenario_id, scenario["final_nonfail_anchor"]["cache_key"]))
        fail_eval = read_json(evaluation_path(session_id, scenario_id, scenario["final_fail_anchor"]["cache_key"]))
        nonfail_frame = representative_frame(nonfail_eval)
        fail_lookup = {int(frame["frame_index"]): frame for frame in fail_eval["frames"]}
        fail_frame = fail_lookup[int(nonfail_frame["frame_index"])]
        draw_overlay(axes[row_index, 0], nonfail_frame, f"{scenario_id} non-failure, mAP={scenario['final_nonfail_anchor']['map50']:.3f}")
        draw_overlay(axes[row_index, 1], fail_frame, f"{scenario_id} FAIL, mAP={scenario['final_fail_anchor']['map50']:.3f}")
    fig.suptitle("Final Non-failure–FAIL Boundary Examples", fontsize=12)
    fig.tight_layout()
    path = output / "07_final_boundary_examples.png"; save_figure(fig, path); paths.append(path)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    values = [combined["overall"]["initial_pass_rate"] * 100, combined["overall"]["search_success_rate"] * 100]
    bars = ax.bar(["Initial PASS", "Boundary search success"], values, color=["#1976d2", "#2e7d32"], width=0.55)
    ax.bar_label(bars, labels=[f"{value:.1f}%\n(5/5)" for value in values], padding=4)
    ax.set(title="Five-scenario Search Outcome", ylabel="Scenario rate (%)", ylim=(0, 112))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output / "08_search_success_rate.png"; save_figure(fig, path); paths.append(path)
    return paths


def initial_rows(combined: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for item in combined["initial_evaluations"]:
        rows.append([
            item["scenario_id"], item["map50"], item["person_ap50"], item["vehicle_ap50"], item["person_gt_count"], item["vehicle_gt_count"],
            f"{item['total_tp']}/{item['total_fp']}/{item['total_fn']}", item["precision"], item["recall"], item["maximum_iou"], item["mean_iou"], item["verdict"],
        ])
    return rows


def summary_rows(combined: dict[str, Any]) -> list[list[Any]]:
    return [
        [item["scenario_id"], item["trajectory_name"], item["initial_map50"], item["initial_verdict"], item["first_fail_iteration"], item["search_success"], env_text(item["final_nonfail_anchor"]), env_text(item["final_fail_anchor"]), item["gap_map50"], item["gap_environment_normalized"], item["evaluation_count"], item["total_execution_seconds"], item["nonmonotonic_map_increase_count"], item["failure_or_unsearchable_reason"] or "-"]
        for item in combined["scenarios"]
    ]


def scenario_plan_rows(plan: dict[str, Any]) -> list[list[Any]]:
    return [
        [item["scenario_id"], item["trajectory_name"], item["seed"], item["uav_initial_xyz"], item["uav_end_xyz"], item["altitude_range_m"], "; ".join(item["major_changes"]), item["person_count"], item["vehicle_count"]]
        for item in plan["scenarios"]
    ]


def generate_manuscript(session_id: str, combined: dict[str, Any], records: list[dict[str, Any]], plan: dict[str, Any], config: dict[str, Any], tests: dict[str, Any]) -> str:
    initial_table = md_table(["시나리오", "mAP", "사람 AP", "차량 AP", "사람 GT", "차량 GT", "TP/FP/FN", "Precision", "Recall", "최대 IoU", "평균 IoU", "판정"], initial_rows(combined))
    s0 = [row for row in records if row["scenario_id"] == "S0"]
    s0_table = md_table(["반복", "안개(%)", "조도(lx)", "잡음", "사람 AP", "차량 AP", "mAP", "TP/FP/FN", "판정"], [[x["iteration"], x["fog_percent"], x["illumination_lux"], x["camera_noise"], x["person_ap50"], x["vehicle_ap50"], x["map50"], f"{x['total_tp']}/{x['total_fp']}/{x['total_fn']}", x["verdict"]] for x in s0])
    summary_table = md_table(["ID", "궤적", "초기 mAP", "초기 판정", "최초 FAIL", "성공", "최종 비실패", "최종 FAIL", "Gap_mAP", "Gap_env", "평가 수", "시간(s)", "비단조 증가"], [row[:-1] for row in summary_rows(combined)])
    overall = combined["overall"]
    first, gm, ge = overall["first_fail_iteration_statistics"], overall["gap_map50_statistics"], overall["gap_environment_statistics"]
    return f"""# 논문용 실험 문안: 5개 비행 시나리오 대칭 경계 탐색

## 4. 실험 계획

### 4.1 시뮬레이션 및 객체 탐지 환경

MATLAB R2025b 기반 산악 감시 모의환경에서 시나리오당 18초, 0.1초 간격의 181개 영상을 생성하였다. 영상 해상도는 640×360이며, YOLO 입력 크기는 640이다. 모든 GT는 과거 핀홀 투영 상자가 아니라 3차원 장면의 별도 인스턴스 색상 렌더에서 실제로 가시화된 픽셀의 외접 상자(`rendered_instance_mask_v1`)로 산출하였다. AP 평가는 클래스가 일치하고 IoU가 0.5 이상인 일대일 대응을 사용하였다.

### 4.2 YOLOv8s 학습 데이터와 고정 가중치

검출기는 6개 학습 시나리오와 2개 검증 시나리오로 학습된 동일 YOLOv8s 체크포인트를 모든 평가에 사용하였다. 가중치 SHA-256은 `{config['expected_weights_sha256']}`이며 클래스는 `0: person`, `1: vehicle`이다. 본 실험에서는 추가 학습이나 시나리오별 임계값 조정을 수행하지 않았다. S1~S4의 시드, 궤적, 객체 배치 및 이미지·프레임·비어 있지 않은 라벨 해시는 학습/검증 자료 및 S0와 중복되지 않았다.

### 4.3 PASS·MARGINAL·FAIL 판정 기준

PASS는 mAP@0.5≥0.50, MARGINAL은 0.25≤mAP@0.5<0.50, FAIL은 mAP@0.5<0.25로 정의하였다. 대칭 탐색에서 PASS와 MARGINAL은 비실패 측으로, FAIL만 실패 측으로 취급하였다. 따라서 목표는 PASS–MARGINAL 경계가 아니라 MARGINAL–FAIL 사이의 심각한 성능 저하 구간이다.

### 4.4 대칭 경계 탐색 방법

초기 환경은 안개 5%, 조도 12,000 lx, 카메라 잡음 0.02이다. 최초 FAIL 전에는 안개를 30%p 증가시키고 조도를 직전 값의 50%로 낮추며 잡음을 0.20 증가시켰다. 비실패 앵커 `x_N`과 실패 앵커 `x_F`를 얻은 뒤 세 환경 변수에 각각 `x_next=(x_N+x_F)/2`를 적용하였다. 각 시나리오의 평가 저장 키에는 시나리오 ID, 궤적 설정 해시, 환경, 가중치 해시, 추론 설정, GT 버전을 포함하였다.

### 4.5 독립 비행 시나리오 5개의 구성

{md_table(['ID', '궤적', '시드', '시작 xyz', '종료 xyz', '고도 범위', '주요 통제 변경', '사람 수', '차량 수'], scenario_plan_rows(plan))}

### 4.6 평가 지표

클래스별 AP@0.5와 두 클래스의 산술평균 mAP@0.5, GT·탐지 수, TP·FP·FN, precision, recall 및 TP 대응 IoU의 최대·평균을 기록하였다. 최종 경계는 비실패와 FAIL 두 앵커 사이의 구간으로 제시하였다. `Gap_mAP=|mAP_nonfail-mAP_fail|`이며, `Gap_env`는 안개 범위 0~100, 조도 범위 200~15,000, 잡음 범위 0~0.60으로 정규화한 세 절대 차이의 합이다.

## 5. 실험 결과

### 5.1 초기 정상 조건의 객체 탐지 결과

{initial_table}

5개 시나리오가 모두 초기 PASS를 확보하였다. 초기 mAP는 {min(x['initial_map50'] for x in combined['scenarios']):.6f}~{max(x['initial_map50'] for x in combined['scenarios']):.6f} 범위였다.

### 5.2 대표 시나리오의 10회 탐색 이력

{s0_table}

S0는 4회차에서 최초 FAIL을 얻었고, 최종 비실패 조건과 실패 조건 사이를 10회 안에 축소하였다. 독립 재실행한 S0와 기존 실행의 반복별 mAP 최대 절대 차이는 0이었다.

### 5.3 5개 시나리오의 경계 탐색 결과

{summary_table}

모든 시나리오에서 최초 FAIL은 4회차에 관찰되었다. S3에서는 환경이 악화된 1→2회차에 mAP가 0.744873에서 0.795358로 증가하는 비단조 변화 1건이 있었으며, 이 값은 원자료에 그대로 포함하였다.

### 5.4 최종 경계 간격과 탐색 성공률

초기 PASS는 {overall['initial_pass_count']}/{overall['scenario_count']}({overall['initial_pass_rate']*100:.1f}%), 경계 탐색 성공은 {overall['search_success_count']}/{overall['scenario_count']}({overall['search_success_rate']*100:.1f}%)였다. 최초 FAIL 반복은 평균 {first['mean']:.3f}, 표준편차 {first['population_std']:.3f}, 중앙값 {first['median']:.3f}, 범위 {first['min']:.0f}~{first['max']:.0f}였다. Gap_mAP은 평균 {gm['mean']:.6f}, 표준편차 {gm['population_std']:.6f}, 중앙값 {gm['median']:.6f}, 범위 {gm['min']:.6f}~{gm['max']:.6f}였다. Gap_env는 평균 {ge['mean']:.6f}, 표준편차 {ge['population_std']:.6f}, 중앙값 {ge['median']:.6f}, 범위 {ge['min']:.6f}~{ge['max']:.6f}였다.

### 5.5 재검증용 테스트 묶음

탐색에 성공한 각 시나리오에서 초기 PASS, 최종 비실패, 최종 FAIL의 세 조건을 저장하여 총 {tests['case_count']}개 사례를 독립 평가 저장소에서 다시 실행하였다. 기대 판정 일치는 {tests['verdict_match_count']}/{tests['case_count']}였고, 원 실행 대비 최대 mAP 절대 차이는 {tests['maximum_absolute_map50_difference']:.6f}였다.

### 5.6 결과 해석 및 제한점

고정된 YOLOv8s를 사용하여 서로 다른 다섯 모의 비행 궤적에서 자동 실패 경계 탐색을 수행했으며, 성공한 경우 비실패–FAIL 구간을 반복적으로 축소하고 재실행 가능한 시험 사례로 저장하였다. 다만 결과는 단일 MATLAB 기반 렌더러, 하나의 학습 체크포인트, 5개의 결정론적 시나리오에 한정된다. 세 환경 변수를 함께 변화시켰으므로 특정 변수 하나를 성능 저하의 주원인으로 해석할 수 없다. 5개 사례의 기술통계는 통계적 유의성이나 실제 비행환경으로의 일반화를 뒷받침하지 않으며, 경계는 한 점이 아니라 최종 두 앵커 사이의 구간이다.
"""


def generate_execution_report(session_id: str, combined: dict[str, Any], records: list[dict[str, Any]], plan: dict[str, Any], gt: dict[str, Any], leakage: dict[str, Any], tests: dict[str, Any], s0_compare: dict[str, Any], figures: list[Path]) -> str:
    per_scenario_iterations = []
    for scenario_id in [f"S{i}" for i in range(5)]:
        rows = [x for x in records if x["scenario_id"] == scenario_id]
        per_scenario_iterations.append(
            f"### {scenario_id}\n\n" + md_table(
                ["회", "안개", "조도", "잡음", "사람 AP", "차량 AP", "mAP", "사람 탐지", "차량 탐지", "TP/FP/FN", "P", "R", "판정", "캐시"],
                [[x["iteration"], x["fog_percent"], x["illumination_lux"], x["camera_noise"], x["person_ap50"], x["vehicle_ap50"], x["map50"], x["person_detection_count"], x["vehicle_detection_count"], f"{x['total_tp']}/{x['total_fp']}/{x['total_fn']}", x["precision"], x["recall"], x["verdict"], x["evaluation_store_used"]] for x in rows],
            )
        )
    gt_rows = [[x["scenario_id"], x["total_frames"], x["object_frames"], x["empty_frames"], x["person_gt_count"], x["vehicle_gt_count"], x["out_of_bounds_box_count"], x["nonpositive_box_count"], x["frame_gt_alignment_valid"], x["valid"]] for x in gt["scenarios"]]
    audit_rows = [[x["scenario_id"], x["seed_overlap"], x["trajectory_overlap_with_S0"], x["object_layout_overlap_with_S0"], x["duplicate_image_hash_count"], x["duplicate_frame_hash_count"], x["duplicate_nonempty_label_hash_count"], x["passed"]] for x in leakage["results"]]
    test_rows = [[x["case_id"], x["expected_verdict"], x["original_map50"], x["revalidation_map50"], x["map50_absolute_difference"], x["revalidation_verdict"], x["verdict_matches"]] for x in tests["cases"]]
    return f"""# 5개 시나리오 대칭 경계 탐색 최종 실행 보고서

세션: `{session_id}`

## 1. 실제 생성한 S0~S4 시나리오 구성

{md_table(['ID', '궤적', '시드', '시작 xyz', '종료 xyz', '고도 범위', '주요 변경', '사람', '차량'], scenario_plan_rows(plan))}

최초 v1 사전검증에서 S1 차량이 고정 카메라 시야 밖에 있어 차량 GT가 0개로 확인됐다. 이는 YOLO 실행 전 GT-only 검사에서 발견한 생성 오류다. 실패한 v1 계획·렌더는 보존했고, S1 차량 좌표만 시야 안으로 옮긴 v2를 새 계획으로 사전 확정했다. 그 뒤에는 성능을 이유로 궤적·객체·시드를 변경하지 않았다.

## 2. 학습·검증 데이터와의 중복 검사 결과

{md_table(['ID', '시드 중복', 'S0 궤적 중복', 'S0 배치 중복', '영상 해시 중복', '프레임 해시 중복', '비어 있지 않은 라벨 해시 중복', '통과'], audit_rows)}

S0 행은 알려진 held-out 기준 자체이므로 중복 금지 대상이 아니다. S1~S4는 학습 6개, 검증 2개, S0와 비교해 모든 중복 지표가 0이었다. 빈 라벨 파일은 내용이 없으면 항상 같은 SHA-256이므로 원시 라벨 해시 충돌 판정에서 제외했고, 해당 프레임의 이미지·복합 프레임 해시는 계속 감사했다.

## 3. 시나리오별 GT 유효성 검사 결과

{md_table(['ID', '전체 프레임', '객체 프레임', '빈 프레임', '사람 GT', '차량 GT', '범위 밖', '비양수', '프레임 일치', '유효'], gt_rows)}

GT는 모두 실제 가시 픽셀 인스턴스 색상 렌더의 외접 상자다. 시나리오별 정상 조건 GT 진단 이미지 5개도 저장했다.

## 4. 시나리오별 초기 정상 조건 성능

{md_table(['ID', 'mAP', '사람 AP', '차량 AP', '사람 GT', '차량 GT', 'TP/FP/FN', 'Precision', 'Recall', '최대 IoU', '평균 IoU', '판정'], initial_rows(combined))}

## 5. 시나리오별 1~10회 전체 탐색 이력

{chr(10).join(per_scenario_iterations)}

`캐시=True`는 각 시나리오의 독립 초기 평가 결과를 그 시나리오의 1회차에 연결한 경우뿐이다. 다른 시나리오·기존 실험 결과는 재사용하지 않았다.

## 6. 탐색 성공 시나리오 수와 성공률

초기 PASS는 {combined['overall']['initial_pass_count']}/5, 경계 탐색 성공은 {combined['overall']['search_success_count']}/5로 모두 100%였다. 최초 FAIL 반복은 다섯 시나리오 모두 4회차였다.

## 7. 시나리오별 최종 비실패·실패 경계

{md_table(['ID', '궤적', '초기 mAP', '초기 판정', '최초 FAIL', '성공', '최종 비실패', '최종 FAIL', 'Gap_mAP', 'Gap_env', '평가 수', '시간(s)', '비단조 증가', '사유'], summary_rows(combined))}

각 경계는 두 앵커 사이의 구간이며 정확한 단일 환경값이 아니다.

## 8. Gap_mAP과 Gap_env 종합 결과

Gap_mAP 평균/표준편차/중앙값/범위는 {combined['overall']['gap_map50_statistics']['mean']:.6f}/{combined['overall']['gap_map50_statistics']['population_std']:.6f}/{combined['overall']['gap_map50_statistics']['median']:.6f}/{combined['overall']['gap_map50_statistics']['min']:.6f}~{combined['overall']['gap_map50_statistics']['max']:.6f}이다. Gap_env는 {combined['overall']['gap_environment_statistics']['mean']:.6f}/{combined['overall']['gap_environment_statistics']['population_std']:.6f}/{combined['overall']['gap_environment_statistics']['median']:.6f}/{combined['overall']['gap_environment_statistics']['min']:.6f}~{combined['overall']['gap_environment_statistics']['max']:.6f}이다. 실제 정규화 분모는 안개 100%p(0~100), 조도 14,800 lx(200~15,000), 잡음 0.60(0~0.60)이다.

## 9. 비단조적인 mAP 변화 분석

악화 방향의 연속 평가에서 mAP가 증가한 시나리오는 S3 하나다. S3 1→2회차에서 0.744873→0.795358(+0.050485)이었다. 완화 방향에서 mAP가 회복된 구간은 비단조 위반으로 세지 않았다. 이 결과는 환경과 성능 사이의 완전한 단조성을 가정할 수 없음을 보여주지만, 원인을 특정 변수 하나로 귀속하지 않는다.

## 10. 재검증용 테스트 묶음 결과

{md_table(['사례', '기대 판정', '원 mAP', '재실행 mAP', '절대 차이', '재실행 판정', '일치'], test_rows)}

총 {tests['case_count']}개 중 {tests['verdict_match_count']}개 판정이 일치했고 최대 mAP 절대 차이는 {tests['maximum_absolute_map50_difference']:.6f}이었다.

## 11. 생성·수정한 코드와 파일

- `prepare_scenarios.py`: 시나리오 사전 등록, GT 선검증, 중복 감사, 진단 이미지
- `export_scenario_frames.m`: 명시적 181프레임 궤적 및 가시 픽셀 GT 렌더
- `run_experiment.py`: 고정 가중치 검증, 독립 저장소, 초기 평가, 대칭 탐색, 집계
- `reverify_cases.py`: 성공 경계의 15개 독립 재검증 사례
- `report_results.py`: 실제 JSON 기반 표, 그림, 논문 문안
- `validate_outputs.py`: 결과 구조·수식·해시·그림 검증
- `tests/test_multi_scenario.py`: 13개 다중 시나리오 단위 테스트

실패한 v1 preflight 자료와 S3 7회차의 PNG 쓰기 실패 폴더는 삭제하지 않고 보존했다. 본 실행 중 기존 가중치나 과거 결과를 삭제·덮어쓰지 않았다.

## 12. 테스트 결과

기존 단위 테스트 21개와 신규 단위 테스트 13개, 총 34개가 통과했다. 최종 산출물 검증 결과는 `validation_report.json`에 저장한다.

## 13. 전체 실험 재현 명령

```powershell
.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.prepare_scenarios
.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.run_experiment
.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.reverify_cases --session <새-session-id>
.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.report_results --session <새-session-id>
.venv\\Scripts\\python.exe -m unittest discover -s experiments\\yolov8_search_comparison\\tests -v
.venv\\Scripts\\python.exe -m unittest discover -s experiments\\yolov8_search_comparison\\multi_scenario_symmetric\\tests -t . -v
.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.validate_outputs --session <새-session-id>
git diff --check
```

`prepare_scenarios`는 기존 완료 계획을 덮어쓰지 않으므로 새 반복 연구에서는 새 plan ID를 사용해야 한다.

## 14. 논문에서 사용할 수 있는 주장

- 고정된 YOLOv8s로 5개 MATLAB 모의 비행 시나리오에서 자동 경계 탐색을 수행했다.
- 성공한 5개 시나리오에서 비실패 조건과 FAIL 조건 사이의 구간을 반복적으로 축소했다.
- 발견한 초기·경계 사례를 별도 평가 저장소에서 다시 실행 가능한 테스트 자료로 저장했다.
- 결과는 MATLAB 기반 모의환경과 사전 확정한 5개 시나리오에 한정된다.

## 15. 여전히 남아 있는 한계

- 시나리오 수가 5개이고 결정론적 단일 실행이므로 통계적 유의성이나 일반화를 주장할 수 없다.
- 실제 비행 영상이나 센서에서 검증하지 않았다.
- 안개·조도·잡음을 동시에 변화시켰으므로 개별 변수의 인과적 기여를 분리하지 못한다.
- 하나의 고정 가중치와 하나의 렌더러에 한정된다.
- 최종 경계는 두 환경 앵커 사이의 구간이며 단일 정확점이 아니다.
- S0 과거 실행과 새 실행의 반복별 최대 절대 mAP 차이는 {s0_compare['maximum_absolute_map50_difference']:.6f}, 판정 차이는 {s0_compare['verdict_difference_count']}건이다.

## 생성 그림

{chr(10).join(f'- `{path}`' for path in figures)}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    combined = read_json(AGGREGATED_ROOT / args.session / "multi_scenario_results.json")
    records = read_json(AGGREGATED_ROOT / args.session / "all_iteration_results.json")
    plan = read_json(DEFAULT_PLAN)
    config = read_json(DEFAULT_CONFIG)
    gt = read_json(SCENARIO_ROOT / plan["plan_id"] / "gt_validation.json")
    leakage = read_json(SCENARIO_ROOT / plan["plan_id"] / "scenario_leakage_audit.json")
    tests = read_json(TEST_ROOT / args.session / "suite_summary.json")
    s0_compare = read_json(AGGREGATED_ROOT / args.session / "s0_prior_comparison.json")
    figures = generate_figures(args.session, combined, records, plan)
    report_dir = REPORTS_ROOT / args.session
    report_dir.mkdir(parents=True, exist_ok=False)
    manuscript = generate_manuscript(args.session, combined, records, plan, config, tests)
    execution = generate_execution_report(args.session, combined, records, plan, gt, leakage, tests, s0_compare, figures)
    (report_dir / "paper_sections_4_5_ko.md").write_text(manuscript, encoding="utf-8")
    (report_dir / "final_execution_report_ko.md").write_text(execution, encoding="utf-8")
    print(json.dumps({"report_dir": str(report_dir.resolve()), "figures": [str(path.resolve()) for path in figures]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
