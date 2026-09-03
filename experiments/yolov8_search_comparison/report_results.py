"""Create publication tables, figures, and Korean text from measured JSON."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _table_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "iteration": r["iteration"],
            "fog_percent": r["fog_percent"],
            "illumination_lux": r["illumination_lux"],
            "camera_noise": r["camera_noise"],
            "person_ap50": r["person_ap50"],
            "vehicle_ap50": r["vehicle_ap50"],
            "map50": r["map50"],
            "verdict": r["verdict"],
            "detected_intruder_count": r["detected_intruder_count"],
            "visible_intruder_count": r["visible_intruder_count"],
            "person_detection_count": r["person_detection_count"],
            "vehicle_detection_count": r["vehicle_detection_count"],
            "person_true_positive_count": r["person_true_positive_count"],
            "person_false_positive_count": r["person_false_positive_count"],
            "person_false_negative_count": r["person_false_negative_count"],
            "vehicle_true_positive_count": r["vehicle_true_positive_count"],
            "vehicle_false_positive_count": r["vehicle_false_positive_count"],
            "vehicle_false_negative_count": r["vehicle_false_negative_count"],
            "total_true_positive_count": r["total_true_positive_count"],
            "total_false_positive_count": r["total_false_positive_count"],
            "total_false_negative_count": r["total_false_negative_count"],
            "nonfail_anchor": r["nonfail_anchor"],
            "fail_anchor": r["fail_anchor"],
            "evaluation_cache_hit": r["evaluation_cache_hit"],
            "simulation_seconds": r["simulation_seconds"],
            "yolov8_inference_seconds": r["yolov8_inference_seconds"],
            "accounted_iteration_seconds": r["accounted_iteration_seconds"],
            "next_condition_calculation": r["next_condition_calculation"],
        }
        for r in records
    ]


def _plot_outputs(
    config: dict[str, Any], results: dict[str, Any], figures_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=False)
    colors = {"symmetric": "#3569b7", "asymmetric": "#d46a2f"}
    pass_threshold = float(config["thresholds"]["pass_map50"])
    fail_threshold = float(config["thresholds"]["fail_map50"])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method, result in results.items():
        records = result["records"]
        ax.plot(
            [r["iteration"] for r in records],
            [r["map50"] for r in records],
            marker="o",
            linewidth=2,
            label=f"{method} / {config['detector']}",
            color=colors.get(method),
        )
    ax.axhline(pass_threshold, color="#25814d", linestyle="--", label="PASS = 0.50")
    ax.axhline(fail_threshold, color="#a83232", linestyle="--", label="FAIL boundary = 0.25")
    ax.set(xlabel="Evaluation", ylabel="mAP@0.5", title="YOLOv8s mAP@0.5 by boundary-search evaluation")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "map50_by_iteration.png", dpi=180)
    plt.close(fig)

    environment_fields = (
        ("fog_percent", "Fog (%)", "fog_by_iteration.png", "Fog by boundary-search evaluation"),
        ("illumination_lux", "Illumination (lx)", "illumination_by_iteration.png", "Illumination by boundary-search evaluation"),
        ("camera_noise", "Camera noise", "camera_noise_by_iteration.png", "Camera noise by boundary-search evaluation"),
    )
    for field, ylabel, filename, title in environment_fields:
        fig, ax = plt.subplots(figsize=(8.2, 4.5))
        for method, result in results.items():
            records = result["records"]
            ax.plot(
                [r["iteration"] for r in records],
                [r[field] for r in records],
                marker="o",
                linewidth=2,
                label=method,
                color=colors.get(method),
            )
        ax.set(xlabel="Evaluation", ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plotted = False
    for method, result in results.items():
        records = result["records"]
        x = [r["iteration"] for r in records]
        nonfail = [
            math.nan if r["nonfail_anchor"] is None else r["nonfail_anchor"]["map50"]
            for r in records
        ]
        fail = [
            math.nan if r["fail_anchor"] is None else r["fail_anchor"]["map50"]
            for r in records
        ]
        ax.plot(x, nonfail, marker="o", color=colors.get(method), label=f"{method} non-FAIL anchor")
        ax.plot(x, fail, marker="x", linestyle="--", color=colors.get(method), label=f"{method} FAIL anchor")
        plotted = True
    ax.axhline(pass_threshold, color="#25814d", linestyle=":", label="PASS = 0.50")
    ax.axhline(fail_threshold, color="#a83232", linestyle=":", label="FAIL boundary = 0.25")
    ax.set(xlabel="Evaluation", ylabel="Anchor mAP@0.5", title="MARGINAL–FAIL boundary convergence")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "boundary_convergence.png", dpi=180)
    plt.close(fig)

    methods = list(results)
    summaries = [results[m]["summary"] for m in methods]
    for field, ylabel, filename, title in (
        ("gap_map50", "Final |mAP_nonfail - mAP_fail|", "gap_map50.png", "Final mAP boundary gap"),
        ("gap_environment_normalized", "Normalized environment gap", "gap_environment.png", "Final normalized environment boundary gap"),
    ):
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        values = [s[field] if s[field] is not None else 0.0 for s in summaries]
        bars = ax.bar(methods, values, color=[colors.get(m, "#777777") for m in methods])
        for bar, summary in zip(bars, summaries):
            label = "N/A" if summary[field] is None else f"{summary[field]:.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom")
        ax.set_ylabel(ylabel)
        ax.set_title(title + f" / {config['detector']}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=180)
        plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.4))
    first_fail = [s["first_fail_evaluation"] or 0 for s in summaries]
    runtime = [s["accounted_total_seconds"] for s in summaries]
    bars1 = ax1.bar(methods, first_fail, color=[colors.get(m, "#777777") for m in methods])
    bars2 = ax2.bar(methods, runtime, color=[colors.get(m, "#777777") for m in methods])
    ax1.set(title="First FAIL evaluation", ylabel="Evaluation count")
    ax2.set(title="Accounted execution cost", ylabel="Seconds")
    for ax, bars, vals in ((ax1, bars1, first_fail), (ax2, bars2, runtime)):
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Discovery and execution cost / {config['detector']}")
    fig.tight_layout()
    fig.savefig(figures_dir / "first_fail_and_runtime.png", dpi=180)
    plt.close(fig)

    anchor_rows: list[tuple[str, str, dict[str, Any]]] = []
    for method, result in results.items():
        summary = result["summary"]
        for side, key in (("N", "final_nonfail_anchor"), ("F", "final_fail_anchor")):
            anchor = summary.get(key)
            if anchor is not None:
                anchor_rows.append((method, side, anchor))
    if anchor_rows:
        labels = [f"{method}-{side}" for method, side, _ in anchor_rows]
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
        fields = (
            ("map50", "mAP@0.5"),
            ("fog_percent", "Fog (%)"),
            ("illumination_lux", "Illumination (lx)"),
            ("camera_noise", "Camera noise"),
        )
        bar_colors = [colors.get(method, "#777777") for method, _, _ in anchor_rows]
        for ax, (field, title) in zip(axes.flat, fields):
            values = [float(anchor[field]) for _, _, anchor in anchor_rows]
            bars = ax.bar(labels, values, color=bar_colors)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
            if field == "map50":
                ax.axhline(pass_threshold, color="#25814d", linestyle=":")
                ax.axhline(fail_threshold, color="#a83232", linestyle=":")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=8)
        fig.suptitle("Closest final non-FAIL (N) and FAIL (F) anchor cases")
        fig.tight_layout()
        fig.savefig(figures_dir / "final_anchor_cases.png", dpi=180)
        plt.close(fig)


def _paper_text(config: dict[str, Any], results: dict[str, Any]) -> str:
    model = next(iter(results.values()))["records"][0] if results and next(iter(results.values()))["records"] else {}
    lines = [
        "# KCI 논문용 추가 실험 결과 문안",
        "",
        "## 4. 실험 계획",
        "",
        f"본 실험은 `{config['scenario_id']}` 단일 최종 시험 시나리오에서 {model.get('weights_provenance', 'YOLOv8s')}를 이용하여 대칭 탐색과 제안 비대칭 탐색을 비교하였다. 초기 조건은 안개 5%, 조도 12,000 lx, 카메라 잡음 0.02이며 최대 평가 횟수는 10회로 제한하였다.",
        "",
        "현재 영상 생성, 카메라 잡음, 보행 궤적, 비행 경로와 객체 배치는 모두 결정적이므로 같은 조건을 30회 반복하지 않았다. 따라서 아래 결과는 단일 결정적 시나리오의 기술통계이며 평균, 표준편차, 유의확률을 제시하지 않는다.",
        "",
        "### 4.1 YOLOv8 평가 환경",
        "",
        f"가중치 파일은 `{model.get('model_weights_filename', 'N/A')}`이고 SHA-256은 `{model.get('model_weights_sha256', 'N/A')}`이다. 입력 영상 전체를 독립적으로 추론했으며 정답 박스는 AP@0.5 계산 단계에서만 사용하였다. 신뢰도 기준은 {config['yolov8']['confidence_threshold']}, NMS IoU 기준은 {config['yolov8']['nms_iou_threshold']}, AP IoU 기준은 {config['yolov8']['ap_iou_threshold']}이다. 체크포인트의 `model.names`를 기준으로 사람과 차량 클래스를 동적으로 매핑하였다.",
        "",
        "### 4.2 데이터와 GT 구성",
        "",
        "학습·검증 데이터는 프레임이 아니라 전체 시나리오 단위로 분리하였고, 최종 시험의 variant 0 및 초기 정상 조건 181프레임은 학습과 모델 선택에서 제외하였다. GT는 MATLAB 3차원 렌더러의 z-buffer가 반영된 인스턴스 색상 패스에서 실제 가시 픽셀의 외접 사각형으로 산출하였다.",
        "",
        "### 4.3 PASS·MARGINAL·FAIL 기준",
        "",
        "mAP@0.5가 0.50 이상이면 PASS, 0.25 이상 0.50 미만이면 MARGINAL, 0.25 미만이면 FAIL로 판정하였다. PASS와 MARGINAL은 비실패 측, FAIL만 실패 측으로 두고 MARGINAL–FAIL 경계를 탐색하였다.",
        "",
        "### 4.4 대칭·비대칭 탐색 설정과 비교 지표",
        "",
        "두 방법은 최초 FAIL 전까지 안개 +30%p, 조도 50% 감소, 잡음 +0.20의 동일한 악화 규칙을 사용하였다. 최초 FAIL 이후 대칭 탐색은 비실패·실패 앵커의 50:50 중간값을 선택하였다. 비대칭 탐색은 비실패에서 실패로 탐색할 때 실패 앵커에 65%, FAIL에서 회복할 때 비실패 앵커에 75%의 가중치를 적용하였다. 탐지기, 가중치, 영상, 비행 경로, 객체 배치, 임계값, 최대 횟수 및 시드는 동일하게 통제하였다.",
        "비교 지표는 최종 두 앵커의 절대 mAP 차이인 Gap_mAP와 환경 차이를 설정 범위로 정규화해 합산한 Gap_env이다. 정규화 분모는 안개 100%p(0–100), 조도 14,800 lx(200–15,000), 잡음 0.60(0–0.60)이다.",
        "",
        "## 5. 실험 결과",
        "",
        f"### 5.1 초기 정상 조건 검증\n\n초기 조건의 mAP@0.5는 {model.get('map50', float('nan')):.4f}로 {model.get('verdict', 'N/A')}였으며, 이 PASS 확인 후에만 경계 탐색을 수행하였다.",
        "",
        "### 5.2 대칭·비대칭 탐색 이력",
        "",
    ]
    for method, result in results.items():
        summary = result["summary"]
        records = result["records"]
        first = records[0]
        final = records[-1]
        lines.append(
            f"- {method}: {summary['evaluation_count']}회 평가했으며 최초 평가는 mAP@0.5={first['map50']:.4f}({first['verdict']}), 마지막 평가는 mAP@0.5={final['map50']:.4f}({final['verdict']})였다. 사람 탐지는 {first['person_detection_count']}건(TP {first['person_true_positive_count']}건), 차량 탐지는 {first['vehicle_detection_count']}건(TP {first['vehicle_true_positive_count']}건)이었고 동일 클래스 GT와의 최대 IoU는 {_fmt(first['maximum_same_class_iou'], 5)}였다. 최초 FAIL 발견 횟수는 {summary['first_fail_evaluation'] if summary['first_fail_evaluation'] is not None else '발견 못함'}이고, 경계 확보 성공 여부는 {summary['boundary_success']}이다. Gap_mAP={_fmt(summary['gap_map50'])}, Gap_env={_fmt(summary['gap_environment_normalized'])}, 비용 환산 실행시간={summary['accounted_total_seconds']:.3f}초였다. 중단 사유는 `{summary['stop_reason']}`이다."
        )
    lines.extend(["", "### 5.3 두 방법의 비교", ""])
    successful = [m for m, r in results.items() if r["summary"]["boundary_success"]]
    if len(successful) == len(results) and len(results) >= 2:
        sym = results.get("symmetric", {}).get("summary")
        asym = results.get("asymmetric", {}).get("summary")
        if sym and asym:
            map_winner = "비대칭" if asym["gap_map50"] < sym["gap_map50"] else "대칭" if sym["gap_map50"] < asym["gap_map50"] else "동일"
            env_winner = "비대칭" if asym["gap_environment_normalized"] < sym["gap_environment_normalized"] else "대칭" if sym["gap_environment_normalized"] < asym["gap_environment_normalized"] else "동일"
            lines.append(f"단일 결정적 시나리오에서 mAP 경계 차이는 {map_winner} 탐색이 더 작았고, 정규화 환경 경계 차이는 {env_winner} 탐색이 더 작았다. 이 결과는 해당 시나리오에서의 수렴 양상만 보여 주며 모집단 수준의 우월성을 뜻하지 않는다.")
    else:
        lines.append("하나 이상의 탐색에서 비실패 사례와 FAIL 사례를 모두 확보하지 못했으므로 두 규칙의 경계 수렴 성능을 비교하거나 비대칭 탐색의 우월성을 주장할 수 없다. 특히 초기 조건 자체가 FAIL이면 탐색 전제인 비실패 시작 앵커가 성립하지 않는다.")
    lines.extend(
        [
            "",
            "### 5.4 현재 결과로 주장할 수 있는 범위",
            "",
            "측정된 결과는 지정된 가중치, 현재의 렌더링 장면, 객체 배치, 궤적과 단일 시드에 한정된다. 구현된 실행 경로가 전체 영상 YOLO 추론과 클래스별 AP@0.5, 그리고 두 결정적 탐색 규칙을 재현 가능하게 기록한다는 점은 확인할 수 있다. 단일 결정적 실행으로 통계적 유의성이나 일반적 우월성은 주장하지 않는다.",
            "",
            "### 5.5 결과의 한계",
            "",
            "학습과 최종 평가는 MATLAB 합성 도메인에 한정되며 실제 항공 영상으로의 일반화는 확인하지 않았다. 또한 탐색 비교는 단일 결정적 시험 시나리오이므로 통계적 유의성이나 모집단 수준의 우월성을 주장할 수 없다.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_outputs(
    config: dict[str, Any],
    results: dict[str, Any],
    aggregate_dir: Path,
    figures_dir: Path,
) -> None:
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for method, result in results.items():
        rows = _table_rows(result["records"])
        _write_csv(aggregate_dir / f"{method}_iterations.csv", rows)
        all_rows.extend({"search_method": method, **row} for row in rows)
        s = result["summary"]
        summary_rows.append(
            {
                "search_method": method,
                "detector": s["detector"],
                "scenario_id": s["scenario_id"],
                "random_seed": s["random_seed"],
                "evaluation_count": s["evaluation_count"],
                "logical_evaluation_count": s["logical_evaluation_count"],
                "actual_new_evaluation_count": s["actual_new_evaluation_count"],
                "cache_hit_count": s["cache_hit_count"],
                "first_fail_evaluation": s["first_fail_evaluation"],
                "boundary_success": s["boundary_success"],
                "gap_map50": s["gap_map50"],
                "gap_environment_normalized": s["gap_environment_normalized"],
                "simulation_total_seconds": s["simulation_total_seconds"],
                "yolov8_inference_total_seconds": s["yolov8_inference_total_seconds"],
                "next_scenario_total_seconds": s["next_scenario_total_seconds"],
                "accounted_total_seconds": s["accounted_total_seconds"],
                "driver_wall_seconds": s["driver_wall_seconds"],
                "stop_reason": s["stop_reason"],
            }
        )
    _write_csv(aggregate_dir / "all_iteration_results.csv", all_rows)
    _write_csv(aggregate_dir / "comparison_summary.csv", summary_rows)
    _plot_outputs(config, results, figures_dir)
    (aggregate_dir / "paper_results_ko.md").write_text(
        _paper_text(config, results), encoding="utf-8"
    )
