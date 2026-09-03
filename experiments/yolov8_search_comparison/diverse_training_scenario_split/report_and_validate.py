"""Create paper artifacts and execute the final evidence-backed validation suite."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from ..search_core import Environment, classify_verdict, environment_gap, map_gap
from .plan import CONFIG_ROOT, DATA_ROOT, FIXED_EVAL_ROOT, FIXED_PLAN, FIXED_SCENARIOS, FIXED_SESSION, HERE, OLD_BEST, OLD_TRAINING_ROOT, PLAN_ID, REPO_ROOT, SELECTED_FRAMES, file_sha256


EVALUATION_ROOT = HERE / "evaluation"
OLD_RESULTS = FIXED_EVAL_ROOT / "aggregated" / FIXED_SESSION / "multi_scenario_results.json"
OLD_RECORDS = FIXED_EVAL_ROOT / "aggregated" / FIXED_SESSION / "all_iteration_results.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_new(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_json_new(path: Path, value: Any) -> None:
    write_text_new(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


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


def load_training_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key.strip(): float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def generate_figures(session: str, training: dict[str, Any], combined: dict[str, Any], records: list[dict[str, Any]]) -> list[Path]:
    figure_root = HERE / "figures" / session
    figure_root.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []

    train_rows = load_training_rows(Path(training["best_weights"]).parents[1] / "results.csv")
    epochs = [row["epoch"] for row in train_rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(epochs, [x["train/box_loss"] for x in train_rows], label="train")
    axes[0, 0].plot(epochs, [x["val/box_loss"] for x in train_rows], label="validation")
    axes[0, 0].set_title("Box loss"); axes[0, 0].legend()
    axes[0, 1].plot(epochs, [x["train/cls_loss"] for x in train_rows], label="train")
    axes[0, 1].plot(epochs, [x["val/cls_loss"] for x in train_rows], label="validation")
    axes[0, 1].set_title("Classification loss"); axes[0, 1].legend()
    axes[1, 0].plot(epochs, [x["metrics/mAP50(B)"] for x in train_rows], label="mAP@0.5")
    axes[1, 0].plot(epochs, [x["metrics/mAP50-95(B)"] for x in train_rows], label="mAP@0.5:0.95")
    axes[1, 0].set_ylim(0, 1.02); axes[1, 0].set_title("Validation AP"); axes[1, 0].legend()
    axes[1, 1].plot(epochs, [x["metrics/precision(B)"] for x in train_rows], label="precision")
    axes[1, 1].plot(epochs, [x["metrics/recall(B)"] for x in train_rows], label="recall")
    axes[1, 1].set_ylim(0, 1.02); axes[1, 1].set_title("Validation precision/recall"); axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.25)
    fig.suptitle("Diverse scenario training curves")
    fig.tight_layout()
    path = figure_root / "01_training_curves.png"; fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for scenario_id in [f"S{i}" for i in range(5)]:
        items = [x for x in records if x["scenario_id"] == scenario_id]
        ax.plot([x["iteration"] for x in items], [x["map50"] for x in items], marker="o", label=scenario_id)
    ax.axhline(0.5, color="green", linestyle="--", label="PASS threshold")
    ax.axhline(0.25, color="red", linestyle="--", label="FAIL threshold")
    ax.set(xlabel="Search iteration", ylabel="mAP@0.5", title="S0-S4 symmetric boundary-search trajectories", xticks=range(1, 11), ylim=(0, 1.03))
    ax.grid(alpha=0.25); ax.legend(ncol=4)
    fig.tight_layout()
    path = figure_root / "02_map_search_trajectories.png"; fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig); paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fields = [("fog_percent", "Fog (%)"), ("illumination_lux", "Illumination (lx)"), ("camera_noise", "Camera noise")]
    for scenario_id in [f"S{i}" for i in range(5)]:
        items = [x for x in records if x["scenario_id"] == scenario_id]
        for ax, (field, label) in zip(axes, fields):
            ax.plot([x["iteration"] for x in items], [x[field] for x in items], marker=".", label=scenario_id)
            ax.set_ylabel(label); ax.grid(alpha=0.25)
    axes[0].legend(ncol=5); axes[-1].set_xlabel("Search iteration"); axes[-1].set_xticks(range(1, 11))
    fig.suptitle("Symmetric environment-condition sequence")
    fig.tight_layout()
    path = figure_root / "03_environment_search_trajectories.png"; fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig); paths.append(path)

    def evaluation_for(scenario_id: str, key: str) -> dict[str, Any]:
        return read_json(EVALUATION_ROOT / "runs" / session / scenario_id / "evaluations" / key / "evaluation.json")

    def representative(evaluation: dict[str, Any]) -> dict[str, Any]:
        frames = evaluation["frames"]
        both = [frame for frame in frames if {x["class_name"] for x in frame["ground_truth"]} == {"person", "vehicle"}]
        return min(both or frames, key=lambda x: abs(int(x["frame_index"]) - 91))

    fig, axes = plt.subplots(5, 2, figsize=(12, 15))
    for row, scenario in enumerate(combined["scenarios"]):
        for col, (kind, anchor) in enumerate((("non-failure", scenario["final_nonfail_anchor"]), ("FAIL", scenario["final_fail_anchor"]))):
            frame = representative(evaluation_for(scenario["scenario_id"], anchor["cache_key"]))
            with Image.open(frame["image_path"]).convert("RGB") as image:
                axes[row, col].imshow(image)
            for gt in frame["ground_truth"]:
                x, y, w, h = gt["bbox_xywh"]
                axes[row, col].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1.2))
            for det in frame["detections"]:
                x, y, w, h = det["bbox_xywh"]
                color = "red" if det["class_name"] == "person" else "deepskyblue"
                axes[row, col].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.0, linestyle=":"))
            axes[row, col].set_title(f"{scenario['scenario_id']} {kind}: mAP={anchor['map50']:.3f}\nfog={anchor['fog_percent']:.2f}, lux={anchor['illumination_lux']:.1f}, noise={anchor['camera_noise']:.4f}")
            axes[row, col].axis("off")
    fig.suptitle("Final non-failure (left) and FAIL (right) boundary examples\nGT=green, person detection=red, vehicle detection=blue")
    fig.tight_layout()
    path = figure_root / "04_final_nonfail_fail_examples.png"; fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig); paths.append(path)
    return paths


def old_internal_validation() -> dict[str, Any]:
    import torch

    checkpoint = torch.load(OLD_BEST, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("train_metrics", {})
    return {
        "map50": float(metrics.get("metrics/mAP50(B)")),
        "map50_95": float(metrics.get("metrics/mAP50-95(B)")),
        "precision": float(metrics.get("metrics/precision(B)")),
        "recall": float(metrics.get("metrics/recall(B)")),
    }


def boundary_text(item: dict[str, Any]) -> str:
    left, right = item["final_nonfail_anchor"], item["final_fail_anchor"]
    if not left or not right:
        return "미확보"
    return f"NF({left['fog_percent']:.2f}%, {left['illumination_lux']:.1f} lx, {left['camera_noise']:.4f}, {left['map50']:.4f})–F({right['fog_percent']:.2f}%, {right['illumination_lux']:.1f} lx, {right['camera_noise']:.4f}, {right['map50']:.4f})"


def comparison_rows(training: dict[str, Any], old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    old_val = old_internal_validation()
    new_val = training["validation_metrics_recomputed_from_best"]
    rows = []
    for label, train_scenarios, train_images, val_scenarios, val_map, result in (
        ("기존 모델", 6, 366, 2, old_val["map50"], old),
        ("새 다양화 모델", 18, 360, 5, new_val["map50"], new),
    ):
        scenarios = result["scenarios"]
        rows.append(
            {
                "model": label,
                "training_scenario_count": train_scenarios,
                "training_image_count": train_images,
                "mean_images_per_training_scenario": train_images / train_scenarios,
                "validation_scenario_count": val_scenarios,
                "internal_validation_map50": val_map,
                "S0_S4_initial_pass_count": result["overall"]["initial_pass_count"],
                "S0_S4_initial_map50_min": min(x["initial_map50"] for x in scenarios),
                "S0_S4_initial_map50_max": max(x["initial_map50"] for x in scenarios),
                "boundary_search_success_count": result["overall"]["search_success_count"],
                "mean_gap_map50": result["overall"]["gap_map50_statistics"]["mean"],
                "final_boundary_intervals": {x["scenario_id"]: boundary_text(x) for x in scenarios},
                "nonmonotonic_increase_count": sum(x["nonmonotonic_map_increase_count"] for x in scenarios),
            }
        )
    return rows


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: Any) -> None:
        self.items.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(x["passed"] for x in self.items)


def run_unit_tests() -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "experiments/yolov8_search_comparison/tests", "-v"],
        [sys.executable, "-m", "unittest", "discover", "-s", "experiments/yolov8_search_comparison/multi_scenario_symmetric/tests", "-t", ".", "-v"],
        [sys.executable, "-m", "unittest", "discover", "-s", "experiments/yolov8_search_comparison/diverse_training_scenario_split/tests", "-t", ".", "-v"],
    ]
    outputs, count, ok = [], 0, True
    for command in commands:
        completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        output = completed.stdout + completed.stderr
        outputs.append(output)
        ok = ok and completed.returncode == 0
        for line in output.splitlines():
            if line.startswith("Ran ") and " tests" in line:
                count += int(line.split()[1])
    return {"passed": ok, "test_count": count, "output": "\n".join(outputs)}


def validate(session: str, training: dict[str, Any], config: dict[str, Any], combined: dict[str, Any], records: list[dict[str, Any]], figures: list[Path], report_dir: Path) -> dict[str, Any]:
    checks = Checks()
    plan = read_json(CONFIG_ROOT / "scenario_plan.json")
    dataset = read_json(DATA_ROOT / "dataset_summary.json")
    gt = read_json(DATA_ROOT / "ground_truth_validation.json")
    split = read_json(DATA_ROOT / "scenario_split.json")
    exact = read_json(DATA_ROOT / "exact_duplicate_audit.json")
    near = read_json(DATA_ROOT / "near_duplicate_audit.json")
    preservation = read_json(DATA_ROOT / "existing_results_preservation_check.json")
    gate = read_json(EVALUATION_ROOT / "aggregated" / session / "initial_pass_gate.json")

    checks.add("scenario_level_split_18_train_5_validation", len(split["train_scenarios"]) == 18 and len(split["validation_scenarios"]) == 5 and not set(split["train_scenarios"]) & set(split["validation_scenarios"]), split)
    checks.add("dataset_counts_360_train_100_validation", dataset["train_image_count"] == 360 and dataset["validation_image_count"] == 100, {"train": dataset["train_image_count"], "validation": dataset["validation_image_count"]})
    checks.add("uniform_20_frame_selection", all(x["selected_frame_indices"] == SELECTED_FRAMES and x["selected_frame_count"] == 20 for x in gt["scenarios"]), SELECTED_FRAMES)
    checks.add("both_classes_in_every_scenario", all(x["person_gt_count"] > 0 and x["vehicle_gt_count"] > 0 and x["both_class_frame_count"] >= 3 for x in gt["scenarios"]), [(x["scenario_id"], x["person_gt_count"], x["vehicle_gt_count"], x["both_class_frame_count"]) for x in gt["scenarios"]])
    checks.add("ground_truth_coordinate_class_alignment", gt["all_valid"] and all(x["out_of_bounds_or_nonpositive_box_count"] == 0 and x["invalid_class_count"] == 0 and x["frame_label_alignment_error_count"] == 0 for x in gt["scenarios"]), "all visible-pixel boxes and frame-label pairs valid")
    checks.add("five_diagnostics_per_scenario", all(x["diagnostic_image_count"] == 5 for x in gt["scenarios"]), dataset["diagnostic_image_count"])
    checks.add("train_validation_test_identity_separation", exact["identity_collision_count"] == 0, exact["identity_collisions"])
    checks.add("exact_content_duplicate_audit", exact["passed"], {"images": len(exact["image_collisions"]), "frames": len(exact["frame_pair_collisions"]), "nonempty_labels": len(exact["nonempty_label_collisions"])})
    checks.add("near_duplicate_audit_and_structural_review", near["review_passed"], [(x["comparison"], x["minimum_hamming_distance"], x["warning_pair_count"]) for x in near["comparisons"]])
    checks.add("pre_registered_hashes_unchanged", file_sha256(CONFIG_ROOT / "scenario_plan.json") == read_json(CONFIG_ROOT / "pre_registration.json")["scenario_plan_sha256"] and all(file_sha256(Path(x["scenario_config_path"])) == x["scenario_config_sha256"] for x in plan["scenarios"]), file_sha256(CONFIG_ROOT / "scenario_plan.json"))
    checks.add("new_weight_hash", file_sha256(Path(training["best_weights"])) == training["best_weights_sha256"] == config["expected_weights_sha256"], training["best_weights_sha256"])
    checks.add("validation_only_weight_selection", training["S0_S4_accessed_for_training_or_selection"] is False, training["validation_metrics_recomputed_from_best"])
    checks.add("global_initial_pass_gate", gate["all_initial_pass"] and gate["pass_count"] == 5 and gate["search_started"], {"pass_count": gate["pass_count"], "search_started": gate["search_started"]})
    checks.add("fifty_search_records", len(records) == 50 and all(sum(x["scenario_id"] == f"S{i}" for x in records) == 10 for i in range(5)), len(records))
    checks.add("fixed_threshold_verdicts", all(classify_verdict(float(x["map50"]), 0.5, 0.25) == x["verdict"] for x in records), "PASS>=0.50; MARGINAL>=0.25; FAIL<0.25")
    checks.add("symmetric_midpoint_policy", all("common_pre_fail" in x["next_condition_calculation"] or "symmetric_midpoint" in x["next_condition_calculation"] for x in records), "all 50 policy labels")
    checks.add("ten_unique_new_weight_evaluations_per_scenario", all(len(list((EVALUATION_ROOT / "runs" / session / f"S{i}" / "evaluations").glob("*/evaluation.json"))) == 10 for i in range(5)), "10 content-addressed evaluation files per scenario")
    old_keys = {x["evaluation_cache_key"] for x in read_json(OLD_RECORDS)}
    new_keys = {x["evaluation_cache_key"] for x in records}
    checks.add("evaluation_cache_separated_from_old_weight", not old_keys & new_keys and all(x["weights_sha256"] == training["best_weights_sha256"] for x in records), {"old_keys": len(old_keys), "new_keys": len(new_keys), "intersection": len(old_keys & new_keys)})
    gap_ok = True
    gap_details = []
    for item in combined["scenarios"]:
        left, right = item["final_nonfail_anchor"], item["final_fail_anchor"]
        if not left or not right:
            gap_ok = False
            continue
        map_value = map_gap(left["map50"], right["map50"])
        env_value = environment_gap(Environment.from_dict(left), Environment.from_dict(right), config["environment_bounds"])
        passed = abs(map_value - item["gap_map50"]) < 1e-12 and abs(env_value - item["gap_environment_normalized"]) < 1e-12 and left["verdict"] != "FAIL" and right["verdict"] == "FAIL"
        gap_ok = gap_ok and passed
        gap_details.append({"scenario": item["scenario_id"], "gap_map50": map_value, "gap_env": env_value, "passed": passed})
    checks.add("gap_formulas_and_boundary_sides", gap_ok, gap_details)
    checks.add("population_standard_deviation_explicit", "population_std" in combined["overall"]["gap_map50_statistics"] and "population_std" in combined["overall"]["gap_environment_statistics"], combined["overall"]["gap_map50_statistics"])
    checks.add("all_existing_results_preserved", preservation["all_unchanged"], preservation)
    checks.add("required_figures_300dpi", len(figures) == 4 and all(min(Image.open(path).info.get("dpi", (0, 0))) >= 299 for path in figures), [str(x) for x in figures])

    json_errors, csv_errors = [], []
    for path in HERE.rglob("*.json"):
        try:
            read_json(path)
        except Exception as exc:
            json_errors.append(f"{path}: {exc}")
    for path in HERE.rglob("*.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                list(csv.reader(handle))
        except Exception as exc:
            csv_errors.append(f"{path}: {exc}")
    checks.add("json_csv_parse", not json_errors and not csv_errors, {"json_errors": json_errors, "csv_errors": csv_errors})
    unit = run_unit_tests()
    checks.add("unit_tests", unit["passed"], {"test_count": unit["test_count"]})
    diff = subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    checks.add("git_diff_check", diff.returncode == 0, diff.stdout + diff.stderr)
    required = ["final_execution_report_ko.md", "training_report_ko.md", "paper_sections_4_5_ko.md", "limitations_and_claims_ko.md", "README.md"]
    checks.add("required_documents_created", all((report_dir / x).is_file() for x in required), required)
    report = {
        "session_id": session,
        "passed": checks.passed,
        "passed_count": sum(x["passed"] for x in checks.items),
        "failed_count": sum(not x["passed"] for x in checks.items),
        "unit_test_count": unit["test_count"],
        "checks": checks.items,
    }
    test_root = HERE / "tests" / "results" / session
    test_root.mkdir(parents=True, exist_ok=False)
    write_json_new(test_root / "test_results.json", report)
    write_csv_new(test_root / "test_results.csv", checks.items)
    write_text_new(test_root / "test_results.txt", unit["output"] + "\n\nGIT DIFF --CHECK\n" + diff.stdout + diff.stderr)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    args = parser.parse_args()
    session = args.session
    training = read_json(args.training_summary.resolve())
    config = read_json(args.evaluation_config.resolve())
    aggregate = EVALUATION_ROOT / "aggregated" / session
    combined = read_json(aggregate / "multi_scenario_results.json")
    records = read_json(aggregate / "all_iteration_results.json")
    old = read_json(OLD_RESULTS)
    dataset = read_json(DATA_ROOT / "dataset_summary.json")
    exact = read_json(DATA_ROOT / "exact_duplicate_audit.json")
    near = read_json(DATA_ROOT / "near_duplicate_audit.json")
    comparison = comparison_rows(training, old, combined)
    write_csv_new(aggregate / "old_new_model_comparison.csv", comparison)
    write_json_new(aggregate / "old_new_model_comparison.json", comparison)
    boundary_rows = [
        {
            "scenario_id": item["scenario_id"],
            "initial_map50": item["initial_map50"],
            "initial_verdict": item["initial_verdict"],
            "first_fail_iteration": item["first_fail_iteration"],
            "final_nonfail": item["final_nonfail_anchor"],
            "final_fail": item["final_fail_anchor"],
            "gap_map50": item["gap_map50"],
            "gap_env": item["gap_environment_normalized"],
            "nonmonotonic_increase_count": item["nonmonotonic_map_increase_count"],
        }
        for item in combined["scenarios"]
    ]
    write_csv_new(aggregate / "final_boundary_table.csv", boundary_rows)
    figures = generate_figures(session, training, combined, records)
    report_dir = HERE / "reports" / session
    report_dir.mkdir(parents=True, exist_ok=False)

    initial_table = md_table(
        ["시나리오", "사람 AP", "차량 AP", "mAP", "TP/FP/FN", "정밀도", "재현율", "최대 IoU", "TP 평균 IoU", "판정"],
        [[x["scenario_id"], x["person_ap50"], x["vehicle_ap50"], x["map50"], f"{x['total_tp']}/{x['total_fp']}/{x['total_fn']}", x["precision"], x["recall"], x["maximum_iou"], x["mean_iou"], x["verdict"]] for x in combined["initial_evaluations"]],
    )
    boundary_table = md_table(
        ["시나리오", "최초 FAIL", "최종 비실패", "최종 FAIL", "Gap_mAP", "Gap_env", "비단조 증가"],
        [[x["scenario_id"], x["first_fail_iteration"], boundary_text(x).split("–F(")[0], "F(" + boundary_text(x).split("–F(")[1] if "–F(" in boundary_text(x) else "-", x["gap_map50"], x["gap_environment_normalized"], x["nonmonotonic_map_increase_count"]] for x in combined["scenarios"]],
    )
    comp_table = md_table(
        ["모델", "학습 시나리오", "학습 영상", "시나리오당", "검증 시나리오", "내부 검증 mAP", "초기 PASS", "초기 mAP 범위", "탐색 성공", "평균 Gap_mAP", "비단조 증가"],
        [[x["model"], x["training_scenario_count"], x["training_image_count"], x["mean_images_per_training_scenario"], x["validation_scenario_count"], x["internal_validation_map50"], x["S0_S4_initial_pass_count"], f"{x['S0_S4_initial_map50_min']:.6f}–{x['S0_S4_initial_map50_max']:.6f}", x["boundary_search_success_count"], x["mean_gap_map50"], x["nonmonotonic_increase_count"]] for x in comparison],
    )
    val = training["validation_metrics_recomputed_from_best"]
    hw = training["hardware"]
    usage = training["resource_usage"]
    execution = f"""# 최종 실행 보고서

## 1. 기존 자료 감사 결과

기존 자료는 학습 6개 시나리오 366장, 검증 2개 시나리오 122장이며 181프레임에서 매 3프레임(0.3초) 간격으로 61장을 추출했다. 분할은 시나리오 단위였다. 기존 설정은 YOLOv8s, 640 px, batch 4, SGD, 최대 50 epoch, patience 12, seed 42이다. 기존 `best.pt`의 SHA-256은 `{file_sha256(OLD_BEST)}`이다. 기존 S0~S4 계획·렌더·GT·결과의 핵심 해시는 작업 전후 모두 일치했다.

## 2. 새 학습·검증 시나리오 구성

`{PLAN_ID}`로 학습 18개와 검증 5개 시나리오를 YOLO 실행 전에 확정했다. 각 18초/181프레임에서 {SELECTED_FRAMES}의 20개 시점만 사용하여 360장과 100장을 구성했다.

최초 사전등록 `diverse_training_scenario_split_v1`은 9개 시나리오가 카메라 시야 밖에 오래 머물러 가시 객체·진단 영상 요건을 충족하지 못했다. 품질 게이트가 YOLO 실행 전에 학습을 차단했으며, v1 계획·460장·감사 결과는 그대로 보존했다. 이후 분할·시드·환경·추출 프레임은 유지하고 해당 시나리오의 카메라-객체 거리 및 종점만 시야 내로 조정하여 별도 v2 계획을 사전등록했다.

## 3. 다양화 요소와 미지원 요소

동서·남북·대각선, 상승·하강, 접근·이탈, 시작/종료 위치, 고도, 객체 배치·간격, 관측 거리·영상 크기, 근접 배치에 의한 부분 가림, 보행 위상, 정상 범위 환경(안개 0~20%, 조도 8,000~15,000 lx, 잡음 0~0.05)을 변화시켰다. 카메라 방향·기울기는 현 렌더러의 `+x`/하향 60° 고정 구조 때문에 바꾸지 않았고, 지형·배경 역시 단일 결정론적 배치만 지원하여 다양화하지 않았다.

## 4. 자료 수와 클래스별 정답 수

- 학습: 360장, 사람 GT {dataset['train_person_gt_count']}개, 차량 GT {dataset['train_vehicle_gt_count']}개
- 검증: 100장, 사람 GT {dataset['validation_person_gt_count']}개, 차량 GT {dataset['validation_vehicle_gt_count']}개
- 빈 프레임: {dataset['empty_frame_count']}장, 진단 이미지: {dataset['diagnostic_image_count']}장

## 5. 분리 및 중복 감사

시나리오 ID·시드·궤적·객체 배치 및 영상/프레임/비어 있지 않은 라벨의 정확 중복은 모두 0건으로 통과했다. dHash 경고 기준은 해밍 거리 ≤4이며, 비교별 최소 거리와 경고 수는 {[(x['comparison'], x['minimum_hamming_distance'], x['warning_pair_count']) for x in near['comparisons']]}이다. 경고 쌍은 정확 SHA가 다르고 구조적 시나리오 식별자가 모두 달라 검토 통과로 기록했다.

## 6. 새 YOLOv8s 학습

{training['training_epochs_completed']} epoch에서 종료되었고 선택 epoch는 {training['best_epoch_one_based']}이다. 내부 검증은 사람 AP@0.5 {val['person_ap50']:.6f}, 차량 AP@0.5 {val['vehicle_ap50']:.6f}, mAP@0.5 {val['map50']:.6f}, mAP@0.5:0.95 {val['map50_95']:.6f}, 정밀도 {val['mean_precision']:.6f}, 재현율 {val['mean_recall']:.6f}였다. 학습 시간은 {training['training_seconds']:.3f}초다. `best.pt`는 {training['best_weights_size_bytes']} bytes, SHA-256 `{training['best_weights_sha256']}`이다. 호스트는 {hw['cpu']['name']} ({hw['cpu']['physical_cores']}코어/{hw['cpu']['logical_processors']}스레드), RAM {hw['system_memory']['total_gib']:.2f} GiB, {hw['gpu'].get('name')} {hw['gpu'].get('memory_total_mib')} MiB이다. 학습 중 최대 프로세스 RSS는 {usage['peak_process_rss_bytes'] / 2**30:.2f} GiB, 최대 시스템 메모리 사용량은 {usage['peak_system_memory_used_bytes'] / 2**30:.2f} GiB, PyTorch 최대 GPU 할당/예약은 각각 {usage['torch_peak_gpu_allocated_bytes'] / 2**20:.1f}/{usage['torch_peak_gpu_reserved_bytes'] / 2**20:.1f} MiB였다.

## 7. S0~S4 초기 정상 조건

{initial_table}

전역 초기 게이트는 {combined['overall']['initial_pass_count']}/5 PASS였다. 다섯 초기 추론이 완료된 뒤에만 탐색을 시작했다.

## 8. 전체 50회 탐색

5개 시나리오 각각 10회, 총 {combined['overall']['total_evaluation_count']}개 기록을 새 가중치·새 캐시 공간에서 계산했다. 최초 FAIL 평균은 {combined['overall']['first_fail_iteration_statistics']['mean']:.3f}회, 모집단 표준편차는 {combined['overall']['first_fail_iteration_statistics']['population_std']:.3f}이다. 비단조 증가 횟수는 {sum(x['nonmonotonic_map_increase_count'] for x in combined['scenarios'])}회다.

## 9. 최종 비실패–FAIL 경계

{boundary_table}

Gap_env가 유사한 것은 동일 초기 구간과 고정 10회 이분 탐색 구조의 결과이며 성능 향상 근거가 아니다. 경계는 한 점이 아니라 두 앵커 사이 구간이다. Gap 통계에는 모집단 표준편차(`statistics.pstdev`)를 사용했다.

## 10. 기존 모델과 새 모델의 기술적 비교

{comp_table}

내부 검증 자료 자체가 다르므로 우월성·인과성·통계적 유의성을 주장하지 않는다.

## 11. 시험 결과

최종 검증 결과는 아래 생성되는 `test_results.json`을 기준으로 한다.

## 12. 허용/금지 주장

허용: 연속 프레임 중심 자료를 시나리오 다양화 자료로 재구성, 시나리오 단위 분리, 고정 YOLOv8s로 S0~S4 경계 탐색, 성공 시나리오의 비실패–FAIL 구간 식별. 금지: 실제 환경 일반화 입증, 통계적 우월성, 합성 자료만으로 실기 성능 보장, 환경 변수별 인과 기여, 단일 정확 경계값, S0~S4를 완전히 보지 않은 외부 시험으로 표현.

## 13. 남은 한계

단일 MATLAB 렌더러·지형·카메라 방향, 23개 학습/검증 시나리오, 합성 영상, 1개 난수 시드 학습, 고정 S0~S4 재사용에 한정된다. 지각 해시는 장면 유사성 경고 도구이지 의미론적 독립성의 증명이 아니다.

## 14. 주요 파일과 재현

- 사전등록: `{CONFIG_ROOT / 'pre_registration.json'}`
- 자료 감사: `{DATA_ROOT / 'dataset_audit_ko.md'}`
- 학습 요약: `{args.training_summary.resolve()}`
- 평가 집계: `{aggregate}`
- 그림: `{HERE / 'figures' / session}`
"""
    training_report = f"""# YOLOv8s 학습 보고서

새 모델은 COCO `yolov8s.pt`에서 시작하여 기존 설정(640, batch 4, SGD, 최대 50 epoch, patience 12, seed 42)을 유지했다. 모델 선택에는 5개 사전등록 검증 시나리오만 사용했고 S0~S4는 사용하지 않았다.

| 항목 | 값 |
|---|---:|
| 완료 epoch | {training['training_epochs_completed']} |
| 선택 epoch | {training['best_epoch_one_based']} |
| 사람 AP@0.5 | {val['person_ap50']:.6f} |
| 차량 AP@0.5 | {val['vehicle_ap50']:.6f} |
| mAP@0.5 | {val['map50']:.6f} |
| mAP@0.5:0.95 | {val['map50_95']:.6f} |
| 정밀도 | {val['mean_precision']:.6f} |
| 재현율 | {val['mean_recall']:.6f} |
| 학습 시간(초) | {training['training_seconds']:.3f} |
| best.pt 크기(bytes) | {training['best_weights_size_bytes']} |
| best.pt SHA-256 | `{training['best_weights_sha256']}` |
| CPU | {hw['cpu']['name']} ({hw['cpu']['physical_cores']}코어/{hw['cpu']['logical_processors']}스레드) |
| GPU / VRAM | {hw['gpu'].get('name')} / {hw['gpu'].get('memory_total_mib')} MiB |
| 시스템 RAM | {hw['system_memory']['total_gib']:.2f} GiB |
| 최대 프로세스 RSS | {usage['peak_process_rss_bytes'] / 2**30:.2f} GiB |
| 최대 시스템 메모리 사용량 | {usage['peak_system_memory_used_bytes'] / 2**30:.2f} GiB ({usage['peak_system_memory_percent']:.1f}%) |
| PyTorch 최대 GPU 할당/예약 | {usage['torch_peak_gpu_allocated_bytes'] / 2**20:.1f}/{usage['torch_peak_gpu_reserved_bytes'] / 2**20:.1f} MiB |

이는 새로운 MATLAB 모의환경 시나리오에서 모델 선택에 사용한 내부 검증 결과이며 실제 UAV 일반화 성능이 아니다.
"""
    paper = f"""# 논문용 제4장·제5장 초안

## 4. 실험 설계

### 4.1 장면 다양화 자료

기존 6개 학습 시나리오의 연속 추출 366장과 유사한 규모를 유지하면서, 18개 독립 학습 시나리오에서 시간적으로 균등한 20프레임씩 총 360장을 구성하였다. 검증은 겹치지 않는 5개 시나리오 100장으로 구성하였다.

### 4.2 시나리오 단위 분리

하나의 궤적에서 나온 모든 프레임은 하나의 분할에만 속하도록 했으며 ID, 시드, 궤적, 객체 배치, SHA-256 및 dHash 감사를 수행하였다.

### 4.3 모델 학습

YOLOv8s를 640 입력, batch 4, SGD, 최대 50 epoch, patience 12, seed 42로 학습하였다. 최종 가중치는 검증 시나리오만으로 선택되었다. 검증 mAP@0.5는 {val['map50']:.6f}였다.

### 4.4 고정 평가

기존에 사전 확정된 S0~S4를 변경하지 않고 새 가중치로 모든 영상의 추론과 AP를 다시 계산했다. 5개 모두 초기 정상 조건에서 PASS한 뒤 각 10회 대칭 탐색을 수행했다.

## 5. 결과 및 논의

### 5.1 초기 성능

{initial_table}

### 5.2 경계 탐색

{boundary_table}

### 5.3 기존 모델과의 기술적 비교

{comp_table}

### 5.4 해석

성공한 시나리오에서 최종 비실패 앵커와 FAIL 앵커 사이의 구간을 식별하였다. Gap_env의 유사성은 고정 반복 이분 탐색의 구조적 결과다.

### 5.5 타당성 위협

합성 단일 지형, 고정 카메라 방향, 제한된 시나리오 수와 단일 학습 시드 때문에 실제 비행환경 일반화나 통계적 우월성을 주장할 수 없다.
"""
    limits = """# 주장 범위와 한계

## 사용할 수 있는 주장

- 연속 프레임 중심의 기존 자료를 다양한 시나리오 중심 자료로 재구성하였다.
- 학습·검증·시험 자료를 시나리오 단위로 분리하였다.
- 고정된 YOLOv8s를 이용하여 S0~S4에서 경계 탐색을 수행하였다.
- 성공한 시나리오에서 비실패–FAIL 구간을 식별하였다.

## 사용하면 안 되는 주장

- 실제 비행환경 일반화가 입증되었다.
- 새 모델이 기존 모델보다 통계적으로 우수하다.
- 합성 데이터만으로 실제 UAV 탐지 성능이 보장된다.
- 안개·조도·잡음 각각의 인과적 기여가 확인되었다.
- 최종 경계가 하나의 정확한 환경값이다.
- S0~S4가 완전히 보지 않은 외부 시험 자료이다.

## 남은 한계

단일 합성 렌더러/지형, 고정 카메라 방향과 기울기, 23개 개발 시나리오, 5개 기존 고정 평가 시나리오, 단일 학습 시드에 한정된다. 서로 다른 자료에 대한 내부 검증 수치는 직접적인 우월성 검정이 아니다.
"""
    readme = f"""# Diverse training scenario split experiment

Plan: `{PLAN_ID}`  
Training run: `{training['run_id']}`  
Evaluation session: `{session}`

Run from the repository root in PowerShell. The commands below reproduce the run in an artifact-free checkout. This working tree intentionally refuses to overwrite the completed plan/run paths, so use a clean copy or assign new plan/run IDs for another execution. MATLAB batch rendering is shown explicitly because it was the successful renderer used for this run.

```powershell
.\\.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.diverse_training_scenario_split.plan
& 'C:\\Program Files\\MATLAB\\R2025b\\bin\\matlab.exe' -batch "addpath(fullfile(pwd,'experiments','yolov8_search_comparison','diverse_training_scenario_split')); render_registered_dataset"
.\\.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.diverse_training_scenario_split.prepare_dataset --skip-render
.\\.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.diverse_training_scenario_split.train_diverse
.\\.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.diverse_training_scenario_split.configure_evaluation --training-summary "{args.training_summary.resolve()}"
.\\.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.run_experiment --config "{args.evaluation_config.resolve()}" --plan "{FIXED_PLAN}" --scenario-root "{FIXED_SCENARIOS.parent}" --runs-root "{EVALUATION_ROOT / 'runs'}" --aggregated-root "{EVALUATION_ROOT / 'aggregated'}"
.\\.venv\\Scripts\\python.exe -m experiments.yolov8_search_comparison.diverse_training_scenario_split.report_and_validate --session "{session}" --training-summary "{args.training_summary.resolve()}" --evaluation-config "{args.evaluation_config.resolve()}"
```
"""
    write_text_new(report_dir / "final_execution_report_ko.md", execution)
    write_text_new(report_dir / "training_report_ko.md", training_report)
    write_text_new(report_dir / "paper_sections_4_5_ko.md", paper)
    write_text_new(report_dir / "limitations_and_claims_ko.md", limits)
    write_text_new(report_dir / "README.md", readme)

    validation = validate(session, training, config, combined, records, figures, report_dir)
    validation_path = EVALUATION_ROOT / "aggregated" / session / "validation_report.json"
    write_json_new(validation_path, validation)
    write_csv_new(EVALUATION_ROOT / "aggregated" / session / "validation_report.csv", validation["checks"])
    print(json.dumps({"session": session, "reports": str(report_dir.resolve()), "figures": [str(x.resolve()) for x in figures], "validation": validation}, ensure_ascii=False, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
