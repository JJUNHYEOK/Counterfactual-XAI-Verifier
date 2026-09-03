"""Freeze and document the actually executed KCI experiment conditions.

This command is intentionally read-only with respect to every existing experiment
artifact.  It refuses to write anything unless the designated best.pt exists and
matches the expected SHA-256, and it creates a new immutable output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import ultralytics
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EXP_ROOT = REPO_ROOT / "experiments" / "yolov8_search_comparison"
DIV_ROOT = EXP_ROOT / "diverse_training_scenario_split"
PLAN_ID = "diverse_training_scenario_split_v2"
RUN_ID = "diverse_yolov8s_seed42__20260903_010327_583537"
SESSION_ID = "kci_diverse_training_s0_s4_v1__20260903_011624_021544"
EXPECTED_WEIGHTS_SHA256 = (
    "E141B3DC0104CBC03AD2EE573FFA451C369C9D93D8690E560E07C146078F5D9E"
)
FREEZE_STATEMENT = (
    "이 시점 이후 S0~S4 또는 신규 시험 결과를 보고 가중치, 판정 기준, "
    "탐색 방법 및 환경 범위를 변경하지 않는다."
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def file_record(path: Path, group: str, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required lock target does not exist: {resolved}")
    return {
        "group": group,
        "role": role,
        "path": str(resolved),
        "repository_relative_path": rel(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def gather_hash_targets(
    training_plan: dict[str, Any], fixed_plan: dict[str, Any]
) -> list[dict[str, Any]]:
    config_root = DIV_ROOT / "config" / PLAN_ID
    data_root = DIV_ROOT / "data" / PLAN_ID
    training_root = DIV_ROOT / "training" / PLAN_ID
    run_root = training_root / RUN_ID
    evaluation_root = DIV_ROOT / "evaluation"
    visual_root = HERE / "outputs" / "near_duplicate_visual_audit_v1"

    targets: list[tuple[Path, str, str]] = [
        (run_root / "weights" / "best.pt", "weights", "frozen final weights"),
        (run_root / "args.yaml", "training", "Ultralytics training arguments"),
        (training_root / f"{RUN_ID}_training_summary.json", "training", "training result and runtime record"),
        (config_root / "pre_registration.json", "training_plan", "training pre-registration"),
        (config_root / "scenario_plan.json", "training_plan", "train/validation scenario plan"),
        (config_root / "scenario_plan.csv", "training_plan", "tabular train/validation scenario plan"),
        (data_root / "data.yaml", "dataset", "Ultralytics dataset definition"),
        (data_root / "dataset_summary.json", "dataset", "dataset and GT summary"),
        (data_root / "dataset_manifest.csv", "dataset", "selected frame and label manifest"),
        (data_root / "scenario_split.json", "dataset", "scenario-disjoint split audit"),
        (data_root / "exact_duplicate_audit.json", "dataset_audit", "exact image duplicate audit"),
        (data_root / "near_duplicate_audit.json", "dataset_audit", "dHash near-duplicate audit"),
        (visual_root / "near_duplicate_visual_audit.csv", "visual_audit", "manual review table for all warnings"),
        (visual_root / "near_duplicate_visual_audit_ko.md", "visual_audit", "manual review report"),
        (visual_root / "near_duplicate_visual_audit_summary.json", "visual_audit", "manual review summary"),
        (
            EXP_ROOT / "multi_scenario_symmetric" / "config" / "multi_scenario_symmetric_plan_v2" / "scenario_plan.json",
            "fixed_test_plan",
            "frozen S0-S4 plan",
        ),
        (
            evaluation_root / "config" / f"evaluation_config__{RUN_ID}.json",
            "evaluation",
            "executed evaluation settings",
        ),
        (
            evaluation_root / "config" / f"evaluation_preflight__{RUN_ID}.json",
            "evaluation",
            "S0-S4 preflight and cache firewall",
        ),
        (
            evaluation_root / "aggregated" / SESSION_ID / "scenario_summary.csv",
            "evaluation_result",
            "executed S0-S4 aggregate",
        ),
        (EXP_ROOT / "search_core.py", "code", "verdict, search, environment gap, and AP implementation"),
        (EXP_ROOT / "model_mapping.py", "code", "person/vehicle conversion rules"),
        (EXP_ROOT / "run_comparison.py", "code", "YOLO inference and evaluation implementation"),
        (
            EXP_ROOT / "multi_scenario_symmetric" / "run_experiment.py",
            "code",
            "S0-S4 fixed evaluation driver used with the frozen v2 evaluation config",
        ),
        (DIV_ROOT / "plan.py", "code", "training/validation scenario planning"),
        (DIV_ROOT / "prepare_dataset.py", "code", "dataset rendering and audits"),
        (DIV_ROOT / "train_diverse.py", "code", "training implementation"),
        (DIV_ROOT / "export_training_scenario_frames.m", "code", "training/validation GT rendering"),
        (
            EXP_ROOT / "multi_scenario_symmetric" / "export_scenario_frames.m",
            "code",
            "fixed S0-S4 GT rendering",
        ),
        (REPO_ROOT / "render_eo_image.m", "simulator", "EO renderer"),
        (REPO_ROOT / "eo_camera_3d_render.m", "simulator", "3-D camera renderer and visible instance pass"),
        (REPO_ROOT / "init_uav_workspace.m", "simulator", "simulation workspace/environment setup"),
        (HERE / "visual_audit.py", "audit_code", "manual visual audit artifact builder"),
        (HERE / "config" / "near_duplicate_visual_audit_v1" / "manual_decisions.json", "audit_decisions", "manual visual decisions"),
        (HERE / "lock_experiment.py", "lock_code", "this lock generator"),
    ]

    for scenario in training_plan["scenarios"]:
        targets.append(
            (
                Path(scenario["scenario_config_path"]),
                "training_scenario_config" if scenario["split"] == "train" else "validation_scenario_config",
                scenario["scenario_id"],
            )
        )
    for scenario in fixed_plan["scenarios"]:
        targets.append(
            (Path(scenario["scenario_config_path"]), "fixed_test_scenario_config", scenario["scenario_id"])
        )

    seen: set[Path] = set()
    records: list[dict[str, Any]] = []
    for path, group, role in targets:
        resolved = path.resolve()
        if resolved not in seen:
            records.append(file_record(resolved, group, role))
            seen.add(resolved)
    return records


def fixed_scenario_summary(fixed_plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": item["scenario_id"],
            "seed": item["seed"],
            "trajectory_name": item["trajectory_name"],
            "uav_initial_xyz": item["uav_initial_xyz"],
            "uav_end_xyz": item["uav_end_xyz"],
            "altitude_range_m": item["altitude_range_m"],
            "person_count": item["person_count"],
            "vehicle_count": item["vehicle_count"],
            "scenario_config_path": item["scenario_config_path"],
            "scenario_config_sha256": item["scenario_config_sha256"],
        }
        for item in fixed_plan["scenarios"]
    ]


def build_lock() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_root = DIV_ROOT / "config" / PLAN_ID
    data_root = DIV_ROOT / "data" / PLAN_ID
    training_root = DIV_ROOT / "training" / PLAN_ID
    run_root = training_root / RUN_ID
    evaluation_root = DIV_ROOT / "evaluation"
    weights = run_root / "weights" / "best.pt"
    actual_weights_hash = sha256(weights)
    if actual_weights_hash != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(
            "Designated best.pt hash mismatch; no lock files were written. "
            f"expected={EXPECTED_WEIGHTS_SHA256}, actual={actual_weights_hash}, path={weights.resolve()}"
        )

    evaluation_config_path = evaluation_root / "config" / f"evaluation_config__{RUN_ID}.json"
    evaluation_config = read_json(evaluation_config_path)
    if str(evaluation_config["expected_weights_sha256"]).upper() != actual_weights_hash:
        raise RuntimeError("Evaluation config expected hash does not match the designated best.pt")

    training_plan = read_json(config_root / "scenario_plan.json")
    pre_registration = read_json(config_root / "pre_registration.json")
    dataset_summary = read_json(data_root / "dataset_summary.json")
    scenario_split = read_json(data_root / "scenario_split.json")
    training_summary = read_json(training_root / f"{RUN_ID}_training_summary.json")
    fixed_plan = read_json(
        EXP_ROOT / "multi_scenario_symmetric" / "config" / "multi_scenario_symmetric_plan_v2" / "scenario_plan.json"
    )
    visual_summary = read_json(HERE / "outputs" / "near_duplicate_visual_audit_v1" / "near_duplicate_visual_audit_summary.json")
    sample_training_manifest = read_json(
        data_root / "rendered_scenarios" / "T01_east_level_low" / "frame_manifest.json"
    )
    evaluation_preflight = read_json(
        evaluation_root / "config" / f"evaluation_preflight__{RUN_ID}.json"
    )

    sample_evaluation_path = next(
        (
            evaluation_root / "runs" / SESSION_ID / "S0" / "evaluations"
        ).glob("*/evaluation.json")
    )
    sample_evaluation = read_json(sample_evaluation_path)
    model_record = sample_evaluation["model"]

    model = YOLO(str(weights.resolve()))
    checkpoint = model.ckpt or {}
    train_args = checkpoint.get("train_args") or {}
    model_names = {str(key): str(value) for key, value in model.names.items()}
    if model_names != evaluation_config["expected_model_class_names"]:
        raise RuntimeError(
            f"Checkpoint class names differ from the frozen evaluation config: {model_names}"
        )

    class_rules = {
        "ground_truth": {
            "simulator_class_1": "person",
            "simulator_class_2": "vehicle",
            "conversion": "MATLAB visible-instance colour mask IDs are converted to person/vehicle bounding boxes; only visible positive-area boxes are emitted.",
        },
        "detector": {
            "actual_checkpoint_names": model_names,
            "actual_mapping": model_record["class_mapping"],
            "general_alias_rule": {
                "person": ["person", "pedestrian", "people", "human"],
                "vehicle": ["vehicle", "car", "motorcycle", "motorbike", "bus", "truck", "van"],
                "other_classes": "excluded",
            },
            "actual_two_class_effect": "checkpoint class 0 -> person; checkpoint class 1 -> vehicle; no many-to-one merge was needed for this checkpoint",
        },
    }

    thresholds = evaluation_config["thresholds"]
    bounds = evaluation_config["environment_bounds"]
    verdict_definition = {
        "PASS": f"mAP@0.5 >= {thresholds['pass_map50']}",
        "MARGINAL": f"{thresholds['fail_map50']} <= mAP@0.5 < {thresholds['pass_map50']}",
        "FAIL": f"mAP@0.5 < {thresholds['fail_map50']}",
        "boundary_anchor_priority": "FAIL updates the failure anchor; PASS or MARGINAL updates the non-failure anchor.",
        "comparison_precedence": "Evaluate PASS first (>=0.50), then FAIL (<0.25), otherwise MARGINAL. Equality 0.50 is PASS and equality 0.25 is MARGINAL.",
    }
    denominators = {
        key: float(value[1]) - float(value[0]) for key, value in bounds.items()
    }
    formula = (
        "Gap_env = |nf_fog-f_fog|/100 + |nf_lux-f_lux|/14800 + "
        "|nf_noise-f_noise|/0.6"
    )

    hardware = training_summary["hardware"]
    package_record = training_summary["packages"]
    software = {
        "python_executed_now": sys.version.replace("\n", " "),
        "python_training_record": package_record["python"],
        "matlab": {
            "version": sample_training_manifest["matlab_version"],
            "release": sample_training_manifest["matlab_release"],
        },
        "pytorch_executed_now": torch.__version__,
        "pytorch_training_record": package_record["torch"],
        "ultralytics_executed_now": ultralytics.__version__,
        "ultralytics_training_record": package_record["ultralytics"],
        "cuda_runtime_executed_now": torch.version.cuda,
        "cuda_training_record": package_record["torch_cuda"],
        "cudnn_executed_now": torch.backends.cudnn.version(),
        "cudnn_training_record": package_record["cudnn"],
        "platform_executed_now": platform.platform(),
    }

    hash_records = gather_hash_targets(training_plan, fixed_plan)
    lock = {
        "schema_version": "1.0",
        "lock_id": "final_experiment_lock_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "KCI final reproducibility lock; documents actual executed settings without retraining or threshold adjustment.",
        "freeze_statement": FREEZE_STATEMENT,
        "weights_verification": {
            "status": "MATCH",
            "expected_sha256": EXPECTED_WEIGHTS_SHA256,
            "actual_sha256": actual_weights_hash,
            "path": str(weights.resolve()),
            "size_bytes": weights.stat().st_size,
        },
        "model": {
            "description": "다양화된 합성자료로 재학습한 YOLOv8s two-class object detector",
            "task": model.task,
            "architecture": "YOLOv8s",
            "classes": 2,
            "model_class_names": model_names,
            "parameter_count": sum(parameter.numel() for parameter in model.model.parameters()),
            "stride_pixels": [float(value) for value in model.model.stride.tolist()],
            "checkpoint_training_model": train_args.get("model"),
            "checkpoint_training_data": train_args.get("data"),
            "checkpoint_training_epochs": train_args.get("epochs"),
            "checkpoint_training_batch": train_args.get("batch"),
            "checkpoint_training_seed": train_args.get("seed"),
        },
        "image_and_detector_settings": {
            "rendered_input_video_frame_width": sample_training_manifest["image_width"],
            "rendered_input_video_frame_height": sample_training_manifest["image_height"],
            "rendered_input_video_frame_format": "PNG frame sequence",
            "yolo_input_size": evaluation_config["yolov8"]["input_size"],
            "confidence_threshold": evaluation_config["yolov8"]["confidence_threshold"],
            "nms_iou_threshold": evaluation_config["yolov8"]["nms_iou_threshold"],
            "ap_iou_threshold": evaluation_config["yolov8"]["ap_iou_threshold"],
            "configured_inference_batch_size": evaluation_config["yolov8"]["batch_size"],
            "observed_effective_inference_batch_size": sample_evaluation["effective_inference_batch_size"],
            "configured_device": evaluation_config["yolov8"]["device"],
            "resolved_device": model_record["device_argument"],
        },
        "class_definitions_and_conversion": class_rules,
        "verdict": verdict_definition,
        "environment": {
            "initial": evaluation_config["initial_environment"],
            "bounds": bounds,
            "pre_fail_degradation": evaluation_config["degradation"],
            "rounding_after_clamp": {
                "fog_percent_decimals": 2,
                "illumination_lux_decimals": 1,
                "camera_noise_decimals": 4,
            },
        },
        "search": {
            "method": evaluation_config["search_method"],
            "initial_point": evaluation_config["initial_environment"],
            "maximum_evaluations_per_scenario": evaluation_config["max_evaluations"],
            "before_first_fail": "fog'=clamp(fog+30); illumination'=clamp(illumination*0.5); noise'=clamp(noise+0.2)",
            "after_first_fail_symmetric": "next = clamp_and_round(0.50*nonfailure_anchor + 0.50*failure_anchor), component-wise",
            "anchor_rule": verdict_definition["boundary_anchor_priority"],
            "boundary_success": "both a non-failure anchor (PASS/MARGINAL) and a FAIL anchor exist",
            "gap_map_formula": "Gap_mAP = |mAP50_nonfailure - mAP50_FAIL|",
            "gap_environment_formula": formula,
            "gap_environment_denominators": denominators,
            "normalization_aggregation": "sum of the three normalized absolute component differences; not an average",
        },
        "randomness": {
            "evaluation_seed": 42,
            "training_seed": training_summary["settings"]["seed"],
            "training_scenario_seeds": {
                item["scenario_id"]: item["seed"] for item in training_plan["scenarios"] if item["split"] == "train"
            },
            "validation_scenario_seeds": {
                item["scenario_id"]: item["seed"] for item in training_plan["scenarios"] if item["split"] == "val"
            },
            "fixed_test_scenario_seeds": {
                item["scenario_id"]: item["seed"] for item in fixed_plan["scenarios"]
            },
            "determinism": "Python random, NumPy and PyTorch are seeded; CUDA is seeded; deterministic algorithms warn-only; cuDNN benchmark disabled and deterministic enabled. MATLAB renderer uses each scenario seed with twister.",
        },
        "ground_truth": {
            "mode": evaluation_config["ground_truth_mode"],
            "version": evaluation_config["gt_version"],
            "renderer": sample_training_manifest["renderer"],
            "method": "A second no-fog/no-noise visible-instance colour rendering pass identifies visible pixels per object. The tight min/max pixel rectangle becomes the GT bbox; fully hidden/out-of-view objects emit no box.",
            "frame_stride": evaluation_config["frame_stride"],
            "expected_fixed_test_frame_count": evaluation_config["expected_frame_count"],
        },
        "map_calculation": {
            "metric": "mAP@0.5",
            "matching": "Per class, detections over all frames are sorted by descending confidence and greedily matched one-to-one to the highest-IoU unused GT in the same frame; IoU >= 0.5 is TP.",
            "ap": "Integral under the monotonic precision envelope at recall-change points.",
            "map": "Unweighted arithmetic mean of person AP50 and vehicle AP50 for classes with at least one GT.",
        },
        "training_and_validation": {
            "plan_id": training_plan["plan_id"],
            "frozen_at_utc": pre_registration["frozen_at_utc"],
            "split_policy": training_plan["split_policy"],
            "training_scenarios": [item["scenario_id"] for item in training_plan["scenarios"] if item["split"] == "train"],
            "validation_scenarios": [item["scenario_id"] for item in training_plan["scenarios"] if item["split"] == "val"],
            "selected_frames_per_scenario": training_plan["selected_frame_count_per_scenario"],
            "selected_frame_indices": training_plan["selected_frame_indices"],
            "training_image_count": dataset_summary["train_image_count"],
            "validation_image_count": dataset_summary["validation_image_count"],
            "scenario_disjoint_check": scenario_split,
            "selection_policy": training_summary["settings"]["selection_policy"],
            "training_settings": training_summary["settings"],
        },
        "fixed_S0_S4": {
            "plan_id": fixed_plan["plan_id"],
            "created_at_utc": fixed_plan["created_at_utc"],
            "registration_rule": fixed_plan["registration_rule"],
            "scenarios": fixed_scenario_summary(fixed_plan),
            "preflight": evaluation_preflight,
            "description_limit": "S0-S4 are previously pre-registered fixed evaluation scenarios. They must not be described as independent, completely unseen, or external test data because the visual audit found material near-similarity.",
        },
        "near_duplicate_visual_audit": visual_summary,
        "software": software,
        "hardware": hardware,
        "git": {
            "branch": git_text("branch", "--show-current"),
            "status_porcelain_v1": git_text("status", "--short", "--branch"),
            "commit": git_text("rev-parse", "HEAD"),
            "working_tree_clean": not bool(git_text("status", "--porcelain")),
        },
        "cache_and_retention": {
            "rendered_frames_retained": evaluation_config["retain_rendered_frames"],
            "executed_policy": evaluation_config["cache_policy"],
            "preflight_old_detection_or_metric_cache_reuse_allowed": evaluation_preflight["old_detection_or_metric_cache_reuse_allowed"],
        },
        "documented_discrepancies": [
            {
                "field": "evaluation JSON model.weights_provenance",
                "recorded_value": model_record["weights_provenance"],
                "actual_value": "diversified synthetic-data retrained YOLOv8s best.pt",
                "basis": "The checkpoint train_args identifies the v2 data.yaml, epochs=50, batch=4, seed=42; its hash equals the preflight/config/training summary expected hash.",
                "cause": "run_comparison.py provenance heuristic recognises only the older yolov8s_sim_20260902 path substring and therefore mislabels the v2 training path.",
                "resolution": "Use the checkpoint fields and verified hash; do not use the erroneous provenance string in the paper.",
            },
            {
                "field": "dataset_summary.near_duplicate_review_passed",
                "recorded_value": dataset_summary["near_duplicate_review_passed"],
                "actual_value": "The automated audit passed its configured warning policy, but subsequent mandatory visual review found 9 high-risk category-3 pairs.",
                "basis": "near_duplicate_visual_audit.csv and near_duplicate_visual_audit_ko.md",
                "resolution": "Do not describe S0-S4 as independent/unseen/external test data.",
            },
        ],
        "claims": {
            "allowed": [
                "The designated diversified synthetic-data retrained YOLOv8s and evaluation conditions were frozen with hashes.",
                "S0-S4 are pre-registered fixed evaluation scenarios in the existing MATLAB simulation environment.",
                "No exact image SHA-256 duplicate was found between the selected training images and S0-S4.",
            ],
            "prohibited": [
                "S0-S4 are independent, completely unseen, or external test data.",
                "Exact-hash separation proves absence of train-test visual leakage.",
                "The results generalize to real flight environments.",
                "Synthetic data alone guarantees real-world detection performance.",
            ],
        },
        "hash_manifest": {
            "csv_filename": "final_experiment_lock_files.csv",
            "file_count": len(hash_records),
        },
    }
    return lock, hash_records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_markdown(path: Path, lock: dict[str, Any], hashes: list[dict[str, Any]]) -> None:
    w = lock["weights_verification"]
    image = lock["image_and_detector_settings"]
    env = lock["environment"]
    search = lock["search"]
    train = lock["training_and_validation"]
    fixed = lock["fixed_S0_S4"]
    sw = lock["software"]
    hw = lock["hardware"]
    audit = lock["near_duplicate_visual_audit"]

    category_counts = audit["category_counts"]
    risk_counts = audit["risk_counts"]
    lines = [
        "# 최종 실험 조건 고정 문서",
        "",
        f"- 고정 ID: `{lock['lock_id']}`",
        f"- 생성 시각(UTC): `{lock['created_at_utc']}`",
        f"- 목적: {lock['purpose']}",
        "",
        f"> {lock['freeze_statement']}",
        "",
        "## 1. 최종 가중치",
        "",
        f"- 구조: {lock['model']['description']} (탐지, 2개 클래스, {lock['model']['parameter_count']:,} parameters)",
        f"- 경로: `{w['path']}`",
        f"- 크기: {w['size_bytes']:,} bytes",
        f"- 예상 SHA-256: `{w['expected_sha256']}`",
        f"- 실제 SHA-256: `{w['actual_sha256']}`",
        f"- 대조 결과: **{w['status']}**",
        "",
        "## 2. 영상·탐지·평가 설정",
        "",
        f"- 렌더 영상: {image['rendered_input_video_frame_width']}×{image['rendered_input_video_frame_height']} PNG 프레임 시퀀스, frame stride 1, S0~S4 각 181프레임",
        f"- YOLO 입력 크기: {image['yolo_input_size']}",
        f"- 탐지 confidence 기준: {image['confidence_threshold']}",
        f"- 중복 탐지 제거 NMS IoU: {image['nms_iou_threshold']}",
        f"- AP 일치 IoU: {image['ap_iou_threshold']}",
        f"- 추론 batch: 설정 {image['configured_inference_batch_size']}, 관측 유효 batch {image['observed_effective_inference_batch_size']}",
        f"- 장치: 설정 `{image['configured_device']}`, 실제 `{image['resolved_device']}`",
        "- 클래스: 체크포인트 0=`person`, 1=`vehicle`. 현재 체크포인트에서는 추가 병합이 없었다.",
        "- 일반 변환 규칙: person/pedestrian/people/human은 person, vehicle/car/motorcycle/motorbike/bus/truck/van은 vehicle, 나머지는 제외한다.",
        "",
        "## 3. GT와 mAP",
        "",
        f"- GT 모드: `{lock['ground_truth']['mode']}`",
        f"- 렌더러: `{lock['ground_truth']['renderer']}`",
        f"- 생성 방식: {lock['ground_truth']['method']}",
        f"- AP: {lock['map_calculation']['matching']} {lock['map_calculation']['ap']}",
        f"- mAP: {lock['map_calculation']['map']}",
        "",
        "## 4. 판정식과 우선순위",
        "",
        f"- PASS: `{lock['verdict']['PASS']}`",
        f"- MARGINAL: `{lock['verdict']['MARGINAL']}`",
        f"- FAIL: `{lock['verdict']['FAIL']}`",
        f"- 비교 우선순위: {lock['verdict']['comparison_precedence']}",
        f"- 앵커 우선순위: {lock['verdict']['boundary_anchor_priority']}",
        "",
        "## 5. 환경과 대칭 탐색",
        "",
        f"- 초기점: `{json.dumps(env['initial'], ensure_ascii=False, sort_keys=True)}`",
        f"- 범위: `{json.dumps(env['bounds'], ensure_ascii=False, sort_keys=True)}`",
        f"- 최초 FAIL 이전: `{search['before_first_fail']}`",
        f"- 최초 FAIL 이후: `{search['after_first_fail_symmetric']}`",
        f"- 최대 평가 횟수: {search['maximum_evaluations_per_scenario']}회",
        f"- Gap_mAP: `{search['gap_map_formula']}`",
        f"- Gap_env: `{search['gap_environment_formula']}` (세 정규화 항의 합이며 평균이 아님)",
        "- clamp 후 반올림: fog 2자리, illumination 1자리, noise 4자리",
        "",
        "## 6. 난수와 자료 계획",
        "",
        f"- 학습 난수값: {lock['randomness']['training_seed']}; 평가 공통 난수값: {lock['randomness']['evaluation_seed']}",
        f"- 학습: {len(train['training_scenarios'])}개 시나리오 × {train['selected_frames_per_scenario']}프레임 = {train['training_image_count']}장",
        f"- 검증: {len(train['validation_scenarios'])}개 시나리오 × {train['selected_frames_per_scenario']}프레임 = {train['validation_image_count']}장",
        f"- 학습 시나리오: {', '.join(train['training_scenarios'])}",
        f"- 검증 시나리오: {', '.join(train['validation_scenarios'])}",
        f"- 가중치 선택: {train['selection_policy']}",
        f"- 고정 시험 계획: `{fixed['plan_id']}`, S0~S4 ({', '.join(item['scenario_id'] for item in fixed['scenarios'])})",
        "",
        "## 7. 실행 환경",
        "",
        f"- CPU: {hw['cpu']['name']} ({hw['cpu']['physical_cores']} cores/{hw['cpu']['logical_processors']} threads)",
        f"- RAM: {hw['system_memory']['total_gib']:.2f} GiB ({hw['system_memory']['total_bytes']:,} bytes)",
        f"- GPU: {hw['gpu']['name']}, {hw['gpu']['memory_total_mib']} MiB, driver {hw['gpu']['driver_version']}",
        f"- Python: {sw['python_training_record']}",
        f"- MATLAB: {sw['matlab']['version']}, release {sw['matlab']['release']}",
        f"- PyTorch: {sw['pytorch_training_record']}; CUDA runtime {sw['cuda_training_record']}; cuDNN {sw['cudnn_training_record']}",
        f"- Ultralytics: {sw['ultralytics_training_record']}",
        f"- OS: {hw['platform']}",
        "",
        "## 8. Git 상태",
        "",
        f"- 브랜치: `{lock['git']['branch']}`",
        f"- HEAD: `{lock['git']['commit']}`",
        f"- clean 여부: `{lock['git']['working_tree_clean']}`",
        "```text",
        lock["git"]["status_porcelain_v1"],
        "```",
        "",
        "## 9. 유사 영상 감사와 주장 제한",
        "",
        f"- 시각 검토: {audit['pair_count']}쌍, 정확 SHA 중복 {audit['exact_duplicate_count']}쌍",
        f"- 분류: 배경/빈 화면 유사 {category_counts['1']}쌍, 객체 구성 충분히 다름 {category_counts['2']}쌍, 객체·크기·배경 매우 유사 {category_counts['3']}쌍",
        f"- 위험도: 낮음 {risk_counts.get('낮음', 0)}쌍, 중간 {risk_counts.get('중간', 0)}쌍, 높음 {risk_counts.get('높음', 0)}쌍",
        f"- 결론: {fixed['description_limit']}",
        "",
        "## 10. 확인된 불일치",
        "",
    ]
    for item in lock["documented_discrepancies"]:
        lines.extend(
            [
                f"- `{item['field']}`: 기록값 `{item['recorded_value']}`. 실제 판단은 `{item['actual_value']}`이다. {item['resolution']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 11. 고정 파일 해시",
            "",
            f"총 {len(hashes)}개 파일의 경로·크기·SHA-256을 `final_experiment_lock_files.csv`에 기록했다.",
            "",
            "## 12. 사용할 수 없는 주장",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in lock["claims"]["prohibited"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        # The first v1 attempt is intentionally retained as a two-file partial
        # failure record; this complete directory is created without overwriting it.
        default=HERE / "outputs" / "final_experiment_lock_v1_complete",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable lock output: {output_dir}")
    lock, hash_records = build_lock()
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "final_experiment_lock.json"
    csv_path = output_dir / "final_experiment_lock_files.csv"
    md_path = output_dir / "final_experiment_lock_ko.md"
    json_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, hash_records)
    write_markdown(md_path, lock, hash_records)
    print(json.dumps({"output_dir": str(output_dir), "weights_status": "MATCH", "hash_file_count": len(hash_records)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
