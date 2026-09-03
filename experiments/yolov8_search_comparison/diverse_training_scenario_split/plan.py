"""Audit the prior experiment and pre-register 23 diverse train/val scenarios."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
PLAN_ID = "diverse_training_scenario_split_v2"
CONFIG_ROOT = HERE / "config" / PLAN_ID
DATA_ROOT = HERE / "data" / PLAN_ID
OLD_TRAINING_ROOT = EXPERIMENT_ROOT / "training" / "yolov8s_sim_20260902"
OLD_BEST = OLD_TRAINING_ROOT / "runs" / "full_yolov8s_seed42" / "weights" / "best.pt"
FIXED_EVAL_ROOT = EXPERIMENT_ROOT / "multi_scenario_symmetric"
FIXED_PLAN = FIXED_EVAL_ROOT / "config" / "multi_scenario_symmetric_plan_v2" / "scenario_plan.json"
FIXED_SCENARIOS = FIXED_EVAL_ROOT / "scenarios" / "multi_scenario_symmetric_plan_v2"
FIXED_SESSION = "kci_multi_scenario_symmetric_v1__20260902_201715_587801"
SELECTED_FRAMES = [1, 10, 20, 29, 39, 48, 58, 67, 77, 86, 96, 105, 115, 124, 134, 143, 153, 162, 172, 181]
GT_VERSION = "rendered_instance_mask_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite pre-registered file: {path}")
        return
    path.write_bytes(data)


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite pre-registered file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [
        {key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()}
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def object_layout(stations: list[tuple[float, float]], separation: float = 2.4, overlap: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for station_index, (x, y) in enumerate(stations):
        delta = 0.25 if overlap else separation / 2
        longitudinal = 0.8 if overlap else 2.6 + 0.2 * station_index
        rows.extend(
            [
                {
                    "object_id": len(rows) + 1,
                    "class_id": 1,
                    "class_name": "person",
                    "xy": [round(x, 3), round(y - delta, 3)],
                    "radius_height": [0.5, 1.8],
                },
                {
                    "object_id": len(rows) + 2,
                    "class_id": 2,
                    "class_name": "vehicle",
                    "xy": [round(x + longitudinal, 3), round(y + delta, 3)],
                    "radius_height": [1.2, 1.6],
                },
            ]
        )
    return rows


def definitions() -> list[dict[str, Any]]:
    specs = [
        ("T01_east_level_low", "train", 4101, "동서 직선 저고도 순항", [-32, -4, 52], [18, -4, 52], [(-10, -5), (10, -2), (32, -5)], 0, 15000, 0.00, 2.4, False, ["동서 직선", "저고도", "중간 간격"]),
        ("T02_east_level_high", "train", 4102, "동서 직선 고고도 순항", [-28, 6, 60], [20, 6, 60], [(-5, 4), (15, 8), (36, 5)], 4, 14200, 0.01, 3.4, False, ["동서 직선", "고고도", "넓은 횡간격"]),
        ("T03_east_ascending", "train", 4103, "동쪽 진행 상승", [-35, -8, 46], [15, -8, 62], [(-12, -10), (9, -6), (31, -9)], 8, 13200, 0.02, 2.0, False, ["상승", "동서 직선", "거리 감소"]),
        ("T04_east_descending", "train", 4104, "동쪽 진행 하강", [-30, 9, 67], [18, 9, 49], [(-8, 7), (13, 11), (35, 8)], 12, 12000, 0.03, 2.8, False, ["하강", "동서 직선", "크기 증가"]),
        ("T05_west_retreat_low", "train", 4105, "서쪽 이탈 저고도", [5, -6, 54], [-10, -6, 54], [(15, -8), (21, -4), (27, -7)], 16, 11000, 0.04, 2.2, False, ["근거리-원거리 이탈", "역방향 궤적"]),
        ("T06_west_retreat_high", "train", 4106, "서쪽 이탈 고고도", [7, 8, 62], [-8, 8, 62], [(17, 6), (23, 10), (29, 7)], 20, 10000, 0.05, 3.2, False, ["근거리-원거리 이탈", "고고도"]),
        ("T07_north_lateral", "train", 4107, "남-북 횡방향", [-20, -18, 55], [-20, 18, 55], [(-10, -10), (-3, 0), (4, 10)], 2, 14500, 0.01, 2.6, False, ["남북 횡방향", "고정 고도"]),
        ("T08_south_lateral", "train", 4108, "북-남 횡방향", [-16, 20, 62], [-16, -20, 62], [(-6, 12), (1, 0), (8, -12)], 6, 13600, 0.02, 3.0, False, ["남북 역방향", "고고도"]),
        ("T09_diagonal_swnE", "train", 4109, "남서-북동 대각선", [-32, -16, 56], [12, 14, 56], [(-8, -10), (12, 1), (33, 12)], 10, 12600, 0.03, 2.0, False, ["대각선", "화면 횡이동"]),
        ("T10_diagonal_nwse", "train", 4110, "북서-남동 대각 상승", [-30, 16, 51], [14, -14, 61], [(-7, 10), (14, 0), (35, -10)], 14, 11600, 0.04, 3.6, False, ["대각선", "상승", "넓은 간격"]),
        ("T11_far_to_near", "train", 4111, "원거리 시작 접근 및 하강", [-48, -2, 64], [0, -2, 52], [(-15, -4), (8, 0), (31, -3)], 18, 10500, 0.05, 2.4, False, ["원거리-근거리 접근", "하강"]),
        ("T12_near_to_far", "train", 4112, "근거리 시작 이탈 및 상승", [0, 3, 50], [-10, 3, 65], [(11, 1), (18, 5), (25, 2)], 1, 14800, 0.00, 2.8, False, ["근거리-원거리 이탈", "상승"]),
        ("T13_high_short", "train", 4113, "고고도 단거리 통과", [-20, -11, 70], [10, -11, 70], [(15, -13), (30, -9), (45, -12)], 5, 13800, 0.01, 4.0, False, ["고고도", "짧은 궤적", "작은 객체 크기"]),
        ("T14_low_close", "train", 4114, "저고도 근접 통과", [-25, 12, 44], [15, 12, 47], [(-7, 10), (10, 14), (28, 11)], 9, 12800, 0.02, 2.0, False, ["저고도", "근접", "큰 객체 크기"]),
        ("T15_lateral_ascending", "train", 4115, "횡방향 상승", [-22, -16, 48], [-22, 16, 63], [(-12, -10), (-5, 0), (2, 10)], 13, 11800, 0.03, 3.2, False, ["남북 횡방향", "상승"]),
        ("T16_lateral_descending", "train", 4116, "횡방향 하강", [-14, 18, 65], [-14, -18, 48], [(-4, 10), (3, 0), (10, -10)], 17, 10800, 0.04, 2.2, False, ["남북 횡방향", "하강"]),
        ("T17_partial_occlusion", "train", 4117, "근접 배치 부분 가림 접근", [-36, 0, 57], [8, 0, 57], [(-10, 0), (10, 0.4), (30, -0.4)], 20, 9800, 0.05, 0.5, True, ["사람-차량 근접 배치", "부분 가림", "접근"]),
        ("T18_wide_diagonal", "train", 4118, "넓은 배치 대각 하강", [-28, -12, 59], [16, 12, 50], [(-4, -8), (17, 1), (38, 10)], 3, 14200, 0.01, 7.0, False, ["대각선", "하강", "넓은 객체 간격"]),
        ("V01_east_ascending", "val", 5101, "검증 동쪽 상승", [-34, 3, 49], [13, 3, 66], [(-10, 1), (11, 5), (33, 2)], 7, 13400, 0.02, 2.6, False, ["상승", "동서 직선"]),
        ("V02_west_retreat", "val", 5102, "검증 서쪽 이탈", [3, -10, 56], [-10, -10, 62], [(13, -12), (19, -8), (25, -11)], 11, 12400, 0.03, 3.4, False, ["이탈", "역방향", "상승"]),
        ("V03_north_lateral", "val", 5103, "검증 남-북 횡방향", [-18, -20, 58], [-18, 17, 58], [(-8, -12), (-1, 0), (6, 11)], 15, 11400, 0.04, 2.0, False, ["남북 횡방향", "고정 고도"]),
        ("V04_diagonal_descending", "val", 5104, "검증 대각 하강", [-31, 13, 64], [13, -12, 50], [(-7, 9), (14, 0), (35, -8)], 19, 10400, 0.05, 3.0, False, ["대각선", "하강"]),
        ("V05_occluded_approach", "val", 5105, "검증 부분 가림 접근", [-45, 7, 62], [1, 7, 54], [(-13, 6.5), (9, 7.3), (31, 6.7)], 0, 15000, 0.00, 0.5, True, ["원거리-근거리 접근", "부분 가림"]),
    ]
    scenarios: list[dict[str, Any]] = []
    for sid, split, seed, name, start, end, stations, fog, illumination, noise, sep, overlap, factors in specs:
        scenarios.append(
            {
                "schema_version": "1.0",
                "scenario_id": sid,
                "split": split,
                "seed": seed,
                "duration_seconds": 18.0,
                "total_frames": 181,
                "selected_frame_indices": SELECTED_FRAMES,
                "selected_time_seconds": [round((frame - 1) * 0.1, 1) for frame in SELECTED_FRAMES],
                "trajectory": {"name": name, "start_xyz": start, "end_xyz": end},
                "camera": {
                    "intrinsics": [600.0, 600.0, 320.0, 180.0, 60.0],
                    "image_size": [640, 360],
                    "boresight": "+x world axis, pitched 60 degrees downward (fixed renderer capability)",
                },
                "objects": object_layout(stations, sep, overlap),
                "person_motion": {"radius_m": 1.0, "omega_rad_s": 0.45, "phase_offset_rad": round(0.17 * (seed % 23), 4)},
                "environment": {"fog_percent": float(fog), "illumination_lux": float(illumination), "camera_noise": float(noise)},
                "diversity_factors": factors,
                "ground_truth_version": GT_VERSION,
            }
        )
    return scenarios


def trajectory_hash(scenario: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes({"start": scenario["trajectory"]["start_xyz"], "end": scenario["trajectory"]["end_xyz"]})).hexdigest().upper()


def layout_hash(scenario: dict[str, Any]) -> str:
    identity = [{"class_id": x["class_id"], "xy": x["xy"], "radius_height": x["radius_height"]} for x in scenario["objects"]]
    return hashlib.sha256(canonical_bytes(identity)).hexdigest().upper()


def hardware_info() -> dict[str, Any]:
    cpu_name = platform.processor()
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        pass
    gpu: dict[str, Any] = {}
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        name, memory_mib, driver = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
        gpu = {"name": name, "memory_total_mib": int(memory_mib), "driver_version": driver}
    except Exception as exc:
        gpu = {"query_error": str(exc)}
    memory = psutil.virtual_memory()
    return {
        "cpu": {"name": cpu_name, "physical_cores": psutil.cpu_count(logical=False), "logical_processors": psutil.cpu_count(logical=True)},
        "gpu": gpu,
        "system_memory": {"total_bytes": memory.total, "total_gib": memory.total / 1024**3},
        "platform": platform.platform(),
        "python": sys.version,
    }


def existing_state_audit() -> dict[str, Any]:
    old_plan_path = EXPERIMENT_ROOT / "training" / "scenario_plan.json"
    old_plan = json.loads(old_plan_path.read_text(encoding="utf-8"))
    manifest_path = OLD_TRAINING_ROOT / "frame_manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        frame_rows = list(csv.DictReader(handle))
    args_path = OLD_TRAINING_ROOT / "runs" / "full_yolov8s_seed42" / "args.yaml"
    fixed_plan = json.loads(FIXED_PLAN.read_text(encoding="utf-8"))
    preservation_paths = [
        old_plan_path,
        manifest_path,
        args_path,
        OLD_BEST,
        FIXED_PLAN,
        FIXED_EVAL_ROOT / "aggregated" / FIXED_SESSION / "multi_scenario_results.json",
        FIXED_EVAL_ROOT / "aggregated" / FIXED_SESSION / "all_iteration_results.csv",
    ]
    fixed_inputs = []
    for scenario in fixed_plan["scenarios"]:
        manifest = FIXED_SCENARIOS / scenario["scenario_id"] / "initial_condition" / "frame_manifest.json"
        frames = FIXED_SCENARIOS / scenario["scenario_id"] / "initial_condition" / "frames"
        preservation_paths.extend([Path(scenario["scenario_config_path"]), manifest])
        fixed_inputs.append(
            {
                "scenario_id": scenario["scenario_id"],
                "scenario_config_path": scenario["scenario_config_path"],
                "scenario_config_sha256": file_sha256(Path(scenario["scenario_config_path"])),
                "initial_manifest_path": str(manifest.resolve()),
                "initial_manifest_sha256": file_sha256(manifest),
                "rendered_image_count": len(list(frames.glob("frame_*.png"))),
            }
        )
    usage = shutil.disk_usage(REPO_ROOT)
    split_counts: dict[str, int] = {}
    scenario_counts: dict[str, int] = {}
    for row in frame_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        scenario_counts[row["scenario_id"]] = scenario_counts.get(row["scenario_id"], 0) + 1
    return {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(REPO_ROOT),
        "branch": subprocess.run(["git", "branch", "--show-current"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        "old_dataset": {
            "plan_path": str(old_plan_path.resolve()),
            "plan_id": old_plan["plan_id"],
            "train_scenario_count": sum(x["split"] == "train" for x in old_plan["scenarios"]),
            "validation_scenario_count": sum(x["split"] == "val" for x in old_plan["scenarios"]),
            "frame_stride": old_plan["frame_stride"],
            "nominal_frame_interval_seconds": old_plan["frame_stride"] * 0.1,
            "split_policy": old_plan["split_policy"],
            "image_counts": split_counts,
            "scenario_image_counts": scenario_counts,
            "manifest_path": str(manifest_path.resolve()),
        },
        "old_training": {
            "args_path": str(args_path.resolve()),
            "initial_weights": str((REPO_ROOT / "yolov8s.pt").resolve()),
            "selected_best_path": str(OLD_BEST.resolve()),
            "selected_best_size_bytes": OLD_BEST.stat().st_size,
            "selected_best_sha256": file_sha256(OLD_BEST),
        },
        "fixed_S0_S4": {
            "plan_path": str(FIXED_PLAN.resolve()),
            "plan_sha256": file_sha256(FIXED_PLAN),
            "scenario_inputs": fixed_inputs,
            "prior_run_path": str((FIXED_EVAL_ROOT / "runs" / FIXED_SESSION).resolve()),
            "prior_aggregate_path": str((FIXED_EVAL_ROOT / "aggregated" / FIXED_SESSION).resolve()),
            "prior_report_path": str((FIXED_EVAL_ROOT / "reports" / FIXED_SESSION).resolve()),
        },
        "storage": {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free, "free_gib": usage.free / 1024**3},
        "hardware": hardware_info(),
        "preservation_snapshot": [
            {"path": str(path.resolve()), "exists": path.exists(), "sha256": file_sha256(path) if path.is_file() else None}
            for path in preservation_paths
        ],
        "policy": "Existing training data, weights, fixed S0-S4 inputs, runs, aggregates, and reports are read-only inputs for this experiment.",
    }


def register() -> dict[str, Any]:
    if CONFIG_ROOT.exists():
        raise FileExistsError(f"Plan ID already exists and is immutable: {CONFIG_ROOT}")
    CONFIG_ROOT.mkdir(parents=True)
    scenarios = definitions()
    if len(scenarios) != 23 or sum(x["split"] == "train" for x in scenarios) != 18 or sum(x["split"] == "val" for x in scenarios) != 5:
        raise AssertionError("The registered design must contain 18 train and 5 validation scenarios")
    ids = [x["scenario_id"] for x in scenarios]
    seeds = [x["seed"] for x in scenarios]
    traj = [trajectory_hash(x) for x in scenarios]
    layouts = [layout_hash(x) for x in scenarios]
    if any(len(values) != len(set(values)) for values in (ids, seeds, traj, layouts)):
        raise ValueError("Scenario IDs, seeds, trajectories, and layouts must each be unique")
    fixed = json.loads(FIXED_PLAN.read_text(encoding="utf-8"))
    fixed_seeds = {int(x["seed"]) for x in fixed["scenarios"]}
    fixed_traj = {
        hashlib.sha256(canonical_bytes({"start": x["uav_initial_xyz"], "end": x["uav_end_xyz"]})).hexdigest().upper()
        for x in fixed["scenarios"]
    }
    fixed_layouts = {
        hashlib.sha256(canonical_bytes([{"class_id": o["class_id"], "xy": o["xy"], "radius_height": o["radius_height"]} for o in x["object_layout"]])).hexdigest().upper()
        for x in fixed["scenarios"]
    }
    if set(seeds) & fixed_seeds or set(traj) & fixed_traj or set(layouts) & fixed_layouts:
        raise ValueError("New scenario design overlaps fixed S0-S4 identity")
    records = []
    for scenario in scenarios:
        config_path = CONFIG_ROOT / "scenario_configs" / f"{scenario['scenario_id']}.json"
        data = canonical_bytes(scenario)
        write_new(config_path, data)
        delta = [scenario["trajectory"]["end_xyz"][i] - scenario["trajectory"]["start_xyz"][i] for i in range(3)]
        norm = math.sqrt(sum(value * value for value in delta))
        records.append(
            {
                "scenario_id": scenario["scenario_id"],
                "split": scenario["split"],
                "seed": scenario["seed"],
                "trajectory_name": scenario["trajectory"]["name"],
                "start_xyz": scenario["trajectory"]["start_xyz"],
                "end_xyz": scenario["trajectory"]["end_xyz"],
                "flight_direction_unit": [value / norm for value in delta],
                "altitude_range_m": sorted([scenario["trajectory"]["start_xyz"][2], scenario["trajectory"]["end_xyz"][2]]),
                "camera": scenario["camera"],
                "object_count": len(scenario["objects"]),
                "person_count": sum(x["class_id"] == 1 for x in scenario["objects"]),
                "vehicle_count": sum(x["class_id"] == 2 for x in scenario["objects"]),
                "object_layout": scenario["objects"],
                "environment": scenario["environment"],
                "selected_frame_indices": scenario["selected_frame_indices"],
                "selected_time_seconds": scenario["selected_time_seconds"],
                "diversity_factors": scenario["diversity_factors"],
                "trajectory_sha256": trajectory_hash(scenario),
                "object_layout_sha256": layout_hash(scenario),
                "scenario_config_path": str(config_path.resolve()),
                "scenario_config_sha256": file_sha256(config_path),
            }
        )
    plan = {
        "schema_version": "1.0",
        "plan_id": PLAN_ID,
        "supersedes_failed_plan": "diverse_training_scenario_split_v1",
        "pre_YOLO_correction": "The v1 visible-pixel GT gate failed for nine camera-frustum-limited lateral/retreat scenarios. Before any YOLO run, only their endpoint distance and/or forward object x positions were reduced; failed v1 plans, renders, manifests, labels, and audits remain preserved.",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_rule": "Frozen before dataset rendering, training, validation, and fixed S0-S4 inference; no detector result may alter this plan.",
        "ground_truth_version": GT_VERSION,
        "class_names": {"0": "person", "1": "vehicle"},
        "source_frame_count_per_scenario": 181,
        "selected_frame_count_per_scenario": 20,
        "selected_frame_indices": SELECTED_FRAMES,
        "frame_interval_pattern": [b - a for a, b in zip(SELECTED_FRAMES, SELECTED_FRAMES[1:])],
        "split_policy": "Each scenario belongs wholly to train or validation. Fixed S0-S4 remain evaluation-only.",
        "training_scenario_count": 18,
        "validation_scenario_count": 5,
        "training_image_target": 360,
        "validation_image_target": 100,
        "fixed_evaluation_plan_path": str(FIXED_PLAN.resolve()),
        "fixed_evaluation_plan_sha256": file_sha256(FIXED_PLAN),
        "scenarios": records,
    }
    plan_path = CONFIG_ROOT / "scenario_plan.json"
    write_new(plan_path, canonical_bytes(plan))
    csv_path = CONFIG_ROOT / "scenario_plan.csv"
    write_csv_new(csv_path, records)
    audit = existing_state_audit()
    audit_path = CONFIG_ROOT / "existing_state_audit.json"
    write_new(audit_path, canonical_bytes(audit))
    preregistration = {
        "schema_version": "1.0",
        "plan_id": PLAN_ID,
        "supersedes_failed_plan": "diverse_training_scenario_split_v1",
        "failed_plan_artifacts": {
            "config_root": str((HERE / "config" / "diverse_training_scenario_split_v1").resolve()),
            "data_root": str((HERE / "data" / "diverse_training_scenario_split_v1").resolve()),
            "training_gate_sha256": file_sha256(HERE / "data" / "diverse_training_scenario_split_v1" / "training_gate.json"),
            "training_was_started": False,
        },
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_plan_path": str(plan_path.resolve()),
        "scenario_plan_sha256": file_sha256(plan_path),
        "scenario_plan_csv_path": str(csv_path.resolve()),
        "scenario_plan_csv_sha256": file_sha256(csv_path),
        "existing_state_audit_path": str(audit_path.resolve()),
        "existing_state_audit_sha256": file_sha256(audit_path),
        "scenario_config_hashes": {row["scenario_id"]: row["scenario_config_sha256"] for row in records},
        "training_configuration": {
            "model": "YOLOv8s two-class detector",
            "initial_weights": str((REPO_ROOT / "yolov8s.pt").resolve()),
            "initial_weights_sha256": file_sha256(REPO_ROOT / "yolov8s.pt"),
            "imgsz": 640,
            "batch": 4,
            "optimizer": "SGD",
            "epochs": 50,
            "patience": 12,
            "seed": 42,
            "selection_data": "five validation scenarios only",
        },
        "supported_diversification": ["trajectory start/end", "flight axis", "ascending/descending altitude", "object placement and spacing", "person walking phase", "camera-object distance", "fog", "illumination", "camera noise"],
        "unsupported_diversification": {
            "camera_direction_and_tilt": "The current 3-D renderer fixes the boresight to world +x and reads a fixed 60-degree downward pitch from intrinsics.",
            "multiple_terrain_or_background_layouts": "The current workspace builder provides one deterministic terrain and scenery layout.",
        },
        "test_data_firewall": "S0-S4 results are forbidden for training termination, weight selection, or parameter adjustment.",
    }
    prereg_path = CONFIG_ROOT / "pre_registration.json"
    write_new(prereg_path, canonical_bytes(preregistration))
    sidecar = f"{file_sha256(prereg_path)}  pre_registration.json\n"
    write_new(CONFIG_ROOT / "pre_registration.sha256", sidecar.encode("ascii"))
    return {
        "plan_id": PLAN_ID,
        "plan": str(plan_path.resolve()),
        "plan_sha256": file_sha256(plan_path),
        "pre_registration": str(prereg_path.resolve()),
        "pre_registration_sha256": file_sha256(prereg_path),
        "existing_audit": str(audit_path.resolve()),
        "free_gib": audit["storage"]["free_gib"],
    }


def main() -> int:
    result = register()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
