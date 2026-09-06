from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY = Path(r"C:\Users\lab\Counterfactual-XAI-Verifier")
WORKTREE = REPOSITORY / ".k"
DIRECT = WORKTREE / "final_kci_direct_yolo_shap_v1"
PREPARATION = DIRECT / "independent_shap_variable_validation_v1_preparation"
OUTPUT = DIRECT / "independent_shap_variable_validation_run_001_lab"
CONFIG_ROOT = PREPARATION / "scenario_configs"
V3 = REPOSITORY / "experiments" / "yolov8_search_comparison" / "shap_counterfactual_boundary_v3"
PILOT = DIRECT / "pilot_run_001"
PROSPECTIVE = DIRECT / "prospective_top_variable_pilot_run_001_lab"
PROTECTED_SOURCE = DIRECT / "ordered_shap_search_development_v1_preparation" / "protected_hashes_before.csv"

EXPECTED_WEIGHT_SHA256 = "E141B3DC0104CBC03AD2EE573FFA451C369C9D93D8690E560E07C146078F5D9E"
P = {"fog_percent": 5.0, "illumination_lux": 12000.0, "camera_noise": 0.02}
Q = {"fog_percent": 76.0, "illumination_lux": 2450.0, "camera_noise": 0.49}
VALIDATION = {
    "fog": {"fog_percent": 70.0, "illumination_lux": 12000.0, "camera_noise": 0.02},
    "illumination": {"fog_percent": 5.0, "illumination_lux": 4640.0, "camera_noise": 0.02},
    "noise": {"fog_percent": 5.0, "illumination_lux": 12000.0, "camera_noise": 0.42},
}
VARIABLES = ("fog", "illumination", "noise")
ENV_KEYS = ("fog_percent", "illumination_lux", "camera_noise")
TOL = 1e-12
NOW = datetime.now(timezone.utc).isoformat()


SCENARIO_SPECS = [
    ("IVS00", 9700, [-52, -17, 66], [8, -4, 54], 61, [(-18, -13), (7, -9), (33, -5)], 0.92, 0.72, 0.25, "descending southwest approach; medium-small objects"),
    ("IVS01", 9701, [-47, 16, 52], [12, 5, 64], 56, [(-15, 13), (9, 9), (36, 5)], 1.06, 1.05, 0.85, "ascending northeast approach; medium-large objects"),
    ("IVS02", 9702, [-50, -2, 72], [5, 10, 62], 68, [(-20, 0), (5, 4), (30, 8)], 0.88, 1.48, 1.45, "high descending lateral approach; small objects"),
    ("IVS03", 9703, [-42, 19, 48], [15, 0, 55], 52, [(-10, 15), (12, 9), (38, 3)], 1.14, 0.88, 2.05, "low ascending cross-track approach; large objects"),
    ("IVS04", 9704, [-54, 7, 60], [3, -10, 60], 59, [(-19, 5), (6, -1), (31, -7)], 0.98, 1.22, 2.65, "level crossing trajectory; nominal objects"),
    ("IVS05", 9705, [-36, -20, 45], [18, -6, 51], 48, [(-8, -13), (14, -9), (39, -5)], 1.18, 0.64, 3.10, "near low ascending approach; largest planned objects"),
    ("IVS06", 9706, [-51, 13, 75], [-2, -8, 67], 69, [(-21, 10), (3, 3), (28, -5)], 0.88, 1.72, 3.70, "far high descending approach; small objects"),
    ("IVS07", 9707, [-46, -12, 58], [14, 14, 49], 57, [(-15, -9), (10, 0), (35, 9)], 1.08, 0.96, 4.25, "south-to-north crossing descent; medium-large objects"),
    ("IVS08", 9708, [-39, 4, 50], [17, -14, 62], 54, [(-12, 2), (13, -5), (39, -11)], 1.12, 1.34, 4.90, "north-to-south crossing ascent; large objects"),
    ("IVS09", 9709, [-53, -6, 69], [9, 8, 57], 64, [(-19, -4), (6, 1), (32, 6)], 0.94, 1.58, 5.45, "high diagonal descent; medium-small objects"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv_new(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def object_rows(centers: list[tuple[int, int]], scale: float) -> list[dict[str, Any]]:
    people_r = (0.44, 0.49, 0.54)
    people_h = (1.68, 1.76, 1.84)
    vehicle_r = (1.00, 1.12, 1.24)
    vehicle_h = (1.42, 1.52, 1.62)
    offsets = ((2.8, 1.8), (3.1, -1.7), (3.4, 1.6))
    rows: list[dict[str, Any]] = []
    object_id = 1
    for index, center in enumerate(centers):
        rows.append({
            "object_id": object_id,
            "class_id": 1,
            "class_name": "person",
            "xy": [float(center[0]), float(center[1])],
            "radius_height": [round(people_r[index] * scale, 6), round(people_h[index] * scale, 6)],
        })
        object_id += 1
        rows.append({
            "object_id": object_id,
            "class_id": 2,
            "class_name": "vehicle",
            "xy": [round(center[0] + offsets[index][0], 6), round(center[1] + offsets[index][1], 6)],
            "radius_height": [round(vehicle_r[index] * scale, 6), round(vehicle_h[index] * scale, 6)],
        })
        object_id += 1
    return rows


def make_scenario(spec: tuple[Any, ...]) -> dict[str, Any]:
    scenario_id, seed, start, end, pitch, centers, scale, motion_radius, phase, rationale = spec
    return {
        "schema_version": "1.0",
        "scenario_id": scenario_id,
        "seed": seed,
        "split": "independent_direct_shap_variable_validation",
        "duration_seconds": 18.0,
        "total_frames": 181,
        "trajectory": {"name": f"locked independent direct-Shapley trajectory {scenario_id}", "start_xyz": start, "end_xyz": end},
        "camera": {
            "boresight": "+x world axis, pitched downward",
            "image_size": [640, 360],
            "intrinsics": [600.0, 600.0, 320.0, 180.0, float(pitch)],
        },
        "objects": object_rows(centers, scale),
        "person_motion": {"radius_m": motion_radius, "omega_rad_s": round(0.60 + 0.06 * (seed - 9700), 6), "phase_offset_rad": phase},
        "gt_version": "rendered_instance_mask_v1",
        "terrain_condition": "fixed geometry_reference_v1 terrain; renderer does not expose a scenario-level terrain selector",
        "diversity_factors": [rationale, f"object_scale={scale}", f"camera_pitch={pitch}"],
        "selection_basis": "result-blind scene metadata only; no normal or degraded mAP used",
    }


def structure_core(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "duration_seconds": config.get("duration_seconds"),
        "total_frames": config.get("total_frames"),
        "trajectory": {"start_xyz": config["trajectory"]["start_xyz"], "end_xyz": config["trajectory"]["end_xyz"]},
        "camera": config["camera"],
        "objects": config["objects"],
        "person_motion": config["person_motion"],
        "gt_version": config.get("gt_version"),
        "terrain_condition": config.get("terrain_condition"),
    }


def discover_existing_scenario_configs() -> list[dict[str, Any]]:
    roots = [
        REPOSITORY / "experiments" / "yolov8_search_comparison",
        WORKTREE / "experiments" / "yolov8_search_comparison",
        WORKTREE / "final_kci_shap_guidance_v1",
        WORKTREE / "final_kci_shap_validity_confirmatory_v1",
    ]
    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            resolved = path.resolve()
            if resolved in seen or PREPARATION in path.parents:
                continue
            seen.add(resolved)
            try:
                text = path.read_text(encoding="utf-8-sig")
                if '"trajectory"' not in text or '"objects"' not in text:
                    continue
                value = json.loads(text)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and all(key in value for key in ("scenario_id", "trajectory", "camera", "objects", "person_motion")):
                rows.append({
                    "path": str(path.resolve()),
                    "scenario_id": str(value["scenario_id"]),
                    "file_sha256": sha256(path),
                    "structure_sha256": value_hash(structure_core(value)),
                })
    return rows


def renderer_transfer(env: dict[str, float]) -> dict[str, float]:
    low_light = max(0.0, (3000.0 - env["illumination_lux"]) / 3000.0)
    return {
        "fog_norm": max(0.0, min(1.0, env["fog_percent"] / 100.0)),
        "low_light": low_light,
        "illumination_multiplier": max(0.10, 1.0 - 0.85 * low_light) if low_light > 0 else 1.0,
        "noise_amplitude": env["camera_noise"] * 0.40,
    }


def severity(env: dict[str, float]) -> dict[str, float]:
    return {
        "fog": env["fog_percent"] / 100.0,
        "illumination": (15000.0 - env["illumination_lux"]) / 14800.0,
        "noise": env["camera_noise"] / 0.6,
    }


def cube_environment(mask: str) -> dict[str, float]:
    return {
        "fog_percent": Q["fog_percent"] if mask[0] == "1" else P["fog_percent"],
        "illumination_lux": Q["illumination_lux"] if mask[1] == "1" else P["illumination_lux"],
        "camera_noise": Q["camera_noise"] if mask[2] == "1" else P["camera_noise"],
    }


def setting_identity(scene_hash: str, condition_id: str, env: dict[str, float], source_hashes: dict[str, str]) -> str:
    return value_hash({
        "condition_id": condition_id,
        "scenario_config_sha256": scene_hash,
        "environment": env,
        "model_seed": 42,
        "weights_sha256": EXPECTED_WEIGHT_SHA256,
        "evaluation_core_sha256": source_hashes["evaluation_core"],
        "renderer_sha256": source_hashes["renderer"],
        "exporter_sha256": source_hashes["exporter"],
        "geometry_reference_sha256": source_hashes["geometry_reference"],
        "detector_settings": {"input_size": 640, "confidence_threshold": 0.25, "nms_iou_threshold": 0.7, "ap_iou_threshold": 0.5, "batch_size": 16, "device": "cuda:0"},
    })


def build_conditions(scenarios: list[dict[str, Any]], source_hashes: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scenarios:
        for mask in ("000", "001", "010", "011", "100", "101", "110", "111"):
            env = cube_environment(mask)
            condition_id = f"{scene['scenario_id']}__CUBE_{mask}"
            transfer = renderer_transfer(env)
            rows.append({
                "condition_id": condition_id,
                "scenario_id": scene["scenario_id"],
                "scenario_seed": scene["seed"],
                "scenario_config_sha256": scene["scenario_config_sha256"],
                "role": "exact_shapley_cube",
                "cube_mask": mask,
                "active_q_variables": "+".join(VARIABLES[index] for index, bit in enumerate(mask) if bit == "1") or "P_baseline",
                "validation_variable": "",
                **env,
                **{f"{key}_severity": value for key, value in severity(env).items()},
                **{f"renderer_{key}": value for key, value in transfer.items()},
                "reuse_candidate": False,
                "reuse_reason": "new independent scenario ID and new scenario-config hash",
                "setting_identity_sha256": setting_identity(scene["scenario_config_sha256"], condition_id, env, source_hashes),
            })
        for variable in VARIABLES:
            env = VALIDATION[variable]
            condition_id = f"{scene['scenario_id']}__VALID_{variable.upper()}_S070"
            transfer = renderer_transfer(env)
            rows.append({
                "condition_id": condition_id,
                "scenario_id": scene["scenario_id"],
                "scenario_seed": scene["seed"],
                "scenario_config_sha256": scene["scenario_config_sha256"],
                "role": "separate_actual_rank_validation",
                "cube_mask": "",
                "active_q_variables": "",
                "validation_variable": variable,
                **env,
                **{f"{key}_severity": value for key, value in severity(env).items()},
                **{f"renderer_{key}": value for key, value in transfer.items()},
                "reuse_candidate": False,
                "reuse_reason": "new independent scenario ID and new scenario-config hash",
                "setting_identity_sha256": setting_identity(scene["scenario_config_sha256"], condition_id, env, source_hashes),
            })
    return rows


def protected_baseline() -> list[dict[str, Any]]:
    rows = read_csv(PROTECTED_SOURCE)
    if len(rows) != 129:
        raise RuntimeError(f"Expected 129 protected files, got {len(rows)}")
    output: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["path"])
        expected = row.get("expected_sha256") or row.get("sha256_before")
        if not expected:
            raise RuntimeError(f"Missing protected hash column: {path}")
        actual = sha256(path) if path.is_file() else ""
        output.append({"path": str(path), "expected_sha256": expected.upper(), "actual_sha256": actual, "exists": path.is_file(), "matches": actual == expected.upper()})
    if not all(row["matches"] for row in output):
        raise RuntimeError("Protected files changed before independent-validation preparation")
    return output


def measured_cost_basis() -> dict[str, Any]:
    event_path = PILOT / "stage_events.jsonl"
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = datetime.fromisoformat(events[0]["at_utc"])
    end = datetime.fromisoformat(events[-1]["at_utc"])
    elapsed_minutes = (end - start).total_seconds() / 60.0
    render_sizes = []
    for folder in (PILOT / "cache" / "renders").iterdir():
        if folder.is_dir():
            render_sizes.append(sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()))
    render_sizes.sort()
    median_bytes = render_sizes[len(render_sizes) // 2] if render_sizes else 0
    return {
        "source_stage_events": str(event_path.resolve()),
        "source_completed_conditions": 33,
        "source_new_renders": 30,
        "source_elapsed_minutes": elapsed_minutes,
        "linear_runtime_estimate_minutes_for_110_new": elapsed_minutes * 110.0 / 30.0,
        "reported_runtime_range_hours": [2.3, 3.2],
        "hard_overall_timeout_seconds": 14400,
        "source_render_cache_directories": len(render_sizes),
        "source_median_render_bytes": median_bytes,
        "median_storage_estimate_gb_for_110": median_bytes * 110.0 / (1024.0 ** 3),
        "conservative_storage_planning_gb": 11.0,
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"Result folder already exists: {OUTPUT}")
    required_new = [
        "experiment_plan_ko.md", "independent_scenarios.csv", "independent_conditions.csv", "independent_conditions.json",
        "locked_gate_definition.json", "fixed_settings.json", "source_manifest.csv", "protected_hashes_before.csv", "README.md",
    ]
    for name in required_new:
        if (PREPARATION / name).exists():
            raise FileExistsError(PREPARATION / name)
    if CONFIG_ROOT.exists():
        raise FileExistsError(CONFIG_ROOT)

    if read_json(DIRECT / "pilot_plan.json").get("Q") != Q or read_json(DIRECT / "pilot_plan.json").get("P") != P:
        raise RuntimeError("Actual pilot P/Q lock differs from the requested expected values")
    prospective = read_json(PROSPECTIVE / "prospective_gate_results.json")
    if prospective.get("overall_status") != "PROSPECTIVE_TOP_VARIABLE_PILOT_PASS" or prospective.get("shapley_recalculated") is not False:
        raise RuntimeError("Prospective direct-Shapley evidence is not the confirmed locked result")

    weight_path = Path(next(row["path"] for row in read_json(DIRECT / "pilot_lock.json")["source_hashes"] if row["sha256"] == EXPECTED_WEIGHT_SHA256))
    if sha256(weight_path) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Seed-42 weights hash mismatch")
    source_files = {
        "evaluation_core": V3 / "evaluation_core.py",
        "renderer": V3 / "matlab" / "eo_camera_3d_render.m",
        "exporter": V3 / "matlab" / "export_v3_scenario_frames.m",
        "render_effects": V3 / "matlab" / "render_eo_image.m",
        "geometry_reference": REPOSITORY / "experiments" / "yolov8_search_comparison" / "geometry_reference_v1.mat",
    }
    source_hashes = {name: sha256(path) for name, path in source_files.items()}

    existing = discover_existing_scenario_configs()
    existing_ids = {row["scenario_id"] for row in existing}
    existing_structures = {row["structure_sha256"] for row in existing}
    proposed_configs = [make_scenario(spec) for spec in SCENARIO_SPECS]
    if len(proposed_configs) != 10 or len({row["scenario_id"] for row in proposed_configs}) != 10:
        raise RuntimeError("Exactly ten unique independent scenario IDs are required")
    if any(row["scenario_id"] in existing_ids for row in proposed_configs):
        raise RuntimeError("A proposed independent scenario ID was previously used")
    proposed_structures = [value_hash(structure_core(row)) for row in proposed_configs]
    if len(set(proposed_structures)) != 10 or any(value in existing_structures for value in proposed_structures):
        raise RuntimeError("A proposed scenario structure duplicates an existing or another proposed scenario")

    CONFIG_ROOT.mkdir(parents=True, exist_ok=False)
    scenario_rows: list[dict[str, Any]] = []
    locked_scenarios: list[dict[str, Any]] = []
    for index, config in enumerate(proposed_configs, start=1):
        config_path = CONFIG_ROOT / f"{config['scenario_id']}.json"
        write_json_new(config_path, config)
        config_hash = sha256(config_path)
        structure_hash = value_hash(structure_core(config))
        objects = config["objects"]
        people = [row for row in objects if row["class_name"] == "person"]
        vehicles = [row for row in objects if row["class_name"] == "vehicle"]
        centroid_x = sum(row["xy"][0] for row in objects) / len(objects)
        centroid_y = sum(row["xy"][1] for row in objects) / len(objects)
        start = config["trajectory"]["start_xyz"]
        end = config["trajectory"]["end_xyz"]
        midpoint = [(start[i] + end[i]) / 2.0 for i in range(3)]
        horizontal_ranges = [math.hypot(point[0] - centroid_x, point[1] - centroid_y) for point in (start, midpoint, end)]
        locked = {"scenario_id": config["scenario_id"], "seed": config["seed"], "scenario_config_path": str(config_path.resolve()), "scenario_config_sha256": config_hash, "structure_sha256": structure_hash}
        locked_scenarios.append(locked)
        scenario_rows.append({
            "independent_validation_id": f"IVV{index:02d}",
            **locked,
            "selection_basis": "result-blind metadata only",
            "prior_scenario_id_found": False,
            "exact_structure_duplicate_found": False,
            "existing_config_records_compared": len(existing),
            "start_x": start[0], "start_y": start[1], "start_z": start[2],
            "end_x": end[0], "end_y": end[1], "end_z": end[2],
            "altitude_change_m": end[2] - start[2],
            "camera_pitch_deg": config["camera"]["intrinsics"][4],
            "person_count": len(people), "vehicle_count": len(vehicles),
            "mean_person_height_m": sum(row["radius_height"][1] for row in people) / len(people),
            "mean_vehicle_radius_m": sum(row["radius_height"][0] for row in vehicles) / len(vehicles),
            "object_centroid_x": centroid_x, "object_centroid_y": centroid_y,
            "horizontal_range_start_m": horizontal_ranges[0], "horizontal_range_mid_m": horizontal_ranges[1], "horizontal_range_end_m": horizontal_ranges[2],
            "person_motion_radius_m": config["person_motion"]["radius_m"],
            "person_motion_omega_rad_s": config["person_motion"]["omega_rad_s"],
            "static_renderer_schema_valid": True,
            "static_gt_schema_valid": all(row["object_id"] == position and row["class_id"] in {1, 2} for position, row in enumerate(objects, start=1)),
            "actual_normal_rendering_not_yet_verified": True,
            "actual_gt_visibility_not_yet_verified": True,
        })

    conditions = build_conditions(locked_scenarios, source_hashes)
    if len(conditions) != 110 or len({row["condition_id"] for row in conditions}) != 110:
        raise RuntimeError("Condition ledger must contain 110 unique IDs")
    for scene in locked_scenarios:
        own = [row for row in conditions if row["scenario_id"] == scene["scenario_id"]]
        masks = {row["cube_mask"] for row in own if row["role"] == "exact_shapley_cube"}
        validations = {row["validation_variable"] for row in own if row["role"] == "separate_actual_rank_validation"}
        envs = {canonical_json({key: row[key] for key in ENV_KEYS}) for row in own}
        if len(own) != 11 or masks != {"000", "001", "010", "011", "100", "101", "110", "111"} or validations != set(VARIABLES) or len(envs) != 11:
            raise RuntimeError(f"Condition completeness/uniqueness failed: {scene['scenario_id']}")

    protected = protected_baseline()
    cost = measured_cost_basis()
    fixed_settings = {
        "schema_version": "1.0", "locked_at_utc": NOW, "status": "LOCKED_BEFORE_ANY_INDEPENDENT_RESULT",
        "worktree": str(WORKTREE), "git_branch": "kci", "scenes": locked_scenarios,
        "P": P, "Q": Q, "validation_severity": 0.70, "validation_conditions": VALIDATION,
        "severity_transform": {"fog": "fog_percent/100", "illumination": "(15000-illumination_lux)/14800", "noise": "camera_noise/0.6"},
        "renderer_transfer": {
            "fog": "fog_norm=clip(fog_percent/100,0,1); blend weight=0.85*fog_norm",
            "illumination": "low_light=max(0,(3000-lux)/3000); multiplier=max(0.10,1-0.85*low_light)",
            "noise": "noise_amplitude=0.40*camera_noise",
            "P": renderer_transfer(P), "Q": renderer_transfer(Q),
            "validation": {name: renderer_transfer(env) for name, env in VALIDATION.items()},
        },
        "exact_shapley": {
            "function": "D(S)=mAP(P)-mAP(x_S)", "variables_in_mask_order": list(VARIABLES), "coalitions": 8,
            "formula": "phi_i=sum_{S subseteq N\\{i}} |S|!(3-|S|-1)!/3! * [D(S union {i})-D(S)]",
            "independent_recalculation": "mean marginal contribution across all 3!=6 permutations",
            "positive_means": "actual mAP degradation", "sign_preserved": True,
            "additivity_and_cross_calculation_tolerance": 1e-12,
        },
        "detector": {"architecture": "YOLOv8s", "model_seed": 42, "weights_path": str(weight_path.resolve()), "weights_sha256": EXPECTED_WEIGHT_SHA256, "input_size": 640, "confidence_threshold": 0.25, "nms_iou_threshold": 0.7, "ap_iou_threshold": 0.5, "batch_size": 16, "device": "cuda:0"},
        "source_files": {name: {"path": str(path.resolve()), "sha256": source_hashes[name]} for name, path in source_files.items()},
        "execution": {"account": r"DESKTOP-CMOIPGE\lab", "MATLAB_PREFDIR_must_be_absent": True, "matlab_options": "-nodesktop -nosplash", "matlab_startup_timeout_seconds": 300, "render_timeout_per_condition_seconds": 300, "yolo_timeout_per_condition_seconds": 120, "condition_timeout_seconds": 420, "overall_timeout_seconds": 14400, "automatic_retry": False, "unrelated_process_termination": False},
        "logical_conditions": 110, "frames_per_condition": 181, "logical_frames": 19910, "reuse_candidates": 0,
        "measured_cost_basis": cost,
        "excluded_scenes": ["DTR0", "F02", "F09", "H00-H09", "G00-G04", "all previously rendered training/validation/test scenario IDs"],
        "scientific_scope": "independent validation of direct exact-Shapley top-variable selection only; no boundary-search efficiency comparison",
        "RF_or_surrogate_or_virtual_map_used": False,
    }
    write_json_new(PREPARATION / "fixed_settings.json", fixed_settings)

    gate = {
        "schema_version": "1.0", "locked_at_utc": NOW, "status": "LOCKED_BEFORE_ANY_INDEPENDENT_RESULT",
        "scene_count": 10, "conditions_per_scene": 11, "total_conditions": 110,
        "technical_requirements": ["all 110 results and completion manifests verified", "all scene/config/environment/weight/evaluation hashes match", "protected-file mismatches=0", "exact Shapley additivity and independent permutation recalculation pass"],
        "scene_rule": {
            "shapley_top_tolerance": 1e-12, "actual_tie_tolerance_map": 0.01,
            "MATCH": "actual first exceeds second by >0.01 mAP and the unique Shapley top is the actual top",
            "INCONCLUSIVE": "actual first-second difference <=0.01 mAP or the Shapley top is not unique within 1e-12",
            "MISMATCH": "a variable other than the unique Shapley top causes >0.01 mAP more degradation",
        },
        "overall_rule": {
            "INDEPENDENT_SHAP_VARIABLE_SELECTION_PASS": "technical requirements pass and MATCH>=7 of 10",
            "INDEPENDENT_SHAP_VARIABLE_SELECTION_FAIL": "technical requirements pass, MATCH<7, and at least one MISMATCH",
            "INDEPENDENT_SHAP_VARIABLE_SELECTION_INCONCLUSIVE": "technical requirements pass, MATCH<7, MISMATCH=0, and the shortfall is due only to INCONCLUSIVE scenes",
            "TECHNICAL_INCOMPLETE": "any technical requirement is incomplete or fails",
        },
        "auxiliary_only": ["direction agreement", "per-scene and aggregate Spearman rank correlation"],
        "post_result_rule_changes_forbidden": True,
    }
    write_json_new(PREPARATION / "locked_gate_definition.json", gate)
    write_json_new(PREPARATION / "independent_conditions.json", {"schema_version": "1.0", "locked_at_utc": NOW, "records": conditions})

    scenario_fields = list(scenario_rows[0].keys())
    condition_fields = list(conditions[0].keys())
    write_csv_new(PREPARATION / "independent_scenarios.csv", scenario_rows, scenario_fields)
    write_csv_new(PREPARATION / "independent_conditions.csv", conditions, condition_fields)
    write_csv_new(PREPARATION / "protected_hashes_before.csv", protected, ["path", "expected_sha256", "actual_sha256", "exists", "matches"])

    plan = f"""# 직접 Shapley 변수 선택 독립 검증 계획

## 목적

개발에 사용하지 않은 IVS00~IVS09에서 실제 MATLAB 영상과 seed-42 YOLOv8s mAP만 사용하여 `8개 실제 평가 → 정확 Shapley → 최상위 변수 → 별도 심각도 0.70 단독조건 → 실제 최상위 변수` 연결을 검증한다. RF·대리모델·가상 mAP는 사용하지 않는다.

## 독립 장면

10개 장면은 결과를 생성하기 전에 새 ID, 새 seed, 새 구성 해시로 고정했다. 선정에는 궤적·고도·카메라 pitch·객체 배치/크기·보행 설정만 사용했다. 기존 {len(existing)}개 시나리오 구성 기록과 비교해 ID 및 구조의 정확 중복이 없음을 확인했다. 실제 정상 렌더 가능성과 GT 가시성은 아직 실행하지 않았으므로 기술 관문에서 확인한다.

## 조건과 계산

- P: 안개 5%, 조도 12,000 lx, 잡음 0.02
- Q: 안개 76%, 조도 2,450 lx, 잡음 0.49
- 장면당 8개 P/Q 조합과 별도 검증 3조건(70%, 4,640 lx, 0.42), 총 110조건·19,910프레임
- `D(S)=mAP(P)-mAP(x_S)`이며 부호를 보존한다.
- 조합식 계산과 6개 순열 평균 계산을 독립 수행하고 1e-12 이내 일치 및 가산성을 확인한다.

## 판정

실제 1·2위 차이가 0.01 이하이거나 Shapley 최상위가 고유하지 않으면 INCONCLUSIVE다. 실제 우위가 0.01을 초과하고 최상위가 같으면 MATCH, 다른 변수가 Shapley 최상위보다 0.01 초과 큰 저하를 만들면 MISMATCH다. 기술 완료 10/10과 MATCH 7/10 이상일 때만 `INDEPENDENT_SHAP_VARIABLE_SELECTION_PASS`다.

## 실행·보존

일반 `DESKTOP-CMOIPGE\\lab` PowerShell, 기본 MATLAB 설정, 기존 렌더러·GT·평가 코드와 seed-42 가중치를 사용한다. 직접 생성한 worker만 제한적으로 관리하고 자동 재시도하지 않는다. 결과를 본 뒤 조건·기준·장면을 변경하지 않는다. 이 계획은 경계 탐색 효율을 검증하지 않는다.
"""
    (PREPARATION / "experiment_plan_ko.md").write_text(plan, encoding="utf-8")

    readme = f"""# 직접 Shapley 변수 선택 독립 검증 실행 준비

IVS00~IVS09의 새 장면 10개에서 장면당 11조건, 총 110조건을 평가하도록 잠갔다. 실제 실행 전 `static_validation.json`이 `STATIC_VALIDATION_PASS`인지 확인한다.

## 실행

다음 명령은 `DESKTOP-CMOIPGE\\lab` 일반 PowerShell에서만 실행한다. `MATLAB_PREFDIR`을 설정하지 않는다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{PREPARATION / 'run_independent_validation_lab.ps1'}"
```

자동 재시도는 없다. 명시적으로 승인된 재개만 `-Resume`을 사용하며, 결과·완료 manifest·해시가 모두 검증된 조건만 재사용한다. 첫 실행 예상은 약 {cost['linear_runtime_estimate_minutes_for_110_new']:.0f}분이며 전체 제한은 4시간이다. 중앙값 기반 저장공간 추정은 약 {cost['median_storage_estimate_gb_for_110']:.1f}GB이고 보수적으로 11GB를 확보한다.
"""
    (PREPARATION / "README.md").write_text(readme, encoding="utf-8")

    source_specs: list[tuple[str, Path, str]] = [
        ("pilot_plan", DIRECT / "pilot_plan.json", "actual locked P/Q and exact-Shapley formula"),
        ("pilot_lock", DIRECT / "pilot_lock.json", "direct-Shapley source and weight hashes"),
        ("pilot_conditions", DIRECT / "pilot_conditions.csv", "P/Q cube construction reference"),
        ("pilot_shap_scores", PILOT / "pilot_shap_scores.csv", "stored development evidence; not reused as independent results"),
        ("pilot_shap_terms", PILOT / "pilot_shap_terms.csv", "exact combinatorial calculation reference"),
        ("prospective_gate", PROSPECTIVE / "prospective_gate_results.json", "completed development-stage prospective evidence"),
        ("evaluation_core", source_files["evaluation_core"], "unchanged MATLAB/YOLO/GT/mAP evaluation code"),
        ("renderer", source_files["renderer"], "unchanged renderer"),
        ("exporter", source_files["exporter"], "unchanged 181-frame exporter"),
        ("render_effects", source_files["render_effects"], "unchanged environment transfer"),
        ("geometry_reference", source_files["geometry_reference"], "fixed terrain and geometry source used by the exporter"),
        ("weights", weight_path, "seed-42 YOLOv8s weights"),
        ("known_good_supervisor", DIRECT / "ordered_shap_search_development_v1_preparation" / "run_ordered_search_managed.py", "safe direct-Popen ownership pattern"),
        ("builder", PREPARATION / "build_preparation.py", "result-blind plan builder"),
        ("worker", PREPARATION / "independent_validation_worker.py", "future execution worker"),
        ("supervisor", PREPARATION / "run_independent_validation_managed.py", "safe future supervisor"),
        ("launcher", PREPARATION / "run_independent_validation_lab.ps1", "interactive lab launcher"),
        ("static_validator", PREPARATION / "static_validate.py", "MATLAB-free static and mock tests"),
        ("fixed_settings", PREPARATION / "fixed_settings.json", "locked execution and scientific settings"),
        ("gate", PREPARATION / "locked_gate_definition.json", "locked result interpretation"),
        ("scenarios", PREPARATION / "independent_scenarios.csv", "independent scene ledger"),
        ("conditions_csv", PREPARATION / "independent_conditions.csv", "110-condition ledger"),
        ("conditions_json", PREPARATION / "independent_conditions.json", "machine-readable condition ledger"),
        ("protected_baseline", PREPARATION / "protected_hashes_before.csv", "129-file protection ledger"),
        ("plan", PREPARATION / "experiment_plan_ko.md", "pre-result experiment plan"),
        ("readme", PREPARATION / "README.md", "execution instructions"),
    ]
    source_specs.extend(("scenario_config", Path(row["scenario_config_path"]), f"new locked independent scene {row['scenario_id']}") for row in locked_scenarios)
    source_rows = [{"source_type": kind, "path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size, "role": role} for kind, path, role in source_specs]
    write_csv_new(PREPARATION / "source_manifest.csv", source_rows, ["source_type", "path", "sha256", "size_bytes", "role"])

    print(json.dumps({
        "status": "INDEPENDENT_VALIDATION_PREPARATION_CREATED",
        "scenes": len(locked_scenarios), "conditions": len(conditions), "reuse_candidates": sum(bool(row["reuse_candidate"]) for row in conditions),
        "protected_files": len(protected), "existing_scenario_configs_compared": len(existing), "output_folder_absent": not OUTPUT.exists(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
