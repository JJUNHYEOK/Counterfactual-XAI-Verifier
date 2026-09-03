"""Pre-register and GT-audit T0--T2 before any detector inference.

Commands are deliberately separated so the plan/config hashes exist before the
MATLAB-only visibility audit, and the visibility audit exists before the YOLO
evaluation driver can run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EXP_ROOT = REPO_ROOT / "experiments" / "yolov8_search_comparison"
DIV_ROOT = EXP_ROOT / "diverse_training_scenario_split"
PLAN_ID = "new_fixed_tests_t0_t2_v1"
SUPERSEDES_PLAN: str | None = None
CORRECTION_NOTE: str | None = None
CONFIG_ROOT = HERE / "config" / PLAN_ID
SCENARIO_ROOT = HERE / "scenarios" / PLAN_ID
OUTPUT_ROOT = HERE / "outputs" / PLAN_ID
BASE_EVALUATION_CONFIG = (
    DIV_ROOT
    / "evaluation"
    / "config"
    / "evaluation_config__diverse_yolov8s_seed42__20260903_010327_583537.json"
)
BASE_LOCK = HERE / "outputs" / "final_experiment_lock_v1_complete" / "final_experiment_lock.json"
TRAINING_DATA_ROOT = DIV_ROOT / "data" / "diverse_training_scenario_split_v2"
FIXED_ROOT = EXP_ROOT / "multi_scenario_symmetric" / "scenarios" / "multi_scenario_symmetric_plan_v2"
FIXED_PLAN = (
    EXP_ROOT
    / "multi_scenario_symmetric"
    / "config"
    / "multi_scenario_symmetric_plan_v2"
    / "scenario_plan.json"
)
GT_VERSION = "rendered_instance_mask_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json_new(path: Path, value: Any, *, canonical: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if canonical
        else json.dumps(value, ensure_ascii=False, indent=2)
    )
    path.write_text(text + "\n", encoding="utf-8")


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in row.items()
                    }
                )


def camera() -> dict[str, Any]:
    return {
        "intrinsics": [600.0, 600.0, 320.0, 180.0, 60.0],
        "image_size": [640, 360],
        "boresight": "+x world axis, pitched 60 degrees downward (fixed renderer capability)",
    }


def obj(
    object_id: int,
    class_id: int,
    xy: list[float],
    radius_height: list[float],
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "class_id": class_id,
        "class_name": "person" if class_id == 1 else "vehicle",
        "xy": xy,
        "radius_height": radius_height,
    }


def scenario_definitions() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "1.0",
            "scenario_id": "T0",
            "seed": 5200,
            "duration_seconds": 18.0,
            "total_frames": 181,
            "trajectory": {
                "name": "서향 이탈·상승 원거리 관측",
                "start_xyz": [22.0, -13.0, 48.0],
                "end_xyz": [-34.0, -7.0, 64.0],
            },
            "camera": camera(),
            "objects": [
                obj(1, 1, [48.0, -16.0], [0.44, 1.65]),
                obj(2, 2, [52.5, -12.0], [1.30, 1.55]),
                obj(3, 1, [57.0, -7.0], [0.56, 1.95]),
                obj(4, 2, [62.5, -2.5], [1.05, 1.45]),
                obj(5, 1, [68.0, 2.0], [0.48, 1.75]),
                obj(6, 2, [74.0, -9.0], [1.38, 1.72]),
            ],
            "person_motion": {"radius_m": 1.8, "omega_rad_s": 0.72, "phase_offset_rad": 0.85},
            "gt_version": GT_VERSION,
            "major_changes": [
                "x축 역방향 이탈과 16 m 상승을 결합",
                "객체별 물리 크기를 달리하고 비대칭 횡간격을 사용",
                "기존보다 큰 보행 반경과 별도 위상을 사용",
            ],
            "difference_from_existing": "학습·검증·S0~S4에 없는 서향 이탈과 상승의 동시 조합",
        },
        {
            "schema_version": "1.0",
            "scenario_id": "T1",
            "seed": 5201,
            "duration_seconds": 18.0,
            "total_frames": 181,
            "trajectory": {
                "name": "북서-남동 대각 하강 접근",
                "start_xyz": [-48.0, 16.0, 66.0],
                "end_xyz": [2.0, -12.0, 45.0],
            },
            "camera": camera(),
            "objects": [
                obj(1, 2, [-8.0, 10.0], [1.10, 1.50]),
                obj(2, 1, [3.0, 7.0], [0.46, 1.70]),
                obj(3, 1, [16.0, 2.0], [0.60, 2.00]),
                obj(4, 2, [30.0, -3.5], [1.42, 1.75]),
                obj(5, 1, [43.0, -8.0], [0.50, 1.82]),
                obj(6, 2, [55.0, -11.0], [1.18, 1.58]),
            ],
            "person_motion": {"radius_m": 0.7, "omega_rad_s": 1.05, "phase_offset_rad": 1.55},
            "gt_version": GT_VERSION,
            "major_changes": [
                "x·y·z가 동시에 변하는 장거리 대각 하강 접근",
                "차량-사람-사람-차량 순의 불규칙 종방향 배치",
                "빠른 보행 위상과 작은 보행 반경을 사용",
            ],
            "difference_from_existing": "21 m 하강과 28 m 횡이동을 결합한 경사 접근으로 기존 경로와 시작·종료점이 다름",
        },
        {
            "schema_version": "1.0",
            "scenario_id": "T2",
            "seed": 5202,
            "duration_seconds": 18.0,
            "total_frames": 181,
            "trajectory": {
                "name": "남서-북동 상승 통과·부분 가림",
                "start_xyz": [-20.0, -20.0, 50.0],
                "end_xyz": [28.0, 14.0, 58.0],
            },
            "camera": camera(),
            "objects": [
                obj(1, 2, [4.0, -10.2], [1.36, 1.70]),
                obj(2, 1, [4.7, -9.8], [0.43, 1.62]),
                obj(3, 1, [24.0, 2.7], [0.58, 1.98]),
                obj(4, 2, [24.8, 3.1], [1.08, 1.48]),
                obj(5, 2, [46.0, 16.2], [1.44, 1.78]),
                obj(6, 1, [46.6, 16.6], [0.51, 1.80]),
            ],
            "person_motion": {"radius_m": 2.0, "omega_rad_s": 0.85, "phase_offset_rad": 2.35},
            "gt_version": GT_VERSION,
            "major_changes": [
                "48 m 전진·34 m 횡이동·8 m 상승의 대각 통과",
                "사람과 차량을 세 개의 근접 쌍으로 배치해 부분 가림 유도",
                "객체별 크기와 보행 위상을 다르게 고정",
            ],
            "difference_from_existing": "세 근접 쌍의 교대 클래스 전후관계와 큰 횡이동을 함께 사용",
        },
    ]


def register() -> None:
    if CONFIG_ROOT.exists():
        raise FileExistsError(f"Plan already exists and is immutable: {CONFIG_ROOT}")
    base_config = read_json(BASE_EVALUATION_CONFIG)
    if sha256(Path(base_config["weights_path"])) != str(base_config["expected_weights_sha256"]).upper():
        raise RuntimeError("Base locked weights no longer match before T0-T2 registration")
    if not BASE_LOCK.is_file():
        raise FileNotFoundError(BASE_LOCK)

    scenario_rows: list[dict[str, Any]] = []
    for scenario in scenario_definitions():
        config_path = CONFIG_ROOT / "scenario_configs" / f"{scenario['scenario_id']}.json"
        write_json_new(config_path, scenario, canonical=True)
        start = scenario["trajectory"]["start_xyz"]
        end = scenario["trajectory"]["end_xyz"]
        delta = [float(b) - float(a) for a, b in zip(start, end)]
        norm = math.sqrt(sum(value * value for value in delta))
        scenario_rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "trajectory_name": scenario["trajectory"]["name"],
                "seed": scenario["seed"],
                "uav_initial_xyz": start,
                "uav_end_xyz": end,
                "flight_direction_unit": [value / norm for value in delta],
                "altitude_range_m": [min(start[2], end[2]), max(start[2], end[2])],
                "person_count": sum(x["class_id"] == 1 for x in scenario["objects"]),
                "vehicle_count": sum(x["class_id"] == 2 for x in scenario["objects"]),
                "total_frames": scenario["total_frames"],
                "major_changes": scenario["major_changes"],
                "difference_from_existing": scenario["difference_from_existing"],
                "scenario_config_path": str(config_path.resolve()),
                "scenario_config_sha256": sha256(config_path),
            }
        )

    locked_config = dict(base_config)
    locked_config["experiment_id"] = "kci_new_fixed_tests_t0_t2_v1"
    locked_config["test_plan_id"] = PLAN_ID
    locked_config["source_lock_path"] = str(BASE_LOCK.resolve())
    locked_config["source_lock_sha256"] = sha256(BASE_LOCK)
    locked_config_path = CONFIG_ROOT / "evaluation_config.json"
    write_json_new(locked_config_path, locked_config, canonical=True)

    plan = {
        "schema_version": "1.0",
        "plan_id": PLAN_ID,
        "supersedes_failed_visibility_plan": SUPERSEDES_PLAN,
        "pre_YOLO_visibility_correction": CORRECTION_NOTE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_rule": "Frozen before GT rendering and before any T0-T2 YOLO inference. Detector results may not alter scenario definitions.",
        "existing_environment": "기존 MATLAB 시뮬레이션 환경",
        "ground_truth_version": GT_VERSION,
        "initial_environment": locked_config["initial_environment"],
        "evaluation_config_path": str(locked_config_path.resolve()),
        "evaluation_config_sha256": sha256(locked_config_path),
        "base_experiment_lock_path": str(BASE_LOCK.resolve()),
        "base_experiment_lock_sha256": sha256(BASE_LOCK),
        "weights_sha256": locked_config["expected_weights_sha256"],
        "allowed_preinference_action": "Only GT visibility/range/class validation and duplicate/similarity audit. If visibility fails, preserve this plan and create a new version before YOLO.",
        "forbidden_post_result_action": "Do not modify a scenario after seeing any T0-T2 YOLO result.",
        "scenarios": scenario_rows,
    }
    plan_path = CONFIG_ROOT / "scenario_plan.json"
    write_json_new(plan_path, plan, canonical=True)
    write_csv_new(CONFIG_ROOT / "scenario_plan.csv", scenario_rows)
    prereg = {
        "plan_id": PLAN_ID,
        "supersedes_failed_visibility_plan": SUPERSEDES_PLAN,
        "pre_YOLO_visibility_correction": CORRECTION_NOTE,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256(plan_path),
        "evaluation_config_path": str(locked_config_path.resolve()),
        "evaluation_config_sha256": sha256(locked_config_path),
        "scenario_config_hashes": {
            item["scenario_id"]: item["scenario_config_sha256"] for item in scenario_rows
        },
        "yolo_inference_performed": False,
    }
    write_json_new(CONFIG_ROOT / "pre_registration.json", prereg)
    print(json.dumps(prereg, ensure_ascii=True))


class MatlabExporter:
    def __init__(self) -> None:
        self.engine = None

    def start(self) -> None:
        if self.engine is not None:
            return
        import matlab.engine

        self.engine = matlab.engine.start_matlab("-nodesktop -nosplash")
        self.engine.cd(str(REPO_ROOT), nargout=0)
        self.engine.addpath(str(REPO_ROOT), nargout=0)
        self.engine.addpath(str(EXP_ROOT), nargout=0)
        self.engine.addpath(str(EXP_ROOT / "multi_scenario_symmetric"), nargout=0)

    def render(self, scenario: dict[str, Any], env: dict[str, float], output_dir: Path) -> dict[str, Any]:
        self.start()
        assert self.engine is not None
        output_dir.mkdir(parents=True, exist_ok=False)
        raw = self.engine.export_scenario_frames(
            float(env["fog_percent"]),
            float(env["illumination_lux"]),
            float(env["camera_noise"]),
            str(output_dir.resolve()),
            float(scenario["seed"]),
            1.0,
            "yolov8",
            GT_VERSION,
            scenario["scenario_config_path"],
            scenario["scenario_config_sha256"],
            nargout=1,
        )
        return json.loads(str(raw))

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            finally:
                self.engine = None


def dhash(image_path: Path) -> int:
    with Image.open(image_path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    result = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            result = (result << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return result


def reference_images() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ("train", "val"):
        for path in sorted((TRAINING_DATA_ROOT / "dataset" / "images" / split).glob("*.png")):
            records.append({"scope": split, "scenario_id": path.stem.split("__")[0], "path": path})
    for scenario_id in ("S0", "S1", "S2", "S3", "S4"):
        for path in sorted((FIXED_ROOT / scenario_id / "initial_condition" / "frames").glob("*.png")):
            records.append({"scope": "S0-S4", "scenario_id": scenario_id, "path": path})
    if len(records) != 1365:
        raise RuntimeError(f"Expected 460 train/val + 905 S0-S4 reference frames, found {len(records)}")
    return records


def validate_manifest(manifest: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    frames = manifest.get("frames", [])
    if manifest.get("scenario_id") != scenario["scenario_id"]:
        errors.append("scenario_id mismatch")
    if manifest.get("scenario_config_sha256") != scenario["scenario_config_sha256"]:
        errors.append("scenario_config_sha256 mismatch")
    if manifest.get("ground_truth_mode") != GT_VERSION:
        errors.append("GT version mismatch")
    if len(frames) != 181:
        errors.append(f"frame count {len(frames)} != 181")
    class_counts = {"person": 0, "vehicle": 0}
    invalid_class_count = 0
    invalid_box_count = 0
    object_frames = 0
    both_class_frames = 0
    empty_frames = 0
    for expected, frame in enumerate(frames, 1):
        if int(frame.get("frame_index", -1)) != expected:
            errors.append(f"frame index mismatch at position {expected}")
        classes: set[str] = set()
        items = frame.get("ground_truth", [])
        if items:
            object_frames += 1
        else:
            empty_frames += 1
        for gt in items:
            name = gt.get("class_name")
            classes.add(str(name))
            if name not in class_counts:
                invalid_class_count += 1
                continue
            class_counts[name] += 1
            x, y, width, height = (float(value) for value in gt["bbox_xywh"])
            if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > 640.001 or y + height > 360.001:
                invalid_box_count += 1
        if classes == {"person", "vehicle"}:
            both_class_frames += 1
    if not all(class_counts.values()):
        errors.append(f"one or more classes have zero visible GT: {class_counts}")
    if invalid_class_count:
        errors.append(f"invalid classes: {invalid_class_count}")
    if invalid_box_count:
        errors.append(f"invalid boxes: {invalid_box_count}")
    return {
        "scenario_id": scenario["scenario_id"],
        "manifest_path": str((SCENARIO_ROOT / scenario["scenario_id"] / "initial_condition" / "frame_manifest.json").resolve()),
        "frame_count": len(frames),
        "person_gt_count": class_counts["person"],
        "vehicle_gt_count": class_counts["vehicle"],
        "object_frame_count": object_frames,
        "both_class_frame_count": both_class_frames,
        "empty_frame_count": empty_frames,
        "invalid_class_count": invalid_class_count,
        "invalid_box_count": invalid_box_count,
        "valid": not errors,
        "errors": errors,
    }


def make_comparison_sheets(pairs: list[dict[str, Any]], output_dir: Path) -> list[str]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default()
    paths: list[str] = []
    for sheet_no, start in enumerate(range(0, min(len(pairs), 24), 2), 1):
        batch = pairs[start : start + 2]
        sheet = Image.new("RGB", (1320, 850), "white")
        draw = ImageDraw.Draw(sheet)
        for row, pair in enumerate(batch):
            y = 20 + row * 415
            with Image.open(pair["new_path"]) as left, Image.open(pair["reference_path"]) as right:
                sheet.paste(left.convert("RGB"), (10, y + 35))
                sheet.paste(right.convert("RGB"), (670, y + 35))
            draw.text((10, y), f"{pair['pair_id']} NEW {pair['new_scenario']} frame {pair['new_frame']} dHash={pair['dhash_distance']}", fill="black", font=font)
            draw.text((670, y), f"{pair['reference_scope']} {pair['reference_scenario']} frame {pair['reference_frame']}", fill="black", font=font)
        path = output_dir / f"closest_pairs_{start + 1:02d}_{start + len(batch):02d}.png"
        sheet.save(path, optimize=True)
        paths.append(str(path.resolve()))
    return paths


def preflight() -> None:
    prereg_path = CONFIG_ROOT / "pre_registration.json"
    prereg = read_json(prereg_path)
    plan_path = Path(prereg["plan_path"])
    if sha256(plan_path) != prereg["plan_sha256"]:
        raise RuntimeError("Frozen T0-T2 scenario plan hash changed")
    plan = read_json(plan_path)
    config = read_json(Path(plan["evaluation_config_path"]))
    for item in plan["scenarios"]:
        if sha256(Path(item["scenario_config_path"])) != item["scenario_config_sha256"]:
            raise RuntimeError(f"Frozen scenario config changed: {item['scenario_id']}")
    if OUTPUT_ROOT.exists() or SCENARIO_ROOT.exists():
        raise FileExistsError("Refusing to overwrite prior T0-T2 preflight output")

    started = time.perf_counter()
    exporter = MatlabExporter()
    validations: list[dict[str, Any]] = []
    try:
        for item in plan["scenarios"]:
            out = SCENARIO_ROOT / item["scenario_id"] / "initial_condition"
            manifest = exporter.render(item, config["initial_environment"], out)
            # The MATLAB function writes this file; compare the returned object
            # with it before continuing the audit.
            manifest_path = out / "frame_manifest.json"
            disk_manifest = read_json(manifest_path)
            if manifest != disk_manifest:
                raise RuntimeError(f"Returned/disk manifest mismatch: {item['scenario_id']}")
            validations.append(validate_manifest(manifest, item))
    finally:
        exporter.close()

    new_images: list[dict[str, Any]] = []
    for item in plan["scenarios"]:
        for path in sorted((SCENARIO_ROOT / item["scenario_id"] / "initial_condition" / "frames").glob("*.png")):
            new_images.append(
                {
                    "scenario_id": item["scenario_id"],
                    "frame": int(path.stem.split("_")[-1]),
                    "path": path,
                    "sha256": sha256(path),
                    "dhash": dhash(path),
                }
            )
    references = reference_images()
    for item in references:
        path = item["path"]
        item["frame"] = int(path.stem.split("_")[-1])
        item["sha256"] = sha256(path)
        item["dhash"] = dhash(path)

    ref_sha: dict[str, list[dict[str, Any]]] = {}
    for item in references:
        ref_sha.setdefault(item["sha256"], []).append(item)
    exact: list[dict[str, Any]] = []
    near: list[dict[str, Any]] = []
    closest: list[dict[str, Any]] = []
    pair_no = 0
    for new in new_images:
        for ref in ref_sha.get(new["sha256"], []):
            exact.append(
                {
                    "new_scenario": new["scenario_id"],
                    "new_frame": new["frame"],
                    "new_path": str(new["path"].resolve()),
                    "reference_scope": ref["scope"],
                    "reference_scenario": ref["scenario_id"],
                    "reference_frame": ref["frame"],
                    "reference_path": str(ref["path"].resolve()),
                }
            )
        candidates: list[tuple[int, dict[str, Any]]] = []
        for ref in references:
            distance = int((new["dhash"] ^ ref["dhash"]).bit_count())
            candidates.append((distance, ref))
            if distance <= 4:
                pair_no += 1
                near.append(
                    {
                        "pair_id": f"N{pair_no:05d}",
                        "new_scenario": new["scenario_id"],
                        "new_frame": new["frame"],
                        "new_path": str(new["path"].resolve()),
                        "reference_scope": ref["scope"],
                        "reference_scenario": ref["scenario_id"],
                        "reference_frame": ref["frame"],
                        "reference_path": str(ref["path"].resolve()),
                        "dhash_distance": distance,
                        "exact_sha256": new["sha256"] == ref["sha256"],
                    }
                )
        for distance, ref in sorted(candidates, key=lambda value: value[0])[:3]:
            closest.append(
                {
                    "new_scenario": new["scenario_id"],
                    "new_frame": new["frame"],
                    "new_path": str(new["path"].resolve()),
                    "reference_scope": ref["scope"],
                    "reference_scenario": ref["scenario_id"],
                    "reference_frame": ref["frame"],
                    "reference_path": str(ref["path"].resolve()),
                    "dhash_distance": distance,
                    "exact_sha256": new["sha256"] == ref["sha256"],
                }
            )
    near.sort(key=lambda row: (row["dhash_distance"], row["new_scenario"], row["new_frame"]))
    for index, row in enumerate(near, 1):
        row["pair_id"] = f"N{index:05d}"
    closest_unique = sorted(
        closest,
        key=lambda row: (row["dhash_distance"], row["new_scenario"], row["new_frame"]),
    )
    for index, row in enumerate(closest_unique, 1):
        row["pair_id"] = f"C{index:05d}"

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    comparison_source = near if near else closest_unique
    sheets = make_comparison_sheets(comparison_source, OUTPUT_ROOT / "similarity_comparison_sheets")
    gt_result = {
        "plan_id": PLAN_ID,
        "performed_before_yolo": True,
        "all_valid": all(item["valid"] for item in validations),
        "scenario_count": len(validations),
        "validations": validations,
    }
    audit = {
        "plan_id": PLAN_ID,
        "performed_before_yolo": True,
        "new_frame_count": len(new_images),
        "reference_frame_count": len(references),
        "comparison_count": len(new_images) * len(references),
        "reference_scopes": {"train": 360, "val": 100, "S0-S4": 905},
        "exact_duplicate_count": len(exact),
        "dhash_warning_threshold": 4,
        "near_duplicate_warning_count": len(near),
        "minimum_dhash_distance": min(row["dhash_distance"] for row in closest_unique),
        "warnings_require_visual_interpretation": bool(near),
        "visual_review_status": "pending" if near else "not_required_no_dhash_le_4_pairs",
        "new_scenarios_passed": not exact,
        "gate_definition": "No exact SHA duplicate, all scenario/config hashes differ, and the GT audit passes. dHash warnings are disclosed and visually reviewed before YOLO but do not by themselves rewrite the frozen plan.",
        "comparison_sheets": sheets,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "closest_pairs_sample": closest_unique[:24],
    }
    write_json_new(OUTPUT_ROOT / "gt_validation.json", gt_result)
    write_csv_new(OUTPUT_ROOT / "gt_validation.csv", validations)
    write_json_new(OUTPUT_ROOT / "similarity_audit.json", audit)
    write_csv_new(OUTPUT_ROOT / "near_duplicate_warnings.csv", near)
    write_csv_new(OUTPUT_ROOT / "closest_pairs_sample.csv", closest_unique[:24])
    completion = {
        "plan_id": PLAN_ID,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_seconds": time.perf_counter() - started,
        "plan_sha256": sha256(plan_path),
        "evaluation_config_sha256": sha256(Path(plan["evaluation_config_path"])),
        "gt_validation_sha256": sha256(OUTPUT_ROOT / "gt_validation.json"),
        "similarity_audit_sha256": sha256(OUTPUT_ROOT / "similarity_audit.json"),
        "yolo_inference_performed": False,
    }
    write_json_new(OUTPUT_ROOT / "preflight_completion.json", completion)
    print(json.dumps({"gt": gt_result, "audit_summary": {k: audit[k] for k in ("new_frame_count", "reference_frame_count", "exact_duplicate_count", "near_duplicate_warning_count", "minimum_dhash_distance")}}, ensure_ascii=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "preflight"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "register":
        register()
    else:
        preflight()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
