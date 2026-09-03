"""Pre-register, render, and GT-audit S0--S4 before any YOLO inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
CONFIG_DIR = HERE / "config"
SCENARIOS_ROOT = HERE / "scenarios"
TRAINING_ROOT = EXPERIMENT_ROOT / "training" / "yolov8s_sim_20260902"
INITIAL_ENVIRONMENT = {"fog_percent": 5.0, "illumination_lux": 12000.0, "camera_noise": 0.02}
GT_VERSION = "rendered_instance_mask_v1"
PLAN_ID = "multi_scenario_symmetric_plan_v2"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_new_or_identical(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite non-identical existing file: {path}")
        return
    path.write_bytes(data)


def objects(rows: list[tuple[float, float, int]]) -> list[dict[str, Any]]:
    result = []
    for index, (x, y, class_id) in enumerate(rows, start=1):
        result.append(
            {
                "object_id": index,
                "class_id": class_id,
                "class_name": "person" if class_id == 1 else "vehicle",
                "xy": [x, y],
                "radius_height": [0.5, 1.8] if class_id == 1 else [1.2, 1.6],
            }
        )
    return result


def definitions() -> list[dict[str, Any]]:
    z0 = 54.25444453897675
    common = {
        "duration_seconds": 18.0,
        "total_frames": 181,
        "camera": {
            "intrinsics": [600.0, 600.0, 320.0, 180.0, 60.0],
            "image_size": [640, 360],
            "boresight": "+x, pitched 60 degrees downward",
        },
        "person_motion": {"radius_m": 1.5, "omega_rad_s": 0.6},
        "gt_version": GT_VERSION,
    }
    specs = [
        {
            "scenario_id": "S0",
            "seed": 42,
            "trajectory": {"name": "기준 동서 직선 순항", "start_xyz": [-15.0, 0.0, z0], "end_xyz": [39.0, 0.0, z0]},
            "person_motion": {"phase_offset_rad": 0.0},
            "objects": objects([(-5, 2, 1), (5, -2, 1), (15, 3, 1), (25, -2, 2), (35, 2, 2)]),
            "major_changes": ["기존 대표 시나리오를 최종 렌더러로 독립 재실행"],
            "difference_from_training": "held-out variant 0; never included in the train/validation split",
        },
        {
            "scenario_id": "S1",
            "seed": 3101,
            "trajectory": {"name": "남쪽에서 북쪽으로 횡방향 관측", "start_xyz": [-12.0, -14.0, z0], "end_xyz": [-12.0, 14.0, z0]},
            "person_motion": {"phase_offset_rad": 0.31},
            "objects": objects([(-2, -5, 1), (5, 0, 1), (12, 5, 1), (0, -7, 2), (6, 7, 2)]),
            "major_changes": ["UAV 진행축을 x축에서 y축으로 변경", "표적을 측면 관측 폭에 맞춰 재배치"],
            "difference_from_training": "explicit lateral path; training variants only used constant xyz offsets of the S0 eastbound path",
        },
        {
            "scenario_id": "S2",
            "seed": 3102,
            "trajectory": {"name": "고도 점진 하강 접근", "start_xyz": [-18.0, 0.0, 64.0], "end_xyz": [30.0, 0.0, 49.0]},
            "person_motion": {"phase_offset_rad": 0.62},
            "objects": objects([(-6, 1, 1), (5, -3, 1), (16, 4, 1), (28, -1, 2), (40, 3, 2)]),
            "major_changes": ["18초 동안 고도 15 m 하강", "진행방향 전방에 표적 간격 유지"],
            "difference_from_training": "time-varying altitude; training variants used constant altitude offsets",
        },
        {
            "scenario_id": "S3",
            "seed": 3103,
            "trajectory": {"name": "남서-북동 대각선 통과", "start_xyz": [-18.0, -10.0, 56.0], "end_xyz": [30.0, 10.0, 56.0]},
            "person_motion": {"phase_offset_rad": 0.93},
            "objects": objects([(-6, -6, 1), (4, -2, 1), (14, 2, 1), (24, 6, 2), (34, 10, 2)]),
            "major_changes": ["x와 y를 동시에 변화시키는 대각선 경로", "대각선 통로를 따라 표적 배치"],
            "difference_from_training": "diagonal velocity and diagonal target corridor absent from train/validation variants",
        },
        {
            "scenario_id": "S4",
            "seed": 3104,
            "trajectory": {"name": "원거리 시작 후 전방 접근", "start_xyz": [-42.0, 0.0, 58.0], "end_xyz": [12.0, 0.0, 58.0]},
            "person_motion": {"phase_offset_rad": 1.24},
            "objects": objects([(-8, -1, 1), (3, 3, 1), (16, -3, 1), (30, 2, 2), (44, -2, 2)]),
            "major_changes": ["S0보다 27 m 먼 위치에서 시작", "접근 거리에 따라 표적 영상 크기가 증가하도록 표적열 확장"],
            "difference_from_training": "long-range start and expanded target spacing absent from train/validation variants",
        },
    ]
    merged = []
    for spec in specs:
        row = {**common, **spec}
        row["person_motion"] = {**common["person_motion"], **spec["person_motion"]}
        merged.append(row)
    return merged


def direction_and_altitude(scenario: dict[str, Any]) -> tuple[list[float], list[float]]:
    start = scenario["trajectory"]["start_xyz"]
    end = scenario["trajectory"]["end_xyz"]
    delta = [end[i] - start[i] for i in range(3)]
    norm = math.sqrt(sum(value * value for value in delta))
    return [value / norm for value in delta], [min(start[2], end[2]), max(start[2], end[2])]


def register_plan() -> dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    plan_dir = CONFIG_DIR / PLAN_ID
    scenario_dir = plan_dir / "scenario_configs"
    records = []
    for scenario in definitions():
        path = scenario_dir / f"{scenario['scenario_id']}.json"
        data = canonical_json_bytes(scenario)
        write_new_or_identical(path, data)
        direction, altitude_range = direction_and_altitude(scenario)
        records.append(
            {
                "scenario_id": scenario["scenario_id"],
                "trajectory_name": scenario["trajectory"]["name"],
                "seed": scenario["seed"],
                "uav_initial_xyz": scenario["trajectory"]["start_xyz"],
                "uav_end_xyz": scenario["trajectory"]["end_xyz"],
                "flight_direction_unit": direction,
                "altitude_range_m": altitude_range,
                "camera": scenario["camera"],
                "object_layout": scenario["objects"],
                "person_count": sum(item["class_id"] == 1 for item in scenario["objects"]),
                "vehicle_count": sum(item["class_id"] == 2 for item in scenario["objects"]),
                "total_frames": scenario["total_frames"],
                "difference_from_training_validation_test": scenario["difference_from_training"],
                "major_changes": scenario["major_changes"],
                "scenario_config_path": str(path.resolve()),
                "scenario_config_sha256": sha256_bytes(data),
            }
        )
    plan = {
        "schema_version": "1.0",
        "plan_id": PLAN_ID,
        "created_at_utc": "2026-09-02T12:00:00+00:00",
        "supersedes_preflight_plan": "multi_scenario_symmetric_plan_v1",
        "preflight_correction": "S1 vehicle x/y positions moved into the fixed camera frustum after v1 GT-only validation found zero visible vehicle boxes; no YOLO inference had been run.",
        "registration_rule": "Scenario definitions are frozen before YOLO inference and may not be selected or edited based on detector performance.",
        "initial_environment": INITIAL_ENVIRONMENT,
        "ground_truth_version": GT_VERSION,
        "scenarios": records,
    }
    plan_path = plan_dir / "scenario_plan.json"
    write_new_or_identical(plan_path, (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    csv_path = plan_dir / "scenario_plan.csv"
    rows = []
    for record in records:
        rows.append({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in record.items()})
    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return plan


def label_text(frame: dict[str, Any], width: int, height: int) -> str:
    lines = []
    for gt in frame.get("ground_truth", []):
        x, y, w, h = (float(v) for v in gt["bbox_xywh"])
        class_id = 0 if gt["class_name"] == "person" else 1
        lines.append(f"{class_id} {(x + w / 2) / width:.10f} {(y + h / 2) / height:.10f} {w / width:.10f} {h / height:.10f}")
    return "\n".join(lines) + ("\n" if lines else "")


def validate_manifest(manifest: dict[str, Any], expected: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = manifest["frames"]
    width, height = int(manifest["image_width"]), int(manifest["image_height"])
    errors: list[str] = []
    if len(frames) != 181 or int(manifest["evaluated_frame_count"]) != 181:
        errors.append(f"frame_count={len(frames)}")
    expected_indices = list(range(1, 182))
    actual_indices = [int(frame["frame_index"]) for frame in frames]
    if actual_indices != expected_indices:
        errors.append("frame_index_sequence_mismatch")
    counts = {"person": 0, "vehicle": 0}
    frame_rows = []
    object_frames = 0
    for frame in frames:
        path = Path(frame["image_path"])
        if path.stem != f"frame_{int(frame['frame_index']):04d}" or not path.is_file():
            errors.append(f"frame_path_mismatch:{frame['frame_index']}")
        for gt in frame.get("ground_truth", []):
            name = gt.get("class_name")
            if name not in counts:
                errors.append(f"invalid_class:{name}")
                continue
            counts[name] += 1
            x, y, w, h = (float(v) for v in gt["bbox_xywh"])
            if w <= 0 or h <= 0:
                errors.append(f"nonpositive_box:{frame['frame_index']}")
            if x < 0 or y < 0 or x + w > width or y + h > height:
                errors.append(f"out_of_bounds:{frame['frame_index']}:{gt['bbox_xywh']}")
        if frame.get("ground_truth"):
            object_frames += 1
        labels = label_text(frame, width, height)
        frame_rows.append(
            {
                "scenario_id": expected["scenario_id"],
                "frame_index": int(frame["frame_index"]),
                "image_path": str(path),
                "image_sha256": file_sha256(path),
                "label_sha256": sha256_bytes(labels.encode("utf-8")),
                "frame_sha256": sha256_bytes((file_sha256(path) + "|" + labels).encode("utf-8")),
                "person_gt": sum(gt["class_name"] == "person" for gt in frame.get("ground_truth", [])),
                "vehicle_gt": sum(gt["class_name"] == "vehicle" for gt in frame.get("ground_truth", [])),
                "label_text": labels,
            }
        )
    if counts["person"] < 20 or counts["vehicle"] < 20:
        errors.append(f"insufficient_class_gt:{counts}")
    summary = {
        "scenario_id": expected["scenario_id"],
        "scenario_config_sha256": expected["scenario_config_sha256"],
        "total_frames": len(frames),
        "object_frames": object_frames,
        "empty_frames": len(frames) - object_frames,
        "person_gt_count": counts["person"],
        "vehicle_gt_count": counts["vehicle"],
        "out_of_bounds_box_count": sum(item.startswith("out_of_bounds") for item in errors),
        "nonpositive_box_count": sum(item.startswith("nonpositive") for item in errors),
        "frame_gt_alignment_valid": not any("frame_" in item and "mismatch" in item for item in errors),
        "class_ids_valid": not any(item.startswith("invalid_class") for item in errors),
        "minimum_20_each_class_valid": counts["person"] >= 20 and counts["vehicle"] >= 20,
        "valid": not errors,
        "errors": errors,
    }
    return summary, frame_rows


def draw_gt_examples(manifest: dict[str, Any], output_dir: Path) -> list[str]:
    candidates = [frame for frame in manifest["frames"] if frame.get("ground_truth")]
    positions = [round(i * (len(candidates) - 1) / 4) for i in range(5)]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for order, pos in enumerate(positions, start=1):
        frame = candidates[pos]
        with Image.open(frame["image_path"]).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for gt in frame["ground_truth"]:
                x, y, w, h = gt["bbox_xywh"]
                color = (255, 50, 50) if gt["class_name"] == "person" else (30, 120, 255)
                draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
                draw.text((x, max(0, y - 12)), gt["class_name"], fill=color)
            target = output_dir / f"gt_example_{order:02d}_frame_{int(frame['frame_index']):04d}.png"
            image.save(target)
            outputs.append(str(target.resolve()))
    return outputs


def load_training_hashes() -> dict[str, set[str]]:
    images: set[str] = set()
    labels: set[str] = set()
    frames: set[str] = set()
    with (TRAINING_ROOT / "frame_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            images.add(row["image_sha256"].upper())
            if row.get("original_training_png_sha256"):
                images.add(row["original_training_png_sha256"].upper())
            label_path = Path(row["label_path"])
            label = label_path.read_text(encoding="utf-8") if label_path.is_file() else ""
            label_hash = sha256_bytes(label.encode("utf-8"))
            if label.strip():
                labels.add(label_hash)
            source_image_hash = row.get("original_training_png_sha256") or row["image_sha256"]
            frames.add(sha256_bytes((source_image_hash + "|" + label).encode("utf-8")))
    return {"images": images, "labels_nonempty": labels, "frames": frames}


def canonical_layout_hash(objects_value: list[dict[str, Any]]) -> str:
    identity = [{"class_id": x["class_id"], "xy": x["xy"], "radius_height": x["radius_height"]} for x in objects_value]
    return sha256_bytes(canonical_json_bytes(identity))


def canonical_trajectory_hash(record: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes({"start": record["uav_initial_xyz"], "end": record["uav_end_xyz"], "camera": record["camera"]}))


def leakage_audit(plan: dict[str, Any], all_frame_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    training = load_training_hashes()
    training_plan = json.loads((EXPERIMENT_ROOT / "training" / "scenario_plan.json").read_text(encoding="utf-8"))
    training_seeds = {int(item["seed"]) for item in training_plan["scenarios"]}
    held_out = training_plan["held_out_test"]
    training_seeds.add(int(held_out["seed"]))
    s0 = next(item for item in plan["scenarios"] if item["scenario_id"] == "S0")
    s0_layout = canonical_layout_hash(s0["object_layout"])
    s0_trajectory = canonical_trajectory_hash(s0)
    results = []
    for record in plan["scenarios"]:
        if record["scenario_id"] == "S0":
            results.append(
                {
                    "scenario_id": "S0",
                    "scope": "known held-out reference (not a new-scenario leakage target)",
                    "seed_overlap": int(record["seed"]) == int(held_out["seed"]),
                    "object_layout_overlap_with_S0": True,
                    "trajectory_overlap_with_S0": True,
                    "duplicate_image_hash_count": 0,
                    "duplicate_frame_hash_count": 0,
                    "duplicate_nonempty_label_hash_count": 0,
                    "passed": True,
                }
            )
            continue
        rows = all_frame_rows[record["scenario_id"]]
        image_overlap = sorted({row["image_sha256"] for row in rows} & training["images"])
        label_overlap = sorted({row["label_sha256"] for row in rows if row["label_text"].strip()} & training["labels_nonempty"])
        frame_overlap = sorted({row["frame_sha256"] for row in rows} & training["frames"])
        item = {
            "scenario_id": record["scenario_id"],
            "scope": "S1-S4 versus six train, two validation, and S0",
            "seed_overlap": int(record["seed"]) in training_seeds,
            "object_layout_overlap_with_S0": canonical_layout_hash(record["object_layout"]) == s0_layout,
            "trajectory_overlap_with_S0": canonical_trajectory_hash(record) == s0_trajectory,
            "duplicate_image_hash_count": len(image_overlap),
            "duplicate_frame_hash_count": len(frame_overlap),
            "duplicate_nonempty_label_hash_count": len(label_overlap),
            "image_hash_examples": image_overlap[:5],
            "frame_hash_examples": frame_overlap[:5],
            "label_hash_examples": label_overlap[:5],
        }
        item["passed"] = not any(
            [item["seed_overlap"], item["object_layout_overlap_with_S0"], item["trajectory_overlap_with_S0"], image_overlap, frame_overlap, label_overlap]
        )
        results.append(item)
    return {
        "schema_version": "1.0",
        "audited_against": {
            "training_scenarios": 6,
            "validation_scenarios": 2,
            "held_out_S0": True,
            "training_frame_manifest": str((TRAINING_ROOT / "frame_manifest.csv").resolve()),
            "empty_label_note": "Empty labels are excluded from raw label-hash collision because every empty file has the same unavoidable SHA-256; frame and image hashes remain audited.",
        },
        "results": results,
        "new_scenarios_passed": all(item["passed"] for item in results if item["scenario_id"] != "S0"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        flattened.append({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    fieldnames = list(dict.fromkeys(key for row in flattened for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def render_and_audit(plan: dict[str, Any]) -> Path:
    import matlab.engine

    plan_root = SCENARIOS_ROOT / plan["plan_id"]
    plan_root.mkdir(parents=True, exist_ok=True)
    if (plan_root / "gt_validation.json").exists() or (plan_root / "scenario_leakage_audit.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed scenario audit set: {plan_root}")
    engine = matlab.engine.start_matlab("-nodesktop -nosplash")
    summaries = []
    all_rows: dict[str, list[dict[str, Any]]] = {}
    try:
        engine.cd(str(REPO_ROOT), nargout=0)
        engine.addpath(str(REPO_ROOT), nargout=0)
        engine.addpath(str(EXPERIMENT_ROOT), nargout=0)
        engine.addpath(str(HERE), nargout=0)
        for record in plan["scenarios"]:
            scenario_dir = plan_root / record["scenario_id"] / "initial_condition"
            raw = engine.export_scenario_frames(
                5.0, 12000.0, 0.02, str(scenario_dir), float(record["seed"]), 1.0,
                "yolov8", GT_VERSION, record["scenario_config_path"], record["scenario_config_sha256"], nargout=1,
            )
            manifest = json.loads(str(raw))
            summary, rows = validate_manifest(manifest, record)
            summary["diagnostic_images"] = draw_gt_examples(manifest, plan_root / record["scenario_id"] / "gt_diagnostics")
            summaries.append(summary)
            all_rows[record["scenario_id"]] = rows
            hash_rows = [{key: value for key, value in row.items() if key != "label_text"} for row in rows]
            write_csv(plan_root / record["scenario_id"] / "frame_hashes.csv", hash_rows)
    finally:
        engine.quit()

    audit = leakage_audit(plan, all_rows)
    (plan_root / "gt_validation.json").write_text(json.dumps({"scenarios": summaries, "all_valid": all(x["valid"] for x in summaries)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(plan_root / "gt_validation.csv", summaries)
    (plan_root / "scenario_leakage_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(plan_root / "scenario_leakage_audit.csv", audit["results"])
    if not all(item["valid"] for item in summaries):
        raise RuntimeError("GT validation failed; YOLO inference is blocked. See gt_validation.json")
    if not audit["new_scenarios_passed"]:
        raise RuntimeError("Scenario leakage audit failed; YOLO inference is blocked.")
    return plan_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    plan = register_plan()
    print(json.dumps({"plan": str((CONFIG_DIR / PLAN_ID / 'scenario_plan.json').resolve()), "registered": len(plan["scenarios"])}, ensure_ascii=False))
    if not args.register_only:
        root = render_and_audit(plan)
        print(json.dumps({"scenario_root": str(root.resolve()), "gt_and_leakage_valid": True}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
