"""Generate scenario-separated MATLAB simulation data in YOLO format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from .run_comparison import HERE, REPO_ROOT, Environment, MatlabFrameExporter, write_json


DEFAULT_PLAN = HERE / "training" / "scenario_plan.json"
DEFAULT_OUTPUT = HERE / "training" / "yolov8s_sim_20260902"
CLASS_IDS = {"person": 0, "vehicle": 1}


def link_or_copy(source: Path, target: Path) -> None:
    """Prefer a same-volume hard link so rendered PNG bytes are stored once."""
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def xywh_pixels_to_yolo(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = (float(value) for value in box)
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"Out-of-range rendered GT: {[x, y, w, h]}")
    return (x + w / 2) / width, (y + h / 2) / height, w / width, h / height


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_plan(plan: dict[str, Any]) -> None:
    scenarios = plan["scenarios"]
    ids = [row["scenario_id"] for row in scenarios]
    variants = [int(row["scenario_variant"]) for row in scenarios]
    if len(ids) != len(set(ids)) or len(variants) != len(set(variants)):
        raise ValueError("Scenario IDs and variants must be unique")
    if any(row["split"] not in {"train", "val"} for row in scenarios):
        raise ValueError("Generated scenarios may only belong to train or val")
    held_out = plan["held_out_test"]
    if held_out.get("included_in_training_dataset") is not False:
        raise ValueError("Held-out test must be explicitly excluded")
    if int(held_out["scenario_variant"]) in variants or held_out["scenario_id"] in ids:
        raise ValueError("Held-out test overlaps a training/validation scenario")


def manifest_is_complete(path: Path, scenario: dict[str, Any], frame_stride: int) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return (
            int(manifest["scenario_variant"]) == int(scenario["scenario_variant"])
            and int(manifest["random_seed"]) == int(scenario["seed"])
            and int(manifest["frame_stride"]) == frame_stride
            and all(Path(frame["image_path"]).is_file() for frame in manifest["frames"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def copy_frame_and_label(
    frame: dict[str, Any], scenario: dict[str, Any], dataset_root: Path, width: int, height: int
) -> dict[str, Any]:
    split = scenario["split"]
    stem = f"{scenario['scenario_id']}__frame_{int(frame['frame_index']):04d}"
    source_image = Path(frame["image_path"])
    image_out = dataset_root / "images" / split / f"{stem}{source_image.suffix.lower()}"
    label_out = dataset_root / "labels" / split / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    link_or_copy(source_image, image_out)

    label_lines: list[str] = []
    person_count = 0
    vehicle_count = 0
    for item in frame.get("ground_truth", []):
        class_name = str(item["class_name"])
        if class_name not in CLASS_IDS:
            raise ValueError(f"Unsupported class {class_name!r}")
        try:
            xc, yc, wn, hn = xywh_pixels_to_yolo(item["bbox_xywh"], width, height)
        except ValueError as exc:
            raise ValueError(f"{exc} in {stem}") from exc
        label_lines.append(f"{CLASS_IDS[class_name]} {xc:.9f} {yc:.9f} {wn:.9f} {hn:.9f}")
        person_count += int(class_name == "person")
        vehicle_count += int(class_name == "vehicle")
    label_out.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="ascii")
    return {
        "scenario_id": scenario["scenario_id"],
        "split": split,
        "scenario_variant": int(scenario["scenario_variant"]),
        "frame_index": int(frame["frame_index"]),
        "image_path": str(image_out.resolve()),
        "label_path": str(label_out.resolve()),
        "image_sha256": sha256(image_out),
        "person_labels": person_count,
        "vehicle_labels": vehicle_count,
    }


def make_smoke_subset(dataset_root: Path, frame_rows: list[dict[str, Any]]) -> Path:
    selected: list[dict[str, Any]] = []
    limits = {"train": 16, "val": 8}
    for split, limit in limits.items():
        candidates = [row for row in frame_rows if row["split"] == split and row["person_labels"] + row["vehicle_labels"] > 0]
        if len(candidates) < limit:
            candidates = [row for row in frame_rows if row["split"] == split]
        step = max(1, len(candidates) // limit)
        selected.extend(candidates[::step][:limit])
    for row in selected:
        split = row["split"]
        image_source = Path(row["image_path"])
        label_source = Path(row["label_path"])
        image_target = dataset_root / "smoke" / "images" / split / image_source.name
        label_target = dataset_root / "smoke" / "labels" / split / label_source.name
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(image_source, image_target)
        link_or_copy(label_source, label_target)
    smoke_yaml = dataset_root / "data_smoke.yaml"
    smoke_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str((dataset_root / "smoke").resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person", 1: "vehicle"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return smoke_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = json.loads(args.plan.resolve().read_text(encoding="utf-8"))
    validate_plan(plan)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "scenario_plan_snapshot.json", plan)

    exporter = MatlabFrameExporter(REPO_ROOT, HERE)
    frame_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    try:
        for scenario in plan["scenarios"]:
            scenario_dir = output_root / "rendered_scenarios" / scenario["scenario_id"]
            manifest_path = scenario_dir / "frame_manifest.json"
            if manifest_is_complete(manifest_path, scenario, int(plan["frame_stride"])):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                manifest = exporter.export(
                    Environment(
                        float(scenario["fog_percent"]),
                        float(scenario["illumination_lux"]),
                        float(scenario["camera_noise"]),
                    ),
                    scenario_dir,
                    int(scenario["seed"]),
                    int(plan["frame_stride"]),
                    "yolov8",
                    str(plan["ground_truth_mode"]),
                    int(scenario["scenario_variant"]),
                )
            start = len(frame_rows)
            for frame in manifest["frames"]:
                frame_rows.append(
                    copy_frame_and_label(
                        frame,
                        scenario,
                        output_root / "dataset",
                        int(manifest["image_width"]),
                        int(manifest["image_height"]),
                    )
                )
            created = frame_rows[start:]
            scenario_rows.append(
                {
                    **scenario,
                    "sampled_frame_count": len(created),
                    "person_label_count": sum(row["person_labels"] for row in created),
                    "vehicle_label_count": sum(row["vehicle_labels"] for row in created),
                    "manifest_path": str(manifest_path.resolve()),
                }
            )
    finally:
        exporter.close()

    hashes: dict[str, set[str]] = {}
    for row in frame_rows:
        hashes.setdefault(row["image_sha256"], set()).add(row["split"])
    cross_split_duplicates = sorted(key for key, splits in hashes.items() if len(splits) > 1)
    train_scenarios = {row["scenario_id"] for row in frame_rows if row["split"] == "train"}
    val_scenarios = {row["scenario_id"] for row in frame_rows if row["split"] == "val"}
    leakage = {
        "scenario_id_overlap_train_val": sorted(train_scenarios & val_scenarios),
        "image_sha256_overlap_train_val": cross_split_duplicates,
        "held_out_test_present_in_generated_rows": any(
            row["scenario_id"] == plan["held_out_test"]["scenario_id"] for row in frame_rows
        ),
        "passed": not (train_scenarios & val_scenarios) and not cross_split_duplicates,
        "held_out_test": plan["held_out_test"],
    }
    if not leakage["passed"] or leakage["held_out_test_present_in_generated_rows"]:
        write_json(output_root / "leakage_audit.json", leakage)
        raise RuntimeError("Dataset leakage audit failed")

    dataset_root = output_root / "dataset"
    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "person", 1: "vehicle"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    smoke_yaml = make_smoke_subset(dataset_root, frame_rows)
    write_csv(output_root / "scenario_manifest.csv", scenario_rows)
    write_json(output_root / "scenario_manifest.json", scenario_rows)
    write_csv(output_root / "frame_manifest.csv", frame_rows)
    write_json(output_root / "leakage_audit.json", leakage)
    summary = {
        "data_yaml": str(data_yaml),
        "smoke_yaml": str(smoke_yaml),
        "scenario_count": len(scenario_rows),
        "train_scenario_count": len(train_scenarios),
        "val_scenario_count": len(val_scenarios),
        "train_image_count": sum(row["split"] == "train" for row in frame_rows),
        "val_image_count": sum(row["split"] == "val" for row in frame_rows),
        "person_label_count": sum(row["person_labels"] for row in frame_rows),
        "vehicle_label_count": sum(row["vehicle_labels"] for row in frame_rows),
        "leakage_audit_passed": leakage["passed"],
    }
    write_json(output_root / "dataset_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
