"""Render, validate, split, hash-audit, and package the pre-registered dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image, ImageDraw

from .plan import (
    CONFIG_ROOT,
    DATA_ROOT,
    FIXED_PLAN,
    FIXED_SCENARIOS,
    GT_VERSION,
    HERE,
    REPO_ROOT,
    SELECTED_FRAMES,
    canonical_bytes,
    file_sha256,
)


CLASS_IDS = {"person": 0, "vehicle": 1}
NEAR_DUPLICATE_WARNING_DISTANCE = 4


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_safe(path: Path, value: Any) -> None:
    data = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite differing artifact: {path}")
        return
    path.write_bytes(data)


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def write_csv_safe(path: Path, rows: list[dict[str, Any]]) -> None:
    data = csv_bytes(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite differing artifact: {path}")
        return
    path.write_bytes(data)


def link_or_copy_safe(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(source) != file_sha256(target):
            raise FileExistsError(f"Existing dataset file differs: {target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def label_text(frame: dict[str, Any], width: int, height: int) -> str:
    lines = []
    for item in frame.get("ground_truth", []):
        class_name = str(item.get("class_name"))
        if class_name not in CLASS_IDS:
            raise ValueError(f"Unsupported class name: {class_name}")
        x, y, w, h = (float(value) for value in item["bbox_xywh"])
        if width <= 0 or height <= 0 or w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(f"Invalid visible-pixel box: {item['bbox_xywh']}")
        values = ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(f"Out-of-range normalized box: {values}")
        lines.append(f"{CLASS_IDS[class_name]} " + " ".join(f"{value:.10f}" for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def draw_diagnostics(manifest: dict[str, Any], target_dir: Path) -> list[str]:
    candidates = [frame for frame in manifest["frames"] if frame.get("ground_truth")]
    if len(candidates) < 5:
        raise ValueError(f"At least five object-bearing diagnostic candidates are required: {manifest['scenario_id']}")
    positions = [round(index * (len(candidates) - 1) / 4) for index in range(5)]
    outputs = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for order, position in enumerate(positions, start=1):
        frame = candidates[position]
        with Image.open(frame["image_path"]).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for gt in frame["ground_truth"]:
                x, y, w, h = (float(v) for v in gt["bbox_xywh"])
                color = (255, 40, 40) if gt["class_name"] == "person" else (30, 120, 255)
                draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
                draw.text((x, max(0, y - 12)), gt["class_name"], fill=color)
            output = target_dir / f"gt_{order:02d}_frame_{int(frame['frame_index']):04d}.png"
            if output.exists():
                raise FileExistsError(output)
            image.save(output)
            outputs.append(str(output.resolve()))
    return outputs


def validate_and_package(
    manifest: dict[str, Any], scenario: dict[str, Any], dataset_root: Path, diagnostic_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[str] = []
    width, height = int(manifest["image_width"]), int(manifest["image_height"])
    frames = manifest["frames"]
    actual_indices = [int(frame["frame_index"]) for frame in frames]
    if actual_indices != SELECTED_FRAMES or int(manifest["evaluated_frame_count"]) != 20:
        errors.append(f"selected_frame_mismatch:{actual_indices}")
    if manifest.get("scenario_id") != scenario["scenario_id"] or manifest.get("split") != scenario["split"]:
        errors.append("scenario_or_split_mismatch")
    if manifest.get("ground_truth_mode") != GT_VERSION:
        errors.append("ground_truth_version_mismatch")
    if manifest.get("scenario_config_sha256") != scenario["scenario_config_sha256"]:
        errors.append("scenario_config_hash_mismatch")

    rows: list[dict[str, Any]] = []
    person_total = vehicle_total = object_frames = both_class_frames = 0
    invalid_boxes = invalid_classes = alignment_errors = 0
    for frame in frames:
        frame_index = int(frame["frame_index"])
        source = Path(frame["image_path"])
        if not source.is_file() or source.stem != f"frame_{frame_index:04d}":
            alignment_errors += 1
            errors.append(f"source_frame_alignment:{frame_index}")
            continue
        frame_person = sum(x.get("class_name") == "person" for x in frame.get("ground_truth", []))
        frame_vehicle = sum(x.get("class_name") == "vehicle" for x in frame.get("ground_truth", []))
        person_total += frame_person
        vehicle_total += frame_vehicle
        object_frames += int(frame_person + frame_vehicle > 0)
        both_class_frames += int(frame_person > 0 and frame_vehicle > 0)
        try:
            text = label_text(frame, width, height)
        except ValueError as exc:
            errors.append(f"label_validation:{frame_index}:{exc}")
            invalid_boxes += 1
            text = ""
        for gt in frame.get("ground_truth", []):
            invalid_classes += int(gt.get("class_name") not in CLASS_IDS)
        stem = f"{scenario['scenario_id']}__frame_{frame_index:04d}"
        image_target = dataset_root / "images" / scenario["split"] / f"{stem}.png"
        label_target = dataset_root / "labels" / scenario["split"] / f"{stem}.txt"
        link_or_copy_safe(source, image_target)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        encoded = text.encode("ascii")
        if label_target.exists() and label_target.read_bytes() != encoded:
            raise FileExistsError(f"Existing label differs: {label_target}")
        if not label_target.exists():
            label_target.write_bytes(encoded)
        image_hash = file_sha256(image_target)
        label_hash = file_sha256(label_target)
        frame_pair_hash = hashlib.sha256((image_hash + "|" + label_hash).encode("ascii")).hexdigest().upper()
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "split": scenario["split"],
                "seed": scenario["seed"],
                "source_frame_number": frame_index,
                "time_seconds": float(frame["time_seconds"]),
                "uav_xyz": frame["uav_xyz"],
                "image_path": str(image_target.resolve()),
                "label_path": str(label_target.resolve()),
                "image_sha256": image_hash,
                "label_sha256": label_hash,
                "frame_pair_sha256": frame_pair_hash,
                "perceptual_dhash64": f"{dhash(image_target):016X}",
                "person_gt": frame_person,
                "vehicle_gt": frame_vehicle,
                "label_empty": not bool(text.strip()),
            }
        )
    if person_total <= 0 or vehicle_total <= 0:
        errors.append("missing_required_class")
    if person_total < 3 or vehicle_total < 3 or both_class_frames < 3:
        errors.append("insufficient_visible_class_examples")
    if invalid_classes:
        errors.append(f"invalid_classes:{invalid_classes}")
    diagnostic_paths = draw_diagnostics(manifest, diagnostic_root) if object_frames >= 5 else []
    if len(diagnostic_paths) < 5:
        errors.append("insufficient_diagnostic_images")
    summary = {
        "scenario_id": scenario["scenario_id"],
        "split": scenario["split"],
        "seed": scenario["seed"],
        "source_frame_count": int(manifest["total_simulation_frames"]),
        "selected_frame_count": len(frames),
        "selected_frame_indices": actual_indices,
        "person_gt_count": person_total,
        "vehicle_gt_count": vehicle_total,
        "object_frame_count": object_frames,
        "both_class_frame_count": both_class_frames,
        "empty_frame_count": len(frames) - object_frames,
        "out_of_bounds_or_nonpositive_box_count": invalid_boxes,
        "invalid_class_count": invalid_classes,
        "frame_label_alignment_error_count": alignment_errors,
        "diagnostic_image_count": len(diagnostic_paths),
        "diagnostic_images": diagnostic_paths,
        "manifest_path": str((Path(manifest["frames"][0]["image_path"]).parent.parent / "frame_manifest.json").resolve()),
        "scenario_config_sha256": scenario["scenario_config_sha256"],
        "valid": not errors,
        "errors": errors,
    }
    return summary, rows


def collision_groups(rows: list[dict[str, Any]], key: str, *, exclude_empty_labels: bool = False) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if exclude_empty_labels and row.get("label_empty"):
            continue
        groups[str(row[key])].append(row)
    collisions = []
    for value, items in groups.items():
        identities = {(x["partition"], x["scenario_id"], int(x["source_frame_number"])) for x in items}
        if len(identities) > 1:
            collisions.append(
                {
                    "sha256": value,
                    "count": len(items),
                    "partitions": sorted({x["partition"] for x in items}),
                    "members": [
                        {"partition": x["partition"], "scenario_id": x["scenario_id"], "source_frame_number": x["source_frame_number"], "image_path": x["image_path"]}
                        for x in items
                    ],
                }
            )
    return collisions


def fixed_evaluation_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = read_json(FIXED_PLAN)
    rows = []
    manifest_hashes = {}
    for scenario in plan["scenarios"]:
        manifest_path = FIXED_SCENARIOS / scenario["scenario_id"] / "initial_condition" / "frame_manifest.json"
        manifest = read_json(manifest_path)
        manifest_hashes[scenario["scenario_id"]] = file_sha256(manifest_path)
        width, height = int(manifest["image_width"]), int(manifest["image_height"])
        for frame in manifest["frames"]:
            path = Path(frame["image_path"])
            text = label_text(frame, width, height)
            image_hash = file_sha256(path)
            label_hash = hashlib.sha256(text.encode("ascii")).hexdigest().upper()
            rows.append(
                {
                    "partition": "test",
                    "scenario_id": scenario["scenario_id"],
                    "source_frame_number": int(frame["frame_index"]),
                    "image_path": str(path.resolve()),
                    "image_sha256": image_hash,
                    "label_sha256": label_hash,
                    "frame_pair_sha256": hashlib.sha256((image_hash + "|" + label_hash).encode("ascii")).hexdigest().upper(),
                    "perceptual_dhash64": f"{dhash(path):016X}",
                    "label_empty": not bool(text.strip()),
                }
            )
    return rows, {"plan_path": str(FIXED_PLAN.resolve()), "plan_sha256": file_sha256(FIXED_PLAN), "manifest_sha256": manifest_hashes}


def near_summary(left: list[dict[str, Any]], right: list[dict[str, Any]], name: str) -> dict[str, Any]:
    histogram = [0] * 65
    closest: list[dict[str, Any]] = []
    warning_count = 0
    minimum = 65
    for a in left:
        ah = int(a["perceptual_dhash64"], 16)
        for b in right:
            distance = (ah ^ int(b["perceptual_dhash64"], 16)).bit_count()
            histogram[distance] += 1
            minimum = min(minimum, distance)
            warning_count += int(distance <= NEAR_DUPLICATE_WARNING_DISTANCE)
            candidate = {
                "distance": distance,
                "left_scenario": a["scenario_id"],
                "left_frame": a["source_frame_number"],
                "left_image": a["image_path"],
                "right_scenario": b["scenario_id"],
                "right_frame": b["source_frame_number"],
                "right_image": b["image_path"],
            }
            if len(closest) < 20 or distance < closest[-1]["distance"]:
                closest.append(candidate)
                closest.sort(key=lambda x: (x["distance"], x["left_scenario"], x["left_frame"], x["right_scenario"], x["right_frame"]))
                del closest[20:]
    total = sum(histogram)
    distances: list[int] = []
    for value, count in enumerate(histogram):
        distances.extend([value] * count)
    return {
        "comparison": name,
        "left_image_count": len(left),
        "right_image_count": len(right),
        "pair_count": total,
        "minimum_hamming_distance": minimum if total else None,
        "median_hamming_distance": statistics.median(distances) if distances else None,
        "mean_hamming_distance": statistics.fmean(distances) if distances else None,
        "warning_threshold": f"dHash Hamming distance <= {NEAR_DUPLICATE_WARNING_DISTANCE}",
        "warning_pair_count": warning_count,
        "distance_histogram": {str(index): count for index, count in enumerate(histogram) if count},
        "closest_pairs": closest,
    }


def verify_preservation() -> dict[str, Any]:
    snapshot = read_json(CONFIG_ROOT / "existing_state_audit.json")["preservation_snapshot"]
    checks = []
    for item in snapshot:
        path = Path(item["path"])
        exists = path.exists()
        actual = file_sha256(path) if path.is_file() else None
        checks.append({"path": str(path), "exists": exists, "expected_sha256": item["sha256"], "actual_sha256": actual, "unchanged": exists and actual == item["sha256"]})
    return {"all_unchanged": all(x["unchanged"] for x in checks), "checks": checks}


def dataset_audit_markdown(summary: dict[str, Any], exact: dict[str, Any], near: dict[str, Any], preservation: dict[str, Any]) -> str:
    scenario_lines = "\n".join(
        f"| {x['scenario_id']} | {x['split']} | {x['selected_frame_count']} | {x['person_gt_count']} | {x['vehicle_gt_count']} | {x['both_class_frame_count']} | {x['empty_frame_count']} | {'PASS' if x['valid'] else 'FAIL'} |"
        for x in summary["scenarios"]
    )
    near_lines = "\n".join(
        f"| {x['comparison']} | {x['pair_count']} | {x['minimum_hamming_distance']} | {x['median_hamming_distance']} | {x['mean_hamming_distance']:.3f} | {x['warning_pair_count']} |"
        for x in near["comparisons"]
    )
    return f"""# 시나리오 다양화 자료 감사

## 판정

- 전체 자료 감사: **{'PASS' if summary['all_valid'] and exact['passed'] and near['review_passed'] and preservation['all_unchanged'] else 'FAIL'}**
- 학습/검증: {summary['train_scenario_count']}/{summary['validation_scenario_count']}개 시나리오, {summary['train_image_count']}/{summary['validation_image_count']}장
- 선택 프레임: 18초 181프레임 중 사전 확정한 20개({SELECTED_FRAMES})
- GT: 실제 가시 픽셀 인스턴스 색상 렌더(`{GT_VERSION}`)
- 기존 핵심 파일 보존: {'PASS' if preservation['all_unchanged'] else 'FAIL'}

## 시나리오별 GT

| ID | 분할 | 영상 | 사람 GT | 차량 GT | 두 클래스 동시 프레임 | 빈 프레임 | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|
{scenario_lines}

범위 밖/비양수 상자, 잘못된 클래스, 프레임-라벨 번호 불일치는 모두 0이어야 통과한다. 각 시나리오마다 5장의 오버레이 진단 영상을 생성하였다.

## 시나리오 단위 분리와 정확 중복

- ID/시드/궤적/객체 배치 중복: {exact['identity_collision_count']}건
- 영상 SHA-256 중복: {len(exact['image_collisions'])}건
- 영상+라벨 쌍 중복: {len(exact['frame_pair_collisions'])}건
- 비어 있지 않은 정답 SHA-256 중복: {len(exact['nonempty_label_collisions'])}건
- 정확 중복 판정: **{'PASS' if exact['passed'] else 'FAIL'}**

빈 정답 파일은 내용상 같은 SHA-256이 불가피하므로 원시 라벨 중복 판정에서 제외하되, 해당 프레임의 영상+라벨 결합 해시는 계속 검사하였다.

## 지각 해시 근접 중복

64비트 dHash의 해밍 거리를 사용했으며 경고 기준은 ≤ {NEAR_DUPLICATE_WARNING_DISTANCE}이다. 경고는 삭제나 재구성을 자동 요구하지 않고, 정확 SHA 및 사전 확정된 궤적·배치가 서로 다른지 함께 검토한다.

| 비교 | 쌍 수 | 최소 거리 | 중앙값 | 평균 | 경고 쌍 |
|---|---:|---:|---:|---:|---:|
{near_lines}

근접 중복 검토 판정: **{'PASS' if near['review_passed'] else 'FAIL'}**. 가장 가까운 영상 쌍은 `near_duplicate_audit.json`에 경로와 함께 기록하였다.

## 해석 범위

자료 다양화는 궤적 축·방향, 상승/하강, 시작·종료 위치, 고도, 객체 배치·간격·부분 가림, 관측 거리, 정상 범위 환경 조건을 대상으로 했다. 카메라 방향·기울기와 다중 지형/배경은 현 렌더러가 고정값만 지원하므로 변경하지 않았다.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=CONFIG_ROOT / "scenario_plan.json")
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--skip-render", action="store_true", help="Package manifests already rendered by render_registered_dataset.m")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = read_json(plan_path)
    prereg = read_json(CONFIG_ROOT / "pre_registration.json")
    if file_sha256(plan_path) != prereg["scenario_plan_sha256"]:
        raise RuntimeError("Pre-registered scenario plan hash changed")
    for scenario in plan["scenarios"]:
        if file_sha256(Path(scenario["scenario_config_path"])) != scenario["scenario_config_sha256"]:
            raise RuntimeError(f"Scenario config hash changed: {scenario['scenario_id']}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = output_root / "dataset"

    engine = None
    if not args.skip_render:
        import matlab.engine

        engine = matlab.engine.start_matlab("-nodesktop -nosplash")
    scenario_summaries: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        if engine is not None:
            engine.cd(str(REPO_ROOT), nargout=0)
            engine.addpath(str(REPO_ROOT), nargout=0)
            engine.addpath(str(HERE.parent), nargout=0)
            engine.addpath(str(HERE), nargout=0)
        for scenario in plan["scenarios"]:
            scenario_dir = output_root / "rendered_scenarios" / scenario["scenario_id"]
            manifest_path = scenario_dir / "frame_manifest.json"
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
            elif engine is not None:
                raw = engine.export_training_scenario_frames(
                    str(scenario_dir), scenario["scenario_config_path"], scenario["scenario_config_sha256"], nargout=1
                )
                manifest = json.loads(str(raw))
            else:
                raise FileNotFoundError(f"Missing pre-rendered manifest in --skip-render mode: {manifest_path}")
            summary, rows = validate_and_package(manifest, scenario, dataset_root, output_root / "diagnostics" / scenario["scenario_id"])
            scenario_summaries.append(summary)
            new_rows.extend(rows)
            print(json.dumps({"scenario": scenario["scenario_id"], "valid": summary["valid"], "person_gt": summary["person_gt_count"], "vehicle_gt": summary["vehicle_gt_count"]}, ensure_ascii=False), flush=True)
    finally:
        if engine is not None:
            engine.quit()
    render_wall_seconds = time.perf_counter() - started

    for row in new_rows:
        row["partition"] = row.pop("split")
    test_rows, fixed_hashes = fixed_evaluation_rows()
    all_rows = new_rows + test_rows

    plan_ids = [x["scenario_id"] for x in plan["scenarios"]]
    plan_seeds = [int(x["seed"]) for x in plan["scenarios"]]
    plan_traj = [x["trajectory_sha256"] for x in plan["scenarios"]]
    plan_layout = [x["object_layout_sha256"] for x in plan["scenarios"]]
    fixed_plan = read_json(FIXED_PLAN)
    fixed_ids = [x["scenario_id"] for x in fixed_plan["scenarios"]]
    fixed_seeds = [int(x["seed"]) for x in fixed_plan["scenarios"]]
    fixed_traj = [hashlib.sha256(canonical_bytes({"start": x["uav_initial_xyz"], "end": x["uav_end_xyz"]})).hexdigest().upper() for x in fixed_plan["scenarios"]]
    fixed_layout = [hashlib.sha256(canonical_bytes([{"class_id": o["class_id"], "xy": o["xy"], "radius_height": o["radius_height"]} for o in x["object_layout"]])).hexdigest().upper() for x in fixed_plan["scenarios"]]
    identity_collisions = {
        "scenario_id": sorted(set(plan_ids) & set(fixed_ids)),
        "seed": sorted(set(plan_seeds) & set(fixed_seeds)),
        "trajectory": sorted(set(plan_traj) & set(fixed_traj)),
        "object_layout": sorted(set(plan_layout) & set(fixed_layout)),
        "within_new_scenario_id": len(plan_ids) - len(set(plan_ids)),
        "within_new_seed": len(plan_seeds) - len(set(plan_seeds)),
        "within_new_trajectory": len(plan_traj) - len(set(plan_traj)),
        "within_new_object_layout": len(plan_layout) - len(set(plan_layout)),
    }
    identity_collision_count = sum(len(v) if isinstance(v, list) else int(v) for v in identity_collisions.values())
    image_collisions = collision_groups(all_rows, "image_sha256")
    frame_collisions = collision_groups(all_rows, "frame_pair_sha256")
    label_collisions = collision_groups(all_rows, "label_sha256", exclude_empty_labels=True)
    exact = {
        "schema_version": "1.0",
        "audited_partition_counts": {name: sum(x["partition"] == name for x in all_rows) for name in ("train", "val", "test")},
        "identity_collisions": identity_collisions,
        "identity_collision_count": identity_collision_count,
        "image_collisions": image_collisions,
        "frame_pair_collisions": frame_collisions,
        "nonempty_label_collisions": label_collisions,
        "empty_label_policy": "Empty label hashes are excluded from raw label collision failure; image+label pair hashes remain included.",
        "fixed_evaluation_hashes": fixed_hashes,
    }
    exact["passed"] = identity_collision_count == 0 and not image_collisions and not frame_collisions and not label_collisions
    write_json_safe(output_root / "exact_duplicate_audit.json", exact)

    partitions = {name: [x for x in all_rows if x["partition"] == name] for name in ("train", "val", "test")}
    comparisons = [
        near_summary(partitions["train"], partitions["val"], "train-vs-validation"),
        near_summary(partitions["train"], partitions["test"], "train-vs-S0-S4"),
        near_summary(partitions["val"], partitions["test"], "validation-vs-S0-S4"),
    ]
    near = {
        "schema_version": "1.0",
        "algorithm": "64-bit difference hash (9x8 grayscale LANCZOS, horizontal adjacent-pixel comparisons)",
        "warning_distance": NEAR_DUPLICATE_WARNING_DISTANCE,
        "comparisons": comparisons,
        "exact_sha_collision_count": len(image_collisions),
        "structural_review": "All warning pairs are accepted only if exact image SHA differs and scenario seed, trajectory, and object layout audits pass.",
    }
    near["review_passed"] = not image_collisions and identity_collision_count == 0
    write_json_safe(output_root / "near_duplicate_audit.json", near)

    split = {
        "plan_id": plan["plan_id"],
        "policy": plan["split_policy"],
        "train_scenarios": [x["scenario_id"] for x in plan["scenarios"] if x["split"] == "train"],
        "validation_scenarios": [x["scenario_id"] for x in plan["scenarios"] if x["split"] == "val"],
        "fixed_test_scenarios": fixed_ids,
        "scenario_overlap": [],
        "passed": not identity_collision_count,
    }
    write_json_safe(output_root / "scenario_split.json", split)

    manifest_rows = [{key: value for key, value in row.items() if key not in {"partition"}} | {"split": row["partition"]} for row in new_rows]
    write_csv_safe(output_root / "dataset_manifest.csv", manifest_rows)
    write_csv_safe(output_root / "ground_truth_validation.csv", scenario_summaries)
    write_json_safe(output_root / "ground_truth_validation.json", {"scenarios": scenario_summaries, "all_valid": all(x["valid"] for x in scenario_summaries)})
    preservation = verify_preservation()
    write_json_safe(output_root / "existing_results_preservation_check.json", preservation)

    train_rows = partitions["train"]
    val_rows = partitions["val"]
    summary = {
        "plan_id": plan["plan_id"],
        "plan_path": str(plan_path),
        "plan_sha256": file_sha256(plan_path),
        "pre_registration_path": str((CONFIG_ROOT / "pre_registration.json").resolve()),
        "pre_registration_sha256": file_sha256(CONFIG_ROOT / "pre_registration.json"),
        "render_and_audit_wall_seconds": render_wall_seconds,
        "train_scenario_count": len(split["train_scenarios"]),
        "validation_scenario_count": len(split["validation_scenarios"]),
        "train_image_count": len(train_rows),
        "validation_image_count": len(val_rows),
        "person_gt_count": sum(int(x["person_gt_count"]) for x in scenario_summaries),
        "vehicle_gt_count": sum(int(x["vehicle_gt_count"]) for x in scenario_summaries),
        "train_person_gt_count": sum(int(x["person_gt"]) for x in train_rows),
        "train_vehicle_gt_count": sum(int(x["vehicle_gt"]) for x in train_rows),
        "validation_person_gt_count": sum(int(x["person_gt"]) for x in val_rows),
        "validation_vehicle_gt_count": sum(int(x["vehicle_gt"]) for x in val_rows),
        "empty_frame_count": sum(int(x["empty_frame_count"]) for x in scenario_summaries),
        "diagnostic_image_count": sum(int(x["diagnostic_image_count"]) for x in scenario_summaries),
        "all_valid": all(x["valid"] for x in scenario_summaries),
        "exact_duplicate_audit_passed": exact["passed"],
        "near_duplicate_review_passed": near["review_passed"],
        "scenario_split_passed": split["passed"],
        "existing_results_preserved": preservation["all_unchanged"],
        "scenarios": scenario_summaries,
    }
    write_json_safe(output_root / "dataset_summary.json", summary)

    data_yaml = output_root / "data.yaml"
    yaml_data = yaml.safe_dump({"path": str(dataset_root.resolve()), "train": "images/train", "val": "images/val", "names": {0: "person", 1: "vehicle"}}, sort_keys=False).encode("utf-8")
    if data_yaml.exists() and data_yaml.read_bytes() != yaml_data:
        raise FileExistsError(data_yaml)
    if not data_yaml.exists():
        data_yaml.write_bytes(yaml_data)
    audit_md = dataset_audit_markdown(summary, exact, near, preservation).encode("utf-8")
    audit_path = output_root / "dataset_audit_ko.md"
    if audit_path.exists() and audit_path.read_bytes() != audit_md:
        raise FileExistsError(audit_path)
    if not audit_path.exists():
        audit_path.write_bytes(audit_md)

    passed = all(
        [
            summary["all_valid"],
            exact["passed"],
            near["review_passed"],
            split["passed"],
            preservation["all_unchanged"],
            len(train_rows) == 360,
            len(val_rows) == 100,
        ]
    )
    gate = {"training_allowed": passed, "checks": summary, "data_yaml": str(data_yaml.resolve())}
    write_json_safe(output_root / "training_gate.json", gate)
    print(json.dumps({"output_root": str(output_root), "training_allowed": passed, "summary": summary}, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("Dataset pre-training gate failed; YOLO training is blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
