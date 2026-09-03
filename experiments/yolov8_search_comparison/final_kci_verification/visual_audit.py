"""Prepare and finalize direct visual review of dHash warning pairs.

The preparation stage creates lossless, GT-annotated side-by-side sheets.  The
finalization stage requires an explicit manual_decisions.json, so perceptual
warnings cannot be accepted solely from hashes or scenario metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DIVERSE_ROOT = REPO_ROOT / "experiments" / "yolov8_search_comparison" / "diverse_training_scenario_split"
DATA_ROOT = DIVERSE_ROOT / "data" / "diverse_training_scenario_split_v2"
FIXED_ROOT = REPO_ROOT / "experiments" / "yolov8_search_comparison" / "multi_scenario_symmetric"
FIXED_PLAN_ID = "multi_scenario_symmetric_plan_v2"
AUDIT_ID = "near_duplicate_visual_audit_v1"
OUTPUT_ROOT = HERE / "outputs" / AUDIT_ID
SHEET_ROOT = OUTPUT_ROOT / "comparison_sheets"
PAIR_ROOT = OUTPUT_ROOT / "pair_images"
DECISIONS_PATH = HERE / "config" / AUDIT_ID / "manual_decisions.json"
WARNING_THRESHOLD = 4


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json_new(path: Path, value: Any) -> None:
    write_text_new(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\malgun.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_train_manifest() -> dict[tuple[str, int], dict[str, str]]:
    path = DATA_ROOT / "dataset_manifest.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["scenario_id"], int(row["source_frame_number"])): row for row in rows}


def yolo_boxes(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id_text, cx_text, cy_text, w_text, h_text = line.split()
        class_id = int(class_id_text)
        cx, cy, bw, bh = map(float, (cx_text, cy_text, w_text, h_text))
        boxes.append(
            {
                "class_name": "person" if class_id == 0 else "vehicle",
                "bbox_xywh": [
                    (cx - bw / 2) * width,
                    (cy - bh / 2) * height,
                    bw * width,
                    bh * height,
                ],
            }
        )
    return boxes


def fixed_frame(scenario_id: str, frame_index: int) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = FIXED_ROOT / "scenarios" / FIXED_PLAN_ID / scenario_id / "initial_condition" / "frame_manifest.json"
    manifest = read_json(manifest_path)
    frame = next(item for item in manifest["frames"] if int(item["frame_index"]) == frame_index)
    return Path(frame["image_path"]), frame["ground_truth"]


def draw_annotated(image_path: Path, boxes: list[dict[str, Any]], title: str) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(16)
    for box in boxes:
        x, y, width, height = map(float, box["bbox_xywh"])
        color = (255, 40, 40) if box["class_name"] == "person" else (0, 220, 255)
        draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
        draw.text((max(0, x), max(0, y - 18)), box["class_name"], fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    header_height = 58
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    canvas.paste(image, (0, header_height))
    ImageDraw.Draw(canvas).multiline_text((8, 6), title, fill="black", font=font, spacing=2)
    return canvas


def prepare() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)
    SHEET_ROOT.mkdir(parents=True)
    PAIR_ROOT.mkdir(parents=True)

    near = read_json(DATA_ROOT / "near_duplicate_audit.json")
    comparison = next(item for item in near["comparisons"] if item["comparison"] == "train-vs-S0-S4")
    warnings = [item for item in comparison["closest_pairs"] if int(item["distance"]) <= WARNING_THRESHOLD]
    if len(warnings) != int(comparison["warning_pair_count"]):
        raise RuntimeError(f"Warning count mismatch: reconstructed={len(warnings)}, recorded={comparison['warning_pair_count']}")
    train_manifest = load_train_manifest()
    review_rows: list[dict[str, Any]] = []
    pair_images: list[Path] = []
    for index, warning in enumerate(warnings, start=1):
        pair_id = f"P{index:02d}"
        left_key = (warning["left_scenario"], int(warning["left_frame"]))
        left_row = train_manifest[left_key]
        left_path = Path(left_row["image_path"])
        with Image.open(left_path) as image:
            left_size = image.size
        left_boxes = yolo_boxes(Path(left_row["label_path"]), *left_size)
        right_path, right_boxes = fixed_frame(warning["right_scenario"], int(warning["right_frame"]))
        left_counts = Counter(box["class_name"] for box in left_boxes)
        right_counts = Counter(box["class_name"] for box in right_boxes)
        left_title = (
            f"{pair_id} TRAIN {warning['left_scenario']} frame {int(warning['left_frame']):04d}\n"
            f"person={left_counts['person']} vehicle={left_counts['vehicle']}"
        )
        right_title = (
            f"dHash={warning['distance']}  FIXED {warning['right_scenario']} frame {int(warning['right_frame']):04d}\n"
            f"person={right_counts['person']} vehicle={right_counts['vehicle']}"
        )
        left_annotated = draw_annotated(left_path, left_boxes, left_title)
        right_annotated = draw_annotated(right_path, right_boxes, right_title)
        pair_canvas = Image.new("RGB", (left_annotated.width + right_annotated.width, max(left_annotated.height, right_annotated.height)), "white")
        pair_canvas.paste(left_annotated, (0, 0))
        pair_canvas.paste(right_annotated, (left_annotated.width, 0))
        pair_path = PAIR_ROOT / f"{pair_id}.png"
        pair_canvas.save(pair_path, format="PNG", dpi=(300, 300), optimize=True)
        pair_images.append(pair_path)
        review_rows.append(
            {
                "pair_id": pair_id,
                "dhash_distance": int(warning["distance"]),
                "train_scenario": warning["left_scenario"],
                "train_frame": int(warning["left_frame"]),
                "train_image_path": str(left_path.resolve()),
                "train_image_sha256": sha256(left_path),
                "train_person_gt": left_counts["person"],
                "train_vehicle_gt": left_counts["vehicle"],
                "train_empty": not left_boxes,
                "test_scenario": warning["right_scenario"],
                "test_frame": int(warning["right_frame"]),
                "test_image_path": str(right_path.resolve()),
                "test_image_sha256": sha256(right_path),
                "test_person_gt": right_counts["person"],
                "test_vehicle_gt": right_counts["vehicle"],
                "test_empty": not right_boxes,
                "pair_image_path": str(pair_path.resolve()),
            }
        )

    sheet_paths: list[str] = []
    for sheet_index in range(0, len(pair_images), 2):
        selected = pair_images[sheet_index : sheet_index + 2]
        opened: list[Image.Image] = []
        for path in selected:
            with Image.open(path) as image:
                opened.append(image.convert("RGB"))
        width = max(image.width for image in opened)
        height = sum(image.height for image in opened)
        sheet = Image.new("RGB", (width, height), "white")
        y = 0
        for image in opened:
            sheet.paste(image, (0, y))
            y += image.height
        start = sheet_index + 1
        end = sheet_index + len(opened)
        sheet_path = SHEET_ROOT / f"sheet_{start:02d}_{end:02d}.png"
        sheet.save(sheet_path, format="PNG", dpi=(300, 300), optimize=True)
        sheet_paths.append(str(sheet_path.resolve()))

    manifest = {
        "audit_id": AUDIT_ID,
        "source_near_duplicate_audit": str((DATA_ROOT / "near_duplicate_audit.json").resolve()),
        "source_sha256": sha256(DATA_ROOT / "near_duplicate_audit.json"),
        "recorded_warning_count": comparison["warning_pair_count"],
        "reconstructed_warning_count": len(review_rows),
        "warning_threshold": f"dHash Hamming distance <= {WARNING_THRESHOLD}",
        "annotation_legend": {"person": "red", "vehicle": "cyan"},
        "pairs": review_rows,
        "comparison_sheets": sheet_paths,
        "manual_decisions_required": str(DECISIONS_PATH.resolve()),
    }
    write_json_new(OUTPUT_ROOT / "review_manifest.json", manifest)
    write_csv_new(OUTPUT_ROOT / "review_metadata.csv", review_rows)
    print(json.dumps({"warning_count": len(review_rows), "sheets": sheet_paths, "manifest": str((OUTPUT_ROOT / 'review_manifest.json').resolve())}, ensure_ascii=False, indent=2))


def finalize() -> None:
    manifest = read_json(OUTPUT_ROOT / "review_manifest.json")
    decisions_doc = read_json(DECISIONS_PATH)
    decisions = {item["pair_id"]: item for item in decisions_doc["decisions"]}
    expected_ids = {item["pair_id"] for item in manifest["pairs"]}
    if set(decisions) != expected_ids:
        raise RuntimeError(f"Manual decision IDs differ: expected={sorted(expected_ids)}, actual={sorted(decisions)}")
    allowed_categories = {
        1: "빈 화면 또는 배경만 유사",
        2: "객체가 있지만 구성이 충분히 다름",
        3: "객체 위치·크기·배경까지 매우 유사",
    }
    allowed_risks = {"낮음", "중간", "높음", "판단 불가"}
    final_rows: list[dict[str, Any]] = []
    for metadata in manifest["pairs"]:
        decision = decisions[metadata["pair_id"]]
        category = int(decision["classification"])
        if category not in allowed_categories:
            raise ValueError(f"Invalid classification for {metadata['pair_id']}: {category}")
        if decision["risk"] not in allowed_risks:
            raise ValueError(f"Invalid risk for {metadata['pair_id']}: {decision['risk']}")
        final_rows.append(
            {
                **metadata,
                "object_position_similarity": decision["object_position_similarity"],
                "object_size_similarity": decision["object_size_similarity"],
                "occlusion_similarity": decision["occlusion_similarity"],
                "background_similarity": decision["background_similarity"],
                "classification": category,
                "classification_label": allowed_categories[category],
                "risk": decision["risk"],
                "judgement_basis": decision["judgement_basis"],
                "reviewer": decisions_doc["reviewer"],
                "reviewed_at": decisions_doc["reviewed_at"],
            }
        )
    category_counts = Counter(row["classification"] for row in final_rows)
    risk_counts = Counter(row["risk"] for row in final_rows)
    exact_duplicate_count = sum(row["train_image_sha256"] == row["test_image_sha256"] for row in final_rows)
    object_scene_pairs = [row for row in final_rows if not row["train_empty"] or not row["test_empty"]]
    high_similarity_count = sum(row["classification"] == 3 for row in final_rows)
    summary = {
        "audit_id": AUDIT_ID,
        "pair_count": len(final_rows),
        "category_counts": {str(key): category_counts.get(key, 0) for key in allowed_categories},
        "risk_counts": dict(risk_counts),
        "exact_duplicate_count": exact_duplicate_count,
        "object_involved_pair_count": len(object_scene_pairs),
        "high_similarity_object_scene_count": high_similarity_count,
        "final_judgement": decisions_doc["final_judgement"],
        "independence_statement": decisions_doc["independence_statement"],
        "allowed_claims": decisions_doc["allowed_claims"],
        "forbidden_claims": decisions_doc["forbidden_claims"],
        "comparison_sheets": manifest["comparison_sheets"],
    }
    csv_path = OUTPUT_ROOT / "near_duplicate_visual_audit.csv"
    md_path = OUTPUT_ROOT / "near_duplicate_visual_audit_ko.md"
    write_csv_new(csv_path, final_rows)
    lines = [
        "# 학습–S0~S4 근접 영상 직접 시각 감사",
        "",
        f"- 원 감사 경고: {manifest['recorded_warning_count']}쌍",
        f"- 직접 재구성 경고: {manifest['reconstructed_warning_count']}쌍",
        f"- 기준: {manifest['warning_threshold']}",
        f"- 정확 영상 SHA-256 중복: {exact_duplicate_count}쌍",
        f"- 객체가 한쪽 이상 포함된 쌍: {len(object_scene_pairs)}쌍",
        f"- 분류 1/2/3: {category_counts.get(1, 0)}/{category_counts.get(2, 0)}/{category_counts.get(3, 0)}쌍",
        f"- 위험도: {dict(risk_counts)}",
        "",
        "## 쌍별 직접 판정",
        "",
        "| ID | 거리 | 학습 장면/GT | 고정시험 장면/GT | 객체·배경 비교 | 분류 | 위험 | 판단 근거 |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for row in final_rows:
        comparison = (
            f"위치 {row['object_position_similarity']}; 크기 {row['object_size_similarity']}; "
            f"가림 {row['occlusion_similarity']}; 배경 {row['background_similarity']}"
        )
        lines.append(
            f"| {row['pair_id']} | {row['dhash_distance']} | {row['train_scenario']}:{row['train_frame']} "
            f"P{row['train_person_gt']}/V{row['train_vehicle_gt']} | {row['test_scenario']}:{row['test_frame']} "
            f"P{row['test_person_gt']}/V{row['test_vehicle_gt']} | {comparison} | "
            f"{row['classification']}. {row['classification_label']} | {row['risk']} | {row['judgement_basis']} |"
        )
    lines.extend(
        [
            "",
            "## 최종 판단",
            "",
            decisions_doc["final_judgement"],
            "",
            "### 시험자료 독립성 표현",
            "",
            decisions_doc["independence_statement"],
            "",
            "### 사용할 수 있는 주장",
            "",
            *[f"- {item}" for item in decisions_doc["allowed_claims"]],
            "",
            "### 사용하면 안 되는 주장",
            "",
            *[f"- {item}" for item in decisions_doc["forbidden_claims"]],
            "",
            "## 비교 이미지",
            "",
            *[f"- `{path}`" for path in manifest["comparison_sheets"]],
            "",
        ]
    )
    write_text_new(md_path, "\n".join(lines))
    write_json_new(OUTPUT_ROOT / "near_duplicate_visual_audit_summary.json", summary)
    # Keep the completion message compatible with the Windows cp949 console;
    # UTF-8 artifacts above retain their original Korean text.
    print(json.dumps({"csv": str(csv_path.resolve()), "report": str(md_path.resolve()), "summary": summary}, ensure_ascii=True, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "finalize"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
