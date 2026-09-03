"""Compact completed training images while keeping the YOLO dataset usable."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()
    root = args.root.resolve()
    dataset = root / "dataset"
    plan = json.loads((root / "scenario_plan_snapshot.json").read_text(encoding="utf-8"))

    smoke_stems = {
        path.stem
        for path in (dataset / "smoke" / "images").rglob("*.png")
    }
    for target in (dataset / "images", dataset / "smoke" / "images"):
        resolved = target.resolve()
        if resolved.is_dir():
            if root not in resolved.parents:
                raise RuntimeError(f"Refusing to remove directory outside training root: {resolved}")
            shutil.rmtree(resolved)

    index: dict[tuple[str, int], tuple[Path, str, str]] = {}
    before = after = converted = 0
    split_by_scenario = {row["scenario_id"]: row["split"] for row in plan["scenarios"]}
    for manifest_path in sorted((root / "rendered_scenarios").glob("*/frame_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenario_id = manifest_path.parent.name
        split = split_by_scenario[scenario_id]
        for frame in manifest["frames"]:
            source = Path(frame["image_path"])
            original_hash = digest(source)
            target = source.with_suffix(".jpg")
            size_before = source.stat().st_size
            with Image.open(source) as image:
                image.convert("RGB").save(target, "JPEG", quality=args.quality, optimize=True)
            with Image.open(target) as check:
                check.verify()
            source.unlink()
            frame["image_path"] = str(target.resolve())
            new_hash = digest(target)
            before += size_before
            after += target.stat().st_size
            converted += 1
            stem = f"{scenario_id}__frame_{int(frame['frame_index']):04d}"
            dataset_image = dataset / "images" / split / f"{stem}.jpg"
            dataset_image.parent.mkdir(parents=True, exist_ok=True)
            os.link(target, dataset_image)
            if stem in smoke_stems:
                smoke_image = dataset / "smoke" / "images" / split / f"{stem}.jpg"
                smoke_image.parent.mkdir(parents=True, exist_ok=True)
                os.link(target, smoke_image)
            index[(scenario_id, int(frame["frame_index"]))] = (dataset_image.resolve(), original_hash, new_hash)
        manifest["image_storage"] = {
            "format": "JPEG",
            "quality": args.quality,
            "note": "Original PNGs were used for the completed training; exact PNGs are reproducible from the scenario plan.",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = root / "frame_manifest.csv"
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path, original_hash, new_hash = index[(row["scenario_id"], int(row["frame_index"]))]
        row["original_training_png_sha256"] = original_hash
        row["image_path"] = str(path)
        row["image_sha256"] = new_hash
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "converted_images": converted,
        "original_png_bytes": before,
        "compacted_jpeg_bytes": after,
        "bytes_saved": before - after,
        "quality": args.quality,
        "training_input_note": "Completed training used the original PNGs; compact JPEGs are retained for inspection only.",
    }
    (root / "image_compaction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
