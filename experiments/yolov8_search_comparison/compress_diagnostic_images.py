"""Replace diagnostic PNG overlays with visually equivalent compact JPEGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="+")
    parser.add_argument("--quality", type=int, default=86)
    args = parser.parse_args()
    converted = 0
    bytes_before = 0
    bytes_after = 0
    for directory in args.directory:
        root = directory.resolve()
        for source in sorted(root.rglob("*.png")):
            target = source.with_suffix(".jpg")
            before = source.stat().st_size
            with Image.open(source) as image:
                image.convert("RGB").save(target, "JPEG", quality=args.quality, optimize=True)
            with Image.open(target) as check:
                check.verify()
            after = target.stat().st_size
            source.unlink()
            converted += 1
            bytes_before += before
            bytes_after += after
    result = {
        "converted_files": converted,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "bytes_saved": bytes_before - bytes_after,
        "jpeg_quality": args.quality,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
