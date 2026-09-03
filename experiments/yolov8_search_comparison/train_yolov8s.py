"""Train the required YOLOv8s model with fully recorded settings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from .run_comparison import REPO_ROOT, file_sha256, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    initial_weights = (REPO_ROOT / "yolov8s.pt").resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    settings = {
        "initial_weights": str(initial_weights),
        "initial_weights_sha256": file_sha256(initial_weights),
        "data": str(args.data.resolve()),
        "project": str(output_root),
        "name": args.name,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "seed": args.seed,
        "deterministic": True,
        "device": str(device),
        "workers": 0,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at": datetime.now().astimezone().isoformat(),
    }
    write_json(output_root / f"{args.name}_requested_settings.json", settings)

    model = YOLO(str(initial_weights))
    model.train(
        data=str(args.data.resolve()),
        project=str(output_root),
        name=args.name,
        exist_ok=False,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        seed=args.seed,
        deterministic=True,
        device=device,
        workers=0,
        cache=False,
        plots=True,
        verbose=True,
    )
    run_dir = output_root / args.name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    result = {
        "run_dir": str(run_dir),
        "best_weights": str(best),
        "best_weights_sha256": file_sha256(best) if best.is_file() else None,
        "last_weights": str(last),
        "last_weights_sha256": file_sha256(last) if last.is_file() else None,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    write_json(output_root / f"{args.name}_result.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
