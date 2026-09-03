"""Train YOLOv8s on the frozen diverse dataset and evaluate only its validation split."""

from __future__ import annotations

import argparse
import csv
import contextlib
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
import ultralytics
from ultralytics import YOLO

from .plan import CONFIG_ROOT, DATA_ROOT, HERE, PLAN_ID, REPO_ROOT, file_sha256, hardware_info


TRAINING_ROOT = HERE / "training" / PLAN_ID


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ResourceMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.process = psutil.Process()
        self.samples: list[dict[str, float]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.process.cpu_percent(None)
        while not self.stop_event.wait(1.0):
            try:
                memory = psutil.virtual_memory()
                self.samples.append(
                    {
                        "elapsed_seconds": float(len(self.samples) + 1),
                        "process_rss_bytes": float(self.process.memory_info().rss),
                        "process_cpu_percent": float(self.process.cpu_percent(None)),
                        "system_memory_used_bytes": float(memory.used),
                        "system_memory_percent": float(memory.percent),
                    }
                )
            except psutil.Error:
                continue

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=5)
        return {
            "sample_interval_seconds": 1,
            "sample_count": len(self.samples),
            "peak_process_rss_bytes": max((x["process_rss_bytes"] for x in self.samples), default=None),
            "peak_process_cpu_percent": max((x["process_cpu_percent"] for x in self.samples), default=None),
            "peak_system_memory_used_bytes": max((x["system_memory_used_bytes"] for x in self.samples), default=None),
            "peak_system_memory_percent": max((x["system_memory_percent"] for x in self.samples), default=None),
        }


def read_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row in rows:
        converted.append({key.strip(): float(value) for key, value in row.items()})
    return converted


def validation_metrics(result: Any) -> dict[str, Any]:
    box = result.box
    indices = box.ap_class_index.tolist() if hasattr(box.ap_class_index, "tolist") else list(box.ap_class_index)
    ap50_values = box.ap50.tolist() if hasattr(box.ap50, "tolist") else list(box.ap50)
    per_class = {str(result.names[int(index)]): float(value) for index, value in zip(indices, ap50_values)}
    return {
        "person_ap50": per_class.get("person"),
        "vehicle_ap50": per_class.get("vehicle"),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "mean_precision": float(box.mp),
        "mean_recall": float(box.mr),
        "per_class_ap50": per_class,
        "selection_scope": "Five pre-registered validation scenarios only; S0-S4 were not accessed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_ROOT / "data.yaml")
    parser.add_argument("--output-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gate = json.loads((DATA_ROOT / "training_gate.json").read_text(encoding="utf-8"))
    if not gate.get("training_allowed"):
        raise RuntimeError("Dataset audit gate does not permit training")
    prereg = json.loads((CONFIG_ROOT / "pre_registration.json").read_text(encoding="utf-8"))
    initial_weights = (REPO_ROOT / "yolov8s.pt").resolve()
    if file_sha256(initial_weights) != prereg["training_configuration"]["initial_weights_sha256"]:
        raise RuntimeError("Initial YOLOv8s weights changed after pre-registration")
    expected = prereg["training_configuration"]
    actual_core = {"imgsz": args.imgsz, "batch": args.batch, "optimizer": "SGD", "epochs": args.epochs, "patience": args.patience, "seed": args.seed}
    if any(actual_core[key] != expected[key] for key in actual_core):
        raise RuntimeError(f"Requested training settings differ from pre-registration: {actual_core}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.name or f"diverse_yolov8s_seed42__{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = output_root / run_name
    if run_dir.exists() or (output_root / f"{run_name}_training_summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite training run: {run_name}")
    settings = {
        "run_id": run_name,
        "plan_id": PLAN_ID,
        "pre_registration_sha256": file_sha256(CONFIG_ROOT / "pre_registration.json"),
        "dataset_summary_sha256": file_sha256(DATA_ROOT / "dataset_summary.json"),
        "initial_weights": str(initial_weights),
        "initial_weights_sha256": file_sha256(initial_weights),
        "data": str(args.data.resolve()),
        "project": str(output_root),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
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
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "workers": 0,
        "selection_policy": "Ultralytics best.pt selected from the five-scenario validation split only.",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_new(output_root / f"{run_name}_requested_settings.json", settings)
    if not torch.cuda.is_available():
        raise RuntimeError("The pre-registered GPU training environment is unavailable")
    torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor()
    monitor.start()
    wall_started = time.perf_counter()
    console_log = output_root / f"{run_name}_training_console.log"
    try:
        with console_log.open("x", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(Tee(sys.stdout, log_handle)), contextlib.redirect_stderr(Tee(sys.stderr, log_handle)):
                model = YOLO(str(initial_weights))
                model.train(
                    data=str(args.data.resolve()),
                    project=str(output_root),
                    name=run_name,
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
                    device=0,
                    workers=0,
                    cache=False,
                    plots=True,
                    verbose=True,
                )
                training_seconds = time.perf_counter() - wall_started
                best = run_dir / "weights" / "best.pt"
                last = run_dir / "weights" / "last.pt"
                if not best.is_file():
                    raise FileNotFoundError(best)
                validation_started = time.perf_counter()
                selected = YOLO(str(best))
                val_result = selected.val(
                    data=str(args.data.resolve()),
                    split="val",
                    imgsz=args.imgsz,
                    batch=args.batch,
                    device=0,
                    workers=0,
                    plots=True,
                    project=str(run_dir),
                    name="best_validation",
                    exist_ok=False,
                    verbose=True,
                )
                validation_seconds = time.perf_counter() - validation_started
    finally:
        resources = monitor.stop()
    resources["torch_peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    resources["torch_peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved())

    rows = read_results(run_dir / "results.csv")
    best_row = max(rows, key=lambda row: row["metrics/mAP50-95(B)"])
    metrics = validation_metrics(val_result)
    packages = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics": ultralytics.__version__,
        "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout.splitlines(),
    }
    summary = {
        "run_id": run_name,
        "plan_id": PLAN_ID,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "training_epochs_completed": len(rows),
        "best_epoch_one_based": int(best_row["epoch"]),
        "best_epoch_metrics_from_training_log": best_row,
        "validation_metrics_recomputed_from_best": metrics,
        "best_weights": str(best.resolve()),
        "best_weights_size_bytes": best.stat().st_size,
        "best_weights_sha256": file_sha256(best),
        "last_weights": str(last.resolve()) if last.is_file() else None,
        "last_weights_size_bytes": last.stat().st_size if last.is_file() else None,
        "last_weights_sha256": file_sha256(last) if last.is_file() else None,
        "training_curve": str((run_dir / "results.png").resolve()),
        "training_console_log": str(console_log.resolve()),
        "settings": settings,
        "hardware": hardware_info(),
        "resource_usage": resources,
        "packages": packages,
        "interpretation_limit": "Internal validation on five new MATLAB scenarios; not evidence of real-UAV generalization.",
        "S0_S4_accessed_for_training_or_selection": False,
    }
    write_json_new(output_root / f"{run_name}_training_summary.json", summary)
    write_json_new(run_dir / "validation_metrics.json", metrics)
    print(json.dumps({"run_id": run_name, "summary": str((output_root / f'{run_name}_training_summary.json').resolve()), "best_weights": str(best.resolve()), "best_weights_sha256": summary["best_weights_sha256"], "validation": metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
