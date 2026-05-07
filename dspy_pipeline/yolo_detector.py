"""yolo_detector.py — Real ML object detector for the border-surveillance UAV.

Replaces the geometric-oracle F_detector (which knew where intruders were
because they were labelled in OBSTACLES_*) with an actual ML detector that
classifies pixels in a rendered camera image. This is what makes the
end-to-end claim "the system detects unauthorised intruders" defensible.

Pipeline (per frame):
    rendered RGB image  ──►  YOLOv8s  ──►  list[Detection]
                                           ▲
                                           filter to {person, vehicle*}
                                           classes only

* "vehicle" = COCO ids {2: car, 3: motorcycle, 5: bus, 7: truck}.
  Bicycles (id 1) are excluded by default — toggle in target_classes.

Standalone test:
    python -m dspy_pipeline.yolo_detector --image assets/sim_step_001.png
    python -m dspy_pipeline.yolo_detector --folder assets/ --save-overlay
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from ultralytics import YOLO
    import torch
    _ULTRALYTICS_OK = True
except ImportError:
    _ULTRALYTICS_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """One YOLO detection in a single image."""
    cls_id:    int                     # COCO class id
    cls_name:  str                     # mapped target class ('person' | 'vehicle')
    conf:      float                   # confidence in [0, 1]
    xyxy:      tuple[float, float, float, float]   # x1, y1, x2, y2 (pixels)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1, y1, x2 - x1, y2 - y1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["xywh"] = list(self.xywh)
        d["xyxy"] = list(self.xyxy)
        return d


@dataclass
class FrameResult:
    """Per-image detection output, easy to serialise."""
    image_path:   str | None
    image_shape:  tuple[int, int, int]                  # (H, W, C)
    detections:   list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "image_path":   self.image_path,
            "image_shape":  list(self.image_shape),
            "n_detections": len(self.detections),
            "detections":   [d.to_dict() for d in self.detections],
            "inference_ms": round(self.inference_ms, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

# Default mapping: COCO class id → our mission target name.
# Anything NOT in this dict is filtered out (e.g. trees, rocks, animals).
DEFAULT_TARGET_CLASSES: dict[int, str] = {
    0: "person",
    2: "vehicle",      # car
    3: "vehicle",      # motorcycle
    5: "vehicle",      # bus
    7: "vehicle",      # truck
}


class UAVDetector:
    """Thin wrapper around Ultralytics YOLOv8 with target-class filtering."""

    def __init__(
        self,
        weights:        str  = "yolov8s.pt",
        target_classes: dict[int, str] | None = None,
        device:         str  = "auto",
        imgsz:          int  = 640,
    ) -> None:
        if not _ULTRALYTICS_OK:
            raise ImportError(
                "ultralytics not installed. Run:\n"
                "  pip install ultralytics torch --index-url https://download.pytorch.org/whl/cu121"
            )
        self.target_classes = target_classes or dict(DEFAULT_TARGET_CLASSES)
        self.imgsz          = imgsz
        self.device         = self._resolve_device(device)
        print(f"[UAVDetector] Loading {weights} on device={self.device}…")
        self.model = YOLO(weights)
        # Move to the chosen device (cuda:0 / cpu)
        try:
            self.model.to(self.device)
        except Exception:
            # Older ultralytics versions handle device per-call instead
            pass
        print(f"[UAVDetector] Ready. Targets: {sorted(set(self.target_classes.values()))}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if _ULTRALYTICS_OK and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    # ── Inference ─────────────────────────────────────────────────────────

    def detect(
        self,
        image:           np.ndarray | str | Path,
        conf_threshold:  float = 0.30,
    ) -> FrameResult:
        """Run YOLO on one image. Returns FrameResult filtered to target classes."""
        import time
        t0 = time.perf_counter()
        # Ultralytics handles ndarray, file path, PIL — pass through
        img_arg = str(image) if isinstance(image, (str, Path)) else image
        results = self.model.predict(
            img_arg,
            conf    = conf_threshold,
            imgsz   = self.imgsz,
            device  = self.device,
            verbose = False,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        if not results:
            return FrameResult(
                image_path  = str(image) if isinstance(image, (str, Path)) else None,
                image_shape = (0, 0, 3),
                inference_ms = elapsed,
            )
        r0 = results[0]
        H, W = r0.orig_shape[:2]
        boxes = r0.boxes
        detections: list[Detection] = []
        if boxes is not None and boxes.xyxy is not None:
            xyxy_all = boxes.xyxy.cpu().numpy()
            cls_all  = boxes.cls.cpu().numpy().astype(int)
            conf_all = boxes.conf.cpu().numpy()
            for box, cls_id, conf in zip(xyxy_all, cls_all, conf_all):
                if cls_id in self.target_classes:
                    detections.append(Detection(
                        cls_id   = int(cls_id),
                        cls_name = self.target_classes[int(cls_id)],
                        conf     = float(conf),
                        xyxy     = tuple(float(v) for v in box),
                    ))
        return FrameResult(
            image_path   = str(image) if isinstance(image, (str, Path)) else None,
            image_shape  = (int(H), int(W), 3),
            detections   = detections,
            inference_ms = elapsed,
        )

    def detect_many(
        self,
        images:          Iterable[np.ndarray | str | Path],
        conf_threshold:  float = 0.30,
    ) -> list[FrameResult]:
        return [self.detect(img, conf_threshold=conf_threshold) for img in images]

    # ── Visualisation helper ──────────────────────────────────────────────

    @staticmethod
    def overlay_detections(
        image:        np.ndarray,
        detections:   list[Detection],
        color_person: tuple[int, int, int] = (0, 200, 255),     # cyan
        color_vehicle:tuple[int, int, int] = (255, 120, 0),     # orange
    ) -> np.ndarray:
        """Draw bboxes + labels on a copy of the image. Requires opencv."""
        import cv2
        out = image.copy()
        for d in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in d.xyxy)
            color = color_person if d.cls_name == "person" else color_vehicle
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.cls_name} {d.conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - lh - 4)), (x1 + lw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main() -> None:
    p = argparse.ArgumentParser(description="Standalone YOLOv8s detector test for UAV imagery")
    p.add_argument("--image",   type=str, default=None,
                   help="Single image path")
    p.add_argument("--folder",  type=str, default=None,
                   help="Folder of *.png images (sim_step_*.png by default)")
    p.add_argument("--pattern", type=str, default="sim_step_*.png")
    p.add_argument("--weights", type=str, default="yolov8s.pt")
    p.add_argument("--conf",    type=float, default=0.30)
    p.add_argument("--save-overlay", action="store_true",
                   help="Save overlay image as <stem>_yolo.png next to input")
    p.add_argument("--save-json",    type=str, default=None,
                   help="Path to save aggregated JSON of all detections")
    args = p.parse_args()

    if not args.image and not args.folder:
        p.error("Provide --image PATH or --folder PATH")

    detector = UAVDetector(weights=args.weights)

    paths: list[Path]
    if args.image:
        paths = [Path(args.image)]
    else:
        paths = sorted(Path(args.folder).glob(args.pattern))
        if not paths:
            print(f"[WARN] No files match {args.folder}/{args.pattern}")
            return

    all_results = []
    for path in paths:
        result = detector.detect(path, conf_threshold=args.conf)
        n_pp = sum(1 for d in result.detections if d.cls_name == "person")
        n_vv = sum(1 for d in result.detections if d.cls_name == "vehicle")
        print(f"{path.name:30s}  person={n_pp:2d}  vehicle={n_vv:2d}  "
              f"({result.inference_ms:.0f} ms)")
        for d in result.detections:
            print(f"    {d.cls_name:8s}  conf={d.conf:.3f}  xyxy={d.xyxy}")
        all_results.append(result.to_dict())

        if args.save_overlay:
            try:
                import cv2
                img = cv2.imread(str(path))
                if img is None: continue
                overlay = UAVDetector.overlay_detections(img, result.detections)
                out_path = path.with_name(f"{path.stem}_yolo.png")
                cv2.imwrite(str(out_path), overlay)
                print(f"    → overlay saved: {out_path.name}")
            except ImportError:
                print("    (opencv-python not installed; --save-overlay skipped)")

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(all_results, indent=2),
                                        encoding="utf-8")
        print(f"\n[Done] JSON saved → {args.save_json}")


if __name__ == "__main__":
    _main()
