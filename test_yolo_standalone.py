#!/usr/bin/env python3
"""test_yolo_standalone.py — Phase C smoke test.

Runs YOLOv8s on the assets/sim_step_*.png images that the previous
matlab visualizer captured. Confirms:
  1. Ultralytics + CUDA work on this box
  2. UAVDetector loads and filters to person/vehicle classes
  3. Detection metrics (IoU, mAP50) compute end-to-end on synthetic GT

The synthetic ground truth here is INTENTIONALLY weak — these are the
matlab-rendered images that look only loosely like real photos, so YOLO
won't necessarily detect the painted silhouettes. Useful for proving
the *pipeline* works; real evaluation needs Unreal Engine renders (Phase B).

Usage:
    python test_yolo_standalone.py
    python test_yolo_standalone.py --conf 0.10 --save-overlay
    python test_yolo_standalone.py --image my_image.jpg --conf 0.25 --save-overlay
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

ASSETS_DIR = ROOT / "assets"
TEST_OUT   = ROOT / "data" / "yolo_phase_c_test.json"

from dspy_pipeline.yolo_detector     import UAVDetector
from dspy_pipeline.detection_metrics import (
    GroundTruth, evaluate_mission,
)


def _image_iter(args) -> list[Path]:
    if args.image:
        return [Path(args.image)]
    if not ASSETS_DIR.exists():
        sys.exit(f"[Error] assets/ folder not found at {ASSETS_DIR}")
    paths = sorted(ASSETS_DIR.glob(args.pattern))
    # Filter out our own overlay outputs so re-runs don't double-process
    paths = [p for p in paths if not p.stem.endswith("_yolo")]
    if not paths:
        sys.exit(f"[Error] No images match {ASSETS_DIR}/{args.pattern}.\n"
                 f"        Run app_demo.py first to generate sim_step_*.png, "
                 f"or pass --pattern '*.jpg' / '*.png' / specific path.")
    return paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image",   type=str, default=None,
                   help="Single image path (overrides --pattern)")
    p.add_argument("--pattern", type=str, default="sim_step_*.png",
                   help="Glob inside assets/ (default: sim_step_*.png)")
    p.add_argument("--weights", type=str, default="yolov8s.pt")
    p.add_argument("--conf",    type=float, default=0.30,
                   help="YOLO conf threshold (default 0.30)")
    p.add_argument("--save-overlay", action="store_true",
                   help="Save overlay PNGs with bboxes drawn")
    args = p.parse_args()

    paths = _image_iter(args)
    print(f"[Phase-C] Loading YOLOv8s ({args.weights})…")
    detector = UAVDetector(weights=args.weights)

    # Treat each image as a "frame" with empty GT for now (synthetic eval)
    # In Phase B we'll feed real GT from Unreal actor positions.
    detections_per_frame: list[list] = []
    gt_per_frame:         list[list[GroundTruth]] = []

    print(f"\n[Phase-C] Running YOLO on {len(paths)} image(s)  conf={args.conf}\n")
    print(f"  {'image':30s}  {'persons':>8}  {'vehicles':>8}  {'top conf':>9}  {'ms':>5}")
    print("  " + "─" * 70)
    rows = []
    for path in paths:
        result = detector.detect(path, conf_threshold=args.conf)
        n_p = sum(1 for d in result.detections if d.cls_name == "person")
        n_v = sum(1 for d in result.detections if d.cls_name == "vehicle")
        top = max((d.conf for d in result.detections), default=0.0)
        print(f"  {path.name:30s}  {n_p:>8d}  {n_v:>8d}  "
              f"{top:>9.3f}  {result.inference_ms:>5.0f}")
        rows.append(result.to_dict())
        detections_per_frame.append(result.detections)
        gt_per_frame.append([])     # synthetic GT unavailable in Phase C

        if args.save_overlay:
            try:
                import cv2
                img = cv2.imread(str(path))
                if img is not None:
                    overlay = UAVDetector.overlay_detections(img, result.detections)
                    out_p = path.with_name(f"{path.stem}_yolo.png")
                    cv2.imwrite(str(out_p), overlay)
            except ImportError:
                print("  (opencv-python not installed; --save-overlay skipped)")

    # --- Mission-level evaluation (vacuous without GT, but proves the wiring) ---
    eval_result = evaluate_mission(
        detections_per_frame, gt_per_frame,
        map_threshold=0.50, continuity_threshold=3,
    )

    print()
    print(f"  ── Mission evaluation (synthetic GT empty — see note) ──")
    print(f"  REQ-1 mAP50: {eval_result['req1']['value']:.3f}  "
          f"(th={eval_result['req1']['threshold']}) "
          f"{'PASS' if eval_result['req1']['passed'] else 'FAIL'}")
    print(f"  REQ-3 worst_run: {eval_result['req3']['value']}  "
          f"(th={eval_result['req3']['threshold']}) "
          f"{'PASS' if eval_result['req3']['passed'] else 'FAIL'}")
    print(f"  Stats: TP={eval_result['stats']['tp']}  "
          f"FP={eval_result['stats']['fp']}  "
          f"FN={eval_result['stats']['fn']}  "
          f"GT_total={eval_result['stats']['total_gt']}")
    print()
    print("  NOTE  GT is empty in this Phase C test, so REQ-1 trivially passes.")
    print("        Phase B will feed real GT from Unreal actor positions and")
    print("        the same evaluate_mission() will produce real mAP50.")

    TEST_OUT.parent.mkdir(exist_ok=True)
    TEST_OUT.write_text(json.dumps({
        "n_images":   len(paths),
        "weights":    args.weights,
        "conf":       args.conf,
        "frames":     rows,
        "evaluation": {k: v for k, v in eval_result.items() if k != "per_frame"},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Done] Per-image JSON saved → {TEST_OUT}")


if __name__ == "__main__":
    main()
