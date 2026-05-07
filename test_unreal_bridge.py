#!/usr/bin/env python3
"""test_unreal_bridge.py — Phase B-2 end-to-end smoke test.

Verifies the full pipeline works:
    Python → MATLAB Engine → Unreal sim → camera PNG → YOLOv8s → detections

Usage:
    python test_unreal_bridge.py
    python test_unreal_bridge.py --no-yolo            # skip YOLO inference
    python test_unreal_bridge.py --conf 0.10          # lower YOLO threshold
    python test_unreal_bridge.py --runs 3             # multi-step test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dspy_pipeline.unreal_bridge import UnrealSimulinkBridge
from dspy_pipeline.yolo_detector import UAVDetector


ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-yolo", action="store_true",
                    help="Skip YOLO inference (just capture PNGs)")
    ap.add_argument("--conf",    type=float, default=0.30,
                    help="YOLO confidence threshold (default 0.30)")
    ap.add_argument("--weights", type=str,   default="yolov8s.pt")
    ap.add_argument("--runs",    type=int,   default=1,
                    help="Number of sim steps to run (default 1)")
    args = ap.parse_args()

    print("=" * 60)
    print("Phase B-2 End-to-End Smoke Test")
    print("=" * 60)

    # 1) Boot detector first (so MATLAB delay is at the end)
    detector = None
    if not args.no_yolo:
        print(f"\n[1/3] Loading YOLOv8s ({args.weights})…")
        detector = UAVDetector(weights=args.weights)
    else:
        print("\n[1/3] Skipping YOLO (--no-yolo)")

    # 2) Boot Unreal bridge (this opens MATLAB + Unreal — slow)
    print("\n[2/3] Starting Unreal bridge (MATLAB + Unreal)…")
    bridge = UnrealSimulinkBridge(model_dir=str(ROOT))
    try:
        bridge.start()
    except Exception as exc:
        print(f"\n[FAIL] Bridge start failed: {exc}")
        return

    # 3) Run N sims, capture frames, run YOLO
    print(f"\n[3/3] Running {args.runs} simulation step(s)…\n")
    results = []
    for step in range(1, args.runs + 1):
        fp_png = str(ASSETS / f"unreal_fp_{step:03d}.png")
        tp_png = str(ASSETS / f"unreal_tp_{step:03d}.png")

        print(f"--- Step {step}/{args.runs} ---")
        t0 = time.perf_counter()
        try:
            result = bridge.run_step(
                env_params  = {"fog_density_percent": 0,
                               "illumination_lux":   8000,
                               "camera_noise_level": 0},
                save_paths  = {"first_person": fp_png, "third_person": tp_png},
                detector    = detector,
                detect_conf = args.conf,
            )
        except Exception as exc:
            print(f"[FAIL] sim step {step}: {exc}")
            continue

        wall = time.perf_counter() - t0
        print(f"  {result.summary()}  total={wall:.1f}s")
        if result.fp_png:
            print(f"    1st-person PNG → {result.fp_png}")
        else:
            print(f"    1st-person PNG: NOT SAVED (cam1p_log empty?)")
        if result.tp_png:
            print(f"    3rd-person PNG → {result.tp_png}")
        else:
            print(f"    3rd-person PNG: NOT SAVED (cam3p_log empty?)")

        for d in result.detections:
            print(f"    YOLO: {d.cls_name:8s}  conf={d.conf:.2f}  "
                  f"bbox=({d.xyxy[0]:.0f}, {d.xyxy[1]:.0f}, "
                  f"{d.xyxy[2]:.0f}, {d.xyxy[3]:.0f})")

        results.append({
            "step":      step,
            "fp_png":    result.fp_png,
            "tp_png":    result.tp_png,
            "n_frames":  result.n_frames,
            "n_persons": result.n_persons,
            "n_vehicles":result.n_vehicles,
            "elapsed_s": result.elapsed_s,
        })
        print()

    bridge.stop()

    # Save summary
    out = ROOT / "data" / "unreal_phase_b2_test.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Done] Summary → {out}")

    # Verdict
    if results and results[0]["fp_png"]:
        print("\n[VERDICT] B-2 통과 ✅")
        print("        다음: Streamlit UI에 PNG를 표시하는 B-4로 진행 가능")
    else:
        print("\n[VERDICT] B-2 부분 실패 ⚠️")
        print("        cam1p_log이 비어있거나 PNG 저장 실패. MATLAB에서:")
        print("          whos cam1p_log")
        print("        실행 후 결과 알려주세요.")


if __name__ == "__main__":
    main()
