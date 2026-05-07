"""unreal_bridge.py — Python wrapper around the Unreal-photoreal Sim3D pipeline.

Runs `mountain_uav_unreal.slx` via MATLAB Engine, captures the two camera
streams (1st-person nadir + 3rd-person chase), saves the last frame of each
as a PNG (lightweight handoff to Python — full N-frame array transfer would
be slow), and optionally feeds the 1st-person frame to YOLOv8s for real
object detection.

This is the Phase B counterpart to dspy_pipeline/matlab_bridge.py
(which targets the synthetic geometric oracle model). Both can coexist;
the orchestrator picks one based on `sim_mode`.

Typical usage:
    from dspy_pipeline.unreal_bridge import UnrealSimulinkBridge
    from dspy_pipeline.yolo_detector  import UAVDetector

    bridge   = UnrealSimulinkBridge(model_dir=".")
    bridge.start()
    detector = UAVDetector("yolov8s.pt")

    result = bridge.run_step(
        env_params      = {"fog_density_percent": 0, "illumination_lux": 8000, "camera_noise_level": 0},
        save_paths      = {"first_person": "assets/unreal_fp_001.png",
                           "third_person": "assets/unreal_tp_001.png"},
        detector        = detector,
    )
    print(f"YOLO found {len(result.detections)} objects on the nadir frame")
    bridge.stop()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import matlab.engine                # type: ignore
    _MATLAB_OK = True
except ImportError:
    _MATLAB_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UnrealStepResult:
    """One simulation step's output."""
    fp_png:      str | None     = None       # 1st-person nadir PNG path
    tp_png:      str | None     = None       # 3rd-person chase PNG path
    n_frames:    int            = 0
    detections:  list           = field(default_factory=list)   # list[Detection]
    n_persons:   int            = 0
    n_vehicles:  int            = 0
    elapsed_s:   float          = 0.0
    sim_seconds: float          = 0.0
    raw:         dict           = field(default_factory=dict)

    def summary(self) -> str:
        return (f"frames={self.n_frames}  detections={len(self.detections)} "
                f"(persons={self.n_persons}, vehicles={self.n_vehicles})  "
                f"sim={self.sim_seconds:.1f}s  wall={self.elapsed_s:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Bridge
# ─────────────────────────────────────────────────────────────────────────────

class UnrealSimulinkBridge:
    """MATLAB Engine bridge for the Unreal photoreal simulation."""

    MODEL_NAME = "mountain_uav_unreal"
    BUILD_FN   = "build_mountain_uav_unreal"

    def __init__(self, model_dir: str | Path) -> None:
        if not _MATLAB_OK:
            raise ImportError(
                "matlab.engine not installed.\n"
                "Install via: pip install matlabengine"
            )
        self.model_dir = Path(model_dir).resolve()
        self._eng      = None
        print(f"[UnrealBridge] Initialised  model_dir={self.model_dir}")

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._eng is not None:
            return
        print("[UnrealBridge] Starting MATLAB Engine (-desktop)…")
        self._eng = matlab.engine.start_matlab("-desktop")
        self._eng.cd(str(self.model_dir), nargout=0)

        slx = self.model_dir / f"{self.MODEL_NAME}.slx"
        if not slx.exists():
            print(f"[UnrealBridge] {self.MODEL_NAME}.slx not found — building…")
            self._eng.eval(f"{self.BUILD_FN}(false)", nargout=0)
        else:
            print(f"[UnrealBridge] Reusing existing {slx.name}")

        self._eng.eval(
            f"if ~bdIsLoaded('{self.MODEL_NAME}'); load_system('{self.MODEL_NAME}'); end",
            nargout=0,
        )
        self._eng.eval(f"open_system('{self.MODEL_NAME}')", nargout=0)
        print("[UnrealBridge] Ready.")

    def stop(self) -> None:
        if self._eng is not None:
            try:
                self._eng.quit()
            except Exception:
                pass
            self._eng = None

    def __enter__(self) -> "UnrealSimulinkBridge":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── main step API ─────────────────────────────────────────────────────

    def run_step(
        self,
        env_params:  dict[str, float] | None = None,
        save_paths:  dict[str, str]    | None = None,
        detector              = None,                    # optional UAVDetector
        detect_conf: float    = 0.30,
    ) -> UnrealStepResult:
        """Run one Unreal sim, capture frames, optionally run YOLO.

        Args:
            env_params:  fog/illum/noise (currently logged only — Phase B-5
                         will wire these into Unreal scene parameters)
            save_paths:  optional dict with keys 'first_person', 'third_person'
                         — last frame of each camera is saved as PNG
            detector:    UAVDetector instance for ML detection on FP frame
            detect_conf: YOLO confidence threshold

        Returns:
            UnrealStepResult
        """
        if self._eng is None:
            raise RuntimeError("Bridge not started. Call .start() first.")
        env_params = env_params or {}
        save_paths = save_paths or {}

        t0 = time.perf_counter()

        # Ensure logs from any previous run are cleared so we don't read stale frames
        self._eng.eval(
            "for v = {'cam1p_log','cam3p_log'}; "
            "  if evalin('base', sprintf('exist(''%s'',''var'')==1', v{1})); "
            "    evalin('base', sprintf('clear %s', v{1})); end; end",
            nargout=0,
        )

        # Run the sim (Unreal window pops up)
        self._eng.eval(f"out_unreal = sim('{self.MODEL_NAME}');", nargout=0)
        sim_seconds = float(self._eng.eval(
            "get_param('" + self.MODEL_NAME + "', 'StopTime')", nargout=1))

        # IMPORTANT: when sim() is called programmatically, To Workspace block
        # data goes into the returned SimulationOutput, NOT the base workspace.
        # Copy each known log over to base so _save_last_frame can read it.
        # Tries .get(), then dot-access, then dataset find.
        self._eng.eval(
            "for v = {'cam1p_log','cam3p_log','uav_pose_log'}; "
            "  vname = v{1}; "
            "  copied = false; "
            "  try; val = out_unreal.get(vname); "
            "    if ~isempty(val); assignin('base', vname, val); copied = true; end; end; "
            "  if ~copied; "
            "    try; val = out_unreal.(vname); "
            "      if ~isempty(val); assignin('base', vname, val); copied = true; end; end; "
            "  end; "
            "  if ~copied; "
            "    try; ds = out_unreal.logsout; "
            "      el = ds.getElement(vname); val = el.Values; "
            "      assignin('base', vname, val); end; "
            "  end; "
            "end",
            nargout=0,
        )

        # Debug: list what's actually in the SimulationOutput
        try:
            self._eng.eval("disp('[UnrealBridge debug] out_unreal contents:'); disp(out_unreal)", nargout=0)
        except Exception:
            pass

        # --- Save the LAST frame of each camera as PNG (Python-friendly handoff)
        fp_png = save_paths.get("first_person")
        tp_png = save_paths.get("third_person")
        n_frames = 0

        if fp_png:
            n = self._save_last_frame("cam1p_log", fp_png)
            if n > 0: n_frames = max(n_frames, n)
        if tp_png:
            n = self._save_last_frame("cam3p_log", tp_png)
            if n > 0: n_frames = max(n_frames, n)

        # --- Optional: run YOLO on the 1st-person frame
        detections: list = []
        n_p = n_v = 0
        if detector is not None and fp_png and Path(fp_png).exists():
            r = detector.detect(fp_png, conf_threshold=detect_conf)
            detections = r.detections
            n_p = sum(1 for d in detections if d.cls_name == "person")
            n_v = sum(1 for d in detections if d.cls_name == "vehicle")

        elapsed = time.perf_counter() - t0

        return UnrealStepResult(
            fp_png      = fp_png if fp_png and Path(fp_png).exists() else None,
            tp_png      = tp_png if tp_png and Path(tp_png).exists() else None,
            n_frames    = n_frames,
            detections  = detections,
            n_persons   = n_p,
            n_vehicles  = n_v,
            elapsed_s   = elapsed,
            sim_seconds = sim_seconds,
            raw         = {"env_params": env_params},
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _save_last_frame(self, log_name: str, png_path: str) -> int:
        """Have MATLAB write the last frame of a camera log to PNG.

        Returns frame count (0 if log missing or empty).
        """
        png_path_mat = png_path.replace("\\", "/")
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        # Sanity-check log exists, then imwrite the last frame
        check_script = (
            f"frame_count = 0; "
            f"if exist('{log_name}','var') && isstruct({log_name}) "
            f"   && isfield({log_name},'signals') && ~isempty({log_name}.signals.values); "
            f"  v = {log_name}.signals.values; "
            f"  sz = size(v); "
            f"  if numel(sz) >= 3; "
            f"    if numel(sz) == 4; frame_count = sz(4); else; frame_count = 1; end; "
            f"    if frame_count > 0; "
            f"      if numel(sz) == 4; last_frame = v(:,:,:,end); else; last_frame = v; end; "
            f"      if isa(last_frame,'uint8'); imwrite(last_frame, '{png_path_mat}'); "
            f"      else; imwrite(uint8(last_frame*255), '{png_path_mat}'); end; "
            f"    end; "
            f"  end; "
            f"end"
        )
        self._eng.eval(check_script, nargout=0)
        try:
            return int(self._eng.eval("frame_count", nargout=1))
        except Exception:
            return 0
