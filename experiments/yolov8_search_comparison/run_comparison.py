"""Run the deterministic YOLOv8 symmetric/asymmetric search comparison.

Typical use from the repository root::

    .venv\\Scripts\\python -m experiments.yolov8_search_comparison.run_comparison

Use ``--detector heuristic`` only for a clearly labelled legacy control.  The
KCI experiment config defaults to YOLOv8 and never passes GT boxes to YOLO.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .model_mapping import resolve_target_class_mapping
from .search_core import (
    Environment,
    bbox_iou_xywh,
    choose_next_environment,
    classify_verdict,
    compute_map50,
    environment_gap,
    map_gap,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config" / "experiment_config.json"
DEFAULT_OUTPUT_ROOT = HERE


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_frame_alignment(frames: list[dict[str, Any]], frame_stride: int) -> None:
    """Reject manifest-order/frame-name mismatches before detector inference."""
    expected_indices = list(range(1, 1 + frame_stride * len(frames), frame_stride))
    actual_indices = [int(frame["frame_index"]) for frame in frames]
    if actual_indices != expected_indices:
        raise ValueError(f"Frame index sequence mismatch: expected {expected_indices[:5]}, got {actual_indices[:5]}")
    for frame in frames:
        expected_stem = f"frame_{int(frame['frame_index']):04d}"
        if Path(frame["image_path"]).stem != expected_stem:
            raise ValueError(
                f"Frame/GT path mismatch: index={frame['frame_index']} path={frame['image_path']}"
            )


def resolve_weights(config: dict[str, Any]) -> Path:
    path = Path(config["weights_path"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO weights were not found at {path}. Supply the intended "
            "pretrained yolov8s.pt or the explicitly named custom weights; "
            "the experiment will not auto-download an arbitrary model."
        )
    return path


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


class MatlabFrameExporter:
    """One MATLAB Engine session shared by every environment and method."""

    def __init__(self, repo_root: Path, experiment_dir: Path) -> None:
        self.repo_root = repo_root
        self.experiment_dir = experiment_dir
        self.engine = None
        self.startup_seconds = 0.0

    def start(self) -> None:
        if self.engine is not None:
            return
        import matlab.engine

        started = time.perf_counter()
        self.engine = matlab.engine.start_matlab("-nodesktop -nosplash")
        self.engine.cd(str(self.repo_root), nargout=0)
        self.engine.addpath(str(self.repo_root), nargout=0)
        self.engine.addpath(str(self.experiment_dir), nargout=0)
        self.startup_seconds = time.perf_counter() - started

    def export(
        self,
        env: Environment,
        output_dir: Path,
        seed: int,
        frame_stride: int,
        detector: str,
        ground_truth_mode: str,
        scenario_variant: int = 0,
    ) -> dict[str, Any]:
        self.start()
        assert self.engine is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = self.engine.export_experiment_frames(
            float(env.fog_percent),
            float(env.illumination_lux),
            float(env.camera_noise),
            str(output_dir),
            float(seed),
            float(frame_stride),
            detector,
            ground_truth_mode,
            float(scenario_variant),
            nargout=1,
        )
        return json.loads(str(raw))

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            finally:
                self.engine = None


class YoloFrameDetector:
    """Whole-frame YOLO inference; this object never receives GT boxes."""

    def __init__(self, weights: Path, settings: dict[str, Any]) -> None:
        import torch
        import ultralytics
        from ultralytics import YOLO

        self.torch = torch
        self.ultralytics = ultralytics
        self.settings = settings
        requested = str(settings.get("device", "auto"))
        self.device = (
            "cuda:0" if requested == "auto" and torch.cuda.is_available()
            else "cpu" if requested == "auto"
            else requested
        )
        self.model = YOLO(str(weights))
        try:
            self.model.to(self.device)
        except Exception:
            pass
        # Resolve from the checkpoint itself. This supports both standard COCO
        # IDs and custom {0: person, 1: vehicle} models without confusing them.
        self.mapping = resolve_target_class_mapping(self.model.names)

    def metadata(self, weights: Path, weights_hash: str) -> dict[str, Any]:
        device_name = "CPU"
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            device_name = self.torch.cuda.get_device_name(0)
        checkpoint = self.model.ckpt or {}
        train_args = checkpoint.get("train_args") or {}
        training_data = train_args.get("data")
        if training_data and "yolov8s_sim_20260902" in str(training_data):
            provenance = "custom YOLOv8s fine-tuned on scenario-separated MATLAB simulation data"
        else:
            provenance = "general Ultralytics COCO pretrained YOLOv8s; not custom-trained"
        return {
            "detector": "yolov8",
            "weights_filename": weights.name,
            "weights_path": str(weights),
            "weights_sha256": weights_hash,
            "weights_provenance": provenance,
            "checkpoint_model_definition": train_args.get("model"),
            "checkpoint_training_data": training_data,
            "checkpoint_training_epochs": train_args.get("epochs"),
            "checkpoint_training_batch": train_args.get("batch"),
            "checkpoint_training_seed": train_args.get("seed"),
            "checkpoint_version": checkpoint.get("version"),
            "checkpoint_date": checkpoint.get("date"),
            "ultralytics_version": self.ultralytics.__version__,
            "python_version": platform.python_version(),
            "torch_version": self.torch.__version__,
            "device_argument": self.device,
            "device_name": device_name,
            "input_size": int(self.settings["input_size"]),
            "confidence_threshold": float(self.settings["confidence_threshold"]),
            "nms_iou_threshold": float(self.settings["nms_iou_threshold"]),
            "ap_iou_threshold": float(self.settings["ap_iou_threshold"]),
            "class_mapping": self.mapping,
            "class_mapping_note": (
                "Resolved from model.names. Recognised person aliases map to person; "
                "car/motorcycle/bus/truck/van/vehicle aliases merge to vehicle."
            ),
            "model_class_names": {int(k): str(v) for k, v in self.model.names.items()},
        }

    def detect_paths(self, image_paths: list[str]) -> tuple[list[list[dict[str, Any]]], float, list[dict[str, float]]]:
        """Infer only from image paths; scoring and GT access happen later."""
        started = time.perf_counter()
        configured_batch = max(1, int(self.settings.get("batch_size", 16)))
        batch_attempts: list[int] = []
        batch = configured_batch
        while batch not in batch_attempts:
            batch_attempts.append(batch)
            batch = max(1, batch // 2)
        last_error: BaseException | None = None
        for batch in batch_attempts:
            all_detections: list[list[dict[str, Any]]] = []
            speed_rows: list[dict[str, float]] = []
            try:
                results = self.model.predict(
                    source=image_paths,
                    conf=float(self.settings["confidence_threshold"]),
                    iou=float(self.settings["nms_iou_threshold"]),
                    imgsz=int(self.settings["input_size"]),
                    device=self.device,
                    batch=batch,
                    stream=True,
                    verbose=False,
                )
                for result in results:
                    detections: list[dict[str, Any]] = []
                    boxes = result.boxes
                    if boxes is not None and boxes.xyxy is not None:
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        classes = boxes.cls.detach().cpu().numpy().astype(int)
                        confidences = boxes.conf.detach().cpu().numpy()
                        for box, source_class, confidence in zip(xyxy, classes, confidences):
                            if int(source_class) not in self.mapping:
                                continue
                            x1, y1, x2, y2 = (float(value) for value in box)
                            detections.append(
                                {
                                    "source_class_id": int(source_class),
                                    "source_class_name": str(self.model.names[int(source_class)]),
                                    "class_name": self.mapping[int(source_class)],
                                    "confidence": float(confidence),
                                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                                    "bbox_xyxy": [x1, y1, x2, y2],
                                }
                            )
                    all_detections.append(detections)
                    speed_rows.append({key: float(value) for key, value in result.speed.items()})
                if batch != configured_batch:
                    logging.warning(
                        "YOLO CUDA OOM recovery succeeded with batch=%d (configured=%d)",
                        batch,
                        configured_batch,
                    )
                self.last_effective_batch_size = batch
                return all_detections, time.perf_counter() - started, speed_rows
            except (self.torch.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                last_error = exc
                logging.warning("YOLO CUDA OOM with batch=%d; retrying smaller batch", batch)
                all_detections.clear()
                speed_rows.clear()
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
        assert last_error is not None
        raise last_error


class EvaluationRepository:
    """Content-addressed render/inference cache shared by both search arms."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        cache_root: Path,
        exporter: MatlabFrameExporter,
        yolo_detector: YoloFrameDetector | None,
        model_metadata: dict[str, Any],
    ) -> None:
        self.config = config
        self.cache_root = cache_root
        self.exporter = exporter
        self.yolo_detector = yolo_detector
        self.model_metadata = model_metadata

    def _key(self, env: Environment) -> str:
        identity = {
            "scenario_id": self.config["scenario_id"],
            "scenario_variant": int(self.config.get("scenario_variant", 0)),
            "environment": env.as_dict(),
            "seed": self.config["random_seed"],
            "frame_stride": self.config["frame_stride"],
            "ground_truth_mode": self.config["ground_truth_mode"],
            "detector": self.config["detector"],
            "model_sha256": self.model_metadata.get("weights_sha256", "heuristic"),
            "yolo_settings": self.config.get("yolov8", {}),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:20]

    def _load_complete_manifest(self, cache_dir: Path, env: Environment) -> dict[str, Any] | None:
        """Reuse a completed render after an interrupted inference run."""
        manifest_path = cache_dir / "frame_manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            expected_environment = env.as_dict()
            if manifest.get("environment") != expected_environment:
                return None
            if int(manifest.get("random_seed", -1)) != int(self.config["random_seed"]):
                return None
            if int(manifest.get("frame_stride", -1)) != int(self.config["frame_stride"]):
                return None
            if manifest.get("detector_mode") != self.config["detector"]:
                return None
            if manifest.get("ground_truth_mode") != self.config["ground_truth_mode"]:
                return None
            if int(manifest.get("scenario_variant", 0)) != int(self.config.get("scenario_variant", 0)):
                return None
            frames = manifest.get("frames", [])
            if len(frames) != int(manifest.get("evaluated_frame_count", -1)):
                return None
            if not frames or not all(Path(frame["image_path"]).is_file() for frame in frames):
                return None
            logging.info("Reusing complete rendered manifest after interrupted inference: %s", manifest_path)
            return manifest
        except (OSError, ValueError, KeyError, TypeError):
            logging.exception("Ignoring invalid partial render cache: %s", manifest_path)
            return None

    def evaluate(self, env: Environment) -> tuple[dict[str, Any], bool]:
        key = self._key(env)
        cache_dir = self.cache_root / key
        evaluation_path = cache_dir / "evaluation.json"
        if evaluation_path.is_file():
            with evaluation_path.open("r", encoding="utf-8") as handle:
                return json.load(handle), True

        # A previously interrupted MATLAB run may leave an incomplete cache
        # directory. Re-rendering into that exact content-addressed directory
        # is safe; only a completed evaluation.json is treated as a cache hit.
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._load_complete_manifest(cache_dir, env)
        if manifest is None:
            manifest = self.exporter.export(
                env,
                cache_dir,
                int(self.config["random_seed"]),
                int(self.config["frame_stride"]),
                str(self.config["detector"]),
                str(self.config["ground_truth_mode"]),
                int(self.config.get("scenario_variant", 0)),
            )
        image_paths = [str(frame["image_path"]) for frame in manifest["frames"]]
        validate_frame_alignment(manifest["frames"], int(manifest["frame_stride"]))

        inference_seconds = 0.0
        speed_rows: list[dict[str, float]] = []
        if self.config["detector"] == "yolov8":
            if self.yolo_detector is None:
                raise RuntimeError("YOLO detector was not initialized")
            # Deliberately pass image paths only. Ground truth stays in manifest.
            detections, inference_seconds, speed_rows = self.yolo_detector.detect_paths(image_paths)
        else:
            detections = [frame.get("heuristic_detections", []) for frame in manifest["frames"]]

        scoring_started = time.perf_counter()
        scored_frames: list[dict[str, Any]] = []
        for frame, frame_detections in zip(manifest["frames"], detections):
            scored_frames.append(
                {
                    "frame_index": frame["frame_index"],
                    "time_seconds": frame["time_seconds"],
                    "image_path": frame["image_path"],
                    "ground_truth": frame.get("ground_truth", []),
                    "detections": frame_detections,
                }
            )
        metrics = compute_map50(
            scored_frames,
            iou_threshold=float(self.config["yolov8"]["ap_iou_threshold"]),
        )
        scoring_seconds = time.perf_counter() - scoring_started
        evaluation = {
            "cache_key": key,
            "environment": env.as_dict(),
            "scenario_id": self.config["scenario_id"],
            "random_seed": self.config["random_seed"],
            "model": self.model_metadata,
            "matlab": {
                "version": manifest["matlab_version"],
                "release": manifest["matlab_release"],
                "image_width": manifest["image_width"],
                "image_height": manifest["image_height"],
                "total_simulation_frames": manifest["total_simulation_frames"],
                "evaluated_frame_count": manifest["evaluated_frame_count"],
                "frame_stride": manifest["frame_stride"],
                "ground_truth_mode": manifest["ground_truth_mode"],
                "scenario_variant": int(manifest.get("scenario_variant", 0)),
                "renderer": manifest["renderer"],
            },
            "metrics": metrics,
            "timings": {
                "geometry_simulation_seconds": float(manifest["geometry_simulation_seconds"]),
                "frame_rendering_seconds": float(manifest["frame_rendering_seconds"]),
                "yolov8_inference_seconds": inference_seconds,
                "metric_scoring_seconds": scoring_seconds,
            },
            "ultralytics_speed_ms_per_frame": speed_rows,
            "effective_inference_batch_size": (
                getattr(self.yolo_detector, "last_effective_batch_size", None)
                if self.yolo_detector is not None else None
            ),
            "rendered_frames_retained": bool(self.config.get("retain_rendered_frames", True)),
            "frames": scored_frames,
        }
        write_json(evaluation_path, evaluation)
        if not self.config.get("retain_rendered_frames", True):
            frames_dir = (cache_dir / "frames").resolve()
            resolved_cache = cache_dir.resolve()
            if frames_dir.parent != resolved_cache:
                raise RuntimeError(f"Refusing to remove render directory outside cache: {frames_dir}")
            if frames_dir.is_dir():
                shutil.rmtree(frames_dir)
                logging.info("Removed transient rendered frames after evaluation: %s", frames_dir)
        return evaluation, False


def anchor_record(env: Environment | None, evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if env is None or evaluation is None:
        return None
    map50 = float(evaluation["metrics"]["map50"])
    return {
        **env.as_dict(),
        "map50": map50,
        "verdict": classify_verdict(map50),
        "cache_key": evaluation["cache_key"],
    }


def maximum_same_class_iou(frames: list[dict[str, Any]]) -> float | None:
    values = [
        bbox_iou_xywh(det["bbox_xywh"], gt["bbox_xywh"])
        for frame in frames
        for det in frame.get("detections", [])
        for gt in frame.get("ground_truth", [])
        if det["class_name"] == gt["class_name"]
    ]
    return max(values) if values else None


def run_search_method(
    method: str,
    config: dict[str, Any],
    evaluations: EvaluationRepository,
    run_dir: Path,
    session_id: str,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    started_wall = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    current = Environment.from_dict(config["initial_environment"])
    nonfail_env: Environment | None = None
    nonfail_eval: dict[str, Any] | None = None
    fail_env: Environment | None = None
    fail_eval: dict[str, Any] | None = None
    first_fail_evaluation: int | None = None
    records: list[dict[str, Any]] = []
    stop_reason = "maximum_evaluations_reached"

    for iteration in range(1, int(config["max_evaluations"]) + 1):
        iteration_started = time.perf_counter()
        evaluation, cache_hit = evaluations.evaluate(current)
        map50 = float(evaluation["metrics"]["map50"])
        thresholds = config["thresholds"]
        verdict = classify_verdict(
            map50,
            float(thresholds["pass_map50"]),
            float(thresholds["fail_map50"]),
        )
        first_fail_discovered = verdict == "FAIL" and first_fail_evaluation is None
        if first_fail_discovered:
            first_fail_evaluation = iteration

        if verdict == "FAIL":
            fail_env, fail_eval = current, evaluation
        else:  # PASS and MARGINAL are the non-failure side by design.
            nonfail_env, nonfail_eval = current, evaluation

        scenario_started = time.perf_counter()
        next_env, calculation = choose_next_environment(
            method=method,
            current=current,
            current_verdict=verdict,
            nonfail_anchor=nonfail_env,
            fail_anchor=fail_env,
            bounds=config["environment_bounds"],
            degradation=config["degradation"],
        )
        scenario_seconds = time.perf_counter() - scenario_started
        timings = evaluation["timings"]
        accounted_seconds = (
            float(timings["geometry_simulation_seconds"])
            + float(timings["frame_rendering_seconds"])
            + float(timings["yolov8_inference_seconds"])
            + float(timings["metric_scoring_seconds"])
            + scenario_seconds
        )
        record = {
            "experiment_id": config["experiment_id"],
            "session_id": session_id,
            "search_method": method,
            "detector": config["detector"],
            "scenario_id": config["scenario_id"],
            "random_seed": config["random_seed"],
            "repetition": 1,
            "iteration": iteration,
            "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **current.as_dict(),
            "detected_intruder_count": evaluation["metrics"]["detected_intruder_count"],
            "visible_intruder_count": evaluation["metrics"]["visible_intruder_count"],
            "target_detection_count": sum(
                item["n_detections"] for item in evaluation["metrics"]["per_class"].values()
            ),
            "person_detection_count": evaluation["metrics"]["per_class"]["person"]["n_detections"],
            "vehicle_detection_count": evaluation["metrics"]["per_class"]["vehicle"]["n_detections"],
            "person_true_positive_count": evaluation["metrics"]["per_class"]["person"]["tp"],
            "person_false_positive_count": evaluation["metrics"]["per_class"]["person"]["fp"],
            "person_false_negative_count": evaluation["metrics"]["per_class"]["person"]["fn"],
            "vehicle_true_positive_count": evaluation["metrics"]["per_class"]["vehicle"]["tp"],
            "vehicle_false_positive_count": evaluation["metrics"]["per_class"]["vehicle"]["fp"],
            "vehicle_false_negative_count": evaluation["metrics"]["per_class"]["vehicle"]["fn"],
            "total_true_positive_count": sum(
                item["tp"] for item in evaluation["metrics"]["per_class"].values()
            ),
            "total_false_positive_count": sum(
                item["fp"] for item in evaluation["metrics"]["per_class"].values()
            ),
            "total_false_negative_count": sum(
                item["fn"] for item in evaluation["metrics"]["per_class"].values()
            ),
            "maximum_same_class_iou": maximum_same_class_iou(evaluation["frames"]),
            "person_ap50": evaluation["metrics"]["person_ap50"],
            "vehicle_ap50": evaluation["metrics"]["vehicle_ap50"],
            "map50": map50,
            "verdict": verdict,
            "nonfail_anchor": anchor_record(nonfail_env, nonfail_eval),
            "fail_anchor": anchor_record(fail_env, fail_eval),
            "next_environment": None if next_env is None else next_env.as_dict(),
            "next_condition_calculation": calculation,
            "first_fail_discovered": first_fail_discovered,
            "cache_key": evaluation["cache_key"],
            "evaluation_cache_hit": cache_hit,
            "model_weights_filename": model_metadata.get("weights_filename"),
            "model_weights_sha256": model_metadata.get("weights_sha256"),
            "weights_provenance": model_metadata.get("weights_provenance"),
            "ultralytics_version": model_metadata.get("ultralytics_version"),
            "python_version": model_metadata.get("python_version"),
            "matlab_version": evaluation["matlab"]["version"],
            "matlab_release": evaluation["matlab"]["release"],
            "device_name": model_metadata.get("device_name"),
            "actual_image_width": evaluation["matlab"]["image_width"],
            "actual_image_height": evaluation["matlab"]["image_height"],
            "yolo_input_size": model_metadata.get("input_size"),
            "confidence_threshold": model_metadata.get("confidence_threshold"),
            "nms_iou_threshold": model_metadata.get("nms_iou_threshold"),
            "ap_iou_threshold": model_metadata.get("ap_iou_threshold"),
            "class_mapping": model_metadata.get("class_mapping"),
            "simulation_seconds": float(timings["geometry_simulation_seconds"])
            + float(timings["frame_rendering_seconds"]),
            "geometry_simulation_seconds": float(timings["geometry_simulation_seconds"]),
            "frame_rendering_seconds": float(timings["frame_rendering_seconds"]),
            "yolov8_inference_seconds": float(timings["yolov8_inference_seconds"]),
            "metric_scoring_seconds": float(timings["metric_scoring_seconds"]),
            "next_scenario_seconds": scenario_seconds,
            "accounted_iteration_seconds": accounted_seconds,
            "iteration_wall_seconds": time.perf_counter() - iteration_started,
        }
        records.append(record)
        logging.info(
            "%s iter=%d env=%s mAP50=%.6f verdict=%s cache=%s",
            method,
            iteration,
            current.as_dict(),
            map50,
            verdict,
            cache_hit,
        )

        if next_env is None:
            stop_reason = calculation
            break
        if next_env == current:
            stop_reason = "stopped_no_environment_progress_at_bounds_or_deterministic_boundary"
            break
        current = next_env

    boundary_success = nonfail_env is not None and fail_env is not None
    gap_map = None
    gap_environment = None
    if boundary_success and nonfail_eval is not None and fail_eval is not None:
        gap_map = map_gap(
            float(nonfail_eval["metrics"]["map50"]),
            float(fail_eval["metrics"]["map50"]),
        )
        gap_environment = environment_gap(
            nonfail_env, fail_env, config["environment_bounds"]
        )

    summary = {
        "experiment_id": config["experiment_id"],
        "session_id": session_id,
        "search_method": method,
        "detector": config["detector"],
        "scenario_id": config["scenario_id"],
        "random_seed": config["random_seed"],
        "repetition": 1,
        "evaluation_count": len(records),
        "logical_evaluation_count": len(records),
        "actual_new_evaluation_count": sum(not r["evaluation_cache_hit"] for r in records),
        "cache_hit_count": sum(r["evaluation_cache_hit"] for r in records),
        "first_fail_evaluation": first_fail_evaluation,
        "boundary_success": boundary_success,
        "gap_map50": gap_map,
        "gap_environment_normalized": gap_environment,
        "final_nonfail_anchor": anchor_record(nonfail_env, nonfail_eval),
        "final_fail_anchor": anchor_record(fail_env, fail_eval),
        "stop_reason": stop_reason,
        "accounted_total_seconds": sum(r["accounted_iteration_seconds"] for r in records),
        "simulation_total_seconds": sum(r["simulation_seconds"] for r in records),
        "yolov8_inference_total_seconds": sum(r["yolov8_inference_seconds"] for r in records),
        "next_scenario_total_seconds": sum(r["next_scenario_seconds"] for r in records),
        "driver_wall_seconds": time.perf_counter() - started_wall,
        "statistical_note": (
            "Single deterministic scenario. No artificial 30-run replication and no "
            "inferential statistics are reported."
        ),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "iterations.json", records)
    write_json(run_dir / "run_summary.json", summary)
    write_json(
        run_dir / "run_metadata.json",
        {
            "experiment_id": config["experiment_id"],
            "session_id": session_id,
            "search_method": method,
            "started_at_utc": started_at_utc,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_metadata,
            "scenario_id": config["scenario_id"],
            "random_seed": config["random_seed"],
            "thresholds": config["thresholds"],
            "environment_bounds": config["environment_bounds"],
            "frame_stride": config["frame_stride"],
            "deterministic_single_scenario": True,
            "statistical_note": summary["statistical_note"],
        },
    )
    write_iteration_csv(run_dir / "iterations.csv", records)
    return {"records": records, "summary": summary, "run_dir": str(run_dir)}


def write_iteration_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    rows: list[dict[str, Any]] = []
    for record in records:
        row = deepcopy(record)
        for key in ("nonfail_anchor", "fail_anchor", "next_environment", "class_mapping"):
            row[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def heuristic_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector": "heuristic",
        "implementation": "image_detector.m",
        "weights_filename": None,
        "weights_sha256": None,
        "python_version": platform.python_version(),
        "ap_iou_threshold": float(config["yolov8"]["ap_iou_threshold"]),
        "warning": "Legacy GT-conditioned detector; never label these results as YOLOv8.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--detector", choices=("yolov8", "heuristic"))
    parser.add_argument("--search-method", choices=("symmetric", "asymmetric", "both"), default="both")
    parser.add_argument("--frame-stride", type=int, help="Smoke tests only; paper config uses 1.")
    parser.add_argument("--max-evaluations", type=int, help="Smoke tests only; paper config uses 10.")
    parser.add_argument("--weights", type=Path, help="Explicit local weights; recorded in the config snapshot.")
    parser.add_argument("--force", action="store_true", help="Ignore prior cache by creating a fresh cache namespace.")
    parser.add_argument(
        "--cache-namespace",
        help="Explicit cache directory name for resuming an interrupted content-addressed run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    if args.detector:
        config["detector"] = args.detector
    if args.frame_stride is not None:
        config["frame_stride"] = max(1, args.frame_stride)
    if args.max_evaluations is not None:
        config["max_evaluations"] = max(1, args.max_evaluations)
    if args.weights is not None:
        config["weights_path"] = str(args.weights.resolve())
    methods = config["search_methods"] if args.search_method == "both" else [args.search_method]
    seed_everything(int(config["random_seed"]))

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    session_id = f"{config['experiment_id']}__{timestamp}__seed{config['random_seed']}"
    output_root = args.output_root.resolve()
    raw_root = output_root / "raw"
    aggregate_root = output_root / "aggregated" / session_id
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_root / f"{session_id}.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    weights: Path | None = None
    detector: YoloFrameDetector | None = None
    if config["detector"] == "yolov8":
        weights = resolve_weights(config)
        weights_hash = file_sha256(weights)
        detector = YoloFrameDetector(weights, config["yolov8"])
        model_metadata = detector.metadata(weights, weights_hash)
    else:
        model_metadata = heuristic_metadata(config)

    run_config = deepcopy(config)
    run_config["execution"] = {
        "session_id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPO_ROOT),
        "host_platform": platform.platform(),
        "python_executable": sys.executable,
        "deterministic_single_scenario": True,
    }
    run_config["model"] = model_metadata
    aggregate_root.mkdir(parents=True, exist_ok=False)
    write_json(aggregate_root / "config_snapshot.json", run_config)

    cache_namespace = "cache"
    if args.cache_namespace:
        if not args.cache_namespace.replace("_", "").replace("-", "").isalnum():
            raise ValueError("--cache-namespace may contain only letters, digits, '_' and '-'")
        cache_namespace = args.cache_namespace
    elif args.force:
        cache_namespace = f"cache_{timestamp}"
    exporter = MatlabFrameExporter(REPO_ROOT, HERE)
    repository = EvaluationRepository(
        config=config,
        cache_root=raw_root / "_shared_evaluations" / cache_namespace,
        exporter=exporter,
        yolo_detector=detector,
        model_metadata=model_metadata,
    )
    results: dict[str, Any] = {}
    try:
        for method in methods:
            run_dir = raw_root / method / session_id
            results[method] = run_search_method(
                method,
                config,
                repository,
                run_dir,
                session_id,
                model_metadata,
            )
    finally:
        exporter.close()

    comparison = {
        "session_id": session_id,
        "experiment_id": config["experiment_id"],
        "detector": config["detector"],
        "model": model_metadata,
        "matlab_engine_startup_seconds": exporter.startup_seconds,
        "methods": {method: result["summary"] for method, result in results.items()},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(aggregate_root / "comparison_results.json", comparison)

    # Generate tables, figures, and Korean manuscript text from actual JSON.
    from .report_results import generate_outputs
    from .validate_results import validate_outputs

    generate_outputs(config, results, aggregate_root, output_root / "figures" / session_id)
    validate_outputs(config, results, output_root, aggregate_root)
    print(json.dumps({"session_id": session_id, "results": str(aggregate_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
