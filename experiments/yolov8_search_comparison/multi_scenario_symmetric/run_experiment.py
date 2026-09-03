"""Execute independent S0--S4 YOLOv8s symmetric boundary searches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import statistics
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..run_comparison import YoloFrameDetector, file_sha256, seed_everything, validate_frame_alignment
from ..search_core import Environment, bbox_iou_xywh, choose_next_environment, classify_verdict, compute_map50, environment_gap, map_gap


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG = HERE / "config" / "experiment_config.json"
DEFAULT_PLAN = HERE / "config" / "multi_scenario_symmetric_plan_v2" / "scenario_plan.json"
SCENARIO_ROOT = HERE / "scenarios"
RUNS_ROOT = HERE / "runs"
AGGREGATED_ROOT = HERE / "aggregated"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def flatten_csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)) or value is None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten_csv_value(row.get(key)) for key in fields})


class ScenarioMatlabExporter:
    def __init__(self) -> None:
        self.engine = None
        self.startup_seconds = 0.0

    def start(self) -> None:
        if self.engine is not None:
            return
        import matlab.engine

        started = time.perf_counter()
        self.engine = matlab.engine.start_matlab("-nodesktop -nosplash")
        self.engine.cd(str(REPO_ROOT), nargout=0)
        self.engine.addpath(str(REPO_ROOT), nargout=0)
        self.engine.addpath(str(EXPERIMENT_ROOT), nargout=0)
        self.engine.addpath(str(HERE), nargout=0)
        self.startup_seconds = time.perf_counter() - started

    def export(
        self,
        env: Environment,
        output_dir: Path,
        scenario: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.start()
        assert self.engine is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        raw = self.engine.export_scenario_frames(
            float(env.fog_percent),
            float(env.illumination_lux),
            float(env.camera_noise),
            str(output_dir),
            float(scenario["seed"]),
            float(config["frame_stride"]),
            "yolov8",
            str(config["ground_truth_mode"]),
            str(scenario["scenario_config_path"]),
            str(scenario["scenario_config_sha256"]),
            nargout=1,
        )
        return json.loads(str(raw))

    def close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            finally:
                self.engine = None


def matched_iou_summary(frames: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    values: list[float] = []
    per_class: dict[str, list[float]] = {"person": [], "vehicle": []}
    for frame in frames:
        for class_name in per_class:
            gts = [gt for gt in frame.get("ground_truth", []) if gt["class_name"] == class_name]
            detections = sorted(
                [det for det in frame.get("detections", []) if det["class_name"] == class_name],
                key=lambda det: -float(det["confidence"]),
            )
            used: set[int] = set()
            for detection in detections:
                candidates = [
                    (bbox_iou_xywh(detection["bbox_xywh"], gt["bbox_xywh"]), index)
                    for index, gt in enumerate(gts)
                    if index not in used
                ]
                if not candidates:
                    continue
                iou, index = max(candidates)
                if iou >= threshold:
                    used.add(index)
                    values.append(float(iou))
                    per_class[class_name].append(float(iou))
    return {
        "definition": "IoU values of confidence-ordered, one-to-one, class-matched true positives at the AP IoU threshold",
        "matched_tp_count": len(values),
        "maximum_iou": max(values) if values else None,
        "mean_iou": statistics.fmean(values) if values else None,
        "per_class": {
            name: {
                "matched_tp_count": len(items),
                "maximum_iou": max(items) if items else None,
                "mean_iou": statistics.fmean(items) if items else None,
            }
            for name, items in per_class.items()
        },
    }


def precision_recall(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    totals = {name: sum(int(item[field]) for item in metrics["per_class"].values()) for name, field in (("tp", "tp"), ("fp", "fp"), ("fn", "fn"))}
    result["total_precision"] = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    result["total_recall"] = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    for class_name, item in metrics["per_class"].items():
        result[f"{class_name}_precision"] = int(item["tp"]) / max(1, int(item["tp"]) + int(item["fp"]))
        result[f"{class_name}_recall"] = int(item["tp"]) / max(1, int(item["tp"]) + int(item["fn"]))
    return result


class ScenarioEvaluationRepository:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        scenario: dict[str, Any],
        cache_root: Path,
        exporter: ScenarioMatlabExporter,
        detector: YoloFrameDetector,
        model_metadata: dict[str, Any],
        weights_path: Path,
    ) -> None:
        self.config = config
        self.scenario = scenario
        self.cache_root = cache_root
        self.exporter = exporter
        self.detector = detector
        self.model_metadata = model_metadata
        self.weights_path = weights_path

    def identity(self, env: Environment) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario["scenario_id"],
            "trajectory_config_sha256": self.scenario["scenario_config_sha256"],
            "environment": env.as_dict(),
            "seed": int(self.scenario["seed"]),
            "frame_stride": int(self.config["frame_stride"]),
            "gt_version": self.config["gt_version"],
            "detector": "yolov8",
            "weights_sha256": self.model_metadata["weights_sha256"],
            "inference_settings": self.config["yolov8"],
        }

    def key(self, env: Environment) -> str:
        raw = json.dumps(self.identity(env), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def _verify_immutable_inputs(self) -> None:
        actual_weight_hash = file_sha256(self.weights_path)
        if actual_weight_hash != self.config["expected_weights_sha256"]:
            raise RuntimeError(f"Weights SHA-256 changed during experiment: {actual_weight_hash}")
        scenario_path = Path(self.scenario["scenario_config_path"])
        actual_scenario_hash = file_sha256(scenario_path)
        if actual_scenario_hash != self.scenario["scenario_config_sha256"]:
            raise RuntimeError(f"Scenario config changed during experiment: {self.scenario['scenario_id']} {actual_scenario_hash}")

    def _validate_manifest(self, manifest: dict[str, Any], env: Environment) -> None:
        if manifest.get("scenario_id") != self.scenario["scenario_id"]:
            raise ValueError("Scenario ID mismatch in manifest")
        if manifest.get("scenario_config_sha256") != self.scenario["scenario_config_sha256"]:
            raise ValueError("Scenario config hash mismatch in manifest")
        if manifest.get("ground_truth_mode") != self.config["gt_version"]:
            raise ValueError("GT version mismatch in manifest")
        if manifest.get("environment") != env.as_dict():
            raise ValueError("Environment mismatch in manifest")
        if int(manifest.get("evaluated_frame_count", -1)) != int(self.config["expected_frame_count"]):
            raise ValueError("Manifest must contain exactly 181 evaluated frames")
        validate_frame_alignment(manifest["frames"], int(self.config["frame_stride"]))
        if not all(Path(frame["image_path"]).is_file() for frame in manifest["frames"]):
            raise FileNotFoundError("One or more manifest images are missing")

    def evaluate(self, env: Environment, initial_manifest: Path | None = None) -> tuple[dict[str, Any], bool]:
        self._verify_immutable_inputs()
        key = self.key(env)
        cache_dir = self.cache_root / key
        evaluation_path = cache_dir / "evaluation.json"
        if evaluation_path.is_file():
            evaluation = read_json(evaluation_path)
            if evaluation.get("evaluation_identity") != self.identity(env):
                raise RuntimeError(f"Evaluation identity collision at {evaluation_path}")
            return evaluation, True

        if cache_dir.exists():
            resolved_cache = cache_dir.resolve()
            resolved_root = self.cache_root.resolve()
            if resolved_cache.parent != resolved_root:
                raise RuntimeError(f"Refusing to archive partial cache outside scenario repository: {resolved_cache}")
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive = cache_dir.with_name(f"{cache_dir.name}__incomplete__{suffix}")
            cache_dir.rename(archive)
            logging.warning("Preserved incomplete evaluation directory before retry: %s", archive)
        cache_dir.mkdir(parents=True, exist_ok=False)
        if initial_manifest is not None:
            manifest = read_json(initial_manifest)
            render_source = "preinference_gt_validated_initial_manifest"
        else:
            manifest = self.exporter.export(env, cache_dir, self.scenario, self.config)
            render_source = "new_scenario_specific_render"
        self._validate_manifest(manifest, env)
        paths = [str(frame["image_path"]) for frame in manifest["frames"]]
        detections, inference_seconds, speed_rows = self.detector.detect_paths(paths)
        scoring_started = time.perf_counter()
        frames = [
            {
                "frame_index": int(frame["frame_index"]),
                "time_seconds": float(frame["time_seconds"]),
                "uav_xyz": frame.get("uav_xyz"),
                "image_path": frame["image_path"],
                "ground_truth": frame.get("ground_truth", []),
                "detections": frame_detections,
            }
            for frame, frame_detections in zip(manifest["frames"], detections)
        ]
        metrics = compute_map50(frames, float(self.config["yolov8"]["ap_iou_threshold"]))
        metrics.update(precision_recall(metrics))
        metrics["matched_iou"] = matched_iou_summary(frames, float(self.config["yolov8"]["ap_iou_threshold"]))
        scoring_seconds = time.perf_counter() - scoring_started
        timings = {
            "simulation_seconds": float(manifest["geometry_simulation_seconds"]),
            "frame_rendering_seconds": float(manifest["frame_rendering_seconds"]),
            "yolov8_inference_seconds": float(inference_seconds),
            "metric_scoring_seconds": float(scoring_seconds),
        }
        timings["total_execution_seconds"] = sum(timings.values())
        evaluation = {
            "schema_version": "2.0",
            "cache_key": key,
            "evaluation_identity": self.identity(env),
            "scenario_id": self.scenario["scenario_id"],
            "scenario_config_sha256": self.scenario["scenario_config_sha256"],
            "environment": env.as_dict(),
            "random_seed": int(self.scenario["seed"]),
            "model": self.model_metadata,
            "gt_version": self.config["gt_version"],
            "render_source": render_source,
            "matlab": {
                "version": manifest["matlab_version"],
                "release": manifest["matlab_release"],
                "renderer": manifest["renderer"],
                "image_width": int(manifest["image_width"]),
                "image_height": int(manifest["image_height"]),
                "evaluated_frame_count": int(manifest["evaluated_frame_count"]),
            },
            "metrics": metrics,
            "timings": timings,
            "ultralytics_speed_ms_per_frame": speed_rows,
            "effective_inference_batch_size": getattr(self.detector, "last_effective_batch_size", None),
            "frames": frames,
        }
        with evaluation_path.open("w", encoding="utf-8") as handle:
            json.dump(evaluation, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return evaluation, False


def anchor(env: Environment | None, evaluation: dict[str, Any] | None, config: dict[str, Any]) -> dict[str, Any] | None:
    if env is None or evaluation is None:
        return None
    score = float(evaluation["metrics"]["map50"])
    return {
        **env.as_dict(),
        "map50": score,
        "verdict": classify_verdict(score, config["thresholds"]["pass_map50"], config["thresholds"]["fail_map50"]),
        "cache_key": evaluation["cache_key"],
    }


def result_record(
    *,
    session_id: str,
    scenario: dict[str, Any],
    iteration: int,
    env: Environment,
    evaluation: dict[str, Any],
    verdict: str,
    cache_hit: bool,
    nonfail_anchor: dict[str, Any] | None,
    fail_anchor: dict[str, Any] | None,
    next_env: Environment | None,
    calculation: str,
    policy_seconds: float,
) -> dict[str, Any]:
    metrics = evaluation["metrics"]
    person, vehicle = metrics["per_class"]["person"], metrics["per_class"]["vehicle"]
    tp, fp, fn = (sum(int(x[name]) for x in (person, vehicle)) for name in ("tp", "fp", "fn"))
    timings = evaluation["timings"]
    return {
        "session_id": session_id,
        "scenario_id": scenario["scenario_id"],
        "trajectory_name": scenario["trajectory_name"],
        "iteration": iteration,
        **env.as_dict(),
        "person_ap50": float(metrics["person_ap50"]),
        "vehicle_ap50": float(metrics["vehicle_ap50"]),
        "map50": float(metrics["map50"]),
        "person_gt_count": int(person["total_gt"]),
        "vehicle_gt_count": int(vehicle["total_gt"]),
        "person_detection_count": int(person["n_detections"]),
        "vehicle_detection_count": int(vehicle["n_detections"]),
        "person_tp": int(person["tp"]),
        "person_fp": int(person["fp"]),
        "person_fn": int(person["fn"]),
        "vehicle_tp": int(vehicle["tp"]),
        "vehicle_fp": int(vehicle["fp"]),
        "vehicle_fn": int(vehicle["fn"]),
        "total_tp": tp,
        "total_fp": fp,
        "total_fn": fn,
        "precision": float(metrics["total_precision"]),
        "recall": float(metrics["total_recall"]),
        "person_precision": float(metrics["person_precision"]),
        "person_recall": float(metrics["person_recall"]),
        "vehicle_precision": float(metrics["vehicle_precision"]),
        "vehicle_recall": float(metrics["vehicle_recall"]),
        "maximum_iou": metrics["matched_iou"]["maximum_iou"],
        "mean_iou": metrics["matched_iou"]["mean_iou"],
        "verdict": verdict,
        "nonfail_anchor": nonfail_anchor,
        "fail_anchor": fail_anchor,
        "next_environment": next_env.as_dict() if next_env else None,
        "next_condition_calculation": calculation,
        "simulation_seconds": float(timings["simulation_seconds"]),
        "frame_rendering_seconds": float(timings["frame_rendering_seconds"]),
        "yolov8_inference_seconds": float(timings["yolov8_inference_seconds"]),
        "metric_scoring_seconds": float(timings["metric_scoring_seconds"]),
        "policy_seconds": policy_seconds,
        "total_execution_seconds": float(timings["total_execution_seconds"]) + policy_seconds,
        "evaluation_store_used": cache_hit,
        "evaluation_cache_key": evaluation["cache_key"],
        "trajectory_config_sha256": scenario["scenario_config_sha256"],
        "weights_sha256": evaluation["model"]["weights_sha256"],
        "gt_version": evaluation["gt_version"],
    }


def monotonicity_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for previous, current in zip(records, records[1:]):
        worsened = (
            current["fog_percent"] >= previous["fog_percent"]
            and current["illumination_lux"] <= previous["illumination_lux"]
            and current["camera_noise"] >= previous["camera_noise"]
            and (
                current["fog_percent"] > previous["fog_percent"]
                or current["illumination_lux"] < previous["illumination_lux"]
                or current["camera_noise"] > previous["camera_noise"]
            )
        )
        if worsened and current["map50"] > previous["map50"]:
            events.append(
                {
                    "from_iteration": previous["iteration"],
                    "to_iteration": current["iteration"],
                    "from_environment": {key: previous[key] for key in ("fog_percent", "illumination_lux", "camera_noise")},
                    "to_environment": {key: current[key] for key in ("fog_percent", "illumination_lux", "camera_noise")},
                    "from_map50": previous["map50"],
                    "to_map50": current["map50"],
                    "map50_increase": current["map50"] - previous["map50"],
                }
            )
    return events


def run_scenario(
    *,
    session_id: str,
    scenario: dict[str, Any],
    config: dict[str, Any],
    repository: ScenarioEvaluationRepository,
    initial_manifest: Path,
    scenario_run_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    initial_env = Environment.from_dict(config["initial_environment"])
    initial_evaluation, initial_cache_hit = repository.evaluate(initial_env, initial_manifest)
    initial_score = float(initial_evaluation["metrics"]["map50"])
    initial_verdict = classify_verdict(initial_score, config["thresholds"]["pass_map50"], config["thresholds"]["fail_map50"])
    initial_summary = result_record(
        session_id=session_id,
        scenario=scenario,
        iteration=0,
        env=initial_env,
        evaluation=initial_evaluation,
        verdict=initial_verdict,
        cache_hit=initial_cache_hit,
        nonfail_anchor=None,
        fail_anchor=None,
        next_env=None,
        calculation="independent_initial_condition_evaluation_before_search",
        policy_seconds=0.0,
    )
    scenario_run_dir.mkdir(parents=True, exist_ok=resume)
    if (scenario_run_dir / "initial_evaluation.json").exists():
        if not resume:
            raise FileExistsError(f"Existing initial evaluation without resume: {scenario_run_dir}")
        initial_summary = read_json(scenario_run_dir / "initial_evaluation.json")
    else:
        write_json(scenario_run_dir / "initial_evaluation.json", initial_summary)
        write_csv(scenario_run_dir / "initial_evaluation.csv", [initial_summary])

    records: list[dict[str, Any]] = []
    nonfail_env: Environment | None = None
    nonfail_eval: dict[str, Any] | None = None
    fail_env: Environment | None = None
    fail_eval: dict[str, Any] | None = None
    first_fail: int | None = None
    if initial_verdict == "PASS":
        current = initial_env
        for iteration in range(1, int(config["max_evaluations"]) + 1):
            evaluation, cache_hit = repository.evaluate(current)
            score = float(evaluation["metrics"]["map50"])
            verdict = classify_verdict(score, config["thresholds"]["pass_map50"], config["thresholds"]["fail_map50"])
            if verdict == "FAIL":
                fail_env, fail_eval = current, evaluation
                if first_fail is None:
                    first_fail = iteration
            else:
                nonfail_env, nonfail_eval = current, evaluation
            policy_started = time.perf_counter()
            next_env, calculation = choose_next_environment(
                method="symmetric",
                current=current,
                current_verdict=verdict,
                nonfail_anchor=nonfail_env,
                fail_anchor=fail_env,
                bounds=config["environment_bounds"],
                degradation=config["degradation"],
            )
            policy_seconds = time.perf_counter() - policy_started
            record = result_record(
                    session_id=session_id,
                    scenario=scenario,
                    iteration=iteration,
                    env=current,
                    evaluation=evaluation,
                    verdict=verdict,
                    cache_hit=cache_hit,
                    nonfail_anchor=anchor(nonfail_env, nonfail_eval, config),
                    fail_anchor=anchor(fail_env, fail_eval, config),
                    next_env=next_env,
                    calculation=calculation,
                    policy_seconds=policy_seconds,
                )
            records.append(record)
            checkpoint = scenario_run_dir / "iteration_records" / f"iteration_{iteration:02d}.json"
            if checkpoint.exists():
                previous_record = read_json(checkpoint)
                stable_fields = ("scenario_id", "iteration", "fog_percent", "illumination_lux", "camera_noise", "map50", "verdict", "evaluation_cache_key")
                if any(previous_record.get(key) != record.get(key) for key in stable_fields):
                    raise RuntimeError(f"Stable iteration result changed during resume: {checkpoint}")
                record = previous_record
                records[-1] = previous_record
            else:
                write_json(checkpoint, record)
            logging.info("%s iteration=%d mAP50=%.6f verdict=%s env=%s", scenario["scenario_id"], iteration, score, verdict, current.as_dict())
            if next_env is None or next_env == current:
                break
            current = next_env

    boundary_success = initial_verdict == "PASS" and nonfail_env is not None and fail_env is not None and len(records) <= 10
    nonfail_result = anchor(nonfail_env, nonfail_eval, config)
    fail_result = anchor(fail_env, fail_eval, config)
    delta = None
    if boundary_success and nonfail_result and fail_result:
        delta = {
            "fog_percent_points": abs(nonfail_env.fog_percent - fail_env.fog_percent),
            "illumination_lux": abs(nonfail_env.illumination_lux - fail_env.illumination_lux),
            "camera_noise": abs(nonfail_env.camera_noise - fail_env.camera_noise),
            "map50": map_gap(nonfail_result["map50"], fail_result["map50"]),
        }
    events = monotonicity_events(records)
    summary = {
        "session_id": session_id,
        "scenario_id": scenario["scenario_id"],
        "trajectory_name": scenario["trajectory_name"],
        "major_changes": scenario["major_changes"],
        "scenario_config_sha256": scenario["scenario_config_sha256"],
        "initial_map50": initial_score,
        "initial_verdict": initial_verdict,
        "initial_pass": initial_verdict == "PASS",
        "first_fail_iteration": first_fail,
        "search_success": boundary_success,
        "final_nonfail_anchor": nonfail_result,
        "final_fail_anchor": fail_result,
        "gap_map50": delta["map50"] if delta else None,
        "gap_environment_normalized": environment_gap(nonfail_env, fail_env, config["environment_bounds"]) if boundary_success else None,
        "boundary_physical_differences": delta,
        "environment_range_denominators": {
            key: {"min": values[0], "max": values[1], "range": values[1] - values[0]}
            for key, values in config["environment_bounds"].items()
        },
        "evaluation_count": len(records),
        "actual_new_inference_count": sum(not item["evaluation_store_used"] for item in records),
        "evaluation_store_use_count": sum(bool(item["evaluation_store_used"]) for item in records),
        "total_execution_seconds": sum(float(item["total_execution_seconds"]) for item in records),
        "scenario_driver_wall_seconds": time.perf_counter() - wall_started,
        "nonmonotonic_map_increase_count": len(events),
        "nonmonotonic_map_increase_events": events,
        "failure_or_unsearchable_reason": (
            None
            if boundary_success
            else "초기 PASS 미확보로 인한 탐색 불가"
            if initial_verdict != "PASS"
            else "10회 이내 유효한 비실패/FAIL 앵커 미확보"
        ),
        "boundary_interpretation": "The boundary is the interval between final non-failure (PASS or MARGINAL) and FAIL anchors, not a single exact environment point.",
    }
    write_json(scenario_run_dir / "iterations.json", records)
    write_csv(scenario_run_dir / "iterations.csv", records)
    write_json(scenario_run_dir / "scenario_summary.json", summary)
    return {"initial": initial_summary, "records": records, "summary": summary}


def stat_summary(values: list[float | int]) -> dict[str, Any]:
    numeric = [float(x) for x in values]
    if not numeric:
        return {"count": 0, "mean": None, "population_std": None, "median": None, "min": None, "max": None, "range": None}
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "population_std": statistics.pstdev(numeric),
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "range": max(numeric) - min(numeric),
    }


def compare_s0(records: list[dict[str, Any]]) -> dict[str, Any]:
    prior = EXPERIMENT_ROOT / "aggregated" / "kci_yolov8s_boundary_search_v1__20260902_173932_443628__seed42" / "symmetric_iterations.csv"
    with prior.open("r", encoding="utf-8-sig", newline="") as handle:
        old = list(csv.DictReader(handle))
    comparisons = []
    for new, previous in zip(records, old):
        comparisons.append(
            {
                "iteration": new["iteration"],
                "new_environment": {key: new[key] for key in ("fog_percent", "illumination_lux", "camera_noise")},
                "old_environment": {key: float(previous[key]) for key in ("fog_percent", "illumination_lux", "camera_noise")},
                "new_map50": new["map50"],
                "old_map50": float(previous["map50"]),
                "map50_difference": new["map50"] - float(previous["map50"]),
                "new_verdict": new["verdict"],
                "old_verdict": previous["verdict"],
            }
        )
    return {
        "prior_result_path": str(prior.resolve()),
        "new_result_is_independent": True,
        "compared_iteration_count": len(comparisons),
        "maximum_absolute_map50_difference": max(abs(x["map50_difference"]) for x in comparisons) if comparisons else None,
        "verdict_difference_count": sum(x["new_verdict"] != x["old_verdict"] for x in comparisons),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--scenario-root", type=Path, default=SCENARIO_ROOT)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--aggregated-root", type=Path, default=AGGREGATED_ROOT)
    parser.add_argument("--resume-session", help="Resume an interrupted session without reusing any other session.")
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    plan_path = args.plan.resolve()
    plan = read_json(plan_path)
    scenario_root = args.scenario_root.resolve()
    runs_root = args.runs_root.resolve()
    aggregated_root = args.aggregated_root.resolve()
    gt_audit = read_json(scenario_root / plan["plan_id"] / "gt_validation.json")
    leakage = read_json(scenario_root / plan["plan_id"] / "scenario_leakage_audit.json")
    if not gt_audit.get("all_valid") or not leakage.get("new_scenarios_passed"):
        raise RuntimeError("Pre-inference GT/leakage gate did not pass")

    weights_path = (REPO_ROOT / config["weights_path"]).resolve()
    actual_hash = file_sha256(weights_path)
    if actual_hash != config["expected_weights_sha256"]:
        raise RuntimeError(f"Weights SHA-256 mismatch: expected {config['expected_weights_sha256']} actual {actual_hash}")
    seed_everything(42)
    detector = YoloFrameDetector(weights_path, config["yolov8"])
    model_metadata = detector.metadata(weights_path, actual_hash)
    actual_names = {str(key): value for key, value in model_metadata["model_class_names"].items()}
    if actual_names != config["expected_model_class_names"]:
        raise RuntimeError(f"Checkpoint class mapping mismatch: {actual_names}")
    model_text = str(model_metadata.get("checkpoint_model_definition") or "").lower()
    if "yolov8s" not in model_text:
        raise RuntimeError(f"Checkpoint is not identified as YOLOv8s: {model_text}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    session_id = args.resume_session or f"{config['experiment_id']}__{timestamp}"
    session_root = runs_root / session_id
    aggregate_dir = aggregated_root / session_id
    session_root.mkdir(parents=True, exist_ok=bool(args.resume_session))
    aggregate_dir.mkdir(parents=True, exist_ok=bool(args.resume_session))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(session_root / "experiment.log", encoding="utf-8"), logging.StreamHandler()],
    )
    snapshot = {
        "session_id": session_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": file_sha256(args.config.resolve()),
        "scenario_plan_path": str(plan_path),
        "scenario_plan_sha256": file_sha256(plan_path),
        "scenario_plan": plan,
        "model": model_metadata,
        "host": {"platform": platform.platform(), "python": sys.version, "executable": sys.executable},
        "preinference_gate": {"gt_validation_passed": True, "leakage_audit_passed": True},
    }
    snapshot_path = aggregate_dir / "config_snapshot.json"
    if snapshot_path.exists():
        if not args.resume_session:
            raise FileExistsError(snapshot_path)
        previous_snapshot = read_json(snapshot_path)
        if previous_snapshot["config_sha256"] != snapshot["config_sha256"] or previous_snapshot["scenario_plan_sha256"] != snapshot["scenario_plan_sha256"]:
            raise RuntimeError("Resume config/plan does not match the interrupted session")
    else:
        write_json(snapshot_path, snapshot)

    exporter = ScenarioMatlabExporter()
    results: dict[str, Any] = {}
    try:
        # Evaluate every initial condition before starting any boundary search.
        # This global gate prevents an early scenario from being searched before
        # a later S0--S4 scenario is discovered not to PASS initially.
        initial_gate_rows = []
        for scenario in plan["scenarios"]:
            repository = ScenarioEvaluationRepository(
                config=config,
                scenario=scenario,
                cache_root=session_root / scenario["scenario_id"] / "evaluations",
                exporter=exporter,
                detector=detector,
                model_metadata=model_metadata,
                weights_path=weights_path,
            )
            manifest_path = scenario_root / plan["plan_id"] / scenario["scenario_id"] / "initial_condition" / "frame_manifest.json"
            evaluation, cache_hit = repository.evaluate(Environment.from_dict(config["initial_environment"]), manifest_path)
            score = float(evaluation["metrics"]["map50"])
            verdict = classify_verdict(score, config["thresholds"]["pass_map50"], config["thresholds"]["fail_map50"])
            initial_gate_rows.append(
                result_record(
                    session_id=session_id,
                    scenario=scenario,
                    iteration=0,
                    env=Environment.from_dict(config["initial_environment"]),
                    evaluation=evaluation,
                    verdict=verdict,
                    cache_hit=cache_hit,
                    nonfail_anchor=None,
                    fail_anchor=None,
                    next_env=None,
                    calculation="global_initial_pass_gate_before_any_search",
                    policy_seconds=0.0,
                )
            )
        initial_gate = {
            "all_initial_pass": all(row["verdict"] == "PASS" for row in initial_gate_rows),
            "pass_count": sum(row["verdict"] == "PASS" for row in initial_gate_rows),
            "scenario_count": len(initial_gate_rows),
            "search_started": all(row["verdict"] == "PASS" for row in initial_gate_rows),
            "evaluations": initial_gate_rows,
        }
        gate_path = aggregate_dir / "initial_pass_gate.json"
        if gate_path.exists():
            previous_gate = read_json(gate_path)
            stable = [(x["scenario_id"], x["map50"], x["verdict"], x["evaluation_cache_key"]) for x in previous_gate["evaluations"]]
            current = [(x["scenario_id"], x["map50"], x["verdict"], x["evaluation_cache_key"]) for x in initial_gate_rows]
            if stable != current:
                raise RuntimeError("Initial PASS gate changed during resume")
        else:
            write_json(gate_path, initial_gate)
            write_csv(aggregate_dir / "initial_condition_results.csv", initial_gate_rows)
        if not initial_gate["all_initial_pass"]:
            raise RuntimeError(
                f"Global initial PASS gate failed ({initial_gate['pass_count']}/{initial_gate['scenario_count']}); no boundary search was started"
            )

        for scenario in plan["scenarios"]:
            scenario_id = scenario["scenario_id"]
            completed_summary = session_root / scenario_id / "search" / "scenario_summary.json"
            if args.resume_session and completed_summary.exists():
                results[scenario_id] = {
                    "initial": read_json(session_root / scenario_id / "search" / "initial_evaluation.json"),
                    "records": read_json(session_root / scenario_id / "search" / "iterations.json"),
                    "summary": read_json(completed_summary),
                }
                logging.info("Loaded completed scenario from the same resumed session: %s", scenario_id)
                continue
            repository = ScenarioEvaluationRepository(
                config=config,
                scenario=scenario,
                cache_root=session_root / scenario_id / "evaluations",
                exporter=exporter,
                detector=detector,
                model_metadata=model_metadata,
                weights_path=weights_path,
            )
            manifest_path = scenario_root / plan["plan_id"] / scenario_id / "initial_condition" / "frame_manifest.json"
            results[scenario_id] = run_scenario(
                session_id=session_id,
                scenario=scenario,
                config=config,
                repository=repository,
                initial_manifest=manifest_path,
                scenario_run_dir=session_root / scenario_id / "search",
                resume=bool(args.resume_session),
            )
    finally:
        exporter.close()

    summaries = [results[key]["summary"] for key in sorted(results)]
    all_records = [record for key in sorted(results) for record in results[key]["records"]]
    success = [item for item in summaries if item["search_success"]]
    initial_pass_count = sum(item["initial_pass"] for item in summaries)
    overall = {
        "session_id": session_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(summaries),
        "initial_pass_count": initial_pass_count,
        "initial_pass_rate": initial_pass_count / len(summaries),
        "search_success_count": len(success),
        "search_success_rate": len(success) / len(summaries),
        "first_fail_iteration_statistics": stat_summary([item["first_fail_iteration"] for item in success]),
        "gap_map50_statistics": stat_summary([item["gap_map50"] for item in success]),
        "gap_environment_statistics": stat_summary([item["gap_environment_normalized"] for item in success]),
        "total_evaluation_count": sum(item["evaluation_count"] for item in summaries),
        "mean_execution_seconds_per_scenario": statistics.fmean(item["total_execution_seconds"] for item in summaries),
        "scenario_with_nonmonotonic_increase_count": sum(item["nonmonotonic_map_increase_count"] > 0 for item in summaries),
        "environment_bounds": config["environment_bounds"],
        "scope_note": "Five deterministic MATLAB simulation scenarios only; no statistical significance or general superiority claim is made.",
    }
    combined = {"overall": overall, "scenarios": summaries, "initial_evaluations": [results[key]["initial"] for key in sorted(results)]}
    write_json(aggregate_dir / "multi_scenario_results.json", combined)
    write_csv(aggregate_dir / "scenario_summary.csv", summaries)
    write_json(aggregate_dir / "all_iteration_results.json", all_records)
    write_csv(aggregate_dir / "all_iteration_results.csv", all_records)
    s0_comparison = compare_s0(results["S0"]["records"])
    write_json(aggregate_dir / "s0_prior_comparison.json", s0_comparison)
    write_csv(aggregate_dir / "s0_prior_comparison.csv", s0_comparison["comparisons"])
    print(json.dumps({"session_id": session_id, "aggregate_dir": str(aggregate_dir.resolve()), "overall": overall}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
