from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPOSITORY = Path(r"C:\Users\lab\Counterfactual-XAI-Verifier")
WORKTREE = REPOSITORY / ".k"
DIRECT = WORKTREE / "final_kci_direct_yolo_shap_v1"
PREPARATION = DIRECT / "independent_shap_variable_validation_v1_preparation"
OUTPUT = DIRECT / "independent_shap_variable_validation_run_001_lab"
V3 = REPOSITORY / "experiments" / "yolov8_search_comparison" / "shap_counterfactual_boundary_v3"
RESULTS = OUTPUT / "condition_results"
MANIFESTS = OUTPUT / "condition_completion_manifests"
CACHE = OUTPUT / "cache"
EVENTS = OUTPUT / "stage_events.jsonl"
REGISTRY = OUTPUT / "completion_registry.json"
SOURCE_MANIFEST = PREPARATION / "source_manifest.csv"
PROTECTED_BASELINE = PREPARATION / "protected_hashes_before.csv"
SETTINGS_PATH = PREPARATION / "fixed_settings.json"
GATE_PATH = PREPARATION / "locked_gate_definition.json"
CONDITIONS_PATH = PREPARATION / "independent_conditions.json"

EXPECTED_ACCOUNT = r"desktop-cmoipge\lab"
EXPECTED_WEIGHT_SHA256 = "E141B3DC0104CBC03AD2EE573FFA451C369C9D93D8690E560E07C146078F5D9E"
VARIABLES = ("fog", "illumination", "noise")
ENV_KEYS = ("fog_percent", "illumination_lux", "camera_noise")
TOL = 1e-12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    write_json_atomic(path, value)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_event(stage: str, status: str, **fields: Any) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at_utc": utc_now(), "stage": stage, "status": status, **fields}, ensure_ascii=False, allow_nan=False) + "\n")


def current_account() -> str:
    completed = subprocess.run(["whoami"], capture_output=True, text=True, timeout=10, check=True)
    return completed.stdout.strip().lower()


def verdict(map50: float) -> str:
    if map50 >= 0.50:
        return "PASS"
    if map50 >= 0.25:
        return "MARGINAL"
    return "FAIL"


def verify_source_manifest() -> None:
    rows = read_csv(SOURCE_MANIFEST)
    if not rows:
        raise RuntimeError("source manifest is empty")
    mismatches = []
    for row in rows:
        path = Path(row["path"])
        actual = sha256(path) if path.is_file() else ""
        if actual != row["sha256"].upper():
            mismatches.append(str(path))
    if mismatches:
        raise RuntimeError(f"source-manifest mismatch: {mismatches[:5]}")


def protected_snapshot(path: Path) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    for item in read_csv(PROTECTED_BASELINE):
        target = Path(item["path"])
        expected = item["expected_sha256"].upper()
        actual = sha256(target) if target.is_file() else ""
        rows.append({
            "path": str(target), "expected_sha256": expected,
            "actual_sha256": actual, "exists": target.is_file(), "matches": actual == expected,
        })
    write_csv_atomic(path, rows, ["path", "expected_sha256", "actual_sha256", "exists", "matches"])
    return rows, len(rows) == 129 and all(row["matches"] for row in rows)


def verify_preflight(resume: bool) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if current_account() != EXPECTED_ACCOUNT:
        raise RuntimeError("interactive lab account gate failed")
    if os.environ.get("MATLAB_PREFDIR"):
        raise RuntimeError("MATLAB_PREFDIR must be absent")
    if not os.environ.get("DIRECT_SHAP_INDEPENDENT_VALIDATION_POPEN_TOKEN"):
        raise RuntimeError("direct-Popen supervisor token is absent")
    branch = subprocess.run(
        ["git", "-C", str(WORKTREE), "branch", "--show-current"],
        capture_output=True, text=True, timeout=20, check=True,
    ).stdout.strip()
    if branch != "kci":
        raise RuntimeError(f"branch changed: {branch}")
    verify_source_manifest()
    _, protected_ok = protected_snapshot(OUTPUT / "protected_hashes_worker_before.csv")
    if not protected_ok:
        raise RuntimeError("protected-file preflight mismatch")
    settings = read_json(SETTINGS_PATH)
    gate = read_json(GATE_PATH)
    conditions = read_json(CONDITIONS_PATH)["records"]
    if settings.get("status") != "LOCKED_BEFORE_ANY_INDEPENDENT_RESULT" or gate.get("status") != "LOCKED_BEFORE_ANY_INDEPENDENT_RESULT":
        raise RuntimeError("settings or gate is not pre-result locked")
    if settings.get("RF_or_surrogate_or_virtual_map_used") is not False:
        raise RuntimeError("forbidden surrogate flag")
    if settings["detector"]["weights_sha256"] != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("locked weight hash differs")
    if len(conditions) != 110 or len({row["condition_id"] for row in conditions}) != 110:
        raise RuntimeError("110-condition lock is incomplete")
    if not resume and (RESULTS.exists() or MANIFESTS.exists() or REGISTRY.exists()):
        raise RuntimeError("first-run output already contains condition state")
    return settings, gate, conditions


def install_instrumentation(service: Any, ev: Any) -> None:
    original_render = service.renderer.render
    original_matlab_start = service.renderer._start
    original_infer = service.detector.infer
    original_model = service.detector.model
    original_compute = ev.compute_metrics

    def render(*args: Any, **kwargs: Any) -> Any:
        startup_started = time.perf_counter()
        append_event("matlab_startup", "started", timeout_seconds=300)
        try:
            original_matlab_start()
        except Exception as exc:
            append_event("matlab_startup", "failed", error=repr(exc))
            raise
        append_event("matlab_startup", "completed", wall_seconds=time.perf_counter() - startup_started)
        started = time.perf_counter()
        append_event("render", "started", scenario_id=args[1]["scenario_id"], timeout_seconds=300)
        try:
            value = original_render(*args, **kwargs)
        except Exception as exc:
            append_event("render", "failed", error=repr(exc))
            raise
        append_event("render", "completed", wall_seconds=time.perf_counter() - started)
        return value

    def infer(*args: Any, **kwargs: Any) -> Any:
        model_started = time.perf_counter()
        append_event("detector_startup", "started", timeout_seconds=180)
        try:
            original_model(int(args[0]))
        except Exception as exc:
            append_event("detector_startup", "failed", error=repr(exc))
            raise
        append_event("detector_startup", "completed", wall_seconds=time.perf_counter() - model_started)
        started = time.perf_counter()
        append_event("yolo_inference", "started", frame_count=len(args[1]), timeout_seconds=120)
        try:
            value = original_infer(*args, **kwargs)
        except Exception as exc:
            append_event("yolo_inference", "failed", error=repr(exc))
            raise
        append_event("yolo_inference", "completed", wall_seconds=time.perf_counter() - started, effective_batch_size=value[2])
        return value

    def score(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        append_event("scoring", "started", timeout_seconds=60)
        try:
            value = original_compute(*args, **kwargs)
        except Exception as exc:
            append_event("scoring", "failed", error=repr(exc))
            raise
        append_event("scoring", "completed", wall_seconds=time.perf_counter() - started)
        return value

    service.renderer.render = render
    service.detector.infer = infer
    ev.compute_metrics = score


def completion_paths(condition_id: str) -> tuple[Path, Path]:
    return RESULTS / f"{condition_id}.json", MANIFESTS / f"{condition_id}.json"


def validate_completed(condition: dict[str, Any], ev: Any, service: Any) -> dict[str, Any] | None:
    result_path, completion_path = completion_paths(condition["condition_id"])
    if not result_path.exists() and not completion_path.exists():
        return None
    if not result_path.is_file() or not completion_path.is_file():
        raise RuntimeError(f"partial completion pair cannot be reused: {condition['condition_id']}")
    result = read_json(result_path)
    completion = read_json(completion_path)
    if completion.get("complete") is not True or completion.get("condition_id") != condition["condition_id"]:
        raise RuntimeError(f"invalid completion record: {condition['condition_id']}")
    if completion.get("setting_identity_sha256") != condition["setting_identity_sha256"]:
        raise RuntimeError(f"setting identity mismatch on resume: {condition['condition_id']}")
    if sha256(result_path) != completion.get("result_sha256"):
        raise RuntimeError(f"result hash mismatch on resume: {condition['condition_id']}")
    for key in ("manifest", "inference"):
        target = Path(completion[f"{key}_path"])
        if not target.is_file() or sha256(target) != completion[f"{key}_sha256"]:
            raise RuntimeError(f"{key} hash mismatch on resume: {condition['condition_id']}")
    manifest = read_json(Path(completion["manifest_path"]))
    inference = read_json(Path(completion["inference_path"]))
    env = ev.Environment.from_dict({key: condition[key] for key in ENV_KEYS}).canonical()
    ev._validate_manifest(manifest, service.scenarios[condition["scenario_id"]], env)
    if len(inference.get("detections", [])) != 181 or result.get("weights_sha256") != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"frame or weight mismatch on resume: {condition['condition_id']}")
    metrics = ev.compute_metrics(manifest, inference["detections"])
    if not math.isclose(float(metrics["map50"]), float(result["metrics"]["map50"]), rel_tol=0.0, abs_tol=TOL):
        raise RuntimeError(f"mAP mismatch on resume: {condition['condition_id']}")
    return result


def persist_result(condition: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    condition_id = condition["condition_id"]
    result_path, completion_path = completion_paths(condition_id)
    manifest_path = Path(evaluation["manifest_path"])
    inference_path = Path(evaluation["inference_path"])
    metrics = evaluation["metrics"]
    if evaluation["weights_sha256"] != EXPECTED_WEIGHT_SHA256 or evaluation["verdict"] != verdict(float(metrics["map50"])):
        raise RuntimeError(f"weight or verdict mismatch: {condition_id}")
    if not evaluation["cache"]["actual_new_render"] or not evaluation["cache"]["actual_new_inference"]:
        raise RuntimeError(f"unlocked internal cache reuse: {condition_id}")
    result = {
        "schema_version": "1.0", "created_at_utc": utc_now(),
        "condition_id": condition_id, "scenario_id": condition["scenario_id"],
        "role": condition["role"], "cube_mask": condition["cube_mask"],
        "validation_variable": condition["validation_variable"],
        "environment": {key: float(condition[key]) for key in ENV_KEYS},
        "setting_identity_sha256": condition["setting_identity_sha256"],
        "model_seed": 42, "weights_sha256": evaluation["weights_sha256"],
        "metrics": metrics, "verdict": evaluation["verdict"], "frame_count": 181,
        "manifest_path": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path),
        "inference_path": str(inference_path.resolve()), "inference_sha256": sha256(inference_path),
        "render_key": evaluation["render_key"], "inference_key": evaluation["inference_key"],
        "timings": evaluation.get("timings", {}),
        "actual_new_render": True, "actual_new_inference": True, "reused": False,
        "automatic_retry": False, "RF_or_other_surrogate_used": False, "virtual_map_used": False,
    }
    write_json_new(result_path, result)
    completion = {
        "schema_version": "1.0", "completed_at_utc": utc_now(), "complete": True,
        "condition_id": condition_id, "scenario_id": condition["scenario_id"],
        "setting_identity_sha256": condition["setting_identity_sha256"],
        "result_path": str(result_path.resolve()), "result_sha256": sha256(result_path),
        "manifest_path": result["manifest_path"], "manifest_sha256": result["manifest_sha256"],
        "inference_path": result["inference_path"], "inference_sha256": result["inference_sha256"],
        "weights_sha256": result["weights_sha256"], "frame_count": 181,
        "map50": metrics["map50"], "person_ap50": metrics["person_ap50"], "vehicle_ap50": metrics["vehicle_ap50"],
        "verdict": result["verdict"], "actual_new_render": True, "actual_new_inference": True,
        "retry_performed": False,
    }
    write_json_new(completion_path, completion)
    return result


def update_registry(completed: list[dict[str, Any]]) -> None:
    write_json_atomic(REGISTRY, {
        "schema_version": "1.0", "updated_at_utc": utc_now(),
        "completed_logical_evaluations": len(completed), "expected_logical_evaluations": 110,
        "actual_new_renders": sum(int(not row.get("reused", False)) for row in completed),
        "actual_new_inferences": sum(int(not row.get("reused", False)) for row in completed),
        "verified_resume_reuse_count": sum(int(row.get("reused", False)) for row in completed),
        "completed_condition_ids": [row["condition_id"] for row in completed],
        "automatic_retries": 0,
    })


def exact_shapley_subset(values: dict[str, float]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    degradation = {mask: values["000"] - value for mask, value in values.items()}
    scores: dict[str, float] = {}
    terms: list[dict[str, Any]] = []
    n = 3
    for index, variable in enumerate(VARIABLES):
        total = 0.0
        others = [position for position in range(n) if position != index]
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                before_bits = ["0"] * n
                for position in subset:
                    before_bits[position] = "1"
                before = "".join(before_bits)
                after_bits = list(before_bits)
                after_bits[index] = "1"
                after = "".join(after_bits)
                weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                marginal = degradation[after] - degradation[before]
                weighted = weight * marginal
                total += weighted
                terms.append({
                    "variable": variable, "subset_without_variable": before,
                    "subset_with_variable": after, "weight": weight,
                    "degradation_before": degradation[before], "degradation_after": degradation[after],
                    "signed_marginal": marginal, "weighted_term": weighted,
                })
        scores[variable] = total
    return scores, terms


def exact_shapley_permutations(values: dict[str, float]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    degradation = {mask: values["000"] - value for mask, value in values.items()}
    totals = {variable: 0.0 for variable in VARIABLES}
    rows: list[dict[str, Any]] = []
    permutations = list(itertools.permutations(range(3)))
    for order_index, order in enumerate(permutations, start=1):
        bits = ["0", "0", "0"]
        for step, position in enumerate(order, start=1):
            before = "".join(bits)
            bits[position] = "1"
            after = "".join(bits)
            marginal = degradation[after] - degradation[before]
            variable = VARIABLES[position]
            totals[variable] += marginal
            rows.append({
                "permutation_index": order_index,
                "permutation": ">".join(VARIABLES[item] for item in order),
                "step": step, "variable": variable, "before_mask": before,
                "after_mask": after, "signed_marginal": marginal,
            })
    return {key: value / len(permutations) for key, value in totals.items()}, rows


def average_descending_ranks(values: dict[str, float], tolerance: float = TOL) -> dict[str, float]:
    ordered = sorted(values, key=lambda key: (-values[key], key))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and abs(values[ordered[end]] - values[ordered[start]]) <= tolerance:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for key in ordered[start:end]:
            ranks[key] = rank
        start = end
    return ranks


def spearman_three(a: dict[str, float], b: dict[str, float], tolerance_a: float = TOL, tolerance_b: float = TOL) -> float | None:
    left = average_descending_ranks(a, tolerance_a)
    right = average_descending_ranks(b, tolerance_b)
    x = [left[key] for key in VARIABLES]
    y = [right[key] for key in VARIABLES]
    xm = sum(x) / 3.0
    ym = sum(y) / 3.0
    numerator = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
    denominator = math.sqrt(sum((xi - xm) ** 2 for xi in x) * sum((yi - ym) ** 2 for yi in y))
    return None if denominator <= TOL else numerator / denominator


def sign(value: float) -> int:
    return 1 if value > TOL else -1 if value < -TOL else 0


def classify_scene(shapley: dict[str, float], actual: dict[str, float]) -> dict[str, Any]:
    shap_sorted = sorted(VARIABLES, key=lambda key: (-shapley[key], key))
    actual_sorted = sorted(VARIABLES, key=lambda key: (-actual[key], key))
    shap_gap = shapley[shap_sorted[0]] - shapley[shap_sorted[1]]
    actual_gap = actual[actual_sorted[0]] - actual[actual_sorted[1]]
    if shap_gap <= TOL:
        status = "INCONCLUSIVE"
        reason = "Shapley top is not unique"
    elif actual_gap <= 0.01 + TOL:
        status = "INCONCLUSIVE"
        reason = "actual top-second gap is at most 0.01 mAP"
    elif shap_sorted[0] == actual_sorted[0]:
        status = "MATCH"
        reason = "unique Shapley top equals distinct actual top"
    else:
        status = "MISMATCH"
        reason = "another variable causes more than 0.01 greater actual degradation"
    direction_matches = sum(sign(shapley[key]) == sign(actual[key]) for key in VARIABLES)
    return {
        "scene_status": status, "reason": reason,
        "shapley_rank": ">".join(shap_sorted), "actual_rank": ">".join(actual_sorted),
        "shapley_top": shap_sorted[0], "actual_top": actual_sorted[0],
        "shapley_top_second_gap": shap_gap, "actual_top_second_gap_map": actual_gap,
        "top_matches": shap_sorted[0] == actual_sorted[0],
        "spearman": spearman_three(shapley, actual, TOL, 0.01),
        "direction_matches": direction_matches, "direction_total": 3,
        "direction_agreement": direction_matches / 3.0,
    }


def aggregate_status(scene_rows: list[dict[str, Any]], technical_complete: bool) -> str:
    if not technical_complete or len(scene_rows) != 10:
        return "TECHNICAL_INCOMPLETE"
    match = sum(row["scene_status"] == "MATCH" for row in scene_rows)
    mismatch = sum(row["scene_status"] == "MISMATCH" for row in scene_rows)
    if match >= 7:
        return "INDEPENDENT_SHAP_VARIABLE_SELECTION_PASS"
    if mismatch > 0:
        return "INDEPENDENT_SHAP_VARIABLE_SELECTION_FAIL"
    return "INDEPENDENT_SHAP_VARIABLE_SELECTION_INCONCLUSIVE"


def finalize(completed: list[dict[str, Any]]) -> None:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        by_scene.setdefault(row["scenario_id"], []).append(row)
    if len(completed) != 110 or len(by_scene) != 10:
        raise RuntimeError("finalization requires all 110 conditions and 10 scenes")

    condition_rows: list[dict[str, Any]] = []
    shapley_terms: list[dict[str, Any]] = []
    permutation_terms: list[dict[str, Any]] = []
    shapley_rows: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    additivity_all = True
    independent_all = True
    for scene_id in sorted(by_scene):
        rows = by_scene[scene_id]
        cube = {row["cube_mask"]: float(row["metrics"]["map50"]) for row in rows if row["role"] == "exact_shapley_cube"}
        validation = {row["validation_variable"]: float(row["metrics"]["map50"]) for row in rows if row["role"] == "separate_actual_rank_validation"}
        if set(cube) != {"000", "001", "010", "011", "100", "101", "110", "111"} or set(validation) != set(VARIABLES):
            raise RuntimeError(f"scene condition roles incomplete: {scene_id}")
        subset_scores, subset_rows = exact_shapley_subset(cube)
        permutation_scores, permutation_rows = exact_shapley_permutations(cube)
        total_degradation = cube["000"] - cube["111"]
        additivity_residual = sum(subset_scores.values()) - total_degradation
        independent_residual = max(abs(subset_scores[key] - permutation_scores[key]) for key in VARIABLES)
        additivity_pass = abs(additivity_residual) <= TOL
        independent_pass = independent_residual <= TOL
        additivity_all = additivity_all and additivity_pass
        independent_all = independent_all and independent_pass
        for item in subset_rows:
            shapley_terms.append({"scenario_id": scene_id, **item})
        for item in permutation_rows:
            permutation_terms.append({"scenario_id": scene_id, **item})
        p_map = cube["000"]
        actual = {key: p_map - validation[key] for key in VARIABLES}
        classification = classify_scene(subset_scores, actual)
        for variable in VARIABLES:
            shapley_rows.append({
                "scenario_id": scene_id, "variable": variable, "shapley_signed_degradation": subset_scores[variable],
                "permutation_recalculation": permutation_scores[variable],
                "cross_calculation_difference": subset_scores[variable] - permutation_scores[variable],
                "additivity_residual_scene": additivity_residual,
            })
            actual_rows.append({
                "scenario_id": scene_id, "variable": variable, "normal_map50": p_map,
                "validation_map50": validation[variable], "actual_signed_absolute_map_degradation": actual[variable],
                "actual_signed_relative_degradation": actual[variable] / p_map if abs(p_map) > TOL else None,
            })
            rank_rows.append({
                "scenario_id": scene_id, "variable": variable,
                "shapley_score": subset_scores[variable], "actual_effect": actual[variable],
                "shapley_rank": average_descending_ranks(subset_scores)[variable],
                "actual_rank": average_descending_ranks(actual, 0.01)[variable],
                "direction_match": sign(subset_scores[variable]) == sign(actual[variable]),
                "scene_status": classification["scene_status"],
            })
        scene_rows.append({
            "scenario_id": scene_id, **classification,
            "normal_map50": p_map, "q_all_map50": cube["111"], "total_degradation": total_degradation,
            "additivity_residual": additivity_residual, "additivity_pass": additivity_pass,
            "independent_recalculation_max_difference": independent_residual,
            "independent_recalculation_pass": independent_pass,
        })
        for row in rows:
            metrics = row["metrics"]
            condition_rows.append({
                "condition_id": row["condition_id"], "scenario_id": scene_id,
                "role": row["role"], "cube_mask": row["cube_mask"], "validation_variable": row["validation_variable"],
                **row["environment"], "person_ap50": metrics["person_ap50"], "vehicle_ap50": metrics["vehicle_ap50"],
                "map50": metrics["map50"], "verdict": row["verdict"], "frame_count": row["frame_count"],
                "render_seconds": row["timings"].get("render_seconds"),
                "inference_seconds": row["timings"].get("inference_seconds"),
                "scoring_seconds": row["timings"].get("scoring_seconds"),
                "reused_after_verified_resume": row.get("reused", False),
                "result_path": str((RESULTS / f"{row['condition_id']}.json").resolve()),
            })

    protected_rows, protected_ok = protected_snapshot(OUTPUT / "protected_hashes_after.csv")
    technical_complete = additivity_all and independent_all and protected_ok
    overall = aggregate_status(scene_rows, technical_complete)
    counts = {name: sum(row["scene_status"] == name for row in scene_rows) for name in ("MATCH", "MISMATCH", "INCONCLUSIVE")}
    write_csv_atomic(OUTPUT / "condition_results.csv", condition_rows, list(condition_rows[0]))
    write_csv_atomic(OUTPUT / "shapley_terms.csv", shapley_terms, list(shapley_terms[0]))
    write_csv_atomic(OUTPUT / "shapley_permutation_terms.csv", permutation_terms, list(permutation_terms[0]))
    write_csv_atomic(OUTPUT / "shapley_scores.csv", shapley_rows, list(shapley_rows[0]))
    write_csv_atomic(OUTPUT / "actual_effects.csv", actual_rows, list(actual_rows[0]))
    write_csv_atomic(OUTPUT / "rank_comparison.csv", rank_rows, list(rank_rows[0]))
    write_csv_atomic(OUTPUT / "scene_summary.csv", scene_rows, list(scene_rows[0]))
    gate_result = {
        "schema_version": "1.0", "completed_at_utc": utc_now(), "overall_status": overall,
        "technical_complete": technical_complete, "completed_conditions": len(completed),
        "completed_manifests": len(list(MANIFESTS.glob("*.json"))), "scene_counts": counts,
        "match_threshold": 7, "all_additivity_pass": additivity_all,
        "all_independent_recalculations_pass": independent_all,
        "protected_file_count": len(protected_rows), "protected_mismatches": sum(not row["matches"] for row in protected_rows),
        "RF_or_other_surrogate_used": False, "virtual_map_used": False,
        "boundary_search_executed": False,
    }
    write_json_atomic(OUTPUT / "gate_results.json", gate_result)
    write_json_atomic(OUTPUT / "protection_integrity_result.json", {
        "protected_file_count": len(protected_rows), "mismatch_count": sum(not row["matches"] for row in protected_rows),
        "all_unchanged": protected_ok,
    })
    report = (
        "# 직접 Shapley 변수 선택 독립 검증 결과\n\n"
        f"- 최종 판정: `{overall}`\n"
        f"- 장면 판정: MATCH {counts['MATCH']}, MISMATCH {counts['MISMATCH']}, INCONCLUSIVE {counts['INCONCLUSIVE']}\n"
        f"- 기술 완료: {len(completed)}/110조건, 정확 Shapley 가산성 및 독립 순열 재계산: {'통과' if additivity_all and independent_all else '실패'}\n"
        f"- 보호 파일 불일치: {sum(not row['matches'] for row in protected_rows)}건\n"
        "- RF·대리모델·가상 mAP는 사용하지 않았고, 경계 탐색 효율은 이 실행의 검증 대상이 아니다.\n"
    )
    (OUTPUT / "independent_validation_report_ko.md").write_text(report, encoding="utf-8")


def write_artifact_manifest() -> None:
    excluded = {"artifact_manifest.csv", "python_stdout.log", "python_stderr.log", "stage_events.jsonl"}
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in excluded and ".tmp" not in path.name:
            rows.append({"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    write_csv_atomic(OUTPUT / "artifact_manifest.csv", rows, ["path", "sha256", "size_bytes"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="reuse only cryptographically verified condition completions")
    args = parser.parse_args()
    service = None
    try:
        append_event("preflight", "started", timeout_seconds=120, resume=args.resume)
        settings, _, conditions = verify_preflight(args.resume)
        append_event("preflight", "completed")

        append_event("module_import", "started", timeout_seconds=120)
        if str(REPOSITORY) not in sys.path:
            sys.path.insert(0, str(REPOSITORY))
        from experiments.yolov8_search_comparison.shap_counterfactual_boundary_v3 import evaluation_core as ev
        append_event("module_import", "completed", module_path=str(Path(ev.__file__).resolve()))

        RESULTS.mkdir(parents=True, exist_ok=True)
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        service = ev.EvaluationService(
            "direct_exact_shapley_independent_variable_validation_v1",
            cache_root=CACHE,
            force_render_new=True,
        )
        service.scenarios = {row["scenario_id"]: row for row in settings["scenes"]}
        install_instrumentation(service, ev)
        completed: list[dict[str, Any]] = []
        milestones = {22, 44, 66, 88, 110}
        for condition in conditions:
            previous = validate_completed(condition, ev, service) if args.resume else None
            if previous is not None:
                previous = dict(previous)
                previous["reused"] = True
                completed.append(previous)
                update_registry(completed)
                continue
            append_event("condition", "started", condition_id=condition["condition_id"], timeout_seconds=420)
            env = ev.Environment.from_dict({key: condition[key] for key in ENV_KEYS})
            evaluation = service.evaluate(
                condition["scenario_id"], env, 42,
                logical=False, forbid_inference_cache=True,
            )
            result = persist_result(condition, evaluation)
            completed.append(result)
            update_registry(completed)
            append_event("condition", "completed", condition_id=condition["condition_id"])
            if len(completed) in milestones:
                print(f"[progress] completed={len(completed)}/110 ({len(completed) // 11}/10 scenes)", flush=True)

        append_event("finalization", "started", timeout_seconds=300)
        finalize(completed)
        write_artifact_manifest()
        append_event("finalization", "completed")
        return 0
    except Exception as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        append_event("worker", "failed", error=repr(exc))
        failure = {
            "schema_version": "1.0", "failed_at_utc": utc_now(), "status": "TECHNICAL_INCOMPLETE",
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
            "scientific_gate_calculated": False, "automatic_retry": False,
            "RF_or_other_surrogate_used": False, "virtual_map_used": False,
        }
        path = OUTPUT / f"technical_failure_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        write_json_new(path, failure)
        try:
            protected_snapshot(OUTPUT / "protected_hashes_failure_after.csv")
        except Exception:
            pass
        return 1
    finally:
        if service is not None:
            append_event("matlab_shutdown", "started", timeout_seconds=60)
            service.close()
            append_event("matlab_shutdown", "completed")


if __name__ == "__main__":
    raise SystemExit(main())
