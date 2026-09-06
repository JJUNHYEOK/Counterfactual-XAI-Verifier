from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.io import loadmat


REPOSITORY = Path(r"C:\Users\lab\Counterfactual-XAI-Verifier")
WORKTREE = REPOSITORY / ".k"
DIRECT = WORKTREE / "final_kci_direct_yolo_shap_v1"
PREPARATION = DIRECT / "independent_shap_variable_validation_v1_preparation"
OUTPUT = DIRECT / "independent_shap_variable_validation_run_001_lab"
WORKER_PATH = PREPARATION / "independent_validation_worker.py"
SUPERVISOR_PATH = PREPARATION / "run_independent_validation_managed.py"
LAUNCHER_PATH = PREPARATION / "run_independent_validation_lab.ps1"
CONDITIONS_PATH = PREPARATION / "independent_conditions.json"
SCENARIOS_PATH = PREPARATION / "independent_scenarios.csv"
SETTINGS_PATH = PREPARATION / "fixed_settings.json"
GATE_PATH = PREPARATION / "locked_gate_definition.json"
SOURCE_MANIFEST = PREPARATION / "source_manifest.csv"
PROTECTED_BASELINE = PREPARATION / "protected_hashes_before.csv"
VARIABLES = ("fog", "illumination", "noise")
ENV_KEYS = ("fog_percent", "illumination_lux", "camera_noise")
P = {"fog_percent": 5.0, "illumination_lux": 12000.0, "camera_noise": 0.02}
Q = {"fog_percent": 76.0, "illumination_lux": 2450.0, "camera_noise": 0.49}
VALIDATION = {
    "fog": {"fog_percent": 70.0, "illumination_lux": 12000.0, "camera_noise": 0.02},
    "illumination": {"fog_percent": 5.0, "illumination_lux": 4640.0, "camera_noise": 0.02},
    "noise": {"fog_percent": 5.0, "illumination_lux": 12000.0, "camera_noise": 0.42},
}
FORBIDDEN_IMPORT_ROOTS = {"shap", "sklearn", "xgboost", "lightgbm", "catboost", "tensorflow", "keras", "joblib"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def next_attempt_path(base_name: str) -> Path:
    base = PREPARATION / base_name
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    number = 2
    while (PREPARATION / f"{stem}_attempt_{number:03d}{suffix}").exists():
        number += 1
    return PREPARATION / f"{stem}_attempt_{number:03d}{suffix}"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(record: list[dict[str, Any]], name: str, passed: bool, evidence: Any) -> None:
    record.append({"check": name, "passed": bool(passed), "evidence": evidence})


def imports_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            values.append(node.module)
    return sorted(set(values))


def expected_cube(mask: str) -> dict[str, float]:
    return {
        "fog_percent": Q["fog_percent"] if mask[0] == "1" else P["fog_percent"],
        "illumination_lux": Q["illumination_lux"] if mask[1] == "1" else P["illumination_lux"],
        "camera_noise": Q["camera_noise"] if mask[2] == "1" else P["camera_noise"],
    }


def same_environment(row: dict[str, Any], expected: dict[str, float]) -> bool:
    return all(math.isclose(float(row[key]), expected[key], rel_tol=0.0, abs_tol=1e-12) for key in ENV_KEYS)


def main() -> None:
    checks: list[dict[str, Any]] = []
    mock: list[dict[str, Any]] = []
    if OUTPUT.exists():
        raise RuntimeError(f"result folder must remain absent during preparation: {OUTPUT}")

    scripts = [PREPARATION / "build_preparation.py", WORKER_PATH, SUPERVISOR_PATH, PREPARATION / "static_validate.py"]
    for path in scripts:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    check(checks, "python_compile", True, [str(path) for path in scripts])

    all_imports = {str(path): imports_in(path) for path in scripts}
    forbidden = {
        path: [name for name in names if name.split(".")[0].lower() in FORBIDDEN_IMPORT_ROOTS]
        for path, names in all_imports.items()
    }
    check(checks, "forbidden_imports_absent", not any(forbidden.values()), forbidden)
    worker_tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
    top_imports = []
    for node in worker_tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.append(node.module)
    check(checks, "worker_import_has_no_scientific_execution_module", not any("evaluation_core" in name or name.startswith("matlab") for name in top_imports), top_imports)

    launcher_escaped = str(LAUNCHER_PATH).replace("'", "''")
    ps_command = (
        "$tokens=$null;$errors=$null;"
        f"[System.Management.Automation.Language.Parser]::ParseFile('{launcher_escaped}',[ref]$tokens,[ref]$errors)|Out-Null;"
        "if($errors.Count -gt 0){$errors|ForEach-Object{$_.Message};exit 1};exit 0"
    )
    ps_result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps_command], capture_output=True, text=True, timeout=30)
    check(checks, "powershell_syntax", ps_result.returncode == 0, {"returncode": ps_result.returncode, "stderr": ps_result.stderr.strip(), "stdout": ps_result.stdout.strip()})

    settings = read_json(SETTINGS_PATH)
    gate = read_json(GATE_PATH)
    conditions = read_json(CONDITIONS_PATH)["records"]
    scenario_rows = read_csv(SCENARIOS_PATH)
    check(checks, "fixed_p", settings["P"] == P, settings["P"])
    check(checks, "fixed_q", settings["Q"] == Q, settings["Q"])
    check(checks, "no_rf_surrogate_virtual_map", settings.get("RF_or_surrogate_or_virtual_map_used") is False, settings.get("RF_or_surrogate_or_virtual_map_used"))
    check(checks, "scenario_count_and_ids", len(scenario_rows) == 10 and {row["scenario_id"] for row in scenario_rows} == {f"IVS{i:02d}" for i in range(10)}, [row["scenario_id"] for row in scenario_rows])
    excluded = {"DTR0", "F02", "F09", *{f"H{i:02d}" for i in range(10)}, *{f"G{i:02d}" for i in range(5)}}
    check(checks, "excluded_scenes_not_used", not ({row["scenario_id"] for row in scenario_rows} & excluded), sorted({row["scenario_id"] for row in scenario_rows} & excluded))
    check(checks, "result_blind_and_unique_scenes", all(row["selection_basis"] == "result-blind metadata only" and row["prior_scenario_id_found"] == "False" and row["exact_structure_duplicate_found"] == "False" for row in scenario_rows), len(scenario_rows))

    geometry_path = REPOSITORY / "experiments" / "yolov8_search_comparison" / "geometry_reference_v1.mat"
    geometry = loadmat(geometry_path)["geometry"]
    terrain_x = geometry["terrain_x"][0, 0]
    terrain_y = geometry["terrain_y"][0, 0]
    terrain_bounds = {
        "x_min": float(terrain_x.min()), "x_max": float(terrain_x.max()),
        "y_min": float(terrain_y.min()), "y_max": float(terrain_y.max()),
        "path": str(geometry_path), "sha256": sha256(geometry_path),
    }
    scenario_details = []
    structure_hashes = []
    for row in scenario_rows:
        path = Path(row["scenario_config_path"])
        config = read_json(path)
        objects = config["objects"]
        coords = [item["xy"] for item in objects]
        radii = [item["radius_height"][0] for item in objects]
        heights = [item["radius_height"][1] for item in objects]
        trajectory_points = [config["trajectory"]["start_xyz"], config["trajectory"]["end_xyz"]]
        within_prior_observed_envelope = (
            all(-54 <= point[0] <= 18 and -20 <= point[1] <= 19 and 44 <= point[2] <= 76 for point in trajectory_points)
            and all(-22 <= xy[0] <= 82.3 and -14 <= xy[1] <= 15 for xy in coords)
            and all(0.3864 <= value <= 1.548 for value in radii)
            and all(1.218 <= value <= 2.232 for value in heights)
            and 45 <= config["camera"]["intrinsics"][4] <= 70
            and 0.55 <= config["person_motion"]["radius_m"] <= 2.1
            and 0.58 <= config["person_motion"]["omega_rad_s"] <= 1.25
        )
        within_renderer_geometry = (
            all(terrain_bounds["x_min"] <= point[0] <= terrain_bounds["x_max"] and terrain_bounds["y_min"] <= point[1] <= terrain_bounds["y_max"] and 0 < point[2] <= 100 for point in trajectory_points)
            and all(terrain_bounds["x_min"] <= xy[0] <= terrain_bounds["x_max"] and terrain_bounds["y_min"] <= xy[1] <= terrain_bounds["y_max"] for xy in coords)
            and all(value > 0 for value in radii + heights)
            and 0 < config["camera"]["intrinsics"][4] < 90
            and config["person_motion"]["radius_m"] > 0
            and config["person_motion"]["omega_rad_s"] > 0
        )
        valid_objects = (
            len(objects) == 6
            and [item["object_id"] for item in objects] == list(range(1, 7))
            and sum(item["class_id"] == 1 and item["class_name"] == "person" for item in objects) == 3
            and sum(item["class_id"] == 2 and item["class_name"] == "vehicle" for item in objects) == 3
        )
        file_hash_ok = sha256(path) == row["scenario_config_sha256"]
        structure_hashes.append(row["structure_sha256"])
        scenario_details.append({"scenario_id": row["scenario_id"], "within_renderer_geometry": within_renderer_geometry, "within_prior_observed_metadata_envelope_auxiliary": within_prior_observed_envelope, "object_schema_valid": valid_objects, "file_hash_ok": file_hash_ok})
    scenario_gate = all(row["within_renderer_geometry"] and row["object_schema_valid"] and row["file_hash_ok"] for row in scenario_details)
    check(checks, "scenario_static_schema_geometry_and_hash", scenario_gate, {"terrain_bounds": terrain_bounds, "scenarios": scenario_details, "note": "prior observed metadata envelope is auxiliary, not a renderer hard bound"})
    check(checks, "scenario_structures_unique", len(set(structure_hashes)) == 10, structure_hashes)

    check(checks, "condition_count_unique", len(conditions) == 110 and len({row["condition_id"] for row in conditions}) == 110, len(conditions))
    condition_details = []
    for scene_id in sorted({row["scenario_id"] for row in conditions}):
        own = [row for row in conditions if row["scenario_id"] == scene_id]
        cube = [row for row in own if row["role"] == "exact_shapley_cube"]
        validation = [row for row in own if row["role"] == "separate_actual_rank_validation"]
        masks = {row["cube_mask"] for row in cube}
        cube_ok = len(cube) == 8 and masks == {f"{value:03b}" for value in range(8)} and all(same_environment(row, expected_cube(row["cube_mask"])) for row in cube)
        validation_ok = len(validation) == 3 and {row["validation_variable"] for row in validation} == set(VARIABLES) and all(same_environment(row, VALIDATION[row["validation_variable"]]) for row in validation)
        unique_env = len({tuple(float(row[key]) for key in ENV_KEYS) for row in own}) == 11
        condition_details.append({"scenario_id": scene_id, "count": len(own), "cube_ok": cube_ok, "validation_ok": validation_ok, "unique_environment_count": unique_env})
    check(checks, "condition_roles_values_and_uniqueness", all(row["count"] == 11 and row["cube_ok"] and row["validation_ok"] and row["unique_environment_count"] for row in condition_details), condition_details)
    check(checks, "no_reuse_candidates", sum(bool(row["reuse_candidate"]) for row in conditions) == 0, sum(bool(row["reuse_candidate"]) for row in conditions))
    check(checks, "logical_frame_count", len(conditions) * 181 == 19910, len(conditions) * 181)

    worker = load_module("independent_validation_worker_for_static_test", WORKER_PATH)
    additive_values = {}
    for value in range(8):
        mask = f"{value:03b}"
        fog, illumination, noise = (int(bit) for bit in mask)
        degradation = 0.10 * fog + 0.05 * illumination + 0.03 * noise + 0.02 * fog * noise
        additive_values[mask] = 0.80 - degradation
    subset, subset_terms = worker.exact_shapley_subset(additive_values)
    permutations, permutation_terms = worker.exact_shapley_permutations(additive_values)
    expected_scores = {"fog": 0.11, "illumination": 0.05, "noise": 0.04}
    shap_ok = all(math.isclose(subset[key], expected_scores[key], rel_tol=0.0, abs_tol=1e-12) and math.isclose(subset[key], permutations[key], rel_tol=0.0, abs_tol=1e-12) for key in VARIABLES)
    shap_ok = shap_ok and math.isclose(sum(subset.values()), additive_values["000"] - additive_values["111"], rel_tol=0.0, abs_tol=1e-12)
    mock.append({"test": "exact_shapley_two_independent_algorithms_and_additivity", "passed": shap_ok, "scores": subset, "expected": expected_scores, "subset_terms": len(subset_terms), "permutation_terms": len(permutation_terms)})

    match_case = worker.classify_scene({"fog": 0.2, "illumination": 0.05, "noise": 0.1}, {"fog": 0.3, "illumination": 0.1, "noise": 0.15})
    tie_case = worker.classify_scene({"fog": 0.2, "illumination": 0.05, "noise": 0.1}, {"fog": 0.20, "illumination": 0.195, "noise": 0.1})
    mismatch_case = worker.classify_scene({"fog": 0.2, "illumination": 0.05, "noise": 0.1}, {"fog": 0.1, "illumination": 0.25, "noise": 0.15})
    shap_tie_case = worker.classify_scene({"fog": 0.2, "illumination": 0.05, "noise": 0.2}, {"fog": 0.3, "illumination": 0.1, "noise": 0.15})
    scene_gate_ok = [match_case["scene_status"], tie_case["scene_status"], mismatch_case["scene_status"], shap_tie_case["scene_status"]] == ["MATCH", "INCONCLUSIVE", "MISMATCH", "INCONCLUSIVE"]
    mock.append({"test": "scene_gate_statuses", "passed": scene_gate_ok, "statuses": [match_case["scene_status"], tie_case["scene_status"], mismatch_case["scene_status"], shap_tie_case["scene_status"]]})
    overall_cases = {
        "pass": worker.aggregate_status([{"scene_status": "MATCH"}] * 7 + [{"scene_status": "MISMATCH"}] * 3, True),
        "fail": worker.aggregate_status([{"scene_status": "MATCH"}] * 6 + [{"scene_status": "MISMATCH"}] + [{"scene_status": "INCONCLUSIVE"}] * 3, True),
        "inconclusive": worker.aggregate_status([{"scene_status": "MATCH"}] * 6 + [{"scene_status": "INCONCLUSIVE"}] * 4, True),
        "technical": worker.aggregate_status([{"scene_status": "MATCH"}] * 10, False),
    }
    expected_overall = {
        "pass": "INDEPENDENT_SHAP_VARIABLE_SELECTION_PASS",
        "fail": "INDEPENDENT_SHAP_VARIABLE_SELECTION_FAIL",
        "inconclusive": "INDEPENDENT_SHAP_VARIABLE_SELECTION_INCONCLUSIVE",
        "technical": "TECHNICAL_INCOMPLETE",
    }
    mock.append({"test": "overall_gate_statuses", "passed": overall_cases == expected_overall, "actual": overall_cases, "expected": expected_overall})

    supervisor = load_module("independent_validation_supervisor_for_static_test", SUPERVISOR_PATH)
    worker_command = [str(supervisor.PYTHON), str(supervisor.WORKER)]
    base_identity = {"pid": 1234, "create_time": 1000.0, "executable": str(supervisor.PYTHON), "command_line": worker_command}
    safety_cases = [
        ("direct_worker", True, base_identity, dict(base_identity), True),
        ("other_repository_python", False, base_identity, {**base_identity, "command_line": [str(supervisor.PYTHON), r"C:\OtherRepo\job.py"]}, False),
        ("new_unowned_python", False, base_identity, dict(base_identity), False),
        ("unowned_matlab", False, base_identity, {**base_identity, "executable": r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"}, False),
        ("same_pid_different_creation", True, base_identity, {**base_identity, "create_time": 2000.0}, False),
        ("insufficient_identity", True, base_identity, {"pid": 1234}, False),
    ]
    safety_results = []
    for name, owned, launch, observed, expected in safety_cases:
        eligible, reason = supervisor.mock_cleanup_eligibility(direct_handle_owned=owned, launch=launch, observed=observed)
        safety_results.append({"case": name, "eligible": eligible, "expected": expected, "passed": eligible == expected, "reason": reason, "real_process_termination_performed": False})
    mock.append({"test": "supervisor_six_case_safety", "passed": all(row["passed"] for row in safety_results), "cases": safety_results})
    check(checks, "mock_tests", all(row["passed"] for row in mock), mock)

    source_rows = read_csv(SOURCE_MANIFEST)
    source_mismatches = []
    for row in source_rows:
        path = Path(row["path"])
        actual = sha256(path) if path.is_file() else ""
        if actual != row["sha256"].upper():
            source_mismatches.append(str(path))
    check(checks, "source_manifest_hashes", not source_mismatches, {"records": len(source_rows), "mismatches": source_mismatches})
    protected = read_csv(PROTECTED_BASELINE)
    protected_mismatches = []
    for row in protected:
        path = Path(row["path"])
        actual = sha256(path) if path.is_file() else ""
        if actual != row["expected_sha256"].upper():
            protected_mismatches.append(str(path))
    check(checks, "protected_129_unchanged", len(protected) == 129 and not protected_mismatches, {"records": len(protected), "mismatches": protected_mismatches})
    check(checks, "locked_gate_thresholds", gate["scene_count"] == 10 and gate["conditions_per_scene"] == 11 and gate["scene_rule"]["actual_tie_tolerance_map"] == 0.01, gate)
    check(checks, "output_absent_and_no_execution", not OUTPUT.exists(), {"output_exists": OUTPUT.exists(), "matlab_or_yolo_invoked_by_validator": False})

    status = "STATIC_VALIDATION_PASS" if all(item["passed"] for item in checks) else "STATIC_VALIDATION_FAIL"
    static_result = {
        "schema_version": "1.0", "validated_at_utc": utc_now(), "status": status,
        "checks": checks, "matlab_started": False, "rendering_started": False,
        "yolo_started": False, "actual_shapley_started": False,
    }
    mock_result = {
        "schema_version": "1.0", "validated_at_utc": utc_now(),
        "status": "MOCK_VALIDATION_PASS" if all(row["passed"] for row in mock) else "MOCK_VALIDATION_FAIL",
        "tests": mock, "real_process_termination_performed": False,
        "matlab_started": False, "rendering_started": False, "yolo_started": False,
    }
    write_json_new(next_attempt_path("mock_validation.json"), mock_result)
    write_json_new(next_attempt_path("static_validation.json"), static_result)
    print(json.dumps({"status": status, "checks": len(checks), "mock_status": mock_result["status"]}, ensure_ascii=False, indent=2))
    if status != "STATIC_VALIDATION_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
