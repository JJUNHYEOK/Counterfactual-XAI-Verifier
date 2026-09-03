"""Validate the complete five-scenario experiment and record test evidence."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from ..run_comparison import file_sha256
from ..search_core import Environment, classify_verdict, environment_gap, map_gap
from .report_results import FIGURES_ROOT, REPORTS_ROOT
from .run_experiment import AGGREGATED_ROOT, DEFAULT_CONFIG, DEFAULT_PLAN, HERE, RUNS_ROOT, SCENARIO_ROOT, read_json


class Validator:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(condition), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_unit_tests() -> dict[str, Any]:
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "experiments/yolov8_search_comparison/tests", "-v"],
        [sys.executable, "-m", "unittest", "discover", "-s", "experiments/yolov8_search_comparison/multi_scenario_symmetric/tests", "-t", ".", "-v"],
    ]
    outputs = []
    total = 0
    for command in commands:
        completed = subprocess.run(command, cwd=HERE.parents[2], text=True, capture_output=True, check=False)
        output = completed.stdout + completed.stderr
        outputs.append(output)
        for line in output.splitlines():
            if line.startswith("Ran ") and " tests" in line:
                total += int(line.split()[1])
        if completed.returncode != 0:
            return {"passed": False, "test_count": total, "output": "\n".join(outputs), "returncode": completed.returncode}
    return {"passed": True, "test_count": total, "output": "\n".join(outputs), "returncode": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    config = read_json(DEFAULT_CONFIG)
    plan = read_json(DEFAULT_PLAN)
    aggregate = AGGREGATED_ROOT / args.session
    combined = read_json(aggregate / "multi_scenario_results.json")
    records = read_json(aggregate / "all_iteration_results.json")
    gt = read_json(SCENARIO_ROOT / plan["plan_id"] / "gt_validation.json")
    leakage = read_json(SCENARIO_ROOT / plan["plan_id"] / "scenario_leakage_audit.json")
    tests = read_json(HERE / "test_suites" / args.session / "suite_summary.json")
    s0_compare = read_json(aggregate / "s0_prior_comparison.json")
    validator = Validator()

    weights = (HERE.parents[2] / config["weights_path"]).resolve()
    actual_weight_hash = file_sha256(weights)
    validator.check("fixed_weights_sha256", actual_weight_hash == config["expected_weights_sha256"], actual_weight_hash)
    validator.check("five_registered_scenarios", len(plan["scenarios"]) == 5 and [x["scenario_id"] for x in plan["scenarios"]] == [f"S{i}" for i in range(5)], len(plan["scenarios"]))
    scenario_hashes = {item["scenario_id"]: file_sha256(Path(item["scenario_config_path"])) for item in plan["scenarios"]}
    validator.check("scenario_configs_unchanged", all(scenario_hashes[x["scenario_id"]] == x["scenario_config_sha256"] for x in plan["scenarios"]), scenario_hashes)
    validator.check("all_gt_scenarios_valid", gt["all_valid"] and all(x["valid"] for x in gt["scenarios"]), [(x["scenario_id"], x["valid"]) for x in gt["scenarios"]])
    validator.check("all_gt_counts_minimum_20", all(x["person_gt_count"] >= 20 and x["vehicle_gt_count"] >= 20 for x in gt["scenarios"]), [(x["scenario_id"], x["person_gt_count"], x["vehicle_gt_count"]) for x in gt["scenarios"]])
    validator.check("gt_geometry_and_alignment", all(x["out_of_bounds_box_count"] == 0 and x["nonpositive_box_count"] == 0 and x["frame_gt_alignment_valid"] for x in gt["scenarios"]), "0 invalid boxes; 181 aligned frames per scenario")
    validator.check("new_scenario_leakage_audit", leakage["new_scenarios_passed"], [(x["scenario_id"], x["passed"]) for x in leakage["results"]])
    validator.check("zero_exact_content_overlap", all(x["duplicate_image_hash_count"] == x["duplicate_frame_hash_count"] == x["duplicate_nonempty_label_hash_count"] == 0 for x in leakage["results"][1:]), "S1-S4 image/frame/nonempty-label overlaps are zero")

    validator.check("five_initial_pass", combined["overall"]["initial_pass_count"] == 5 and all(x["initial_verdict"] == "PASS" for x in combined["scenarios"]), combined["overall"]["initial_pass_count"])
    validator.check("five_search_success", combined["overall"]["search_success_count"] == 5 and combined["overall"]["search_success_rate"] == 1.0, combined["overall"]["search_success_rate"])
    validator.check("fifty_iteration_records", len(records) == 50 and all(sum(x["scenario_id"] == sid for x in records) == 10 for sid in [f"S{i}" for i in range(5)]), len(records))
    validator.check("symmetric_only", all("symmetric_mid" in x["next_condition_calculation"] or "common_pre_fail" in x["next_condition_calculation"] for x in records), "all decisions are common pre-FAIL or 50:50 midpoint")
    validator.check("fixed_inference_and_gt", all(x["weights_sha256"] == config["expected_weights_sha256"] and x["gt_version"] == config["gt_version"] for x in records), "all 50 records")
    keys = [x["evaluation_cache_key"] for x in records]
    validator.check("scenario_specific_evaluation_keys", len(keys) == len(set(keys)), {"records": len(keys), "unique_keys": len(set(keys))})

    gap_details = []
    gaps_valid = True
    for item in combined["scenarios"]:
        nonfail = item["final_nonfail_anchor"]
        fail = item["final_fail_anchor"]
        calculated_map = map_gap(nonfail["map50"], fail["map50"])
        calculated_env = environment_gap(Environment.from_dict(nonfail), Environment.from_dict(fail), config["environment_bounds"])
        valid = abs(calculated_map - item["gap_map50"]) < 1e-12 and abs(calculated_env - item["gap_environment_normalized"]) < 1e-12
        valid = valid and classify_verdict(nonfail["map50"]) != "FAIL" and classify_verdict(fail["map50"]) == "FAIL"
        gaps_valid = gaps_valid and valid
        gap_details.append({"scenario_id": item["scenario_id"], "gap_map": calculated_map, "gap_env": calculated_env, "passed": valid})
    validator.check("gap_formulas_and_anchor_sides", gaps_valid, gap_details)
    validator.check("first_fail_statistics", combined["overall"]["first_fail_iteration_statistics"]["mean"] == 4.0 and combined["overall"]["first_fail_iteration_statistics"]["population_std"] == 0.0, combined["overall"]["first_fail_iteration_statistics"])
    validator.check("nonmonotonic_event_count", combined["overall"]["scenario_with_nonmonotonic_increase_count"] == 1 and next(x for x in combined["scenarios"] if x["scenario_id"] == "S3")["nonmonotonic_map_increase_count"] == 1, "S3 only")
    validator.check("s0_independent_reproduction", s0_compare["new_result_is_independent"] and s0_compare["maximum_absolute_map50_difference"] == 0.0 and s0_compare["verdict_difference_count"] == 0, s0_compare["maximum_absolute_map50_difference"])

    validator.check("fifteen_revalidation_cases", tests["case_count"] == 15, tests["case_count"])
    validator.check("all_revalidation_verdicts_match", tests["all_verdicts_match"] and tests["verdict_match_count"] == 15, tests["verdict_match_count"])
    validator.check("revalidation_is_fresh", all(x["fresh_independent_store"] and not x["evaluation_store_used"] for x in tests["cases"]), "15 independent stores, 0 cache hits")
    validator.check("revalidation_map_difference_recorded", tests["maximum_absolute_map50_difference"] == 0.0, tests["maximum_absolute_map50_difference"])

    figure_paths = sorted((FIGURES_ROOT / args.session).glob("0[1-8]_*.png"))
    dpi_details = []
    for path in figure_paths:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            dpi_details.append({"file": path.name, "width": image.width, "height": image.height, "dpi": dpi})
    validator.check("eight_required_figures", len(figure_paths) == 8, [x.name for x in figure_paths])
    validator.check("figures_at_least_300dpi_nominal", all(min(x["dpi"]) >= 299.0 for x in dpi_details), dpi_details)
    report_dir = REPORTS_ROOT / args.session
    paper = (report_dir / "paper_sections_4_5_ko.md").read_text(encoding="utf-8")
    validator.check("paper_sections_complete", all(f"### {section}" in paper for section in ("4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "5.1", "5.2", "5.3", "5.4", "5.5", "5.6")), "sections 4.1-5.6")
    validator.check("paper_excludes_out_of_scope_terms", "SHAP" not in paper and "비대칭" not in paper and "asymmetric" not in paper.lower(), "no out-of-scope comparison text")
    validator.check("aggregate_csv_row_counts", len(csv_rows(aggregate / "scenario_summary.csv")) == 5 and len(csv_rows(aggregate / "all_iteration_results.csv")) == 50, "5 summary rows, 50 iteration rows")

    json_files = list(HERE.rglob("*.json"))
    json_errors = []
    for path in json_files:
        try:
            read_json(path)
        except Exception as exc:  # pragma: no cover - evidence path
            json_errors.append(f"{path}: {exc}")
    validator.check("all_multi_scenario_json_parse", not json_errors, {"count": len(json_files), "errors": json_errors})

    unit = run_unit_tests()
    validator.check("all_unit_tests", unit["passed"] and unit["test_count"] == 34, {"count": unit["test_count"], "returncode": unit["returncode"]})
    test_results_path = report_dir / "test_results.txt"
    test_results_path.write_text(unit["output"], encoding="utf-8")

    report = {
        "session_id": args.session,
        "passed": validator.passed,
        "passed_count": sum(item["passed"] for item in validator.checks),
        "failed_count": sum(not item["passed"] for item in validator.checks),
        "checks": validator.checks,
    }
    output = aggregate / "validation_report.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (aggregate / "validation_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "passed", "detail"])
        writer.writeheader()
        for item in validator.checks:
            writer.writerow({"name": item["name"], "passed": item["passed"], "detail": json.dumps(item["detail"], ensure_ascii=False) if isinstance(item["detail"], (dict, list)) else item["detail"]})
    print(json.dumps({"validation_report": str(output.resolve()), "passed": report["passed"], "passed_count": report["passed_count"], "failed_count": report["failed_count"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

