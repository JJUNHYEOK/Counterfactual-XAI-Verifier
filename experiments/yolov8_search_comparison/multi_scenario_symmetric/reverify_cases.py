"""Create and independently re-run reproducibility cases for every boundary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..run_comparison import YoloFrameDetector, file_sha256, seed_everything
from ..search_core import Environment, classify_verdict
from .run_experiment import (
    AGGREGATED_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_PLAN,
    HERE,
    REPO_ROOT,
    ScenarioEvaluationRepository,
    ScenarioMatlabExporter,
    read_json,
    write_csv,
    write_json,
)


TEST_ROOT = HERE / "test_suites"


def build_cases(session_id: str, combined: dict[str, Any], plan: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = {item["scenario_id"]: item for item in plan["scenarios"]}
    cases = []
    for result in combined["scenarios"]:
        if not result["search_success"]:
            continue
        scenario_id = result["scenario_id"]
        values = [
            ("initial_pass", config["initial_environment"], result["initial_map50"], result["initial_verdict"]),
            ("final_nonfail", result["final_nonfail_anchor"], result["final_nonfail_anchor"]["map50"], result["final_nonfail_anchor"]["verdict"]),
            ("final_fail", result["final_fail_anchor"], result["final_fail_anchor"]["map50"], result["final_fail_anchor"]["verdict"]),
        ]
        for role, environment, score, verdict in values:
            case_id = f"{scenario_id}_{role}"
            cases.append(
                {
                    "case_id": case_id,
                    "case_role": role,
                    "original_scenario_id": scenario_id,
                    "environment": {key: float(environment[key]) for key in ("fog_percent", "illumination_lux", "camera_noise")},
                    "scenario_config_sha256": scenarios[scenario_id]["scenario_config_sha256"],
                    "weights_sha256": config["expected_weights_sha256"],
                    "gt_version": config["gt_version"],
                    "expected_verdict": verdict,
                    "original_map50": float(score),
                    "rerun_command": (
                        ".venv\\Scripts\\python.exe -m "
                        "experiments.yolov8_search_comparison.multi_scenario_symmetric.reverify_cases "
                        f"--session {session_id} --case {case_id}"
                    ),
                }
            )
    return cases


def metadata(config: dict[str, Any], weights_path: Path) -> tuple[YoloFrameDetector, dict[str, Any]]:
    actual = file_sha256(weights_path)
    if actual != config["expected_weights_sha256"]:
        raise RuntimeError(f"Weights SHA-256 mismatch before revalidation: {actual}")
    detector = YoloFrameDetector(weights_path, config["yolov8"])
    info = detector.metadata(weights_path, actual)
    names = {str(key): value for key, value in info["model_class_names"].items()}
    if names != config["expected_model_class_names"]:
        raise RuntimeError(f"Unexpected model classes during revalidation: {names}")
    return detector, info


def execute_cases(
    *,
    session_id: str,
    cases: list[dict[str, Any]],
    scenarios: dict[str, dict[str, Any]],
    config: dict[str, Any],
    target_root: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    weights_path = (REPO_ROOT / config["weights_path"]).resolve()
    detector, model_metadata = metadata(config, weights_path)
    exporter = ScenarioMatlabExporter()
    results = []
    try:
        for case in cases:
            case_root = target_root / "runs" / case["case_id"]
            result_path = case_root / "case_result.json"
            if resume and result_path.exists():
                results.append(read_json(result_path))
                continue
            repository = ScenarioEvaluationRepository(
                config=config,
                scenario=scenarios[case["original_scenario_id"]],
                cache_root=case_root / "independent_evaluation_store",
                exporter=exporter,
                detector=detector,
                model_metadata=model_metadata,
                weights_path=weights_path,
            )
            evaluation, cache_hit = repository.evaluate(Environment.from_dict(case["environment"]))
            score = float(evaluation["metrics"]["map50"])
            verdict = classify_verdict(score, config["thresholds"]["pass_map50"], config["thresholds"]["fail_map50"])
            result = {
                **case,
                "revalidation_map50": score,
                "map50_signed_difference": score - float(case["original_map50"]),
                "map50_absolute_difference": abs(score - float(case["original_map50"])),
                "revalidation_verdict": verdict,
                "verdict_matches": verdict == case["expected_verdict"],
                "fresh_independent_store": True,
                "evaluation_store_used": cache_hit,
                "evaluation_cache_key": evaluation["cache_key"],
                "timings": evaluation["timings"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            case_root.mkdir(parents=True, exist_ok=True)
            write_json(result_path, result)
            results.append(result)
            print(json.dumps({"case": case["case_id"], "map50": score, "verdict": verdict, "matches": result["verdict_matches"]}, ensure_ascii=False), flush=True)
    finally:
        exporter.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--case")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    config = read_json(args.config.resolve())
    plan = read_json(args.plan.resolve())
    combined = read_json(AGGREGATED_ROOT / args.session / "multi_scenario_results.json")
    scenarios = {item["scenario_id"]: item for item in plan["scenarios"]}
    cases = build_cases(args.session, combined, plan, config)
    seed_everything(42)

    if args.case:
        selected = [case for case in cases if case["case_id"] == args.case]
        if not selected:
            raise KeyError(f"Unknown case ID: {args.case}")
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        target = TEST_ROOT / args.session / "manual_replays" / f"{args.case}__{stamp}"
        target.mkdir(parents=True, exist_ok=False)
        results = execute_cases(session_id=args.session, cases=selected, scenarios=scenarios, config=config, target_root=target, resume=False)
        write_json(target / "suite_summary.json", {"cases": results, "all_verdicts_match": all(item["verdict_matches"] for item in results)})
        return 0

    target = TEST_ROOT / args.session
    if target.exists() and not args.resume:
        raise FileExistsError(f"Test suite already exists; use --resume: {target}")
    target.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = target / "test_suite_manifest.json"
    if not manifest_path.exists():
        write_json(
            manifest_path,
            {
                "schema_version": "1.0",
                "source_session_id": args.session,
                "created_before_revalidation": True,
                "case_count": len(cases),
                "cases": cases,
            },
        )
        write_csv(target / "test_suite_manifest.csv", cases)
    else:
        expected = read_json(manifest_path)["cases"]
        if expected != cases:
            raise RuntimeError("Existing test suite manifest differs from current source results")

    results = execute_cases(
        session_id=args.session,
        cases=cases,
        scenarios=scenarios,
        config=config,
        target_root=target,
        resume=args.resume,
    )
    summary = {
        "source_session_id": args.session,
        "case_count": len(results),
        "verdict_match_count": sum(item["verdict_matches"] for item in results),
        "all_verdicts_match": all(item["verdict_matches"] for item in results),
        "maximum_absolute_map50_difference": max(item["map50_absolute_difference"] for item in results),
        "mean_absolute_map50_difference": sum(item["map50_absolute_difference"] for item in results) / len(results),
        "cases": results,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = target / "suite_summary.json"
    if summary_path.exists():
        if not args.resume:
            raise FileExistsError(summary_path)
    else:
        write_json(summary_path, summary)
        write_csv(target / "suite_summary.csv", results)
    print(json.dumps({"test_suite": str(target.resolve()), "case_count": len(results), "all_verdicts_match": summary["all_verdicts_match"], "maximum_absolute_map50_difference": summary["maximum_absolute_map50_difference"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

