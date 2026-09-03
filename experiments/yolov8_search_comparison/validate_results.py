"""Validate raw JSON/CSV parity and the fairness invariants of one session."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .search_core import Environment, classify_verdict


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_outputs(
    config: dict[str, Any],
    results: dict[str, Any],
    output_root: Path,
    aggregate_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        _assert(condition, f"{name}: {detail}")
        checks.append({"name": name, "passed": True, "detail": detail})

    bounds = config["environment_bounds"]
    shared_evaluations: dict[tuple[Any, ...], tuple[str, float, float, float]] = {}
    for method, result in results.items():
        run_dir = Path(result["run_dir"])
        json_path = run_dir / "iterations.json"
        csv_path = run_dir / "iterations.csv"
        with json_path.open("r", encoding="utf-8") as handle:
            json_rows = json.load(handle)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        check(
            f"{method}_json_csv_row_count",
            len(json_rows) == len(csv_rows) == len(result["records"]),
            f"rows={len(json_rows)}",
        )
        for record in json_rows:
            expected = classify_verdict(
                float(record["map50"]),
                float(config["thresholds"]["pass_map50"]),
                float(config["thresholds"]["fail_map50"]),
            )
            check(
                f"{method}_iter_{record['iteration']}_verdict",
                record["verdict"] == expected,
                f"mAP50={record['map50']} verdict={record['verdict']}",
            )
            env = Environment.from_dict(record)
            for key, value in env.as_dict().items():
                check(
                    f"{method}_iter_{record['iteration']}_{key}_bounds",
                    float(bounds[key][0]) <= value <= float(bounds[key][1]),
                    f"{key}={value}",
                )
            identity = (
                record["scenario_id"],
                int(record["random_seed"]),
                env.fog_percent,
                env.illumination_lux,
                env.camera_noise,
            )
            signature = (
                record["cache_key"],
                float(record["person_ap50"]),
                float(record["vehicle_ap50"]),
                float(record["map50"]),
            )
            if identity in shared_evaluations:
                check(
                    f"shared_evaluation_{record['cache_key']}",
                    shared_evaluations[identity] == signature,
                    "identical scenario/environment/seed has identical cache key and AP values",
                )
            else:
                shared_evaluations[identity] = signature

        check(
            f"{method}_metadata_exists",
            (run_dir / "run_metadata.json").is_file(),
            str(run_dir / "run_metadata.json"),
        )

    if {"symmetric", "asymmetric"}.issubset(results):
        sym = results["symmetric"]["records"]
        asym = results["asymmetric"]["records"]
        common = min(len(sym), len(asym))
        for idx in range(common):
            if sym[idx]["first_fail_discovered"] or asym[idx]["first_fail_discovered"]:
                break
            check(
                f"common_pre_fail_environment_{idx + 1}",
                all(sym[idx][key] == asym[idx][key] for key in (
                    "fog_percent", "illumination_lux", "camera_noise"
                )),
                "both methods use the same pre-FAIL degradation path",
            )

    report = {
        "schema_version": "1.0",
        "passed": True,
        "check_count": len(checks),
        "checks": checks,
        "interpretation": (
            "Validation covers thresholds, configured bounds, JSON/CSV row parity, "
            "and identical detector results for identical scenario/environment/seed. "
            "Policy weight equations are covered by unit tests."
        ),
    }
    path = aggregate_dir / "validation_report.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
