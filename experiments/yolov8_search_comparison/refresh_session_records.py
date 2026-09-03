"""Backfill display fields from immutable cached evaluations without rerunning inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report_results import _paper_text, _table_rows, _write_csv
from .run_comparison import HERE, write_iteration_csv, write_json
from .validate_results import validate_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--cache-namespace", default="cache")
    args = parser.parse_args()
    root = args.output_root.resolve()
    aggregate = root / "aggregated" / args.session_id
    config = json.loads((aggregate / "config_snapshot.json").read_text(encoding="utf-8"))
    results = {}
    all_rows = []
    for method in config["search_methods"]:
        run_dir = root / "raw" / method / args.session_id
        records = json.loads((run_dir / "iterations.json").read_text(encoding="utf-8"))
        for record in records:
            evaluation_path = (
                root / "raw" / "_shared_evaluations" / args.cache_namespace
                / record["cache_key"] / "evaluation.json"
            )
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            person = evaluation["metrics"]["per_class"]["person"]
            vehicle = evaluation["metrics"]["per_class"]["vehicle"]
            record.update(
                {
                    "person_false_positive_count": person["fp"],
                    "person_false_negative_count": person["fn"],
                    "vehicle_false_positive_count": vehicle["fp"],
                    "vehicle_false_negative_count": vehicle["fn"],
                    "total_true_positive_count": person["tp"] + vehicle["tp"],
                    "total_false_positive_count": person["fp"] + vehicle["fp"],
                    "total_false_negative_count": person["fn"] + vehicle["fn"],
                }
            )
        write_json(run_dir / "iterations.json", records)
        write_iteration_csv(run_dir / "iterations.csv", records)
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        results[method] = {"records": records, "summary": summary, "run_dir": str(run_dir)}
        rows = _table_rows(records)
        _write_csv(aggregate / f"{method}_iterations.csv", rows)
        all_rows.extend({"search_method": method, **row} for row in rows)
    _write_csv(aggregate / "all_iteration_results.csv", all_rows)
    (aggregate / "paper_results_ko.md").write_text(_paper_text(config, results), encoding="utf-8")
    validate_outputs(config, results, root, aggregate)
    print(json.dumps({"session_id": args.session_id, "methods": list(results), "updated": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
