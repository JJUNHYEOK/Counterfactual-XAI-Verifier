from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from xai import generate_counterfactual_and_boundary, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SHAP-guided simulation-grounded counterfactual attribution for XAI outputs.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/counterfactual_case_input.json",
        help="Path to XAI input JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_counterfactual",
        help="Directory where output JSON files will be written.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "simulation_grounded", "kernelshap", "map50_proxy", "sim_dummy"],
        default="auto",
        help="Compatibility option. Analysis uses simulation-grounded KernelSHAP.",
    )
    parser.add_argument(
        "--target_status",
        type=str,
        choices=["PASS", "FAIL", "pass", "fail"],
        default=None,
        help="Optional target status override for legacy compatibility.",
    )
    parser.add_argument(
        "--num_counterfactuals",
        type=int,
        default=3,
        help="Number of legacy counterfactual candidate rows to emit.",
    )
    parser.add_argument(
        "--num_boundaries",
        type=int,
        default=5,
        help="Number of boundary candidate rows to emit.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Compatibility argument; analysis is deterministic from provided evidence/callable.",
    )
    parser.add_argument(
        "--shap_json_path",
        type=str,
        default=None,
        help="Deprecated compatibility option. Ignored.",
    )
    parser.add_argument(
        "--shap_case_id",
        type=str,
        default=None,
        help="Deprecated compatibility option. Ignored.",
    )
    return parser.parse_args()


def _resolve_step_filename(scene_id: str) -> str:
    match = re.search(r"(\d+)", scene_id)
    if match:
        return f"xai_signals_step_{match.group(1)}.json"
    return "xai_signals_step_latest.json"


def _build_xai_signals_payload(counterfactual_output: dict[str, Any]) -> dict[str, Any]:
    top_features = counterfactual_output.get("top_features")
    top_features = top_features if isinstance(top_features, list) else []

    dominant_factors = []
    for row in top_features:
        if not isinstance(row, dict):
            continue
        name = str(row.get("feature", "")).strip()
        if not name:
            continue
        dominant_factors.append(
            {
                "name": name,
                "importance": float(row.get("shap_importance", row.get("attribution_score", 0.0))),
                "shap_value": float(row.get("shap_value", 0.0)),
                "direction": str(row.get("direction", "keep")),
            }
        )

    summary = counterfactual_output.get("llm_guidance", "")

    return {
        "scene_id": str(counterfactual_output.get("scene_id", "unknown_scene")),
        "method": str(counterfactual_output.get("method", "simulation_grounded_kernelshap")),
        "result_status": str(counterfactual_output.get("result_status", "UNKNOWN")),
        "risk_level": str(counterfactual_output.get("risk_level", "unknown")),
        "top_features": top_features,
        "boundary_focus_features": counterfactual_output.get("boundary_focus_features", []),
        "llm_guidance": str(summary),
        "xai_signals": {
            "method": str(counterfactual_output.get("method", "simulation_grounded_kernelshap")),
            "dominant_factors": dominant_factors,
            "attention_summary": str(summary),
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(input_path)

    if args.shap_json_path or args.shap_case_id:
        print("[XAI] Notice: legacy SHAP JSON options are ignored in KernelSHAP-direct mode.")

    counterfactual_output, boundary_output = generate_counterfactual_and_boundary(
        payload=payload,
        target_status=args.target_status,
        mode=args.mode,
        num_counterfactuals=args.num_counterfactuals,
        num_boundary_candidates=args.num_boundaries,
        random_seed=args.random_seed,
    )

    counterfactual_path = save_json(
        output_dir / "counterfactual_explanations.json",
        counterfactual_output,
    )
    boundary_path = save_json(
        output_dir / "boundary_candidates.json",
        boundary_output,
    )

    scene_id = str(counterfactual_output.get("scene_id", payload.get("scene_id", "unknown_scene")))
    xai_signals_payload = _build_xai_signals_payload(counterfactual_output)
    xai_signals_path = save_json(
        output_dir / _resolve_step_filename(scene_id),
        xai_signals_payload,
    )

    print("[XAI] Simulation-grounded KernelSHAP attribution complete")
    print(f"- input: {input_path}")
    print(f"- method: {counterfactual_output.get('method')}")
    print(f"- result_status: {counterfactual_output.get('result_status')}")
    print(f"- risk_level: {counterfactual_output.get('risk_level')}")
    print(f"- counterfactual_explanations: {counterfactual_path}")
    print(f"- boundary_candidates: {boundary_path}")
    print(f"- xai_signals: {xai_signals_path}")
    print("")

    base_case = counterfactual_output.get("base_case", {})
    print("[Base case]")
    print(
        "status={status}, decision_margin={margin:.4f}, distance_to_boundary={distance:.4f}".format(
            status=base_case.get("predicted_status", "UNKNOWN"),
            margin=float(base_case.get("decision_margin", 0.0)),
            distance=float(base_case.get("distance_to_boundary", 0.0)),
        )
    )

    top_rows = counterfactual_output.get("minimal_change_candidates")
    top_rows = top_rows if isinstance(top_rows, list) else []
    if top_rows:
        top_cf = top_rows[0]
        print("\n[Top candidate]")
        print(
            "{cid}: status={status}, margin={margin:.4f}, distance={distance:.4f}".format(
                cid=str(top_cf.get("candidate_id", "cf_00")),
                status=str(top_cf.get("predicted_status", "UNKNOWN")),
                margin=float(top_cf.get("decision_margin", 0.0)),
                distance=float(top_cf.get("distance_l1_normalized", 0.0)),
            )
        )
        print(str(top_cf.get("summary_explanation", "")))


if __name__ == "__main__":
    main()
