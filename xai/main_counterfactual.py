from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from xai import generate_counterfactual_and_boundary, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XAI counterfactual / boundary 탐색 실행기",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/counterfactual_case_input.json",
        help="XAI 입력 JSON 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_counterfactual",
        help="출력 디렉터리",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "sim_dummy", "map50_proxy"],
        default="auto",
        help="평가 모드(auto 권장)",
    )
    parser.add_argument(
        "--target_status",
        type=str,
        choices=["PASS", "FAIL", "pass", "fail"],
        default=None,
        help="목표 상태 미지정 시 현재 상태를 자동 반전",
    )
    parser.add_argument(
        "--num_counterfactuals",
        type=int,
        default=3,
        help="최소 변화 counterfactual 후보 개수",
    )
    parser.add_argument(
        "--num_boundaries",
        type=int,
        default=5,
        help="경계 인접 후보 개수",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="재현성을 위한 랜덤 시드",
    )
    parser.add_argument(
        "--shap_json_path",
        type=str,
        default=None,
        help="tabular_xai의 xai_llm_output.json 경로(선택). 제공 시 SHAP 신호를 counterfactual 탐색에 결합",
    )
    parser.add_argument(
        "--shap_case_id",
        type=str,
        default=None,
        help="SHAP JSON에서 사용할 failure_case_id(선택)",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_shap_signals_from_llm_json(
    shap_payload: dict[str, Any],
    scene_id: str | None,
    requested_case_id: str | None,
) -> dict[str, Any]:
    global_rows = (
        shap_payload.get("global_explanation", {}).get("top_contributing_features", [])
        if isinstance(shap_payload.get("global_explanation"), dict)
        else []
    )
    global_rows = global_rows if isinstance(global_rows, list) else []

    global_factors: list[dict[str, Any]] = []
    for row in global_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("feature", "")).strip()
        if not name:
            continue
        global_factors.append(
            {
                "name": name,
                "importance": abs(
                    _safe_float(row.get("mean_abs_shap", row.get("importance", 0.0)), 0.0)
                ),
            }
        )

    case_rows = shap_payload.get("failure_case_explanations", [])
    case_rows = case_rows if isinstance(case_rows, list) else []
    selected_case: dict[str, Any] | None = None

    if requested_case_id:
        for case in case_rows:
            if not isinstance(case, dict):
                continue
            if str(case.get("failure_case_id", "")) == str(requested_case_id):
                selected_case = case
                break

    if selected_case is None and scene_id:
        for case in case_rows:
            if not isinstance(case, dict):
                continue
            if str(case.get("failure_case_id", "")) == str(scene_id):
                selected_case = case
                break

    if selected_case is None and case_rows:
        selected_case = next((case for case in case_rows if isinstance(case, dict)), None)

    local_factors: list[dict[str, Any]] = []
    selected_case_id = None
    if isinstance(selected_case, dict):
        selected_case_id = str(selected_case.get("failure_case_id", ""))
        top_rows = selected_case.get("top_contributing_features", [])
        top_rows = top_rows if isinstance(top_rows, list) else []
        for row in top_rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("feature", row.get("name", ""))).strip()
            if not name:
                continue
            local_factors.append(
                {
                    "name": name,
                    "contribution_score": _safe_float(row.get("contribution_score", 0.0), 0.0),
                    "abs_contribution_score": abs(
                        _safe_float(
                            row.get("abs_contribution_score", row.get("contribution_score", 0.0)),
                            0.0,
                        )
                    ),
                    "direction": row.get("direction"),
                }
            )

    return {
        "source_schema": str(shap_payload.get("schema_version", "unknown")),
        "selected_failure_case_id": selected_case_id,
        "global_feature_importance": global_factors,
        "local_feature_contributions": local_factors,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(input_path)
    cf_request = payload.get("counterfactual_request")
    cf_request = cf_request if isinstance(cf_request, dict) else {}

    if args.shap_json_path:
        shap_payload = load_json(args.shap_json_path)
        shap_signals = _build_shap_signals_from_llm_json(
            shap_payload=shap_payload,
            scene_id=str(payload.get("scene_id", "")),
            requested_case_id=args.shap_case_id,
        )
        payload["shap_signals"] = shap_signals

        xai_signals = payload.get("xai_signals")
        if not isinstance(xai_signals, dict):
            xai_signals = {}
            payload["xai_signals"] = xai_signals
        if not isinstance(xai_signals.get("dominant_factors"), list) or not xai_signals.get("dominant_factors"):
            xai_signals["dominant_factors"] = [
                {"name": row["name"], "importance": row["importance"]}
                for row in shap_signals["global_feature_importance"][:5]
            ]

    resolved_target = args.target_status or cf_request.get("target_status")
    resolved_mode = args.mode
    if args.mode == "auto" and isinstance(cf_request.get("evaluator"), str):
        request_mode = str(cf_request["evaluator"]).strip().lower()
        if request_mode in {"sim_dummy", "map50_proxy"}:
            resolved_mode = request_mode

    counterfactual_output, boundary_output = generate_counterfactual_and_boundary(
        payload=payload,
        target_status=resolved_target,
        mode=resolved_mode,
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

    print("[XAI] Counterfactual / Boundary search complete")
    print(f"- input: {input_path}")
    print(f"- mode: {counterfactual_output['search_mode']}")
    if args.shap_json_path:
        print(f"- shap_guidance: {args.shap_json_path}")
    print(f"- counterfactual_explanations: {counterfactual_path}")
    print(f"- boundary_candidates: {boundary_path}")
    print("")

    base_case = counterfactual_output["base_case"]
    print("[Base case]")
    print(
        f"status={base_case['predicted_status']}, "
        f"decision_margin={base_case['decision_margin']:.4f}, "
        f"distance_to_boundary={base_case['distance_to_boundary']:.4f}"
    )

    print("\n[Top minimal-change candidate]")
    top_cf = counterfactual_output["minimal_change_candidates"][0]
    print(
        f"{top_cf['candidate_id']}: status={top_cf['predicted_status']}, "
        f"margin={top_cf['decision_margin']:.4f}, "
        f"distance={top_cf['distance_l1_normalized']:.4f}"
    )
    print(top_cf["summary_explanation"])


if __name__ == "__main__":
    main()
