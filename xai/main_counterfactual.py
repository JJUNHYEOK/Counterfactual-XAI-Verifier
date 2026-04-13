from __future__ import annotations

import argparse
from pathlib import Path

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(input_path)
    cf_request = payload.get("counterfactual_request")
    cf_request = cf_request if isinstance(cf_request, dict) else {}

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
