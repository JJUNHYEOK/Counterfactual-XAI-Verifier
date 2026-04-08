from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from core.detector import (
    build_dummy_llm_input,
    load_llm_input,
    save_llm_input,
)
from core.schema import SchemaValidationError, validate_llm_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM 파이프라인 실행용 진입점"
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default="data/real_xai_input.json",
        help="입력 JSON 경로",
    )
    parser.add_argument(
        "--output_yaml",
        type=str,
        default="data/mock_simulink_output.yaml",
        help="출력 YAML 경로",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4-turbo",
        help="GPTGenerator 사용 시 모델 이름",
    )
    parser.add_argument(
        "--make_dummy",
        action="store_true",
        help="더미 입력 JSON을 먼저 생성한 뒤 실행",
    )

    # 더미 입력 생성용 옵션
    parser.add_argument("--scene_id", type=str, default="run_001_frame_0231")
    parser.add_argument("--detector_name", type=str, default="yolox-small")
    parser.add_argument("--image_width", type=int, default=1920)
    parser.add_argument("--image_height", type=int, default=1080)
    parser.add_argument("--corruption_type", type=str, default="fog")
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--baseline_confidence", type=float, default=0.91)
    parser.add_argument("--current_confidence", type=float, default=0.63)
    parser.add_argument(
        "--missed_detection",
        action="store_true",
        help="완전 놓침 상황으로 더미 생성",
    )

    return parser.parse_args()


def generate_or_load_input(args: argparse.Namespace) -> dict:
    input_path = ROOT_DIR / args.input_json

    if args.make_dummy or not input_path.exists():
        dummy_input = build_dummy_llm_input(
            frame_id=args.scene_id,
            detector_name=args.detector_name,
            image_size=(args.image_width, args.image_height),
            corruption_type=args.corruption_type,
            severity=args.severity,
            baseline_confidence=args.baseline_confidence,
            current_confidence=args.current_confidence,
            missed_detection=args.missed_detection,
        )
        save_llm_input(dummy_input, input_path)
        print(f"[1] 더미 입력 JSON 생성 완료: {input_path}")

    data = load_llm_input(input_path)
    print(f"[2] 입력 JSON 로드 완료: {input_path}")
    return data


def run_llm_generation(xai_input: dict, model_name: str):
    """
    저장소 안의 LLM 코드 형태가 다를 수 있어서 여러 방식으로 시도한다.
    """
    errors: list[str] = []

    # 1) 함수형 API: core.gpt_generator.generate_scenario
    try:
        from llm_agent.gpt_generator import generate_scenario

        return generate_scenario(xai_input)
    except Exception as e:
        errors.append(f"core.gpt_generator.generate_scenario 실패: {e}")

    # 2) 클래스형 API: core.gpt_generator.GPTGenerator
    try:
        from llm_agent.gpt_generator import GPTGenerator

        generator = GPTGenerator(model_name=model_name)

        if hasattr(generator, "generate_counterfactual"):
            return generator.generate_counterfactual(xai_input)
        if hasattr(generator, "generate_scenario"):
            return generator.generate_scenario(xai_input)

        raise AttributeError("GPTGenerator 안에 generate_counterfactual/generate_scenario가 없습니다.")
    except Exception as e:
        errors.append(f"core.gpt_generator.GPTGenerator 실패: {e}")

    # 3) 클래스형 API: llm_agent.gpt_generator.GPTGenerator
    try:
        from llm_agent.gpt_generator import GPTGenerator

        generator = GPTGenerator(model_name=model_name)

        if hasattr(generator, "generate_counterfactual"):
            return generator.generate_counterfactual(xai_input)
        if hasattr(generator, "generate_scenario"):
            return generator.generate_scenario(xai_input)

        raise AttributeError("llm_agent GPTGenerator 안에 generate_counterfactual/generate_scenario가 없습니다.")
    except Exception as e:
        errors.append(f"llm_agent.gpt_generator.GPTGenerator 실패: {e}")

    error_text = "\n".join(errors)
    raise RuntimeError(
        "LLM 생성기 호출에 실패했습니다.\n"
        "가능한 원인:\n"
        "1) gpt_generator.py 위치가 다름\n"
        "2) 함수/메서드 이름이 다름\n"
        "3) API 키 또는 환경 설정 문제\n\n"
        f"세부 오류:\n{error_text}"
    )


def export_yaml_output(result, output_path: Path) -> None:
    """
    기존 yaml_exporter가 있으면 우선 사용하고,
    없거나 형식이 다르면 안전하게 직접 저장한다.
    """
    # 1) 기존 exporter 우선 시도
    try:
        from parsers.yaml_exporter import export_yaml

        export_yaml(result, str(output_path))
        print(f"[4] yaml_exporter 사용 완료: {output_path}")
        return
    except Exception as e:
        print(f"[4-1] 기존 yaml_exporter 사용 실패, 직접 저장으로 전환: {e}")

    # 2) 직접 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(result, str):
        text = result
    else:
        try:
            import yaml
            text = yaml.safe_dump(
                result,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        except Exception:
            text = json.dumps(result, indent=2, ensure_ascii=False)

    output_path.write_text(text, encoding="utf-8")
    print(f"[4] 출력 저장 완료: {output_path}")


def main():
    args = parse_args()

    try:
        xai_input = generate_or_load_input(args)
        validate_llm_input(xai_input)
        print("[3] 입력 JSON 스키마 검증 완료")
    except SchemaValidationError as e:
        raise RuntimeError(f"입력 JSON 형식 오류:\n{e}") from e

    result = run_llm_generation(
        xai_input=xai_input,
        model_name=args.model_name,
    )

    output_path = ROOT_DIR / args.output_yaml
    export_yaml_output(result, output_path)

    print("\n===== 최종 입력 JSON =====")
    print(json.dumps(xai_input, indent=2, ensure_ascii=False))

    print("\n===== LLM 생성 결과 =====")
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()