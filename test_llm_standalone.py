# test_llm_standalone.py
import sys
import os
import json
from dotenv import load_dotenv

# [핵심] 현재 폴더를 경로에 추가하여 llm_agent와 prompts를 인식하게 함
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_agent.gpt_generator import GPTGenerator

load_dotenv()

def main():
    # 1. 가짜 XAI 데이터 (객체 탐지용)
    mock_xai_data = {
        "scenario_id": "rescue_sim_001",
        "current_requirement": {"target": "mAP50", "threshold": 0.85, "actual": 0.91},
        "current_environment": {"fog_density_percent": 15.0, "illumination_lux": 8000.0},
        "xai_analysis": {
            "feature_importance": {"fog_density_percent": 0.7, "illumination_lux": 0.3},
            "insight": "안개가 탐지율 저하의 주된 원인입니다."
        }
    }

    print("🤖 LLM 카운터팩추얼 시나리오 생성 시작...")
    generator = GPTGenerator(model_name="gpt-4-turbo")
    result = generator.generate_counterfactual(mock_xai_data)

    if result:
        print("\n[생성된 가혹 조건 시나리오]")
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()