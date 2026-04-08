import sys
import os
import json
import time
from dotenv import load_dotenv

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_agent.gpt_generator import GPTGenerator

load_dotenv()

def main():
    generator = GPTGenerator(model_name="gpt-4-turbo")
    
    # [설정] 초기 상태 및 방어선
    current_score = 92.5  # 초기 mAP
    current_env = {"fog_density_percent": 0.0, "illumination_lux": 10000.0}
    THRESHOLD = 85.0
    
    print(f"🚀 임계점({THRESHOLD}%) 정밀 탐색 시작 (Baseline: {current_score}%)")

    for i in range(1, 6):  # 최대 5번의 시나리오 업데이트 시도
        print(f"\n--- [Step {i}] 현재 탐지율: {current_score:.2f}% ---")
        
        # 가상의 XAI 데이터 (실제로는 XAI 브랜치에서 넘어올 데이터 구조)
        mock_xai_data = {
            "feature_importance": {"fog_density_percent": 0.75, "illumination_lux": 0.25},
            "insight": "안개 농도가 시각적 인지에 지배적인 영향을 미침"
        }

        # [핵심] 점수와 함께 LLM에게 시나리오 요청
        result = generator.generate_counterfactual(mock_xai_data, current_score)
        
        if result:
            status = result.get("search_status", "UNKNOWN")
            next_env = result.get("environment_parameters", {})
            reason = result.get("adjustment_reasoning", "No reason provided")
            
            print(f"📡 상태: {status}")
            print(f"🤖 제안 파라미터: {next_env}")
            print(f"📝 논리: {reason}")

            # [가정] 실제 시뮬레이터를 돌려 점수가 하락했다고 가정 (Simulink 연동 부분)
            current_env = next_env
            current_score -= 4.0  # 임의 하락치 (테스트용)

            if current_score < THRESHOLD:
                print(f"\n🎯 [탐색 완료] 방어선 붕괴 지점 포착!")
                print(f"최종 임계 조건: {current_env}")
                print(f"최종 탐지율: {current_score:.2f}%")
                break
        else:
            print("❌ LLM 응답 실패")
            break
            
        time.sleep(1)

if __name__ == "__main__":
    main()