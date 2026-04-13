import sys
import os
import json
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_agent.gpt_generator import GPTGenerator

load_dotenv()

def main():
    generator = GPTGenerator(model_name="gpt-4-turbo")
    
    current_score = 92.5
    THRESHOLD = 85.0
    
    current_env = {
        "weather_conditions": {"fog_density_percent": 0.0, "rain_intensity": 0.0},
        "sensor_noise": {"gaussian_noise_level": 0.0, "low_contrast_factor": 0.0},
        "uav_blur_effects": {"motion_blur_intensity": 0.0, "zoom_blur_intensity": 0.0}
    }
    
    print(f"임계점({THRESHOLD}%) 정밀 탐색 시작 (Baseline: {current_score}%)")

    for i in range(1, 6):
        print(f"\n--- [Step {i}] 현재 탐지율: {current_score:.2f}% ---")
        
        mock_xai_data = {
            "feature_importance": {"fog_density_percent": 0.6, "motion_blur_intensity": 0.4},
            "insight": "드론의 움직임과 안개가 결합될 때 탐지 실패 확률이 급증함"
        }


        result = generator.generate_counterfactual(mock_xai_data, current_score, current_env)
        
        if result:
            status = result.get("search_status", "UNKNOWN")
            next_env = result.get("environment_parameters", current_env) # 없으면 기존값 유지
            reason = result.get("adjustment_reasoning", "No reason provided")
            
            print(f"상태: {status}")
            print(f"제안 파라미터: {json.dumps(next_env, ensure_ascii=False)}")
            print(f"논리: {reason}")

            current_env = next_env
            current_score -= 4.0  

            if current_score < THRESHOLD:
                print(f"\n[탐색 완료] 방어선 붕괴 지점 포착!")
                print(f"최종 임계 조건: {json.dumps(current_env, ensure_ascii=False, indent=2)}")
                print(f"최종 탐지율: {current_score:.2f}%")
                break
        else:
            print("❌ LLM 응답 실패")
            break
            
        time.sleep(1)

if __name__ == "__main__":
    main()
    
