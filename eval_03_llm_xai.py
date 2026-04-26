import time
import json
import cv2
import albumentations as A
from pathlib import Path

from xai.real_analyzer import RealXAIAnalyzer
from llm_agent.gpt_generator import GPTGenerator

# --- [경로 및 설정] ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = ROOT_DIR / "assets"
CLEAN_IMAGE_PATH = str(IMAGE_DIR / "step_1.jpg")

SAFETY_LINE = 0.4829
MAX_STEPS_PER_TRIAL = 50
TOTAL_TRIALS = 5

print("📦 LLM+XAI (제안 시스템) 평가기 로드 중...")
xai = RealXAIAnalyzer()
llm = GPTGenerator(model_name="gpt-5.4-nano")

def clean_val(val, default=0.0):
    try:
        s = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
        return float(s)
    except:
        return default

def _normalize_dominant_factors(feature_importance: list[dict]) -> list[dict]:
    if not isinstance(feature_importance, list): return []
    normalized = []
    for row in feature_importance:
        if isinstance(row, dict) and str(row.get("name", "")).strip():
            normalized.append({
                "name": str(row.get("name", "")).strip(),
                "importance": float(clean_val(row.get("importance", 0.0))),
            })
    return normalized

def run_llm_xai_evaluation():
    base_img = cv2.imread(CLEAN_IMAGE_PATH)
    if base_img is None:
        print("❌ 원본 이미지를 찾을 수 없습니다.")
        return

    print(f"\n🚀 총 {TOTAL_TRIALS}회의 LLM+XAI (제안 시스템) 실험을 시작합니다.")
    print("목표: XAI 피드백을 통해 최소 횟수(최저 StFF)로 시스템 붕괴 유도\n")
    experiment_results = []
    total_stff = 0

    for trial in range(1, TOTAL_TRIALS + 1):
        print(f"▶️ [Trial {trial}/{TOTAL_TRIALS}] 지능형 탐색 시작...")
        start_time = time.time()
        
        stff = MAX_STEPS_PER_TRIAL
        final_map = 0.0
        found_failure = False
        failed_params = {}
        
        # [핵심] 초기 상태 설정
        current_state = {
            "map50": 0.63, 
            "xai_signals": {"dominant_factors": []},
            "message": "초기 상태입니다. 환경 변수를 조합하여 UAV 객체 탐지율을 낮추세요."
        }
        
        for step in range(1, MAX_STEPS_PER_TRIAL + 1):
            print(f"  [Step {step}] XAI 피드백 기반 시나리오 생성 중...")
            scenario = llm.generate_scenario(current_state)
            env = scenario.get("environment_parameters", {}) if scenario else {}
            
            f = clean_val(env.get("fog_density_percent", 0))
            n = clean_val(env.get("camera_noise_level", 0))
            m = int(clean_val(env.get("motion_blur_intensity", 0)))
            lux = clean_val(env.get("illumination_lux", 1000))
            
            # 이미지 변조
            aug_list = []
            if f > 0: aug_list.append(A.RandomFog(fog_coef=min(f/100.0, 1.0), p=1.0))
            if n > 0: aug_list.append(A.GaussNoise(var_limit=(10.0, max(10.1, n*5)), p=1.0))
            if m > 2: 
                m_odd = m if m % 2 != 0 else m + 1
                aug_list.append(A.MotionBlur(blur_limit=(3, max(3, m_odd)), p=1.0))
            if lux != 1000:
                darken = max(-0.8, (lux - 1000) / 1000.0)
                aug_list.append(A.RandomBrightnessContrast(brightness_limit=(darken, darken), p=1.0))
            
            degraded_img = A.Compose(aug_list)(image=base_img)['image'] if aug_list else base_img.copy()
            temp_path = str(IMAGE_DIR / f"temp_eval_llm_xai_{trial}.jpg")
            cv2.imwrite(temp_path, degraded_img)
            
            # [핵심] XAI 분석 수행 및 피드백 추출
            new_map50, feature_importance = xai.analyze(temp_path, scenario)
            dominant_factors = _normalize_dominant_factors(feature_importance)
            
            print(f"    -> 적용: 안개 {f}%, 노이즈 {n}, 블러 {m} | 결과 mAP: {new_map50:.4f}")
            
            if new_map50 < SAFETY_LINE:
                stff = step
                final_map = new_map50
                found_failure = True
                failed_params = env
                break
                
            # [핵심] LLM에게 XAI의 분석 결과를 무기로 쥐여줌
            current_state = {
                "map50": float(new_map50),
                "xai_signals": {"dominant_factors": dominant_factors}, # <-- 여기가 논문의 Secret Sauce!
                "message": f"현재 mAP는 {new_map50:.4f}입니다. 첨부된 XAI dominant_factors를 분석하여, 모델이 가장 취약해하는 요인을 극대화하는 방향으로 다음 파라미터를 생성하세요."
            }

        end_time = time.time()
        total_stff += stff
        
        trial_data = {
            "trial": trial,
            "stff": stff,
            "failure_found": found_failure,
            "final_map": float(final_map),
            "time_taken_sec": round(end_time - start_time, 2),
            "failed_parameters": failed_params
        }
        experiment_results.append(trial_data)
        
        if found_failure:
            print(f"  🚨 XAI 타겟팅 성공! StFF: {stff}회 | mAP: {final_map:.4f}")
        else:
            print(f"  ✅ {MAX_STEPS_PER_TRIAL}회 탐색 내에 실패를 찾지 못했습니다.")

    avg_stff = total_stff / TOTAL_TRIALS
    print(f"\n📊 [평가 완료] LLM+XAI (제안 시스템) 평균 StFF: {avg_stff}회")
    
    with open(DATA_DIR / "eval_result_llm_xai.json", "w", encoding="utf-8") as f_out:
        json.dump({
            "experiment": "LLM_XAI_Proposed",
            "total_trials": TOTAL_TRIALS,
            "average_stff": avg_stff,
            "trials": experiment_results
        }, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_llm_xai_evaluation()
