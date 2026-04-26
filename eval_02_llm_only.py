import time
import json
import cv2
import albumentations as A
from pathlib import Path

# 기존 모듈 임포트
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

print("📦 LLM-only 베이스라인 평가기 로드 중...")
xai = RealXAIAnalyzer()
llm = GPTGenerator(model_name="gpt-5.4-nano")

# 값 정제 유틸리티
def clean_val(val, default=0.0):
    try:
        s = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
        return float(s)
    except:
        return default

def run_llm_only_evaluation():
    base_img = cv2.imread(CLEAN_IMAGE_PATH)
    if base_img is None:
        print("❌ 원본 이미지를 찾을 수 없습니다.")
        return

    print(f"\n🧠 총 {TOTAL_TRIALS}회의 LLM-only 실험을 시작합니다. (XAI 피드백 없음)")
    experiment_results = []
    total_stff = 0

    for trial in range(1, TOTAL_TRIALS + 1):
        print(f"\n▶️ [Trial {trial}/{TOTAL_TRIALS}] 탐색 시작...")
        start_time = time.time()
        
        stff = MAX_STEPS_PER_TRIAL
        final_map = 0.0
        found_failure = False
        failed_params = {}
        
        # 초기 상태: XAI 정보(xai_signals)를 의도적으로 비워둡니다.
        current_state = {
            "map50": 0.63, # 임의의 초기 베이스라인
            "xai_signals": {"dominant_factors": []}, # 핵심: XAI 힌트 차단
            "message": "UAV 객체 탐지 시스템을 무너뜨리기 위한 새로운 환경 결함 조합을 생성하세요."
        }
        
        for step in range(1, MAX_STEPS_PER_TRIAL + 1):
            print(f"  [Step {step}] LLM 시나리오 생성 중...")
            scenario = llm.generate_scenario(current_state)
            
            if not scenario:
                print("  ⚠️ LLM 응답 실패, 기본값으로 진행합니다.")
                env = {"fog_density_percent": 30, "camera_noise_level": 10, "motion_blur_intensity": 3, "illumination_lux": 800}
            else:
                env = scenario.get("environment_parameters", {})
            
            # LLM이 제안한 파라미터 추출
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
            
            if aug_list:
                degraded_img = A.Compose(aug_list)(image=base_img)['image']
            else:
                degraded_img = base_img.copy()
                
            temp_path = str(IMAGE_DIR / f"temp_eval_llm_{trial}.jpg")
            cv2.imwrite(temp_path, degraded_img)
            
            # 분석 수행
            new_map50, _ = xai.analyze(temp_path, scenario)
            
            print(f"    -> 적용: 안개 {f}%, 노이즈 {n}, 블러 {m}, 조도 {lux}lx | 결과 mAP: {new_map50:.4f}")
            
            # 실패 조건 도달 확인
            if new_map50 < SAFETY_LINE:
                stff = step
                final_map = new_map50
                found_failure = True
                failed_params = env
                break
                
            # 피드백 업데이트 (여전히 XAI 정보는 주지 않음)
            current_state = {
                "map50": float(new_map50),
                "xai_signals": {"dominant_factors": []},
                "message": f"이전 조합으로 시스템 붕괴에 실패했습니다(mAP: {new_map50:.4f}). 파라미터를 완전히 다르게 재조합하여 더 강력한 공격을 시도하세요."
            }

        # Trial 종료 후 기록
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
            print(f"  🚨 실패 발견! StFF: {stff}회 | mAP: {final_map:.4f}")
        else:
            print(f"  ✅ {MAX_STEPS_PER_TRIAL}회 탐색 내에 실패를 찾지 못했습니다.")

    # 최종 결과 저장
    avg_stff = total_stff / TOTAL_TRIALS
    print(f"\n📊 [평가 완료] LLM-only 평균 StFF: {avg_stff}회")
    
    with open(DATA_DIR / "eval_result_llm_only.json", "w", encoding="utf-8") as f_out:
        json.dump({
            "experiment": "LLM_Only_Baseline",
            "total_trials": TOTAL_TRIALS,
            "average_stff": avg_stff,
            "trials": experiment_results
        }, f_out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_llm_only_evaluation()
