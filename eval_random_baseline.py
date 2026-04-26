import time
import random
import json
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from xai.real_analyzer import RealXAIAnalyzer

# --- [경로 및 설정] ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = ROOT_DIR / "assets"
CLEAN_IMAGE_PATH = str(IMAGE_DIR / "step_1.jpg") # 맑은 상태의 원본 이미지

SAFETY_LINE = 0.4829 # 실패로 간주하는 mAP 임계점
MAX_STEPS_PER_TRIAL = 50 # 한 번의 시도에서 최대 탐색 횟수 (이 안에 못 찾으면 50으로 기록)
TOTAL_TRIALS = 5 # 평균을 내기 위한 총 실험 횟수

# 결과 저장용 리스트
experiment_results = []

print("📦 Random Search 베이스라인 평가기 로드 중...")
xai = RealXAIAnalyzer()

def run_random_evaluation():
    base_img = cv2.imread(CLEAN_IMAGE_PATH)
    if base_img is None:
        print("❌ 원본 이미지를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    print(f"\n🎲 총 {TOTAL_TRIALS}회의 Random Search 실험을 시작합니다.")
    print(f"목표: mAP가 {SAFETY_LINE} 미만으로 떨어지는 최초의 탐색 횟수(StFF) 측정\n")
    
    total_stff = 0

    for trial in range(1, TOTAL_TRIALS + 1):
        print(f"▶️ [Trial {trial}/{TOTAL_TRIALS}] 탐색 시작...")
        start_time = time.time()
        
        stff = MAX_STEPS_PER_TRIAL
        final_map = 0.0
        found_failure = False
        failed_params = {}
        
        for step in range(1, MAX_STEPS_PER_TRIAL + 1):
            # 1. 무지성 랜덤 파라미터 생성 (탐색 공간: 안개, 노이즈, 조도, 블러)
            f = random.uniform(0.0, 1.0)       # 안개 (0~100%)
            n = random.uniform(0.0, 50.0)      # 노이즈 (0~50)
            lux = random.uniform(200, 1000)    # 조도 (200~1000 Lux)
            m = random.choice([0, 3, 5, 7])    # 모션 블러 커널 크기 (0은 적용 안 함)
            
            # 2. 파라미터를 이미지에 합성
            aug_list = []
            if f > 0: aug_list.append(A.RandomFog(fog_coef=f, p=1.0))
            if n > 0: aug_list.append(A.GaussNoise(var_limit=(10.0, max(10.1, n)), p=1.0))
            if m > 0: aug_list.append(A.MotionBlur(blur_limit=(3, max(3, m)), p=1.0))
            if lux < 1000:
                darken = max(-0.8, (lux - 1000) / 1000.0)
                aug_list.append(A.RandomBrightnessContrast(brightness_limit=(darken, darken), p=1.0))
            
            degraded_img = A.Compose(aug_list)(image=base_img)['image']
            
            # 임시 저장 후 분석 (I/O 병목이 있지만 베이스라인 측정이므로 감수)
            temp_path = str(IMAGE_DIR / f"temp_eval_{trial}.jpg")
            cv2.imwrite(temp_path, degraded_img)
            
            new_map50, _ = xai.analyze(temp_path, {})
            
            # 3. 붕괴(Failure) 확인
            if new_map50 < SAFETY_LINE:
                stff = step
                final_map = new_map50
                found_failure = True
                failed_params = {"fog": f, "noise": n, "lux": lux, "blur": m}
                break

        # Trial 결과 기록
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
            print(f"  🚨 실패 발견! StFF: {stff}회 | mAP: {final_map:.4f} | 소요 시간: {trial_data['time_taken_sec']}초")
        else:
            print(f"  ✅ {MAX_STEPS_PER_TRIAL}회 탐색 내에 실패를 찾지 못했습니다. (StFF={MAX_STEPS_PER_TRIAL} 기록)")

    # 4. 최종 통계 산출 및 저장
    avg_stff = total_stff / TOTAL_TRIALS
    print(f"\n📊 [평가 완료] Random Search 평균 StFF (최초 실패까지의 탐색 횟수): {avg_stff}회")
    
    with open(DATA_DIR / "eval_result_random.json", "w", encoding="utf-8") as f_out:
        json.dump({
            "experiment": "Random_Search_Baseline",
            "total_trials": TOTAL_TRIALS,
            "average_stff": avg_stff,
            "trials": experiment_results
        }, f_out, indent=2, ensure_ascii=False)
        
    print("📝 결과가 data/eval_result_random.json 에 저장되었습니다.")

if __name__ == "__main__":
    run_random_evaluation()
