import sys
import os
from pathlib import Path
import time
import json
import cv2
import albumentations as A
import re

# ---------------------------------------------------------
# [경로 설정] 파이썬이 xai, llm_agent 폴더를 정상 인식하도록 강제 고정
# ---------------------------------------------------------
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from llm_agent.gpt_generator import GPTGenerator
from xai.real_analyzer import RealXAIAnalyzer

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = ROOT_DIR / "assets"

def save_json(data: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def clean_val(val):
    """어떤 지독한 문자열이 들어와도 숫자만 추출하는 방탄 함수"""
    try:
        s = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
        return float(s)
    except:
        return 0.0

def run_dynamic_pipeline():
    print("🚀 UAV 자율 검증 파이프라인 가동 (실시간 능동 변조 모드)")
    
    llm = GPTGenerator(model_name="gpt-5.4-nano")
    xai = RealXAIAnalyzer()
    
    # 초기 상태 세팅
    current_state = {
        "map50": 0.95,
        "xai_signals": {"dominant_factors": []},
        "message": "초기 상태입니다. 원본 이미지 분석부터 시작합니다."
    }
    
    # 3단계 강제 탐색 루프 (발표 시연용)
    for step in range(1, 4):
        print(f"\n{'='*50}\n🔄 [Iteration {step}] 탐색 및 동적 변조 루프 시작\n{'='*50}")
        
        # ---------------------------------------------------------
        # Step 1: LLM 공격 시나리오 생성
        # ---------------------------------------------------------
        print("🤖 [LLM] 이전 피드백을 분석하여 악화된 반사실적 시나리오를 설계 중...")
        max_retries = 3  # 최대 3번까지 다시 물어봅니다
        scenario = None
        
        for attempt in range(max_retries):
            scenario = llm.generate_scenario(current_state)
            if scenario is not None:
                break # 성공적으로 답변을 받으면 재시도 중단!
                
            # 503 에러 등으로 None이 반환되었을 때의 처리
            print(f"  ⚠️ [API 지연] OpenAI 서버가 바쁩니다. 5초 후 다시 시도합니다... ({attempt + 1}/{max_retries})")
            time.sleep(5) # 5초 동안 숨 고르기
            
        if scenario is None:
            print("❌ OpenAI 서버 오류가 지속되어 파이프라인을 중단합니다. 1~2분 뒤에 다시 실행해 주세요.")
            break
            
        print(f"  [LLM 디버그 로그]: {json.dumps(scenario, ensure_ascii=False)[:200]}...") # 너무 길면 잘라서 출력

        
        # ---------------------------------------------------------
        # Step 2: 실시간 이미지 훼손 시뮬레이터 (Albumentations)
        # ---------------------------------------------------------
        print("🌍 [Simulator] LLM 파라미터 파싱 및 실시간 이미지 변조 중...")
        
        base_image_path = str(IMAGE_DIR / "step_1.jpg")
        if not os.path.exists(base_image_path):
            print(f"❌ 에러: {base_image_path} 파일이 없습니다. (원본 이미지를 준비해주세요)")
            break
            
        base_img = cv2.imread(base_image_path)
        env_params = scenario.get("environment_parameters", {})
        
        # LLM이 Flat하게 주든 Nested(중첩)하게 주든 무조건 찾아내는 방어 로직
        def get_param(flat_key, nested_parent, nested_child):
            if flat_key in env_params:
                return env_params[flat_key]
            if nested_parent in env_params and isinstance(env_params[nested_parent], dict):
                if nested_child in env_params[nested_parent]:
                    return env_params[nested_parent][nested_child]
            return 0.0

        f_raw = get_param("fog_density_percent", "weather_conditions", "fog_density_percent")
        n_raw = get_param("camera_noise_level", "sensor_noise", "gaussian_noise_level")
        m_raw = get_param("motion_blur_intensity", "uav_blur_effects", "motion_blur_intensity")
        z_raw = get_param("zoom_blur_intensity", "uav_blur_effects", "zoom_blur_intensity")

        target_fog = clean_val(f_raw)
        target_noise = clean_val(n_raw)
        target_m_blur = int(clean_val(m_raw))
        target_z_blur = clean_val(z_raw)

        print(f"  👉 [적용 스펙] 안개: {target_fog}%, 노이즈: {target_noise}, M-블러: {target_m_blur}, Z-블러: {target_z_blur}")

        # 육안으로 확 띄게 스케일링 (체감 증폭)
        aug_list = []
        if target_fog > 0:
            f_val = min(target_fog / 100.0, 1.0)
            # fog_coef_lower 대신 alpha_coef 만으로 안개 조절 (버전 충돌 방지)
            aug_list.append(A.RandomFog(alpha_coef=f_val, p=1.0))
            
        if target_noise > 0:
            # var_limit 대신 var_limit 또는 최신 버전의 std_range 사용. 
            # 가장 안전한 방법은 기본 GaussNoise에 p값만 조절하는 것입니다.
            # 노이즈 값이 클수록 적용 확률(p)과 강도를 높임
            p_val = min(target_noise * 10.0, 1.0) # 0.1이면 100% 적용되도록
            if p_val > 0.01:
                # 구버전/신버전 호환을 위해 가장 기본적인 파라미터 사용
                aug_list.append(A.GaussNoise(p=p_val))
            
        if target_m_blur > 2:
            b_val = target_m_blur if target_m_blur % 2 != 0 else target_m_blur + 1
            b_val = max(3, min(b_val, 31)) # 51은 너무 커서 아예 안 보일 수 있으니 31로 제한
            aug_list.append(A.MotionBlur(blur_limit=(3, b_val), p=1.0))
            
        if target_z_blur > 0:
            # max_factor는 대부분의 버전에서 지원
            z_val = 1.0 + (target_z_blur / 50.0)
            aug_list.append(A.ZoomBlur(max_factor=z_val, step_factor=(0.01, 0.02), p=1.0))

        # 효과 적용 및 새 이미지 저장
        if aug_list:
            degraded_img = A.Compose(aug_list)(image=base_img)['image']
        else:
            print("  ⚠️ 주의: 추출된 변조 수치가 모두 0이어서 원본이 그대로 유지됩니다.")
            degraded_img = base_img

        current_image_name = f"current_iter_{step}.jpg"
        current_image_path = str(IMAGE_DIR / current_image_name)
        cv2.imwrite(current_image_path, degraded_img)

        # ---------------------------------------------------------
        # Step 3: XAI & YOLO 분석 (진짜 데이터 추출)
        # ---------------------------------------------------------
        print("🔍 [XAI] 변조된 이미지를 YOLO로 탐지하고 SHAP 기여도를 분석 중...")
        new_map50, feature_importance = xai.analyze(current_image_path, scenario)
            
        print(f"  👉 추출된 실제 mAP50: {new_map50:.4f}")
        
        # 다음 루프를 위한 상태 업데이트
        current_state = {
            "map50": new_map50,
            "xai_signals": {"dominant_factors": feature_importance}
        }
        
        # ---------------------------------------------------------
        # Step 4: 대시보드 연동용 JSON 저장
        # ---------------------------------------------------------
        dashboard_data = {
            "iteration": step,
            "panel_1_visual": {
                "params": {
                    "fog": target_fog,
                    "noise": target_noise,
                    "motion_blur": target_m_blur,
                    "zoom_blur": target_z_blur
                },
                "map50_score": round(new_map50, 4),
                "rendered_image": current_image_name
            },
            "panel_2_xai": feature_importance,
            "panel_3_llm": {
                "hypothesis": scenario.get("target_hypothesis", ""),
                "composite_strategy": scenario.get("composite_strategy", "N/A"),
                "reasoning": scenario.get("llm_reasoning", "")
            }
        }
        
        out_file = DATA_DIR / f"dashboard_step_{step}.json"
        save_json(dashboard_data, out_file)
        print(f"✅ 대시보드용 데이터 갱신 완료: {out_file.name}")

    print(f"\n🎯 [탐색 완료] 총 4단계의 동적 변조 및 능동 탐색 루프를 성공적으로 마쳤습니다.")

if __name__ == "__main__":
    run_dynamic_pipeline()