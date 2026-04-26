import time
import json
import threading
import numpy as np
import cv2
import albumentations as A
from pathlib import Path

from xai.real_analyzer import RealXAIAnalyzer
from llm_agent.gpt_generator import GPTGenerator

# --- [경로 및 설정] ---
ROOT_DIR = Path(__file__).resolve().parent
TEMP_STREAM_PATH = ROOT_DIR / "assets" / "simulink_stream.jpg"

SAFETY_LINE = 0.4829
BASE_MAP = 0.63

print("📦 비동기 실시간 관제탑 로드 중...")
xai = RealXAIAnalyzer()
llm = GPTGenerator(model_name="gpt-5.4-nano")

# --- [공유 상태 (Shared State)] ---
# 두 개의 스레드(Simulink 메인 스레드 & LLM 백그라운드 스레드)가 공유하는 데이터
shared_state = {
    "step_count": 0,
    "current_env": {"fog_density_percent": 0, "camera_noise_level": 0, "motion_blur_intensity": 0, "illumination_lux": 1000},
    "latest_map50": 0.63,
    "latest_xai_factors": [],
    "is_llm_thinking": False,
    "is_crashed": False
}

def clean_val(val, default=0.0):
    try:
        s = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
        return float(s)
    except:
        return default

def _normalize_dominant_factors(feature_importance):
    if not isinstance(feature_importance, list): return []
    return [{"name": str(r.get("name", "")).strip(), "importance": float(clean_val(r.get("importance", 0.0)))} for r in feature_importance if isinstance(r, dict) and str(r.get("name", "")).strip()]

# --- [느린 루프: 백그라운드 LLM 스레드] ---
def llm_worker():
    global shared_state
    print("🧠 [LLM Worker] 백그라운드 스레드 가동 완료.")
    
    while not shared_state["is_crashed"]:
        time.sleep(3) # 3초마다 상황을 보고 공격 시나리오 업데이트
        
        if shared_state["step_count"] < 10 or shared_state["is_llm_thinking"]:
            continue # 초반이거나 이미 생각 중이면 대기
            
        shared_state["is_llm_thinking"] = True
        print("\n🤔 [LLM] 현재 XAI 지표 분석 및 새로운 결함 시나리오 생성 중...")
        
        prompt_state = {
            "map50": shared_state["latest_map50"],
            "xai_signals": {"dominant_factors": shared_state["latest_xai_factors"]},
            "message": "UAV 실시간 비행 중입니다. XAI 요인을 바탕으로 mAP를 떨어뜨릴 환경 변수를 즉각 생성하세요."
        }
        
        scenario = llm.generate_scenario(prompt_state)
        
        if scenario and "environment_parameters" in scenario:
            new_env = scenario["environment_parameters"]
            # 상태 업데이트 (다음 프레임부터 이 날씨가 적용됨)
            shared_state["current_env"]["fog_density_percent"] = clean_val(new_env.get("fog_density_percent", 0))
            shared_state["current_env"]["camera_noise_level"] = clean_val(new_env.get("camera_noise_level", 0))
            shared_state["current_env"]["motion_blur_intensity"] = int(clean_val(new_env.get("motion_blur_intensity", 0)))
            shared_state["current_env"]["illumination_lux"] = clean_val(new_env.get("illumination_lux", 1000))
            print(f"⚡ [LLM 업데이트 적용] 안개: {shared_state['current_env']['fog_density_percent']}%, 노이즈: {shared_state['current_env']['camera_noise_level']}")
            
        shared_state["is_llm_thinking"] = False

# LLM 스레드 시작 (데몬 스레드로 설정하여 메인 프로그램 종료 시 함께 종료)
threading.Thread(target=llm_worker, daemon=True).start()

# --- [빠른 루프: Simulink 실시간 호출 함수] ---
def analyze_realtime(np_img):
    global shared_state
    
    try:
        if shared_state["is_crashed"]:
            return {"map50": 0.0, "safety_margin": 0.0, "is_violated": True}

        frame = np.array(np_img).astype(np.uint8)
        shared_state["step_count"] += 1

        # 1. 현재 적용된 날씨(LLM이 업데이트한 날씨) 가져오기
        env = shared_state["current_env"]
        f = env["fog_density_percent"]
        n = env["camera_noise_level"]
        m = env["motion_blur_intensity"]
        lux = env["illumination_lux"]

        # 2. 이미지 실시간 변조
        aug_list = []
        if f > 0: aug_list.append(A.RandomFog(fog_coef=min(f/100.0, 1.0), p=1.0))
        if n > 0: aug_list.append(A.GaussNoise(var_limit=(10.0, max(10.1, n*5)), p=1.0))
        if m > 2: aug_list.append(A.MotionBlur(blur_limit=(3, max(3, m if m % 2 != 0 else m + 1)), p=1.0))
        if lux != 1000:
            darken = max(-0.8, (lux - 1000) / 1000.0)
            aug_list.append(A.RandomBrightnessContrast(brightness_limit=(darken, darken), p=1.0))

        degraded_frame = A.Compose(aug_list)(image=frame)['image'] if aug_list else frame

        # 3. 분석 및 상태 갱신
        img_path = str(TEMP_STREAM_PATH)
        cv2.imwrite(img_path, degraded_frame)
        new_map50, feature_importance = xai.analyze(img_path, {})
        
        shared_state["latest_map50"] = new_map50
        shared_state["latest_xai_factors"] = _normalize_dominant_factors(feature_importance)

        # 4. 제어 신호 계산
        margin_val = (new_map50 - SAFETY_LINE) / (BASE_MAP - SAFETY_LINE) * 100
        safety_margin = max(0.0, float(margin_val))
        is_violated = bool(new_map50 < SAFETY_LINE)

        # 5. 콘솔 모니터링 (매 프레임)
        print(f"📡 [비행 중: 프레임 {shared_state['step_count']}] mAP: {new_map50:.4f} | Margin: {safety_margin:.1f}%")

        if is_violated:
            print("\n🚨 [위험 감지] 탐지율 붕괴! 제어권 회수(Fail-safe) 발동.\n")
            shared_state["is_crashed"] = True

        return {
            "map50": float(new_map50),
            "safety_margin": safety_margin,
            "is_violated": is_violated
        }

    except Exception as e:
        print(f"❌ 분석 에러: {e}")
        return {"map50": 0.0, "safety_margin": 0.0, "is_violated": True}
