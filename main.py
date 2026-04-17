import sys
import os
from pathlib import Path
import time
import json
import cv2
import albumentations as A

# [실시간 출력 함수]
def s_print(msg):
    print(msg, flush=True)

# 경로 설정
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from llm_agent.gpt_generator import GPTGenerator
from xai.real_analyzer import RealXAIAnalyzer
from xai import generate_counterfactual_and_boundary, save_json

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = ROOT_DIR / "assets"

def clean_val(val):
    try:
        s = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
        return float(s)
    except:
        return 0.0


def _normalize_dominant_factors(feature_importance: list[dict]) -> list[dict]:
    if not isinstance(feature_importance, list):
        return []
    normalized: list[dict] = []
    for row in feature_importance:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "importance": float(clean_val(row.get("importance", 0.0))),
            }
        )
    return normalized

def run_dynamic_pipeline():
    s_print("UAV 객체 탐지 자율 검증 파이프라인 가동")
    
    # 💡 gpt-4o-mini 모델 권장
    llm = GPTGenerator(model_name="gpt-5.4-nano")
    xai = RealXAIAnalyzer() 
    
    # 초기 상태
    current_state = {
        "map50": 0.95, 
        "xai_signals": {"dominant_factors": []},
        "message": "첫 번째 루프입니다. 85% 경계선을 탐색하기 위한 가벼운 복합 결함을 설계하세요."
    }
    
    for step in range(1, 6):
        s_print(f"\n{'='*50}\n🔄 [Iteration {step}] 루프 시작\n{'='*50}")
        
        # ---------------------------------------------------------
        # [신규] Step 0: 실시간 점수 피드백 생성
        # ---------------------------------------------------------
        score = current_state["map50"]
        if step > 1: # 2회차 루프부터 피드백 적용
            if score < 0.70:
                feedback = f"경고: 현재 점수({score:.2f})가 너무 낮습니다. 파라미터 강도를 50% 이상 낮춰서 0.85 근처로 복구하세요."
            elif score > 0.90:
                feedback = f"정보: 모델이 너무 평온합니다({score:.2f}). 강도를 20% 높여서 85% 선 아래로 떨어뜨리세요."
            else:
                feedback = f"목표 근접: 현재 점수({score:.2f})가 85% 경계선 근처입니다. 아주 미세하게만 조정하세요."
            
            current_state["message"] = feedback
            s_print(f"[System Feedback] {feedback}")

        # ---------------------------------------------------------
        # Step 1: LLM 시나리오 생성
        # ---------------------------------------------------------
        s_print("[LLM] 시나리오 설계 중...")
        scenario = llm.generate_scenario(current_state)
        if not scenario: break
        s_print(f"가설: {scenario.get('target_hypothesis')}")
        
        # ---------------------------------------------------------
        # Step 2: 이미지 변조 (Damping 및 정수화 로직)
        # ---------------------------------------------------------
        s_print("[Simulator] 이미지 변조 중...")
        base_img = cv2.imread(str(IMAGE_DIR / "step_1.jpg"))
        env = scenario.get("environment_parameters", {})
        
        damping = 0.05 # 공격 강도 완화 계수
        
        f = clean_val(env.get("fog_density_percent")) * damping
        n = clean_val(env.get("camera_noise_level")) * damping
        m_raw = clean_val(env.get("motion_blur_intensity")) * damping
        m = int(m_raw) 
        
        aug_list = []
        if f > 0:
            aug_list.append(A.RandomFog(alpha_coef=min(f/100.0, 1.0), p=1.0))
        if n > 0:
            aug_list.append(A.GaussNoise(p=min(n*5, 1.0))) 
        if m > 2:
            m_odd = m if m % 2 != 0 else m + 1
            aug_list.append(A.MotionBlur(blur_limit=(3, max(3, m_odd)), p=1.0))

        if aug_list:
            degraded_img = A.Compose(aug_list)(image=base_img)['image']
        else:
            degraded_img = base_img
        
        img_path = str(IMAGE_DIR / f"current_iter_{step}.jpg")
        cv2.imwrite(img_path, degraded_img)

        # ---------------------------------------------------------
        # Step 3: XAI & YOLO 분석
        # ---------------------------------------------------------
        s_print("🔍 [XAI] YOLO11x 탐지 및 SHAP 분석 중...")
        
        # 분석을 실행하면서 바운딩 박스가 그려진 이미지를 저장하도록 합니다.
        # RealXAIAnalyzer 내부에 이 기능을 추가하거나, 아래처럼 처리합니다.
        new_map50, feature_importance = xai.analyze(img_path, scenario)
        
        # 💡 [핵심 추가] 탐지 결과가 그려진 이미지를 별도로 저장
        # xai.analyze 내부에서 결과를 그리는 코드가 있다면 그 파일명을 'annotated_...'로 하시면 됩니다.
        # 만약 xai.analyze가 결과를 저장하지 않는다면, 그 안에서 results[0].save()를 호출해야 합니다.
        annotated_path = IMAGE_DIR / f"annotated_iter_{step}.jpg"
        
        # (예시: xai 객체가 마지막 결과를 저장하는 기능을 가지고 있다고 가정하거나 직접 구현)
        # xai.save_annotated_image(annotated_path) 
        
        s_print(f"  ✅ 결과 mAP50: {new_map50:.4f}")
        
        # ---------------------------------------------------------
        # Step 4: Counterfactual / Boundary 탐색 (SHAP+CF 결합)
        # ---------------------------------------------------------
        dominant_factors = _normalize_dominant_factors(feature_importance)
        cf_input = {
            "scene_id": f"iter_{step:03d}",
            "task": "uav_object_detection",
            "goal": "Find boundary-near counterfactual scenario around mAP50 threshold",
            "perception": {
                "detector": "yolo11x",
                "input_resolution": [640, 360],
                "detections": [],
            },
            "current_requirement": {
                "metric": "map50",
                "threshold": 0.85,
                "requirement_violated": bool(new_map50 < 0.85),
            },
            "scenario": scenario,
            "performance_signals": {
                "confidence_trend": "decreasing" if new_map50 < current_state.get("map50", 0.95) else "stable",
                "miss_rate_trend": "increasing" if new_map50 < 0.85 else "stable",
                "risk_score": float(max(0.0, 0.85 - new_map50)),
                "failure_type": "detection_performance_drop" if new_map50 < 0.85 else "normal_operation",
                "map50": float(new_map50),
                "threshold": 0.85,
            },
            "xai_signals": {
                "method": "yolo_xgb_shap_posthoc",
                "dominant_factors": dominant_factors,
                "attention_summary": f"iter={step}, mAP50={new_map50:.4f}",
            },
            "counterfactual_request": {
                "evaluator": "map50_proxy",
                "guidance_blend": {"xai": 0.4, "shap_global": 0.3, "shap_local": 0.3},
            },
            "scenario_constraints": {
                "allow_weather_change": True,
                "allow_lighting_change": True,
                "allow_obstacle_density_change": True,
            },
        }
        try:
            cf_output, boundary_output = generate_counterfactual_and_boundary(
                payload=cf_input,
                mode="map50_proxy",
                random_seed=42 + step,
            )
            cf_path = save_json(DATA_DIR / f"counterfactual_explanations_step_{step}.json", cf_output)
            bd_path = save_json(DATA_DIR / f"boundary_candidates_step_{step}.json", boundary_output)
            top_cf = (cf_output.get("minimal_change_candidates") or [{}])[0]
            cf_summary = str(top_cf.get("summary_explanation", "no counterfactual candidate"))
            s_print(f"  [CF] saved: {cf_path.name}, {bd_path.name}")
        except Exception as cf_err:
            cf_output, boundary_output = {}, {}
            cf_summary = f"counterfactual search failed: {cf_err}"
            s_print(f"  [CF] error: {cf_err}")

        # ---------------------------------------------------------
        # Step 5: 결과 저장 및 상태 갱신
        # ---------------------------------------------------------
        out_file = DATA_DIR / f"dashboard_step_{step}.json"
        with open(out_file, "w", encoding="utf-8") as f_out:
            json.dump({
                "iteration": step,
                "panel_1_visual": {"map50_score": new_map50, "params": env, "rendered_image": f"current_iter_{step}.jpg"},
                "panel_2_xai": feature_importance,
                "panel_3_llm": {"hypothesis": scenario.get("target_hypothesis"), "reasoning": scenario.get("llm_reasoning")},
                "panel_4_counterfactual": {
                    "summary": cf_summary,
                    "counterfactual_file": f"counterfactual_explanations_step_{step}.json",
                    "boundary_file": f"boundary_candidates_step_{step}.json",
                    "top_candidate": (cf_output.get("minimal_change_candidates") or [{}])[0],
                },
            }, f_out, indent=2, ensure_ascii=False)
            
        current_state = {
            "map50": new_map50, 
            "xai_signals": {"dominant_factors": dominant_factors},
            "message": cf_summary,
        }

    s_print("\n모든 루프가 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    run_dynamic_pipeline()
