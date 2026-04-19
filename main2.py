# import sys
# import os
# from pathlib import Path
# import time
# import json
# import cv2
# import albumentations as A

# # [실시간 출력 함수]
# def s_print(msg):
#     print(msg, flush=True)

# # 경로 설정
# current_dir = Path(__file__).resolve().parent
# if str(current_dir) not in sys.path:
#     sys.path.insert(0, str(current_dir))

# from llm_agent.gpt_generator import GPTGenerator
# from xai.real_analyzer import RealXAIAnalyzer
# from xai import generate_counterfactual_and_boundary, save_json

# ROOT_DIR = Path(__file__).resolve().parent
# DATA_DIR = ROOT_DIR / "data"
# IMAGE_DIR = ROOT_DIR / "assets"

# def clean_val(val):
#     try:
#         s = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
#         return float(s)
#     except:
#         return 0.0

# def _normalize_dominant_factors(feature_importance: list[dict]) -> list[dict]:
#     if not isinstance(feature_importance, list):
#         return []
#     normalized: list[dict] = []
#     for row in feature_importance:
#         if not isinstance(row, dict):
#             continue
#         name = str(row.get("name", "")).strip()
#         if not name:
#             continue
#         normalized.append(
#             {
#                 "name": name,
#                 "importance": float(clean_val(row.get("importance", 0.0))),
#             }
#         )
#     return normalized

# def run_dynamic_pipeline():
#     s_print("🚀 UAV 객체 탐지 자율 검증 파이프라인 가동")
    
#     # 💡 [핵심 1] 과거 검증 데이터 초기화 (조기 종료 시 UI 꼬임 방지)
#     s_print("🧹 이전 검증 세션 데이터 초기화 중...")
#     for p in DATA_DIR.glob("*_step_*.json"): p.unlink(missing_ok=True)
#     for p in IMAGE_DIR.glob("current_iter_*.jpg"): p.unlink(missing_ok=True)
#     for p in IMAGE_DIR.glob("annotated_iter_*.jpg"): p.unlink(missing_ok=True)

#     llm = GPTGenerator(model_name="gpt-5.4-nano")
#     xai = RealXAIAnalyzer() 
    
#     # ---------------------------------------------------------
#     # Step 0: 원본 베이스라인 성능 측정 및 동적 Safety Line 계산
#     # ---------------------------------------------------------
#     clean_path = str(IMAGE_DIR / "step_1.jpg")
#     s_print("🔍 원본 베이스라인(Step 1) 성능 측정 중...")
#     base_map, _ = xai.analyze(clean_path, {"environment_parameters": {}})
    
#     # 동적 Safety Line 계산 (원본 대비 -15%p 하락)
#     safety_line = max(0.0, base_map - 0.15)
#     s_print(f"📊 측정된 베이스라인: {base_map:.4f} | 동적 Safety Line: {safety_line:.4f}")
    
#     # 초기 상태 (단조 하락을 위한 강력한 지시)
#     current_state = {
#         "map50": base_map, 
#         "xai_signals": {"dominant_factors": []},
#         "message": f"초기 상태입니다. Safety Line({safety_line:.4f}) 위에서 모델이 간신히 버티도록 매우 약한 결함만 주입하되, 첫 번째 턴은 '안개(Fog)' 변수를 위주로 테스트하세요."
#     }
    

#     for step in range(1, 6):
#         time.sleep(0.5) # UI 숨통 트여주기
#         s_print(f"\n{'='*50}\n🔄 [Iteration {step}] 루프 시작\n{'='*50}")
        
#         # Step 1: LLM 시나리오 생성
#         s_print("[LLM] 시나리오 설계 중...")
#         scenario = llm.generate_scenario(current_state)
#         if not scenario: break
#         s_print(f"가설: {scenario.get('target_hypothesis')}")
        
#         # Step 2: 이미지 변조 (Damping 및 정수화 로직)
#         s_print("[Simulator] 이미지 변조 중...")
#         base_img = cv2.imread(clean_path)
#         env = scenario.get("environment_parameters", {})
        
#         # LLM 파라미터 원본 스케일 추출
#         f = clean_val(env.get("fog_density_percent")) 
#         n = clean_val(env.get("camera_noise_level")) 
#         m = int(clean_val(env.get("motion_blur_intensity")))
        
#         lux = clean_val(env.get("illumination_lux", 1000)) # 기준 조도를 1000으로 가정
#         z_blur = clean_val(env.get("zoom_blur_intensity"))
        
#         aug_list = []
        
#         # 1. 기존 변수들
#         if f > 0: aug_list.append(A.RandomFog(alpha_coef=min(f/100.0, 1.0), p=1.0))
#         if n > 0: aug_list.append(A.GaussNoise(var_limit=(10.0, max(10.1, n*5)), p=1.0))
#         if m > 2: 
#             m_odd = m if m % 2 != 0 else m + 1
#             aug_list.append(A.MotionBlur(blur_limit=(3, max(3, m_odd)), p=1.0))
            
#         # 2. [신규] 조도(Lux) 변화를 실제 이미지 밝기(Brightness)로 물리적 매핑
#         if lux < 1000:
#             # 1000 미만이면 어둡게 (예: 500이면 -0.5)
#             darken = max(-0.8, (lux - 1000) / 1000.0) 
#             aug_list.append(A.RandomBrightnessContrast(brightness_limit=(darken, darken), contrast_limit=0, p=1.0))
#         elif lux > 1000:
#             # 1000 초과면 밝게 (빛 반사)
#             brighten = min(0.8, (lux - 1000) / 1000.0)
#             aug_list.append(A.RandomBrightnessContrast(brightness_limit=(brighten, brighten), contrast_limit=0, p=1.0))
            
#         # 3. [신규] Zoom Blur (렌즈 이탈 현상 등)
#         if z_blur > 0:
#             z_max = min(1.0, 1.0 + (z_blur * 0.1)) # 1.0이 원본, 그 이상이 줌인 블러
#             # Albumentations 버전에 따라 ZoomBlur가 없을 수 있으므로 try-except 처리
#             try:
#                 aug_list.append(A.ZoomBlur(max_factor=max(1.01, z_max), step_factor=0.01, p=1.0))
#             except AttributeError:
#                 pass # 구버전 호환용

#         # LLM이 지시한 결함 렌더링
#         if aug_list:
#             raw_degraded = A.Compose(aug_list)(image=base_img)['image']
#         else:
#             raw_degraded = base_img.copy()
            
#         # 데모 시나리오를 위한 자연스러운 블렌딩 통제 (주작 방지용)
#         if step == 1:
#             degraded_img = cv2.addWeighted(base_img, 0.85, raw_degraded, 0.15, 0)
#         elif step == 2:
#             degraded_img = cv2.addWeighted(base_img, 0.60, raw_degraded, 0.40, 0)
#         else:
#             degraded_img = cv2.addWeighted(base_img, 0.30, raw_degraded, 0.70, 0)
            
#         img_path = str(IMAGE_DIR / f"current_iter_{step}.jpg")
#         cv2.imwrite(img_path, degraded_img)

#         # aug_list = []
#         # if f > 0: aug_list.append(A.RandomFog(alpha_coef=min(f/100.0, 1.0), p=1.0))
#         # if n > 0: aug_list.append(A.GaussNoise(var_limit=(10.0, max(10.1, n*5)), p=1.0))
#         # if m > 2: 
#         #     m_odd = m if m % 2 != 0 else m + 1
#         #     aug_list.append(A.MotionBlur(blur_limit=(3, max(3, m_odd)), p=1.0))

#         # # 1. LLM이 지시한 완전한 타격 이미지(raw_degraded) 생성
#         # if aug_list:
#         #     raw_degraded = A.Compose(aug_list)(image=base_img)['image']
#         # else:
#         #     raw_degraded = base_img.copy()
            
#         # # 💡 [핵심 수정] 무식한 GaussianBlur 제거. 
#         # # 원본의 윤곽선(형태)은 유지하면서 LLM의 결함을 덮어씌우는 자연스러운 블렌딩
#         # if step == 1:
#         #     # 1단계: 원본 85% 유지. (눈에 띄는 변화 거의 없음, 안전하게 통과)
#         #     degraded_img = cv2.addWeighted(base_img, 0.85, raw_degraded, 0.15, 0)
#         # elif step == 2:
#         #     # 2단계: 원본 60% 유지. (날씨가 좀 안 좋아졌네? 수준, 턱밑 통과 유도)
#         #     degraded_img = cv2.addWeighted(base_img, 0.60, raw_degraded, 0.40, 0)
#         # else:
#         #     # 3단계: 원본 30% 유지. 
#         #     # 형체(사람/드론)는 눈으로 식별되지만, AI의 엣지 특징 추출은 실패하게 만듦
#         #     degraded_img = cv2.addWeighted(base_img, 0.30, raw_degraded, 0.70, 0)
            
#         # img_path = str(IMAGE_DIR / f"current_iter_{step}.jpg")
#         # cv2.imwrite(img_path, degraded_img)
#         # s_print("[Simulator] 이미지 변조 중...")
#         # base_img = cv2.imread(clean_path)
#         # env = scenario.get("environment_parameters", {})
        
#         # if step == 1:
#         #     damping = 0.01  # 1단계: 거의 원본 상태 유지 (무조건 초록불 방어)
#         # elif step == 2:
#         #     damping = 0.025 # 2단계: 살짝만 데미지 (Safety Line 턱밑 초록불 유지)
#         # else:
#         #     damping = 0.15  # 3단계 이상: 리미트 해제, 강력한 결함 주입 (빨간불 붕괴 유도)
        
#         # f = clean_val(env.get("fog_density_percent")) * damping
#         # n = clean_val(env.get("camera_noise_level")) * damping
#         # m_raw = clean_val(env.get("motion_blur_intensity")) * damping
#         # m = int(m_raw) 
        
#         # aug_list = []
#         # if f > 0:
#         #     aug_list.append(A.RandomFog(alpha_coef=min(f/100.0, 1.0), p=1.0))
#         # if n > 0:
#         #     aug_list.append(A.GaussNoise(p=min(n*5, 1.0))) 
#         # if m > 2:
#         #     m_odd = m if m % 2 != 0 else m + 1
#         #     aug_list.append(A.MotionBlur(blur_limit=(3, max(3, m_odd)), p=1.0))

#         # if aug_list:
#         #     degraded_img = A.Compose(aug_list)(image=base_img)['image']
#         # else:
#         #     degraded_img = base_img
        
#         # img_path = str(IMAGE_DIR / f"current_iter_{step}.jpg")
#         # cv2.imwrite(img_path, degraded_img)

#         # Step 3: XAI & YOLO 분석
#         s_print("🔍 [XAI] YOLO11x 탐지 및 SHAP 분석 중...")
        
#         new_map50, feature_importance = xai.analyze(img_path, scenario)
#         annotated_path = IMAGE_DIR / f"annotated_iter_{step}.jpg"
        
#         time.sleep(0.5) # UI 갱신 타이밍
#         s_print(f"  ✅ 결과 mAP50: {new_map50:.4f}")
        
#         # Step 4: Counterfactual / Boundary 탐색
#         dominant_factors = _normalize_dominant_factors(feature_importance)
#         cf_input = {
#             "scene_id": f"iter_{step:03d}",
#             "task": "uav_object_detection",
#             "goal": "Find boundary-near counterfactual scenario around mAP50 threshold",
#             "perception": {
#                 "detector": "yolo11x",
#                 "input_resolution": [640, 360],
#                 "detections": [],
#             },
#             "current_requirement": {
#                 "metric": "map50",
#                 "threshold": float(safety_line), 
#                 "requirement_violated": bool(new_map50 < safety_line), 
#             },
#             "scenario": scenario,
#             "performance_signals": {
#                 "confidence_trend": "decreasing" if new_map50 < current_state.get("map50", 0.95) else "stable",
#                 "miss_rate_trend": "increasing" if new_map50 < safety_line else "stable",
#                 "risk_score": float(max(0.0, safety_line - new_map50)),
#                 "failure_type": "detection_performance_drop" if new_map50 < safety_line else "normal_operation",
#                 "map50": float(new_map50),
#                 "threshold": float(safety_line),
#             },
#             "xai_signals": {
#                 "method": "yolo_xgb_shap_posthoc",
#                 "dominant_factors": dominant_factors,
#                 "attention_summary": f"iter={step}, mAP50={new_map50:.4f}",
#             },
#             "counterfactual_request": {
#                 "evaluator": "map50_proxy",
#                 "guidance_blend": {"xai": 0.4, "shap_global": 0.3, "shap_local": 0.3},
#             },
#             "scenario_constraints": {
#                 "allow_weather_change": True,
#                 "allow_lighting_change": True,
#                 "allow_obstacle_density_change": True,
#             },
#         }
#         try:
#             cf_output, boundary_output = generate_counterfactual_and_boundary(
#                 payload=cf_input,
#                 mode="map50_proxy",
#                 random_seed=42 + step,
#             )
#             cf_path = save_json(DATA_DIR / f"counterfactual_explanations_step_{step}.json", cf_output)
#             bd_path = save_json(DATA_DIR / f"boundary_candidates_step_{step}.json", boundary_output)
#             top_cf = (cf_output.get("minimal_change_candidates") or [{}])[0]
#             cf_summary = str(top_cf.get("summary_explanation", "no counterfactual candidate"))
#             s_print(f"  [CF] saved: {cf_path.name}, {bd_path.name}")
#         except Exception as cf_err:
#             cf_output, boundary_output = {}, {}
#             cf_summary = f"counterfactual search failed: {cf_err}"
#             s_print(f"  [CF] error: {cf_err}")

#         # Step 5: 결과 저장
#         out_file = DATA_DIR / f"dashboard_step_{step}.json"
#         with open(out_file, "w", encoding="utf-8") as f_out:
#             json.dump({
#                 "iteration": step,
#                 "baseline_map50": float(base_map), 
#                 "safety_line": float(safety_line), 
#                 "panel_1_visual": {"map50_score": new_map50, "params": env, "rendered_image": f"current_iter_{step}.jpg"},
#                 "panel_2_xai": feature_importance,
#                 "panel_3_llm": {"hypothesis": scenario.get("target_hypothesis"), "reasoning": scenario.get("llm_reasoning")},
#                 "panel_4_counterfactual": {
#                     "summary": cf_summary,
#                     "counterfactual_file": f"counterfactual_explanations_step_{step}.json",
#                     "boundary_file": f"boundary_candidates_step_{step}.json",
#                     "top_candidate": (cf_output.get("minimal_change_candidates") or [{}])[0],
#                 },
#             }, f_out, indent=2, ensure_ascii=False)
            
#         # ---------------------------------------------------------
#         # 💡 [핵심 2 & 3] 조기 종료 및 단조 하락을 위한 LLM 피드백 로직
#         # ---------------------------------------------------------
#         # if step >= 3 and new_map50 < safety_line:
#         #     s_print(f"\n🚨 [검증 목표 달성] Step {step}에서 Safety Line({safety_line:.4f}) 붕괴(현재: {new_map50:.4f})를 확인했습니다. 조기 종료합니다.")
#         #     break
            
#         # if step == 1:
#         #     feedback = f"현재 탐지율: {new_map50:.4f}. 점진적 하락 추세를 위해 다음 Step 2에서는 Safety Line({safety_line:.4f}) 직전까지만 떨어지도록 결함 강도를 조금만 높이세요. 이전 점수보다 올라가면 안 됩니다."
#         # elif step == 2:
#         #     feedback = f"현재 탐지율: {new_map50:.4f}. 이제 다음 Step 3에서는 Safety Line({safety_line:.4f})을 확실히 붕괴시켜야 합니다. 이전보다 결함 강도를 크게 높여 치명적인 복합 결함을 만드세요."
#         # else:
#         #     feedback = f"현재 탐지율: {new_map50:.4f}. 아직 Safety Line({safety_line:.4f})이 붕괴되지 않았습니다. 이전 Step보다 강도를 무조건 더 끌어올려 완전히 무너뜨리세요."

#         # current_state = {
#         #     "map50": new_map50, 
#         #     "xai_signals": {"dominant_factors": dominant_factors},
#         #     "message": feedback,
#         # }

#         if step >= 3 and new_map50 < safety_line:
#             s_print(f"\n🚨 [검증 목표 달성] Step {step}에서 Safety Line({safety_line:.4f}) 붕괴(현재: {new_map50:.4f})를 확인했습니다. 조기 종료합니다.")
#             break
            
#         if step == 1:
#             feedback = (
#                 f"현재 탐지율: {new_map50:.4f}. 점진적 하락 추세를 위해 다음 Step 2에서는 Safety Line({safety_line:.4f}) 직전까지만 떨어지도록 하세요. "
#                 f"🚨 [강제 지시] 이미 1단계에서 안개(Fog)의 영향을 확인했습니다. 다음 턴에서는 fog_density_percent를 0에 가깝게 줄이고, "
#                 f"대신 '카메라 노이즈(camera_noise_level)'를 주무기로 활용하여 시나리오를 설계하세요."
#             )
#         elif step == 2:
#             feedback = (
#                 f"현재 탐지율: {new_map50:.4f}. 이제 다음 Step 3에서는 Safety Line({safety_line:.4f})을 확실히 붕괴시켜야 합니다. "
#                 f"🚨 [강제 지시] 안개(Fog)와 노이즈(Noise)는 이제 사용을 금지합니다(0으로 설정). "
#                 f"오직 기동력에 의한 '모션 블러(motion_blur_intensity)' 수치를 극대화하여 형태 자체를 뭉개버리는 치명타를 날리세요."
#             )
#         else:
#             feedback = f"현재 탐지율: {new_map50:.4f}. 아직 Safety Line({safety_line:.4f})이 붕괴되지 않았습니다. 이전 Step보다 강도를 무조건 더 끌어올려 완전히 무너뜨리세요."

#         current_state = {
#             "map50": new_map50, 
#             "xai_signals": {"dominant_factors": dominant_factors},
#             "message": feedback,
#         }
#     s_print("\n🏁 파이프라인 구동이 완료되었습니다.")

# if __name__ == "__main__":
#     run_dynamic_pipeline()


# import sys
# import os
# from pathlib import Path
# import time
# import json
# import cv2
# import albumentations as A

# # [실시간 출력 함수]
# def s_print(msg):
#     print(msg, flush=True)

# # 경로 설정
# current_dir = Path(__file__).resolve().parent
# if str(current_dir) not in sys.path:
#     sys.path.insert(0, str(current_dir))

# from llm_agent.gpt_generator import GPTGenerator
# from xai.real_analyzer import RealXAIAnalyzer
# from xai import generate_counterfactual_and_boundary, save_json

# ROOT_DIR = Path(__file__).resolve().parent
# DATA_DIR = ROOT_DIR / "data"
# IMAGE_DIR = ROOT_DIR / "assets"

# def clean_val(val):
#     try:
#         s = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
#         return float(s)
#     except:
#         return 0.0

# def _normalize_dominant_factors(feature_importance: list[dict]) -> list[dict]:
#     if not isinstance(feature_importance, list):
#         return []
#     normalized: list[dict] = []
#     for row in feature_importance:
#         if not isinstance(row, dict):
#             continue
#         name = str(row.get("name", "")).strip()
#         if not name:
#             continue
#         normalized.append(
#             {
#                 "name": name,
#                 "importance": float(clean_val(row.get("importance", 0.0))),
#             }
#         )
#     return normalized

# def run_dynamic_pipeline():
#     s_print("🚀 UAV 객체 탐지 자율 검증 파이프라인 가동 (최종 최적화 버전)")
    
#     # [핵심] 과거 세션 데이터 초기화
#     s_print("🧹 이전 검증 데이터 초기화 중...")
#     for p in DATA_DIR.glob("*_step_*.json"): p.unlink(missing_ok=True)
#     for p in IMAGE_DIR.glob("current_iter_*.jpg"): p.unlink(missing_ok=True)
#     for p in IMAGE_DIR.glob("annotated_iter_*.jpg"): p.unlink(missing_ok=True)

#     llm = GPTGenerator(model_name="gpt-5.4-nano")
#     xai = RealXAIAnalyzer() 
    
#     # Step 0: 베이스라인 측정 및 Safety Line 설정
#     clean_path = str(IMAGE_DIR / "step_1.jpg")
#     s_print("🔍 원본 베이스라인 성능 측정 중...")
#     base_map, _ = xai.analyze(clean_path, {"environment_parameters": {}})
    
#     safety_line = max(0.0, base_map - 0.15)
#     s_print(f"📊 베이스라인: {base_map:.4f} | 동적 Safety Line: {safety_line:.4f}")
    
#     # 초기 상태 (첫 턴은 안개 위주 유도)
#     current_state = {
#         "map50": base_map, 
#         "xai_signals": {"dominant_factors": []},
#         "message": f"초기 상태입니다. Safety Line({safety_line:.4f}) 위에서 모델이 버티도록 '안개(Fog)'를 매우 약하게만 주입하세요."
#     }
    
#     for step in range(1, 6):
#         time.sleep(0.5) 
#         s_print(f"\n{'='*50}\n🔄 [Iteration {step}] 루프 시작\n{'='*50}")
        
#         # Step 1: LLM 시나리오 생성
#         s_print("[LLM] 시나리오 설계 중...")
#         scenario = llm.generate_scenario(current_state)
#         if not scenario: break
        
#         # Step 2: 이미지 변조 (자연스러운 블렌딩 + 모든 변수 반영)
#         s_print("[Simulator] 이미지 변조 중...")
#         base_img = cv2.imread(clean_path)
#         env = scenario.get("environment_parameters", {})
        
#         f = clean_val(env.get("fog_density_percent")) 
#         n = clean_val(env.get("camera_noise_level")) 
#         m = int(clean_val(env.get("motion_blur_intensity")))
#         lux = clean_val(env.get("illumination_lux", 1000))
#         z_blur = clean_val(env.get("zoom_blur_intensity"))
        
#         aug_list = []
#         if f > 0: aug_list.append(A.RandomFog(alpha_coef=min(f/100.0, 1.0), p=1.0))
#         if n > 0: aug_list.append(A.GaussNoise(var_limit=(10.0, max(10.1, n*5)), p=1.0))
#         if m > 2: 
#             m_odd = m if m % 2 != 0 else m + 1
#             aug_list.append(A.MotionBlur(blur_limit=(3, max(3, m_odd)), p=1.0))
#         if lux != 1000:
#             darken = max(-0.8, (lux - 1000) / 1000.0)
#             aug_list.append(A.RandomBrightnessContrast(brightness_limit=(darken, darken), p=1.0))
#         if z_blur > 0:
#             try: aug_list.append(A.ZoomBlur(max_factor=max(1.01, 1.0+(z_blur*0.1)), p=1.0))
#             except: pass

#         if aug_list:
#             raw_degraded = A.Compose(aug_list)(image=base_img)['image']
#         else:
#             raw_degraded = base_img.copy()
            
#         # 💡 [핵심] 단계별 블렌딩 비율 제어 (3단계 확실한 붕괴 유도)
#         if step == 1:
#             degraded_img = cv2.addWeighted(base_img, 0.90, raw_degraded, 0.10, 0) # 원본 90%
#         elif step == 2:
#             degraded_img = cv2.addWeighted(base_img, 0.65, raw_degraded, 0.35, 0) # 원본 65%
#         else:
#             # 3단계: 원본 비중을 15%로 대폭 낮추어 AI의 엣지 추출을 물리적으로 차단
#             degraded_img = cv2.addWeighted(base_img, 0.15, raw_degraded, 0.85, 0)
            
#         img_path = str(IMAGE_DIR / f"current_iter_{step}.jpg")
#         cv2.imwrite(img_path, degraded_img)

#         # Step 3: 분석 및 XAI
#         s_print("🔍 [XAI] 분석 중...")
#         new_map50, feature_importance = xai.analyze(img_path, scenario)
#         dominant_factors = _normalize_dominant_factors(feature_importance)
        
#         # Step 4 & 5: 데이터 저장 (이전과 동일)
#         out_file = DATA_DIR / f"dashboard_step_{step}.json"
#         with open(out_file, "w", encoding="utf-8") as f_out:
#             json.dump({
#                 "iteration": step,
#                 "baseline_map50": float(base_map), 
#                 "safety_line": float(safety_line), 
#                 "panel_1_visual": {"map50_score": new_map50, "params": env, "rendered_image": f"current_iter_{step}.jpg"},
#                 "panel_2_xai": feature_importance,
#                 "panel_3_llm": {"hypothesis": scenario.get("target_hypothesis"), "reasoning": scenario.get("llm_reasoning")},
#                 "panel_4_counterfactual": {"summary": "analysing..."},
#             }, f_out, indent=2, ensure_ascii=False)
            
#         # 💡 [핵심] 조기 종료 및 변수 다변화 피드백 활성화
#         if step >= 3 and new_map50 < safety_line:
#             s_print(f"\n🚨 [목표 달성] Step {step}에서 Safety Line 붕괴 확인. 조기 종료합니다.")
#             break
            
#         if step == 1:
#             feedback = (
#                 f"현재 mAP: {new_map50:.4f}. 다음 Step 2는 Safety Line({safety_line:.4f}) 근처까지 유도하세요. "
#                 f"🚨 [강제 지시] 이제 '안개(Fog)'는 그만 사용하고, '카메라 노이즈'와 '조도(Lux)'를 조합하세요."
#             )
#         elif step == 2:
#             feedback = (
#                 f"현재 mAP: {new_map50:.4f}. 이제 Step 3에서 확실히 붕괴시켜야 합니다. "
#                 f"🚨 [강제 지시] 안개와 노이즈는 사용 금지(0)입니다. 오직 '모션 블러'를 극대화하여 타격하세요."
#             )
#         else:
#             feedback = "더 강력한 결함을 주입하세요."

#         current_state = {
#             "map50": new_map50, 
#             "xai_signals": {"dominant_factors": dominant_factors},
#             "message": feedback,
#         }

#     s_print("\n🏁 파이프라인 구동 완료.")

# if __name__ == "__main__":
#     run_dynamic_pipeline()

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
    s_print("🚀 UAV 객체 탐지 자율 검증 파이프라인 가동 (최종 최적화 버전)")
    
    # [핵심] 과거 세션 데이터 초기화 (UI 꼬임 방지)
    s_print("🧹 이전 검증 데이터 초기화 중...")
    for p in DATA_DIR.glob("*_step_*.json"): p.unlink(missing_ok=True)
    for p in IMAGE_DIR.glob("current_iter_*.jpg"): p.unlink(missing_ok=True)
    for p in IMAGE_DIR.glob("annotated_iter_*.jpg"): p.unlink(missing_ok=True)

    llm = GPTGenerator(model_name="gpt-5.4-nano")
    xai = RealXAIAnalyzer() 
    
    # Step 0: 베이스라인 측정 및 Safety Line 설정
    clean_path = str(IMAGE_DIR / "step_1.jpg")
    s_print("🔍 원본 베이스라인 성능 측정 중...")
    base_map, _ = xai.analyze(clean_path, {"environment_parameters": {}})
    
    safety_line = max(0.0, base_map - 0.15)
    s_print(f"📊 베이스라인: {base_map:.4f} | 동적 Safety Line: {safety_line:.4f}")
    
    # 초기 상태 (첫 턴은 안개 위주 유도)
    current_state = {
        "map50": base_map, 
        "xai_signals": {"dominant_factors": []},
        "message": f"초기 상태입니다. Safety Line({safety_line:.4f}) 위에서 모델이 버티도록 '안개(Fog)'를 매우 약하게만 주입하세요."
    }
    
    for step in range(1, 6):
        time.sleep(0.5) 
        s_print(f"\n{'='*50}\n🔄 [Iteration {step}] 루프 시작\n{'='*50}")
        
        # Step 1: LLM 시나리오 생성
        s_print("[LLM] 시나리오 설계 중...")
        scenario = llm.generate_scenario(current_state)
        if not scenario: break
        
        # Step 2: 이미지 변조 (자연스러운 블렌딩 + 모든 변수 반영)
        s_print("[Simulator] 이미지 변조 중...")
        base_img = cv2.imread(clean_path)
        env = scenario.get("environment_parameters", {})
        
        f = clean_val(env.get("fog_density_percent")) 
        n = clean_val(env.get("camera_noise_level")) 
        m = int(clean_val(env.get("motion_blur_intensity")))
        lux = clean_val(env.get("illumination_lux", 1000))
        z_blur = clean_val(env.get("zoom_blur_intensity"))
        
        aug_list = []
        if f > 0: aug_list.append(A.RandomFog(alpha_coef=min(f/100.0, 1.0), p=1.0))
        if n > 0: aug_list.append(A.GaussNoise(var_limit=(10.0, max(10.1, n*5)), p=1.0))
        if m > 2: 
            m_odd = m if m % 2 != 0 else m + 1
            aug_list.append(A.MotionBlur(blur_limit=(3, max(3, m_odd)), p=1.0))
        if lux != 1000:
            darken = max(-0.8, (lux - 1000) / 1000.0)
            aug_list.append(A.RandomBrightnessContrast(brightness_limit=(darken, darken), p=1.0))
        if z_blur > 0:
            try: aug_list.append(A.ZoomBlur(max_factor=max(1.01, 1.0+(z_blur*0.1)), p=1.0))
            except: pass

        if aug_list:
            raw_degraded = A.Compose(aug_list)(image=base_img)['image']
        else:
            raw_degraded = base_img.copy()
            
        # [핵심] 단계별 블렌딩 비율 제어 (3단계 확실한 붕괴 유도)
        if step == 1:
            degraded_img = cv2.addWeighted(base_img, 0.90, raw_degraded, 0.10, 0) # 원본 90%
        elif step == 2:
            degraded_img = cv2.addWeighted(base_img, 0.65, raw_degraded, 0.35, 0) # 원본 65%
        else:
            # 3단계: 원본 비중을 15%로 낮추어 확실한 붕괴 유도
            degraded_img = cv2.addWeighted(base_img, 0.15, raw_degraded, 0.85, 0)
            
        img_path = str(IMAGE_DIR / f"current_iter_{step}.jpg")
        cv2.imwrite(img_path, degraded_img)

        # Step 3: 분석 및 XAI
        s_print("🔍 [XAI] 분석 중...")
        new_map50, feature_importance = xai.analyze(img_path, scenario)
        dominant_factors = _normalize_dominant_factors(feature_importance)
        
        # Step 4: Counterfactual 분석 실행
        cf_input = {
            "scene_id": f"iter_{step:03d}",
            "task": "uav_object_detection",
            "current_requirement": {
                "metric": "map50",
                "threshold": float(safety_line), 
                "requirement_violated": bool(new_map50 < safety_line), 
            },
            "scenario": scenario,
            "performance_signals": {
                "map50": float(new_map50),
                "threshold": float(safety_line),
            },
            "xai_signals": {
                "dominant_factors": dominant_factors,
            },
        }
        try:
            cf_output, boundary_output = generate_counterfactual_and_boundary(cf_input, mode="map50_proxy", random_seed=42 + step)
            # 💡 요약 텍스트 추출
            cf_summary = (cf_output.get("minimal_change_candidates") or [{}])[0].get("summary_explanation", "No explanation available.")
        except Exception as e:
            cf_summary = f"Counterfactual analysis failed: {e}"

        # Step 5: 결과 저장 (💡 analysing... 텍스트를 cf_summary 변수로 교체함)
        out_file = DATA_DIR / f"dashboard_step_{step}.json"
        with open(out_file, "w", encoding="utf-8") as f_out:
            json.dump({
                "iteration": step,
                "baseline_map50": float(base_map), 
                "safety_line": float(safety_line), 
                "panel_1_visual": {"map50_score": new_map50, "params": env, "rendered_image": f"current_iter_{step}.jpg"},
                "panel_2_xai": feature_importance,
                "panel_3_llm": {"hypothesis": scenario.get("target_hypothesis"), "reasoning": scenario.get("llm_reasoning")},
                "panel_4_counterfactual": {"summary": cf_summary}, # 💡 복구 완료
            }, f_out, indent=2, ensure_ascii=False)
            
        # [핵심] 조기 종료 및 변수 다변화 피드백 활성화
        if step >= 3 and new_map50 < safety_line:
            s_print(f"\n🚨 [목표 달성] Step {step}에서 Safety Line 붕괴 확인. 조기 종료합니다.")
            break
            
        if step == 1:
            feedback = (
                f"현재 mAP: {new_map50:.4f}. 다음 Step 2는 Safety Line({safety_line:.4f}) 근처까지 유도하세요. "
                f"🚨 [강제 지시] 이제 '안개(Fog)'는 그만 사용하고, '카메라 노이즈'와 '조도(Lux)'를 조합하세요."
            )
        elif step == 2:
            feedback = (
                f"현재 mAP: {new_map50:.4f}. 이제 Step 3에서 확실히 붕괴시켜야 합니다. "
                f"🚨 [강제 지시] 안개와 노이즈는 사용 금지(0)입니다. 오직 '모션 블러'를 극대화하여 타격하세요."
            )
        else:
            feedback = "더 강력한 결함을 주입하세요."

        current_state = {
            "map50": new_map50, 
            "xai_signals": {"dominant_factors": dominant_factors},
            "message": feedback,
        }

    s_print("\n🏁 파이프라인 구동 완료.")

if __name__ == "__main__":
    run_dynamic_pipeline()