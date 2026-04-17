import numpy as np
import xgboost as xgb
import shap
import os  # <--- 경로 처리를 위해 추가
from ultralytics import YOLO

def absolute_cleaner(val):
    try:
        s = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
        return float(s)
    except:
        return 0.0

class RealXAIAnalyzer:
    def __init__(self):
        # 모델 로드 (최초 실행 시 다운로드될 수 있음)
        self.yolo_model = YOLO("yolo11x.pt")
        self.history_X = [[0.0, 5000.0, 0.0, 0.0, 0.0], [100.0, 100.0, 1.0, 20.0, 10.0]]
        self.history_y = [0.95, 0.10]

    def analyze(self, image_path: str, scenario_data: dict):
        # 1. YOLO 객체 탐지 수행
        results = self.yolo_model(image_path, verbose=False)
        
        # ---------------------------------------------------------
        # [신규 추가] 바운딩 박스 이미지 저장 로직
        # ---------------------------------------------------------
        # [수정된 바운딩 박스 이미지 저장 로직] 대시보드 실시간 출력용
        try:
            base_name = os.path.basename(image_path)
            
            # 💡 [버그 수정] 원본 파일을 덮어쓰지 않도록 예외 처리
            if "current" in base_name:
                annotated_name = base_name.replace("current", "annotated")
            else:
                annotated_name = f"annotated_{base_name}" # 예: annotated_step_1.jpg
                
            save_dir = os.path.dirname(image_path)
            annotated_path = os.path.join(save_dir, annotated_name)
            results[0].save(filename=annotated_path) 
        except Exception as e:
            print(f"⚠️ 이미지 저장 실패: {e}")
        # ---------------------------------------------------------

        # 2. 성능 지표(mAP50 대용) 계산
        boxes = results[0].boxes
        current_map50 = float(np.mean(boxes.conf.cpu().numpy())) if len(boxes) > 0 else 0.10

        # 3. 환경 파라미터 파싱
        env = scenario_data.get("environment_parameters", {})
        
        # 기존 클리너 사용
        fog = absolute_cleaner(env.get("fog_density_percent", 0.0))
        lux = absolute_cleaner(env.get("illumination_lux", 5000.0))
        noise = absolute_cleaner(env.get("camera_noise_level", 0.0))
        m_blur = absolute_cleaner(env.get("motion_blur_intensity", 0.0))
        z_blur = absolute_cleaner(env.get("zoom_blur_intensity", 0.0))
        
        current_X = [fog, lux, noise, m_blur, z_blur]
        self.history_X.append(current_X)
        self.history_y.append(current_map50)

        # 4. XGBoost & SHAP 분석
        try:
            X_arr = np.array(self.history_X, dtype=np.float64)
            y_arr = np.array(self.history_y, dtype=np.float64)
            
            model = xgb.XGBRegressor(n_estimators=30).fit(X_arr, y_arr)
            explainer = shap.TreeExplainer(model)
            shap_v = np.abs(explainer.shap_values(X_arr)[-1])
            
            total = np.sum(shap_v) + 1e-9
            feature_importance = [
                {"name": "fog", "importance": float(shap_v[0]/total)},
                {"name": "low_light", "importance": float(shap_v[1]/total)},
                {"name": "noise", "importance": float(shap_v[2]/total)},
                {"name": "motion_blur", "importance": float(shap_v[3]/total)},
                {"name": "zoom_blur", "importance": float(shap_v[4]/total)}
            ]
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        except Exception as e:
            print(f"⚠️ SHAP 분석 에러: {e}")
            feature_importance = [{"name": "error_fallback", "importance": 1.0}]
            
        return current_map50, feature_importance