import numpy as np
import xgboost as xgb
import shap
from ultralytics import YOLO

def absolute_cleaner(val):
    # 대괄호, 따옴표 등 온갖 쓰레기 문자를 다 날리고 숫자만 남깁니다.
    try:
        s = str(val).replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
        return float(s)
    except:
        return 0.0

class RealXAIAnalyzer:
    def __init__(self):
        self.yolo_model = YOLO("yolo11x.pt")
        # 무조건 2개의 초기 데이터가 있어야 XGBoost가 에러를 안 냅니다.
        self.history_X = [[0.0, 5000.0, 0.0, 0.0, 0.0], [100.0, 100.0, 1.0, 20.0, 10.0]]
        self.history_y = [0.95, 0.10]

    def analyze(self, image_path: str, scenario_data: dict):
        # 1. YOLO 객체 탐지
        results = self.yolo_model(image_path, verbose=False)
        boxes = results[0].boxes
        current_map50 = float(np.mean(boxes.conf.cpu().numpy())) if len(boxes) > 0 else 0.10

        # 2. 파라미터 파싱 (지독한 클리너 적용)
        env = scenario_data.get("environment_parameters", {})
        
        fog = absolute_cleaner(env.get("weather_conditions", {}).get("fog_density_percent", 0.0))
        lux = absolute_cleaner(env.get("sensor_noise", {}).get("low_contrast_factor", 5000.0))
        noise = absolute_cleaner(env.get("sensor_noise", {}).get("gaussian_noise_level", 0.0))
        m_blur = absolute_cleaner(env.get("uav_blur_effects", {}).get("motion_blur_intensity", 0.0))
        z_blur = absolute_cleaner(env.get("uav_blur_effects", {}).get("zoom_blur_intensity", 0.0))
        
        current_X = [fog, lux, noise, m_blur, z_blur]
        self.history_X.append(current_X)
        self.history_y.append(current_map50)

        # 3. XGBoost & SHAP 훈련
        try:
            X_arr = np.array(self.history_X, dtype=np.float64)
            y_arr = np.array(self.history_y, dtype=np.float64)
            
            model = xgb.XGBRegressor(n_estimators=30).fit(X_arr, y_arr)
            explainer = shap.TreeExplainer(model)
            shap_v = np.abs(explainer.shap_values(X_arr)[-1])
            
            total = np.sum(shap_v) + 1e-9
            feature_importance = [
                {"name": "fog", "importance": float(shap_v[0]/total)},
                {"name": "low_contrast", "importance": float(shap_v[1]/total)},
                {"name": "noise", "importance": float(shap_v[2]/total)},
                {"name": "motion_blur", "importance": float(shap_v[3]/total)},
                {"name": "zoom_blur", "importance": float(shap_v[4]/total)}
            ]
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        except Exception as e:
            print(f"⚠️ SHAP 분석 에러 (무시하고 진행): {e}")
            feature_importance = [{"name": "error_fallback", "importance": 1.0}]
            
        return current_map50, feature_importance