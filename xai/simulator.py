import time


class Simulator:
    def run_sim_dummy(self, params: dict) -> dict:
        time.sleep(2)
        scenario_id = str(params.get("scenario_id", "sim_run_001"))
        wind_speed = float(params.get("wind_speed", 0.0))
        delay = float(params.get("delay", 0.0))
        obstacle_density = float(params.get("obstacle_density", 0.0))

        wind_norm = min(max(wind_speed / 10.0, 0.0), 1.0)
        delay_norm = min(max(delay / 5.0, 0.0), 1.0)
        obstacle_norm = min(max(obstacle_density, 0.0), 1.0)

        risk_score = (0.50 * wind_norm) + (0.30 * delay_norm) + (0.20 * obstacle_norm)
        min_distance = 3.2 - (1.8 * wind_norm + 1.0 * delay_norm + 0.9 * obstacle_norm)
        min_distance = round(max(0.4, min_distance), 2)

        collision = (
            min_distance < 1.0
            or risk_score >= 0.78
            or (wind_speed >= 8.0 and obstacle_density >= 0.7)
        )
        mission_completed = not collision and risk_score < 0.90
        status = "FAIL" if collision or not mission_completed else "SUCCESS"

        if status == "FAIL":
            message = "Mission failed due to high combined environmental risk."
        elif risk_score >= 0.45 or min_distance < 1.8:
            message = "Mission completed with narrow safety margin."
        else:
            message = "Mission completed in safe operating conditions."

        return {
            "scenario_id": scenario_id,
            "status": status,
            "min_distance": min_distance,
            "wind_speed": round(wind_speed, 2),
            "delay": round(delay, 2),
            "obstacle_density": round(obstacle_density, 2),
            "collision": collision,
            "mission_completed": mission_completed,
            "message": message,
        }
