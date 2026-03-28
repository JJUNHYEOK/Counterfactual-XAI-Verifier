import time


class Simulator:
    def run_sim_dummy(self, params: dict) -> dict:
        time.sleep(2)
        wind = float(params.get("wind_speed", 0))
        if wind >= 6.0:
            return {
                "status": "failure",
                "min_distance": 1.2,
                "wind_speed": wind,
                "collision": True,
                "message": "Wind too strong",
            }
        return {
            "status": "success",
            "min_distance": 2.4,
            "wind_speed": wind,
            "collision": False,
            "message": "Conditions within safe range",
        }
