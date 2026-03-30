import time


class Simulator:
    def run_sim_dummy(self, params: dict) -> dict:
        wind = float(params.get("wind_speed", 0))
        if wind >= 6.0:
            return {
                "status": "failure",
                "collision": True,
                "message": "Wind too strong",
            }
        return {
            "status": "success",
            "collision": False,
            "message": "Conditions within safe range",
        }
