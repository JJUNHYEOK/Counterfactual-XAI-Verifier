import json

from simulator import Simulator
from xai.dummy_analyzer import analyze_xai_dummy, get_example_sim_log


def generate_params_dummy() -> dict:
    return {
        "scenario_id": "sim_run_001",
        "wind_speed": 7.2,
        "delay": 1.4,
        "obstacle_density": 0.55,
    }


def main() -> None:
    print("[LLM] Generating test parameters...")
    params = generate_params_dummy()
    print(json.dumps(params, indent=2))

    print("[Simulator] Running...")
    sim = Simulator()
    result = sim.run_sim_dummy(params)
    print(json.dumps(result, indent=2))

    print("[XAI] Analyzing outcome...")
    xai_out = analyze_xai_dummy(result)
    print(json.dumps(xai_out, indent=2))

    print("[XAI] Example input test...")
    example_sim_log = get_example_sim_log()
    example_xai_out = analyze_xai_dummy(example_sim_log)
    print(json.dumps(example_sim_log, indent=2))
    print(json.dumps(example_xai_out, indent=2))

    print("[Pipeline] Done: LLM -> Simulator -> XAI")


if __name__ == "__main__":
    main()
