from simulator import Simulator
from xai.dummy_analyzer import analyze_xai_dummy, get_example_sim_log


def generate_params_dummy() -> dict:
    return {"wind_speed": 4.5, "delay": 0.1}


def main() -> None:
    print("[LLM] Generating test parameters...")
    params = generate_params_dummy()
    print(f"  params: {params}")

    print("[Simulator] Running...")
    sim = Simulator()
    result = sim.run_sim_dummy(params)
    print(f"  result: {result}")

    print("[XAI] Analyzing outcome...")
    xai_out = analyze_xai_dummy(result)
    print(f"  xai: {xai_out}")

    print("[XAI] Example input test...")
    example_sim_log = get_example_sim_log()
    example_xai_out = analyze_xai_dummy(example_sim_log)
    print(f"  example_sim_log: {example_sim_log}")
    print(f"  example_xai: {example_xai_out}")

    print("[Pipeline] Done: LLM -> Simulator -> XAI")


if __name__ == "__main__":
    main()
