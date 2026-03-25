from simulator import Simulator


def generate_params_dummy() -> dict:
    return {"wind_speed": 4.5, "delay": 0.1}


def analyze_xai_dummy(result: dict) -> dict:
    return {"feature_importance": {"wind_speed": 0.82, "delay": 0.18}}


def main() -> None:
    print("[LLM] Generating test parameters...")
    params = generate_params_dummy()
    print(f"  params: {params}")

    print("[Simulator] Running...")
    sim = Simulator()
    result = sim.run(params)
    print(f"  result: {result}")

    print("[XAI] Analyzing outcome...")
    xai_out = analyze_xai_dummy(result)
    print(f"  xai: {xai_out}")

    print("[Pipeline] Done: LLM -> Simulator -> XAI")


if __name__ == "__main__":
    main()
