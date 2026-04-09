function simIn = build_sim_input(mdl, scenario)
    simIn = Simulink.SimulationInput(mdl);

    env = scenario.environment_parameters;

    simIn = setVariable(simIn, "FOG_DENSITY_PERCENT", env.fog_density_percent);
    simIn = setVariable(simIn, "ILLUMINATION_LUX", env.illumination_lux);
    simIn = setVariable(simIn, "CAMERA_NOISE_LEVEL", env.camera_noise_level);

    % 필요하면 StopTime, solver 등도 여기서 설정 가능
    simIn = setModelParameter(simIn, "StopTime", "20");
end