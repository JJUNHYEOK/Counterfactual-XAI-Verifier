function xaiInput = dummy_xai(scenario, analysis, iter)
    xaiInput.scenario_id = sprintf("iter_%03d", iter);
    xaiInput.current_requirement.target = "collision_free_navigation";
    xaiInput.current_requirement.threshold = 1;
    xaiInput.current_requirement.actual = double(analysis.requirement_satisfied);

    xaiInput.current_environment = scenario.environment_parameters;

    xaiInput.xai_analysis.feature_importance = struct( ...
        "fog_density_percent", 0.7, ...
        "illumination_lux", 0.3);

    xaiInput.xai_analysis.insight = analysis.summary;
end