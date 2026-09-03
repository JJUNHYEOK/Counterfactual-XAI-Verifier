function render_registered_dataset(plan_path, output_root)
%RENDER_REGISTERED_DATASET Render every scenario in an immutable JSON plan.
plan = jsondecode(fileread(plan_path));
for idx = 1:numel(plan.scenarios)
    scenario = plan.scenarios(idx);
    output_dir = fullfile(output_root, 'rendered_scenarios', scenario.scenario_id);
    manifest_path = fullfile(output_dir, 'frame_manifest.json');
    if isfile(manifest_path)
        fprintf('REUSE %s\n', scenario.scenario_id);
        continue;
    end
    export_training_scenario_frames(output_dir, scenario.scenario_config_path, scenario.scenario_config_sha256);
    fprintf('RENDERED %s (%d/%d)\n', scenario.scenario_id, idx, numel(plan.scenarios));
end
end

