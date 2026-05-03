# Data Folder (SI <-> XAI <-> LLM)

Input examples for simulation-grounded XAI:

- `counterfactual_case_input.json` (FAIL case with `scenario_history` + `counterfactual_replay_results`)
- `counterfactual_case_sim_dummy.json` (PASS case with degradation trend in `scenario_history`)
- `scenario_iter_001.json`
- `sim_result_iter_001.json`
- `eval_iter_001.json`

Output examples from XAI:

- `counterfactual_explanations.json`
- `boundary_candidates.json`
- `xai_input.json` (legacy sample)

Recommended optional evidence fields:

- `scenario_history`
- `counterfactual_replay_results` or `counterfactual_replays`

These fields improve attribution confidence without changing LLM/SI execution code.
