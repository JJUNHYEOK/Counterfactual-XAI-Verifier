# Data Folder (SI <-> XAI <-> LLM)

This folder uses the same file naming style already used in the SI branch.

Input examples for XAI:
- `scenario_iter_001.json`
- `sim_result_iter_001.json`
- `eval_iter_001.json`
- `counterfactual_case_input.json` (counterfactual / boundary 탐색용 샘플)
- `counterfactual_case_sim_dummy.json` (simulator dummy 기반 탐색 샘플)

Output example from XAI:
- `xai_input.json`

Keep these names for easier merge between `MATLAB/Simulink/SI`, `XAI`, and `LLM`.
