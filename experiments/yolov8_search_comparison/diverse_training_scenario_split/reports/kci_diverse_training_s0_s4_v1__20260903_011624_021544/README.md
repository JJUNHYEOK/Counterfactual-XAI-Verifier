# Diverse training scenario split experiment

Plan: `diverse_training_scenario_split_v2`  
Training run: `diverse_yolov8s_seed42__20260903_010327_583537`  
Evaluation session: `kci_diverse_training_s0_s4_v1__20260903_011624_021544`

Run from the repository root in PowerShell. The commands below reproduce the run in an artifact-free checkout. This working tree intentionally refuses to overwrite the completed plan/run paths, so use a clean copy or assign new plan/run IDs for another execution. MATLAB batch rendering is shown explicitly because it was the successful renderer used for this run.

```powershell
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.diverse_training_scenario_split.plan
& 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe' -batch "addpath(fullfile(pwd,'experiments','yolov8_search_comparison','diverse_training_scenario_split')); render_registered_dataset"
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.diverse_training_scenario_split.prepare_dataset --skip-render
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.diverse_training_scenario_split.train_diverse
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.diverse_training_scenario_split.configure_evaluation --training-summary "C:\Users\lab\Counterfactual-XAI-Verifier\experiments\yolov8_search_comparison\diverse_training_scenario_split\training\diverse_training_scenario_split_v2\diverse_yolov8s_seed42__20260903_010327_583537_training_summary.json"
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.run_experiment --config "C:\Users\lab\Counterfactual-XAI-Verifier\experiments\yolov8_search_comparison\diverse_training_scenario_split\evaluation\config\evaluation_config__diverse_yolov8s_seed42__20260903_010327_583537.json" --plan "C:\Users\lab\Counterfactual-XAI-Verifier\experiments\yolov8_search_comparison\multi_scenario_symmetric\config\multi_scenario_symmetric_plan_v2\scenario_plan.json" --scenario-root "C:\Users\lab\Counterfactual-XAI-Verifier\experiments\yolov8_search_comparison\multi_scenario_symmetric\scenarios" --runs-root "C:\Users\lab\Counterfactual-XAI-Verifier\experiments\yolov8_search_comparison\diverse_training_scenario_split\evaluation\runs" --aggregated-root "C:\Users\lab\Counterfactual-XAI-Verifier\experiments\yolov8_search_comparison\diverse_training_scenario_split\evaluation\aggregated"
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.diverse_training_scenario_split.report_and_validate --session "kci_diverse_training_s0_s4_v1__20260903_011624_021544" --training-summary "C:\Users\lab\Counterfactual-XAI-Verifier\experiments\yolov8_search_comparison\diverse_training_scenario_split\training\diverse_training_scenario_split_v2\diverse_yolov8s_seed42__20260903_010327_583537_training_summary.json" --evaluation-config "C:\Users\lab\Counterfactual-XAI-Verifier\experiments\yolov8_search_comparison\diverse_training_scenario_split\evaluation\config\evaluation_config__diverse_yolov8s_seed42__20260903_010327_583537.json"
```
