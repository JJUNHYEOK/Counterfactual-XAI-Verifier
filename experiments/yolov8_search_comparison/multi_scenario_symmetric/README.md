# Five-scenario symmetric boundary-search experiment

This directory contains the fixed-weight S0--S4 experiment. It uses only the
symmetric 50:50 boundary policy. The registered v2 plan is immutable after
the pre-inference GT and leakage gate.

## Fixed inputs

- Weights: `../training/yolov8s_sim_20260902/runs/full_yolov8s_seed42/weights/best.pt`
- SHA-256: `C5C549611CBE39EBCDB051863EE80F92AE0072266FEEB71A21CC7AABEBF57ECA`
- Classes: `0: person`, `1: vehicle`
- GT: `rendered_instance_mask_v1`
- Frames: 181 per evaluation
- Initial environment: fog 5%, illumination 12,000 lx, noise 0.02
- Verdicts: PASS >= 0.50; MARGINAL >= 0.25 and < 0.50; FAIL < 0.25

PASS and MARGINAL are the non-failure side. The reported boundary is the
interval between the final non-failure and FAIL anchors, not a single point.

## Reproduction

Run from the repository root. Because completed plans and result folders are
never overwritten, assign a new plan ID in `prepare_scenarios.py` for a new
independent study run.

```powershell
.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.prepare_scenarios
.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.run_experiment
.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.reverify_cases --session <session-id>
.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.report_results --session <session-id>
.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.validate_outputs --session <session-id>
.venv\Scripts\python.exe -m unittest discover -s experiments\yolov8_search_comparison\tests -v
.venv\Scripts\python.exe -m unittest discover -s experiments\yolov8_search_comparison\multi_scenario_symmetric\tests -t . -v
git diff --check
```

Interrupted experiment runs can be resumed only within their own evaluation
stores:

```powershell
.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.run_experiment --resume-session <session-id>
.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.multi_scenario_symmetric.reverify_cases --session <session-id> --resume
```

## Completed result

The completed session is
`kci_multi_scenario_symmetric_v1__20260902_201715_587801`. All five scenarios
passed initially and all five produced valid non-failure--FAIL intervals in
10 evaluations. The 15 independently rendered revalidation cases all matched
their expected verdicts.

