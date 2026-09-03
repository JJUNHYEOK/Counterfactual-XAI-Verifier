# YOLOv8s boundary-search comparison

This experiment is isolated from the dashboard's SHAP/LLM path. YOLO receives
only the complete rendered EO image. GT is used only by the AP@0.5 scorer.

The paper run uses pixel-exact visible-object boxes from a flat instance-colour
render pass (`rendered_instance_mask_v1`). The legacy pinhole projection is
retained only as a diagnostic field because it does not exactly match MATLAB's
3-D camera renderer.

## Final measured run

- Session: `kci_yolov8s_boundary_search_v1__20260902_173932_443628__seed42`
- Selected weight: `training/yolov8s_sim_20260902/runs/full_yolov8s_seed42/weights/best.pt`
- SHA-256: `C5C549611CBE39EBCDB051863EE80F92AE0072266FEEB71A21CC7AABEBF57ECA`
- Baseline mAP@0.5: `0.9750427353718847` (`PASS`)
- Symmetric final gap: mAP `0.02924052270668906`, environment `0.010947747747747785`
- Asymmetric final gap: mAP `0.20727131582352176`, environment `0.0812765765765765`

## Reproduction

From the repository root in PowerShell:

```powershell
# Unit tests
.\.venv\Scripts\python.exe -m unittest discover -s experiments\yolov8_search_comparison\tests -v

# Recreate scenario-separated train/validation data
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.prepare_training_data `
  --plan experiments\yolov8_search_comparison\training\scenario_plan.json `
  --output-root experiments\yolov8_search_comparison\training\yolov8s_sim_20260902

# One-epoch smoke test
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.train_yolov8s `
  --data experiments\yolov8_search_comparison\training\yolov8s_sim_20260902\dataset\data_smoke.yaml `
  --output-root experiments\yolov8_search_comparison\training\yolov8s_sim_20260902\runs `
  --name smoke_epoch1 --epochs 1 --patience 1 --batch 2 --imgsz 640 --seed 42

# Full training
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.train_yolov8s `
  --data experiments\yolov8_search_comparison\training\yolov8s_sim_20260902\data.yaml `
  --output-root experiments\yolov8_search_comparison\training\yolov8s_sim_20260902\runs `
  --name full_yolov8s_seed42 --epochs 50 --patience 12 --batch 4 --imgsz 640 --seed 42

# Held-out baseline validation
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.validate_baseline `
  --evaluation experiments\yolov8_search_comparison\baseline_validation\raw\_shared_evaluations\cache_20260902_121729_173239\ed54cdee262b7169997e\evaluation.json `
  --weights experiments\yolov8_search_comparison\training\yolov8s_sim_20260902\runs\full_yolov8s_seed42\weights\best.pt `
  --output-dir experiments\yolov8_search_comparison\training\yolov8s_sim_20260902\baseline_validation_20260902_1521

# Ten evaluations per search rule
.\.venv\Scripts\python.exe -m experiments.yolov8_search_comparison.run_comparison `
  --config experiments\yolov8_search_comparison\config\experiment_config.json `
  --weights experiments\yolov8_search_comparison\training\yolov8s_sim_20260902\runs\full_yolov8s_seed42\weights\best.pt `
  --search-method both --max-evaluations 10
```

Rendered boundary frames are transient when `retain_rendered_frames=false`;
GT, predictions, timings, and model metadata remain in content-addressed
`evaluation.json` files. Identical scenario/environment/seed/model inputs share
one cache key across both methods.

## Class mapping

Mapping is resolved from `model.names`, not fixed IDs. COCO models map person
and merge car/motorcycle/bus/truck/van to vehicle. The custom model maps
`0: person` and `1: vehicle` directly.
