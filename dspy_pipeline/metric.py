"""DSPy Metric: adversarial reward signal for UAV scenario evaluation.

The metric calls the Simulink bridge on each LLM-generated prediction and
returns a scalar in [0.0, 1.0]:

  1.0     — All 3 requirements violated  → maximum adversarial success
  0.85    — 2 requirements violated
  0.70    — 1 requirement violated
  0.0–0.4 — Mission passed; partial credit for proximity to failure boundary
  0.0     — Invalid prediction (parse error / exception)

DSPy optimizers (BootstrapFewShot, MIPROv2) call this function for every
(example, prediction) pair during the compilation phase.  High-scoring
predictions become few-shot demonstrations that steer the LLM toward
generating more adversarial scenarios.

Typical setup:
    from dspy_pipeline.metric import init_metric_bridge, simulink_adversarial_metric

    init_metric_bridge('/path/to/project', mode='mock')   # or 'engine'

    optimizer = BootstrapFewShot(metric=simulink_adversarial_metric)
    compiled  = optimizer.compile(module, trainset=trainset)
"""

from __future__ import annotations

import dspy

from dspy_pipeline.matlab_bridge import MatlabSimulinkBridge, SimulationResult

# Module-level bridge singleton (shared across all metric calls in one run)
_bridge: MatlabSimulinkBridge | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Bridge lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def init_metric_bridge(
    model_dir: str,
    mode: str = "auto",
) -> MatlabSimulinkBridge:
    """Initialise (and start) the global Simulink bridge used by the metric.

    Must be called once before any metric evaluations.

    Args:
        model_dir: Absolute path to the Simulink project root.
        mode:      'engine' | 'subprocess' | 'mock' | 'auto'

    Returns:
        The started MatlabSimulinkBridge instance.
    """
    global _bridge
    _bridge = MatlabSimulinkBridge(model_dir, mode=mode)
    _bridge.start()
    print(f"[Metric] Bridge ready  mode={_bridge.mode}")
    return _bridge


def get_bridge() -> MatlabSimulinkBridge:
    if _bridge is None:
        raise RuntimeError(
            "Metric bridge not initialised. Call init_metric_bridge() first."
        )
    return _bridge


# ─────────────────────────────────────────────────────────────────────────────
# DSPy metric function
# ─────────────────────────────────────────────────────────────────────────────

def simulink_adversarial_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """DSPy metric: reward the LLM for generating mission-failure scenarios.

    Implements the adversarial testbed objective:
      "Find environment variable combinations where the UAV fails to detect
       all 5 objects (Mission_Success = 0)."

    The metric feeds the LLM-generated environment parameters directly into
    the Simulink bridge and scores the result:

      Adversarial success (any requirement violated):
        3 violations → 1.00
        2 violations → 0.85
        1 violation  → 0.70

      Mission passed (no violation):
        Partial credit based on mAP50 distance from the 0.85 threshold:
          mAP50 = 0.85 → 0.40  (right at the boundary)
          mAP50 = 1.00 → 0.00  (safely above threshold)

    Args:
        example:    DSPy training example (input fields only; no gold label needed)
        prediction: Output of AdversarialScenarioGenerator.forward()
        trace:      DSPy optimiser trace (pass-through, not used)

    Returns:
        float in [0.0, 1.0]
    """
    # Extract environment parameters from the prediction
    env_params = getattr(prediction, "environment_parameters", None)
    if not isinstance(env_params, dict):
        return 0.0

    try:
        result: SimulationResult = get_bridge().run_simulation(env_params)
    except Exception as exc:
        print(f"[Metric] Simulation error: {exc}")
        return 0.0

    score = _compute_score(result)

    # Verbose logging during optimisation (trace is truthy when bootstrapping)
    if trace:
        status = "FAIL" if not result.all_passed else "PASS"
        print(
            f"[Metric] fog={env_params.get('fog_density_percent',0):.1f}% "
            f"illum={env_params.get('illumination_lux',0):.0f}lx "
            f"noise={env_params.get('camera_noise_level',0):.3f} "
            f"→ mAP50={result.map50:.4f} "
            f"violated={result.violated_count} "
            f"[{status}] score={score:.3f}"
        )

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Internal scoring helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(result: SimulationResult) -> float:
    """Convert a SimulationResult into a scalar reward in [0.0, 1.0]."""
    if not result.all_passed:
        # Adversarial success: scale with violation count
        violation_reward = {1: 0.70, 2: 0.85, 3: 1.00}
        return violation_reward.get(min(result.violated_count, 3), 0.70)

    # Mission passed: partial credit for being close to the failure boundary.
    # mAP50 margin above threshold: 0 → 0.40 score; 0.15+ → 0.0 score.
    margin = result.map50 - MatlabSimulinkBridge.MAP_THRESHOLD
    proximity = max(0.0, 0.40 - margin * (0.40 / 0.15))
    return round(proximity, 4)
