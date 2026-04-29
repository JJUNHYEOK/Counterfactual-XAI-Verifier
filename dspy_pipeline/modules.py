"""DSPy Module: AdversarialScenarioGenerator.

Wraps UAVAdversarialScenario with ChainOfThought so that DSPy optimizers
(BootstrapFewShot, MIPROv2) can automatically improve the prompts using
Simulink simulation feedback as the reward signal.
"""

from __future__ import annotations

import json
import re
from typing import Any

import dspy

from dspy_pipeline.signatures import UAVAdversarialScenario

# Parameter bounds (mirror build_mountain_uav_model.m / gpt_for_simulink.py)
_BOUNDS: dict[str, tuple[float, float]] = {
    "fog_density_percent": (0.0, 100.0),
    "illumination_lux":    (200.0, 20000.0),
    "camera_noise_level":  (0.0, 0.6),
}

_DEFAULTS: dict[str, float] = {
    "fog_density_percent": 30.0,
    "illumination_lux":    4000.0,
    "camera_noise_level":  0.1,
}


def _clip(name: str, value: Any) -> float:
    lo, hi = _BOUNDS.get(name, (-1e9, 1e9))
    return max(lo, min(hi, float(value)))


def _parse_env_params(raw: str) -> dict[str, float]:
    """Parse environment parameters from LLM-generated string.

    Handles three failure modes:
      1. LLM wraps JSON in markdown fences or prose
      2. LLM uses single quotes instead of double
      3. LLM outputs partial / malformed JSON
    """
    if not raw or not raw.strip():
        return dict(_DEFAULTS)

    text = raw.strip()

    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip("` \n")

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and any(k in data for k in _BOUNDS):
            return {k: _clip(k, data.get(k, _DEFAULTS[k])) for k in _BOUNDS}
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract first {...} block that contains at least one of our keys
    for match in re.finditer(r"\{[^{}]*\}", text, re.DOTALL):
        try:
            candidate = match.group(0).replace("'", '"')
            data = json.loads(candidate)
            if isinstance(data, dict) and any(k in data for k in _BOUNDS):
                return {k: _clip(k, data.get(k, _DEFAULTS[k])) for k in _BOUNDS}
        except (json.JSONDecodeError, ValueError):
            continue

    # Last resort: extract individual numeric values with regex
    result = dict(_DEFAULTS)
    for param in _BOUNDS:
        pattern = rf'["\']?{re.escape(param)}["\']?\s*:\s*([0-9]+(?:\.[0-9]+)?)'
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result[param] = _clip(param, float(m.group(1)))

    return result


class AdversarialScenarioGenerator(dspy.Module):
    """DSPy module that generates adversarial UAV environment scenarios.

    Uses ChainOfThought on UAVAdversarialScenario so the LLM reasons
    step-by-step before committing to environment parameters.

    DSPy optimizers automatically improve the few-shot demonstrations
    in this module by measuring each prediction against the Simulink
    adversarial metric (mission_success == 0 → high reward).

    Typical usage:
        generator = AdversarialScenarioGenerator()

        # After optimization:
        optimizer.compile(generator, trainset=trainset)

        prediction = generator(
            iteration_history="...",
            xai_analysis="...",
            current_performance="...",
        )
        env = prediction.environment_parameters  # dict
    """

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(UAVAdversarialScenario)

    def forward(
        self,
        iteration_history: str,
        xai_analysis: str,
        current_performance: str,
    ) -> dspy.Prediction:
        pred = self.generate(
            iteration_history=iteration_history,
            xai_analysis=xai_analysis,
            current_performance=current_performance,
        )

        env_params = _parse_env_params(
            getattr(pred, "environment_parameters_json", "")
        )

        return dspy.Prediction(
            reasoning=getattr(pred, "reasoning", ""),
            analysis=getattr(pred, "analysis", ""),
            environment_parameters=env_params,
            target_hypothesis=getattr(pred, "target_hypothesis", ""),
            raw_json=getattr(pred, "environment_parameters_json", ""),
        )
