"""DSPy-based adversarial scenario generation pipeline for UAV Simulink testbed.

Architecture:
  signatures.py   - DSPy Signature: defines LLM inputs/outputs
  modules.py      - DSPy Module: AdversarialScenarioGenerator (ChainOfThought)
  matlab_bridge.py- Simulink backend: Engine / subprocess / mock
  metric.py       - DSPy Metric: rewards mission failure (Mission_Success=0)
  dataset.py      - Build training set from existing iteration JSON files
  optimizer.py    - BootstrapFewShot / MIPROv2 teleprompter setup
"""

from dspy_pipeline.signatures import UAVAdversarialScenario
from dspy_pipeline.modules import AdversarialScenarioGenerator

__all__ = ["UAVAdversarialScenario", "AdversarialScenarioGenerator"]
