"""KernelSHAP-based parameter importance for the boundary-search loop.

This module preserves the previous payload contract used by:
  - run_dspy_adversarial._build_xai_signals()
  - xai/counterfactual_boundary.py (shap_signals payload)

but removes the XGBoost dependency. Instead, it adapts the existing
KernelSHAP counterfactual analyzer in xai.generate_counterfactual_and_boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from xai import generate_counterfactual_and_boundary

    _KERNEL_SHAP_AVAILABLE = True
except Exception:
    _KERNEL_SHAP_AVAILABLE = False

try:
    from dspy_pipeline.matlab_bridge import MatlabSimulinkBridge

    _DEFAULT_THRESHOLD = float(MatlabSimulinkBridge.MAP_THRESHOLD)
except Exception:
    _DEFAULT_THRESHOLD = 0.85


FEATURES = ["fog_density_percent", "illumination_lux", "camera_noise_level"]


@dataclass
class ShapSignals:
    """KernelSHAP analysis result for one (history, current_env) pair."""

    global_importance: list[dict]
    local_contributions: list[dict]
    base_value: float
    n_samples: int
    model_r2: float | None

    def to_payload(self) -> dict:
        return {
            "global_feature_importance": self.global_importance,
            "local_feature_contributions": self.local_contributions,
            "base_value": self.base_value,
            "n_samples": self.n_samples,
            "model_r2": self.model_r2,
        }


def is_available() -> bool:
    return _KERNEL_SHAP_AVAILABLE


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_status(map50: float, threshold: float) -> str:
    return "PASS" if map50 >= threshold else "FAIL"


def _build_history_nodes(rows: list[dict], threshold: float) -> list[dict]:
    nodes: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        env = {k: _as_float(row.get(k, 0.0), 0.0) for k in FEATURES}
        map50 = _as_float(row.get("map50", 0.0), 0.0)
        status = _to_status(map50, threshold)
        nodes.append(
            {
                "scenario_id": f"history_{idx:03d}",
                "scenario": {"environment_parameters": env},
                "performance_signals": {"map50": map50, "status": status},
                "result_status": status,
                "safety_line": threshold,
            }
        )
    return nodes


def _normalize_current_env(current_env: dict[str, float]) -> dict[str, float]:
    return {k: _as_float(current_env.get(k, 0.0), 0.0) for k in FEATURES}


def _feature_direction(value: float) -> str:
    return "increase_failure" if value < 0.0 else "decrease_failure"


def _adapt_from_kernelshap(
    counterfactual_output: dict[str, Any],
    n_samples: int,
) -> ShapSignals:
    top = counterfactual_output.get("top_features")
    top = top if isinstance(top, list) else []

    by_name: dict[str, dict] = {}
    for row in top:
        if not isinstance(row, dict):
            continue
        name = str(row.get("feature", "")).strip()
        if not name:
            continue
        by_name[name] = row

    global_importance: list[dict] = []
    local_contributions: list[dict] = []
    for name in FEATURES:
        row = by_name.get(name, {})
        shap_value = _as_float(row.get("shap_value", 0.0), 0.0)
        shap_importance = abs(_as_float(row.get("shap_importance", 0.0), 0.0))

        global_importance.append(
            {
                "name": name,
                "importance": round(shap_importance, 4),
                "mean_abs_shap": round(shap_importance, 5),
                "direction": _feature_direction(shap_value),
            }
        )
        local_contributions.append(
            {
                "name": name,
                "shap_value": round(shap_value, 5),
                "abs_contribution_score": round(abs(shap_value), 5),
                "direction": _feature_direction(shap_value),
            }
        )

    global_importance.sort(key=lambda d: d["importance"], reverse=True)
    local_contributions.sort(key=lambda d: d["abs_contribution_score"], reverse=True)

    analysis_context = {}
    guidance = counterfactual_output.get("guidance")
    if isinstance(guidance, dict):
        ctx = guidance.get("analysis_context")
        if isinstance(ctx, dict):
            analysis_context = ctx

    base_value = _as_float(
        analysis_context.get("kernelshap_expected_value", 0.0),
        _as_float(counterfactual_output.get("base_case", {}).get("map50", 0.0), 0.0),
    )

    return ShapSignals(
        global_importance=global_importance,
        local_contributions=local_contributions,
        base_value=base_value,
        n_samples=n_samples,
        model_r2=None,
    )


def compute_shap_signals(
    history: list[dict],
    current_env: dict[str, float],
) -> ShapSignals | None:
    """Compute SHAP-like signals for DSPy guidance without XGBoost.

    Args:
        history:     list of rows with FEATURES keys and map50.
        current_env: current environment parameters.

    Returns:
        ShapSignals or None when KernelSHAP backend is unavailable.
    """
    if not _KERNEL_SHAP_AVAILABLE:
        return None

    rows = [h for h in history if all(k in h for k in FEATURES) and "map50" in h]
    if len(rows) < 1:
        return _uniform_fallback(current_env, n_samples=0)

    threshold = _DEFAULT_THRESHOLD
    current = _normalize_current_env(current_env)
    current_map50 = _as_float(rows[-1].get("map50", 0.0), 0.0)
    current_status = _to_status(current_map50, threshold)

    payload = {
        "scene_id": "dspy_current",
        "current_scenario": {
            "scenario_id": "dspy_current",
            "environment_parameters": current,
            "result_status": current_status,
            "map50": current_map50,
            "safety_line": threshold,
        },
        "scenario_history": _build_history_nodes(rows, threshold),
        "result_status": current_status,
        "map50": current_map50,
        "safety_line": threshold,
    }

    try:
        counterfactual_output, _ = generate_counterfactual_and_boundary(
            payload=payload,
            target_status=None,
            mode="simulation_grounded",
            num_counterfactuals=3,
            num_boundary_candidates=5,
            random_seed=0,
            simulink_callable=None,
        )
        return _adapt_from_kernelshap(counterfactual_output, n_samples=len(rows))
    except Exception:
        return _uniform_fallback(current_env, n_samples=len(rows))


def _uniform_fallback(current_env: dict[str, float], n_samples: int) -> ShapSignals:
    del current_env
    uniform = round(1.0 / len(FEATURES), 4)
    return ShapSignals(
        global_importance=[
            {
                "name": k,
                "importance": uniform,
                "mean_abs_shap": 0.0,
                "direction": "increase_failure",
            }
            for k in FEATURES
        ],
        local_contributions=[
            {
                "name": k,
                "shap_value": 0.0,
                "abs_contribution_score": 0.0,
                "direction": "increase_failure",
            }
            for k in FEATURES
        ],
        base_value=0.0,
        n_samples=n_samples,
        model_r2=None,
    )
