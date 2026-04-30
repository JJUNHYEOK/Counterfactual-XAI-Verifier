"""SHAP-based parameter importance for the boundary-search loop.

Trains a small XGBoost regressor on accumulated (env_params, mAP50) pairs
and produces SHAP signals in the schema consumed by:
  * dspy_pipeline.signatures.UAVAdversarialScenario (xai_analysis input)
  * xai/counterfactual_boundary.Map50ProxyEvaluator (shap_signals payload)

Why XGBoost+SHAP rather than gradient/perturbation:
  - Captures non-monotone interactions (fog x noise) once enough samples exist
  - SHAP gives both global importance and a *signed* local contribution per
    feature, which the LLM uses to decide which knob to push for PASS->FAIL
    or relax for FAIL->PASS.

The module degrades gracefully:
  - With <3 samples it returns a uniform-importance fallback (no model fit)
  - If xgboost/shap are not installed it returns None (caller falls back to
    the heuristic _infer_dominant_factors in run_dspy_adversarial.py)
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
    import xgboost as xgb
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


FEATURES = ["fog_density_percent", "illumination_lux", "camera_noise_level"]


@dataclass
class ShapSignals:
    """SHAP analysis result for one (history, current_env) pair."""

    global_importance: list[dict]      # mean |SHAP| across history
    local_contributions: list[dict]    # signed SHAP for current_env
    base_value: float                  # model expected value
    n_samples: int                     # rows used to fit
    model_r2: float | None             # in-sample R^2 (sanity only)

    def to_payload(self) -> dict:
        """Schema matching xai/counterfactual_boundary.Map50ProxyEvaluator."""
        return {
            "global_feature_importance":  self.global_importance,
            "local_feature_contributions": self.local_contributions,
            "base_value":                  self.base_value,
            "n_samples":                   self.n_samples,
            "model_r2":                    self.model_r2,
        }


def is_available() -> bool:
    return _SHAP_AVAILABLE


def compute_shap_signals(
    history: list[dict],
    current_env: dict[str, float],
) -> ShapSignals | None:
    """Fit XGBoost on (env -> mAP50) history and explain current_env.

    Args:
        history:     list of dicts each containing the FEATURES keys + 'map50'
        current_env: env params for which we want a local SHAP explanation

    Returns:
        ShapSignals or None if SHAP/XGBoost unavailable.
    """
    if not _SHAP_AVAILABLE:
        return None

    rows = [h for h in history if all(k in h for k in FEATURES) and "map50" in h]
    if len(rows) < 3:
        return _uniform_fallback(current_env, n_samples=len(rows))

    X = np.array([[float(r[k]) for k in FEATURES] for r in rows], dtype=float)
    y = np.array([float(r["map50"]) for r in rows], dtype=float)

    # Tiny tree ensemble — cheap and robust for ~10-50 samples
    model = xgb.XGBRegressor(
        n_estimators=64,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        reg_lambda=1.0,
        random_state=0,
        verbosity=0,
    )
    model.fit(X, y)

    # In-sample R^2 (sanity check; we don't trust it for generalisation)
    y_pred = model.predict(X)
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None

    explainer  = shap.TreeExplainer(model)
    shap_train = explainer.shap_values(X)            # (n, 3) — for global importance
    x_curr     = np.array([[float(current_env.get(k, 0.0)) for k in FEATURES]])
    shap_curr  = explainer.shap_values(x_curr)[0]    # (3,) — local for current

    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    # Global: mean absolute SHAP, with sign hint from average signed direction
    mean_abs = np.abs(shap_train).mean(axis=0)
    mean_signed = shap_train.mean(axis=0)
    g_total = float(mean_abs.sum()) or 1.0
    global_importance = [
        {
            "name":          FEATURES[i],
            "importance":    round(float(mean_abs[i]) / g_total, 4),
            "mean_abs_shap": round(float(mean_abs[i]), 5),
            "direction":     "increase_failure"
                if mean_signed[i] < 0      # negative shap on map50 == push toward FAIL
                else "decrease_failure",
        }
        for i in range(len(FEATURES))
    ]
    global_importance.sort(key=lambda d: d["importance"], reverse=True)

    # Local: signed SHAP for current_env
    local_contributions = [
        {
            "name":               FEATURES[i],
            "shap_value":         round(float(shap_curr[i]), 5),
            "abs_contribution_score": round(abs(float(shap_curr[i])), 5),
            "direction":          "increase_failure"
                if shap_curr[i] < 0
                else "decrease_failure",
        }
        for i in range(len(FEATURES))
    ]
    local_contributions.sort(key=lambda d: d["abs_contribution_score"], reverse=True)

    return ShapSignals(
        global_importance=global_importance,
        local_contributions=local_contributions,
        base_value=base_value,
        n_samples=len(rows),
        model_r2=round(r2, 4) if r2 is not None else None,
    )


def _uniform_fallback(current_env: dict[str, float], n_samples: int) -> ShapSignals:
    uniform = round(1.0 / len(FEATURES), 4)
    return ShapSignals(
        global_importance=[
            {"name": k, "importance": uniform, "mean_abs_shap": 0.0,
             "direction": "increase_failure"}
            for k in FEATURES
        ],
        local_contributions=[
            {"name": k, "shap_value": 0.0, "abs_contribution_score": 0.0,
             "direction": "increase_failure"}
            for k in FEATURES
        ],
        base_value=0.0,
        n_samples=n_samples,
        model_r2=None,
    )
