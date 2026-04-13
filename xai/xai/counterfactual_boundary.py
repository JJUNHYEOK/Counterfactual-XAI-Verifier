from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from simulator import Simulator


FAIL_STATUS = "FAIL"
PASS_STATUS = "PASS"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    weight: float
    mutable: bool

    def span(self) -> float:
        span = self.upper - self.lower
        return span if span > 1e-9 else 1e-9

    def clamp(self, value: float) -> float:
        return min(self.upper, max(self.lower, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def extract_environment_parameters(payload: dict[str, Any]) -> dict[str, float]:
    candidates: list[dict[str, Any]] = []

    current_environment = payload.get("current_environment")
    if isinstance(current_environment, dict):
        candidates.append(current_environment)

    scenario = payload.get("scenario")
    if isinstance(scenario, dict):
        scenario_environment = scenario.get("environment_parameters")
        if isinstance(scenario_environment, dict):
            candidates.append(scenario_environment)

    direct_environment = payload.get("environment_parameters")
    if isinstance(direct_environment, dict):
        candidates.append(direct_environment)

    result: dict[str, float] = {}
    for mapping in candidates:
        for key, value in mapping.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = float(value)

    if not result:
        raise ValueError(
            "환경 파라미터를 찾지 못했습니다. "
            "current_environment 또는 scenario.environment_parameters를 제공해 주세요."
        )
    return result


def _default_bounds(name: str, value: float) -> tuple[float, float]:
    lowered = name.lower()
    if "percent" in lowered:
        return 0.0, 100.0
    if "lux" in lowered:
        return 0.0, 20000.0
    if "wind" in lowered:
        return 0.0, 30.0
    if "delay" in lowered and "ms" in lowered:
        return 0.0, 1000.0
    if "delay" in lowered:
        return 0.0, 20.0
    if "noise" in lowered or "obstacle_density" in lowered:
        return 0.0, 1.0
    if 0.0 <= value <= 1.0:
        return 0.0, 1.0

    spread = max(abs(value) * 0.75, 1.0)
    lower = value - spread
    upper = value + spread
    if value >= 0.0 and lower < 0.0:
        lower = 0.0
    return lower, upper


def _is_weather_param(name: str) -> bool:
    lowered = name.lower()
    tokens = ("wind", "fog", "rain", "snow", "weather")
    return any(token in lowered for token in tokens)


def _is_lighting_param(name: str) -> bool:
    lowered = name.lower()
    tokens = ("illumination", "light", "brightness")
    return any(token in lowered for token in tokens)


def _is_obstacle_param(name: str) -> bool:
    lowered = name.lower()
    tokens = ("obstacle", "density")
    return any(token in lowered for token in tokens)


def _apply_scenario_constraints(
    specs: list[ParameterSpec],
    payload: dict[str, Any],
) -> list[ParameterSpec]:
    constraints = payload.get("scenario_constraints")
    if not isinstance(constraints, dict):
        return specs

    weather_allowed = bool(constraints.get("allow_weather_change", True))
    lighting_allowed = bool(constraints.get("allow_lighting_change", True))
    obstacle_allowed = bool(constraints.get("allow_obstacle_density_change", True))

    updated: list[ParameterSpec] = []
    for spec in specs:
        mutable = spec.mutable
        if not weather_allowed and _is_weather_param(spec.name):
            mutable = False
        if not lighting_allowed and _is_lighting_param(spec.name):
            mutable = False
        if not obstacle_allowed and _is_obstacle_param(spec.name):
            mutable = False
        updated.append(
            ParameterSpec(
                name=spec.name,
                lower=spec.lower,
                upper=spec.upper,
                weight=spec.weight,
                mutable=mutable,
            )
        )
    return updated


def build_parameter_specs(
    payload: dict[str, Any],
    base_environment: dict[str, float],
) -> list[ParameterSpec]:
    search_space = payload.get("search_space")
    search_space = search_space if isinstance(search_space, dict) else {}

    bounds_config = search_space.get("bounds")
    bounds_config = bounds_config if isinstance(bounds_config, dict) else {}

    weights_config = search_space.get("weights")
    weights_config = weights_config if isinstance(weights_config, dict) else {}

    mutable_parameters = search_space.get("mutable_parameters")
    if isinstance(mutable_parameters, list):
        mutable_set = {str(item) for item in mutable_parameters}
    else:
        mutable_set = None

    specs: list[ParameterSpec] = []
    for name, value in base_environment.items():
        configured_bounds = bounds_config.get(name)
        if isinstance(configured_bounds, list) and len(configured_bounds) == 2:
            lower = _safe_float(configured_bounds[0], value)
            upper = _safe_float(configured_bounds[1], value)
            if upper < lower:
                lower, upper = upper, lower
            if abs(upper - lower) < 1e-9:
                upper = lower + 1.0
        else:
            lower, upper = _default_bounds(name, value)

        weight = max(1e-6, _safe_float(weights_config.get(name), 1.0))
        mutable = True if mutable_set is None else name in mutable_set
        specs.append(
            ParameterSpec(
                name=name,
                lower=lower,
                upper=upper,
                weight=weight,
                mutable=mutable,
            )
        )

    return _apply_scenario_constraints(specs, payload)


class BaseEvaluator:
    mode_name = "base"

    def evaluate(self, params: dict[str, float]) -> dict[str, Any]:
        raise NotImplementedError

    def target_hint(self, target_status: str, param_name: str, base_value: float) -> float:
        return 0.0


class SimDummyEvaluator(BaseEvaluator):
    mode_name = "sim_dummy"

    def __init__(self, scene_id: str):
        self._scene_id = scene_id
        self._simulator = Simulator(sleep_seconds=0.0)

    def evaluate(self, params: dict[str, float]) -> dict[str, Any]:
        sim_input = {"scenario_id": self._scene_id}
        sim_input.update(params)
        sim_result = self._simulator.run_sim_dummy(sim_input)

        min_distance = _safe_float(sim_result.get("min_distance"), 0.0)
        margin = 1.0 - min_distance

        status_text = str(sim_result.get("status", "")).upper()
        predicted_fail = status_text == FAIL_STATUS or margin > 0.0
        if predicted_fail and margin <= 0.0:
            margin = 0.01
        if (not predicted_fail) and margin >= 0.0:
            margin = -0.01

        predicted_status = FAIL_STATUS if predicted_fail else PASS_STATUS
        return {
            "predicted_status": predicted_status,
            "decision_margin": float(margin),
            "threshold_condition": (
                "decision_margin > 0 이면 FAIL, "
                "decision_margin <= 0 이면 PASS "
                "(decision_margin = 1.0 - min_distance)"
            ),
            "raw_sim_result": sim_result,
        }

    def target_hint(self, target_status: str, param_name: str, base_value: float) -> float:
        lowered = param_name.lower()
        harmful_when_increasing = (
            "wind" in lowered
            or "delay" in lowered
            or "obstacle" in lowered
            or "fog" in lowered
            or "noise" in lowered
        )
        if not harmful_when_increasing:
            return 0.0
        return -1.0 if target_status == PASS_STATUS else 1.0


class Map50ProxyEvaluator(BaseEvaluator):
    mode_name = "map50_proxy"

    def __init__(
        self,
        payload: dict[str, Any],
        specs: list[ParameterSpec],
        base_environment: dict[str, float],
    ):
        self._payload = payload
        self._specs = specs
        self._base_environment = base_environment

        perf = payload.get("performance_signals")
        perf = perf if isinstance(perf, dict) else {}
        eval_result = payload.get("eval_result")
        eval_result = eval_result if isinstance(eval_result, dict) else {}
        req = payload.get("current_requirement")
        req = req if isinstance(req, dict) else {}

        self._threshold = _safe_float(
            perf.get("threshold", eval_result.get("requirement_threshold", req.get("threshold", 0.85))),
            0.85,
        )
        self._base_map50 = _safe_float(
            perf.get("map50", eval_result.get("map50", self._threshold - 0.05)),
            self._threshold - 0.05,
        )
        self._base_margin = self._threshold - self._base_map50
        self._guidance_weights = self._resolve_guidance_weights()
        self._guidance_sources: list[str] = []
        self._factor_effects = self._build_factor_effects()

    def _resolve_guidance_weights(self) -> dict[str, float]:
        request = self._payload.get("counterfactual_request")
        request = request if isinstance(request, dict) else {}
        blend = request.get("guidance_blend")
        blend = blend if isinstance(blend, dict) else {}

        xai_weight = max(0.0, _safe_float(blend.get("xai"), 0.4))
        shap_global_weight = max(0.0, _safe_float(blend.get("shap_global"), 0.3))
        shap_local_weight = max(0.0, _safe_float(blend.get("shap_local"), 0.3))

        total = xai_weight + shap_global_weight + shap_local_weight
        if total <= 1e-12:
            return {"xai": 1.0, "shap_global": 0.0, "shap_local": 0.0}
        return {
            "xai": xai_weight / total,
            "shap_global": shap_global_weight / total,
            "shap_local": shap_local_weight / total,
        }

    def _build_factor_effects(self) -> dict[str, float]:
        def direction_to_sign(direction: Any) -> float | None:
            if not isinstance(direction, str):
                return None
            lowered = direction.strip().lower()
            if lowered in {"increase_failure", "increase", "positive", "up"}:
                return 1.0
            if lowered in {"decrease_failure", "decrease", "negative", "down"}:
                return -1.0
            return None

        def add_effect(
            effects: dict[str, float],
            spec_names: list[str],
            factor_name: str,
            magnitude: float,
            source_weight: float,
            direction_hint: float | None = None,
            fallback_sign_from_name: bool = True,
        ) -> bool:
            if not factor_name or magnitude <= 0.0 or source_weight <= 0.0:
                return False
            matched = _match_parameter_name(factor_name, spec_names)
            if not matched:
                return False
            if direction_hint is None and fallback_sign_from_name:
                direction_hint = _infer_effect_sign(matched, self._base_environment.get(matched, 0.0))
            if direction_hint is None:
                direction_hint = 1.0
            effects[matched] = effects.get(matched, 0.0) + (source_weight * direction_hint * magnitude)
            return True

        xai_signals = self._payload.get("xai_signals")
        xai_signals = xai_signals if isinstance(xai_signals, dict) else {}
        dominant_factors = xai_signals.get("dominant_factors")
        dominant_factors = dominant_factors if isinstance(dominant_factors, list) else []

        shap_signals = self._payload.get("shap_signals")
        shap_signals = shap_signals if isinstance(shap_signals, dict) else {}
        shap_global = shap_signals.get("global_feature_importance")
        shap_global = shap_global if isinstance(shap_global, list) else []
        shap_local = shap_signals.get("local_feature_contributions")
        shap_local = shap_local if isinstance(shap_local, list) else []

        spec_names = [spec.name for spec in self._specs]
        effects: dict[str, float] = {}

        xai_added = False
        for factor in dominant_factors:
            if not isinstance(factor, dict):
                continue
            factor_name = str(factor.get("name", "")).strip()
            if not factor_name:
                continue
            importance = abs(_safe_float(factor.get("importance"), 0.0))
            direction_hint = direction_to_sign(factor.get("direction"))
            xai_added = add_effect(
                effects=effects,
                spec_names=spec_names,
                factor_name=factor_name,
                magnitude=importance,
                source_weight=self._guidance_weights["xai"],
                direction_hint=direction_hint,
            ) or xai_added
        if xai_added:
            self._guidance_sources.append("xai_signals.dominant_factors")

        shap_global_added = False
        for factor in shap_global:
            if not isinstance(factor, dict):
                continue
            factor_name = str(factor.get("name", factor.get("feature", ""))).strip()
            importance = abs(
                _safe_float(
                    factor.get("importance", factor.get("mean_abs_shap", factor.get("abs_contribution_score", 0.0))),
                    0.0,
                )
            )
            direction_hint = direction_to_sign(factor.get("direction"))
            shap_global_added = add_effect(
                effects=effects,
                spec_names=spec_names,
                factor_name=factor_name,
                magnitude=importance,
                source_weight=self._guidance_weights["shap_global"],
                direction_hint=direction_hint,
            ) or shap_global_added
        if shap_global_added:
            self._guidance_sources.append("shap_signals.global_feature_importance")

        shap_local_added = False
        for factor in shap_local:
            if not isinstance(factor, dict):
                continue
            factor_name = str(factor.get("name", factor.get("feature", ""))).strip()
            signed_score = _safe_float(
                factor.get("contribution_score", factor.get("shap_value", factor.get("importance", 0.0))),
                0.0,
            )
            magnitude = abs(
                _safe_float(
                    factor.get("abs_contribution_score", abs(signed_score)),
                    abs(signed_score),
                )
            )
            direction_hint = direction_to_sign(factor.get("direction"))
            if direction_hint is None and abs(signed_score) > 1e-12:
                direction_hint = 1.0 if signed_score > 0.0 else -1.0
            shap_local_added = add_effect(
                effects=effects,
                spec_names=spec_names,
                factor_name=factor_name,
                magnitude=magnitude,
                source_weight=self._guidance_weights["shap_local"],
                direction_hint=direction_hint,
            ) or shap_local_added
        if shap_local_added:
            self._guidance_sources.append("shap_signals.local_feature_contributions")

        if not effects:
            self._guidance_sources.append("fallback_uniform_effects")
            for spec in self._specs:
                sign = _infer_effect_sign(spec.name, self._base_environment.get(spec.name, 0.0))
                effects[spec.name] = sign * (1.0 / max(1, len(self._specs)))
            return effects

        norm = sum(abs(value) for value in effects.values())
        if norm <= 1e-12:
            return effects
        return {name: value / norm for name, value in effects.items()}

    def evaluate(self, params: dict[str, float]) -> dict[str, Any]:
        margin = self._base_margin
        contributions: dict[str, float] = {}

        for spec in self._specs:
            name = spec.name
            delta_norm = (params[name] - self._base_environment[name]) / spec.span()
            effect = self._factor_effects.get(name, 0.0)
            contribution = effect * delta_norm
            contributions[name] = contribution
            margin += contribution

        predicted_status = FAIL_STATUS if margin > 0.0 else PASS_STATUS
        map50_proxy = self._threshold - margin
        return {
            "predicted_status": predicted_status,
            "decision_margin": float(margin),
            "threshold_condition": (
                "decision_margin = threshold - map50_proxy. "
                "decision_margin > 0 이면 FAIL, decision_margin <= 0 이면 PASS"
            ),
            "raw_proxy_result": {
                "map50_proxy": float(map50_proxy),
                "threshold": float(self._threshold),
                "base_map50": float(self._base_map50),
                "factor_effects": self._factor_effects,
                "factor_contributions": contributions,
                "guidance_weights": self._guidance_weights,
                "guidance_sources": self._guidance_sources,
            },
        }

    def target_hint(self, target_status: str, param_name: str, base_value: float) -> float:
        effect = self._factor_effects.get(param_name, 0.0)
        if abs(effect) <= 1e-12:
            return 0.0
        if target_status == PASS_STATUS:
            return -1.0 if effect > 0.0 else 1.0
        return 1.0 if effect > 0.0 else -1.0


def _match_parameter_name(factor_name: str, parameter_names: list[str]) -> str | None:
    normalized_factor = _normalize_name(factor_name)
    if not normalized_factor:
        return None

    normalized_params = {name: _normalize_name(name) for name in parameter_names}
    for name, normalized_param in normalized_params.items():
        if normalized_factor in normalized_param or normalized_param in normalized_factor:
            return name

    synonym_map = {
        "fogdensity": ("fogdensitypercent", "fog"),
        "camera": ("cameranoise", "cameranoiselevel"),
        "noise": ("cameranoise", "cameranoiselevel", "noiselevel"),
        "illumination": ("illuminationlux", "light"),
        "wind": ("windspeed", "windspeedmps"),
        "delay": ("delay", "communicationdelayms"),
        "obstacle": ("obstacledensity",),
    }
    for name, normalized_param in normalized_params.items():
        for keyword, candidates in synonym_map.items():
            if keyword in normalized_factor and any(token in normalized_param for token in candidates):
                return name
    return None


def _infer_effect_sign(parameter_name: str, current_value: float) -> float:
    lowered = parameter_name.lower()
    if any(token in lowered for token in ("fog", "noise", "wind", "delay", "obstacle", "rain", "snow")):
        return 1.0
    if any(token in lowered for token in ("illumination", "light", "brightness")):
        return -1.0 if current_value < 5000.0 else 1.0
    return 1.0


def _select_evaluator(
    payload: dict[str, Any],
    specs: list[ParameterSpec],
    base_environment: dict[str, float],
    mode: str,
) -> BaseEvaluator:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "sim_dummy":
        return SimDummyEvaluator(scene_id=str(payload.get("scene_id", "cf_scene")))
    if normalized_mode == "map50_proxy":
        return Map50ProxyEvaluator(payload=payload, specs=specs, base_environment=base_environment)

    sim_related = {"wind_speed", "delay", "obstacle_density"}
    if sim_related.issubset(set(base_environment.keys())):
        return SimDummyEvaluator(scene_id=str(payload.get("scene_id", "cf_scene")))
    return Map50ProxyEvaluator(payload=payload, specs=specs, base_environment=base_environment)


def _project(params: dict[str, float], specs: list[ParameterSpec]) -> dict[str, float]:
    projected = params.copy()
    for spec in specs:
        projected[spec.name] = spec.clamp(projected[spec.name])
    return projected


def _distance_l1_normalized(
    base_environment: dict[str, float],
    candidate_environment: dict[str, float],
    specs: list[ParameterSpec],
) -> float:
    distance = 0.0
    for spec in specs:
        base = base_environment[spec.name]
        candidate = candidate_environment[spec.name]
        distance += spec.weight * abs((candidate - base) / spec.span())
    return float(distance)


def _build_start_points(
    base_environment: dict[str, float],
    specs: list[ParameterSpec],
    evaluator: BaseEvaluator,
    target_status: str,
    random_seed: int,
    num_starts: int,
) -> list[dict[str, float]]:
    rng = random.Random(random_seed)
    starts = [base_environment.copy()]

    directed = base_environment.copy()
    directed_strong = base_environment.copy()
    directed_full = base_environment.copy()
    for spec in specs:
        if not spec.mutable:
            continue
        hint = evaluator.target_hint(target_status, spec.name, base_environment[spec.name])
        if abs(hint) <= 1e-12:
            continue
        directed[spec.name] = spec.clamp(base_environment[spec.name] + (0.2 * spec.span() * hint))
        directed_strong[spec.name] = spec.clamp(base_environment[spec.name] + (0.75 * spec.span() * hint))
        directed_full[spec.name] = spec.upper if hint > 0.0 else spec.lower
    if directed != base_environment:
        starts.append(directed)
    if directed_strong != base_environment:
        starts.append(directed_strong)
    if directed_full != base_environment:
        starts.append(directed_full)

    # 한 번에 한 파라미터씩 크게 이동한 시드를 추가해 경계 탐색 실패를 줄인다.
    for spec in specs:
        if not spec.mutable:
            continue
        hint = evaluator.target_hint(target_status, spec.name, base_environment[spec.name])
        if abs(hint) <= 1e-12:
            continue
        single = base_environment.copy()
        single[spec.name] = spec.upper if hint > 0.0 else spec.lower
        starts.append(single)

    while len(starts) < num_starts:
        point = base_environment.copy()
        for spec in specs:
            if not spec.mutable:
                continue
            jitter = rng.uniform(-0.35, 0.35) * spec.span()
            point[spec.name] = spec.clamp(base_environment[spec.name] + jitter)
        starts.append(point)
    return starts


def _optimize_projected(
    start: dict[str, float],
    specs: list[ParameterSpec],
    objective_fn,
    random_seed: int,
    num_steps: int = 90,
    step_size: float = 0.25,
    finite_diff_ratio: float = 0.03,
) -> tuple[dict[str, float], float, dict[str, Any]]:
    rng = random.Random(random_seed)
    current = _project(start, specs)
    best = current.copy()
    best_obj, best_eval = objective_fn(current)

    for step in range(num_steps):
        grad: dict[str, float] = {}
        for spec in specs:
            if not spec.mutable:
                continue
            eps = max(spec.span() * finite_diff_ratio, 1e-4)
            plus = current.copy()
            minus = current.copy()
            plus[spec.name] = spec.clamp(current[spec.name] + eps)
            minus[spec.name] = spec.clamp(current[spec.name] - eps)
            plus_obj, _ = objective_fn(plus)
            minus_obj, _ = objective_fn(minus)
            denom = plus[spec.name] - minus[spec.name]
            grad[spec.name] = 0.0 if abs(denom) < 1e-12 else (plus_obj - minus_obj) / denom

        lr = step_size * (0.985 ** step)
        updated = current.copy()
        for spec in specs:
            if not spec.mutable:
                continue
            noise = 0.0
            if step > 0 and step % 25 == 0:
                noise = rng.uniform(-0.01, 0.01) * spec.span()
            updated[spec.name] = spec.clamp(updated[spec.name] - (lr * grad[spec.name]) + noise)

        current = _project(updated, specs)
        current_obj, current_eval = objective_fn(current)
        if current_obj < best_obj:
            best_obj = current_obj
            best = current.copy()
            best_eval = current_eval

    return best, float(best_obj), best_eval


def _build_parameter_changes(
    base_environment: dict[str, float],
    candidate_environment: dict[str, float],
    specs: list[ParameterSpec],
) -> list[dict[str, float | str]]:
    changes: list[dict[str, float | str]] = []
    for spec in specs:
        start = base_environment[spec.name]
        end = candidate_environment[spec.name]
        delta = end - start
        if abs(delta) < 1e-10:
            continue
        changes.append(
            {
                "name": spec.name,
                "from": float(start),
                "to": float(end),
                "delta": float(delta),
                "normalized_delta": float(delta / spec.span()),
            }
        )
    changes.sort(key=lambda item: abs(_safe_float(item["normalized_delta"])), reverse=True)
    return changes


def _build_candidate_summary(
    source_status: str,
    target_status: str,
    predicted_status: str,
    changes: list[dict[str, float | str]],
    decision_margin: float,
) -> str:
    if not changes:
        if source_status == target_status:
            return "기준 시나리오가 이미 목표 상태에 있습니다."
        return "현재 탐색 설정에서는 추가 변화 없이 목표 상태 전환이 불가능합니다."

    top = changes[0]
    name = str(top["name"])
    delta = _safe_float(top["delta"])
    arrow = "증가" if delta > 0 else "감소"
    return (
        f"{name} 값을 {abs(delta):.4g}만큼 {arrow}시키는 변화가 "
        f"{source_status}->{predicted_status} 결과 변화에 가장 크게 기여했습니다. "
        f"(목표: {target_status}) "
        f"(candidate_margin={decision_margin:.4f})"
    )


def _search_counterfactual_candidates(
    base_environment: dict[str, float],
    specs: list[ParameterSpec],
    evaluator: BaseEvaluator,
    target_status: str,
    random_seed: int,
    num_candidates: int,
) -> list[dict[str, Any]]:
    starts = _build_start_points(
        base_environment=base_environment,
        specs=specs,
        evaluator=evaluator,
        target_status=target_status,
        random_seed=random_seed,
        num_starts=max(8, num_candidates * 4),
    )

    penalty = 8.0

    def objective(candidate_env: dict[str, float]) -> tuple[float, dict[str, Any]]:
        evaluation = evaluator.evaluate(candidate_env)
        margin = _safe_float(evaluation.get("decision_margin"), 0.0)
        distance = _distance_l1_normalized(base_environment, candidate_env, specs)
        if target_status == PASS_STATUS:
            violation = max(0.0, margin)
        else:
            violation = max(0.0, -margin)
        value = distance + (penalty * violation) + (0.1 * abs(margin))
        return float(value), evaluation

    raw_candidates: list[dict[str, Any]] = []
    for idx, start in enumerate(starts):
        start_obj, start_eval = objective(start)
        start_distance = _distance_l1_normalized(base_environment, start, specs)
        start_status = str(start_eval.get("predicted_status", PASS_STATUS)).upper()
        raw_candidates.append(
            {
                "environment": _project(start, specs),
                "objective": start_obj,
                "distance_l1_normalized": start_distance,
                "evaluation": start_eval,
                "meets_target": start_status == target_status,
            }
        )

        candidate_env, objective_value, evaluation = _optimize_projected(
            start=start,
            specs=specs,
            objective_fn=objective,
            random_seed=random_seed + idx + 17,
        )
        distance = _distance_l1_normalized(base_environment, candidate_env, specs)
        predicted_status = str(evaluation.get("predicted_status", PASS_STATUS)).upper()
        raw_candidates.append(
            {
                "environment": candidate_env,
                "objective": objective_value,
                "distance_l1_normalized": distance,
                "evaluation": evaluation,
                "meets_target": predicted_status == target_status,
            }
        )

    dedup: dict[tuple[tuple[str, float], ...], dict[str, Any]] = {}
    for candidate in raw_candidates:
        key = tuple(
            (spec.name, round(candidate["environment"][spec.name], 7))
            for spec in specs
        )
        existing = dedup.get(key)
        if existing is None or candidate["objective"] < existing["objective"]:
            dedup[key] = candidate

    ordered = sorted(
        dedup.values(),
        key=lambda item: (
            not item["meets_target"],
            item["distance_l1_normalized"],
            abs(_safe_float(item["evaluation"].get("decision_margin"), 0.0)),
        ),
    )
    if ordered:
        first = ordered[0]
        if (not first["meets_target"]) and first["distance_l1_normalized"] <= 1e-12 and len(ordered) > 1:
            ordered = ordered[1:] + [first]
    return ordered[:num_candidates]


def _search_boundary_candidates(
    base_environment: dict[str, float],
    specs: list[ParameterSpec],
    evaluator: BaseEvaluator,
    random_seed: int,
    num_candidates: int,
) -> list[dict[str, Any]]:
    starts = _build_start_points(
        base_environment=base_environment,
        specs=specs,
        evaluator=evaluator,
        target_status=PASS_STATUS,
        random_seed=random_seed + 300,
        num_starts=max(10, num_candidates * 5),
    )

    def objective(candidate_env: dict[str, float]) -> tuple[float, dict[str, Any]]:
        evaluation = evaluator.evaluate(candidate_env)
        margin = _safe_float(evaluation.get("decision_margin"), 0.0)
        distance = _distance_l1_normalized(base_environment, candidate_env, specs)
        return float((5.0 * abs(margin)) + distance), evaluation

    raw_candidates: list[dict[str, Any]] = []
    for idx, start in enumerate(starts):
        candidate_env, objective_value, evaluation = _optimize_projected(
            start=start,
            specs=specs,
            objective_fn=objective,
            random_seed=random_seed + idx + 401,
            num_steps=70,
            step_size=0.22,
        )
        distance = _distance_l1_normalized(base_environment, candidate_env, specs)
        raw_candidates.append(
            {
                "environment": candidate_env,
                "objective": objective_value,
                "distance_l1_normalized": distance,
                "evaluation": evaluation,
            }
        )

    dedup: dict[tuple[tuple[str, float], ...], dict[str, Any]] = {}
    for candidate in raw_candidates:
        key = tuple(
            (spec.name, round(candidate["environment"][spec.name], 7))
            for spec in specs
        )
        existing = dedup.get(key)
        if existing is None or candidate["objective"] < existing["objective"]:
            dedup[key] = candidate

    ordered = sorted(
        dedup.values(),
        key=lambda item: (
            abs(_safe_float(item["evaluation"].get("decision_margin"), 0.0)),
            item["distance_l1_normalized"],
        ),
    )
    return ordered[:num_candidates]


def _format_candidate_payload(
    candidate_id: str,
    source_status: str,
    target_status: str | None,
    base_environment: dict[str, float],
    specs: list[ParameterSpec],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    evaluation = candidate["evaluation"]
    predicted_status = str(evaluation.get("predicted_status", PASS_STATUS)).upper()
    decision_margin = _safe_float(evaluation.get("decision_margin"), 0.0)
    changes = _build_parameter_changes(base_environment, candidate["environment"], specs)
    payload = {
        "candidate_id": candidate_id,
        "predicted_status": predicted_status,
        "decision_margin": float(decision_margin),
        "distance_l1_normalized": float(candidate["distance_l1_normalized"]),
        "parameter_changes": changes,
        "counterfactual_environment": candidate["environment"],
        "summary_explanation": _build_candidate_summary(
            source_status=source_status,
            target_status=target_status or predicted_status,
            predicted_status=predicted_status,
            changes=changes,
            decision_margin=decision_margin,
        ),
    }
    raw_sim_result = evaluation.get("raw_sim_result")
    if isinstance(raw_sim_result, dict):
        payload["sim_result_preview"] = {
            "status": raw_sim_result.get("status"),
            "min_distance": raw_sim_result.get("min_distance"),
            "message": raw_sim_result.get("message"),
        }
    raw_proxy_result = evaluation.get("raw_proxy_result")
    if isinstance(raw_proxy_result, dict):
        payload["proxy_result_preview"] = {
            "map50_proxy": raw_proxy_result.get("map50_proxy"),
            "threshold": raw_proxy_result.get("threshold"),
        }
    return payload


def generate_counterfactual_and_boundary(
    payload: dict[str, Any],
    target_status: str | None = None,
    mode: str = "auto",
    num_counterfactuals: int = 3,
    num_boundary_candidates: int = 5,
    random_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if num_counterfactuals <= 0:
        raise ValueError("num_counterfactuals는 1 이상이어야 합니다.")
    if num_boundary_candidates <= 0:
        raise ValueError("num_boundary_candidates는 1 이상이어야 합니다.")

    base_environment = extract_environment_parameters(payload)
    specs = build_parameter_specs(payload, base_environment)
    if not any(spec.mutable for spec in specs):
        raise ValueError("변경 가능한 파라미터가 없습니다. scenario_constraints/search_space를 확인해 주세요.")

    evaluator = _select_evaluator(payload, specs, base_environment, mode=mode)
    base_eval = evaluator.evaluate(base_environment)
    source_status = str(base_eval.get("predicted_status", PASS_STATUS)).upper()
    decision_margin = _safe_float(base_eval.get("decision_margin"), 0.0)
    raw_proxy = base_eval.get("raw_proxy_result")
    raw_proxy = raw_proxy if isinstance(raw_proxy, dict) else {}
    guidance_info = {
        "guidance_sources": raw_proxy.get("guidance_sources", []),
        "guidance_weights": raw_proxy.get("guidance_weights", {}),
    }

    resolved_target = (target_status or "").strip().upper()
    if resolved_target not in (PASS_STATUS, FAIL_STATUS):
        resolved_target = PASS_STATUS if source_status == FAIL_STATUS else FAIL_STATUS

    cf_candidates_raw = _search_counterfactual_candidates(
        base_environment=base_environment,
        specs=specs,
        evaluator=evaluator,
        target_status=resolved_target,
        random_seed=random_seed,
        num_candidates=num_counterfactuals,
    )
    boundary_candidates_raw = _search_boundary_candidates(
        base_environment=base_environment,
        specs=specs,
        evaluator=evaluator,
        random_seed=random_seed,
        num_candidates=num_boundary_candidates,
    )

    cf_candidates = [
        _format_candidate_payload(
            candidate_id=f"cf_{idx + 1:02d}",
            source_status=source_status,
            target_status=resolved_target,
            base_environment=base_environment,
            specs=specs,
            candidate=candidate,
        )
        for idx, candidate in enumerate(cf_candidates_raw)
    ]
    boundary_candidates = [
        _format_candidate_payload(
            candidate_id=f"bd_{idx + 1:02d}",
            source_status=source_status,
            target_status=None,
            base_environment=base_environment,
            specs=specs,
            candidate=candidate,
        )
        for idx, candidate in enumerate(boundary_candidates_raw)
    ]

    scene_id = str(payload.get("scene_id", payload.get("scenario_id", "cf_scene")))
    threshold_condition = str(base_eval.get("threshold_condition", "decision_margin 기준"))
    mutable_parameters = [spec.name for spec in specs if spec.mutable]

    counterfactual_output = {
        "schema_version": "xai_counterfactual_v1",
        "scene_id": scene_id,
        "search_mode": evaluator.mode_name,
        "search_method": "projected_gradient_multi_start",
        "source_status": source_status,
        "target_status": resolved_target,
        "threshold_condition": threshold_condition,
        "mutable_parameters": mutable_parameters,
        "guidance": guidance_info,
        "base_case": {
            "current_environment": base_environment,
            "predicted_status": source_status,
            "decision_margin": float(decision_margin),
            "distance_to_boundary": float(abs(decision_margin)),
        },
        "minimal_change_candidates": cf_candidates,
    }

    closest_boundary = boundary_candidates[0] if boundary_candidates else None
    boundary_output = {
        "schema_version": "xai_boundary_v1",
        "scene_id": scene_id,
        "search_mode": evaluator.mode_name,
        "search_method": "projected_gradient_multi_start",
        "threshold_condition": threshold_condition,
        "guidance": guidance_info,
        "base_case": {
            "current_environment": base_environment,
            "predicted_status": source_status,
            "decision_margin": float(decision_margin),
            "distance_to_boundary": float(abs(decision_margin)),
        },
        "nearest_boundary_candidates": boundary_candidates,
        "boundary_analysis": {
            "base_distance_to_boundary": float(abs(decision_margin)),
            "closest_boundary_distance": (
                float(closest_boundary["distance_l1_normalized"]) if closest_boundary else None
            ),
            "closest_boundary_margin_abs": (
                float(abs(_safe_float(closest_boundary["decision_margin"], 0.0)))
                if closest_boundary
                else None
            ),
            "interpretation": (
                "distance_to_boundary가 작을수록 pass/fail 경계에 가까운 시나리오입니다."
            ),
        },
    }

    return counterfactual_output, boundary_output
