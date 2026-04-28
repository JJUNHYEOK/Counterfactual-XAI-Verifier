from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import shap


PASS_STATUS = "PASS"
FAIL_STATUS = "FAIL"
METHOD_NAME = "simulation_grounded_kernelshap"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    weight: float
    mutable: bool

    def span(self) -> float:
        return max(self.upper - self.lower, 1e-9)

    def clamp(self, value: float) -> float:
        return min(self.upper, max(self.lower, value))


@dataclass
class RunPoint:
    run_id: str
    env: dict[str, float]
    map50: float | None
    safety_line: float | None
    status: str

    def safety_margin(self) -> float | None:
        if self.map50 is None or self.safety_line is None:
            return None
        return float(self.map50 - self.safety_line)

    def decision_margin(self) -> float:
        margin = self.safety_margin()
        if margin is not None:
            return float(-margin)
        return 0.01 if self.status == FAIL_STATUS else -0.01


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fo(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        parsed = float(value)
        return None if parsed != parsed else parsed
    except (TypeError, ValueError):
        return None


def _read(node: Any, *path: str) -> Any:
    cur = node
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _status(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text in {PASS_STATUS, "SUCCESS", "SAFE", "OK"}:
        return PASS_STATUS
    if text in {FAIL_STATUS, "FAILED", "FAILURE", "UNSAFE", "ERROR", "VIOLATED"}:
        return FAIL_STATUS
    return None


def _extract_environment(node: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(node, dict):
        return out

    candidates = (
        node.get("current_environment"),
        _read(node, "scenario", "environment_parameters"),
        _read(node, "current_scenario", "environment_parameters"),
        node.get("environment_parameters"),
        node.get("current_scenario"),
    )
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        for key, value in cand.items():
            fv = _fo(value)
            if fv is not None:
                out[str(key)] = float(fv)
    return out


def _extract_map50(node: dict[str, Any]) -> float | None:
    for v in (
        node.get("map50"),
        _read(node, "performance_signals", "map50"),
        _read(node, "sim_result", "map50"),
        _read(node, "eval_result", "map50"),
        _read(node, "metrics", "map50"),
        _read(node, "result", "map50"),
    ):
        fv = _fo(v)
        if fv is not None:
            return fv
    return None


def _extract_safety_line(node: dict[str, Any], fallback: float | None = None) -> float | None:
    for v in (
        node.get("safety_line"),
        node.get("threshold"),
        _read(node, "current_requirement", "threshold"),
        _read(node, "performance_signals", "threshold"),
        _read(node, "sim_result", "threshold"),
        _read(node, "eval_result", "requirement_threshold"),
        fallback,
    ):
        fv = _fo(v)
        if fv is not None:
            return fv
    return None


def _extract_status(node: dict[str, Any], map50: float | None, safety_line: float | None) -> str:
    req = node.get("current_requirement") if isinstance(node.get("current_requirement"), dict) else {}
    if isinstance(req.get("requirement_violated"), bool):
        return FAIL_STATUS if bool(req.get("requirement_violated")) else PASS_STATUS

    for v in (
        node.get("result_status"),
        node.get("status"),
        _read(node, "performance_signals", "status"),
        _read(node, "sim_result", "status"),
    ):
        s = _status(v)
        if s:
            return s

    if map50 is not None and safety_line is not None:
        return FAIL_STATUS if map50 < safety_line else PASS_STATUS

    failure_type = str(_read(node, "performance_signals", "failure_type") or "").strip().lower()
    return PASS_STATUS if failure_type in {"", "none", "normal_operation", "safe", "nominal"} else FAIL_STATUS


def _run_from_node(node: dict[str, Any], idx: int, fallback_safety_line: float | None) -> RunPoint | None:
    env = _extract_environment(node)
    if not env:
        return None
    map50 = _extract_map50(node)
    safety_line = _extract_safety_line(node, fallback_safety_line)
    status = _extract_status(node, map50, safety_line)
    run_id = str(
        node.get("scene_id")
        or node.get("scenario_id")
        or _read(node, "scenario", "scenario_id")
        or _read(node, "current_scenario", "scenario_id")
        or f"history_{idx:03d}"
    )
    return RunPoint(run_id=run_id, env=env, map50=map50, safety_line=safety_line, status=status)


def _same_env(a: dict[str, float], b: dict[str, float], tol: float = 1e-9) -> bool:
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= tol for k in a)


def _default_bounds(name: str, value: float) -> tuple[float, float]:
    lowered = name.lower()
    if "percent" in lowered:
        return 0.0, 100.0
    if "lux" in lowered:
        return 0.0, 20000.0
    if any(token in lowered for token in ("noise", "density")):
        return 0.0, 1.0
    if "wind" in lowered:
        return 0.0, 30.0
    if "delay" in lowered:
        return 0.0, 20.0
    span = max(abs(value) * 0.75, 1.0)
    lo = value - span
    hi = value + span
    if value >= 0 and lo < 0:
        lo = 0.0
    return lo, hi


def _is_weather(name: str) -> bool:
    l = name.lower()
    return any(t in l for t in ("wind", "fog", "rain", "snow", "weather"))


def _is_light(name: str) -> bool:
    l = name.lower()
    return any(t in l for t in ("illumination", "light", "brightness", "lux", "low_light"))


def _is_obstacle(name: str) -> bool:
    l = name.lower()
    return any(t in l for t in ("obstacle", "density"))


def _is_harmful_when_increasing(name: str) -> bool:
    lowered = name.lower()
    tokens = ("fog", "noise", "blur", "wind", "delay", "obstacle", "density", "rain", "snow")
    return any(token in lowered for token in tokens)


def _resolve_feature_ranges(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("feature_ranges"), dict):
        return payload["feature_ranges"]
    search_space = payload.get("search_space") if isinstance(payload.get("search_space"), dict) else {}
    bounds = search_space.get("bounds") if isinstance(search_space.get("bounds"), dict) else {}
    return bounds


def extract_environment_parameters(payload: dict[str, Any]) -> dict[str, float]:
    current_node = payload.get("current_scenario") if isinstance(payload.get("current_scenario"), dict) else payload
    env = _extract_environment(current_node)
    if env:
        return env
    env = _extract_environment(payload)
    if env:
        return env
    history = payload.get("scenario_history")
    if isinstance(history, list) and history:
        env = _extract_environment(history[-1])
        if env:
            return env
    raise ValueError("Environment parameters are missing.")


def build_parameter_specs(payload: dict[str, Any], env: dict[str, float]) -> list[ParameterSpec]:
    feature_ranges = _resolve_feature_ranges(payload)
    search_space = payload.get("search_space") if isinstance(payload.get("search_space"), dict) else {}
    weights = search_space.get("weights") if isinstance(search_space.get("weights"), dict) else {}
    mutable = search_space.get("mutable_parameters")
    mutable_set = set(map(str, mutable)) if isinstance(mutable, list) else None

    constraints = payload.get("scenario_constraints") if isinstance(payload.get("scenario_constraints"), dict) else {}
    allow_weather = bool(constraints.get("allow_weather_change", True))
    allow_light = bool(constraints.get("allow_lighting_change", True))
    allow_obstacle = bool(constraints.get("allow_obstacle_density_change", True))

    specs: list[ParameterSpec] = []
    for name, value in env.items():
        cfg = feature_ranges.get(name)
        if isinstance(cfg, dict):
            lo = _f(cfg.get("min"), value)
            hi = _f(cfg.get("max"), value)
        elif isinstance(cfg, list) and len(cfg) == 2:
            lo = _f(cfg[0], value)
            hi = _f(cfg[1], value)
        else:
            lo, hi = _default_bounds(name, value)

        if hi < lo:
            lo, hi = hi, lo
        if abs(hi - lo) < 1e-9:
            hi = lo + 1.0

        is_mutable = True if mutable_set is None else name in mutable_set
        if not allow_weather and _is_weather(name):
            is_mutable = False
        if not allow_light and _is_light(name):
            is_mutable = False
        if not allow_obstacle and _is_obstacle(name):
            is_mutable = False

        specs.append(
            ParameterSpec(
                name=name,
                lower=float(lo),
                upper=float(hi),
                weight=max(1e-6, _f(weights.get(name), 1.0)),
                mutable=is_mutable,
            )
        )
    return specs


def _collect_history_runs(payload: dict[str, Any], current_run: RunPoint) -> list[RunPoint]:
    nodes: list[dict[str, Any]] = []

    previous = payload.get("previous_scenario")
    if isinstance(previous, dict):
        nodes.append(previous)

    for key in ("scenario_history", "history", "execution_history", "run_history"):
        value = payload.get(key)
        if isinstance(value, list):
            nodes.extend([row for row in value if isinstance(row, dict)])
            if value:
                break

    runs = [run for idx, node in enumerate(nodes) if (run := _run_from_node(node, idx, current_run.safety_line)) is not None]
    if not runs:
        return [current_run]

    last = runs[-1]
    if last.run_id == current_run.run_id or _same_env(last.env, current_run.env):
        runs[-1] = current_run
    else:
        runs.append(current_run)
    return runs


def _find_previous_pass(runs: list[RunPoint], current_index: int) -> RunPoint | None:
    for i in range(current_index - 1, -1, -1):
        if runs[i].status == PASS_STATUS:
            return runs[i]
    return None


def _match_feature_name(name: str, names: list[str]) -> str | None:
    needle = _normalize_name(name)
    table = {_normalize_name(item): item for item in names}
    if needle in table:
        return table[needle]

    aliases = {
        "fog": "fog_density_percent",
        "noise": "camera_noise_level",
        "lowlight": "illumination_lux",
        "motionblur": "motion_blur_intensity",
        "zoomblur": "zoom_blur_intensity",
    }
    if needle in aliases and _normalize_name(aliases[needle]) in table:
        return table[_normalize_name(aliases[needle])]

    for normalized, original in table.items():
        if needle and (needle in normalized or normalized in needle):
            return original
    return None


class SimulationResultFunction:
    def __init__(
        self,
        payload: dict[str, Any],
        specs: list[ParameterSpec],
        runs: list[RunPoint],
        current_run: RunPoint,
        simulink_callable: Callable[..., Any] | None = None,
    ):
        self._payload = payload
        self._specs = specs
        self._feature_names = [spec.name for spec in specs]
        self._span = {spec.name: spec.span() for spec in specs}
        self._callable = simulink_callable or self._resolve_callable(payload)
        self.source = "simulink_callable" if self._callable is not None else "observed_results_fallback"

        self._observed_points: list[tuple[dict[str, float], float]] = []
        for run in runs:
            if run.map50 is None:
                continue
            self._observed_points.append((dict(run.env), float(run.map50)))
        if current_run.map50 is not None:
            self._observed_points.append((dict(current_run.env), float(current_run.map50)))

        for key in ("counterfactual_replay_results", "counterfactual_replays", "counterfactual_results", "counterfactual_replay_history"):
            value = payload.get(key)
            if not isinstance(value, list):
                continue
            for row in value:
                if not isinstance(row, dict):
                    continue
                env = _extract_environment(row)
                map50 = _extract_map50(row)
                if env and map50 is not None:
                    self._observed_points.append((env, float(map50)))

    def _resolve_callable(self, payload: dict[str, Any]) -> Callable[..., Any] | None:
        for key in (
            "simulink_callable",
            "simulation_callable",
            "simulink_result_callable",
            "sim_result_callable",
            "simulink_result_function",
        ):
            value = payload.get(key)
            if callable(value):
                return value
        return None

    def _extract_map50_from_result(self, result: Any) -> float | None:
        if isinstance(result, (int, float)) and not isinstance(result, bool):
            return float(result)
        if not isinstance(result, dict):
            return None
        return _extract_map50(result)

    def _invoke_callable(self, env: dict[str, float]) -> float | None:
        if self._callable is None:
            return None

        fn = self._callable
        result = None
        errors: list[Exception] = []

        try:
            result = fn(env)
        except Exception as exc:
            errors.append(exc)

        if result is None:
            try:
                vector = [float(env[name]) for name in self._feature_names]
                result = fn(vector)
            except Exception as exc:
                errors.append(exc)

        if result is None:
            try:
                vector = [float(env[name]) for name in self._feature_names]
                result = fn(*vector)
            except Exception as exc:
                errors.append(exc)

        map50 = self._extract_map50_from_result(result)
        if map50 is None and errors:
            return None
        return map50

    def _distance(self, a: dict[str, float], b: dict[str, float]) -> float:
        accum = 0.0
        for name in self._feature_names:
            da = float(a.get(name, 0.0))
            db = float(b.get(name, 0.0))
            accum += ((da - db) / self._span[name]) ** 2
        return float(np.sqrt(accum))

    def _fallback_from_observed(self, env: dict[str, float]) -> float | None:
        if not self._observed_points:
            return None

        for ref_env, map50 in self._observed_points:
            if self._distance(ref_env, env) <= 1e-9:
                return float(map50)

        scored: list[tuple[float, float]] = []
        for ref_env, map50 in self._observed_points:
            dist = self._distance(ref_env, env)
            scored.append((dist, float(map50)))
        scored.sort(key=lambda item: item[0])

        k = min(5, len(scored))
        nearest = scored[:k]
        weighted_sum = 0.0
        total_weight = 0.0
        for dist, map50 in nearest:
            weight = 1.0 / (dist + 1e-6)
            weighted_sum += weight * map50
            total_weight += weight

        if total_weight <= 1e-12:
            return float(nearest[0][1])
        return float(weighted_sum / total_weight)

    def evaluate_map50(self, env: dict[str, float]) -> float:
        map50 = self._invoke_callable(env)
        if map50 is None:
            map50 = self._fallback_from_observed(env)
        if map50 is None:
            raise ValueError("Unable to evaluate simulation result function for KernelSHAP.")
        return float(map50)

    def predict_batch(
        self,
        x_matrix: np.ndarray,
        target_metric: str,
        safety_line: float | None,
    ) -> np.ndarray:
        x_arr = np.asarray(x_matrix, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        outputs: list[float] = []
        for row in x_arr:
            env = {name: float(row[idx]) for idx, name in enumerate(self._feature_names)}
            map50 = self.evaluate_map50(env)
            if target_metric == "safety_margin" and safety_line is not None:
                outputs.append(float(map50 - safety_line))
            else:
                outputs.append(float(map50))
        return np.asarray(outputs, dtype=float)


def _build_background_matrix(
    specs: list[ParameterSpec],
    runs: list[RunPoint],
    current_run: RunPoint,
) -> np.ndarray:
    names = [spec.name for spec in specs]

    rows: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()

    def push(env: dict[str, float]) -> None:
        vector = [float(env.get(name, current_run.env.get(name, 0.0))) for name in names]
        key = tuple(round(v, 9) for v in vector)
        if key in seen:
            return
        seen.add(key)
        rows.append(vector)

    for run in runs:
        push(run.env)
    push(current_run.env)

    if len(rows) < 2:
        midpoint = [float((spec.lower + spec.upper) / 2.0) for spec in specs]
        rows.append(midpoint)

    if len(rows) < 3:
        lower = [float(spec.lower) for spec in specs]
        upper = [float(spec.upper) for spec in specs]
        rows.extend([lower, upper])

    return np.asarray(rows, dtype=float)


def _compute_kernel_shap(
    sim_fn: SimulationResultFunction,
    specs: list[ParameterSpec],
    runs: list[RunPoint],
    current_run: RunPoint,
    target_metric: str,
) -> tuple[dict[str, float], float, str | None]:
    names = [spec.name for spec in specs]
    current_vector = np.asarray([float(current_run.env[name]) for name in names], dtype=float)
    background = _build_background_matrix(specs=specs, runs=runs, current_run=current_run)

    def predict(x_matrix: np.ndarray) -> np.ndarray:
        return sim_fn.predict_batch(
            x_matrix=x_matrix,
            target_metric=target_metric,
            safety_line=current_run.safety_line,
        )

    try:
        explainer = shap.KernelExplainer(predict, background)
        nsamples = max(40, min(220, 20 + (10 * len(names))))
        raw_values = explainer.shap_values(current_vector.reshape(1, -1), nsamples=nsamples)

        if isinstance(raw_values, list):
            arr = np.asarray(raw_values[0], dtype=float)
        else:
            arr = np.asarray(raw_values, dtype=float)
        if arr.ndim == 2:
            arr = arr[0]

        expected = explainer.expected_value
        if isinstance(expected, (list, tuple, np.ndarray)):
            expected_value = float(np.asarray(expected).reshape(-1)[0])
        else:
            expected_value = float(expected)

        shap_values = {name: float(arr[idx]) for idx, name in enumerate(names)}
        return shap_values, expected_value, None
    except Exception as exc:
        fallback = {name: 0.0 for name in names}
        baseline = current_run.map50 if current_run.map50 is not None else 0.0
        return fallback, float(baseline), f"KernelSHAP fallback: {exc}"


def _shap_prior(payload: dict[str, Any], specs: list[ParameterSpec]) -> dict[str, float]:
    xai_signals = payload.get("xai_signals") if isinstance(payload.get("xai_signals"), dict) else {}
    rows = xai_signals.get("dominant_factors") if isinstance(xai_signals.get("dominant_factors"), list) else []
    names = [spec.name for spec in specs]

    out: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        matched = _match_feature_name(str(row.get("name", "")), names)
        if not matched:
            continue
        imp = abs(_f(row.get("importance"), 0.0))
        if imp > 0.0:
            out[matched] = max(out.get(matched, 0.0), imp)

    max_value = max(out.values()) if out else 0.0
    return {k: v / max_value for k, v in out.items()} if max_value > 1e-12 else {}


def _pass_direction(feature: str) -> str:
    if _is_light(feature):
        return "decrease"
    if _is_harmful_when_increasing(feature):
        return "increase"
    return "adjust_near_boundary"


def _fail_direction(feature: str) -> str:
    if _is_light(feature):
        return "increase"
    if _is_harmful_when_increasing(feature):
        return "decrease"
    return "adjust_near_boundary"


def _risk_level(current_run: RunPoint, previous_run: RunPoint | None) -> tuple[str, float, float]:
    map_drop = 0.0
    margin_drop = 0.0
    if previous_run and previous_run.map50 is not None and current_run.map50 is not None:
        map_drop = max(0.0, previous_run.map50 - current_run.map50)
    prev_margin = previous_run.safety_margin() if previous_run else None
    cur_margin = current_run.safety_margin()
    if prev_margin is not None and cur_margin is not None:
        margin_drop = max(0.0, prev_margin - cur_margin)

    if current_run.status == FAIL_STATUS:
        return "failed", map_drop, margin_drop

    cur_margin = current_run.safety_margin()
    if cur_margin is not None and cur_margin <= 0.03:
        return "near_boundary", map_drop, margin_drop
    if map_drop > 1e-9 or margin_drop > 1e-9:
        return "degrading", map_drop, margin_drop
    return "safe", map_drop, margin_drop


def _build_top_features(
    specs: list[ParameterSpec],
    shap_values: dict[str, float],
    prior: dict[str, float],
    current_run: RunPoint,
    status: str,
    previous_run: RunPoint | None,
    shap_error: str | None,
    target_metric: str,
) -> list[dict[str, Any]]:
    abs_sum = sum(abs(v) for v in shap_values.values())
    if abs_sum <= 1e-12:
        abs_sum = 1.0
    low_signal = max((abs(v) for v in shap_values.values()), default=0.0) <= 1e-9
    prior_sum = sum(max(prior.get(spec.name, 0.0), 0.0) for spec in specs)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.name not in shap_values:
            continue
        value = float(shap_values[spec.name])
        if low_signal and prior_sum > 1e-12:
            importance = max(prior.get(spec.name, 0.0), 0.0) / prior_sum
        else:
            importance = abs(value) / abs_sum
            if importance <= 1e-12 and prior.get(spec.name, 0.0) > 0.0:
                importance = 0.1 * prior[spec.name]

        direction = _pass_direction(spec.name) if status == PASS_STATUS else _fail_direction(spec.name)

        if previous_run and spec.name in previous_run.env:
            observed = (
                f"{spec.name} 값이 이전 실행 {previous_run.env[spec.name]:.4f}에서 "
                f"현재 {current_run.env[spec.name]:.4f}로 변했습니다."
            )
        else:
            observed = f"{spec.name}의 현재 값이 성능 변화에 기여하고 있습니다."

        metric_label = "safety_margin" if target_metric == "safety_margin" else "map50"
        if low_signal and prior_sum > 1e-12:
            reason = (
                f"KernelSHAP 신호가 약해 실행 이력 기반 우선순위를 사용했습니다. "
                f"LLM 반사실 생성 시 {spec.name}을(를) '{direction}' 방향으로 우선 조정하세요."
            )
        else:
            reason = (
                f"{metric_label} 기준 KernelSHAP 값 {value:+.5f}를 바탕으로 "
                f"{spec.name}을(를) '{direction}' 방향으로 조정하는 것이 유효합니다."
            )
        if shap_error:
            reason = f"{reason} (참고: {shap_error})"

        rows.append(
            {
                "feature": spec.name,
                "shap_value": round(value, 6),
                "shap_importance": round(importance, 6),
                "attribution_score": round(importance, 6),
                "direction": direction,
                "observed_effect": observed,
                "reason": reason,
            }
        )

    rows.sort(key=lambda row: row.get("shap_importance", 0.0), reverse=True)
    return rows[:8]


def _boundary_focus(status: str, risk_level: str, top_features: list[dict[str, Any]]) -> list[str]:
    if status == FAIL_STATUS:
        return [str(row.get("feature", "")) for row in top_features[:3] if row.get("feature")]
    if risk_level in {"degrading", "near_boundary"}:
        return [str(row.get("feature", "")) for row in top_features[:2] if row.get("feature")]
    return []


def _guidance(status: str, risk_level: str, top_features: list[dict[str, Any]], previous_pass: RunPoint | None) -> str:
    if not top_features:
        if status == PASS_STATUS:
            return "현재 실행은 PASS입니다. 경계 근처 이력을 추가 수집해 KernelSHAP 가이던스를 안정화하세요."
        return "현재 실행은 FAIL입니다. 단일 변수 경계 탐색을 추가 수행해 KernelSHAP 기여도를 재계산하세요."

    focus = ", ".join(f"{row['feature']} ({row['direction']})" for row in top_features[:3])
    if status == PASS_STATUS:
        prefix = "현재 실행은 PASS이며 안전합니다." if risk_level == "safe" else (
            "현재 실행은 PASS지만 성능 저하가 관찰됩니다." if risk_level == "degrading" else "현재 실행은 PASS지만 경계에 가깝습니다."
        )
        return (
            f"{prefix} 다음 반사실 시나리오 생성에서는 {focus}를 중심으로 점진적으로 가혹도를 높이되 현실성을 유지하세요."
        )

    base = f"최근 PASS 기준: {previous_pass.run_id}" if previous_pass else "PASS 기준 시나리오 없음"
    return (
        f"현재 실행은 FAIL입니다. 경계 탐색은 {focus} 중심으로 수행하고({base}), 한 번에 한 변수씩 조정하세요."
    )


def _guidance_sources(runs: list[RunPoint], sim_source: str, shap_error: str | None) -> list[str]:
    src = ["scenario_history" if len(runs) > 1 else "current_run", sim_source, "kernelshap"]
    if shap_error:
        src.append("kernelshap_fallback")
    return src


def _legacy_candidates(
    top_features: list[dict[str, Any]],
    current_run: RunPoint,
    specs: list[ParameterSpec],
    limit: int,
    prefix: str,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    smap = {spec.name: spec for spec in specs}
    base_margin = current_run.decision_margin()
    selected = top_features[:limit] if top_features else []
    if not selected:
        selected = [
            {
                "feature": "no_feature_detected",
                "shap_importance": 0.0,
                "direction": "keep",
                "reason": "유효한 SHAP 신호를 찾지 못했습니다.",
                "observed_effect": "현재 입력만으로는 설명 가능한 변수를 충분히 추출하지 못했습니다.",
            }
        ]

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        feature = str(row.get("feature", ""))
        direction = str(row.get("direction", "keep"))
        base_value = float(current_run.env.get(feature, 0.0))
        env = dict(current_run.env)

        normalized_distance = 0.0
        candidate_value = base_value
        spec = smap.get(feature)
        if spec is not None and feature in current_run.env:
            step = spec.span() * max(0.02, 0.15 * _f(row.get("shap_importance"), 0.0))
            if direction == "increase":
                candidate_value = spec.clamp(base_value + step)
            elif direction == "decrease":
                candidate_value = spec.clamp(base_value - step)
            elif direction == "adjust_near_boundary":
                candidate_value = spec.clamp(base_value + (0.1 * step if current_run.status == PASS_STATUS else -0.1 * step))
            env[feature] = float(candidate_value)
            normalized_distance = abs(candidate_value - base_value) / spec.span()

        swing = 0.02 + 0.06 * max(0.0, min(_f(row.get("shap_importance"), 0.0), 1.0))
        est_margin = base_margin + swing if current_run.status == PASS_STATUS else base_margin - swing
        pred = FAIL_STATUS if est_margin > 0.0 else PASS_STATUS

        changes = [
            {
                "parameter": feature,
                "from": float(base_value),
                "to": float(candidate_value),
                "delta": float(candidate_value - base_value),
                "normalized_l1": float(normalized_distance),
            }
        ]

        rows.append(
            {
                "candidate_id": f"{prefix}_{idx:02d}",
                "predicted_status": pred,
                "decision_margin": float(est_margin),
                "distance_to_boundary": float(abs(est_margin)),
                "distance_l1_normalized": float(normalized_distance),
                "changed_parameters": changes,
                "parameter_changes": changes,
                "counterfactual_environment": env,
                "summary_explanation": f"{row.get('reason', '')} {row.get('observed_effect', '')}".strip(),
                "attribution_score": float(_f(row.get("shap_importance"), 0.0)),
                "direction": direction,
                "shap_value": float(_f(row.get("shap_value"), 0.0)),
                "shap_importance": float(_f(row.get("shap_importance"), 0.0)),
                "observed_effect": str(row.get("observed_effect", "")),
            }
        )
    return rows


def generate_counterfactual_and_boundary(
    payload: dict[str, Any],
    target_status: str | None = None,
    mode: str = "auto",
    num_counterfactuals: int = 3,
    num_boundary_candidates: int = 5,
    random_seed: int = 42,
    simulink_callable: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    del mode, random_seed
    if num_counterfactuals <= 0 or num_boundary_candidates <= 0:
        raise ValueError("num_counterfactuals and num_boundary_candidates must be >= 1")

    current_node = payload.get("current_scenario") if isinstance(payload.get("current_scenario"), dict) else payload
    current_env = extract_environment_parameters(payload)
    specs = build_parameter_specs(payload, current_env)
    if not any(spec.mutable for spec in specs):
        raise ValueError("No mutable parameter found. Check scenario_constraints/search_space/feature_ranges.")

    current_map50 = _extract_map50(current_node)
    if current_map50 is None:
        current_map50 = _extract_map50(payload)

    safety_line = _extract_safety_line(current_node, _extract_safety_line(payload, None))
    status = _extract_status(current_node if isinstance(current_node, dict) else payload, current_map50, safety_line)

    current_run = RunPoint(
        run_id=str(payload.get("scene_id") or payload.get("scenario_id") or _read(current_node, "scenario_id") or "current_scene"),
        env=current_env,
        map50=current_map50,
        safety_line=safety_line,
        status=status,
    )

    runs = _collect_history_runs(payload, current_run)
    current_idx = len(runs) - 1
    previous_run = runs[current_idx - 1] if current_idx > 0 else None
    previous_pass = _find_previous_pass(runs, current_idx)

    target_metric = str(payload.get("shap_target_metric", "map50")).strip().lower()
    if target_metric not in {"map50", "safety_margin"}:
        target_metric = "map50"

    sim_fn = SimulationResultFunction(
        payload=payload,
        specs=specs,
        runs=runs,
        current_run=current_run,
        simulink_callable=simulink_callable,
    )

    shap_values, expected_value, shap_error = _compute_kernel_shap(
        sim_fn=sim_fn,
        specs=specs,
        runs=runs,
        current_run=current_run,
        target_metric=target_metric,
    )

    prior = _shap_prior(payload, specs)
    top_features = _build_top_features(
        specs=specs,
        shap_values=shap_values,
        prior=prior,
        current_run=current_run,
        status=current_run.status,
        previous_run=previous_run,
        shap_error=shap_error,
        target_metric=target_metric,
    )

    risk_level, map_drop, margin_drop = _risk_level(current_run, previous_run)
    boundary_focus = _boundary_focus(current_run.status, risk_level, top_features)
    llm_guidance = _guidance(current_run.status, risk_level, top_features, previous_pass)

    guidance_sources = _guidance_sources(runs, sim_fn.source, shap_error)
    guidance_weights = {source: round(1.0 / len(guidance_sources), 4) for source in guidance_sources}

    resolved_target = str(target_status or "").strip().upper()
    if resolved_target not in {PASS_STATUS, FAIL_STATUS}:
        resolved_target = PASS_STATUS if current_run.status == FAIL_STATUS else FAIL_STATUS

    counterfactual_candidates = _legacy_candidates(
        top_features=top_features,
        current_run=current_run,
        specs=specs,
        limit=num_counterfactuals,
        prefix="cf",
    )
    boundary_candidates = _legacy_candidates(
        top_features=top_features,
        current_run=current_run,
        specs=specs,
        limit=num_boundary_candidates,
        prefix="bd",
    )

    decision_margin = current_run.decision_margin()
    mutable_parameters = [spec.name for spec in specs if spec.mutable]

    threshold_condition = "PASS if map50 >= safety_line, FAIL if map50 < safety_line"
    if current_run.map50 is None or current_run.safety_line is None:
        threshold_condition = "PASS/FAIL inferred from result_status or requirement_violated"

    analysis_context = {
        "history_size": len(runs),
        "used_previous_pass": previous_pass is not None,
        "kernelshap_target": target_metric,
        "kernelshap_expected_value": round(expected_value, 6),
        "kernelshap_error": shap_error,
        "map_drop_from_previous": round(map_drop, 6),
        "margin_drop_from_previous": round(margin_drop, 6),
    }

    counterfactual_output = {
        "schema_version": "xai_counterfactual_v1",
        "scene_id": current_run.run_id,
        "search_mode": "kernelshap_direct_simulation",
        "search_method": METHOD_NAME,
        "source_status": current_run.status,
        "target_status": resolved_target,
        "threshold_condition": threshold_condition,
        "mutable_parameters": mutable_parameters,
        "guidance": {
            "guidance_sources": guidance_sources,
            "guidance_weights": guidance_weights,
            "analysis_context": analysis_context,
        },
        "base_case": {
            "current_environment": current_run.env,
            "predicted_status": current_run.status,
            "decision_margin": float(decision_margin),
            "distance_to_boundary": float(abs(decision_margin)),
            "map50": current_run.map50,
            "safety_line": current_run.safety_line,
            "safety_margin": current_run.safety_margin(),
        },
        "minimal_change_candidates": counterfactual_candidates,
        "method": METHOD_NAME,
        "result_status": current_run.status,
        "risk_level": risk_level,
        "top_features": top_features,
        "boundary_focus_features": boundary_focus,
        "llm_guidance": llm_guidance,
    }

    closest = boundary_candidates[0] if boundary_candidates else None
    boundary_output = {
        "schema_version": "xai_boundary_v1",
        "scene_id": current_run.run_id,
        "search_mode": "kernelshap_direct_simulation",
        "search_method": METHOD_NAME,
        "threshold_condition": threshold_condition,
        "guidance": {
            "guidance_sources": guidance_sources,
            "guidance_weights": guidance_weights,
            "analysis_context": analysis_context,
        },
        "base_case": {
            "current_environment": current_run.env,
            "predicted_status": current_run.status,
            "decision_margin": float(decision_margin),
            "distance_to_boundary": float(abs(decision_margin)),
            "map50": current_run.map50,
            "safety_line": current_run.safety_line,
            "safety_margin": current_run.safety_margin(),
        },
        "nearest_boundary_candidates": boundary_candidates,
        "boundary_analysis": {
            "base_distance_to_boundary": float(abs(decision_margin)),
            "closest_boundary_distance": float(closest["distance_l1_normalized"]) if closest else None,
            "closest_boundary_margin_abs": float(abs(_f(closest["decision_margin"], 0.0))) if closest else None,
            "interpretation": "KernelSHAP explains the direct simulation result function and highlights boundary-sensitive variables.",
        },
        "method": METHOD_NAME,
        "result_status": current_run.status,
        "risk_level": risk_level,
        "top_features": top_features,
        "boundary_focus_features": boundary_focus,
        "llm_guidance": llm_guidance,
    }

    return counterfactual_output, boundary_output


