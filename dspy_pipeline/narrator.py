"""LLM-based natural-language narrator for the demo UI.

Three DSPy modules turn raw simulation telemetry into Korean security-report
prose for the Streamlit dashboard:

    EdgeCaseDescriber  — one scenario  → 환경/결과/임무 영향 (3~5문장)
    ScenarioComparison — prev vs curr → 환경 변화 + 결과 변화 (2~3문장)
    MissionSummary     — 전체 history → 종합 보고서 (요약·경계·시사점·권고)

All three take `mission_context` (loaded from mission_context.json) so the
LLM grounds every output in the border-mountain-surveillance brief.
The reports avoid raw numbers and prefer qualitative language ('짙은 안개',
'저조도', '임계 진입' 등).
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy


# ─────────────────────────────────────────────────────────────────────────────
# Mission context loading (singleton)
# ─────────────────────────────────────────────────────────────────────────────

_MISSION_CONTEXT: str | None = None


def load_mission_context(path: str | Path = "mission_context.json") -> str:
    global _MISSION_CONTEXT
    if _MISSION_CONTEXT is None:
        _MISSION_CONTEXT = Path(path).read_text(encoding="utf-8")
    return _MISSION_CONTEXT


# ─────────────────────────────────────────────────────────────────────────────
# Signatures
# ─────────────────────────────────────────────────────────────────────────────

class EdgeCaseDescriber(dspy.Signature):
    """국경 산악 감시 임무에서 한 시나리오의 환경 조건과 시뮬 결과를
    안보·국방 보고서 톤으로 자연어로 풀어 설명한다.

    원칙:
      - 수치(0.45, 6500 lx 등)를 그대로 노출하지 말고 정성적 표현 사용
        (예: '저조도', '짙은 안개', '센서 잡음 누적')
      - 임무 영향을 항상 마지막에 명시 ('침입자 식별 가능', '보고 누락 위험' 등)
      - 3~5문장의 단일 문단
    """

    mission_context: str = dspy.InputField(desc="임무 정의 JSON")
    env_summary:     str = dspy.InputField(desc="현재 환경 (정성적 dict): fog_level, illum_level, noise_level, mAP_band, intruders_missed")
    sim_outcome:     str = dspy.InputField(desc="시뮬 결과: PASS/FAIL, 어떤 REQ가 위반되었는지")
    boundary_status: str = dspy.InputField(desc="현재 시나리오가 boundary 근처인지 (예: 'far_inside_pass', 'near_boundary', 'just_failed')")

    edge_case_narrative: str = dspy.OutputField(
        desc="3~5문장의 자연어 한 단락. 환경 → 카메라 영향 → 탐지 결과 → 임무 의미 순서.")


class ScenarioComparison(dspy.Signature):
    """이전 시나리오와 현재 시나리오를 비교해 무엇이 더 가혹해졌고
    그 결과 임무가 어떻게 변화했는지를 안보 보고서 톤으로 2~3문장으로 기술한다.
    """

    mission_context:    str = dspy.InputField(desc="임무 정의 JSON")
    previous_summary:   str = dspy.InputField(desc="이전 시나리오 요약 (env + result + edge_case_narrative)")
    current_summary:    str = dspy.InputField(desc="현재 시나리오 요약 (env + result + edge_case_narrative)")

    comparison_narrative: str = dspy.OutputField(
        desc="2~3문장. 환경 변화의 본질 + 결과 전환의 의미 + 안보적 함의.")


class MissionSummary(dspy.Signature):
    """전체 시뮬 이력을 임무 보고서 형식으로 요약한다.
    국경 산악 감시 관점에서 발견된 실패 경계와 시사점을 명시.
    """

    mission_context:   str = dspy.InputField(desc="임무 정의 JSON")
    iteration_history: str = dspy.InputField(desc="N개 시나리오의 압축된 history JSON 배열")
    boundary_findings: str = dspy.InputField(desc="발견된 PASS/FAIL 경계 위치 (정성적 기술)")
    shap_dominant:     str = dspy.InputField(desc="가장 영향력 큰 환경 변수 (SHAP 기준)")

    executive_summary:     str = dspy.OutputField(desc="3~4문장 요약: 임무 검증 N회, 무엇을 발견했는지.")
    failure_boundary_text: str = dspy.OutputField(desc="발견된 임무 실패 경계의 정성적 기술 (수치 없이).")
    security_implications: str = dspy.OutputField(desc="안보 관점 시사점, '- ' 로 시작하는 bullet 3~5개.")
    recommendations:       str = dspy.OutputField(desc="운용 권고사항, '- ' 로 시작하는 bullet 2~4개.")


# ─────────────────────────────────────────────────────────────────────────────
# Modules (thin wrappers — main pipeline already configures dspy.LM)
# ─────────────────────────────────────────────────────────────────────────────

class Narrator(dspy.Module):
    """One-stop narrator: bundles the three signatures."""

    def __init__(self) -> None:
        super().__init__()
        self.describe_edge_case = dspy.ChainOfThought(EdgeCaseDescriber)
        self.compare_scenarios  = dspy.ChainOfThought(ScenarioComparison)
        self.summarise_mission  = dspy.ChainOfThought(MissionSummary)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — translate raw telemetry into qualitative inputs for the LLM
# ─────────────────────────────────────────────────────────────────────────────

def qualitative_env(record: dict) -> dict:
    """Convert numeric env params to qualitative bands the LLM can use."""
    fog   = record.get("fog_density_percent", 0)
    illum = record.get("illumination_lux",    8000)
    noise = record.get("camera_noise_level",  0)
    map50 = record.get("map50",               0.7)
    return {
        "fog_level":      _band(fog,   [(15, "맑음"), (35, "옅은 안개"), (60, "짙은 안개"), (101, "극심한 안개")]),
        "illum_level":    _band(illum, [(2000, "야간 수준 저조도"), (5500, "흐린 오후"), (10000, "정오 일조"), (20001, "강한 일조")]),
        "noise_level":    _band(noise, [(0.10, "잡음 거의 없음"), (0.25, "경미한 잡음"), (0.45, "중간 잡음"), (0.61, "심한 잡음")]),
        "mAP_band":       _band(map50, [(0.30, "탐지 거의 실패"), (0.50, "임계 미달"), (0.65, "임계 근접"), (1.01, "안정")]),
        "all_passed":     bool(record.get("all_passed", False)),
        "violated_count": int(record.get("violated_count", 0)),
        "worst_run":      int(record.get("worst_run", 0)),
    }


def _band(value: float, breaks: list[tuple[float, str]]) -> str:
    for upper, label in breaks:
        if value < upper:
            return label
    return breaks[-1][1]


def boundary_status(record: dict, last_pass_env: dict | None,
                    last_fail_env: dict | None) -> str:
    """Categorise where the scenario sits relative to the boundary."""
    if record["all_passed"]:
        if last_fail_env is None:
            return "far_inside_pass"
        return "near_boundary_pass" if record["map50"] < 0.60 else "comfortable_pass"
    else:
        if last_pass_env is None:
            return "always_failing_regime"
        return "just_failed" if record["map50"] > 0.40 else "deep_failure"
