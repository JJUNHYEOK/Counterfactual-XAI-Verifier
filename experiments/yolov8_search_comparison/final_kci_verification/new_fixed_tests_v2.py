"""Visibility-corrected T0--T2 v2 plan; v1 remains preserved."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import new_fixed_tests as base


PLAN_ID = "new_fixed_tests_t0_t2_v2"
V1_SCENARIO_DEFINITIONS = base.scenario_definitions


def configure_v2_globals() -> None:
    base.PLAN_ID = PLAN_ID
    base.SUPERSEDES_PLAN = "new_fixed_tests_t0_t2_v1"
    base.CORRECTION_NOTE = (
        "The v1 MATLAB-only visible-GT gate failed before any T0-T2 YOLO inference: "
        "T0 had zero visible objects and T1 had zero visible persons. Only T0/T1 "
        "trajectory endpoints and object positions/sizes were brought into the supported "
        "camera/terrain range. T2 was copied unchanged. All v1 configs, renders and audits remain preserved."
    )
    base.CONFIG_ROOT = base.HERE / "config" / PLAN_ID
    base.SCENARIO_ROOT = base.HERE / "scenarios" / PLAN_ID
    base.OUTPUT_ROOT = base.HERE / "outputs" / PLAN_ID


def scenario_definitions_v2() -> list[dict[str, Any]]:
    scenarios = V1_SCENARIO_DEFINITIONS()
    t0, t1, _t2 = scenarios

    t0["trajectory"] = {
        "name": "서향 이탈·상승 중거리 관측 (가시성 교정)",
        "start_xyz": [10.0, -11.0, 47.0],
        "end_xyz": [-28.0, -6.0, 60.0],
    }
    t0["objects"] = [
        base.obj(1, 1, [15.0, -13.0], [0.58, 1.95]),
        base.obj(2, 2, [20.5, -9.5], [1.30, 1.55]),
        base.obj(3, 1, [26.0, -5.5], [0.62, 2.05]),
        base.obj(4, 2, [32.0, -1.5], [1.05, 1.45]),
        base.obj(5, 1, [38.0, 2.0], [0.56, 1.90]),
        base.obj(6, 2, [43.0, -7.0], [1.38, 1.72]),
    ]
    t0["major_changes"] = [
        "x축 역방향 이탈과 13 m 상승을 결합",
        "v1 GT 실패 후 YOLO 이전에 끝점 거리와 객체 x 위치를 지원 범위로 축소",
        "객체별 물리 크기·비대칭 횡간격·별도 보행 위상은 유지",
    ]
    t0["difference_from_existing"] = "학습·검증·S0~S4에 없는 서향 이탈과 상승의 동시 조합"

    t1["trajectory"] = {
        "name": "북서-남동 대각 하강 접근 (가시성 교정)",
        "start_xyz": [-38.0, 12.0, 62.0],
        "end_xyz": [6.0, -10.0, 47.0],
    }
    t1["objects"] = [
        base.obj(1, 2, [-3.0, 8.0], [1.18, 1.56]),
        base.obj(2, 1, [6.0, 5.0], [0.62, 2.05]),
        base.obj(3, 1, [16.0, 1.0], [0.66, 2.10]),
        base.obj(4, 2, [27.0, -3.0], [1.38, 1.72]),
        base.obj(5, 1, [37.0, -6.5], [0.60, 2.00]),
        base.obj(6, 2, [44.0, -9.0], [1.20, 1.60]),
    ]
    t1["major_changes"] = [
        "x·y·z가 동시에 변하는 대각 하강 접근",
        "v1 GT 실패 후 YOLO 이전에 경로 거리와 사람 위치·크기를 지원 범위로 교정",
        "불규칙 클래스 순서와 빠른 보행 위상은 유지",
    ]
    t1["difference_from_existing"] = "15 m 하강과 22 m 횡이동을 결합한 경사 접근으로 기존 경로와 시작·종료점이 다름"
    return scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "preflight"))
    args = parser.parse_args()
    configure_v2_globals()
    base.scenario_definitions = scenario_definitions_v2
    if args.command == "register":
        base.register()
    else:
        base.preflight()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
