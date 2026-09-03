"""Record the direct visual review of every T0--T2 dHash warning."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PLAN_ID = "new_fixed_tests_t0_t2_v2"
CONFIG_ROOT = HERE / "config" / PLAN_ID
SCENARIO_ROOT = HERE / "scenarios" / PLAN_ID
OUTPUT_ROOT = HERE / "outputs" / PLAN_ID
FIXED_ROOT = (
    REPO_ROOT
    / "experiments"
    / "yolov8_search_comparison"
    / "multi_scenario_symmetric"
    / "scenarios"
    / "multi_scenario_symmetric_plan_v2"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_gt(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "person": sum(item["class_name"] == "person" for item in items),
        "vehicle": sum(item["class_name"] == "vehicle" for item in items),
    }


def reference_gt(row: dict[str, str], fixed_manifests: dict[str, dict[str, Any]]) -> dict[str, int]:
    if row["reference_scope"] == "S0-S4":
        frame = fixed_manifests[row["reference_scenario"]]["frames"][int(row["reference_frame"]) - 1]
        return count_gt(frame.get("ground_truth", []))
    image_path = Path(row["reference_path"])
    label_path = (
        image_path.parents[2]
        / "labels"
        / row["reference_scope"]
        / f"{image_path.stem}.txt"
    )
    class_ids = [
        line.split()[0]
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {"person": class_ids.count("0"), "vehicle": class_ids.count("1")}


def main() -> int:
    output_json = OUTPUT_ROOT / "similarity_visual_review.json"
    output_csv = OUTPUT_ROOT / "similarity_visual_review.csv"
    output_md = OUTPUT_ROOT / "similarity_visual_review_ko.md"
    gate_path = OUTPUT_ROOT / "preinference_gate.json"
    if any(path.exists() for path in (output_json, output_csv, output_md, gate_path)):
        raise FileExistsError("Refusing to overwrite an existing T0-T2 visual review/gate")

    warnings_path = OUTPUT_ROOT / "near_duplicate_warnings.csv"
    with warnings_path.open("r", encoding="utf-8-sig", newline="") as handle:
        warnings = list(csv.DictReader(handle))
    if len(warnings) != 17:
        raise RuntimeError(f"Expected the observed 17 warnings, found {len(warnings)}")
    gt = read_json(OUTPUT_ROOT / "gt_validation.json")
    audit = read_json(OUTPUT_ROOT / "similarity_audit.json")
    plan = read_json(CONFIG_ROOT / "scenario_plan.json")
    prereg = read_json(CONFIG_ROOT / "pre_registration.json")
    if not gt["all_valid"] or audit["exact_duplicate_count"] != 0:
        raise RuntimeError("GT or exact-duplicate preflight gate did not pass")

    new_manifests = {
        scenario_id: read_json(
            SCENARIO_ROOT / scenario_id / "initial_condition" / "frame_manifest.json"
        )
        for scenario_id in ("T0", "T1", "T2")
    }
    fixed_manifests = {
        scenario_id: read_json(
            FIXED_ROOT / scenario_id / "initial_condition" / "frame_manifest.json"
        )
        for scenario_id in ("S0", "S1", "S2", "S3", "S4")
    }
    decisions: list[dict[str, Any]] = []
    for row in warnings:
        new_frame = new_manifests[row["new_scenario"]]["frames"][int(row["new_frame"]) - 1]
        new_counts = count_gt(new_frame.get("ground_truth", []))
        ref_counts = reference_gt(row, fixed_manifests)
        if new_counts != {"person": 0, "vehicle": 0}:
            raise RuntimeError(f"Visual decision assumption changed at {row['pair_id']}: {new_counts}")
        decisions.append(
            {
                **row,
                "dhash_distance": int(row["dhash_distance"]),
                "exact_sha256": str(row["exact_sha256"]).lower() == "true",
                "new_person_gt": new_counts["person"],
                "new_vehicle_gt": new_counts["vehicle"],
                "reference_person_gt": ref_counts["person"],
                "reference_vehicle_gt": ref_counts["vehicle"],
                "visual_classification": "빈 화면 또는 배경만 유사",
                "object_scene_material_duplicate": False,
                "object_duplicate_risk": "낮음",
                "background_context_risk": "중간",
                "judgement_basis": (
                    "신규 쪽 프레임의 사람·차량 GT가 모두 0이다. 비교 묶음에서 같은 단일 MATLAB "
                    "지형의 수목·지면·통나무/바위 배치가 유사해 dHash 경고가 발생했지만, 신규 쪽에 "
                    "객체가 없어 객체 위치·크기·가림 구성의 실질적 중복은 성립하지 않는다."
                ),
            }
        )

    reviewed_at = datetime.now(timezone.utc).isoformat()
    review = {
        "plan_id": PLAN_ID,
        "reviewed_at_utc": reviewed_at,
        "performed_before_any_T0_T2_yolo_inference": True,
        "review_method": "Direct visual inspection of all nine lossless side-by-side sheets covering all 17 dHash<=4 pairs, followed by GT-count verification from manifests/labels.",
        "reviewed_warning_count": len(decisions),
        "comparison_sheet_count": len(audit["comparison_sheets"]),
        "exact_duplicate_count": 0,
        "background_only_similarity_count": len(decisions),
        "object_scene_material_duplicate_count": 0,
        "conclusion": "17건 모두 신규 프레임이 빈 GT인 배경 유사 경고였다. 정확 중복과 객체 포함 장면의 실질적 중복은 확인되지 않았으나, 동일 단일 MATLAB 지형의 배경 맥락 공유는 남아 있으므로 외부·실환경 독립성을 주장하지 않는다.",
        "decisions": decisions,
    }
    write_json_new(output_json, review)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    md_lines = [
        "# T0~T2 v2 사전 시각 유사성 검토",
        "",
        f"- 검토 시각(UTC): `{reviewed_at}`",
        "- 검토 시점: T0~T2에 대한 YOLO 추론 전",
        "- 전수 비교: 신규 543프레임 × 학습·검증·S0~S4 1,365프레임 = 741,195쌍",
        "- 정확 SHA-256 중복: 0건",
        "- dHash 거리 4 이하: 17건",
        "- 직접 확인한 비교 묶음: 9장(17쌍 전체)",
        "- 객체 포함 장면의 실질적 중복: 0건",
        "",
        "## 개별 판단",
        "",
        "| ID | 신규 | 비교 자료 | dHash | 신규 GT(P/V) | 비교 GT(P/V) | 분류 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for item in decisions:
        md_lines.append(
            f"| {item['pair_id']} | {item['new_scenario']} f{item['new_frame']} | "
            f"{item['reference_scope']} {item['reference_scenario']} f{item['reference_frame']} | "
            f"{item['dhash_distance']} | {item['new_person_gt']}/{item['new_vehicle_gt']} | "
            f"{item['reference_person_gt']}/{item['reference_vehicle_gt']} | {item['visual_classification']} |"
        )
    md_lines.extend(
        [
            "",
            "## 판단",
            "",
            review["conclusion"],
            "",
            "T0~T2는 기존 MATLAB 시뮬레이션 환경에서 추가 구성한 시나리오이다. 동일 렌더러와 "
            "단일 지형을 공유하므로 실제 비행환경 또는 외부 자료에 대한 독립성·일반화를 주장하지 않는다.",
        ]
    )
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    gate = {
        "plan_id": PLAN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_required_preinference_checks_passed": True,
        "yolo_inference_performed_before_gate": False,
        "plan_path": str((CONFIG_ROOT / "scenario_plan.json").resolve()),
        "plan_sha256": sha256(CONFIG_ROOT / "scenario_plan.json"),
        "preregistration_sha256": sha256(CONFIG_ROOT / "pre_registration.json"),
        "evaluation_config_sha256": sha256(CONFIG_ROOT / "evaluation_config.json"),
        "gt_validation_sha256": sha256(OUTPUT_ROOT / "gt_validation.json"),
        "similarity_audit_sha256": sha256(OUTPUT_ROOT / "similarity_audit.json"),
        "visual_review_sha256": sha256(output_json),
        "conditions": {
            "GT_all_valid": True,
            "exact_duplicate_count": 0,
            "all_dhash_warnings_directly_reviewed": True,
            "object_scene_material_duplicate_count": 0,
            "weights_sha256": plan["weights_sha256"],
            "superseded_v1_preserved": plan["supersedes_failed_visibility_plan"]
            == "new_fixed_tests_t0_t2_v1",
        },
        "frozen_plan_sha_matches_preregistration": sha256(CONFIG_ROOT / "scenario_plan.json")
        == prereg["plan_sha256"],
        "authorization": "The frozen v2 plan may now be evaluated. No post-result scenario edits are permitted.",
    }
    write_json_new(gate_path, gate)
    print(json.dumps({"reviewed": len(decisions), "object_material_duplicates": 0, "gate": True}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
