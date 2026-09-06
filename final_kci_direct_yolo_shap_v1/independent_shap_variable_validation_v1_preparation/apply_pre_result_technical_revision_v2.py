from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY = Path(r"C:\Users\lab\Counterfactual-XAI-Verifier")
PREPARATION = REPOSITORY / ".k" / "final_kci_direct_yolo_shap_v1" / "independent_shap_variable_validation_v1_preparation"
GEOMETRY = REPOSITORY / "experiments" / "yolov8_search_comparison" / "geometry_reference_v1.mat"
FILES_TO_PRESERVE = (
    "fixed_settings.json",
    "independent_conditions.csv",
    "independent_conditions.json",
    "source_manifest.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    history = PREPARATION / "preparation_revision_history" / "attempt_001"
    history.mkdir(parents=True, exist_ok=False)
    for name in FILES_TO_PRESERVE:
        shutil.copy2(PREPARATION / name, history / name)

    settings_path = PREPARATION / "fixed_settings.json"
    settings = read_json(settings_path)
    geometry_entry = {"path": str(GEOMETRY.resolve()), "sha256": sha256(GEOMETRY)}
    settings["source_files"]["geometry_reference"] = geometry_entry
    settings["pre_result_technical_revision"] = {
        "revision": 2,
        "reason": "lock the geometry file actually loaded by the unchanged MATLAB exporter; no scene, environment, gate, model, or result changed",
        "previous_files_preserved_at": str(history.resolve()),
    }
    write_json(settings_path, settings)

    source_hashes = {name: item["sha256"] for name, item in settings["source_files"].items()}
    conditions_json_path = PREPARATION / "independent_conditions.json"
    conditions_value = read_json(conditions_json_path)
    conditions = conditions_value["records"]
    detector = {
        "input_size": 640, "confidence_threshold": 0.25,
        "nms_iou_threshold": 0.7, "ap_iou_threshold": 0.5,
        "batch_size": 16, "device": "cuda:0",
    }
    for row in conditions:
        environment = {key: float(row[key]) for key in ("fog_percent", "illumination_lux", "camera_noise")}
        row["setting_identity_sha256"] = value_hash({
            "condition_id": row["condition_id"],
            "scenario_config_sha256": row["scenario_config_sha256"],
            "environment": environment, "model_seed": 42,
            "weights_sha256": settings["detector"]["weights_sha256"],
            "evaluation_core_sha256": source_hashes["evaluation_core"],
            "renderer_sha256": source_hashes["renderer"],
            "exporter_sha256": source_hashes["exporter"],
            "geometry_reference_sha256": source_hashes["geometry_reference"],
            "detector_settings": detector,
        })
    write_json(conditions_json_path, conditions_value)
    write_csv(PREPARATION / "independent_conditions.csv", conditions, list(conditions[0]))

    revision_path = PREPARATION / "preparation_revision_002.json"
    revision = {
        "schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision": 2, "status": "PRE_RESULT_TECHNICAL_REVISION",
        "reason": "The first static validator treated a prior observed metadata envelope as a renderer hard bound. The unchanged exporter has no such object-coordinate bound and loads a -100..100 m terrain grid. The actual geometry input was also added to the source lock.",
        "scientific_changes": [],
        "unchanged": ["ten scene configurations", "P", "Q", "three severity-0.70 validation conditions", "110-condition set", "Shapley formula", "0.01 tie rule", "7/10 gate", "YOLO weights and evaluator"],
        "mechanical_changes": ["static validator uses actual geometry grid bounds", "geometry-reference hash added to settings, source manifest, and condition setting identities"],
        "original_files_preserved_at": str(history.resolve()),
        "matlab_render_yolo_shap_executed": False,
    }
    write_json(revision_path, revision)

    manifest_path = PREPARATION / "source_manifest.csv"
    manifest = read_csv(manifest_path)
    by_path = {str(Path(row["path"]).resolve()).lower(): row for row in manifest}
    additions = [
        ("geometry_reference", GEOMETRY, "fixed terrain and geometry source used by the exporter"),
        ("pre_result_revision_script", PREPARATION / "apply_pre_result_technical_revision_v2.py", "reproducible pre-result technical lock correction"),
        ("pre_result_revision_record", revision_path, "audit record for pre-result technical correction"),
    ]
    for kind, path, role in additions:
        key = str(path.resolve()).lower()
        if key not in by_path:
            row = {"source_type": kind, "path": str(path.resolve()), "sha256": "", "size_bytes": "", "role": role}
            manifest.append(row)
            by_path[key] = row
    for row in manifest:
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        row["sha256"] = sha256(path)
        row["size_bytes"] = path.stat().st_size
    write_csv(manifest_path, manifest, ["source_type", "path", "sha256", "size_bytes", "role"])

    print(json.dumps({
        "status": "PRE_RESULT_TECHNICAL_REVISION_V2_APPLIED",
        "preserved_files": len(FILES_TO_PRESERVE), "conditions": len(conditions),
        "geometry_sha256": geometry_entry["sha256"], "source_manifest_records": len(manifest),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
