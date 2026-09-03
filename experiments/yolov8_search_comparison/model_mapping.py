"""Resolve detector-native class names to the experiment's two classes."""

from __future__ import annotations

from typing import Mapping


PERSON_NAMES = {"person", "pedestrian", "people", "human"}
VEHICLE_NAMES = {
    "vehicle",
    "car",
    "motorcycle",
    "motorbike",
    "bus",
    "truck",
    "van",
}


def resolve_target_class_mapping(model_names: Mapping[int, str]) -> dict[int, str]:
    """Map COCO, custom 2-class, and compatible model names without ID assumptions.

    Unrelated classes are omitted. A model must expose at least one recognised
    person or vehicle name; otherwise it is incompatible with this experiment.
    """
    resolved: dict[int, str] = {}
    for raw_id, raw_name in model_names.items():
        class_id = int(raw_id)
        name = str(raw_name).strip().lower().replace("-", " ").replace("_", " ")
        if name in PERSON_NAMES:
            resolved[class_id] = "person"
        elif name in VEHICLE_NAMES:
            resolved[class_id] = "vehicle"
    if not resolved:
        raise ValueError(
            "Model has no recognised person/vehicle class names: "
            f"{dict(model_names)}"
        )
    return resolved

