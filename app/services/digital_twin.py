"""
Backend orchestration for the semantic 3D digital twin (AeroTwin AI
Module 17). Reads Phase 5 (heightfield) + Phase 6 (semantic labels,
height-above-ground, adjusted confidence) outputs, extracts discrete
objects, and persists a manifest.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np

from depthwizard.digital_twin.objects import DigitalTwinObject, extract_objects
from depthwizard.semantics.segmenter import SemanticClass
from depthwizard.utils.logging_setup import get_logger

logger = get_logger("depthwizard.backend.digital_twin")


@dataclasses.dataclass
class DigitalTwinSummary:
    object_count: int
    counts_by_class: dict[str, int]
    manifest_path: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def build_digital_twin(result_dir: str | Path, min_object_pixels: int = 8) -> tuple[DigitalTwinSummary, list[DigitalTwinObject]]:
    result_dir = Path(result_dir)
    reconstruction_dir = result_dir / "reconstruction"
    semantics_dir = result_dir / "semantics"

    elevation_path = reconstruction_dir / "heightfield_elevation.npy"
    confidence_path = reconstruction_dir / "heightfield_confidence.npy"
    source_frame_ids_path = reconstruction_dir / "heightfield_source_frame_ids.npy"
    meta_path = reconstruction_dir / "heightfield_meta.json"
    labels_path = semantics_dir / "labels.npy"
    height_above_ground_path = semantics_dir / "height_above_ground.npy"
    adjusted_confidence_path = semantics_dir / "confidence_adjusted.npy"

    missing = [p for p in [elevation_path, meta_path, labels_path, height_above_ground_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required inputs for digital twin construction: {[str(p) for p in missing]} -- "
            f"has Phase 5 (dense reconstruction) and Phase 6 (semantics) both run for this job?"
        )

    elevation = np.load(elevation_path)
    labels = np.load(labels_path)
    height_above_ground = np.load(height_above_ground_path)
    confidence = np.load(adjusted_confidence_path) if adjusted_confidence_path.exists() else np.load(confidence_path)
    source_frame_ids = np.load(source_frame_ids_path) if source_frame_ids_path.exists() else np.full(elevation.shape, -1, dtype=np.int32)
    meta = json.loads(meta_path.read_text())

    objects = extract_objects(
        labels=labels, elevation=elevation, height_above_ground=height_above_ground,
        confidence=confidence, source_frame_ids=source_frame_ids,
        origin_xy=tuple(meta["origin_xy"]), cell_size=meta["cell_size"],
        min_object_pixels=min_object_pixels,
    )

    output_dir = result_dir / "digital_twin"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "objects.json"
    manifest_path.write_text(json.dumps({
        "scale_disclaimer": (
            "All dimensions are in the reconstruction's own relative world units. "
            "No metric or georeferencing calibration (Module 19) has run for this job -- "
            "these are NOT meters, and should not be presented to a user as such."
        ),
        "objects": [obj.to_dict() for obj in objects],
    }, indent=2))

    counts_by_class: dict[str, int] = {}
    for obj in objects:
        counts_by_class[obj.semantic_class] = counts_by_class.get(obj.semantic_class, 0) + 1

    summary = DigitalTwinSummary(object_count=len(objects), counts_by_class=counts_by_class, manifest_path=str(manifest_path))
    logger.info("Digital twin: %d objects extracted (%s)", len(objects), counts_by_class)
    return summary, objects
