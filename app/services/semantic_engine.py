"""
Backend orchestration for semantic understanding (AeroTwin AI Module 16).

Runs on the heightfield persisted by Phase 5's `reconstruction_engine.py`
(world-space RGB + elevation), not per-keyframe images -- this classifies
the merged reconstruction once, rather than every source frame
redundantly, and produces labels already in the same grid the mesh/digital
twin (Phase 7) will consume.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np

from depthwizard.semantics.segmenter import (
    CLASS_NAMES,
    HeuristicSemanticSegmenter,
    SemanticClass,
    SemanticConfidenceWeights,
    apply_semantic_confidence_adjustment,
)
from depthwizard.utils.logging_setup import get_logger

logger = get_logger("depthwizard.backend.semantic_engine")

_CLASS_COLORS = {  # BGR, for the visualization PNG only
    SemanticClass.UNKNOWN: (128, 128, 128),
    SemanticClass.GROUND: (90, 140, 160),
    SemanticClass.VEGETATION: (60, 160, 60),
    SemanticClass.BUILDING: (60, 60, 200),
    SemanticClass.ROAD: (80, 80, 80),
    SemanticClass.WATER: (200, 120, 40),
}


@dataclasses.dataclass
class SemanticResultSummary:
    method: str
    class_fractions: dict[str, float]
    labels_path: str
    height_above_ground_path: str
    confidence_adjusted_path: str
    visualization_path: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def run_semantic_segmentation(
    result_dir: str | Path,
    confidence_weights: SemanticConfidenceWeights | None = None,
) -> SemanticResultSummary:
    result_dir = Path(result_dir)
    reconstruction_dir = result_dir / "reconstruction"
    elevation_path = reconstruction_dir / "heightfield_elevation.npy"
    rgb_path = reconstruction_dir / "heightfield_rgb.png"
    confidence_path = reconstruction_dir / "heightfield_confidence.npy"

    if not elevation_path.exists() or not rgb_path.exists():
        raise FileNotFoundError(
            f"No heightfield found under {reconstruction_dir} -- has Phase 5 dense "
            f"reconstruction run for this job yet?"
        )

    elevation = np.load(elevation_path)
    rgb_bgr = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    segmenter = HeuristicSemanticSegmenter()
    result = segmenter.segment(rgb, elevation)

    output_dir = result_dir / "semantics"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = output_dir / "labels.npy"
    np.save(labels_path, result.labels)

    height_above_ground_path = output_dir / "height_above_ground.npy"
    np.save(height_above_ground_path, result.height_above_ground)

    confidence_adjusted_path = output_dir / "confidence_adjusted.npy"
    if confidence_path.exists():
        base_confidence = np.load(confidence_path)
        adjusted = apply_semantic_confidence_adjustment(base_confidence, result.labels, weights=confidence_weights)
        np.save(confidence_adjusted_path, adjusted)
    else:
        logger.warning("No heightfield confidence found at %s -- skipping semantic confidence adjustment", confidence_path)
        confidence_adjusted_path = None

    vis_path = output_dir / "semantic_visualization.png"
    _save_visualization(result.labels, vis_path)

    (output_dir / "semantic_summary.json").write_text(json.dumps({
        "method": result.method, "class_fractions": result.class_fractions,
    }, indent=2))

    logger.info("Semantic segmentation (%s): %s", result.method, result.class_fractions)

    return SemanticResultSummary(
        method=result.method, class_fractions=result.class_fractions,
        labels_path=str(labels_path), height_above_ground_path=str(height_above_ground_path),
        confidence_adjusted_path=str(confidence_adjusted_path) if confidence_adjusted_path else "",
        visualization_path=str(vis_path),
    )


def _save_visualization(labels: np.ndarray, path: Path) -> None:
    vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for cls, color in _CLASS_COLORS.items():
        vis[labels == cls] = color
    cv2.imwrite(str(path), vis)
