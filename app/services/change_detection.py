"""
Backend orchestration for temporal change detection (AeroTwin AI Module 20).
Compares two already-completed reconstruction jobs' heightfields.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np

from depthwizard.change_detection.change_map import ChangeClass, compute_change_map
from depthwizard.utils.logging_setup import get_logger

logger = get_logger("depthwizard.backend.change_detection")

_CHANGE_COLORS = {  # BGR
    ChangeClass.UNKNOWN: (128, 128, 128),
    ChangeClass.NO_CHANGE: (180, 180, 180),
    ChangeClass.NEW_STRUCTURE: (60, 180, 60),
    ChangeClass.REMOVED_STRUCTURE: (60, 60, 200),
}


@dataclasses.dataclass
class ChangeDetectionSummary:
    registration_trusted: bool
    registration_rmse: float
    reasons: list[str]
    class_fractions: dict[str, float]
    change_map_path: str
    visualization_path: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _load_heightfield(result_dir: Path) -> tuple[np.ndarray, tuple[float, float], float]:
    reconstruction_dir = result_dir / "reconstruction"
    elevation = np.load(reconstruction_dir / "heightfield_elevation.npy")
    meta = json.loads((reconstruction_dir / "heightfield_meta.json").read_text())
    return elevation, tuple(meta["origin_xy"]), meta["cell_size"]


def run_change_detection(
    result_dir_a: str | Path,
    result_dir_b: str | Path,
    output_dir: str | Path,
    change_threshold_fraction_of_range: float = 0.15,
) -> ChangeDetectionSummary:
    """`change_threshold_fraction_of_range`: the minimum elevation
    difference to count as real change, expressed as a fraction of flight
    A's own elevation range -- since reconstructions are in arbitrary
    relative scale, an absolute threshold (e.g. "2.0 units") means nothing
    across different jobs; a scene-relative fraction is at least
    internally meaningful. Once Module 19 georeferencing exists, an
    absolute metric threshold becomes possible and preferable.
    """
    result_dir_a = Path(result_dir_a)
    result_dir_b = Path(result_dir_b)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elevation_a, origin_a, cell_size_a = _load_heightfield(result_dir_a)
    elevation_b, origin_b, cell_size_b = _load_heightfield(result_dir_b)

    valid_a = elevation_a[np.isfinite(elevation_a)]
    elev_range = float(valid_a.max() - valid_a.min()) if valid_a.size > 1 else 1.0
    change_threshold = max(elev_range * change_threshold_fraction_of_range, 1e-6)

    result = compute_change_map(
        elevation_a, origin_a, cell_size_a, elevation_b, origin_b, cell_size_b,
        change_threshold=change_threshold,
    )

    change_map_path = output_dir / "change_map.npy"
    np.save(change_map_path, result.change_labels)
    np.save(output_dir / "elevation_difference.npy", result.elevation_difference)

    vis_path = output_dir / "change_visualization.png"
    _save_visualization(result.change_labels, vis_path)

    (output_dir / "change_summary.json").write_text(json.dumps({
        "registration_trusted": result.registration_trusted,
        "registration_final_rmse": result.registration.final_rmse,
        "registration_converged": result.registration.converged,
        "change_threshold_used": change_threshold,
        "reasons": result.reasons,
        "class_fractions": result.class_fractions,
    }, indent=2))

    logger.info(
        "Change detection: registration_trusted=%s, rmse=%.4f, fractions=%s",
        result.registration_trusted, result.registration.final_rmse, result.class_fractions,
    )

    return ChangeDetectionSummary(
        registration_trusted=result.registration_trusted, registration_rmse=result.registration.final_rmse,
        reasons=result.reasons, class_fractions=result.class_fractions,
        change_map_path=str(change_map_path), visualization_path=str(vis_path),
    )


def _save_visualization(change_labels: np.ndarray, path: Path) -> None:
    vis = np.zeros((*change_labels.shape, 3), dtype=np.uint8)
    for cls, color in _CHANGE_COLORS.items():
        vis[change_labels == cls] = color
    cv2.imwrite(str(path), vis)
