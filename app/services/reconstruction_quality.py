"""
Backend orchestration for reconstruction quality gating (AeroTwin AI
Module 5). Thin wrapper over `depthwizard.reconstruction.quality`.
"""
from __future__ import annotations

from depthwizard.reconstruction.colmap_pipeline import SfMResult
from depthwizard.reconstruction.quality import (
    QualityThresholds,
    ReconstructionQualityReport,
    evaluate_reconstruction_quality,
)
from depthwizard.utils.config import load_config


def check_reconstruction_quality(sfm_result: SfMResult) -> ReconstructionQualityReport:
    cfg = load_config().reconstruction_quality
    thresholds = QualityThresholds(
        good_registration_rate=cfg.good_registration_rate,
        fair_registration_rate=cfg.fair_registration_rate,
        good_reprojection_error=cfg.good_reprojection_error,
        fair_reprojection_error=cfg.fair_reprojection_error,
        min_points_for_dense=cfg.min_points_for_dense,
    )
    return evaluate_reconstruction_quality(sfm_result, thresholds=thresholds)
