"""
Backend orchestration for keyframe selection (AeroTwin AI Module 3).

Thin wrapper over `depthwizard.video.keyframes` -- see
`video_preprocessing.py` for why the split exists.
"""
from __future__ import annotations

from pathlib import Path

from depthwizard.utils.config import load_config
from depthwizard.video.keyframes import KeyframeSelectionConfig, KeyframeSelectionResult, select_keyframes


def run_keyframe_selection(video_path: str | Path, video_id: str, result_dir: str | Path) -> KeyframeSelectionResult:
    """Run Phase-1 candidate scoring + keyframe selection for one video,
    writing selected frames and a manifest under `result_dir/keyframes/`.
    """
    cfg = load_config().video
    kf_config = KeyframeSelectionConfig(
        analysis_fps=cfg.analysis_fps,
        max_duration_seconds=cfg.max_duration_seconds,
        blur_weight=cfg.blur_weight, exposure_weight=cfg.exposure_weight,
        contrast_weight=cfg.contrast_weight, feature_weight=cfg.feature_weight,
        motion_weight=cfg.motion_weight,
        novelty_weight=cfg.novelty_weight, feature_richness_weight=cfg.feature_richness_weight,
        quality_weight=cfg.quality_weight, coverage_gain_weight=cfg.coverage_gain_weight,
        redundancy_weight=cfg.redundancy_weight,
        information_score_threshold=cfg.information_score_threshold,
        min_keyframe_spacing_seconds=cfg.min_keyframe_spacing_seconds,
        max_keyframes=cfg.max_keyframes, min_keyframes_warning=cfg.min_keyframes_warning,
        min_acceptable_quality=cfg.min_acceptable_quality,
    )
    keyframe_dir = Path(result_dir) / "keyframes"
    return select_keyframes(video_path, video_id, keyframe_dir, config=kf_config)
