"""
Backend orchestration for the AI depth engine (AeroTwin AI Module 10).

Thin wrapper over `depthwizard.video.depth_pass` -- see
`video_preprocessing.py` for why the split exists.

Shares a single lazily-constructed `DepthEstimator` at module scope,
matching the existing single-image `pipeline.py`'s `get_estimator()`
pattern -- model load is a real ~6s cost (weights + device placement)
that must happen once per process, not once per job or once per frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from depthwizard.depth.inference import DepthEstimator
from depthwizard.utils.config import load_config
from depthwizard.video.depth_pass import DepthBatchResult, run_depth_for_keyframes

_estimator: Optional[DepthEstimator] = None


def get_video_depth_estimator() -> DepthEstimator:
    """Same MiDaS_small model, same checkpoint/device config as the
    single-image pipeline -- deliberately not a second model instance, so
    the two pipelines can eventually share one loaded model in-process."""
    global _estimator
    if _estimator is None:
        cfg = load_config().model
        _estimator = DepthEstimator(
            checkpoint_path=cfg.checkpoint_path,
            device_preference=cfg.device,
            input_size=cfg.input_size,
        )
    return _estimator


def run_ai_depth(keyframe_dir: str | Path, result_dir: str | Path) -> DepthBatchResult:
    cfg = load_config().inference
    estimator = get_video_depth_estimator()
    depth_dir = Path(result_dir) / "depth"
    return run_depth_for_keyframes(
        keyframe_dir, depth_dir, estimator,
        max_full_image_side=cfg.max_full_image_side, tile_size=cfg.tile_size, overlap=cfg.overlap,
    )
