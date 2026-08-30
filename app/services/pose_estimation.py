"""
Backend orchestration for SfM / camera pose estimation (AeroTwin AI Module 4).

Thin wrapper over `depthwizard.reconstruction.colmap_pipeline` -- see
`video_preprocessing.py` for why the split exists.
"""
from __future__ import annotations

from pathlib import Path

from depthwizard.reconstruction.colmap_pipeline import SfMResult, run_sfm
from depthwizard.utils.config import load_config


def estimate_camera_poses(keyframe_dir: str | Path, result_dir: str | Path) -> SfMResult:
    """Run COLMAP feature extraction -> matching -> mapping over a
    directory of selected keyframes, writing the COLMAP workspace under
    `result_dir/colmap_workspace/` and returning the parsed result."""
    cfg = load_config().colmap
    workspace_dir = Path(result_dir) / "colmap_workspace"
    return run_sfm(
        image_dir=keyframe_dir,
        workspace_dir=workspace_dir,
        executable=cfg.executable,
        camera_model=cfg.camera_model,
        single_camera=cfg.single_camera,
        matcher=cfg.matcher,
        use_gpu=cfg.use_gpu,
        timeout_seconds=cfg.timeout_seconds,
    )
