"""
Backend orchestration for video ingestion (AeroTwin AI Module 1).

Thin wrapper over `depthwizard.video.ingestion` -- the actual algorithms
live in `src/depthwizard/` alongside the rest of the reconstruction
science, matching the existing repo's separation between core algorithms
(`src/depthwizard/*`) and API/job orchestration (`backend/app/services/*`).
This keeps the same architecture the depth/DSM/calibration modules already
use, rather than introducing a second convention for video-specific code.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from depthwizard.utils.config import load_config
from depthwizard.video.ingestion import VideoMetadata, VideoValidationError, probe_video, save_thumbnail, iter_candidate_frames

VIDEO_UPLOAD_DIR = Path("data/uploads/videos")
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def ingest_video(saved_path: str | Path, original_filename: str) -> VideoMetadata:
    """Probe an already-saved video file and return its metadata record.

    Raises VideoValidationError only for conditions the caller should
    treat as a hard 400 (e.g. size limit exceeded before we even try to
    open it); anything OpenCV itself rejects comes back as a
    status="INVALID" VideoMetadata rather than an exception, so the
    caller can decide how to surface it.
    """
    cfg = load_config()
    video_id = uuid.uuid4().hex[:12]
    metadata = probe_video(saved_path, video_id=video_id, max_size_mb=cfg.video.max_video_size_mb)

    if metadata.status == "VALID" and metadata.duration_seconds > cfg.video.max_duration_seconds:
        metadata.status = "INVALID"
        metadata.error = (
            f"Video duration {metadata.duration_seconds:.0f}s exceeds "
            f"max_duration_seconds={cfg.video.max_duration_seconds:.0f}s "
            f"(demo-mode limit -- see Module 31)."
        )

    metadata.filename = original_filename or metadata.filename
    return metadata


def generate_preview_thumbnail(video_path: str | Path, output_path: str | Path) -> bool:
    """Save a single thumbnail from the first decodable frame, for the
    upload-confirmation UI. Returns False (does not raise) if the video
    has no candidate frames -- this is cosmetic, not a hard failure."""
    cfg = load_config()
    try:
        for _frame_id, _ts, frame in iter_candidate_frames(video_path, analysis_fps=cfg.video.analysis_fps):
            save_thumbnail(frame, output_path, size=tuple(cfg.video.thumbnail_size))
            return True
    except VideoValidationError:
        return False
    return False
