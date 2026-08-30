"""
Backend orchestration for standalone video quality analysis (AeroTwin AI
Module 2). Thin wrapper over `depthwizard.video.quality` -- see
`video_preprocessing.py` for why the split exists.

This is deliberately separate from `keyframe_selection.py`: keyframe
selection consumes per-frame quality scores as one term in the
InformationScore, but a standalone quality report (e.g. "is this video
even worth uploading?") shouldn't require running full keyframe selection.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

from depthwizard.utils.config import load_config
from depthwizard.video.ingestion import iter_candidate_frames
from depthwizard.video.quality import FrameQualityScore, score_frame


@dataclasses.dataclass
class VideoQualityReport:
    video_id: str
    frames_analyzed: int
    mean_quality: float
    min_quality: float
    max_quality: float
    low_quality_fraction: float  # fraction of frames below min_acceptable_quality
    frame_scores: list[FrameQualityScore]

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def analyze_video_quality(video_path: str | Path, video_id: str, sample_limit: Optional[int] = 500) -> VideoQualityReport:
    """Score up to `sample_limit` candidate frames and summarize overall
    video quality. `sample_limit` bounds cost for a quick pre-upload
    quality check; pass None to score every candidate frame (as keyframe
    selection does internally)."""
    cfg = load_config().video
    weights = {
        "blur": cfg.blur_weight, "exposure": cfg.exposure_weight,
        "contrast": cfg.contrast_weight, "feature": cfg.feature_weight,
        "motion": cfg.motion_weight,
    }

    scores: list[FrameQualityScore] = []
    prev_gray = None
    for frame_id, timestamp, frame in iter_candidate_frames(video_path, analysis_fps=cfg.analysis_fps):
        score, gray, _feature_count = score_frame(frame, frame_id, timestamp, prev_gray=prev_gray, weights=weights)
        scores.append(score)
        prev_gray = gray
        if sample_limit is not None and len(scores) >= sample_limit:
            break

    overall_values = [s.overall for s in scores] or [0.0]
    low_quality_count = sum(1 for v in overall_values if v < cfg.min_acceptable_quality)

    return VideoQualityReport(
        video_id=video_id,
        frames_analyzed=len(scores),
        mean_quality=round(sum(overall_values) / len(overall_values), 4),
        min_quality=round(min(overall_values), 4),
        max_quality=round(max(overall_values), 4),
        low_quality_fraction=round(low_quality_count / len(overall_values), 4),
        frame_scores=scores,
    )
