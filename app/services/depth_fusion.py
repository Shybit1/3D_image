"""
Backend orchestration for depth-geometry fusion (AeroTwin AI Module 8).

Ties together Phase 2 (SfM poses/points), Phase 3 (AI depth), and the new
`depthwizard.reconstruction.{sparse_depth,depth_fusion}` core to fuse
every registered keyframe, persisting per-frame confidence/difference maps
and a job-level manifest.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from depthwizard.reconstruction.colmap_pipeline import SfMResult
from depthwizard.reconstruction.depth_fusion import FrameFusionResult, FusionConfig, classify_confidence, fuse_frame
from depthwizard.reconstruction.sparse_depth import extract_sparse_frame_depth
from depthwizard.reconstruction.uncertainty import ReliabilityReport, build_reliability_report
from depthwizard.utils.config import load_config
from depthwizard.utils.logging_setup import get_logger
from depthwizard.video.depth_pass import DepthBatchResult, load_depth_array

logger = get_logger("depthwizard.backend.depth_fusion")


@dataclasses.dataclass
class FusionSummary:
    video_id: str
    frames_fused: int
    frames_skipped: int
    skip_reasons: list[str]
    mean_confidence: Optional[float]
    mean_geometric_coverage_fraction: float
    fusion_dir: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def run_depth_fusion(
    sfm_result: SfMResult,
    depth_batch: DepthBatchResult,
    result_dir: str | Path,
) -> tuple[FusionSummary, ReliabilityReport]:
    """Fuse every frame that has BOTH a registered camera pose (Phase 2)
    AND an AI depth map (Phase 3). Frames missing either are skipped with
    an explicit reason -- never silently ignored (Module 5/29 pattern).

    Returns both the fusion summary (Module 8) and the reliability report
    (Module 9) -- built from the same in-memory `FrameFusionResult`s
    rather than re-deriving stats from disk twice.
    """
    cfg = load_config().video  # reuse height_proxy shape assumptions from video config where relevant
    fusion_dir = Path(result_dir) / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    depth_by_frame = {r.frame_id: r for r in depth_batch.records}
    confidences = []
    coverage_fractions = []
    skipped = 0
    skip_reasons = []
    fused_count = 0
    fusion_results: list[FrameFusionResult] = []

    for pose in sfm_result.poses:
        depth_record = depth_by_frame.get(pose.frame_id)
        if depth_record is None:
            skipped += 1
            reason = f"frame {pose.frame_id}: no AI depth available (not in depth batch)"
            skip_reasons.append(reason)
            fusion_results.append(FrameFusionResult(
                frame_id=pose.frame_id, image_name=pose.image_name, status="INSUFFICIENT_GEOMETRIC_COVERAGE",
                reason=reason, calibration=None, aligned_ai_depth=None, geometric_depth=None,
                depth_difference=None, confidence=None, geometric_coverage_fraction=0.0,
            ))
            continue

        height_proxy_png = cv2.imread(depth_record.height_proxy_png_path, cv2.IMREAD_UNCHANGED)
        if height_proxy_png is None:
            skipped += 1
            reason = f"frame {pose.frame_id}: could not read height-proxy PNG"
            skip_reasons.append(reason)
            fusion_results.append(FrameFusionResult(
                frame_id=pose.frame_id, image_name=pose.image_name, status="INSUFFICIENT_GEOMETRIC_COVERAGE",
                reason=reason, calibration=None, aligned_ai_depth=None, geometric_depth=None,
                depth_difference=None, confidence=None, geometric_coverage_fraction=0.0,
            ))
            continue
        height_proxy = height_proxy_png.astype(np.float32) / 65535.0

        sparse = extract_sparse_frame_depth(pose, sfm_result.points3d_xyz)
        result = fuse_frame(height_proxy, sparse, config=FusionConfig())
        fusion_results.append(result)

        if result.status != "FUSED":
            skipped += 1
            skip_reasons.append(f"frame {pose.frame_id}: {result.reason}")
            continue

        fused_count += 1
        confidences.append(float(np.nanmean(result.confidence)))
        coverage_fractions.append(result.geometric_coverage_fraction)

        np.save(fusion_dir / f"confidence_{pose.frame_id:06d}.npy", result.confidence)
        np.save(fusion_dir / f"depth_difference_{pose.frame_id:06d}.npy", result.depth_difference)
        np.save(fusion_dir / f"aligned_depth_{pose.frame_id:06d}.npy", result.aligned_ai_depth)
        _save_confidence_visual(result.confidence, fusion_dir / f"confidence_{pose.frame_id:06d}.png")
        classes = classify_confidence(result.confidence)
        np.save(fusion_dir / f"confidence_class_{pose.frame_id:06d}.npy", classes)

    summary = FusionSummary(
        video_id=depth_batch.video_id, frames_fused=fused_count, frames_skipped=skipped,
        skip_reasons=skip_reasons, mean_confidence=float(np.mean(confidences)) if confidences else None,
        mean_geometric_coverage_fraction=float(np.mean(coverage_fractions)) if coverage_fractions else 0.0,
        fusion_dir=str(fusion_dir),
    )
    summary.write_json(fusion_dir / "fusion_summary.json")

    reliability_report = build_reliability_report(fusion_results)
    Path(fusion_dir / "reliability_report.json").write_text(json.dumps(reliability_report.to_dict(), indent=2))

    logger.info(
        "Depth fusion: %d fused, %d skipped, mean_confidence=%s",
        fused_count, skipped, f"{summary.mean_confidence:.3f}" if summary.mean_confidence is not None else "N/A",
    )
    return summary, reliability_report


def _save_confidence_visual(confidence: np.ndarray, path: Path) -> None:
    """Red-yellow-green heatmap PNG for quick visual inspection; NaN
    (no geometric coverage) rendered as neutral gray, not a color that
    could be misread as a low-but-real confidence value."""
    valid = np.isfinite(confidence)
    vis = np.zeros((*confidence.shape, 3), dtype=np.uint8)
    vis[:] = (128, 128, 128)  # gray = no coverage
    if valid.any():
        clipped = np.clip(confidence[valid], 0.0, 1.0)
        # green channel rises, red channel falls, with confidence
        red = ((1.0 - clipped) * 255).astype(np.uint8)
        green = (clipped * 255).astype(np.uint8)
        vis[valid] = np.stack([np.zeros_like(red), green, red], axis=-1)  # BGR for cv2.imwrite
    cv2.imwrite(str(path), vis)
