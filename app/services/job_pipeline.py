"""
Full pipeline orchestrator (AeroTwin AI Module 26/29).

Runs Phases 1-7 for one video job (change detection and spatial AI are
separate, explicit follow-on calls per Module 40's API surface, not part
of the linear per-video pipeline). Updates a `VideoJobRecord` at every
stage so `GET /api/reconstruction/{job_id}/progress` reflects real
progress, not a guess.

Fallback behavior (Module 29): if COLMAP/SfM fails or quality is POOR,
the pipeline does NOT silently continue into dense reconstruction --
it stops, records what it actually achieved, and reports an honest
`final_mode` rather than pretending full reconstruction happened.
"""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

from app.services.depth_engine import run_ai_depth
from app.services.depth_fusion import run_depth_fusion
from app.services.digital_twin import build_digital_twin
from app.services.keyframe_selection import run_keyframe_selection
from app.services.pose_estimation import estimate_camera_poses
from app.services.reconstruction_engine import run_dense_reconstruction
from app.services.reconstruction_quality import check_reconstruction_quality
from app.services.semantic_engine import run_semantic_segmentation
from app.services.video_job_store import SqliteVideoJobStore, VideoJobRecord
from depthwizard.utils.logging_setup import get_logger

logger = get_logger("depthwizard.backend.job_pipeline")


def run_pipeline_job(job_id: str, video_path: str, result_dir: str, store: SqliteVideoJobStore) -> None:
    """Synchronous pipeline body -- callers (the API layer) launch this on
    a background thread so the HTTP request that starts a job returns
    immediately (Module 26: no long job inside a request handler)."""
    job = store[job_id]
    result_dir_path = Path(result_dir)

    try:
        job.status = "running"

        job.advance_stage("VIDEO_ANALYSIS")
        # (Video probing already happened at upload time -- see routes_video.py)

        job.advance_stage("KEYFRAME_SELECTION")
        kf_result = run_keyframe_selection(video_path, job.video_id, result_dir)
        job.metrics["keyframes_selected"] = kf_result.keyframes_selected
        job.mark_stage_complete("KEYFRAME_SELECTION")

        job.advance_stage("POSE_ESTIMATION")
        sfm_result = estimate_camera_poses(result_dir_path / "keyframes", result_dir)
        job.metrics["sfm_status"] = sfm_result.status
        job.metrics["registered_images"] = sfm_result.registered_images
        job.metrics["total_images"] = sfm_result.total_images
        _write_camera_trajectory(sfm_result, result_dir_path)
        job.mark_stage_complete("POSE_ESTIMATION")

        job.advance_stage("SPARSE_RECONSTRUCTION")
        quality = check_reconstruction_quality(sfm_result)
        job.metrics["reconstruction_quality"] = quality.quality
        job.mark_stage_complete("SPARSE_RECONSTRUCTION")

        if sfm_result.status == "FAILED" or quality.quality == "FAILED":
            job.status = "completed_with_fallback"
            job.final_mode = "FALLBACK_MODE"
            job.error = f"SfM failed: {sfm_result.error or quality.reasons}"
            logger.warning("Job %s stopped after SfM failure: %s", job_id, job.error)
            return

        job.advance_stage("AI_DEPTH")
        depth_batch = run_ai_depth(result_dir_path / "keyframes", result_dir)
        job.metrics["depth_frames"] = depth_batch.total_frames
        job.mark_stage_complete("AI_DEPTH")

        if not quality.proceed_to_dense:
            job.status = "completed_with_fallback"
            job.final_mode = "GEOMETRIC_MODE"
            job.error = f"Reconstruction quality insufficient for dense reconstruction: {quality.reasons}"
            logger.warning("Job %s stopped before fusion: %s", job_id, job.error)
            return

        job.advance_stage("DEPTH_FUSION")
        fusion_summary, reliability = run_depth_fusion(sfm_result, depth_batch, result_dir)
        job.metrics["frames_fused"] = fusion_summary.frames_fused
        job.metrics["mean_confidence"] = fusion_summary.mean_confidence
        job.mark_stage_complete("DEPTH_FUSION")

        job.advance_stage("UNCERTAINTY")
        job.metrics["overall_high_confidence_fraction"] = reliability.overall_high_fraction
        job.mark_stage_complete("UNCERTAINTY")

        if fusion_summary.frames_fused == 0:
            job.status = "completed_with_fallback"
            job.final_mode = "AI_DEPTH_MODE"
            job.error = "No frames had sufficient geometric coverage to fuse -- AI depth exists but is unverified."
            logger.warning("Job %s stopped after fusion produced nothing: %s", job_id, job.error)
            return

        job.advance_stage("DENSE_RECONSTRUCTION")
        recon = run_dense_reconstruction(
            sfm_result, result_dir_path / "fusion", result_dir_path / "keyframes", result_dir,
            min_confidence=0.4, backproject_stride=2,
        )
        job.metrics["reconstruction_status"] = recon.status
        job.metrics["point_count"] = recon.point_count
        job.metrics["mesh_vertex_count"] = recon.mesh_vertex_count
        job.mark_stage_complete("DENSE_RECONSTRUCTION")

        if recon.status != "SUCCESS":
            job.status = "completed_with_fallback"
            job.final_mode = "AI_DEPTH_MODE"
            job.error = recon.error
            logger.warning("Job %s dense reconstruction failed: %s", job_id, job.error)
            return

        job.advance_stage("SEMANTIC_ANALYSIS")
        try:
            semantic_summary = run_semantic_segmentation(result_dir)
            job.metrics["semantic_class_fractions"] = semantic_summary.class_fractions
        except Exception as exc:
            # Semantic failure must not take down an otherwise-successful
            # reconstruction (Module 29: "Semantic model failure -> geometry-only mode").
            logger.warning("Job %s semantic segmentation failed, continuing without it: %s", job_id, exc)
            job.metrics["semantic_error"] = str(exc)
        job.mark_stage_complete("SEMANTIC_ANALYSIS")

        job.advance_stage("DIGITAL_TWIN")
        try:
            dt_summary, _objects = build_digital_twin(result_dir)
            job.metrics["object_count"] = dt_summary.object_count
            job.metrics["counts_by_class"] = dt_summary.counts_by_class
        except Exception as exc:
            logger.warning("Job %s digital twin extraction failed, continuing without it: %s", job_id, exc)
            job.metrics["digital_twin_error"] = str(exc)
        job.mark_stage_complete("DIGITAL_TWIN")

        job.advance_stage("FINALIZATION")
        job.status = "completed"
        job.final_mode = "FULL_RECONSTRUCTION"
        job.mark_stage_complete("FINALIZATION")
        logger.info("Job %s completed: %s", job_id, job.metrics)

    except Exception as exc:
        job.status = "failed"
        job.error = f"{exc}\n{traceback.format_exc()}"
        logger.error("Job %s crashed: %s", job_id, exc)


def _write_camera_trajectory(sfm_result, result_dir_path: Path) -> None:
    """Persist a lightweight camera-trajectory JSON for the
    /camera-trajectory endpoint, independent of the full COLMAP workspace
    (which callers shouldn't need to parse directly)."""
    import json
    trajectory = {
        "poses": [
            {
                "frame_id": p.frame_id, "image_name": p.image_name,
                "qw": p.qw, "qx": p.qx, "qy": p.qy, "qz": p.qz,
                "tx": p.tx, "ty": p.ty, "tz": p.tz,
            }
            for p in sorted(sfm_result.poses, key=lambda p: p.frame_id)
        ],
        "registered_images": sfm_result.registered_images, "total_images": sfm_result.total_images,
    }
    (result_dir_path / "camera_trajectory.json").write_text(json.dumps(trajectory, indent=2))


def launch_pipeline_job_async(job_id: str, video_path: str, result_dir: str, store: SqliteVideoJobStore) -> threading.Thread:
    """Module 26: lightweight background worker, acceptable for this
    prototype's scale -- a real deployment would swap this for a proper
    task queue without changing the pipeline body above."""
    thread = threading.Thread(
        target=run_pipeline_job, args=(job_id, video_path, result_dir, store), daemon=True,
    )
    thread.start()
    return thread
