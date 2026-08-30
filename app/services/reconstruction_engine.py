"""
Backend orchestration for dense 3D reconstruction (AeroTwin AI Module 14).

Pipeline: Phase 4's fused, confidence-scored per-frame depth -> back-project
every frame into world space -> merge into one point cloud -> rasterize to
a 2.5D heightfield -> reuse the existing (unmodified) terrain mesher ->
export both the existing JSON mesh format (current viewer compatibility)
and a real GLB (Module 46 web-viewer requirement).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from depthwizard.reconstruction.colmap_pipeline import SfMResult
from depthwizard.reconstruction.gltf_export import export_glb
from depthwizard.reconstruction.heightfield import fill_small_gaps, rasterize_to_heightfield
from depthwizard.reconstruction.mesh_builder import build_terrain_mesh, mesh_to_json
from depthwizard.reconstruction.point_cloud import (
    PointCloudResult,
    backproject_frame,
    merge_point_clouds,
    write_ply,
)
from depthwizard.utils.logging_setup import get_logger

logger = get_logger("depthwizard.backend.reconstruction_engine")


@dataclasses.dataclass
class ReconstructionResult:
    status: str  # "SUCCESS" | "FAILED"
    error: Optional[str]
    point_count: int
    mesh_vertex_count: int
    mesh_face_count: int
    heightfield_coverage_fraction: float
    ply_path: Optional[str]
    glb_path: Optional[str]
    mesh_json_path: Optional[str]


def run_dense_reconstruction(
    sfm_result: SfMResult,
    fusion_dir: str | Path,
    keyframe_dir: str | Path,
    result_dir: str | Path,
    min_confidence: float = 0.5,
    backproject_stride: int = 4,
    voxel_size: Optional[float] = None,
    max_mesh_resolution: int = 200,
) -> ReconstructionResult:
    fusion_dir = Path(fusion_dir)
    keyframe_dir = Path(keyframe_dir)
    result_dir = Path(result_dir)
    output_dir = result_dir / "reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics_by_camera = {c.camera_id: c for c in sfm_result.cameras}
    clouds: list[PointCloudResult] = []

    for pose in sfm_result.poses:
        confidence_path = fusion_dir / f"confidence_{pose.frame_id:06d}.npy"
        # aligned_ai_depth wasn't persisted directly in Phase 4 (only
        # confidence/difference were) -- reconstruct it from height-proxy +
        # calibration would require re-loading the fit; instead Phase 4's
        # depth_difference + confidence are enough IF we also persist
        # aligned depth. We read it back from disk here; if absent for a
        # frame (fusion skipped it), that frame simply contributes no points.
        aligned_path = fusion_dir / f"aligned_depth_{pose.frame_id:06d}.npy"
        if not confidence_path.exists() or not aligned_path.exists():
            continue

        intrinsics = intrinsics_by_camera.get(pose.camera_id)
        if intrinsics is None:
            continue

        rgb_path = keyframe_dir / pose.image_name
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        confidence = np.load(confidence_path)
        aligned_depth = np.load(aligned_path)

        cloud = backproject_frame(
            pose, intrinsics, aligned_depth, confidence, rgb,
            min_confidence=min_confidence, stride=backproject_stride,
        )
        clouds.append(cloud)

    if not clouds or sum(c.points.shape[0] for c in clouds) == 0:
        return ReconstructionResult(
            status="FAILED",
            error="No points survived back-projection -- either no frames had both confidence "
                  "and aligned-depth artifacts, or every frame's confidence was below "
                  "min_confidence. Falls back to point-cloud/geometry-only mode being "
                  "unavailable too; check that Phase 4 fusion actually ran and persisted "
                  "aligned depth for this job.",
            point_count=0, mesh_vertex_count=0, mesh_face_count=0, heightfield_coverage_fraction=0.0,
            ply_path=None, glb_path=None, mesh_json_path=None,
        )

    merged = merge_point_clouds(clouds, voxel_size=voxel_size)
    ply_path = output_dir / "dense_point_cloud.ply"
    write_ply(merged, ply_path)

    try:
        xy_range = float(np.ptp(merged.points[:, :2], axis=0).max())
        heightfield = rasterize_to_heightfield(merged, cell_size=xy_range / 200.0)
    except ValueError as exc:
        return ReconstructionResult(
            status="FAILED", error=str(exc), point_count=int(merged.points.shape[0]),
            mesh_vertex_count=0, mesh_face_count=0, heightfield_coverage_fraction=0.0,
            ply_path=str(ply_path), glb_path=None, mesh_json_path=None,
        )

    filled_elevation = fill_small_gaps(heightfield)
    mesh = build_terrain_mesh(
        filled_elevation, max_resolution=max_mesh_resolution,
        vertical_exaggeration=1.0, pixel_size_xy=heightfield.cell_size,
    )

    rows, cols = mesh.resolution
    colors_resized = cv2.resize(heightfield.rgb, (cols, rows), interpolation=cv2.INTER_NEAREST)
    vertex_colors = colors_resized.reshape(-1, 3)

    mesh_json_path = output_dir / "mesh.json"
    mesh_json_path.write_text(json.dumps(mesh_to_json(mesh)))

    glb_path = output_dir / "model.glb"
    export_glb(mesh, glb_path, vertex_colors=vertex_colors)

    # Persist the full-resolution heightfield too (not just the downsampled
    # mesh) -- Phase 6 (semantics) classifies at heightfield resolution,
    # and re-rasterizing from the point cloud a second time would be
    # wasted, non-deterministic-order work for no benefit.
    np.save(output_dir / "heightfield_elevation.npy", heightfield.elevation)
    np.save(output_dir / "heightfield_confidence.npy", heightfield.confidence)
    np.save(output_dir / "heightfield_source_frame_ids.npy", heightfield.source_frame_ids)
    cv2.imwrite(str(output_dir / "heightfield_rgb.png"), cv2.cvtColor(heightfield.rgb, cv2.COLOR_RGB2BGR))
    (output_dir / "heightfield_meta.json").write_text(json.dumps({
        "origin_xy": heightfield.origin_xy, "cell_size": heightfield.cell_size,
        "coverage_fraction": heightfield.coverage_fraction, "shape": list(heightfield.elevation.shape),
    }))

    logger.info(
        "Dense reconstruction: %d points -> %d vertices / %d faces, heightfield coverage %.1f%%",
        merged.points.shape[0], mesh.vertices.shape[0], mesh.faces.shape[0], heightfield.coverage_fraction * 100,
    )

    return ReconstructionResult(
        status="SUCCESS", error=None, point_count=int(merged.points.shape[0]),
        mesh_vertex_count=int(mesh.vertices.shape[0]), mesh_face_count=int(mesh.faces.shape[0]),
        heightfield_coverage_fraction=heightfield.coverage_fraction,
        ply_path=str(ply_path), glb_path=str(glb_path), mesh_json_path=str(mesh_json_path),
    )
