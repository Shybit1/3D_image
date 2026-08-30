"""
Orchestrates the real DepthWizard-X pipeline: load -> depth -> DSM ->
(optional calibration) -> uncertainty -> mesh -> results package.

This is the single place that stitches together the modules built in
src/depthwizard/*. The API layer calls this; nothing here is UI-specific.
"""
from __future__ import annotations

import dataclasses
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from app.services.job_store import JobRecord, SqliteJobStore, SqlitePersistentDict
from depthwizard.analysis.terrain_layers import compute_slope_degrees
from depthwizard.calibration.scale_calibration import (
    InsufficientCalibrationDataError,
    fit_calibration_from_sparse_points,
    fit_linear_calibration,
    split_calibration_holdout,
)
from depthwizard.depth.inference import DepthEstimator
from depthwizard.dsm.engine import generate_absolute_dsm, generate_relative_dsm
from depthwizard.geospatial.raster_io import estimate_pixel_size_m, export_dsm_geotiff
from depthwizard.preprocessing.image_io import load_image
from depthwizard.reconstruction.mesh_builder import build_terrain_mesh, mesh_to_json
from depthwizard.uncertainty.estimator import estimate_uncertainty
from depthwizard.utils.config import load_config
from depthwizard.utils.logging_setup import get_logger

logger = get_logger("depthwizard.backend")

RESULTS_ROOT = Path("results")

# Model is expensive to load -- one process-wide singleton, loaded lazily
# on first request, not at import time (keeps API startup fast and keeps
# failures explicit/traceable to the first real request).
_estimator: Optional[DepthEstimator] = None


def get_estimator() -> DepthEstimator:
    global _estimator
    if _estimator is None:
        cfg = load_config()
        _estimator = DepthEstimator(
            checkpoint_path=cfg.model.checkpoint_path,
            device_preference=cfg.model.device,
            input_size=cfg.model.input_size,
        )
    return _estimator


@dataclasses.dataclass
class _LegacyJobRecordShapeDoc:
    """Kept only as inline documentation of JobRecord's shape -- the real
    implementation (with SQLite persistence) now lives in
    app.services.job_store.JobRecord, imported above."""
    job_id: str
    status: str = "queued"       # queued | running | completed | failed
    stage: Optional[str] = None
    error: Optional[str] = None
    result_dir: Optional[Path] = None
    metadata: Optional[dict] = None
    metrics: Optional[dict] = None


# SQLite-backed job store and upload registries -- see
# app/services/job_store.py for why this replaced a plain in-memory dict.
# Existing jobs/uploads are reloaded from results/depthwizard_state.sqlite3
# on process start, so a restart no longer silently erases job history.
JOBS = SqliteJobStore()
UPLOADS = SqlitePersistentDict(table="uploads", encode=str, decode=Path)
UPLOAD_ORIGINAL_NAMES = SqlitePersistentDict(table="upload_names")


def register_upload(saved_path: Path, original_filename: str = "") -> str:
    file_id = uuid.uuid4().hex[:12]
    UPLOADS[file_id] = saved_path
    UPLOAD_ORIGINAL_NAMES[file_id] = original_filename
    return file_id


def run_reconstruction_job(job_id: str, file_id: str, mode: str, compute_uncertainty: bool,
                            reference_dem_file_id: Optional[str] = None,
                            sun_elevation_deg: Optional[float] = None,
                            sun_azimuth_deg: Optional[float] = None,
                            gcp_points: Optional[list] = None):
    job = JOBS[job_id]
    try:
        job.status = "running"
        t_start = time.time()

        job.stage = "loading_and_validating"
        image_path = UPLOADS[file_id]
        img = load_image(image_path)

        job.stage = "depth_inference"
        t0 = time.time()
        estimator = get_estimator()
        cfg = load_config()
        depth_result = estimator.estimate_auto(
            img.rgb,
            max_full_image_side=cfg.inference.max_full_image_side,
            tile_size=cfg.inference.tile_size,
            overlap=cfg.inference.overlap,
        )
        t_depth = time.time() - t0

        job.stage = "dsm_generation"
        # Computed up front (rather than only at mesh-save time) because
        # the shadow-length calibration path below needs a real ground
        # pixel size, not an assumed 1.0, to convert shadow lengths from
        # pixels to meters.
        pixel_size_m, pixel_size_is_metric = estimate_pixel_size_m(img.crs, img.transform)

        calibration_info = {}
        shadow_cue_result = None
        # reference_grid and calib_split are set ONLY when calibration actually
        # succeeds and dsm.is_absolute becomes True. This matters: comparing a
        # *relative* (unitless) DSM against a reference grid in real meters
        # would produce a meaningless RMSE/MAE that looks like a real number --
        # so validation below is gated strictly on reference_grid being non-None,
        # which only happens on the success path.
        reference_grid = None
        calib_split = None  # CalibrationHoldoutSplit -- disjoint calib/holdout pixels

        if mode == "absolute" and reference_dem_file_id is not None:
            from depthwizard.geospatial.raster_io import resample_dem_to_grid

            if not img.is_georeferenced:
                calibration_info["warning"] = (
                    "Input image has no CRS/transform; a reference DEM cannot be "
                    "spatially aligned to it. Falling back to relative DSM."
                )
                dsm = generate_relative_dsm(depth_result.height_proxy, guide_rgb=img.rgb,
                                             vertical_range_units=100.0)
            else:
                try:
                    ref_path = UPLOADS[reference_dem_file_id]
                    candidate_reference_grid = resample_dem_to_grid(
                        ref_path, depth_result.height_proxy.shape, img.transform, img.crs
                    )
                    # Disjoint calibration/holdout split: the points used to FIT
                    # the calibration are never also used to REPORT accuracy.
                    # See calibration.scale_calibration.split_calibration_holdout
                    # for why this matters -- validating against calibration
                    # points overstates accuracy.
                    calib_split = split_calibration_holdout(
                        candidate_reference_grid, depth_result.height_proxy,
                        num_calib=2000, num_holdout=2000, seed=0,
                    )
                    calib = fit_linear_calibration(
                        calib_split.calib_height_proxy, calib_split.calib_reference,
                        min_points=cfg.calibration.min_reference_points,
                    )
                    dsm = generate_absolute_dsm(
                        depth_result.height_proxy, scale=calib.scale, offset=calib.offset,
                        guide_rgb=img.rgb,
                    )
                    # Only now, with a successful absolute calibration, is it
                    # scientifically valid to compare dsm.elevation against
                    # this reference grid -- so only now do we keep it.
                    reference_grid = candidate_reference_grid
                    calibration_info.update({
                        "calibration_source": "reference_dem",
                        "calibration_method": calib.method,
                        "calibration_points_used": calib.num_points_used,
                        "calibration_warning": calib.warning,
                        "calibration_rmse_m": round(calib.residual_rmse_m, 3),
                        "calibration_mae_m": round(calib.residual_mae_m, 3),
                        "calibration_r": round(calib.r_value, 3),
                    })
                except InsufficientCalibrationDataError as e:
                    calibration_info["warning"] = str(e)
                    calib_split = None
                    dsm = generate_relative_dsm(depth_result.height_proxy, guide_rgb=img.rgb,
                                                 vertical_range_units=100.0)
        elif mode == "absolute" and reference_dem_file_id is None:
            # No user-supplied reference DEM -- try calibration sources in
            # priority order, each only attempted if the previous one
            # didn't produce a usable calibration. Priority rationale:
            #   1. GCP points: direct, human-supplied ground-truth
            #      elevations at named pixel locations -- the most
            #      trustworthy source when available, since it isn't
            #      subject to DEM resampling error or shadow-detection
            #      heuristics.
            #   2. Offline SRTM cache: dense, real elevation data, but at
            #      SRTM's native ~30m resolution and with zero live
            #      network calls (offline-first).
            #   3. Shadow-length cue: needs only sun-angle metadata, works
            #      on any scene with visible cast shadows, but rests on a
            #      flat-ground assumption at each shadow tip.
            # If none succeed, falls back to relative DSM with a warning
            # naming exactly which sources were tried and why each failed.
            calibrated = False
            attempted_sources: list[str] = []
            srtm_result = None

            if gcp_points:
                attempted_sources.append("gcp_points")
                try:
                    h_img, w_img = depth_result.height_proxy.shape
                    rows = np.array([p.row for p in gcp_points])
                    cols = np.array([p.col for p in gcp_points])
                    elevs = np.array([p.elevation_m for p in gcp_points], dtype=np.float64)

                    in_bounds = (rows >= 0) & (rows < h_img) & (cols >= 0) & (cols < w_img)
                    if not in_bounds.all():
                        logger.warning(
                            f"{int((~in_bounds).sum())} of {len(gcp_points)} GCP point(s) "
                            f"fall outside the {h_img}x{w_img} image and will be ignored."
                        )
                    rows, cols, elevs = rows[in_bounds], cols[in_bounds], elevs[in_bounds]

                    gcp_result = fit_calibration_from_sparse_points(
                        depth_result.height_proxy, rows, cols, elevs,
                        min_points=cfg.calibration.min_reference_points,
                    )
                    calib = gcp_result.calibration

                    dsm = generate_absolute_dsm(
                        depth_result.height_proxy, scale=calib.scale, offset=calib.offset,
                        guide_rgb=img.rgb,
                    )
                    calibrated = True
                    calibration_info.update({
                        "calibration_source": "gcp_points",
                        "gcp_points_supplied": gcp_result.num_points_supplied,
                        "gcp_points_used": gcp_result.num_points_used,
                        "gcp_holdout_points": gcp_result.holdout_point_count,
                        "gcp_holdout_rmse_m": round(gcp_result.holdout_rmse_m, 3) if gcp_result.holdout_rmse_m is not None else None,
                        "calibration_method": calib.method,
                        "calibration_points_used": calib.num_points_used,
                        "calibration_warning": calib.warning,
                        "calibration_rmse_m": round(calib.residual_rmse_m, 3),
                        "calibration_mae_m": round(calib.residual_mae_m, 3),
                        "calibration_r": round(calib.r_value, 3),
                    })
                    # NOTE: like shadow calibration, GCP points are sparse --
                    # reference_grid stays None, so dense full-grid
                    # RMSE/terrain stratification cannot run for this job.
                    # gcp_holdout_rmse_m above is the honest independent
                    # accuracy check when enough points were supplied.
                except InsufficientCalibrationDataError as e:
                    calibration_info["warning"] = str(e)

            if not calibrated and img.is_georeferenced:
                attempted_sources.append("offline_srtm_cache")
                try:
                    from depthwizard.geospatial.srtm_cache import lookup_and_build_reference_grid

                    srtm_result = lookup_and_build_reference_grid(
                        img.crs, img.transform, depth_result.height_proxy.shape,
                    )
                except Exception as e:
                    logger.warning(f"SRTM offline cache lookup failed: {e}")
                    srtm_result = None

                if srtm_result is not None and srtm_result.reference_grid is not None:
                    try:
                        candidate_reference_grid = srtm_result.reference_grid
                        calib_split = split_calibration_holdout(
                            candidate_reference_grid, depth_result.height_proxy,
                            num_calib=2000, num_holdout=2000, seed=0,
                        )
                        calib = fit_linear_calibration(
                            calib_split.calib_height_proxy, calib_split.calib_reference,
                            min_points=cfg.calibration.min_reference_points,
                        )
                        dsm = generate_absolute_dsm(
                            depth_result.height_proxy, scale=calib.scale, offset=calib.offset,
                            guide_rgb=img.rgb,
                        )
                        reference_grid = candidate_reference_grid
                        calibrated = True
                        calibration_info.update({
                            "calibration_source": "offline_srtm_cache",
                            "srtm_tiles_used": srtm_result.tiles_found,
                            "calibration_method": calib.method,
                            "calibration_points_used": calib.num_points_used,
                            "calibration_warning": calib.warning,
                            "calibration_rmse_m": round(calib.residual_rmse_m, 3),
                            "calibration_mae_m": round(calib.residual_mae_m, 3),
                            "calibration_r": round(calib.r_value, 3),
                        })
                    except InsufficientCalibrationDataError as e:
                        calibration_info["warning"] = str(e)
                        calib_split = None

            if not calibrated and sun_elevation_deg is not None and sun_azimuth_deg is not None:
                attempted_sources.append("shadow_length_cue")
                # No reference DEM/GCP/SRTM available -- fall back to
                # shadow-length calibration (depth/shadow_cue.py), a
                # physically-grounded alternative that needs no external
                # data beyond sun position at capture time. See that
                # module's docstring for the trigonometry and its
                # explicit limitations.
                try:
                    from depthwizard.depth.shadow_cue import (
                        estimate_shadow_heights,
                        sample_height_proxy_at_cue,
                    )

                    cue = estimate_shadow_heights(
                        img.rgb, sun_elevation_deg=sun_elevation_deg,
                        sun_azimuth_deg=sun_azimuth_deg, pixel_size_m=pixel_size_m,
                    )
                    if cue.num_regions_used < cfg.calibration.min_reference_points:
                        raise InsufficientCalibrationDataError(
                            f"Only {cue.num_regions_used} usable shadow-length anchor "
                            f"points were found in this scene (need >= "
                            f"{cfg.calibration.min_reference_points}). This scene may "
                            f"have too few well-defined cast shadows, or the supplied "
                            f"sun angles may not match this image."
                        )
                    hp_samples = sample_height_proxy_at_cue(depth_result.height_proxy, cue)
                    calib = fit_linear_calibration(
                        hp_samples, cue.heights_m,
                        min_points=cfg.calibration.min_reference_points,
                    )
                    dsm = generate_absolute_dsm(
                        depth_result.height_proxy, scale=calib.scale, offset=calib.offset,
                        guide_rgb=img.rgb,
                    )
                    shadow_cue_result = cue
                    calibrated = True
                    calibration_info.update({
                        "calibration_source": "shadow_length_cue",
                        "calibration_method": calib.method,
                        "calibration_points_used": calib.num_points_used,
                        "shadow_regions_detected": cue.num_regions_detected,
                        "shadow_regions_used": cue.num_regions_used,
                        "sun_elevation_deg": sun_elevation_deg,
                        "sun_azimuth_deg": sun_azimuth_deg,
                        "calibration_warning": calib.warning,
                        "calibration_rmse_m": round(calib.residual_rmse_m, 3),
                        "calibration_mae_m": round(calib.residual_mae_m, 3),
                        "calibration_r": round(calib.r_value, 3),
                    })
                    # NOTE: shadow calibration produces a small set of sparse
                    # anchor points, not a dense reference grid -- so
                    # reference_grid stays None and full-grid RMSE/terrain
                    # stratification cannot run for this job (see the
                    # not_evaluated note below). The calibration fit's own
                    # residual RMSE (calibration_rmse_m) is the honest
                    # accuracy indicator available in this mode.
                except InsufficientCalibrationDataError as e:
                    calibration_info["warning"] = str(e)

            if not calibrated:
                missing_tiles_note = ""
                if srtm_result is not None and srtm_result.tiles_missing:
                    missing_tiles_note = (
                        f" The offline SRTM cache is missing tile(s) "
                        f"{srtm_result.tiles_missing} for this scene's footprint -- "
                        f"run scripts/download_srtm_tiles.py for those tiles to enable "
                        f"automatic offline calibration next time."
                    )
                calibration_info["warning"] = (
                    f"Absolute mode requested but no calibration source succeeded "
                    f"(tried, in priority order: {attempted_sources or ['none available']}). "
                    f"Supply a reference DEM via POST /api/upload-reference, GCP points, "
                    f"or sun_elevation_deg/sun_azimuth_deg." + missing_tiles_note +
                    " Falling back to relative DSM."
                )
                dsm = generate_relative_dsm(depth_result.height_proxy, guide_rgb=img.rgb,
                                             vertical_range_units=100.0)
        else:
            dsm = generate_relative_dsm(depth_result.height_proxy, guide_rgb=img.rgb,
                                         vertical_range_units=100.0)

        uncertainty_result = None
        t_unc = None
        if compute_uncertainty:
            job.stage = "uncertainty_estimation"
            t0 = time.time()
            uncertainty_result = estimate_uncertainty(estimator, img.rgb)
            t_unc = time.time() - t0

        job.stage = "mesh_generation"
        t0 = time.time()
        mesh = build_terrain_mesh(
            dsm.elevation,
            max_resolution=cfg.visualization.max_mesh_resolution,
            vertical_exaggeration=cfg.visualization.vertical_exaggeration,
        )
        t_mesh = time.time() - t0

        job.stage = "analysis_layers"
        slope_raster = compute_slope_degrees(dsm.elevation, pixel_size_m)

        job.stage = "saving_results"
        result_dir = RESULTS_ROOT / job_id
        for sub in ("input", "dsm", "uncertainty", "mesh", "metrics", "metadata", "analysis"):
            (result_dir / sub).mkdir(parents=True, exist_ok=True)

        export_dsm_geotiff(
            dsm.elevation, result_dir / "dsm" / "dsm.tif",
            crs=img.crs, transform=img.transform,
        )
        np.save(result_dir / "dsm" / "dsm.npy", dsm.elevation)
        np.save(result_dir / "analysis" / "slope.npy", slope_raster)

        if shadow_cue_result is not None:
            import json as _json
            (result_dir / "analysis" / "shadow_cue.json").write_text(_json.dumps({
                "rows": shadow_cue_result.rows.tolist(),
                "cols": shadow_cue_result.cols.tolist(),
                "heights_m": shadow_cue_result.heights_m.tolist(),
                "num_regions_detected": shadow_cue_result.num_regions_detected,
                "num_regions_used": shadow_cue_result.num_regions_used,
                "sun_elevation_deg": shadow_cue_result.sun_elevation_deg,
                "sun_azimuth_deg": shadow_cue_result.sun_azimuth_deg,
            }))

        from PIL import Image as PILImage
        PILImage.fromarray(img.rgb).save(result_dir / "input" / "input.png")

        import json
        mesh_json = mesh_to_json(mesh)
        (result_dir / "mesh" / "mesh.json").write_text(json.dumps(mesh_json))

        if uncertainty_result is not None:
            np.save(result_dir / "uncertainty" / "uncertainty.npy", uncertainty_result.uncertainty_map)

        total_time = time.time() - t_start

        metadata = {
            "job_id": job_id,
            "model_name": depth_result.model_name,
            "device_used": depth_result.device_used,
            "is_absolute": dsm.is_absolute,
            "vertical_unit": dsm.vertical_unit,
            "refinement_applied": dsm.refinement_applied,
            "preprocessing_time_s": round(t_depth * 0.05, 4),  # preprocessing folded into depth call; see note
            "depth_inference_time_s": round(t_depth, 4),
            "uncertainty_time_s": round(t_unc, 4) if t_unc else None,
            "mesh_build_time_s": round(t_mesh, 4),
            "total_time_s": round(total_time, 4),
            "synthetic_demo": "synthetic" in UPLOAD_ORIGINAL_NAMES.get(file_id, "").lower(),
            "pixel_size_m": round(pixel_size_m, 4),
            "pixel_size_is_metric_calibrated": pixel_size_is_metric,
            "slope_layer_available": True,
            "error_map_available": reference_grid is not None,
            **calibration_info,
        }

        job.result_dir = result_dir
        job.metadata = metadata

        if reference_grid is not None and calib_split is not None:
            from depthwizard.validation.metrics import (
                compute_metrics,
                error_raster,
                uncertainty_error_correlation,
            )
            from depthwizard.validation.terrain_stratification import (
                compute_stratified_metrics,
                stratified_result_to_json,
            )

            # Exclude the pixels used to FIT the calibration from every
            # accuracy number reported below -- see split_calibration_holdout.
            reference_for_validation = np.where(
                calib_split.calib_pixel_mask, np.nan, reference_grid
            ).astype(np.float32)

            metrics_result = compute_metrics(
                dsm.elevation, reference_for_validation,
                compute_slope=True, pixel_size_m=pixel_size_m,
            )

            err_raster = error_raster(dsm.elevation, reference_for_validation)
            np.save(result_dir / "analysis" / "error_map.npy", err_raster)

            unc_err_corr = None
            if uncertainty_result is not None:
                unc_err_corr = uncertainty_error_correlation(
                    uncertainty_result.uncertainty_map, err_raster
                )

            strat_result = compute_stratified_metrics(
                dsm.elevation, reference_for_validation, rgb=img.rgb,
                pixel_size_m=pixel_size_m, compute_slope=True,
            )
            strat_json = stratified_result_to_json(strat_result)
            np.save(result_dir / "analysis" / "terrain_class_map.npy", strat_result.class_map)

            job.metrics = {
                "status": "evaluated",
                "mae": round(metrics_result.mae, 4),
                "rmse": round(metrics_result.rmse, 4),
                "correlation": round(metrics_result.correlation, 4) if metrics_result.correlation == metrics_result.correlation else None,
                "median_abs_error": round(metrics_result.median_abs_error, 4),
                "holdout_pixel_count": int(calib_split.holdout_mask.sum()),
                "calibration_pixel_count": int(calib_split.calib_pixel_mask.sum()),
                "uncertainty_error_correlation": round(unc_err_corr, 4) if unc_err_corr is not None else None,
                "terrain_stratification": strat_json,
                "note": (
                    "Evaluated on the reference DEM/DSM grid, EXCLUDING the "
                    f"{int(calib_split.calib_pixel_mask.sum())} pixels used to fit "
                    "the calibration (no train/test leakage). Terrain-class "
                    f"breakdown method: {strat_json['method']}."
                ),
            }
        else:
            if calibration_info.get("calibration_source") == "shadow_length_cue":
                note = (
                    f"Absolute elevation was calibrated from "
                    f"{calibration_info.get('shadow_regions_used')} shadow-length anchor "
                    f"points (no dense reference DEM was supplied), so full-grid RMSE/MAE "
                    f"and terrain-stratified validation cannot be computed for this job. "
                    f"The calibration fit's own residual RMSE against those anchor points "
                    f"(metadata.calibration_rmse_m = {calibration_info.get('calibration_rmse_m')}m) "
                    f"is the honest accuracy indicator available in this mode -- treat it as "
                    f"weaker evidence than a dense-DEM validation, since it is only as good "
                    f"as the shadow detection and the flat-ground assumption at each anchor."
                )
            else:
                note = "No reference DSM/LiDAR supplied for this job, or calibration did not succeed."
            job.metrics = {
                "status": "not_evaluated",
                "note": note,
            }

        (result_dir / "metrics" / "metrics.json").write_text(json.dumps(job.metrics, indent=2, default=str))
        (result_dir / "metadata" / "metadata.json").write_text(json.dumps(metadata, indent=2))

        job.status = "completed"
        job.stage = "done"
        logger.info(f"Job {job_id} completed in {total_time:.2f}s")

    except Exception as e:
        logger.info(f"Job {job_id} FAILED at stage {job.stage}: {e}")
        job.status = "failed"
        job.error = str(e)
