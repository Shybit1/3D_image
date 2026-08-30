from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas.models import (
    JobStatus,
    MetricsResponse,
    ReconstructionMetadata,
    ReconstructRequest,
    UploadResponse,
)
from app.services.pipeline import JOBS, JobRecord, UPLOADS, register_upload, run_reconstruction_job
from depthwizard.preprocessing.image_io import ImageValidationError, load_image

router = APIRouter(prefix="/api")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXT}")

    saved_name = f"{uuid.uuid4().hex[:12]}{suffix}"
    saved_path = UPLOAD_DIR / saved_name
    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        img = load_image(saved_path)
    except ImageValidationError as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(400, str(e))

    file_id = register_upload(saved_path, original_filename=file.filename or "")

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        is_georeferenced=img.is_georeferenced,
        width=img.width,
        height=img.height,
        crs=img.crs,
    )


@router.post("/upload-reference", response_model=UploadResponse)
async def upload_reference_dem(file: UploadFile = File(...)):
    """
    Upload a reference DEM/DSM (GeoTIFF, single-band elevation in meters)
    to be used as a calibration/validation reference for a subsequent
    /api/reconstruct call. This is what makes 'absolute' mode actually
    produce real metric elevation instead of falling back to relative.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        raise HTTPException(400, "Reference DEM must be a GeoTIFF (.tif/.tiff).")

    saved_name = f"ref_{uuid.uuid4().hex[:12]}{suffix}"
    saved_path = UPLOAD_DIR / saved_name
    with saved_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        img = load_image(saved_path)
    except ImageValidationError as e:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(400, str(e))

    if not img.is_georeferenced:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(400, "Reference DEM has no CRS/transform -- cannot be used for calibration.")

    file_id = register_upload(saved_path, original_filename=file.filename or "")

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        is_georeferenced=True,
        width=img.width,
        height=img.height,
        crs=img.crs,
    )


@router.post("/reconstruct", response_model=JobStatus)
async def reconstruct(req: ReconstructRequest, background_tasks: BackgroundTasks):
    if req.file_id not in UPLOADS:
        raise HTTPException(404, f"Unknown file_id '{req.file_id}'. Upload the image first.")
    if req.reference_dem_file_id and req.reference_dem_file_id not in UPLOADS:
        raise HTTPException(404, f"Unknown reference_dem_file_id '{req.reference_dem_file_id}'.")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = JobRecord(job_id=job_id)
    background_tasks.add_task(
        run_reconstruction_job, job_id, req.file_id, req.mode, req.compute_uncertainty,
        req.reference_dem_file_id, req.sun_elevation_deg, req.sun_azimuth_deg, req.gcp_points,
    )
    return JobStatus(job_id=job_id, status="queued")


@router.get("/job/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    job = _get_job_or_404(job_id)
    return JobStatus(job_id=job.job_id, status=job.status, stage=job.stage, error=job.error)


@router.get("/metadata/{job_id}", response_model=ReconstructionMetadata)
async def get_metadata(job_id: str):
    job = _get_job_or_404(job_id)
    if job.status != "completed" or job.metadata is None:
        raise HTTPException(409, f"Job not completed yet (status={job.status}).")
    return ReconstructionMetadata(**job.metadata)


@router.get("/metrics/{job_id}", response_model=MetricsResponse)
async def get_metrics(job_id: str):
    job = _get_job_or_404(job_id)
    if job.status != "completed":
        raise HTTPException(409, f"Job not completed yet (status={job.status}).")
    return MetricsResponse(**job.metrics)


@router.get("/mesh/{job_id}")
async def get_mesh(job_id: str):
    job = _get_job_or_404(job_id)
    _require_completed(job)
    mesh_path = job.result_dir / "mesh" / "mesh.json"
    if not mesh_path.exists():
        raise HTTPException(404, "Mesh not found for this job.")
    return json.loads(mesh_path.read_text())


@router.get("/dsm/{job_id}")
async def get_dsm(job_id: str):
    job = _get_job_or_404(job_id)
    _require_completed(job)
    tif_path = job.result_dir / "dsm" / "dsm.tif"
    if not tif_path.exists():
        raise HTTPException(404, "DSM raster not found for this job.")
    return FileResponse(tif_path, filename=f"dsm_{job_id}.tif")


@router.get("/uncertainty/{job_id}")
async def get_uncertainty(job_id: str):
    import numpy as np

    job = _get_job_or_404(job_id)
    _require_completed(job)
    unc_path = job.result_dir / "uncertainty" / "uncertainty.npy"
    if not unc_path.exists():
        raise HTTPException(404, "Uncertainty map was not computed for this job.")
    arr = np.load(unc_path)
    return {"shape": list(arr.shape), "values": arr.astype(float).tolist()}


@router.get("/slope/{job_id}")
async def get_slope(job_id: str):
    """
    Per-pixel terrain slope in degrees, computed from the job's DSM.
    Always available (does not require a reference DEM), but only
    physically exact when `pixel_size_is_metric_calibrated` in the job's
    metadata is true -- otherwise the pixel size (and thus the slope
    angle) is an approximation. The frontend should surface that flag,
    not just the numbers.
    """
    import numpy as np

    job = _get_job_or_404(job_id)
    _require_completed(job)
    slope_path = job.result_dir / "analysis" / "slope.npy"
    if not slope_path.exists():
        raise HTTPException(404, "Slope layer not found for this job.")
    arr = np.load(slope_path)
    return {
        "shape": list(arr.shape),
        "unit": "degrees",
        "pixel_size_is_metric_calibrated": job.metadata.get("pixel_size_is_metric_calibrated") if job.metadata else None,
        "values": np.nan_to_num(arr, nan=-1.0).astype(float).tolist(),
    }


@router.get("/error-map/{job_id}")
async def get_error_map(job_id: str):
    """
    Signed error raster (predicted - reference) in meters. Only available
    when a reference DEM was supplied AND calibration succeeded -- i.e.
    exactly when `metrics.status == "evaluated"` for this job. This is
    computed on holdout pixels only where it matters for accuracy claims;
    see the `metrics` endpoint for the aggregate numbers this raster
    supports.
    """
    import numpy as np

    job = _get_job_or_404(job_id)
    _require_completed(job)
    err_path = job.result_dir / "analysis" / "error_map.npy"
    if not err_path.exists():
        raise HTTPException(
            404,
            "Error map not available for this job (no reference DEM supplied, "
            "or absolute calibration did not succeed).",
        )
    arr = np.load(err_path)
    return {
        "shape": list(arr.shape),
        "unit": "meters",
        "values": np.nan_to_num(arr, nan=0.0).astype(float).tolist(),
        "valid_mask": np.isfinite(arr).astype(int).tolist(),
    }


@router.get("/shadow-cue/{job_id}")
async def get_shadow_cue(job_id: str):
    """
    Sparse shadow-length calibration anchor points (only present when
    absolute mode was calibrated via shadow length rather than a reference
    DEM -- see depth/shadow_cue.py). Returned as point lists rather than a
    dense raster since the cue is inherently sparse (typically tens to a
    few hundred anchors, not one per pixel).
    """
    job = _get_job_or_404(job_id)
    _require_completed(job)
    cue_path = job.result_dir / "analysis" / "shadow_cue.json"
    if not cue_path.exists():
        raise HTTPException(
            404,
            "No shadow-length calibration was performed for this job "
            "(either a reference DEM was used instead, or absolute mode wasn't requested).",
        )
    import json

    return json.loads(cue_path.read_text())


@router.get("/terrain-map/{job_id}")
async def get_terrain_map(job_id: str):
    """
    Per-pixel terrain class (0=urban, 1=hilly, 2=forested, 3=sparse) used
    for the stratified validation breakdown. Only available alongside the
    error map (same gating condition). See `metrics.terrain_stratification`
    for the per-class RMSE/MAE/correlation numbers this map corresponds to.
    """
    import numpy as np

    job = _get_job_or_404(job_id)
    _require_completed(job)
    class_path = job.result_dir / "analysis" / "terrain_class_map.npy"
    if not class_path.exists():
        raise HTTPException(404, "Terrain classification not available for this job.")
    arr = np.load(class_path)
    return {
        "shape": list(arr.shape),
        "class_names": {"0": "urban", "1": "hilly", "2": "forested", "3": "sparse"},
        "values": arr.astype(int).tolist(),
    }


@router.get("/texture/{job_id}")
async def get_texture(job_id: str):
    job = _get_job_or_404(job_id)
    _require_completed(job)
    input_path = job.result_dir / "input" / "input.png"
    if not input_path.exists():
        raise HTTPException(404, "Input texture not found for this job.")
    return FileResponse(input_path, media_type="image/png")


def _get_job_or_404(job_id: str) -> JobRecord:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id '{job_id}'")
    return job


def _require_completed(job: JobRecord):
    if job.status != "completed":
        raise HTTPException(409, f"Job not completed yet (status={job.status}).")
