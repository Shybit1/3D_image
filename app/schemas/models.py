from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    is_georeferenced: bool
    width: int
    height: int
    crs: Optional[str] = None


class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    status: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    error: Optional[str] = None


class ReconstructionStartRequest(BaseModel):
    video_id: str


class ReconstructionStartResponse(BaseModel):
    job_id: str
    status: str


class ReconstructionStatusResponse(BaseModel):
    job_id: str
    video_id: str
    status: str
    current_stage: Optional[str] = None
    stages_completed: list[str] = []
    progress_pct: float = 0.0
    error: Optional[str] = None
    final_mode: Optional[str] = None
    metrics: dict[str, Any] = {}


class MeasureRequest(BaseModel):
    object_id: str


class ChangeDetectionStartRequest(BaseModel):
    job_id_a: str
    job_id_b: str


class SpatialQueryRequest(BaseModel):
    job_id: str
    query: str
    change_job_id_a: Optional[str] = None
    change_job_id_b: Optional[str] = None


class GCPPoint(BaseModel):
    """A single ground-control point: a pixel location in the uploaded
    image paired with its known real-world elevation (e.g. from a survey,
    a known landmark height, or manual measurement). row/col are 0-indexed
    pixel coordinates in the ORIGINAL uploaded image."""
    row: int
    col: int
    elevation_m: float


class ReconstructRequest(BaseModel):
    file_id: str
    mode: Literal["relative", "absolute"] = "relative"
    compute_uncertainty: bool = True
    reference_dem_file_id: Optional[str] = None
    # Optional sun-position metadata enabling shadow-length height
    # calibration (see depth/shadow_cue.py) as an alternative to a
    # reference DEM, or read from scene metadata if the upload pipeline
    # extracts it in the future. NEVER defaulted/guessed server-side --
    # if omitted, shadow-based calibration is simply skipped.
    sun_elevation_deg: Optional[float] = None
    sun_azimuth_deg: Optional[float] = None
    # Optional ground control points for direct, human-supplied absolute
    # calibration -- see calibration source priority order in
    # services/pipeline.py. Preferred over SRTM/shadow cues when supplied,
    # since these are direct measurements rather than estimates.
    gcp_points: Optional[list[GCPPoint]] = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: Optional[str] = None
    error: Optional[str] = None


class MetricsResponse(BaseModel):
    status: Literal["evaluated", "not_evaluated"]
    mae: Optional[float] = None
    rmse: Optional[float] = None
    correlation: Optional[float] = None
    median_abs_error: Optional[float] = None
    note: Optional[str] = None
    # Honest-accuracy fields: metrics below are computed ONLY on pixels held
    # out from calibration sampling -- see calibration.scale_calibration.
    # split_calibration_holdout and services/pipeline.py for how the split
    # is enforced.
    holdout_pixel_count: Optional[int] = None
    calibration_pixel_count: Optional[int] = None
    # Correlation between the model's measured per-pixel uncertainty
    # (ensemble std) and its actual absolute error against the reference
    # DEM, on the held-out pixels. None (not 0.0) if not computable.
    uncertainty_error_correlation: Optional[float] = None
    # Per-terrain-class (urban/hilly/forested/sparse) RMSE/MAE/correlation
    # breakdown -- see validation.terrain_stratification. Structure:
    # {"method": "...", "classes": {"urban": {...}, "hilly": {...}, ...}}
    terrain_stratification: Optional[dict[str, Any]] = None


class ReconstructionMetadata(BaseModel):
    job_id: str
    model_name: str
    device_used: str
    is_absolute: bool
    vertical_unit: str
    calibration_method: Optional[str] = None
    calibration_points_used: Optional[int] = None
    calibration_warning: Optional[str] = None
    calibration_rmse_m: Optional[float] = None
    calibration_mae_m: Optional[float] = None
    calibration_r: Optional[float] = None
    refinement_applied: bool
    preprocessing_time_s: float
    depth_inference_time_s: float
    uncertainty_time_s: Optional[float] = None
    mesh_build_time_s: float
    total_time_s: float
    synthetic_demo: bool = False
