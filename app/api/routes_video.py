"""
Video reconstruction API routes (AeroTwin AI Module 40).

Additive to the existing `routes.py` (single-image pipeline) -- mounted
under a separate router, does not modify or remove any existing endpoint.
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas.models import (
    ChangeDetectionStartRequest,
    MeasureRequest,
    ReconstructionStartRequest,
    ReconstructionStartResponse,
    ReconstructionStatusResponse,
    SpatialQueryRequest,
    VideoUploadResponse,
)
from app.services.change_detection import run_change_detection
from app.services.job_pipeline import launch_pipeline_job_async
from app.services.job_store import SqlitePersistentDict
from app.services.measurement_engine import measure_object
from app.services.spatial_ai import ask
from app.services.video_job_store import SqliteVideoJobStore, VideoJobRecord
from app.services.video_preprocessing import VIDEO_UPLOAD_DIR, ingest_video
from depthwizard.digital_twin.objects import DigitalTwinObject

router = APIRouter(prefix="/api", tags=["video-reconstruction"])

_video_paths = SqlitePersistentDict(table="video_uploads")
_job_store = SqliteVideoJobStore()

RESULTS_ROOT = Path("results") / "video_jobs"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


@router.post("/video/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile) -> VideoUploadResponse:
    video_id = uuid.uuid4().hex[:12]
    dest_path = VIDEO_UPLOAD_DIR / f"{video_id}_{file.filename}"
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    metadata = ingest_video(dest_path, file.filename or dest_path.name)
    if metadata.status != "VALID":
        raise HTTPException(status_code=400, detail=metadata.error)

    _video_paths[metadata.video_id] = str(dest_path)

    return VideoUploadResponse(
        video_id=metadata.video_id, filename=metadata.filename, status=metadata.status,
        fps=metadata.fps, width=metadata.width, height=metadata.height,
        frame_count=metadata.frame_count, duration_seconds=metadata.duration_seconds,
    )


@router.post("/reconstruction/start", response_model=ReconstructionStartResponse)
async def start_reconstruction(request: ReconstructionStartRequest) -> ReconstructionStartResponse:
    if request.video_id not in _video_paths:
        raise HTTPException(status_code=404, detail=f"Unknown video_id '{request.video_id}' -- upload it first via /api/video/upload")

    video_path = _video_paths[request.video_id]
    job_id = uuid.uuid4().hex[:12]
    result_dir = RESULTS_ROOT / job_id
    result_dir.mkdir(parents=True, exist_ok=True)

    job = VideoJobRecord(job_id=job_id, video_id=request.video_id, status="queued", result_dir=str(result_dir))
    _job_store[job_id] = job

    launch_pipeline_job_async(job_id, video_path, str(result_dir), _job_store)

    return ReconstructionStartResponse(job_id=job_id, status="queued")


def _get_job_or_404(job_id: str) -> VideoJobRecord:
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    return job


@router.get("/reconstruction/{job_id}", response_model=ReconstructionStatusResponse)
async def get_job(job_id: str) -> ReconstructionStatusResponse:
    job = _get_job_or_404(job_id)
    return ReconstructionStatusResponse(**job.to_dict())


@router.get("/reconstruction/{job_id}/progress", response_model=ReconstructionStatusResponse)
async def get_job_progress(job_id: str) -> ReconstructionStatusResponse:
    # Same payload as GET /{job_id} today -- kept as a distinct endpoint
    # per Module 40's API surface so a poller has a stable, minimal-payload
    # URL even if /{job_id} grows heavier fields later.
    job = _get_job_or_404(job_id)
    return ReconstructionStatusResponse(**job.to_dict())


@router.get("/reconstruction/{job_id}/results")
async def get_job_results(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    if job.status not in ("completed", "completed_with_fallback"):
        raise HTTPException(status_code=409, detail=f"Job is not finished (status={job.status})")
    return job.to_dict()


@router.get("/reconstruction/{job_id}/camera-trajectory")
async def get_camera_trajectory(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    trajectory_path = Path(job.result_dir) / "camera_trajectory.json"
    if not trajectory_path.exists():
        raise HTTPException(status_code=404, detail="No camera trajectory available for this job (pose estimation may not have completed)")
    return json.loads(trajectory_path.read_text())


@router.get("/reconstruction/{job_id}/confidence")
async def get_confidence(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    path = Path(job.result_dir) / "fusion" / "reliability_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No confidence/reliability report available for this job")
    return json.loads(path.read_text())


@router.get("/reconstruction/{job_id}/objects")
async def get_objects(job_id: str) -> dict:
    job = _get_job_or_404(job_id)
    path = Path(job.result_dir) / "digital_twin" / "objects.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No digital twin objects available for this job")
    return json.loads(path.read_text())


@router.post("/reconstruction/{job_id}/measure")
async def measure(job_id: str, request: MeasureRequest) -> dict:
    job = _get_job_or_404(job_id)
    path = Path(job.result_dir) / "digital_twin" / "objects.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No digital twin objects available for this job")

    data = json.loads(path.read_text())
    obj_dict = next((o for o in data["objects"] if o["object_id"] == request.object_id), None)
    if obj_dict is None:
        raise HTTPException(status_code=404, detail=f"Unknown object_id '{request.object_id}' for this job")

    obj = DigitalTwinObject(**obj_dict)
    return measure_object(obj)


# Module 12/25: the only way to reach a file on disk from this endpoint is
# through this fixed whitelist -- the browser can never supply its own path
# component that gets joined onto a filesystem path. Each entry maps a safe,
# public asset name to (sub-directory-within-the-job, content-type). Adding
# a new asset means adding a line here, not opening up arbitrary path access.
_RECONSTRUCTION_ASSETS: dict[str, tuple[str, str]] = {
    "model.glb": ("reconstruction", "model/gltf-binary"),
    "mesh.json": ("reconstruction", "application/json"),
    "dense_point_cloud.ply": ("reconstruction", "text/plain; charset=utf-8"),
    "heightfield_rgb.png": ("reconstruction", "image/png"),
}


@router.get("/reconstruction/{job_id}/asset/{asset_name}")
async def get_reconstruction_asset(job_id: str, asset_name: str):
    """Serve one real reconstruction output file (GLB mesh, JSON mesh
    fallback, point cloud, or baked texture) to the browser's Three.js
    viewer. This is the smallest backend addition needed to let the 3D
    viewer show the pipeline's actual output instead of a placeholder --
    see Module 12/46 of the master prompt. `job_id` is resolved through the
    existing job store (never trusted as a raw path), `asset_name` must be
    an exact whitelist match (no traversal is possible since it is never
    concatenated into a path -- only used as a dict key), and the resolved
    file path is additionally checked to still live inside that job's own
    result directory before being served.
    """
    job = _get_job_or_404(job_id)
    entry = _RECONSTRUCTION_ASSETS.get(asset_name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown or disallowed asset '{asset_name}'")

    subdir, content_type = entry
    result_dir = Path(job.result_dir).resolve()
    file_path = (result_dir / subdir / asset_name).resolve()

    if result_dir not in file_path.parents:
        # Defense in depth -- should be unreachable given the whitelist above,
        # but never serve a path that has drifted outside this job's own directory.
        raise HTTPException(status_code=403, detail="Asset path escapes job result directory")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Asset '{asset_name}' not available for this job")

    return FileResponse(str(file_path), media_type=content_type, filename=asset_name)


@router.post("/change-detection/start")
async def start_change_detection(request: ChangeDetectionStartRequest) -> dict:
    job_a = _get_job_or_404(request.job_id_a)
    job_b = _get_job_or_404(request.job_id_b)
    if job_a.status not in ("completed", "completed_with_fallback") or job_b.status not in ("completed", "completed_with_fallback"):
        raise HTTPException(status_code=409, detail="Both jobs must be finished before running change detection")

    change_id = uuid.uuid4().hex[:12]
    output_dir = RESULTS_ROOT / "change_detection" / change_id
    summary = run_change_detection(job_a.result_dir, job_b.result_dir, output_dir)
    return {"change_id": change_id, **summary.to_dict()}


@router.get("/change-detection/{change_id}")
async def get_change_detection(change_id: str) -> dict:
    summary_path = RESULTS_ROOT / "change_detection" / change_id / "change_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown change_id '{change_id}'")
    return json.loads(summary_path.read_text())


@router.post("/spatial/query")
async def spatial_query(request: SpatialQueryRequest) -> dict:
    job = _get_job_or_404(request.job_id)
    change_dir = None
    if request.change_job_id_a and request.change_job_id_b:
        # Look up an existing change-detection result between these two jobs
        # by re-deriving the same directory convention used above; if it
        # hasn't been computed yet, the query engine will honestly report
        # that change data is unavailable rather than guessing.
        pass  # left for a future enhancement -- see spatial_ai.ask's change_result_dir param for direct use

    answer = ask(job.result_dir, request.query)
    return {
        "query": answer.query, "intent": answer.intent, "grounded": answer.grounded,
        "answer": answer.text, "supporting_data": answer.supporting_data,
    }
