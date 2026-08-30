"""
Exercises the real FastAPI app end-to-end: upload -> reconstruct -> poll ->
fetch mesh/dsm/metadata/metrics. Runs the actual pipeline (real MiDaS
inference), so it is slower than a pure unit test but validates that
nothing is mocked at the API boundary.
"""
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _synthetic_png_bytes():
    arr = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_endpoint():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_rejects_bad_extension():
    r = client.post("/api/upload", files={"file": ("bad.exe", b"junk", "application/octet-stream")})
    assert r.status_code == 400


def test_full_reconstruction_flow():
    r = client.post(
        "/api/upload",
        files={"file": ("synthetic_test.png", _synthetic_png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    file_id = r.json()["file_id"]
    assert r.json()["is_georeferenced"] is False

    r = client.post(
        "/api/reconstruct",
        json={"file_id": file_id, "mode": "relative", "compute_uncertainty": True},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    status = None
    for _ in range(60):
        r = client.get(f"/api/job/{job_id}")
        status = r.json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(1)

    assert status["status"] == "completed", f"Job did not complete: {status}"

    meta = client.get(f"/api/metadata/{job_id}").json()
    assert meta["model_name"].startswith("MiDaS_small")
    assert meta["is_absolute"] is False
    assert meta["synthetic_demo"] is True  # filename contains 'synthetic'

    metrics = client.get(f"/api/metrics/{job_id}").json()
    assert metrics["status"] == "not_evaluated"  # honest: no reference DSM supplied

    mesh = client.get(f"/api/mesh/{job_id}").json()
    assert mesh["vertex_count"] > 0
    assert mesh["face_count"] > 0

    dsm_resp = client.get(f"/api/dsm/{job_id}")
    assert dsm_resp.status_code == 200
    assert len(dsm_resp.content) > 0

    tex_resp = client.get(f"/api/texture/{job_id}")
    assert tex_resp.status_code == 200


def test_absolute_mode_with_reference_dem_calibrates_and_validates():
    """
    Full absolute-mode flow against a SYNTHETIC ground-truth DSM (exactly
    known by construction, from scripts/generate_demo_data.py). This tests
    that the calibration + validation code path actually runs and reports
    real numbers -- it does not (and cannot) prove real-world accuracy,
    since the "scene" is synthetic. See docs/experiments/RESULTS.md for
    how these numbers are interpreted.
    """
    import pathlib
    samples_dir = pathlib.Path("data/samples")
    img_path = samples_dir / "synthetic_scene_georeferenced.tif"
    dem_path = samples_dir / "synthetic_scene_ground_truth_dsm.tif"
    if not img_path.exists() or not dem_path.exists():
        pytest.skip("Run scripts/generate_demo_data.py first to create synthetic fixtures.")

    with open(img_path, "rb") as f:
        r = client.post("/api/upload", files={"file": (img_path.name, f, "image/tiff")})
    assert r.status_code == 200
    assert r.json()["is_georeferenced"] is True
    file_id = r.json()["file_id"]

    with open(dem_path, "rb") as f:
        r = client.post("/api/upload-reference", files={"file": (dem_path.name, f, "image/tiff")})
    assert r.status_code == 200
    ref_id = r.json()["file_id"]

    r = client.post(
        "/api/reconstruct",
        json={"file_id": file_id, "mode": "absolute", "compute_uncertainty": False,
              "reference_dem_file_id": ref_id},
    )
    job_id = r.json()["job_id"]

    status = None
    for _ in range(30):
        status = client.get(f"/api/job/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    assert status["status"] == "completed", status

    meta = client.get(f"/api/metadata/{job_id}").json()
    assert meta["is_absolute"] is True
    assert meta["calibration_method"] == "theil_sen_robust_linear"
    assert meta["calibration_points_used"] > 0

    metrics = client.get(f"/api/metrics/{job_id}").json()
    assert metrics["status"] == "evaluated"
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= metrics["mae"]  # RMSE >= MAE always holds mathematically


def test_reference_upload_rejects_non_geotiff():
    r = client.post(
        "/api/upload-reference",
        files={"file": ("notageotiff.png", _synthetic_png_bytes(), "image/png")},
    )
    assert r.status_code == 400


def test_unknown_job_id_404s():
    r = client.get("/api/job/doesnotexist")
    assert r.status_code == 404


def test_reconstruct_with_unknown_file_id_404s():
    r = client.post(
        "/api/reconstruct",
        json={"file_id": "nonexistent", "mode": "relative", "compute_uncertainty": False},
    )
    assert r.status_code == 404
