from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.routes_video import router as video_router

app = FastAPI(
    title="DepthWizard-X API",
    description="Single-view satellite image -> DSM -> 3D reconstruction pipeline (SIH26175), "
                 "now also hosting the AeroTwin AI drone-video reconstruction pipeline.",
    version="0.2.0-aerotwin-phase10",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(video_router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "depthwizard-x-backend"}


FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend_static"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
