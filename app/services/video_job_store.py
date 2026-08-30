"""
Video reconstruction job store (AeroTwin AI Module 26/27).

Extends the existing `job_store.py` SQLite pattern rather than replacing
it -- the original image-pipeline jobs table is untouched; this adds a
SEPARATE `video_jobs` table with the multi-stage progress fields the
video pipeline needs (Module 27's 13-stage list) that the single-image
job record never needed.

Per Module 26: SQLite is an intentional, stated "not a real distributed
queue" choice appropriate for a demo/small-deployment scale, same
reasoning as the existing job_store.py module.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from app.services.job_store import DEFAULT_DB_PATH

# Module 27's stage list, in pipeline order. A job's `current_stage` is
# always one of these (or None before it starts); `stages_completed` is
# the prefix of this list actually finished so far.
PIPELINE_STAGES = [
    "VIDEO_ANALYSIS", "KEYFRAME_SELECTION", "POSE_ESTIMATION", "SPARSE_RECONSTRUCTION",
    "AI_DEPTH", "DEPTH_FUSION", "UNCERTAINTY", "DENSE_RECONSTRUCTION",
    "SEMANTIC_ANALYSIS", "DIGITAL_TWIN", "FINALIZATION",
]

# Module 29's final-mode report.
FINAL_MODES = ("FULL_RECONSTRUCTION", "GEOMETRIC_MODE", "AI_DEPTH_MODE", "FALLBACK_MODE")


class VideoJobRecord:
    __slots__ = (
        "job_id", "video_id", "status", "current_stage", "stages_completed",
        "progress_pct", "error", "result_dir", "final_mode", "metrics",
        "_store", "_persist_enabled",
    )

    def __init__(
        self, job_id: str, video_id: str, status: str = "queued",
        current_stage: Optional[str] = None, stages_completed: Optional[list] = None,
        progress_pct: float = 0.0, error: Optional[str] = None,
        result_dir: Optional[str] = None, final_mode: Optional[str] = None,
        metrics: Optional[dict] = None,
    ):
        object.__setattr__(self, "_store", None)
        object.__setattr__(self, "_persist_enabled", False)
        self.job_id = job_id
        self.video_id = video_id
        self.status = status
        self.current_stage = current_stage
        self.stages_completed = stages_completed or []
        self.progress_pct = progress_pct
        self.error = error
        self.result_dir = result_dir
        self.final_mode = final_mode
        self.metrics = metrics or {}
        object.__setattr__(self, "_persist_enabled", True)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in ("_store", "_persist_enabled"):
            return
        if self._persist_enabled and self._store is not None:
            self._store._persist(self)

    def advance_stage(self, stage: str) -> None:
        """Mark `stage` as the currently-running stage and everything
        before it in PIPELINE_STAGES as completed -- keeps progress_pct
        derivable purely from position in the known stage list, not a
        second hand-maintained number that can drift out of sync."""
        if stage not in PIPELINE_STAGES:
            raise ValueError(f"Unknown stage '{stage}', must be one of {PIPELINE_STAGES}")
        idx = PIPELINE_STAGES.index(stage)
        self.stages_completed = PIPELINE_STAGES[:idx]
        self.current_stage = stage
        self.progress_pct = round(100.0 * idx / len(PIPELINE_STAGES), 1)

    def mark_stage_complete(self, stage: str) -> None:
        idx = PIPELINE_STAGES.index(stage)
        self.stages_completed = PIPELINE_STAGES[: idx + 1]
        self.progress_pct = round(100.0 * (idx + 1) / len(PIPELINE_STAGES), 1)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id, "video_id": self.video_id, "status": self.status,
            "current_stage": self.current_stage, "stages_completed": self.stages_completed,
            "progress_pct": self.progress_pct, "error": self.error,
            "result_dir": self.result_dir, "final_mode": self.final_mode, "metrics": self.metrics,
        }


class SqliteVideoJobStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, VideoJobRecord] = {}
        self._init_schema()
        self._load_all()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_jobs (
                    job_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT,
                    stages_completed_json TEXT,
                    progress_pct REAL DEFAULT 0.0,
                    error TEXT,
                    result_dir TEXT,
                    final_mode TEXT,
                    metrics_json TEXT,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)

    def _load_all(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id, video_id, status, current_stage, stages_completed_json, "
                "progress_pct, error, result_dir, final_mode, metrics_json FROM video_jobs"
            ).fetchall()
        for job_id, video_id, status, current_stage, stages_json, progress_pct, error, result_dir, final_mode, metrics_json in rows:
            rec = VideoJobRecord(
                job_id=job_id, video_id=video_id, status=status, current_stage=current_stage,
                stages_completed=json.loads(stages_json) if stages_json else [],
                progress_pct=progress_pct or 0.0, error=error, result_dir=result_dir,
                final_mode=final_mode, metrics=json.loads(metrics_json) if metrics_json else {},
            )
            object.__setattr__(rec, "_store", self)
            self._cache[job_id] = rec

    def _persist(self, rec: VideoJobRecord):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO video_jobs (job_id, video_id, status, current_stage, stages_completed_json,
                    progress_pct, error, result_dir, final_mode, metrics_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(job_id) DO UPDATE SET
                    video_id=excluded.video_id, status=excluded.status, current_stage=excluded.current_stage,
                    stages_completed_json=excluded.stages_completed_json, progress_pct=excluded.progress_pct,
                    error=excluded.error, result_dir=excluded.result_dir, final_mode=excluded.final_mode,
                    metrics_json=excluded.metrics_json, updated_at=excluded.updated_at
                """,
                (
                    rec.job_id, rec.video_id, rec.status, rec.current_stage,
                    json.dumps(rec.stages_completed), rec.progress_pct, rec.error,
                    rec.result_dir, rec.final_mode, json.dumps(rec.metrics, default=str),
                ),
            )

    def __setitem__(self, job_id: str, rec: VideoJobRecord):
        object.__setattr__(rec, "_store", self)
        self._cache[job_id] = rec
        self._persist(rec)

    def __getitem__(self, job_id: str) -> VideoJobRecord:
        return self._cache[job_id]

    def get(self, job_id: str, default=None):
        return self._cache.get(job_id, default)

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._cache
