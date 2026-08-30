"""
SQLite-backed job store and upload registry.

WHY THIS EXISTS: the previous baseline used a plain in-memory dict for
both job status/results (`JOBS`) and the file_id -> path upload mapping
(`UPLOADS`). That meant any process restart (a deploy, a crash, a dev
auto-reload) silently erased every in-flight and completed job -- a
person polling GET /api/jobs/{id} would just start getting 404s, and the
underlying result files on disk (which DO survive a restart) would
become permanently unreachable because the job_id -> result_dir mapping
that pointed to them was gone.

This module fixes that using SQLite -- no new external service, no new
dependency (it's in the Python standard library), keeping this an
"offline-first" appropriate fix for a demo/small-deployment scale.

WHAT THIS IS NOT: a substitute for a real distributed task queue
(Celery/RQ + Redis) at multi-worker production scale. SQLite's
single-writer model means concurrent writes from multiple worker
PROCESSES serialize rather than truly parallelize, and there's no
built-in retry/dead-letter/priority handling here. For a genuinely
concurrent multi-user production deployment, swap SqliteJobStore for a
real queue -- but that is a materially bigger infrastructure change than
"stop losing job state on restart", which is the actual problem this
module solves.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_DB_PATH = Path("results") / "depthwizard_state.sqlite3"


class JobRecord:
    """
    Same shape/usage as the original plain dataclass (job.status = "...",
    job.stage = "...", job.metadata = {...}, etc.) so the many call sites
    scattered through services/pipeline.py that mutate a JobRecord's
    fields directly do not need to change at all.

    The difference: once a JobRecord is registered into a SqliteJobStore
    (via `store[job_id] = record`), every subsequent attribute write
    automatically persists the full record to SQLite. This is what makes
    `job.stage = "depth_inference"` (etc.) durable across a restart
    without rewriting every one of those call sites to something like
    `store.update(job_id, stage=...)`.
    """

    __slots__ = (
        "job_id", "status", "stage", "error", "result_dir", "metadata", "metrics",
        "_store", "_persist_enabled",
    )

    def __init__(self, job_id: str, status: str = "queued", stage: Optional[str] = None,
                 error: Optional[str] = None, result_dir: Optional[Path] = None,
                 metadata: Optional[dict] = None, metrics: Optional[dict] = None):
        object.__setattr__(self, "_store", None)
        object.__setattr__(self, "_persist_enabled", False)
        self.job_id = job_id
        self.status = status
        self.stage = stage
        self.error = error
        self.result_dir = result_dir
        self.metadata = metadata
        self.metrics = metrics
        object.__setattr__(self, "_persist_enabled", True)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in ("_store", "_persist_enabled"):
            return
        if self._persist_enabled and self._store is not None:
            self._store._persist(self)


class SqliteJobStore:
    """
    Drop-in replacement for the previous `JOBS: dict[str, JobRecord] = {}`
    global -- supports the same `JOBS[job_id]`, `JOBS[job_id] = record`,
    `JOBS.get(job_id)`, `job_id in JOBS` usage already used throughout the
    API layer, backed by SQLite instead of process memory. Existing jobs
    are reloaded from disk on construction, so a fresh process picks up
    where the last one left off.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, JobRecord] = {}
        self._init_schema()
        self._load_all()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT,
                    error TEXT,
                    result_dir TEXT,
                    metadata_json TEXT,
                    metrics_json TEXT,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)

    def _load_all(self):
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id, status, stage, error, result_dir, metadata_json, metrics_json FROM jobs"
            ).fetchall()
        for job_id, status, stage, error, result_dir, metadata_json, metrics_json in rows:
            rec = JobRecord(
                job_id=job_id, status=status, stage=stage, error=error,
                result_dir=Path(result_dir) if result_dir else None,
                metadata=json.loads(metadata_json) if metadata_json else None,
                metrics=json.loads(metrics_json) if metrics_json else None,
            )
            object.__setattr__(rec, "_store", self)
            self._cache[job_id] = rec

    def _persist(self, rec: JobRecord):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, stage, error, result_dir, metadata_json, metrics_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status, stage=excluded.stage, error=excluded.error,
                    result_dir=excluded.result_dir, metadata_json=excluded.metadata_json,
                    metrics_json=excluded.metrics_json, updated_at=excluded.updated_at
                """,
                (
                    rec.job_id, rec.status, rec.stage, rec.error,
                    str(rec.result_dir) if rec.result_dir else None,
                    json.dumps(rec.metadata, default=str) if rec.metadata is not None else None,
                    json.dumps(rec.metrics, default=str) if rec.metrics is not None else None,
                ),
            )

    def __setitem__(self, job_id: str, rec: JobRecord):
        object.__setattr__(rec, "_store", self)
        self._cache[job_id] = rec
        self._persist(rec)

    def __getitem__(self, job_id: str) -> JobRecord:
        return self._cache[job_id]

    def get(self, job_id: str, default=None):
        return self._cache.get(job_id, default)

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._cache

    def __len__(self) -> int:
        return len(self._cache)


class SqlitePersistentDict:
    """
    Generic SQLite-backed key/value store with a dict-like interface
    (`d[k] = v`, `d[k]`, `d.get(k, default)`, `k in d`), used here for the
    file_id -> upload-path and file_id -> original-filename mappings.
    The underlying uploaded files were already persisted to disk; only
    the id -> path/name mapping was previously memory-only and is what
    this fixes.

    `encode`/`decode` let callers store non-string values (e.g. Path)
    while keeping the storage layer itself simple (one TEXT column).
    """

    def __init__(self, table: str, db_path: Path = DEFAULT_DB_PATH,
                 encode: Callable[[Any], str] = str,
                 decode: Callable[[str], Any] = str):
        self.table = table
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._encode = encode
        self._decode = decode
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with self._lock, self._connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

    def __setitem__(self, key: str, value: Any):
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO {self.table} (key, value) VALUES (?, ?) "
                f"ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, self._encode(value)),
            )

    def __getitem__(self, key: str) -> Any:
        with self._lock, self._connect() as conn:
            row = conn.execute(f"SELECT value FROM {self.table} WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return self._decode(row[0])

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(f"SELECT 1 FROM {self.table} WHERE key=?", (key,)).fetchone()
        return row is not None
