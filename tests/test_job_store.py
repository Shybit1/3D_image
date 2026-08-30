"""
Tests the SQLite-backed job store and upload registry in isolation from
the rest of the pipeline (no torch/model dependency needed here) --
specifically the property that matters: state survives a fresh process
(simulated by constructing a brand new store instance against the same
db file, exactly as would happen on an actual server restart).
"""
from pathlib import Path

import pytest

from app.services.job_store import JobRecord, SqlitePersistentDict, SqliteJobStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_state.sqlite3"


def test_job_record_field_writes_persist_automatically(db_path):
    store = SqliteJobStore(db_path)
    rec = JobRecord(job_id="job1")
    store["job1"] = rec

    rec.status = "running"
    rec.stage = "depth_inference"
    rec.metadata = {"model_name": "MiDaS_small"}
    rec.metrics = {"status": "not_evaluated", "note": "test"}

    # No explicit "save" call anywhere above -- every attribute write on
    # rec should have auto-persisted via JobRecord.__setattr__.
    fresh_store = SqliteJobStore(db_path)
    reloaded = fresh_store["job1"]

    assert reloaded.status == "running"
    assert reloaded.stage == "depth_inference"
    assert reloaded.metadata == {"model_name": "MiDaS_small"}
    assert reloaded.metrics == {"status": "not_evaluated", "note": "test"}


def test_job_store_survives_simulated_restart_with_multiple_jobs(db_path):
    store = SqliteJobStore(db_path)
    for i in range(5):
        rec = JobRecord(job_id=f"job{i}")
        store[f"job{i}"] = rec
        rec.status = "completed" if i % 2 == 0 else "failed"

    fresh_store = SqliteJobStore(db_path)
    assert len(fresh_store) == 5
    for i in range(5):
        expected = "completed" if i % 2 == 0 else "failed"
        assert fresh_store[f"job{i}"].status == expected


def test_job_store_contains_and_get_with_default(db_path):
    store = SqliteJobStore(db_path)
    store["known"] = JobRecord(job_id="known")

    assert "known" in store
    assert "unknown" not in store
    assert store.get("unknown", "fallback") == "fallback"
    assert store.get("known") is not None


def test_job_record_result_dir_round_trips_as_path(db_path):
    store = SqliteJobStore(db_path)
    rec = JobRecord(job_id="job_with_dir")
    store["job_with_dir"] = rec
    rec.result_dir = Path("results") / "job_with_dir"

    fresh_store = SqliteJobStore(db_path)
    reloaded = fresh_store["job_with_dir"]
    assert isinstance(reloaded.result_dir, Path)
    assert reloaded.result_dir == Path("results") / "job_with_dir"


def test_persistent_dict_round_trips_with_custom_encode_decode(db_path):
    uploads = SqlitePersistentDict("uploads", db_path, encode=str, decode=Path)
    uploads["file1"] = Path("/tmp/foo.png")

    fresh = SqlitePersistentDict("uploads", db_path, encode=str, decode=Path)
    result = fresh["file1"]
    assert isinstance(result, Path)
    assert result == Path("/tmp/foo.png")


def test_persistent_dict_raises_keyerror_for_missing_key(db_path):
    uploads = SqlitePersistentDict("uploads", db_path)
    with pytest.raises(KeyError):
        uploads["does_not_exist"]


def test_persistent_dict_get_with_default(db_path):
    names = SqlitePersistentDict("upload_names", db_path)
    names["file1"] = "original.png"
    assert names.get("file1") == "original.png"
    assert names.get("missing", "") == ""


def test_persistent_dict_contains(db_path):
    d = SqlitePersistentDict("some_table", db_path)
    d["k"] = "v"
    assert "k" in d
    assert "other" not in d


def test_persistent_dict_overwrite_updates_value(db_path):
    d = SqlitePersistentDict("some_table", db_path)
    d["k"] = "v1"
    d["k"] = "v2"
    assert d["k"] == "v2"


def test_two_tables_in_same_db_file_dont_collide(db_path):
    # uploads and upload_names share one sqlite file but different tables --
    # make sure a key in one doesn't leak into the other.
    uploads = SqlitePersistentDict("uploads", db_path, encode=str, decode=Path)
    names = SqlitePersistentDict("upload_names", db_path)
    uploads["id1"] = Path("/tmp/a.png")
    names["id1"] = "a.png"

    assert uploads["id1"] == Path("/tmp/a.png")
    assert names["id1"] == "a.png"
    assert "id1" in uploads
    assert "id1" in names
