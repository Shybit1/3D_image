"""
Backend orchestration for the Spatial AI Assistant (AeroTwin AI Module 21).
Ties `scene_database` (load) + `query_engine` (answer) together for one job.

For two-flight questions ("what changed between Flight 1 and Flight 2"),
pass `change_result_dir` -- the change_summary.json produced by
`change_detection.run_change_detection` for that specific job pair.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from depthwizard.spatial_ai.query_engine import SpatialAIAnswer, answer_query
from depthwizard.spatial_ai.scene_database import SceneDatabase, load_scene_database


def ask(result_dir: str | Path, query: str, change_result_dir: Optional[str | Path] = None) -> SpatialAIAnswer:
    result_dir = Path(result_dir)
    db = load_scene_database(result_dir)

    if change_result_dir is not None:
        import json
        change_summary_path = Path(change_result_dir) / "change_summary.json"
        if change_summary_path.exists():
            db.change_summary = json.loads(change_summary_path.read_text())
            if "change_detection" not in db.available_data:
                db.available_data.append("change_detection")

    return answer_query(query, db)
