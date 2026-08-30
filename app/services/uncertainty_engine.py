"""
Backend orchestration for uncertainty/reliability reporting (AeroTwin AI
Module 9). The actual reliability report is computed inline during
`depth_fusion.run_depth_fusion` (it needs the in-memory per-frame fusion
results, not just the persisted arrays) -- this module provides the
read-back path for API routes that want the already-computed report
without re-running fusion.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_reliability_report(result_dir: str | Path) -> dict:
    path = Path(result_dir) / "fusion" / "reliability_report.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No reliability report at {path} -- has depth fusion (Phase 4) run for this job yet?"
        )
    return json.loads(path.read_text())
