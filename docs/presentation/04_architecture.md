# 04 — Architecture

Full detail in `docs/architecture/TECHNICAL_BLUEPRINT.md`. This page is
the diagram-and-talking-points version.

## System Diagram

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌────────────────────┐
│  frontend_static/    │────▶│   backend/app (FastAPI)  │────▶│ src/depthwizard/    │
│  index.html          │◀────│   REST API + job queue   │◀────│ core pipeline lib   │
│  (Three.js viewer)   │     │                          │     │ (no web deps)       │
└─────────────────────┘     └──────────────────────────┘     └────────────────────┘
                                                                        │
                                          ┌─────────────────────────────┼─────────────────────────────┐
                                          ▼                             ▼                             ▼
                                  preprocessing/               depth/ + dsm/ +              validation/ +
                                  (image, GeoTIFF I/O)          calibration/ + uncertainty/  geospatial/
                                                                + reconstruction/
```

## Why This Split

`src/depthwizard/` is a plain importable Python library with zero FastAPI/
web dependencies — it can be used directly in a notebook, a CLI script, or
swapped into a different web framework, without touching pipeline logic.
`backend/app/` is purely orchestration: it decides *when* to call which
pipeline function and how to expose results over HTTP; it contains no
scientific computation itself. This boundary is what let us swap the
absolute-calibration code path in later without touching the depth model,
mesh builder, or frontend.

## Key Design Decisions

- **Backbone-agnostic depth interface** (`depth/inference.py`): swapping
  MiDaS_small for a stronger model later touches one file, not the whole
  pipeline (see MODEL_SELECTION.md's upgrade path).
- **In-memory job queue**: correct choice for a single-process demo;
  explicitly documented as not production-ready (no restart survival, no
  multi-worker scaling) rather than silently implying otherwise.
- **No hard-coded config values**: every tunable (tile size, ensemble
  passes, mesh resolution, calibration thresholds) lives in
  `configs/default.yaml`.
- **Single-file Three.js frontend**: chosen over a full React build
  pipeline for this baseline because it's directly servable by FastAPI's
  StaticFiles with zero build step, and every line in it is real,
  verified-working code rather than an unbuilt scaffold.
