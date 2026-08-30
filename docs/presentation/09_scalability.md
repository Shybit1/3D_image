# 09 — Scalability

## Current Limits (honest, from the actual implementation)

- **Job queue**: in-memory Python dict, single process. Fine for a demo
  or single-analyst tool; does not survive a restart and does not scale
  across multiple backend workers. Documented explicitly in code comments
  (`backend/app/services/pipeline.py`) rather than silently assumed away.
- **Image size**: `preprocessing/image_io.py` enforces a 16,000px per-side
  ceiling; genuinely large satellite scenes (30,000px+) are not yet
  tiled/stitched — `configs/default.yaml` already has `tile_size` and
  `overlap` fields reserved for this, but the tiling logic itself is
  PLANNED, not implemented.
- **Mesh resolution**: capped at 200×200 vertices by default
  (`visualization.max_mesh_resolution`) specifically to keep the mesh
  light enough for a browser — configurable, but full-resolution meshing
  of a multi-thousand-pixel DSM is not attempted (nor advisable for a
  WebGL viewer without LOD, which is also PLANNED).

## Path to Production Scale

1. **Job queue** → swap the in-memory dict for Redis/Celery or a DB-backed
   queue; the `run_reconstruction_job` function signature is already
   decoupled from the FastAPI request/response cycle, so this swap doesn't
   touch pipeline logic.
2. **Tiling** → implement the tile/overlap/stitch logic the config already
   anticipates, running each tile through the existing single-tile
   pipeline and merging results — no change to the depth/DSM/mesh code
   itself, only to how images are chunked before it.
3. **GPU fleet** → `utils/hardware.py`'s auto-detection already means no
   code change is needed to run on a CUDA machine; only infrastructure
   (container GPU scheduling) needs to be added.
4. **Mesh LOD** → straightforward Three.js extension (multiple mesh
   resolutions swapped by camera distance), independent of the backend.

## What We Deliberately Did Not Over-Engineer

We chose not to build a Kubernetes/microservices deployment or a
distributed task queue for this baseline, because doing so before the
core scientific pipeline was even verified correct would have been
premature — better to have a small, fully-tested, honestly-scoped system
than a large, partially-faked one.
