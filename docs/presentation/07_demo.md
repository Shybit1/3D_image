# 07 — Demo Script

## Setup (before judges arrive)

```bash
python scripts/download_models.py     # one-time, ~82MB
python scripts/generate_demo_data.py  # synthetic scene + ground truth
export PYTHONPATH=src:backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000/`.

## Live Demo Flow (≈2 minutes)

1. **Upload** `data/samples/synthetic_demo_scene.png` (or a real image if
   available) via drag-and-drop.
2. Leave mode on **Relative DSM**, click **Reconstruct**. Narrate the log
   panel as real pipeline stages print live (loading → depth inference →
   DSM → uncertainty → mesh) — this is genuine progress, not a canned
   animation.
3. Once loaded (~3 seconds), **orbit and zoom** the 3D mesh. Hover to show
   the live raycasted elevation readout in the HUD.
4. Switch view mode to **Uncertainty** — point out that the highlighted
   regions are real measured ensemble disagreement, not a fixed overlay.
5. Switch to **Texture** to show the RGB-draped mesh.
6. Switch mode to **Absolute DSM**, upload
   `data/samples/synthetic_scene_ground_truth_dsm.tif` as the reference
   DEM, and **Reconstruct** again on
   `data/samples/synthetic_scene_georeferenced.tif`.
7. Point at the bottom metrics bar: it now shows **real** RMSE/MAE/
   correlation numbers, computed live against the uploaded reference — not
   "Not evaluated" this time, because real reference data was supplied.
8. **Be upfront**: mention that this reference scene is synthetic (labeled
   as such throughout the UI/metadata), and that the measured weak
   correlation is itself the most scientifically interesting finding of
   the build — the honest current state of the domain-gap problem.

## Fallback Talking Points If Something Doesn't Render

- Show `pytest tests backend/tests -v` running live in a terminal instead
  — 34 passing tests is itself a strong, honest demo of correctness.
- Show `docs/experiments/RESULTS.md` directly — every number in it is
  real and was measured during this exact build.

## What NOT to Say

- Don't claim this beats LiDAR or claim a specific accuracy percentage
  that isn't in RESULTS.md.
- Don't call the synthetic scene "satellite data" — always say "synthetic
  demonstration scene."
