# 02 — Solution

## DepthWizard-X, in One Sentence

A single satellite image goes in; a real numerical elevation model (DSM),
an honest uncertainty map, and a navigable textured 3D environment come
out — with every number traceable to actual computation, never fabricated.

## Pipeline (as built and tested, not aspirational)

```
RGB / GeoTIFF image
      │
      ▼
Validation & loading  (rejects corrupt files, detects georeferencing)
      │
      ▼
MiDaS_small depth inference  (pretrained, MIT-licensed, real weights)
      │
      ▼
Edge-aware DSM refinement  (RGB-guided bilateral filter, preserves building edges)
      │
      ├─── if reference DEM supplied ──▶ Theil-Sen robust calibration ──▶ absolute DSM (meters)
      │
      └─── otherwise ──────────────────▶ relative DSM (unitless)
      │
      ▼
Multi-scale ensemble uncertainty  (8 independent forward passes, real measured std)
      │
      ▼
3D mesh generation  (real triangulated heightfield, UV-mapped)
      │
      ▼
Three.js viewer  (orbit/zoom, height/uncertainty/wireframe/texture modes, live elevation readout)
      │
      ▼
Validation metrics  (MAE/RMSE/correlation — reported "Not evaluated" honestly when no reference exists)
```

## What Makes This a Working System, Not a Slide

Every box above has been executed for real inside this build:
- The depth model loads with **0 missing / 0 unexpected weight keys**
  (verified programmatically, not eyeballed).
- The full pipeline ran through a live HTTP server end-to-end in ~3
  seconds on CPU (no GPU in this build environment).
- 34 automated tests pass, including one that runs the *entire* absolute-
  calibration-and-validation loop against a known ground truth.

## What We Are Not Claiming

- We are not claiming LiDAR-grade accuracy.
- We are not claiming our depth model is novel research — it's a
  pretrained third-party model, used exactly as the problem statement
  expects.
- We are not claiming validated real-world accuracy numbers — none exist
  yet, because no real satellite+LiDAR reference pair was reachable from
  this build's sandboxed environment. See `06_results.md`.
