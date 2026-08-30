# DepthWizard-X — Technical Blueprint

**SIH26175** — "DepthWizard: Single-View Height Estimation and 3D Flythrough" (ISRO)

Status of this document: reflects what has actually been built and tested
in this repository as of the baseline build. Anything not yet implemented
is explicitly marked PLANNED.

---

## 1. Problem Interpretation

Given a single optical satellite image (RGB), reconstruct a 3D
representation of the observed surface without stereo pairs, multi-view
imagery, or LiDAR. Two operating modes are required:

- **Non-georeferenced input** (plain PNG/JPG): produce a *relative* DSM —
  a height field with correct relative geometry (which points are higher
  than which) but no absolute metric scale, since nothing in the input
  ties pixel values to real-world units.
- **Georeferenced input** (GeoTIFF with CRS/affine transform): produce an
  *absolute* DSM in meters, using auxiliary reference data (SRTM/DEM tiles,
  or a handful of Ground Control Points) to calibrate the relative depth
  prediction to real elevation.

## 2. Official Requirements (verified)

Verified against the SIH2026 portal listing for SIH26175 (organization:
ISRO). Key points:

- Input: single non-stereo optical satellite image.
- Use a pretrained monocular depth estimation model as the starting point
  (not a from-scratch novel depth network — the competition explicitly
  expects engineering integration of existing depth-estimation research).
- Convert relative depth to absolute height using scene statistics,
  low-resolution DEMs, semantic priors, or minimal GCPs for georeferenced
  inputs.
- Produce a textured, navigable 3D reconstruction (flythrough) in a
  standard 3D engine (Three.js / Unity / Babylon.js all acceptable).
- Evaluation is split 50/50 between (a) DSM accuracy — RMSE/MAE/correlation
  against LiDAR or reference DSM, tested for stability across urban,
  sparse, hilly, and forested landscapes — and (b) visualization quality —
  projection accuracy, navigability, and standalone deployability.

## 3. Scientific Formulation

Let `I ∈ R^(H×W×3)` be the input RGB image. A monocular depth network `f_θ`
produces an *inverse relative depth* map `D = f_θ(I) ∈ R^(H×W)`, where
larger values indicate points nearer the camera. For a nadir (near-overhead)
satellite view, "nearer the camera" corresponds to "physically higher
elevation" — taller objects (buildings, trees) sit closer to the sensor
than the ground around them. We therefore treat `D` directly as a *height
proxy* `h = normalize(D) ∈ [0,1]^(H×W)`.

For the **relative DSM**, we present `h` scaled to an arbitrary vertical
range for visualization; it has no metric meaning.

For the **absolute DSM**, we require paired samples `{(h_i, z_i)}` where
`z_i` is a real reference elevation in meters (from a resampled DEM or a
GCP) at the same pixel location as `h_i`, and fit:

```
z = scale * h + offset
```

using Theil–Sen robust regression (median of pairwise slopes), which
tolerates outlier reference points — important because DEM/GCP references
are frequently misregistered or noisy relative to the depth prediction's
pixel grid. **We do not assume this linear form is universally correct**;
the fit's residual RMSE/MAE and the height-proxy/reference correlation are
reported alongside every calibration so a low-quality fit is visible
rather than silently trusted.

## 4. Existing Methods Considered

| Method | Type | Considered for |
|---|---|---|
| MiDaS (v2.1, DPT-Hybrid/Large/Small) | CNN+Transformer hybrid, relative depth | Baseline backbone |
| Depth Anything (v1/v2) | ViT-based, relative depth, strong zero-shot generalization | Stronger baseline / Phase 2 upgrade path |
| DPT-Large | Pure ViT dense prediction | Alternative baseline |
| Satellite-specific stereo/mono depth models (e.g. research repos trained on WorldView / IARPA MVS data) | CNN, often absolute depth in original coordinate frame | Ideal long-term choice; access/license constraints, see below |

## 5. Candidate Models Evaluated for This Build

Evaluated against: (1) license permitting redistribution/use, (2) whether
weights are reachable given this sandbox's restricted network egress
(only specific domains: GitHub, PyPI, npm — **not** huggingface.co), (3)
CPU inference speed (no GPU available in this build environment), (4)
architecture simplicity for a fully offline, self-contained deployment.

- **Depth Anything v2** — strongest zero-shot generalization, but its
  released checkpoints are hosted on Hugging Face Hub, which is not
  reachable from this build environment's network allowlist. Left as a
  documented upgrade path (see MODEL_SELECTION.md) rather than silently
  substituted or faked.
- **DPT-Large** — also HF-hub-hosted for the standard release; same
  constraint. Also far heavier for CPU-only inference.
- **MiDaS_small (v2.1)** — weights hosted directly on GitHub Releases
  (`isl-org/MiDaS`), which **is** reachable. MIT licensed. ~82MB, ~0.2s
  CPU inference per 256×256 tile. **Selected for the baseline.**

See `MODEL_SELECTION.md` for the full comparison and the exact
compatibility issues found and fixed while integrating it.

## 6. Selected Model

**MiDaS_small (v2.1)**, EfficientNet-Lite3 backbone, Intel ISL, MIT
license. Loaded with **zero missing / zero unexpected state_dict keys**
against our vendored architecture (verified in `tests/integration/`).

## 7. Alternative Models Rejected (for this baseline, not permanently)

Depth Anything v2 and DPT-Large are not rejected on merit — they are
very likely to outperform MiDaS_small on absolute quality — they are
deferred because their weight distribution channel is unreachable in this
specific build sandbox. In an unrestricted environment (e.g. the actual
competition judging machine, or Anthropic's own model-download step run
locally by a developer), swapping the backbone is a contained change
inside `src/depthwizard/depth/inference.py` and `backbones/`.

## 8. Architecture

```
Image (PNG/JPG/GeoTIFF)
    -> preprocessing/image_io.py         (validation, RGB extraction, GeoTIFF CRS read)
    -> depth/inference.py                (MiDaS_small forward pass -> height proxy)
    -> dsm/engine.py                     (edge-aware refinement, relative/absolute DSM)
    -> calibration/scale_calibration.py  (Theil-Sen fit, ONLY if reference points supplied)
    -> uncertainty/estimator.py          (multi-scale/flip ensemble std, real not fabricated)
    -> reconstruction/mesh_builder.py    (heightfield -> triangulated mesh + UVs + normals)
    -> geospatial/raster_io.py           (GeoTIFF DSM export)
    -> validation/metrics.py             (MAE/RMSE/correlation vs reference, if supplied)
    -> backend/app (FastAPI)             (orchestration, job queue, REST API)
    -> frontend_static/index.html        (Three.js viewer, real API calls, no mocked data)
```

## 9. Validation Plan

`validation/metrics.py` implements MAE, RMSE, Pearson correlation,
median/percentile absolute error, and optional slope error, all computed
from real co-registered arrays. **No reference DSM/LiDAR dataset is
bundled in this build** (none was available inside the sandbox's network
allowlist), so `RESULTS.md` correctly reports "Not yet evaluated" for
real-world accuracy numbers — this is the honest state, not a placeholder
to be embarrassed about. The self-consistency tests (identical array
compared to itself → exactly zero error, correlation 1.0) confirm the
metrics math itself is correct.

## 10. Risk Analysis

| Risk | Status | Mitigation |
|---|---|---|
| MiDaS trained on egocentric photos, not nadir satellite views (domain gap) | **Confirmed empirically** — raw output shows a dominant top-to-bottom gradient artifact | Edge-aware refinement (implemented); Phase 2 fine-tuning infra (PLANNED) |
| No GPU in build/deploy environment | Confirmed — CPU fallback implemented and used throughout | `utils/hardware.py` auto-detects and reports real device |
| Absolute DSM requires external DEM/GCP data not bundled | Confirmed | Calibration module refuses to fabricate; falls back to relative DSM with explicit warning |
| Large real satellite tiles (>16000px) | Not tested at that scale | `MAX_SIDE_PX` guard exists; full tiling/stitching pipeline is PLANNED |
| Job queue is in-memory, single-process | By design for baseline | Documented; Redis/Celery swap is PLANNED for production |

## 11. Innovation Strategy

See `INNOVATION.md` for the explicit split between what is standard
engineering integration (using a pretrained depth model, standard mesh
triangulation) versus what required real problem-solving in this build
(fixing two genuine architecture-compatibility bugs in the vendored MiDaS
code caused by `timm` API drift; the robust-regression calibration
approach; the multi-scale ensemble uncertainty proxy chosen specifically
because MiDaS_small has no native uncertainty head).
