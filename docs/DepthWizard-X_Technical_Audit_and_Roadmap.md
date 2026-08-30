# DepthWizard-X — Technical Audit & Elevation Roadmap
### SIH26175: Single-View Satellite Image → DSM → Interactive 3D Environment

---

## 0. Top-line verdict

This is **not** a generic student demo. It's a genuine engineering baseline: real
rasterio-backed GeoTIFF I/O, a real Theil-Sen calibration fit with residual
reporting, a real multi-scale/flip test-time uncertainty ensemble, real
triangulated mesh generation, a 34-test pytest suite, and — most tellingly — a
README and RESULTS.md that **report a weak r≈0.29 correlation** for the
baseline depth model instead of hiding it. That last point matters more than
any single feature: most SIH repos fabricate an accuracy number. Yours found
its own weakness and documented it. That is exactly the posture a jury
evaluating "scientific honesty" will reward — don't lose it as you extend
this.

The gaps are real, but they're gaps of **coverage and integration**, not of
fabrication. Below is the audit, then the prioritized build plan.

---

## 1. Audit — Present vs. Missing

### 1.1 Geospatial pipeline

| Capability | Status | Evidence |
|---|---|---|
| Rasterio-backed GeoTIFF read/write | ✅ Present | `geospatial/raster_io.py`, `preprocessing/image_io.py` |
| CRS + affine transform propagation | ✅ Present | `LoadedImage.crs/.transform`, honestly `None` for non-georeferenced PNG/JPG (no invented CRS) |
| DEM reprojection onto image grid | ✅ Present | `resample_dem_to_grid()` via `rasterio.warp.reproject` |
| 16-bit satellite band handling | ✅ Present | `_robust_stretch_to_uint8()` (2–98 percentile stretch) |
| Pixel↔world coordinate helpers | ✅ Present | `pixel_to_world()` |
| **Auto SRTM/DEM acquisition (offline-bundled)** | ❌ Missing | Reference DEM must be manually uploaded per-job; nothing bundles/looks up SRTM tiles by scene footprint |
| **Large-scene tiling** | ❌ Missing (documented as PLANNED) | Hard 16000px ceiling in `image_io.py`, but no tile/stitch path — `InferenceConfig.tile_size` exists in config but is **never read** by any code path |
| Slope raster as a first-class product | ⚠️ Partial | `_slope_error()` computes slope internally for validation only; not exposed as a raster/endpoint |

### 1.2 Absolute elevation calibration

| Capability | Status | Evidence |
|---|---|---|
| Height-proxy → meters via robust regression | ✅ Present | Theil-Sen, resistant to DEM outliers (`scale_calibration.py`) |
| Refuses to fabricate absolute output when data is insufficient | ✅ Present | `InsufficientCalibrationDataError`, `min_points` guard |
| Residual RMSE/MAE/r reported alongside the fit | ✅ Present | `CalibrationResult` |
| GCP (point-list) calibration path | ⚠️ Partial | Math supports arbitrary paired samples; **no API/UI path** exists — only raster-DEM upload is wired |
| **Held-out validation split** | ❌ Missing (self-documented) | Validation in `pipeline.py` runs against the *same* reference grid used for calibration sampling (2000 of ~100k+ px overlap) — this inflates apparent agreement and is explicitly flagged as a known shortcut in the code comment |

### 1.3 Uncertainty estimation

| Capability | Status | Evidence |
|---|---|---|
| Genuine per-pixel measured variance (not a fixed confidence score) | ✅ Present | Multi-scale × flip ensemble, real `std()` across passes (`uncertainty/estimator.py`) |
| Documented limitation (no MC-Dropout/heteroscedastic head available for this backbone) | ✅ Present | Docstring is explicit about *why* this method was chosen |
| Uncertainty exposed to frontend as a color layer | ✅ Present | `/api/uncertainty/{job_id}`, nearest-neighbor UV sampling onto mesh |
| **Uncertainty calibration check** (does high measured std actually correlate with high real error?) | ❌ Missing | You have both an error raster (possible) and an uncertainty raster, but nothing cross-validates one against the other — this is the single most convincing "uncertainty-aware" proof a jury can be shown, and it's one function away |

### 1.4 Scientific validation engine

| Capability | Status | Evidence |
|---|---|---|
| RMSE / MAE / correlation / percentile errors | ✅ Present | `validation/metrics.py` |
| Slope-error metric | ✅ Present (internal only) | `_slope_error()` |
| Signed error raster | ✅ Present | `error_raster()` — **written but never called anywhere in `pipeline.py` or exposed via any route** |
| **Terrain-category stratification (Urban / Hilly / Forested / Sparse)** | ❌ Missing | Explicitly listed in your own README as `PLANNED — blocked on real labeled reference data`. This is the single item named verbatim in the SIH26175 evaluation rubric and it does not exist in any form — not even a stub interface |
| Real-world reference-data validation run | ❌ Not yet done | Only a synthetic ground-truth test exists (honestly labeled as such) |

### 1.5 3D environment / UI

| Capability | Status | Evidence |
|---|---|---|
| Real Three.js BufferGeometry mesh from real elevation data | ✅ Present | `mesh_builder.py` → `mesh_to_json()` → `buildScene()` |
| Hand-rolled orbit/zoom camera, raycasted elevation readout | ✅ Present | `frontend_static/index.html` |
| View modes: Height, Uncertainty, Wireframe, Texture | ✅ Present | 4 of the 4 implemented are genuinely wired to real data |
| **Slope layer** | ❌ Missing | Named explicitly in your prompt's required analysis layers; no button, no shader, no data path |
| **Error-map layer** | ❌ Missing | `error_raster()` exists in Python and is dead code — nothing serializes it to the frontend |
| Legend / colorbar per active layer | ❌ Missing | Color ramps (`applyHeightColoring`, `applyUncertaintyColoring`) exist but the UI never tells the viewer what a given color *means* in meters/degrees/σ — this undercuts scientific credibility more than it looks |
| Per-terrain metrics panel | ❌ Missing (depends on 1.4) | |

### 1.6 Offline-first execution

| Capability | Status | Evidence |
|---|---|---|
| No live external API in the core math path | ✅ Present | Depth model is a locally-loaded checkpoint; calibration and metrics are pure NumPy/SciPy |
| Explicit, non-silent model download step | ✅ Present | `scripts/download_models.py` prints license + size before fetching |
| Fully offline single-image relative-DSM flow | ✅ Present | Works with zero network calls once the checkpoint exists locally |
| **Offline SRTM tile bundling for absolute mode** | ❌ Missing | Absolute mode currently *requires* the user to have their own DEM file already — there's no bundled/offline SRTM cache, so "offline-first absolute elevation" doesn't actually work end-to-end without external data the user must source themselves |
| Graceful degradation on missing checkpoint / bad calibration data | ✅ Present | `FileNotFoundError` with actionable message; `InsufficientCalibrationDataError` triggers documented fallback to relative DSM rather than crashing |

### 1.7 Software engineering hygiene

| Capability | Status |
|---|---|
| Config-driven (no hard-coded magic numbers in processing modules) | ✅ Mostly — but `InferenceConfig.tile_size/overlap` are defined and **unused**, a config-drift smell |
| Structured logging | ✅ `utils/logging_setup.py`, used consistently |
| Dataclass-based, typed internal contracts | ✅ Consistent across all modules |
| Explicit failure handling (no silent fallback to fake data) | ✅ This is the strongest trait of the repo — every fallback path is logged/flagged, never silent |
| Test coverage | ✅ 34 tests across unit/integration/API layers, but **zero tests cover terrain stratification, slope layer, or error-map export** because those features don't exist yet |
| Job durability | ❌ In-memory `dict` job store — documented honestly as demo-only, not production-grade |
| CI pipeline | ❌ No `.github/workflows` — tests exist but nothing runs them automatically |

---

## 2. Root scientific risk (read this before adding features)

Your own `RESULTS.md` reports **r ≈ 0.29** between MiDaS_small's relative
depth and true elevation on a synthetic nadir scene. This is a domain-gap
problem, not a bug: MiDaS was trained on oblique, ground-level photography
where "close to camera" reliably means "physically near." In a nadir
satellite view, that heuristic partially breaks down — a tall building's
*roof* is close to the sensor, but so is a bright rooftop with no height at
all if the network is keying off texture/shading priors instead of true
parallax, which monocular networks always are to some degree.

**Adding a terrain-stratification validator or a slope layer will not fix
this** — they will make the weakness *more visible*, which is correct and
good, but you should walk in prepared to explain it rather than be surprised
by it. Two honest paths forward, in order of effort:

1. **Cheap, defensible**: keep MiDaS_small as-is, but add a **shadow-length
   height cue** as a second, independent signal, fused with the network
   output via a small learned or heuristic weighting. Shadow length is a
   classical, physically grounded satellite height-estimation technique
   (height = shadow_length × tan(sun_elevation_angle)), it requires only sun
   angle metadata (often in image EXIF or scene metadata) and a shadow mask,
   and it doesn't require any labeled training data — it's exactly the kind
   of "semantic prior" your problem statement's second milestone already
   names.
2. **Higher effort, higher payoff**: fine-tune on a public nadir height
   dataset (e.g. DFC2019/2018 Data Fusion Contest DSM+RGB pairs, or IEEE
   GRSS benchmark data) — this is your documented Phase 2 and is the right
   call, but it's weeks not days.

Do (1) before the competition; mention (2) as roadmap. A jury will trust "we
found a real domain-gap, measured it, and added an independent physical cue
to compensate" far more than a suspiciously high accuracy number.

---

## 3. Prioritized roadmap

### P0 — Required to satisfy the stated evaluation rubric

1. **Terrain-stratified validation engine** (Urban/Hilly/Forested/Sparse
   RMSE/MAE/correlation). This is named verbatim in SIH26175's eval
   criteria and currently doesn't exist even as a stub.
2. **Fix the calibration/validation data leakage** — hold out the DEM
   sample points used for calibration from the points used for reported
   metrics. Currently the same reference grid serves both roles.
3. **Expose slope and error-map rasters** end-to-end (they're already
   computed or trivially computable in Python) and add matching Three.js
   view-mode buttons with a legend/colorbar.
4. **Uncertainty-vs-error cross-validation** — when a reference DEM is
   supplied, correlate the measured uncertainty map against the actual
   error raster and report that correlation. This is the single strongest
   "uncertainty-aware" proof you can show a jury, and both rasters already
   exist independently.

### P1 — Competition-grade polish

5. Wire the already-implemented GCP point-list calibration path into the
   API/UI (currently math-only).
6. Shadow-length height-cue fusion (see §2) as a second independent height
   signal, with its own confidence contribution to the uncertainty map.
7. Bundle a small offline SRTM tile cache (a handful of tiles covering your
   demo AOIs) so "offline-first absolute mode" is actually demonstrable
   without the user sourcing their own DEM.
8. Wire `InferenceConfig.tile_size/overlap` into an actual tiling/stitching
   path in `depth/inference.py` for large scenes — right now it's config
   that nothing reads, which is worse than not having the option, because a
   reviewer who reads `configs/default.yaml` will expect it to work.

### P2 — Production hardening (mention in roadmap, defer past the deadline)

9. Replace the in-memory job dict with a SQLite-backed queue (still no
   external services, still offline-first, but durable across restarts).
10. GitHub Actions CI running the existing 34+ tests on every push.
11. Depth-Anything-V2 backbone as a swappable alternative behind the
    existing backbone interface, A/B'd against MiDaS_small's r=0.29 baseline.

---

## 4. What I'll build next

Given the scope, I'd suggest we tackle **P0 items 1–4** first since they
directly answer the stated evaluation rubric and reuse code that already
exists (`error_raster()`, `_slope_error()`) rather than requiring new
infrastructure. Concretely, that means:

- A new `validation/terrain_stratification.py` module: given a reference
  DEM/DSM plus either (a) a supplied land-cover/terrain mask or (b) a
  built-in NDVI/texture-based terrain classifier as a fallback when no
  external mask is available, compute per-class `ValidationMetrics`.
- Extending `pipeline.py` to call this when a reference DEM is present and
  to hold out calibration points from the validation set.
- A `/api/slope/{job_id}` and `/api/error-map/{job_id}` endpoint pair, plus
  two new Three.js view-mode buttons and a colorbar legend component.
- An `uncertainty_error_correlation` field in the metrics response.

Want me to proceed with implementation in that order, or do you want to
reprioritize (e.g. shadow-cue fusion first, since it addresses the
underlying r=0.29 finding rather than just measuring it more granularly)?
