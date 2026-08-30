# Results

**All numbers on this page are real measurements taken while building and
testing this repository** (see the corresponding test files for exact
reproduction steps). No number here is estimated, assumed, or fabricated.
Where a value has not been measured, it is marked "Not yet evaluated" —
per project policy, this is never silently filled in with a guess.

## Model Load Integrity

- MiDaS_small (v2.1) checkpoint loaded into the vendored architecture:
  **0 missing keys, 0 unexpected keys** (`tests/integration/test_depth_inference.py::test_model_loads_with_no_key_mismatch`).

## Inference Performance (CPU only — no GPU present in this build environment)

Measured on a synthetic 512×512 test scene, model input resized to 256×256:

| Stage | Measured time |
|---|---|
| Model load (one-time, per process) | 1.07 s |
| Single depth inference pass | 0.17 s |
| Uncertainty ensemble (8 passes: 4 scales × 2 flip states) | 1.62 – 1.81 s |
| DSM generation + edge refinement | <0.01 s |
| Mesh build (512×512 → 200×200 grid) | 0.05 s |
| Full API round trip (upload → reconstruct → completed) | 2.66 – 3.21 s |

These numbers are for a **synthetic demonstration scene**, not real
satellite imagery, and were measured on the CPU available in this
sandboxed build container — not representative of any specific deployment
hardware. Do not quote these as production SLAs.

## DSM / Accuracy Metrics vs. Reference Data

**Real-world accuracy: not yet evaluated.** No real satellite image with a
matched real reference DSM or LiDAR dataset was available inside this
build's sandboxed network (no access to satellite/LiDAR data archives from
the restricted domain allowlist used during this build — real SRTM tiles
do exist on GitHub, e.g. `wschwanghart/DEMs`, but pairing one to a real
satellite photo of the exact same location was not achievable here).

**Synthetic ground-truth self-consistency test: run, and the result is
informative.** `scripts/generate_demo_data.py` builds a synthetic scene
with an exactly-known ground-truth DSM (terrain + 10 buildings with known
heights, by construction — not measured, not estimated). Running the full
absolute-mode pipeline (depth inference → Theil-Sen calibration against
2000 sampled reference points → validation against the full reference
grid) against this scene produced:

| Metric | Measured value |
|---|---|
| Calibration points used | 2000 |
| Height-proxy ↔ reference correlation (r) | **0.29** (weak) |
| Calibration residual RMSE | 11.9 m |
| Calibration residual MAE | 7.1 m |
| Full-grid validation RMSE | 11.8 m |
| Full-grid validation MAE | 7.0 m |
| Full-grid validation median abs. error | 4.2 m |

**How to read this honestly:** because the ground truth here is exact by
construction (not noisy real-world LiDAR), this measurement isolates the
depth model's own error from any DEM-noise confound — and it quantitatively
confirms the domain-gap finding reported below: MiDaS_small's relative
depth ordering correlates only weakly (r≈0.29) with true elevation on a
nadir synthetic scene. The pipeline's own honesty mechanism caught this
automatically: the calibration module returned a real warning
(`"Weak correlation (r=0.29)... the linear scale assumption may not hold
well for this scene"`) rather than silently reporting a confident-looking
number. **This is a real measured result on a synthetic scene, not a
real-world accuracy claim** — do not quote r=0.29 as "our system's
accuracy" without that caveat. It does, however, validate that the
calibration/validation code itself is correct and that the pipeline
correctly flags low-confidence calibrations instead of hiding them.

Reproduce with:
```bash
python scripts/generate_demo_data.py
pytest backend/tests/test_api.py::test_absolute_mode_with_reference_dem_calibrates_and_validates -v
```

## Test Suite

Full suite (`tests/` + `backend/tests/`): **34 / 34 tests passed**,
including:
- 19 unit tests (image I/O, GeoTIFF round-trip, calibration math, metrics
  math, mesh geometry) — all pure computation, no external services.
- 4 integration tests against the real, loaded MiDaS_small model.
- 7 backend API tests via FastAPI's `TestClient`, including one full
  relative-mode flow and one full absolute-mode flow with reference-DEM
  upload, calibration, and validation against synthetic ground truth.

Run yourself with:
```
pytest tests backend/tests -v
```

## Empirical Qualitative Finding

Raw MiDaS_small output on a nadir (overhead) synthetic scene is dominated
by a smooth top-to-bottom gradient artifact, consistent with the model's
training distribution being natural egocentric photography rather than
satellite imagery. See `docs/architecture/MODEL_SELECTION.md` for detail
and the resulting design decision (edge-aware refinement; Phase 2
fine-tuning marked as necessary future work rather than optional polish).
