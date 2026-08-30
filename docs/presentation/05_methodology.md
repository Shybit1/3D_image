# 05 — Methodology

Full equations in `docs/science/MATHEMATICAL_METHOD.md`.

## Depth → Height

MiDaS_small outputs inverse relative depth `D`. We normalize to `h ∈
[0,1]` and treat it directly as a height proxy, reasoning that for a
near-overhead sensor, "closer to the camera" ≈ "physically higher."

## Edge-Aware Refinement

An RGB-guided (joint) bilateral filter smooths `h` within visually
homogeneous regions while preserving discontinuities aligned with real RGB
edges — reducing the blurred-building-edge artifact typical of raw
upsampled network output.

## Calibration (only when reference elevation data is available)

`Z = scale·h + offset`, fit via **Theil-Sen** (median of pairwise slopes),
chosen specifically for robustness to the outlier/misregistration noise
common in DEM/GCP reference data. Fit quality (residual RMSE/MAE,
correlation) is always reported — never hidden even when it's bad, which
is exactly what happened in our synthetic ground-truth test (see
`06_results.md`).

## Uncertainty

8 independent forward passes (4 input scales × horizontal-flip on/off) of
the *same* model; per-pixel standard deviation across those passes is the
uncertainty signal. This is a genuine empirical measurement, not a learned
confidence head (the backbone has none) and not a fixed number.

## Validation

Standard MAE/RMSE/Pearson-correlation/percentile-error metrics, computed
only against real reference elevation when supplied. When no reference
exists, the system reports "Not evaluated" rather than omitting the field
or guessing.

## Why This Ordering of Methods

Each stage was chosen to fail *visibly* rather than *silently*: refinement
can report whether it actually ran; calibration reports its own quality;
metrics only ever report on data that was actually supplied. This was a
deliberate response to the scientific-integrity requirements in the
project brief, not an afterthought.
