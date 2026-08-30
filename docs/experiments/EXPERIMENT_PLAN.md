# Experiment Plan

## Baseline Validation (COMPLETED — see RESULTS.md)

Goal: confirm the pipeline runs correctly end-to-end and produces
internally-consistent outputs (not accuracy against ground truth, which
requires reference data not available in this build).

- Model load integrity check (0 missing / 0 unexpected state_dict keys).
- Inference output sanity (non-constant, bounded [0,1], correct shape).
- Metrics module self-consistency (identical arrays -> exactly zero error).
- Full API integration flow (upload -> reconstruct -> mesh/DSM/texture
  retrieval).

## Real-World Accuracy Evaluation (PLANNED — not yet run)

Requires: a real satellite image with a matched reference DSM or LiDAR
elevation dataset, covering at minimum one urban scene. None was available
inside this build's network-restricted sandbox (no access to satellite
imagery archives or LiDAR repositories from the allowed domain list).

Plan once data is available:
1. Acquire a paired (satellite RGB, reference DSM) sample, e.g. from an
   open dataset such as an IARPA/SpaceNet-style multi-view stereo release
   with ground truth, or a national open LiDAR/DSM portal.
2. Run the full pipeline in `absolute` mode using the reference DSM as the
   calibration source (via `calibration.sample_dem_at_pixels`).
3. Compute `validation.metrics.compute_metrics` against a held-out subset
   of the same reference DSM (not the calibration points themselves, to
   avoid leaking ground truth into the reported accuracy).
4. Repeat across landscape categories (urban, sparse, hilly, forested) per
   the SIH26175 evaluation criteria, if multiple scenes are available.

## Ablation Plan (PLANNED)

- With vs. without edge-aware refinement (`dsm.engine.refine_edges`):
  measure RMSE/MAE difference against reference DSM.
- Effect of ensemble size (`K` in uncertainty estimation) on
  uncertainty-map stability (would need a fixed scene, varying K, checking
  map-to-map correlation as K increases).
- MiDaS_small vs. a larger MiDaS variant (DPT-Hybrid) on the same scene,
  if/when the larger checkpoint is downloaded — CPU inference time vs.
  accuracy tradeoff.

None of the above ablations have been run yet; do not report ablation
numbers until they have.
