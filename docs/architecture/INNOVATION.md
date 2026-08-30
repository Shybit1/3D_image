# Innovation Documentation

Per the master build spec's requirement to separate existing technology
from actual engineering contribution honestly.

## Existing Technology (not ours, used as-is or via standard technique)

- MiDaS_small depth estimation model and weights (Intel ISL, MIT license).
- EfficientNet-Lite3 backbone architecture (`timm`).
- Standard grid-mesh triangulation for heightfields (a textbook technique,
  not novel).
- Theil–Sen robust regression (a well-established 1968 statistical method,
  used here, not invented here).
- Three.js for WebGL rendering.
- FastAPI/Pydantic for the REST layer.

## Our Engineering Contribution (real, but ordinary engineering — not claimed as research novelty)

- Fixing two genuine compatibility breaks between the released MiDaS
  checkpoint and current `timm`'s module structure (documented in
  `MODEL_SELECTION.md`), verified via a strict zero-missing-keys check.
- Designing the pipeline so the depth backbone is fully swappable without
  touching DSM/calibration/uncertainty/mesh/API code.
- The specific choice to treat MiDaS's inverse-relative-depth output as a
  direct height proxy for nadir imagery (rather than naively treating it
  as ground-plane depth), based on the physical reasoning that "nearer the
  sensor" == "taller" in an overhead view — and then empirically
  confirming and characterizing where that assumption breaks down (the
  gradient artifact).
- Building a calibration module that refuses to fabricate an absolute
  scale when insufficient reference data exists, rather than defaulting
  to a guessed constant — this is a deliberate scientific-integrity
  design choice, not a default library behavior.
- The multi-scale/flip test-time ensemble as a practical, honestly-labeled
  uncertainty proxy, chosen specifically because the selected backbone has
  no native variance/dropout head — an adaptation to a real constraint,
  not a novel uncertainty-quantification method.
- End-to-end wiring: a working FastAPI job orchestration layer and a
  from-scratch Three.js viewer (no scaffolding template used) with real
  raycasted elevation readout, height/uncertainty/wireframe/texture view
  modes driven entirely by real backend data.

## Potential Research Contribution (PLANNED, not yet attempted)

- Fine-tuning a depth backbone specifically on nadir satellite imagery
  with DSM ground truth (Phase 2 in the roadmap) — this is where genuine
  novel-ish contribution could occur (a satellite-domain-adapted monocular
  height estimator), but it requires a labeled dataset and compute this
  build did not have access to.
- Investigating whether semantic segmentation priors (building/vegetation/
  water masks) measurably improve calibration or edge preservation versus
  the current RGB-guided bilateral filter — currently unimplemented and
  therefore unevaluated, not claimed as working.

## Explicit Non-Claims

- We do not claim LiDAR-level accuracy.
- We do not claim our depth model is novel — it is a pretrained
  third-party model used as intended by its authors and by the SIH
  problem statement itself (which explicitly expects use of an existing
  pretrained backbone).
- We do not claim the uncertainty map is a calibrated probability — it is
  a relative, empirically-measured disagreement signal across an ensemble,
  useful for flagging likely-unreliable regions, not a formal confidence
  interval.
