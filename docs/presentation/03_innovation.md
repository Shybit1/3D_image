# 03 — Innovation

Full detail in `docs/architecture/INNOVATION.md`. Summary for a judge Q&A
setting:

## What's Standard (we didn't invent this)

- The MiDaS_small pretrained depth model itself.
- Grid-mesh triangulation of a heightfield — a textbook technique.
- Theil-Sen robust regression — a well-known 1968 statistic.

## What Required Real Engineering (ours, but not "novel research")

1. **Fixing two genuine breaking bugs** in getting a 2019-era pretrained
   model to load correctly against a current `timm` library version — a
   `torch.hub` network dependency that fails in constrained environments,
   and an architecture-index mismatch caused by API drift in the backbone
   library. Verified fixed via a strict zero-missing-keys check, not
   assumed.
2. **Treating inverse relative depth as a height proxy for nadir views**,
   based on physical reasoning (nearer sensor = taller object, for an
   overhead shot) — then empirically testing that assumption and finding
   (and reporting) exactly where it breaks down.
3. **A calibration module that refuses to fabricate scale/offset** when
   too little reference data exists, and that reports its own fit quality
   (residual RMSE, correlation) rather than presenting every calibration
   as equally trustworthy.
4. **A practical, honestly-labeled uncertainty proxy** (multi-scale/flip
   ensemble) built specifically because the chosen backbone has no native
   variance head — an adaptation to a real constraint.

## What's Genuinely Open for Future Research

- Fine-tuning a depth backbone specifically on labeled nadir satellite
  imagery with real DSM ground truth (Phase 2, not attempted — no labeled
  dataset was reachable in this build's sandbox).
- Whether semantic priors (building/vegetation/water masks) measurably
  improve calibration accuracy — unimplemented, therefore unevaluated,
  therefore not claimed.

We'd rather a judge push back on "is this actually novel?" and hear "no,
and here's exactly which 20% is ours" than discover later that we
overclaimed.
