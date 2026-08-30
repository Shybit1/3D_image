# 08 — Impact

## Who This Helps

- **Disaster response teams** needing a rapid 3D sense of an affected area
  from a single, most-recently-available satellite pass — when a stereo
  pair or LiDAR flight simply isn't available in time.
- **Urban planning and infrastructure monitoring** in regions with sparse
  LiDAR coverage, where a single-image approximate DSM is better than no
  3D information at all, provided its limitations (see `06_results.md`)
  are understood and communicated — which this system does automatically
  via its uncertainty and calibration-warning outputs.
- **Education and public engagement** — a navigable, textured 3D fly-
  through of a place is a far more intuitive artifact than a flat satellite
  photo or a raw elevation raster.

## Where It's Genuinely Useful Today (Baseline State)

- As a fast, honest, first-look 3D visualization tool for a single image,
  with built-in warnings when the underlying calibration is unreliable —
  rather than a black box that always looks confident.
- As open, MIT-licensed, fully-vendored software with no dependency on any
  paid cloud AI API — runs entirely offline once the ~82MB model weights
  are downloaded once.

## Where It Needs More Work Before Being Operationally Trusted

Per the measured r≈0.29 domain-gap finding, this baseline's absolute
elevation numbers should not be trusted for anything safety-critical
(e.g. structural height compliance checks) without either (a) Phase 2
satellite-domain fine-tuning, or (b) a real reference DEM with a large
enough sample of well-distributed calibration points and a reported strong
correlation for that specific scene. The system already surfaces this
distinction automatically via the calibration warning field — the honest
path to impact is trusting the tool exactly as much as its own reported
diagnostics say to, not more.
