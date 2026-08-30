# 10 — Anticipated Judge Questions

**Q: What's your actual DSM accuracy against real LiDAR?**
A: Not yet measured — no real satellite+LiDAR paired dataset was reachable
during this build. What we *have* measured is a synthetic ground-truth
test that isolates the depth model's own error (r≈0.29, RMSE≈11.8m) — see
`06_results.md`. We'd rather tell you exactly that than give you a number
we can't back up.

**Q: Isn't r=0.29 a bad result? Why show it?**
A: Yes, it's weak, and it's the single most useful finding from this
build: it quantifies the domain gap between a natural-photo-trained depth
model and nadir satellite imagery, which is exactly the problem the SIH
statement flags as the hard part. It also proves our calibration module's
safety mechanism works — it detected the weak fit and reported a warning
instead of a confident-looking wrong number.

**Q: Is MiDaS your own model?**
A: No — it's Intel ISL's pretrained MiDaS_small, MIT licensed, used as-is.
The SIH problem statement explicitly expects use of an existing pretrained
depth backbone rather than a from-scratch network. Our contribution is
everything around it: the pipeline, the calibration safety logic, the
uncertainty estimation, the mesh/viewer, and the domain-gap analysis.

**Q: Why not use a stronger model like Depth Anything v2?**
A: We evaluated it and preferred it in principle, but its released weights
are hosted on Hugging Face Hub, which was unreachable from this specific
build sandbox's network policy. MiDaS_small's weights are hosted on GitHub
Releases, which was reachable — verified by directly downloading and
loading them with zero key mismatches. The backbone is swappable by design
(`depth/inference.py`); this is a sandbox constraint, not an architectural
limitation.

**Q: How do you handle non-georeferenced images?**
A: They produce a relative DSM only — no absolute elevation is fabricated.
This is enforced in code, not just documentation: `calibration/
scale_calibration.py` raises an explicit error rather than guessing when
insufficient reference data exists, and the pipeline catches that and
falls back cleanly.

**Q: What happens with a very large real satellite image?**
A: Currently capped at 16,000px per side with a clear validation error
above that. Tiling/stitching for larger images is planned but not
implemented — the config already has reserved fields for it
(`inference.tile_size`, `inference.overlap`).

**Q: Is this production-ready?**
A: No, and we say so directly in the README's "What's Implemented vs.
Planned" section. It's a working, tested baseline (34/34 tests passing,
full pipeline verified end-to-end) with clearly scoped follow-on work: a
real accuracy dataset, satellite-domain fine-tuning, semantic
segmentation, and a production-grade job queue.

**Q: Can I run this myself right now?**
A: Yes — `pip install -r requirements.txt`, `python scripts/
download_models.py`, `uvicorn app.main:app`, open the browser. No API
keys, no cloud dependency, fully offline after the one-time model
download.
