# Model Selection

## Decision: MiDaS_small (v2.1), Intel ISL, MIT License

## Comparison Table

| Model | License | Weight host | Reachable in this build? | CPU speed (approx, 256px) | Params |
|---|---|---|---|---|---|
| MiDaS_small v2.1 | MIT | GitHub Releases (`isl-org/MiDaS`) | **Yes** | ~0.17s/frame (measured) | ~21M |
| MiDaS DPT-Hybrid | MIT | GitHub Releases | Yes | Untested (heavier) | ~123M |
| DPT-Large | MIT | Hugging Face Hub | **No** (domain not in network allowlist) | N/A | ~343M |
| Depth Anything v2 (small/base/large) | Apache 2.0 / mixed | Hugging Face Hub | **No** | N/A | 25M–335M |
| Satellite-specific mono-depth research models | Varies, often research-only | Varies (often not released) | N/A | N/A | N/A |

## Why MiDaS_small

1. **Actually downloadable.** This is not a minor point: several stronger
   candidates (Depth Anything v2, DPT-Large's standard release) are only
   distributed through Hugging Face Hub, which is outside this sandbox's
   network allowlist (`pypi.org`, `github.com`, `npmjs.com`, etc. — not
   `huggingface.co`). We verified this directly: `torch.hub.load` for the
   MiDaS repo itself also failed (a proxy header bug in this sandbox), but
   a direct `curl` to the GitHub Releases URL succeeded and returned a
   real 85MB checkpoint.
2. **MIT licensed**, safe to vendor and redistribute.
3. **Small enough for real-time CPU inference** in a demo context — no GPU
   is present in this build/deploy environment, and judged demos often run
   on judges' own laptops.
4. **Simple architecture** (EfficientNet-Lite3 encoder + custom decoder),
   easy to fully vendor into the project rather than depending on a live
   `torch.hub` call at runtime (which is itself fragile, per point 1).

## Real Integration Issues Found and Fixed

This section exists because the master build instructions require
documenting actual engineering work, not just a model name.

**Issue 1 — `torch.hub.load` fails with `KeyError('Authorization')`.**
The official MiDaS `hubconf.py` and internal blocks.py call
`torch.hub.load("rwightman/gen-efficientnet-pytorch", ...)` to build the
EfficientNet-Lite3 backbone. In this sandbox, that call fails due to a
proxy/auth header issue in the network egress layer. **Fix:** replaced the
`torch.hub.load` call with `timm.create_model("tf_efficientnet_lite3",
pretrained=False)`, which builds the identical architecture with zero
network calls (the pretrained weights come from the MiDaS checkpoint
itself, not from this backbone-construction step).

**Issue 2 — `timm` API drift breaks state_dict key alignment.** Current
`timm` fuses the stem activation into `bn1` (`BatchNormAct2d`, activation
applied internally) instead of exposing a separate `act1` module, which
is what older `timm` (and the original MiDaS integration code, and the
released checkpoint's key names) assumed. This shifts every subsequent
module's index in the `nn.Sequential`, which changes the state_dict key
strings (e.g. `pretrained.layer1.3...` vs `pretrained.layer1.2...|`),
causing a silent-looking but actually broken weight load. **Fix:** inserted
a parameter-free `nn.Identity()` placeholder in the old `act1` slot to
preserve the original module indices/key names exactly, since `bn1`
already performs the activation internally — functionally identical
output, correct key alignment.

**Verification:** after both fixes, loading the real downloaded checkpoint
into the reconstructed architecture produces **0 missing keys and 0
unexpected keys** (asserted in `tests/integration/test_depth_inference.py`
via the constructor's own strict check, which raises if this is ever not
true).

## Empirical Finding: Domain Gap on Nadir Imagery

Running the loaded, correctly-weighted model on a synthetic nadir-view
test scene produced a depth map dominated by a smooth top-to-bottom
gradient, with the actual synthetic building signal only weakly visible
underneath it. This matches the expected failure mode: MiDaS was trained
on natural egocentric photographs, where "top of frame" reliably
correlates with "far/sky" and "bottom of frame" with "near/ground" — a
strong learned prior that does not hold for a satellite's overhead view.

This is reported here as a measured limitation, not smoothed over. It is
the primary scientific justification for `dsm/engine.py`'s edge-aware
refinement pass and for treating Phase 2 (satellite-domain fine-tuning
infrastructure) as necessary future work rather than a nice-to-have.

## Upgrade Path (PLANNED)

Swapping to Depth Anything v2 or a satellite-specific model requires:
1. A network path to the weight host (Hugging Face Hub or equivalent),
   not available in this sandbox but trivial on an unrestricted machine.
2. Implementing the corresponding architecture in
   `src/depthwizard/depth/backbones/` alongside the existing MiDaS one.
3. Updating `depth/inference.py`'s `DepthEstimator` to select backbone by
   config (`configs/model.yaml`), which the config schema already
   supports (`model.name` field).

No other module (DSM engine, calibration, uncertainty, mesh builder, API,
frontend) depends on which backbone produced the height proxy — this
boundary was designed in from the start specifically to make this swap
low-risk.
