# Spatial detail allocation: a complexity map for the detail envelope and gated eta

**Date:** 2026-10-16  
**Status:** **Hypothesis**. No experiment has been run. Nothing here is a KreaPhoton default.  
**Trigger:** BudgetPix (arXiv:2610.12307) — compute-adaptive tokenization for pixel-space diffusion.  
**Related backlog:** R2 (dark smooth-region blotches) and R10 (detail vs. acutance) in
[Krea 2 Turbo landscape](krea2-turbo-landscape-2026-10.md).

## Source and what it does *not* give us

BudgetPix splits the image with an entropy-guided quadtree. Flat regions merge into coarse
tokens, so the backbone runs on a shorter sequence. A small token-local refiner then restores
detail inside each merged token. The authors report MiniT2I-L (512²) and PixelDiT (1024²)
fidelity at ~25% of the compute (**Paper result**: pixel-space models, ImageNet and small T2I).

The method is not usable here as-is:

- It is defined for **pixel-space** denoisers. Krea 2 is latent (Wan 2.1 VAE, 16 ch, /8), and
  the paper does not discuss latent models.
- It needs **training**: zero-initialized merge and refine modules that are fine-tuned onto the
  base model. It is not a training-free sampler trick.
- The released checkpoints are research-scale. PixelDiT carries a non-commercial license.

What carries over is the principle: **compute and detail pressure should not be uniform across
the frame; allocate them by local visual complexity.**

## Current state in KreaPhoton

Both detail-related controls are **temporal only**. They are one scalar per step, applied to
the whole frame:

- **Detail boost** (`sampling.py`, `_detail_envelope_sigma` / `_detail_envelope`): the model is
  called at `sigma_model = s_cur - a * (s_cur - s_next)`. A positive `a` tells the model the
  state is cleaner than it is, so it removes less noise and keeps more texture. `a` follows the
  sigma envelope `DETAIL_SIGMA = (0.97, 0.80, 0.55)`, the same value for every pixel.
- **Gated eta** (`_gated_eta`): ancestral re-noise is ramped to exactly 0 below `sigma_gate`.
  The M5 finding behind it is that terminal ancestral injection is the dark-blotch driver. The
  gate is again one scalar per step.

There is precedent for pointwise adaptivity. `trajectory.trust_clamp` already limits the x0
extrapolation per pixel by the local 3×3 RMS of the reference.

The community pain list matches a spatial mismatch:

- Blotches, banding and noise concentrate on **dark and smooth** surfaces: flat walls,
  gradients, black fabric, dim skin.
- Detail overshoot shows up on **textured** areas: pores pushed into blemishes, "aged" skin.

One uniform knob cannot be low on the first set and high on the second at the same time.

## Hypothesis

> A soft, frozen complexity map `m(x, y) ∈ [0, 1]` computed from the x0 estimate at the end of
> the structure phase, used to modulate **(A)** ancestral re-noise and **(B)** detail pressure
> spatially, reduces blotches and banding on smooth and dark regions without losing texture on
> high-complexity regions. On the R2 stratum it beats any single uniform setting of the same
> controls.

### Map construction (training-free)

1. Take `denoised` (x0 estimate) at the first model call at or below `composition_end`. This is
   already captured as `x0_anchor` for `refine_mode="auto"`. Layout is settled there and
   texture has not yet been committed.
2. Compute a complexity statistic in latent space. Candidates are local RMS of the
   mid-frequency band (reusing `trajectory._bands` / `local_rms`), gradient energy, or a local
   entropy proxy. Average over channels.
3. Blur, normalize per image (percentile-based, not min/max), and push through a smoothstep, so
   the map has no hard edges. BudgetPix's hard quadtree is deliberately not copied.
4. **Freeze** the map for the remaining steps, so it cannot chase its own injected noise.

### Two variants, cheapest first

- **A — spatial eta (zero extra model calls).** The ancestral noise is already per pixel, so
  scale it pointwise: `eta(x, y) = eta_gated(sigma) * lerp(eta_floor, 1, m(x, y))`. Smooth and
  dark regions get less late stochasticity, which is the M5 blotch mechanism. Textured regions
  keep the current behavior. This variant directly targets R2.
- **B — spatial detail boost (+1 model call per envelope step).** `sigma_model` is a scalar
  timestep input to the DiT, so a per-pixel nudge is not native. Run the nudged and un-nudged
  calls and blend the two x0 estimates: `x0 = lerp(x0_plain, x0_nudged, m)`. The cost is
  bounded because the envelope is nonzero only for sigma in (0.55, 0.97), a few steps. This
  variant targets R10.

## Risks and things that would falsify it

- **Skin is mid-complexity.** Cheeks can read as "flat" in latent statistics and lose pores.
  This is the most likely failure. Inspect skin crops specifically; consider a floor on `m`.
- **Seams.** Texture discontinuity at map transitions shows up as halos or "pasted"
  patches. The map must stay soft. Check it with a seam diagnostic along high `∂m` contours.
- **Dark ≠ flat.** Low latent variance in dark regions may be real detail (dark hair, black
  fabric weave) rather than flatness. A luminance-aware term may be needed.
- **Uniform-setting baseline wins.** If a single tuned `sigma_gate` / `eta` / detail value
  matches the spatial variant on R2 and R10 under blind review, the hypothesis adds complexity
  for nothing. **Record this outcome as a negative result.**
- **Checkpoint and precision dependence.** Calibrate per checkpoint. Quantization already
  shifts trajectories (see the landscape note).

## Minimal experiment

| Item | Plan |
|---|---|
| Prompts | R2 stratum (dark hair, black fabric, dim skin, flat walls, gradients, low light) + R10 stratum (portraits, hair, fabric, still life) + unseen holdout prompts |
| Seeds | ≥ 3 per prompt, matched across arms |
| Arms | baseline · best uniform eta/gate · best uniform detail · A · B · A+B |
| Fixed | checkpoint, precision, resolution, LoRAs, VAE, step grid |
| Outcomes | blind pairwise preference; dark-region chroma variance and banding energy; skin-crop pore detail vs. blemish density (scored separately); seam score at map contours; model calls and wall time |
| Order | A first (free); B only if A leaves a texture deficit |

Debug output: save `m` as an image next to each result, so a failure can be attributed to the
map rather than to the modulation.

## Literature update (2026-10-16)

A follow-up scan is recorded in [the Chinese-lab scan](cn-lab-scan-2026-10.md), section 1. It
changes the design as follows. All points remain **Hypothesis** until measured.

- **Supports variant A.** SANI (2609.18466) gates injected noise per pixel and reports that
  smooth regions are degraded by it. That work uses small DDPM/DDIM models, not a flow DiT.
- **Supports the frozen, post-structure map.** RAS (2502.10389) and SDiT (2601.12283) find the
  complexity regions stable across steps once structure is formed, and both need dense early
  steps.
- **Design amendments:**
  - **Feather or dilate the map, never threshold it.** SDiT measured that 1–2 px dilation beats
    hard regions.
  - **Cap the max/min ratio** of per-pixel eta and nudge. AsynDM (2510.04504) shows residual
    noise and blur when region noise levels diverge too far.
  - **Protect strong edges in variant B.** SAMG (2604.26503) lowers strength at high-energy
    boundaries; this pairs with `trust_clamp`.
  - **Add the per-patch std of the predicted velocity as a second map signal,** as in RAS. It
    comes free from the existing model calls.
- **Counter-evidence.** FreeEnhance (2409.07451) injects *more* noise into flat regions to grow
  detail, in an img2img setting. This sharpens the main risk: the map must not suppress
  stochasticity where flatness means underdeveloped texture (skin) rather than intended
  smoothness (sky).
- **Sibling candidate.** A high-pass-only detail / extrapolation delta (scan item C1) targets the
  same R2 failure with no map at all. Run it as an arm in the same experiment. If it alone
  matches the spatial variants, prefer it.
- **Novelty.** No source found tests per-pixel ancestral eta on a distilled flow-matching DiT, or
  measures halos from soft maps.

## Longer horizon

If pixel-space models (for example Iris-3B, Apache 2.0) become practical, BudgetPix itself
becomes relevant as a speed-up. That would be a separate track that requires training, not
part of this hypothesis.
