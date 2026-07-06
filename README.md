# ComfyUI-KreaPhoton

Photorealism-focused sampling nodes for **Krea 2 Turbo** in ComfyUI. Custom sampler
mathematics built specifically for this model — analytic restart schedules, detail
σ-nudge, gated ancestral stochasticity, σ-window guidance, photo-manifold noise
contraction, and two-axis seed variety — every calibrated constant validated by
pre-registered protocols on real generations (400+ images across the E/M/V research
cycle), not copied from SD/SDXL folklore.

> Krea 2 Turbo is a CFG-distilled rectified-flow DiT on the Wan21 16-channel latent.
> Most classic sampler tricks (naive CFG, SDE ancestral defaults, SD-tuned schedules)
> either do nothing or actively break it. This pack is the result of measuring what
> actually works on this exact model.

---

## Nodes

| Node | Purpose |
|---|---|
| **KreaPhoton Sampler** | All-in-one: seed / preset / variety. Everything else computed from validated presets. |
| **KreaPhoton Sampler (Advanced)** | Same engine, SIGMAS input, every parameter exposed. |
| **KreaPhoton Scheduler** | SIGMAS generator with the restart segment encoded (see warning below). |
| **KreaPhoton Empty Latent** | Photo aspect ratios / megapixel tiers for Krea2 (16-channel latent). |

### KreaPhoton Sampler

Minimum knobs by design. Inputs:

| Input | Type | Notes |
|---|---|---|
| `model` | MODEL | Krea 2 Turbo checkpoint |
| `positive` | CONDITIONING | |
| `latent_image` | LATENT | use KreaPhoton Empty Latent |
| `seed` | INT | drives sampling, restart re-noise, ancestral RNG and variety |
| `preset` | combo | `turbo/fast` (8 steps) / `turbo/balanced` (12, default) / `turbo/quality` (16, euler_2m) / `raw/experimental` (36) |
| `variety` | combo | `off` / `low` / `medium` / `high` — inter-seed decorrelation, see Variety |
| `preview_method` | combo | live per-step preview: `auto` / `latent2rgb` / `taesd` / `none` |
| `negative` (opt) | CONDITIONING | enables the σ-window guidance (see Guidance) |
| `clean_model` (opt) | MODEL | anti-mutation composition split (see clean_model) |
| `vae` (opt) | VAE | connect to get the decoded result as a thumbnail on the node |
| `seed_b` (opt) | INT | second seed for the composition blend (−1 = off) |
| `blend` (opt) | FLOAT | 0 = off; >0 spherically interpolates the composition toward `seed_b` (see Composition blend) |

Output: `LATENT`.

### KreaPhoton Sampler (Advanced)

Adds explicit control over everything the presets compute: `sigmas` (SIGMAS input),
`sampler_order` (euler / euler_2m), detail envelope (`detail_amount/start/end/peak`),
gated ancestral (`eta0`, `sigma_gate`), noise `contraction` (+ `per_channel_contraction`
using measured per-channel manifold stats), guidance (`guidance_mode` off/flat/window,
`flat_cfg`, `delta`, `guidance_lo/hi`), variety axes separately (`variety_a_latent`,
`variety_a_cond`, `variety_end`), `composition_end` for the clean_model split, plus the
same `preview_method` / `negative` / `clean_model` / `vae` / `seed_b` / `blend` as the
simple node. Also exposes `variety_seed` — the variety realization seed decoupled from
the generation seed (−1 = use the generation seed); fix the generation seed and vary this
to explore variety realizations of the same base.

### KreaPhoton Scheduler

`steps`, `alpha` (schedule steepness; default 3.158 = e^1.15, the stock Krea2
`ModelSamplingFlux` shift — verified equivalent to live `calculate_sigmas` within 1e-6),
`restart_frac`, `sigma_r`, `plunge` → SIGMAS.

⚠️ **A restart schedule encodes one ascending σ-jump.** The stock `SamplerCustom` /
k-diffusion samplers do not understand ascending sigmas — feed KreaPhoton SIGMAS only
into KreaPhoton samplers.

### KreaPhoton Empty Latent

Megapixel tiers S (~1.0 MP) / M (~1.4) / L (~1.7, default) / XL (~2.1) × aspects
1:1, 4:3, 3:2 (default), 16:9, 9:16. All dimensions divisible by 16 (VAE /8 × DiT
patch 2×2).

---

## Quick start

```
CheckpointLoader ──► KreaPhoton Sampler ──► VAEDecode ──► SaveImage
CLIPTextEncode ────► (positive)   ▲
KreaPhoton Empty Latent ──────────┘
```

1. Clone into `ComfyUI/custom_nodes/` (see Installation), restart ComfyUI.
2. Wire the graph above, pick a preset, hit Queue.
3. Optionally connect the VAE to the sampler's `vae` input — the finished image shows
   directly on the node; `preview_method=auto` (default) shows the image forming from
   noise every step, so you can cancel early.

---

## What the engine actually does

One custom KSAMPLER loop orchestrated through a `CFGGuider` subclass (the stock
`comfy.samplers.sample()` hardcodes its guider, so the whole prepare/guide/sample chain
is assembled manually, mirroring `SamplerCustomAdvanced`):

- **Analytic restart schedule (M3)** — a re-noise segment (`restart_frac`, `sigma_r`,
  optional terminal `plunge`) encoded directly in the SIGMAS array as an ascending jump;
  the loop performs a proper rectified-flow re-noise at that boundary with a dedicated
  RNG stream. Validated (V2): recovers shadow/fabric texture without stamping artifacts.
- **Detail σ-nudge (M2)** — step-relative shift of the σ the model is evaluated at,
  shaped by a smoothstep envelope (`detail_start/end/peak`); skipped on plunge and
  final steps.
- **Gated-eta ancestral (M5)** — CONST/rectified-flow ancestral stepping (same math as
  comfy's `sample_euler_ancestral_RF`) with `eta(σ) = eta0 · smoothstep(σ; σ_gate, 0.35)`:
  exactly zero near the end (terminal ancestral injection is the dark-blotch driver),
  full strength mid-phase. Validated (V4): `eta0=1.0` gives the *cleanest* shadows.
- **σ-window guidance (M6, V1)** — real CFG only inside σ∈[0.7, 0.9] with `delta=1.25`
  (Δ=1.5 catastrophically hallucinates — validated), implemented as a `CFGGuider`
  subclass overriding `predict_noise` only. Outside the window comfy's cfg=1.0
  optimization skips the uncond forward entirely, so a connected `negative` costs extra
  NFE only inside the window (~15–25% of steps), not 2× the whole run.
- **Manifold noise contraction (V5)** — initial unit noise contracted toward the photo
  manifold (`contraction=0.70`, measured from real photographs through the VAE:
  global σ=0.4666). Cleaner shadows with *more* inter-seed diversity, not less.
  Advanced node can use measured per-channel std/mean instead of the scalar.
- **Two-axis seed variety (M4, V3)** — at a σ-boundary (default 0.90): low-frequency
  FFT band re-composition of the latent (AC-only — Wan21 channels are not zero-mean)
  + rotation of semantic conditioning taps (7–10) of the packed 30720-dim Krea2 cond.
  Mutation-cap validated: no identity/pose/composition breaks at any level.
- **clean_model split (anti-mutation)** — optional: composition phase (σ > 0.85) runs
  on a clean checkpoint, the LoRA identity/detail phase on `model`. ZPhoton-proven
  pattern; unvalidated on Krea2 LoRA stacks yet.

### Previews

- **Live per-step preview** (`preview_method` widget): the node temporarily overrides
  the global ComfyUI preview method for the duration of sampling and restores it
  afterwards — it works regardless of `--preview-method` CLI flags, Manager config or
  frontend settings. `auto` = latent2rgb (instant color projection); `taesd` needs
  `lighttaew2_1` in `models/vae_approx` (falls back to latent2rgb if absent).
- **On-node thumbnail** (`vae` input): one VAE decode at the end, displayed on the node
  like KSampler (Efficient); the LATENT output is unchanged.

### Composition blend (`seed_b` + `blend`)

On Krea 2 Turbo the composition is set by the **initial noise field** — a fresh seed
moves the whole layout. `blend` spherically interpolates (slerp) the initial noise
between `seed` and `seed_b`, giving a coherent, on-manifold composition dial: `0` = pure
`seed`, `1` = pure `seed_b`, in between = a genuinely new intermediate composition the
distillate coheres into a photoreal image (validated — full-frame composition spread at
the midpoint reaches ~96% of a seed change, with no ghosting/artifacts).

This is the composition-variety lever a latent perturbation cannot provide: measured on
this model, the variety knob and other variance-preserving latent edits move **texture**,
not composition — only a change to the whole noise field moves layout. **Caveat:**
because composition and identity are coupled through the noise field here, `blend` moves
identity *with* composition — it is a composition explorer between two seeds, not a
fixed-identity variety control. `blend = 0` (default) is a bit-exact no-op.

---

## Presets (all values validated or honestly labeled)

| | fast | balanced (default) | quality | raw/experimental |
|---|---|---|---|---|
| steps | 8 | 12 | 16 | 36 |
| sampler | euler | euler | euler_2m (AB2) | euler_2m |
| restart `frac` / `σ_r` / plunge | 0.25 / 0.65 / on | 0.25 / 0.65 / on | 0.25 / 0.65 / on | 0.20 / 0.45 / off |
| detail | 0.50 | 0.60 | 0.70 | 0.50 |
| eta0 / σ_gate | 1.0 / 0.10 | 1.0 / 0.10 | 1.0 / 0.10 | 1.0 / 0.10 |
| contraction | 0.70 | 0.70 | 0.70 | 1.00 (uncalibrated) |

`σ_r=0.65`, `plunge`, `eta0=1.0`, `contraction=0.70`, `delta=1.25` are V-protocol
validated on real generations (pre-registered seeds/cells/accept-rules — no p-hacking);
`detail_a` values are design hypotheses from the M2 working range, labeled as such in
`presets.py`. `raw/experimental` targets the non-distilled RAW mode (real CFG 3.5) and
is largely unexplored.

---

## Installation

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Kostik2702/ComfyUI-KreaPhoton.git
```

Restart ComfyUI. No extra Python dependencies — torch and comfy only.

Requirements: ComfyUI (0.2x, tested on 0.26) + a Krea 2 Turbo checkpoint
(Wan21 16-channel latent family).

---

## Known limitations

- **eta0 is auto-disabled whenever a segment split occurs** (variety ≠ off or
  clean_model connected). Ancestral stepping combined with any split produced a
  reproducible speckle/mosaic artifact (isolated to the split round-trip mechanism,
  suspected bf16 error amplification through 1/(1−σ_boundary); under investigation).
  A safety guard silently forces eta0=0 in that combination so a broken image is
  never shipped.
- Variety dose-response is weak/non-monotonic beyond `low` (V3 open finding): the
  off→low jump captures most of the decorrelation; a magnitude re-sweep is planned.
- `raw/experimental` preset: schedule steepness for the dynamic-μ RAW canon is
  uncalibrated.
- Restart SIGMAS are KreaPhoton-only (ascending jump, see Scheduler warning).

## Testing

Plain-assert test suite (`tests/run_tests.py`, no pytest dependency — runs under the
ComfyUI embedded interpreter). Tests assert invariants, not literal preset values:
schedule equivalence to stock `calculate_sigmas` at features-off, restart variance
balance, NFE Σ-batch-dim invariant of the guidance window, bit-exactness of untouched
conditioning taps, `corr(lf′, lf) = √(1−a²)` for the LF re-composition.

```sh
python tests/run_tests.py
```

## License

MIT © 2026 Kostiantyn Hrytsuk
