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

Current version **1.6.1** — ten nodes, one engine (`kreaphoton/sampling.py`), MIT.
Everything that is measured is labeled measured; everything that is a design hypothesis
is labeled as such, in the README and in `presets.py`.

---

## Nodes

| Node | Purpose |
|---|---|
| **KreaPhoton Sampler** | All-in-one: seed / preset / variety (+ `denoise` for refine, `coherence`, phase models). Everything else computed from validated presets. |
| **KreaPhoton Sampler (Advanced)** | Same engine, SIGMAS input, every parameter exposed (+ `x0_extrapolation`, `variety_seed`, `pag_*`, `guidance_rescale`). |
| **KreaPhoton Scheduler** | SIGMAS generator with the restart segment encoded (see warning below); optional `self_refine` second descent. |
| **KreaPhoton Empty Latent** | Photo aspect ratios / megapixel tiers for Krea2 (16-channel latent). |
| **KreaPhoton Encode** | krea2-native text encode (plain `KREA2_TEMPLATE` path) with an optional calibrated style directive. |
| **KreaPhoton LoRA Phase** | Records a LoRA + strength + phase (composition / identity / texture / …) in a plan the model carries; the KreaPhoton samplers, upscalers and the Face Detailer expand the plan into phase models with comfy's ordinary LoRA patching (works on int8 / fp8 checkpoints). Chain one per LoRA. |
| **KreaPhoton Save Image** | Save with folder picker, timestamp+counter unique names, PNG/JPEG/WebP metadata. Local-only power-user feature (the folder browser has no path allowlist - do not expose a `--listen` server). |
| **KreaPhoton Upscale** | Faithful tiled ×1.25–×2.0 upscale with the same Krea 2 model: latent tiles re-blended every step (no seams), back-projection to the source (the downscaled result *is* the source), presets only, optional SR-model base. |
| **KreaPhoton Upscale v2** | The same faithful refine plus empty-tile skipping, per-step grid shift and optional noise inversion; its own preset calibration (`polish` / `detail` / `strong`), shipped separately so v1 stays as calibrated. |
| **KreaPhoton Face Detailer** | YOLO face detection → the `max_faces` largest faces redrawn at 1024 → 1536 px with the same sampler on the LoRA plan's texture model (identity LoRAs boosted ×1.5), ArcFace identity gate with keep-best retry, feathered paste. Presets `subtle` / `standard` / `strong`. |

Typical pipeline: `Empty Latent` + `Encode` → `LoRA Phase` (one per LoRA) → `Sampler` →
`VAEDecode` → `Upscale v2` → `Face Detailer` → `Save Image`. Every IMAGE-stage node takes
the same `model` (with its plan) and `positive`.

### KreaPhoton Sampler

Minimum knobs by design. Inputs:

| Input | Type | Notes |
|---|---|---|
| `model` | MODEL | Krea 2 Turbo checkpoint |
| `positive` | CONDITIONING | |
| `latent_image` | LATENT | use KreaPhoton Empty Latent |
| `seed` | INT | drives sampling, restart re-noise, ancestral RNG and variety |
| `preset` | combo | `turbo/fast` (8 steps) / `turbo/balanced` (12, default) / `turbo/quality` (16, euler_2m) / `turbo/candid` (16, euler_2m, texture push dialled back — documentary / lifestyle, "as the eye sees it") / `raw/experimental` (36) |
| `variety` | combo | `off` / `low` / `medium` / `high` — inter-seed decorrelation, see Variety |
| `preview_method` | combo | live per-step preview: `auto` / `latent2rgb` / `taesd` / `none` |
| `denoise` | FLOAT | 1.0 = txt2img (default). Below 1.0 = refine / img2img of the connected latent on a clean partial descent: 0.2–0.4 polish, 0.5–0.7 enhance + vary (restart/plunge/blend/eta are txt2img-only and skipped) |
| `negative` (opt) | CONDITIONING | Turbo presets: enables the σ-window guidance (see Guidance). `raw/experimental`: real full-trajectory CFG 3.5 always, with a zeroed unconditional when nothing is connected |
| `clean_model` (opt) | MODEL | anti-mutation composition split (see clean_model) |
| `vae` (opt) | VAE | connect to get the decoded result as a thumbnail on the node |
| `seed_b` (opt) | INT | second seed for the composition blend (−1 = off) |
| `blend` (opt) | FLOAT | 0 = off; >0 spherically interpolates the composition toward `seed_b` (see Composition blend) |
| `restart_enhance` (opt) | combo | `off` (default) / `dlss5 default` / `dlss5 natural` / `dlss5 cinematic` — DLSS 5 Photoreal Enhance at the restart boundary; experimental, needs `vae` and the `ComfyUI-dlss-enhancer` pack (see v1.4 notes) |
| `texture_model` (opt) | MODEL | model for the texture phase (the restart segment, σ ≤ 0.65) — see Phase models |
| `coherence` (opt) | combo | `off` (default) / `jump` / `self_refine` / `jump+self_refine` — see Coherence tools |

Output: `LATENT`. A model carrying a LoRA Phase plan is expanded into phase models
automatically; explicitly connected `clean_model` / `texture_model` override the plan for
their phase.

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
to explore variety realizations of the same base — and `x0_extrapolation` (v1.3): a
terminal x0-trajectory extrapolation weight (0 = off). At the last step the model's x0
estimate is continued linearly past the last evaluation toward σ=0
(`denoised + w·f·(denoised − previous denoised)`, `f = σ_last/(σ_prev − σ_last)`, capped
at 2; exactly 1.0 on the calibrated restart tail; skipped on plunge steps). Same mechanism
as the third-party *Krea 2 Turbo Preset Sampler*'s `zero_extrapolation`; sharper
micro-detail / local contrast, uncalibrated on KreaPhoton grids — try 0.3–0.5.

Every SIGMAS array is validated before any model is touched (finite, 1-D, values in
[0, 1], at most one ascending jump); a malformed or foreign schedule fails with a message
naming the offending indices instead of integrating a meaningless trajectory.

v1.4 additions on the Advanced node (all off by default, experimental): `pag_scale` /
`pag_lo` / `pag_hi` / `pag_blocks` — perturbed-attention guidance, a second conditional
forward with identity self-attention in the chosen DiT blocks inside a σ-window,
`denoised += scale·(cond − perturbed)`; +1 model call per step in the window. Measured on
Krea 2 Turbo (merge checkpoint, 5 scenes): scale ≥ 1.0 produces sparkle artifacts,
saturation and face distortion, ≤ 0.5 is benign but showed no anatomy benefit — keep it
at 0 unless you are experimenting. `restart_enhance` (both samplers, needs `vae`) — at the
restart boundary the plunge x0 is decoded, run through NVIDIA DLSS 5 Photoreal Enhance V2
(the `ComfyUI-dlss-enhancer` pack must be installed), re-encoded and re-noised into the
texture phase. Measured: the restart steps re-synthesise texture and the DLSS pass is
practically invisible in the result — run DLSS after decoding instead.

Resolution-aware shift (v1.4): Krea 2's canonical schedule shift is linear in the image
token count (μ 0.5 at 256 tokens → 1.15 at 6400); ComfyUI pins 1.15. The simple sampler
now uses the canonical α for the connected latent — the L/XL tiers (≥ 6400 tokens) are
unchanged, a 1024² grid gets α = 2.47 instead of 3.16. The Scheduler does the same when
its optional `latent` input is connected.

### Coherence tools (`coherence` on the simple node)

Two Power-Nodes-derived mechanisms, both pure sampling (no detector, no second model):

- **`jump`** — a one-time jump-back on the first step at/below σ 0.93: the state is
  rescaled by (1−σ_decl)/(1−σ) and the model is told σ_decl = σ + 0.19·(1−σ). That is
  what comfy's noise-scaling round-trip does to Power Nodes' 0.920 → 0.935 stage boundary,
  made explicit: the model sees the correct signal-to-noise ratio at 20 % less amplitude
  than the declared σ implies and commits harder to structure on the composition step.
  Free (no extra model call). Advanced: `coherence_jump` / `coherence_jump_sigma`.
- **`self_refine`** — after the plunge draft, re-noise to σ 0.85 and descend 4 extra
  steps to the plunge floor, plunge again, then the usual restart. The model re-decides
  faces, bodies and clothing with the whole draft as a prior (0.85 ≈ img2img 0.6: layout
  kept, incoherent detail re-rendered). +4 model calls. Scheduler: `refine_steps` /
  `refine_sigma` (a second ascending jump in the SIGMAS; the validator accepts two).
- `jump+self_refine` — both. `off` is bit-identical to v1.3.

Measured (23 frames, int8 checkpoint + LoRA rig, one seed per cell): `self_refine` kept the
layout in every scene and visibly repaired hands (potter's clay-mush hand → distinct fingers;
a raised hand in a two-person LoRA scene), identity and wardrobe preserved, no artifacts, about
+25 % time. `jump` was subtle and harmless. Recommendation: `self_refine` for LoRA stacks,
groups and hand-heavy scenes; leave `off` when the plain result is already fine.

### KreaPhoton LoRA Phase

Sits on the MODEL line like a LoRA loader: `model` → LoRA Phase (character LoRA,
`identity`) → LoRA Phase (style LoRA, `texture`) → KreaPhoton sampler. It patches
nothing itself; it records the LoRA, strength and phase in a plan the model carries, and
the sampler expands the plan into phase models with comfy's ordinary LoRA patching — so
it works on every checkpoint, including int8 / fp8-quantized ones (comfy weight hooks
cannot patch those; that path was dropped). `phase`: `composition` (σ 1.0→0.85, layout /
framing / pose), `identity` (σ 0.85→0, face / body / clothing — includes the texture
phase), `texture` (σ 0.65→0, the restart segment: skin, fabric, grain),
`composition+identity`, `all` (classic loader). Character LoRA → `identity` keeps the face
without letting the LoRA steer the layout; style LoRA → `texture` adds its look without
fighting the prompt's composition. Model-only: the text encoder is not patched. Other
samplers see the unpatched model. Identical LoRA sets share one patcher, so a plan with a
single identity-phase LoRA costs one extra re-patch, not two.

### Phase models (`clean_model` / `texture_model`)

The same mechanism, wired by hand: both samplers accept up to three models, one per
schedule phase — `clean_model` for the composition phase (σ > 0.85), `model` for the
identity phase, `texture_model` for the texture phase (the restart segment, σ ≤ 0.65;
`texture_start` on the Advanced node). Build them with ordinary LoRA loaders; an
explicitly connected phase model overrides the plan for that phase. Each phase switch
costs one LoRA re-patch (a few seconds on a 12B model). The segments share one global
detail-envelope index and one guider setup, so the split integrates the exact sigma
sequence of a single run.

**A masked run never splits.** If the latent carries a `noise_mask` (inpaint, the Face
Detailer crop pass) and the plan or the inputs would split the schedule, the whole schedule
runs in one lifecycle on the model of the phase that covers most of it (a console line
says which). Reason, measured 2026-09-15: comfy's `KSamplerX0Inpaint` re-noises the
unmasked band from the *source* latent with the *original* noise at every step and writes
the source back at the end; a second segment hands it the mid-trajectory state and zero
noise, and the feather band decodes to coloured speckle (the "confetti ring").

Per-phase prompts (the Power-Nodes `positive_stg2/stg3` idea) need no extra input: build
them with the stock `ConditioningSetTimestepRange` + `ConditioningCombine` nodes — the
guider honours conditioning timestep ranges like any ComfyUI sampler. Cover the whole
range, or a step with no active positive falls back to comfy's zeroed conditioning.

### KreaPhoton Upscale

Faithful tiled ×1.25–×2.0 upscale with the same Krea 2 Turbo checkpoint: a light
diffusion refine adds real micro-texture and edge definition, and a source-consistency
step guarantees that the result reproduces the source *exactly* when scaled back down —
layout, tone, faces and objects cannot change. Calibrated on existing frames (21 in two surveys: hands, freckled faces, groups of three
to five people, coins, phones, cards, watches, hair, screen text, JPEG sources, night
scenes; research design doc 07 §11 — the numbered research docs are not shipped with the
pack, their conclusions are in this README and in `presets.py` comments), not on prompts.

| Input | Type | Notes |
|---|---|---|
| `model` | MODEL | Krea 2 Turbo; a LoRA Phase plan is honoured (identity / texture LoRAs apply, composition ones never do) |
| `positive` | CONDITIONING | the frame's prompt, shared by every tile (measured: an empty prompt gives the same result — the source drives everything) |
| `image` | IMAGE | a batch is processed one image at a time (seed + index) |
| `vae` | VAE | the krea2 Wan 2.1 VAE |
| `seed` | INT | refine noise |
| `preset` | combo | `polish` (denoise 0.06, 4 steps: clean, slightly crisper) / `detail` (0.12, 6 — default: natural skin / fabric / hair micro-texture, sharper edges) / `strong` (0.25, 8: more model texture and grain; source artefacts get emphasised too) |
| `scale` | FLOAT 1.25–2.0 | pixel upscale before the refine (1.5 and 2.0 calibrated). Target dims are rounded to a multiple of 16 px. No 1.0: under the source-consistency guarantee a same-size run is either a no-op or a rewrite (measured) |
| `negative` (opt) | CONDITIONING | σ-window guidance like the Sampler, but every preset starts far below the window (σ 0.14–0.45) — no effect on Turbo today |
| `upscale_model` (opt) | UPSCALE_MODEL | pixel base instead of Lanczos (its own factor, then resized to `scale`). Measured with 4xNomos8kSCHAT-L and 4xNomos8kDAT: about twice the edge sharpness of the Lanczos base (Laplacian 3.2–3.8× vs 1.9× Lanczos-relative), the SR models' waxy / painted look is replaced by the diffusion's natural skin, but a source artefact such as crosshatch skin gets sharper too; +55–75 s per 1.7 MP frame on an RTX 5090 |
| `tune` (opt) | STRING | calibration override (JSON), leave empty — see the tooltip |

Output: `IMAGE`.

How it works: the image is upscaled (Lanczos or the model), encoded once with the tiled
VAE, and the ordinary KreaPhoton refine schedule runs on the **whole** latent. Every
model call cuts the state into overlapping 1024 px tiles (128 px feathered overlap),
runs them in batches of 4 with the shared prompt and merges the x0 predictions — the
tiles are re-blended at every step, so seams cannot form (MultiDiffusion /
Mixture-of-Diffusers idea, done inside the sampler loop). One tiled decode, then
iterated back-projection: `result += Up(source − Down(result))`, five passes with an
antialiased `Down` (a plain area/box downscale bins unevenly at ×1.5 and left a fine grid on
skin — measured on the second survey), so the downscaled result matches the source to
within the PNG rounding. Only the detail the model added
below the source scale survives. Gated-eta, the detail nudge, restart / plunge / blend /
coherence are all off (each was measured to push texture or rewrite content here).

What the calibration found (2026-09-13, native-resolution crops on seven V17 frames):
any denoise of 0.20 or more — on tiles or on one whole-image tile, with any prompt,
sampler order, contraction or nudge — rewrote the picture: clay grain became a smooth
glove with a fine mesh texture and the tone drifted several levels. The latent
low-frequency anchor of the original design helped only partially (kept in `tiling.py`,
reachable through `tune`, off in every preset). Back-projection fixed it outright, and
with it the useful denoise band is 0.04–0.25. Consequence, stated plainly: **this node
sharpens and re-textures, it does not repair geometry** — a deformed watch stays a
deformed watch, because the denoise that could re-draw it rewrites the whole frame. A
"repair" mode that locked the source on a coarser grid (2 source px) was measured worse
(smeared edges, amplified crosshatch, waxy freckles) and is not offered.

Cost on an RTX 5090, ×2 of a 1088×1600 frame: `polish` ≈ 17–22 s, `detail` ≈ 21–29 s,
`strong` ≈ 27–38 s (models already loaded); add the SR model's time when one is
connected. VRAM: the batch of four 1024² tiles fits with room to spare.

### KreaPhoton Upscale v2

The same faithful refine as v1 with three mechanisms added and measured on the same frames
(research design doc 07 §12), shipped as a separate node so v1 stays as calibrated:

- **Empty-tile skipping** — tiles whose source carries no detail (bokeh, night sky, plain
  walls; absolute luma-Laplacian activity below a threshold at the tile's 90th percentile)
  get no model call and take the source latent instead. Measured identical output with 10–25 %
  fewer tile forwards on the calibration frames; the log line on the console reports the count.
- **Per-step grid shift** — the interior tiles move by a seeded random offset inside the
  grid's slack at every model call (SpotDiffusion idea), so no blend band sits at a fixed
  position; overlap 64 px instead of 128, same tile count.
- **Noise inversion** (off by default, `tune` `{"invert_steps": n_steps-1}`) — the source
  latent is first walked *up* the rectified flow on the descent grid so the descent starts
  from the source's own noise instead of a fresh draw. Measured: +0.2–0.3 dB and no visible
  difference at twice the model calls, because the back-projection already pins the result
  to the source. Kept as an option, not a default.

Presets `polish` 0.10 / 6, `detail` 0.20 / 8, `strong` 0.35 / 10 (contraction 1.0). Same
inputs and `tune` override as v1 plus `invert_steps`, `skip`, `shift`, `start_noise`.

### KreaPhoton Face Detailer

IMAGE → IMAGE, after the sampler or the upscaler. Finds faces (ultralytics YOLO
`face_yolov8m.pt`), redraws the `max_faces` largest ones at guide resolution with the
KreaPhoton sampler, measures the result's identity against the original face (ArcFace),
retries when it drifted, and pastes the winner back through a feathered ellipse.

Inputs: `model` (may carry a LoRA Phase plan), `positive`, `image`, `vae`, `seed`, `preset`,
`max_faces` (1–8, largest bbox first; faces under 48 px are skipped), `identity_boost`
(default 1.5 — strength multiplier for the plan's identity LoRAs in the face pass, see
**LoRA likeness**). Optional: `negative` (nothing at cfg 1), `face_positive` — a conditioning
for the crop only (measured 2026-09-16: a short "trigger, close-up portrait, natural skin
texture" prompt scored *lower* on likeness than the scene `positive`, 0.60 vs 0.66 — leave it
unconnected unless the scene prompt says nothing about the face; the report notes when it is
connected), `reference_image` — photo(s) of the character, a batch of several is best: identity
is measured against the mean of their embeddings, retries explore seeds at the pass denoise,
and the original face is kept if the redraw loses likeness, `upscale_model` — SR model to
enlarge the crop (else lanczos), `tune` — JSON overrides. Outputs: `image`, `mask` (union of
the pasted faces, for chaining), `report` (per face: bbox, every attempt's denoise / seed /
id_sim, the verdict; also printed to the console).

Per face, per pass: square crop of `2.0 ×` the bbox (aligned to 16 px) → resized so its long
side is the pass's guide → VAE encode → `refine_schedule(n_steps, denoise)` with an elliptic
`noise_mask` (bbox dilated 10 %, 6 % feather) → decode → ArcFace cosine to the reference.
Below `id_threshold` the pass is retried (up to 3 attempts) — without a reference with
`denoise − 0.05` and `seed + 1000` (the original crop is the target, a retry moves toward it),
with a reference at the same denoise and `seed + 1000` (the character is the target, a retry
explores); the attempt with the highest similarity wins, never a worse one. Pass 2 starts from
the pass-1 winner at its full guide resolution, so the detail built at 1024 feeds 1536; only the
final winner is resized down to the crop and pasted.

**LoRA likeness.** Every preset pass starts below σ 0.66 (`refine_schedule` at denoise ≤ 0.45),
i.e. inside the plan's texture segment: the crop runs on the plan's texture set exactly as
Upscale v2 does — a character LoRA in `identity` or `all`, and a style LoRA in `texture`, act on
the face; a `composition`-only LoRA does not — with the identity LoRAs (`all` / `identity`) at
`strength × identity_boost`. Measured 2026-09-16 on a real graph (AnnaMin_v3 at 1.0 / `all`,
ArcFace cosine of the detailed face to the centroid of the LoRA's 120 training faces; a real
photo of the character scores ~0.79 there, the base generation 0.61, the detailer's input after
Upscale v2 0.56): at ×1.0 **every** redraw lost likeness (0.47–0.49) whatever the LoRA set
(identity set, texture set, all six LoRAs), denoise (0.35–0.6) or prompt; at ×1.3 it broke even
(0.60), at ×1.5 it gained (0.62–0.66 over four seeds), at ×1.8 it plateaued (0.645). On a large
crop the face has far more authority than in the full frame, and at the nominal strength the
base model's face prior wins the redraw. Without a plan (classic LoRA loader) nothing is
boosted and the report says so. The measurement rig: `mb search "Face Detailer identity boost"`.

Presets (`guide / denoise / steps` per pass, then `id_threshold`):

| preset | pass 1 | pass 2 | id_threshold | use |
|---|---|---|---|---|
| `subtle` | 1024 / 0.25 / 6 | — | 0.70 | portraits, LoRA identity first |
| `standard` | 1024 / 0.35 / 6 | 1536 / 0.15 / 6 | 0.65 | default; skin texture on pass 2 |
| `strong` | 1024 / 0.45 / 7 | 1536 / 0.20 / 6 | 0.60 | small faces on full-body frames |

Steps follow the effective-step rule of the Impact Pack measurement on krea2 (18 × 0.35 ≈ 6
effective ↔ our 6 steps at denoise 0.35 — `refine_schedule` oversamples the descent at
`n_steps / denoise`). **Honestly labeled: the numbers are derived from that Impact Pack
measurement (2026-09-01) and are not yet validated on the KreaPhoton sampler.**

`tune` keys: `crop_factor`, `bbox_threshold` (0.45; 0.6 details the subject only),
`min_face_px`, `feather`, `dilation`, `retry_max`, `retry_denoise_step`, `retry_seed_step`,
`sampler`, `guidance`, `detail_a`, `invert` (1 = inversion-anchored refine, the Upscale v2
mechanism, unmeasured on faces), `id_threshold`, `guide1` / `denoise1` / `steps1`,
`guide2` / `denoise2` / `steps2` (`guide2: 0` drops pass 2).

Models and packages: `ultralytics` + `models/ultralytics/bbox/face_yolov8m.pt` are required
(the Impact Subpack installs both; otherwise `pip install ultralytics` and download the
detector from `huggingface.co/Bingsu/adetailer`). `insightface` + `onnxruntime` +
`models/insightface/models/buffalo_l/` are optional — without them the identity gate is OFF
(one attempt per pass, the report says so). The gate runs on CPU: ≈1.2 s per embedding on a
desktop CPU, i.e. ≈4 s for the usual one face / two passes / no retry.

### KreaPhoton Scheduler

`steps`, `alpha` (schedule steepness; default 3.158 = e^1.15, the stock Krea2
`ModelSamplingFlux` shift — verified equivalent to live `calculate_sigmas` within 1e-6),
`restart_frac`, `sigma_r`, `plunge` → SIGMAS. Model calls always equal `steps` (the restart
segment is dropped below 4 steps, a plunge needs 2 structure points).

⚠️ **A restart schedule encodes one ascending σ-jump.** The stock `SamplerCustom` /
k-diffusion samplers do not understand ascending sigmas — feed KreaPhoton SIGMAS only
into KreaPhoton samplers.

### KreaPhoton Empty Latent

Megapixel tiers S (~1.0 MP) / M (~1.4) / L (~1.7, default) / XL (~2.1) × aspects
1:1, 4:3, 3:2 (default), 16:9, 9:16, plus `batch_size`. All dimensions divisible by 16
(VAE /8 × DiT patch 2×2). Exact sizes (W × H):

| tier | 1:1 | 4:3 | 3:2 | 16:9 | 9:16 |
|---|---|---|---|---|---|
| S (~1.0 MP) | 1024×1024 | 1152×864 | 896×1344 | 1344×768 | 768×1344 |
| M (~1.4 MP) | 1184×1184 | 1344×1008 | 1040×1568 | 1568×880 | 880×1568 |
| L (~1.7 MP) | 1312×1312 | 1504×1120 | 1088×1600 | 1728×960 | 960×1728 |
| XL (~2.1 MP) | 1440×1440 | 1664×1248 | 1184×1776 | 1920×1088 | 1088×1920 |

Note that the `3:2` bucket is **portrait** (2:3, W < H — the default 1088×1600 frame
quoted throughout this README); `4:3` and `16:9` are landscape, `9:16` portrait.

### KreaPhoton Encode

`clip` + `text` → CONDITIONING through krea2's plain `KREA2_TEMPLATE` path (the same
encode the stock CLIPTextEncode does for this model), with an optional `style` directive
prepended to the prompt: `off` (default, faithful / literal), `editorial`, `cinematic`,
`natural` (short calibrated nudges — krea2's Qwen encoder responds strongly to leading
instruction text) or `custom` (uses `custom_directive`). Krea2 renders media nouns
literally — avoid "magazine", "cover", "snapshot", "poster" in prompts, they leak as text
and borders.

### KreaPhoton Save Image

`images`, `folder_path` (absolute server folder, **Browse** button; empty = the standard
`output/`; created if missing), `filename_prefix` (default `KreaPhoton`), `format`
`png` / `jpg` / `webp`, `quality` (jpg / webp), `save_metadata` (PNG text chunks — drag
the PNG back into ComfyUI to restore the graph; EXIF for jpg / webp). Names are
`prefix_YYYYMMDD-HHMMSS-mmm_00001.ext` (timestamp to the millisecond + batch index; on a
collision the counter continues past it), unique across runs. The folder browser is served by
`server_routes.py` + `web/kreaphoton_save.js` and has **no path allowlist** — it lists any
directory the ComfyUI process can read. Local use only; do not expose a `--listen` server
with this pack installed.

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

The full photo pipeline (LoRA character + style, upscale, faces, save):

```
CheckpointLoader ─► LoRA Phase (character, identity) ─► LoRA Phase (style, texture) ─┬─► KreaPhoton Sampler ─► VAEDecode ─► KreaPhoton Upscale v2 ─► KreaPhoton Face Detailer ─► KreaPhoton Save Image
KreaPhoton Encode (positive) ─────────────────────────────────────────────────────────┴─► (positive of every KreaPhoton node)
KreaPhoton Empty Latent ─► (latent_image)
```

The plan travels with the MODEL: the sampler runs each LoRA only in its phase, Upscale
honours the identity / texture LoRAs (never the composition ones), the Face Detailer runs
the crop on the texture set with the identity LoRAs at `×identity_boost`.

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
- **Two-axis seed variety (M4, V3)** — at a σ-boundary (default 0.96, i.e. the third
  model call): low-frequency FFT band re-composition of the latent (AC-only — Wan21
  channels are not zero-mean) + rotation of semantic conditioning taps (7–10) of the
  packed 30720-dim Krea2 cond. Mutation-cap validated: no identity/pose/composition
  breaks at any level. **What it moves:** texture / micro-detail (skin, hair, fabric
  realization) at a fixed composition — on this model a variance-preserving latent
  perturbation cannot move layout (measured: ≤2% of a seed change's composition
  authority even at the top of the trajectory). Composition variety = a different seed
  or `blend`. Since v1.3 both axes are applied *inside* the single sampler lifecycle
  (conditioning switched per step by the guider), so the validated gated eta stays on
  and there is no latent round-trip between segments. The latent axis re-composes only
  the **noise component** of the state, `x = (1−σ)·x̂₀ + σ·ε` with `x̂₀` the model's
  previous prediction: re-composing the whole state made its low frequencies disagree
  with the committed structure, which the guidance window amplified into glowing blobs
  on some seeds (found and fixed 2026-09-07).
- **clean_model split (anti-mutation)** — optional: composition phase (σ > 0.85) runs
  on a clean checkpoint, the LoRA identity/detail phase on `model`. ZPhoton pattern. The
  two segments now share one global detail-envelope index, so the split integrates the
  exact sigma sequence of a single run; gated eta stays on. Smoke-validated on a 4-LoRA
  Krea2 stack (character LoRA @1.0) at two seeds — identity, hands and clothing intact.
  Whether it *reduces* LoRA mutation on Krea2 is not measured yet.

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

| | fast | balanced (default) | quality | candid | raw/experimental |
|---|---|---|---|---|---|
| steps | 8 | 12 | 16 | 16 | 36 |
| sampler | euler | euler | euler_2m (AB2) | euler_2m (AB2) | euler_2m |
| restart `frac` / `σ_r` / plunge | 0.25 / 0.65 / on | 0.25 / 0.65 / on | 0.25 / 0.65 / on | 0.25 / 0.65 / on | 0.20 / 0.45 / off |
| detail | 0.50 | 0.60 | 0.70 | 0.30 | 0.50 |
| eta0 / σ_gate | 1.0 / 0.10 | 1.0 / 0.10 | 1.0 / 0.10 | 1.0 / 0.10 | 1.0 / 0.10 |
| contraction | 0.70 | 0.70 | 0.70 | 0.85 | 1.00 (uncalibrated) |
| guidance | window Δ1.25 (with negative) | window Δ1.25 (with negative) | window Δ1.25 (with negative) | window Δ1.25 (with negative) | flat CFG 3.5 always |
| x0_extrapolation | 0 | 0 | 0 | 0 | 0 |

`turbo/candid` is the quality grid with the texture push dialled back (detail 0.30,
noise amplitude 0.85). On a 12-scene set (cinematic, documentary, lifestyle, groups of
3–5, hands, night interiors, action, landscape) plus a 6-frame LoRA story it read
consistently a little softer and more natural than `quality` in hard light — skin, wet
fabric, hair — with the same structure, hands and prompt adherence. The difference is
subtle: most "hyper-detail" in Krea2 output comes from prompt words (pores, texture,
detailed skin) and polished LoRAs, not from the sampler.

`σ_r=0.65`, `plunge`, `eta0=1.0`, `contraction=0.70`, `delta=1.25` were selected by the
V-protocol on real generations (pre-registered cells and accept rules) — on a narrow
domain: three prompts, seeds 1001–1003, the official Turbo checkpoint. `detail_a` values
are design hypotheses from the M2 working range, labeled as such in `presets.py`;
`contraction` is an empirical aesthetic control (cleaner shadows, more inter-seed
diversity at 0.70), not a "correct manifold" — the flow prior expects unit noise.
`raw/experimental` targets the non-distilled RAW mode and is largely unexplored.

---

## Installation

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/Kostik2702/ComfyUI-KreaPhoton.git
```

Restart ComfyUI. No extra Python dependencies for the samplers, scheduler, encode, save
and upscalers — torch and comfy only (no `requirements.txt`, nothing is pip-installed).

Per-feature dependencies (the Face Detailer's detector is the only hard one — the others
degrade with a console line when absent):

| Feature | Needs |
|---|---|
| Face Detailer detection | `ultralytics` + `models/ultralytics/bbox/face_yolov8m.pt` (the Impact Subpack installs both; else `pip install ultralytics` and the detector from `huggingface.co/Bingsu/adetailer`) — **required** for the node to run |
| Face Detailer identity gate | `insightface` + `onnxruntime` + `models/insightface/models/buffalo_l/` — without them one attempt per pass, no retry |
| `taesd` preview | `lighttaew2_1` in `models/vae_approx` (falls back to latent2rgb) |
| `restart_enhance` | the `ComfyUI-dlss-enhancer` pack (experimental, measured practically invisible) |
| SR-model base in Upscale / Face Detailer | any ESRGAN-class `UPSCALE_MODEL` (measured with 4xNomos8kSCHAT-L / 4xNomos8kDAT) |

Requirements: ComfyUI 0.2x–0.3x (developed on 0.26, current on 0.30.1; the samplers go
through `comfy.samplers` / `CFGGuider` and `SamplerCustomAdvanced`'s prepare chain, the
Face Detailer through `KSamplerX0Inpaint`) + a Krea 2 Turbo checkpoint (Wan21 16-channel
latent family; bf16 and int8 / fp8-quantized both work, LoRA plans included).

---

## Known limitations

- **Variety + negative share a hallucination budget.** With the latent variety axis
  active, the guidance window runs at `delta · (1 − 0.45·a_latent)` (high → 0.88,
  medium → 1.02). Measured on a merge checkpoint: at the full Δ=1.25 one seed in six
  grew a glowing / mesh region; at the reduced Δ (or at medium) the same seeds are
  clean. The Advanced node applies the same rule to its explicit `delta`.
- Variety is a texture/detail knob, not a composition knob (measured, see above). Level
  dose-response at the 0.96 boundary has been checked for artifacts and mutation-cap,
  not re-swept for monotonic strength.
- `guidance_rescale` (Advanced) is a standard CFG-rescale, exposed as an option; it is
  not part of any preset and was not needed once the budget rule was in place.
- `x0_extrapolation` and `detail_a` values are uncalibrated / design hypotheses.
- `raw/experimental` preset: schedule steepness for the dynamic-μ RAW canon is
  uncalibrated.
- Restart SIGMAS are KreaPhoton-only (ascending jump, see Scheduler warning).
- `KreaPhoton Upscale` cannot repair geometry: it is source-consistent by construction,
  so a deformed hand or watch in the source stays deformed (sharper). Any denoise that
  could re-draw it (≥ 0.35 here) was measured to rewrite the whole frame.
- The validation domain is narrow (3 prompts × 3 seeds × official Turbo). Faces of
  different ages/skin tones, groups, hands, text, night/interior scenes, LoRA stacks
  and quantized checkpoints are not covered by the evidence base.
- `KreaPhoton Face Detailer`: the preset numbers (guide / denoise / steps) are derived
  from an Impact Pack measurement on krea2, not validated on the KreaPhoton sampler; the
  `identity_boost` default was measured on one character LoRA and one graph. The ArcFace
  gate runs on CPU (≈1.2 s per embedding). The detector needs `ultralytics` — without it
  the node fails with a clear message rather than passing the image through.
- `KreaPhoton Save Image`'s folder browser has no path allowlist (local use only).
- The pack's own `WEB_DIRECTORY` ships one JS extension (`web/kreaphoton_save.js`); the
  on-node previews and thumbnails use ComfyUI's stock preview channel.

## Testing

Plain-assert test suite (`tests/run_tests.py`, no pytest dependency — runs under the
ComfyUI embedded interpreter, 9 files, each a subprocess so an import failure is a red
file, not a skipped one). Tests assert invariants, not literal preset values: schedule
equivalence to stock `calculate_sigmas` at features-off, model calls == steps for every
step count, SIGMAS validation accept/reject table, restart variance balance, guidance
window exactness + variety cond switch, bit-exactness of untouched conditioning taps,
`corr(lf′, lf) = √(1−a²)` for the LF re-composition, in-loop variety == `lf_recompose`
of the boundary state, terminal extrapolation formula, the preset → `run_sampling`
contract (RAW CFG, INPUT_TYPES backward compatibility), tile grid / feather / merge
identities (`test_tiling`), the masked-run single-lifecycle rule (`test_sampling`), Save
Image naming / collision / metadata (`test_save`), and the Face Detailer geometry, retry
policy, keep-best choice, seed range and `build_face_model` boost (`test_face`).

```sh
"<ComfyUI>/python_embeded/python.exe" tests/run_tests.py
```

Last run before 1.6.1 was tagged: `ALL 9 TEST FILES PASSED` on ComfyUI 0.30.1
(torch 2.9 / cu130 / cp313 portable).

## Changelog

**1.6.1**
- Face Detailer: `identity_boost` (default 1.5) — the plan's identity LoRAs (`all` /
  `identity`) run the face pass at `strength × boost` (`lora_phase.build_face_model`).
  Measured on the owner's graph against the character's training set: at ×1.0 every redraw
  lost likeness to the character (0.557 → 0.47–0.49), at ×1.5 it gains (0.62–0.66); neither
  the LoRA set nor denoise nor the prompt was the lever. `reference_image` accepts a batch
  (centroid of the found faces); with a reference the retries keep the pass denoise and
  explore seeds. `face_positive` tooltip / README: a short trigger prompt measured lower
  than the scene positive, leave it unconnected. Report lines for all three.
- Fix, "confetti ring" — coloured speckle in the feather band of a detailed face. A
  `noise_mask` must never cross a phase-model split (comfy's `KSamplerX0Inpaint` re-noises
  the unmasked band from the source with the original noise; segment 2+ handed it the
  mid-trajectory state and zero noise). Engine: a masked run collapses to one lifecycle on
  the longest phase's model (console note); Face Detailer: the crop pass always runs on the
  plan's texture patcher, no split. Reproduced deterministically before the fix
  (`retry_max=1`, denoise 0.45 → σ₀ 0.658); regression test in `test_sampling.py`.
- Fix: per-face / per-attempt seeds (`seed + batch + face·7919 + pass·104729 +
  retry·retry_seed_step`, masked to 64 bits) — a seed near the top of the range no longer
  overflows comfy's RNG.
- README: Upscale v2 in the node table, Encode / Save Image / Empty Latent sizes
  documented, Sampler table completed (`restart_enhance`, `texture_model`, `coherence`),
  masked-run rule under Phase models, installation matrix of optional dependencies.

**1.6.0**
- `KreaPhoton Face Detailer` node (`face_detailer.py`, `face_geometry.py`, `face_detect.py`):
  YOLO face detection, `max_faces` largest faces redrawn per preset (`subtle` / `standard` /
  `strong`, two passes 1024 → 1536) with the KreaPhoton sampler on the LoRA plan's phase
  model, elliptic noise mask, ArcFace identity gate with keep-best retry, optional
  `reference_image`, `face_positive`, SR-model crop enlarge, `tune`; `mask` and `report`
  outputs. Presets derived from the Impact Pack krea2 measurement, labeled unvalidated.
- `tests/test_face.py` (9 test files now).

**1.5.0**
- `KreaPhoton Upscale` node: faithful tiled refine ×1.25–×2.0 on the same checkpoint —
  latent tiles re-blended every step inside the sampler loop, post-decode
  back-projection to the source, presets `polish` / `detail` / `strong`, optional SR
  model base, LoRA Phase plans honoured. Calibrated on existing frames (docs/07 §11).
- Sampler loop: optional `tiler` / `x0_hook` (both `None` = bit-exact v1.4 behaviour).
- New module `tiling.py` + `tests/test_tiling.py` (8 test files now).
- `KreaPhoton Upscale v2` node (`upscale_v2.py`): empty-tile skipping, per-step grid shift,
  optional noise inversion (`sampling.run_inversion`); measured, see README section.

**1.4.0** (experimental features, each behind an off-by-default switch)
- Phase models: `texture_model` input (restart / texture phase) next to `clean_model`
  (composition) — LoRA phase scheduling that works on quantized checkpoints.
- `KreaPhoton LoRA Phase` node: LoRA active only in one schedule phase (shipped as comfy
  weight hooks with σ keyframes, bf16 checkpoints only; replaced in 1.4.1 by the plan →
  phase-models mechanism described above, which also covers int8 / fp8 checkpoints).
- Perturbed-attention guidance on the Advanced node (`pag_*`).
- `restart_enhance`: DLSS 5 Photoreal Enhance V2 at the restart boundary (needs the
  `ComfyUI-dlss-enhancer` pack and `vae`).
- Resolution-aware schedule shift (canonical Krea 2 μ by token count); L/XL unchanged.
- Coherence tools: `jump` (one-time jump-back) and `self_refine` (second descent from the
  plunge draft) — simple-node combo `coherence`, Scheduler `refine_steps`.
- `turbo/candid` preset.

**1.3.0**
- **Root cause of the "split speckle / mosaic / droplets" found and fixed**: a segment
  split (old variety split, `clean_model`) restarted the detail-envelope progress at
  zero, so the restart steps were evaluated at a far lower σ than the state actually
  had (σ̂≈0.52 for σ=0.65) and the model under-removed its own re-noise; the leftover
  decoded as coloured confetti. It was never bf16 or the noise-scaling round-trip. The
  envelope now uses the global step index; the `eta0=0` guard is gone.
- Variety runs inside one sampler lifecycle (no split, no latent round-trip, gated eta
  kept on) and re-composes only the noise component of the state.
- Variety and the guidance window share a hallucination budget (`delta` scaled by the
  latent variety amplitude).
- `raw/experimental` really runs full-trajectory CFG 3.5 (the preset's `cfg` was never
  read before).
- SIGMAS validation on every sampler run; Scheduler model calls == steps for all N ≥ 1.
- Numerical guards: degenerate guidance windows, foreign/zeroed conditioning in variety,
  clear errors for non-Krea2 latents.
- New Advanced knobs: `x0_extrapolation` (terminal x0-trajectory extrapolation),
  `guidance_rescale` (CFG-rescale on guided steps), `contraction` range up to 1.5.
- README/pyproject synced with the shipped nodes (6 nodes, version 1.3.0).

## License

MIT © 2026 Kostiantyn Hrytsuk
