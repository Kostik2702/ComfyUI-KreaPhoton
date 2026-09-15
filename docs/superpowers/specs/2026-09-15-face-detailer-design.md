# KreaPhoton Face Detailer — design spec

Date: 2026-09-15. Owner decision (chat, this date): own implementation on ultralytics,
ArcFace identity gate (original crop + optional reference), largest faces first.
Prior art: vault `DESIGN-KleinPhoto-nodePack` Node 5 (FaceRefiner), `SPEC-face-refiner-retry-logic`,
`photorealism-settings` (Krea 2 FaceDetailer numbers measured 2026-09-01 on Impact Pack).

## 1. Goal

One node, `KreaPhoton Face Detailer`, that takes a finished IMAGE (after the sampler or
the upscaler), finds faces, redraws the N largest at higher resolution with the
KreaPhoton sampler on the identity-phase model of the LoRA plan, and pastes them back.
Simple UI (preset + `max_faces`), maximum face detail, and — when a character LoRA is in
the plan — the LoRA's likeness, not a generic face. Identity is measured (ArcFace), not
assumed.

Non-goals (v1): hands, SAM segmentation masks, per-face prompts, video/batch
temporal consistency, GPU insightface (CPU ORT is enough: ~0.1 s per face).

## 2. Node interface

Category `KreaPhoton`, class `KreaPhotonFaceDetailer`, file `kreaphoton/face_detailer.py`
(same layout as `upscale_v2.py`: node class in its own module, registered from `nodes.py`).

required
- `model` MODEL — may carry a LoRA plan (`lora_phase.PLAN_KEY`).
- `positive` CONDITIONING — the main prompt; used for the face pass when `face_positive`
  is not connected.
- `image` IMAGE — (B, H, W, C) 0..1; every batch item is processed independently.
- `vae` VAE — Krea 2 (Wan 2.1) VAE; latents are (1, 16, 1, h, w), `LATENT_PX = 8`.
- `seed` INT.
- `preset` `["subtle", "standard", "strong"]`, default `standard`.
- `max_faces` INT 1..8, default 1.

optional (appended in this order; never reordered — `test_nodes.py` widget-order rule)
- `negative` CONDITIONING — passed through to the guider; at cfg 1 it does nothing
  (documented, same as the samplers).
- `face_positive` CONDITIONING — prompt for the crop only (character-LoRA trigger +
  "close-up portrait, natural skin texture"). Absent → `positive`.
- `reference_image` IMAGE — a photo of the character; the identity gate measures
  against its largest face instead of the original crop.
- `upscale_model` UPSCALE_MODEL — used to enlarge the crop before encoding
  (`nodes._upscale_pixels` already supports it); absent → lanczos.
- `tune` STRING — JSON object overriding preset fields (same format as
  `upscale_v2.apply_tune_v2`); unknown key → ValueError naming the key.

outputs
- `IMAGE` — the detailed image, same size as input.
- `MASK` — union of the pasted face masks (B, H, W) 0..1, for chaining
  (e.g. a second pass, or an inspector).
- `report` STRING — one line per face: index, bbox, area px, attempts, chosen
  denoise, id_sim per attempt, gate state; plus global warnings (see §7).

## 3. Preset schema (`presets.FACE_PRESETS`, validated at import like the others)

```
FACE_PRESETS = {
  "subtle":   {"passes": [(1024, 0.25, 6)],                    "id_threshold": 0.70},
  "standard": {"passes": [(1024, 0.35, 6), (1536, 0.15, 6)],  "id_threshold": 0.65},
  "strong":   {"passes": [(1024, 0.45, 7), (1536, 0.20, 6)],  "id_threshold": 0.60},
}
FACE_COMMON = {
  "crop_factor": 2.0,      # bbox side x factor (vault: 3.0 shrinks the face in the crop)
  "bbox_threshold": 0.45,  # YOLO confidence
  "min_face_px": 48,       # smaller bboxes are skipped (no detail to build)
  "feather": 0.06,         # fraction of crop side; mask feather + paste feather
  "dilation": 0.10,        # fraction of bbox side added around the bbox for the mask
  "retry_max": 3,          # attempts per pass (1 = no retry)
  "retry_denoise_step": 0.05,   # denoise -= step per retry
  "retry_seed_step": 1000,      # seed += step per retry (distilled model: seed+1 is ~identical)
  "sampler": "euler", "guidance": "window", "detail_a": 0.0, "invert": 0,
}
```

`passes` = `(guide_px, denoise, n_steps)`: the crop is resized so its long side equals
`guide_px`, then `schedules.refine_schedule(n_steps, alpha, denoise)` runs `n_steps`
steps of a schedule oversampled at `n_steps/denoise` (the vault "effective steps" rule:
18×0.35 ≈ 6 on Impact ↔ our 6 steps at denoise 0.35). `alpha` from `alpha_for_latent`
(resolution-aware, as Upscale v2). Pass 2 (if any) starts from the pass-1 result.

Honesty label (README + presets comment): numbers are DERIVED from the Impact Pack
measurement of 2026-09-01 (er_sde, 18×0.35, guide 1024, crop 2.0) and NOT yet validated
on the KreaPhoton sampler; validation = owner's live A/B on the rig, tracked in
`learnings/active/KREA2-NODES.md`.

`tune` keys: any `FACE_COMMON` key, `id_threshold`, and `denoise1`, `steps1`, `guide1`,
`denoise2`, `steps2`, `guide2` (override the pass tuple; `guide2=0` removes pass 2;
`invert=1` enables the inversion-anchored variant, §5).

## 4. Pipeline per image

```
detect(image) -> faces sorted by bbox area desc
  filter conf >= bbox_threshold and min(side) >= min_face_px
  take first max_faces
for each face:
    crop_box  = square(bbox center, side = max(bbox side) * crop_factor), aligned to
                ALIGN_PX (16), clamped to the image (shift, then shrink if the image is
                smaller than the crop)
    crop      = image[crop_box]
    mask_c    = ellipse inscribed in the dilated bbox (in crop coords), feathered
    ref_emb   = arcface(reference face)   # reference_image's largest face, else crop
    best      = None
    for pass in preset.passes:
        src = crop if first pass else best.crop
        for attempt in range(retry_max):
            denoise = pass.denoise - attempt * retry_denoise_step (floor 0.05)
            seed_a  = seed + face_index * 7919 + attempt * retry_seed_step
            px      = upscale(src, guide_px)                      # lanczos | upscale_model
            z       = vae.encode(px)  (tiled if > 1024)
            out     = run_sampling(identity_model, face_positive, negative,
                                   {"samples": z, "noise_mask": mask_c at latent res},
                                   refine_schedule(n_steps, alpha, denoise), seed=seed_a,
                                   guidance from preset, texture_model=plan texture
                                   for pass 2 when sigmas[0] <= UPSCALE_TEXTURE_START)
            cand    = vae.decode(out) resized back to crop size (lanczos)
            sim     = cos(arcface(cand), ref_emb)     # None when gate is off
            record attempt; if sim is None or sim >= id_threshold: break
        best = attempt with max sim (gate on) | the last attempt (gate off)
    paste: image[crop_box] = crop * (1 - mask_p) + best.crop * mask_p
           (mask_p = the same feathered ellipse in pixel space; faces are pasted in
            area order so an overlap is resolved by the larger face)
    mask_out |= mask_p placed in image coords
```

Model selection: `lora_phase.build_phase_models(model)` → `(clean, identity, texture)`.
The face pass always runs on `identity` (the character LoRA is there by construction);
`texture` is handed to `run_sampling` as `texture_model` so a style LoRA in the texture
phase acts on the restart-free low-sigma tail exactly as in Upscale v2. No plan →
`model` as is.

Batch: `image` batch items are processed sequentially; `seed + b` per item as the
upscalers do.

## 5. Inversion-anchored variant (`invert=1`, off by default)

Same as §4 but the crop latent is first inverted with `sampling.run_inversion` on
`upscale_v2.inversion_sigmas(sigmas, n_steps - 1)` and resampled with
`add_noise=False`. Measured +0.2–0.3 dB source fidelity on Upscale v2 at denoise
0.06–0.25; unmeasured at face denoise 0.35–0.45. Exposed for the owner's A/B only.

## 6. Identity gate

- Backend: insightface `FaceAnalysis(name="buffalo_l", root=<models/insightface>,
  providers=["CPUExecutionProvider"])`, `det_size=(640, 640)`, loaded lazily once per
  process and cached. Root resolved via `folder_paths.models_dir` + `insightface`.
- Embedding of a crop = normed embedding of the largest detected face in that crop;
  no face detected in a candidate → `sim = 0.0` (a destroyed face is the worst case,
  same "empty detection = 0" rule as the KleinPhoto retry spec).
- Reference: `reference_image` largest face when connected (no face in it → warning in
  report, fall back to the original crop); otherwise the original crop.
- Gate off when insightface (or its model pack) is unavailable: single attempt per
  pass, `report` says `identity gate: OFF (insightface not available)`. Never a hard
  error — the node still details.
- Keep-best: among attempts, the highest `sim`; ties → earliest. Never returns an
  attempt below the best one. With `reference_image`, if the best attempt's sim is
  below the ORIGINAL crop's sim to the reference by more than 0.05, the original crop
  is kept and the report says `kept original (identity regressed)`.

## 7. Report warnings

- LoRA plan present and `face_positive` not connected → `note: LoRA plan has N entries;
  face_positive is not connected — make sure the character trigger is in positive`.
- No faces detected → `no faces >= min_face_px at conf >= threshold` (image returned
  unchanged, MASK all zeros).
- More faces than `max_faces` → `k faces skipped (max_faces)`.

## 8. Files

- `kreaphoton/face_geometry.py` — pure torch/py, NO comfy imports at module level
  (like `tiling.py`): `select_faces(boxes, confs, max_faces, min_face_px, threshold)`,
  `crop_box(bbox, crop_factor, image_hw, align)`, `face_mask(crop_hw, bbox_in_crop,
  dilation, feather)`, `paste(image, crop_box, patch, mask)`, `retry_schedule(denoise,
  seed, retry_max, denoise_step, seed_step)`, `choose_best(attempts, threshold,
  orig_sim)`.
- `kreaphoton/face_detect.py` — lazy wrappers: `detect_faces(image_bhwc) ->
  [(x0,y0,x1,y1,conf)]` via ultralytics YOLO (`models/ultralytics/bbox/face_yolov8m.pt`,
  resolved through `folder_paths.get_folder_paths("ultralytics_bbox")` when registered,
  else `models_dir/ultralytics/bbox`), `ArcFaceGate` (§6). Both cache their model
  objects per process.
- `kreaphoton/face_detailer.py` — the node; `apply_tune_face`, the §4 loop, report.
- `kreaphoton/presets.py` — `FACE_PRESETS`, `FACE_COMMON`, `validate_face_presets`.
- `kreaphoton/nodes.py` — register class + display name `KreaPhoton Face Detailer`.
- `tests/test_face.py` — plain-assert, runnable with the embedded python without a
  ComfyUI tree for the geometry module; the node's `run_sampling`, VAE, detector and
  gate are monkeypatched.
- `tests/test_nodes.py` — structural check of the new node's INPUT_TYPES (required
  order, optional order).
- `README.md` — node section + changelog; `pyproject.toml` version 1.6.0.

## 9. Tests (what must be red before green)

Geometry: crop at an image corner is clamped and still aligned; crop larger than the
image shrinks to the image; sort by area is stable and `max_faces` truncates; faces
under `min_face_px` and under threshold are dropped; mask is 1 at the bbox center,
0 at the crop corner, monotone across the feather band; paste with a zero mask is the
identity, with a one mask replaces the bbox; retry schedule floors denoise at 0.05 and
uses distinct seeds; `choose_best` keeps the max sim, returns the original on
regression > 0.05, and picks the last attempt when sims are `None`.

Node (patched): the identity model from the plan is what reaches `run_sampling`;
`noise_mask` is present and latent-sized; pass 2 gets `texture_model`; gate OFF path
runs exactly one attempt; report lines contain every attempt; `tune` unknown key raises.

Live (owner, not automated): A/B `standard` vs Impact Pack FaceDetailer on the rig
(`E:/CUI portable/claude/ab_kit/score.py` id_sim), close-up / medium / full-body frames,
with and without a character LoRA in the plan.

## 10. Risks / open questions

- Preset numbers unvalidated on this sampler (labeled). Mitigation: `tune` exposes every
  number; validation task in `learnings/active/KREA2-NODES.md`.
- `noise_mask` blending in comfy's `KSamplerX0Inpaint` blends `latent_image` per step —
  inside the feather band the model sees a mix; feather 6% is the Impact default
  region; if seams appear, `feather` is a tune key.
- ArcFace on CPU: buffalo_l ≈ 60–120 ms per embedding; ≤ 3 attempts × 2 passes × 8
  faces worst case ≈ 6 s — acceptable next to the sampling cost.
- ultralytics import time (~1 s) is paid once per process, lazily on first run.
