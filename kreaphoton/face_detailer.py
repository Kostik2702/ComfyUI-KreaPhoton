"""
KreaPhoton Face Detailer (v1.6): detect faces (ultralytics YOLO), redraw the N
largest at guide resolution with the KreaPhoton sampler on the LoRA plan's phase
model, gate every attempt with ArcFace identity similarity (keep-best retry), and
paste the winner back through a feathered ellipse mask.

Per face, per pass (guide_px, denoise, n_steps):
  crop (bbox x crop_factor, square, /16) -> resize long side to guide_px (lanczos or
  upscale_model) -> VAE encode -> run_sampling(refine_schedule(n_steps, denoise),
  noise_mask = ellipse at latent res) -> decode -> ArcFace sim to the reference
  (reference_image's face, else the ORIGINAL crop) -> retry with denoise - step and
  seed + step while sim < id_threshold -> keep the best attempt.
Pass 2 starts from the pass-1 winner AT ITS GUIDE RESOLUTION (no round trip through
the crop size), so the detail built at 1024 feeds the 1536 pass. Only the final
winner is resized down to the crop and pasted.

Phase model: the crop pass runs as ONE lifecycle on the plan's texture set
(identity- and texture-phase LoRAs act, a composition-only LoRA does not) - the
pass lives at sigma <= ~0.66, texture territory, and a noise_mask cannot cross a
phase split (comfy's inpaint blend needs the source latent and the original noise
in every segment; a split hands segment 2 the noisy state, and the mask band
decodes to a ring of coloured speckle - found 2026-09-15 on the owner's graph,
where a composition+identity LoRA made the texture set differ from the identity
set and denoise 0.45 started one step above 0.65). Without a plan the model runs
as is.

Identity boost (v1.6.1): the plan's identity-carrying LoRAs (phase all / identity)
run the crop pass at strength x identity_boost (default 1.5) through
lora_phase.build_face_model. Measured 2026-09-16 on the owner's graph against the
character's training set (ArcFace to the 120-face centroid): at x1.0 EVERY redraw
lost likeness (input 0.557 -> 0.47-0.49) whatever the LoRA set, denoise or prompt;
at x1.5 it gained (0.62-0.66, above the base generation's 0.607), x1.8 plateaued.
The face has more authority on a 1024-px crop than in the full frame, and at the
nominal strength the base model's face prior wins the redraw.

Reference: `reference_image` may be a BATCH (several photos of the character); the
gate measures against the L2-normalised mean of their embeddings - a centroid is a
far steadier target than one photo (measured: a real photo of the character scores
~0.79 to the centroid, ~0.63 to another single photo). With a reference the retries
explore SEEDS at the pass's denoise (the reference is the target, the redraw should
move toward it), without one they lower denoise (the original crop is the target,
a retry must move toward it) - keep-best either way.

Module-level seams (_detect, _gate, _run_sampling, _run_inversion, _build_phase_models,
_build_face_model, _upscale) exist so tests can replace the heavy parts; the node
calls them by name.
"""
import json

import torch
import torch.nn.functional as F

from . import face_detect
from .face_geometry import choose_best, crop_box, face_mask, paste, place_mask, retry_schedule, select_faces
from .lora_phase import PLAN_KEY, build_face_model, build_phase_models
from .presets import (DEFAULT_FACE_PRESET, FACE_COMMON, FACE_PRESETS, GUIDANCE, MANIFOLD_MEAN, MANIFOLD_STD,
                      preset_guidance, validate_face_presets)
from .sampling import run_inversion, run_sampling
from .schedules import ALPHA, SHIFT, alpha_for_latent, refine_schedule

_PRESET_TOOLTIP = ("subtle: one pass 1024 px / denoise 0.25 (LoRA-identity safe). standard: 1024 / 0.35 then "
                   "1536 / 0.15 for skin texture. strong: 1024 / 0.45 then 1536 / 0.20 (small faces on "
                   "full-body frames). Steps follow the effective-step rule (6-7 per pass). Numbers derived "
                   "from the Impact Pack krea2 measurement, not yet validated on this sampler.")
_MAX_FACES_TOOLTIP = "How many faces to detail, largest bbox first. Faces below min_face_px (48) are skipped."
_FACE_POSITIVE_TOOLTIP = ("Optional prompt for the face crop only. Measured 2026-09-16: a short 'trigger, close-up "
                          "portrait, natural skin texture' prompt scored LOWER on likeness (0.60) than leaving this "
                          "unconnected and using the scene positive (0.66). Leave unconnected unless the scene "
                          "prompt says nothing about the face.")
_REFERENCE_TOOLTIP = ("Photo(s) of the character - a batch of several is best: the identity gate measures every "
                      "attempt against their mean embedding, retries explore seeds at the pass denoise, and the "
                      "original face is kept when the redraw loses likeness.")
_BOOST_TOOLTIP = ("Strength multiplier for the LoRA plan's identity LoRAs (phase all / identity) in the face pass. "
                  "1.0 = plan strengths as they are (measured: every redraw then LOSES likeness); 1.5 = measured "
                  "gain; 1.8 = plateau. Needs a KreaPhoton LoRA Phase plan on model - a classic LoRA loader is "
                  "not touched (report says so).")
_UPSCALE_MODEL_TOOLTIP = "Optional ESRGAN-class model to enlarge the crop before the pass (else lanczos)."
_TUNE_TOOLTIP = ("Calibration override, leave EMPTY. JSON keys: crop_factor, bbox_threshold, min_face_px, "
                 "feather, dilation, retry_max, retry_denoise_step, retry_seed_step, sampler, guidance, "
                 "detail_a, invert (0/1), id_threshold, guide1/denoise1/steps1, guide2/denoise2/steps2 "
                 "(guide2=0 drops pass 2).")

_FACE_SEED_STRIDE = 7919      # per-face seed offset (prime) so faces do not share a noise draw
_PASS_SEED_STRIDE = 104729    # per-pass seed offset
_REGRESS_MARGIN = 0.05        # with a reference: keep the original if the redraw loses more than this

# seams (see the module docstring)
_detect = face_detect.detect_faces
_gate = face_detect.get_gate
_run_sampling = run_sampling
_run_inversion = run_inversion
_build_phase_models = build_phase_models
_build_face_model = build_face_model


def reference_embedding(gate, images):
    """(centroid, n_faces) over a (B, H, W, C) batch: the L2-normalised mean of the
    per-image embeddings of the images where the gate finds a face; (None, 0) when
    it finds none."""
    embs = []
    for b in range(int(images.shape[0])):
        e = gate.embed(images[b:b + 1])
        if e is not None:
            embs.append(e)
    if not embs:
        return None, 0
    c = torch.stack(embs, dim=0).mean(dim=0)
    return c / c.norm().clamp_min(1e-8), len(embs)


def _fmt_loras(pairs):
    return ", ".join("%s %.2f" % (name.replace("\\", "/").split("/")[-1], s) for name, s in pairs)


def _upscale(image, scale: float, upscale_model=None):
    from .nodes import _upscale_pixels
    return _upscale_pixels(image, scale, upscale_model)


def _resize(image, h: int, w: int):
    """(1, H, W, C) -> (1, h, w, C), lanczos (comfy.utils), lazy import."""
    if (int(image.shape[1]), int(image.shape[2])) == (h, w):
        return image
    import comfy.utils
    return comfy.utils.common_upscale(image.movedim(-1, 1), w, h, "lanczos", "disabled").movedim(1, -1)


def apply_tune_face(preset: dict, common: dict, tune):
    """(passes, id_threshold, common) with the JSON `tune` overrides applied and
    validated. Unknown key -> ValueError naming it."""
    passes = [tuple(p) for p in preset["passes"]]
    thr = float(preset["id_threshold"])
    c = dict(common)
    text = (tune or "").strip()
    if not text:
        return passes, thr, c
    try:
        over = json.loads(text)
    except ValueError as e:
        raise ValueError(f"KreaPhoton Face Detailer: tune must be a JSON object ({e})")
    if not isinstance(over, dict):
        raise ValueError("KreaPhoton Face Detailer: tune must be a JSON object")
    pass_over = {}
    for k, v in over.items():
        if k in c:
            c[k] = v
        elif k == "id_threshold":
            thr = float(v)
        elif k[:-1] in ("guide", "denoise", "steps") and k[-1] in "12":
            pass_over.setdefault(int(k[-1]) - 1, {})[k[:-1]] = v
        else:
            raise ValueError(f"KreaPhoton Face Detailer: unknown tune key {k!r}")
    for idx in sorted(pass_over):
        o = pass_over[idx]
        if idx >= len(passes):
            passes.append((0, passes[-1][1], passes[-1][2]))
        guide, denoise, steps = passes[idx]
        passes[idx] = (int(o.get("guide", guide)), float(o.get("denoise", denoise)), int(o.get("steps", steps)))
    passes = [p for p in passes if int(p[0]) > 0]
    validate_face_presets({"tune": {"passes": passes, "id_threshold": thr}}, c)
    return passes, thr, c


class KreaPhotonFaceDetailer:
    """See the module docstring."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "preset": (list(FACE_PRESETS.keys()), {"default": DEFAULT_FACE_PRESET, "tooltip": _PRESET_TOOLTIP}),
                "max_faces": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": _MAX_FACES_TOOLTIP}),
                "identity_boost": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 3.0, "step": 0.05,
                                             "tooltip": _BOOST_TOOLTIP}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": "Optional; at cfg 1 (the default) it does nothing."}),
                "face_positive": ("CONDITIONING", {"tooltip": _FACE_POSITIVE_TOOLTIP}),
                "reference_image": ("IMAGE", {"tooltip": _REFERENCE_TOOLTIP}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": _UPSCALE_MODEL_TOOLTIP}),
                "tune": ("STRING", {"default": "", "multiline": False, "tooltip": _TUNE_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "report")
    FUNCTION = "detail"
    CATEGORY = "KreaPhoton"

    def detail(self, model, positive, image, vae, seed, preset, max_faces, identity_boost=1.5, negative=None,
               face_positive=None, reference_image=None, upscale_model=None, tune=""):
        from .nodes import LATENT_PX, ALIGN_PX, _ORDER_FROM_SAMPLER_NAME, _decode_tiled, _encode_tiled
        passes, id_threshold, c = apply_tune_face(FACE_PRESETS[preset], FACE_COMMON, tune)
        if c["sampler"] not in _ORDER_FROM_SAMPLER_NAME:
            raise ValueError(f"KreaPhoton Face Detailer: sampler must be one of {list(_ORDER_FROM_SAMPLER_NAME)}")
        order = _ORDER_FROM_SAMPLER_NAME[c["sampler"]]
        cond = face_positive if face_positive is not None else positive
        guidance_mode, flat_cfg = preset_guidance(c, negative is not None)
        plan = list((getattr(model, "model_options", None) or {}).get(PLAN_KEY, []))
        boost = float(identity_boost)

        gate = _gate()
        report = []
        if gate.available:
            report.append("identity gate: ON (%s)" % face_detect.ARCFACE_PACK)
        else:
            report.append("identity gate: OFF (%s)" % (gate.reason or "insightface not available"))
        # The masked crop pass is ONE lifecycle on the plan's texture set (the pass lives at
        # sigma <= ~0.66, texture territory). Never a phase split: comfy's inpaint blend needs
        # the source latent + noise in every segment - a split hands segment 2 the noisy state,
        # the mask band decodes to coloured speckle (confetti ring, 2026-09-15).
        if plan and boost != 1.0:
            run_model, boosted, unchanged = _build_face_model(model, boost)
            report.append("identity boost x%.2f: %s%s" % (boost, _fmt_loras(boosted) or "no identity-phase LoRA in the plan",
                                                        ("; unchanged: " + _fmt_loras(unchanged)) if unchanged else ""))
        else:
            _, identity, plan_texture = _build_phase_models(model)
            run_model = plan_texture if plan_texture is not None else identity
            if not plan and boost != 1.0:
                report.append("identity boost x%.2f ignored: model carries no KreaPhoton LoRA Phase plan (a classic "
                              "LoRA loader is not touched)" % boost)
        if face_positive is not None:
            report.append("note: face_positive is connected - measured 2026-09-16, a short trigger prompt scored "
                          "lower on likeness than the scene positive; try it unconnected")
        ref_emb, ref_is_external = None, False
        if reference_image is not None and gate.available:
            ref_emb, n_ref = reference_embedding(gate, reference_image)
            if ref_emb is None:
                report.append("warning: no face found in reference_image - measuring against the original crop")
            else:
                ref_is_external = True
                report.append("reference: %d face(s) in %d image(s), retries explore seeds at the pass denoise"
                              % (n_ref, int(reference_image.shape[0])))

        out_images, out_masks = [], []
        for b in range(int(image.shape[0])):
            img = image[b:b + 1, ..., :3]
            H, W = int(img.shape[1]), int(img.shape[2])
            boxes, confs = _detect(img, threshold=min(0.25, float(c["bbox_threshold"])))
            selected = select_faces(boxes, confs, max_faces=int(max_faces), min_face_px=int(c["min_face_px"]),
                                    threshold=float(c["bbox_threshold"]))
            eligible = len(select_faces(boxes, confs, max_faces=len(boxes) or 1, min_face_px=int(c["min_face_px"]),
                                        threshold=float(c["bbox_threshold"])))
            mask_out = torch.zeros((H, W), dtype=torch.float32)
            if not selected:
                report.append("image %d: no faces >= %d px at conf >= %.2f (%d raw detections) - unchanged"
                              % (b, int(c["min_face_px"]), float(c["bbox_threshold"]), len(boxes)))
                out_images.append(img)
                out_masks.append(mask_out)
                continue
            if eligible > len(selected):
                report.append("image %d: %d faces skipped (max_faces=%d)" % (b, eligible - len(selected), int(max_faces)))
            cur = img
            for fi, (_, box) in enumerate(selected):
                cb = crop_box(box, float(c["crop_factor"]), (H, W), ALIGN_PX)
                x0, y0, x1, y1 = cb
                side_h, side_w = y1 - y0, x1 - x0
                crop = cur[:, y0:y1, x0:x1, :]
                bbox_in_crop = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
                mask_c = face_mask((side_h, side_w), bbox_in_crop, dilation=float(c["dilation"]),
                                   feather=float(c["feather"]))
                lines = ["face %d: bbox (%d,%d,%d,%d) %dx%d px, crop %dx%d px at (%d,%d)"
                         % (fi + 1, box[0], box[1], box[2], box[3], box[2] - box[0], box[3] - box[1],
                            side_w, side_h, x0, y0)]
                # identity reference for this face
                face_ref, orig_sim, gate_on = ref_emb, None, gate.available
                if gate.available:
                    orig_emb = gate.embed(crop)
                    if ref_is_external:
                        orig_sim = gate.sim(orig_emb, face_ref) if orig_emb is not None else 0.0
                    elif orig_emb is None:
                        gate_on = False
                        lines.append("  identity: no face found in the original crop - gate skipped for this face")
                    else:
                        face_ref = orig_emb
                src = crop           # current best, at its own resolution
                for pi, (guide, denoise, n_steps) in enumerate(passes):
                    scale = float(guide) / float(max(int(src.shape[1]), int(src.shape[2])))
                    px = _upscale(src, scale, upscale_model)
                    z = _encode_tiled(vae, px)
                    h, w = int(z.shape[-2]), int(z.shape[-1])
                    alpha = alpha_for_latent(h, w, ALPHA) if SHIFT["resolution_aware"] else ALPHA
                    noise_mask = F.interpolate(mask_c.view(1, 1, side_h, side_w), size=(h, w), mode="area")
                    noise_mask = noise_mask.view(1, 1, 1, h, w).to(z.device)
                    attempts, cands = [], []
                    base_seed = (int(seed) + b + fi * _FACE_SEED_STRIDE + pi * _PASS_SEED_STRIDE) & 0xffffffffffffffff
                    # with a reference the target is the character: retries explore seeds at the
                    # pass denoise; without one the target is the original crop: retries lower denoise
                    retries = retry_schedule(float(denoise), base_seed, retry_max=int(c["retry_max"]) if gate_on else 1,
                                             denoise_step=0.0 if ref_is_external else float(c["retry_denoise_step"]),
                                             seed_step=int(c["retry_seed_step"]))
                    for ai, (d_a, seed_a) in enumerate(retries):
                        sigmas = refine_schedule(int(n_steps), alpha=alpha, denoise=float(d_a))
                        latent_in = {"samples": z, "noise_mask": noise_mask}
                        add_noise = True
                        if int(c["invert"]):
                            from .upscale_v2 import inversion_sigmas
                            asc = inversion_sigmas(sigmas, int(n_steps) - 1)
                            latent_in = _run_inversion(run_model, cond, negative, {"samples": z}, asc, seed=seed_a,
                                                       guidance_mode=guidance_mode, flat_cfg=flat_cfg,
                                                       delta=GUIDANCE["delta"], lo=GUIDANCE["lo"], hi=GUIDANCE["hi"])
                            latent_in["noise_mask"] = noise_mask
                            add_noise = False
                        out = _run_sampling(
                            run_model, cond, negative, latent_in, sigmas, seed=seed_a,
                            guidance_mode=guidance_mode, flat_cfg=flat_cfg,
                            delta=GUIDANCE["delta"], lo=GUIDANCE["lo"], hi=GUIDANCE["hi"],
                            add_noise=add_noise, contraction=1.0, per_channel_contraction=False,
                            manifold_std=MANIFOLD_STD, manifold_mean=MANIFOLD_MEAN,
                            detail_amount=float(c["detail_a"]), order=order,
                            eta0=0.0, sigma_gate=0.10,   # refine: ancestral noise = speckle (2026-07-07)
                        )
                        cand = _decode_tiled(vae, out["samples"])[..., :3].clamp(0.0, 1.0)
                        sim = None
                        if gate_on:
                            emb = gate.embed(cand)
                            sim = gate.sim(emb, face_ref) if emb is not None else 0.0
                        attempts.append(sim)
                        cands.append(cand)
                        if sim is None or sim >= id_threshold:
                            break
                    best, reason = choose_best(attempts, threshold=id_threshold, orig_sim=orig_sim,
                                               regress_margin=_REGRESS_MARGIN)
                    desc = "; ".join("attempt %d denoise %.3f seed %d%s"
                                     % (k + 1, retries[k][0], retries[k][1],
                                        "" if s is None else " id_sim %.3f" % s)
                                     for k, s in enumerate(attempts))
                    lines.append("  pass %d (guide %d, %d steps): %s -> %s%s"
                                 % (pi + 1, int(guide), int(n_steps), desc, reason,
                                    "" if best is None else " (attempt %d)" % (best + 1)))
                    if best is None:
                        src = None
                        break
                    src = cands[best]
                if src is not None and src is not crop:
                    patch = _resize(src, side_h, side_w).to(cur)
                    cur = paste(cur, cb, patch, mask_c)
                    mask_out = torch.maximum(mask_out, place_mask(mask_c, cb, (H, W)))
                report.extend(lines)
            out_images.append(cur)
            out_masks.append(mask_out)
        text = "\n".join(report)
        print("[KreaPhoton Face Detailer]\n" + text)
        return (torch.cat(out_images, dim=0), torch.stack(out_masks, dim=0), text)
