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

Phase model: every preset pass starts below sigma 0.66 (refine_schedule at denoise
<= 0.45), i.e. inside the plan's texture segment - the crop runs on the texture
patcher exactly like Upscale v2 (identity- and texture-phase LoRAs act, a
composition-only LoRA does not). Without a plan the model runs as is.

Module-level seams (_detect, _gate, _run_sampling, _run_inversion, _build_phase_models,
_upscale) exist so tests can replace the heavy parts; the node calls them by name.
"""
import json

import torch
import torch.nn.functional as F

from . import face_detect
from .face_geometry import choose_best, crop_box, face_mask, paste, place_mask, retry_schedule, select_faces
from .lora_phase import PLAN_KEY, build_phase_models
from .presets import (DEFAULT_FACE_PRESET, FACE_COMMON, FACE_PRESETS, GUIDANCE, MANIFOLD_MEAN, MANIFOLD_STD,
                      UPSCALE_TEXTURE_START, preset_guidance, validate_face_presets)
from .sampling import run_inversion, run_sampling
from .schedules import ALPHA, SHIFT, alpha_for_latent, refine_schedule

_PRESET_TOOLTIP = ("subtle: one pass 1024 px / denoise 0.25 (LoRA-identity safe). standard: 1024 / 0.35 then "
                   "1536 / 0.15 for skin texture. strong: 1024 / 0.45 then 1536 / 0.20 (small faces on "
                   "full-body frames). Steps follow the effective-step rule (6-7 per pass). Numbers derived "
                   "from the Impact Pack krea2 measurement, not yet validated on this sampler.")
_MAX_FACES_TOOLTIP = "How many faces to detail, largest bbox first. Faces below min_face_px (48) are skipped."
_FACE_POSITIVE_TOOLTIP = ("Prompt for the face crop only: the character LoRA trigger + 'close-up portrait, "
                          "natural skin texture'. Not connected -> positive is used. Without the trigger in "
                          "either, a character LoRA redraws a generic face.")
_REFERENCE_TOOLTIP = ("A photo of the character: the identity gate measures every attempt against its face "
                      "instead of the original crop, and keeps the original when the redraw loses identity.")
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

    def detail(self, model, positive, image, vae, seed, preset, max_faces, negative=None, face_positive=None,
               reference_image=None, upscale_model=None, tune=""):
        from .nodes import LATENT_PX, ALIGN_PX, _ORDER_FROM_SAMPLER_NAME, _decode_tiled, _encode_tiled
        passes, id_threshold, c = apply_tune_face(FACE_PRESETS[preset], FACE_COMMON, tune)
        if c["sampler"] not in _ORDER_FROM_SAMPLER_NAME:
            raise ValueError(f"KreaPhoton Face Detailer: sampler must be one of {list(_ORDER_FROM_SAMPLER_NAME)}")
        order = _ORDER_FROM_SAMPLER_NAME[c["sampler"]]
        cond = face_positive if face_positive is not None else positive
        guidance_mode, flat_cfg = preset_guidance(c, negative is not None)
        _, identity, plan_texture = _build_phase_models(model)
        plan = list((getattr(model, "model_options", None) or {}).get(PLAN_KEY, []))

        gate = _gate()
        report = []
        if gate.available:
            report.append("identity gate: ON (%s)" % face_detect.ARCFACE_PACK)
        else:
            report.append("identity gate: OFF (%s)" % (gate.reason or "insightface not available"))
        if plan and face_positive is None:
            report.append("note: LoRA plan has %d entries; face_positive is not connected - make sure the "
                          "character trigger is in positive" % len(plan))
        ref_emb, ref_is_external = None, False
        if reference_image is not None and gate.available:
            ref_emb = gate.embed(reference_image[:1])
            if ref_emb is None:
                report.append("warning: no face found in reference_image - measuring against the original crop")
            else:
                ref_is_external = True

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
                    retries = retry_schedule(float(denoise), base_seed, retry_max=int(c["retry_max"]) if gate_on else 1,
                                             denoise_step=float(c["retry_denoise_step"]),
                                             seed_step=int(c["retry_seed_step"]))
                    for ai, (d_a, seed_a) in enumerate(retries):
                        sigmas = refine_schedule(int(n_steps), alpha=alpha, denoise=float(d_a))
                        run_model, texture_model = identity, plan_texture
                        if plan_texture is not None and float(sigmas[0]) <= UPSCALE_TEXTURE_START:
                            run_model, texture_model = plan_texture, None
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
                            texture_model=texture_model, texture_start=UPSCALE_TEXTURE_START,
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
