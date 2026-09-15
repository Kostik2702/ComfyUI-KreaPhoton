"""
KreaPhoton Upscale v2 (2026-09-13): the v1 tiled faithful refine plus the three tier-1
mechanisms from the post-calibration review, as a SEPARATE node so v1 stays as measured:

  1. noise inversion   - instead of descending from a fresh noise draw, the source latent
                         is first inverted along the same schedule (sampling.run_inversion:
                         Euler up the rectified flow) so the descent starts from the noise
                         the source itself would produce. Hypothesis under test: a higher
                         denoise then adds real sub-source detail without rewriting.
  2. empty-tile skipping - tiles with no detail in the source (bokeh, sky, walls) get no
                         model call; their x0 is the source latent (tiling.LatentTilerV2).
  3. grid shift        - the tile grid moves by a seeded random offset at every model call,
                         so blend bands never sit at fixed positions and the overlap can be
                         halved (TILE_V2).

Everything else (pixel base, tiled VAE, presets-only UI, `tune` override, back-projection
to the source, LoRA-plan handling) is shared with v1 through nodes.py helpers.
"""
import torch

from .lora_phase import build_phase_models
from .presets import (ANCHOR, DEFAULT_UPSCALE_V2_PRESET, FIDELITY, GUIDANCE, MANIFOLD_MEAN, MANIFOLD_STD,
                      TILE_V2, UPSCALE_TEXTURE_START, UPSCALE_V2_PRESETS, preset_guidance)
from .sampling import run_inversion, run_sampling
from .schedules import ALPHA, SHIFT, alpha_for_latent, refine_schedule
from .tiling import LFAnchor, LatentTilerV2, activity_map

_V2_PRESET_TOOLTIP = ("v2 presets: polish 0.10 / 6, detail 0.20 / 8, strong 0.35 / 10 steps, contraction 1.0; "
                      "tiles without detail (bokeh, sky, walls) are skipped and the tile grid moves every "
                      "step. Result is back-projected to the source like v1. Noise inversion (start the "
                      "descent from the source's own inverted noise) is available through tune "
                      "{\"invert_steps\": n_steps-1} - measured: no visible gain at twice the time.")
_V2_TUNE_TOOLTIP = ("Calibration override, leave EMPTY. JSON keys: preset keys denoise, n_steps, invert_steps, "
                    "sampler, detail_a, anchor, contraction, skip; tile keys tile, overlap, batch, shift "
                    "(0/1); anchor keys radius, release_sigma; extras alpha, eta0, bp_iters, bp_lock, "
                    "start_noise.")
_V2_PRESET_KEYS = ("denoise", "n_steps", "invert_steps", "sampler", "detail_a", "anchor", "contraction", "skip")
_V2_TILE_KEYS = {"tile": "size", "overlap": "overlap", "batch": "batch", "shift": "shift"}
_V2_ANCHOR_KEYS = ("radius", "release_sigma")
_V2_EXTRA_KEYS = ("alpha", "eta0", "bp_iters", "bp_lock", "start_noise")


def apply_tune_v2(preset: dict, tune):
    from .nodes import _ORDER_FROM_SAMPLER_NAME
    p, tile, anchor, extra = dict(preset), dict(TILE_V2), dict(ANCHOR), {}
    text = (tune or "").strip()
    if not text:
        return p, tile, anchor, extra
    import json
    try:
        over = json.loads(text)
    except ValueError as e:
        raise ValueError(f"KreaPhoton Upscale v2: tune must be a JSON object ({e})")
    if not isinstance(over, dict):
        raise ValueError("KreaPhoton Upscale v2: tune must be a JSON object")
    for k, v in over.items():
        if k in _V2_PRESET_KEYS:
            p[k] = v
        elif k in _V2_TILE_KEYS:
            tile[_V2_TILE_KEYS[k]] = bool(v) if k == "shift" else int(v)
        elif k in _V2_ANCHOR_KEYS:
            anchor[k] = float(v)
        elif k in _V2_EXTRA_KEYS:
            extra[k] = v
        else:
            raise ValueError(f"KreaPhoton Upscale v2: unknown tune key {k!r}")
    if p["sampler"] not in _ORDER_FROM_SAMPLER_NAME:
        raise ValueError(f"KreaPhoton Upscale v2: tune sampler must be one of {list(_ORDER_FROM_SAMPLER_NAME)}")
    p["n_steps"] = int(p["n_steps"])
    p["invert_steps"] = int(p["invert_steps"])
    return p, tile, anchor, extra


def inversion_sigmas(sigmas_desc: torch.Tensor, invert_steps: int) -> torch.Tensor:
    """Ascending schedule for the inversion from the descent schedule: the descent's own
    grid (minus the final 0) reversed when invert_steps == n_steps - 1 (the n non-zero
    sigmas of an n-step descent span n - 1 inversion steps - the symmetric choice, the
    only one whose Euler round trip is exact up to integration error), otherwise a
    uniform resample to invert_steps + 1 points. Starts at the smallest non-zero sigma of
    the descent, ends at its first sigma."""
    desc = [float(v) for v in sigmas_desc if float(v) > 1e-6]
    asc = list(reversed(desc))
    k = int(invert_steps)
    if k <= 0:
        return torch.tensor([], dtype=torch.float32)
    if k + 1 == len(asc):
        return torch.tensor(asc, dtype=torch.float32)
    lo, hi = asc[0], asc[-1]
    return torch.tensor([lo + (hi - lo) * i / k for i in range(k + 1)], dtype=torch.float32)


class KreaPhotonUpscaleV2:
    """See the module docstring."""

    @classmethod
    def INPUT_TYPES(cls):
        from .nodes import _UPSCALE_MODEL_TOOLTIP, _UPSCALE_NEGATIVE_TOOLTIP, _UPSCALE_SCALE_TOOLTIP
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "preset": (list(UPSCALE_V2_PRESETS.keys()), {"default": DEFAULT_UPSCALE_V2_PRESET,
                                                             "tooltip": _V2_PRESET_TOOLTIP}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.25, "max": 2.0, "step": 0.05,
                                    "tooltip": _UPSCALE_SCALE_TOOLTIP}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": _UPSCALE_NEGATIVE_TOOLTIP}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": _UPSCALE_MODEL_TOOLTIP}),
                "tune": ("STRING", {"default": "", "multiline": False, "tooltip": _V2_TUNE_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "KreaPhoton"

    def upscale(self, model, positive, image, vae, seed, preset, scale, negative=None, upscale_model=None,
                tune=""):
        from .nodes import (LATENT_PX, _ORDER_FROM_SAMPLER_NAME, _decode_tiled, _encode_tiled, _upscale_pixels,
                            back_project)
        p, tile, anchor_cfg, extra = apply_tune_v2(UPSCALE_V2_PRESETS[preset], tune)
        _, model, plan_texture = build_phase_models(model)
        guidance_mode, flat_cfg = preset_guidance(p, negative is not None)
        lat_tile = tile["size"] // LATENT_PX
        lat_overlap = tile["overlap"] // LATENT_PX
        outs = []
        for b in range(int(image.shape[0])):
            pixels = _upscale_pixels(image[b:b + 1], float(scale), upscale_model)
            z = _encode_tiled(vae, pixels)
            h, w = int(z.shape[-2]), int(z.shape[-1])
            run_model = model
            sigmas = None
            alpha = alpha_for_latent(min(lat_tile, h), min(lat_tile, w), ALPHA) if SHIFT["resolution_aware"] else ALPHA
            if extra.get("alpha"):
                alpha = float(extra["alpha"])
            sigmas = refine_schedule(p["n_steps"], alpha=alpha, denoise=p["denoise"])
            texture_model = plan_texture
            if plan_texture is not None and float(sigmas[0]) <= UPSCALE_TEXTURE_START:
                run_model, texture_model = plan_texture, None
            z_ref = run_model.model.process_latent_in(z)
            # activity at latent resolution: one cell per activity_block px -> repeat to latent px
            act = activity_map(pixels, TILE_V2["activity_block"])
            rep = TILE_V2["activity_block"] // LATENT_PX
            act = act.repeat_interleave(rep, 0).repeat_interleave(rep, 1)[:h, :w]
            if act.shape[0] < h or act.shape[1] < w:
                pad = torch.zeros((h, w), dtype=act.dtype)
                pad[:act.shape[0], :act.shape[1]] = act
                act = pad
            tiler = LatentTilerV2(min(lat_tile, h), min(lat_tile, w), lat_overlap, tile["batch"],
                                  shift_seed=int(seed) + b, shift=bool(tile["shift"]),
                                  activity=act, skip_threshold=float(p["skip"]), fallback=z_ref)
            anchor = LFAnchor(z_ref, anchor_cfg["radius"], p["anchor"], float(sigmas[0]),
                              anchor_cfg["release_sigma"])
            latent_in = {"samples": z}
            add_noise = True
            k = int(p["invert_steps"])
            if k > 0:
                asc = inversion_sigmas(sigmas, k)
                latent_in = run_inversion(run_model, positive, negative, latent_in, asc, seed=int(seed) + b,
                                          guidance_mode=guidance_mode, flat_cfg=flat_cfg,
                                          delta=GUIDANCE["delta"], lo=GUIDANCE["lo"], hi=GUIDANCE["hi"],
                                          start_noise=float(extra.get("start_noise", 1.0)),
                                          tiler=tiler, x0_hook=anchor)
                add_noise = False
            out = run_sampling(
                run_model, positive, negative, latent_in, sigmas, seed=int(seed) + b,
                guidance_mode=guidance_mode, flat_cfg=flat_cfg,
                delta=GUIDANCE["delta"], lo=GUIDANCE["lo"], hi=GUIDANCE["hi"],
                add_noise=add_noise,
                contraction=p["contraction"], per_channel_contraction=False,
                manifold_std=MANIFOLD_STD, manifold_mean=MANIFOLD_MEAN,
                detail_amount=p["detail_a"], order=_ORDER_FROM_SAMPLER_NAME[p["sampler"]],
                eta0=float(extra.get("eta0", 0.0)), sigma_gate=0.10,
                texture_model=texture_model, texture_start=UPSCALE_TEXTURE_START,
                tiler=tiler, x0_hook=anchor,
            )
            print("[KreaPhoton Upscale v2] %s x%.2f: %d tile forwards, %d empty tiles skipped, %d model calls"
                  % (preset, float(scale), tiler.calls, tiler.skipped, tiler.step))
            decoded = _decode_tiled(vae, out["samples"])
            decoded = back_project(decoded, image[b:b + 1, ..., :3],
                                   int(extra.get("bp_iters", FIDELITY["bp_iters"])),
                                   float(extra.get("bp_lock", p.get("bp_lock", 1.0))))
            outs.append(decoded)
        return (torch.cat(outs, dim=0),)
