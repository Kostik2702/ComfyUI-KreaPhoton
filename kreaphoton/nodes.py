"""KreaPhoton node definitions (S7). Four v1-scope nodes (docs/04):
Sampler, Sampler (Advanced), Scheduler, Empty Latent. Relative imports only
(hyphenated custom_nodes folder import trap - planning-council H12)."""
import contextlib

import torch

from .presets import (DEFAULT_PRESET, DEFAULT_RESOLUTION_ASPECT, DEFAULT_RESOLUTION_SIZE,
                      GUIDANCE, MANIFOLD_MEAN, MANIFOLD_STD, PRESETS,
                      RESOLUTION_ASPECTS, RESOLUTION_BUCKETS, VARIETY_COND_TAPS,
                      VARIETY_END, VARIETY_LEVELS)
from .noise import slerp_noise
from .sampling import run_sampling
from .save import KreaPhotonSaveImage
from .schedules import ALPHA, build_schedule, refine_schedule

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

CATEGORY = "KreaPhoton"
_ORDER_FROM_SAMPLER_NAME = {"euler": 1, "euler_2m": 2}

_VAE_PREVIEW_TOOLTIP = ("Optional: connect a VAE to show the decoded result as a "
                        "thumbnail on this node (KSampler-Efficient style). Adds one "
                        "VAE decode at the end of sampling; the LATENT output is unchanged.")

PREVIEW_METHODS = ["auto", "latent2rgb", "taesd", "none"]
_PREVIEW_METHOD_TOOLTIP = ("Live per-step preview of the forming image, independent of "
                           "server/frontend preview settings. auto=latent2rgb (instant "
                           "color projection); taesd needs lighttaew2_1 in models/vae_approx "
                           "(falls back to latent2rgb if absent).")

_BLEND_TOOLTIP = ("Composition blend toward seed_b (0 = off, pure seed; 1 = seed_b's "
                  "composition). Spherically interpolates the two seeds' initial noise "
                  "(on-manifold) to walk a coherent composition path between them. Note: "
                  "identity moves with composition on krea2 - a composition explorer "
                  "between two seeds, not fixed-identity variety. Needs seed_b set.")
_SEED_B_TOOLTIP = "Second seed for the composition blend (see `blend`). Ignored when blend<=0."

_DENOISE_TOOLTIP = ("Refine / img2img strength (standard KSampler denoise semantics). "
                    "1.0 = OFF: normal txt2img from the connected latent (default). Below 1.0 "
                    "REFINES the connected latent instead of generating from scratch - feed a "
                    "VAE-encoded image into `latent_image`: 0.2-0.4 = polish/detail, 0.5-0.7 = "
                    "enhance + vary. Uses a clean partial descent (restart/plunge/blend are "
                    "full-txt2img-only and skipped).")


def _blend_noise(model, latent_image, seed, seed_b, blend):
    """Composed initial noise for the composition blend, or None (use default
    per-seed noise) when the blend is off. slerp(noise(seed), noise(seed_b))."""
    if blend <= 0.0 or seed_b < 0:
        return None
    import comfy.sample  # lazy: top-level ComfyUI module, absent in unit tests
    latent5d = comfy.sample.fix_empty_latent_channels(model, latent_image["samples"])
    n_a = comfy.sample.prepare_noise(latent5d, int(seed))
    n_b = comfy.sample.prepare_noise(latent5d, int(seed_b))
    return slerp_noise(n_a, n_b, float(min(1.0, blend)))


@contextlib.contextmanager
def _live_preview(method):
    """KSampler-Efficient trick (efficiency_nodes.py:501/:724): temporarily
    override the GLOBAL preview method for the duration of sampling, restore
    in finally. Node execution happens AFTER the core's per-prompt reset
    (execution.py:727 -> latent_preview.py:136), so the node's own widget
    wins regardless of CLI flags, Manager config, or frontend settings.
    Sequential node execution makes the global mutation safe in practice
    (same long-standing pattern as efficiency-nodes)."""
    try:
        import latent_preview
        from comfy.cli_args import args
    except ImportError:  # plain-assert unit tests without a ComfyUI tree
        yield
        return
    prev = args.preview_method
    args.preview_method = {
        "auto": latent_preview.LatentPreviewMethod.Auto,
        "latent2rgb": latent_preview.LatentPreviewMethod.Latent2RGB,
        "taesd": latent_preview.LatentPreviewMethod.TAESD,
    }.get(method, latent_preview.LatentPreviewMethod.NoPreviews)
    try:
        yield
    finally:
        args.preview_method = prev


def _result_with_preview(out, vae):
    """LATENT result, plus an on-node thumbnail when a VAE is connected:
    decode -> PreviewImage temp files -> ui.images (execution.py ships ui
    for ANY executed node, OUTPUT_NODE not required - verified at
    execution.py:560-575). 5D video-shaped decode flattened to a 4D image
    batch exactly like stock VAEDecode (nodes.py:313-314)."""
    if vae is None:
        return (out,)
    import nodes as comfy_nodes  # lazy: top-level ComfyUI module, absent in unit tests
    images = vae.decode(out["samples"])
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    ui = comfy_nodes.PreviewImage().save_images(images, filename_prefix="KreaPhoton")["ui"]
    return {"ui": ui, "result": (out,)}


class KreaPhotonSampler:
    """All-in-one: seed / preset / variety (docs/04 principle: minimum knobs,
    everything else computed under the hood from presets.py)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "preset": (list(PRESETS.keys()), {"default": DEFAULT_PRESET}),
                "variety": (list(VARIETY_LEVELS.keys()), {"default": "off"}),
                "preview_method": (PREVIEW_METHODS, {"default": "auto",
                                                     "tooltip": _PREVIEW_METHOD_TOOLTIP}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": _DENOISE_TOOLTIP}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "clean_model": ("MODEL", {
                    "tooltip": "Optional: composition phase (sigma > composition_end) runs on "
                               "this clean checkpoint, LoRA identity/detail phase on `model` "
                               "(anti-mutation, ZPhoton-proven pattern; unvalidated on krea2 "
                               "LoRA stacks per docs/04 item 7)."}),
                "vae": ("VAE", {"tooltip": _VAE_PREVIEW_TOOLTIP}),
                "seed_b": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff,
                                   "tooltip": _SEED_B_TOOLTIP}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": _BLEND_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, latent_image, seed, preset, variety,
              preview_method="auto", denoise=1.0, negative=None, clean_model=None, vae=None,
              seed_b=-1, blend=0.0):
        p = PRESETS[preset]
        a_latent, a_cond = VARIETY_LEVELS[variety]
        if denoise < 1.0:
            # refine / img2img: partial clean descent on the input latent (feed a
            # VAE-encoded image into latent_image). Fresh partial noise; restart/
            # plunge and blend are full-txt2img-only and skipped here.
            sigmas = refine_schedule(p["n_steps"], alpha=p["alpha"], denoise=denoise)
            noise = None
        else:
            sigmas = build_schedule(p["n_steps"], alpha=p["alpha"], restart_frac=p["restart_frac"],
                                    sigma_r=p["sigma_r"], plunge=p["plunge"])
            noise = _blend_noise(model, latent_image, seed, seed_b, blend)

        # refine (denoise<1) disables gated-eta: ancestral noise injected onto an
        # already-formed image shows up as a speckle/dust artifact at denoise>=0.35
        # (KREA2-NODES 2026-07-07). Full txt2img keeps the preset's eta0.
        eta0 = 0.0 if denoise < 1.0 else p["eta0"]

        if negative is None:
            guidance_mode = "off"
        else:
            guidance_mode = "window" if GUIDANCE["enabled_by_default"] else "flat"
            # Negative is not None -> mode-dependent NFE cost (docs/04 item 6):
            # pre-E1b "flat" runs full-trajectory cfg (2x NFE the whole run);
            # "window" (post-E1b) only costs extra calls inside the Delta-window.

        with _live_preview(preview_method):
            out = run_sampling(
                model, positive, negative, latent_image, sigmas, seed=seed,
                guidance_mode=guidance_mode, flat_cfg=GUIDANCE["flat_cfg"],
                delta=GUIDANCE["delta"], lo=GUIDANCE["lo"], hi=GUIDANCE["hi"],
                noise=noise,
                contraction=p["contraction"], per_channel_contraction=False,
                manifold_std=MANIFOLD_STD, manifold_mean=MANIFOLD_MEAN,
                detail_amount=p["detail_a"], order=_ORDER_FROM_SAMPLER_NAME[p["sampler"]],
                eta0=eta0, sigma_gate=p["sigma_gate"],
                clean_model=clean_model, composition_end=0.85,
                variety_a_latent=a_latent, variety_a_cond=a_cond, variety_seed=seed,
                variety_end=VARIETY_END, variety_cond_taps=VARIETY_COND_TAPS,
            )
        return _result_with_preview(out, vae)


class KreaPhotonSamplerAdvanced:
    """Same engine, SIGMAS input, every parameter explicit (variety axes
    exposed separately, per docs/04 scope note)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "sigmas": ("SIGMAS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "sampler_order": (list(_ORDER_FROM_SAMPLER_NAME.keys()), {"default": "euler"}),
                "detail_amount": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "detail_start": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_end": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_peak": ("FLOAT", {"default": 0.6, "min": 0.05, "max": 0.95, "step": 0.01}),
                "eta0": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "sigma_gate": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "contraction": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
                "per_channel_contraction": ("BOOLEAN", {"default": False}),
                "guidance_mode": (["off", "flat", "window"], {"default": "off"}),
                "flat_cfg": ("FLOAT", {"default": GUIDANCE["flat_cfg"], "min": 1.0, "max": 4.0, "step": 0.01}),
                "delta": ("FLOAT", {"default": GUIDANCE["delta"], "min": 0.0, "max": 3.0, "step": 0.01}),
                "guidance_lo": ("FLOAT", {"default": GUIDANCE["lo"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_hi": ("FLOAT", {"default": GUIDANCE["hi"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "variety_a_latent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "variety_a_cond": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "variety_end": ("FLOAT", {"default": VARIETY_END, "min": 0.0, "max": 1.0, "step": 0.01}),
                "preview_method": (PREVIEW_METHODS, {"default": "auto",
                                                     "tooltip": _PREVIEW_METHOD_TOOLTIP}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "clean_model": ("MODEL",),
                "composition_end": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "vae": ("VAE", {"tooltip": _VAE_PREVIEW_TOOLTIP}),
                "variety_seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff,
                                         "tooltip": "Seed for the variety realization (lf_recompose "
                                                    "/ cond rotation), decoupled from the generation "
                                                    "seed. -1 = use the generation seed (default, "
                                                    "identical to the simple node). Fix the "
                                                    "generation seed and vary this to explore "
                                                    "variety realizations of the SAME base."}),
                "seed_b": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff,
                                   "tooltip": _SEED_B_TOOLTIP}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": _BLEND_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, latent_image, sigmas, seed, sampler_order,
              detail_amount, detail_start, detail_end, detail_peak, eta0, sigma_gate,
              contraction, per_channel_contraction, guidance_mode, flat_cfg, delta,
              guidance_lo, guidance_hi, variety_a_latent, variety_a_cond, variety_end,
              preview_method="auto", negative=None, clean_model=None, composition_end=0.85,
              vae=None, variety_seed=-1, seed_b=-1, blend=0.0):
        v_seed = seed if variety_seed < 0 else int(variety_seed)
        noise = _blend_noise(model, latent_image, seed, seed_b, blend)
        with _live_preview(preview_method):
            out = run_sampling(
                model, positive, negative, latent_image, sigmas, seed=seed,
                guidance_mode=guidance_mode, flat_cfg=flat_cfg, delta=delta, lo=guidance_lo, hi=guidance_hi,
                noise=noise,
                contraction=contraction, per_channel_contraction=per_channel_contraction,
                manifold_std=MANIFOLD_STD, manifold_mean=MANIFOLD_MEAN,
                detail_amount=detail_amount, detail_start=detail_start, detail_end=detail_end,
                detail_peak=detail_peak, order=_ORDER_FROM_SAMPLER_NAME[sampler_order],
                eta0=eta0, sigma_gate=sigma_gate,
                clean_model=clean_model, composition_end=composition_end,
                variety_a_latent=variety_a_latent, variety_a_cond=variety_a_cond, variety_seed=v_seed,
                variety_end=variety_end, variety_cond_taps=VARIETY_COND_TAPS,
            )
        return _result_with_preview(out, vae)


class KreaPhotonScheduler:
    """SIGMAS generator. restart_frac>0 encodes a restart as an ascending
    jump - the stock SamplerCustom will NOT understand it (use KreaPhoton
    samplers, or KreaPhoton Sampler Advanced)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 12, "min": 1, "max": 64}),
                "alpha": ("FLOAT", {"default": ALPHA, "min": 1.0, "max": 10.0, "step": 0.001,
                                    "tooltip": "Schedule steepness. Stock krea2 shift (mu=1.15) "
                                               "== e^1.15 = 3.158 (default). Higher = softer/less "
                                               "detail but more seed variance; critic-verified stock "
                                               "is not arbitrary (docs/04)."}),
                "restart_frac": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.6, "step": 0.01,
                                          "tooltip": "Fraction of steps spent in the restart "
                                                     "(re-noise) segment. 0 = plain descent, no "
                                                     "restart jump encoded."}),
                "sigma_r": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "plunge": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, steps, alpha, restart_frac, sigma_r, plunge):
        return (build_schedule(steps, alpha=alpha, restart_frac=restart_frac,
                               sigma_r=sigma_r, plunge=plunge),)


class KreaPhotonEmptyLatent:
    """Photo aspect ratios/sizes for krea2 (16ch, 4D output - comfy's own
    fix_empty_latent_channels handles the 4D->5D Wan21 unsqueeze downstream,
    inside the KreaPhoton samplers - planning-council D7/F11)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size": (list(RESOLUTION_BUCKETS.keys()), {"default": DEFAULT_RESOLUTION_SIZE}),
                "aspect": (RESOLUTION_ASPECTS, {"default": DEFAULT_RESOLUTION_ASPECT}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, size, aspect, batch_size):
        width, height = RESOLUTION_BUCKETS[size][aspect]
        latent = torch.zeros([batch_size, 16, height // 8, width // 8])
        return ({"samples": latent},)


# Aesthetic "style directive" prepended to the prompt. krea2's Qwen encoder responds
# strongly to leading instruction text (measured: it shifts composition/subject/polish),
# but is LITERAL - media/layout nouns (magazine, cover, snapshot, poster, photo, print)
# render into the image as text/borders/covers. Presets therefore use mood/quality
# ADJECTIVES only, never media nouns (KREA2-NODES prefix-sweep 2026-07-07).
STYLE_DIRECTIVES = {
    "off":         "",
    "editorial":   "professionally styled, refined color grading, flattering soft studio lighting, ",
    "cinematic":   "cinematic lighting, anamorphic shallow depth of field, dramatic filmic mood, ",
    "natural":     "candid, natural available light, true-to-life, ",
    "custom":      None,   # use custom_directive
}

_STYLE_TOOLTIP = ("Aesthetic directive prepended to the prompt (krea2 responds strongly to "
                  "leading instruction text). off = faithful/literal (safest default). "
                  "editorial / cinematic / natural = calibrated nudges. custom = use "
                  "`custom_directive`. WARNING: krea2 renders media nouns literally - avoid "
                  "'magazine', 'cover', 'snapshot', 'poster' (they leak as text/borders).")
_CUSTOM_DIRECTIVE_TOOLTIP = ("Your own leading style directive (used only when style=custom). "
                             "Use mood/quality adjectives, NOT media nouns - 'magazine cover' / "
                             "'snapshot' render as literal text/frames in the image.")


class KreaPhotonEncode:
    """krea2-native text encode with an aesthetic style directive. Feeds the prompt
    through the correct KREA2_TEMPLATE path (plain clip.tokenize, unlike
    CLIPTextEncodeLumina2 which injects a foreign Lumina2 system prefix), optionally
    prepending a calibrated style directive (KREA2-NODES prefix-sweep)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "style": (list(STYLE_DIRECTIVES.keys()), {"default": "off",
                                                          "tooltip": _STYLE_TOOLTIP}),
            },
            "optional": {
                "custom_directive": ("STRING", {"default": "", "multiline": True,
                                                "tooltip": _CUSTOM_DIRECTIVE_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, clip, text, style, custom_directive=""):
        if style == "custom":
            directive = custom_directive.strip()
            if directive and not directive.endswith((" ", ",", ".", ":", ";")):
                directive += ", "
        else:
            directive = STYLE_DIRECTIVES.get(style) or ""
        tokens = clip.tokenize(directive + text)
        return (clip.encode_from_tokens_scheduled(tokens),)


NODE_CLASS_MAPPINGS.update({
    "KreaPhotonSampler": KreaPhotonSampler,
    "KreaPhotonSamplerAdvanced": KreaPhotonSamplerAdvanced,
    "KreaPhotonScheduler": KreaPhotonScheduler,
    "KreaPhotonEmptyLatent": KreaPhotonEmptyLatent,
    "KreaPhotonEncode": KreaPhotonEncode,
    "KreaPhotonSaveImage": KreaPhotonSaveImage,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "KreaPhotonSampler": "KreaPhoton Sampler",
    "KreaPhotonSamplerAdvanced": "KreaPhoton Sampler (Advanced)",
    "KreaPhotonScheduler": "KreaPhoton Scheduler",
    "KreaPhotonEmptyLatent": "KreaPhoton Empty Latent",
    "KreaPhotonEncode": "KreaPhoton Encode",
    "KreaPhotonSaveImage": "KreaPhoton Save Image",
})
