"""KreaPhoton node definitions (S7). Four v1-scope nodes (docs/04):
Sampler, Sampler (Advanced), Scheduler, Empty Latent. Relative imports only
(hyphenated custom_nodes folder import trap - planning-council H12)."""
import contextlib

import torch

from .presets import (ANCHOR, COHERENCE, COHERENCE_OPTIONS, DEFAULT_PRESET, DEFAULT_RESOLUTION_ASPECT,
                      DEFAULT_RESOLUTION_SIZE, DEFAULT_UPSCALE_PRESET,
                      DLSS_PRESETS, FIDELITY, GUIDANCE, LORA_PHASES, MANIFOLD_MEAN, MANIFOLD_STD, PAG,
                      PRESETS, RESOLUTION_ASPECTS, RESOLUTION_BUCKETS, RESTART_ENHANCE_OPTIONS,
                      TILE, UPSCALE_PRESETS, UPSCALE_TEXTURE_START,
                      VARIETY_COND_TAPS, VARIETY_END, VARIETY_LEVELS, preset_guidance)
from .enhance import make_dlss_restart_hook
from .lora_phase import add_to_plan, build_phase_models
from .noise import slerp_noise
from .sampling import run_sampling
from .save import KreaPhotonSaveImage
from .schedules import ALPHA, SHIFT, alpha_for_latent, build_schedule, refine_schedule
from .tiling import LatentTiler, LFAnchor

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

_X0_EXTRAP_TOOLTIP = ("Terminal x0-trajectory extrapolation (0 = off). At the last step the model's "
                      "x0 estimate is extrapolated linearly past the last evaluation toward sigma=0 "
                      "(denoised + w*f*(denoised - previous denoised), f = last step ratio, capped at 2) "
                      "- sharper micro-detail / local contrast, same mechanism as the Krea 2 Turbo "
                      "Preset Sampler's zero_extrapolation. Skipped on plunge steps. UNCALIBRATED "
                      "on KreaPhoton grids: try 0.3-0.5, 1.0 can over-sharpen skin/freckles.")

_DENOISE_TOOLTIP = ("Refine / img2img strength (standard KSampler denoise semantics). "
                    "1.0 = OFF: normal txt2img from the connected latent (default). Below 1.0 "
                    "REFINES the connected latent instead of generating from scratch - feed a "
                    "VAE-encoded image into `latent_image`: 0.2-0.4 = polish/detail, 0.5-0.7 = "
                    "enhance + vary. Uses a clean partial descent (restart/plunge/blend are "
                    "full-txt2img-only and skipped).")


_RESTART_ENHANCE_TOOLTIP = ("EXPERIMENTAL (v1.4): at the restart boundary the plunge x0 is decoded, "
                            "run through NVIDIA DLSS 5 Photoreal Enhance V2 (custom node pack "
                            "ComfyUI-dlss-enhancer must be installed), re-encoded and re-noised into "
                            "the texture phase - the last steps re-synthesise texture around the "
                            "enhanced lighting/materials. Needs `vae` connected. Adds one VAE decode, "
                            "one encode and one DLSS pass per image.")
_TEXTURE_MODEL_TOOLTIP = ("Optional phase model (v1.4): the restart / texture phase (sigma <= 0.65) runs "
                          "on THIS model, the identity phase on `model`, the composition phase on "
                          "`clean_model` if connected. LoRA phase scheduling that works on any checkpoint "
                          "(incl. int8/fp8 quantized): e.g. model = base + character LoRA, clean_model = "
                          "base, texture_model = base + character + style LoRAs. Each phase switch costs "
                          "one LoRA re-patch (a few seconds on a 12B model).")
_COHERENCE_TOOLTIP = ("Power-Nodes-derived coherence tools (v1.4). jump = one-time jump-back on the "
                      "composition step (the model sees the state cleaner than declared and commits "
                      "harder to structure; free). self_refine = after the plunge draft, re-noise to "
                      "sigma 0.85 and re-descend 4 steps, plunge again, then the usual restart: the "
                      "model re-decides faces/bodies/clothing with the whole draft as a prior (+4 model "
                      "calls). Use self_refine when LoRA stacks or crowded scenes come out incoherent.")
_PAG_SCALE_TOOLTIP = ("EXPERIMENTAL (v1.4): perturbed-attention guidance. Inside [pag_lo, pag_hi] a "
                      "second conditional forward runs with identity self-attention in pag_blocks and "
                      "denoised += pag_scale * (cond - perturbed): structure/anatomy guidance that "
                      "needs no negative. 0 = off. +1 model call per step inside the window.")


def _latent_hw(latent_image):
    s = latent_image["samples"]
    return int(s.shape[-2]), int(s.shape[-1])


def _preset_alpha(p, latent_image):
    """Resolution-aware shift (v1.4): the preset alpha is the e^1.15 cap, smaller
    grids get the Krea 2 canonical (lower) alpha. L/XL tiers are unchanged."""
    if not SHIFT["resolution_aware"]:
        return p["alpha"]
    h, w = _latent_hw(latent_image)
    return alpha_for_latent(h, w, p["alpha"])


def _restart_hook_for(restart_enhance, vae):
    if restart_enhance is None or restart_enhance == "off":
        return None
    if restart_enhance not in DLSS_PRESETS:
        raise ValueError(f"KreaPhoton: unknown restart_enhance {restart_enhance!r}")
    return make_dlss_restart_hook(vae, DLSS_PRESETS[restart_enhance])


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


def _explicit_phase_model(explicit, plan_model, phase):
    """Phase model for the composition ("clean") or texture slot.

    An explicitly connected clean_model / texture_model wins over the plan on
    `model`; but that input may itself be a KreaPhoton LoRA Phase chain - a
    model that CARRIES a plan and patches nothing (lora_phase.add_to_plan). Handed
    to run_sampling raw it is the bare checkpoint: the restart segment then runs
    without the character LoRA and the face drifts on the last steps ([EDITORAL]
    KREA6, 2026-09-15). So the explicit input is expanded for ITS phase - its
    composition / texture patcher, or its identity patcher when that phase shares
    the identity LoRA set (no split). A plan-less input is returned unchanged."""
    if explicit is None:
        return plan_model
    clean, identity, texture = build_phase_models(explicit)
    own = clean if phase == "composition" else texture
    return own if own is not None else identity


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
                               "LoRA stacks per docs/04 item 7). EXPERIMENTAL: this is the one "
                               "remaining two-lifecycle split and it forces eta0=0."}),
                "vae": ("VAE", {"tooltip": _VAE_PREVIEW_TOOLTIP}),
                "seed_b": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff,
                                   "tooltip": _SEED_B_TOOLTIP}),
                "blend": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": _BLEND_TOOLTIP}),
                "restart_enhance": (RESTART_ENHANCE_OPTIONS, {"default": "off",
                                                              "tooltip": _RESTART_ENHANCE_TOOLTIP}),
                "texture_model": ("MODEL", {"tooltip": _TEXTURE_MODEL_TOOLTIP}),
                "coherence": (COHERENCE_OPTIONS, {"default": "off", "tooltip": _COHERENCE_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, latent_image, seed, preset, variety,
              preview_method="auto", denoise=1.0, negative=None, clean_model=None, vae=None,
              seed_b=-1, blend=0.0, restart_enhance="off", texture_model=None, coherence="off"):
        p = PRESETS[preset]
        a_latent, a_cond = VARIETY_LEVELS[variety]
        alpha = _preset_alpha(p, latent_image)
        use_jump = coherence in ("jump", "jump+self_refine")
        refine_steps = COHERENCE["refine_steps"] if coherence in ("self_refine", "jump+self_refine") else 0
        # LoRA phase plan (KreaPhoton LoRA Phase nodes upstream) -> phase models;
        # explicitly connected clean_model / texture_model win for their phase
        plan_clean, model, plan_texture = build_phase_models(model)
        clean_model = _explicit_phase_model(clean_model, plan_clean, "composition")
        texture_model = _explicit_phase_model(texture_model, plan_texture, "texture")
        if denoise < 1.0:
            # refine / img2img: partial clean descent on the input latent (feed a
            # VAE-encoded image into latent_image). Fresh partial noise; restart/
            # plunge and blend are full-txt2img-only and skipped here.
            sigmas = refine_schedule(p["n_steps"], alpha=alpha, denoise=denoise)
            noise = None
        else:
            sigmas = build_schedule(p["n_steps"], alpha=alpha, restart_frac=p["restart_frac"],
                                    sigma_r=p["sigma_r"], plunge=p["plunge"],
                                    refine_steps=refine_steps if p["plunge"] else 0,
                                    refine_sigma=COHERENCE["refine_sigma"])
            noise = _blend_noise(model, latent_image, seed, seed_b, blend)
        restart_hook = _restart_hook_for(restart_enhance, vae)

        # refine (denoise<1) disables gated-eta: ancestral noise injected onto an
        # already-formed image shows up as a speckle/dust artifact at denoise>=0.35
        # (KREA2-NODES 2026-07-07). Full txt2img keeps the preset's eta0.
        eta0 = 0.0 if denoise < 1.0 else p["eta0"]

        # Guidance is a preset contract (presets.preset_guidance): Turbo presets
        # run the M6 window only with a connected negative (mode-dependent NFE
        # cost - docs/04 item 6), RAW runs real full-trajectory CFG always.
        guidance_mode, flat_cfg = preset_guidance(p, negative is not None)

        with _live_preview(preview_method):
            out = run_sampling(
                model, positive, negative, latent_image, sigmas, seed=seed,
                guidance_mode=guidance_mode, flat_cfg=flat_cfg,
                delta=GUIDANCE["delta"], lo=GUIDANCE["lo"], hi=GUIDANCE["hi"],
                noise=noise,
                contraction=p["contraction"], per_channel_contraction=False,
                manifold_std=MANIFOLD_STD, manifold_mean=MANIFOLD_MEAN,
                detail_amount=p["detail_a"], order=_ORDER_FROM_SAMPLER_NAME[p["sampler"]],
                eta0=eta0, sigma_gate=p["sigma_gate"],
                x0_extrapolation=p["x0_extrapolation"],
                clean_model=clean_model, composition_end=0.85,
                variety_a_latent=a_latent, variety_a_cond=a_cond, variety_seed=seed,
                variety_end=VARIETY_END, variety_cond_taps=VARIETY_COND_TAPS,
                restart_hook=restart_hook, texture_model=texture_model,
                coherence_jump=COHERENCE["jump"] if (use_jump and denoise >= 1.0) else 0.0,
                coherence_jump_sigma=COHERENCE["jump_sigma"],
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
                "contraction": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.5, "step": 0.01,
                                          "tooltip": "Initial noise amplitude scale. 1.0 = stock unit noise; "
                                                     "presets use 0.70 (V5: cleaner shadows, more inter-seed "
                                                     "diversity, softer photographic look). Above 1.0 = the "
                                                     "'intensity' direction of Z-Image Power Nodes: more contrast, "
                                                     "sharper edges, more saturation (up to 1.4 there). Photo "
                                                     "styles usually prefer <= 1.0."}),
                "per_channel_contraction": ("BOOLEAN", {"default": False}),
                "guidance_mode": (["off", "flat", "window"], {"default": "off"}),
                "flat_cfg": ("FLOAT", {"default": GUIDANCE["flat_cfg"], "min": 1.0, "max": 4.0, "step": 0.01}),
                "delta": ("FLOAT", {"default": GUIDANCE["delta"], "min": 0.0, "max": 3.0, "step": 0.01,
                                    "tooltip": "Window guidance strength: g(sigma) = 1 + delta * smoothstep "
                                               "in [guidance_lo, guidance_hi]. 1.25 V1-validated; 1.5 "
                                               "hallucinates. When variety_a_latent > 0 the effective delta "
                                               "is delta * (1 - 0.45 * a_latent) - variety consumes the "
                                               "hallucination budget (V16b)."}),
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
                "x0_extrapolation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                               "tooltip": _X0_EXTRAP_TOOLTIP}),
                "guidance_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                               "tooltip": "CFG-rescale phi (Lin et al. 2023) applied only "
                                                          "on steps where the guidance scale is > 1: the "
                                                          "guided x0 is re-normalised to the conditional "
                                                          "prediction's std, damping the saturated / glowing "
                                                          "blobs of over-guidance. 0 = off. Try 0.5-0.7 when "
                                                          "combining a negative with variety."}),
                "pag_scale": ("FLOAT", {"default": PAG["scale"], "min": 0.0, "max": 5.0, "step": 0.05,
                                        "tooltip": _PAG_SCALE_TOOLTIP}),
                "pag_lo": ("FLOAT", {"default": PAG["lo"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "pag_hi": ("FLOAT", {"default": PAG["hi"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "pag_blocks": ("STRING", {"default": PAG["blocks"],
                                          "tooltip": "DiT blocks (of 28) whose self-attention is perturbed, "
                                                     "e.g. '8-15' or '6,9,12'."}),
                "restart_enhance": (RESTART_ENHANCE_OPTIONS, {"default": "off",
                                                              "tooltip": _RESTART_ENHANCE_TOOLTIP}),
                "texture_model": ("MODEL", {"tooltip": _TEXTURE_MODEL_TOOLTIP}),
                "texture_start": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "Sigma at/below which texture_model takes over "
                                                       "(0.65 = exactly the restart segment on the "
                                                       "calibrated grids)."}),
                "coherence_jump": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.6, "step": 0.01,
                                             "tooltip": "One-time jump-back strength on the first step at/"
                                                        "below coherence_jump_sigma (0.19 = Power Nodes' "
                                                        "0.920->0.935 boundary). 0 = off. Self-refine lives "
                                                        "in the Scheduler (refine_steps)."}),
                "coherence_jump_sigma": ("FLOAT", {"default": COHERENCE["jump_sigma"], "min": 0.5, "max": 1.0,
                                                   "step": 0.01}),
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
              vae=None, variety_seed=-1, seed_b=-1, blend=0.0, x0_extrapolation=0.0,
              guidance_rescale=0.0, pag_scale=0.0, pag_lo=PAG["lo"], pag_hi=PAG["hi"],
              pag_blocks=PAG["blocks"], restart_enhance="off", texture_model=None, texture_start=0.65,
              coherence_jump=0.0, coherence_jump_sigma=COHERENCE["jump_sigma"]):
        v_seed = seed if variety_seed < 0 else int(variety_seed)
        plan_clean, model, plan_texture = build_phase_models(model)
        clean_model = _explicit_phase_model(clean_model, plan_clean, "composition")
        texture_model = _explicit_phase_model(texture_model, plan_texture, "texture")
        noise = _blend_noise(model, latent_image, seed, seed_b, blend)
        restart_hook = _restart_hook_for(restart_enhance, vae)
        with _live_preview(preview_method):
            out = run_sampling(
                model, positive, negative, latent_image, sigmas, seed=seed,
                guidance_mode=guidance_mode, flat_cfg=flat_cfg, delta=delta, lo=guidance_lo, hi=guidance_hi,
                guidance_rescale=guidance_rescale,
                pag_scale=pag_scale, pag_lo=pag_lo, pag_hi=pag_hi, pag_blocks=pag_blocks,
                restart_hook=restart_hook, texture_model=texture_model, texture_start=texture_start,
                coherence_jump=coherence_jump, coherence_jump_sigma=coherence_jump_sigma,
                noise=noise,
                contraction=contraction, per_channel_contraction=per_channel_contraction,
                manifold_std=MANIFOLD_STD, manifold_mean=MANIFOLD_MEAN,
                detail_amount=detail_amount, detail_start=detail_start, detail_end=detail_end,
                detail_peak=detail_peak, order=_ORDER_FROM_SAMPLER_NAME[sampler_order],
                eta0=eta0, sigma_gate=sigma_gate, x0_extrapolation=x0_extrapolation,
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
                                                     "restart jump encoded. Dropped automatically "
                                                     "below 4 steps (model calls always == steps)."}),
                "sigma_r": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "plunge": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "latent": ("LATENT", {"tooltip": "Optional: connect the latent to make alpha resolution-"
                                                 "aware (Krea 2 canonical shift: mu 0.5 at 256 tokens -> "
                                                 "1.15 at 6400; alpha = min(alpha, e^mu)). L/XL tiers are "
                                                 "unchanged, 1 MP grids get the softer canonical schedule."}),
                "refine_steps": ("INT", {"default": 0, "min": 0, "max": 12,
                                         "tooltip": "Self-refine pass (v1.4): after the plunge draft, re-noise "
                                                    "to refine_sigma and descend this many extra model calls "
                                                    "to the plunge floor, plunge again, then the restart. "
                                                    "Needs plunge=true. 0 = off. Try 4."}),
                "refine_sigma": ("FLOAT", {"default": COHERENCE["refine_sigma"], "min": 0.76, "max": 0.99,
                                           "step": 0.01,
                                           "tooltip": "Re-noise level of the self-refine pass. 0.85 keeps the "
                                                      "draft's layout (~img2img 0.6) and re-renders bodies / "
                                                      "faces / clothing; 0.93+ lets the model re-decide more."}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, steps, alpha, restart_frac, sigma_r, plunge, latent=None, refine_steps=0,
              refine_sigma=COHERENCE["refine_sigma"]):
        if latent is not None:
            h, w = _latent_hw(latent)
            alpha = alpha_for_latent(h, w, alpha)
        return (build_schedule(steps, alpha=alpha, restart_frac=restart_frac,
                               sigma_r=sigma_r, plunge=plunge,
                               refine_steps=refine_steps if plunge else 0, refine_sigma=refine_sigma),)


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


class KreaPhotonLoraPhase:
    """LoRA phase scheduling (v1.4.1): sits on the MODEL line like a LoRA loader
    but patches nothing - it records {lora, strength, phase} in a plan carried by
    the model. A KreaPhoton sampler expands the plan into composition / identity
    / texture phase models with comfy's ordinary LoRA patching (works on int8 /
    fp8-quantized checkpoints, unlike weight hooks) and runs its validated
    multi-segment split. Chain one node per LoRA. Model-only: the text encoder
    is not patched."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            loras = folder_paths.get_filename_list("loras")
        except Exception:  # unit tests without a ComfyUI tree
            loras = []
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (loras,),
                "strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "phase": (list(LORA_PHASES.keys()), {
                    "default": "identity",
                    "tooltip": "Phase of the KreaPhoton schedule in which this LoRA is active: "
                               "composition = layout/pose (sigma 1.0-0.85); identity = face/body/"
                               "clothing (0.85-0, incl. the texture phase); texture = restart segment "
                               "skin/fabric (0.65-0); all = classic LoRA loader. Character LoRA -> "
                               "identity (keeps the face, stops it steering the layout); style LoRA "
                               "-> texture. Applied only by KreaPhoton samplers; other samplers see "
                               "the unpatched model."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model, lora_name, strength, phase):
        return (add_to_plan(model, lora_name, strength, phase),)


LATENT_PX = 8          # Wan 2.1 VAE spatial compression (the sampler contract: 16ch /8 latent)
ALIGN_PX = 16          # target image dims: latent /8 x DiT patch 2 -> even latent dims

_UPSCALE_PRESET_TOOLTIP = ("How much the diffusion model may re-texture below the source scale; the "
                           "result always reproduces the source exactly when downscaled (back-projection), "
                           "so layout, tone, faces and objects never change. polish: denoise 0.06, 4 steps - "
                           "a clean, slightly crisper upscale. detail (default): 0.12, 6 - natural skin / "
                           "fabric / hair micro-texture, edges sharpened. strong: 0.25, 8 - more model "
                           "texture, a little more grain; source artefacts (e.g. crosshatch skin) get "
                           "emphasised too. None of them can re-draw a deformed object - that needs a "
                           "denoise that rewrites the whole frame (measured), so it is not offered.")
_UPSCALE_SCALE_TOOLTIP = ("Pixel upscale factor before the tiled refine (Lanczos, or upscale_model when "
                          "connected); 1.5 and 2.0 are the calibrated points. Target dims are rounded to a "
                          "multiple of 16 px. There is no 1.0: with the source-consistency guarantee a "
                          "same-size run can only be a no-op or a rewrite (measured on an 11 MP frame).")
_UPSCALE_NEGATIVE_TOOLTIP = ("Optional. Enables the sigma-window guidance (0.7-0.9) exactly like the "
                             "Sampler - but every upscale preset starts BELOW that window (sigma "
                             "0.38-0.66), so on Krea 2 Turbo this input has no effect today. Kept for "
                             "parity / future presets.")
_UPSCALE_MODEL_TOOLTIP = ("Optional ESRGAN-class upscale model for the pixel base (its own factor, then "
                          "resized to `scale`). Without it: Lanczos. The diffusion refine supplies "
                          "the detail either way; a model base mainly helps at scale 2.0 on soft sources.")


_UPSCALE_TUNE_TOOLTIP = ("Calibration override, leave EMPTY for normal use. A JSON object whose keys "
                         "replace preset / constant values for this run: preset keys denoise, n_steps, "
                         "sampler, detail_a, anchor, contraction; tile keys tile, overlap, batch (px); "
                         "anchor keys radius, release_sigma; extras alpha, eta0. Example: "
                         "{\"denoise\": 0.25, \"anchor\": 1.0, \"detail_a\": 0}")
_TUNE_PRESET_KEYS = ("denoise", "n_steps", "sampler", "detail_a", "anchor", "contraction")
_TUNE_TILE_KEYS = {"tile": "size", "overlap": "overlap", "batch": "batch"}
_TUNE_ANCHOR_KEYS = ("radius", "release_sigma")
_TUNE_EXTRA_KEYS = ("alpha", "eta0", "bp_iters", "bp_lock")


def _apply_tune(preset: dict, tune):
    """(preset, tile, anchor, extra) with the JSON overrides of `tune` applied.
    Empty / whitespace -> the untouched preset and presets.TILE / ANCHOR."""
    p, tile, anchor, extra = dict(preset), dict(TILE), dict(ANCHOR), {}
    text = (tune or "").strip()
    if not text:
        return p, tile, anchor, extra
    import json
    try:
        over = json.loads(text)
    except ValueError as e:
        raise ValueError(f"KreaPhoton Upscale: tune must be a JSON object ({e})")
    if not isinstance(over, dict):
        raise ValueError("KreaPhoton Upscale: tune must be a JSON object")
    for k, v in over.items():
        if k in _TUNE_PRESET_KEYS:
            p[k] = v
        elif k in _TUNE_TILE_KEYS:
            tile[_TUNE_TILE_KEYS[k]] = int(v)
        elif k in _TUNE_ANCHOR_KEYS:
            anchor[k] = float(v)
        elif k in _TUNE_EXTRA_KEYS:
            extra[k] = v
        else:
            raise ValueError(f"KreaPhoton Upscale: unknown tune key {k!r}")
    if p["sampler"] not in _ORDER_FROM_SAMPLER_NAME:
        raise ValueError(f"KreaPhoton Upscale: tune sampler must be one of {list(_ORDER_FROM_SAMPLER_NAME)}")
    p["n_steps"] = int(p["n_steps"])
    return p, tile, anchor, extra


def _target_size(h: int, w: int, scale: float):
    th = max(ALIGN_PX, int(round(h * scale / ALIGN_PX)) * ALIGN_PX)
    tw = max(ALIGN_PX, int(round(w * scale / ALIGN_PX)) * ALIGN_PX)
    return th, tw


def _upscale_pixels(image, scale: float, upscale_model=None):
    """(1, H, W, C) float 0..1 -> (1, th, tw, 3) at the aligned target size."""
    import comfy.utils  # lazy: top-level ComfyUI module
    h, w = int(image.shape[1]), int(image.shape[2])
    th, tw = _target_size(h, w, scale)
    img = image[..., :3]
    if upscale_model is not None:
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel  # core node, lazy
        # v3 node: `upscale` is a classmethod returning io.NodeOutput (indexable, verified
        # comfy_api/latest/_io.py NodeOutput.__getitem__); no instance needed
        img = ImageUpscaleWithModel.upscale(upscale_model, img)[0][..., :3]
    if (int(img.shape[1]), int(img.shape[2])) != (th, tw):
        img = comfy.utils.common_upscale(img.movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1)
    return img.clamp(0.0, 1.0)


def _encode_tiled(vae, pixels):
    z = vae.encode_tiled(pixels, tile_x=TILE["vae_tile"], tile_y=TILE["vae_tile"], overlap=TILE["vae_overlap"])
    if z.ndim == 4:
        z = z.unsqueeze(2)
    return z


def back_project(result, source, iters: int, lock_scale: float = 1.0):
    """Source-consistency (v1.5 live calibration, 2026-09-13): iterated back-projection
    result += Up(source - Down(result)). Down = area average to the lock grid, Up =
    bicubic. After it the result reproduces the SOURCE exactly at the lock scale -
    tone drift, re-lit surfaces and re-drawn mid-scale structure (the failure modes
    measured on S08: MAE 5-10 / 255 even at denoise 0.12) are all removed, and only the
    detail the model added BELOW that scale survives. (B,H,W,C) float 0..1 in, same
    out; iters 0 = off.

    lock_scale 1.0 = the source pixel grid (strict fidelity: nothing the source shows
    can change) - every preset. > 1 locks a coarser grid (source / lock_scale): structure
    above lock_scale source px is the source's, finer content is the model's. Kept for
    the `tune` input only: lock 2 was measured worse (smeared edges, amplified source
    artefacts) - see presets.py."""
    import torch.nn.functional as F
    if int(iters) <= 0:
        return result
    lock = max(1.0, float(lock_scale))
    if lock == 1.0 and result.shape[1:3] == source.shape[1:3]:
        return result
    r = result.movedim(-1, 1).float()
    s = source.movedim(-1, 1).float().to(r.device)
    size = (max(1, int(round(s.shape[-2] / lock))), max(1, int(round(s.shape[-1] / lock))))
    # Down = antialiased bilinear, NOT "area": adaptive average pooling at a non-integer
    # factor (x1.5, or x2 after the 16-px rounding) bins unevenly and leaves a periodic
    # residual that shows as a fine grid on skin / leather (batch-2 survey, 2026-09-13:
    # residual 0.0054 vs 0.0002 at x1.5 on a smooth field; PSNR 36 vs 57 dB on frames).
    def down(t):
        return F.interpolate(t, size=size, mode="bilinear", antialias=True, align_corners=False)
    if lock != 1.0:
        s = down(s)
    for _ in range(int(iters)):
        err = F.interpolate(s - down(r), size=(int(r.shape[-2]), int(r.shape[-1])), mode="bicubic",
                            align_corners=False)
        r = (r + err).clamp(0.0, 1.0)
    return r.movedim(1, -1).to(result.dtype)


def _decode_tiled(vae, samples):
    c = int(vae.spacial_compression_decode())
    images = vae.decode_tiled(samples, tile_x=TILE["vae_tile"] // c, tile_y=TILE["vae_tile"] // c,
                              overlap=TILE["vae_overlap"] // c)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


class KreaPhotonUpscale:
    """Tiled detail refine of an upscaled image (v1.5, docs/07). Pixel upscale ->
    one tiled VAE encode -> KreaPhoton refine on the whole latent with the model
    called on overlapping 1024 px tiles that are re-blended every step
    (tiling.LatentTiler) and the low-frequency band anchored to the source
    (tiling.LFAnchor) -> one tiled VAE decode. Presets only; tile geometry and
    anchor constants live in presets.TILE / ANCHOR. Works with the LoRA Phase
    plan: the refine starts below sigma 0.85 so composition-phase LoRAs never
    take part; identity/texture LoRAs do."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "preset": (list(UPSCALE_PRESETS.keys()), {"default": DEFAULT_UPSCALE_PRESET,
                                                          "tooltip": _UPSCALE_PRESET_TOOLTIP}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.25, "max": 2.0, "step": 0.05,
                                    "tooltip": _UPSCALE_SCALE_TOOLTIP}),
            },
            "optional": {
                "negative": ("CONDITIONING", {"tooltip": _UPSCALE_NEGATIVE_TOOLTIP}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": _UPSCALE_MODEL_TOOLTIP}),
                "tune": ("STRING", {"default": "", "multiline": False, "tooltip": _UPSCALE_TUNE_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, model, positive, image, vae, seed, preset, scale, negative=None, upscale_model=None,
                tune=""):
        p, tile, anchor_cfg, extra = _apply_tune(UPSCALE_PRESETS[preset], tune)
        # LoRA plan -> phase models. No composition phase here (refine starts below
        # 0.85), so the clean model is unused; the texture model owns sigma <= 0.65.
        _, model, plan_texture = build_phase_models(model)
        guidance_mode, flat_cfg = preset_guidance(p, negative is not None)
        lat_tile = tile["size"] // LATENT_PX
        lat_overlap = tile["overlap"] // LATENT_PX
        outs = []
        for b in range(int(image.shape[0])):
            pixels = _upscale_pixels(image[b:b + 1], float(scale), upscale_model)
            z = _encode_tiled(vae, pixels)
            h, w = int(z.shape[-2]), int(z.shape[-1])
            tiler = LatentTiler(min(lat_tile, h), min(lat_tile, w), lat_overlap, tile["batch"])
            # schedule steepness for the grid the model actually sees (the tile), same
            # resolution-aware policy as the simple Sampler
            alpha = alpha_for_latent(tiler.tile_h, tiler.tile_w, ALPHA) if SHIFT["resolution_aware"] else ALPHA
            if extra.get("alpha"):
                alpha = float(extra["alpha"])
            sigmas = refine_schedule(p["n_steps"], alpha=alpha, denoise=p["denoise"])
            # plan_segments can only cut at 0 < i < n-1: when the descent already starts
            # inside the texture phase the whole run belongs to the texture model
            # (it carries the identity LoRAs too - lora_phase.PHASE_SEGMENTS).
            run_model, texture_model = model, plan_texture
            if plan_texture is not None and float(sigmas[0]) <= UPSCALE_TEXTURE_START:
                run_model, texture_model = plan_texture, None
            z_ref = run_model.model.process_latent_in(z)     # the loop's state space
            anchor = LFAnchor(z_ref, anchor_cfg["radius"], p["anchor"], float(sigmas[0]),
                              anchor_cfg["release_sigma"])
            out = run_sampling(
                run_model, positive, negative, {"samples": z}, sigmas, seed=int(seed) + b,
                guidance_mode=guidance_mode, flat_cfg=flat_cfg,
                delta=GUIDANCE["delta"], lo=GUIDANCE["lo"], hi=GUIDANCE["hi"],
                contraction=p["contraction"], per_channel_contraction=False,
                manifold_std=MANIFOLD_STD, manifold_mean=MANIFOLD_MEAN,
                detail_amount=p["detail_a"], order=_ORDER_FROM_SAMPLER_NAME[p["sampler"]],
                eta0=float(extra.get("eta0", 0.0)), sigma_gate=0.10,   # refine: ancestral noise = speckle (2026-07-07)
                texture_model=texture_model, texture_start=UPSCALE_TEXTURE_START,
                tiler=tiler, x0_hook=anchor,
            )
            decoded = _decode_tiled(vae, out["samples"])
            decoded = back_project(decoded, image[b:b + 1, ..., :3],
                                   int(extra.get("bp_iters", FIDELITY["bp_iters"])),
                                   float(extra.get("bp_lock", p.get("bp_lock", 1.0))))
            outs.append(decoded)
        return (torch.cat(outs, dim=0),)


from .upscale_v2 import KreaPhotonUpscaleV2  # noqa: E402  (imports nodes.py helpers lazily inside methods)
from .face_detailer import KreaPhotonFaceDetailer  # noqa: E402  (same pattern)

NODE_CLASS_MAPPINGS.update({
    "KreaPhotonSampler": KreaPhotonSampler,
    "KreaPhotonSamplerAdvanced": KreaPhotonSamplerAdvanced,
    "KreaPhotonScheduler": KreaPhotonScheduler,
    "KreaPhotonEmptyLatent": KreaPhotonEmptyLatent,
    "KreaPhotonEncode": KreaPhotonEncode,
    "KreaPhotonSaveImage": KreaPhotonSaveImage,
    "KreaPhotonLoraPhase": KreaPhotonLoraPhase,
    "KreaPhotonUpscale": KreaPhotonUpscale,
    "KreaPhotonUpscaleV2": KreaPhotonUpscaleV2,
    "KreaPhotonFaceDetailer": KreaPhotonFaceDetailer,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "KreaPhotonSampler": "KreaPhoton Sampler",
    "KreaPhotonSamplerAdvanced": "KreaPhoton Sampler (Advanced)",
    "KreaPhotonScheduler": "KreaPhoton Scheduler",
    "KreaPhotonEmptyLatent": "KreaPhoton Empty Latent",
    "KreaPhotonEncode": "KreaPhoton Encode",
    "KreaPhotonSaveImage": "KreaPhoton Save Image",
    "KreaPhotonLoraPhase": "KreaPhoton LoRA Phase",
    "KreaPhotonUpscale": "KreaPhoton Upscale",
    "KreaPhotonUpscaleV2": "KreaPhoton Upscale v2",
    "KreaPhotonFaceDetailer": "KreaPhoton Face Detailer",
})
