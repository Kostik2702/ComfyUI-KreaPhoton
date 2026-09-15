"""
KreaPhoton guider (M6 sigma-adaptive guidance window + variety cond switch).
The ONLY sanctioned way to implement phase-windowed guidance for KreaPhoton
(planning-council B4/D10-D14/H29), and since v1.3 the single guider every
KreaPhoton sampler path uses ("off" / "flat" / "window" are three cond_scale
policies of one class - no more stock-CFGGuider special case).

Ground truth (E:\\CUI portable\\ComfyUI-torch2.9-cu130-cp313-v1.2\\ComfyUI):
  comfy/samplers.py:609-612  sampling_function(): at math.isclose(cond_scale,1.0)
                              and disable_cfg1_optimization not set -> uncond_=None,
                              ZERO extra NFE for that step (stock optimization).
  comfy/samplers.py:1217-1218 CFGGuider.predict_noise (the ONE method we override
                              besides set_conds, which only adds the variety key)
  comfy/samplers.py:592-602   sampler_cfg_function / sampler_post_cfg_function hooks
                              receive a ZEROED uncond at cfg=1.0 (calc_cond_batch
                              leaves the unconsumed accumulator at zeros) - silent
                              garbage for exactly our default (cfg=1.0) path.
  comfy/samplers.py:1038-1075 process_conds iterates over EVERY key of the conds
                              dict, so an extra "positive_variety" entry is prepared
                              (areas, hooks, model extra_conds) exactly like
                              positive/negative - verified in the live 0.30 source.

RULE: guidance/variety windows are NEVER implemented via model_options hooks
(sampler_cfg_function / sampler_post_cfg_function). Only a CFGGuider subclass
sees whether uncond was actually computed this step.

Variety cond axis (v1.3, single-lifecycle): the rotated positive is stored as a
third cond entry and selected per model call by a flag the sampler loop sets in
model_options at the variety boundary (VARIETY_FLAG). The loop owns the boundary
decision (it is the same step at which the latent axis fires), the guider only
reads it - no second guider.sample() lifecycle, no latent round-trip, and the
validated gated-eta stays on (audit F02/F03).
"""
import comfy.samplers
import torch

VARIETY_FLAG = "kreaphoton_variety_active"       # model_options key written by the loop
VARIETY_COND_KEY = "positive_variety"            # conds key holding the rotated positive
GUIDANCE_MODES = ("off", "flat", "window")


def smoothstep01(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def g_window(sigma: float, delta: float, lo: float = 0.7, hi: float = 0.9) -> float:
    """g(sigma) = 1 + delta * smoothstep((sigma-lo)/(hi-lo)).

    MUST return exactly 1.0 outside [lo, hi] (smoothstep01 clamps its argument to
    [0,1] before the cubic, so u<=0 -> smoothstep=0 -> g=1.0 exactly; no epsilon
    residue, unlike a sigmoid-form window would leave). This is what lets the
    stock cfg1-optimization (samplers.py:609) fire for free outside the window.

    Degenerate windows (audit F07): hi < lo is swapped, hi == lo becomes a hard
    step at lo (1+delta at/above, exactly 1.0 below) instead of a division by 0.
    """
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo <= 1e-9:
        return 1.0 + delta if sigma >= lo else 1.0
    return 1.0 + delta * smoothstep01((sigma - lo) / (hi - lo))


PAG_STASH_KEY = "_kreaphoton_pag_v"


def parse_blocks(spec) -> frozenset:
    """'8-15' / '8,10,12' / '8-11,20' / iterable of ints -> frozenset of block indices."""
    if spec is None:
        return frozenset()
    if not isinstance(spec, str):
        return frozenset(int(b) for b in spec)
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return frozenset(out)


def _pag_stash_v(q, k, v, pe=None, attn_mask=None, extra_options=None):
    """attn1_patch: keep v for the output patch of the SAME attention call
    (krea2 passes one extra_options dict to both patch lists)."""
    if extra_options is not None:
        extra_options[PAG_STASH_KEY] = v
    return {}


def _pag_identity_output(blocks: frozenset):
    def patch(out, extra_options):
        if extra_options.get("block_index") not in blocks:
            return out
        v = extra_options.pop(PAG_STASH_KEY, None)
        if v is None:
            return out
        # v: (B, Hkv, L, D) before GQA expansion; out: (B, L, H*D)
        b, hkv, seq, d = v.shape
        heads = out.shape[-1] // d
        if heads != hkv:
            v = v.repeat_interleave(heads // hkv, dim=1)
        return v.transpose(1, 2).reshape(b, seq, heads * d).to(out.dtype)
    return patch


def pag_model_options(model_options: dict, blocks: frozenset) -> dict:
    """Clone of model_options with the PAG attention patches appended for this
    forward only (never mutates the caller's dicts)."""
    mo = dict(model_options)
    to = dict(mo.get("transformer_options", {}))
    patches = dict(to.get("patches", {}))
    patches["attn1_patch"] = list(patches.get("attn1_patch", [])) + [_pag_stash_v]
    patches["attn1_output_patch"] = list(patches.get("attn1_output_patch", [])) + [_pag_identity_output(blocks)]
    to["patches"] = patches
    mo["transformer_options"] = to
    return mo


class KreaPhotonGuider(comfy.samplers.CFGGuider):
    """CFGGuider subclass implementing the M6 guidance window + variety cond switch.

    Overrides predict_noise (comfy/samplers.py:1217-1218 is a 1-line method) and
    set_conds (adds the optional variety key); everything else - prepare_sampling,
    process_conds, device management, wrapper executors - is inherited unchanged,
    verified against the live 0.30 source.

    self.conds is populated by the base class's inner_sample/process_conds before
    any predict_noise call; never read self.original_conds here.

    mode: "off"    -> cond_scale 1.0 every step (stock cfg1 optimization, 0 uncond NFE)
          "flat"   -> cond_scale = cfg every step (full-trajectory CFG, 2x NFE; RAW)
          "window" -> cond_scale = g_window(sigma) (M6, uncond only inside [lo, hi])
    """

    def __init__(self, model_patcher, mode: str = "off", cfg: float = 1.0,
                 delta: float = 1.25, lo: float = 0.7, hi: float = 0.9,
                 rescale: float = 0.0, pag_scale: float = 0.0, pag_lo: float = 0.72,
                 pag_hi: float = 0.93, pag_blocks="8-15"):
        super().__init__(model_patcher)
        if mode not in GUIDANCE_MODES:
            raise ValueError(f"KreaPhoton: guidance_mode must be one of {GUIDANCE_MODES}, got {mode!r}")
        self.mode = mode
        self.set_cfg(float(cfg))
        self.delta = float(delta)
        self.lo = float(min(lo, hi))
        self.hi = float(max(lo, hi))
        self.rescale = float(max(0.0, min(1.0, rescale)))
        self.pag_scale = float(max(0.0, pag_scale))
        self.pag_lo = float(min(pag_lo, pag_hi))
        self.pag_hi = float(max(pag_lo, pag_hi))
        self.pag_blocks = parse_blocks(pag_blocks)

    # ------------------------------------------------------------------
    # Perturbed-attention guidance (v1.4, experimental)
    # ------------------------------------------------------------------
    def pag_active(self, sigma: float) -> bool:
        return (getattr(self, "pag_scale", 0.0) > 0.0 and self.pag_lo - 1e-9 <= sigma <= self.pag_hi + 1e-9
                and len(self.pag_blocks) > 0)

    def _pag_post_cfg(self, args):
        """comfy sampler_post_cfg_function: one extra conditional forward with
        identity self-attention in self.pag_blocks (comfy/ldm/krea2/model.py
        Attention.forward: attn1_patch sees q/k/v BEFORE RoPE and the same
        extra_options dict reaches attn1_output_patch, so v is stashed there and
        the attention output replaced by v = each token attends only to itself -
        the PAG perturbation). denoised += scale * (cond - perturbed).
        cond_denoised is real on every step (at cfg=1.0 only the UNCOND is a
        zeroed dummy - see the header), so this is safe inside the cfg=1 path."""
        cond_pred = args["cond_denoised"]
        cond = args["cond"]
        x = args["input"]
        sigma_t = args["sigma"]
        model = args["model"]
        model_options = pag_model_options(args["model_options"], self.pag_blocks)
        (perturbed,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma_t, model_options)
        return args["denoised"] + self.pag_scale * (cond_pred - perturbed)

    def _rescale_post_cfg(self, args):
        """CFG-rescale (Lin et al. 2023, phi = self.rescale) as a comfy
        sampler_post_cfg_function. Registered per call ONLY when cond_scale>1,
        so cond_denoised/uncond_denoised are real (at cfg=1.0 comfy hands hooks
        a zeroed uncond - the trap documented at the top of this file)."""
        cfg_result = args["denoised"]
        cond = args["cond_denoised"]
        dims = tuple(range(1, cfg_result.ndim))
        std_cfg = cfg_result.std(dim=dims, keepdim=True).clamp_min(1e-8)
        std_cond = cond.std(dim=dims, keepdim=True)
        rescaled = cfg_result * (std_cond / std_cfg)
        return self.rescale * rescaled + (1.0 - self.rescale) * cfg_result

    def set_conds(self, positive, negative, positive_variety=None):
        conds = {"positive": positive, "negative": negative}
        if positive_variety is not None:
            conds[VARIETY_COND_KEY] = positive_variety
        self.inner_set_conds(conds)

    def cond_scale_at(self, sigma: float) -> float:
        if self.mode == "window":
            return g_window(sigma, self.delta, self.lo, self.hi)
        if self.mode == "flat":
            return float(self.cfg)
        return 1.0

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}
        sigma = float(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else float(timestep)
        cond_scale = self.cond_scale_at(sigma)
        positive = self.conds.get("positive", None)
        if model_options.get(VARIETY_FLAG, False) and VARIETY_COND_KEY in self.conds:
            positive = self.conds[VARIETY_COND_KEY]
        extra_post = []
        if getattr(self, "rescale", 0.0) > 0.0 and cond_scale > 1.0 + 1e-9:
            extra_post.append(self._rescale_post_cfg)
        if self.pag_active(sigma):
            extra_post.append(self._pag_post_cfg)
        if extra_post:
            model_options = dict(model_options)
            model_options["sampler_post_cfg_function"] = (
                list(model_options.get("sampler_post_cfg_function", [])) + extra_post)
        return comfy.samplers.sampling_function(
            self.inner_model, x, timestep,
            self.conds.get("negative", None), positive,
            cond_scale, model_options=model_options, seed=seed,
        )
