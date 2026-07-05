"""
KreaPhoton sampler core (S6). Single custom KSAMPLER loop (euler/euler_2m +
M2 step-relative sigma-nudge + M3 restart re-noise + M5 gated-eta ancestral
stochastic component), orchestrated through KreaPhotonGuider — NOT
comfy.sample.sample_custom (planning-council D11: comfy.samplers.sample()
hardcodes the stock CFGGuider, verified at comfy/samplers.py:1330-1334, and
cannot carry a custom guider). Multi-segment orchestration (clean_model /
variety boundaries) ported from ComfyUI-ZPhoton's run_sampling, which shares
segment-boundary sigmas across sub-calls so comfy's noise_scaling /
inverse_noise_scaling rescale cancels out exactly (verified in M3).
"""
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import torch

from .guidance import KreaPhotonGuider
from .noise import contract_noise
from .variety import cond_tap_rotation, lf_recompose


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _detail_envelope(p: float, start: float, end: float, peak: float) -> float:
    if end <= start:
        return 0.0
    u = (p - start) / (end - start)
    if u <= 0.0 or u >= 1.0:
        return 0.0
    peak = max(0.05, min(0.95, peak))
    w = u / peak if u < peak else (1.0 - u) / (1.0 - peak)
    return _smoothstep(w)


GATE_HI = 0.35   # M5-proven upper edge of the gated-eta ramp; lower edge is the
                  # calibratable preset field sigma_gate (M5-proven default 0.10).


def _gated_eta(sigma_next: float, eta0: float, sigma_gate: float, gate_hi: float = GATE_HI) -> float:
    """eta(sigma_next) = eta0 * smoothstep(sigma_next; sigma_gate, gate_hi).
    Exactly 0 below sigma_gate (M5: terminal ancestral injection is the
    dark-blotch driver), exactly eta0 at/above gate_hi (M5: mid-phase unchanged)."""
    if gate_hi <= sigma_gate:
        return eta0
    u = (sigma_next - sigma_gate) / (gate_hi - sigma_gate)
    return eta0 * _smoothstep(u)


@torch.no_grad()
def kreaphoton_sampler_loop(model, x, sigmas, extra_args=None, callback=None, disable=None,
                            detail_amount=0.0, detail_start=0.15, detail_end=0.95, detail_peak=0.6,
                            order=1, eta0=0.0, sigma_gate=0.10, restart_seed=0):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    gen = torch.Generator(device="cpu").manual_seed((int(restart_seed) + 0x5EED) & 0xffffffffffffffff)
    old_d = None

    for i in range(n):
        s_cur = float(sigmas[i])
        s_next = float(sigmas[i + 1])

        # --- restart segment: ascending jump -> proper flow re-noise (M3) ---
        if s_next > s_cur + 1e-6:
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = (1.0 - s_next) * x + s_next * eps
            old_d = None
            continue

        if s_cur <= 1e-6:
            continue

        # --- detail boost: step-relative sigma nudge (M2), skip on plunge/final ---
        p = i / max(n - 1, 1)
        is_final = s_next <= 1e-6
        is_plunge = (s_cur - s_next) > 0.25
        if is_final or is_plunge:
            a = 0.0
        else:
            a = detail_amount * _detail_envelope(p, detail_start, detail_end, detail_peak)
            a = max(-1.0, min(1.0, a))
        sigma_model = max(1e-4, s_cur - a * (s_cur - s_next))

        denoised = model(x, sigma_model * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i],
                      "sigma_hat": sigmas[i], "denoised": denoised})

        if is_final:
            x = denoised
            old_d = None
            continue

        eta = _gated_eta(s_next, eta0, sigma_gate)
        if eta > 0.0:
            # CONST/rectified-flow ancestral step, ported from comfy's own
            # sample_euler_ancestral_RF (k_diffusion/sampling.py:240-266) so our
            # detail-nudged sigma_model integrates with the SAME math CONST
            # models expect. old_d reset - AB2 memory is not meaningful across
            # a stochastic jump.
            downstep_ratio = 1.0 + (s_next / s_cur - 1.0) * eta
            sigma_down = s_next * downstep_ratio
            alpha_next = 1.0 - s_next
            alpha_down = 1.0 - sigma_down
            renoise_coeff = max(0.0, s_next ** 2 - sigma_down ** 2 * alpha_next ** 2 / alpha_down ** 2) ** 0.5
            ratio = sigma_down / s_cur
            x = ratio * x + (1.0 - ratio) * denoised
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = (alpha_next / alpha_down) * x + eps * renoise_coeff
            old_d = None
        else:
            d = (x - denoised) / s_cur
            dt = s_next - s_cur
            if order >= 2 and old_d is not None and abs(dt) <= 0.25:
                d_use = 1.5 * d - 0.5 * old_d   # Adams-Bashforth 2
            else:
                d_use = d
            x = x + d_use * dt
            old_d = d

    return x


def zero_conditioning(cond):
    """Zeroed-out copy of a conditioning (honest unconditional when no real
    negative is provided - ZPhoton precedent, safe regardless of guidance mode
    since cfg=1.0/flat/window all treat an empty negative identically)."""
    out = []
    for t, d in cond:
        d = d.copy()
        pooled = d.get("pooled_output")
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(t), d])
    return out


def _rotate_conditioning(cond, taps, a, seed):
    """Apply cond_tap_rotation to the TENSOR component of a comfy CONDITIONING
    list ([tensor, dict], ...), preserving dict metadata untouched."""
    out = []
    for t, d in cond:
        out.append([cond_tap_rotation(t, taps, a, seed), d.copy()])
    return out


def _run_one(model, positive, negative, latent_dict, sigmas, *, seed,
             guidance_mode, flat_cfg, delta, lo, hi,
             noise, add_noise, contraction, per_channel_contraction, manifold_std, manifold_mean,
             detail_amount, detail_start, detail_end, detail_peak, order, eta0, sigma_gate):
    latent = comfy.sample.fix_empty_latent_channels(model, latent_dict["samples"])
    assert latent.ndim == 5, f"expected 5D Wan21 latent, got shape {tuple(latent.shape)}"

    if negative is None:
        negative = zero_conditioning(positive)

    if not add_noise:
        noise = torch.zeros(latent.shape, dtype=latent.dtype, device="cpu")
    elif noise is None:
        noise = comfy.sample.prepare_noise(latent, seed)
    noise = contract_noise(noise, strength=contraction, per_channel=per_channel_contraction,
                            manifold_std=manifold_std, manifold_mean=manifold_mean)

    sampler = comfy.samplers.KSAMPLER(kreaphoton_sampler_loop, extra_options={
        "detail_amount": float(detail_amount), "detail_start": float(detail_start),
        "detail_end": float(detail_end), "detail_peak": float(detail_peak),
        "order": int(order), "eta0": float(eta0), "sigma_gate": float(sigma_gate),
        "restart_seed": int(seed),
    })

    # guidance_mode: "off"/"flat" use the STOCK CFGGuider (cfg constant, no
    # windowing - trivial special case, no need for our custom guider at all);
    # "window" uses KreaPhotonGuider (M6, gated behind E1b per docs/04).
    if guidance_mode == "window":
        guider = KreaPhotonGuider(model, delta=delta, lo=lo, hi=hi)
    else:
        guider = comfy.samplers.CFGGuider(model)
        guider.set_cfg(flat_cfg if guidance_mode == "flat" else 1.0)
    guider.set_conds(positive, negative)

    try:
        import latent_preview
        callback = latent_preview.prepare_callback(model, len(sigmas) - 1)
    except Exception:
        callback = None
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(noise, latent, sampler, sigmas,
                            denoise_mask=latent_dict.get("noise_mask"),
                            callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(device=comfy.model_management.intermediate_device(),
                         dtype=comfy.model_management.intermediate_dtype())
    out = latent_dict.copy()
    out["samples"] = samples
    return out


def run_sampling(model, positive, negative, latent_dict, sigmas, *, seed,
                 guidance_mode="off", flat_cfg=1.15, delta=1.25, lo=0.7, hi=0.9,
                 noise=None, add_noise=True,
                 contraction=1.0, per_channel_contraction=False,
                 manifold_std=None, manifold_mean=None,
                 detail_amount=0.0, detail_start=0.15, detail_end=0.95, detail_peak=0.6,
                 order=1, eta0=0.0, sigma_gate=0.10,
                 clean_model=None, composition_end=0.85,
                 variety_a_latent=0.0, variety_a_cond=0.0, variety_seed=0,
                 variety_end=0.90, variety_cond_taps=(7, 8, 9, 10)):
    """Shared entry point for all KreaPhoton sampler nodes.

    Segment orchestration (ZPhoton precedent, M3-verified exact rescale
    cancellation at shared boundary sigmas): clean_model splits the
    trajectory at composition_end (composition on the clean model, LoRA
    identity/detail phase on `model` - anti-mutation, unvalidated on krea2
    LoRA stack per docs/04 item 7); variety splits at variety_end and applies
    lf_recompose (latent axis) + cond_tap_rotation (cond axis, on `positive`)
    once, at that boundary. restart is NOT a split point here - it's encoded
    directly in the sigmas array and handled per-step inside
    kreaphoton_sampler_loop (the s_next > s_cur predicate), exactly like
    composition/variety segments still run the SAME schedule around it.

    NOTE on latent space at the variety boundary: `cur["samples"]` here is the
    RAW-VAE-space output of the previous segment's guider.sample() call
    (CFGGuider.inner_sample applies process_latent_out before returning -
    comfy/samplers.py:1230), NOT the normalized space lf_recompose's own
    docstring anchors its Wan21/MANIFOLD_MEAN discussion to. This matches
    ZPhoton's own proven call site exactly (same architecture, same space at
    that point) and is harmless: lf_recompose's mean/std restore is
    self-referential (matches whatever space the input tensor is already in),
    so composition-rotation correctness does not depend on which space it
    receives - only contract_noise's MANIFOLD_STD/MEAN constants are
    space-specific, and those are applied strictly to the pre-noise_scaling
    noise tensor inside _run_one, never here.
    """
    kw = dict(guidance_mode=guidance_mode, flat_cfg=flat_cfg, delta=delta, lo=lo, hi=hi,
              contraction=contraction, per_channel_contraction=per_channel_contraction,
              manifold_std=manifold_std, manifold_mean=manifold_mean,
              detail_amount=detail_amount, detail_start=detail_start, detail_end=detail_end,
              detail_peak=detail_peak, order=order, eta0=eta0, sigma_gate=sigma_gate)

    sig_list = [float(s) for s in sigmas]

    def first_idx_below(th):
        for i, s in enumerate(sig_list):
            if s <= th:
                return i
        return None

    use_comp = clean_model is not None
    use_variety = variety_a_latent > 0.0 or variety_a_cond > 0.0

    split_idxs = set()
    variety_idx = None
    if use_comp:
        i = first_idx_below(composition_end)
        if i is not None and 0 < i < len(sig_list) - 1:
            split_idxs.add(i)
    if use_variety:
        i = first_idx_below(variety_end)
        if i is not None and 0 < i < len(sig_list) - 1:
            split_idxs.add(i)
            variety_idx = i

    if not split_idxs:
        return _run_one(model, positive, negative, latent_dict, sigmas,
                        seed=seed, noise=noise, add_noise=add_noise, **kw)

    bounds = [0] + sorted(split_idxs) + [len(sig_list) - 1]
    cur = latent_dict
    cur_positive = positive
    cur_noise, cur_add = noise, add_noise
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        if a == b:
            continue
        if variety_idx is not None and a == variety_idx:
            cur = dict(cur)
            if variety_a_latent > 0.0:
                cur["samples"] = lf_recompose(cur["samples"], seed_v=variety_seed, a=variety_a_latent)
            if variety_a_cond > 0.0:
                cur_positive = _rotate_conditioning(positive, variety_cond_taps, variety_a_cond, variety_seed)
        seg = sigmas[a:b + 1]
        sig_start = sig_list[a]
        m = clean_model if (use_comp and sig_start > composition_end) else model
        cur = _run_one(m, cur_positive, negative, cur, seg,
                       seed=seed, noise=cur_noise, add_noise=cur_add, **kw)
        cur_noise, cur_add = None, False
    return cur
