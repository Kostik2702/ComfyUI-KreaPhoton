"""
KreaPhoton sampler core (S6; single-lifecycle rework v1.3). One custom KSAMPLER
loop (euler/euler_2m + M2 step-relative sigma-nudge + M3 restart re-noise + M5
gated-eta ancestral stochastic component + M4 variety at its sigma boundary +
terminal x0-trajectory extrapolation), orchestrated through KreaPhotonGuider -
NOT comfy.sample.sample_custom (planning-council D11: comfy.samplers.sample()
hardcodes the stock CFGGuider, verified at comfy/samplers.py, and cannot carry
a custom guider).

v1.3 (audit F02/F03): variety no longer splits the trajectory into two
guider.sample() lifecycles. The latent axis (lf_recompose) fires INSIDE the
loop at the first model-call step at/below variety_end, on the sampler's own
state x (lf_recompose commutes with the per-channel affine + 1/(1-sigma)
rescale the old split path went through - proven in tests/test_sampling.py
[6]); the cond axis is a third cond entry the guider switches to via a flag the
loop writes into model_options at the same step. Consequences: no
process_latent_out/in round-trip, no forced eta0=0 (the V4-validated gated eta
stays on with variety), one preview/progress lifecycle, and the boundary is
now bit-identical between the simple and Advanced nodes.

The only remaining multi-segment path is the optional clean_model split
(composition on a clean checkpoint, LoRA phase on `model`) - ZPhoton pattern,
still experimental on krea2 LoRA stacks, and it keeps the eta0 safety guard.
"""
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import torch

from .guidance import VARIETY_FLAG, KreaPhotonGuider
from .noise import contract_noise
from .presets import effective_delta
from .schedules import validate_sigmas
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
PLUNGE_DSIGMA = 0.25          # a descending step larger than this is a plunge: M2 nudge
                               # skip, AB2 fallback to euler, no x0-extrapolation.
X0_EXTRAP_MAX_FACTOR = 2.0    # cap on the linear x0-trajectory extrapolation factor
                               # sigma_last/(sigma_prev - sigma_last); the calibrated
                               # restart grid (0.4333 -> 0.2166 -> 0) gives exactly 1.0,
                               # a foreign grid with a tiny last step could otherwise
                               # extrapolate wildly.
VARIETY_STATES = ("off", "pending", "active")


def _gated_eta(sigma_next: float, eta0: float, sigma_gate: float, gate_hi: float = GATE_HI) -> float:
    """eta(sigma_next) = eta0 * smoothstep(sigma_next; sigma_gate, gate_hi).
    Exactly 0 below sigma_gate (M5: terminal ancestral injection is the
    dark-blotch driver), exactly eta0 at/above gate_hi (M5: mid-phase unchanged)."""
    if gate_hi <= sigma_gate:
        return eta0
    u = (sigma_next - sigma_gate) / (gate_hi - sigma_gate)
    return eta0 * _smoothstep(u)


def variety_perturb(x, sigma: float, x0_hat, seed_v: int, a: float):
    """Latent variety axis applied to the NOISE component only (v1.3.1).

    The loop state is x = (1-sigma)*x0 + sigma*eps (CONST). Re-composing the
    LF band of the whole state also re-composes the LF band of the model's
    committed structure estimate x0 - a state whose low frequencies no longer
    agree with its high frequencies. Without guidance the distillate absorbs
    that (V3/V13: texture-only variety, identity kept), but inside the M6
    guidance window the cond/uncond disagreement it creates is amplified up to
    2.25x and surfaces as glowing LF blobs on some seeds (V14b D2, 2026-09-07).

    Fix: recover the current noise realization from the previous model call,
    eps = (x - (1-sigma)*x0_hat)/sigma (exact for euler on CONST, effective
    noise after an ancestral step), re-compose ONLY its LF band and rebuild the
    state around the untouched x0_hat. Variance-preserving on eps, structure
    untouched, guidance sees a self-consistent state. With no previous
    prediction (boundary at the first step, x is pure noise) the whole state is
    the noise and lf_recompose applies to it directly.
    """
    if x0_hat is None or sigma <= 1e-6:
        return lf_recompose(x, seed_v=seed_v, a=a)
    eps = (x - (1.0 - sigma) * x0_hat) / sigma
    eps = lf_recompose(eps, seed_v=seed_v, a=a)
    return (1.0 - sigma) * x0_hat + sigma * eps


def plan_segments(sig_list, *, composition_end=0.85, has_clean=False, texture_start=0.65,
                  has_texture=False):
    """Segment plan for the multi-model split (v1.4 phase models). Returns a list of
    (start_idx, end_idx, phase) with phase in {"composition", "identity", "texture"};
    a single ("identity") segment when no phase model is connected.

    composition: sigma > composition_end, run on clean_model (layout without LoRA
                 steering); identity: the rest, on `model`; texture: from the first
                 index with sigma <= texture_start - on the calibrated grids that is
                 the plunge readout (sigma 0) right before the restart jump, so the
                 texture model owns exactly the restart segment.
    Pure, unit-tested (tests/test_sampling.py [13])."""
    n = len(sig_list)
    cuts = []
    if has_clean:
        i = next((k for k, s in enumerate(sig_list) if s <= composition_end), None)
        if i is not None and 0 < i < n - 1:
            cuts.append((i, "identity"))
    if has_texture:
        # texture phase = the restart segment: from the readout before the LAST
        # ascending jump when that jump lands at/below texture_start (a self-
        # refine jump lands higher and stays in the identity phase); a plain
        # descending schedule falls back to the first sigma <= texture_start.
        jumps = [k for k in range(n - 1) if sig_list[k + 1] > sig_list[k] + 1e-6]
        i = None
        if jumps and sig_list[jumps[-1] + 1] <= texture_start + 1e-9:
            i = jumps[-1]
        if i is None:
            i = next((k for k, s in enumerate(sig_list) if s <= texture_start), None)
        if i is not None and 0 < i < n - 1:
            cuts.append((i, "texture"))
    cuts.sort()
    phase0 = "composition" if has_clean else "identity"
    segs = []
    start, phase = 0, phase0
    for idx, next_phase in cuts:
        if idx <= start:
            phase = next_phase
            continue
        segs.append((start, idx, phase))
        start, phase = idx, next_phase
    segs.append((start, n - 1, phase))
    return [s for s in segs if s[1] > s[0]]


def variety_boundary_index(sigmas, variety_end: float):
    """Index of the first MODEL-CALL step (not the restart jump, not a sigma=0
    readout) whose sigma is at/below variety_end - the step at which both variety
    axes fire. None if no such step exists (variety never applies)."""
    s = [float(v) for v in sigmas]
    for i in range(len(s) - 1):
        if s[i + 1] > s[i] + 1e-6:
            continue
        if s[i] <= 1e-6:
            continue
        if s[i] <= variety_end + 1e-9:
            return i
    return None


@torch.no_grad()
def kreaphoton_sampler_loop(model, x, sigmas, extra_args=None, callback=None, disable=None,
                            detail_amount=0.0, detail_start=0.15, detail_end=0.95, detail_peak=0.6,
                            order=1, eta0=0.0, sigma_gate=0.10, restart_seed=0,
                            variety_a_latent=0.0, variety_end=0.96, variety_seed=0,
                            variety_state="off", x0_extrapolation=0.0,
                            progress_offset=0, progress_total=0, restart_hook=None,
                            coherence_jump=0.0, coherence_jump_sigma=0.93,
                            tiler=None, x0_hook=None):
    """The KSAMPLER sampler_function. Pure numerical loop, testable with a mock
    model callable (tests/test_sampling.py).

    tiler (v1.5 Upscale): optional callable(model, x, sigma_model, extra_args)
                   -> denoised that replaces the single model call - tiling.LatentTiler
                   cuts x into overlapping tiles, batches them through the model
                   and merges the x0 predictions with feathered weights, so every
                   step re-blends neighbouring tiles (no seams). None = one call.
    x0_hook: optional callable(denoised, sigma_model) -> denoised applied right
                   after every model call (before the integration step and the
                   preview callback) - tiling.LFAnchor pulls the low-frequency
                   band back to a reference latent. None = bit-exact.

    coherence_jump (v1.4, Power-Nodes jump-back): ONCE, at the first model-call
                   step with sigma <= coherence_jump_sigma, the state is rescaled
                   by (1-s_decl)/(1-s_cur) and the model is told s_decl =
                   s_cur + coherence_jump*(1-s_cur); the step then integrates from
                   s_decl. That is exactly what comfy's noise-scaling round-trip
                   does to Power Nodes' 0.920 -> 0.935 stage boundary (0.19 in
                   these units), made explicit: the model sees a state with the
                   correct signal-to-noise ratio but 20% less amplitude than the
                   declared sigma implies, and commits harder to structure on the
                   composition-phase step. 0 = off (bit-exact).

    restart_hook: optional callable(x, base_model) -> x applied to the plunge
                   readout x0 right before the restart re-noise (v1.4 restart
                   enhance, see enhance.py). base_model is the comfy BaseModel
                   behind the guider (process_latent_in/out), or None under a
                   plain mock model.

    progress_offset / progress_total: global step index of this segment's first
                   step and the total step count of the FULL schedule, so the M2
                   detail envelope p = (offset + i) / (total - 1) is identical
                   whether the schedule runs in one lifecycle or is split at a
                   clean_model boundary. ROOT CAUSE of the S9 "split speckle /
                   mosaic" and the 2026-07-06 "variety@0.90 droplets" (v1.3
                   finding, V15 L5 on a LoRA rig): a segment restarted its envelope
                   at p=0, so the restart steps got a nudge of ~0.58 x step instead
                   of ~0.18 (sigma_model 0.52 at a true sigma of 0.65) - the model
                   then under-removes the re-noise and the leftover decodes as
                   coloured confetti. 0 = single-lifecycle default.

    variety_state: "off"     - no boundary logic, guider flag stays False;
                   "pending" - at the boundary step apply lf_recompose (if
                               variety_a_latent>0) and raise the guider flag;
                   "active"  - flag raised from the first step (a segment that
                               starts after the boundary, clean_model split only).
    x0_extrapolation: 0..1, terminal step only. Linear extrapolation of the x0
                   prediction trajectory to sigma=0 (denoised + f*(denoised -
                   prev_denoised), f = sigma_last/(sigma_prev - sigma_last),
                   capped at X0_EXTRAP_MAX_FACTOR), blended by this weight into
                   the plain final denoised. Skipped on a plunge step and when
                   the segment has no previous model call.
    """
    extra_args = {} if extra_args is None else extra_args
    if variety_state not in VARIETY_STATES:
        raise ValueError(f"variety_state must be one of {VARIETY_STATES}, got {variety_state!r}")
    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    n_total = int(progress_total) if int(progress_total) > 0 else n
    p_off = int(progress_offset)
    gen = torch.Generator(device="cpu").manual_seed((int(restart_seed) + 0x5EED) & 0xffffffffffffffff)
    old_d = None
    prev_denoised = None
    prev_sigma_model = None

    # Variety hand-off to the guider (cond axis). extra_args["model_options"] is
    # the per-lifecycle clone comfy passes to every model call of this loop
    # (CFGGuider.inner_sample -> KSAMPLER.sample -> KSamplerX0Inpaint ->
    # guider.predict_noise), so a key written here is visible in predict_noise.
    model_options = extra_args.get("model_options")
    if model_options is None:
        model_options = {}
        extra_args["model_options"] = model_options
    variety_done = variety_state != "pending"
    model_options[VARIETY_FLAG] = variety_state == "active"
    jump_done = coherence_jump <= 0.0

    for i in range(n):
        s_cur = float(sigmas[i])
        s_next = float(sigmas[i + 1])

        # --- restart segment: ascending jump -> proper flow re-noise (M3) ---
        if s_next > s_cur + 1e-6:
            if restart_hook is not None:
                base_model = getattr(getattr(model, "inner_model", None), "inner_model", None)
                x = restart_hook(x, base_model)
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = (1.0 - s_next) * x + s_next * eps
            old_d = None
            prev_denoised = None
            prev_sigma_model = None
            continue

        if s_cur <= 1e-6:
            continue

        # --- variety boundary (M4): once, at the first model-call step at/below
        # variety_end. Latent axis on the loop state x (normalized model space;
        # lf_recompose is self-referential in mean/std, so the space is
        # irrelevant to it - see variety.py), cond axis via the guider flag. ---
        if not variety_done and s_cur <= variety_end + 1e-9:
            if variety_a_latent > 0.0:
                x = variety_perturb(x, s_cur, prev_denoised, int(variety_seed), float(variety_a_latent))
                old_d = None          # AB2 memory is not meaningful across a re-composition
                prev_denoised = None
                prev_sigma_model = None
            model_options[VARIETY_FLAG] = True
            variety_done = True

        # --- coherence jump-back (v1.4): once, first model-call step at/below
        # coherence_jump_sigma. State rescaled, declared sigma raised; the step
        # integrates from the declared sigma (see docstring). ---
        if not jump_done and s_cur <= coherence_jump_sigma + 1e-9:
            s_decl = s_cur + float(coherence_jump) * (1.0 - s_cur)
            s_decl = min(s_decl, 0.999)
            if s_decl > s_next + 1e-6:
                x = x * ((1.0 - s_decl) / (1.0 - s_cur))
                s_cur = s_decl
                old_d = None
                prev_denoised = None
                prev_sigma_model = None
            jump_done = True

        # --- detail boost: step-relative sigma nudge (M2), skip on plunge/final ---
        p = (p_off + i) / max(n_total - 1, 1)
        is_final = s_next <= 1e-6
        is_plunge = (s_cur - s_next) > PLUNGE_DSIGMA
        if is_final or is_plunge:
            a = 0.0
        else:
            a = detail_amount * _detail_envelope(p, detail_start, detail_end, detail_peak)
            a = max(-1.0, min(1.0, a))
        sigma_model = max(1e-4, s_cur - a * (s_cur - s_next))

        if tiler is not None:
            denoised = tiler(model, x, sigma_model, extra_args)
        else:
            denoised = model(x, sigma_model * s_in, **extra_args)
        if x0_hook is not None:
            denoised = x0_hook(denoised, sigma_model)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i],
                      "sigma_hat": sigmas[i], "denoised": denoised})

        if is_final:
            x = denoised
            if (x0_extrapolation > 0.0 and not is_plunge and prev_denoised is not None
                    and prev_sigma_model is not None and prev_sigma_model - sigma_model > 1e-6):
                # terminal x0-trajectory extrapolation: the model's x0 estimate
                # sharpens as sigma -> 0; continue that trend linearly past the
                # last evaluation point instead of stopping at it.
                factor = min(X0_EXTRAP_MAX_FACTOR, sigma_model / (prev_sigma_model - sigma_model))
                x = denoised + float(x0_extrapolation) * factor * (denoised - prev_denoised)
            old_d = None
            prev_denoised = None
            prev_sigma_model = None
            continue

        eta = _gated_eta(s_next, eta0, sigma_gate)
        if eta > 0.0:
            # CONST/rectified-flow ancestral step, ported from comfy's own
            # sample_euler_ancestral_RF (k_diffusion/sampling.py) so our
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
            if order >= 2 and old_d is not None and abs(dt) <= PLUNGE_DSIGMA:
                d_use = 1.5 * d - 0.5 * old_d   # Adams-Bashforth 2
            else:
                d_use = d
            x = x + d_use * dt
            old_d = d

        prev_denoised = denoised
        prev_sigma_model = sigma_model

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
             detail_amount, detail_start, detail_end, detail_peak, order, eta0, sigma_gate,
             x0_extrapolation=0.0, positive_variety=None, variety_a_latent=0.0,
             variety_end=0.96, variety_seed=0, variety_state="off",
             progress_offset=0, progress_total=0, guidance_rescale=0.0,
             pag_scale=0.0, pag_lo=0.72, pag_hi=0.93, pag_blocks="8-15", restart_hook=None,
             coherence_jump=0.0, coherence_jump_sigma=0.93, tiler=None, x0_hook=None):
    latent = comfy.sample.fix_empty_latent_channels(model, latent_dict["samples"])
    if latent.ndim != 5 or latent.shape[1] != 16:
        raise ValueError(
            f"KreaPhoton: expected a Krea 2 (Wan21) latent of shape (B, 16, 1, H, W), got "
            f"{tuple(latent.shape)} - use KreaPhoton Empty Latent / EmptySD3LatentImage (16ch) "
            f"or a VAE-encoded image from the krea2 VAE, and a Krea 2 checkpoint as `model`")

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
        "variety_a_latent": float(variety_a_latent), "variety_end": float(variety_end),
        "variety_seed": int(variety_seed), "variety_state": variety_state,
        "x0_extrapolation": float(x0_extrapolation),
        "progress_offset": int(progress_offset), "progress_total": int(progress_total),
        "restart_hook": restart_hook,
        "coherence_jump": float(coherence_jump), "coherence_jump_sigma": float(coherence_jump_sigma),
        "tiler": tiler, "x0_hook": x0_hook,
    })

    # One guider for every path: "off" is cond_scale 1.0 (stock cfg1 optimization,
    # no uncond NFE), "flat" constant cfg (RAW), "window" the M6 window.
    guider = KreaPhotonGuider(model, mode=guidance_mode, cfg=flat_cfg, delta=delta, lo=lo, hi=hi,
                              rescale=guidance_rescale, pag_scale=pag_scale, pag_lo=pag_lo,
                              pag_hi=pag_hi, pag_blocks=pag_blocks)
    guider.set_conds(positive, negative, positive_variety=positive_variety)

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


@torch.no_grad()
def kreaphoton_invert_loop(model, x, sigmas, extra_args=None, callback=None, disable=None,
                           tiler=None, x0_hook=None):
    """Noise inversion for the CONST / rectified-flow model (v2 Upscale): the KSAMPLER
    sampler_function for an ASCENDING sigma array. Euler steps run the flow backwards -
    d = (x - x0_hat) / sigma is exactly dx/dsigma of x = (1-sigma)*x0 + sigma*eps - so
    the state climbs from the (almost) clean source latent at sigmas[0] to sigmas[-1]
    carrying the source's OWN noise realisation: the noise a forward descent from there
    would turn back into the source. Deterministic (no RNG). tiler / x0_hook exactly as
    in kreaphoton_sampler_loop. comfy hands us x = sigmas[0]*noise + (1-sigmas[0])*z and
    divides the result by (1-sigmas[-1]) - the descent lifecycle that follows multiplies
    by the same factor (CONST noise_scaling with zero noise), so the hand-over is exact."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    for i in range(n):
        s_cur = float(sigmas[i])
        s_next = float(sigmas[i + 1])
        if s_next <= s_cur + 1e-9 or s_cur <= 1e-6:
            raise ValueError("kreaphoton_invert_loop: sigmas must be strictly ascending and > 0 "
                             "(got %.4f -> %.4f at %d)" % (s_cur, s_next, i))
        if tiler is not None:
            denoised = tiler(model, x, s_cur, extra_args)
        else:
            denoised = model(x, s_cur * s_in, **extra_args)
        if x0_hook is not None:
            denoised = x0_hook(denoised, s_cur)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        d = (x - denoised) / s_cur
        x = x + d * (s_next - s_cur)
    return x


def run_inversion(model, positive, negative, latent_dict, sigmas_asc, *, seed,
                  guidance_mode="off", flat_cfg=1.15, delta=1.25, lo=0.7, hi=0.9,
                  start_noise=1.0, tiler=None, x0_hook=None):
    """Invert `latent_dict` (source latent) along the ASCENDING `sigmas_asc` with
    kreaphoton_invert_loop. Returns a LATENT dict whose samples, fed to run_sampling with
    add_noise=False on the matching DESCENDING schedule, descend back to (approximately)
    the source - the v2 Upscale's fidelity mechanism.

    start_noise: amplitude of the fresh noise comfy mixes in at sigmas_asc[0] (a tiny
    sigma): 1.0 = unit noise (a proper x_sigma state), 0 = none (the loop then inverts a
    slightly darkened clean latent; kept as a tune option)."""
    sigmas_asc = torch.as_tensor(sigmas_asc, dtype=torch.float32)
    latent = comfy.sample.fix_empty_latent_channels(model, latent_dict["samples"])
    if latent.ndim != 5 or latent.shape[1] != 16:
        raise ValueError(f"KreaPhoton: expected a Krea 2 (Wan21) latent of shape (B, 16, 1, H, W), got "
                         f"{tuple(latent.shape)}")
    if negative is None:
        negative = zero_conditioning(positive)
    noise = comfy.sample.prepare_noise(latent, seed) * float(start_noise)
    sampler = comfy.samplers.KSAMPLER(kreaphoton_invert_loop, extra_options={"tiler": tiler, "x0_hook": x0_hook})
    guider = KreaPhotonGuider(model, mode=guidance_mode, cfg=flat_cfg, delta=delta, lo=lo, hi=hi)
    guider.set_conds(positive, negative)
    try:
        import latent_preview
        callback = latent_preview.prepare_callback(model, len(sigmas_asc) - 1)
    except Exception:
        callback = None
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(noise, latent, sampler, sigmas_asc, denoise_mask=None,
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
                 order=1, eta0=0.0, sigma_gate=0.10, x0_extrapolation=0.0,
                 clean_model=None, composition_end=0.85,
                 variety_a_latent=0.0, variety_a_cond=0.0, variety_seed=0,
                 variety_end=0.96, variety_cond_taps=(7, 8, 9, 10),
                 guidance_rescale=0.0, pag_scale=0.0, pag_lo=0.72, pag_hi=0.93, pag_blocks="8-15",
                 restart_hook=None, texture_model=None, texture_start=0.65,
                 coherence_jump=0.0, coherence_jump_sigma=0.93, tiler=None, x0_hook=None):
    """Shared entry point for all KreaPhoton sampler nodes.

    tiler / x0_hook (v1.5 Upscale): forwarded to every segment's loop (see
    kreaphoton_sampler_loop) - the tiled model call and the LF anchor apply to
    the identity and texture phase models alike.

    Variety (both axes) is applied INSIDE the single sampler lifecycle at the
    boundary step variety_boundary_index(sigmas, variety_end): lf_recompose on
    the loop state (latent axis) and a guider-side switch to the tap-rotated
    positive (cond axis). Restart is likewise not a split point - it is encoded
    in the sigmas array and handled per-step by the loop.

    The clean_model split (composition_end) is the one remaining multi-segment
    path: composition phase (sigma > composition_end) on the clean checkpoint,
    LoRA identity/detail phase on `model`. ZPhoton precedent, M3-verified exact
    rescale cancellation at the shared boundary sigma, still unvalidated on
    krea2 LoRA stacks (docs/04 item 7) and therefore still guarded: eta0 is
    forced to 0 whenever the split occurs (S9 speckle/mosaic precedent).
    """
    sigmas = validate_sigmas(sigmas)
    sig_list = [float(s) for s in sigmas]

    # variety consumes guidance budget (presets.GUIDANCE['variety_budget'], V16b)
    delta = effective_delta(delta, guidance_mode, variety_a_latent)

    kw = dict(guidance_mode=guidance_mode, flat_cfg=flat_cfg, delta=delta, lo=lo, hi=hi,
              contraction=contraction, per_channel_contraction=per_channel_contraction,
              manifold_std=manifold_std, manifold_mean=manifold_mean,
              detail_amount=detail_amount, detail_start=detail_start, detail_end=detail_end,
              detail_peak=detail_peak, order=order, eta0=eta0, sigma_gate=sigma_gate,
              x0_extrapolation=x0_extrapolation, guidance_rescale=guidance_rescale,
              pag_scale=pag_scale, pag_lo=pag_lo, pag_hi=pag_hi, pag_blocks=pag_blocks,
              restart_hook=restart_hook, coherence_jump_sigma=coherence_jump_sigma,
              tiler=tiler, x0_hook=x0_hook)
    # the jump belongs to the first segment only (the step at/below its sigma
    # lives in the composition/identity head of the schedule); later segments
    # must not re-apply it
    kw_jump = float(coherence_jump)

    # --- variety: rotated positive prepared once, boundary index computed once ---
    positive_variety = None
    if variety_a_cond > 0.0:
        positive_variety = _rotate_conditioning(positive, variety_cond_taps, variety_a_cond, variety_seed)
    use_variety = variety_a_latent > 0.0 or positive_variety is not None
    variety_idx = variety_boundary_index(sig_list, variety_end) if use_variety else None
    vkw = dict(positive_variety=positive_variety if variety_idx is not None else None,
               variety_end=variety_end, variety_seed=variety_seed)

    # --- phase models (v1.4): composition on clean_model, identity on model,
    # texture (the restart segment) on texture_model. Multi-segment split; the
    # old S9 "eta0 + split -> speckle" guard is gone - its root cause was the
    # per-segment detail envelope (kreaphoton_sampler_loop progress_offset):
    # every segment now gets its global step offset, so the split integrates
    # the SAME sigma_model sequence as one lifecycle (V16 F_clean validated).
    segments = plan_segments(sig_list, composition_end=composition_end, has_clean=clean_model is not None,
                             texture_start=texture_start, has_texture=texture_model is not None)
    if len(segments) == 1:
        return _run_one(model, positive, negative, latent_dict, sigmas,
                        seed=seed, noise=noise, add_noise=add_noise,
                        variety_a_latent=variety_a_latent if variety_idx is not None else 0.0,
                        variety_state="pending" if variety_idx is not None else "off",
                        coherence_jump=kw_jump, **vkw, **kw)

    phase_models = {"composition": clean_model, "identity": model, "texture": texture_model}
    cur = latent_dict
    cur_noise, cur_add = noise, add_noise
    for k, (a, b, phase) in enumerate(segments):
        seg = sigmas[a:b + 1]
        m = phase_models[phase]
        if variety_idx is None:
            state, a_lat = "off", 0.0
        elif a <= variety_idx < b:
            state, a_lat = "pending", variety_a_latent
        elif variety_idx < a:
            state, a_lat = "active", 0.0
        else:
            state, a_lat = "off", 0.0
        # Each segment's KSAMPLER seeds its OWN ancestral/restart RNG from
        # `restart_seed`. Reusing the plain `seed` for every segment would replay
        # the IDENTICAL draw sequence at segment 2's first step as at segment 1's
        # - a "double-exposure" of one frozen noise realization at two points of
        # the trajectory (S9 mosaic precedent). Large per-segment offset
        # decorrelates each segment's stream.
        seg_seed = (int(seed) + k * 1_000_003) & 0xffffffffffffffff
        # jump only in the segment that contains its trigger step
        jump_here = 0.0
        if kw_jump > 0.0:
            trig = variety_boundary_index(sig_list, coherence_jump_sigma)
            if trig is not None and a <= trig < b:
                jump_here = kw_jump
        cur = _run_one(m, positive, negative, cur, seg,
                       seed=seg_seed, noise=cur_noise, add_noise=cur_add,
                       variety_a_latent=a_lat, variety_state=state,
                       progress_offset=a, progress_total=len(sig_list) - 1,
                       coherence_jump=jump_here, **vkw, **kw)
        cur_noise, cur_add = None, False
    return cur
