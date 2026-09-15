"""
Analytic sigma-schedule family for Krea2 (flow matching, CONST) — ported from
research/models/m1_schedule_family.py (all M1 proofs: identity vs stock
sgm_uniform grid at <1e-6, N=6..24 scaling, restart = single ascending jump).

Comfy references (E:\\CUI portable\\ComfyUI-torch2.9-cu130-cp313-v1.2\\ComfyUI):
  comfy/model_sampling.py:382  flux_time_shift(mu, s, t) = e^mu/(e^mu+(1/t-1)^s)
  comfy/model_sampling.py:395  ModelSamplingFlux sigma table = sigma(arange(1,10001)/10000)
  comfy/model_sampling.py:408  timestep(sigma) = sigma  (raw sigma fed to the model)
  comfy/samplers.py:670        normal_scheduler(..., sgm=True)
"""
from dataclasses import dataclass

import torch

MU = 1.15
ALPHA = 2.718281828459045 ** MU        # e^1.15 = 3.15819...; kept as a literal-derived
                                        # constant so this module has zero torch/math
                                        # dependency surprises at import time.
TIMESTEPS = 10000                      # ModelSamplingFlux table size (comfy/model_sampling.py:395)
PLUNGE_SIGMA_FLOOR = 0.75              # M1 default. ACTIVE: all turbo presets ship plunge=True
                                        # (V2, 2026-07-05), so composition crystallizes at this
                                        # floor's plunge readout; anything re-noised below 0.75
                                        # (e.g. restart @ sigma_r=0.65) can only affect texture.
MIN_STEPS_FOR_RESTART = 4              # below this the restart segment is dropped (F05 contract:
                                        # model calls == steps for EVERY steps >= 1)
SIGMA_MAX_TOLERANCE = 1e-4             # flow sigma lives in [0, 1]; anything above is a foreign
                                        # (EPS/VP) schedule the CONST re-noise math cannot integrate
MAX_ASCENDING_JUMPS = 2                # self-refine pass + restart segment (v1.4); 3+ = foreign/malformed


def sigma_from_t(t: float, alpha: float) -> float:
    """time_snr_shift form: sigma(t) = alpha*t / (1 + (alpha-1)*t).
    Algebraically identical to comfy's flux_time_shift(mu, 1, t) with alpha = e^mu
    (proven in research/models/m1_schedule_family.py [A0], max diff < 1e-14)."""
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def t_from_sigma(sigma: float, alpha: float) -> float:
    return sigma / (alpha - (alpha - 1.0) * sigma)


def flux_time_shift(mu: float, t: float) -> float:
    """comfy/model_sampling.py:382 (sigma exponent = 1.0). Cross-check only —
    build_schedule() itself uses sigma_from_t/alpha, proven algebraically identical."""
    import math
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))


# --- Resolution-aware shift (v1.4). Krea 2 canon (docs/01 §2, diffusers pipeline): the
#     schedule shift mu is linear in the image token count, base 0.5 at 256 tokens ->
#     max 1.15 at 6400 tokens (Flux-style). ComfyUI's Krea2 config pins 1.15 for every
#     size; so did every KreaPhoton preset (alpha = e^1.15). Above 6400 tokens (the L/XL
#     tiers, 1088x1600 = 6800) nothing changes; below it the structure phase gets the
#     canonical, softer schedule (1024x1024 = 4096 tokens -> mu 0.906, alpha 2.47). ---
SHIFT = {
    "base_mu": 0.5,
    "max_mu": MU,
    "base_tokens": 256,
    "max_tokens": 6400,
    "resolution_aware": True,      # simple-node policy; the Scheduler needs its `latent` input
}


def canonical_mu(tokens: int) -> float:
    """Krea 2 dynamic shift for an image of `tokens` DiT tokens (latent h*w / 4)."""
    t0, t1 = SHIFT["base_tokens"], SHIFT["max_tokens"]
    m0, m1 = SHIFT["base_mu"], SHIFT["max_mu"]
    u = (float(tokens) - t0) / float(t1 - t0)
    return m0 + (m1 - m0) * max(0.0, min(1.0, u))


def alpha_for_latent(h_lat: int, w_lat: int, alpha_max: float = ALPHA) -> float:
    """Resolution-aware schedule steepness (v1.4): e^mu with the Krea 2 canonical
    dynamic shift for this latent grid (tokens = h*w/4, DiT patch 2x2), capped by
    alpha_max (the preset's alpha == e^1.15, so the L/XL tiers are unchanged)."""
    tokens = (int(h_lat) * int(w_lat)) // 4
    return min(float(alpha_max), 2.718281828459045 ** canonical_mu(tokens))


def build_schedule(n_steps: int, alpha: float = ALPHA, restart_frac: float = 0.0,
                    sigma_r: float = 0.6, plunge: bool = False, q: float = 1.0,
                    refine_steps: int = 0, refine_sigma: float = 0.85) -> torch.Tensor:
    """Analytic schedule family for krea2.

    Structure segment in t-space: n_m points
        t_i = t_hi - (t_hi - t_lo) * (i / n_m)^q      (q>1 = denser at high sigma)
    mapped through sigma_from_t, then 0.0 appended.

    plunge=True: structure stops at PLUNGE_SIGMA_FLOOR, the appended 0.0 becomes a
    plunge step (distilled model one-shots x0 from mid sigma). ACTIVE on all turbo
    presets since V2 (2026-07-05); raw/experimental keeps it False.

    restart_frac>0: ascending jump to sigma_r encoded directly in SIGMAS, then linear
    descent back to 0 (n_r model calls carved out of the n_steps budget — total model
    calls == n_steps regardless of restart_frac, per M1[C]).

    q is fixed at 1.0 for every v1 preset (required for the sgm_uniform identity to
    hold); exposed as a kwarg only for the Advanced node / future calibration.

    refine_steps>0 (v1.4 self-refine, Power-Nodes "coherence pass" idea): after the
    plunge readout a SECOND ascending jump re-noises the draft x0 to refine_sigma
    and refine_steps model calls descend to PLUNGE_SIGMA_FLOOR, plunge again, then
    the restart segment follows. The model re-decides the identity-phase content
    with the whole draft as a prior (0.85 ~ img2img denoise 0.6: layout kept,
    incoherent detail re-rendered). Adds refine_steps model calls on top of n_steps
    (documented: total calls == n_steps + refine_steps).
    """
    n_steps = max(1, int(n_steps))
    n_r = 0
    # Small-N contract (audit F05): a restart needs >= 2 structure points + a
    # plunge/terminal + >= 1 restart step, so below MIN_STEPS_FOR_RESTART the
    # restart segment is dropped instead of being force-fitted (the old
    # max(1, ...) clamp produced 3 model calls for steps=1/2).
    if restart_frac > 0.0 and n_steps >= MIN_STEPS_FOR_RESTART:
        n_r = max(1, min(int(round(n_steps * restart_frac)), n_steps - 2))
    n_m = n_steps - n_r
    if n_m < 2:
        plunge = False   # a plunge needs >= 2 structure points; 1-step = plain one-shot

    sigs = []
    t_hi = 1.0
    if plunge:
        t_lo_eff = t_from_sigma(PLUNGE_SIGMA_FLOOR, alpha)
        pts = max(2, n_m)
        for i in range(pts):
            u = i / (pts - 1)
            t = t_hi - (t_hi - t_lo_eff) * (u ** q)
            sigs.append(sigma_from_t(t, alpha))
    else:
        t_lo_eff = alpha / (alpha + (TIMESTEPS - 1.0))
        for i in range(n_m):
            u = i / n_m
            t = t_hi - (t_hi - t_lo_eff) * (u ** q)
            sigs.append(sigma_from_t(t, alpha))
    sigs.append(0.0)

    k = int(refine_steps)
    if k > 0:
        # self-refine segment: refine_sigma -> ... -> floor (k points, uniform in t),
        # then a plunge readout; only meaningful with a plunge structure, so the
        # descent always ends at the plunge floor and re-plunges.
        rs = max(PLUNGE_SIGMA_FLOOR + 1e-3, min(0.999, float(refine_sigma)))
        t_a, t_b = t_from_sigma(rs, alpha), t_from_sigma(PLUNGE_SIGMA_FLOOR, alpha)
        for i in range(k):
            u = i / max(k - 1, 1)
            sigs.append(sigma_from_t(t_a - (t_a - t_b) * u, alpha) if k > 1 else rs)
        sigs.append(0.0)

    if n_r > 0:
        for j in range(n_r):
            sigs.append(sigma_r * (1.0 - j / n_r))
        sigs.append(0.0)

    return torch.tensor(sigs, dtype=torch.float32)


def refine_schedule(n_steps: int, alpha: float = ALPHA, denoise: float = 1.0) -> torch.Tensor:
    """Partial descent for img2img / refine (denoise < 1.0).

    Oversample a PLAIN descent (no restart, no plunge - both are full-txt2img
    detail-recovery tricks that fight a partial refine) at round(n_steps/denoise)
    steps, then take the last n_steps+1 sigmas. This is the standard denoise-strength
    truncation (comfy's own convention; parses cleanly through infer_segment_map,
    which is explicitly written to treat a denoise<1 truncation as no-restart).

    Returns exactly n_steps+1 sigmas descending to 0. denoise>=1.0 collapses to the
    full plain descent (n_steps+1 values) - callers use build_schedule for the real
    txt2img path and only reach here when denoise < 1.0.
    """
    d = max(1e-3, min(1.0, float(denoise)))
    total = max(int(n_steps), int(round(n_steps / d)))
    full = build_schedule(total, alpha=alpha, restart_frac=0.0, plunge=False)
    return full[-(int(n_steps) + 1):]


def count_model_calls(sigmas) -> int:
    """Number of model evaluations kreaphoton_sampler_loop performs on this
    SIGMAS array: every step except the ascending restart jump (free re-noise)
    and steps starting at sigma=0 (plunge readout already happened)."""
    s = [float(v) for v in sigmas]
    calls = 0
    for i in range(len(s) - 1):
        if s[i + 1] > s[i] + 1e-6:
            continue
        if s[i] <= 1e-6:
            continue
        calls += 1
    return calls


def validate_sigmas(sigmas) -> torch.Tensor:
    """Production validation boundary for every SIGMAS array the samplers run
    (own presets AND foreign arrays on the Advanced node's SIGMAS input -
    audit F04). Raises ValueError with the offending indices; never silently
    falls back, because a malformed schedule integrates into a meaningless
    trajectory without any visible error (multiple ascending jumps re-noise
    several times, sigma>1 breaks the CONST re-noise formula, NaN poisons x).

    Accepted: finite 1-D float tensor, >= 2 values, every value in [0, 1],
    first value > 0, at most MAX_ASCENDING_JUMPS strict ascending jumps (the
    self-refine pass and the restart segment), descending (or flat) everywhere
    else. A final 0.0 is NOT required (a partial descent for a multi-stage graph
    is legal comfy usage).
    """
    if not torch.is_tensor(sigmas):
        try:
            sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
        except Exception as e:  # noqa: BLE001 - user input, report it whole
            raise ValueError(f"KreaPhoton: SIGMAS must be a 1-D tensor, got {type(sigmas).__name__} ({e})")
    if sigmas.ndim != 1:
        raise ValueError(f"KreaPhoton: SIGMAS must be 1-D, got shape {tuple(sigmas.shape)}")
    if sigmas.numel() < 2:
        raise ValueError(f"KreaPhoton: SIGMAS needs at least 2 values (got {sigmas.numel()})")
    if not bool(torch.isfinite(sigmas).all()):
        bad = [i for i, v in enumerate(sigmas.tolist()) if v != v or v in (float("inf"), float("-inf"))]
        raise ValueError(f"KreaPhoton: SIGMAS contains NaN/Inf at indices {bad[:8]}")
    s = sigmas.tolist()
    neg = [i for i, v in enumerate(s) if v < 0.0]
    if neg:
        raise ValueError(f"KreaPhoton: SIGMAS has negative values at indices {neg[:8]} - krea2 flow sigma lives in [0, 1]")
    high = [i for i, v in enumerate(s) if v > 1.0 + SIGMA_MAX_TOLERANCE]
    if high:
        raise ValueError(f"KreaPhoton: SIGMAS has values > 1.0 at indices {high[:8]} (values {[round(s[i], 4) for i in high[:4]]}) "
                         f"- this is not a krea2/flow schedule (CONST models use sigma in [0, 1])")
    if s[0] <= 1e-6:
        raise ValueError("KreaPhoton: SIGMAS starts at 0 - nothing to sample")
    jumps = [i for i in range(len(s) - 1) if s[i + 1] > s[i] + 1e-6]
    if len(jumps) > MAX_ASCENDING_JUMPS:
        raise ValueError(f"KreaPhoton: SIGMAS has {len(jumps)} ascending jumps at indices {jumps} - a KreaPhoton "
                         f"schedule encodes at most {MAX_ASCENDING_JUMPS} ascending re-noise jumps (self-refine + "
                         f"restart); feed only KreaPhoton Scheduler output (or a plain descending schedule) into "
                         f"the samplers")
    return sigmas


@dataclass
class SegmentMap:
    """Restart/plunge boundary map inferred from a SIGMAS tensor (own or foreign).

    restart_start is the load-bearing field (drives the run_sampling re-noise step);
    plunge_idx is best-effort only (plunge is not reliably detectable from an
    arbitrary external SIGMAS array — no calibrated preset uses it yet, so this is
    forward-compat scaffolding, not asserted in tests beyond a sanity heuristic).
    """
    structure_end: int
    plunge_idx: int | None
    restart_start: int | None
    ambiguous: bool


def infer_segment_map(sigmas: torch.Tensor) -> SegmentMap:
    """Graceful-degrade parser for external SIGMAS (Advanced node input): denoise<1
    truncation and duplicate boundary values must not false-positive a restart.

    Restart predicate (ZPhoton-style, M1[C]): s[i+1] > s[i] + 1e-6, i.e. a STRICT
    ascending jump. Multiple jumps -> ambiguous=True, restart_start=None (never
    guess which one is the real restart; callers must fall back to plain
    integration, no re-noise segment machinery).
    """
    n = sigmas.shape[0]
    jumps = [i for i in range(n - 1) if float(sigmas[i + 1]) > float(sigmas[i]) + 1e-6]

    if len(jumps) == 0:
        return SegmentMap(structure_end=n - 1, plunge_idx=None, restart_start=None, ambiguous=False)

    if len(jumps) > 1:
        return SegmentMap(structure_end=n - 1, plunge_idx=None, restart_start=None, ambiguous=True)

    j = jumps[0]
    # best-effort plunge heuristic: an unusually large single descending step just
    # before the restart jump (relative to the median step size in that segment)
    seg = sigmas[:j + 1]
    plunge_idx = None
    if seg.shape[0] >= 2:
        steps = (seg[:-1] - seg[1:]).abs()
        if steps.numel() >= 2:
            median_step = steps.median().item()
            last_step = steps[-1].item()
            if median_step > 0 and last_step > 4.0 * median_step:
                plunge_idx = j

    return SegmentMap(structure_end=j, plunge_idx=plunge_idx, restart_start=j + 1, ambiguous=False)
