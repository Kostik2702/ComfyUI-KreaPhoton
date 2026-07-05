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
PLUNGE_SIGMA_FLOOR = 0.75              # M1 default; inert while every v1 preset has plunge=False


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


def build_schedule(n_steps: int, alpha: float = ALPHA, restart_frac: float = 0.0,
                    sigma_r: float = 0.6, plunge: bool = False, q: float = 1.0) -> torch.Tensor:
    """Analytic schedule family for krea2.

    Structure segment in t-space: n_m points
        t_i = t_hi - (t_hi - t_lo) * (i / n_m)^q      (q>1 = denser at high sigma)
    mapped through sigma_from_t, then 0.0 appended.

    plunge=True: structure stops at PLUNGE_SIGMA_FLOOR, the appended 0.0 becomes a
    plunge step (distilled model one-shots x0 from mid sigma) — v1 presets keep this
    False pending calibration (docs/04 KNOWN UNKNOWNS).

    restart_frac>0: ascending jump to sigma_r encoded directly in SIGMAS, then linear
    descent back to 0 (n_r model calls carved out of the n_steps budget — total model
    calls == n_steps regardless of restart_frac, per M1[C]).

    q is fixed at 1.0 for every v1 preset (required for the sgm_uniform identity to
    hold); exposed as a kwarg only for the Advanced node / future calibration.
    """
    n_steps = int(n_steps)
    n_r = 0
    if restart_frac > 0.0:
        n_r = max(1, min(int(round(n_steps * restart_frac)), n_steps - 2))
    n_m = n_steps - n_r

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

    if n_r > 0:
        for j in range(n_r):
            sigs.append(sigma_r * (1.0 - j / n_r))
        sigs.append(0.0)

    return torch.tensor(sigs, dtype=torch.float32)


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
