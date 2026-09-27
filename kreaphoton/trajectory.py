"""
Pure math on the model's x0-prediction trajectory (Sampler v2, iteration 2; proposal
E:\\dk\\KreaPhoton_Sampler_v2_Technical_Proposal.md §18, §19, §22):

  x0_zero_delta  terminal extrapolation of the x0 estimate to sigma = 0 - linear (the v1
                 formula, same float operation order) or quadratic Lagrange through the
                 last three model calls (§19).
  trust_clamp    the §18.1 trust region with a LOCAL reference. As written in the
                 proposal (reference = the GLOBAL rms of the last x0 step) it is
                 degenerate for linear extrapolation: delta is a scalar multiple of that
                 step, so the clamp is just a cap on the scalar. Against the 3x3 local rms
                 of the step it limits isolated spikes - the overshoot risk - and leaves
                 smooth extrapolation alone.
  motion         relative change of the x0 estimate between two model calls, split into
                 low / mid / high frequency bands by average-pool pyramids (§22 P3.1) -
                 the signal the adaptive predictor-corrector and self-refine decide on.

No comfy import: unit-tested standalone (tests/test_trajectory.py).
"""
import torch
import torch.nn.functional as F

QUAD_MAX_WEIGHT = 8.0     # |Lagrange weight| above this = ill-conditioned (near-duplicate sigmas):
                          # the quadratic falls back to linear. A uniform grid gives (1, -3, 3).
LF_POOL = 8               # latent px: LF band = 8x8 block mean (64 image px)
HF_POOL = 2               # latent px: HF band = residual below 2x2 block mean


def x0_zero_delta(hist, weight=1.0, order=1, max_factor=2.0):
    """weight * (x0(sigma=0) - x0_last) from hist = [(x0, sigma), ...] oldest first, or None
    when there is nothing to extrapolate (< 2 points, zero spacing).

    order 1: weight * f * (x0_last - x0_prev), f = s_last / (s_prev - s_last) capped at
             max_factor - bit-for-bit the v1 terminal formula.
    order 2: Lagrange through the last three points, evaluated at 0; linear with < 3
             points or ill-conditioned weights (QUAD_MAX_WEIGHT)."""
    if len(hist) < 2:
        return None
    (xp, sp), (xl, sl) = hist[-2], hist[-1]
    if order >= 2 and len(hist) >= 3:
        (xa, sa), (xb, sb), (xc, sc) = hist[-3], hist[-2], hist[-1]
        den_a, den_b, den_c = (sa - sb) * (sa - sc), (sb - sa) * (sb - sc), (sc - sa) * (sc - sb)
        if min(abs(den_a), abs(den_b), abs(den_c)) > 1e-12:
            la, lb, lc = sb * sc / den_a, sa * sc / den_b, sa * sb / den_c
            if max(abs(la), abs(lb), abs(lc)) <= QUAD_MAX_WEIGHT:
                return float(weight) * (la * xa + lb * xb + (lc - 1.0) * xc)
    if sp - sl <= 1e-6:
        return None
    factor = min(float(max_factor), sl / (sp - sl))
    return float(weight) * factor * (xl - xp)


def local_rms(t, k=3):
    """Per-channel RMS over a k x k window of the last two dims (edge windows shrink)."""
    shape = t.shape
    flat = t.reshape(-1, 1, shape[-2], shape[-1])
    m = F.avg_pool2d(flat * flat, k, stride=1, padding=k // 2, count_include_pad=False)
    return m.sqrt().reshape(shape)


def trust_clamp(delta, ref, k):
    """delta limited pointwise to +-k * local_rms(ref); k <= 0 returns delta unchanged."""
    if k <= 0.0:
        return delta
    lim = float(k) * local_rms(ref)
    return torch.maximum(torch.minimum(delta, lim), -lim)


def _pool_up(t, p):
    shape = t.shape
    flat = t.reshape(-1, 1, shape[-2], shape[-1])
    low = F.avg_pool2d(flat, p, stride=p, ceil_mode=True)
    return F.interpolate(low, size=(shape[-2], shape[-1]), mode="nearest").reshape(shape)


def _bands(t):
    lf = _pool_up(t, LF_POOL)
    below_hf = _pool_up(t, HF_POOL)
    return {"all": t, "lf": lf, "mf": below_hf - lf, "hf": t - below_hf}


def motion(x0, x0_ref, eps=1e-8):
    """Relative change of the x0 estimate per band, {'all','lf','mf','hf'}: rms(band(x0 -
    x0_ref)) / rms(band(x0_ref)), plus 'cos' (cosine similarity). Per sample; the batch
    reports its most-moved sample (max change, min cos) - a decision taken for the whole
    batch must cover the sample that needs it."""
    diff = _bands((x0 - x0_ref).float())
    ref = _bands(x0_ref.float())
    dims = tuple(range(1, x0.ndim))
    out = {}
    for key in ("all", "lf", "mf", "hf"):
        num = diff[key].pow(2).mean(dim=dims).sqrt()
        den = ref[key].pow(2).mean(dim=dims).sqrt()
        out[key] = float((num / (den + eps)).max())
    a, b = x0.float().flatten(1), x0_ref.float().flatten(1)
    out["cos"] = float(F.cosine_similarity(a, b, dim=1).min())
    return out
