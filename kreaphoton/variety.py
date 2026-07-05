"""
Two-axis variety for Krea2 — ported from research/models/m4_variety_ops.py
(all M4 proofs: LF-mode grid-invariance, variance preservation, HF carry,
corr(lf',lf)=sqrt(1-a^2), cond-rotation orthogonalization, untouched-tap
bit-exactness, SVE-additive std-inflation comparison).

(a) lf_recompose: latent composition axis. AC-only correction against the
    ZPhoton original (planning-council M4): Wan21 channels are NOT zero-mean
    (research/results/E3_real.json normalized_per_channel.mean, presets.py
    MANIFOLD_MEAN) — rotating the full LF band (DC+AC) would fold a per-channel
    palette offset into the energy-matching step and break the corr law.
    Here the DC (per-channel spatial mean) is split off BEFORE rotation and
    added back untouched.

(b) cond_tap_rotation: conditioning semantic axis, replacing KreaSeedVarianceEnhancer's
    additive U-noise (not variance-preserving, +20% std inflation at equal
    decorrelation — proven in M4[b2]) with an orthogonalized rotation. Ground
    truth for the tap-major layout: comfy/text_encoders/krea2.py:63-65
    `out.permute(0, 2, 1, 3).reshape(b, seq, n*h)` — tap k occupies the
    CONTIGUOUS slice [:, :, k*2560:(k+1)*2560] (verified against the source,
    not assumed; community bug precedent: Krea2T-Enhancer sliced 24x1280
    chunks instead of 12x2560 taps, crossing a tap boundary).
"""
import torch

N_CYC_REF = 16          # ZPhoton anchor: 128 * 0.25 / 2 (M4)
TAP_DIM = 2560
N_TAPS = 12


# ======================================================================
# (a) lf_recompose — latent composition axis
# ======================================================================
def _lf_mask(h: int, w: int, cutoff_y: float, cutoff_x: float, device, dtype) -> torch.Tensor:
    """Hard elliptical low-pass mask in rfft layout (h, w//2+1)."""
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype).unsqueeze(1)
    fx = torch.fft.rfftfreq(w, device=device, dtype=dtype).unsqueeze(0)
    r2 = (fy / (cutoff_y * 0.5)) ** 2 + (fx / (cutoff_x * 0.5)) ** 2
    return (r2 <= 1.0).to(dtype)


def _split_bands(x: torch.Tensor, mask: torch.Tensor):
    X = torch.fft.rfft2(x)
    lf = torch.fft.irfft2(X * mask, s=x.shape[-2:])
    return lf, x - lf


def auto_cutoff(h: int, w: int, n_cyc: float = N_CYC_REF):
    """cutoff_axis(G) = 2*N_cyc/G (planning-council M4 scaling rule) — keeps a
    fixed number of composition cycles-per-image regardless of latent grid size."""
    return min(1.0, 2.0 * n_cyc / h), min(1.0, 2.0 * n_cyc / w)


def lf_recompose(x5: torch.Tensor, seed_v: int, a: float, cutoff: tuple | None = None) -> torch.Tensor:
    """x5: 5D (B, C, 1, H, W) Wan21 latent, NORMALIZED space. Returns same shape.

    a=0 -> exact no-op (bit-identical, no FFT/RNG work performed).
    """
    a = float(max(0.0, min(1.0, a)))
    if a <= 0.0:
        return x5

    b, c, t, h, w = x5.shape
    assert t == 1, f"lf_recompose expects a single-frame 5D latent, got T={t}"
    x = x5[:, :, 0].double()
    device, dtype = x.device, x.dtype

    cy, cx = cutoff if cutoff is not None else auto_cutoff(h, w)
    mask = _lf_mask(h, w, cy, cx, device, dtype)

    lf, hf = _split_bands(x, mask)
    # split off DC (per-channel palette): rotate only the AC part of the band
    lf_dc = lf.mean(dim=(2, 3), keepdim=True)
    lf_ac = lf - lf_dc

    gen = torch.Generator(device="cpu").manual_seed(seed_v)
    g = torch.randn((b, c, h, w), generator=gen, dtype=dtype).to(device)
    g, _ = _split_bands(g, mask)
    g = g - g.mean(dim=(2, 3), keepdim=True)

    lf_n = lf_ac.reshape(b, c, -1).norm(dim=2).view(b, c, 1, 1)
    g_n = g.reshape(b, c, -1).norm(dim=2).clamp_min(1e-6).view(b, c, 1, 1)
    g_hat = g * (lf_n / g_n)   # band-energy match on the AC part

    y = lf_dc + (1.0 - a * a) ** 0.5 * lf_ac + a * g_hat + hf

    # restore per-channel mean/std (palette/style) exactly
    ym = y.mean(dim=(2, 3), keepdim=True)
    ys = y.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    xm = x.mean(dim=(2, 3), keepdim=True)
    xs = x.std(dim=(2, 3), keepdim=True)
    y = (y - ym) / ys * xs + xm

    return y.unsqueeze(2).to(x5.dtype)


# ======================================================================
# (b) cond_tap_rotation — conditioning semantic axis
# ======================================================================
def cond_tap_rotation(cond: torch.Tensor, taps: tuple, a: float, seed: int) -> torch.Tensor:
    """cond: (B, seq, 12*2560) tap-major (comfy/text_encoders/krea2.py:63-65).
    Variance-preserving rotation of the selected taps toward fresh orthogonal
    noise; untouched taps are returned bit-exact (never even read/written)."""
    a = float(max(0.0, min(1.0, a)))
    out = cond.clone()
    if a <= 0.0:
        return out

    b = cond.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for k in taps:
        sl = slice(k * TAP_DIM, (k + 1) * TAP_DIM)
        for bi in range(b):
            v = cond[bi, :, sl].double()
            mu = v.mean()
            vc = (v - mu).reshape(-1)
            g = torch.randn(vc.shape, generator=gen, dtype=torch.float64)
            g = g - g.mean()
            g = g - (g @ vc) / (vc @ vc) * vc          # exact orthogonalization vs vc
            g = g * (vc.norm() / g.norm())              # band-energy match
            vpc = (1.0 - a * a) ** 0.5 * vc + a * g
            out[bi, :, sl] = (vpc.reshape(v.shape) + mu).to(cond.dtype)
    return out


def apply_variety(level: str, variety_levels: dict, latent5d: torch.Tensor = None,
                  cond: torch.Tensor = None, seed: int = 0,
                  cond_taps: tuple = (7, 8, 9, 10)) -> tuple:
    """level -> (a_latent, a_cond) from presets.VARIETY_LEVELS. Dispatches to
    lf_recompose / cond_tap_rotation for whichever inputs are given; the
    Advanced node calls lf_recompose/cond_tap_rotation directly with
    independent a values instead (axes exposed separately)."""
    a_latent, a_cond = variety_levels[level]
    out_latent = lf_recompose(latent5d, seed_v=seed, a=a_latent) if latent5d is not None else None
    out_cond = cond_tap_rotation(cond, cond_taps, a_cond, seed=seed) if cond is not None else None
    return out_latent, out_cond
