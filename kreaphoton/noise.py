"""
Initial-noise manifold contraction (research/results/E3_real.json).

The sampler loop lives in NORMALIZED (Wan21 model) space (planning-council
B1/F9 code trace: comfy/sample.py's prepare_noise draws unit N(0,1) directly
in that space; comfy/samplers.py:1214-1216 CFGGuider.inner_sample applies
process_latent_in ONLY to a non-empty latent_image, never to the noise
tensor). MANIFOLD_STD/MEAN from presets.py therefore apply DIRECTLY here, no
rescale through the Wan21 latent_format constants needed.

Measured fact (research/results/E3_real.json, 8 real photographs through
qwen_image_vae): the photo manifold's normalized std is ~0.467 globally
(per-channel 0.31-0.50) - the sampler's stock unit-N(0,1) initial noise is
~2.1x WIDER than the manifold it is meant to converge onto. contract_noise
gives a knob to shrink it, strength=1.0 (stock, no-op) down to strength=0.47
(full contraction to the measured manifold std).
"""
import torch


def contract_noise(noise: torch.Tensor, strength: float, per_channel: bool = False,
                    manifold_std=None, manifold_mean=None) -> torch.Tensor:
    """noise: 5D unit N(0,1) tensor, shape (B, 16, 1, H, W), NORMALIZED space.

    strength: 1.0 = stock (exact no-op, bit-identical). Values below 1.0 scale
    the noise toward the measured manifold std; per_channel=True additionally
    applies presets.MANIFOLD_STD/MEAN as a per-channel vector instead of the
    scalar strength (Advanced node only).

    Applied to the RAW unit-noise tensor BEFORE noise_scaling
    (sigma*eps + (1-sigma)*x0, comfy/model_sampling.py CONST) - never inside
    an EmptyLatent node, where a non-zero "empty" latent would incorrectly
    pass process_latent_in and get shifted (planning-council F11/R12).
    """
    if strength == 1.0 and not per_channel:
        return noise

    if per_channel:
        assert manifold_std is not None, "per_channel=True requires manifold_std"
        std = torch.as_tensor(manifold_std, dtype=noise.dtype, device=noise.device)
        std = std.view(1, -1, 1, 1, 1)
        out = noise * std
        if manifold_mean is not None:
            mean = torch.as_tensor(manifold_mean, dtype=noise.dtype, device=noise.device)
            out = out + mean.view(1, -1, 1, 1, 1)
        return out

    return noise * strength


def slerp_noise(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation of two noise fields, norm-interpolated so the
    result stays ~unit-N(0,1) (on-manifold) at every t.

    On krea2-turbo the composition is set by the INITIAL noise field (proven
    2026-07-06: a fresh seed moves layout, a variance-preserving LF-band rotation
    does not). slerp between two seeds' noise therefore walks a coherent
    composition path between them - t=0 is a's composition, t=1 is b's, and the
    distillate coheres every intermediate into a valid photoreal image. Note:
    this moves identity WITH composition (they are coupled on this model), so it
    is a composition explorer between two seeds, not fixed-identity variety.

    t<=0 returns a unchanged (bit-identical), t>=1 returns b. High-dim gaussians
    are near-orthogonal, so the interpolation angle is well away from 0/pi."""
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    af, bf = a.reshape(-1).double(), b.reshape(-1).double()
    na, nb = af.norm(), bf.norm()
    ua = af / na.clamp_min(1e-12)
    ub = bf / nb.clamp_min(1e-12)
    dot = (ua * ub).sum().clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        out = (1.0 - t) * af + t * bf
    else:
        out = (torch.sin((1.0 - t) * omega) / so) * af + (torch.sin(t * omega) / so) * bf
    target_norm = (1.0 - t) * na + t * nb
    out = out / out.norm().clamp_min(1e-12) * target_norm
    return out.reshape(a.shape).to(a.dtype)
