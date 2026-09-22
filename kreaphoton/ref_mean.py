"""Reference-mean guidance (spike, 2026-09-22) - "Follow the Mean: Reference-Guided
Flow Matching" (arXiv 2605.10302) transplanted onto the KreaPhoton sampler loop.

The idea: a rectified-flow velocity is (up to the interpolant) the posterior mean of
the clean endpoint given the noisy state. With a bank of reference latents R = {r_n}
that mean has a closed form under the Gaussian noise model of the loop state
x = (1 - s) x0 + s eps  (comfy CONST model space, s = sigma):

    w_n(x, s) = softmax_n( [ (1-s) <x, r_n> - 1/2 (1-s)^2 ||r_n||^2 ] / (s^2 tau) )
    mu_hat(x, s) = sum_n w_n r_n

and the model's own x0 estimate (`denoised`) is pulled toward it:

    denoised' = denoised + beta(s) * m * (mu_hat - denoised)

m is the (latent-resolution) face mask: the inner products are taken over the mask
only (the crop background must not vote), and the shift is confined to it (comfy's
inpaint wrapper has already blended the source latent outside; the paste discards
it anyway). beta(s) ramps linearly from 0 at `lo` to `beta` at `hi` and stays there
above - at low sigma the posterior is a hard nearest-neighbour and the shift would
paste one reference's pixels into the crop, so it is faded out.

tau = 1 is the exact posterior (in 2.6e5 dimensions that is a near-argmax over the
bank); larger tau flattens the weights toward the plain bank mean.

MEASURED 2026-09-22 (research/face_identity_rig ab8_*.log, 20 runs, 3 seeds, base
AnnaMin 2.0, 8 references): NO identity gain beyond seed noise. ArcFace to the 120-face
centroid: baseline 0.660 vs beta 0.4 with a flat bank mean (tau 1e5) 0.674, per-seed
-0.014..+0.038. The exact posterior (tau 1) is a hard argmax toward one reference and
HURTS (pass-1 gate 0.678 -> 0.637 at beta 0.4, 0.537 at 0.7); beta 0.7 visibly blurs
the face toward the mean. Kept as an experimental, off-by-default knob (rm_beta 0 is
bit-exact); do not expect it to move likeness.

Everything here is pure tensor math on (B, 16, 1, H, W) latents; the bank enters in
VAE latent space and is moved into the model's normalised space lazily on the first
call via the comfy BaseModel handed in by the loop (mock loops pass None = as is).
"""
import torch
import torch.nn.functional as F

REF_MEAN_EXPERIMENTAL_BETA = 0.4   # what the node's `ref_mean_guidance` switch enables (tune rm_beta overrides)
REF_MEAN_DEFAULTS = {
    "rm_beta": 0.0,     # 0 = off (bit-exact)
    "rm_lo": 0.20,      # beta = 0 at/below this sigma
    "rm_hi": 0.40,      # full beta at/above this sigma
    "rm_tau": 1.0e5,    # posterior temperature: 1 = exact (= argmax, measured harmful), 1e5 = flat bank mean
    "rm_pass": 1,       # which detailer pass(es) the shift acts in: 1, 2 or 12
}


def beta_at(sigma: float, beta: float, lo: float, hi: float) -> float:
    """Linear ramp: 0 at/below lo, beta at/above hi."""
    if beta <= 0.0:
        return 0.0
    if hi <= lo + 1e-9:
        return float(beta) if sigma >= hi else 0.0
    t = (float(sigma) - lo) / (hi - lo)
    return float(beta) * max(0.0, min(1.0, t))


def posterior_weights(x: torch.Tensor, bank: torch.Tensor, sigma: float, mask=None, tau: float = 1.0):
    """(N,) softmax weights of the bank entries given the noisy state x (1, C, T, H, W).
    bank: (N, C, T, H, W) in the same space as x. mask: broadcastable to x, in [0, 1],
    or None = every element votes."""
    s = float(sigma)
    if s <= 1e-6:
        s = 1e-6
    xf = x.reshape(1, -1).to(torch.float32)
    rf = bank.reshape(bank.shape[0], -1).to(torch.float32)
    if mask is not None:
        m = mask.expand_as(x).reshape(1, -1).to(torch.float32)
        xf = xf * m
        rf_m = rf * m
    else:
        rf_m = rf
    a = 1.0 - s
    dots = (rf_m * xf).sum(dim=1)                 # <x, r_n>_m
    norms = (rf_m * rf).sum(dim=1)                # ||r_n||_m^2
    logits = (a * dots - 0.5 * a * a * norms) / (s * s * max(float(tau), 1e-6))
    return torch.softmax(logits, dim=0)


def ref_mean_shift(x, denoised, bank, sigma, *, beta, lo, hi, tau=1.0, mask=None):
    """denoised + beta(sigma) * m * (mu_hat - denoised); bit-exact `denoised` when
    beta(sigma) == 0."""
    b = beta_at(sigma, beta, lo, hi)
    if b <= 0.0:
        return denoised
    w = posterior_weights(x, bank, sigma, mask=mask, tau=tau)
    mu = (w.view(-1, *([1] * (bank.ndim - 1))).to(bank.dtype) * bank).sum(dim=0, keepdim=True)
    shift = (mu.to(denoised) - denoised) * b
    if mask is not None:
        shift = shift * mask.to(denoised)
    return denoised + shift


class RefMeanHook:
    """The loop's ref_hook: callable(x, denoised, sigma, base_model) -> denoised.
    Holds the bank in VAE latent space and normalises it on first use."""

    def __init__(self, bank_vae, mask_latent, *, beta, lo, hi, tau=1.0):
        if bank_vae.ndim != 5:
            raise ValueError(f"ref_mean: bank must be (N, C, T, H, W), got {tuple(bank_vae.shape)}")
        self.bank_vae = bank_vae
        self.mask = mask_latent
        self.beta, self.lo, self.hi, self.tau = float(beta), float(lo), float(hi), float(tau)
        self._bank = None
        self.calls = 0          # steps where the shift acted (report)
        self.last_weights = None

    def _bank_for(self, x, base_model):
        if self._bank is None:
            b = self.bank_vae.to(device=x.device)
            if base_model is not None and hasattr(base_model, "process_latent_in"):
                b = base_model.process_latent_in(b)
            if b.shape[-2:] != x.shape[-2:]:
                n, c, t = b.shape[:3]
                b = F.interpolate(b.reshape(n, c * t, *b.shape[-2:]), size=x.shape[-2:],
                                  mode="bilinear", align_corners=False).reshape(n, c, t, *x.shape[-2:])
            self._bank = b.to(dtype=x.dtype)
        return self._bank

    def __call__(self, x, denoised, sigma, base_model=None):
        if beta_at(sigma, self.beta, self.lo, self.hi) <= 0.0:
            return denoised
        bank = self._bank_for(x, base_model)
        mask = None if self.mask is None else self.mask.to(device=x.device)
        self.last_weights = posterior_weights(x, bank, sigma, mask=mask, tau=self.tau).detach().cpu()
        self.calls += 1
        return ref_mean_shift(x, denoised, bank, sigma, beta=self.beta, lo=self.lo, hi=self.hi,
                              tau=self.tau, mask=mask)


def build_reference_bank(reference_image, *, detect, crop_box_fn, encode, resize, crop_factor: float,
                         target_hw, align: int, bbox_threshold: float):
    """VAE-encode the largest face of every reference image, cropped with the same
    crop_factor geometry as the detailer's own crop and resized to the crop's pixel
    size target_hw = (h, w). Returns (bank (N, 16, 1, h/8, w/8) or None, n_used, n_skipped)."""
    lats, skipped = [], 0
    th, tw = int(target_hw[0]), int(target_hw[1])
    for b in range(int(reference_image.shape[0])):
        img = reference_image[b:b + 1, ..., :3]
        H, W = int(img.shape[1]), int(img.shape[2])
        boxes, confs = detect(img, threshold=min(0.25, float(bbox_threshold)))
        best = None
        for box, conf in zip(boxes, confs):
            if conf < float(bbox_threshold):
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            if best is None or area > best[0]:
                best = (area, box)
        if best is None:
            skipped += 1
            continue
        x0, y0, x1, y1 = crop_box_fn(best[1], float(crop_factor), (H, W), align)
        crop = img[:, y0:y1, x0:x1, :]
        px = resize(crop, th, tw)
        lats.append(encode(px))
    if not lats:
        return None, 0, skipped
    return torch.cat(lats, dim=0), len(lats), skipped
