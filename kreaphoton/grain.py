# -*- coding: utf-8 -*-
"""KreaPhoton Film Grain (2026-10-18): the finishing grain / dither step.

Why (vault: KreaPhoton research - retouching and photo practice scan, 2026-10-18): a smooth
8-bit gradient bands, and noise of about half a level rms or more before quantization turns the
staircase into an even gradient (triangular dither of +-1 level = 0.41 rms decorrelates the
quantization error). Retouchers add grain as the last step at the final size, monochrome, and use
it to tie retouched or composited areas together. It does not hide the 20-200 px dark blotches a
VAE can leave: those are low-frequency, grain is not.

Real photos carry little noise at this size: a full-frame base-ISO file downscaled to 1-2 MP
keeps about 0.2 levels, ISO 3200-6400 about 1-2. The default here is 1.0 level at middle grey.

Model: out = clamp(image + amount / 255 * w(Y) * n, 0, 1).
  n     unit-variance Gaussian field; size > 0 blurs it with a Gaussian of that sigma (px) and
        rescales it to unit variance; lag-d autocorrelation sum(k_i k_i+d) / sum(k_i^2) of the sampled
        kernel k, i.e. about exp(-d^2 / (4 size^2)) from size 1 up.
        colour c mixes a shared (luma) field with per-channel fields: corr(R, G) = 1 - c^2.
  w(Y)  response over the image's luma Y (Rec. 709 weights on the sRGB values): "sensor" = shot +
        read noise sigma_lin = sqrt(x + r^2) carried through the sRGB curve (sigma * dY/dx) - more
        visible in the shadows, less in the highlights; "flat" = 1. Both are 1 at middle grey.
Pure torch, deterministic per seed (image b of a batch uses seed + b)."""
import torch

from .ui_meta import CAT_FINISHING

RESPONSES = ("sensor", "flat")
MID_GREY = 0.4613           # sRGB value of 18 % linear grey
READ_NOISE = 0.01           # read noise relative to the shot noise at full scale (sensor curve shape)
_LUMA = (0.2126, 0.7152, 0.0722)


def srgb_to_linear(y):
    return torch.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)


def _sigma_srgb(y):
    x = srgb_to_linear(y.clamp(0.0, 1.0))
    slope = torch.where(x <= 0.0031308, torch.full_like(x, 12.92),
                        1.055 / 2.4 * x.clamp_min(0.0031308) ** (1.0 / 2.4 - 1.0))
    return torch.sqrt(x + READ_NOISE ** 2) * slope


def response_weight(y, response):
    """w(Y): grain strength relative to middle grey."""
    if response == "flat":
        return torch.ones_like(y)
    if response != "sensor":
        raise ValueError("response must be one of %s, got %r" % (list(RESPONSES), response))
    return _sigma_srgb(y) / _sigma_srgb(torch.tensor(MID_GREY, dtype=y.dtype))


def grain_kernel(size):
    """The sampled 1-D Gaussian (sigma `size` px, radius 3 sigma) the grain field is blurred with."""
    radius = max(1, int(round(3.0 * size)))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / size) ** 2)
    return k / k.sum()


def _gaussian_field(shape, size, gen):
    """Unit-variance Gaussian noise of shape (..., H, W), correlated by a Gaussian of sigma `size`."""
    n = torch.randn(shape, generator=gen, dtype=torch.float32)
    if size <= 0.0:
        return n
    k = grain_kernel(size)
    radius = (len(k) - 1) // 2
    flat = n.reshape(-1, 1, *shape[-2:])
    flat = torch.nn.functional.pad(flat, (radius, radius, radius, radius), mode="circular")
    flat = torch.nn.functional.conv2d(flat, k.view(1, 1, 1, -1))
    flat = torch.nn.functional.conv2d(flat, k.view(1, 1, -1, 1))
    # separable kernel k (x) k: the blurred field's variance is (sum k^2)^2
    return flat.reshape(shape) / float((k * k).sum())


def add_grain(image, amount, size=0.0, color=0.0, response="sensor", seed=0):
    """image (B, H, W, C) in [0, 1] -> image with grain on the first three channels."""
    if amount <= 0.0:
        return image
    if response not in RESPONSES:
        raise ValueError("response must be one of %s, got %r" % (list(RESPONSES), response))
    c = float(min(1.0, max(0.0, color)))
    rgb = image[..., :3].float()
    b, h, w = rgb.shape[:3]
    luma = sum(rgb[..., i] * _LUMA[i] for i in range(3))
    weight = response_weight(luma, response)
    out = image.clone()
    for i in range(b):
        gen = torch.Generator(device="cpu").manual_seed((int(seed) + i) & 0xffffffffffffffff)
        shared = _gaussian_field((h, w), size, gen)
        n = shared.unsqueeze(-1).expand(h, w, 3)
        if c > 0.0:
            own = _gaussian_field((3, h, w), size, gen).permute(1, 2, 0)
            n = (1.0 - c * c) ** 0.5 * n + c * own
        grain = (float(amount) / 255.0) * weight[i].unsqueeze(-1) * n.to(rgb.device)
        out[i, ..., :3] = (rgb[i] + grain).clamp(0.0, 1.0).to(image.dtype)
    return out


class KreaPhotonFilmGrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05,
                                     "tooltip": "Grain strength in 8-bit levels (rms) at middle grey. 0.5 or "
                                                "more dithers away banding in smooth gradients; 1-2 looks like a "
                                                "high-ISO photo at this size; 0 passes the image through."}),
                "size": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05,
                                   "tooltip": "Grain size in pixels. 0 = per-pixel noise; 0.5-1 is about what a "
                                              "camera file shows after demosaicing; larger is coarse film grain."}),
                "color": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "0 = neutral grain, the same in red, green and blue (what "
                                               "retouchers use); higher adds colour noise."}),
                "response": (list(RESPONSES), {"default": "sensor",
                                               "tooltip": "sensor: stronger in the shadows and weaker in "
                                                          "the highlights, like camera noise; flat: the same "
                                                          "everywhere."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The image with grain; connect it straight to Save Image.",)
    FUNCTION = "apply"
    CATEGORY = CAT_FINISHING
    DESCRIPTION = ("Adds fine photographic grain as the last step: smooths 8-bit banding in gradients and "
                   "ties the image together. Put it after upscaling, right before saving.")
    SEARCH_ALIASES = ["film grain", "add noise", "dither", "grain", "banding"]

    def apply(self, image, seed, amount, size, color, response):
        return (add_grain(image, float(amount), float(size), float(color), response, int(seed)),)
