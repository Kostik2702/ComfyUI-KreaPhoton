# -*- coding: utf-8 -*-
"""Plain-assert test for kreaphoton/grain.py (KreaPhoton Film Grain, 2026-10-18): the finishing
grain / dither step. Every property is checked against its analytic value, and the dither test
shows the purpose: on a dark 8-bit gradient the grain removes the quantization staircase that the
same image has without it.

Run: <embedded python> tests/test_grain.py
"""
import importlib
import importlib.util
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KREAPHOTON_DIR = os.path.join(REPO_ROOT, "kreaphoton")
COMFYUI_ROOT = r"E:\CUI portable\ComfyUI-torch2.9-cu130-cp313-v1.2\ComfyUI"


def _load():
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)
    if "kreaphoton" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "kreaphoton", os.path.join(KREAPHOTON_DIR, "__init__.py"), submodule_search_locations=[KREAPHOTON_DIR])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["kreaphoton"] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module("kreaphoton.grain"), importlib.import_module("kreaphoton.nodes")


def main():
    import torch
    G, N = _load()
    print("=" * 78)
    print("test_grain: finishing grain / dither (KreaPhoton Film Grain)")
    print("=" * 78)

    def flat(y, h=256, w=256, b=1, c=3):
        return torch.full((b, h, w, c), float(y))

    def rms_levels(out, img):
        return float(((out - img)[..., :3] * 255.0).double().pow(2).mean().sqrt())

    mid = G.MID_GREY

    print("[1] amount 0 returns the image untouched")
    img = torch.rand(1, 32, 48, 3)
    assert G.add_grain(img, 0.0, 1.0, 0.5, "sensor", 3) is img
    print("     identity")

    print("[2] amount = rms in 8-bit levels at middle grey, for white and correlated grain")
    for size in (0.0, 0.6, 1.5):
        for resp in ("flat", "sensor"):
            r = rms_levels(G.add_grain(flat(mid), 1.2, size, 0.0, resp, 7), flat(mid))
            assert abs(r - 1.2) / 1.2 < 0.03, (size, resp, r)
    print("     1.2 levels requested -> within 3 % at sizes 0 / 0.6 / 1.5, flat and sensor")

    print("[3] response: sensor follows the sRGB-encoded shot + read noise curve, flat is constant")
    ys = (0.03, mid, 0.9)
    want = [G.response_weight(torch.tensor(y), "sensor").item() for y in ys]
    got = [rms_levels(G.add_grain(flat(y), 2.0, 0.0, 0.0, "sensor", 11), flat(y)) / 2.0 for y in ys]
    for wv, gv, y in zip(want, got, ys):
        assert abs(gv - wv) / wv < 0.04, (y, wv, gv)
    assert got[0] > got[1] > got[2], got
    flat_got = [rms_levels(G.add_grain(flat(y), 2.0, 0.0, 0.0, "flat", 11), flat(y)) / 2.0 for y in ys]
    assert all(abs(v - 1.0) < 0.04 for v in flat_got), flat_got
    # the curve itself: sigma_srgb = sqrt(x + r^2) * dY/dx, normalised at middle grey
    x = G.srgb_to_linear(torch.tensor(0.9))
    slope = 1.055 / 2.4 * x ** (1 / 2.4 - 1)
    ref = math.sqrt(float(x) + G.READ_NOISE ** 2) * float(slope)
    xm = G.srgb_to_linear(torch.tensor(mid))
    refm = math.sqrt(float(xm) + G.READ_NOISE ** 2) * float(1.055 / 2.4 * xm ** (1 / 2.4 - 1))
    assert abs(want[2] - ref / refm) < 1e-4, (want[2], ref / refm)
    print("     sensor: shadows %.2f, mid 1.00, highlights %.2f (measured %.2f / %.2f / %.2f); flat 1.00"
          % (want[0], want[2], *got))

    print("[4] colour: 0 = the same grain in R, G, B; c = channel correlation 1 - c^2")
    base = flat(mid, 192, 192)
    o = G.add_grain(base, 1.0, 0.0, 0.0, "flat", 5)
    assert torch.equal(o[..., 0], o[..., 1]) and torch.equal(o[..., 1], o[..., 2]), "colour 0 must be neutral"
    for c in (0.5, 1.0):
        d = (G.add_grain(base, 1.0, 0.0, c, "flat", 5) - base)[0]
        rg = torch.corrcoef(torch.stack([d[..., 0].flatten(), d[..., 1].flatten()]))[0, 1].item()
        assert abs(rg - (1 - c * c)) < 0.03, (c, rg)
    print("     colour 0 neutral; correlation 0.75 at 0.5, 0 at 1.0")

    print("[5] size: Gaussian-correlated grain; lag-1 correlation = the sampled kernel's, ~exp(-1/(4 size^2))")
    big = flat(mid, 384, 384)

    def lag1(size):
        d = (G.add_grain(big, 1.0, size, 0.0, "flat", 9) - big)[0, ..., 0].double()
        a, b = d[:, :-1].flatten(), d[:, 1:].flatten()
        return float(torch.corrcoef(torch.stack([a, b]))[0, 1])

    assert abs(lag1(0.0)) < 0.03, lag1(0.0)
    for s in (0.6, 1.0, 1.5):
        k = G.grain_kernel(s).double()
        exact = float((k[:-1] * k[1:]).sum() / (k * k).sum())
        assert abs(lag1(s) - exact) < 0.03, (s, lag1(s), exact)
        if s >= 1.0:
            assert abs(exact - math.exp(-1.0 / (4 * s * s))) < 0.02, (s, exact)
    print("     size 0 white; 0.6 / 1.0 / 1.5 match the kernel's lag-1 correlation (continuous form from 1 px)")

    print("[6] seeds: deterministic, per image in a batch")
    a = G.add_grain(base, 1.0, 0.6, 0.2, "sensor", 42)
    assert torch.equal(a, G.add_grain(base, 1.0, 0.6, 0.2, "sensor", 42))
    assert not torch.equal(a, G.add_grain(base, 1.0, 0.6, 0.2, "sensor", 43))
    pair = G.add_grain(base.repeat(2, 1, 1, 1), 1.0, 0.6, 0.2, "sensor", 42)
    assert torch.equal(pair[0:1], a) and not torch.equal(pair[0], pair[1]), "image b uses seed + b"
    print("     same seed same grain; batch image b gets seed + b")

    print("[7] the purpose: dither removes the 8-bit staircase on a dark gradient")
    h, w = 256, 1024
    truth = torch.linspace(0.02, 0.06, w).view(1, 1, w, 1).expand(1, h, w, 3).contiguous()
    q = lambda t: torch.round(t * 255.0) / 255.0                    # what the PNG writer does

    def col_error(t):
        # systematic error of the 8-bit result: means over 256 rows x 8 columns (one level of the
        # gradient spans ~100 columns, so the staircase survives; the grain's own noise averages to
        # ~0.02 levels)
        err = (q(t)[0, :, :, 0].double() - truth[0, :, :, 0].double()) * 255
        return float(err.reshape(h, w // 8, 8).mean(dim=(0, 2)).abs().max())

    plain_err = col_error(truth)
    grain_err = col_error(G.add_grain(truth, 1.0, 0.0, 0.0, "flat", 1))
    assert plain_err > 0.45, plain_err                   # the staircase: up to half a level off
    assert grain_err < 0.12, grain_err                   # the column average follows the gradient
    print("     column-mean error %.2f levels without grain, %.2f with 1.0 level of grain" % (plain_err, grain_err))

    print("[8] range and alpha")
    white = G.add_grain(flat(1.0, c=4), 4.0, 0.6, 0.5, "flat", 2)
    assert float(white.max()) <= 1.0 and float(white.min()) >= 0.0
    assert torch.equal(white[..., 3], flat(1.0, c=4)[..., 3]), "alpha is not grain"
    half = flat(0.5).to(torch.float16)
    assert G.add_grain(half, 1.0, 0.6, 0.0, "sensor", 2).dtype == torch.float16
    print("     clamped to [0, 1]; alpha untouched; dtype kept")

    print("[9] the node")
    node = N.NODE_CLASS_MAPPINGS["KreaPhotonFilmGrain"]
    spec = node.INPUT_TYPES()["required"]
    assert list(spec) == ["image", "seed", "amount", "size", "color", "response"], list(spec)
    (out,) = node().apply(base, 42, 1.0, 0.6, 0.2, "sensor")
    assert torch.equal(out, a)
    assert spec["response"][0] == list(G.RESPONSES)
    print("     KreaPhotonFilmGrain wraps add_grain")

    print("\ntest_grain: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
