# -*- coding: utf-8 -*-
"""
Plain-assert test for kreaphoton/trajectory.py (Sampler v2 iteration 2): the pure math
on the model's x0-prediction trajectory - terminal x0(sigma=0) extrapolation (linear /
quadratic), the local trust region, and the band-split motion metrics that drive the
adaptive predictor-corrector and self-refine.

Run: <embedded python> tests/test_trajectory.py
"""
import importlib
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KREAPHOTON_DIR = os.path.join(REPO_ROOT, "kreaphoton")
COMFYUI_ROOT = r"E:\CUI portable\ComfyUI-torch2.9-cu130-cp313-v1.2\ComfyUI"


def _load():
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)
    if "kreaphoton" not in sys.modules:
        spec = importlib.util.spec_from_file_location("kreaphoton", os.path.join(KREAPHOTON_DIR, "__init__.py"),
                                                      submodule_search_locations=[KREAPHOTON_DIR])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["kreaphoton"] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module("kreaphoton.trajectory")


def main():
    import torch

    tr = _load()
    print("=" * 78)
    print("test_trajectory: kreaphoton.trajectory (x0 extrapolation, trust region, motion)")
    print("=" * 78)
    torch.manual_seed(0)
    shape = (1, 16, 1, 12, 10)

    # --- (1) linear x0(0): the v1 formula (factor sigma_last/(sigma_prev - sigma_last), capped) ---
    print("[1] linear extrapolation == the v1 terminal formula")
    a, b = torch.randn(shape), torch.randn(shape)
    d = tr.x0_zero_delta([(a, 0.4), (b, 0.2)], order=1, max_factor=2.0)
    assert torch.equal(d, 1.0 * (b - a)), "0.2/(0.4-0.2) = 1"
    w = 0.37
    assert torch.equal(tr.x0_zero_delta([(a, 0.43), (b, 0.21)], weight=w), float(w) * (0.21 / (0.43 - 0.21)) * (b - a)), \
        "same float operation order as the v1 loop (bit-exact default path)"
    d = tr.x0_zero_delta([(a, 0.3), (b, 0.25)], order=1, max_factor=2.0)
    assert torch.allclose(d, 2.0 * (b - a)), "factor 5 capped at 2"
    assert tr.x0_zero_delta([(b, 0.2)], order=1) is None, "one point: nothing to extrapolate"
    assert tr.x0_zero_delta([(a, 0.2), (b, 0.2)], order=1) is None, "degenerate spacing"
    print("     factor 1 on a halving grid, capped at 2, None on < 2 points / zero spacing")

    # --- (2) quadratic x0(0) is EXACT for an x0 trajectory quadratic in sigma ---
    print("[2] quadratic Lagrange extrapolation exact for x0(s) = A + B s + C s^2")
    A, B, C = torch.randn(shape), torch.randn(shape), torch.randn(shape)
    x0 = lambda s: A + B * s + C * s * s
    for sig in ((0.4875, 0.325, 0.1625), (0.6, 0.35, 0.2), (0.47, 0.30, 0.13)):
        hist = [(x0(s), s) for s in sig]
        d = tr.x0_zero_delta(hist, order=2)
        assert torch.allclose(hist[-1][0] + d, A, atol=1e-4), sig
    lin = tr.x0_zero_delta([(x0(0.325), 0.325), (x0(0.1625), 0.1625)], order=1)
    assert not torch.allclose(x0(0.1625) + lin, A, atol=1e-3), "linear is not exact on a curved trajectory"
    two = tr.x0_zero_delta([(x0(0.325), 0.325), (x0(0.1625), 0.1625)], order=2)
    assert torch.equal(two, lin), "order 2 with two points falls back to linear"
    wild = tr.x0_zero_delta([(A, 0.30), (B, 0.2999), (C, 0.1)], order=2)
    assert torch.equal(wild, tr.x0_zero_delta([(B, 0.2999), (C, 0.1)], order=1)), \
        "ill-conditioned weights (near-duplicate sigmas) fall back to linear"
    print("     exact to 1e-4 on three grids; linear is not; 2 points / ill-conditioned -> linear")

    # --- (3) trust region: pointwise |delta| <= k * local rms of the reference step; k<=0 = off ---
    print("[3] trust_clamp: local spike limiter")
    ref = torch.randn(shape) * 0.1
    delta = ref * 1.5
    delta[0, 3, 0, 6, 5] = 50.0                                   # one extrapolation spike
    assert torch.equal(tr.trust_clamp(delta, ref, 0.0), delta), "k = 0 -> unchanged"
    cl = tr.trust_clamp(delta, ref, 2.0)
    lim = 2.0 * tr.local_rms(ref)
    assert bool((cl.abs() <= lim + 1e-7).all())
    assert float(cl[0, 3, 0, 6, 5]) < 1.0, "the spike is limited"
    keep = (delta.abs() <= lim)
    assert torch.equal(cl[keep], delta[keep]), "inside the region nothing changes"
    c = torch.ones(shape) * 3.0
    assert torch.allclose(tr.local_rms(c), c), "local rms of a constant is the constant (edge windows shrink)"
    print("     spike 50 -> %.3f, in-region values untouched, k=0 bit-exact" % float(cl[0, 3, 0, 6, 5]))

    # --- (4) band motion: relative change per band, invariant to scale, zero for no change ---
    print("[4] motion(): relative LF / MF / HF change of the x0 estimate")
    # an image-like reference: energy concentrated in low frequencies (a white-noise reference
    # would make every band's RELATIVE change equal - the metric is per band on purpose)
    smooth = lambda h, w, a: a * torch.nn.functional.interpolate(torch.randn(16, 1, h, w), size=(64, 48),
                                                                  mode="bilinear").reshape(1, 16, 1, 64, 48)
    z = smooth(16, 12, 1.0) + 0.1 * torch.randn(1, 16, 1, 64, 48)
    m0 = tr.motion(z, z)
    assert all(abs(v) < 1e-9 for k, v in m0.items() if k != "cos") and abs(m0["cos"] - 1.0) < 1e-6
    lf_bump = z + smooth(4, 3, 0.5)
    hf_bump = z + 0.1 * torch.randn_like(z)
    ml, mh = tr.motion(lf_bump, z), tr.motion(hf_bump, z)
    assert ml["lf"] > ml["hf"] and mh["hf"] > mh["lf"], (ml, mh)
    assert abs(tr.motion(3 * lf_bump, 3 * z)["mf"] - ml["mf"]) < 1e-5, "scale invariant"
    two = torch.cat([z, z], 0)
    moved = torch.cat([z, hf_bump], 0)
    assert abs(tr.motion(moved, two)["hf"] - mh["hf"]) < 1e-5, "batch: the most-moved sample decides"
    print("     no change -> 0 / cos 1; LF bump -> lf %.3f > hf %.3f; HF bump -> hf %.3f; batch max"
          % (ml["lf"], ml["hf"], mh["hf"]))

    print("\ntest_trajectory: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
