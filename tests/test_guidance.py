# -*- coding: utf-8 -*-
"""
Plain-assert test for kreaphoton/guidance.py. Pure math (g_window, exposure
constants from M6) + structural checks (predict_noise is the ONLY override,
cond_scale correctly threaded to comfy.samplers.sampling_function). No live
model/checkpoint needed — that's S3.5(b)'s job (real guided generation).

Run: <embedded python> tests/test_guidance.py
"""
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFYUI_ROOT = r"E:\CUI portable\ComfyUI-torch2.9-cu130-cp313-v1.2\ComfyUI"


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)
    import comfy.samplers

    gd = _load("kreaphoton_guidance", "kreaphoton/guidance.py")

    print("=" * 78)
    print("test_guidance: kreaphoton.guidance g_window + KreaPhotonGuider structure")
    print("=" * 78)

    # --- (1) g_window == 1.0 EXACTLY outside [lo, hi] ---
    print("[1] g_window boundary exactness")
    for sigma in (0.0, 0.3, 0.5, 0.7, 0.7 - 1e-9):
        g = gd.g_window(sigma, delta=1.5, lo=0.7, hi=0.9)
        assert g == 1.0, f"sigma={sigma}: expected exactly 1.0, got {g}"
    for sigma in (0.9, 0.95, 1.0):
        g = gd.g_window(sigma, delta=1.5, lo=0.7, hi=0.9)
        assert abs(g - 2.5) < 1e-12, f"sigma={sigma}: expected 1+delta=2.5, got {g}"
    print("     g(sigma<=lo)=1.0 exactly, g(sigma>=hi)=1+delta exactly")

    # --- (2) monotonic increasing inside the window ---
    xs = [0.7 + i * 0.02 for i in range(11)]
    gs = [gd.g_window(x, delta=1.5) for x in xs]
    assert all(gs[i] <= gs[i + 1] for i in range(len(gs) - 1)), "g_window must be monotonic in [lo,hi]"
    print("[2] monotonic increasing in [0.7, 0.9]: OK")

    # --- (3) exposure constants match M6 proof (docs/03, research/models/m6) ---
    print("[3] exposure candidates (N=12 stock grid, E_safe=0.30, E_broken=0.80)")

    def flux_time_shift(mu, t):
        import math
        return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))

    def sgm_uniform(n, mu=1.15, timesteps=10000):
        smin = flux_time_shift(mu, 1.0 / timesteps)
        ts = [1.0 - i * (1.0 - smin) / n for i in range(n)]
        return [flux_time_shift(mu, t) for t in ts] + [0.0]

    def exposure_discrete(sigmas, delta):
        return sum((gd.g_window(sigmas[i], delta) - 1.0) * (sigmas[i] - sigmas[i + 1])
                    for i in range(len(sigmas) - 1))

    sig12 = sgm_uniform(12)
    e_125 = exposure_discrete(sig12, 1.25)
    e_150 = exposure_discrete(sig12, 1.50)
    print("     Delta=1.25 -> E=%.4f (expected < 0.300)" % e_125)
    print("     Delta=1.50 -> E=%.4f (expected 0.343, i.e. > 0.300 - HONEST, not a bug)" % e_150)
    assert e_125 < 0.300, "Delta=1.25 must be strictly under the safe exposure budget"
    assert abs(e_150 - 0.343) < 0.005, "Delta=1.50 exposure should match the M6-proven honest result"
    assert e_150 < 0.45 * 0.80, "Delta=1.50 must stay well under the breakage budget"

    # --- (4) KreaPhotonGuider overrides ONLY predict_noise ---
    print("[4] KreaPhotonGuider structural check (D10/F15)")
    Guider = gd.KreaPhotonGuider
    assert issubclass(Guider, comfy.samplers.CFGGuider)
    for method in ("inner_sample", "outer_sample", "sample", "set_conds", "set_cfg", "__call__"):
        base = getattr(comfy.samplers.CFGGuider, method)
        derived = getattr(Guider, method)
        assert derived is base, f"{method} must be INHERITED, not overridden"
    assert Guider.predict_noise is not comfy.samplers.CFGGuider.predict_noise
    print("     only predict_noise is overridden; inner_sample/outer_sample/sample/"
          "set_conds/set_cfg/__call__ inherited untouched")

    # --- (5) predict_noise threads cond_scale = g_window(sigma) into sampling_function ---
    print("[5] predict_noise cond_scale threading (mocked sampling_function)")
    captured = {}
    real_sampling_function = comfy.samplers.sampling_function

    def fake_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None):
        captured["cond_scale"] = cond_scale
        captured["timestep"] = timestep
        return x  # anything torch-shaped; predict_noise just returns it

    comfy.samplers.sampling_function = fake_sampling_function
    try:
        g = Guider.__new__(Guider)  # bypass __init__ (no real model_patcher needed for this check)
        g.delta, g.lo, g.hi = 1.25, 0.7, 0.9
        g.conds = {"positive": ["POS"], "negative": ["NEG"]}
        g.inner_model = "MOCK_MODEL"
        import torch
        for sigma_val, expected_scale in ((0.5, 1.0), (0.8, gd.g_window(0.8, 1.25)), (1.0, 1.0 + 1.25)):
            g.predict_noise(torch.zeros(1), torch.tensor([sigma_val]), model_options={}, seed=0)
            # float32 round-trip through the tensor -> ~1e-7 tolerance, not float64 exactness
            assert abs(captured["cond_scale"] - expected_scale) < 1e-5, \
                f"sigma={sigma_val}: expected cond_scale={expected_scale}, got {captured['cond_scale']}"
        print("     cond_scale correctly computed from g_window(sigma) for sigma=0.5/0.8/1.0")
    finally:
        comfy.samplers.sampling_function = real_sampling_function

    print("\ntest_guidance: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
