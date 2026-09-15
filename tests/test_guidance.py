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

    # --- (3b) degenerate windows (audit F07): no division by zero, exact 1.0 below ---
    print("[3b] degenerate windows hi==lo / hi<lo")
    assert gd.g_window(0.69, 1.25, lo=0.7, hi=0.7) == 1.0
    assert gd.g_window(0.70, 1.25, lo=0.7, hi=0.7) == 2.25
    assert gd.g_window(0.80, 1.25, lo=0.9, hi=0.7) == gd.g_window(0.80, 1.25, lo=0.7, hi=0.9)
    print("     hi==lo -> hard step at lo; hi<lo -> swapped")

    # --- (4) KreaPhotonGuider overrides ONLY predict_noise + set_conds ---
    print("[4] KreaPhotonGuider structural check (D10/F15)")
    Guider = gd.KreaPhotonGuider
    assert issubclass(Guider, comfy.samplers.CFGGuider)
    for method in ("inner_sample", "outer_sample", "sample", "set_cfg", "__call__", "outer_predict_noise"):
        base = getattr(comfy.samplers.CFGGuider, method)
        derived = getattr(Guider, method)
        assert derived is base, f"{method} must be INHERITED, not overridden"
    assert Guider.predict_noise is not comfy.samplers.CFGGuider.predict_noise
    assert Guider.set_conds is not comfy.samplers.CFGGuider.set_conds
    print("     only predict_noise + set_conds overridden; inner_sample/outer_sample/sample/"
          "set_cfg/__call__/outer_predict_noise inherited untouched")

    # --- (5) predict_noise threads cond_scale per mode into sampling_function,
    #         and switches to the variety positive when the loop's flag is up ---
    print("[5] predict_noise cond_scale threading + variety cond switch (mocked sampling_function)")
    captured = {}
    real_sampling_function = comfy.samplers.sampling_function

    def fake_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None):
        captured["cond_scale"] = cond_scale
        captured["timestep"] = timestep
        captured["cond"] = cond
        captured["uncond"] = uncond
        return x  # anything torch-shaped; predict_noise just returns it

    comfy.samplers.sampling_function = fake_sampling_function
    try:
        import torch
        g = Guider.__new__(Guider)  # bypass __init__ (no real model_patcher needed for this check)
        g.mode, g.cfg, g.delta, g.lo, g.hi, g.rescale = "window", 1.0, 1.25, 0.7, 0.9, 0.0
        g.conds = {"positive": ["POS"], "negative": ["NEG"], gd.VARIETY_COND_KEY: ["POS_VAR"]}
        g.inner_model = "MOCK_MODEL"
        for sigma_val, expected_scale in ((0.5, 1.0), (0.8, gd.g_window(0.8, 1.25)), (1.0, 1.0 + 1.25)):
            g.predict_noise(torch.zeros(1), torch.tensor([sigma_val]), model_options={}, seed=0)
            # float32 round-trip through the tensor -> ~1e-7 tolerance, not float64 exactness
            assert abs(captured["cond_scale"] - expected_scale) < 1e-5, \
                f"sigma={sigma_val}: expected cond_scale={expected_scale}, got {captured['cond_scale']}"
            assert captured["cond"] == ["POS"] and captured["uncond"] == ["NEG"]
        print("     window: cond_scale from g_window(sigma) for sigma=0.5/0.8/1.0, positive used")

        g.mode = "flat"; g.cfg = 3.5
        g.predict_noise(torch.zeros(1), torch.tensor([0.3]), model_options={}, seed=0)
        assert captured["cond_scale"] == 3.5
        g.mode = "off"
        g.predict_noise(torch.zeros(1), torch.tensor([0.95]), model_options={}, seed=0)
        assert captured["cond_scale"] == 1.0
        print("     flat: constant cfg; off: exactly 1.0 (stock cfg1 optimization path)")

        # variety switch: flag False -> positive; flag True -> positive_variety; flag
        # True without a variety cond -> positive (flag is inert)
        g.predict_noise(torch.zeros(1), torch.tensor([0.9]), model_options={gd.VARIETY_FLAG: False}, seed=0)
        assert captured["cond"] == ["POS"]
        g.predict_noise(torch.zeros(1), torch.tensor([0.9]), model_options={gd.VARIETY_FLAG: True}, seed=0)
        assert captured["cond"] == ["POS_VAR"]
        del g.conds[gd.VARIETY_COND_KEY]
        g.predict_noise(torch.zeros(1), torch.tensor([0.9]), model_options={gd.VARIETY_FLAG: True}, seed=0)
        assert captured["cond"] == ["POS"]
        print("     variety flag selects positive_variety only when present")

        # CFG-rescale: registered as a post-cfg hook ONLY on guided steps (cond_scale>1)
        captured_mo = {}

        def fake_sf2(model, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None):
            captured_mo["mo"] = model_options
            captured_mo["cond_scale"] = cond_scale
            return x

        comfy.samplers.sampling_function = fake_sf2
        g.mode, g.rescale = "window", 0.7
        base_mo = {"sampler_post_cfg_function": ["EXISTING"]}
        g.predict_noise(torch.zeros(1), torch.tensor([0.5]), model_options=base_mo, seed=0)   # g == 1.0
        assert captured_mo["mo"] is base_mo and captured_mo["mo"]["sampler_post_cfg_function"] == ["EXISTING"]
        g.predict_noise(torch.zeros(1), torch.tensor([0.9]), model_options=base_mo, seed=0)   # g == 2.25
        hooks = captured_mo["mo"]["sampler_post_cfg_function"]
        assert hooks[0] == "EXISTING" and hooks[-1] == g._rescale_post_cfg and len(hooks) == 2
        assert base_mo["sampler_post_cfg_function"] == ["EXISTING"], "caller's model_options must not be mutated"
        g.rescale = 0.0
        g.predict_noise(torch.zeros(1), torch.tensor([0.9]), model_options=base_mo, seed=0)
        assert captured_mo["mo"] is base_mo
        # the hook itself: phi=1 rescales the guided std to the cond std exactly, phi=0 is identity
        g.rescale = 1.0
        cond = torch.randn(2, 16, 1, 8, 8)
        cfg_res = cond * 3.0 + 0.5
        out = g._rescale_post_cfg({"denoised": cfg_res, "cond_denoised": cond})
        for b in range(2):
            assert abs(float(out[b].std()) - float(cond[b].std())) < 1e-5
        g.rescale = 0.0
        assert torch.equal(g._rescale_post_cfg({"denoised": cfg_res, "cond_denoised": cond}), cfg_res)
        print("     CFG-rescale hook only on guided steps, appended after existing hooks, std law exact")
    finally:
        comfy.samplers.sampling_function = real_sampling_function

    # --- (6) set_conds registers the variety key only when given ---
    print("[6] set_conds variety key")

    class _FakePatcher:
        model_options = {}

        def is_dynamic(self):
            return False

    g = Guider.__new__(Guider)
    g.model_patcher = _FakePatcher()
    g.original_conds = {}
    import torch
    pos = [[torch.zeros(1, 4, 8), {}]]
    neg = [[torch.zeros(1, 4, 8), {}]]
    g.set_conds(pos, neg)
    assert set(g.original_conds.keys()) == {"positive", "negative"}
    g.set_conds(pos, neg, positive_variety=pos)
    assert set(g.original_conds.keys()) == {"positive", "negative", gd.VARIETY_COND_KEY}
    print("     positive/negative only by default; positive_variety added on demand")

    # --- (7) invalid mode rejected at construction ---
    try:
        Guider.__init__(Guider.__new__(Guider), _FakePatcher(), mode="banana")
        raise AssertionError("invalid guidance mode must raise")
    except ValueError as e:
        assert "guidance_mode" in str(e)
    print("[7] invalid mode rejected with a naming ValueError")

    # --- (8) perturbed-attention guidance plumbing (v1.4) ---
    print("[8] PAG: block spec, patch cloning, identity-attention output, window gating")
    assert gd.parse_blocks("8-15") == frozenset(range(8, 16))
    assert gd.parse_blocks("6, 9,12") == frozenset({6, 9, 12})
    assert gd.parse_blocks("15-8,20") == frozenset(set(range(8, 16)) | {20})
    assert gd.parse_blocks([1, 2]) == frozenset({1, 2}) and gd.parse_blocks("") == frozenset()
    base = {"transformer_options": {"patches": {"attn1_patch": ["X"]}, "other": 1}, "k": 2}
    mo = gd.pag_model_options(base, frozenset({3}))
    assert base["transformer_options"]["patches"] == {"attn1_patch": ["X"]}, "caller's options must not be mutated"
    assert mo["transformer_options"]["patches"]["attn1_patch"][0] == "X"
    assert mo["transformer_options"]["patches"]["attn1_patch"][-1] is gd._pag_stash_v
    assert len(mo["transformer_options"]["patches"]["attn1_output_patch"]) == 1
    assert mo["transformer_options"]["other"] == 1 and mo["k"] == 2
    # identity attention: the stash keeps v in the shared extra_options, the output patch
    # returns v expanded over the GQA heads in (B, L, H*D) layout - only in selected blocks
    B, Hkv, L, D, H = 1, 2, 5, 4, 4
    v = torch.randn(B, Hkv, L, D)
    extra = {"block_index": 3}
    assert gd._pag_stash_v(None, None, v, extra_options=extra) == {}
    out_in = torch.zeros(B, L, H * D)
    out = gd._pag_identity_output(frozenset({3}))(out_in, extra)
    expected = v.repeat_interleave(H // Hkv, dim=1).transpose(1, 2).reshape(B, L, H * D)
    assert torch.equal(out, expected) and gd.PAG_STASH_KEY not in extra
    extra2 = {"block_index": 4, gd.PAG_STASH_KEY: v}
    assert gd._pag_identity_output(frozenset({3}))(out_in, extra2) is out_in, "other blocks untouched"
    # guider gating: post-cfg hook registered only inside [pag_lo, pag_hi] and only with scale>0
    comfy.samplers.sampling_function = fake_sf2
    try:
        g = Guider.__new__(Guider)
        g.mode, g.cfg, g.delta, g.lo, g.hi, g.rescale = "off", 1.0, 1.25, 0.7, 0.9, 0.0
        g.pag_scale, g.pag_lo, g.pag_hi, g.pag_blocks = 1.0, 0.72, 0.93, frozenset(range(8, 16))
        g.conds = {"positive": ["POS"], "negative": ["NEG"]}
        g.inner_model = "MOCK_MODEL"
        base_mo = {}
        g.predict_noise(torch.zeros(1), torch.tensor([0.5]), model_options=base_mo, seed=0)
        assert captured_mo["mo"] is base_mo
        g.predict_noise(torch.zeros(1), torch.tensor([0.85]), model_options=base_mo, seed=0)
        assert captured_mo["mo"]["sampler_post_cfg_function"] == [g._pag_post_cfg]
        assert "sampler_post_cfg_function" not in base_mo
        g.pag_scale = 0.0
        g.predict_noise(torch.zeros(1), torch.tensor([0.85]), model_options=base_mo, seed=0)
        assert captured_mo["mo"] is base_mo
        # the hook math: denoised + scale*(cond - perturbed) with a fake calc_cond_batch
        real_ccb = comfy.samplers.calc_cond_batch
        comfy.samplers.calc_cond_batch = lambda model, conds, x, t, mo: (x * 0.5,)
        try:
            g.pag_scale = 2.0
            xin = torch.ones(1, 4)
            res = g._pag_post_cfg({"cond_denoised": xin * 3.0, "cond": ["POS"], "input": xin, "sigma": torch.tensor([0.85]),
                                   "model": None, "model_options": {}, "denoised": xin * 3.0})
            assert torch.allclose(res, xin * (3.0 + 2.0 * (3.0 - 0.5)))
        finally:
            comfy.samplers.calc_cond_batch = real_ccb
    finally:
        comfy.samplers.sampling_function = real_sampling_function
    print("     blocks parsed, options cloned not mutated, identity output == v over heads, window gating, hook math")

    print("\ntest_guidance: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
