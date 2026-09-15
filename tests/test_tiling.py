# -*- coding: utf-8 -*-
"""
Plain-assert test for kreaphoton/tiling.py (v1.5 KreaPhoton Upscale): tile grid,
feather weights, LatentTiler merge (per-step MultiDiffusion-style blending of
the model's x0 prediction), lowpass + LFAnchor, and the upscale presets /
constants in presets.py. Pure tensor math - no ComfyUI, no checkpoint.

Run: <embedded python> tests/test_tiling.py
"""
import importlib
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KREAPHOTON_DIR = os.path.join(REPO_ROOT, "kreaphoton")
COMFYUI_ROOT = r"E:\CUI portable\ComfyUI-torch2.9-cu130-cp313-v1.2\ComfyUI"


def _load_kreaphoton_package():
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)
    if "kreaphoton" not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            "kreaphoton", os.path.join(KREAPHOTON_DIR, "__init__.py"),
            submodule_search_locations=[KREAPHOTON_DIR])
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules["kreaphoton"] = pkg
        pkg_spec.loader.exec_module(pkg)
    return (importlib.import_module("kreaphoton.tiling"),
            importlib.import_module("kreaphoton.presets"),
            importlib.import_module("kreaphoton.schedules"),
            importlib.import_module("kreaphoton.sampling"))


def main():
    import torch

    tl, presets, sch, smp = _load_kreaphoton_package()

    print("=" * 78)
    print("test_tiling: kreaphoton.tiling grid / feather / LatentTiler / LFAnchor + upscale presets")
    print("=" * 78)
    torch.manual_seed(0)

    # --- (1) tile_grid: full coverage, even starts, last tile ends at length ---
    print("[1] tile_grid coverage / alignment")
    g = tl.tile_grid(400, 128, 16)
    assert len(g) == 4, g
    assert g[0] == (0, 128) and g[-1] == (272, 400), g
    assert all((b - a) == 128 for a, b in g), g
    assert all(a % 2 == 0 for a, _ in g), "tile starts must be even (DiT 2x2 patch)"
    for (a0, b0), (a1, b1) in zip(g, g[1:]):
        assert a1 > a0 and (b0 - a1) >= 14, "neighbour overlap must stay >= overlap - 2 (alignment)"
    covered = set()
    for a, b in g:
        covered.update(range(a, b))
    assert covered == set(range(400))
    assert tl.tile_grid(272, 128, 16) == [(0, 128), (72, 200), (144, 272)]
    assert tl.tile_grid(100, 128, 16) == [(0, 100)], "shorter than a tile -> one tile of the full length"
    assert tl.tile_grid(128, 128, 16) == [(0, 128)]
    print("     400/128/16 -> %s ; 272 -> 3 tiles ; 100 -> single" % g)

    # --- (2) feather_weight ---
    print("[2] feather_weight")
    w = tl.feather_weight(128, 128, 16)
    assert w.shape == (128, 128)
    assert float(w.max()) <= 1.0 + 1e-6 and float(w.min()) > 0.0
    assert float(w[64, 64]) == 1.0 and float(w[0, 0]) < 0.05
    assert float(w[64, 0]) == float(w[64, 127]) and float(w[0, 64]) == float(w[127, 64]), "symmetric ramps"
    assert torch.equal(tl.feather_weight(32, 40, 0), torch.ones(32, 40))
    print("     centre 1.0, corner %.4f, symmetric, overlap 0 -> ones" % float(w[0, 0]))

    # --- (3) LatentTiler: pointwise mock -> tiled == untiled; batching; single tile; batch guard ---
    print("[3] LatentTiler merge / batching")
    x = torch.randn(1, 16, 1, 272, 400)
    calls = []

    def mock(xt, sigma, **kw):
        calls.append((tuple(xt.shape), tuple(sigma.shape), kw))
        return 0.5 * xt + 1.0

    tiler = tl.LatentTiler(128, 128, 16, batch=4)
    out = tiler(mock, x, 0.5, {"model_options": {"k": 1}, "seed": 7})
    assert out.shape == x.shape
    assert torch.allclose(out, 0.5 * x + 1.0, atol=1e-6), "pointwise model: tiled result must equal the untiled one"
    assert len(calls) == 3, "12 tiles / batch 4 -> 3 model calls, got %d" % len(calls)
    assert calls[0][0] == (4, 16, 1, 128, 128) and calls[0][1] == (4,)
    assert calls[0][2] == {"model_options": {"k": 1}, "seed": 7}, "extra_args must reach the model"
    calls.clear()
    small = torch.randn(1, 16, 1, 100, 100)
    out_s = tiler(mock, small, 0.3, {})
    assert len(calls) == 1 and calls[0][0] == (1, 16, 1, 100, 100) and calls[0][1] == (1,)
    assert torch.allclose(out_s, 0.5 * small + 1.0, atol=1e-6)
    calls.clear()
    out4 = tiler(mock, torch.randn(1, 16, 272, 400), 0.5, {})
    assert out4.shape == (1, 16, 272, 400) and len(calls) == 3 and calls[0][0] == (4, 16, 128, 128), "4D latents too"
    try:
        tiler(mock, torch.randn(2, 16, 1, 272, 400), 0.5, {})
        raise AssertionError("batch > 1 must be rejected")
    except ValueError as e:
        assert "batch" in str(e)
    print("     12 tiles -> 3 calls of (4,16,1,128,128), sigma (4,), result == untiled; single tile; B>1 rejected")

    # --- (4) lowpass / LFAnchor ---
    print("[4] lowpass + LFAnchor")
    z = torch.randn(1, 16, 1, 64, 80)
    lp = tl.lowpass(z, 4)
    assert lp.shape == z.shape
    # reflect padding is not exactly mean-preserving on a random field (border re-weighting);
    # the exact property is the constant case below - here only "no gross shift"
    assert abs(float(lp.mean()) - float(z.mean())) < 1e-2, "gaussian blur must not shift the mean"
    assert float((lp - z).abs().mean()) < float(z.abs().mean()), "blur removes energy"
    const = torch.full((1, 16, 1, 32, 32), 0.7)
    assert torch.allclose(tl.lowpass(const, 4), const, atol=1e-5), "constant is its own lowpass (reflect padding)"
    assert tl.lowpass(torch.randn(1, 16, 40, 40), 4).shape == (1, 16, 40, 40), "4D too"
    z_ref = torch.randn(1, 16, 1, 64, 80)
    anchor = tl.LFAnchor(z_ref, radius=4, w_max=1.0, sigma_start=0.7, release_sigma=0.3)
    d = torch.randn(1, 16, 1, 64, 80)
    out = anchor(d, 0.7)
    corr = out - d
    assert torch.allclose(corr, tl.lowpass(z_ref, 4) - tl.lowpass(d, 4), atol=1e-6), \
        "w=1: the correction is exactly LF(ref) - LF(d)"
    # the correction lives in the low band: its own high-frequency residue is a few
    # percent of the prediction's (a gaussian is not an idempotent projector, so an
    # exact band-split identity does not hold - this is the honest invariant)
    hf = lambda t: t - tl.lowpass(t, 4)
    ratio = float(hf(corr).norm() / hf(d).norm())
    assert ratio < 0.1, "correction must be low-frequency, HF ratio %.3f" % ratio
    # and it moves the prediction's low band toward the reference
    before = float((tl.lowpass(d, 4) - tl.lowpass(z_ref, 4)).norm())
    after = float((tl.lowpass(out, 4) - tl.lowpass(z_ref, 4)).norm())
    assert after < 0.5 * before, "LF distance to the reference must shrink (%.3f -> %.3f)" % (before, after)
    assert anchor(d, 0.3) is d and anchor(d, 0.1) is d, "released: identity object"
    assert abs(anchor.weight(0.5) - 0.5) < 1e-9, "smoothstep midpoint"
    assert anchor.weight(0.9) == 1.0
    half = tl.LFAnchor(z_ref, radius=4, w_max=0.5, sigma_start=0.7, release_sigma=0.3)
    assert abs(half.weight(0.7) - 0.5) < 1e-9
    assert tl.LFAnchor(z_ref, 4, 0.0, 0.7, 0.3)(d, 0.7) is d, "w_max 0 -> off"
    # dtype/device follow the denoised tensor
    out_h = anchor(d.to(torch.float64), 0.7)
    assert out_h.dtype == torch.float64
    print("     LF follows reference at w=1, HF untouched, release below sigma 0.3, dtype follows input")

    # --- (5) upscale presets + constants ---
    print("[5] presets.UPSCALE_PRESETS / TILE / ANCHOR")
    assert presets.validate_upscale_presets() is True
    assert set(presets.UPSCALE_PRESETS) == {"polish", "detail", "strong"}
    assert presets.DEFAULT_UPSCALE_PRESET == "detail"
    try:
        presets.validate_upscale_presets({"broken": {"denoise": 0.3}})
        raise AssertionError("incomplete upscale preset must be rejected")
    except ValueError as e:
        assert "missing keys" in str(e)
    for name, p in presets.UPSCALE_PRESETS.items():
        assert 0.0 < p["denoise"] <= 0.5, name
        assert p["sampler"] in ("euler", "euler_2m") and p["guidance"] == "window", name
        assert p["detail_a"] == 0.0, "%s: the M2 nudge is a texture push, measured 2026-09-13" % name
        assert p["bp_lock"] >= 1.0, name
    assert presets.UPSCALE_PRESETS["strong"]["denoise"] <= 0.30, "measured 2026-09-13: >= 0.35 rewrites the frame"
    assert presets.UPSCALE_PRESETS["polish"]["denoise"] < presets.UPSCALE_PRESETS["detail"]["denoise"] \
        < presets.UPSCALE_PRESETS["strong"]["denoise"]
    assert all(q["bp_lock"] == 1.0 for q in presets.UPSCALE_PRESETS.values()), "lock 2 measured worse"
    assert presets.FIDELITY["bp_iters"] >= 1
    assert presets.TILE["size"] % 16 == 0 and presets.TILE["overlap"] % 8 == 0 and presets.TILE["batch"] >= 1
    assert presets.ANCHOR["radius"] >= 1 and 0.0 < presets.ANCHOR["release_sigma"] < 0.5
    # start sigma of every preset on a 1024^2 tile is below the composition boundary
    lat = presets.TILE["size"] // 8
    alpha = sch.alpha_for_latent(lat, lat, sch.ALPHA)
    expect = {"polish": 0.14, "detail": 0.25, "strong": 0.45}   # refine_schedule tail index = total - n
    for name, p in presets.UPSCALE_PRESETS.items():
        sig = sch.refine_schedule(p["n_steps"], alpha=alpha, denoise=p["denoise"])
        assert len(sig) == p["n_steps"] + 1 and float(sig[-1]) == 0.0
        assert float(sig[0]) < 0.85, "%s starts at %.3f - composition phase must not exist" % (name, float(sig[0]))
        assert abs(float(sig[0]) - expect[name]) < 0.03, (name, float(sig[0]))
        segs = smp.plan_segments([float(s) for s in sig], has_clean=False,
                                 texture_start=presets.UPSCALE_TEXTURE_START, has_texture=True)
        assert all(ph != "composition" for _, _, ph in segs), segs
        if float(sig[0]) <= presets.UPSCALE_TEXTURE_START:
            assert segs == [(0, p["n_steps"], "identity")], ("no cut possible at index 0 -> the node must "
                                                             "run the texture model directly", segs)
        else:
            assert [ph for _, _, ph in segs] == ["identity", "texture"], segs
    print("     alpha(tile) %.3f; sigma0 polish/detail/strong = %.3f / %.3f / %.3f (< 0.85)" % (
        alpha, *[float(sch.refine_schedule(p["n_steps"], alpha=alpha, denoise=p["denoise"])[0])
                 for p in presets.UPSCALE_PRESETS.values()]))

    # --- (6) v2: shifted grid, activity map, LatentTilerV2 skip + shift ---
    print("[6] v2: tile_grid_shifted / activity_map / LatentTilerV2")
    base = tl.tile_grid(400, 128, 8)
    seen_grids = set()
    for shift in (0, 10, 37, 111, 5000, 65535):
        g = tl.tile_grid_shifted(400, 128, 8, shift)
        assert len(g) == len(base), "the shift must not change the tile count (%d vs %d)" % (len(g), len(base))
        cov = set()
        for a, b in g:
            assert b - a == 128 and a % 2 == 0 and 0 <= a <= 272, g
            cov.update(range(a, b))
        assert cov == set(range(400)), "shift %d must keep full coverage" % shift
        for (a0, b0), (a1, b1) in zip(g, g[1:]):
            assert b0 - a1 >= 8, "neighbour overlap must stay >= overlap under shift %d: %s" % (shift, g)
        assert g[0][0] == 0 and g[-1][1] == 400
        seen_grids.add(tuple(g))
    assert len(seen_grids) >= 3, "different shifts must give different interior positions"
    assert tl.tile_grid_shifted(100, 128, 8, 50) == [(0, 100)]
    assert tl.tile_grid_shifted(400, 128, 8, 0) == base, "shift 0 == the base grid"
    assert tl.tile_grid_shifted(240, 128, 8, 40) == tl.tile_grid(240, 128, 8) and len(tl.tile_grid(240, 128, 8)) == 2, "2 tiles -> nothing to move"
    img = torch.zeros(1, 64, 96, 3)
    img[:, :, 48:, :] = 1.0                               # vertical edge in the right half
    act = tl.activity_map(img, 16)
    assert act.shape == (4, 6)
    # absolute units: the 0->1 step at column 48 puts ONE column of +1 into block 2 and one
    # column of -1 into block 3 (16 of 256 pixels) -> std sqrt(1/16) = 0.25; flat blocks 0
    assert float(act[:, :2].max()) == 0.0 and float(act[:, 4:].max()) == 0.0
    assert abs(float(act[:, 2:4].min()) - 0.25) < 0.01 and abs(float(act[:, 2:4].max()) - 0.25) < 0.01
    x = torch.randn(1, 16, 1, 272, 400)
    calls = []

    def mock(xt, sigma, **kw):
        calls.append(tuple(xt.shape))
        return 0.5 * xt + 1.0
    t2 = tl.LatentTilerV2(128, 128, 8, batch=4, shift_seed=5, shift=True)
    out_a = t2(mock, x, 0.5, {})
    out_b = t2(mock, x, 0.5, {})
    assert torch.allclose(out_a, 0.5 * x + 1.0, atol=1e-5) and torch.allclose(out_b, 0.5 * x + 1.0, atol=1e-5)
    assert t2.step == 2 and t2.calls >= 24 and t2.skipped == 0
    b1 = t2.boxes(272, 400, t2._draw_shift(272, 400))
    b2 = t2.boxes(272, 400, t2._draw_shift(272, 400))
    assert b1 != b2, "the grid must move between draws"
    t3 = tl.LatentTilerV2(128, 128, 8, batch=4, shift=False)
    assert t3.boxes(272, 400, (0, 0)) == tl.LatentTiler(128, 128, 8).boxes(272, 400), "shift off == v1 grid"
    act_lat = torch.ones(272, 400)
    act_lat[:, :200] = 0.0
    fallback = torch.full_like(x, 7.0)
    t4 = tl.LatentTilerV2(128, 128, 8, batch=4, shift=False, activity=act_lat, skip_threshold=0.05, fallback=fallback)
    calls.clear()
    out_c = t4(mock, x, 0.5, {})
    assert t4.skipped > 0 and t4.calls + t4.skipped == len(t4.boxes(272, 400, (0, 0)))
    assert torch.allclose(out_c[..., :, :88], torch.full_like(out_c[..., :, :88], 7.0), atol=1e-5), \
        "a region covered only by empty tiles is exactly the fallback"
    assert torch.allclose(out_c[..., :, 300:], 0.5 * x[..., :, 300:] + 1.0, atol=1e-5), \
        "a region covered only by live tiles is exactly the model output"
    print("     shifted grids cover fully; activity edge 0.25 / flat 0; %d tiles skipped, live/empty regions exact"
          % t4.skipped)

    print("\ntest_tiling: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
