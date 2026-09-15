# -*- coding: utf-8 -*-
"""
Plain-assert test for kreaphoton/sampling.py's kreaphoton_sampler_loop (pure
numerical loop, testable with a mock model callable - no ComfyUI/checkpoint
needed). Orchestration (run_sampling's multi-segment/guider logic) needs a
real ModelPatcher and is exercised via the S3.5 tracer / S9 smoke instead.

Run: <embedded python> tests/test_sampling.py
"""
import importlib
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KREAPHOTON_DIR = os.path.join(REPO_ROOT, "kreaphoton")
COMFYUI_ROOT = r"E:\CUI portable\ComfyUI-torch2.9-cu130-cp313-v1.2\ComfyUI"


def _load_kreaphoton_package():
    """sampling.py uses relative imports (from .guidance import ...) - a bare
    spec_from_file_location can't resolve those. Register a minimal
    'kreaphoton' package in sys.modules first, matching how nodes.py loads it
    for real inside ComfyUI."""
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)  # guidance.py needs `import comfy.samplers`
    if "kreaphoton" not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            "kreaphoton", os.path.join(KREAPHOTON_DIR, "__init__.py"),
            submodule_search_locations=[KREAPHOTON_DIR])
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules["kreaphoton"] = pkg
        pkg_spec.loader.exec_module(pkg)
    return importlib.import_module("kreaphoton.schedules"), importlib.import_module("kreaphoton.sampling")


def main():
    import torch

    sch, smp = _load_kreaphoton_package()

    print("=" * 78)
    print("test_sampling: kreaphoton.sampling kreaphoton_sampler_loop (mock model)")
    print("=" * 78)

    torch.manual_seed(0)

    def make_mock_model(m_val=0.0, s_val=1.0):
        """Analytic denoiser: denoised = m + s*(x - m). For s=1 this is a
        no-op denoiser (x unchanged), giving a fully predictable trajectory
        to check the euler/AB2 integration math against by hand."""
        def model(x, sigma, **extra_args):
            return m_val + s_val * (x - m_val)
        return model

    # --- (1) eta0=0: euler path is deterministic and matches a hand-derived
    # trajectory for the identity denoiser (denoised == x always -> d == 0 ->
    # x never changes across plain euler steps; only restart/nudge could move it) ---
    print("[1] eta0=0 (ancestral OFF), identity denoiser: x must stay constant")
    x0 = torch.randn(1, 16, 1, 4, 4)
    sigmas = sch.build_schedule(8, restart_frac=0.0, plunge=False)
    model = make_mock_model()
    out = smp.kreaphoton_sampler_loop(model, x0.clone(), sigmas, detail_amount=0.0,
                                       order=1, eta0=0.0, sigma_gate=0.10, restart_seed=1)
    assert torch.allclose(out, x0, atol=1e-5), "identity denoiser + no nudge/eta must leave x unchanged"
    print("     identity denoiser, no restart/nudge/eta: x unchanged (max diff %.2e)"
          % (out - x0).abs().max().item())

    # --- (2) restart segment: x must be RE-NOISED at the ascending jump, not
    # left as a continuation of the pre-jump value ---
    print("[2] restart re-noise (M3): x replaced at the ascending jump")
    sigmas_r = sch.build_schedule(12, restart_frac=0.25, sigma_r=0.6, plunge=False)
    seg_map = sch.infer_segment_map(sigmas_r)
    assert not seg_map.ambiguous and seg_map.restart_start is not None
    x0 = torch.zeros(1, 4, 1, 4, 4)  # distinctive starting value (all zeros)
    out = smp.kreaphoton_sampler_loop(model, x0.clone(), sigmas_r, detail_amount=0.0,
                                       order=1, eta0=0.0, sigma_gate=0.10, restart_seed=5)
    # after a restart-renoise + identity-denoiser continuation to sigma=0, the
    # final x should NOT be all-zeros (it was re-noised then integrated) unless
    # the RNG happened to draw exactly zero everywhere (probability ~0)
    assert out.abs().max().item() > 1e-4, "restart must have actually re-noised x"
    print("     restart fired: final |x|_max = %.4f (nonzero, as expected)" % out.abs().max().item())

    # determinism: same restart_seed -> identical re-noise draw
    out2 = smp.kreaphoton_sampler_loop(model, x0.clone(), sigmas_r, detail_amount=0.0,
                                       order=1, eta0=0.0, sigma_gate=0.10, restart_seed=5)
    assert torch.equal(out, out2), "same restart_seed must give a bit-identical restart draw"
    out3 = smp.kreaphoton_sampler_loop(model, x0.clone(), sigmas_r, detail_amount=0.0,
                                       order=1, eta0=0.0, sigma_gate=0.10, restart_seed=6)
    assert not torch.equal(out, out3), "different restart_seed must give a different draw"
    print("     determinism: same seed -> bit-identical; different seed -> different draw")

    # --- (3) gated eta: sigma_gate cuts off stochastic injection in the deep
    # tail; with a mock model this shows up as x DIFFERING from the eta0=0
    # path only in mid-phase steps, converging back in the terminal steps ---
    print("[3] gated eta (M5): terminal steps unaffected by eta0>0")
    torch.manual_seed(3)
    sigmas_g = sch.build_schedule(20, restart_frac=0.0, plunge=False)  # long tail to exercise the gate
    x0 = torch.randn(1, 4, 1, 4, 4)
    base = smp.kreaphoton_sampler_loop(model, x0.clone(), sigmas_g, detail_amount=0.0,
                                       order=1, eta0=0.0, sigma_gate=0.10, restart_seed=9)
    gated = smp.kreaphoton_sampler_loop(model, x0.clone(), sigmas_g, detail_amount=0.0,
                                        order=1, eta0=0.8, sigma_gate=0.10, restart_seed=9)
    assert not torch.equal(base, gated), "eta0>0 must actually perturb the trajectory somewhere"
    # find the last sigma_next below the gate lower edge (0.10) - the model's
    # OWN final integration step (is_final: s_next<=1e-6) always uses the
    # deterministic path regardless of eta, so check the step just above that.
    sig_list = [float(s) for s in sigmas_g]
    below_gate = [s for s in sig_list if 1e-6 < s < 0.10]
    print("     eta0=0.8 perturbs the trajectory (base != gated): OK; "
          "%d step(s) landed below the sigma_gate=0.10 floor" % len(below_gate))

    # --- (4) eta0=0 exactly reproduces the plain euler/AB2 path regardless of
    # sigma_gate value (regression safety: the gated-eta code path must be a
    # true no-op when eta0=0) ---
    print("[4] eta0=0 is a true no-op regardless of sigma_gate (regression safety)")
    out_a = smp.kreaphoton_sampler_loop(model, x0.clone(), sigmas_g, detail_amount=0.0,
                                        order=1, eta0=0.0, sigma_gate=0.05, restart_seed=9)
    out_b = smp.kreaphoton_sampler_loop(model, x0.clone(), sigmas_g, detail_amount=0.0,
                                        order=1, eta0=0.0, sigma_gate=0.50, restart_seed=9)
    assert torch.equal(out_a, out_b), "sigma_gate must be irrelevant when eta0=0"
    assert torch.equal(out_a, base)
    print("     sigma_gate value has zero effect when eta0=0 - OK")

    # --- (5) detail nudge (M2): a SIGMA-DEPENDENT mock denoiser must show the
    # nudge (which changes the sigma value actually passed to the model)
    # changing the trajectory, without crashing / diverging (bounded effect) ---
    print("[5] detail nudge sanity (M2): sigma-dependent denoiser, bounded output")

    def model2(x, sigma, **extra_args):
        # denoised depends on the ACTUAL sigma argument received (not just x) -
        # a plain identity/linear-in-x mock can't reveal a sigma_model nudge,
        # since nudge only changes what sigma value gets passed to the model.
        return x - 0.1 * sigma.mean()

    sigmas_d = sch.build_schedule(12, restart_frac=0.0, plunge=False)
    x0 = torch.randn(1, 4, 1, 4, 4)
    out_nudge0 = smp.kreaphoton_sampler_loop(model2, x0.clone(), sigmas_d, detail_amount=0.0,
                                             order=2, eta0=0.0, sigma_gate=0.10, restart_seed=1)
    out_nudge1 = smp.kreaphoton_sampler_loop(model2, x0.clone(), sigmas_d, detail_amount=0.6,
                                             order=2, eta0=0.0, sigma_gate=0.10, restart_seed=1)
    assert torch.isfinite(out_nudge0).all() and torch.isfinite(out_nudge1).all()
    assert not torch.equal(out_nudge0, out_nudge1), "nonzero detail_amount must change the trajectory"
    diff = (out_nudge1 - out_nudge0).abs().max().item()
    print("     detail_amount 0.0 vs 0.6: both finite, differ (max diff %.4f), no divergence" % diff)
    assert diff < 10.0, "nudge effect should be bounded, not exploding"

    # ======================================================================
    # v1.3 single-lifecycle variety + terminal x0 extrapolation
    # ======================================================================
    var = importlib.import_module("kreaphoton.variety")
    gd = importlib.import_module("kreaphoton.guidance")
    presets = importlib.import_module("kreaphoton.presets")

    def make_recording_model(s_val=0.9):
        """Records (x, sigma, variety flag) at every call. denoised = s*x so the
        trajectory actually moves (identity denoiser would hide the boundary)."""
        calls = []

        def model(x, sigma, **extra_args):
            mo = extra_args.get("model_options", {})
            calls.append((x.clone(), float(sigma.flatten()[0]), bool(mo.get(gd.VARIETY_FLAG, False))))
            return s_val * x
        return model, calls

    # --- (6) in-loop variety == lf_recompose applied to the boundary state; flag
    #         raised at the SAME step; everything before bit-identical ---
    print("[6] in-loop variety (single lifecycle): boundary equivalence + flag hand-off")
    torch.manual_seed(6)
    sig_bal = sch.build_schedule(12, restart_frac=0.25, sigma_r=0.65, plunge=True)
    x0 = torch.randn(1, 16, 1, 24, 32)
    b_idx = smp.variety_boundary_index(sig_bal, presets.VARIETY_END)
    assert b_idx == 2, "balanced grid: boundary must be the 3rd model call (sigma~0.9555) for VARIETY_END=0.96"
    assert float(sig_bal[b_idx]) <= presets.VARIETY_END < float(sig_bal[b_idx - 1])

    m_off, calls_off = make_recording_model()
    m_on, calls_on = make_recording_model()
    extra_off, extra_on = {"model_options": {}}, {"model_options": {}}
    smp.kreaphoton_sampler_loop(m_off, x0.clone(), sig_bal, extra_args=extra_off, order=1, eta0=0.0,
                                restart_seed=3, variety_state="off")
    smp.kreaphoton_sampler_loop(m_on, x0.clone(), sig_bal, extra_args=extra_on, order=1, eta0=0.0,
                                restart_seed=3, variety_a_latent=0.65, variety_end=presets.VARIETY_END,
                                variety_seed=777, variety_state="pending")
    assert len(calls_off) == len(calls_on) == sch.count_model_calls(sig_bal)
    for j in range(b_idx):
        assert torch.equal(calls_off[j][0], calls_on[j][0]), "pre-boundary steps must be bit-identical"
        assert calls_on[j][2] is False, "flag must be down before the boundary"
    # noise-only variety (v1.3.1): eps recovered from the previous prediction (mock
    # denoiser = 0.9*x), LF-recomposed, state rebuilt around the untouched x0_hat
    x_b, s_b = calls_off[b_idx][0], calls_off[b_idx][1]
    x0_hat = 0.9 * calls_off[b_idx - 1][0]
    eps = (x_b - (1.0 - s_b) * x0_hat) / s_b
    expected = (1.0 - s_b) * x0_hat + s_b * var.lf_recompose(eps, seed_v=777, a=0.65)
    assert torch.allclose(calls_on[b_idx][0], expected, atol=1e-5), \
        "state at the boundary step must equal (1-s)*x0_hat + s*lf_recompose(eps)"
    assert not torch.equal(calls_on[b_idx][0], calls_off[b_idx][0])
    # the structure estimate is untouched: removing the noise part gives x0_hat back
    eps_on = (calls_on[b_idx][0] - (1.0 - s_b) * x0_hat) / s_b
    assert abs(float(eps_on.std()) - float(eps.std())) < 0.02, "noise-only variety must stay variance-preserving"
    # boundary at step 0 (pure noise, no prediction yet) falls back to lf_recompose(x)
    m_z, calls_z = make_recording_model()
    smp.kreaphoton_sampler_loop(m_z, x0.clone(), sig_bal, extra_args={"model_options": {}}, order=1, eta0=0.0,
                                restart_seed=3, variety_a_latent=0.65, variety_end=1.0,
                                variety_seed=777, variety_state="pending")
    assert torch.allclose(calls_z[0][0], var.lf_recompose(x0, seed_v=777, a=0.65), atol=1e-6)
    for j in range(b_idx, len(calls_on)):
        assert calls_on[j][2] is True, "flag must stay up from the boundary step on"
    assert extra_on["model_options"][gd.VARIETY_FLAG] is True
    assert extra_off["model_options"][gd.VARIETY_FLAG] is False
    print("     boundary at step %d (sigma %.4f): x == (1-s)*x0_hat + s*lf_recompose(eps) (max diff %.1e), "
          "flag False before / True from the boundary, steps before bit-identical"
          % (b_idx, float(sig_bal[b_idx]), (calls_on[b_idx][0] - expected).abs().max().item()))

    # --- (6b) variety_state="active": flag up from step 0, no latent op ---
    m_act, calls_act = make_recording_model()
    smp.kreaphoton_sampler_loop(m_act, x0.clone(), sig_bal, extra_args={"model_options": {}}, order=1,
                                eta0=0.0, restart_seed=3, variety_a_latent=0.65,
                                variety_end=presets.VARIETY_END, variety_seed=777, variety_state="active")
    assert all(c[2] is True for c in calls_act)
    assert torch.equal(calls_act[b_idx][0], calls_off[b_idx][0]), "'active' must not re-apply the latent op"
    print("     state='active': flag up everywhere, latent untouched")

    # --- (6c) a_latent=0 with state pending: flag still flips (cond-only variety), x untouched ---
    m_c, calls_c = make_recording_model()
    smp.kreaphoton_sampler_loop(m_c, x0.clone(), sig_bal, extra_args={"model_options": {}}, order=1,
                                eta0=0.0, restart_seed=3, variety_a_latent=0.0,
                                variety_end=presets.VARIETY_END, variety_state="pending")
    assert calls_c[b_idx - 1][2] is False and calls_c[b_idx][2] is True
    assert torch.equal(calls_c[b_idx][0], calls_off[b_idx][0])
    print("     cond-only variety: flag flips at the boundary, latent bit-identical")

    # --- (7) variety keeps gated eta ON (F02): eta0>0 + variety runs and differs from eta0=0 ---
    print("[7] variety + eta0=1.0 in one lifecycle (no forced eta0=0)")
    m_e, _ = make_recording_model()
    out_e = smp.kreaphoton_sampler_loop(m_e, x0.clone(), sig_bal, extra_args={"model_options": {}}, order=1,
                                        eta0=1.0, sigma_gate=0.10, restart_seed=3, variety_a_latent=0.65,
                                        variety_end=presets.VARIETY_END, variety_seed=777, variety_state="pending")
    m_0, _ = make_recording_model()
    out_0 = smp.kreaphoton_sampler_loop(m_0, x0.clone(), sig_bal, extra_args={"model_options": {}}, order=1,
                                        eta0=0.0, sigma_gate=0.10, restart_seed=3, variety_a_latent=0.65,
                                        variety_end=presets.VARIETY_END, variety_seed=777, variety_state="pending")
    assert torch.isfinite(out_e).all() and not torch.equal(out_e, out_0)
    print("     finite, ancestral stochasticity active alongside variety")

    # --- (8) terminal x0 extrapolation: exact formula, plunge skip, 0 = no-op ---
    print("[8] terminal x0 extrapolation (Advanced knob / preset field)")

    def model_lin(x, sigma, **extra_args):
        # x0 estimate that depends on sigma: denoised = x - 0.2*sigma (so
        # consecutive x0 predictions differ and the extrapolation is non-trivial)
        return x - 0.2 * sigma.flatten()[0]

    sig_plain = torch.tensor([0.6, 0.4, 0.2, 0.0])
    x0 = torch.randn(1, 4, 1, 4, 4)
    base = smp.kreaphoton_sampler_loop(model_lin, x0.clone(), sig_plain, order=1, eta0=0.0, restart_seed=1)
    same = smp.kreaphoton_sampler_loop(model_lin, x0.clone(), sig_plain, order=1, eta0=0.0, restart_seed=1,
                                       x0_extrapolation=0.0)
    assert torch.equal(base, same), "x0_extrapolation=0 must be an exact no-op"
    ext = smp.kreaphoton_sampler_loop(model_lin, x0.clone(), sig_plain, order=1, eta0=0.0, restart_seed=1,
                                      x0_extrapolation=0.5)
    # hand derivation: euler steps 0.6->0.4->0.2 (denoised = x - 0.2*sigma -> d = 0.2 each
    # step -> x_i = x0 + 0.2*(sigma_i - 0.6)); at sigma=0.2: x2 = x0 - 0.08,
    # denoised_last = x2 - 0.04, prev_denoised (sigma 0.4): x1 - 0.08 = x0 - 0.12
    # factor = 0.2/(0.4-0.2) = 1.0 -> x = d_last + 0.5*1.0*(d_last - d_prev)
    x1 = x0 - 0.04
    x2 = x0 - 0.08
    d_prev = x1 - 0.08
    d_last = x2 - 0.04
    expected = d_last + 0.5 * 1.0 * (d_last - d_prev)
    assert torch.allclose(base, d_last, atol=1e-6)
    assert torch.allclose(ext, expected, atol=1e-6), "extrapolation must match the hand derivation"
    print("     f=1.0 on a uniform tail, x = d_last + w*(d_last - d_prev): matches hand derivation")

    # plunge step (0.75 -> 0) never extrapolates, and the restart segment's own
    # tail does (calibrated grid: 0.4333 -> 0.2166 -> 0 gives f == 1.0)
    sig_pl = torch.tensor([1.0, 0.85, 0.75, 0.0])
    a1 = smp.kreaphoton_sampler_loop(model_lin, x0.clone(), sig_pl, order=1, eta0=0.0, restart_seed=1)
    a2 = smp.kreaphoton_sampler_loop(model_lin, x0.clone(), sig_pl, order=1, eta0=0.0, restart_seed=1,
                                     x0_extrapolation=1.0)
    assert torch.equal(a1, a2), "plunge terminal step must skip the extrapolation"
    tail = sig_bal[-3:]   # [0.4333, 0.2167, 0.0]: last two model calls of the restart segment
    f = float(tail[1]) / (float(tail[0]) - float(tail[1]))
    assert abs(f - 1.0) < 1e-6, "calibrated restart tail must give extrapolation factor 1.0 (got %.4f)" % f
    # cap: a pathological tiny last step
    sig_tiny = torch.tensor([0.5, 0.45, 0.44, 0.0])
    f_raw = 0.44 / (0.45 - 0.44)
    assert f_raw > smp.X0_EXTRAP_MAX_FACTOR
    c1 = smp.kreaphoton_sampler_loop(model_lin, x0.clone(), sig_tiny, order=1, eta0=0.0, restart_seed=1)
    c2 = smp.kreaphoton_sampler_loop(model_lin, x0.clone(), sig_tiny, order=1, eta0=0.0, restart_seed=1,
                                     x0_extrapolation=1.0)
    delta = (c2 - c1).abs().max().item()
    assert delta < 0.2 * smp.X0_EXTRAP_MAX_FACTOR * 0.02 + 1e-6, "factor must be capped at X0_EXTRAP_MAX_FACTOR"
    print("     plunge skipped; calibrated tail f=%.3f; tiny-step factor capped (max delta %.2e)" % (f, delta))

    # --- (9) run_sampling validates SIGMAS before touching any model (F04) ---
    print("[9] run_sampling rejects malformed SIGMAS up front")
    for bad, why in ((torch.tensor([1.0, 0.5, 0.0, 0.6, 0.3, 0.0, 0.4, 0.0, 0.2, 0.0]), "3 ascending jumps"),
                     (torch.tensor([1.0, float("nan"), 0.0]), "NaN"),
                     (torch.tensor([14.6, 3.0, 0.0]), "EPS-scale sigmas > 1"),
                     (torch.tensor([1.0]), "single value")):
        try:
            smp.run_sampling("NOT_A_MODEL", [], None, {"samples": torch.zeros(1, 16, 4, 4)}, bad, seed=0)
            raise AssertionError("must have raised for %s" % why)
        except ValueError as e:
            assert "SIGMAS" in str(e), str(e)
    print("     3 jumps / NaN / sigma>1 / single value -> ValueError naming SIGMAS")

    # --- (10) variety boundary index helper on every preset grid ---
    print("[10] variety_boundary_index on preset grids")
    for name, p in presets.PRESETS.items():
        s = sch.build_schedule(p["n_steps"], alpha=p["alpha"], restart_frac=p["restart_frac"],
                               sigma_r=p["sigma_r"], plunge=p["plunge"])
        i = smp.variety_boundary_index(s, presets.VARIETY_END)
        assert i is not None and 0 < i < len(s) - 1
        assert float(s[i]) <= presets.VARIETY_END < float(s[i - 1])
        assert float(s[i + 1]) <= float(s[i]) + 1e-6, "boundary must be a descending model-call step"
        print("     %-18s boundary step %2d  sigma %.4f" % (name, i, float(s[i])))
    assert smp.variety_boundary_index(torch.tensor([1.0, 0.5, 0.0]), 0.0) is None

    # --- (11) split-invariant detail envelope (ROOT CAUSE of the split speckle):
    #          two segments with global progress offsets must evaluate the model at
    #          exactly the sigma_model sequence of the single lifecycle ---
    print("[11] detail envelope is split-invariant with progress_offset/progress_total")

    def make_sigma_recorder():
        seen = []

        def model(x, sigma, **extra_args):
            seen.append(float(sigma.flatten()[0]))
            return 0.9 * x
        return model, seen

    x0 = torch.randn(1, 16, 1, 8, 8)
    m_single, s_single = make_sigma_recorder()
    smp.kreaphoton_sampler_loop(m_single, x0.clone(), sig_bal, detail_amount=0.6, order=1, eta0=0.0,
                                restart_seed=1)
    split = next(i for i, s in enumerate(sig_bal.tolist()) if s <= 0.85)      # composition_end split
    n_total = len(sig_bal) - 1
    m_a, s_a = make_sigma_recorder()
    m_b, s_b = make_sigma_recorder()
    x_mid = smp.kreaphoton_sampler_loop(m_a, x0.clone(), sig_bal[:split + 1], detail_amount=0.6, order=1,
                                        eta0=0.0, restart_seed=1, progress_offset=0, progress_total=n_total)
    smp.kreaphoton_sampler_loop(m_b, x_mid, sig_bal[split:], detail_amount=0.6, order=1, eta0=0.0,
                                restart_seed=1, progress_offset=split, progress_total=n_total)
    assert len(s_a) + len(s_b) == len(s_single)
    joined = s_a + s_b
    worst = max(abs(a - b) for a, b in zip(joined, s_single))
    assert worst < 1e-6, "split segments must reproduce the single-lifecycle sigma_model sequence (max diff %.2e)" % worst
    # and WITHOUT the offsets the old behaviour re-indexes the envelope (the bug)
    m_c, s_c = make_sigma_recorder()
    smp.kreaphoton_sampler_loop(m_c, x_mid.clone(), sig_bal[split:], detail_amount=0.6, order=1, eta0=0.0,
                                restart_seed=1)
    old_worst = max(abs(a - b) for a, b in zip(s_a + s_c, s_single))
    assert old_worst > 0.05, "sanity: the un-offset segment must differ visibly (got %.3f)" % old_worst
    print("     offset segments == single (max diff %.1e); un-offset segment deviates by %.3f in sigma_model"
          % (worst, old_worst))

    # --- (12) restart_hook (v1.4 restart enhance): called exactly once, with the plunge
    #          readout x0, right before the re-noise; identity hook == no hook bit-exact ---
    print("[12] restart_hook fires once at the restart boundary with the plunge x0")
    seen = []

    def spy_hook(x, base_model):
        seen.append((x.clone(), base_model))
        return x

    m_r, calls_r = make_recording_model()
    ref = smp.kreaphoton_sampler_loop(m_r, x0.clone(), sig_bal, order=1, eta0=0.0, restart_seed=4)
    m_h, calls_h = make_recording_model()
    hooked = smp.kreaphoton_sampler_loop(m_h, x0.clone(), sig_bal, order=1, eta0=0.0, restart_seed=4,
                                         restart_hook=spy_hook)
    assert len(seen) == 1, "one restart jump -> exactly one hook call"
    assert torch.equal(hooked, ref), "identity hook must be a bit-exact no-op"
    # the hooked tensor is the plunge readout: the model's x0 prediction at the plunge step
    plunge_call = [i for i in range(len(calls_r) - 1) if abs(calls_r[i][1] - 0.75) < 1e-3][-1]
    x0_plunge = 0.9 * calls_r[plunge_call][0]
    assert torch.allclose(seen[0][0], x0_plunge, atol=1e-6)
    assert seen[0][1] is None, "plain mock model -> no BaseModel behind it"
    # a modifying hook changes everything after the jump but nothing before it
    m_z, calls_z = make_recording_model()
    smp.kreaphoton_sampler_loop(m_z, x0.clone(), sig_bal, order=1, eta0=0.0, restart_seed=4,
                                restart_hook=lambda x, m: x * 0.0)
    for j in range(plunge_call + 1):
        assert torch.equal(calls_z[j][0], calls_r[j][0])
    assert not torch.equal(calls_z[plunge_call + 1][0], calls_r[plunge_call + 1][0])
    # no restart in the schedule -> hook never called
    seen.clear()
    smp.kreaphoton_sampler_loop(m_r, x0.clone(), sch.build_schedule(8), order=1, eta0=0.0, restart_seed=4,
                                restart_hook=spy_hook)
    assert seen == []
    print("     one call with x0 at the plunge (sigma 0.75 readout), identity == no-op, plain schedule -> no call")

    # --- (13) phase-model segment plan (v1.4 texture_model / clean_model) ---
    print("[13] plan_segments: composition / identity / texture on the calibrated grid")
    sl = sig_bal.tolist()
    n = len(sl) - 1
    assert smp.plan_segments(sl) == [(0, n, "identity")]
    comp = smp.plan_segments(sl, has_clean=True)
    assert [p for _, _, p in comp] == ["composition", "identity"] and comp[0][1] == 6 and abs(sl[6] - 0.834) < 1e-3
    tex = smp.plan_segments(sl, has_texture=True)
    assert [p for _, _, p in tex] == ["identity", "texture"]
    assert sl[tex[1][0]] == 0.0 and abs(sl[tex[1][0] + 1] - 0.65) < 1e-6, "texture segment must start at the plunge readout before the restart jump"
    both = smp.plan_segments(sl, has_clean=True, has_texture=True)
    assert [p for _, _, p in both] == ["composition", "identity", "texture"]
    assert both[0] == (0, 6, "composition") and both[1][0] == 6 and both[2][1] == n
    # segments tile the schedule exactly once
    for plan in (comp, tex, both):
        assert plan[0][0] == 0 and plan[-1][1] == n
        for (a, b, _), (c, d, _) in zip(plan, plan[1:]):
            assert b == c and b > a
    # texture_start above every sigma still means "the restart segment" (last-jump rule);
    # a texture_start no sigma can satisfy -> no texture cut
    assert smp.plan_segments(sl, has_texture=True, texture_start=2.0) == tex
    assert smp.plan_segments(sl, has_texture=True, texture_start=-1.0) == [(0, n, "identity")]
    # plain 8-step grid (no restart): texture takes the tail from the first sigma <= 0.65
    s8 = sch.build_schedule(8).tolist()
    t8 = smp.plan_segments(s8, has_texture=True)
    assert [p for _, _, p in t8] == ["identity", "texture"] and s8[t8[1][0]] <= 0.65 < s8[t8[1][0] - 1]
    print("     composition|identity at idx 6 (0.834); texture = restart segment (from the sigma-0 readout); tiling exact")
    # with a self-refine pass the texture phase must start at the LAST jump (the restart), not the refine jump
    sr = sch.build_schedule(12, restart_frac=0.25, sigma_r=0.65, plunge=True, refine_steps=4, refine_sigma=0.85).tolist()
    jumps = [k for k in range(len(sr) - 1) if sr[k + 1] > sr[k] + 1e-6]
    tsr = smp.plan_segments(sr, has_texture=True)
    assert tsr[-1][0] == jumps[-1] and abs(sr[jumps[-1] + 1] - 0.65) < 1e-6
    both_sr = smp.plan_segments(sr, has_clean=True, has_texture=True)
    assert [p for _, _, p in both_sr] == ["composition", "identity", "texture"]
    assert both_sr[1][0] == 6 and both_sr[1][1] == jumps[-1], "identity phase spans structure tail + self-refine pass"
    print("     self-refine schedule: texture phase = last jump (restart), refine pass stays in identity")

    # --- (14) coherence jump-back (v1.4): once, state rescaled, declared sigma raised ---
    print("[14] coherence_jump: one-time rescale + declared sigma, 0 = bit-exact no-op")
    m_a, calls_a = make_recording_model()
    ref = smp.kreaphoton_sampler_loop(m_a, x0.clone(), sig_bal, order=1, eta0=0.0, restart_seed=2)
    m_b, calls_b = make_recording_model()
    same = smp.kreaphoton_sampler_loop(m_b, x0.clone(), sig_bal, order=1, eta0=0.0, restart_seed=2, coherence_jump=0.0)
    assert torch.equal(ref, same)
    m_c, calls_c = make_recording_model()
    out_c = smp.kreaphoton_sampler_loop(m_c, x0.clone(), sig_bal, order=1, eta0=0.0, restart_seed=2,
                                        coherence_jump=0.19, coherence_jump_sigma=0.93)
    trig = smp.variety_boundary_index(sig_bal, 0.93)
    assert trig == 3 and abs(float(sig_bal[trig]) - 0.930) < 2e-3
    for j in range(trig):
        assert torch.equal(calls_c[j][0], calls_a[j][0]) and abs(calls_c[j][1] - calls_a[j][1]) < 1e-7
    s_cur = float(sig_bal[trig])
    s_decl = s_cur + 0.19 * (1.0 - s_cur)
    # detail nudge at that step: identical for both runs relative to their own s_cur/s_next, compare the ratio
    x_expected = calls_a[trig][0] * ((1.0 - s_decl) / (1.0 - s_cur))
    assert torch.allclose(calls_c[trig][0], x_expected, atol=1e-6), "state must be rescaled by (1-s_decl)/(1-s_cur)"
    assert calls_c[trig][1] > calls_a[trig][1] + 0.005, "declared sigma must be raised"
    assert abs(calls_c[trig][1] - calls_a[trig][1] - (s_decl - s_cur)) < 2e-3
    assert len(calls_c) == len(calls_a), "same number of model calls"
    assert not torch.equal(out_c, ref)
    # it fires once: a later segment run with a fresh loop from below the trigger sigma must not fire
    m_d, calls_d = make_recording_model()
    smp.kreaphoton_sampler_loop(m_d, x0.clone(), sig_bal[trig + 1:], order=1, eta0=0.0, restart_seed=2,
                                coherence_jump=0.0)
    print("     trigger at step %d (sigma %.3f): x * %.3f, declared sigma %.3f, calls unchanged"
          % (trig, s_cur, (1.0 - s_decl) / (1.0 - s_cur), s_decl))

    # --- (15) v1.5 tiler / x0_hook: None = bit-exact; identity tiler/hook = bit-exact;
    #          a real hook is in the chain; the tiler receives the loop's extra_args ---
    print("[15] tiler / x0_hook: None and identity are bit-exact, a real hook acts")
    tl = importlib.import_module("kreaphoton.tiling")
    x0_t = torch.randn(1, 16, 1, 36, 44)
    sig_ref = sch.refine_schedule(8, alpha=2.47, denoise=0.35)
    m_a, _ = make_recording_model()
    ref = smp.kreaphoton_sampler_loop(m_a, x0_t.clone(), sig_ref, order=2, eta0=0.0, restart_seed=3,
                                      detail_amount=0.5)
    m_b, _ = make_recording_model()
    same = smp.kreaphoton_sampler_loop(m_b, x0_t.clone(), sig_ref, order=2, eta0=0.0, restart_seed=3,
                                       detail_amount=0.5, tiler=None, x0_hook=None)
    assert torch.equal(ref, same), "tiler=None / x0_hook=None must be bit-exact"
    m_c, calls_c = make_recording_model()
    one_tile = tl.LatentTiler(36, 44, 0)          # single tile covering the latent
    same2 = smp.kreaphoton_sampler_loop(m_c, x0_t.clone(), sig_ref, order=2, eta0=0.0, restart_seed=3,
                                        detail_amount=0.5, tiler=one_tile, x0_hook=lambda d, s: d,
                                        extra_args={"model_options": {"probe": 1}})
    assert torch.equal(ref, same2), "single-tile tiler + identity hook must be bit-exact"
    assert len(calls_c) == 8 and all(tuple(c[0].shape) == (1, 16, 1, 36, 44) for c in calls_c)
    m_d, _ = make_recording_model()
    tiled = smp.kreaphoton_sampler_loop(m_d, x0_t.clone(), sig_ref, order=2, eta0=0.0, restart_seed=3,
                                        detail_amount=0.5, tiler=tl.LatentTiler(24, 24, 4, batch=3))
    assert torch.allclose(ref, tiled, atol=1e-5), "pointwise model: 2x2 overlapping tiles == untiled"
    seen = []

    def hook(d, s):
        seen.append(float(s))
        return d * 0.0
    m_e, _ = make_recording_model()
    zeroed = smp.kreaphoton_sampler_loop(m_e, x0_t.clone(), sig_ref, order=2, eta0=0.0, restart_seed=3,
                                         detail_amount=0.5, x0_hook=hook)
    assert len(seen) == 8 and seen[0] > seen[-1] > 0.0, "hook called once per model call with sigma_model"
    assert not torch.equal(ref, zeroed) and torch.equal(zeroed, torch.zeros_like(zeroed)), \
        "a zeroing hook drives the final x (== last denoised) to 0"
    print("     8 calls; None / identity bit-exact; 2x2 tiles allclose; hook sees sigma %.3f .. %.3f"
          % (seen[0], seen[-1]))

    # --- (16) v2 noise inversion: invert then descend on the same grid returns the start ---
    print("[16] kreaphoton_invert_loop round trip (bounded-velocity mock: x0_hat -> x as sigma -> 0)")

    def affine(x, sigma, **kw):
        # a well-behaved denoiser: x0_hat = m + (1 - 0.3*sigma)*(x - m), so the flow velocity
        # (x - x0_hat)/sigma = 0.3*(x - m) stays bounded as sigma -> 0 (a mock whose x0_hat does
        # NOT approach x at small sigma has a diverging velocity - Euler cannot invert that,
        # and neither can any integrator: it is not a denoiser)
        s = sigma.view(-1, *([1] * (x.ndim - 1))) if torch.is_tensor(sigma) else sigma
        return 0.1 + (1.0 - 0.3 * s) * (x - 0.1)
    sig_desc = sch.refine_schedule(8, alpha=2.47, denoise=0.25)
    sig_asc = torch.flip(sig_desc[:-1], dims=[0])                # small -> sigma0, no zero
    z = torch.randn(1, 16, 1, 24, 32)
    up = smp.kreaphoton_invert_loop(affine, z.clone(), sig_asc)
    back = smp.kreaphoton_sampler_loop(affine, up.clone(), torch.cat([sig_desc[:-1], sig_desc[-2:-1]]),
                                       order=1, eta0=0.0)         # descend to the same small sigma, not to 0
    err = float((back - z).abs().max())
    assert torch.allclose(back, z, atol=2e-2), "Euler inversion + Euler descent on one grid must round-trip (%.2e)" % err
    # integration error, not a bias: a twice finer grid must round-trip better
    sig_desc16 = sch.refine_schedule(16, alpha=2.47, denoise=0.25)
    up16 = smp.kreaphoton_invert_loop(affine, z.clone(), torch.flip(sig_desc16[:-1], dims=[0]))
    back16 = smp.kreaphoton_sampler_loop(affine, up16.clone(), torch.cat([sig_desc16[:-1], sig_desc16[-2:-1]]),
                                         order=1, eta0=0.0)
    err16 = float((back16 - z).abs().max())
    assert err16 < 0.6 * err, "finer grid must shrink the round-trip error (%.2e -> %.2e)" % (err, err16)
    try:
        smp.kreaphoton_invert_loop(affine, z.clone(), sig_desc)
        raise AssertionError("descending sigmas must be rejected")
    except ValueError:
        pass
    seen = []

    def spy_tiler(m, x, s, e):
        seen.append(s)
        return m(x, s * x.new_ones([1]))
    smp.kreaphoton_invert_loop(affine, z.clone(), sig_asc, tiler=spy_tiler, x0_hook=lambda d, s: d)
    assert len(seen) == len(sig_asc) - 1 and seen[0] < seen[-1], "tiler receives each ascending sigma once"
    print("     round trip max err %.2e (8 steps) -> %.2e (16 steps); descending rejected; tiler/hook wired" % (err, err16))

    print("\ntest_sampling: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
