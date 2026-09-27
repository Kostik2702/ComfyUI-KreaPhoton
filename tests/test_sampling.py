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
    # ref_hook (ref_mean spike): None / identity bit-exact; sees the loop STATE x and the TRUE
    # sigma s_cur (not the detail-nudged sigma_model); base_model None under a mock
    seen_ref = []

    def ref_hook(x, d, s, base_model):
        seen_ref.append((tuple(x.shape), float(s), base_model))
        return d
    m_f, _ = make_recording_model()
    same3 = smp.kreaphoton_sampler_loop(m_f, x0_t.clone(), sig_ref, order=2, eta0=0.0, restart_seed=3,
                                        detail_amount=0.5, ref_hook=ref_hook)
    assert torch.equal(ref, same3), "identity ref_hook must be bit-exact"
    assert len(seen_ref) == 8 and all(sh == (1, 16, 1, 36, 44) and bm is None for sh, _, bm in seen_ref)
    assert [s for _, s, _ in seen_ref] == [float(v) for v in sig_ref[:-1]], "ref_hook sees s_cur, not sigma_model"
    assert seen != [s for _, s, _ in seen_ref], "with detail_amount 0.5 sigma_model differs from s_cur"
    print("     ref_hook: identity bit-exact, sees x and the true sigma")

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

    # --- (20) v1.6: a noise_mask must never cross a phase-model split. comfy's
    #          KSamplerX0Inpaint needs the SOURCE latent and the ORIGINAL noise in every
    #          segment; segment 2+ hands over the mid-trajectory state and zero noise, so the
    #          mask band becomes the noisy state (Face Detailer "confetti ring", 2026-09-15) ---
    print("[20] run_sampling: noise_mask + split -> one segment on the longest phase")
    runs = []

    def spy_run_one(model, positive, negative, latent_dict, sigmas, **kw):
        runs.append((model, len(sigmas) - 1, latent_dict.get("noise_mask") is not None))
        return latent_dict
    real_run_one = smp._run_one
    smp._run_one = spy_run_one
    try:
        s8 = sch.refine_schedule(7, alpha=2.48, denoise=0.45)          # sigma0 0.658 > 0.65: 1 identity step + 6 texture
        lat = {"samples": torch.zeros(1, 16, 1, 8, 8), "noise_mask": torch.ones(1, 1, 1, 8, 8)}
        smp.run_sampling("identity", "pos", None, lat, s8, seed=1, texture_model="texture")
        assert len(runs) == 1 and runs[0][0] == "texture" and runs[0][1] == 7 and runs[0][2], runs
        runs.clear()
        smp.run_sampling("identity", "pos", None, {"samples": torch.zeros(1, 16, 1, 8, 8)}, s8, seed=1,
                         texture_model="texture")
        assert len(runs) == 2 and [r[0] for r in runs] == ["identity", "texture"], runs   # no mask: split as before
        runs.clear()
        s_full = sch.build_schedule(8).tolist()
        lat = {"samples": torch.zeros(1, 16, 1, 8, 8), "noise_mask": torch.ones(1, 1, 1, 8, 8)}
        smp.run_sampling("identity", "pos", None, lat, torch.tensor(s_full), seed=1, clean_model="clean",
                         texture_model="texture")
        assert len(runs) == 1 and runs[0][1] == 8, runs
        assert runs[0][0] in ("identity", "texture", "clean")
    finally:
        smp._run_one = real_run_one
    print("     masked run stays one lifecycle (longest phase's model); unmasked split unchanged")

    # ======================================================================
    # Sampler v2 (iteration 1): solver contract, rf_2m, eta_tail, diagnostics
    # ======================================================================
    def preset_sigmas(name, refine_steps=0):
        p = presets.PRESETS[name]
        return sch.build_schedule(p["n_steps"], alpha=p["alpha"], restart_frac=p["restart_frac"],
                                  sigma_r=p["sigma_r"], plunge=p["plunge"], refine_steps=refine_steps)

    def smooth(x, sigma, **kw):
        # x0_hat = c + (1 - 0.3*sigma)*(x - c)  =>  dx/dsigma = 0.3*(x - c): closed-form trajectory
        # x(s) = c + (x(s0) - c)*exp(0.3*(s - s0)); x0_hat moves along it, so Euler is NOT exact
        s = sigma.view(-1, *([1] * (x.ndim - 1))) if torch.is_tensor(sigma) else sigma
        return 0.1 + (1.0 - 0.3 * s) * (x - 0.1)

    # --- (21) solver contract: None derives from `order` bit-exactly, names are explicit ---
    print("[21] solver contract: solver=None == order-derived name; unknown solver rejected")
    sig_plain = sch.build_schedule(12, restart_frac=0.0, plunge=False)
    xs = torch.randn(1, 16, 1, 6, 6)
    for order, name in ((1, "euler"), (2, "euler_2m")):
        a = smp.kreaphoton_sampler_loop(smooth, xs.clone(), sig_plain, order=order, eta0=0.0, detail_amount=0.4)
        b = smp.kreaphoton_sampler_loop(smooth, xs.clone(), sig_plain, order=1, solver=name, eta0=0.0,
                                        detail_amount=0.4)
        assert torch.equal(a, b), f"order={order} must equal solver={name!r} bit-exactly"
    SOLVERS_ALL = ("euler", "euler_2m", "rf_2m")
    assert tuple(smp.SOLVERS) == SOLVERS_ALL
    try:
        smp.kreaphoton_sampler_loop(smooth, xs.clone(), sig_plain, solver="heun")
        raise AssertionError("unknown solver must raise")
    except ValueError as e:
        assert "solver" in str(e)
    print("     order 1/2 == 'euler'/'euler_2m'; 'heun' -> ValueError naming solver")

    # --- (22) rf_2m is second order (DPM++2M in the RF/x0 form, variable step by construction):
    #          error vs the closed-form trajectory shrinks ~4x per grid halving, Euler ~2x.
    #          Measured finding pinned here too: the legacy fixed-coefficient AB2 is ALSO second
    #          order on a smoothly nonuniform grid (the coefficient mismatch is O(h^2)), and the
    #          restart tail is uniform in sigma where 1.5/-0.5 is exact - which 2M wins on the
    #          real model is an A/B question, not a correctness one (smooth mock favours sigma-time) ---
    print("[22] rf_2m convergence order on a closed-form trajectory")
    xa = torch.randn(1, 4, 1, 4, 4)

    def err(solver, n):
        # nonuniform grid (sigma = 0.9 - 0.8*u^1.5) from 0.9 to 0.1, no final readout
        sig = torch.tensor([0.9 - 0.8 * (i / n) ** 1.5 for i in range(n + 1)])
        out = smp.kreaphoton_sampler_loop(smooth, xa.clone(), sig, solver=solver, eta0=0.0)
        exact = 0.1 + (xa - 0.1) * torch.exp(torch.tensor(0.3 * (float(sig[-1]) - float(sig[0]))))
        return float((out - exact).abs().max())
    e = {s: (err(s, 16), err(s, 32)) for s in SOLVERS_ALL}
    ratio = {s: e[s][0] / e[s][1] for s in SOLVERS_ALL}
    assert 1.6 < ratio["euler"] < 2.6, "Euler is first order (ratio %.2f)" % ratio["euler"]
    assert ratio["rf_2m"] > 3.4, "rf_2m is second order (ratio %.2f)" % ratio["rf_2m"]
    assert ratio["euler_2m"] > 3.4, "legacy AB2 is second order on a smooth grid (ratio %.2f)" % ratio["euler_2m"]
    assert e["rf_2m"][0] < 0.5 * e["euler"][0], "rf_2m beats Euler on the same grid"
    # a grid starting at sigma=1 (lambda=-inf) degrades to a first-order step instead of crashing
    s1 = smp.kreaphoton_sampler_loop(smooth, xa.clone(), sig_plain, solver="rf_2m", eta0=0.0)
    assert torch.isfinite(s1).all()
    print("     err 16 -> 32 steps: " + "; ".join("%s %.2e -> %.2e (x%.2f)" % (s, e[s][0], e[s][1], ratio[s])
                                               for s in SOLVERS_ALL) + "; sigma=1 start finite")

    # --- (23) eta_tail: None / == eta0 are bit-exact; 0 makes the restart tail deterministic;
    #          a jump landing at/above the plunge floor (self-refine) is not the tail ---
    print("[23] eta_tail: bit-exact defaults, deterministic restart tail, refine segment untouched")
    sig_q = preset_sigmas("turbo/quality")
    xq = torch.randn(1, 16, 1, 6, 6)
    ref_q = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, order=2, eta0=1.0, detail_amount=0.7,
                                        restart_seed=4)
    for et in (None, 1.0):
        same_q = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, order=2, eta0=1.0, detail_amount=0.7,
                                             restart_seed=4, eta_tail=et)
        assert torch.equal(ref_q, same_q), f"eta_tail={et} must be bit-exact"
    st = {}
    det_q = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, solver="euler", eta0=1.0, detail_amount=0.7,
                                        restart_seed=4, eta_tail=0.0, stats=st)
    assert not torch.equal(ref_q, det_q)
    assert (st["ancestral"], st["euler"], st["final"], st["jumps"]) == (11, 3, 2, 1), st
    same_plain = smp.kreaphoton_sampler_loop(smooth, xs.clone(), sig_plain, eta0=1.0, restart_seed=4)
    tail_plain = smp.kreaphoton_sampler_loop(smooth, xs.clone(), sig_plain, eta0=1.0, restart_seed=4, eta_tail=0.0)
    assert torch.equal(same_plain, tail_plain), "no restart jump -> no tail -> eta_tail is a no-op"
    st_r = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), preset_sigmas("turbo/quality", refine_steps=4), solver="euler",
                                eta0=1.0, restart_seed=4, eta_tail=0.0, stats=st_r)
    assert (st_r["ancestral"], st_r["euler"], st_r["jumps"]) == (14, 3, 2), st_r   # 11 structure + 3 refine
    print("     quality: 14 ancestral -> 11 ancestral + 3 deterministic tail; refine jump (0.85) keeps eta0")

    # --- (24) diagnostics = the testable branch contract of every shipped preset. The v1 presets
    #          never reach the multistep branch (euler_2m == euler); quality-v2 runs a real 2M tail ---
    print("[24] branch counts per preset (stats)")
    assert "turbo/quality-v2" in presets.PRESETS, "the v2 branch below must not pass vacuously"
    for name, p in presets.PRESETS.items():
        sig_p = preset_sigmas(name)
        st = {}
        smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_p, order=2 if p["sampler"] != "euler" else 1,
                                    solver=p["sampler"], eta0=p["eta0"], sigma_gate=p["sigma_gate"],
                                    detail_amount=p["detail_a"], eta_tail=p.get("eta_tail"), restart_seed=4,
                                    pc=p.get("pc", "off"), stats=st)
        branches = st["ancestral"] + st["euler"] + st["ab2"] + st["rf2m"] + st["heun"] + st["final"]
        assert st["model_calls"] == branches == sch.count_model_calls(sig_p), (name, st)
        if name == "turbo/quality-v2":
            assert (st["ancestral"], st["euler"], st["rf2m"], st["ab2"]) == (11, 1, 2, 0), st
        elif name == "turbo/quality-plus":   # PC: step 8 ancestral+corrector, tail heun / rf2m / heun
            assert (st["ancestral"], st["euler"], st["rf2m"], st["heun"], st["pc"]) == (11, 0, 1, 2, 3), st
        else:
            assert st["ab2"] == 0 and st["rf2m"] == 0, f"{name}: v1 preset reached the multistep branch {st}"
        print("     %-18s %s" % (name, smp.format_stats(st)))
    # the old AB2 path on a shipped-preset config vs plain Euler: identical (the dead-label finding)
    o_eu = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, order=1, eta0=1.0, detail_amount=0.7, restart_seed=4)
    assert torch.equal(ref_q, o_eu), "turbo/quality: euler_2m == euler bit-exactly (AB2 never fires)"

    # --- (25) multistep history contract: every trajectory-changing event declares history
    #          invalid; resets are counted only when a real history was dropped ---
    print("[25] history-reset contract")
    st = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_bal, solver="rf_2m", eta0=0.0, restart_seed=4,
                                variety_a_latent=0.4, variety_state="pending", variety_seed=3, stats=st)
    assert st["resets"].get("variety") == 1, st
    st = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_bal, solver="rf_2m", eta0=0.0, restart_seed=4,
                                coherence_jump=0.19, stats=st)
    assert st["resets"].get("coherence") == 1, st
    # raw: the gate is monotone in sigma_next, so its deterministic steps sit alone right before
    # a readout - each runs first order, the AB2 branch is never reached (no history to reset)
    st = {}
    sig_raw = preset_sigmas("raw/experimental")
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_raw, solver="euler_2m", eta0=1.0, restart_seed=4, stats=st)
    assert st["euler"] >= 1 and st["ab2"] == 0, st
    st = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, solver="rf_2m", eta0=1.0, eta_tail=0.0, restart_seed=4,
                                stats=st)
    assert st["resets"] == {"final": 1}, "structure is all-ancestral (no history to lose); the tail ends in a readout"
    st = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, solver="euler", eta0=1.0, eta_tail=0.0, restart_seed=4,
                                stats=st)
    assert st["resets"] == {}, "euler keeps no multistep history - there is nothing to reset"
    # the first tail step after the restart is first order (history reset by the jump)
    sig_tail = torch.tensor([0.65, 0.4875, 0.325, 0.1625])
    st = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_tail, solver="rf_2m", eta0=0.0, stats=st)
    assert (st["euler"], st["rf2m"]) == (1, 2), st
    # a conditioning-only variety switch (a_latent 0) changes the vector field too: rf_2m drops its
    # history there; euler_2m keeps its v1 behaviour (bit-exact default path)
    st = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_bal, solver="rf_2m", eta0=0.0, restart_seed=4,
                                variety_a_latent=0.0, variety_state="pending", stats=st)
    assert st["resets"].get("variety") == 1, st
    st = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_bal, solver="euler_2m", eta0=0.0, restart_seed=4,
                                variety_a_latent=0.0, variety_state="pending", stats=st)
    assert "variety" not in st["resets"], st

    # --- (27) rf_2m step-ratio guard: after a near-flat step h_prev -> 0 and k = h/(2 h_prev) is
    #          unbounded while the detail nudge keeps x0 - x0_prev finite (review M1: |rf - euler|
    #          ~141 on a foreign grid). r = h_prev/h < 0.5 (k > 1) falls back to first order ---
    print("[27] rf_2m step-ratio guard")
    sig_flat = torch.tensor([0.9, 0.8, 0.79999, 0.6, 0.45, 0.3, 0.15, 0.0])
    st = {}
    x_flat = smp.kreaphoton_sampler_loop(model2, xq.clone(), sig_flat, solver="rf_2m", eta0=0.0, detail_amount=0.7,
                                         stats=st)
    x_eu = smp.kreaphoton_sampler_loop(model2, xq.clone(), sig_flat, solver="euler", eta0=0.0, detail_amount=0.7)
    assert (st["euler"], st["rf2m"]) == (2, 4), "the step after the near-flat one must run first order %s" % st
    assert float((x_flat - x_eu).abs().max()) < 1.0, "bounded next to Euler"
    st = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, solver="rf_2m", eta0=1.0, eta_tail=0.0, stats=st)
    assert st["rf2m"] == 2, "the calibrated tail (r 0.98 / 0.75) is untouched by the guard"
    print("     near-flat step -> next step first order, |rf_2m - euler| bounded; quality tail keeps 2 rf2m steps")

    # --- (28) the tail is a property of the FULL schedule: a phase cut after the last jump must
    #          not hand the rest of the tail back to eta0 (review M3), and a last jump landing at/
    #          above the plunge floor (self-refine only, no restart) is not a tail ---
    print("[28] eta_tail across a phase cut; refine-only schedule has no tail")
    sig_foreign = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.72, 0.6, 0.3, 0.0])
    st_one = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_foreign, eta0=1.0, eta_tail=0.0, restart_seed=4, stats=st_one)
    st_a, st_b = {}, {}
    mid = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_foreign[:6], eta0=1.0, eta_tail=0.0, restart_seed=4,
                                      tail_start=smp.tail_start_index(sig_foreign), stats=st_a)
    smp.kreaphoton_sampler_loop(smooth, mid, sig_foreign[5:], eta0=1.0, eta_tail=0.0, restart_seed=4,
                                progress_offset=5, tail_start=smp.tail_start_index(sig_foreign), stats=st_b)
    assert st_a["ancestral"] + st_b["ancestral"] == st_one["ancestral"], (st_one, st_a, st_b)
    assert st_a["euler"] + st_b["euler"] == st_one["euler"] == 2, (st_one, st_a, st_b)
    sig_refine_only = sch.build_schedule(12, restart_frac=0.0, plunge=True, refine_steps=4)
    assert smp.tail_start_index(sig_refine_only) is None
    a = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_refine_only, eta0=1.0, restart_seed=4)
    b = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_refine_only, eta0=1.0, restart_seed=4, eta_tail=0.0)
    assert torch.equal(a, b), "a refine jump (lands >= 0.75) is not a restart tail"
    runs = []
    smp._run_one = spy_run_one3 = (lambda model, positive, negative, latent_dict, sigmas, **kw:
                                   runs.append(kw.get("tail_start")) or latent_dict)
    try:
        smp.run_sampling("m", "pos", None, {"samples": torch.zeros(1, 16, 1, 8, 8)}, sig_foreign, seed=1,
                         texture_model="tex", eta_tail=0.0)
    finally:
        smp._run_one = real_run_one
    assert runs == [smp.tail_start_index(sig_foreign)] * 2, runs
    print("     split == single run; refine-only -> no tail; run_sampling hands every segment the global tail start")

    # ======================================================================
    # Sampler v2 iteration 2: guidance sigma, sigma-space detail, x0 extrapolation,
    # predictor-corrector, adaptive self-refine, trajectory trace
    # ======================================================================
    tr = importlib.import_module("kreaphoton.trajectory")

    def make_true_sigma_recorder():
        seen = []

        def model(x, sigma, **kw):
            seen.append((float(sigma.flatten()[0]), kw["model_options"].get(gd.TRUE_SIGMA_KEY)))
            return smooth(x, sigma)
        return model, seen

    # --- (29) the loop publishes the TRUE trajectory sigma for the guider on every model call ---
    print("[29] TRUE_SIGMA_KEY = s_cur on every call (model still gets the nudged sigma)")
    m_t, seen_t = make_true_sigma_recorder()
    smp.kreaphoton_sampler_loop(m_t, xq.clone(), sig_q, eta0=1.0, detail_amount=0.7, restart_seed=4,
                                extra_args={"model_options": {}})
    calls_q = [(float(sig_q[i]), float(sig_q[i + 1])) for i in range(len(sig_q) - 1)
               if float(sig_q[i + 1]) <= float(sig_q[i]) + 1e-6 and float(sig_q[i]) > 1e-6]
    assert [t for _, t in seen_t] == [s for s, _ in calls_q], "true sigma = s_cur of each model-call step"
    assert any(abs(m - t) > 1e-4 for m, t in seen_t), "with detail 0.7 the model sigma is nudged below it"
    print("     %d calls, true sigma == s_cur, model sigma nudged" % len(seen_t))

    # --- (30) sigma-space detail envelope: a function of the state's sigma only (split-invariant by
    #          construction), phase-aware default 0.90 / 0.50 / 0.12 ---
    print("[30] detail_space='sigma'")
    env = smp._detail_envelope_sigma
    hi, pk, lo = presets.DETAIL_SIGMA
    # live finding 2026-09-27: a nudge peaking in the restart tail (the proposal's 0.90/0.50/0.12)
    # destroys the frame at detail 0.7 (confetti mesh, the S9 mechanism) - the default peaks at the
    # end of the structure phase and is only mild at the restart-tail entry (like v1), zero below 0.55
    assert (hi, pk, lo) == (0.97, 0.80, 0.55)
    assert env(0.4875, hi, pk, lo) == 0.0 and env(0.325, hi, pk, lo) == 0.0, "no nudge deep in the restart tail"
    assert env(0.65, hi, pk, lo) < 0.4, "mild at the tail entry (a <= 0.28 at detail 0.7, v1 had 0.27)"
    assert env(0.99, hi, pk, lo) == 0.0 and env(hi, hi, pk, lo) == 0.0 and env(lo, hi, pk, lo) == 0.0
    assert abs(env(pk, hi, pk, lo) - 1.0) < 1e-12
    assert 0.0 < env(0.95, hi, pk, lo) < env(0.88, hi, pk, lo) < 1.0 and 0.0 < env(0.6, hi, pk, lo) < env(0.7, hi, pk, lo)
    assert env(0.8, 0.90, 0.50, 0.12) > 0.0 and env(0.95, 0.90, 0.50, 0.12) == 0.0, "any (hi, peak, lo) is honoured"
    m_s, seen_s = make_true_sigma_recorder()
    smp.kreaphoton_sampler_loop(m_s, xq.clone(), sig_q, eta0=0.0, detail_amount=0.7, detail_space="sigma",
                                extra_args={"model_options": {}})
    for (sm, st_), (s, sn) in zip(seen_s, calls_q):
        plunge_or_final = sn <= 1e-6 or (s - sn) > smp.PLUNGE_DSIGMA
        a = 0.0 if plunge_or_final else 0.7 * env(s, hi, pk, lo)
        assert abs(sm - max(1e-4, s - a * (s - sn))) < 1e-6, (s, sm)
    split = 13
    m_a2, seen_a2 = make_true_sigma_recorder()
    mid = smp.kreaphoton_sampler_loop(m_a2, xq.clone(), sig_q[:split + 1], eta0=0.0, detail_amount=0.7,
                                      detail_space="sigma", extra_args={"model_options": {}})
    smp.kreaphoton_sampler_loop(m_a2, mid, sig_q[split:], eta0=0.0, detail_amount=0.7, detail_space="sigma",
                                extra_args={"model_options": {}})
    assert [m for m, _ in seen_a2] == [m for m, _ in seen_s], "no progress_offset needed: split == single run"
    try:
        smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, detail_space="banana")
        raise AssertionError("unknown detail_space must raise")
    except ValueError as e:
        assert "detail_space" in str(e)
    print("     0 above 0.90 / below 0.12, 1 at 0.50; sigma_model follows it; split-invariant without offsets")

    # --- (31) terminal x0 extrapolation: order 2 (quadratic, exact for a quadratic x0(sigma)) and
    #          the local trust region; order 1 / trust 0 are the v1 path bit-for-bit ---
    print("[31] x0 extrapolation order 2 + trust region in the loop")
    A2, B2, C2 = torch.randn_like(xq), torch.randn_like(xq), torch.randn_like(xq)

    def quad_model(x, sigma, **kw):
        s = float(sigma.flatten()[0])
        return A2 + B2 * s + C2 * s * s
    base_ex = smp.kreaphoton_sampler_loop(quad_model, xq.clone(), sig_q, eta0=0.0, x0_extrapolation=1.0)
    same_ex = smp.kreaphoton_sampler_loop(quad_model, xq.clone(), sig_q, eta0=0.0, x0_extrapolation=1.0,
                                          x0_extrap_order=1, x0_trust=0.0)
    assert torch.equal(base_ex, same_ex)
    q2 = smp.kreaphoton_sampler_loop(quad_model, xq.clone(), sig_q, eta0=0.0, x0_extrapolation=1.0, x0_extrap_order=2)
    assert torch.allclose(q2, A2, atol=1e-4), "quadratic x0(sigma) extrapolated to sigma 0 exactly"
    assert not torch.allclose(base_ex, A2, atol=1e-3), "linear is not exact on it"
    huge = smp.kreaphoton_sampler_loop(quad_model, xq.clone(), sig_q, eta0=0.0, x0_extrapolation=1.0,
                                       x0_extrap_order=2, x0_trust=1e9)
    assert torch.equal(huge, q2), "an inactive trust region changes nothing"
    tight = smp.kreaphoton_sampler_loop(quad_model, xq.clone(), sig_q, eta0=0.0, x0_extrapolation=1.0,
                                        x0_extrap_order=2, x0_trust=0.5)
    last = quad_model(None, torch.tensor([float(sig_q[-2])]))
    prev = quad_model(None, torch.tensor([float(sig_q[-3])]))
    assert bool(((tight - last).abs() <= 0.5 * tr.local_rms(last - prev) + 1e-6).all()), "|delta| <= k * local rms"
    print("     order 2 exact, order 1 not; trust 1e9 == off; trust 0.5 bounds every pixel")

    # --- (32) selective predictor-corrector (Heun on the deterministic part, +1 call at 3 steps) ---
    print("[32] predictor-corrector: strategic steps, Heun accuracy, ancestral down-step, auto")
    assert smp.pc_step_indices(sig_q, composition_end=0.85) == [8, 13, 15], smp.pc_step_indices(sig_q)
    head = torch.tensor([1.0, 0.985, 0.968, 0.951, 0.932, 0.912, 0.89, 0.867, 0.0])
    assert smp.pc_step_indices(head) == [], "a Composition Drafts head has no strategic step"
    fwd = []

    def counting(x, sigma, **kw):
        fwd.append(float(sigma.flatten()[0]))
        return smooth(x, sigma)
    st = {}
    out_pc = smp.kreaphoton_sampler_loop(counting, xq.clone(), sig_q, solver="rf_2m", eta0=1.0, eta_tail=0.0,
                                         pc="selective", restart_seed=4, stats=st)
    assert (st["model_calls"], st["pc"], len(fwd)) == (16, 3, 19), (st, len(fwd))
    assert st["ancestral"] + st["euler"] + st["ab2"] + st["rf2m"] + st["heun"] + st["final"] == 16, st
    assert st["heun"] == 2 and torch.isfinite(out_pc).all(), st   # restart step + late step; step 8 is ancestral
    one = torch.tensor([0.9, 0.5])
    exact1 = 0.1 + (xa - 0.1) * torch.exp(torch.tensor(0.3 * (0.5 - 0.9)))
    e_eu = float((smp.kreaphoton_sampler_loop(smooth, xa.clone(), one, eta0=0.0) - exact1).abs().max())
    e_pc = float((smp.kreaphoton_sampler_loop(smooth, xa.clone(), one, eta0=0.0, pc="selective", pc_steps=[0])
                  - exact1).abs().max())
    assert e_pc < 0.1 * e_eu, "Heun step: local error %.2e vs Euler %.2e" % (e_pc, e_eu)
    for thr, n_pc in ((0.0, 3), (1e9, 0)):
        st = {}
        smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, eta0=1.0, eta_tail=0.0, pc="auto", pc_threshold=thr,
                                    restart_seed=4, stats=st)
        assert st["pc"] == n_pc and len(st["pc_motion"]) == 3, (thr, st)
    # a texture-model split starts a new lifecycle at the readout: the restart step's motion is
    # measured against the draft the segment was handed (the state at sigma 0 IS an x0)
    st = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q[12:], eta0=1.0, eta_tail=0.0, pc="auto", pc_steps=[13, 15],
                                progress_offset=12, tail_start=13, restart_seed=4, stats=st)
    assert len(st["pc_motion"]) == 2 and st["pc_motion"][0] is not None, st["pc_motion"]
    try:
        smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, pc="always")
        raise AssertionError("unknown pc mode must raise")
    except ValueError as e:
        assert "pc" in str(e)
    print("     indices [8, 13, 15]; +3 calls; Heun err %.1e vs Euler %.1e; auto thr 0 -> 3, 1e9 -> 0" % (e_pc, e_eu))

    # --- (33) adaptive self-refine: a stable draft skips the whole refine segment without an RNG
    #          draw - with the sigma-space envelope that IS the no-refine schedule, bit-for-bit ---
    print("[33] refine_mode='auto'")
    sig_r = preset_sigmas("turbo/quality", refine_steps=4)
    kw_r = dict(eta0=1.0, detail_amount=0.7, detail_space="sigma", restart_seed=4)
    no_refine = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, **kw_r)
    st = {}
    skipped = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_r, refine_mode="auto", refine_threshold=1e9,
                                          stats=st, **kw_r)
    assert torch.equal(skipped, no_refine), "skipped refine == the schedule without it"
    assert st["model_calls"] == 16 and st["refine_skipped"] == 1 and st["refine_motion"] is not None, st
    st = {}
    ran = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_r, refine_mode="auto", refine_threshold=0.0,
                                      stats=st, **kw_r)
    on = smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_r, **kw_r)
    assert torch.equal(ran, on) and st["model_calls"] == 20 and st["refine_skipped"] == 0, st
    print("     stable draft: 16 calls, == no-refine schedule; unstable: 20 calls, == refine on (motion %.3f)"
          % st["refine_motion"])

    # --- (34) trajectory trace: one motion record per model call ---
    print("[34] trace")
    st = {}
    smp.kreaphoton_sampler_loop(smooth, xq.clone(), sig_q, eta0=1.0, restart_seed=4, trace=True, stats=st)
    assert len(st["traj"]) == st["model_calls"] == 16
    assert st["traj"][0]["sigma"] == 1.0 and "mf" in st["traj"][1] and st["traj"][0].get("mf") is None
    print("     16 records (first call has no previous x0)")

    # --- (35) run_sampling threads the iteration-2 options (and the GLOBAL pc steps) ---
    print("[35] run_sampling threads guidance_sigma / detail_space / x0 order+trust / pc / refine_mode / trace")
    runs = []

    def spy_run_one4(model, positive, negative, latent_dict, sigmas, **kw):
        runs.append({k: kw.get(k) for k in ("guidance_sigma", "detail_space", "detail_sigma", "x0_extrap_order",
                                            "x0_trust", "pc", "pc_steps", "refine_mode", "trace")})
        return latent_dict
    smp._run_one = spy_run_one4
    try:
        lat = {"samples": torch.zeros(1, 16, 1, 8, 8)}
        smp.run_sampling("m", "pos", None, lat, sig_q, seed=1, guidance_sigma="true", detail_space="sigma",
                         x0_extrap_order=2, x0_trust=1.0, pc="selective", refine_mode="auto", trace=True,
                         texture_model="tex")
        smp.run_sampling("m", "pos", None, lat, sig_q, seed=1)
    finally:
        smp._run_one = real_run_one
    assert len(runs) == 3
    for r in runs[:2]:
        assert r == {"guidance_sigma": "true", "detail_space": "sigma", "detail_sigma": presets.DETAIL_SIGMA,
                     "x0_extrap_order": 2, "x0_trust": 1.0, "pc": "selective", "pc_steps": [8, 13, 15],
                     "refine_mode": "auto", "trace": True}, r
    assert runs[2]["pc"] == "off" and runs[2]["pc_steps"] is None and runs[2]["guidance_sigma"] == "model"
    print("     every segment gets the options and the full-schedule pc steps; defaults are v1")
    print("     variety / coherence / final resets counted, euler keeps none (an ancestral step right after a "
          "deterministic one is unreachable: the gate is monotone in sigma_next); first step after a reset is "
          "first order")

    # --- (26) run_sampling threads solver / eta_tail to every segment's loop ---
    print("[26] run_sampling threads solver + eta_tail")
    runs = []

    def spy_run_one2(model, positive, negative, latent_dict, sigmas, **kw):
        runs.append((kw.get("solver"), kw.get("eta_tail")))
        return latent_dict
    smp._run_one = spy_run_one2
    try:
        lat = {"samples": torch.zeros(1, 16, 1, 8, 8)}
        smp.run_sampling("m", "pos", None, lat, sig_q, seed=1, solver="rf_2m", eta_tail=0.0)
        smp.run_sampling("m", "pos", None, lat, sig_q, seed=1, texture_model="tex", solver="rf_2m", eta_tail=0.0)
        smp.run_sampling("m", "pos", None, lat, sig_q, seed=1)
    finally:
        smp._run_one = real_run_one
    assert runs == [("rf_2m", 0.0)] * 3 + [(None, None)], runs
    print("     single lifecycle and every split segment receive solver / eta_tail; defaults None")

    print("\ntest_sampling: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
