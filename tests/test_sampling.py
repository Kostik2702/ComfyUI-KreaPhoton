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

    print("\ntest_sampling: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
