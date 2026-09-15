# -*- coding: utf-8 -*-
"""
Plain-assert test for kreaphoton/schedules.py (no pytest in the ComfyUI embedded
interpreter). Cross-checks against the LIVE comfy.samplers.calculate_sigmas +
ModelSamplingFlux (planning-council H15) — not a numpy replica of comfy's math.

Run: <embedded python> tests/test_schedules.py
"""
import importlib.util
import math
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
    import torch
    import comfy.model_sampling
    import comfy.samplers

    sch = _load("kreaphoton_schedules", "kreaphoton/schedules.py")

    print("=" * 78)
    print("test_schedules: kreaphoton.schedules vs LIVE comfy.samplers.calculate_sigmas")
    print("=" * 78)

    # --- (A0) algebraic identity: flux_time_shift(mu,t) == sigma_from_t(t, e^mu) ---
    diffs = [abs(sch.flux_time_shift(sch.MU, t) - sch.sigma_from_t(t, sch.ALPHA))
             for t in [1e-6 + i * (1.0 - 1e-6) / 2000 for i in range(2001)]]
    dmax_ident = max(diffs)
    print("[A0] flux_time_shift == sigma_from_t(alpha=e^mu): max diff %.3e" % dmax_ident)
    assert dmax_ident < 1e-12, "algebraic identity failed"

    # --- (A) LIVE identity: build_schedule(12) vs calculate_sigmas(ModelSamplingFlux(), "sgm_uniform", 12)
    ms = comfy.model_sampling.ModelSamplingFlux()  # model_config=None -> shift defaults to 1.15
    assert abs(ms.shift - 1.15) < 1e-9
    for n in (6, 8, 12, 16, 20, 24):
        live = comfy.samplers.calculate_sigmas(ms, "sgm_uniform", n)
        ours = sch.build_schedule(n, alpha=sch.ALPHA, restart_frac=0.0, plunge=False)
        dmax = (live.double() - ours.double()).abs().max().item()
        print("     N=%-2d  max |sigma diff| vs LIVE calculate_sigmas = %.3e  (required < 1e-6)" % (n, dmax))
        assert live.shape == ours.shape
        assert dmax < 1e-6, f"N={n}: does not reproduce LIVE stock grid"
    assert abs(float(ours[0]) - 1.0) < 1e-6
    assert float(ours[-1]) == 0.0

    # --- (B) continuous scaling N=6..24 ---
    print("[B] scaling N=6..24")
    max_steps = {}
    for n in range(6, 25):
        s = sch.build_schedule(n)
        assert s.shape[0] == n + 1
        seg = s[:-1]
        assert torch.all(seg[1:] < seg[:-1]), "structure segment must strictly descend"
        assert abs(float(s[0]) - 1.0) < 1e-6 and float(s[-1]) == 0.0
        max_steps[n] = float((seg[:-1] - seg[1:]).max()) if n > 1 else 0.0
    assert max_steps[24] < max_steps[12] < max_steps[6]
    print("     max in-segment step:  N=6 %.4f | N=12 %.4f | N=24 %.4f"
          % (max_steps[6], max_steps[12], max_steps[24]))

    # grid nesting N -> 2N (q=1 default keeps every original node)
    for n in (6, 8, 12):
        a, b = sch.build_schedule(n), sch.build_schedule(2 * n)
        nest = (a[:-1].double() - b[0:-1:2].double()).abs().max().item()
        print("     nesting N=%d in N=%d: max |diff| = %.3e" % (n, 2 * n, nest))
        assert nest < 1e-6

    # --- (C) restart segment: exactly one ascending jump, correctly parsed ---
    print("[C] restart-jump encoding + SegmentMap")
    for n in (8, 12, 16, 24):
        s = sch.build_schedule(n, restart_frac=0.25, sigma_r=0.6)
        seg_map = sch.infer_segment_map(s)
        assert not seg_map.ambiguous
        assert seg_map.restart_start is not None
        assert float(s[seg_map.structure_end]) == 0.0
        assert abs(float(s[seg_map.restart_start]) - 0.6) < 1e-6
        tail = s[seg_map.restart_start:]
        assert torch.all(tail[1:] < tail[:-1]) or tail.shape[0] == 1
        assert float(tail[-1]) == 0.0
        n_model_calls = (s.shape[0] - 1) - 1  # jump consumes no model call
        assert n_model_calls == n
        print("     N=%-2d  restart_start=%d, sigma_r matched, model_calls=%d"
              % (n, seg_map.restart_start, n_model_calls))

    # --- (D) graceful degrade: multi-jump array -> ambiguous, no guess ---
    print("[D] graceful degrade on malformed/foreign SIGMAS")
    malformed = torch.tensor([1.0, 0.5, 0.0, 0.7, 0.0, 0.9, 0.0])  # two ascending jumps
    seg_map = sch.infer_segment_map(malformed)
    assert seg_map.ambiguous and seg_map.restart_start is None
    print("     two-jump array -> ambiguous=True, restart_start=None (safe fallback)")

    # denoise<1 truncation: suffix of a valid schedule, no jump present -> no false positive
    full = sch.build_schedule(12, restart_frac=0.25, sigma_r=0.6)
    truncated = full[3:]  # simulate denoise<1 (starts mid-structure)
    seg_map_trunc = sch.infer_segment_map(truncated[:truncated.shape[0] // 2])  # cut before the jump
    print("     truncated-schedule prefix -> ambiguous=%s, restart_start=%s (must not crash)"
          % (seg_map_trunc.ambiguous, seg_map_trunc.restart_start))

    # duplicate boundary values must not false-positive as ascending
    dup = torch.tensor([1.0, 0.5, 0.5, 0.2, 0.0])
    seg_map_dup = sch.infer_segment_map(dup)
    assert seg_map_dup.restart_start is None and not seg_map_dup.ambiguous
    print("     duplicate-boundary array -> no false restart detected")

    # --- (E) refine_schedule: partial clean descent for img2img (denoise<1) ---
    print("[E] refine_schedule (img2img partial descent)")
    n = 12
    starts = {}
    for d in (1.0, 0.7, 0.5, 0.3):
        s = sch.refine_schedule(n, denoise=d)
        assert s.shape[0] == n + 1, f"refine_schedule must return n_steps+1 sigmas (d={d})"
        assert float(s[-1]) == 0.0, "refine schedule must end at 0"
        assert torch.all(s[1:] < s[:-1]), "refine schedule must strictly descend (no restart/plunge)"
        assert float(s[0]) <= 1.0 + 1e-6, "start sigma cannot exceed sigma_max"
        starts[d] = float(s[0])
        print("     denoise=%.1f  start_sigma=%.4f  (len=%d)" % (d, starts[d], s.shape[0]))
    # lighter denoise -> lower start sigma (less noise injected = gentler refine)
    assert starts[0.3] < starts[0.5] < starts[0.7] < starts[1.0], "start sigma must fall with denoise"
    assert abs(starts[1.0] - 1.0) < 1e-6, "denoise=1.0 must equal the full plain descent (start==sigma_max)"
    # denoise=1.0 refine_schedule == build_schedule plain descent, elementwise
    d1 = (sch.refine_schedule(n, denoise=1.0).double()
          - sch.build_schedule(n, restart_frac=0.0, plunge=False).double()).abs().max().item()
    assert d1 < 1e-6, "denoise=1.0 must be identical to build_schedule(plain)"
    print("     denoise=1.0 == build_schedule(plain): max diff %.3e" % d1)

    # --- (F) NFE contract for EVERY UI step count (audit F05): model calls == steps ---
    print("[F] model_calls == steps for steps 1..24 x every preset restart/plunge combo")
    combos = [(0.0, False), (0.25, True), (0.25, False), (0.20, False), (0.6, True)]
    for n in range(1, 25):
        for frac, plunge in combos:
            s = sch.build_schedule(n, restart_frac=frac, sigma_r=0.65, plunge=plunge)
            calls = sch.count_model_calls(s)
            assert calls == n, f"steps={n} frac={frac} plunge={plunge}: {calls} model calls (sigmas {s.tolist()})"
            assert float(s[0]) == 1.0 and float(s[-1]) == 0.0
            sch.validate_sigmas(s)   # every own schedule passes the production validator
    # restart is dropped below MIN_STEPS_FOR_RESTART, present at/above it
    assert sch.infer_segment_map(sch.build_schedule(3, restart_frac=0.25, plunge=True)).restart_start is None
    assert sch.infer_segment_map(sch.build_schedule(4, restart_frac=0.25, plunge=True)).restart_start is not None
    print("     1..24 steps x 5 combos: exact; restart dropped below %d steps" % sch.MIN_STEPS_FOR_RESTART)

    # --- (G) validate_sigmas: production boundary (audit F04) ---
    print("[G] validate_sigmas accept/reject table")
    ok_cases = {
        "plain": sch.build_schedule(12),
        "restart": sch.build_schedule(12, restart_frac=0.25, sigma_r=0.65, plunge=True),
        "refine": sch.refine_schedule(12, denoise=0.4),
        "partial (no final 0)": torch.tensor([1.0, 0.7, 0.4]),
        "flat duplicate": torch.tensor([1.0, 0.5, 0.5, 0.0]),
        "list input": [1.0, 0.5, 0.0],
    }
    for name, s in ok_cases.items():
        out = sch.validate_sigmas(s)
        assert torch.is_tensor(out) and out.ndim == 1, name
    print("     accepted: %s" % ", ".join(ok_cases))
    sch.validate_sigmas(torch.tensor([1.0, 0.75, 0.0, 0.85, 0.75, 0.0, 0.65, 0.3, 0.0]))   # self-refine + restart
    bad_cases = {
        "three jumps": (torch.tensor([1.0, 0.5, 0.0, 0.7, 0.0, 0.9, 0.0, 0.6, 0.0]), "ascending jumps"),
        "NaN": (torch.tensor([1.0, float("nan"), 0.0]), "NaN"),
        "Inf": (torch.tensor([1.0, float("inf"), 0.0]), "NaN/Inf"),
        "negative": (torch.tensor([1.0, -0.1, 0.0]), "negative"),
        "sigma>1 (EPS)": (torch.tensor([14.6, 1.0, 0.0]), "> 1.0"),
        "starts at 0": (torch.tensor([0.0, 0.0]), "starts at 0"),
        "2-D": (torch.zeros(2, 3), "1-D"),
        "single": (torch.tensor([1.0]), "at least 2"),
    }
    for name, (s, needle) in bad_cases.items():
        try:
            sch.validate_sigmas(s)
            raise AssertionError("%s must be rejected" % name)
        except ValueError as e:
            assert needle in str(e), "%s: message %r lacks %r" % (name, str(e), needle)
    print("     rejected with naming errors: %s" % ", ".join(bad_cases))

    # --- (H) resolution-aware shift (v1.4): canonical Krea 2 mu by token count ---
    print("[H] alpha_for_latent (canonical dynamic shift)")
    assert abs(sch.canonical_mu(256) - 0.5) < 1e-9
    assert abs(sch.canonical_mu(6400) - 1.15) < 1e-9
    assert abs(sch.canonical_mu(4096) - 0.906) < 1e-3, sch.canonical_mu(4096)
    assert sch.canonical_mu(100) == 0.5 and sch.canonical_mu(10000) == 1.15
    a_L = sch.alpha_for_latent(200, 136)          # 1088x1600 -> 6800 tokens -> capped at e^1.15
    a_S = sch.alpha_for_latent(128, 128)          # 1024x1024 -> 4096 tokens
    a_XS = sch.alpha_for_latent(64, 64)           # 512x512   -> 1024 tokens
    assert abs(a_L - sch.ALPHA) < 1e-9, "L tier must keep the preset alpha (bit-identical schedules)"
    assert abs(a_S - math.exp(0.906)) < 3e-3 and a_S < a_L
    assert a_XS < a_S and a_XS >= math.exp(0.5) - 1e-9
    assert torch.equal(sch.build_schedule(12, alpha=a_L), sch.build_schedule(12))
    print("     L 1088x1600 -> alpha %.3f (unchanged) | S 1024^2 -> %.3f | 512^2 -> %.3f" % (a_L, a_S, a_XS))

    # --- (I) self-refine pass (v1.4): second ascending jump, calls == n + refine_steps ---
    print("[I] build_schedule(refine_steps=4, refine_sigma=0.85)")
    base12 = sch.build_schedule(12, restart_frac=0.25, sigma_r=0.65, plunge=True)
    sr = sch.build_schedule(12, restart_frac=0.25, sigma_r=0.65, plunge=True, refine_steps=4, refine_sigma=0.85)
    s = sr.tolist()
    jumps = [i for i in range(len(s) - 1) if s[i + 1] > s[i] + 1e-6]
    assert len(jumps) == 2, jumps
    assert sch.count_model_calls(sr) == 12 + 4
    assert torch.equal(sr[:len(base12) - 4], base12[:len(base12) - 4]), "structure + first plunge unchanged"
    j0 = jumps[0]
    assert s[j0] == 0.0 and abs(s[j0 + 1] - 0.85) < 1e-6, "self-refine re-noises the plunge draft to refine_sigma"
    seg = s[j0 + 1:jumps[1] + 1]
    assert seg[-1] == 0.0 and abs(seg[-2] - sch.PLUNGE_SIGMA_FLOOR) < 1e-6, "refine descends to the floor then plunges"
    assert all(seg[i + 1] < seg[i] for i in range(len(seg) - 1))
    assert abs(s[jumps[1] + 1] - 0.65) < 1e-6 and s[-1] == 0.0, "restart segment follows unchanged"
    assert torch.equal(sr[jumps[1] + 1:], base12[len(base12) - 4:])
    sch.validate_sigmas(sr)
    assert torch.equal(sch.build_schedule(12, restart_frac=0.25, sigma_r=0.65, plunge=True, refine_steps=0), base12)
    one = sch.build_schedule(12, restart_frac=0.25, sigma_r=0.65, plunge=True, refine_steps=1, refine_sigma=0.85)
    assert sch.count_model_calls(one) == 13
    print("     2 jumps, 16 model calls, refine 0.85 -> 0.75 -> plunge, restart unchanged, refine_steps=0 bit-identical")

    print("\ntest_schedules: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
