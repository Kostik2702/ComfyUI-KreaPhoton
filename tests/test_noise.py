# -*- coding: utf-8 -*-
"""Plain-assert test for kreaphoton/noise.py (manifold contraction).

Run: <embedded python> tests/test_noise.py
"""
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    import torch

    noise_mod = _load("kreaphoton_noise", "kreaphoton/noise.py")
    presets = _load("kreaphoton_presets", "kreaphoton/presets.py")

    torch.manual_seed(0)
    print("=" * 78)
    print("test_noise: kreaphoton.noise contract_noise")
    print("=" * 78)

    eps = torch.randn(1, 16, 1, 32, 24)  # 5D unit N(0,1), small grid for speed

    # --- strength=1.0 is an EXACT no-op (bit-identical, not just close) ---
    out = noise_mod.contract_noise(eps, strength=1.0)
    assert out is eps or torch.equal(out, eps)
    print("[1] strength=1.0: bit-exact no-op  OK")

    # --- scalar strength scales the tensor uniformly ---
    for s in (0.82, 0.47, 1.0 / 2.14):
        out = noise_mod.contract_noise(eps, strength=s)
        assert torch.allclose(out, eps * s, atol=1e-7)
        ratio = (out.std() / eps.std()).item()
        print("     strength=%.4f: std ratio out/in = %.4f (expected %.4f)" % (s, ratio, s))
        assert abs(ratio - s) < 0.05, "measured std ratio should track the scalar strength"

    # --- per_channel mode: exact per-channel std/mean application ---
    print("[2] per_channel mode (Advanced): MANIFOLD_STD/MEAN applied exactly")
    out = noise_mod.contract_noise(eps, strength=1.0, per_channel=True,
                                    manifold_std=presets.MANIFOLD_STD,
                                    manifold_mean=presets.MANIFOLD_MEAN)
    std_t = torch.tensor(presets.MANIFOLD_STD).view(1, 16, 1, 1, 1)
    mean_t = torch.tensor(presets.MANIFOLD_MEAN).view(1, 16, 1, 1, 1)
    expected = eps * std_t + mean_t
    assert torch.allclose(out, expected, atol=1e-6)
    print("     per-channel scale+offset matches eps*MANIFOLD_STD+MANIFOLD_MEAN exactly")

    # per_channel without mean: offset must be zero (no silent bias injection)
    out_no_mean = noise_mod.contract_noise(eps, strength=1.0, per_channel=True,
                                            manifold_std=presets.MANIFOLD_STD)
    assert torch.allclose(out_no_mean, eps * std_t, atol=1e-6)
    print("     per_channel without manifold_mean: no offset applied (as documented)")

    print("\ntest_noise: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
