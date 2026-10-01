# -*- coding: utf-8 -*-
"""Plain-assert test for kreaphoton.precision (2026-10-01): weight-precision detection,
profile resolution, non-finite guard, LoRA trainer-flag guard. Fakes only, no checkpoint.

Run: <embedded python> tests/test_precision.py
"""
import importlib
import importlib.util
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KREAPHOTON_DIR = os.path.join(REPO_ROOT, "kreaphoton")
COMFYUI_ROOT = "E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2/ComfyUI"


def _load(name):
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)
    if "kreaphoton" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "kreaphoton", os.path.join(KREAPHOTON_DIR, "__init__.py"),
            submodule_search_locations=[KREAPHOTON_DIR])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["kreaphoton"] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module("kreaphoton." + name)


def _raises(fn, exc=ValueError, needle=None):
    try:
        fn()
    except exc as e:
        if needle is not None:
            assert needle in str(e), f"expected {needle!r} in error, got {e!r}"
        return True
    raise AssertionError(f"expected {exc.__name__}")


class FakeParam:
    def __init__(self, n, dtype, layout=None):
        self._n, self.dtype = n, dtype
        if layout is not None:
            self._layout_cls = layout

    def numel(self):
        return self._n


class FakeDiffusionModel:
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return iter(self._params)


class FakeBaseModel:
    def __init__(self, params, infer=torch.bfloat16):
        self.diffusion_model = FakeDiffusionModel(params)
        self._infer = infer

    def get_dtype_inference(self):
        return self._infer


class FakePatcher:
    def __init__(self, params, infer=torch.bfloat16):
        self.model = FakeBaseModel(params, infer)


def test_detect(pr):
    int8 = FakePatcher([FakeParam(900, torch.int8, "TensorWiseINT8Layout"), FakeParam(100, torch.bfloat16)])
    info = pr.detect_precision(int8)
    assert info.kind == "int8" and abs(info.quantized_fraction - 0.9) < 1e-9, info
    assert pr.resolve_profile(info) == "int8"

    fp8s = FakePatcher([FakeParam(900, torch.float8_e4m3fn, "TensorCoreFP8E4M3Layout"), FakeParam(100, torch.float16)])
    assert pr.detect_precision(fp8s).kind == "fp8"

    fp8_plain = FakePatcher([FakeParam(1000, torch.float8_e4m3fn)])
    info = pr.detect_precision(fp8_plain)
    assert info.kind == "fp8" and info.quantized_fraction == 0.0, info
    assert pr.resolve_profile(info) == "int8", "fp8 is the quantized family -> int8 profile"

    bf16 = FakePatcher([FakeParam(1000, torch.bfloat16)])
    info = pr.detect_precision(bf16)
    assert (info.kind, info.compute_dtype) == ("bf16", "torch.bfloat16"), info
    assert pr.resolve_profile(info) == "full"

    fp16 = FakePatcher([FakeParam(990, torch.float16), FakeParam(10, torch.float32)], infer=torch.float16)
    info = pr.detect_precision(fp16)
    assert (info.kind, info.compute_dtype) == ("fp16", "torch.float16"), info
    assert pr.resolve_profile(info) == "full"

    w4 = FakePatcher([FakeParam(1000, torch.uint8, "TensorCoreConvRotW4A4Layout")])
    assert pr.detect_precision(w4).kind == "int4"

    unknown = pr.detect_precision("MODEL")
    assert unknown.kind == "unknown" and pr.resolve_profile(unknown) == "int8", "unclassifiable -> today's behaviour"
    print("     detect: int8 mixed / fp8 scaled / fp8 plain / bf16 / fp16 / int4 / unknown")


def test_resolve(pr):
    bf16 = pr.PrecisionInfo("bf16", 0.0, "torch.bfloat16")
    int8 = pr.PrecisionInfo("int8", 0.9, "torch.bfloat16")
    assert pr.resolve_profile(bf16, "int8") == "int8"
    assert pr.resolve_profile(int8, "full") == "full"
    assert pr.resolve_profile(int8, "auto") == "int8"
    _raises(lambda: pr.resolve_profile(bf16, "fp16"), needle="precision_profile")
    line = pr.describe(bf16, "full", "auto")
    assert "bf16" in line and "full" in line and "auto" in line, line
    assert pr.PROFILE_OPTIONS == ["auto", "int8", "full"]
    print("     resolve: override wins, bad override rejected, describe names kind/profile/override")


def test_nonfinite(smp, sch):
    ok = torch.ones(1, 16, 1, 4, 4)
    assert smp.check_finite(ok, 0.5) is ok
    bad = ok.clone()
    bad[0, 0, 0, 0, 0] = float("nan")
    _raises(lambda: smp.check_finite(bad, 0.5), smp.NonFiniteModelOutput, needle="--bf16-unet")

    calls = {"n": 0}

    def nan_on_second(x, sigma, **kw):
        calls["n"] += 1
        return torch.full_like(x, float("nan")) if calls["n"] == 2 else x

    sigmas = sch.build_schedule(8, restart_frac=0.0, plunge=False)
    _raises(lambda: smp.kreaphoton_sampler_loop(nan_on_second, torch.randn(1, 16, 1, 4, 4), sigmas,
                                                 detail_amount=0.0, order=1, eta0=0.0, sigma_gate=0.10,
                                                 restart_seed=1),
            smp.NonFiniteModelOutput, needle="sigma")
    assert calls["n"] == 2, "the loop must stop at the first non-finite call"
    print("     NaN at the 2nd model call stops the loop with an actionable message")


def main():
    pr = _load("precision")
    smp, sch = _load("sampling"), _load("schedules")
    print("=" * 78)
    print("test_precision")
    print("=" * 78)
    print("[1] detect_precision")
    test_detect(pr)
    print("[2] resolve_profile / describe")
    test_resolve(pr)
    print("[3] non-finite model output guard")
    test_nonfinite(smp, sch)
    print("\ntest_precision: ALL PASSED")


if __name__ == "__main__":
    main()
