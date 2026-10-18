"""
Weight-precision detection for the loaded Krea 2 model (2026-10-01, full-precision profile).

Every calibrated constant in presets.py was measured on int8 ConvRot checkpoints. A
full-precision Turbo differs in two ways that matter to KreaPhoton: a LoRA patched on the
fly is NOT re-quantized to int8 (comfy/ops.py requantize_from_float - on int8 part of a
character LoRA's delta drowns in the quantization step), and fp16 weights run fp16 compute
(comfy.model_management.unet_dtype) with no overflow clamp in comfy/ldm/krea2.
detect_precision() classifies the diffusion model by the share of parameter ELEMENTS held by
quantized layouts (int8 files keep 5-37 large tensors in bf16/f16, so "any bf16 tensor" is
not a signal); resolve_profile() maps that, or the node's override, to a
presets.PRECISION_PROFILES key. No comfy import: QuantizedTensor is recognised by its
_layout_cls string (comfy/quant_ops.py register_layout_class names).
"""
import json
import os
from dataclasses import dataclass

PROFILE_OPTIONS = ["auto", "int8", "full"]
PRECISION_TOOLTIP = ("Which calibration to use for this checkpoint's weight precision. auto = detect: "
                     "int8 / fp8 checkpoints use the int8 calibration (bit-exact v1.8), bf16 / fp16 "
                     "use the full-precision profile. The detected format is printed in the log.")
PROFILE_PATCH_ENV = "KREAPHOTON_PROFILE_PATCH"   # calibration only: path to {"full": {...}}, re-read per call
FULL_KINDS = ("bf16", "fp16", "fp32")

_DTYPE_KIND = {
    "torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32",
    "torch.float8_e4m3fn": "fp8", "torch.float8_e5m2": "fp8", "torch.int8": "int8",
}


@dataclass(frozen=True)
class PrecisionInfo:
    kind: str                  # int8 | fp8 | int4 | quantized | bf16 | fp16 | fp32 | unknown
    quantized_fraction: float  # share of parameter elements held by quantized layouts
    compute_dtype: str         # str(torch dtype) the model computes in, "unknown" if not exposed


def _layout_of(p):
    for t in (p, getattr(p, "data", None)):
        layout = getattr(t, "_layout_cls", None) if t is not None else None
        if layout:
            return layout if isinstance(layout, str) else getattr(layout, "__name__", str(layout))
    return None


def _layout_kind(layout: str) -> str:
    name = layout.lower()
    if "w4a4" in name or "nvfp4" in name:
        return "int4"
    if "int8" in name:
        return "int8"
    if "fp8" in name:
        return "fp8"
    return "quantized"


def detect_precision(model) -> PrecisionInfo:
    base = getattr(model, "model", None)
    dm = getattr(base, "diffusion_model", None)
    if dm is None or not hasattr(dm, "parameters"):
        return PrecisionInfo("unknown", 0.0, "unknown")
    quantized, by_kind, by_dtype, total = 0, {}, {}, 0
    for p in dm.parameters():
        n = int(p.numel())
        total += n
        layout = _layout_of(p)
        if layout is not None:
            quantized += n
            k = _layout_kind(layout)
            by_kind[k] = by_kind.get(k, 0) + n
        else:
            d = str(p.dtype)
            by_dtype[d] = by_dtype.get(d, 0) + n
    if total == 0:
        return PrecisionInfo("unknown", 0.0, "unknown")
    fraction = quantized / total
    if fraction >= 0.5:
        kind = max(by_kind, key=by_kind.get)
    else:
        kind = _DTYPE_KIND.get(max(by_dtype, key=by_dtype.get), "unknown")
    infer = getattr(base, "get_dtype_inference", None)
    compute = str(infer()) if callable(infer) else "unknown"
    return PrecisionInfo(kind, fraction, compute)


def resolve_profile(info: PrecisionInfo, override: str = "auto") -> str:
    if override not in PROFILE_OPTIONS:
        raise ValueError(f"precision_profile must be one of {PROFILE_OPTIONS}, got {override!r}")
    if override != "auto":
        return override
    return "full" if info.kind in FULL_KINDS else "int8"


def describe(info: PrecisionInfo, profile: str, override: str) -> str:
    return "precision: %s (compute %s, quantized %.0f%%) -> profile %s (%s)" % (
        info.kind, info.compute_dtype, 100.0 * info.quantized_fraction, profile,
        "auto" if override == "auto" else "override")


def resolve_node_profile(model, override):
    """(profile name, profile dict) for a node run, logged. The dict is a copy: a calibration
    patch (KREAPHOTON_PROFILE_PATCH) never touches presets.PRECISION_PROFILES."""
    from .presets import PRECISION_PROFILES, validate_precision_profiles
    info = detect_precision(model)
    name = resolve_profile(info, override)
    profile = dict(PRECISION_PROFILES[name])
    patched = ""
    path = os.environ.get(PROFILE_PATCH_ENV)
    if path:
        with open(path, encoding="utf-8") as f:
            patch = json.load(f).get(name, {})
        if patch:
            profile.update(patch)
            validate_precision_profiles({name: profile})
            patched = " PATCHED by %s %s" % (PROFILE_PATCH_ENV, json.dumps(patch, sort_keys=True))
    print("[KreaPhoton] " + describe(info, name, override) + patched)
    return name, profile
