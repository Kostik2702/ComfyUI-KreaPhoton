# KreaPhoton Full-Precision Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** KreaPhoton runs correctly and is calibrated on full-precision Krea 2 Turbo checkpoints (bf16 / fp16), not only int8 ConvRot: character-LoRA likeness at least int8 + 0.03 ArcFace, and no LoRA-induced clothing mutations — while every int8 run stays bit-exact.

**Architecture:** A new `kreaphoton/precision.py` classifies the loaded diffusion model (int8 / fp8 / int4 / bf16 / fp16 / fp32) from its parameters and resolves a precision profile (`auto` → `int8` | `full`, manual override on the node). `presets.PRECISION_PROFILES` holds only RELATIVE corrections to the int8 calibration (LoRA identity scale, face-boost scale, detail scale) plus the composition/identity boundary; the `int8` profile is the identity, so int8 behaviour cannot change. The `full` values are produced by a measurement campaign (research rig: ArcFace likeness + clothing-mutation metric) on one merge in three precisions — `kreaNoctUncensoredRealism_v40` int8 / bf16 / fp16. Two guards ship with it: a non-finite model-output guard (fp16 overflow) and a LoRA trainer-flag guard (rsLoRA / DoRA scale mismatch).

**Tech Stack:** torch 2.9 / cu130, ComfyUI 0.37.0 (portable, `E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2`), comfy-aimdo dynamic VRAM, insightface 0.7.3 `buffalo_l` (CPU), Qwen3-VL-8B (transformers) as the clothing judge. Tests are plain-assert scripts run with the embedded python (`tests/run_tests.py`), no pytest.

**Spec:** this plan + vault `learnings/active/KREA2-NODES.md`, entries of 2026-10-01 (Step-Back, Investigation, `## Grill`, lora-flag finding). Owner decisions are quoted in "Owner decisions" below.

## Owner decisions (2026-10-01, asked in session)

- Target: full-precision Turbo (fp16/bf16). Target checkpoint: `diffusion_models/kreaNoctUncensoredRealism_v40_bf16.safetensors` (24.48 GiB, BF16, merge v3 0.7 + fineporn 0.3). Its int8 twin `kreaNoctUncensoredRealism_v40_int8.safetensors` is the A/B control; fp16 = local cast of the bf16 file (Task 2).
- Problem statement: not a crash — quality was never tested or optimised off int8; raise character-LoRA likeness, remove clothing mutations under character LoRAs.
- Profile selection: automatic detection + manual override `precision_profile = auto / int8 / full`; the detected format goes to the node log / report.
- Acceptance: clothing mutation = the garment differs from the same-seed frame WITHOUT the character LoRA (plus a Qwen3-VL check against the prompt); likeness = ArcFace cosine to the AnnaMin_v3 dataset centroid ≥ int8 + 0.03, on ≥ 8 seeds.
- The Krea2 memory leak is fixed FIRST (Task 1), before any A/B run.

## Global Constraints

- int8 bit-exactness: `PRECISION_PROFILES["int8"]` is `{"identity_lora_scale": 1.0, "face_boost_scale": 1.0, "detail_a_scale": 1.0, "composition_end": 0.85}` forever; a test pins it. Any int8 output change is a bug.
- Node inputs may only be APPENDED to `optional` (saved-workflow rule, `tests/test_nodes.py:37`). New input name everywhere: `precision_profile`, options `["auto", "int8", "full"]`, default `"auto"`.
- A model that cannot be classified (`"MODEL"` string in tests, missing `diffusion_model`) resolves to the `int8` profile — today's behaviour.
- No comfy import at module level in `precision.py` (pattern: `face_geometry.py`); `QuantizedTensor` is detected by duck typing on `_layout_cls` (a string such as `"TensorWiseINT8Layout"`, `comfy/quant_ops.py:201-209`).
- `presets.py` stays the single source of truth for every constant (planning-council R19); a profile value carries a `MEASURED <date>, N=<n>` comment or the word `UNCALIBRATED`.
- Every new test file is added to `TESTS` in `tests/run_tests.py`. Run: `"E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2/python_embeded/python.exe" tests/run_tests.py`.
- The repo has uncommitted owner WIP (`kreaphoton/think.py`, `tests/test_think.py`, `kreaphoton/nodes.py`, `README.md`, `docs/REFERENCE.md`, `tests/run_tests.py`). Never `git add -A`; stage only the files a task names, and for `nodes.py` / `run_tests.py` stage hunks (`git add -p`) or ask the owner to commit the WIP first.
- GPU work runs through the owner's ComfyUI on this desktop (RTX 5090 32 GB, 64 GB RAM, 20 logical cores). One generation queue at a time; CPU metric jobs (ArcFace on CPU) run sequentially with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8` (vault rule on parallel CPU load). ComfyUI restarts need the owner's go-ahead at that moment.
- Text with backslashes (LoRA names like `krea2\char\AnnaMin_v4_lora.safetensors`) is built from `chr(92)` in scripts, never typed into heredocs (vault rule, rig.py precedent).

## Review Focus

1. **fp16 compute overflow** — an fp16 checkpoint runs fp16 compute (`comfy/model_management.py:1166-1168`) and `comfy/ldm/krea2` has no clamp: a NaN must stop the run with an actionable message, never decode into a black/confetti frame. Pinned in Task 4 (loop test with a NaN-returning model).
2. **Mixed checkpoints** — int8 ConvRot files keep 5-37 large tensors in BF16 or F16 (`cineroKrea2Dark` has F16 ones): detection must classify by the quantized share of parameter elements, not by "any bf16 tensor". Pinned in Task 3 (90 % int8 + 10 % bf16 → `int8`).
3. **Plain fp8 without scales** (`selforaKrea2Realistic_v10`: F8_E4M3, no `weight_scale`) loads as `torch.float8_e4m3fn` parameters with no layout: must be `fp8` → `int8` profile, not `unknown`. Pinned in Task 3.
4. **A character LoRA wired through an explicit `clean_model` / `texture_model` that itself carries a plan** (`_explicit_phase_model`, nodes.py:189) must receive the same `identity_lora_scale` as `model` — otherwise the restart segment runs a weaker identity LoRA than the identity segment. Pinned in Task 6 (explicit-input test).
5. **LoRA trained with rsLoRA / DoRA flags set** — ComfyUI has no rsLoRA support (`grep rslora comfy/` is empty) and would apply such a LoRA √r times weaker, silently. Pinned in Task 5 (flag = 1 → `ValueError` naming the file).

---

## Phase map and budget

| Phase | Tasks | Kind | GPU time (est.) |
|---|---|---|---|
| A. Preconditions | 1 leak fix, 2 fp16 twin | research + core fix | ~1 h |
| B. Pack code | 3 precision, 4 NaN guard, 5 LoRA-flag guard, 6 profiles plumbing | TDD | 0 (live smoke ~10 min) |
| C. Measurement | 7 bench rig, 8 baseline A/B | research | ~2.5 h |
| D. Calibration | 9 boundary + identity scale, 10 face boost, 11 detail | research | ~3.5 h |
| E. Ship | 12 holdout + constants + docs + release, 13 vault | release | ~1 h |

Decision gates (owner): after Task 1 (leak root cause + fix approach), after Task 8 (baseline table — does `full` need calibration at all?), after Task 9 (boundary choice is a behaviour change), after Task 12 (release).

---

### Task 1: Krea2 memory leak — diagnose and fix (owner: fix first)

Context: `WARNING, memory leak with model Krea2` (`comfy/model_management.py:1072`) = the ModelPatcher chain died but `comfy.model_base.Krea2` survived `gc.collect()`; dead `LoadedModel` entries are never removed and accumulated to 41 in the 2026-09-30 session (69 689 warnings, two `gc.collect()` ≈ 0.6 s each per model load). Default cache in 0.37 is RAM_PRESSURE (`main.py:331`), which can evict even the `UNETLoader` output. A diagnostic probe is installed: `ComfyUI/custom_nodes/zz_kp_leak_probe/__init__.py` (walks `gc.get_referrers` from the leaked BaseModel to a root, logs `[LEAK-PROBE]`, max 3 reports).

**Files:**
- Create: `E:/Project/Krea2 Nodes/research/precision/leak_repro.py`
- Fix: decided by the probe output (gate below)
- Delete at the end: `ComfyUI/custom_nodes/zz_kp_leak_probe/`

- [ ] **Step 1: Capture the owner's graph in API form.** Run one generation of `[EDITORAL] KREA6 [BUILD TESTS SHORT]` from the UI, then save its API prompt:

```python
# research/precision/capture_api.py - run after one UI generation
import json, urllib.request
hist = json.load(urllib.request.urlopen("http://127.0.0.1:8188/history?max_items=1"))
pid, item = next(iter(hist.items()))
json.dump(item["prompt"][2], open("E:/Project/Krea2 Nodes/research/precision/owner_api.json", "w"), indent=1)
print("saved prompt", pid, "nodes:", len(item["prompt"][2]))
```

- [ ] **Step 2: Write the reproduction script** — it alternates the UNET file, changes a LoRA strength, and interrupts one run mid-sampling, i.e. the three events seen before the first leak in the 2026-09-30 log:

```python
# research/precision/leak_repro.py
import copy, json, time, urllib.request
URL = "http://127.0.0.1:8188"
LOG = "E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2/ComfyUI/user/comfyui.log"
BASE = json.load(open("E:/Project/Krea2 Nodes/research/precision/owner_api.json"))
UNETS = ["kreaNoctUncensoredRealism_v40_int8.safetensors", "kreaNoctUncensoredRealism_v40_bf16.safetensors"]

def post(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

def queue(prompt):
    return post("/prompt", {"prompt": prompt})["prompt_id"]

def wait(pid, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.load(urllib.request.urlopen(f"{URL}/history/{pid}"))
        if pid in h:
            return h[pid]["status"]["status_str"]
        time.sleep(2)
    raise TimeoutError(pid)

def variant(unet, lora_strength):
    p = copy.deepcopy(BASE)
    for node in p.values():
        if node["class_type"] == "UNETLoader":
            node["inputs"]["unet_name"] = unet
        if node["class_type"] == "KreaPhotonLoraPhase":
            node["inputs"]["strength"] = lora_strength
    return p

def leak_lines():
    return sum(1 for line in open(LOG, encoding="utf-8", errors="replace") if "memory leak with model" in line)

before = leak_lines()
plan = [(UNETS[0], 1.0, False), (UNETS[1], 1.0, False), (UNETS[1], 0.9, True), (UNETS[0], 0.9, False), (UNETS[1], 1.0, False), (UNETS[0], 1.0, False)]
for unet, s, interrupt in plan:
    pid = queue(variant(unet, s))
    if interrupt:
        time.sleep(25)
        post("/interrupt", {"prompt_id": pid})
    print(unet, s, "interrupt" if interrupt else "", wait(pid))
print("new leak warnings:", leak_lines() - before)
```

- [ ] **Step 3: Restart ComfyUI (owner go-ahead), run `leak_repro.py`.** Expected before the fix: `new leak warnings: > 0` and `[LEAK-PROBE] chain ...` lines in `comfyui.log`. If the count is 0 after all 6 runs, add 6 more runs with the cache under pressure (`--cache-ram 2` launch flag) before concluding the trigger is elsewhere.

- [ ] **Step 4: Gate — read the chains.** `grep -n "LEAK-PROBE" comfyui.log`. Classify the root: (a) a module global / class attribute in a custom node → fix in that node (if KreaPhoton: remove the global; if third party: report + local patch decision with the owner); (b) a live frame / traceback (interrupt path) → find who stores the exception; (c) `no-tracked-referrer` → C-level holder (aimdo / prefetch queue) → upstream ComfyUI issue + `--disable-comfy-compiler` A/B. Record the chain and the chosen fix in the KREA2-NODES learnings, ask the owner to approve the fix approach.

- [ ] **Step 5: Implement the approved fix at the source, rerun `leak_repro.py`.** Expected: `new leak warnings: 0`, and `[LEAK-PROBE]` lines absent for the new runs.

- [ ] **Step 6: Remove the probe** (`custom_nodes/zz_kp_leak_probe`), restart, rerun Step 5 once more (the probe must not be what made it pass). Commit only if the fix lives in this repo: `git commit -m "fix: <holder> kept the Krea2 BaseModel alive after its patchers died"`.

---

### Task 2: fp16 twin of the target checkpoint

**Files:**
- Create: `E:/Project/Krea2 Nodes/research/precision/make_fp16.py`
- Output: `ComfyUI/models/diffusion_models/kreaNoctUncensoredRealism_v40_fp16.safetensors` (≈24.5 GiB; E: has 398 GB free)

- [ ] **Step 1: Write the cast script** (streams tensor by tensor, reports range problems before writing):

```python
# research/precision/make_fp16.py
import json, math
import torch
from safetensors import safe_open
from safetensors.torch import save_file
D = "E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2/ComfyUI/models/diffusion_models/"
SRC, DST = D + "kreaNoctUncensoredRealism_v40_bf16.safetensors", D + "kreaNoctUncensoredRealism_v40_fp16.safetensors"
FP16_MAX, FP16_TINY = 65504.0, 6.103515625e-05
out, overflow, underflow, total = {}, 0, 0, 0
with safe_open(SRC, "pt") as f:
    meta = f.metadata() or {}
    for k in f.keys():
        t = f.get_tensor(k)
        if t.dtype == torch.bfloat16:
            a = t.float().abs()
            overflow += int((a > FP16_MAX).sum())
            underflow += int(((a > 0) & (a < FP16_TINY)).sum())
            total += t.numel()
            t = t.to(torch.float16)
        out[k] = t
print(f"bf16 elements {total}, > fp16 max: {overflow}, below fp16 normal range: {underflow} ({underflow / max(total, 1):.2e})")
assert overflow == 0, "weights exceed fp16 range - fp16 twin would be wrong"
meta["kreaphoton_note"] = "fp16 cast of kreaNoctUncensoredRealism_v40_bf16 for precision A/B (2026-10-01)"
save_file(out, DST, metadata=meta)
print("saved", DST)
```

- [ ] **Step 2: Run it** with the embedded python. Expected: `> fp16 max: 0`, a subnormal share printed (record it in the learnings — subnormal weights lose precision in fp16, a possible quality factor separate from activation overflow), file saved.
- [ ] **Step 3: Verify the header:** all 430 tensors present, large tensors `F16`, shapes equal to the bf16 file (reuse the header-compare snippet from the 2026-10-01 session: keys stripped of `model.diffusion_model.`).

---

### Task 3: `precision.py` — classify the loaded model

**Files:**
- Create: `kreaphoton/precision.py`
- Create: `tests/test_precision.py`
- Modify: `tests/run_tests.py` (append `"test_precision.py"` to `TESTS`)

**Interfaces:**
- Produces: `PROFILE_OPTIONS: list[str]`, `PrecisionInfo(kind: str, quantized_fraction: float, compute_dtype: str)`, `detect_precision(model) -> PrecisionInfo`, `resolve_profile(info: PrecisionInfo, override: str = "auto") -> str` (`"int8"` | `"full"`), `describe(info, profile, override) -> str`.

- [ ] **Step 1: Write the failing test**

```python
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


def main():
    pr = _load("precision")
    print("=" * 78)
    print("test_precision")
    print("=" * 78)
    print("[1] detect_precision")
    test_detect(pr)
    print("[2] resolve_profile / describe")
    test_resolve(pr)
    print("\ntest_precision: ALL PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run** `python tests/test_precision.py`. Expected: FAIL `ModuleNotFoundError: No module named 'kreaphoton.precision'`.

- [ ] **Step 3: Implement `kreaphoton/precision.py`**

```python
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
from dataclasses import dataclass

PROFILE_OPTIONS = ["auto", "int8", "full"]
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
```

- [ ] **Step 4: Run** `python tests/test_precision.py`. Expected: `test_precision: ALL PASSED`.
- [ ] **Step 5: Append `"test_precision.py"` to `TESTS` in `tests/run_tests.py`, run `tests/run_tests.py`.** Expected: `ALL 13 TEST FILES PASSED`.
- [ ] **Step 6: Commit** `git add kreaphoton/precision.py tests/test_precision.py` + the `run_tests.py` hunk; `git commit -m "feat(precision): classify the loaded Krea 2 model by weight precision"`.

---

### Task 4: Non-finite model-output guard

**Files:**
- Modify: `kreaphoton/sampling.py` (`call` inside `kreaphoton_sampler_loop` at line 472; `kreaphoton_invert_loop` model call at line 794-797)
- Test: `tests/test_precision.py`

**Interfaces:**
- Produces: `sampling.NonFiniteModelOutput(RuntimeError)`, `sampling.check_finite(den, sigma) -> den`.

- [ ] **Step 1: Add the failing test** to `tests/test_precision.py` (call it from `main()` as `[3]`):

```python
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
```

In `main()`: `smp, sch = _load("sampling"), _load("schedules")`.

- [ ] **Step 2: Run** — expected FAIL `AttributeError: module 'kreaphoton.sampling' has no attribute 'check_finite'`.

- [ ] **Step 3: Implement** in `kreaphoton/sampling.py` (module level, above `kreaphoton_sampler_loop`):

```python
class NonFiniteModelOutput(RuntimeError):
    """The diffusion model returned NaN/Inf. On Krea 2 the known cause is fp16 compute: fp16
    weights run fp16 (comfy.model_management.unet_dtype) and comfy/ldm/krea2 has no overflow
    clamp (2026-10-01). Stopping beats decoding a black / confetti frame."""


def check_finite(den, sigma):
    finite = torch.isfinite(den)
    if not bool(finite.all()):
        raise NonFiniteModelOutput(
            "KreaPhoton: the model returned %d non-finite values at sigma %.4f. On Krea 2 this is "
            "fp16 overflow - run the checkpoint in bf16 (start ComfyUI with --bf16-unet, or use a "
            "bf16 copy of the file)." % (int((~finite).sum()), float(sigma)))
    return den
```

In `call` (line 472) insert after the model/tiler evaluation, before `x0_hook`:

```python
        if tiler is not None:
            den = tiler(model, xx, sig_model, extra_args)
        else:
            den = model(xx, sig_model * s_in, **extra_args)
        den = check_finite(den, sig_true)
```

In `kreaphoton_invert_loop` after line 797: `denoised = check_finite(denoised, s_cur)`.

- [ ] **Step 4: Run** `tests/test_precision.py` and `tests/test_sampling.py`. Expected: both pass (finite mocks are untouched).
- [ ] **Step 5: Commit** `git commit -m "feat(sampling): stop on non-finite model output (fp16 overflow guard)"` (stage the two `sampling.py` hunks + test).

---

### Task 5: LoRA trainer-flag guard

Context (2026-10-01): blissful-tuner LoRAs (`networks.lora_krea2`) carry scalar tensors `use_dora_flag` / `use_rslora_flag`; ComfyUI logs `lora key not loaded` for each (`comfy/lora.py:93`). All 22 local files have both flags = 0 (rank 32 / alpha 32 → scale 1.0, identical to ComfyUI's alpha/r), so today the warning is harmless. ComfyUI has no rsLoRA: a file with `use_rslora_flag = 1` would be applied √r weaker without any error.

**Files:**
- Modify: `kreaphoton/lora_phase.py` (`_default_loader`, line 60)
- Test: `tests/test_precision.py`

**Interfaces:**
- Produces: `lora_phase.strip_trainer_flags(lora: dict, name: str) -> dict`.

- [ ] **Step 1: Add the failing test** (`[4]` in `main()`):

```python
def test_lora_flags(lp):
    w = torch.zeros(4, 4)
    clean = {"a.lora_down.weight": w, "a.use_dora_flag": torch.tensor(0.0), "a.use_rslora_flag": torch.tensor(0.0)}
    out = lp.strip_trainer_flags(dict(clean), "anna.safetensors")
    assert set(out) == {"a.lora_down.weight"}, out.keys()
    rs = dict(clean)
    rs["a.use_rslora_flag"] = torch.tensor(1.0)
    _raises(lambda: lp.strip_trainer_flags(rs, "rs.safetensors"), needle="rs.safetensors")
    dora = dict(clean)
    dora["a.use_dora_flag"] = torch.tensor(1.0)
    _raises(lambda: lp.strip_trainer_flags(dora, "dora.safetensors"), needle="DoRA")
    plain = {"a.lora_down.weight": w}
    assert lp.strip_trainer_flags(plain, "x") is plain, "no flags -> the same dict, no copy"
    print("     zero flags dropped silently, rsLoRA / DoRA flag = 1 rejected with the file name")
```

- [ ] **Step 2: Run** — expected FAIL `AttributeError: ... 'strip_trainer_flags'`.
- [ ] **Step 3: Implement** in `lora_phase.py`:

```python
_TRAINER_FLAGS = {"use_rslora_flag": "rsLoRA (alpha/sqrt(rank) scaling)", "use_dora_flag": "DoRA"}


def strip_trainer_flags(lora: dict, name: str) -> dict:
    """blissful-tuner stores two scalar flags in the state dict; ComfyUI maps only weights and
    warns 'lora key not loaded' for each. Flag 0 = a plain LoRA, scaled exactly as ComfyUI
    scales it (alpha / rank) -> drop the flags. Flag 1 = a scaling ComfyUI does not implement
    (no rsLoRA support at all; DoRA only via dora_scale keys) -> refuse, a silent sqrt(rank)
    weaker character LoRA is the worst failure for likeness (2026-10-01)."""
    flags = [k for k in lora if k.rsplit(".", 1)[-1] in _TRAINER_FLAGS]
    if not flags:
        return lora
    for k in flags:
        if float(lora[k]) != 0.0:
            raise ValueError(f"KreaPhoton LoRA Phase: {name} was trained with "
                             f"{_TRAINER_FLAGS[k.rsplit('.', 1)[-1]]}, which ComfyUI does not apply - "
                             f"the LoRA would act with the wrong strength")
    return {k: v for k, v in lora.items() if k not in flags}
```

In `_default_loader`, after `lora = comfy.utils.load_torch_file(path, safe_load=True)`: `lora = strip_trainer_flags(lora, lora_name)` (before caching).

- [ ] **Step 4: Run** `tests/test_precision.py` + `tests/run_tests.py`. Expected: all pass.
- [ ] **Step 5: Live check** (owner restart): one generation of the owner graph → `comfyui.log` has no `use_dora_flag` / `use_rslora_flag` lines for KreaPhoton LoRA Phase loads.
- [ ] **Step 6: Commit** `git commit -m "fix(lora): drop zero trainer flags, refuse rsLoRA/DoRA LoRAs ComfyUI cannot scale"`.

---

### Task 6: Precision profiles and plumbing (int8 bit-exact)

**Files:**
- Modify: `kreaphoton/presets.py` (new block after `LORA_PHASES`, validator call at the bottom)
- Modify: `kreaphoton/lora_phase.py` (`build_phase_models` gains `identity_scale`)
- Modify: `kreaphoton/nodes.py` (`_explicit_phase_model`, `KreaPhotonSampler`, `KreaPhotonSamplerAdvanced`)
- Modify: `kreaphoton/face_detailer.py` (`KreaPhotonFaceDetailer`)
- Test: `tests/test_precision.py`, `tests/test_nodes.py` (Advanced optional-tail list at line 346)

**Interfaces:**
- Consumes: Task 3 `detect_precision`, `resolve_profile`, `describe`, `PROFILE_OPTIONS`.
- Produces: `presets.PRECISION_PROFILE_KEYS`, `presets.PRECISION_PROFILES`, `presets.validate_precision_profiles(profiles=None) -> True`, `lora_phase.build_phase_models(model, loader=..., apply_lora=..., identity_scale=1.0)`, `nodes._explicit_phase_model(explicit, plan_model, phase, identity_scale=1.0)`, `precision.resolve_node_profile(model, override) -> (name: str, profile: dict)` (lives in `precision.py` because `face_detailer.py` does not import `nodes`), env var `KREAPHOTON_PROFILE_PATCH` = path to a JSON file `{"full": {"composition_end": 0.8}}` re-read on every call (calibration only, same precedent as `KREAPHOTON_TRACE`).

- [ ] **Step 1: Write the failing tests** (`[5]`, `[6]` in `test_precision.py`):

```python
NEUTRAL = {"identity_lora_scale": 1.0, "face_boost_scale": 1.0, "detail_a_scale": 1.0, "composition_end": 0.85}


def test_profiles(presets):
    assert presets.validate_precision_profiles() is True
    assert presets.PRECISION_PROFILES["int8"] == NEUTRAL, "int8 must stay the identity (bit-exact int8 runs)"
    assert set(presets.PRECISION_PROFILES) == {"int8", "full"}
    bad = {"int8": dict(NEUTRAL), "full": dict(NEUTRAL, composition_end=1.2)}
    _raises(lambda: presets.validate_precision_profiles(bad), needle="composition_end")
    bad = {"int8": dict(NEUTRAL), "full": {"identity_lora_scale": 1.0}}
    _raises(lambda: presets.validate_precision_profiles(bad), needle="missing")
    print("     int8 profile is the identity; validator rejects bad ranges and missing keys")


class PlanPatcher:
    def __init__(self, opts=None, applied=None):
        self.model_options = dict(opts or {})
        self.applied = list(applied or [])

    def clone(self):
        return PlanPatcher(self.model_options, self.applied)


def test_identity_scale(lp):
    plan = [{"lora_name": "anna", "strength": 1.0, "phase": "identity"},
            {"lora_name": "style", "strength": 0.5, "phase": "texture"}]
    model = PlanPatcher({lp.PLAN_KEY: plan})

    def apply(p, lora, strength):
        p.applied.append((lora, strength))
        return p

    clean, ident, tex = lp.build_phase_models(model, loader=lambda n: n, apply_lora=apply, identity_scale=1.25)
    assert ident.applied == [("anna", 1.25)], ident.applied
    assert tex.applied == [("anna", 1.25), ("style", 0.5)], tex.applied
    assert clean.applied == [], "composition segment carries no LoRA here"
    _, ident1, tex1 = lp.build_phase_models(model, loader=lambda n: n, apply_lora=apply)
    assert ident1.applied == [("anna", 1.0)] and tex1.applied == [("anna", 1.0), ("style", 0.5)]
    print("     identity_scale multiplies identity-phase entries only; default 1.0 unchanged")


def test_profile_patch(pr, tmp_dir):
    bf16 = pr.PrecisionInfo("bf16", 0.0, "torch.bfloat16")
    real_detect = pr.detect_precision
    pr.detect_precision = lambda m: bf16
    path = os.path.join(tmp_dir, "profile_patch.json")
    try:
        os.environ.pop("KREAPHOTON_PROFILE_PATCH", None)
        name, prof = pr.resolve_node_profile("MODEL", "auto")
        assert name == "full" and prof["composition_end"] == 0.85
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"full": {"composition_end": 0.8, "identity_lora_scale": 1.15}}')
        os.environ["KREAPHOTON_PROFILE_PATCH"] = path
        name, prof = pr.resolve_node_profile("MODEL", "auto")
        assert prof["composition_end"] == 0.8 and prof["identity_lora_scale"] == 1.15, prof
        name, prof = pr.resolve_node_profile("MODEL", "int8")
        assert prof["composition_end"] == 0.85, "a patch for 'full' never touches 'int8'"
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"full": {"composition_end": 1.4}}')
        _raises(lambda: pr.resolve_node_profile("MODEL", "auto"), needle="composition_end")
    finally:
        os.environ.pop("KREAPHOTON_PROFILE_PATCH", None)
        pr.detect_precision = real_detect
    print("     KREAPHOTON_PROFILE_PATCH: re-read per call, validated, scoped to its profile")
```

In `main()` call it as `[7]` with `tmp_dir = tempfile.mkdtemp()` (`import tempfile` at the top).

Add to `tests/test_nodes.py` inside the existing `fake_run_sampling` block (after line 186), the plumbing test:

```python
        # precision profile plumbing (2026-10-01): int8 / unknown model -> neutral profile,
        # full -> profile values reach run_sampling
        import kreaphoton.precision as precision_mod
        real_detect = precision_mod.detect_precision
        saved_full = dict(presets.PRECISION_PROFILES["full"])
        try:
            node.sample("MODEL", pos, latent, 1001, "turbo/balanced", "off", preview_method="none")
            assert captured["composition_end"] == 0.85
            assert captured["detail_amount"] == presets.PRESETS["turbo/balanced"]["detail_a"]
            presets.PRECISION_PROFILES["full"].update(detail_a_scale=0.5, composition_end=0.80)
            precision_mod.detect_precision = lambda m: precision_mod.PrecisionInfo("bf16", 0.0, "torch.bfloat16")
            node.sample("MODEL", pos, latent, 1001, "turbo/balanced", "off", preview_method="none")
            assert captured["composition_end"] == 0.80
            assert captured["detail_amount"] == presets.PRESETS["turbo/balanced"]["detail_a"] * 0.5
            node.sample("MODEL", pos, latent, 1001, "turbo/balanced", "off", preview_method="none",
                        precision_profile="int8")
            assert captured["composition_end"] == 0.85, "override int8 wins over detected bf16"
        finally:
            precision_mod.detect_precision = real_detect
            presets.PRECISION_PROFILES["full"].clear()
            presets.PRECISION_PROFILES["full"].update(saved_full)
        print("     precision profile: unknown -> int8 neutral, bf16 -> full values, override wins")
```

Add to `tests/test_nodes.py` (Review Focus 4 — an explicit `clean_model` / `texture_model` that carries a plan gets the same identity scale as `model`):

```python
    # explicit phase input expanded with the profile's identity scale (2026-10-01)
    seen = []
    real_build = nodes.build_phase_models

    def spy_build(m, identity_scale=1.0, **kw):
        seen.append(identity_scale)
        return None, m, None

    nodes.build_phase_models = spy_build
    try:
        assert nodes._explicit_phase_model("EXPLICIT", "PLAN", "texture", 1.25) == "EXPLICIT"
        assert nodes._explicit_phase_model(None, "PLAN", "texture", 1.25) == "PLAN"
        assert seen == [1.25], seen
    finally:
        nodes.build_phase_models = real_build
    print("     explicit clean/texture input expanded with the profile identity scale")
```

Update the Advanced optional-tail assertion at `tests/test_nodes.py:346`: append `"precision_profile"` to the expected list. Add after line 216: `assert list(s_it["optional"].keys())[-1] == "precision_profile"`.

- [ ] **Step 2: Run** both test files — expected FAIL (`validate_precision_profiles` missing; `precision_profile` unexpected keyword).

- [ ] **Step 3: Implement `presets.py`** (after `LORA_PHASES`):

```python
# --- Precision profiles (2026-10-01, KREAPHOTON-FP16). Every value above was measured on int8
#     ConvRot checkpoints. On int8 a LoRA is patched on the fly and the patched weight is
#     RE-QUANTIZED to int8 (comfy/ops.py requantize_from_float), so part of a character LoRA's
#     delta drowns in the quantization step; the Face Detailer's measured identity_boost 1.5
#     (2026-09-16) compensates that on int8. A full-precision Turbo applies the delta exactly.
#     A profile holds only RELATIVE corrections to the int8 calibration (scales) plus the
#     composition -> identity boundary, so "int8" is the identity and int8 runs stay bit-exact.
#     identity_lora_scale: x strength of identity-phase plan entries (Sampler phase models);
#     face_boost_scale: x the Face Detailer's identity_boost; detail_a_scale: x the preset's
#     detail_a (simple Sampler only); composition_end: the clean -> identity split sigma.
#     "full" is UNCALIBRATED (a copy of int8) until docs/superpowers/plans/
#     2026-10-01-full-precision-profile.md Tasks 9-12 measure it. ---
PRECISION_PROFILE_KEYS = ("identity_lora_scale", "face_boost_scale", "detail_a_scale", "composition_end")
PRECISION_PROFILES = {
    "int8": {"identity_lora_scale": 1.0, "face_boost_scale": 1.0, "detail_a_scale": 1.0, "composition_end": 0.85},
    "full": {"identity_lora_scale": 1.0, "face_boost_scale": 1.0, "detail_a_scale": 1.0, "composition_end": 0.85},
}


def validate_precision_profiles(profiles: dict = None):
    profiles = PRECISION_PROFILES if profiles is None else profiles
    for name, p in profiles.items():
        missing = [k for k in PRECISION_PROFILE_KEYS if k not in p]
        if missing:
            raise ValueError(f"precision profile {name!r} is missing keys {missing}")
        for k in ("identity_lora_scale", "face_boost_scale", "detail_a_scale"):
            if not (0.0 < float(p[k]) <= 3.0):
                raise ValueError(f"precision profile {name!r}: {k} must be in (0, 3], got {p[k]!r}")
        if not (0.5 <= float(p["composition_end"]) <= 0.95):
            raise ValueError(f"precision profile {name!r}: composition_end must be in [0.5, 0.95], "
                             f"got {p['composition_end']!r}")
    return True
```

Bottom of file: `validate_precision_profiles()` next to the other validators.

- [ ] **Step 4: Implement `lora_phase.build_phase_models`** — signature `def build_phase_models(model, loader=_default_loader, apply_lora=_default_apply, identity_scale: float = 1.0):`, and inside `patcher_for`:

```python
        for i in key:
            entry = plan[i]
            strength = float(entry["strength"])
            if entry["phase"] in IDENTITY_PHASES:
                strength *= float(identity_scale)
            p = apply_lora(p, loader(entry["lora_name"]), strength)
```

(`IDENTITY_PHASES` is defined later in the module; it is read at call time, so no move is needed.) Docstring: one sentence on `identity_scale`.

- [ ] **Step 5: Implement `resolve_node_profile` in `precision.py`** (append; `presets` is comfy-free, so the module stays importable in tests). `detect_precision` is looked up through the module global at call time, so tests can replace it:

```python
import json
import os

PRECISION_TOOLTIP = ("Which calibration to use for this checkpoint's weight precision. auto = detect: "
                     "int8 / fp8 checkpoints use the int8 calibration (bit-exact v1.8), bf16 / fp16 "
                     "use the full-precision profile. The detected format is printed in the log.")
PROFILE_PATCH_ENV = "KREAPHOTON_PROFILE_PATCH"   # calibration only: path to {"full": {...}}, re-read per call


def resolve_node_profile(model, override):
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
```

Then the node plumbing in `nodes.py`: `from .precision import PRECISION_TOOLTIP, PROFILE_OPTIONS, resolve_node_profile`.

`_explicit_phase_model(explicit, plan_model, phase, identity_scale=1.0)` passes `identity_scale=identity_scale` to its `build_phase_models(explicit, ...)` call.

`KreaPhotonSampler.INPUT_TYPES` — append as the LAST optional input:
`"precision_profile": (PROFILE_OPTIONS, {"default": "auto", "tooltip": PRECISION_TOOLTIP}),`
`sample(..., coherence="off", precision_profile="auto")`; at the top of `sample`, before `build_phase_models`:

```python
        _, prof = resolve_node_profile(model, precision_profile)
        plan_clean, model, plan_texture = build_phase_models(model, identity_scale=prof["identity_lora_scale"])
        clean_model = _explicit_phase_model(clean_model, plan_clean, "composition", prof["identity_lora_scale"])
        texture_model = _explicit_phase_model(texture_model, plan_texture, "texture", prof["identity_lora_scale"])
```

and in the `run_sampling` call: `detail_amount=p["detail_a"] * prof["detail_a_scale"]`, `composition_end=prof["composition_end"]`.

`KreaPhotonSamplerAdvanced` — same input appended last; resolve the profile the same way; apply ONLY `identity_lora_scale` to its `build_phase_models` / `_explicit_phase_model` calls (the Advanced node's numeric widgets, including `composition_end`, run as typed).

- [ ] **Step 6: Implement the Face Detailer** — `from .precision import PRECISION_TOOLTIP, PROFILE_OPTIONS, resolve_node_profile` next to the other relative imports (face_detailer.py:53-60); optional input `"precision_profile": (PROFILE_OPTIONS, {"default": "auto", "tooltip": PRECISION_TOOLTIP})` appended last; `detail(..., precision_profile="auto")`; in `detail()` before line 217: `name, prof = resolve_node_profile(model, precision_profile)`, then `boost = float(identity_boost) * prof["face_boost_scale"]` (replaces `boost = float(identity_boost)`), and right after the gate lines a report line `report.append("precision profile %s: identity boost x%.2f -> x%.2f" % (name, float(identity_boost), boost))`.

- [ ] **Step 7: Run** `tests/run_tests.py`. Expected: all 13 files pass; every pre-existing assertion unchanged (int8 / unknown models get the neutral profile).

- [ ] **Step 8: Live smoke (owner restart).** Run the owner graph three times: `UNETLoader` = `..._v40_int8`, `..._v40_bf16`, `selforaKrea2Realistic_v10` (plain fp8, in `checkpoints/`, load via the same loader the owner uses for checkpoints). Expected log lines: `precision: int8 ... -> profile int8 (auto)`, `precision: bf16 (compute torch.bfloat16 ...) -> profile full (auto)`, `precision: fp8 ... -> profile int8 (auto)`. The int8 frame must be pixel-identical to a run from the commit before Task 6 (same seed): compare with `numpy.array_equal` on the two PNGs.

- [ ] **Step 9: Commit** `git commit -m "feat(precision): precision_profile (auto/int8/full) on Sampler, Advanced and Face Detailer; int8 bit-exact"`.

---

### Task 7: Measurement rig — likeness and clothing mutation

**Files:**
- Create: `E:/Project/Krea2 Nodes/research/precision/bench.py` (generation driver)
- Create: `E:/Project/Krea2 Nodes/research/precision/score.py` (metrics)
- Create: `E:/Project/Krea2 Nodes/research/precision/prompts.json`
- Reuse: `research/face_identity_rig/idsim.py` (ArcFace, centroid cache `annamin_ds_emb.npy`), `research/lora_stack/score.py` (DeepFashion2 garment segmentation), `research/precision/owner_api.json` (Task 1)

**Metric definitions (owner decision 2026-10-01):**
- `likeness` = cosine of the largest face's ArcFace embedding (insightface `buffalo_l`) to the L2 centroid of the AnnaMin_v3 training set (120 faces). Seed noise ±0.02 (2026-09-16): compare means over ≥ 8 seeds.
- `mutation` per (prompt, seed): the frame WITH the character LoRA vs the same-seed frame WITHOUT it (LoRA Phase strength 0; composition identical by construction). Mutation = garment-mask IoU < `IOU_MIN` OR the Qwen3-VL judge answers "no" to any garment question derived from the prompt. `IOU_MIN` is set in Task 8 from the precision noise floor (IoU between the int8 and bf16 no-LoRA frames of the same seed, 5th percentile), so a mutation means "more change than switching precision causes".

- [ ] **Step 1: Prompt set** `prompts.json` — 6 prompts, each with explicit garments and the questions the judge asks:

```json
{
 "front":  {"text": "AnnaMin, full body photo of a young woman standing on a city street, white linen shirt, blue denim shorts, brown leather sandals, overcast daylight", "garments": ["white linen shirt", "blue denim shorts", "brown leather sandals"]},
 "cafe":   {"text": "AnnaMin, a young woman sitting at a cafe table, black leather skirt, cream knitted sweater, ankle boots, warm window light", "garments": ["black leather skirt", "cream knitted sweater", "ankle boots"]},
 "back34": {"text": "AnnaMin, three-quarter back view of a young woman on a beach boardwalk, red bikini top, light blue denim shorts, straw hat", "garments": ["red bikini top", "light blue denim shorts", "straw hat"]},
 "office": {"text": "AnnaMin, a young woman in an office, grey tailored blazer, white blouse, black trousers, soft fluorescent light", "garments": ["grey tailored blazer", "white blouse", "black trousers"]},
 "night":  {"text": "AnnaMin, a young woman on a neon-lit street at night, green satin slip dress, black leather jacket", "garments": ["green satin slip dress", "black leather jacket"]},
 "sport":  {"text": "AnnaMin, a young woman jogging in a park, pink sports bra, black leggings, white running shoes, morning sun", "garments": ["pink sports bra", "black leggings", "white running shoes"]}
}
```

- [ ] **Step 2: Map the owner graph.** Print what `bench.py` must touch:

```python
# research/precision/inspect_api.py
import json
p = json.load(open("E:/Project/Krea2 Nodes/research/precision/owner_api.json"))
for nid, n in sorted(p.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
    if n["class_type"] in ("UNETLoader", "KreaPhotonLoraPhase", "KreaPhotonSampler", "KreaPhotonFaceDetailer",
                           "VAEDecode", "SaveImage", "PreviewImage", "KreaPhotonPromptBuilder", "KreaPhotonEncode",
                           "easy positive", "StringConcatenate", "CLIPTextEncode"):
        flat = {k: v for k, v in n["inputs"].items() if not isinstance(v, list)}
        print(nid, n["class_type"], json.dumps(flat, ensure_ascii=False)[:200])
```

Write the ids it prints into the `GRAPH` dict at the top of `bench.py` (Step 3): the `UNETLoader`, the LoRA Phase node whose `lora_name` contains `AnnaMin`, the KreaPhoton Sampler, the Face Detailer, the `VAEDecode` fed by the Sampler, and the node + input name that carries the positive text (the first text-bearing node upstream of the Encode node).

- [ ] **Step 3: `bench.py`** — one call = one block on one UNET; output nodes of the owner graph are dropped and one `SaveImage` is attached to the stage under test, so DLSS / SeedVR2 / upscale (not ancestors of it) never execute:

```python
# research/precision/bench.py
# usage: python bench.py <run_name> <unet_file> <stage: sampler|face> <lora: on|off|both> [seeds=1001-1008] [prompts=all]
import copy, json, os, sys, time, urllib.request
SP = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8188"
COMFY = "E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2/ComfyUI"
BASE = json.load(open(os.path.join(SP, "owner_api.json")))
PROMPTS = json.load(open(os.path.join(SP, "prompts.json")))
GRAPH = {  # filled from inspect_api.py (Step 2)
    "unet": "<id>", "lora_char": "<id>", "sampler": "<id>", "face": "<id>", "decode": "<id>",
    "text_node": "<id>", "text_input": "<input name>",
}
OUTPUT_CLASSES = {"SaveImage", "PreviewImage", "Image Save", "SaveImageExtended"}

def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

def run(prompt):
    r = post("/prompt", {"prompt": prompt})
    if r.get("node_errors"):
        raise SystemExit(json.dumps(r, indent=1)[:3000])
    pid, t0 = r["prompt_id"], time.time()
    while True:
        time.sleep(2)
        h = json.load(urllib.request.urlopen(f"{URL}/history/{pid}"))
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            return h[pid], time.time() - t0

def variant(unet, prompt_key, seed, lora_on, stage, tag):
    p = copy.deepcopy(BASE)
    for nid in [k for k, n in p.items() if n["class_type"] in OUTPUT_CLASSES]:
        del p[nid]
    p[GRAPH["unet"]]["inputs"]["unet_name"] = unet
    p[GRAPH["lora_char"]]["inputs"]["strength"] = p[GRAPH["lora_char"]]["inputs"]["strength"] if lora_on else 0.0
    p[GRAPH["text_node"]]["inputs"][GRAPH["text_input"]] = PROMPTS[prompt_key]["text"]
    p[GRAPH["sampler"]]["inputs"]["seed"] = seed
    src = [GRAPH["decode"], 0] if stage == "sampler" else [GRAPH["face"], 0]
    if stage == "face":
        p[GRAPH["face"]]["inputs"]["seed"] = seed
    p["bench_save"] = {"class_type": "SaveImage", "inputs": {"images": src, "filename_prefix": "precision/" + tag}}
    return p

def main():
    run_name, unet, stage, lora = sys.argv[1:5]
    seeds = range(1001, 1009)
    keys = list(PROMPTS)
    for a in sys.argv[5:]:
        if a.startswith("seeds="):
            lo, hi = a[6:].split("-"); seeds = range(int(lo), int(hi) + 1)
        if a.startswith("prompts="):
            keys = a[8:].split(",")
    modes = {"on": [True], "off": [False], "both": [True, False]}[lora]
    rows = open(os.path.join(SP, "runs.jsonl"), "a", encoding="utf-8")
    for key in keys:
        for seed in seeds:
            for on in modes:
                tag = f"{run_name}_{key}_{'lora' if on else 'nolora'}_s{seed}"
                item, sec = run(variant(unet, key, seed, on, stage, tag))
                ok = item["status"]["status_str"] == "success"
                imgs = item["outputs"].get("bench_save", {}).get("images", [])
                path = os.path.join(COMFY, "output", imgs[0]["subfolder"], imgs[0]["filename"]) if (ok and imgs) else None
                msgs = json.dumps(item["status"].get("messages", []))
                rows.write(json.dumps(dict(run=run_name, unet=unet, stage=stage, prompt=key, seed=seed, lora=on,
                                           path=path, seconds=round(sec, 1), ok=ok,
                                           nonfinite="NonFiniteModelOutput" in msgs)) + "\n")
                rows.flush()
                print(tag, "ok" if ok else "FAILED", round(sec, 1))

if __name__ == "__main__":
    main()
```

Profile values for the sweeps (Tasks 9-11) come from `KREAPHOTON_PROFILE_PATCH` (Task 6): start ComfyUI for the campaign from a separate `run_calibration.bat` (a copy of the owner's `run_nvidia_gpu.bat` with `set KREAPHOTON_PROFILE_PATCH=E:\Project\Krea2 Nodes\research\precision\profile_patch.json` before the python line — the owner's bat is not edited) and rewrite `profile_patch.json` between blocks; the log line `PATCHED by ...` confirms each block.

- [ ] **Step 4: `score.py`** — runs AFTER a generation block, with ComfyUI models freed first (`post("/free", {"unload_models": True, "free_memory": True})`); CPU parts with `OMP_NUM_THREADS=8`:

```python
# research/precision/score.py
# usage: python score.py <run_name> [<run_name> ...]   -> scores.csv rows appended + summary printed
import csv, json, os, statistics, sys
import cv2, numpy as np, torch
from ultralytics import YOLO
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SP), "face_identity_rig"))
import idsim  # noqa: E402  insightface buffalo_l + AnnaMin_v3 centroid (idsim.cent)
PROMPTS = json.load(open(os.path.join(SP, "prompts.json")))
SEG = YOLO("E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2/ComfyUI/models/ultralytics/segm/deepfashion2_yolov8s-seg.pt")
JUDGE_DIR = "E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2/ComfyUI/models/LLM/Qwen-VL/Huihui-Qwen3-VL-8B-Instruct-abliterated"
IOU_MIN = float(os.environ.get("IOU_MIN", "0"))   # set from the Task 8 noise floor; 0 = report IoU only

def likeness(img):
    fs = idsim.app.get(img)
    if not fs:
        return None
    f = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    e = f.normed_embedding / np.linalg.norm(f.normed_embedding)
    return float(idsim.cent @ e)

def garment_mask(img):
    r = SEG.predict(img, conf=0.35, verbose=False, retina_masks=True)[0]
    if r.masks is None:
        return np.zeros(img.shape[:2], bool), set()
    m = r.masks.data.cpu().numpy().astype(bool).any(axis=0)
    return m, {r.names[int(k)] for k in r.boxes.cls.cpu().tolist()}

def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)

class Judge:
    def __init__(self):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.proc = AutoProcessor.from_pretrained(JUDGE_DIR)
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(JUDGE_DIR, dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.inference_mode()
    def wearing(self, path, garment):
        msgs = [{"role": "user", "content": [{"type": "image", "image": path},
                 {"type": "text", "text": f"Is the person wearing {garment}? Answer yes or no."}]}]
        inp = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt").to(self.vlm.device)
        out = self.vlm.generate(**inp, max_new_tokens=3, do_sample=False)
        ans = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return ans.strip().lower().startswith("yes")

def main():
    runs = set(sys.argv[1:])
    rows = [json.loads(l) for l in open(os.path.join(SP, "runs.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["run"] in runs]
    by_key = {(r["run"], r["prompt"], r["seed"], r["lora"]): r for r in rows}
    judge = Judge()
    out = []
    for r in rows:
        if not r["lora"] or not r["ok"]:
            continue
        ref = by_key.get((r["run"], r["prompt"], r["seed"], False))
        img = cv2.imread(r["path"])
        rec = dict(run=r["run"], unet=r["unet"], prompt=r["prompt"], seed=r["seed"], likeness=likeness(img),
                   nonfinite=r["nonfinite"])
        m, classes = garment_mask(img)
        if ref and ref["ok"]:
            m0, classes0 = garment_mask(cv2.imread(ref["path"]))
            rec["garment_iou"] = iou(m, m0)
            rec["class_change"] = sorted(classes ^ classes0)
        misses = [g for g in PROMPTS[r["prompt"]]["garments"] if not judge.wearing(r["path"], g)]
        rec["judge_missing"] = misses
        rec["mutation"] = bool(misses) or (IOU_MIN > 0 and rec.get("garment_iou", 1.0) < IOU_MIN)
        out.append(rec)
    path = os.path.join(SP, "scores.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        if new:
            w.writeheader()
        w.writerows(out)
    for run in sorted(runs):
        sub = [o for o in out if o["run"] == run]
        lk = [o["likeness"] for o in sub if o["likeness"] is not None]
        print(f"{run}: n={len(sub)} likeness {statistics.mean(lk):.3f} +- {statistics.pstdev(lk):.3f} "
              f"mutation {sum(o['mutation'] for o in sub)}/{len(sub)} "
              f"iou_p5 {np.percentile([o.get('garment_iou', 1.0) for o in sub], 5):.3f} "
              f"nonfinite {sum(o['nonfinite'] for o in sub)}")

if __name__ == "__main__":
    main()
```

The no-LoRA noise floor for `IOU_MIN` (Task 8) needs IoU between two no-LoRA frames of different UNETs at the same seed: `python -c` over `runs.jsonl` pairs `(int8 run, bf16 run, prompt, seed, lora=False)` with `garment_mask` + `iou` imported from `score.py`, 5th percentile.
- [ ] **Step 5: Rig self-check** — `python bench.py selfcheck_a kreaNoctUncensoredRealism_v40_int8.safetensors sampler both seeds=1001-1002 prompts=front,cafe`, the same again as `selfcheck_b`, then `python score.py selfcheck_a selfcheck_b`: likeness of LoRA frames > no-LoRA frames by ≥ 0.15 (the LoRA works), IoU between the `_a` and `_b` no-LoRA frames = 1.0 (determinism), the judge says "yes" for every prompted garment on at least 3 of 4 no-LoRA frames (judge sanity). A failure here stops the campaign.

---

### Task 8: Baseline A/B — same merge, three precisions, current constants

- [ ] **Step 1: Generate** 6 prompts × 8 seeds × {LoRA 1.0, LoRA 0} × {int8, bf16, fp16} = 288 frames, preset `turbo/balanced`, `precision_profile` = `auto` (full = copy of int8 → identical constants). Blocks by unet; ComfyUI restart between blocks is NOT needed after Task 1 — record the leak counter after each block anyway.
- [ ] **Step 2: Score**, compute `IOU_MIN` (5th percentile of int8-vs-bf16 no-LoRA IoU), recompute mutation with it.
- [ ] **Step 3: Gate (owner).** Present the table. Decision rules: if fp16 produced non-finite runs → fp16 is documented as "use --bf16-unet" (guard from Task 4 already stops it) and calibration targets bf16 only; if bf16 likeness ≥ int8 + 0.03 AND mutation ≤ int8 already → Tasks 9-11 shrink to a confirmation pass; otherwise run them in full.
- [ ] **Step 4:** Append the table and decisions to the KREA2-NODES learnings.

---

### Task 9: Calibrate `composition_end` and `identity_lora_scale` (full profile)

Mechanism: clothing is decided in the identity segment (presets.LORA_PHASES comment: "identity 0.85 -> 0: face, body, clothing decided"). Moving the clean → identity boundary lower keeps the clean model in charge of garments longer (fewer LoRA-induced garment changes) at a likely likeness cost; `identity_lora_scale` buys likeness back. Owner workflow places the character LoRA in the `identity` phase.

- [ ] **Step 1: Boundary sweep** at `identity_lora_scale` 1.0: `composition_end` ∈ {0.85, 0.80, 0.75, 0.70} × 4 prompts (front, cafe, office, sport) × 8 seeds, bf16, LoRA 1.0 (+ the no-LoRA frames from Task 8 as references) = 128 frames (~1 h). Each block: write `research/precision/profile_patch.json` = `{"full": {"composition_end": <value>}}`, run `python bench.py b9_ce<value> kreaNoctUncensoredRealism_v40_bf16.safetensors sampler on prompts=front,cafe,office,sport`, confirm the `PATCHED by KREAPHOTON_PROFILE_PATCH` log line, then `python score.py b9_ce<value>`.
- [ ] **Step 2: Scale sweep** at the best boundary (lowest mutation rate whose likeness ≥ int8 baseline): `identity_lora_scale` ∈ {0.9, 1.0, 1.15, 1.3} × same 4 × 8 = 128 frames. Also visual check of the 1:1 face and skin crops for over-cooking (lora_stack `crops.py`) — a scale that gains ArcFace but adds plastic skin / artefacts is rejected.
- [ ] **Step 3: Gate (owner)** — the boundary is a behaviour change for full-precision runs: present likeness / mutation / crops per cell, owner picks. Record the choice as a vault decision note (`mb note new decision "KreaPhoton full-precision profile - boundary and identity scale"`).

---

### Task 10: Calibrate `face_boost_scale` (Face Detailer, full profile)

- [ ] **Step 1:** On the Task 9 winner frames (2 prompts × 8 seeds), Face Detailer `standard`, `identity_boost` 1.5 × `face_boost_scale` ∈ {0.67, 0.8, 0.9, 1.0} (effective 1.0 / 1.2 / 1.35 / 1.5) = 64 face passes (~15 min, base cached). Metric: likeness after the detailer, and the detailer's own `id_sim` report line (agrees with the 120-face measurement within 0.01, 2026-09-16).
- [ ] **Step 2:** Pick the smallest scale whose likeness is within 0.01 of the best (less boost = less over-cooking risk). Hypothesis being tested: the int8 requantization loss is what x1.5 compensated, so bf16 needs less.

---

### Task 11: `detail_a_scale` regression check (full profile)

- [ ] **Step 1:** `detail_a_scale` ∈ {0.8, 1.0} × 6 prompts × 4 seeds on the winning profile = 48 frames. Metrics: `research/lora_stack/skin.py` (skin Laplacian variance, hf energy, gloss %) + the confetti check from 2026-09-27 (coloured mesh over skin, 1:1 crops). Keep 1.0 unless 0.8 removes a measured artefact; record either way.

---

### Task 12: Holdout validation, constants, docs, release

**Files:**
- Modify: `kreaphoton/presets.py` (`PRECISION_PROFILES["full"]` values + MEASURED comments)
- Modify: `README.md`, `docs/REFERENCE.md`, `pyproject.toml` (version 1.9.0)

- [ ] **Step 1: Holdout** — the final `full` profile vs int8 baseline on holdout seeds 2001-2008 × 6 prompts (+ no-LoRA references) = 192 frames. Acceptance (owner): bf16 likeness mean ≥ int8 mean + 0.03; bf16 mutation rate ≤ int8 mutation rate; zero non-finite runs on bf16. Fail → back to Task 9 with the holdout result recorded (the holdout is then burned; draw new seeds 3001-3008 for the next attempt).
- [ ] **Step 2: Write the constants** into `PRECISION_PROFILES["full"]` with comments `# MEASURED 2026-10-xx, N=<frames>, Noct v4 bf16 vs int8, AnnaMin_v4; holdout likeness <x> vs <y>`. Run `tests/run_tests.py` — `test_profiles` still passes (int8 neutral), the plumbing test still passes (it patches `full` itself).
- [ ] **Step 3: Docs** — README "Precision" section: what auto does, the override, fp16 = bf16-compute advice (`--bf16-unet`), the LoRA-flag guard; REFERENCE.md: the new input on three nodes, the profile table with values and sources.
- [ ] **Step 4: Release** — version 1.9.0 in `pyproject.toml`; `git commit -m "feat(precision): calibrated full-precision profile (bf16/fp16 Turbo)"`, then `git commit -m "chore(release): 1.9.0"`.

---

### Task 13: Vault

- [ ] Decision note (Task 9 gate) exists and links the plan; learnings KREA2-NODES: final table + what transferred from int8 and what did not; `changelog/2026-10.md` entry; `03_Tasks/KREAPHOTON_AUDIT_BACKLOG.md`: KPA-08 "mutation reduction measured" status updated with the Task 9/12 numbers; KPA-10 "benchmark + holdout protocol" → reference `research/precision/`.

---

## Risks and gaps (found while planning)

- **The leak may not reproduce on demand** (it first fired after an interrupt, two earlier interrupts did not). Task 1 Step 3 has the `--cache-ram 2` escalation; if it still does not fire, the A/B blocks run with a ComfyUI restart between models (the owner's alternative) and Task 1 stays open — recorded, not silently dropped.
- **24.5 GiB bf16 + 8.5 GiB text encoder exceed 32 GB VRAM**: dynamic VRAM streams weights, so bf16 runs are slower than int8 — timing is reported but is not an acceptance criterion; if a bf16 frame takes > 3× the int8 time, the budget table above doubles.
- **The LoRA was trained on int8-quantized RAW** (blissful-tuner ConvRot at load, 2026-09-15): bf16 may NOT beat int8 on likeness by itself — that is exactly what Task 8 measures before any tuning, and Task 9's `identity_lora_scale` is the lever if it does not.
- **One character, one merge**: the profile is calibrated on AnnaMin_v4 + Noct v4. Transfer to other characters / merges is not proven; the holdout uses new seeds, not a new character. Optional extension: repeat Task 12 Step 1 with `Erika_v1` (no training-set centroid on disk → needs a reference batch, see Face Detailer `reference_image`).
- **The judge (Qwen3-VL) can be wrong on small garments** (sandals, hats): Task 7 Step 4 measures its yes-rate on no-LoRA frames; garments it fails on are dropped from the question list rather than counted as mutations.
- **Not in scope:** KreaPhoton Upscale / Upscale v2 / Composition Drafts get no profile (their constants are LoRA-independent fidelity settings); Raw (`krea2_raw_bf16`) is a different model variant, not a precision — its `raw/experimental` preset is untouched.
