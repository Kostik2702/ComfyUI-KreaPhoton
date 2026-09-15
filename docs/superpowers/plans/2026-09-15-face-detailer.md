# KreaPhoton Face Detailer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `KreaPhoton Face Detailer` node: YOLO face detection → crop → masked img2img with the KreaPhoton sampler on the LoRA plan's identity model → ArcFace-gated keep-best retry → feathered paste.

**Architecture:** Three modules. `face_geometry.py` is pure tensor/py logic (selection, crop box, mask, paste, retry policy, keep-best) with no comfy imports. `face_detect.py` wraps ultralytics (detection) and insightface (identity embeddings) lazily with per-process caches. `face_detailer.py` is the node: preset + tune → per-face loop calling `run_sampling` (as `upscale_v2.py` does) → report string.

**Tech Stack:** torch, ComfyUI (folder_paths, comfy.utils, VAE), ultralytics 8.3 (`face_yolov8m.pt`), insightface 0.7.3 + onnxruntime CPU (`buffalo_l`). Tests are plain-assert scripts run with the ComfyUI embedded python (`tests/run_tests.py`), no pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-face-detailer-design.md`

## Global Constraints

- No comfy / ultralytics / insightface import at module level in `face_geometry.py`; lazy inside functions in `face_detect.py` and `face_detailer.py` (pattern: `tiling.py`, `upscale_v2.py`).
- Node inputs: required order `model, positive, image, vae, seed, preset, max_faces`; optional order `negative, face_positive, reference_image, upscale_model, tune`. Never reorder (saved-workflow rule, `tests/test_nodes.py`).
- Crop boxes aligned to `ALIGN_PX = 16`; latent = pixels / `LATENT_PX = 8`; latent shape `(1, 16, 1, h, w)`.
- `tune` is a JSON object (same as Upscale v2's `apply_tune_v2`), unknown key → `ValueError` naming the key.
- Missing insightface / model pack → gate OFF with a report line, never an exception. Missing ultralytics or `face_yolov8m.pt` → `RuntimeError` with the expected path (detection is mandatory).
- Preset numbers are labeled "derived from Impact Pack 2026-09-01, unvalidated on the KreaPhoton sampler" in `presets.py` and README.
- Version bump `pyproject.toml` 1.5.0 → 1.6.0; README node section + changelog entry.
- Test runner: `"E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2/python_embeded/python.exe" tests/run_tests.py`; exit code from the command itself.

---

### Task 1: Presets

**Files:** Modify `kreaphoton/presets.py` (append after `validate_upscale_v2_presets`); Test `tests/test_face.py` (new, `_load_kreaphoton_package` boilerplate copied from `tests/test_tiling.py`, returning `face_geometry, presets`).

**Interfaces — Produces:**
```python
FACE_PRESETS = {"subtle": {"passes": [(1024, 0.25, 6)], "id_threshold": 0.70},
                "standard": {"passes": [(1024, 0.35, 6), (1536, 0.15, 6)], "id_threshold": 0.65},
                "strong": {"passes": [(1024, 0.45, 7), (1536, 0.20, 6)], "id_threshold": 0.60}}
DEFAULT_FACE_PRESET = "standard"
FACE_COMMON = {"crop_factor": 2.0, "bbox_threshold": 0.45, "min_face_px": 48, "feather": 0.06,
               "dilation": 0.10, "retry_max": 3, "retry_denoise_step": 0.05, "retry_seed_step": 1000,
               "sampler": "euler", "guidance": "window", "detail_a": 0.0, "invert": 0}
def validate_face_presets(presets=None, common=None) -> True   # raises ValueError
```
Validation rules: every preset has `passes` (non-empty list of 3-tuples, guide multiple of 16 in [256, 2048], denoise in (0, 1], steps int ≥ 1) and `id_threshold` in [0, 1]; `FACE_COMMON` crop_factor ≥ 1, feather/dilation in [0, 0.5), retry_max ≥ 1, guidance in ("flat","window"), sampler in ("euler","euler_2m"). Called at import (`validate_face_presets()` at module bottom, like the others).

- [ ] Write tests: valid presets pass; a preset with `passes=[]` raises; guide 1000 (not /16) raises; `id_threshold=1.5` raises.
- [ ] Run → fail (attribute missing). Implement. Run → pass. Commit `feat(face): presets`.

### Task 2: face_geometry — selection and crop box

**Files:** Create `kreaphoton/face_geometry.py`; Test `tests/test_face.py`.

**Interfaces — Produces:**
```python
def select_faces(boxes, confs, *, max_faces, min_face_px, threshold) -> list[tuple[int, tuple]]
    # boxes: list of (x0,y0,x1,y1) float px; returns [(orig_index, box)] sorted by area desc,
    # stable, filtered conf >= threshold and min(w,h) >= min_face_px, truncated to max_faces
def crop_box(box, crop_factor, image_hw, align=16) -> tuple[int,int,int,int]
    # square of side round_up_to_align(max(w,h)*crop_factor) centred on the box centre,
    # shifted inside the image; if side > min(H,W) the side becomes floor_to_align(min(H,W));
    # returns (x0,y0,x1,y1) ints, x1-x0 == y1-y0 (unless the image itself is not square-able,
    # then each side is clamped independently and aligned)
```
- [ ] Tests: `select_faces` drops conf < threshold, drops small, sorts by area desc, stable for equal areas (original order), truncates; `crop_box` corner face → box starts at 0 and side aligned; face larger than image → box == whole image aligned down; `x1-x0` and `y1-y0` multiples of 16.
- [ ] Run → fail. Implement. Run → pass. Commit `feat(face): geometry — selection, crop box`.

### Task 3: face_geometry — mask and paste

**Interfaces — Produces:**
```python
def face_mask(crop_hw, bbox_in_crop, *, dilation, feather) -> torch.Tensor   # (h, w) float 0..1
    # ellipse inscribed in the bbox dilated by `dilation * bbox side` each side, with a
    # linear feather band of width feather * min(crop_hw) outside the ellipse edge
def paste(image, box, patch, mask) -> torch.Tensor
    # image (1,H,W,3); box (x0,y0,x1,y1); patch (1,h,w,3) with h,w == box size; mask (h,w)
    # returns a NEW image with image[box] = image[box]*(1-m) + patch*m
def place_mask(mask, box, image_hw) -> torch.Tensor  # (H, W) zeros with mask at box
```
- [ ] Tests: mask == 1 at bbox centre, == 0 at crop corner, values in [0,1], monotone non-increasing along the row from centre to edge; `paste` with zeros mask returns image unchanged, with ones mask returns the patch inside the box and the image outside; `place_mask` sums equal `mask.sum()`.
- [ ] Run → fail. Implement. Run → pass. Commit `feat(face): geometry — mask, paste`.

### Task 4: face_geometry — retry policy and keep-best

**Interfaces — Produces:**
```python
def retry_schedule(denoise, seed, *, retry_max, denoise_step, seed_step, floor=0.05) -> list[tuple[float,int]]
    # [(denoise, seed), ...] length retry_max; denoise decreases by step, floored; seeds distinct
def choose_best(attempts, *, threshold, orig_sim=None, regress_margin=0.05) -> tuple[int|None, str]
    # attempts: list of sims (float or None). Returns (index or None, reason)
    #   all None -> (last index, "gate off")
    #   max sim >= threshold -> (that index, "pass")
    #   else if orig_sim is not None and max sim < orig_sim - margin -> (None, "kept original (identity regressed)")
    #   else -> (argmax, "below threshold, kept best")
```
- [ ] Tests: schedule length/floor/distinct seeds; `choose_best` four branches; tie → earliest index.
- [ ] Run → fail. Implement. Run → pass. Commit `feat(face): retry policy, keep-best`.

### Task 5: face_detect — YOLO and ArcFace wrappers

**Files:** Create `kreaphoton/face_detect.py`; tests in `tests/test_face.py` cover only the pure helpers (`_bbox_folder_candidates`, `_largest_face_index`) — the model-loading paths are exercised live.

**Interfaces — Produces:**
```python
def detect_faces(image_bhwc, *, model_name="face_yolov8m.pt", threshold=0.25) -> tuple[list, list]
    # (boxes [(x0,y0,x1,y1)], confs); image (1,H,W,C) 0..1 torch; ultralytics YOLO cached by path
def yolo_model_path(model_name) -> str   # RuntimeError listing tried paths if absent
class ArcFaceGate:
    def __init__(self): ...            # available: bool; reason: str
    def embed(self, image_bhwc) -> torch.Tensor | None   # largest face, L2-normed, None if no face
    @staticmethod
    def sim(a, b) -> float             # cosine
_GATE = None; def get_gate() -> ArcFaceGate   # process cache
```
Path resolution: `folder_paths.get_folder_paths("ultralytics_bbox")` if registered, then `folder_paths.models_dir/ultralytics/bbox`, then `.../ultralytics`. insightface root = `folder_paths.models_dir/insightface`, `FaceAnalysis(name="buffalo_l", root=root, providers=["CPUExecutionProvider"])`, `prepare(ctx_id=-1, det_size=(640, 640))`; any exception → `available=False`, `reason=str(e)`.
Image → numpy BGR uint8 for both libraries.

- [ ] Tests: `_largest_face_index` on fake face objects with `.bbox`; `_to_bgr_uint8` shape/dtype/channel flip.
- [ ] Implement. Run → pass. Commit `feat(face): detector + ArcFace gate wrappers`.

### Task 6: face_detailer node

**Files:** Create `kreaphoton/face_detailer.py`; Modify `kreaphoton/nodes.py` (register after UpscaleV2, both mappings); Test `tests/test_face.py` (node with monkeypatched `run_sampling`, fake vae/model/detector/gate) and `tests/test_nodes.py` (INPUT_TYPES order).

**Interfaces — Consumes:** Tasks 1–5, `lora_phase.build_phase_models`, `sampling.run_sampling`, `sampling.run_inversion`, `upscale_v2.inversion_sigmas`, `schedules.refine_schedule/alpha_for_latent/ALPHA/SHIFT`, `presets.preset_guidance/GUIDANCE/MANIFOLD_*/UPSCALE_TEXTURE_START`, `nodes._upscale_pixels/_encode_tiled/_decode_tiled/LATENT_PX/ALIGN_PX/_ORDER_FROM_SAMPLER_NAME`.

**Produces:**
```python
def apply_tune_face(preset: dict, common: dict, tune: str) -> tuple[list, float, dict]
    # (passes, id_threshold, common) ; keys: FACE_COMMON keys, id_threshold,
    # guide1/denoise1/steps1, guide2/denoise2/steps2 (guide2=0 drops pass 2; sets pass 2 if absent)
class KreaPhotonFaceDetailer:  RETURN_TYPES = ("IMAGE", "MASK", "STRING"); RETURN_NAMES = ("image", "mask", "report")
    FUNCTION = "detail"
    def detail(self, model, positive, image, vae, seed, preset, max_faces, negative=None,
               face_positive=None, reference_image=None, upscale_model=None, tune="")
```
Dependency seams for tests: module-level names `_detect = face_detect.detect_faces`, `_gate = face_detect.get_gate`, `_run_sampling = sampling.run_sampling` — the node calls these names so a test can replace them on the module.

Per-face loop exactly per spec §4; seed per attempt `seed + b + face_i * 7919 + attempt * retry_seed_step`; pass 2 source = best crop of pass 1; `texture_model` passed to pass 2 when `float(sigmas[0]) <= UPSCALE_TEXTURE_START` (else the identity model runs the whole pass, same rule as upscale_v2). `noise_mask` = `face_mask` downsampled to latent (area) shaped `(1, 1, 1, h, w)`.

- [ ] Tests (fakes): `run_sampling` receives the identity model built from a plan-carrying fake model; `latent_dict["noise_mask"]` shape `(1,1,1,h/8,w/8)`; gate OFF → exactly one `run_sampling` call per pass; gate ON with sims `[0.5, 0.8]` → two calls, best = second, report contains both sims; `max_faces=1` with two detected → one face detailed and "skipped" in report; unknown tune key raises `ValueError` mentioning the key; no faces → image returned as-is, mask zeros.
- [ ] `tests/test_nodes.py`: assert required/optional key order of `KreaPhotonFaceDetailer.INPUT_TYPES()`.
- [ ] Add `test_face.py` to `tests/run_tests.py`. Run full suite → pass. Commit `feat(face): KreaPhoton Face Detailer node`.

### Task 7: Docs and version

- [ ] README: node section after Upscale v2 (interface, presets with honesty label, identity gate, report, tune keys, models required + where), changelog `1.6.0`. `pyproject.toml` version `1.6.0`.
- [ ] Commit `docs: Face Detailer README + 1.6.0`.

### Task 8: Live smoke (owner rig)

- [ ] Restart ComfyUI (pack is symlinked), load a workflow with `KreaPhoton Face Detailer` after the sampler, run one close-up and one full-body frame; read the report; note first impressions in `learnings/active/KREA2-NODES.md`. Validation A/B against Impact FaceDetailer remains an owner task.
