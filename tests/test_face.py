# -*- coding: utf-8 -*-
"""
Plain-assert test for the KreaPhoton Face Detailer (v1.6): FACE_PRESETS contract,
face_geometry (selection, crop box, mask, paste, retry policy, keep-best),
face_detect pure helpers, and the node's control flow with run_sampling / VAE /
detector / identity gate replaced by fakes. No checkpoint, no generation.

Run: <embedded python> tests/test_face.py
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
    return (importlib.import_module("kreaphoton.face_geometry"),
            importlib.import_module("kreaphoton.presets"))


def _raises(fn, exc=ValueError, needle=None):
    try:
        fn()
    except exc as e:
        if needle is not None:
            assert needle in str(e), f"expected {needle!r} in error, got {e!r}"
        return True
    raise AssertionError(f"expected {exc.__name__}")


def test_presets(presets):
    assert presets.validate_face_presets() is True
    assert presets.DEFAULT_FACE_PRESET in presets.FACE_PRESETS
    for name, p in presets.FACE_PRESETS.items():
        assert p["passes"], name
        for guide, denoise, steps in p["passes"]:
            assert guide % 16 == 0 and 0.0 < denoise <= 1.0 and steps >= 1, (name, guide, denoise, steps)
    _raises(lambda: presets.validate_face_presets({"x": {"passes": [], "id_threshold": 0.5}}), needle="passes")
    _raises(lambda: presets.validate_face_presets({"x": {"passes": [(1000, 0.3, 6)], "id_threshold": 0.5}}),
            needle="16")
    _raises(lambda: presets.validate_face_presets({"x": {"passes": [(1024, 0.3, 6)], "id_threshold": 1.5}}),
            needle="id_threshold")
    _raises(lambda: presets.validate_face_presets(common=dict(presets.FACE_COMMON, retry_max=0)),
            needle="retry_max")
    print("  [1] face presets contract ... ok")


def test_select_faces(fg):
    boxes = [(0, 0, 10, 10), (100, 100, 200, 200), (300, 300, 340, 340), (0, 0, 100, 100), (500, 500, 600, 600)]
    confs = [0.9, 0.9, 0.9, 0.2, 0.9]
    sel = fg.select_faces(boxes, confs, max_faces=8, min_face_px=20, threshold=0.45)
    # index 0 dropped (small), 3 dropped (conf), sorted by area desc, stable for equal areas (1 before 4)
    assert [i for i, _ in sel] == [1, 4, 2], sel
    sel = fg.select_faces(boxes, confs, max_faces=1, min_face_px=20, threshold=0.45)
    assert [i for i, _ in sel] == [1]
    assert fg.select_faces([], [], max_faces=1, min_face_px=20, threshold=0.45) == []
    print("  [2] select_faces: filter, sort, truncate ... ok")


def test_crop_box(fg):
    # corner face: box shifted inside, aligned to 16, square
    b = fg.crop_box((0, 0, 100, 100), 2.0, (1024, 1024), align=16)
    x0, y0, x1, y1 = b
    assert x0 == 0 and y0 == 0, b
    assert (x1 - x0) % 16 == 0 and (y1 - y0) % 16 == 0 and (x1 - x0) == (y1 - y0), b
    assert x1 - x0 >= 200, b
    # centred face keeps its centre
    b = fg.crop_box((400, 400, 500, 500), 2.0, (1024, 1024), align=16)
    x0, y0, x1, y1 = b
    assert abs((x0 + x1) / 2 - 450) <= 8 and abs((y0 + y1) / 2 - 450) <= 8, b
    # face larger than the image: whole image aligned down
    b = fg.crop_box((10, 10, 700, 700), 2.0, (500, 800), align=16)
    x0, y0, x1, y1 = b
    assert (y1 - y0) == 496 and (x1 - x0) == 496 and 0 <= y0 and y1 <= 500 and 0 <= x0 and x1 <= 800, b
    # non-square small image: the side is capped by the short image axis, stays square
    b = fg.crop_box((10, 10, 60, 60), 2.0, (64, 256), align=16)
    x0, y0, x1, y1 = b
    assert y0 == 0 and y1 == 64 and (x1 - x0) == 64 and x1 <= 256, b
    print("  [3] crop_box: corner, centre, oversize, non-square ... ok")


def test_mask_paste(fg):
    import torch
    m = fg.face_mask((256, 256), (64, 64, 192, 192), dilation=0.1, feather=0.06)
    assert m.shape == (256, 256)
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0
    assert float(m[128, 128]) == 1.0 and float(m[0, 0]) == 0.0
    row = m[128, 128:].tolist()
    assert all(a >= b - 1e-6 for a, b in zip(row, row[1:])), "mask must be non-increasing from the centre"
    assert 0.0 < float(m[128, 200]) < 1.0 or float(m[128, 200]) in (0.0, 1.0)
    # some values strictly between 0 and 1 (the feather band exists)
    assert bool(((m > 0.0) & (m < 1.0)).any())

    img = torch.rand(1, 64, 96, 3)
    patch = torch.rand(1, 32, 32, 3)
    box = (16, 8, 48, 40)
    out = fg.paste(img, box, patch, torch.zeros(32, 32))
    assert torch.equal(out, img) and out is not img
    out = fg.paste(img, box, patch, torch.ones(32, 32))
    assert torch.equal(out[:, 8:40, 16:48], patch)
    assert torch.equal(out[:, :8], img[:, :8]) and torch.equal(out[:, :, 48:], img[:, :, 48:])
    placed = fg.place_mask(torch.ones(32, 32) * 0.5, box, (64, 96))
    assert placed.shape == (64, 96) and abs(float(placed.sum()) - 512.0) < 1e-4
    assert float(placed[0, 0]) == 0.0 and float(placed[20, 20]) == 0.5
    print("  [4] face_mask / paste / place_mask ... ok")


def test_retry_and_best(fg):
    sched = fg.retry_schedule(0.12, 7, retry_max=3, denoise_step=0.05, seed_step=1000)
    assert len(sched) == 3
    assert [round(d, 3) for d, _ in sched] == [0.12, 0.07, 0.05], sched
    assert len({s for _, s in sched}) == 3 and sched[0][1] == 7 and sched[1][1] == 1007
    assert fg.retry_schedule(0.35, 0, retry_max=1, denoise_step=0.05, seed_step=1000) == [(0.35, 0)]

    assert fg.choose_best([None, None], threshold=0.6) == (1, "gate off")
    assert fg.choose_best([0.5, 0.8, 0.7], threshold=0.6) == (1, "pass")
    assert fg.choose_best([0.5, 0.55], threshold=0.6) == (1, "below threshold, kept best")
    assert fg.choose_best([0.5, 0.55], threshold=0.6, orig_sim=0.9) == (None, "kept original (identity regressed)")
    assert fg.choose_best([0.5, 0.55], threshold=0.6, orig_sim=0.58) == (1, "below threshold, kept best")
    assert fg.choose_best([0.7, 0.7], threshold=0.6) == (0, "pass")
    print("  [5] retry_schedule / choose_best ... ok")


def test_detect_helpers():
    import torch
    fd = importlib.import_module("kreaphoton.face_detect")

    class F:
        def __init__(self, bbox):
            self.bbox = bbox
    faces = [F([0, 0, 10, 10]), F([0, 0, 50, 50]), F([0, 0, 20, 20])]
    assert fd._largest_face_index(faces) == 1
    assert fd._largest_face_index([]) is None
    img = torch.zeros(1, 4, 6, 3)
    img[..., 0] = 1.0  # pure red
    bgr = fd._to_bgr_uint8(img)
    assert bgr.shape == (4, 6, 3) and bgr.dtype.name == "uint8"
    assert int(bgr[0, 0, 2]) == 255 and int(bgr[0, 0, 0]) == 0
    print("  [6] face_detect helpers ... ok")


# ---------------------------------------------------------------- node (fakes)
class _FakeModel:
    def __init__(self, model_options=None, name="base"):
        self.model_options = dict(model_options or {})
        self.name = name

    def clone(self):
        import copy
        return _FakeModel(copy.deepcopy(self.model_options), self.name)


class _FakeVAE:
    def encode_tiled(self, px, **kw):
        import torch
        return torch.zeros(1, 16, px.shape[1] // 8, px.shape[2] // 8)

    def encode(self, px):
        return self.encode_tiled(px)

    def spacial_compression_decode(self):
        return 8

    def decode_tiled(self, z, **kw):
        import torch
        return torch.full((1, z.shape[-2] * 8, z.shape[-1] * 8, 3), 0.5)

    def decode(self, z):
        return self.decode_tiled(z)


class _FakeGate:
    def __init__(self, sims, available=True):
        self.sims = list(sims)
        self.available = available
        self.reason = "" if available else "fake off"
        self.calls = 0

    def embed(self, img):
        return ("emb", self.calls)

    def sim(self, a, b):
        i = self.calls
        self.calls += 1
        return self.sims[min(i, len(self.sims) - 1)]


def test_node():
    import torch
    fdl = importlib.import_module("kreaphoton.face_detailer")
    lp = importlib.import_module("kreaphoton.lora_phase")
    presets = importlib.import_module("kreaphoton.presets")

    calls = []

    def fake_run_sampling(model, positive, negative, latent_dict, sigmas, **kw):
        calls.append({"model": model, "positive": positive, "negative": negative,
                      "latent": latent_dict, "sigmas": sigmas, "kw": kw})
        return {"samples": latent_dict["samples"]}

    def fake_build(model, loader=None, apply_lora=None):
        plan = model.model_options.get(lp.PLAN_KEY)
        if not plan:
            return None, model, None
        return None, _FakeModel(name="identity"), _FakeModel(name="texture")

    img = torch.rand(1, 512, 768, 3)
    boxes = [(300, 100, 420, 240), (40, 40, 100, 100)]
    confs = [0.9, 0.8]
    fdl._run_sampling = fake_run_sampling
    fdl._build_phase_models = fake_build
    fdl._detect = lambda image, **kw: (boxes, confs)
    fdl._upscale = lambda image, scale, upscale_model=None: torch.nn.functional.interpolate(
        image.movedim(-1, 1), size=(int(round(image.shape[1] * scale)), int(round(image.shape[2] * scale))),
        mode="bilinear").movedim(1, -1)
    node = fdl.KreaPhotonFaceDetailer()
    model = _FakeModel({lp.PLAN_KEY: [{"lora_name": "a", "strength": 1.0, "phase": "identity"}]})

    # gate OFF -> one run_sampling per pass, identity model, latent-sized noise_mask
    fdl._gate = lambda: _FakeGate([], available=False)
    calls.clear()
    out_img, out_mask, report = node.detail(model, "pos", img, _FakeVAE(), 5, "standard", 1)
    assert out_img.shape == img.shape and out_mask.shape == (1, 512, 768)
    assert len(calls) == 2, len(calls)                      # two passes, one attempt each
    # phase-model rule (== Upscale v2): a pass starting inside the texture segment runs
    # wholly on the texture patcher, otherwise identity + texture_model for the tail
    for c in calls:
        if float(c["sigmas"][0]) <= presets.UPSCALE_TEXTURE_START:
            assert c["model"].name == "texture" and c["kw"]["texture_model"] is None, c["model"].name
        else:
            assert c["model"].name == "identity" and c["kw"]["texture_model"].name == "texture"
    assert calls[0]["latent"]["samples"].shape[-2] == 1024 // 8, calls[0]["latent"]["samples"].shape
    assert calls[1]["latent"]["samples"].shape[-2] == 1536 // 8, calls[1]["latent"]["samples"].shape
    nm = calls[0]["latent"]["noise_mask"]
    z = calls[0]["latent"]["samples"]
    assert nm.shape == (1, 1, 1, z.shape[-2], z.shape[-1]), (nm.shape, z.shape)
    assert float(nm.max()) == 1.0 and float(nm.min()) == 0.0
    assert "identity gate: OFF" in report and "1 faces skipped" in report, report
    assert float(out_mask.sum()) > 0.0
    # the face region changed, a far corner did not
    assert not torch.equal(out_img[:, 100:240, 300:420], img[:, 100:240, 300:420])
    assert torch.equal(out_img[:, 480:, :40], img[:, 480:, :40])

    # gate ON: first attempt below threshold, second passes -> 2 calls in pass 1
    fdl._gate = lambda: _FakeGate([0.5, 0.8, 0.9, 0.9])
    calls.clear()
    _, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1)
    assert len(calls) == 2, len(calls)
    assert "0.500" in report and "0.800" in report and "pass" in report, report
    assert calls[0]["kw"]["seed"] != calls[1]["kw"]["seed"]

    # face_positive replaces positive for the crop; negative passes through
    calls.clear()
    node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1, negative="neg", face_positive="face_pos")
    assert calls[0]["positive"] == "face_pos" and calls[0]["negative"] == "neg"

    # no plan -> the raw model reaches run_sampling
    calls.clear()
    node.detail(_FakeModel(name="raw"), "pos", img, _FakeVAE(), 5, "subtle", 1)
    assert calls[0]["model"].name == "raw"

    # two faces (gate passes first time)
    fdl._gate = lambda: _FakeGate([0.9])
    calls.clear()
    _, out_mask, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 2)
    assert len(calls) == 2 and "face 1" in report and "face 2" in report, report

    # no faces -> unchanged
    fdl._detect = lambda image, **kw: ([], [])
    out_img, out_mask, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1)
    assert torch.equal(out_img, img) and float(out_mask.sum()) == 0.0 and "no faces" in report
    fdl._detect = lambda image, **kw: (boxes, confs)

    # tune: unknown key names it; guide2=0 drops pass 2; steps1 applies
    _raises(lambda: fdl.apply_tune_face(presets.FACE_PRESETS["standard"], presets.FACE_COMMON,
                                        '{"bogus": 1}'), needle="bogus")
    passes, thr, common = fdl.apply_tune_face(presets.FACE_PRESETS["standard"], presets.FACE_COMMON,
                                              '{"guide2": 0, "steps1": 9, "id_threshold": 0.5, "feather": 0.1}')
    assert passes == [(1024, 0.35, 9)] and thr == 0.5 and common["feather"] == 0.1
    passes, _, _ = fdl.apply_tune_face(presets.FACE_PRESETS["subtle"], presets.FACE_COMMON,
                                       '{"guide2": 1536, "denoise2": 0.1, "steps2": 4}')
    assert passes == [(1024, 0.25, 6), (1536, 0.1, 4)]
    _raises(lambda: fdl.apply_tune_face(presets.FACE_PRESETS["subtle"], presets.FACE_COMMON, "not json"),
            needle="JSON")

    # INPUT_TYPES order contract
    it = fdl.KreaPhotonFaceDetailer.INPUT_TYPES()
    assert list(it["required"]) == ["model", "positive", "image", "vae", "seed", "preset", "max_faces"]
    assert list(it["optional"]) == ["negative", "face_positive", "reference_image", "upscale_model", "tune"]
    print("  [7] node control flow with fakes ... ok")


def main():
    fg, presets = _load_kreaphoton_package()
    print("test_face:")
    test_presets(presets)
    test_select_faces(fg)
    test_crop_box(fg)
    test_mask_paste(fg)
    test_retry_and_best(fg)
    test_detect_helpers()
    test_node()
    print("test_face: ALL PASSED")


if __name__ == "__main__":
    main()
