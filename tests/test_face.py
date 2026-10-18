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

    # gender filter helpers (2026-10-01): detector boxes <- insightface verdicts by IoU
    assert fg.GENDERS == ("any", "female", "male")
    assert abs(fg.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9
    assert fg.box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    det = [(300, 100, 420, 240), (40, 40, 100, 100), (600, 50, 640, 90)]
    ins = [((45, 42, 98, 104), "male"), ((305, 95, 425, 235), "female")]
    assert fg.match_genders(det, ins) == ["female", "male", None]   # third box: no insightface face
    # one insightface face cannot be claimed by two detector boxes (greedy by IoU)
    assert fg.match_genders([(0, 0, 10, 10), (1, 0, 11, 10)], [((0, 0, 10, 10), "male")]) == ["male", None]
    b2, c2 = fg.keep_gender(det, [0.9, 0.8, 0.7], ["female", "male", None], "male")
    assert b2 == [det[1]] and c2 == [0.8], (b2, c2)
    confs3 = [0.9, 0.8, 0.7]
    b3, c3 = fg.keep_gender(det, confs3, ["female", "male", None], "any")
    assert b3 is det and c3 is confs3, "'any' passes the detections through untouched"
    print("  [2b] box_iou / match_genders / keep_gender ... ok")


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

    face_builds = []

    def fake_build_face(model, boost):
        face_builds.append(boost)
        return _FakeModel(name="face"), [("a", 1.0 * boost)], []
    fdl._build_face_model = fake_build_face

    # gate OFF -> one run_sampling per pass, boosted face model (default identity_boost 1.5),
    # latent-sized noise_mask
    fdl._gate = lambda: _FakeGate([], available=False)
    calls.clear()
    out_img, out_mask, report = node.detail(model, "pos", img, _FakeVAE(), 5, "standard", 1)
    assert out_img.shape == img.shape and out_mask.shape == (1, 512, 768)
    assert all(c["kw"]["solver"] == presets.FACE_COMMON["sampler"] for c in calls), "sampler NAME reaches the loop"
    assert len(calls) == 2, len(calls)                      # two passes, one attempt each
    assert face_builds == [1.5] and "identity boost x1.50: a 1.50" in report, (face_builds, report)
    # never a phase split (a noise_mask cannot cross a segment boundary - confetti ring 2026-09-15)
    for c in calls:
        assert c["model"].name == "face" and c["kw"].get("texture_model") is None, (c["model"].name, c["kw"])
        assert c["latent"].get("noise_mask") is not None
    # identity_boost 1.0 -> the plan's texture patcher as before, no face build
    calls.clear(); face_builds.clear()
    _, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "standard", 1, identity_boost=1.0)
    assert face_builds == [] and all(c["model"].name == "texture" for c in calls), (face_builds, [c["model"].name for c in calls])
    assert "identity boost" not in report, report
    # precision profile (2026-10-01): the face boost is scaled by the profile's face_boost_scale
    real_profile = fdl.resolve_node_profile
    fdl.resolve_node_profile = lambda m, override: ("full", {"identity_lora_scale": 1.0, "face_boost_scale": 0.8,
                                                             "detail_a_scale": 1.0, "composition_end": 0.85})
    try:
        calls.clear(); face_builds.clear()
        _, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "standard", 1)
        assert [round(b, 6) for b in face_builds] == [1.2], face_builds
        assert "precision profile full: identity boost x1.50 -> x1.20" in report, report
    finally:
        fdl.resolve_node_profile = real_profile
    # no plan + boost -> raw model, report says the boost was ignored
    calls.clear()
    _, _, report = node.detail(_FakeModel(name="raw"), "pos", img, _FakeVAE(), 5, "subtle", 1)
    assert calls[0]["model"].name == "raw" and "identity boost x1.50 ignored" in report, report
    calls.clear()
    out_img, out_mask, report = node.detail(model, "pos", img, _FakeVAE(), 5, "standard", 1)
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

    # gate ON: first attempt below threshold, second passes -> 2 calls in pass 1; without a
    # reference the retry LOWERS denoise (the original crop is the target)
    fdl._gate = lambda: _FakeGate([0.5, 0.8, 0.9, 0.9])
    calls.clear()
    _, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1)
    assert len(calls) == 2, len(calls)
    assert "0.500" in report and "0.800" in report and "pass" in report, report
    assert calls[0]["kw"]["seed"] != calls[1]["kw"]["seed"]
    assert float(calls[1]["sigmas"][0]) < float(calls[0]["sigmas"][0]), (calls[0]["sigmas"], calls[1]["sigmas"])

    # reference BATCH -> centroid of the per-image embeddings; the retry keeps the pass
    # denoise and changes the seed (the character is the target)
    class _VecGate:
        available, reason = True, ""

        def __init__(self, vecs):
            self.vecs = [torch.tensor(v) for v in vecs]

        def embed(self, image):
            return self.vecs.pop(0)

        @staticmethod
        def sim(a, b):
            return float(torch.dot(a, b))
    ref = torch.rand(3, 64, 64, 3)
    fdl._gate = lambda: _VecGate([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0],      # 3 reference images -> centroid ~(0.894, 0.447)
                                  [0.894, 0.447],                          # original crop = centroid -> orig_sim 1.0
                                  [0.0, 1.0],                              # attempt 1 -> 0.447 (below 0.70)
                                  [0.894, 0.447]])                         # attempt 2 -> 1.0 pass
    calls.clear()
    _, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1, reference_image=ref)
    assert "reference: 3 face(s) in 3 image(s)" in report, report
    assert len(calls) == 2 and "pass (attempt 2)" in report, (len(calls), report)
    assert torch.equal(calls[0]["sigmas"], calls[1]["sigmas"]), "with a reference the retry keeps the denoise"
    assert calls[0]["kw"]["seed"] != calls[1]["kw"]["seed"]
    # a reference batch with no face at all -> warning, gate falls back to the original crop
    g = _FakeGate([0.9])
    g.embed = lambda image: None
    fdl._gate = lambda: g
    _, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1, reference_image=ref)
    assert "no face found in reference_image" in report and "no face found in the original crop" in report, report

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
    assert list(it["required"]) == ["model", "positive", "image", "vae", "seed", "preset", "max_faces", "identity_boost"]
    assert it["required"]["identity_boost"][1]["default"] == 1.5
    assert list(it["optional"]) == ["negative", "face_positive", "reference_image", "upscale_model", "tune",
                                    "ref_mean_guidance", "precision_profile", "gender"]
    assert it["optional"]["gender"][0] == ["any", "female", "male"] and it["optional"]["gender"][1]["default"] == "any"
    print("  [7] node control flow with fakes ... ok")


def test_face_model():
    """lora_phase.build_face_model: the plan's texture set with identity-phase entries at
    strength x boost; texture-only entries unchanged; no plan -> (None, [], [])."""
    lp = importlib.import_module("kreaphoton.lora_phase")
    applied = []

    def apply(p, lora, strength):
        applied.append((lora, round(strength, 3)))
        return p
    plan = [{"lora_name": "char", "strength": 1.0, "phase": "all"},
            {"lora_name": "eyes", "strength": 1.5, "phase": "composition+identity"},
            {"lora_name": "face", "strength": 0.8, "phase": "identity"},
            {"lora_name": "style", "strength": 0.4, "phase": "texture"},
            {"lora_name": "layout", "strength": 1.0, "phase": "composition"},
            {"lora_name": "off", "strength": 0.0, "phase": "all"}]
    m = _FakeModel({lp.PLAN_KEY: plan})
    p, boosted, unchanged = lp.build_face_model(m, 1.5, loader=lambda n: n, apply_lora=apply)
    assert p is not None and lp.PLAN_KEY not in p.model_options
    assert applied == [("char", 1.5), ("face", 1.2), ("style", 0.4)], applied      # texture set only, in plan order
    boosted = [(n, round(s, 3)) for n, s in boosted]
    assert boosted == [("char", 1.5), ("face", 1.2)] and unchanged == [("style", 0.4)], (boosted, unchanged)
    assert lp.PLAN_KEY in m.model_options, "the input model keeps its plan"
    assert lp.build_face_model(_FakeModel(), 1.5, loader=lambda n: n, apply_lora=apply) == (None, [], [])
    # the reference centroid helper: mean of the found faces, unit length, count
    import torch
    fdl = importlib.import_module("kreaphoton.face_detailer")

    class G:
        def __init__(self): self.q = [torch.tensor([1.0, 0.0]), None, torch.tensor([0.0, 1.0])]
        def embed(self, image): return self.q.pop(0)
    c, n = fdl.reference_embedding(G(), torch.zeros(3, 8, 8, 3))
    assert n == 2 and torch.allclose(c, torch.tensor([0.7071, 0.7071]), atol=1e-3), (c, n)
    print("  [8] build_face_model / reference_embedding ... ok")


class _FakeGenders:
    def __init__(self, faces, available=True):
        self._faces = list(faces)
        self.available = available
        self.reason = "" if available else "fake genderage off"
        self.calls = 0

    def faces(self, img):
        self.calls += 1
        return list(self._faces)


def test_gender_filter():
    """gender input (2026-10-01): 'any' = v1.8 behaviour (classifier never called); 'female' / 'male'
    keep only that gender's faces, then the usual largest-first max_faces selection; no such face or no
    classifier -> image unchanged with the reason in the report (owner: never detail the wrong person)."""
    import torch
    fdl = importlib.import_module("kreaphoton.face_detailer")
    lp = importlib.import_module("kreaphoton.lora_phase")
    calls = []

    def fake_run_sampling(model, positive, negative, latent_dict, sigmas, **kw):
        calls.append(kw)
        return {"samples": latent_dict["samples"] + 0.5}

    big, small = (300, 100, 420, 240), (40, 40, 100, 100)
    fdl._run_sampling = fake_run_sampling
    fdl._build_phase_models = lambda model, loader=None, apply_lora=None: (None, model, None)
    fdl._detect = lambda image, **kw: ([big, small], [0.9, 0.8])
    fdl._upscale = lambda image, scale, upscale_model=None: torch.nn.functional.interpolate(
        image.movedim(-1, 1), size=(int(round(image.shape[1] * scale)), int(round(image.shape[2] * scale))),
        mode="bilinear").movedim(1, -1)
    fdl._gate = lambda: _FakeGate([], available=False)
    node = fdl.KreaPhotonFaceDetailer()
    model = _FakeModel()
    img = torch.rand(1, 512, 768, 3)

    def region_changed(out, box):
        x0, y0, x1, y1 = box
        return not torch.equal(out[:, y0:y1, x0:x1], img[:, y0:y1, x0:x1])

    genders = _FakeGenders([((45, 42, 98, 104), "male"), ((305, 95, 425, 235), "female")])
    fdl._genders = lambda: genders
    # 'any': the classifier is never asked, the largest face (the woman) is detailed as before
    out, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1, identity_boost=1.0)
    assert genders.calls == 0 and region_changed(out, big) and not region_changed(out, small), report
    # 'male': the smaller face is the only man -> it is detailed, the woman is untouched
    out, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1, identity_boost=1.0, gender="male")
    assert region_changed(out, small) and not region_changed(out, big), report
    assert "gender male: 1 of 2 detections" in report, report
    # 'female' with max_faces 2: only one woman -> one face detailed
    calls.clear()
    out, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 2, identity_boost=1.0, gender="female")
    assert region_changed(out, big) and not region_changed(out, small) and len(calls) == 1, (report, len(calls))
    # no man in the frame -> unchanged, reason reported
    genders._faces = [((305, 95, 425, 235), "female")]
    out, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1, identity_boost=1.0, gender="male")
    assert torch.equal(out, img) and "no male face found - unchanged" in report, report
    # classifier unavailable -> unchanged, reason reported
    fdl._genders = lambda: _FakeGenders([], available=False)
    out, _, report = node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1, identity_boost=1.0, gender="female")
    assert torch.equal(out, img) and "fake genderage off" in report and "unchanged" in report, report
    _raises(lambda: node.detail(model, "pos", img, _FakeVAE(), 5, "subtle", 1, gender="woman"), needle="gender")
    print("  [11] gender filter: any / male / female / none found / no classifier ... ok")


def test_phase_model_parents():
    """Every patcher build_phase_models / build_face_model hands out is a DIRECT clone of the
    input model (2026-10-01, 'memory leak with model Krea2'): when a loaded patcher dies, comfy's
    LoadedModel follows ONE weakref parent level (model_management._switch_parent). A clone of a
    clone collected by the cyclic GC together with its parent breaks that switch and leaves a dead
    LoadedModel entry; research/precision/leak_mech.py reproduces it with the real comfy classes."""
    lp = importlib.import_module("kreaphoton.lora_phase")

    class _ParentModel(_FakeModel):
        def __init__(self, model_options=None, parent=None):
            super().__init__(model_options)
            self.parent = parent

        def clone(self):
            import copy
            return _ParentModel(copy.deepcopy(self.model_options), parent=self)

    plan = [{"lora_name": "char", "strength": 1.0, "phase": "identity"},
            {"lora_name": "layout", "strength": 0.4, "phase": "composition"},
            {"lora_name": "style", "strength": 0.4, "phase": "texture"}]
    m = _ParentModel({lp.PLAN_KEY: plan})
    clean, ident, tex = lp.build_phase_models(m, loader=lambda n: n, apply_lora=lambda p, lora, s: p)
    for name, p in (("clean", clean), ("identity", ident), ("texture", tex)):
        assert p is not None and p.parent is m, (name, p and p.parent)
        assert lp.PLAN_KEY not in p.model_options, name
    assert lp.PLAN_KEY in m.model_options, "the input model keeps its plan"
    face, _, _ = lp.build_face_model(m, 1.5, loader=lambda n: n, apply_lora=lambda p, lora, s: p)
    assert face.parent is m
    print("  [10] phase / face patchers are direct clones of the input ... ok")


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
    test_face_model()
    test_phase_model_parents()
    test_gender_filter()
    test_ref_mean()
    print("test_face: ALL PASSED")


def test_ref_mean():
    """ref_mean.py (spike 2026-09-22): beta 0 / below the ramp is bit-exact; the
    posterior picks the nearest reference at low sigma and flattens with tau; the
    shift is confined to the mask; the bank builder crops the largest face."""
    import torch
    rm = importlib.import_module("kreaphoton.ref_mean")
    torch.manual_seed(0)
    r0 = torch.randn(1, 16, 1, 8, 8)
    r1 = torch.randn(1, 16, 1, 8, 8)
    bank = torch.cat([r0, r1], dim=0)
    s = 0.3
    x = (1.0 - s) * r0 + s * torch.randn_like(r0) * 0.1        # state near r0
    den = torch.randn(1, 16, 1, 8, 8)
    # ramp
    assert rm.beta_at(0.1, 0.5, 0.2, 0.4) == 0.0 and rm.beta_at(0.5, 0.5, 0.2, 0.4) == 0.5
    assert abs(rm.beta_at(0.3, 0.5, 0.2, 0.4) - 0.25) < 1e-9
    # off = bit-exact (same object)
    assert rm.ref_mean_shift(x, den, bank, s, beta=0.0, lo=0.2, hi=0.4) is den
    assert rm.ref_mean_shift(x, den, bank, 0.1, beta=0.5, lo=0.2, hi=0.4) is den
    # exact posterior at low sigma = nearest reference
    w = rm.posterior_weights(x, bank, s)
    assert abs(float(w.sum()) - 1.0) < 1e-5 and float(w[0]) > 0.99, w
    # tau flattens toward uniform
    w_flat = rm.posterior_weights(x, bank, s, tau=1e9)
    assert abs(float(w_flat[0]) - 0.5) < 1e-3, w_flat
    # full-beta shift lands on the weighted mean (here r0)
    out = rm.ref_mean_shift(x, den, bank, 0.5, beta=1.0, lo=0.2, hi=0.4)
    assert torch.allclose(out, r0, atol=1e-4), (out - r0).abs().max()
    # mask confines the shift and drives the vote
    mask = torch.zeros(1, 1, 1, 8, 8); mask[..., :4, :] = 1.0
    out_m = rm.ref_mean_shift(x, den, bank, 0.5, beta=1.0, lo=0.2, hi=0.4, mask=mask)
    assert torch.equal(out_m[..., 4:, :], den[..., 4:, :]) and torch.allclose(out_m[..., :4, :], r0[..., :4, :], atol=1e-4)
    # hook object: normalises the bank through base_model.process_latent_in once, counts calls
    class _BM:
        n = 0
        def process_latent_in(self, z):
            _BM.n += 1
            return z * 2.0
    hook = rm.RefMeanHook(bank, mask, beta=1.0, lo=0.2, hi=0.4)
    o1 = hook(x, den, 0.5, _BM()); o2 = hook(x, den, 0.5, _BM())
    assert _BM.n == 1 and hook.calls == 2 and hook.last_weights.shape == (2,)
    assert torch.allclose(o1[..., :4, :], (2.0 * bank)[int(hook.last_weights.argmax())][None, ..., :4, :], atol=1e-3)
    assert hook(x, den, 0.05, _BM()) is den and hook.calls == 2, "below the ramp: untouched, not counted"
    # bank with a different latent size is resampled to the state's
    hook2 = rm.RefMeanHook(torch.cat([r0, r1], dim=0), None, beta=1.0, lo=0.2, hi=0.4)
    o3 = hook2(torch.randn(1, 16, 1, 12, 10), torch.zeros(1, 16, 1, 12, 10), 0.5, None)
    assert tuple(o3.shape) == (1, 16, 1, 12, 10)
    # bank builder: largest face above threshold, crop geometry via crop_box, encode per image
    fg = importlib.import_module("kreaphoton.face_geometry")
    imgs = torch.rand(3, 256, 256, 3)
    dets = {0: ([(10, 10, 60, 60), (100, 100, 220, 220)], [0.9, 0.9]), 1: ([], []), 2: ([(50, 50, 150, 150)], [0.3])}
    seen = []

    def detect(img, threshold):
        seen.append(img.shape)
        return dets[len(seen) - 1]
    b, n, skipped = rm.build_reference_bank(
        imgs, detect=detect, crop_box_fn=fg.crop_box, encode=lambda px: torch.zeros(1, 16, 1, px.shape[1] // 8, px.shape[2] // 8),
        resize=lambda im, h, w: torch.zeros(1, h, w, 3), crop_factor=2.0, target_hw=(128, 128), align=16, bbox_threshold=0.45)
    assert tuple(b.shape) == (1, 16, 1, 16, 16) and n == 1 and skipped == 2, (b.shape, n, skipped)
    assert rm.build_reference_bank(imgs[1:2], detect=lambda i, threshold: ([], []), crop_box_fn=fg.crop_box, encode=None,
                                   resize=None, crop_factor=2.0, target_hw=(128, 128), align=16, bbox_threshold=0.45)[0] is None
    # the node switch: off -> no ref_hook even if tune sets rm_beta; on -> a RefMeanHook reaches run_sampling
    # (bank built from the reference batch), tune rm_beta overrides the switch's 0.4; on without a reference -> warning
    fdl = importlib.import_module("kreaphoton.face_detailer")
    lp = importlib.import_module("kreaphoton.lora_phase")
    calls = []

    def fake_run_sampling(model, positive, negative, latent_dict, sigmas, **kw):
        calls.append(kw)
        return {"samples": latent_dict["samples"]}
    fdl._run_sampling = fake_run_sampling
    fdl._build_phase_models = lambda model, loader=None, apply_lora=None: (None, model, None)
    fdl._detect = lambda image, **kw: ([(300, 100, 420, 240)], [0.9])
    fdl._upscale = lambda image, scale, upscale_model=None: torch.nn.functional.interpolate(
        image.movedim(-1, 1), size=(int(round(image.shape[1] * scale)), int(round(image.shape[2] * scale))),
        mode="bilinear").movedim(1, -1)
    fdl._gate = lambda: _FakeGate([], available=False)
    node = fdl.KreaPhotonFaceDetailer()
    img = torch.rand(1, 512, 768, 3)
    refs = torch.rand(2, 512, 768, 3)
    _, _, rep = node.detail(_FakeModel(name="raw"), "pos", img, _FakeVAE(), 5, "subtle", 1, identity_boost=1.0,
                            reference_image=refs, tune='{"rm_beta":0.3}')
    assert calls and calls[0].get("ref_hook") is None and "ref-mean" not in rep, "switch off must win over tune"
    calls.clear()
    _, _, rep = node.detail(_FakeModel(name="raw"), "pos", img, _FakeVAE(), 5, "subtle", 1, identity_boost=1.0,
                            reference_image=refs, ref_mean_guidance=True)
    hook = calls[0].get("ref_hook")
    assert isinstance(hook, rm.RefMeanHook) and hook.beta == rm.REF_MEAN_EXPERIMENTAL_BETA and hook.tau == 1.0e5, hook
    assert tuple(hook.bank_vae.shape) == (2, 16, 1, 1024 // 8, 1024 // 8), hook.bank_vae.shape
    assert "ref-mean guidance (EXPERIMENTAL" in rep and "2 face crops encoded" in rep, rep
    calls.clear()
    _, _, rep = node.detail(_FakeModel(name="raw"), "pos", img, _FakeVAE(), 5, "subtle", 1, identity_boost=1.0,
                            reference_image=refs, ref_mean_guidance=True, tune='{"rm_beta":0.15}')
    assert calls[0]["ref_hook"].beta == 0.15, "tune rm_beta overrides the switch's default"
    calls.clear()
    _, _, rep = node.detail(_FakeModel(name="raw"), "pos", img, _FakeVAE(), 5, "subtle", 1, identity_boost=1.0,
                            ref_mean_guidance=True)
    assert calls[0].get("ref_hook") is None and "needs reference_image" in rep, rep
    print("test_ref_mean: OK")


if __name__ == "__main__":
    main()
