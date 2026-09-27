# -*- coding: utf-8 -*-
"""
Plain-assert test for KreaPhoton Composition Drafts (spike 2026-09-22): the schedule
head (composition prefix + terminal plunge), the layout spec parser / scorer, the
duplicate finder and ranking, the sheet renderer, and the node's control flow with
run_sampling / detectors / VAE / preview replaced by fakes. No checkpoint, no generation.

Run: <embedded python> tests/test_composition.py
"""
import importlib
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KREAPHOTON_DIR = os.path.join(REPO_ROOT, "kreaphoton")
COMFYUI_ROOT = r"E:\CUI portable\ComfyUI-torch2.9-cu130-cp313-v1.2\ComfyUI"


def _load():
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)
    if "kreaphoton" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "kreaphoton", os.path.join(KREAPHOTON_DIR, "__init__.py"), submodule_search_locations=[KREAPHOTON_DIR])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["kreaphoton"] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module("kreaphoton.composition")


def _raises(fn, needle=None):
    try:
        fn()
    except ValueError as e:
        if needle is not None:
            assert needle in str(e), (needle, str(e))
        return True
    raise AssertionError("expected ValueError")


class _FakeModel:
    def __init__(self, model_options=None, name="base"):
        self.model_options = dict(model_options or {})
        self.name = name


class _FakeVAE:
    def spacial_compression_decode(self):
        return 8

    def decode_tiled(self, z, **kw):
        import torch
        b, c, t, h, w = z.shape
        # a gradient image whose brightness encodes the batch's seed byte (z mean) so drafts differ
        base = torch.linspace(0.0, 1.0, w * 8).view(1, 1, w * 8, 1).expand(b, h * 8, w * 8, 3).clone()
        return (base * 0.5 + float(z.mean()) * 0.5).clamp(0.0, 1.0)


def test_head(cm):
    import torch
    sch = importlib.import_module("kreaphoton.schedules")
    presets = importlib.import_module("kreaphoton.presets")
    p = presets.PRESETS["turbo/balanced"]
    full = sch.build_schedule(p["n_steps"], alpha=p["alpha"], restart_frac=p["restart_frac"], sigma_r=p["sigma_r"],
                              plunge=p["plunge"])
    head, n, readout = cm.composition_head(full)
    assert n == sum(1 for s in full[:-1].tolist() if s > 0.85) and float(head[-1]) == 0.0 and readout == float(full[n - 1])
    assert torch.equal(head[:-1], full[:n]), "head is the exact schedule prefix"
    h3, n3, r3 = cm.composition_head(full, draft_calls=3)
    assert n3 == 3 and len(h3) == 4 and r3 == float(full[2])
    h99, n99, _ = cm.composition_head(full, draft_calls=99)
    assert n99 == n, "draft_calls beyond the phase clamps to the phase"
    # a restart jump right after the first call must not be walked into
    hj, nj, _ = cm.composition_head(torch.tensor([1.0, 0.9, 0.95, 0.5, 0.0]))
    assert nj == 2 and [round(v, 4) for v in hj.tolist()] == [1.0, 0.9, 0.0], hj
    _raises(lambda: cm.composition_head(torch.tensor([0.8, 0.4, 0.0])), "no model call above")
    print("test_head: OK (balanced: %d composition calls, readout sigma %.3f)" % (n, readout))


def test_layout(cm):
    assert cm.parse_layout("") == {} and cm.parse_layout("  ") == {}
    spec = cm.parse_layout('{"faces": 2, "subject": "left", "shot": "medium", "hands": 1}')
    assert spec == {"faces": 2, "subject": "left", "shot": "medium", "hands": 1}
    _raises(lambda: cm.parse_layout("[1]"), "JSON object")
    _raises(lambda: cm.parse_layout('{"cats": 1}'), "unknown layout key")
    _raises(lambda: cm.parse_layout('{"faces": -1}'), "non-negative")
    _raises(lambda: cm.parse_layout('{"subject": "up"}'), "subject must be")
    _raises(lambda: cm.parse_layout('{"shot": "wide"}'), "shot must be")
    f = {"faces": 2, "persons": 1, "hands": 0, "subject": "left", "shot": "medium"}
    assert cm.layout_score(f, {}) is None
    assert cm.layout_score(f, spec) == (1.0 + 1.0 + 1.0 + 0.5) / 4.0
    assert cm.layout_score(dict(f, faces=0), {"faces": 2}) == 0.0
    assert cm.layout_score(dict(f, shot=None), {"shot": "closeup"}) == 0.0
    print("test_layout: OK")


def test_features(cm):
    import torch
    img = torch.zeros(1, 400, 600, 3)
    boxes = {cm.face_detect.DEFAULT_FACE_MODEL: ([(500, 40, 580, 160), (10, 10, 30, 30)], [0.9, 0.3]),
             cm.PERSON_MODEL: ([(420, 20, 600, 400)], [0.8]),
             cm.HAND_MODEL: ([(100, 100, 130, 140), (200, 100, 230, 140)], [0.7, 0.9])}
    f = cm.draft_features(img, detect=lambda image, model_name, threshold: boxes[model_name])
    assert f["faces"] == 1 and f["persons"] == 1 and f["hands"] == 2, f
    assert f["subject"] == "right" and f["shot"] == "closeup", f      # face 120/400 = 0.30 >= 0.28
    f2 = cm.draft_features(img, detect=lambda image, model_name, threshold: ([], []))
    assert f2["subject"] is None and f2["shot"] is None and f2["faces"] == 0
    print("test_features: OK")


def test_dups_rank_sheet(cm):
    import torch
    # layouts, not noise: a 16x16 thumbnail of uniform noise is ~flat, every such image "duplicates" every other
    a = torch.linspace(0.0, 1.0, 96).view(1, 1, 96, 1).expand(1, 64, 96, 3).clone()   # bright on the right
    b = (a + 0.01 * torch.rand_like(a)).clamp(0.0, 1.0)                                # same layout + noise
    c = torch.linspace(0.0, 1.0, 64).view(1, 64, 1, 1).expand(1, 64, 96, 3).clone()   # bright at the bottom
    thumbs = [cm.luminance_thumb(x) for x in (a, b, c)]
    assert cm.duplicates(thumbs) == [None, 0, None], cm.duplicates(thumbs)
    assert cm.rank([0.5, 0.9, 0.9], [None, None, None]) == [1, 2, 0]
    assert cm.rank([0.5, 0.9, 0.9], [None, 0, None]) == [2, 0, 1], "a duplicate ranks after the originals"
    assert cm.rank([None, None], [None, None]) == [0, 1], "unranked keeps input order"
    sheet = cm.render_sheet([a, b, c], ["#1 seed 1\nfaces 1", "#2", "#3 BEST"], tile=96)
    assert sheet.ndim == 4 and sheet.shape[0] == 1 and sheet.shape[-1] == 3 and sheet.shape[2] == 2 * 96, sheet.shape
    assert float(sheet.max()) <= 1.0 and float(sheet.min()) >= 0.0
    print("test_dups_rank_sheet: OK")


def test_node(cm):
    import torch
    presets = importlib.import_module("kreaphoton.presets")
    calls = []

    def fake_run_sampling(model, positive, negative, latent_dict, sigmas, **kw):
        calls.append({"model": model, "sigmas": sigmas, "kw": kw})
        seed = kw["seed"]
        return {"samples": torch.full_like(latent_dict["samples"], float(seed % 7) / 7.0)}

    counts = {}

    def fake_detect(image, model_name, threshold):
        n = counts.get(model_name, 0)
        return ([(10, 10, 100, 100)] * n, [0.9] * n)
    cm._run_sampling = fake_run_sampling
    cm._detect = fake_detect
    cm._preview_ui = lambda sheet: None
    node = cm.KreaPhotonCompositionDrafts()
    latent = {"samples": torch.zeros(1, 16, 1, 136, 200)}
    model = _FakeModel(name="raw")
    sheet, drafts, best_seed, report = node.draft(model, "pos", latent, _FakeVAE(), 5, "turbo/balanced", 4, 0, 1000, "off")
    assert len(calls) == 4 and [c["kw"]["seed"] for c in calls] == [5, 1005, 2005, 3005]
    p = presets.PRESETS["turbo/balanced"]
    for c in calls:
        assert c["model"] is model and float(c["sigmas"][-1]) == 0.0 and c["kw"]["progress_total"] == p["n_steps"] + 1, c["kw"]
        assert c["kw"]["eta0"] == p["eta0"] and c["kw"]["coherence_jump"] == 0.0 and c["kw"]["x0_extrapolation"] == 0.0
    assert tuple(drafts.shape) == (4, 136 * 8, 200 * 8, 3) and sheet.shape[0] == 1 and best_seed == 5
    assert "no layout spec (unranked)" in report and "best_seed 5 (#1)" in report, report
    # Sampler v2: the head integrates with the preset's own solver / eta_tail, like the Sampler does
    assert all(c["kw"]["solver"] == p["sampler"] and c["kw"]["eta_tail"] is None for c in calls)
    calls.clear()
    node.draft(model, "pos", latent, _FakeVAE(), 5, "turbo/quality-v2", 1, 0, 1000, "off")
    assert calls[0]["kw"]["solver"] == "rf_2m" and calls[0]["kw"]["eta_tail"] == 0.0, calls[0]["kw"]
    calls.clear()
    # draft_calls 2 -> 3-entry head; coherence jump forwarded; explicit clean_model wins for the head
    calls.clear()
    clean = _FakeModel(name="clean")
    node.draft(model, "pos", latent, _FakeVAE(), 5, "turbo/balanced", 1, 2, 1000, "off", clean_model=clean,
               coherence="jump")
    assert len(calls[0]["sigmas"]) == 3 and calls[0]["model"] is clean
    assert calls[0]["kw"]["coherence_jump"] == presets.COHERENCE["jump"]
    # layout ranks: drafts 1..4 see faces 0/1/1/2 -> spec faces 1 picks #2 (first with score 1.0)
    calls.clear()
    seq = iter([0, 1, 1, 2])
    faces_per_call = {}

    def fake_detect2(image, model_name, threshold):
        if model_name == cm.face_detect.DEFAULT_FACE_MODEL:
            k = id(image)
            if k not in faces_per_call:
                faces_per_call[k] = next(seq)
            n = faces_per_call[k]
            return ([(10 + 200 * i, 10, 100 + 200 * i, 100) for i in range(n)], [0.9] * n)
        return ([], [])
    cm._detect = fake_detect2
    sheet, drafts, best_seed, report = node.draft(model, "pos", latent, _FakeVAE(), 5, "turbo/balanced", 4, 0, 1000,
                                                  "off", layout='{"faces": 1}')
    assert best_seed == 1005 and "best_seed 1005 (#2)" in report, report
    assert "#1 seed 5: score 0.50" in report and "#4 seed 3005: score 0.50" in report, report
    # duplicate flag: identical latents -> identical drafts -> DUP
    calls.clear()
    cm._detect = fake_detect
    cm._run_sampling = lambda model, positive, negative, latent_dict, sigmas, **kw: {"samples": latent_dict["samples"]}
    _, _, _, report = node.draft(model, "pos", latent, _FakeVAE(), 5, "turbo/balanced", 3, 0, 1000, "off")
    assert "DUP of #1" in report, report
    _raises(lambda: node.draft(model, "pos", latent, _FakeVAE(), 5, "turbo/balanced", 1, 0, 1000, "off", layout='{"x":1}'),
            "unknown layout key")
    print("test_node: OK")


def main():
    cm = _load()
    print("test_composition:")
    test_head(cm)
    test_layout(cm)
    test_features(cm)
    test_dups_rank_sheet(cm)
    test_node(cm)
    print("test_composition: ALL PASSED")


if __name__ == "__main__":
    main()
