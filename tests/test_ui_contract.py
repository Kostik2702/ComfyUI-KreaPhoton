# -*- coding: utf-8 -*-
"""
Plain-assert test for the node-UI contract (node-ui-concept P0): the frozen input order of all
11 nodes, and the library metadata (CATEGORY, DESCRIPTION, OUTPUT_TOOLTIPS, SEARCH_ALIASES,
display names) with no lab shorthand in user-facing text.

Why the order is frozen: a saved workflow stores widget values as a positional list
(widgets_values). Reordering, renaming or removing an input shifts every later value in every
existing workflow. "required" must stay identical; "optional" may only grow at its end.

Run: <embedded python> tests/test_ui_contract.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_nodes import _load_kreaphoton_package  # noqa: E402

# Input order at v1.8.0 (commit 75fa81c). Append new optional inputs to the END of a node's
# optional list and to this table; never edit an existing entry.
FROZEN_INPUTS = {
    "KreaPhotonSampler": (
        ["model", "positive", "latent_image", "seed", "preset", "variety", "preview_method", "denoise"],
        ["negative", "clean_model", "vae", "seed_b", "blend", "restart_enhance", "texture_model", "coherence"]),
    "KreaPhotonSamplerAdvanced": (
        ["model", "positive", "latent_image", "sigmas", "seed", "sampler_order", "detail_amount", "detail_start",
         "detail_end", "detail_peak", "eta0", "sigma_gate", "contraction", "per_channel_contraction",
         "guidance_mode", "flat_cfg", "delta", "guidance_lo", "guidance_hi", "variety_a_latent", "variety_a_cond",
         "variety_end", "preview_method"],
        ["negative", "clean_model", "composition_end", "vae", "variety_seed", "seed_b", "blend",
         "x0_extrapolation", "guidance_rescale", "pag_scale", "pag_lo", "pag_hi", "pag_blocks", "restart_enhance",
         "texture_model", "texture_start", "coherence_jump", "coherence_jump_sigma", "eta_tail", "guidance_sigma",
         "detail_space", "detail_sigma_hi", "detail_sigma_peak", "detail_sigma_lo", "x0_extrap_order", "x0_trust",
         "pc", "refine_mode"]),
    "KreaPhotonScheduler": (
        ["steps", "alpha", "restart_frac", "sigma_r", "plunge"],
        ["latent", "refine_steps", "refine_sigma", "plunge_sigma", "soft_plunge"]),
    "KreaPhotonEmptyLatent": (["size", "aspect", "batch_size"], []),
    "KreaPhotonEncode": (["clip", "text", "style"], ["custom_directive"]),
    "KreaPhotonSaveImage": (["images", "folder_path", "filename_prefix", "format", "quality", "save_metadata"], []),
    "KreaPhotonLoraPhase": (["model", "lora_name", "strength", "phase"], []),
    "KreaPhotonUpscale": (
        ["model", "positive", "image", "vae", "seed", "preset", "scale"],
        ["negative", "upscale_model", "tune"]),
    "KreaPhotonUpscaleV2": (
        ["model", "positive", "image", "vae", "seed", "preset", "scale"],
        ["negative", "upscale_model", "tune"]),
    "KreaPhotonFaceDetailer": (
        ["model", "positive", "image", "vae", "seed", "preset", "max_faces", "identity_boost"],
        ["negative", "face_positive", "reference_image", "upscale_model", "tune", "ref_mean_guidance"]),
    "KreaPhotonCompositionDrafts": (
        ["model", "positive", "latent_image", "vae", "seed", "preset", "n_drafts", "draft_calls", "seed_step",
         "variety"],
        ["negative", "clean_model", "coherence", "layout"]),
}

# Library order follows the pipeline (node-ui-concept section 8).
EXPECTED_CATEGORY = {
    "KreaPhotonEmptyLatent": "1 Setup", "KreaPhotonEncode": "1 Setup", "KreaPhotonLoraPhase": "1 Setup",
    "KreaPhotonCompositionDrafts": "2 Preview",
    "KreaPhotonSampler": "3 Sampling", "KreaPhotonSamplerAdvanced": "3 Sampling", "KreaPhotonScheduler": "3 Sampling",
    "KreaPhotonUpscale": "4 Finishing", "KreaPhotonUpscaleV2": "4 Finishing", "KreaPhotonFaceDetailer": "4 Finishing",
    "KreaPhotonSaveImage": "5 Output",
}

# Lab shorthand that means nothing outside the project: doc numbers, experiment ids, pack names,
# release tags. Measurement history belongs in docs/REFERENCE.md, not in a tooltip.
# Case-sensitive on purpose: "Upscale v2" is a node name, "V16b" is an experiment id; the one
# product name with a capital V (NVIDIA "Photoreal Enhance V2") is excluded explicitly.
JARGON = re.compile(r"docs/\d|ZPhoton|Power[- ]Nodes|(?<!Enhance )\bV\d+[a-z]?\b|\bM\d\b|critic-verified"
                    r"|planning-council|\(v\d+(\.\d+)*\)|\(Sampler v2\)|KPA-\d")


def input_order(cls):
    it = cls.INPUT_TYPES()
    return list(it.get("required", {}).keys()), list(it.get("optional", {}).keys())


def check_order(name, cls):
    req, opt = input_order(cls)
    f_req, f_opt = FROZEN_INPUTS[name]
    assert req == f_req, "%s: required inputs changed (saved workflows shift): %s != %s" % (name, req, f_req)
    assert opt[:len(f_opt)] == f_opt, ("%s: optional inputs reordered/removed; new ones may only be appended: %s"
                                       % (name, opt))


def user_texts(cls):
    """Every user-facing string of a node: description, output tooltips, input tooltips."""
    texts = [("DESCRIPTION", getattr(cls, "DESCRIPTION", ""))]
    texts += [("output %d" % i, t) for i, t in enumerate(getattr(cls, "OUTPUT_TOOLTIPS", ()))]
    it = cls.INPUT_TYPES()
    for sec in ("required", "optional"):
        for k, v in it.get(sec, {}).items():
            if len(v) > 1 and isinstance(v[1], dict) and v[1].get("tooltip"):
                texts.append((k, v[1]["tooltip"]))
    return texts


def main():
    nodes, _ = _load_kreaphoton_package()
    import importlib
    ui = importlib.import_module("kreaphoton.ui_meta")

    print("=" * 78)
    print("test_ui_contract: frozen input order + library metadata of all nodes")
    print("=" * 78)

    assert set(nodes.NODE_CLASS_MAPPINGS) == set(FROZEN_INPUTS), "a node was added/removed: update FROZEN_INPUTS"

    print("[1] input order frozen at v1.8.0 (required identical, optional append-only)")
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        check_order(name, cls)
    # the check must fail on a planted swap and on a planted insertion - otherwise it checks nothing
    real = FROZEN_INPUTS["KreaPhotonEncode"]

    class Swapped:
        @staticmethod
        def INPUT_TYPES():
            return {"required": {"clip": 0, "style": 0, "text": 0}, "optional": {"custom_directive": 0}}

    class Inserted:
        @staticmethod
        def INPUT_TYPES():
            return {"required": {"clip": 0, "text": 0, "style": 0}, "optional": {"new": 0, "custom_directive": 0}}

    class Appended:
        @staticmethod
        def INPUT_TYPES():
            return {"required": {"clip": 0, "text": 0, "style": 0}, "optional": {"custom_directive": 0, "new": 0}}

    for bad in (Swapped, Inserted):
        try:
            check_order("KreaPhotonEncode", bad)
            raise RuntimeError("planted %s was not caught" % bad.__name__)
        except AssertionError:
            pass
    check_order("KreaPhotonEncode", Appended)
    assert FROZEN_INPUTS["KreaPhotonEncode"] == real
    print("     11/11 match; planted swap and insertion caught, an append passes")

    print("[2] categories follow the pipeline")
    assert ui.CATEGORIES == tuple(ui.ROOT + "/" + c for c in
                                  ("1 Setup", "2 Preview", "3 Sampling", "4 Finishing", "5 Output"))
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert cls.CATEGORY == ui.ROOT + "/" + EXPECTED_CATEGORY[name], (name, cls.CATEGORY)
    # The library shows sub-categories in REGISTRATION order (frontend 1.53.6, checked live
    # 2026-10-17: v1.8 order gave 3, 1, 5, 4, 2), so the mapping must be sorted by pipeline step.
    def first_seen(mapping):
        seen = []
        for cls in mapping.values():
            if cls.CATEGORY not in seen:
                seen.append(cls.CATEGORY)
        return seen

    assert first_seen(nodes.NODE_CLASS_MAPPINGS) == list(ui.CATEGORIES), first_seen(nodes.NODE_CLASS_MAPPINGS)
    v18_order = ["KreaPhotonSampler", "KreaPhotonSamplerAdvanced", "KreaPhotonScheduler", "KreaPhotonEmptyLatent",
                 "KreaPhotonEncode", "KreaPhotonSaveImage", "KreaPhotonLoraPhase", "KreaPhotonUpscale",
                 "KreaPhotonUpscaleV2", "KreaPhotonFaceDetailer", "KreaPhotonCompositionDrafts"]
    planted = {k: nodes.NODE_CLASS_MAPPINGS[k] for k in v18_order}
    assert first_seen(planted) != list(ui.CATEGORIES), "registration-order check is blind to the v1.8 order"
    print("     %s, registered in this order" % ", ".join(ui.CATEGORIES))

    print("[3] DESCRIPTION, OUTPUT_TOOLTIPS, SEARCH_ALIASES on every node")
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        desc = getattr(cls, "DESCRIPTION", "")
        assert isinstance(desc, str) and 20 <= len(desc) <= 200, (name, desc)
        tips = getattr(cls, "OUTPUT_TOOLTIPS", None)
        assert isinstance(tips, tuple) and len(tips) == len(cls.RETURN_TYPES), (name, tips, cls.RETURN_TYPES)
        assert all(isinstance(t, str) and t.strip() for t in tips), (name, tips)
        aliases = getattr(cls, "SEARCH_ALIASES", None)
        assert isinstance(aliases, list) and aliases and all(isinstance(a, str) and a for a in aliases), name
    print("     11/11 complete")

    print("[4] display names: the original Upscale is marked classic, not deprecated")
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["KreaPhotonUpscale"] == "KreaPhoton Upscale (classic)"
    assert not getattr(nodes.KreaPhotonUpscale, "DEPRECATED", False)
    assert all(v.startswith("KreaPhoton ") for v in nodes.NODE_DISPLAY_NAME_MAPPINGS.values())
    print("     ok")

    print("[5] no lab shorthand in user-facing text")
    hits = []
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        for where, text in user_texts(cls):
            m = JARGON.search(text)
            if m:
                hits.append("%s.%s: %r in %r" % (name, where, m.group(0), text[:80]))
    assert not hits, "lab shorthand in tooltips:\n  " + "\n  ".join(hits)
    for planted in ("ZPhoton-proven pattern", "per docs/04 item 7", "V2-validated default", "(V16b)",
                    "window (M6 ramp)", "EXPERIMENTAL (v1.4): x", "EXPERIMENTAL (Sampler v2): x"):
        assert JARGON.search(planted), "jargon guard misses %r" % planted
    for clean in ("Experimental: +1 model call per step.", "Wan 2.1 VAE", "sigma 0.65", "1.5 and 2.0", "v2 presets",
                  "KreaPhoton Upscale v2", "NVIDIA DLSS 5 Photoreal Enhance V2", "M (~1.4 MP)"):
        assert not JARGON.search(clean), "jargon guard flags ordinary text %r" % clean
    print("     none; guard catches every planted id and passes ordinary text")

    print("[6] web/kreaphoton_theme.js colours every node and every LoRA phase (P1)")
    web = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
    js = open(os.path.join(web, "kreaphoton_theme.js"), encoding="utf-8").read()
    fam_block = re.search(r"export const FAMILY = \{(.*?)\};", js, re.S).group(1)
    fam = dict(re.findall(r"(\w+):\s*\"(\w+)\"", fam_block))
    assert set(fam) == set(nodes.NODE_CLASS_MAPPINGS), "FAMILY out of sync with the node list: %s" % (
        set(fam) ^ set(nodes.NODE_CLASS_MAPPINGS))
    assert set(fam.values()) <= {"composition", "identity", "texture", "neutral", "phase"}, fam
    phase_block = re.search(r"const PHASE_FAMILY = \{(.*?)\};", js, re.S).group(1)
    phases = set(re.findall(r"\"?([\w+]+)\"?:\s*\"\w+\"", phase_block))
    presets = importlib.import_module("kreaphoton.presets")
    assert set(presets.LORA_PHASES) <= phases, "LoRA phase without a colour: %s" % (set(presets.LORA_PHASES) - phases)
    save_js = open(os.path.join(web, "kreaphoton_save.js"), encoding="utf-8").read()
    assert "Choose folder" in save_js and not re.search("[\U0001F300-\U0001FAFF]", save_js), "emoji in the Save UI"
    print("     %d nodes, %d LoRA phases mapped; Save UI has no emoji" % (len(fam), len(presets.LORA_PHASES)))

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
