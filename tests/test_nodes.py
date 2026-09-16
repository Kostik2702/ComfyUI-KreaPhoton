# -*- coding: utf-8 -*-
"""
Plain-assert test for kreaphoton/nodes.py node-level contracts (audit F01 /
KPA-01, KPA-05): what the simple Sampler actually hands to run_sampling per
preset (guidance policy, cfg, x0_extrapolation), preset schema validation, and
INPUT_TYPES backward compatibility (old workflows must keep loading).

run_sampling is monkeypatched - no model, no generation.

Run: <embedded python> tests/test_nodes.py
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
    return (importlib.import_module("kreaphoton.nodes"),
            importlib.import_module("kreaphoton.presets"))


# INPUT_TYPES of v1.2.0 (commit e2122e1) - the backward-compat baseline. New
# inputs may only be APPENDED to "optional" (widget order/slots of saved
# workflows must not shift).
V12_SAMPLER_REQUIRED = ["model", "positive", "latent_image", "seed", "preset", "variety",
                        "preview_method", "denoise"]
V12_SAMPLER_OPTIONAL = ["negative", "clean_model", "vae", "seed_b", "blend"]
V12_ADVANCED_REQUIRED = ["model", "positive", "latent_image", "sigmas", "seed", "sampler_order",
                         "detail_amount", "detail_start", "detail_end", "detail_peak", "eta0",
                         "sigma_gate", "contraction", "per_channel_contraction", "guidance_mode",
                         "flat_cfg", "delta", "guidance_lo", "guidance_hi", "variety_a_latent",
                         "variety_a_cond", "variety_end", "preview_method"]
V12_ADVANCED_OPTIONAL = ["negative", "clean_model", "composition_end", "vae", "variety_seed",
                         "seed_b", "blend"]


def main():
    import torch

    nodes, presets = _load_kreaphoton_package()

    print("=" * 78)
    print("test_nodes: kreaphoton.nodes preset -> run_sampling contract + INPUT_TYPES compat")
    print("=" * 78)

    # --- (1) preset schema contract ---
    print("[1] presets.validate_presets: every preset complete, RAW carries cfg")
    assert presets.validate_presets() is True
    for name, p in presets.PRESETS.items():
        assert p["guidance"] in ("window", "flat"), name
    assert presets.PRESETS["raw/experimental"]["guidance"] == "flat"
    assert presets.PRESETS["raw/experimental"]["cfg"] == 3.5
    try:
        presets.validate_presets({"broken": {"n_steps": 8}})
        raise AssertionError("incomplete preset must be rejected")
    except ValueError as e:
        assert "missing keys" in str(e)
    try:
        presets.validate_presets({"broken": dict(presets.PRESETS["turbo/fast"], guidance="flat")})
        raise AssertionError("flat preset without cfg must be rejected")
    except ValueError as e:
        assert "no 'cfg'" in str(e)
    print("     OK (incomplete / flat-without-cfg rejected with a naming error)")

    # --- (2) preset_guidance policy table ---
    print("[2] presets.preset_guidance policy table")
    pg = presets.preset_guidance
    assert pg(presets.PRESETS["turbo/balanced"], False) == ("off", presets.GUIDANCE["flat_cfg"])
    assert pg(presets.PRESETS["turbo/balanced"], True) == ("window", presets.GUIDANCE["flat_cfg"])
    assert pg(presets.PRESETS["raw/experimental"], False) == ("flat", 3.5)
    assert pg(presets.PRESETS["raw/experimental"], True) == ("flat", 3.5)
    print("     turbo: off / window by negative; raw: flat 3.5 always")

    # --- (2b) variety consumes guidance budget (V16b) ---
    print("[2b] presets.effective_delta")
    ed = presets.effective_delta
    assert ed(1.25, "window", 0.0) == 1.25
    assert ed(1.25, "off", 0.65) == 1.25 and ed(1.25, "flat", 0.65) == 1.25
    hi_a, med_a, low_a = (presets.VARIETY_LEVELS[k][0] for k in ("high", "medium", "low"))
    assert abs(ed(1.25, "window", hi_a) - 0.884) < 1e-3, ed(1.25, "window", hi_a)
    assert abs(ed(1.25, "window", med_a) - 1.025) < 1e-3
    assert ed(1.25, "window", low_a) > ed(1.25, "window", med_a) > ed(1.25, "window", hi_a) > 0.0
    print("     window+variety: high -> %.3f, medium -> %.3f, low -> %.3f; off/flat untouched"
          % (ed(1.25, "window", hi_a), ed(1.25, "window", med_a), ed(1.25, "window", low_a)))

    # --- (3) simple Sampler hands the policy to run_sampling (the F01 bug site) ---
    print("[3] KreaPhotonSampler.sample -> run_sampling kwargs per preset")
    captured = {}
    real_run_sampling = nodes.run_sampling

    def fake_run_sampling(model, positive, negative, latent_dict, sigmas, **kw):
        captured.clear()
        captured.update(kw)
        captured["sigmas"] = sigmas
        captured["negative"] = negative
        return {"samples": torch.zeros(1, 16, 1, 4, 4)}

    nodes.run_sampling = fake_run_sampling
    try:
        node = nodes.KreaPhotonSampler()
        latent = {"samples": torch.zeros(1, 16, 4, 4)}
        pos = [[torch.zeros(1, 8, 30720), {}]]
        neg = [[torch.zeros(1, 8, 30720), {}]]

        node.sample("MODEL", pos, latent, 1001, "raw/experimental", "off", preview_method="none")
        assert captured["guidance_mode"] == "flat", captured["guidance_mode"]
        assert captured["flat_cfg"] == 3.5
        assert captured["negative"] is None   # zero_conditioning happens inside run_sampling
        assert captured["x0_extrapolation"] == presets.PRESETS["raw/experimental"]["x0_extrapolation"]
        print("     raw/experimental, no negative  -> flat cfg 3.5 (was: off - F01)")

        node.sample("MODEL", pos, latent, 1001, "raw/experimental", "off", preview_method="none", negative=neg)
        assert captured["guidance_mode"] == "flat" and captured["flat_cfg"] == 3.5
        print("     raw/experimental, with negative -> flat cfg 3.5")

        node.sample("MODEL", pos, latent, 1001, "turbo/balanced", "off", preview_method="none")
        assert captured["guidance_mode"] == "off"
        assert captured["eta0"] == presets.PRESETS["turbo/balanced"]["eta0"]
        assert captured["variety_a_latent"] == 0.0 and captured["variety_a_cond"] == 0.0
        print("     turbo/balanced, no negative     -> off (unchanged)")

        node.sample("MODEL", pos, latent, 1001, "turbo/balanced", "high", preview_method="none", negative=neg)
        assert captured["guidance_mode"] == "window"
        assert captured["delta"] == presets.GUIDANCE["delta"]
        a_lat, a_cond = presets.VARIETY_LEVELS["high"]
        assert captured["variety_a_latent"] == a_lat and captured["variety_a_cond"] == a_cond
        assert captured["variety_end"] == presets.VARIETY_END
        assert captured["eta0"] == presets.PRESETS["turbo/balanced"]["eta0"], \
            "variety must NOT disable eta0 at the node level (single-lifecycle variety, F02)"
        print("     turbo/balanced, negative+variety high -> window, eta0 kept, variety levels passed")

        # refine path: eta0 forced 0, plain partial descent
        node.sample("MODEL", pos, latent, 1001, "turbo/quality", "off", preview_method="none", denoise=0.4)
        assert captured["eta0"] == 0.0
        assert float(captured["sigmas"][0]) < 0.9 and float(captured["sigmas"][-1]) == 0.0
        print("     turbo/quality denoise=0.4       -> eta0=0, partial descent")

        # Advanced node threads x0_extrapolation and variety_seed
        adv = nodes.KreaPhotonSamplerAdvanced()
        sig = torch.tensor([1.0, 0.5, 0.0])
        adv.sample("MODEL", pos, latent, sig, 7, "euler", 0.0, 0.15, 0.95, 0.6, 0.0, 0.1, 1.0, False,
                   "off", 1.15, 1.25, 0.7, 0.9, 0.0, 0.0, 0.96, preview_method="none",
                   variety_seed=42, x0_extrapolation=0.35)
        assert captured["x0_extrapolation"] == 0.35 and captured["variety_seed"] == 42
        adv.sample("MODEL", pos, latent, sig, 7, "euler", 0.0, 0.15, 0.95, 0.6, 0.0, 0.1, 1.0, False,
                   "off", 1.15, 1.25, 0.7, 0.9, 0.0, 0.0, 0.96, preview_method="none")
        assert captured["x0_extrapolation"] == 0.0 and captured["variety_seed"] == 7, \
            "omitted optional inputs must keep v1.2 semantics (variety_seed=seed, no extrapolation)"
        print("     Advanced: x0_extrapolation / variety_seed threaded, defaults keep v1.2 semantics")
    finally:
        nodes.run_sampling = real_run_sampling

    # --- (4) INPUT_TYPES backward compatibility ---
    print("[4] INPUT_TYPES backward compatibility vs v1.2.0")
    s_it = nodes.KreaPhotonSampler.INPUT_TYPES()
    a_it = nodes.KreaPhotonSamplerAdvanced.INPUT_TYPES()
    assert list(s_it["required"].keys()) == V12_SAMPLER_REQUIRED, list(s_it["required"].keys())
    assert list(s_it["optional"].keys())[:len(V12_SAMPLER_OPTIONAL)] == V12_SAMPLER_OPTIONAL
    assert list(a_it["required"].keys()) == V12_ADVANCED_REQUIRED, list(a_it["required"].keys())
    assert list(a_it["optional"].keys())[:len(V12_ADVANCED_OPTIONAL)] == V12_ADVANCED_OPTIONAL
    assert "x0_extrapolation" in a_it["optional"]
    assert a_it["optional"]["x0_extrapolation"][1]["default"] == 0.0
    print("     required lists identical to v1.2.0; new inputs only appended to optional")

    # --- (5) all 7 mappings present ---
    print("[5] NODE_CLASS_MAPPINGS")
    expected = {"KreaPhotonSampler", "KreaPhotonSamplerAdvanced", "KreaPhotonScheduler",
                "KreaPhotonEmptyLatent", "KreaPhotonEncode", "KreaPhotonSaveImage", "KreaPhotonLoraPhase",
                "KreaPhotonUpscale", "KreaPhotonUpscaleV2", "KreaPhotonFaceDetailer"}
    assert set(nodes.NODE_CLASS_MAPPINGS.keys()) == expected
    assert set(nodes.NODE_DISPLAY_NAME_MAPPINGS.keys()) == expected
    print("     10/10 nodes registered with display names")

    # --- (6) v1.4: resolution-aware shift through the simple node ---
    print("[6] simple node: resolution-aware alpha (L unchanged, S softer)")
    import math
    sch = importlib.import_module("kreaphoton.schedules")
    nodes.run_sampling = fake_run_sampling
    try:
        node = nodes.KreaPhotonSampler()
        pos = [[torch.zeros(1, 8, 30720), {}]]
        node.sample("MODEL", pos, {"samples": torch.zeros(1, 16, 200, 136)}, 1, "turbo/balanced", "off",
                    preview_method="none")
        sig_L = captured["sigmas"].clone()
        node.sample("MODEL", pos, {"samples": torch.zeros(1, 16, 128, 128)}, 1, "turbo/balanced", "off",
                    preview_method="none")
        sig_S = captured["sigmas"].clone()
        node.sample("MODEL", pos, {"samples": torch.zeros(1, 16, 1, 128, 128)}, 1, "turbo/balanced", "off",
                    preview_method="none")
        sig_S5 = captured["sigmas"].clone()
        p = presets.PRESETS["turbo/balanced"]
        ref_L = sch.build_schedule(p["n_steps"], alpha=p["alpha"], restart_frac=p["restart_frac"],
                                   sigma_r=p["sigma_r"], plunge=p["plunge"])
        assert torch.equal(sig_L, ref_L), "L tier schedule must be bit-identical to the preset"
        assert not torch.equal(sig_S, ref_L) and torch.equal(sig_S, sig_S5)
        ref_S = sch.build_schedule(p["n_steps"], alpha=math.exp(sch.canonical_mu(4096)),
                                   restart_frac=p["restart_frac"], sigma_r=p["sigma_r"], plunge=p["plunge"])
        assert torch.allclose(sig_S, ref_S, atol=1e-6)
        assert float(sig_S[2]) < float(sig_L[2]), "softer shift -> lower sigma at the same step index"
        # restart_enhance='off' -> no hook; a DLSS preset without vae -> naming error before sampling
        node.sample("MODEL", pos, {"samples": torch.zeros(1, 16, 8, 8)}, 1, "turbo/balanced", "off",
                    preview_method="none", restart_enhance="off")
        assert captured["restart_hook"] is None
        try:
            node.sample("MODEL", pos, {"samples": torch.zeros(1, 16, 8, 8)}, 1, "turbo/balanced", "off",
                        preview_method="none", restart_enhance="dlss5 default")
            raise AssertionError("restart_enhance without vae must raise")
        except ValueError as e:
            assert "vae" in str(e)
    finally:
        nodes.run_sampling = real_run_sampling
    print("     L bit-identical; S uses canonical mu %.3f; restart_enhance guarded" % sch.canonical_mu(4096))

    # --- (7) v1.4.1: LoRA phase plan on the MODEL line -> phase models ---
    print("[7] KreaPhotonLoraPhase plan + lora_phase.build_phase_models")
    lp = importlib.import_module("kreaphoton.lora_phase")

    class FakePatcher:
        def __init__(self, tag="base", options=None, patches=None):
            self.tag = tag
            self.model_options = {} if options is None else options
            self.patches = [] if patches is None else patches

        def clone(self):
            import copy
            return FakePatcher(self.tag, copy.deepcopy(self.model_options), list(self.patches))

        def add_patches(self, patches, strength):
            self.patches.append((patches, strength))

    it = nodes.KreaPhotonLoraPhase.INPUT_TYPES()
    assert list(it["required"].keys()) == ["model", "lora_name", "strength", "phase"]
    assert it["required"]["phase"][0] == list(presets.LORA_PHASES.keys()) == list(lp.PHASE_SEGMENTS.keys())
    assert nodes.KreaPhotonLoraPhase.RETURN_TYPES == ("MODEL",)
    base = FakePatcher()
    node = nodes.KreaPhotonLoraPhase()
    (m1,) = node.apply(base, "char.safetensors", 1.0, "identity")
    (m2,) = node.apply(m1, "style.safetensors", 0.4, "texture")
    assert lp.PLAN_KEY not in base.model_options, "plan must not leak into the input model"
    assert [e["lora_name"] for e in m1.model_options[lp.PLAN_KEY]] == ["char.safetensors"]
    assert [e["lora_name"] for e in m2.model_options[lp.PLAN_KEY]] == ["char.safetensors", "style.safetensors"]
    try:
        node.apply(base, "x", 1.0, "nope")
        raise AssertionError("unknown phase must raise")
    except ValueError:
        pass
    # segment membership
    plan = m2.model_options[lp.PLAN_KEY]
    assert lp.phase_sets(plan) == {"composition": (), "identity": (0,), "texture": (0, 1)}
    assert lp.phase_sets([{"lora_name": "a", "strength": 1.0, "phase": "all"}]) == {"composition": (0,), "identity": (0,), "texture": (0,)}
    assert lp.phase_sets([{"lora_name": "a", "strength": 0.0, "phase": "all"}]) == {"composition": (), "identity": (), "texture": ()}
    # expansion with injected loader/apply: distinct sets -> distinct patchers, shared sets -> shared
    loaded = []
    applied = []

    def loader(name):
        loaded.append(name)
        return "LORA:" + name

    def apply(p, lora, strength):
        applied.append((p.tag, lora, strength))
        p.add_patches(lora, strength)
        return p

    clean, ident, tex = lp.build_phase_models(m2, loader=loader, apply_lora=apply)
    assert clean is not None and tex is not None and ident is not None
    assert lp.PLAN_KEY not in ident.model_options and lp.PLAN_KEY not in clean.model_options
    assert clean.patches == [] and ident.patches == [("LORA:char.safetensors", 1.0)]
    assert tex.patches == [("LORA:char.safetensors", 1.0), ("LORA:style.safetensors", 0.4)]
    # identity-only plan: texture shares the identity patcher -> no texture split, clean = base
    c2, i2, t2 = lp.build_phase_models(m1, loader=loader, apply_lora=apply)
    assert t2 is None and c2 is not None and c2.patches == [] and i2.patches == [("LORA:char.safetensors", 1.0)]
    # 'all' only -> everything shares one patcher -> no split at all
    (m3,) = node.apply(base, "flat.safetensors", 0.5, "all")
    c3, i3, t3 = lp.build_phase_models(m3, loader=loader, apply_lora=apply)
    assert c3 is None and t3 is None and i3.patches == [("LORA:flat.safetensors", 0.5)]
    # no plan -> passthrough
    assert lp.build_phase_models(base, loader=loader, apply_lora=apply) == (None, base, None)
    print("     plan appended per node (no leak), sets composition/identity/texture, shared sets deduped")

    # --- (8) Advanced optional inputs (v1.4 additions appended, not inserted) ---
    a_it = nodes.KreaPhotonSamplerAdvanced.INPUT_TYPES()
    tail = list(a_it["optional"].keys())[len(V12_ADVANCED_OPTIONAL):]
    assert tail == ["x0_extrapolation", "guidance_rescale", "pag_scale", "pag_lo", "pag_hi", "pag_blocks",
                    "restart_enhance", "texture_model", "texture_start", "coherence_jump", "coherence_jump_sigma"], tail
    s_it = nodes.KreaPhotonSampler.INPUT_TYPES()
    assert list(s_it["optional"].keys())[len(V12_SAMPLER_OPTIONAL):] == ["restart_enhance", "texture_model", "coherence"]
    assert s_it["optional"]["restart_enhance"][0][0] == "off"
    assert s_it["optional"]["coherence"][0] == presets.COHERENCE_OPTIONS and s_it["optional"]["coherence"][1]["default"] == "off"
    print("[8] new optional inputs appended after the v1.2 ones (simple: +3; Advanced: +11)")

    # --- (10) coherence combo on the simple node -> schedule / kwargs ---
    print("[10] simple node coherence combo")
    nodes.run_sampling = fake_run_sampling
    try:
        node = nodes.KreaPhotonSampler()
        pos = [[torch.zeros(1, 8, 30720), {}]]
        latent = {"samples": torch.zeros(1, 16, 200, 136)}

        def jumps_of(sig):
            s = sig.tolist()
            return [i for i in range(len(s) - 1) if s[i + 1] > s[i] + 1e-6]

        node.sample("MODEL", pos, latent, 1, "turbo/balanced", "off", preview_method="none", coherence="off")
        assert captured["coherence_jump"] == 0.0 and len(jumps_of(captured["sigmas"])) == 1
        assert sch.count_model_calls(captured["sigmas"]) == 12
        node.sample("MODEL", pos, latent, 1, "turbo/balanced", "off", preview_method="none", coherence="jump")
        assert captured["coherence_jump"] == presets.COHERENCE["jump"] and len(jumps_of(captured["sigmas"])) == 1
        assert captured["coherence_jump_sigma"] == presets.COHERENCE["jump_sigma"]
        node.sample("MODEL", pos, latent, 1, "turbo/balanced", "off", preview_method="none", coherence="self_refine")
        assert captured["coherence_jump"] == 0.0 and len(jumps_of(captured["sigmas"])) == 2
        assert sch.count_model_calls(captured["sigmas"]) == 12 + presets.COHERENCE["refine_steps"]
        node.sample("MODEL", pos, latent, 1, "turbo/balanced", "off", preview_method="none", coherence="jump+self_refine")
        assert captured["coherence_jump"] > 0 and len(jumps_of(captured["sigmas"])) == 2
        # refine (denoise<1) never gets a jump or a self-refine pass
        node.sample("MODEL", pos, latent, 1, "turbo/balanced", "off", preview_method="none", denoise=0.4,
                    coherence="jump+self_refine")
        assert captured["coherence_jump"] == 0.0 and len(jumps_of(captured["sigmas"])) == 0
        # Scheduler node builds the refine pass too
        sig = nodes.KreaPhotonScheduler().build(12, sch.ALPHA, 0.25, 0.65, True, refine_steps=4)[0]
        assert len(jumps_of(sig)) == 2 and sch.count_model_calls(sig) == 16
        sig2 = nodes.KreaPhotonScheduler().build(12, sch.ALPHA, 0.25, 0.65, False, refine_steps=4)[0]
        assert len(jumps_of(sig2)) == 1, "no plunge -> no self-refine pass"
    finally:
        nodes.run_sampling = real_run_sampling
    print("     off / jump / self_refine / both -> jump strength + 1 or 2 ascending jumps; refine path untouched")

    # --- (9) samplers expand a LoRA plan into phase models (mocked run_sampling) ---
    print("[9] simple node expands the LoRA plan into clean/model/texture")
    real_build = nodes.build_phase_models

    def fake_build(model):
        if isinstance(model, dict) and model.get("plan"):
            tag = model.get("tag", "")
            if model.get("shared"):          # every phase set identical -> one patcher
                return (None, "IDENTITY" + tag, None)
            return ("CLEAN" + tag, "IDENTITY" + tag, "TEXTURE" + tag)
        return (None, model, None)

    nodes.build_phase_models = fake_build
    nodes.run_sampling = fake_run_sampling
    try:
        node = nodes.KreaPhotonSampler()
        pos = [[torch.zeros(1, 8, 30720), {}]]
        latent = {"samples": torch.zeros(1, 16, 8, 8)}
        node.sample({"plan": True}, pos, latent, 1, "turbo/balanced", "off", preview_method="none")
        assert captured["clean_model"] == "CLEAN" and captured["texture_model"] == "TEXTURE"
        node.sample({"plan": True}, pos, latent, 1, "turbo/balanced", "off", preview_method="none",
                    clean_model="EXPLICIT")
        assert captured["clean_model"] == "EXPLICIT" and captured["texture_model"] == "TEXTURE", \
            "an explicitly connected phase model must win over the plan"
        node.sample("MODEL", pos, latent, 1, "turbo/balanced", "off", preview_method="none")
        assert captured["clean_model"] is None and captured["texture_model"] is None
        # ROOT CAUSE of "the face changes on the last steps" ([EDITORAL] KREA6,
        # 2026-09-15): an explicitly connected texture_model / clean_model that is
        # itself a KreaPhoton LoRA Phase chain carries a PLAN and no patches - handed
        # to run_sampling raw, the restart segment ran on the bare checkpoint
        # without the character LoRA. The explicit input must be expanded for ITS
        # phase (texture -> its texture patcher, clean -> its composition patcher).
        node.sample({"plan": True}, pos, latent, 1, "turbo/balanced", "off", preview_method="none",
                    texture_model={"plan": True, "tag": "_T"}, clean_model={"plan": True, "tag": "_C"})
        assert captured["texture_model"] == "TEXTURE_T", captured["texture_model"]
        assert captured["clean_model"] == "CLEAN_C", captured["clean_model"]
        # a plan whose texture set equals its identity set (no texture split) -> the
        # identity patcher IS the texture model; same for composition
        node.sample("MODEL", pos, latent, 1, "turbo/balanced", "off", preview_method="none",
                    texture_model={"plan": True, "tag": "_S", "shared": True},
                    clean_model={"plan": True, "tag": "_S", "shared": True})
        assert captured["texture_model"] == "IDENTITY_S" and captured["clean_model"] == "IDENTITY_S"
        adv = nodes.KreaPhotonSamplerAdvanced()
        sig = sch.build_schedule(8, alpha=sch.ALPHA, restart_frac=0.25, sigma_r=0.65, plunge=True)
        adv.sample({"plan": True}, pos, latent, sig, 1, "euler", 0.0, 0.15, 0.95, 0.6, 0.0, 0.10, 1.0,
                   False, "off", 1.0, 1.0, 0.7, 0.9, 0.0, 0.0, 0.96, preview_method="none",
                   texture_model={"plan": True, "tag": "_T"}, clean_model={"plan": True, "tag": "_C"})
        assert captured["texture_model"] == "TEXTURE_T" and captured["clean_model"] == "CLEAN_C"
    finally:
        nodes.build_phase_models = real_build
        nodes.run_sampling = real_run_sampling
    print("     plan -> clean/texture phase models; explicit inputs win and are expanded for their phase")

    # --- (11) v1.5 KreaPhoton Upscale: pixel upscale -> tiled encode -> run_sampling
    #          contract (eta0 0, tiler, anchor, texture rule) -> tiled decode ---
    print("[11] KreaPhotonUpscale contract (mocked vae / model / run_sampling)")
    tl = importlib.import_module("kreaphoton.tiling")
    up_it = nodes.KreaPhotonUpscale.INPUT_TYPES()
    assert list(up_it["required"].keys()) == ["model", "positive", "image", "vae", "seed", "preset", "scale"]
    assert list(up_it["optional"].keys()) == ["negative", "upscale_model", "tune"]
    # tune override: empty = untouched; keys route to preset / tile / anchor / extra; unknown rejected
    p0, t0, a0, e0 = nodes._apply_tune(presets.UPSCALE_PRESETS["detail"], "  ")
    assert p0 == presets.UPSCALE_PRESETS["detail"] and t0 == presets.TILE and a0 == presets.ANCHOR and e0 == {}
    p1, t1, a1, e1 = nodes._apply_tune(presets.UPSCALE_PRESETS["detail"],
                                       '{"denoise": 0.25, "detail_a": 0, "tile": 768, "radius": 4, "alpha": 3.158}')
    assert p1["denoise"] == 0.25 and p1["detail_a"] == 0
    assert p1["n_steps"] == presets.UPSCALE_PRESETS["detail"]["n_steps"], "untouched keys keep the preset value"
    assert t1["size"] == 768 and t1["overlap"] == presets.TILE["overlap"] and a1["radius"] == 4.0 and e1["alpha"] == 3.158
    assert presets.UPSCALE_PRESETS["detail"]["denoise"] != 0.25, "tune must not mutate the preset table"
    for bad in ('{"nope": 1}', '[1, 2]', '{bad json'):
        try:
            nodes._apply_tune(presets.UPSCALE_PRESETS["detail"], bad)
            raise AssertionError("tune %r must be rejected" % bad)
        except ValueError:
            pass
    # back-projection: the result reproduces the source at the source scale; HF the
    # source cannot see survives; 0 iters / same size = identity
    torch.manual_seed(3)
    # a natural-image-like (smooth) source: on white noise the Down(Up(.)) operator converges
    # slowly (residual 0.010 after 3 iters), on smooth content it is < 1e-4 - measured
    src_img = torch.nn.functional.interpolate(torch.rand(1, 3, 8, 12), size=(32, 48), mode="bicubic",
                                              align_corners=False).clamp(0, 1).movedim(1, -1)
    res_img = torch.nn.functional.interpolate(src_img.movedim(-1, 1), scale_factor=2, mode="bilinear",
                                              align_corners=False).movedim(1, -1)
    res_img = (res_img * 0.9 + 0.03).clamp(0, 1)           # tone drift the model would introduce
    hf = torch.zeros(1, 64, 96, 3)
    hf[:, ::2, ::2] += 0.02
    hf[:, 1::2, 1::2] += 0.02
    hf[:, ::2, 1::2] -= 0.02
    hf[:, 1::2, ::2] -= 0.02                                # zero-mean per 2x2 block: invisible at source scale
    res_hf = (res_img + hf).clamp(0.02, 0.98)
    # the node's own downscale operator (antialiased bilinear - "area" bins unevenly at a
    # non-integer factor and leaves a periodic residual, batch-2 survey 2026-09-13)
    def down_aa(t, size):
        return torch.nn.functional.interpolate(t.movedim(-1, 1), size=size, mode="bilinear", antialias=True,
                                               align_corners=False).movedim(1, -1)
    bp = nodes.back_project(res_hf, src_img, presets.FIDELITY["bp_iters"])
    resid = float((down_aa(bp, (32, 48)) - src_img).abs().mean())
    before = float((down_aa(res_hf, (32, 48)) - src_img).abs().mean())
    assert resid < 5e-4 and resid < 0.05 * before, "down(result) must reproduce the source (%.5f vs %.5f)" % (resid, before)
    r1 = float((down_aa(nodes.back_project(res_hf, src_img, 1), (32, 48)) - src_img).abs().mean())
    assert r1 > resid, "more iterations -> smaller residual"
    # the 2x2 zero-mean pattern is invisible at the source scale -> it must survive
    def block_hf(t):
        area = torch.nn.functional.interpolate(t.movedim(-1, 1), scale_factor=0.5, mode="area")
        return t - area.repeat_interleave(2, -2).repeat_interleave(2, -1).movedim(1, -1)
    kept = float((block_hf(bp) * hf).mean()) / float((hf * hf).mean())
    assert kept > 0.8, "sub-source detail must be kept (%.3f of the pattern survives)" % kept
    # non-integer scale (x1.5 -> 48x72): the residual must be as small as at x2 - the area
    # operator gave 0.0054 here vs 0.0002 (the grid-on-skin defect of the survey)
    res15 = torch.nn.functional.interpolate(src_img.movedim(-1, 1), size=(48, 72), mode="bilinear",
                                            align_corners=False).movedim(1, -1) * 0.9 + 0.03
    bp15 = nodes.back_project(res15.clamp(0, 1), src_img, presets.FIDELITY["bp_iters"])
    resid15 = float((down_aa(bp15, (32, 48)) - src_img).abs().mean())
    assert resid15 < 5e-4, "x1.5 back-projection residual %.5f" % resid15
    assert nodes.back_project(res_hf, src_img, 0) is res_hf
    assert nodes.back_project(src_img, src_img, 3) is src_img, "same size -> identity object"
    _, _, _, e2 = nodes._apply_tune(presets.UPSCALE_PRESETS["detail"], '{"bp_iters": 0, "bp_lock": 2}')
    assert e2["bp_iters"] == 0 and e2["bp_lock"] == 2
    # lock_scale 2: the result matches the source on the HALF-resolution grid only, and
    # the 2-row stripe pattern of the SOURCE grid (which lock 1 enforces) is NOT forced
    src_blk = src_img.clone()          # 2-row stripes (period 4 source px): representable under the
    src_blk[:, ::4, :, :] += 0.05      # antialiased operator at lock 1, Nyquist (suppressed) at lock 2
    src_blk[:, 1::4, :, :] += 0.05
    src_blk = src_blk.clamp(0, 1)
    res2 = torch.nn.functional.interpolate(src_img.movedim(-1, 1), scale_factor=2, mode="bilinear",
                                           align_corners=False).movedim(1, -1)
    bp_lock1 = nodes.back_project(res2, src_blk, 3, 1.0)
    bp_lock2 = nodes.back_project(res2, src_blk, 3, 2.0)
    assert float((down_aa(bp_lock1, (32, 48)) - src_blk).abs().mean()) < 2e-3, "lock 1 reproduces the striped source"
    assert float((down_aa(bp_lock2, (32, 48)) - src_blk).abs().mean()) > 0.01, "lock 2 must NOT force the source-grid stripes"
    bp_lock2s = nodes.back_project(res2, src_img, presets.FIDELITY["bp_iters"], 2.0)   # smooth source
    assert float((down_aa(bp_lock2s, (16, 24)) - down_aa(src_img, (16, 24))).abs().mean()) < 1e-3,         "lock 2 reproduces the source on the half grid"
    assert up_it["required"]["preset"][0] == list(presets.UPSCALE_PRESETS.keys())
    assert up_it["required"]["preset"][1]["default"] == presets.DEFAULT_UPSCALE_PRESET
    assert up_it["required"]["scale"][1]["min"] == 1.25 and up_it["required"]["scale"][1]["max"] == 2.0, \
        "scale 1.0 measured useless (no-op or rewrite), survey 2026-09-13"
    assert nodes.KreaPhotonUpscale.RETURN_TYPES == ("IMAGE",)
    assert nodes._target_size(200, 300, 2.0) == (400, 608), nodes._target_size(200, 300, 2.0)
    assert nodes._target_size(1600, 1088, 1.5) == (2400, 1632)
    assert nodes._target_size(1600, 1088, 1.0) == (1600, 1088), "aligned source at 1.0 keeps its size"

    class FakeVAE:
        def __init__(self):
            self.encoded = []
            self.decoded = []

        def encode_tiled(self, pixels, tile_x=None, tile_y=None, overlap=None):
            self.encoded.append((tuple(pixels.shape), tile_x, tile_y, overlap))
            return torch.zeros(1, 16, pixels.shape[1] // 8, pixels.shape[2] // 8)   # 4D on purpose

        def spacial_compression_decode(self):
            return 8

        def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None):
            self.decoded.append((tuple(samples.shape), tile_x, tile_y, overlap))
            h, w = samples.shape[-2] * 8, samples.shape[-1] * 8
            return torch.zeros(1, 1, h, w, 3)                                        # 5D video-shaped

    class FakeBase:
        def __init__(self):
            self.seen = []

        def process_latent_in(self, z):
            self.seen.append(tuple(z.shape))
            return z * 2.0

    class FakeModel:
        def __init__(self, tag="M"):
            self.tag = tag
            self.model_options = {}
            self.model = FakeBase()

        def clone(self):
            return FakeModel(self.tag)

    def fake_run_sampling_up(model, positive, negative, latent_dict, sigmas, **kw):
        captured.clear()
        captured.update(kw)
        captured["model"] = model
        captured["sigmas"] = sigmas
        captured["negative"] = negative
        captured["latent_shape"] = tuple(latent_dict["samples"].shape)
        return {"samples": latent_dict["samples"]}

    nodes.run_sampling = fake_run_sampling_up
    try:
        node = nodes.KreaPhotonUpscale()
        vae = FakeVAE()
        model = FakeModel()
        pos = [[torch.zeros(1, 8, 30720), {}]]
        img = torch.rand(1, 200, 300, 3)
        (out,) = node.upscale(model, pos, img, vae, 1001, "detail", 2.0)
        assert tuple(out.shape) == (1, 400, 608, 3), tuple(out.shape)
        assert vae.encoded[0][0] == (1, 400, 608, 3) and vae.encoded[0][1:] == (1024, 1024, 128)
        assert vae.decoded[0][0] == (1, 16, 1, 50, 76) and vae.decoded[0][1:] == (128, 128, 16)
        assert captured["latent_shape"] == (1, 16, 1, 50, 76), "4D encode result promoted to 5D"
        p = presets.UPSCALE_PRESETS["detail"]
        assert captured["eta0"] == 0.0 and captured["sigma_gate"] == 0.10
        assert captured.get("variety_a_latent", 0.0) == 0.0 and captured.get("variety_a_cond", 0.0) == 0.0
        assert captured["contraction"] == p["contraction"] and captured["detail_amount"] == p["detail_a"]
        assert captured["order"] == nodes._ORDER_FROM_SAMPLER_NAME[p["sampler"]]
        assert captured["guidance_mode"] == "off" and captured["negative"] is None
        assert captured["seed"] == 1001 and captured["model"] is model and captured["texture_model"] is None
        assert isinstance(captured["tiler"], tl.LatentTiler)
        t = captured["tiler"]
        assert (t.tile_h, t.tile_w, t.overlap, t.batch) == (50, 76, 16, presets.TILE["batch"]), \
            "latent smaller than a tile -> the tile is the latent (single tile)"
        assert isinstance(captured["x0_hook"], tl.LFAnchor)
        a = captured["x0_hook"]
        assert a.w_max == p["anchor"] and a.radius == presets.ANCHOR["radius"]
        assert a.release == presets.ANCHOR["release_sigma"] and a.sigma_start == float(captured["sigmas"][0])
        assert model.model.seen == [(1, 16, 1, 50, 76)], "anchor reference goes through process_latent_in"
        assert torch.equal(a.z_ref, torch.zeros(1, 16, 1, 50, 76) * 2.0)
        sig = captured["sigmas"]
        assert len(sig) == p["n_steps"] + 1 and float(sig[-1]) == 0.0 and float(sig[0]) < 0.85
        # alpha of the TILE the model sees (50x76 latent -> canonical mu below the cap)
        exp_alpha = sch.alpha_for_latent(50, 76, sch.ALPHA)
        assert torch.allclose(sig, sch.refine_schedule(p["n_steps"], alpha=exp_alpha, denoise=p["denoise"]))
        # a large image -> full 128 tiles, 2x3 grid with overlap 16
        vae2, big = FakeVAE(), torch.rand(1, 1088, 1600, 3)
        node.upscale(model, pos, big, vae2, 5, "detail", 2.0)
        t = captured["tiler"]
        assert (t.tile_h, t.tile_w) == (128, 128) and vae2.encoded[0][0] == (1, 2176, 3200, 3)
        assert len(t.boxes(272, 400)) == 12
        assert torch.allclose(captured["sigmas"],
                              sch.refine_schedule(p["n_steps"], alpha=sch.alpha_for_latent(128, 128, sch.ALPHA),
                                                  denoise=p["denoise"]))
        # negative -> window policy (even though the window sits above sigma0 - parity with the Sampler)
        node.upscale(model, pos, img, FakeVAE(), 1, "polish", 1.0, negative=pos)
        assert captured["guidance_mode"] == "window" and captured["order"] == 1
        assert captured["flat_cfg"] == presets.GUIDANCE["flat_cfg"]
        # batch of 2 images -> two runs, seeds seed+0 / seed+1, concatenated output
        seeds = []
        real_fake = nodes.run_sampling

        def seed_spy(model, positive, negative, latent_dict, sigmas, **kw):
            seeds.append(kw["seed"])
            return real_fake(model, positive, negative, latent_dict, sigmas, **kw)
        nodes.run_sampling = seed_spy
        (out2,) = node.upscale(model, pos, torch.rand(2, 64, 64, 3), FakeVAE(), 10, "polish", 1.0)
        assert tuple(out2.shape) == (2, 64, 64, 3) and seeds == [10, 11]
        nodes.run_sampling = fake_run_sampling_up
        # LoRA plan texture rule: sigma0 <= 0.65 -> the texture model runs the whole descent;
        # (the identity -> texture split path is exercised through tune below)
        real_build = nodes.build_phase_models
        tex = FakeModel("TEX")
        nodes.build_phase_models = lambda m: (None, m, tex)
        try:
            n_base = len(model.model.seen)
            node.upscale(model, pos, img, FakeVAE(), 1, "detail", 2.0)
            assert captured["model"] is tex and captured["texture_model"] is None, \
                "detail starts inside the texture phase -> whole run on the texture model"
            assert len(tex.model.seen) == 1 and len(model.model.seen) == n_base, \
                "anchor reference built on the run model"
            # every calibrated preset starts inside the texture phase (sigma0 <= 0.45)
            node.upscale(model, pos, big, FakeVAE(), 1, "strong", 2.0)
            assert captured["model"] is tex and captured["texture_model"] is None
            assert float(captured["sigmas"][0]) <= presets.UPSCALE_TEXTURE_START
            # the identity -> texture split path stays reachable through tune (sigma0 0.66 on a
            # full 1024^2 tile at denoise 0.45)
            node.upscale(model, pos, big, FakeVAE(), 1, "strong", 2.0, tune='{"denoise": 0.45, "n_steps": 12}')
            assert captured["model"] is model and captured["texture_model"] is tex
            assert captured["texture_start"] == presets.UPSCALE_TEXTURE_START
            assert float(captured["sigmas"][0]) > presets.UPSCALE_TEXTURE_START >= float(captured["sigmas"][1])
        finally:
            nodes.build_phase_models = real_build
    finally:
        nodes.run_sampling = real_run_sampling
    print("     200x300 x2 -> 400x608, single 50x76 tile; 1088x1600 x2 -> 12 tiles of 128; eta0 0; "
          "anchor w=%.2f; texture rule detail/strong; batch seeds +b" % presets.UPSCALE_PRESETS["detail"]["anchor"])

    # --- (12) v2 node: inversion + descent contract, skip/shift tiler, presets ---
    print("[12] KreaPhotonUpscaleV2 contract (mocked run_inversion / run_sampling)")
    uv2 = importlib.import_module("kreaphoton.upscale_v2")
    assert presets.validate_upscale_v2_presets() is True
    assert set(presets.UPSCALE_V2_PRESETS) == {"polish", "detail", "strong"}
    for name, q in presets.UPSCALE_V2_PRESETS.items():
        assert q["detail_a"] == 0.0 and q["anchor"] == 0.0 and q["bp_lock"] == 1.0, name
    sig_d = sch.refine_schedule(8, alpha=2.47, denoise=0.25)
    asc = uv2.inversion_sigmas(sig_d, 7)                       # 8 non-zero sigmas -> 7 inversion steps
    assert len(asc) == 8 and float(asc[0]) == float(sig_d[-2]) and float(asc[-1]) == float(sig_d[0])
    assert all(float(asc[i + 1]) > float(asc[i]) for i in range(len(asc) - 1))
    asc5 = uv2.inversion_sigmas(sig_d, 5)
    assert len(asc5) == 6 and abs(float(asc5[0]) - float(sig_d[-2])) < 1e-6 and abs(float(asc5[-1]) - float(sig_d[0])) < 1e-6
    assert len(uv2.inversion_sigmas(sig_d, 0)) == 0
    it2 = nodes.KreaPhotonUpscaleV2.INPUT_TYPES()
    assert list(it2["required"].keys()) == ["model", "positive", "image", "vae", "seed", "preset", "scale"]
    assert list(it2["optional"].keys()) == ["negative", "upscale_model", "tune"]
    assert it2["required"]["scale"][1]["min"] == 1.25
    inv_calls, samp_calls = [], []

    def fake_inv(model, positive, negative, latent_dict, sigmas_asc, **kw):
        inv_calls.append(dict(kw, sigmas=sigmas_asc, shape=tuple(latent_dict["samples"].shape)))
        return {"samples": latent_dict["samples"] + 1.0}

    def fake_samp(model, positive, negative, latent_dict, sigmas, **kw):
        samp_calls.append(dict(kw, sigmas=sigmas, latent=latent_dict["samples"]))
        return {"samples": latent_dict["samples"]}
    real_inv, real_samp = uv2.run_inversion, uv2.run_sampling
    uv2.run_inversion, uv2.run_sampling = fake_inv, fake_samp
    try:
        node2 = nodes.KreaPhotonUpscaleV2()
        q = presets.UPSCALE_V2_PRESETS["detail"]
        assert q["invert_steps"] == 0, "measured 2026-09-13: inversion off by default (no visible gain, x2 time)"
        k_inv = q["n_steps"] - 1
        (out2,) = node2.upscale(FakeModel(), pos, torch.rand(1, 200, 300, 3), FakeVAE(), 3, "detail", 2.0,
                                tune='{"invert_steps": %d}' % k_inv)
        assert tuple(out2.shape) == (1, 400, 608, 3)
        assert len(inv_calls) == 1 and len(samp_calls) == 1
        assert len(inv_calls[0]["sigmas"]) == k_inv + 1 == q["n_steps"]
        assert isinstance(inv_calls[0]["tiler"], tl.LatentTilerV2) and inv_calls[0]["tiler"] is samp_calls[0]["tiler"]
        assert samp_calls[0]["add_noise"] is False, "after inversion the descent must not add fresh noise"
        assert samp_calls[0]["eta0"] == 0.0 and samp_calls[0]["contraction"] == q["contraction"]
        assert torch.equal(samp_calls[0]["latent"], torch.zeros(1, 16, 1, 50, 76) + 1.0), "descent starts from the inverted latent"
        t = samp_calls[0]["tiler"]
        assert t.skip_threshold == q["skip"] and t.fallback is not None and tuple(t.activity.shape) == (50, 76)
        assert t.shift is True and (t.tile_h, t.tile_w, t.overlap) == (50, 76, presets.TILE_V2["overlap"] // 8)
        inv_calls.clear()
        samp_calls.clear()
        node2.upscale(FakeModel(), pos, torch.rand(1, 200, 300, 3), FakeVAE(), 3, "detail", 2.0,
                      tune='{"shift": 0, "skip": 0}')                      # preset default: no inversion
        assert not inv_calls and samp_calls[0]["add_noise"] is True
        assert samp_calls[0]["tiler"].shift is False and samp_calls[0]["tiler"].skip_threshold == 0.0
        pp, tt, aa, ee = uv2.apply_tune_v2(presets.UPSCALE_V2_PRESETS["strong"],
                                           '{"denoise": 0.3, "overlap": 96, "start_noise": 0.5}')
        assert pp["denoise"] == 0.3 and tt["overlap"] == 96 and ee["start_noise"] == 0.5 and pp["skip"] == 0.015
        try:
            uv2.apply_tune_v2(presets.UPSCALE_V2_PRESETS["strong"], '{"nope": 1}')
            raise AssertionError("unknown v2 tune key must be rejected")
        except ValueError:
            pass
    finally:
        uv2.run_inversion, uv2.run_sampling = real_inv, real_samp
    print("     inversion on the reversed grid -> descent without fresh noise, v2 tiler shared; invert_steps 0 = control")

    # --- (13) v1.6 Face Detailer: INPUT_TYPES order contract (behaviour: tests/test_face.py) ---
    print("[13] KreaPhotonFaceDetailer INPUT_TYPES order")
    it3 = nodes.KreaPhotonFaceDetailer.INPUT_TYPES()
    assert list(it3["required"].keys()) == ["model", "positive", "image", "vae", "seed", "preset", "max_faces",
                                            "identity_boost"]
    assert list(it3["optional"].keys()) == ["negative", "face_positive", "reference_image", "upscale_model", "tune"]
    assert it3["required"]["preset"][1]["default"] == presets.DEFAULT_FACE_PRESET
    assert it3["required"]["max_faces"][1]["min"] == 1 and it3["required"]["max_faces"][1]["max"] == 8
    assert nodes.KreaPhotonFaceDetailer.RETURN_TYPES == ("IMAGE", "MASK", "STRING")
    print("     required/optional order pinned; preset default %s" % presets.DEFAULT_FACE_PRESET)

    print("\ntest_nodes: ALL ASSERTS PASSED")


if __name__ == "__main__":
    main()
