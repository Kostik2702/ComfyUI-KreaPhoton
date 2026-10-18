# -*- coding: utf-8 -*-
"""Plain-assert test for node UI P3 (kreaphoton/ui_data.py): the Sampler (Advanced) sections, the
relevance rules that dim a widget without effect, and "Start from preset..." (preset_widget_values).

A relevance rule is only trusted when it is proven on the code the node runs: under the false
condition two values of the dimmed widget give the same behaviour, under the true condition they
give different behaviour (the probe is not blind). Every (node, widget, rule) of RELEVANCE must be
proven here - an unproven rule turns the coverage check red.

Run: <embedded python> tests/test_ui_relevance.py
"""
import importlib
import importlib.util
import inspect
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KREAPHOTON_DIR = os.path.join(REPO_ROOT, "kreaphoton")
COMFYUI_ROOT = r"E:\CUI portable\ComfyUI-torch2.9-cu130-cp313-v1.2\ComfyUI"
SOCKETS = ("MODEL", "CONDITIONING", "LATENT", "SIGMAS", "VAE", "CLIP", "IMAGE", "UPSCALE_MODEL", "MASK")


def _load():
    if COMFYUI_ROOT not in sys.path:
        sys.path.insert(0, COMFYUI_ROOT)
    if "kreaphoton" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "kreaphoton", os.path.join(KREAPHOTON_DIR, "__init__.py"), submodule_search_locations=[KREAPHOTON_DIR])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["kreaphoton"] = pkg
        spec.loader.exec_module(pkg)
    return {m: importlib.import_module("kreaphoton." + m)
            for m in ("ui_data", "nodes", "sampling", "guidance", "schedules", "presets", "lora_phase",
                      "server_routes")}


def widget_specs(cls):
    it = cls.INPUT_TYPES()
    spec = {**it["required"], **it.get("optional", {})}
    return {k: v for k, v in spec.items() if isinstance(v[0], list) or v[0] not in SOCKETS}


def main():
    import torch
    M = _load()
    U, N, S, G, SCH, P = (M[k] for k in ("ui_data", "nodes", "sampling", "guidance", "schedules", "presets"))
    print("=" * 78)
    print("test_ui_relevance: sections, relevance rules, Start from preset (node UI P3)")
    print("=" * 78)

    # ------------------------------------------------------------------ [1] structure
    print("[1] sections and rules name real widgets; Advanced sections cover every widget once")
    classes = {k: getattr(N, k) for k in ("KreaPhotonSampler", "KreaPhotonSamplerAdvanced", "KreaPhotonScheduler")}
    specs = {k: widget_specs(c) for k, c in classes.items()}
    adv = [w for s in U.SECTIONS["KreaPhotonSamplerAdvanced"] for w in s["widgets"]]
    assert sorted(adv) == sorted(set(specs["KreaPhotonSamplerAdvanced"]) - {"seed"}), \
        sorted(set(specs["KreaPhotonSamplerAdvanced"]) ^ set(adv + ["seed"]))
    assert len(adv) == len(set(adv)), "a widget sits in two sections"
    more = U.SECTIONS["KreaPhotonSampler"][0]["widgets"]
    opt = [k for k in N.KreaPhotonSampler.INPUT_TYPES()["optional"] if k in specs["KreaPhotonSampler"]]
    assert more == opt, ("More inputs = the simple Sampler's optional widgets, in order", more, opt)
    assert set(U.EXPERIMENTAL) <= set(adv)
    for sec in U.SECTIONS["KreaPhotonSamplerAdvanced"]:
        sm = sec["summary"]
        for w in [sm.get("widget")] + sm.get("widgets", []):
            assert w is None or w in sec["widgets"], (sec["id"], w)

    def cond_widgets(c):
        if "all" in c or "any" in c:
            return [w for x in c.get("all", c.get("any")) for w in cond_widgets(x)]
        return [] if "context" in c else [c["widget"]]

    for node, rules in U.RELEVANCE.items():
        for r in rules:
            assert set(r["widgets"]) <= set(specs[node]), (node, r["widgets"])
            assert set(cond_widgets(r["when"])) <= set(specs[node]), (node, r["when"])
            assert r["why"]
    print("     %d Advanced widgets in %d sections; More inputs = %s"
          % (len(adv), len(U.SECTIONS["KreaPhotonSamplerAdvanced"]), ", ".join(more)))

    # ------------------------------------------------------------------ [2] evaluator
    print("[2] condition evaluator")
    ch = U.condition_holds
    assert ch({"widget": "m", "in": ["a"]}, {"m": "a"}) and not ch({"widget": "m", "in": ["a"]}, {"m": "b"})
    assert ch({"widget": "x", "gt": 0}, {"x": 0.1}) and not ch({"widget": "x", "gt": 0}, {"x": 0})
    assert ch({"widget": "x", "gte": 1}, {"x": 1.0}) and not ch({"widget": "x", "gte": 1}, {"x": 0.99})
    assert ch({"widget": "x", "ne": 0}, {"x": -0.2}) and not ch({"widget": "x", "ne": 0}, {"x": 0.0})
    assert ch({"widget": "b", "is": True}, {"b": True}) and not ch({"widget": "b", "is": True}, {"b": False})
    assert ch({"context": "t"}, {}, {"t": True}) and not ch({"context": "t"}, {}, {})
    assert ch({"all": [{"widget": "x", "gt": 0}, {"widget": "b", "is": True}]}, {"x": 1, "b": True})
    assert not ch({"all": [{"widget": "x", "gt": 0}, {"widget": "b", "is": True}]}, {"x": 1, "b": False})
    assert ch({"any": [{"widget": "x", "gt": 0}, {"widget": "b", "is": True}]}, {"x": 0, "b": True})
    print("     in / gt / gte / ne / is / context / all / any")

    # ------------------------------------------------------------------ [3] per-rule proofs
    print("[3] every rule proven: no effect under the false condition, an effect under the true one")
    defaults = {node: {k: v[1].get("default") for k, v in sp.items()} for node, sp in specs.items()}
    proven = set()

    def prove(node, widget, probe, false_base, true_base, a, b, ctx_false=None, ctx_true=None):
        vf = {**defaults[node], **false_base}
        vt = {**defaults[node], **true_base}
        why = U.dimmed(node, vf, ctx_false)
        assert widget in why, ("%s.%s should be dimmed under %r" % (node, widget, false_base))
        assert widget not in U.dimmed(node, vt, ctx_true), ("%s.%s dimmed under %r" % (node, widget, true_base))
        pf_a, pf_b = probe({**vf, widget: a}, ctx_false), probe({**vf, widget: b}, ctx_false)
        assert pf_a == pf_b, "%s.%s has an effect while dimmed (%s)" % (node, widget, why[widget])
        pt_a, pt_b = probe({**vt, widget: a}, ctx_true), probe({**vt, widget: b}, ctx_true)
        assert pt_a != pt_b, "%s.%s: the probe cannot see it even when it should act - blind proof" % (node, widget)
        for r in U.RELEVANCE[node]:
            if widget in r["widgets"] and not U.condition_holds(r["when"], vf, ctx_false):
                proven.add((node, widget, id(r)))

    # (a) guidance rules: the guider's predict_noise with comfy's sampling_function captured
    real_sf, real_ccb = G.comfy.samplers.sampling_function, G.comfy.samplers.calc_cond_batch
    torch.manual_seed(3)
    t_c, t_u, t_x = torch.randn(1, 4, 6, 6), torch.randn(1, 4, 6, 6), torch.randn(1, 4, 6, 6)

    def fake_ccb(model, conds, x, sigma, model_options):
        patch = model_options["transformer_options"]["patches"]["attn1_output_patch"][-1]
        v, out = torch.ones(1, 1, 2, 2), torch.zeros(1, 2, 2)
        hits = sum(1 for k in range(28)
                   if not torch.equal(patch(out.clone(), {"block_index": k, G.PAG_STASH_KEY: v}), out))
        return [t_c * (1.0 + 0.01 * hits)]

    def fake_sf(model, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None):
        d = t_u + cond_scale * (t_c - t_u)
        outs = [round(float(cond_scale), 6)]
        for f in (model_options or {}).get("sampler_post_cfg_function", []):
            d = f({"cond_denoised": t_c, "uncond_denoised": t_u, "denoised": d, "cond": cond, "input": x,
                   "sigma": timestep, "model": None, "model_options": model_options})
            outs.append(round(float(d.sum()), 4))
        return outs

    class Patcher:
        model_options = {}

    def guide_probe(v, ctx=None):
        g = G.KreaPhotonGuider(Patcher(), mode=v["guidance_mode"], cfg=v["flat_cfg"], delta=v["delta"],
                               lo=v["guidance_lo"], hi=v["guidance_hi"], rescale=v["guidance_rescale"],
                               pag_scale=v["pag_scale"], pag_lo=v["pag_lo"], pag_hi=v["pag_hi"],
                               pag_blocks=v["pag_blocks"], guidance_sigma=v["guidance_sigma"])
        g.conds = {"positive": "P", "negative": "N"}
        g.inner_model = None
        # timestep = the detail-nudged sigma, TRUE_SIGMA_KEY = the trajectory sigma (they differ)
        return [g.predict_noise(t_x, torch.tensor([s]), {G.TRUE_SIGMA_KEY: s - 0.04})
                for s in (0.98, 0.92, 0.86, 0.8, 0.75, 0.7, 0.6, 0.3)]

    G.comfy.samplers.sampling_function, G.comfy.samplers.calc_cond_batch = fake_sf, fake_ccb
    try:
        A = "KreaPhotonSamplerAdvanced"
        prove(A, "flat_cfg", guide_probe, {"guidance_mode": "window"}, {"guidance_mode": "flat"}, 1.5, 2.0)
        for w, a, b in (("delta", 1.0, 1.25), ("guidance_lo", 0.6, 0.7), ("guidance_hi", 0.85, 0.9)):
            for mode in ("off", "flat"):
                prove(A, w, guide_probe, {"guidance_mode": mode}, {"guidance_mode": "window"}, a, b)
            prove(A, w, guide_probe, {"guidance_mode": "off"}, {"guidance_mode": "band"}, a, b)
        for mode in ("off", "flat"):
            prove(A, "guidance_sigma", guide_probe, {"guidance_mode": mode, "pag_scale": 0.0},
                  {"guidance_mode": "window"}, "model", "true")
        prove(A, "guidance_sigma", guide_probe, {"guidance_mode": "off", "pag_scale": 0.0},
              {"guidance_mode": "off", "pag_scale": 1.0}, "model", "true")
        prove(A, "guidance_rescale", guide_probe, {"guidance_mode": "off"}, {"guidance_mode": "window"}, 0.0, 0.7)
        for w, a, b in (("pag_lo", 0.72, 0.5), ("pag_hi", 0.93, 0.78), ("pag_blocks", "8-15", "6,9,12")):
            prove(A, w, guide_probe, {"pag_scale": 0.0}, {"pag_scale": 1.0}, a, b)
    finally:
        G.comfy.samplers.sampling_function, G.comfy.samplers.calc_cond_batch = real_sf, real_ccb

    # (b) loop rules: the sampler loop on a mock model whose output depends on the sigma it sees
    def mock(x, sigma, **kw):
        return 0.6 * x + 0.3 * sigma.reshape(-1, *([1] * (x.ndim - 1)))

    x5 = torch.randn(1, 16, 1, 8, 8, generator=torch.Generator().manual_seed(1))
    sig = SCH.build_schedule(12, restart_frac=0.25, sigma_r=0.65, plunge=True)

    def loop_probe(v, ctx=None):
        out = S.kreaphoton_sampler_loop(
            mock, x5.clone(), sig, detail_amount=v["detail_amount"], detail_start=v["detail_start"],
            detail_end=v["detail_end"], detail_peak=v["detail_peak"], detail_space=v["detail_space"],
            detail_sigma=(v["detail_sigma_hi"], v["detail_sigma_peak"], v["detail_sigma_lo"]),
            x0_extrapolation=v["x0_extrapolation"], x0_extrap_order=v["x0_extrap_order"], x0_trust=v["x0_trust"],
            coherence_jump=v["coherence_jump"], coherence_jump_sigma=v["coherence_jump_sigma"], restart_seed=2)
        return tuple(out.flatten().tolist())          # exact: dimmed must be bit-identical

    det = {"detail_amount": 0.4}
    for w, a, b in (("detail_start", 0.15, 0.4), ("detail_end", 0.95, 0.6), ("detail_peak", 0.6, 0.3)):
        prove(A, w, loop_probe, {"detail_amount": 0.0}, det, a, b)
        prove(A, w, loop_probe, {**det, "detail_space": "sigma"}, det, a, b)
    for w, a, b in (("detail_sigma_hi", 0.97, 0.9), ("detail_sigma_peak", 0.8, 0.6), ("detail_sigma_lo", 0.55, 0.3)):
        prove(A, w, loop_probe, {"detail_amount": 0.0, "detail_space": "sigma"}, {**det, "detail_space": "sigma"}, a, b)
        prove(A, w, loop_probe, {**det, "detail_space": "step"}, {**det, "detail_space": "sigma"}, a, b)
    prove(A, "detail_space", loop_probe, {"detail_amount": 0.0}, det, "step", "sigma")
    for w, a, b in (("x0_extrap_order", 1, 2), ("x0_trust", 0.0, 0.5)):
        prove(A, w, loop_probe, {"x0_extrapolation": 0.0}, {"x0_extrapolation": 0.5}, a, b)
    prove(A, "coherence_jump_sigma", loop_probe, {"coherence_jump": 0.0}, {"coherence_jump": 0.19}, 0.93, 0.8)

    # (c) variety: run_sampling decides the variety state, the loop then ignores end / seed when off
    pos = [[torch.randn(1, 4, G_DIM(M)), {}]]
    real_run_one = S._run_one

    def variety_probe(v, ctx=None):
        cap = {}
        S._run_one = lambda m, *a, **kw: cap.update(kw) or {"samples": None}
        try:
            S.run_sampling("M", pos, None, {"samples": x5}, sig, seed=5, variety_a_latent=v["variety_a_latent"],
                           variety_a_cond=v["variety_a_cond"], variety_end=v["variety_end"],
                           variety_seed=v["variety_seed"] if v["variety_seed"] >= 0 else 5)
        finally:
            S._run_one = real_run_one
        flags = []          # the cond axis: the step at which the guider switches to the rotated positive

        def flag_mock(x, sigma, **kw):
            flags.append(bool((kw.get("model_options") or {}).get(G.VARIETY_FLAG, False)))
            return mock(x, sigma)

        out = S.kreaphoton_sampler_loop(flag_mock, x5.clone(), sig, variety_a_latent=cap["variety_a_latent"],
                                        variety_end=cap["variety_end"], variety_seed=cap["variety_seed"],
                                        variety_state=cap["variety_state"], restart_seed=2)
        pv = cap["positive_variety"]
        return (tuple(out.flatten().tolist()), tuple(flags) if pv is not None else None,
                None if pv is None else tuple(pv[0][0].flatten().tolist()))

    for w, a, b in (("variety_end", 0.96, 0.7), ("variety_seed", 3, 9)):
        prove(A, w, variety_probe, {"variety_a_latent": 0.0, "variety_a_cond": 0.0}, {"variety_a_latent": 0.5}, a, b)
        prove(A, w, variety_probe, {"variety_a_latent": 0.0, "variety_a_cond": 0.0}, {"variety_a_cond": 0.5}, a, b)

    # (d) texture_start: only plan_segments reads it, and only with a texture model
    plain = SCH.build_schedule(16)

    def texture_probe(v, ctx=None):
        segs = []
        S._run_one = lambda m, lp, ln, ld, sg, **kw: segs.append((m, len(sg))) or {"samples": None}
        try:
            S.run_sampling("ID", None, None, {"samples": x5}, plain, seed=1, texture_start=v["texture_start"],
                           texture_model="TEX" if (ctx or {}).get("texture_model") else None)
        finally:
            S._run_one = real_run_one
        return segs

    prove(A, "texture_start", texture_probe, {}, {}, 0.65, 0.45, {"texture_model": False}, {"texture_model": True})
    # the JS context "texture model exists": texture_model linked, or a LoRA Phase whose phase splits the
    # texture set off the identity set - the condition build_phase_models uses (lora_phase.py:155-157)
    LP = M["lora_phase"]
    for ph in LP.PHASE_SEGMENTS:
        sets = LP.phase_sets([{"phase": ph, "strength": 1.0, "lora_name": "x"}])
        assert (sets["texture"] != sets["identity"]) == (ph in U.TEXTURE_SPLIT_PHASES), ph
        sets0 = LP.phase_sets([{"phase": ph, "strength": 0.0, "lora_name": "x"}])
        assert sets0["texture"] == sets0["identity"], "a strength-0 LoRA never splits"
    assert U.ui_static()["texture_split_phases"] == list(U.TEXTURE_SPLIT_PHASES)

    # (e) seed_b: _blend_noise reads seed_b only with blend > 0 (comfy.sample replaced: no model)
    fake_sample = types.ModuleType("comfy.sample")
    fake_sample.fix_empty_latent_channels = lambda m, lat: lat
    fake_sample.prepare_noise = lambda lat, seed: torch.randn(lat.shape, generator=torch.Generator().manual_seed(seed))
    real_sample = sys.modules.get("comfy.sample")
    sys.modules["comfy.sample"] = fake_sample
    import comfy
    real_attr = getattr(comfy, "sample", None)
    comfy.sample = fake_sample
    lat = {"samples": torch.zeros(1, 16, 8, 8)}

    def blend_probe(v, ctx=None):
        n = N._blend_noise("M", lat, 5, v["seed_b"], v["blend"])
        return None if n is None else round(float(n.double().sum()), 6)

    try:
        prove(A, "seed_b", blend_probe, {"blend": 0.0}, {"blend": 0.5}, 7, 11)
        prove("KreaPhotonSampler", "seed_b", blend_probe, {"blend": 0.0}, {"blend": 0.5}, 7, 11)

        # (f) simple Sampler, denoise < 1: what reaches run_sampling. restart_enhance acts through the
        # restart hook, which the loop calls only at an ascending jump (proven right below)
        cap = {}
        real_rs, real_rwp = N.run_sampling, N._result_with_preview
        N.run_sampling = lambda *a, **kw: cap.update(kw, _sigmas=a[4]) or {"samples": None}
        N._result_with_preview = lambda out, vae: (out,)

        def norm(kw):
            out = {}
            for k, val in kw.items():
                if k == "restart_hook":
                    s = [float(x) for x in kw["_sigmas"]]
                    val = val is not None and any(s[i + 1] > s[i] + 1e-6 for i in range(len(s) - 1))
                elif torch.is_tensor(val):
                    val = tuple(round(float(x), 6) for x in val.flatten()[:64].tolist())
                out[k] = val
            return out

        def sampler_probe(v, ctx=None):
            cap.clear()
            N.KreaPhotonSampler().sample("M", pos, lat, 5, "turbo/quality", "off", preview_method="none",
                                         denoise=v["denoise"], vae="VAE", seed_b=v["seed_b"], blend=v["blend"],
                                         restart_enhance=v["restart_enhance"], coherence=v["coherence"])
            return norm(cap)

        SM = "KreaPhotonSampler"
        enh = [o for o in N.KreaPhotonSampler.INPUT_TYPES()["optional"]["restart_enhance"][0] if o != "off"][0]
        lo, hi = {"denoise": 0.6}, {"denoise": 1.0}
        prove(SM, "blend", sampler_probe, {**lo, "seed_b": 7}, {**hi, "seed_b": 7}, 0.0, 0.5)
        prove(SM, "seed_b", sampler_probe, {**lo, "blend": 0.5}, {**hi, "blend": 0.5}, 7, 11)
        prove(SM, "coherence", sampler_probe, lo, hi, "off", "jump+self_refine")
        prove(SM, "restart_enhance", sampler_probe, lo, hi, "off", enh)
    finally:
        N.run_sampling, N._result_with_preview = real_rs, real_rwp
        if real_sample is None:
            sys.modules.pop("comfy.sample", None)
        else:
            sys.modules["comfy.sample"] = real_sample
        if real_attr is None:
            del comfy.sample
        else:
            comfy.sample = real_attr
    calls = []
    hook = lambda x, base: calls.append(1) or x
    S.kreaphoton_sampler_loop(mock, x5.clone(), SCH.refine_schedule(16, denoise=0.6), restart_hook=hook)
    assert calls == [], "the restart hook fired on a refine descent"
    S.kreaphoton_sampler_loop(mock, x5.clone(), sig, restart_hook=hook)
    assert len(calls) == 1, "the restart hook must fire once at the restart jump (probe would be blind)"

    # (g) Scheduler: the schedule itself
    SC = "KreaPhotonScheduler"

    def sched_probe(v, ctx=None):
        return [round(x, 6) for x in U.scheduler_sigmas(
            v["steps"], v["alpha"], v["restart_frac"], v["sigma_r"], v["plunge"], refine_steps=v["refine_steps"],
            refine_sigma=v["refine_sigma"], plunge_sigma=v["plunge_sigma"], soft_plunge=v["soft_plunge"]).tolist()]

    on = {"steps": 12, "restart_frac": 0.25, "plunge": True}
    prove(SC, "sigma_r", sched_probe, {**on, "restart_frac": 0.0}, on, 0.6, 0.5)
    prove(SC, "sigma_r", sched_probe, {**on, "steps": SCH.MIN_STEPS_FOR_RESTART - 1}, on, 0.6, 0.5)
    for w, a, b in (("refine_steps", 0, 4), ("plunge_sigma", 0.75, 0.6), ("soft_plunge", 0, 1)):
        prove(SC, w, sched_probe, {**on, "plunge": False, "refine_steps": 4 if w != "refine_steps" else 0},
              on, a, b)
    prove(SC, "refine_sigma", sched_probe, {**on, "refine_steps": 0}, {**on, "refine_steps": 4}, 0.85, 0.95)
    prove(SC, "refine_sigma", sched_probe, {**on, "plunge": False, "refine_steps": 4}, {**on, "refine_steps": 4},
          0.85, 0.95)

    every = {(node, w, id(r)) for node, rules in U.RELEVANCE.items() for r in rules for w in r["widgets"]}
    missing = sorted((n, w) for n, w, _ in every - proven)
    assert not missing, "rules without a proof: %s" % missing
    print("     %d (node, widget, rule) pairs proven; guider, loop, run_sampling, blend, Sampler, Scheduler"
          % len(every))

    # a wrong rule must fail its proof (the harness is not a rubber stamp)
    real_rules = U.RELEVANCE[A]
    U.RELEVANCE[A] = [{"widgets": ["delta"], "when": {"widget": "guidance_mode", "in": ["off"]}, "why": "planted"}]
    try:
        G.comfy.samplers.sampling_function, G.comfy.samplers.calc_cond_batch = fake_sf, fake_ccb
        prove(A, "delta", guide_probe, {"guidance_mode": "window"}, {"guidance_mode": "off"}, 1.0, 1.25)
        raise RuntimeError("a planted wrong rule passed its proof")
    except AssertionError:
        pass
    finally:
        U.RELEVANCE[A] = real_rules
        G.comfy.samplers.sampling_function, G.comfy.samplers.calc_cond_batch = real_sf, real_ccb
    print("     planted wrong rule (delta dims in window mode) fails its proof")

    # ------------------------------------------------------------------ [4] Start from preset
    print("[4] Start from preset: Advanced + Scheduler hand run_sampling what the simple Sampler does")
    real_rs = N.run_sampling
    captured = []
    N.run_sampling = lambda *a, **kw: captured.append((a, kw)) or {"samples": None}
    sig_rs = inspect.signature(S.run_sampling)

    def effective(a, kw):
        bound = sig_rs.bind(*a, **kw)
        bound.apply_defaults()
        out = dict(bound.arguments)
        out.pop("latent_dict")
        if out["detail_sigma"] is None:
            out["detail_sigma"] = tuple(P.DETAIL_SIGMA)
        out["detail_sigma"] = tuple(float(x) for x in out["detail_sigma"])
        for k, val in list(out.items()):
            if torch.is_tensor(val):
                out[k] = tuple(round(float(x), 7) for x in val.tolist())
            elif isinstance(val, float):
                out[k] = round(val, 9)
        return out

    adv_defaults = defaults["KreaPhotonSamplerAdvanced"]
    run_args = [k for k in adv_defaults if k not in ("seed",)]
    checked = 0
    try:
        for preset in P.PRESETS:
            for neg in (None, [[torch.zeros(1, 4, G_DIM(M)), {}]]):
                for profile in ("auto", "int8", "full"):
                    for hw in ((128, 96), (64, 64)):
                        latent = {"samples": torch.zeros(1, 16, *hw)}
                        captured.clear()
                        N.KreaPhotonSampler().sample("M", pos, latent, 5, preset, "off", preview_method="none",
                                                     negative=neg, precision_profile=profile)
                        vals = U.preset_widget_values(preset, profile, neg is not None)
                        sched = N.KreaPhotonScheduler().build(latent=latent, **vals["scheduler"])[0]
                        kw = {k: adv_defaults[k] for k in run_args}
                        kw.update(vals["sampler"], preview_method="none", precision_profile=profile)
                        N.KreaPhotonSamplerAdvanced().sample("M", pos, latent, sched, 5, negative=neg, **kw)
                        simple, advanced = (effective(*c) for c in captured)
                        diff = {k: (simple[k], advanced[k]) for k in simple if simple[k] != advanced[k]}
                        assert not diff, (preset, profile, neg is not None, hw, diff)
                        checked += 1
        # the comparison sees a wrong value
        captured.clear()
        latent = {"samples": torch.zeros(1, 16, 128, 96)}
        N.KreaPhotonSampler().sample("M", pos, latent, 5, "turbo/quality", "off", preview_method="none")
        vals = U.preset_widget_values("turbo/quality")
        sched = N.KreaPhotonScheduler().build(latent=latent, **vals["scheduler"])[0]
        kw = {k: adv_defaults[k] for k in run_args}
        kw.update(vals["sampler"], preview_method="none")
        kw["eta0"] += 0.01
        N.KreaPhotonSamplerAdvanced().sample("M", pos, latent, sched, 5, **kw)
        simple, advanced = (effective(*c) for c in captured)
        assert simple["eta0"] != advanced["eta0"], "the equivalence check is blind to a changed value"
    finally:
        N.run_sampling = real_rs
    print("     %d combinations (7 presets x negative x 3 profiles x 2 sizes) identical; a planted change is seen"
          % checked)

    # ------------------------------------------------------------------ [5] legal widget values
    print("[5] every value Start from preset writes is a legal widget value")
    for preset in P.PRESETS:
        for neg in (False, True):
            vals = U.preset_widget_values(preset, "auto", neg)
            for node, part in (("KreaPhotonSamplerAdvanced", "sampler"), ("KreaPhotonScheduler", "scheduler")):
                sp = specs[node]
                for k, v in vals[part].items():
                    kind, opts = sp[k][0], sp[k][1]
                    if isinstance(kind, list):
                        assert v in kind, (preset, k, v, kind)
                    elif kind == "BOOLEAN":
                        assert isinstance(v, bool), (preset, k, v)
                    elif kind == "STRING":
                        assert isinstance(v, str), (preset, k, v)
                    else:
                        assert isinstance(v, int if kind == "INT" else (int, float)), (preset, k, v)
                        assert opts.get("min", -1e18) <= v <= opts.get("max", 1e18), (preset, k, v, opts)
            assert set(vals["scheduler"]) == set(specs["KreaPhotonScheduler"]), "the Scheduler gets every widget"
            assert not set(vals["sampler"]) & {"seed", "preview_method", "variety_a_latent", "variety_a_cond",
                                               "variety_end", "variety_seed", "seed_b", "blend", "restart_enhance",
                                               "precision_profile", "texture_start"}, "a user run choice was set"
    for bad in ({"preset": "nope"}, {"profile": "fp8"}, {"negative": "maybe"}):
        try:
            M["server_routes"].preset_payload(bad)
            raise RuntimeError("accepted %r" % bad)
        except ValueError:
            pass
    pay = M["server_routes"].ui_payload()
    assert pay["defaults"]["KreaPhotonSamplerAdvanced"]["flat_cfg"] == P.GUIDANCE["flat_cfg"]
    assert set(pay["defaults"]["KreaPhotonScheduler"]) == set(specs["KreaPhotonScheduler"])
    assert pay["sections"] is U.SECTIONS and pay["relevance"] is U.RELEVANCE
    print("     combo options, ranges and types hold; routes reject bad input; /ui carries widget defaults")

    print("\ntest_ui_relevance: ALL ASSERTS PASSED")


def G_DIM(M):
    v = importlib.import_module("kreaphoton.variety")
    return v.N_TAPS * v.TAP_DIM


if __name__ == "__main__":
    main()
