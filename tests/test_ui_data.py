# -*- coding: utf-8 -*-
"""UI data for the node redesign P2 (node-ui-concept section 7.3): the sigma strip and the
routes describe exactly the schedule the nodes run - one source for the maths.

Run: <embedded python> tests/test_ui_data.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_nodes import _load_kreaphoton_package  # noqa: E402


def old_sampler_sigmas(P, SC, preset, h_lat, w_lat, coherence, denoise):
    """The inline construction of KreaPhotonSampler.sample before P2 (master 83be921), copied
    verbatim as the bit-exact reference for the extracted function."""
    p = P.PRESETS[preset]
    alpha = SC.alpha_for_latent(h_lat, w_lat, p["alpha"]) if SC.SHIFT["resolution_aware"] else p["alpha"]
    explicit_refine = coherence in ("self_refine", "jump+self_refine")
    preset_refine_auto = p.get("refine", "off") == "auto"
    refine_steps = P.COHERENCE["refine_steps"] if (explicit_refine or preset_refine_auto) else 0
    refine_mode = "auto" if (preset_refine_auto and not explicit_refine) else "on"
    if denoise < 1.0:
        sigmas = SC.refine_schedule(p["n_steps"], alpha=alpha, denoise=denoise)
    else:
        sigmas = SC.build_schedule(p["n_steps"], alpha=alpha, restart_frac=p["restart_frac"],
                                   sigma_r=p["sigma_r"], plunge=p["plunge"],
                                   refine_steps=refine_steps if p["plunge"] else 0,
                                   refine_sigma=P.COHERENCE["refine_sigma"],
                                   plunge_sigma=p.get("plunge_sigma", SC.PLUNGE_SIGMA_FLOOR),
                                   soft_plunge=p.get("soft_plunge", 0))
    return sigmas, refine_mode


def main():
    nodes, P = _load_kreaphoton_package()
    SC = importlib.import_module("kreaphoton.schedules")
    U = importlib.import_module("kreaphoton.ui_data")
    LP = importlib.import_module("kreaphoton.lora_phase")

    print("=" * 78)
    print("test_ui_data: sigma strip data == the schedule the nodes run")
    print("=" * 78)

    lat_sizes = [(h // 8, w // 8) for tier in P.RESOLUTION_BUCKETS.values() for (w, h) in tier.values()]

    print("[1] sampler_sigmas bit-exact with the old inline Sampler code")
    n = 0
    for preset in P.PRESETS:
        for coherence in P.COHERENCE_OPTIONS:
            for denoise in (1.0, 0.6):
                for h, w in lat_sizes:
                    ref, ref_mode = old_sampler_sigmas(P, SC, preset, h, w, coherence, denoise)
                    got = U.sampler_sigmas(preset, h, w, coherence=coherence, denoise=denoise)
                    assert got["sigmas"].tolist() == ref.tolist(), (preset, coherence, denoise, h, w)
                    # the value the loop receives (the old call gated it on denoise at run_sampling)
                    assert got["refine_mode"] == (ref_mode if denoise >= 1.0 else "on"), (preset, coherence)
                    assert got["pc"] == (P.PRESETS[preset].get("pc", "off") if denoise >= 1.0 else "off")
                    n += 1
    print("     %d combinations identical" % n)

    print("[2] scheduler_sigmas == KreaPhoton Scheduler node output")
    sched = nodes.KreaPhotonScheduler()
    for args in [(16, 3.158, 0.25, 0.65, True), (36, 3.158, 0.2, 0.45, False), (8, 2.0, 0.0, 0.6, True)]:
        for refine_steps in (0, 4):
            node_out = sched.build(*args, refine_steps=refine_steps)[0]
            got = U.scheduler_sigmas(*args, refine_steps=refine_steps)
            assert got.tolist() == node_out.tolist(), (args, refine_steps)
    print("     identical")

    print("[3] describe_schedule: one cell per model call, phases from the sampler's own split")
    for preset in P.PRESETS:
        for coherence in P.COHERENCE_OPTIONS:
            for denoise in (1.0, 0.6):
                s = U.sampler_sigmas(preset, 200, 136, coherence=coherence, denoise=denoise)
                d = U.describe_schedule(s["sigmas"], composition_end=0.85,
                                        refine_auto=s["refine_mode"] == "auto", pc=s["pc"])
                assert len(d["calls"]) == SC.count_model_calls(s["sigmas"]), (preset, coherence, denoise)
                assert d["counts"]["total"] == len(d["calls"])
                assert sum(d["counts"][k] for k in U.PHASES) == len(d["calls"]), d["counts"]
    q = U.describe_schedule(U.sampler_sigmas("turbo/quality", 200, 136)["sigmas"], composition_end=0.85)
    assert {k: q["counts"][k] for k in U.PHASES} == {"composition": 8, "identity": 3, "draft": 1, "texture": 4,
                                                    "refine": 0}, q["counts"]
    assert q["caption"] == "16 calls, compose 8, identity 3, draft 1, texture 4", q["caption"]
    assert q["restart"] is not None and abs(q["restart"]["sigma"] - 0.65) < 1e-6, q["restart"]
    assert [c["phase"] for c in q["calls"]][11] == "draft" and q["calls"][11]["next"] == 0.0
    print("     turbo/quality: %s" % q["caption"])

    print("[4] quality-plus: the self-refine cells are conditional, corrector steps marked")
    sp = U.sampler_sigmas("turbo/quality-plus", 200, 136)
    qp = U.describe_schedule(sp["sigmas"], composition_end=0.85, refine_auto=sp["refine_mode"] == "auto",
                             pc=sp["pc"])
    refine = [c for c in qp["calls"] if c["phase"] == "refine"]
    assert len(refine) == P.COHERENCE["refine_steps"] and all(c["conditional"] for c in refine), refine
    assert not any(c["conditional"] for c in qp["calls"] if c["phase"] != "refine")
    assert qp["counts"]["total"] == 20 and qp["counts"]["skippable"] == 4, qp["counts"]
    assert qp["counts"]["pc"] == sum(1 for c in qp["calls"] if c["pc"]) and qp["counts"]["pc"] > 0
    assert "if the draft still moves" in qp["caption"], qp["caption"]
    # an explicit self-refine request runs the pass: nothing conditional
    se = U.sampler_sigmas("turbo/quality", 200, 136, coherence="self_refine")
    de = U.describe_schedule(se["sigmas"], composition_end=0.85, refine_auto=se["refine_mode"] == "auto")
    assert de["counts"]["refine"] == 4 and de["counts"]["skippable"] == 0, de["counts"]
    print("     %s" % qp["caption"])

    print("[5] planted error: the classifier must not agree with a wrong split")
    real = U.plan_segments
    try:
        U.plan_segments = lambda sig, **kw: [(0, len(sig) - 1, "identity")]
        bad = U.describe_schedule(U.sampler_sigmas("turbo/quality", 200, 136)["sigmas"], composition_end=0.85)
        assert bad["counts"]["composition"] != 8, "describe_schedule ignores the sampler's plan_segments"
    finally:
        U.plan_segments = real
    print("     a swapped split changes the counts (describe_schedule reads plan_segments)")

    print("[6] ui_static: buckets, presets, LoRA phases, media nouns from the Python sources")
    st = U.ui_static()
    assert st["resolution"]["buckets"]["L (~1.7 MP)"]["3:2"] == [1088, 1600]
    assert st["resolution"]["default"] == {"size": P.DEFAULT_RESOLUTION_SIZE, "aspect": P.DEFAULT_RESOLUTION_ASPECT}
    assert set(st["presets"]) == set(P.PRESETS)
    assert st["presets"]["turbo/quality-plus"]["refine"] == "auto"
    assert set(st["lora_phases"]) == set(LP.PHASE_SEGMENTS)
    for k, v in st["lora_phases"].items():
        assert v == list(LP.PHASE_SEGMENTS[k]), (k, v)
    assert st["media_nouns"] == list(P.MEDIA_NOUNS) and "magazine" in st["media_nouns"]
    assert st["boundaries"]["composition_end"] == 0.85
    print("     ok")

    print("[7] schedule_payload: the route's pure handler")
    R = importlib.import_module("kreaphoton.server_routes")
    a = R.schedule_payload({"node": "sampler", "preset": "turbo/quality", "w": "1088", "h": "1600"})
    assert a["counts"]["total"] == 16 and a["latent"] == {"w": 1088, "h": 1600, "assumed": False}, a["latent"]
    b = R.schedule_payload({"node": "sampler"})
    assert b["preset"] == P.DEFAULT_PRESET and b["latent"]["assumed"] is True
    assert (b["latent"]["w"], b["latent"]["h"]) == tuple(P.RESOLUTION_BUCKETS["L (~1.7 MP)"]["3:2"])
    c = R.schedule_payload({"node": "scheduler", "steps": "16", "alpha": "3.158", "restart_frac": "0.25",
                            "sigma_r": "0.65", "plunge": "true"})
    assert c["sigmas"] == sched.build(16, 3.158, 0.25, 0.65, True)[0].tolist()
    d6 = R.schedule_payload({"node": "sampler", "preset": "turbo/quality", "denoise": "0.6"})
    # a partial descent: no plunge readout, no self-refine; below texture_start the plan's texture
    # model runs it (plan_segments' descending fallback), the strip shows that routing
    assert d6["counts"]["draft"] == 0 and d6["counts"]["refine"] == 0 and d6["counts"]["total"] == 16, d6["counts"]
    assert all(c["phase"] == "texture" for c in d6["calls"] if c["sigma"] <= 0.65), d6["calls"]
    for bad in ({"node": "sampler", "preset": "nope"}, {"node": "x"}, {"node": "sampler", "w": "abc"},
                {"node": "sampler", "denoise": "0"}, {"node": "scheduler", "steps": "0"}):
        try:
            R.schedule_payload(bad)
            raise RuntimeError("accepted %r" % bad)
        except ValueError:
            pass
    print("     defaults, scheduler == node, bad input -> ValueError")

    print("\ntest_ui_data: ALL PASSED")


if __name__ == "__main__":
    main()
