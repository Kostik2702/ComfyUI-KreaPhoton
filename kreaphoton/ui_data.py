# -*- coding: utf-8 -*-
"""Data behind the node UI (node-ui-concept P2): the sigma strip, the Empty Latent readout and
the lint chips. One source for the maths - the samplers build their schedule with
sampler_sigmas / scheduler_sigmas, and describe_schedule classifies the model calls with the
sampler's own split (plan_segments, pc_step_indices), so the strip shows what the node runs.
The JS side only draws what the routes in server_routes.py return.

P3 adds the Sampler (Advanced) sections, the relevance rules that dim a widget without effect and
the values "Start from preset..." writes (preset_widget_values).
"""
from .lora_phase import PHASE_SEGMENTS
from .presets import (COHERENCE, COHERENCE_OPTIONS, DEFAULT_PRESET, DEFAULT_RESOLUTION_ASPECT,
                      DEFAULT_RESOLUTION_SIZE, LORA_PHASES, MEDIA_NOUNS, PRECISION_PROFILES, PRESETS,
                      RESOLUTION_ASPECTS, RESOLUTION_BUCKETS)
from .sampling import pc_step_indices, plan_segments
from .schedules import (MIN_STEPS_FOR_RESTART, PLUNGE_SIGMA_FLOOR, SHIFT, TAIL_SIGMA_MAX, alpha_for_latent,
                        build_schedule, refine_schedule)

# display phase of a model call; the first three are the plan's model segments
PHASES = ("composition", "identity", "draft", "texture", "refine")
TEXTURE_START = LORA_PHASES["texture"][0]     # the Sampler runs run_sampling's default, 0.65
_EPS = 1e-6


def sampler_sigmas(preset, h_lat, w_lat, coherence="off", denoise=1.0):
    """The schedule KreaPhoton Sampler runs for this preset, latent grid, coherence and denoise.
    Returns {sigmas, refine_mode, refine_steps, pc}; refine_mode / pc are what the loop gets at
    denoise 1 (a partial descent runs refine 'on' and no corrector)."""
    p = PRESETS[preset]
    alpha = alpha_for_latent(h_lat, w_lat, p["alpha"]) if SHIFT["resolution_aware"] else p["alpha"]
    # self-refine: an explicit coherence request runs it; a preset with refine 'auto'
    # (Sampler v2, quality-plus) builds the pass and lets the loop skip a stable draft
    explicit_refine = coherence in ("self_refine", "jump+self_refine")
    preset_refine_auto = p.get("refine", "off") == "auto"
    refine_steps = COHERENCE["refine_steps"] if (explicit_refine or preset_refine_auto) else 0
    refine_mode = "auto" if (preset_refine_auto and not explicit_refine) else "on"
    if denoise < 1.0:
        sigmas = refine_schedule(p["n_steps"], alpha=alpha, denoise=denoise)
    else:
        sigmas = build_schedule(p["n_steps"], alpha=alpha, restart_frac=p["restart_frac"],
                                sigma_r=p["sigma_r"], plunge=p["plunge"],
                                refine_steps=refine_steps if p["plunge"] else 0,
                                refine_sigma=COHERENCE["refine_sigma"],
                                plunge_sigma=p.get("plunge_sigma", PLUNGE_SIGMA_FLOOR),
                                soft_plunge=p.get("soft_plunge", 0))
    full = denoise >= 1.0
    return {"sigmas": sigmas, "refine_mode": refine_mode if full else "on", "refine_steps": refine_steps,
            "pc": p.get("pc", "off") if full else "off"}


def scheduler_sigmas(steps, alpha, restart_frac, sigma_r, plunge, h_lat=None, w_lat=None, refine_steps=0,
                     refine_sigma=COHERENCE["refine_sigma"], plunge_sigma=PLUNGE_SIGMA_FLOOR, soft_plunge=0):
    """The schedule KreaPhoton Scheduler outputs for these widgets (and a latent grid, if any)."""
    if h_lat is not None and w_lat is not None:
        alpha = alpha_for_latent(h_lat, w_lat, alpha)
    return build_schedule(steps, alpha=alpha, restart_frac=restart_frac, sigma_r=sigma_r, plunge=plunge,
                          refine_steps=refine_steps if plunge else 0, refine_sigma=refine_sigma,
                          plunge_sigma=float(plunge_sigma), soft_plunge=int(soft_plunge))


def _caption(counts):
    words = [("composition", "compose"), ("identity", "identity"), ("draft", "draft"), ("texture", "texture"),
             ("refine", "self-refine")]
    parts = ["%d calls" % counts["total"]]
    parts += ["%s %d" % (w, counts[k]) for k, w in words if counts[k]]
    text = ", ".join(parts)
    if counts["skippable"]:
        text += " if the draft still moves"     # refine 'auto': the loop skips them on a stable draft
    if counts["pc"]:
        text += "; +%d corrector call%s" % (counts["pc"], "" if counts["pc"] == 1 else "s")
    return text


def describe_schedule(sigmas, composition_end, refine_auto=False, pc="off"):
    """One entry per model call of the KreaPhoton loop on this SIGMAS array:
    {i, sigma, next, phase, segment, conditional, pc}.

    segment = the model that runs the call under a full phase plan (plan_segments with a clean and
    a texture model); phase refines it for display: the readout before a re-noise jump is the
    draft, the calls between a self-refine jump (landing at/above TAIL_SIGMA_MAX) and its readout
    are refine - conditional when refine_mode is auto, since the loop skips them on a stable draft.
    pc marks the steps that get a corrector call (pc_step_indices); the log counts those apart."""
    s = [float(v) for v in sigmas]
    n = len(s)
    segs = plan_segments(s, composition_end=composition_end, has_clean=True, texture_start=TEXTURE_START,
                         has_texture=True)
    seg_of = {}
    for a, b, ph in segs:
        for i in range(a, b):
            seg_of[i] = ph
    refine_idx = set()
    for k in range(n - 1):
        if s[k + 1] > s[k] + _EPS and s[k + 1] >= TAIL_SIGMA_MAX - 1e-9:
            j = k + 1
            while j < n - 1:
                refine_idx.add(j)
                if s[j + 1] <= _EPS:
                    break
                j += 1
    pc_idx = set(pc_step_indices(s, composition_end)) if pc != "off" else set()
    calls = []
    for i in range(n - 1):
        if s[i + 1] > s[i] + _EPS or s[i] <= _EPS:
            continue          # a re-noise jump or a step from a finished readout: no model call
        segment = seg_of.get(i, segs[-1][2])
        if i in refine_idx:
            phase = "refine"
        elif s[i + 1] <= _EPS and i + 1 < n - 1:
            phase = "draft"
        else:
            phase = segment
        calls.append({"i": i, "sigma": s[i], "next": s[i + 1], "phase": phase, "segment": segment,
                      "conditional": phase == "refine" and bool(refine_auto), "pc": i in pc_idx})
    counts = {k: sum(1 for c in calls if c["phase"] == k) for k in PHASES}
    counts.update(total=len(calls), skippable=sum(1 for c in calls if c["conditional"]),
                  pc=sum(1 for c in calls if c["pc"]))
    restart = None
    jumps = [k for k in range(n - 1) if s[k + 1] > s[k] + _EPS]
    if jumps and s[jumps[-1] + 1] < TAIL_SIGMA_MAX:
        restart = {"index": jumps[-1] + 1, "sigma": s[jumps[-1] + 1]}
    return {"sigmas": s, "calls": calls, "counts": counts, "restart": restart, "caption": _caption(counts)}


def shared_composition_end():
    """composition_end for precision_profile auto at edit time (the model is not known yet): the
    value every profile shares, or (int8 value, False) when they disagree."""
    values = {float(p["composition_end"]) for p in PRECISION_PROFILES.values()}
    if len(values) == 1:
        return values.pop(), True
    return float(PRECISION_PROFILES["int8"]["composition_end"]), False


def ui_static():
    """Static data for the node UI: resolution table, preset summaries, LoRA phases, media nouns."""
    return {
        "resolution": {
            "buckets": {size: {asp: [int(w), int(h)] for asp, (w, h) in tier.items()}
                        for size, tier in RESOLUTION_BUCKETS.items()},
            "aspects": list(RESOLUTION_ASPECTS),
            "default": {"size": DEFAULT_RESOLUTION_SIZE, "aspect": DEFAULT_RESOLUTION_ASPECT},
        },
        "presets": {name: {"n_steps": int(p["n_steps"]), "plunge": bool(p["plunge"]),
                           "restart_frac": float(p["restart_frac"]), "refine": p.get("refine", "off"),
                           "pc": p.get("pc", "off"), "experimental": bool(p.get("experimental", False))}
                    for name, p in PRESETS.items()},
        "default_preset": DEFAULT_PRESET,
        "coherence": list(COHERENCE_OPTIONS),
        "lora_phases": {k: list(v) for k, v in PHASE_SEGMENTS.items()},
        "lora_ranges": {k: [float(a), float(b)] for k, (a, b) in LORA_PHASES.items()},
        "media_nouns": list(MEDIA_NOUNS),
        "boundaries": {"composition_end": shared_composition_end()[0], "texture_start": float(TEXTURE_START),
                       "tail_sigma_max": float(TAIL_SIGMA_MAX)},
        "sections": SECTIONS,
        "relevance": RELEVANCE,
        "experimental": list(EXPERIMENTAL),
        "texture_split_phases": list(TEXTURE_SPLIT_PHASES),
    }


# ---------------------------------------------------------------- node UI P3
# Sections, relevance and "Start from preset..." (node-ui-concept s5-s6). The widget order is
# frozen, so a section is a set of widget names, not a range; the JS groups it on screen (the saved order stays frozen).

# summary kinds: value (a widget's value), changed (count against the reference), switch (off
# while the widget is 0, else the text with {v}), any_on (on when any widget is > 0), inputs (the
# connected sockets among `inputs`)
SECTIONS = {
    "KreaPhotonSamplerAdvanced": [
        {"id": "solver", "title": "Solver", "summary": {"kind": "value", "widget": "sampler_order"},
         "widgets": ["sampler_order", "x0_extrapolation", "x0_extrap_order", "x0_trust", "pc"]},
        {"id": "noise", "title": "Noise", "summary": {"kind": "changed"},
         "widgets": ["eta0", "sigma_gate", "eta_tail", "contraction", "per_channel_contraction"]},
        {"id": "detail", "title": "Detail envelope", "summary": {"kind": "changed"},
         "widgets": ["detail_amount", "detail_space", "detail_start", "detail_end", "detail_peak",
                     "detail_sigma_hi", "detail_sigma_peak", "detail_sigma_lo"]},
        {"id": "guidance", "title": "Guidance", "summary": {"kind": "value", "widget": "guidance_mode"},
         "widgets": ["guidance_mode", "flat_cfg", "delta", "guidance_lo", "guidance_hi", "guidance_rescale",
                     "guidance_sigma"]},
        {"id": "pag", "title": "Perturbed attention", "summary": {"kind": "switch", "widget": "pag_scale",
                                                                 "on": "scale {v}"},
         "widgets": ["pag_scale", "pag_lo", "pag_hi", "pag_blocks"]},
        {"id": "variety", "title": "Variety and blend",
         "summary": {"kind": "any_on", "widgets": ["variety_a_latent", "variety_a_cond", "blend"]},
         "widgets": ["variety_a_latent", "variety_a_cond", "variety_end", "variety_seed", "seed_b", "blend"]},
        {"id": "coherence", "title": "Coherence", "summary": {"kind": "switch", "widget": "coherence_jump",
                                                             "on": "jump {v}"},
         "widgets": ["coherence_jump", "coherence_jump_sigma", "refine_mode"]},
        {"id": "phase", "title": "Phase models",
         "summary": {"kind": "inputs", "inputs": ["clean_model", "texture_model", "vae"]},
         "widgets": ["composition_end", "texture_start", "restart_enhance", "precision_profile"]},
        {"id": "preview", "title": "Preview", "summary": {"kind": "value", "widget": "preview_method"},
         "widgets": ["preview_method"]},
    ],
    "KreaPhotonSampler": [
        {"id": "more", "title": "More inputs", "summary": {"kind": "changed"},
         "widgets": ["seed_b", "blend", "restart_enhance", "coherence", "precision_profile"]},
    ],
}

# LoRA Phase phases that give the sampler a texture model of its own: build_phase_models splits it off
# when the texture set differs from the identity set (lora_phase.py), i.e. a phase in one of the two only
TEXTURE_SPLIT_PHASES = tuple(p for p, segs in PHASE_SEGMENTS.items() if ("texture" in segs) != ("identity" in segs))

# Sampler v2 widgets that stay in their section with an "exp" tag (concept s6)
EXPERIMENTAL = ("eta_tail", "guidance_sigma", "detail_space", "detail_sigma_hi", "detail_sigma_peak",
                "detail_sigma_lo", "x0_extrap_order", "x0_trust", "pc", "refine_mode")

# A widget dims while `when` is false. Conditions: {"widget", "in" | "gt" | "gte" | "ne" | "is"},
# {"all": [...]}, {"any": [...]}, {"context": name} (a fact about the graph the JS supplies).
# headline: the reason also shows on the closed section header.
# Every rule is proven on the code the node runs by tests/test_ui_relevance.py.
_GUIDED = {"widget": "guidance_mode", "in": ["window", "band"]}
RELEVANCE = {
    "KreaPhotonSamplerAdvanced": [
        {"widgets": ["flat_cfg"], "when": {"widget": "guidance_mode", "in": ["flat"]},
         "why": "flat_cfg acts only with guidance_mode flat"},
        {"widgets": ["delta", "guidance_lo", "guidance_hi"], "when": _GUIDED,
         "why": "delta and the window act only with guidance_mode window or band"},
        {"widgets": ["guidance_sigma"], "when": {"any": [_GUIDED, {"widget": "pag_scale", "gt": 0}]},
         "why": "guidance_sigma moves the guidance window and the PAG gate: both are off"},
        {"widgets": ["guidance_rescale"], "when": {"widget": "guidance_mode", "in": ["flat", "window", "band"]},
         "why": "guidance_rescale needs a guidance mode"},
        {"widgets": ["detail_space", "detail_sigma_hi", "detail_sigma_peak", "detail_sigma_lo",
                     "detail_start", "detail_end", "detail_peak"],
         "when": {"widget": "detail_amount", "ne": 0},
         "why": "the detail envelope does nothing at detail_amount 0"},
        {"widgets": ["detail_start", "detail_end", "detail_peak"],
         "when": {"widget": "detail_space", "in": ["step"]},
         "why": "detail_start / end / peak shape the step envelope (detail_space step)"},
        {"widgets": ["detail_sigma_hi", "detail_sigma_peak", "detail_sigma_lo"],
         "when": {"widget": "detail_space", "in": ["sigma"]},
         "why": "detail_sigma_* shape the sigma envelope (detail_space sigma)"},
        {"widgets": ["x0_extrap_order", "x0_trust"], "when": {"widget": "x0_extrapolation", "gt": 0},
         "why": "x0_extrap_order and x0_trust need x0_extrapolation > 0"},
        {"widgets": ["pag_lo", "pag_hi", "pag_blocks"], "when": {"widget": "pag_scale", "gt": 0},
         "why": "the PAG window and blocks need pag_scale > 0"},
        {"widgets": ["coherence_jump_sigma"], "when": {"widget": "coherence_jump", "gt": 0},
         "why": "coherence_jump_sigma needs coherence_jump > 0"},
        {"widgets": ["seed_b"], "when": {"widget": "blend", "gt": 0}, "why": "seed_b needs blend > 0"},
        {"widgets": ["variety_end", "variety_seed"],
         "when": {"any": [{"widget": "variety_a_latent", "gt": 0}, {"widget": "variety_a_cond", "gt": 0}]},
         "why": "variety_end and variety_seed need a variety amount > 0"},
        {"widgets": ["texture_start"], "when": {"context": "texture_model"},
         "why": "texture_start needs a texture model: texture_model, or a texture / composition+identity "
                "LoRA Phase upstream"},
    ],
    "KreaPhotonSampler": [
        {"widgets": ["seed_b", "blend", "coherence", "restart_enhance"], "when": {"widget": "denoise", "gte": 1},
         "why": "Refine runs a plain descent: restart, blend and coherence are skipped", "headline": True},
        {"widgets": ["seed_b"], "when": {"widget": "blend", "gt": 0}, "why": "seed_b needs blend > 0"},
    ],
    "KreaPhotonScheduler": [
        {"widgets": ["sigma_r"],
         "when": {"all": [{"widget": "restart_frac", "gt": 0}, {"widget": "steps", "gte": MIN_STEPS_FOR_RESTART}]},
         "why": "sigma_r needs restart_frac > 0 and at least %d steps" % MIN_STEPS_FOR_RESTART},
        {"widgets": ["refine_steps", "plunge_sigma", "soft_plunge"], "when": {"widget": "plunge", "is": True},
         "why": "refine_steps, plunge_sigma and soft_plunge need plunge"},
        {"widgets": ["refine_sigma"],
         "when": {"all": [{"widget": "plunge", "is": True}, {"widget": "refine_steps", "gt": 0}]},
         "why": "refine_sigma needs plunge and refine_steps > 0"},
    ],
}


def condition_holds(cond, values, context=None):
    """Evaluate a RELEVANCE condition on widget values (and graph facts in `context`)."""
    if "all" in cond:
        return all(condition_holds(c, values, context) for c in cond["all"])
    if "any" in cond:
        return any(condition_holds(c, values, context) for c in cond["any"])
    if "context" in cond:
        return bool((context or {}).get(cond["context"]))
    v = values[cond["widget"]]
    if "in" in cond:
        return v in cond["in"]
    if "is" in cond:
        return bool(v) is cond["is"]
    if "ne" in cond:
        return float(v) != float(cond["ne"])
    if "gte" in cond:
        return float(v) >= float(cond["gte"])
    if "gt" in cond:
        return float(v) > float(cond["gt"])
    raise ValueError("unknown condition %r" % (cond,))


def dimmed(node, values, context=None):
    """{widget: why} for the widgets of `node` that have no effect under these values."""
    out = {}
    for rule in RELEVANCE.get(node, []):
        if not condition_holds(rule["when"], values, context):
            for w in rule["widgets"]:
                out.setdefault(w, rule["why"])
    return out


def _profile_values(profile):
    """(profile name, profile dict, note) for the preset values. auto at edit time (the model is
    unknown): the profile values every profile shares, or int8's with a note when they differ."""
    keys = ("detail_a_scale", "composition_end")
    if profile not in ("auto",) + tuple(PRECISION_PROFILES):
        raise ValueError("profile must be auto or one of %s, got %r" % (list(PRECISION_PROFILES), profile))
    if profile != "auto":
        return profile, PRECISION_PROFILES[profile], None
    shared = len({tuple(float(p[k]) for k in keys) for p in PRECISION_PROFILES.values()}) == 1
    return "int8", PRECISION_PROFILES["int8"], None if shared else "precision auto: int8 values shown"


def preset_widget_values(preset, profile="auto", has_negative=False):
    """What "Start from preset..." writes: the Sampler (Advanced) widgets a preset controls and the
    nine Scheduler widgets, so Advanced + Scheduler run exactly what the simple Sampler runs with
    this preset (variety off, coherence off, denoise 1). The user's own run choices (seed, preview,
    variety, blend, restart_enhance, precision_profile, texture_start) are not part of a preset.
    Values the Sampler leaves to run_sampling's defaults are read from its signature."""
    import inspect
    from .presets import DETAIL_SIGMA, GUIDANCE, preset_guidance
    from .sampling import run_sampling
    if preset not in PRESETS:
        raise ValueError("unknown preset %r" % preset)
    p = PRESETS[preset]
    name, prof, note = _profile_values(profile)
    d = {k: v.default for k, v in inspect.signature(run_sampling).parameters.items()}
    mode, cfg = preset_guidance(p, bool(has_negative))
    refine_auto = p.get("refine", "off") == "auto"
    sampler = {
        "sampler_order": p["sampler"],
        "detail_amount": float(p["detail_a"]) * float(prof["detail_a_scale"]),
        "detail_start": d["detail_start"], "detail_end": d["detail_end"], "detail_peak": d["detail_peak"],
        "eta0": float(p["eta0"]), "sigma_gate": float(p["sigma_gate"]),
        "contraction": float(p["contraction"]), "per_channel_contraction": False,
        "guidance_mode": mode, "flat_cfg": float(cfg),
        "delta": float(GUIDANCE["delta"]), "guidance_lo": float(GUIDANCE["lo"]), "guidance_hi": float(GUIDANCE["hi"]),
        "composition_end": float(prof["composition_end"]),
        "x0_extrapolation": float(p["x0_extrapolation"]),
        "guidance_rescale": d["guidance_rescale"],
        "pag_scale": d["pag_scale"], "pag_lo": d["pag_lo"], "pag_hi": d["pag_hi"], "pag_blocks": d["pag_blocks"],
        "coherence_jump": 0.0, "coherence_jump_sigma": float(COHERENCE["jump_sigma"]),
        "eta_tail": -1.0 if p.get("eta_tail") is None else float(p["eta_tail"]),
        "guidance_sigma": p.get("guidance_sigma", "model"), "detail_space": p.get("detail_space", "step"),
        "detail_sigma_hi": DETAIL_SIGMA[0], "detail_sigma_peak": DETAIL_SIGMA[1],
        "detail_sigma_lo": DETAIL_SIGMA[2],
        "x0_extrap_order": int(p.get("x0_extrap_order", 1)), "x0_trust": float(p.get("x0_trust", 0.0)),
        "pc": p.get("pc", "off"), "refine_mode": "auto" if refine_auto else "on",
    }
    scheduler = {
        "steps": int(p["n_steps"]), "alpha": float(p["alpha"]), "restart_frac": float(p["restart_frac"]),
        "sigma_r": float(p["sigma_r"]), "plunge": bool(p["plunge"]),
        "refine_steps": int(COHERENCE["refine_steps"]) if (refine_auto and p["plunge"]) else 0,
        "refine_sigma": float(COHERENCE["refine_sigma"]),
        "plunge_sigma": float(p.get("plunge_sigma", PLUNGE_SIGMA_FLOOR)),
        "soft_plunge": int(p.get("soft_plunge", 0)),
    }
    return {"preset": preset, "sampler": sampler, "scheduler": scheduler, "profile": name, "note": note,
            "scheduler_needs_latent": bool(SHIFT["resolution_aware"])}
