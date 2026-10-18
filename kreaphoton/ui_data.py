# -*- coding: utf-8 -*-
"""Data behind the node UI (node-ui-concept P2): the sigma strip, the Empty Latent readout and
the lint chips. One source for the maths - the samplers build their schedule with
sampler_sigmas / scheduler_sigmas, and describe_schedule classifies the model calls with the
sampler's own split (plan_segments, pc_step_indices), so the strip shows what the node runs.
The JS side only draws what the routes in server_routes.py return.
"""
from .lora_phase import PHASE_SEGMENTS
from .presets import (COHERENCE, COHERENCE_OPTIONS, DEFAULT_PRESET, DEFAULT_RESOLUTION_ASPECT,
                      DEFAULT_RESOLUTION_SIZE, LORA_PHASES, MEDIA_NOUNS, PRECISION_PROFILES, PRESETS,
                      RESOLUTION_ASPECTS, RESOLUTION_BUCKETS)
from .sampling import pc_step_indices, plan_segments
from .schedules import (PLUNGE_SIGMA_FLOOR, SHIFT, TAIL_SIGMA_MAX, alpha_for_latent, build_schedule,
                        refine_schedule)

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
    }
