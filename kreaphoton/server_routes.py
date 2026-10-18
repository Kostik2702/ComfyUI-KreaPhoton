# -*- coding: utf-8 -*-
"""Server-side routes for KreaPhoton (KREA2-NODES 2026-07-08, node UI P2 2026-10-18).

GET /kreaphoton/browse - the directory listing that backs the JS folder picker in
web/kreaphoton_save.js. The browser cannot open a native OS folder dialog against the server
filesystem, so the modal walks the tree via this endpoint. Read-only: it lists sub-directories
only, never reads or writes files.

GET /kreaphoton/ui - static data for the node UI (resolution table, presets, LoRA phases, media
nouns, sections, relevance rules, widget defaults); GET /kreaphoton/schedule - the schedule a
Sampler or Scheduler with these widgets runs, one entry per model call
(ui_data.describe_schedule); GET /kreaphoton/preset_values - what "Start from preset..." writes
into Sampler (Advanced) and its Scheduler. All read-only, computed by the same functions the
nodes run, so the JS never re-implements the maths.

Import is wrapped so the plain-assert test suite (no ComfyUI/aiohttp) can import
the package without a running server."""
from .save import list_dirs
from .ui_data import (describe_schedule, preset_widget_values, sampler_sigmas, scheduler_sigmas,
                      shared_composition_end, ui_static)

_MAX_PX = 16384


def _num(query, key, kind, default, lo=None, hi=None, lo_open=False):
    raw = query.get(key)
    if raw is None or raw == "":
        return default
    try:
        v = kind(raw)
    except (TypeError, ValueError):
        raise ValueError("%s must be a %s, got %r" % (key, kind.__name__, raw))
    if lo is not None and (v <= lo if lo_open else v < lo):
        raise ValueError("%s must be %s %s, got %r" % (key, ">" if lo_open else ">=", lo, raw))
    if hi is not None and v > hi:
        raise ValueError("%s must be <= %s, got %r" % (key, hi, raw))
    return v


def _bool(query, key, default):
    raw = query.get(key)
    if raw is None or raw == "":
        return default
    if str(raw).lower() in ("1", "true", "yes", "on"):
        return True
    if str(raw).lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError("%s must be true or false, got %r" % (key, raw))


def _latent(query):
    """(h_lat, w_lat, info): from w / h in pixels, else the default Empty Latent size (assumed)."""
    from .nodes import LATENT_PX
    from .presets import DEFAULT_RESOLUTION_ASPECT, DEFAULT_RESOLUTION_SIZE, RESOLUTION_BUCKETS
    w = _num(query, "w", int, None, lo=LATENT_PX, hi=_MAX_PX)
    h = _num(query, "h", int, None, lo=LATENT_PX, hi=_MAX_PX)
    assumed = w is None or h is None
    if assumed:
        w, h = RESOLUTION_BUCKETS[DEFAULT_RESOLUTION_SIZE][DEFAULT_RESOLUTION_ASPECT]
    return h // LATENT_PX, w // LATENT_PX, {"w": int(w), "h": int(h), "assumed": assumed}


def _composition_end(profile):
    """The composition -> identity boundary of a precision profile. auto: the value every profile
    shares (the model is unknown at edit time). The research patch KREAPHOTON_PROFILE_PATCH is
    not applied here - it only exists for calibration sweeps."""
    from .precision import PROFILE_OPTIONS
    from .presets import PRECISION_PROFILES
    if profile not in PROFILE_OPTIONS:
        raise ValueError("profile must be one of %s, got %r" % (PROFILE_OPTIONS, profile))
    if profile == "auto":
        return shared_composition_end()
    return float(PRECISION_PROFILES[profile]["composition_end"]), True


def schedule_payload(query):
    """The /kreaphoton/schedule response for a query dict; ValueError on bad input."""
    from .presets import COHERENCE_OPTIONS, DEFAULT_PRESET, PRESETS
    node = query.get("node") or "sampler"
    ce, ce_shared = _composition_end(query.get("profile") or "auto")
    h_lat, w_lat, latent = _latent(query)
    if node == "sampler":
        preset = query.get("preset") or DEFAULT_PRESET
        if preset not in PRESETS:
            raise ValueError("unknown preset %r" % preset)
        coherence = query.get("coherence") or "off"
        if coherence not in COHERENCE_OPTIONS:
            raise ValueError("coherence must be one of %s, got %r" % (list(COHERENCE_OPTIONS), coherence))
        denoise = _num(query, "denoise", float, 1.0, lo=0.0, hi=1.0, lo_open=True)
        s = sampler_sigmas(preset, h_lat, w_lat, coherence=coherence, denoise=denoise)
        out = describe_schedule(s["sigmas"], ce, refine_auto=s["refine_mode"] == "auto", pc=s["pc"])
        out.update(preset=preset, coherence=coherence, denoise=denoise)
    elif node == "scheduler":
        # defaults and ranges are the Scheduler node's own widget specs
        from .nodes import KreaPhotonScheduler
        it = KreaPhotonScheduler.INPUT_TYPES()
        spec = {**it["required"], **it.get("optional", {})}

        def widget(key, kind):
            opts = spec[key][1]
            return _num(query, key, kind, opts["default"], lo=opts.get("min"), hi=opts.get("max"))

        sigmas = scheduler_sigmas(
            widget("steps", int), widget("alpha", float), widget("restart_frac", float),
            widget("sigma_r", float), _bool(query, "plunge", spec["plunge"][1]["default"]),
            h_lat=None if latent["assumed"] else h_lat, w_lat=None if latent["assumed"] else w_lat,
            refine_steps=widget("refine_steps", int), refine_sigma=widget("refine_sigma", float),
            plunge_sigma=widget("plunge_sigma", float), soft_plunge=widget("soft_plunge", int))
        out = describe_schedule(sigmas, ce)
    else:
        raise ValueError("node must be sampler or scheduler, got %r" % node)
    out.update(node=node, latent=latent, composition_end=ce, composition_end_shared=ce_shared)
    return out


# nodes whose widget defaults the UI needs (changed dots, "More inputs" count)
_DEFAULT_NODES = ("KreaPhotonSampler", "KreaPhotonSamplerAdvanced", "KreaPhotonScheduler")


def ui_payload():
    """The /kreaphoton/ui response: ui_static() plus each widget's INPUT_TYPES default."""
    from . import nodes
    out = ui_static()
    defaults = {}
    for name in _DEFAULT_NODES:
        it = getattr(nodes, name).INPUT_TYPES()
        spec = {**it["required"], **it.get("optional", {})}
        defaults[name] = {k: v[1]["default"] for k, v in spec.items()
                          if len(v) > 1 and isinstance(v[1], dict) and "default" in v[1]}
    out["defaults"] = defaults
    return out


def preset_payload(query):
    """The /kreaphoton/preset_values response; ValueError on bad input."""
    from .presets import DEFAULT_PRESET
    return preset_widget_values(query.get("preset") or DEFAULT_PRESET, query.get("profile") or "auto",
                                _bool(query, "negative", False))


_REGISTERED = False
try:
    from server import PromptServer
    from aiohttp import web

    # PromptServer.instance is created at ComfyUI startup, before custom nodes
    # load - but may be absent when the module is imported standalone (unit
    # tests / tooling). Guard on it so a bare import is always a safe no-op.
    if getattr(PromptServer, "instance", None) is not None:

        @PromptServer.instance.routes.get("/kreaphoton/browse")
        async def kreaphoton_browse(request):
            """?path=<dir> -> {path, parent, dirs:[{name,path}], error?}.
            Empty path -> drive roots (Windows) / filesystem root (POSIX)."""
            return web.json_response(list_dirs(request.query.get("path", "")))

        @PromptServer.instance.routes.get("/kreaphoton/ui")
        async def kreaphoton_ui(request):
            return web.json_response(ui_payload())

        @PromptServer.instance.routes.get("/kreaphoton/schedule")
        async def kreaphoton_schedule(request):
            """?node=sampler&preset=&coherence=&denoise=&w=&h=&profile= | ?node=scheduler&steps=&alpha=
            &restart_frac=&sigma_r=&plunge=&refine_steps=&refine_sigma=&plunge_sigma=&soft_plunge=&w=&h="""
            try:
                return web.json_response(schedule_payload(dict(request.query)))
            except ValueError as e:
                return web.json_response({"error": str(e)}, status=400)

        @PromptServer.instance.routes.get("/kreaphoton/preset_values")
        async def kreaphoton_preset_values(request):
            """?preset=&profile=&negative= -> the widget values "Start from preset..." writes."""
            try:
                return web.json_response(preset_payload(dict(request.query)))
            except ValueError as e:
                return web.json_response({"error": str(e)}, status=400)

        _REGISTERED = True
except ImportError:
    # No ComfyUI server (e.g. unit tests) - routes simply not registered.
    pass
