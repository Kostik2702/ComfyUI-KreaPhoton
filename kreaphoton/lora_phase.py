"""
LoRA phase scheduling by phase MODELS (v1.4.1).

The KreaPhoton LoRA Phase node does not patch anything: it clones the model and
appends {lora_name, strength, phase} to a plan stored in model_options (clone()
deep-copies list/dict options, so the plan never leaks into the original model).
When a KreaPhoton sampler receives a model carrying a plan it calls
build_phase_models(): the plan is expanded into up to three ModelPatchers -
composition / identity / texture - each with exactly the LoRAs whose phase
covers that segment, applied through comfy's ordinary add_patches path (the
same one LoRA loaders use, so int8 / fp8-quantized checkpoints work; comfy
weight *hooks* cannot patch those - V18 D, 2026-09-07). The samplers then run
the validated multi-segment split (sampling.plan_segments) on those models.

Phase -> segments:
  all                   composition + identity + texture   (classic loader)
  composition           composition                         sigma 1.0 -> 0.85
  identity              identity + texture                  sigma 0.85 -> 0
  texture               texture                             restart segment, sigma 0.65 -> 0
  composition+identity  composition + identity              sigma 1.0 -> 0.65
"""
import os

PLAN_KEY = "kreaphoton_lora_plan"
SEGMENTS = ("composition", "identity", "texture")
PHASE_SEGMENTS = {
    "all": ("composition", "identity", "texture"),
    "composition": ("composition",),
    "identity": ("identity", "texture"),
    "texture": ("texture",),
    "composition+identity": ("composition", "identity"),
}

_LORA_CACHE = {}          # (path, mtime) -> lora state dict
_LORA_CACHE_MAX = 8


def add_to_plan(model, lora_name: str, strength: float, phase: str):
    """Clone `model` and append one plan entry. Pure w.r.t. the input model."""
    if phase not in PHASE_SEGMENTS:
        raise ValueError(f"KreaPhoton LoRA Phase: unknown phase {phase!r} (choose from {list(PHASE_SEGMENTS)})")
    m = model.clone()
    plan = list(m.model_options.get(PLAN_KEY, []))
    plan.append({"lora_name": str(lora_name), "strength": float(strength), "phase": str(phase)})
    m.model_options[PLAN_KEY] = plan
    return m


def phase_sets(plan):
    """segment -> tuple of plan indices active in that segment (order preserved)."""
    out = {seg: [] for seg in SEGMENTS}
    for i, entry in enumerate(plan):
        if float(entry.get("strength", 0.0)) == 0.0:
            continue
        for seg in PHASE_SEGMENTS[entry["phase"]]:
            out[seg].append(i)
    return {seg: tuple(v) for seg, v in out.items()}


_TRAINER_FLAGS = {"use_rslora_flag": "rsLoRA (alpha/sqrt(rank) scaling)", "use_dora_flag": "DoRA"}


def strip_trainer_flags(lora: dict, name: str) -> dict:
    """blissful-tuner stores two scalar flags in the state dict; ComfyUI maps only weights and
    warns 'lora key not loaded' for each. Flag 0 = a plain LoRA, scaled exactly as ComfyUI
    scales it (alpha / rank) -> drop the flags. Flag 1 = a scaling ComfyUI does not implement
    (no rsLoRA support at all; DoRA only via dora_scale keys) -> refuse, a silent sqrt(rank)
    weaker character LoRA is the worst failure for likeness (2026-10-01)."""
    flags = [k for k in lora if k.rsplit(".", 1)[-1] in _TRAINER_FLAGS]
    if not flags:
        return lora
    for k in flags:
        if float(lora[k]) != 0.0:
            raise ValueError(f"KreaPhoton LoRA Phase: {name} was trained with "
                             f"{_TRAINER_FLAGS[k.rsplit('.', 1)[-1]]}, which ComfyUI does not apply - "
                             f"the LoRA would act with the wrong strength")
    return {k: v for k, v in lora.items() if k not in flags}


def _default_loader(lora_name: str):
    import folder_paths
    import comfy.utils
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise ValueError(f"KreaPhoton LoRA Phase: LoRA {lora_name!r} not found in models/loras")
    key = (path, os.path.getmtime(path))
    lora = _LORA_CACHE.get(key)
    if lora is None:
        lora = strip_trainer_flags(comfy.utils.load_torch_file(path, safe_load=True), lora_name)
        if len(_LORA_CACHE) >= _LORA_CACHE_MAX:
            _LORA_CACHE.pop(next(iter(_LORA_CACHE)))
        _LORA_CACHE[key] = lora
    return lora


def _default_apply(patcher, lora, strength: float):
    """Ordinary LoRA application (comfy.sd.load_lora_for_models, model-only)."""
    import comfy.lora
    import comfy.lora_convert
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map)
    patcher.add_patches(loaded, strength)
    return patcher


def _plan_free_clone(model):
    """A clone of `model` without the plan (clone() deep-copies model_options, so the
    input keeps its plan)."""
    p = model.clone()
    p.model_options.pop(PLAN_KEY, None)
    return p


def build_phase_models(model, loader=_default_loader, apply_lora=_default_apply):
    """(clean_model, model, texture_model) for the samplers.

    No plan -> (None, model, None): the model is used as is. With a plan the
    base is the model minus the plan; each distinct LoRA set becomes one
    patcher (shared when two segments need the same set, so a plan with only
    an identity-phase LoRA yields no texture split and no extra re-patch).

    Every patcher is a DIRECT clone of `model`, never a clone of a clone: when a
    loaded patcher dies, comfy's LoadedModel re-points to its parent through ONE
    weakref level; a two-level chain of throwaway clones freed by the cyclic GC
    in one pass leaves the entry dead - 'memory leak with model Krea2', dead
    entries piling up and a full gc.collect() per model load (2026-10-01,
    research/precision/leak_mech.py)."""
    plan = list((getattr(model, "model_options", None) or {}).get(PLAN_KEY, []))
    if not plan:
        return None, model, None
    base = _plan_free_clone(model)
    sets = phase_sets(plan)
    built = {}

    def patcher_for(indices):
        key = tuple(indices)
        if key in built:
            return built[key]
        p = base if not key else _plan_free_clone(model)
        for i in key:
            entry = plan[i]
            p = apply_lora(p, loader(entry["lora_name"]), float(entry["strength"]))
        built[key] = p
        return p

    identity = patcher_for(sets["identity"])
    clean = patcher_for(sets["composition"])
    texture = patcher_for(sets["texture"])
    return (clean if sets["composition"] != sets["identity"] else None,
            identity,
            texture if sets["texture"] != sets["identity"] else None)


IDENTITY_PHASES = ("all", "identity")   # plan phases that carry a character (face/body) LoRA


def build_face_model(model, boost: float, loader=_default_loader, apply_lora=_default_apply):
    """(patcher, boosted, unchanged) for the Face Detailer crop pass, or (None, [], [])
    when the model carries no plan.

    The crop pass lives in the texture segment (sigma <= ~0.66), so the patcher holds
    exactly the plan's texture set - like build_phase_models' texture patcher - but
    every entry whose phase is in IDENTITY_PHASES runs at strength x boost. Measured
    2026-09-16 on the owner's graph (AnnaMin_v3, 'all', 1.0): at x1.0 every redraw
    LOSES likeness to the character (ArcFace to the dataset centroid 0.557 -> 0.47-0.49),
    at x1.5 it gains (-> 0.62-0.66); the LoRA set itself (identity / texture / all six)
    made no difference. `boosted` / `unchanged` list (lora_name, strength) for the report."""
    plan = list((getattr(model, "model_options", None) or {}).get(PLAN_KEY, []))
    if not plan:
        return None, [], []
    base = model.clone()
    base.model_options.pop(PLAN_KEY, None)
    boosted, unchanged = [], []
    for i in phase_sets(plan)["texture"]:
        entry = plan[i]
        strength = float(entry["strength"])
        if entry["phase"] in IDENTITY_PHASES:
            strength *= float(boost)
            boosted.append((entry["lora_name"], strength))
        else:
            unchanged.append((entry["lora_name"], strength))
        base = apply_lora(base, loader(entry["lora_name"]), strength)
    return base, boosted, unchanged
