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


def _default_loader(lora_name: str):
    import folder_paths
    import comfy.utils
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise ValueError(f"KreaPhoton LoRA Phase: LoRA {lora_name!r} not found in models/loras")
    key = (path, os.path.getmtime(path))
    lora = _LORA_CACHE.get(key)
    if lora is None:
        lora = comfy.utils.load_torch_file(path, safe_load=True)
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


def build_phase_models(model, loader=_default_loader, apply_lora=_default_apply):
    """(clean_model, model, texture_model) for the samplers.

    No plan -> (None, model, None): the model is used as is. With a plan the
    base is the model minus the plan; each distinct LoRA set becomes one
    patcher (shared when two segments need the same set, so a plan with only
    an identity-phase LoRA yields no texture split and no extra re-patch)."""
    plan = list((getattr(model, "model_options", None) or {}).get(PLAN_KEY, []))
    if not plan:
        return None, model, None
    base = model.clone()
    base.model_options.pop(PLAN_KEY, None)
    sets = phase_sets(plan)
    built = {}

    def patcher_for(indices):
        key = tuple(indices)
        if key in built:
            return built[key]
        p = base if not key else base.clone()
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
