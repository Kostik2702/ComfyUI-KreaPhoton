"""
Sigma-adaptive guidance window (M6) — the ONLY sanctioned way to implement
phase-windowed guidance for KreaPhoton (planning-council B4/D10-D14/H29).

Ground truth (E:\\CUI portable\\ComfyUI-torch2.9-cu130-cp313-v1.2\\ComfyUI):
  comfy/samplers.py:609-612  sampling_function(): at math.isclose(cond_scale,1.0)
                              and disable_cfg1_optimization not set -> uncond_=None,
                              ZERO extra NFE for that step (stock optimization).
  comfy/samplers.py:1211-1212 CFGGuider.predict_noise (the ONE method we override)
  comfy/samplers.py:592-602   sampler_cfg_function / sampler_post_cfg_function hooks
                              receive a ZEROED uncond at cfg=1.0 (calc_cond_batch
                              leaves the unconsumed accumulator at zeros) — silent
                              garbage for exactly our default (cfg=1.0) path.

RULE: guidance/variety windows are NEVER implemented via model_options hooks
(sampler_cfg_function / sampler_post_cfg_function). Only a CFGGuider subclass
sees whether uncond was actually computed this step.
"""
import comfy.samplers
import torch


def smoothstep01(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def g_window(sigma: float, delta: float, lo: float = 0.7, hi: float = 0.9) -> float:
    """g(sigma) = 1 + delta * smoothstep((sigma-lo)/(hi-lo)).

    MUST return exactly 1.0 outside [lo, hi] (smoothstep01 clamps its argument to
    [0,1] before the cubic, so u<=0 -> smoothstep=0 -> g=1.0 exactly; no epsilon
    residue, unlike a sigmoid-form window would leave). This is what lets the
    stock cfg1-optimization (samplers.py:609) fire for free outside the window.
    """
    return 1.0 + delta * smoothstep01((sigma - lo) / (hi - lo))


class KreaPhotonGuider(comfy.samplers.CFGGuider):
    """CFGGuider subclass implementing the M6 guidance window.

    Overrides ONLY predict_noise (comfy/samplers.py:1211-1212 is a 1-line method;
    everything else — prepare_sampling, process_conds, device management, wrapper
    executors — is inherited unchanged, verified against the live 0.26 source).

    self.conds is populated by the base class's inner_sample/process_conds before
    any predict_noise call; never read self.original_conds here.
    """

    def __init__(self, model_patcher, delta: float, lo: float = 0.7, hi: float = 0.9):
        super().__init__(model_patcher)
        self.delta = delta
        self.lo = lo
        self.hi = hi

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        if model_options is None:
            model_options = {}
        sigma = float(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else float(timestep)
        cond_scale = g_window(sigma, self.delta, self.lo, self.hi)
        return comfy.samplers.sampling_function(
            self.inner_model, x, timestep,
            self.conds.get("negative", None), self.conds.get("positive", None),
            cond_scale, model_options=model_options, seed=seed,
        )
