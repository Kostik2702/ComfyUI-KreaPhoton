"""KreaPhoton node definitions.

TEMPORARY (S3.5 tracer-bullet, planning-council H27/H28): KreaPhotonTracerSampler
below is a throwaway diagnostic node, NOT part of the v1 scope (docs/04). It
exists to prove KreaPhotonGuider drives a REAL guided generation correctly
(NFE invariant, no composition-level bug that unit tests are structurally
blind to - prior incident: ZIT-Hires-TWOPASS CFG double-count survived 80
unit tests, only surfaced at first real generation). Removed once S7 builds
the real KreaPhoton Sampler nodes.
"""
import comfy.model_management
import comfy.sample
import comfy.samplers
import torch

from .guidance import KreaPhotonGuider
from .schedules import build_schedule

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class KreaPhotonTracerSampler:
    """S3.5(b) tracer only. Plain structure schedule (no restart), KreaPhotonGuider,
    stock euler integrator - built by hand exactly like SamplerCustomAdvanced
    (comfy_extras/nodes_custom_sampler.py:1013), since comfy.samplers.sample()
    hardcodes the stock CFGGuider (D11) and cannot be used with a custom guider."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 64}),
                "delta": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 3.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "nfe_log")
    FUNCTION = "sample"
    CATEGORY = "KreaPhoton/tracer"

    def sample(self, model, positive, negative, latent_image, seed, steps, delta):
        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        assert latent.ndim == 5, f"expected 5D Wan21 latent, got shape {tuple(latent.shape)}"

        sigmas = build_schedule(steps, restart_frac=0.0, plunge=False).to(torch.float32)

        noise = comfy.sample.prepare_noise(latent, seed)

        # --- NFE-invariant instrumentation (planning-council D14): count
        # non-None entries in the `conds` list per predict_noise call. Each
        # non-None entry contributes exactly B rows into the model forward
        # regardless of comfy's internal batch-fusion strategy (samplers.py:233-305),
        # so this is a valid proxy for "Sigma batch-dim / B" without needing to
        # dig into hooked_to_run/to_batch internals. ---
        nfe_log = []
        orig_calc_cond_batch = comfy.samplers.calc_cond_batch

        def _counting_calc_cond_batch(model_, conds, x_in, timestep, model_options):
            n_active = sum(1 for c in conds if c is not None)
            sigma_val = float(timestep.flatten()[0].item())
            nfe_log.append((round(sigma_val, 4), n_active))
            return orig_calc_cond_batch(model_, conds, x_in, timestep, model_options)

        comfy.samplers.calc_cond_batch = _counting_calc_cond_batch
        try:
            guider = KreaPhotonGuider(model, delta=delta)
            guider.set_conds(positive, negative)
            sampler = comfy.samplers.sampler_object("euler")
            samples = guider.sample(noise, latent, sampler, sigmas, seed=seed)
        finally:
            comfy.samplers.calc_cond_batch = orig_calc_cond_batch

        samples = samples.to(device=comfy.model_management.intermediate_device(),
                              dtype=comfy.model_management.intermediate_dtype())

        log_lines = ["sigma=%.4f  n_active=%d  (%s window)" %
                     (s, n, "IN" if n == 2 else "OUT") for s, n in nfe_log]
        n_in = sum(1 for _, n in nfe_log if n == 2)
        n_out = sum(1 for _, n in nfe_log if n == 1)
        summary = ("KreaPhotonTracerSampler NFE log (delta=%.2f, steps=%d)\n" % (delta, steps)
                   + "\n".join(log_lines)
                   + "\n\nsummary: %d steps IN window (n_active=2), %d steps OUT (n_active=1)" % (n_in, n_out))

        return ({"samples": samples}, summary)


NODE_CLASS_MAPPINGS["KreaPhotonTracerSampler"] = KreaPhotonTracerSampler
NODE_DISPLAY_NAME_MAPPINGS["KreaPhotonTracerSampler"] = "KreaPhoton Tracer Sampler (S3.5, temporary)"
