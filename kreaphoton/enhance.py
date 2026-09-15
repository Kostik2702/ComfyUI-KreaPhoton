"""
Restart-boundary pixel-space enhancement (v1.4, experimental).

At the restart boundary the sampler holds the plunge readout x0 (a clean image in
the model's normalized latent space) and is about to re-noise it to sigma_r for
the texture phase. A hook installed here can decode that x0 with the VAE, run a
pixel-space enhancer, encode it back and hand the sampler the enhanced x0 - the
3 restart steps then re-synthesise texture around the enhanced lighting /
materials instead of the raw plunge. First enhancer: NVIDIA DLSS 5 Photoreal
Enhance V2 (custom node pack ComfyUI-dlss-enhancer), called through its node
class straight from nodes.NODE_CLASS_MAPPINGS - a soft dependency, never
imported at module load.

Space bookkeeping: loop state x -> model.process_latent_out -> VAE decode ->
enhance -> VAE encode -> model.process_latent_in -> x' (same device/dtype).
"""
import torch

DLSS_NODE_CLASSES = ("DLSS5PhotorealEnhanceV2", "DLSS5PhotorealEnhance")


def _dlss_node_class():
    try:
        import nodes as comfy_nodes  # top-level ComfyUI module
    except ImportError as e:  # unit tests without a ComfyUI tree
        raise RuntimeError("KreaPhoton restart_enhance needs a running ComfyUI (nodes module)") from e
    for name in DLSS_NODE_CLASSES:
        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
        if cls is not None:
            return name, cls
    raise ValueError("KreaPhoton restart_enhance: the DLSS 5 Photoreal Enhance node is not installed "
                     "(custom node pack ComfyUI-dlss-enhancer) - set restart_enhance to 'off'")


def _images_from_decode(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 5:  # (B, T, H, W, C) video-shaped decode of a single frame
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def make_dlss_restart_hook(vae, params: dict, run_fn=None):
    """Returns hook(x, base_model) -> x' for kreaphoton_sampler_loop(restart_hook=...).

    run_fn: optional override (tests) with the signature of the DLSS node's `run`:
    run_fn(image=(B,H,W,3) float 0..1, **params) -> (image, ...)."""
    if vae is None:
        raise ValueError("KreaPhoton restart_enhance needs the `vae` input connected")

    def hook(x, base_model):
        nonlocal run_fn
        if run_fn is None:
            _, cls = _dlss_node_class()
            node = cls()
            run_fn = node.run
        device, dtype = x.device, x.dtype
        latent_raw = base_model.process_latent_out(x.float())
        images = _images_from_decode(vae.decode(latent_raw))
        out = run_fn(image=images[..., :3].contiguous(), **params)
        enhanced = out[0] if isinstance(out, (tuple, list)) else out
        enhanced = enhanced[..., :3].to(images.device, torch.float32).clamp(0.0, 1.0)
        latent_new = vae.encode(enhanced)
        if latent_new.ndim == 4 and x.ndim == 5:
            latent_new = latent_new.unsqueeze(2)
        if tuple(latent_new.shape) != tuple(x.shape):
            raise RuntimeError(f"KreaPhoton restart_enhance: VAE round-trip changed the latent shape "
                               f"{tuple(x.shape)} -> {tuple(latent_new.shape)} (DLSS mode must be DLAA 1x)")
        x_new = base_model.process_latent_in(latent_new.to(device=device, dtype=torch.float32))
        return x_new.to(device=device, dtype=dtype)

    return hook
