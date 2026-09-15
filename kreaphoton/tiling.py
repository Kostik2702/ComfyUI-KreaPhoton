"""
Latent tiling for the KreaPhoton Upscale node (v1.5) - pure tensor math.

The tiled upscale runs the ordinary KreaPhoton refine (refine_schedule on the
whole upscaled latent) and tiles INSIDE every model call: LatentTiler cuts the
loop state x into overlapping tiles, runs them through the model in batches
with the shared conditioning (comfy repeats a batch-1 cond to the tile batch:
comfy/conds.py process_cond -> repeat_to_batch_size) and merges the x0
predictions with feathered weights. Because neighbouring tiles are re-blended
at every step (MultiDiffusion / Mixture-of-Diffusers idea), seams cannot form
- unlike pixel-space tiling (USDU), which decodes, blends and re-encodes each
tile once. One tiled VAE encode and one tiled decode per image, no per-tile
VAE round-trips.

LFAnchor is the "do not rewrite the picture" guard: after every model call
the low-frequency band of the merged x0 prediction is pulled toward the
source latent with a weight that is w_max at the start of the descent and
releases to 0 below release_sigma. Low frequencies carry layout, light and
tone; small objects on a 2x frame (a wrist watch ~20 latent px) are mid/high
frequency and stay free to be re-rendered. Both objects are installed through
kreaphoton_sampler_loop(tiler=..., x0_hook=...); None keeps the loop bit-exact.

No comfy imports at module level (unit-testable without a ComfyUI tree).
"""
import math

import torch
import torch.nn.functional as F


def tile_grid(length: int, tile: int, overlap: int):
    """Positions [(start, end)] of tiles of size `tile` covering [0, length)
    with at least ~`overlap` shared between neighbours: n = ceil((L-ov)/(T-ov))
    tiles spread evenly so the last one ends exactly at `length`. Starts are
    rounded down to even values (DiT 2x2 patch alignment; the latent dims of a
    16-px-aligned image are even, so the last tile still ends at `length`),
    which can shave the overlap by up to 2. length <= tile -> one tile [0, L)."""
    length, tile, overlap = int(length), int(tile), int(overlap)
    if tile <= 0:
        raise ValueError("tile_grid: tile must be positive")
    if length <= tile:
        return [(0, length)]
    overlap = max(0, min(overlap, tile - 2))
    n = int(math.ceil((length - overlap) / float(tile - overlap)))
    out = []
    for i in range(n):
        start = int(round(i * (length - tile) / float(n - 1)))
        start -= start % 2
        out.append((start, start + tile))
    return out


def feather_weight(h: int, w: int, overlap: int, device=None, dtype=torch.float32):
    """(h, w) blend weight: 1 inside, smoothstep ramp of width `overlap` toward
    every edge (outer image edges too - normalisation by the weight sum in the
    merge restores 1 there). overlap 0 -> all ones."""
    def ramp(n):
        r = torch.ones(n, dtype=dtype, device=device)
        k = int(max(0, min(overlap, n // 2)))
        if k > 0:
            u = (torch.arange(k, dtype=dtype, device=device) + 0.5) / k
            s = u * u * (3.0 - 2.0 * u)
            r[:k] = s
            r[n - k:] = s.flip(0)
        return r
    return ramp(int(h))[:, None] * ramp(int(w))[None, :]


class LatentTiler:
    """Callable installed as kreaphoton_sampler_loop(tiler=...):
    tiler(model_fn, x, sigma_scalar, extra_args) -> denoised of x's shape.

    Tiles are cut on the last two dims (works for (1,C,H,W) and (1,C,T,H,W)),
    concatenated along the batch dim in chunks of `batch`, and the model is
    called with sigma broadcast to that chunk. Batch-1 latents only (the node
    iterates images) - a bigger batch would mix images inside a chunk."""

    def __init__(self, tile_h: int, tile_w: int, overlap: int, batch: int = 4):
        self.tile_h = int(tile_h)
        self.tile_w = int(tile_w)
        self.overlap = int(overlap)
        self.batch = max(1, int(batch))
        self._weights = {}

    def boxes(self, h: int, w: int):
        return [(y0, y1, x0, x1)
                for (y0, y1) in tile_grid(h, self.tile_h, self.overlap)
                for (x0, x1) in tile_grid(w, self.tile_w, self.overlap)]

    def _weight(self, h: int, w: int, like: torch.Tensor):
        key = (h, w, like.device, like.dtype)
        wt = self._weights.get(key)
        if wt is None:
            wt = feather_weight(h, w, self.overlap, device=like.device, dtype=like.dtype)
            self._weights[key] = wt
        return wt

    def __call__(self, model_fn, x: torch.Tensor, sigma: float, extra_args: dict):
        if x.shape[0] != 1:
            raise ValueError("KreaPhoton LatentTiler: batch 1 only (the Upscale node iterates images), "
                             "got batch %d" % int(x.shape[0]))
        h, w = int(x.shape[-2]), int(x.shape[-1])
        boxes = self.boxes(h, w)
        if boxes == [(0, h, 0, w)]:
            return model_fn(x, float(sigma) * x.new_ones([1]), **extra_args)
        acc = torch.zeros_like(x)
        wsum = torch.zeros((h, w), device=x.device, dtype=x.dtype)
        for i in range(0, len(boxes), self.batch):
            chunk = boxes[i:i + self.batch]
            xt = torch.cat([x[..., y0:y1, x0:x1] for (y0, y1, x0, x1) in chunk], dim=0)
            d = model_fn(xt, float(sigma) * xt.new_ones([xt.shape[0]]), **extra_args)
            for k, (y0, y1, x0, x1) in enumerate(chunk):
                wt = self._weight(y1 - y0, x1 - x0, x)
                acc[..., y0:y1, x0:x1] += d[k:k + 1] * wt
                wsum[y0:y1, x0:x1] += wt
        return acc / wsum


def lowpass(z: torch.Tensor, radius: float) -> torch.Tensor:
    """Separable gaussian blur (std = radius latent px, kernel 6*radius+1) over
    the last two dims of a 4D/5D tensor; reflect padding (replicate when the
    tensor is smaller than the pad). Mean-preserving."""
    r = max(1, int(round(float(radius))))
    pad = 3 * r
    k = 2 * pad + 1
    t = torch.arange(k, dtype=torch.float32, device=z.device) - pad
    g = torch.exp(-0.5 * (t / float(r)) ** 2)
    g = g / g.sum()
    shape = z.shape
    z4 = z.reshape(-1, 1, shape[-2], shape[-1]).float()
    mode = "reflect" if (shape[-2] > pad and shape[-1] > pad) else "replicate"
    z4 = F.pad(z4, (pad, pad, pad, pad), mode=mode)
    z4 = F.conv2d(z4, g.view(1, 1, 1, k))
    z4 = F.conv2d(z4, g.view(1, 1, k, 1))
    return z4.reshape(shape).to(z.dtype)


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


class LFAnchor:
    """Callable installed as kreaphoton_sampler_loop(x0_hook=...):
    anchor(denoised, sigma) -> denoised + w(sigma) * (LF(z_ref) - LF(denoised)).

    z_ref must be in the loop's state space (the node passes
    model.model.process_latent_in(source latent)); LF(z_ref) is computed once
    and moved to the denoised tensor's device/dtype lazily. w(sigma) =
    w_max * smoothstep((sigma - release_sigma) / (sigma_start - release_sigma)):
    full strength at the start of the descent, 0 at/below release_sigma, so the
    last steps are free to settle micro-contrast. w_max 0 disables the hook."""

    def __init__(self, z_ref: torch.Tensor, radius: float, w_max: float,
                 sigma_start: float, release_sigma: float):
        self.z_ref = z_ref
        self.radius = float(radius)
        self.w_max = float(w_max)
        self.sigma_start = float(sigma_start)
        self.release = float(release_sigma)
        self._ref_lf = None

    def weight(self, sigma: float) -> float:
        if self.w_max <= 0.0 or self.sigma_start <= self.release:
            return 0.0
        return self.w_max * _smoothstep((float(sigma) - self.release) / (self.sigma_start - self.release))

    def _reference(self, like: torch.Tensor) -> torch.Tensor:
        if self._ref_lf is None:
            self._ref_lf = lowpass(self.z_ref.to(device=like.device, dtype=torch.float32), self.radius)
        return self._ref_lf.to(device=like.device, dtype=like.dtype)

    def __call__(self, denoised: torch.Tensor, sigma: float) -> torch.Tensor:
        w = self.weight(sigma)
        if w <= 0.0:
            return denoised
        ref = self._reference(denoised)
        if tuple(ref.shape) != tuple(denoised.shape):
            raise RuntimeError("KreaPhoton LFAnchor: reference latent shape %s != prediction shape %s"
                               % (tuple(ref.shape), tuple(denoised.shape)))
        return denoised + w * (ref - lowpass(denoised, self.radius))


# --- v2 Upscale: shifted grid + empty-tile skipping -------------------------------------

def tile_grid_shifted(length: int, tile: int, overlap: int, shift: int):
    """tile_grid with the INTERIOR tiles moved by a common offset inside the slack the
    grid has: the first tile stays at 0, the last at length - tile, the tiles between
    move by delta = shift folded into [-slack, +slack], slack = (tile - overlap) - the
    base spacing, so every neighbour overlap stays >= overlap and the tile COUNT is
    unchanged (an earlier variant pinned both ends AND shifted all tiles -> one extra
    tile per axis, +67% model calls on a 3x4 grid; measured 2026-09-13). Even starts
    (DiT patch). No interior tile (n <= 2) or length <= tile -> the base grid."""
    base = tile_grid(length, tile, overlap)
    n = len(base)
    if n <= 2:
        return base
    tile, length = int(tile), int(length)
    overlap = max(0, min(int(overlap), tile - 2))
    spacing = max(b[0] - a[0] for a, b in zip(base, base[1:]))
    slack = (tile - overlap) - spacing
    if slack <= 1:
        return base
    span = 2 * slack + 1
    delta = ((int(shift) + slack) % span) - slack  # fold into [-slack, +slack], shift 0 -> 0
    delta -= delta % 2
    out = [base[0]]
    for s0, _ in base[1:-1]:
        s1 = min(max(0, s0 + delta), length - tile)
        out.append((s1, s1 + tile))
    out.append(base[-1])
    return out


def activity_map(pixels: torch.Tensor, block: int) -> torch.Tensor:
    """(h, w) detail map of an image (B,H,W,C float 0..1) at 1/block resolution: the
    STANDARD DEVIATION of the luma Laplacian inside each block, in absolute luma units
    (no normalisation - a relative "fraction of the busiest block" was measured useless:
    one hard edge makes every other tile look empty). Measured on the calibration
    frames: bokeh / night sky blocks 0.005-0.017, skin and fabric 0.03-0.09, hard
    edges > 0.3. Used with LatentTilerV2 to decide which latent tiles carry no detail
    worth a model call - those fall back to the source latent."""
    g = pixels[..., :3].float().mean(-1)                      # (B,H,W) luma
    g = g[:, None]                                             # (B,1,H,W)
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=g.device).view(1, 1, 3, 3)
    lap = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), k)
    h, w = int(lap.shape[-2]) // block, int(lap.shape[-1]) // block
    lap = lap[..., :h * block, :w * block]
    blocks = lap.reshape(lap.shape[0], 1, h, block, w, block)
    std = blocks.std(dim=(3, 5), unbiased=False)[:, 0]          # (B,h,w)
    return std.mean(0)


class LatentTilerV2(LatentTiler):
    """LatentTiler with (1) a per-call random grid shift (SpotDiffusion idea): the tile
    grid moves by a seeded random offset at every model call, so no blend band stays at
    a fixed position and a smaller overlap suffices; (2) empty-tile skipping: tiles whose
    mean `activity` (tiling.activity_map on the source, latent resolution) is below
    `skip_threshold` (absolute luma-Laplacian std, see activity_map) at their 90th
    percentile are not sent to the model - their x0 prediction is the `fallback`
    latent (the source in the loop's space), which is what back-projection would restore
    there anyway. `calls` / `skipped` count tiles for reporting."""

    def __init__(self, tile_h: int, tile_w: int, overlap: int, batch: int = 4, *, shift_seed: int = 0,
                 shift: bool = True, activity: torch.Tensor = None, skip_threshold: float = 0.0,
                 fallback: torch.Tensor = None):
        super().__init__(tile_h, tile_w, overlap, batch)
        self.shift = bool(shift)
        self.activity = activity
        self.skip_threshold = float(skip_threshold)
        self.fallback = fallback
        self._gen = torch.Generator(device="cpu").manual_seed((int(shift_seed) + 0x7113) & 0xffffffffffffffff)
        self.step = 0
        self.calls = 0
        self.skipped = 0

    def boxes(self, h: int, w: int, shift_hw=(0, 0)):
        dy, dx = shift_hw
        return [(y0, y1, x0, x1)
                for (y0, y1) in tile_grid_shifted(h, self.tile_h, self.overlap, dy)
                for (x0, x1) in tile_grid_shifted(w, self.tile_w, self.overlap, dx)]

    def _draw_shift(self, h: int, w: int):
        """A random offset per call; tile_grid_shifted folds it into the grid's slack."""
        if not self.shift:
            return 0, 0
        dy = int(torch.randint(0, 1 << 16, (1,), generator=self._gen).item())
        dx = int(torch.randint(0, 1 << 16, (1,), generator=self._gen).item())
        return dy, dx

    def _is_empty(self, box):
        """A tile is empty when even its 90th-percentile block (its busiest tenth) is
        below the absolute threshold - a tile with one small detailed object among
        bokeh is NOT empty (a mean would say it is)."""
        if self.activity is None or self.skip_threshold <= 0.0 or self.fallback is None:
            return False
        y0, y1, x0, x1 = box
        blk = self.activity[y0:y1, x0:x1].flatten()
        return float(torch.quantile(blk, 0.9)) < self.skip_threshold

    def __call__(self, model_fn, x: torch.Tensor, sigma: float, extra_args: dict):
        if x.shape[0] != 1:
            raise ValueError("KreaPhoton LatentTilerV2: batch 1 only, got batch %d" % int(x.shape[0]))
        h, w = int(x.shape[-2]), int(x.shape[-1])
        self.step += 1
        boxes = self.boxes(h, w, self._draw_shift(h, w))
        if boxes == [(0, h, 0, w)]:
            self.calls += 1
            return model_fn(x, float(sigma) * x.new_ones([1]), **extra_args)
        live = [b for b in boxes if not self._is_empty(b)]
        empty = [b for b in boxes if self._is_empty(b)]
        self.skipped += len(empty)
        acc = torch.zeros_like(x)
        wsum = torch.zeros((h, w), device=x.device, dtype=x.dtype)
        if empty:
            fb = self.fallback.to(device=x.device, dtype=x.dtype)
            for (y0, y1, x0, x1) in empty:
                wt = self._weight(y1 - y0, x1 - x0, x)
                acc[..., y0:y1, x0:x1] += fb[..., y0:y1, x0:x1] * wt
                wsum[y0:y1, x0:x1] += wt
        for i in range(0, len(live), self.batch):
            chunk = live[i:i + self.batch]
            xt = torch.cat([x[..., y0:y1, x0:x1] for (y0, y1, x0, x1) in chunk], dim=0)
            d = model_fn(xt, float(sigma) * xt.new_ones([xt.shape[0]]), **extra_args)
            self.calls += len(chunk)
            for k, (y0, y1, x0, x1) in enumerate(chunk):
                wt = self._weight(y1 - y0, x1 - x0, x)
                acc[..., y0:y1, x0:x1] += d[k:k + 1] * wt
                wsum[y0:y1, x0:x1] += wt
        return acc / wsum
