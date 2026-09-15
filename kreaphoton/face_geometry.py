"""
Face Detailer geometry and policy (v1.6): face selection, crop box, soft mask,
paste, retry schedule and keep-best choice. Pure torch / python - no comfy,
ultralytics or insightface imports (unit-testable without a ComfyUI tree, the
tiling.py discipline).

Coordinates: boxes are (x0, y0, x1, y1) in pixels, x along width; images are
comfy IMAGE tensors (B, H, W, C) in 0..1.
"""
import math

import torch


def _align_up(v: int, align: int) -> int:
    return int(math.ceil(v / align)) * align


def _align_down(v: int, align: int) -> int:
    return (int(v) // align) * align


def select_faces(boxes, confs, *, max_faces: int, min_face_px: int, threshold: float):
    """[(original_index, box)] of the faces to detail: confidence >= threshold and
    min(w, h) >= min_face_px, sorted by bbox area descending (stable: equal areas keep
    detector order), truncated to max_faces. The subject is almost always the largest
    face; background faces come after it."""
    kept = []
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        x0, y0, x1, y1 = [float(v) for v in box]
        w, h = x1 - x0, y1 - y0
        if float(conf) < float(threshold) or min(w, h) < float(min_face_px):
            continue
        kept.append((i, (x0, y0, x1, y1), w * h))
    kept.sort(key=lambda t: -t[2])  # sort() is stable
    return [(i, box) for i, box, _ in kept[:int(max_faces)]]


def _clamp_axis(center: float, side: int, limit: int):
    """Place a segment of `side` (<= limit) centred at `center` inside [0, limit] by
    shifting it in. Returns (start, end)."""
    start = int(round(center - side / 2))
    start = max(0, min(start, limit - side))
    return start, start + side


def crop_box(box, crop_factor: float, image_hw, align: int = 16):
    """Square crop of side max(bbox side) * crop_factor (aligned up to `align`, capped
    by the shorter image axis aligned down) centred on the bbox and shifted inside the
    image. Always square, always aligned."""
    H, W = int(image_hw[0]), int(image_hw[1])
    x0, y0, x1, y1 = [float(v) for v in box]
    side = _align_up(int(math.ceil(max(x1 - x0, y1 - y0) * float(crop_factor))), align)
    side = max(min(side, _align_down(min(H, W), align)), align)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bx0, bx1 = _clamp_axis(cx, side, W)
    by0, by1 = _clamp_axis(cy, side, H)
    return (bx0, by0, bx1, by1)


def face_mask(crop_hw, bbox_in_crop, *, dilation: float, feather: float) -> torch.Tensor:
    """(h, w) float mask: 1 inside the ellipse inscribed in the bbox dilated by
    `dilation` x bbox side on every edge, linear feather of width feather x min(h, w)
    outside the ellipse edge, 0 beyond. Non-increasing from the centre outward."""
    h, w = int(crop_hw[0]), int(crop_hw[1])
    x0, y0, x1, y1 = [float(v) for v in bbox_in_crop]
    bw, bh = x1 - x0, y1 - y0
    d = float(dilation)
    x0, x1 = x0 - d * bw, x1 + d * bw
    y0, y1 = y0 - d * bh, y1 + d * bh
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = max((x1 - x0) / 2.0, 1.0), max((y1 - y0) / 2.0, 1.0)
    band = max(float(feather) * min(h, w), 1e-6)
    ys = torch.arange(h, dtype=torch.float32).view(h, 1) + 0.5
    xs = torch.arange(w, dtype=torch.float32).view(1, w) + 0.5
    # normalised elliptic radius: 1.0 on the ellipse edge; the feather band is
    # measured in pixels along the shorter semi-axis so it is the same width all round
    r = torch.sqrt(((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2)
    px_outside = (r - 1.0) * min(rx, ry)
    m = 1.0 - px_outside / band
    return m.clamp(0.0, 1.0)


def paste(image: torch.Tensor, box, patch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """New (1, H, W, C) image with image[box] = image[box] * (1 - mask) + patch * mask.
    patch is (1, h, w, C) matching the box, mask is (h, w)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    out = image.clone()
    region = out[:, y0:y1, x0:x1, :]
    m = mask.to(dtype=region.dtype, device=region.device).unsqueeze(0).unsqueeze(-1)
    out[:, y0:y1, x0:x1, :] = region * (1.0 - m) + patch.to(region)[..., :region.shape[-1]] * m
    return out


def place_mask(mask: torch.Tensor, box, image_hw) -> torch.Tensor:
    """(H, W) zeros with `mask` written at `box`."""
    x0, y0, x1, y1 = [int(v) for v in box]
    out = torch.zeros((int(image_hw[0]), int(image_hw[1])), dtype=mask.dtype)
    out[y0:y1, x0:x1] = mask.to(out)
    return out


def retry_schedule(denoise: float, seed: int, *, retry_max: int, denoise_step: float, seed_step: int,
                   floor: float = 0.05):
    """[(denoise, seed)] for the attempts of one pass: denoise decreases by `denoise_step`
    per retry (never below `floor` - the LoRA identity drifts with denoise, so a retry
    always moves toward the source), seed advances by `seed_step` (a distilled model
    barely changes between seed and seed+1)."""
    out = []
    for k in range(int(retry_max)):
        d = max(float(floor), float(denoise) - k * float(denoise_step))
        out.append((d, (int(seed) + k * int(seed_step)) & 0xffffffffffffffff))
    return out


def choose_best(attempts, *, threshold: float, orig_sim=None, regress_margin: float = 0.05):
    """(index or None, reason) over the identity similarities of the attempts (None =
    gate off). Keep-best: the highest similarity wins (ties -> earliest); with a
    reference the redraw must not lose more than `regress_margin` of the ORIGINAL
    face's similarity, else None = keep the original crop."""
    if not attempts:
        return None, "no attempts"
    if all(s is None for s in attempts):
        return len(attempts) - 1, "gate off"
    sims = [(-1.0 if s is None else float(s)) for s in attempts]
    best = max(range(len(sims)), key=lambda i: (sims[i], -i))
    if sims[best] >= float(threshold):
        return best, "pass"
    if orig_sim is not None and sims[best] < float(orig_sim) - float(regress_margin):
        return None, "kept original (identity regressed)"
    return best, "below threshold, kept best"
