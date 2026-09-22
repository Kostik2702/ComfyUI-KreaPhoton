"""KreaPhoton Composition Drafts (spike 2026-09-22): pick the composition BEFORE paying
for the whole image.

On Krea 2 Turbo the layout is decided in the composition phase (sigma > 0.85: 6 of the
12 balanced steps, 8 of 16 on quality/candid) and the distilled model one-shots a
readable x0 estimate from mid sigma (the same readout the plunge step uses). So for N
candidate seeds this node runs ONLY the first `draft_calls` model calls of the preset's
schedule (0 = the whole composition phase), takes the x0 estimate of the last call as
the draft, decodes it, scores every draft against an optional layout spec with YOLO
detectors (faces / persons / hands: count, where the subject sits, how tight the shot
is), renders a labelled contact sheet and returns the best seed. Feed `best_seed` (or
any seed read off the sheet) into the KreaPhoton Sampler: the draft's head was run with
the SAME schedule prefix, noise, guider, RNG stream and detail envelope as the sampler's
full run (progress_total is the full schedule), so the chosen composition reproduces
bit-exactly there.

Cost: N x draft_calls model calls at the working resolution. The composition phase is
half the schedule, so a full-phase draft is ~0.5 image. MEASURED 2026-09-22
(research/composition_probe, the owner's cafe scene, six seeds, turbo/candid, 2.1 MP):
after 2 calls (sigma 0.985, 1/8 of the schedule, ~5 s a draft on an RTX 5090) the
pose, framing and shot are readable, and the SUBJECT's person / face boxes sit within
2 % of the frame of where the finished image (the owner's full Sampler on the same
seed) puts them (depth 4: within 1 %; depth 8 = full phase: within 0.7 %). What
settles deeper is secondary: furniture, the counter's angle, background people, hands
(detected only from 4 calls on). Default draft_calls = 2: where the subject stands
and how the shot is framed is decided by the first two model calls.

Layout spec (`layout`, JSON, every key optional):
  faces    int   expected face count (YOLO face_yolov8m)
  persons  int   expected person count (person_yolov8m-seg boxes; background pedestrians
                 count once the draft is deep enough to show them - a scene with "blurred
                 people behind" wants no `persons` key, or the full-phase depth)
  hands    int   expected visible hands (hand_yolov8n)
  subject  left | center | right   horizontal third of the largest person (else face)
  shot     closeup | medium | full  from the largest face height / image height
           (thresholds SHOT_CLOSEUP / SHOT_MEDIUM below - CALIBRATE placeholders)
Score = mean over the given keys: counts 1 - 0.5 x |diff| (floored at 0), categories
1 / 0. Without a spec the drafts are not ranked (best_seed = the first) and the sheet is
just a sheet. Near-duplicate drafts (16 x 16 luminance thumbnails closer than
DUP_DISTANCE) are flagged in the report - a seed step that lands on the same layout is
wasted budget.

Seams (_run_sampling, _detect, _decode) exist so tests can replace the heavy parts.
"""
import json
import math

import torch
import torch.nn.functional as F

from . import face_detect
from .lora_phase import build_phase_models
from .presets import (COHERENCE, COHERENCE_OPTIONS, DEFAULT_PRESET, GUIDANCE, MANIFOLD_MEAN, MANIFOLD_STD,
                      PRESETS, VARIETY_COND_TAPS, VARIETY_END, VARIETY_LEVELS, preset_guidance)
from .sampling import run_sampling
from .schedules import build_schedule

COMPOSITION_END = 0.85          # the sampler's composition_end (presets.LORA_PHASES "composition")
MAX_DRAFTS = 16
PERSON_MODEL = "person_yolov8m-seg.pt"
HAND_MODEL = "hand_yolov8n.pt"
SHOT_CLOSEUP = 0.28             # largest face height / image height >= this -> closeup   (CALIBRATE)
SHOT_MEDIUM = 0.10              # ... >= this -> medium, else full                       (CALIBRATE)
DUP_DISTANCE = 0.06             # RMS distance of 16x16 luminance thumbs below this = same layout (CALIBRATE)
THUMB = 16
SHEET_TILE = 384                # px, long side of a sheet tile
SUBJECT_ZONES = ("left", "center", "right")
SHOT_TYPES = ("closeup", "medium", "full")
LAYOUT_KEYS = ("faces", "persons", "hands", "subject", "shot")

# seams
_run_sampling = run_sampling
_detect = face_detect.detect_faces


def composition_head(sigmas, composition_end: float = COMPOSITION_END, draft_calls: int = 0):
    """(head sigmas, n_calls, readout_sigma): the descending prefix of `sigmas` whose model
    calls sit at sigma > composition_end - at most draft_calls of them when > 0 - with a
    terminal 0.0 appended (a plunge step: the last call's x0 estimate becomes the state).
    ValueError when the schedule has no composition call."""
    sig = [float(s) for s in sigmas]
    calls = []
    for i in range(len(sig) - 1):
        if sig[i] <= composition_end + 1e-9:
            break
        calls.append(sig[i])
        if sig[i + 1] > sig[i] + 1e-6:      # an ascending jump follows: the phase ends here
            break
    if int(draft_calls) > 0:
        calls = calls[:int(draft_calls)]
    if not calls:
        raise ValueError("KreaPhoton Composition Drafts: the schedule has no model call above sigma %.2f"
                         % composition_end)
    head = torch.tensor(calls + [0.0], dtype=torch.float32)
    return head, len(calls), calls[-1]


def parse_layout(text):
    """JSON layout spec -> validated dict (empty dict = no spec). Unknown key / bad value
    -> ValueError naming it."""
    text = (text or "").strip()
    if not text:
        return {}
    try:
        spec = json.loads(text)
    except ValueError as e:
        raise ValueError(f"KreaPhoton Composition Drafts: layout must be a JSON object ({e})")
    if not isinstance(spec, dict):
        raise ValueError("KreaPhoton Composition Drafts: layout must be a JSON object")
    out = {}
    for k, v in spec.items():
        if k not in LAYOUT_KEYS:
            raise ValueError(f"KreaPhoton Composition Drafts: unknown layout key {k!r} (keys: {LAYOUT_KEYS})")
        if k in ("faces", "persons", "hands"):
            if not isinstance(v, (int, float)) or int(v) < 0:
                raise ValueError(f"KreaPhoton Composition Drafts: layout {k} must be a non-negative integer")
            out[k] = int(v)
        elif k == "subject":
            if v not in SUBJECT_ZONES:
                raise ValueError(f"KreaPhoton Composition Drafts: layout subject must be one of {SUBJECT_ZONES}")
            out[k] = v
        else:
            if v not in SHOT_TYPES:
                raise ValueError(f"KreaPhoton Composition Drafts: layout shot must be one of {SHOT_TYPES}")
            out[k] = v
    return out


def _largest(boxes):
    best = None
    for b in boxes:
        area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        if best is None or area > best[0]:
            best = (area, b)
    return None if best is None else best[1]


def draft_features(image, *, detect=None, bbox_threshold: float = 0.4):
    """Layout features of one (1, H, W, C) image: counts, largest boxes, subject zone,
    shot type. detect(image, model_name=..., threshold=...) -> (boxes, confs)."""
    detect = _detect if detect is None else detect
    H, W = int(image.shape[1]), int(image.shape[2])

    def boxes_of(model_name):
        boxes, confs = detect(image, model_name=model_name, threshold=min(0.25, bbox_threshold))
        return [b for b, c in zip(boxes, confs) if c >= bbox_threshold]

    faces = boxes_of(face_detect.DEFAULT_FACE_MODEL)
    persons = boxes_of(PERSON_MODEL)
    hands = boxes_of(HAND_MODEL)
    face = _largest(faces)
    person = _largest(persons)
    anchor = person if person is not None else face
    subject = None
    if anchor is not None:
        cx = 0.5 * (anchor[0] + anchor[2]) / max(W, 1)
        subject = SUBJECT_ZONES[min(2, int(cx * 3.0))]
    shot = None
    if face is not None:
        fh = (face[3] - face[1]) / max(H, 1)
        shot = "closeup" if fh >= SHOT_CLOSEUP else ("medium" if fh >= SHOT_MEDIUM else "full")
    return {"faces": len(faces), "persons": len(persons), "hands": len(hands),
            "face": face, "person": person, "subject": subject, "shot": shot}


def layout_score(features, spec):
    """Mean over the spec's keys (None without a spec): counts 1 - 0.5|diff| floored at
    0, categories 1/0 (a missing category, e.g. no face for `shot`, scores 0)."""
    if not spec:
        return None
    parts = []
    for k, want in spec.items():
        if k in ("faces", "persons", "hands"):
            parts.append(max(0.0, 1.0 - 0.5 * abs(int(features[k]) - int(want))))
        else:
            parts.append(1.0 if features.get(k) == want else 0.0)
    return sum(parts) / len(parts)


def luminance_thumb(image, size: int = THUMB):
    """(1, H, W, C) -> (size*size,) luminance thumbnail, area-averaged."""
    lum = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).unsqueeze(1).float()
    return F.interpolate(lum, size=(size, size), mode="area").reshape(-1)


def duplicates(thumbs, threshold: float = DUP_DISTANCE):
    """For each draft, the index of the earliest earlier draft within `threshold` RMS
    distance, else None."""
    out = []
    for i, t in enumerate(thumbs):
        dup = None
        for j in range(i):
            if float(((t - thumbs[j]) ** 2).mean().sqrt()) < threshold:
                dup = j
                break
        out.append(dup)
    return out


def rank(scores, dups):
    """Draft order: score desc (None = unranked, keeps input order), duplicates after
    originals, index asc. Returns the list of indices, best first."""
    def key(i):
        s = scores[i]
        return (0 if dups[i] is None else 1, -(s if s is not None else 0.0), i)
    return sorted(range(len(scores)), key=key)


def render_sheet(images, labels, tile: int = SHEET_TILE):
    """Contact sheet (1, H, W, 3) float 0..1 of the (1, h, w, C) images with a text label
    drawn on every tile (PIL)."""
    from PIL import Image, ImageDraw
    n = len(images)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    tiles = []
    for img in images:
        h, w = int(img.shape[1]), int(img.shape[2])
        s = float(tile) / float(max(h, w))
        th, tw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        t = F.interpolate(img[..., :3].movedim(-1, 1).float(), size=(th, tw), mode="area").movedim(1, -1)[0]
        tiles.append(t.clamp(0.0, 1.0).cpu())
    th = max(int(t.shape[0]) for t in tiles)
    tw = max(int(t.shape[1]) for t in tiles)
    sheet = Image.new("RGB", (cols * tw, rows * th), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for k, (t, label) in enumerate(zip(tiles, labels)):
        r, c = divmod(k, cols)
        pil = Image.fromarray((t.numpy() * 255.0 + 0.5).astype("uint8"))
        sheet.paste(pil, (c * tw, r * th))
        x, y = c * tw + 6, r * th + 6
        for line in label.split("\n"):
            bbox = draw.textbbox((x, y), line)
            draw.rectangle((bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2), fill=(0, 0, 0))
            draw.text((x, y), line, fill=(120, 255, 120))
            y = bbox[3] + 4
    arr = torch.from_numpy(__import__("numpy").asarray(sheet).copy()).float() / 255.0
    return arr.unsqueeze(0)


_LAYOUT_TOOLTIP = ('Optional JSON layout spec the drafts are scored against, e.g. {"faces": 2, "subject": "left", '
                   '"shot": "medium"}. Keys: faces / persons / hands (counts), subject (left|center|right), '
                   'shot (closeup|medium|full). Empty = no ranking, best_seed = the first draft.')
_DRAFT_CALLS_TOOLTIP = ("Model calls per draft. Measured 2026-09-22 (one scene, six seeds, turbo/candid): after 2 "
                        "calls (sigma 0.985, 1/8 of the schedule) pose, framing and shot are readable and the "
                        "subject's person/face boxes are within 2% of the frame of the finished image (4 calls: "
                        "1%); furniture, background people and hands settle deeper (hands detect from 4 calls). "
                        "0 = the whole phase above sigma 0.85 (half the schedule). The chosen seed reproduces "
                        "in the Sampler at any depth - the draft is a readout, not a state.")
_SEED_STEP_TOOLTIP = "Draft i uses seed + i x seed_step (1000: neighbouring seeds of a distilled model look alike)."


class KreaPhotonCompositionDrafts:
    """See the module docstring."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "preset": (list(PRESETS.keys()), {"default": DEFAULT_PRESET}),
                "n_drafts": ("INT", {"default": 6, "min": 1, "max": MAX_DRAFTS}),
                "draft_calls": ("INT", {"default": 2, "min": 0, "max": 32, "tooltip": _DRAFT_CALLS_TOOLTIP}),
                "seed_step": ("INT", {"default": 1000, "min": 1, "max": 1_000_000, "tooltip": _SEED_STEP_TOOLTIP}),
                "variety": (list(VARIETY_LEVELS.keys()), {"default": "off"}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "clean_model": ("MODEL", {"tooltip": "Composition-phase model, exactly as on the Sampler."}),
                "coherence": (COHERENCE_OPTIONS, {"default": "off"}),
                "layout": ("STRING", {"default": "", "multiline": False, "tooltip": _LAYOUT_TOOLTIP}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("sheet", "drafts", "best_seed", "report")
    FUNCTION = "draft"
    CATEGORY = "KreaPhoton"

    def draft(self, model, positive, latent_image, vae, seed, preset, n_drafts, draft_calls, seed_step, variety,
              negative=None, clean_model=None, coherence="off", layout=""):
        from .nodes import _ORDER_FROM_SAMPLER_NAME, _decode_tiled, _explicit_phase_model, _preset_alpha
        p = PRESETS[preset]
        spec = parse_layout(layout)
        a_latent, a_cond = VARIETY_LEVELS[variety]
        alpha = _preset_alpha(p, latent_image)
        use_jump = coherence in ("jump", "jump+self_refine")
        plan_clean, identity, _ = build_phase_models(model)
        clean = _explicit_phase_model(clean_model, plan_clean, "composition")
        head_model = clean if clean is not None else identity
        full = build_schedule(p["n_steps"], alpha=alpha, restart_frac=p["restart_frac"], sigma_r=p["sigma_r"],
                              plunge=p["plunge"])
        head, n_calls, readout = composition_head(full, COMPOSITION_END, int(draft_calls))
        guidance_mode, flat_cfg = preset_guidance(p, negative is not None)

        drafts, feats, seeds = [], [], []
        for i in range(int(n_drafts)):
            s_i = (int(seed) + i * int(seed_step)) & 0xffffffffffffffff
            out = _run_sampling(
                head_model, positive, negative, latent_image, head, seed=s_i,
                guidance_mode=guidance_mode, flat_cfg=flat_cfg,
                delta=GUIDANCE["delta"], lo=GUIDANCE["lo"], hi=GUIDANCE["hi"],
                contraction=p["contraction"], per_channel_contraction=False,
                manifold_std=MANIFOLD_STD, manifold_mean=MANIFOLD_MEAN,
                detail_amount=p["detail_a"], order=_ORDER_FROM_SAMPLER_NAME[p["sampler"]],
                eta0=p["eta0"], sigma_gate=p["sigma_gate"], x0_extrapolation=0.0,
                variety_a_latent=a_latent, variety_a_cond=a_cond, variety_seed=s_i,
                variety_end=VARIETY_END, variety_cond_taps=VARIETY_COND_TAPS,
                coherence_jump=COHERENCE["jump"] if use_jump else 0.0,
                coherence_jump_sigma=COHERENCE["jump_sigma"],
                progress_total=len(full) - 1,
            )
            img = _decode_tiled(vae, out["samples"])[..., :3].clamp(0.0, 1.0)
            drafts.append(img)
            feats.append(draft_features(img))
            seeds.append(s_i)

        scores = [layout_score(f, spec) for f in feats]
        dups = duplicates([luminance_thumb(d) for d in drafts])
        order = rank(scores, dups)
        best = order[0]
        labels, lines = [], []
        lines.append("composition drafts: %d seeds x %d call(s) of %d (readout at sigma %.3f), preset %s%s"
                     % (len(drafts), n_calls, len(full) - 1, readout, preset,
                        (", layout %s" % json.dumps(spec)) if spec else ", no layout spec (unranked)"))
        for i, (f, s_i) in enumerate(zip(feats, seeds)):
            sc = "" if scores[i] is None else " score %.2f" % scores[i]
            dup = "" if dups[i] is None else " DUP of #%d" % (dups[i] + 1)
            desc = "faces %d persons %d hands %d%s%s" % (
                f["faces"], f["persons"], f["hands"],
                (" subject %s" % f["subject"]) if f["subject"] else "",
                (" shot %s" % f["shot"]) if f["shot"] else "")
            tag = " BEST" if i == best else ""
            labels.append("#%d seed %d%s%s%s\n%s" % (i + 1, s_i, sc, dup, tag, desc))
            lines.append("  #%d seed %d:%s %s%s%s" % (i + 1, s_i, sc, desc, dup, tag))
        lines.append("best_seed %d (#%d)" % (seeds[best], best + 1))
        sheet = render_sheet(drafts, labels)
        report = "\n".join(lines)
        print("[KreaPhoton Composition Drafts]\n" + report)
        result = (sheet, torch.cat(drafts, dim=0), int(seeds[best]), report)
        ui = _preview_ui(sheet)
        return {"ui": ui, "result": result} if ui is not None else result


def _preview_ui(sheet):
    """On-node thumbnail of the sheet (PreviewImage temp files -> ui.images, the
    _result_with_preview pattern); None outside ComfyUI. Seam for tests."""
    try:
        import nodes as comfy_nodes  # lazy: top-level ComfyUI module
    except ImportError:
        return None
    return comfy_nodes.PreviewImage().save_images(sheet, filename_prefix="KreaPhotonDrafts")["ui"]
