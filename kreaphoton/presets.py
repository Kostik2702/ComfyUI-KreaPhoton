"""
Single source of truth for every calibrated constant (planning-council R19).

No other module hardcodes a preset value or a manifold constant. Values marked
"CALIBRATE: V<n>" are starting hypotheses from docs/04 design math (M1-M6) and
research/results/E*.json, pending the corresponding validation slot in
docs/05-plan-kreaphoton-v1.md §V-protocol (S8.5/S10) — nothing here is final
until that V-slot freezes it. Values NOT marked CALIBRATE are measured facts
(e.g. MANIFOLD_STD/MEAN from research/results/E3_real.json), not guesses.
"""

# --- Manifold constants (MEASURED, research/results/E3_real.json normalized_per_channel,
#     8 real photographs through qwen_image_vae; global std=0.4666, mean=-0.0070) ---
MANIFOLD_STD = [
    0.34837067127227783, 0.34285303950309753, 0.45075058937072754, 0.4954315721988678,
    0.41811972856521606, 0.4532100260257721, 0.3540203869342804, 0.477848619222641,
    0.3242858350276947, 0.3410053551197052, 0.31148362159729004, 0.38901352882385254,
    0.3583352863788605, 0.4181098937988281, 0.36241018772125244, 0.4790935218334198,
]
MANIFOLD_MEAN = [
    0.28196287155151367, 0.40890082716941833, -0.0024451527278870344, -0.08094137161970139,
    -0.20453284680843353, -0.2656249701976776, -0.09275273233652115, -0.4799724817276001,
    -0.24426628649234772, -0.19706667959690094, -0.037885453552007675, 0.37430068850517273,
    0.10810646414756775, -0.04556984826922417, 0.2223597764968872, 0.14363540709018707,
]

# --- Guidance window (M6). V1/E1b RAN 2026-07-05 (docs/06 pre-registered protocol,
#     27 img: {1.25,1.31,1.5} x P1/P2/P3 x seed1001-1003): delta=1.5 catastrophically
#     hallucinates (P1: paint-like mutation on shoulder; P2: ghost face/eyes overlaid
#     on the whole scene - worse than the E1 "crocodile skin" early-warning sign, a
#     full compositional break, not a mild degrade). delta=1.25 and 1.31 both clean
#     across all 9 cells each; per pre-registered tie-break (smallest passing
#     candidate) delta=1.25 wins. Gate satisfied -> window enabled by default now. ---
GUIDANCE = {
    "delta": 1.25,              # V1-VALIDATED (was CALIBRATE placeholder, same value)
    "lo": 0.7,
    "hi": 0.9,
    "enabled_by_default": True,  # V1 gate passed (docs/04: "включается по умолчанию только после E1b")
    "flat_cfg": 1.15,           # fallback for guidance_mode="flat" (explicit full-traj cfg,
                                 # e.g. Advanced node override); "window" is now the Sampler default.
    # Variety consumes guidance budget (V14/V16/V16b, 2026-09-07, cielbleuKrea2): the latent
    # variety axis + the full Delta window hallucinated glowing/mesh regions on ~1 of 6 seeds
    # even after the noise-only re-composition fix; the same seeds are clean at variety
    # medium (a=0.40) with Delta 1.25, and at variety high (a=0.65) with Delta 0.90. Rule:
    # delta_eff = delta * (1 - variety_budget * a_latent)  -> high 0.88, medium 1.02, low 1.14.
    "variety_budget": 0.45,
}


def effective_delta(delta: float, guidance_mode: str, variety_a_latent: float,
                    budget: float = None) -> float:
    """Delta actually used by the window when the latent variety axis is active
    (see GUIDANCE['variety_budget']). Identity for guidance_mode != 'window' or
    a_latent == 0 - the V1-validated window is untouched without variety."""
    if guidance_mode != "window" or variety_a_latent <= 0.0:
        return float(delta)
    b = GUIDANCE["variety_budget"] if budget is None else budget
    return float(delta) * max(0.0, 1.0 - b * float(min(1.0, variety_a_latent)))

# Photo resolution buckets (Wan21 spatial /8 x DiT patch 2x2 -> effective /16;
# every dim below is divisible by 16). Canon anchor: user's ErikaNew4 baseline
# 1088x1600 (3:2 portrait, docs/01 §5) sits in the L tier.
RESOLUTION_BUCKETS = {
    "S (~1.0 MP)":  {"1:1": (1024, 1024), "4:3": (1152, 864),  "3:2": (896, 1344),  "16:9": (1344, 768),  "9:16": (768, 1344)},
    "M (~1.4 MP)":  {"1:1": (1184, 1184), "4:3": (1344, 1008), "3:2": (1040, 1568), "16:9": (1568, 880),  "9:16": (880, 1568)},
    "L (~1.7 MP)":  {"1:1": (1312, 1312), "4:3": (1504, 1120), "3:2": (1088, 1600), "16:9": (1728, 960),  "9:16": (960, 1728)},
    "XL (~2.1 MP)": {"1:1": (1440, 1440), "4:3": (1664, 1248), "3:2": (1184, 1776), "16:9": (1920, 1088), "9:16": (1088, 1920)},
}
RESOLUTION_ASPECTS = list(RESOLUTION_BUCKETS["L (~1.7 MP)"].keys())
DEFAULT_RESOLUTION_SIZE = "L (~1.7 MP)"
DEFAULT_RESOLUTION_ASPECT = "3:2"

# --- Variety mix: level -> (a_latent, a_cond). ZPhoton's own latent-axis mixes are
#     UNVALIDATED (vault decision "ZPhoton - аналитический форк Power Nodes", postscript
#     2026-07-05: merged without the visual A/B its own gate required) — treat these as a
#     fresh starting hypothesis for Krea2, not an inherited prior.
#
#     V3 RAN 2026-07-05 (24 img, off/low/medium/high x P1/P2 x seed1001-1003):
#     mutation-cap=0 CONFIRMED (no identity/pose/composition break at any level on
#     either prompt - values below are SAFE to ship). SVE comparison variant
#     deferred (not a shippable v1 feature; M4 already proved it analytically
#     non-variance-preserving, docs/03) - documented scope trim, not silently
#     dropped. OPEN FINDING (not blocking, values kept as-is): dose-response is
#     weak/non-monotonic - inter-seed SSIM barely moves low->medium->high on P1
#     (0.395->0.396->0.396) and on P2 actually INCREASES at higher levels
#     (0.430->0.431->0.445, i.e. LESS decorrelation at "high" than "low" - the
#     opposite of intended). The off->low jump captures most of the effect;
#     medium/high don't clearly add more variety. Follow-up: re-sweep
#     (a_latent,a_cond) magnitudes with a wider low/medium/high spread. ---
VARIETY_LEVELS = {
    "off":    (0.00, 0.00),
    "low":    (0.20, 0.15),   # texture/detail variation (see VARIETY_END note)
    "medium": (0.40, 0.30),   # texture/detail variation
    "high":   (0.65, 0.50),   # texture/detail variation
}
VARIETY_COND_TAPS = (7, 8, 9, 10)   # semantic taps per Rebalance/Enhancer community consensus
# 2026-07-06 (0.90 -> 0.96) + 2026-09-07 root cause: the "droplet/speckle" seen
# at 0.90 was NOT a bf16 round-trip effect but the per-segment re-indexing of the
# M2 detail envelope in the old two-lifecycle split (see sampling.py
# progress_offset) - a segment starting at sigma 0.87 re-ran the envelope from
# p=0 and over-nudged the structure phase. v1.3 applies variety inside one
# lifecycle, so the boundary is free of that mechanism; 0.96 (third model call,
# sigma~0.955 on the balanced grid) is kept because it is the verified point
# where the perturbation is texture-only: composition/identity preserved
# (mutation-cap holds, V3/V13/V14 B), variety clearly visible. Do NOT move it to
# 1.0: on the initial noise the LF re-draw moves composition AND identity
# (V14c, 2026-09-07 - a partial seed change, not variety).
# NOTE on scope: variety on krea2-turbo is a TEXTURE/DETAIL knob, not composition
# - variance-preserving latent perturbation cannot move composition on this model
# (proven 2026-07-06); for composition variety use a different seed or `blend`.
VARIETY_END = 0.96                  # boundary sigma below which latent variety applies (M4)

# --- Presets (Sampler simple node: seed / preset / variety) ---
#
# Guidance contract (audit F01 / KPA-01): every preset names its guidance
# policy explicitly. "window" = M6 sigma-window with delta (only when a
# `negative` is connected, else off; Turbo is CFG-distilled, a full-trajectory
# cfg breaks it - E1). "flat" = full-trajectory CFG at `cfg` on EVERY run; when
# no `negative` is connected the unconditional is the zeroed conditioning
# (sampling.zero_conditioning - the standard ConditioningZeroOut convention).
#
# x0_extrapolation: terminal x0-trajectory extrapolation weight (sampling.py,
# same mechanism the third-party "Krea 2 Turbo Preset Sampler" ships as
# zero_extrapolation). 0.0 = off in every preset until a V-slot calibrates it -
# UNTESTED on KreaPhoton grids; exposed on the Advanced node.
PRESET_REQUIRED_KEYS = ("n_steps", "alpha", "restart_frac", "sigma_r", "plunge", "detail_a",
                        "eta0", "sigma_gate", "contraction", "sampler", "guidance",
                        "x0_extrapolation")

PRESETS = {
    "turbo/fast": {
        "guidance": "window",
        "x0_extrapolation": 0.0,
        "n_steps": 8,
        "alpha": 3.158,             # e^1.15, stock ModelSamplingFlux shift; critic-verified base
        "restart_frac": 0.25,       # V2-VALIDATED (kept at design hypothesis, 54-img grid didn't test frac itself)
        "sigma_r": 0.65,            # V2-VALIDATED: {0.55,0.60,0.65}x{plunge} grid, 2026-07-05 -
                                     # higher sigma_r recovers shadow/fabric texture (dark_blotch
                                     # +25..98% vs no-restart baseline = RECOVERED DETAIL, confirmed
                                     # visually clean on P2 zoom-crops, not blotch/noise artifacts;
                                     # P1 face crop: freckles naturally varied, no stamping)
        "plunge": True,             # V2-VALIDATED: plunge=True consistently reduces/reverses the
                                     # lap_sharpness regression vs baseline at every sigma_r tested
        "detail_a": 0.50,           # UNTESTED (M2 working range 0.4-0.8; held constant across
                                     # all of V1-V6 - no dedicated V-slot tested this value; not
                                     # "V2" as previously mislabeled - V2 only swept sigma_r/plunge)
        "eta0": 1.0,                # V4-VALIDATED: {0.0,0.5,1.0} x P2 x seed1001-1003 (quality
                                     # preset, N=16), 2026-07-05 - eta0=1.0 gave the CLEANEST
                                     # shadows (lowest hf_noise/dark_blotch of the 3, not just
                                     # visually artifact-free) - highest tested value wins per
                                     # accept-rule, no regression found at this sr=0.65/plunge=True
                                     # config (was the open M5 event-hypothesis risk; resolved safe)
        "sigma_gate": 0.10,         # M5-proven terminal cutoff
        "contraction": 0.70,        # V5-VALIDATED, see turbo/balanced comment
        "sampler": "euler",
    },
    "turbo/balanced": {             # DEFAULT
        "guidance": "window",
        "x0_extrapolation": 0.0,
        "n_steps": 12,
        "alpha": 3.158,
        "restart_frac": 0.25,
        "sigma_r": 0.65,            # V2-VALIDATED, see turbo/fast comment
        "plunge": True,             # V2-VALIDATED
        "detail_a": 0.60,
        "eta0": 1.0,
        "sigma_gate": 0.10,
        "contraction": 0.70,        # V5-VALIDATED: {1.00,0.85,0.70} x P1/P2/P3 x seed1001-1003,
                                     # 2026-07-05 - accept-rule is "most aggressive value with no
                                     # inter-seed SSIM regression vs c=1.00 control" (guards against
                                     # the E2 alpha-tightening "sameness" mechanism). Neither 0.85
                                     # (SSIM +0.9%, flat) nor 0.70 (SSIM -4.7%, MORE diverse, not
                                     # less) regressed - both pass, picked smallest per tie-break.
                                     # hf_noise/dark_blotch improve monotonically with stronger
                                     # contraction (cleaner, not "flatter"); P1/P3 visual check
                                     # confirms no loss of photographic naturalness at 0.70.
        "sampler": "euler",
    },
    "turbo/quality": {
        "guidance": "window",
        "x0_extrapolation": 0.0,
        "n_steps": 16,
        "alpha": 3.158,
        "restart_frac": 0.25,
        "sigma_r": 0.65,            # V2-VALIDATED
        "plunge": True,             # V2-VALIDATED
        "detail_a": 0.70,
        "eta0": 1.0,                # mandatory gated-eta at this step count (design: "gated-eta обязателен")
        "sigma_gate": 0.10,
        "contraction": 0.70,        # V5-VALIDATED, see turbo/balanced comment
        "sampler": "euler_2m",
    },
    "turbo/candid": {              # V17 (2026-09-07): the quality grid with the texture push
        "guidance": "window",      # dialled back - for documentary / lifestyle / "as the eye sees
        "x0_extrapolation": 0.0,   # it" work. On 12 scenes x 2 (groups, hands, night, LoRA
        "n_steps": 16,             # story) it read consistently softer and more natural than
        "alpha": 3.158,            # turbo/quality in hard light (skin, wet fabric, hair), with
        "restart_frac": 0.25,      # the same structure, hands and prompt adherence; the
        "sigma_r": 0.65,           # differences are subtle - most "hyper-detail" comes from
        "plunge": True,            # prompt words (pores, texture) and polished LoRAs, not the
        "detail_a": 0.30,          # sampler.
        "eta0": 1.0,
        "sigma_gate": 0.10,
        "contraction": 0.85,       # softer initial-noise amplitude than 0.70 (V5) - less
                                    # micro-contrast; V16-era smoke, not a V-slot calibration
        "sampler": "euler_2m",
    },
    "raw/experimental": {
        "guidance": "flat",        # RAW is NOT distilled: real full-trajectory CFG (KPA-01 fix;
                                   # before v1.3 `cfg` below was never read - audit F01)
        "x0_extrapolation": 0.0,
        "n_steps": 36,
        "alpha": 3.158,             # CALIBRATE: raw canon is dynamic mu 0.5->1.15 (docs/01); unexplored
        "restart_frac": 0.20,
        "sigma_r": 0.45,
        "plunge": False,
        "detail_a": 0.50,
        "eta0": 1.0,
        "sigma_gate": 0.10,
        "contraction": 1.00,       # no contraction calibration attempted for RAW yet
        "sampler": "euler_2m",
        "cfg": 3.5,                # RAW needs real CFG (negative works, unlike Turbo)
        "experimental": True,
    },
}

DEFAULT_PRESET = "turbo/balanced"

# --- Resolution-aware shift (v1.4): constants + canonical_mu() live in schedules.py
#     (SHIFT / canonical_mu / alpha_for_latent) - schedule math has no preset dependency.
#     The simple-node policy switch is schedules.SHIFT["resolution_aware"]. ---

# --- LoRA phase scheduling (v1.4.1, KreaPhoton LoRA Phase node -> lora_phase.py). Phases
#     are the segments of the KreaPhoton multi-model split, expressed here as the sigma
#     bands they cover (the split boundaries are composition_end 0.85 and texture_start
#     0.65 = the restart segment):
#       composition  sigma 1.00 -> 0.85  layout / framing / pose
#       identity     sigma 0.85 -> 0     face, body, clothing decided (incl. texture phase)
#       texture      sigma 0.65 -> 0     restart segment: skin/fabric/grain
#     A character LoRA at "identity" keeps the face without steering the layout; a style
#     LoRA at "texture" adds its look without fighting the prompt's composition. The
#     segment membership table is lora_phase.PHASE_SEGMENTS. ---
LORA_PHASES = {
    "all":                 (1.00, 0.00),
    "composition":         (1.00, 0.85),
    "identity":            (0.85, 0.00),
    "texture":             (0.65, 0.00),
    "composition+identity": (1.00, 0.65),
}

# --- Coherence (v1.4, Power-Nodes derived, simple-node combo `coherence`):
#     jump  = one-time jump-back on the composition step at/below jump_sigma, strength
#             0.19 == PN's 0.920 -> 0.935 boundary in our units (declared sigma =
#             s + 0.19*(1-s), state x (1-s_decl)/(1-s));
#     self_refine = after the plunge draft, re-noise to refine_sigma and re-descend
#             refine_steps calls to the floor, plunge again, then the restart segment
#             (the model re-decides identity-phase content with the draft as prior).
#     Off in every preset until an A/B says otherwise. ---
COHERENCE = {
    "jump": 0.19,
    "jump_sigma": 0.93,
    "refine_steps": 4,
    "refine_sigma": 0.85,
}
COHERENCE_OPTIONS = ["off", "jump", "self_refine", "jump+self_refine"]

# --- Perturbed-attention guidance (v1.4, experimental). Second forward per step with
#     identity self-attention in the middle DiT blocks, inside a sigma window; the
#     difference cond - perturbed is added with pag_scale. Structure/anatomy guidance
#     that needs no negative. Off in every preset until an A/B says otherwise. ---
PAG = {
    "scale": 0.0,
    "lo": 0.72,
    "hi": 0.93,
    "blocks": "8-15",            # of 28 SingleStreamBlocks
}

# --- DLSS 5 restart enhance (v1.4, experimental): at the restart boundary the plunge
#     x0 is decoded, run through the DLSS 5 Photoreal Enhance V2 node (if installed),
#     re-encoded and re-noised into the texture phase. Parameter sets are the pack's
#     recommended V2 starting point with the three styles. ---
DLSS_PRESETS = {
    "dlss5 default":   dict(mode="DLAA 1x", intensity=1.0, style="Default", local_tone=0.0,
                            local_structure=1.0, skin_structure=-1.0, strength=0.7,
                            preserve_color=0.25, preserve_tone=0.0, preserve_detail=0.15),
    "dlss5 natural":   dict(mode="DLAA 1x", intensity=1.0, style="Natural", local_tone=0.0,
                            local_structure=1.0, skin_structure=-1.0, strength=0.7,
                            preserve_color=0.25, preserve_tone=0.0, preserve_detail=0.15),
    "dlss5 cinematic": dict(mode="DLAA 1x", intensity=1.0, style="Cinematic", local_tone=0.0,
                            local_structure=1.0, skin_structure=-1.0, strength=0.7,
                            preserve_color=0.25, preserve_tone=0.0, preserve_detail=0.15),
}
RESTART_ENHANCE_OPTIONS = ["off"] + list(DLSS_PRESETS.keys())


# --- KreaPhoton Upscale (v1.5, docs/07 + live calibration 2026-09-13 on existing frames).
#     What the calibration measured on Krea 2 Turbo (S08 potter's hands, 2x, native-res
#     crops + PSNR of the downscaled result against the source):
#       * every denoise >= 0.20 (t-space; sigma0 >= 0.38 after the shift) REWROTE the
#         picture - clay grain became a smooth glove with a fine mesh texture, tone drifted
#         -3..-6 levels - regardless of tile size (a single whole-image tile did the same),
#         prompt (empty / generic identical), contraction, sampler order or the M2 nudge;
#       * the latent LF anchor (tiling.LFAnchor) reduced the drift only partially even at
#         full strength (MAE 5.7 vs 10) and could not protect the 4-20 px band;
#       * post-decode back-projection (nodes.back_project, presets.FIDELITY) fixed it
#         outright: PSNR(down(result), source) 24-29 dB -> 55-59 dB, source structure back,
#         only sub-source detail kept from the model;
#       * with back-projection the useful denoise band is 0.04-0.25: sigma0 0.14 / 0.25 /
#         0.45. Above that the model's pixel-scale texture prior (the mesh) dominates.
#     denoise -> sigma0 on a 1024^2 tile (alpha 2.47): polish 0.14, detail 0.25, strong
#     0.45 - all far below the composition boundary (0.85); identity/texture LoRAs apply
#     from the first step (the whole descent runs on the texture model when the plan has
#     one). detail_a 0 (the M2 nudge is a texture push - the opposite of fidelity), anchor
#     0 (superseded by back-projection; tune can re-enable it), eta0 0 (refine speckle
#     fix, 2026-07-07), contraction 0.70 (validated refine path; 1.0 measured worse).
#     bp_lock: back-projection lock grid in source px - 1 = strict (nothing the source
#     shows can change). A lock of 2 was tried as a "repair" mode (room to re-draw a
#     slightly deformed small object below 2 source px): measured WORSE on S08 / S02 -
#     finger edges smeared, the crosshatch skin artefact of the source amplified, freckles
#     waxy, PSNR 31 dB - so every preset locks at 1 and there is no repair preset: with the
#     source-consistency guarantee the node sharpens and re-textures, it cannot re-draw
#     geometry (any denoise that could, >= 0.35, rewrites the whole frame - see above).
#     Owner 2026-09-13: "be careful with repair" -> the strongest preset stays faithful. ---
UPSCALE_REQUIRED_KEYS = ("denoise", "n_steps", "sampler", "detail_a", "anchor", "contraction", "guidance",
                         "bp_lock")
UPSCALE_PRESETS = {
    "polish": {"denoise": 0.06, "n_steps": 4, "sampler": "euler", "detail_a": 0.0, "anchor": 0.0,
               "contraction": 0.70, "guidance": "window", "bp_lock": 1.0},
    "detail": {"denoise": 0.12, "n_steps": 6, "sampler": "euler", "detail_a": 0.0, "anchor": 0.0,
               "contraction": 0.70, "guidance": "window", "bp_lock": 1.0},
    "strong": {"denoise": 0.25, "n_steps": 8, "sampler": "euler", "detail_a": 0.0, "anchor": 0.0,
               "contraction": 0.70, "guidance": "window", "bp_lock": 1.0},
}
DEFAULT_UPSCALE_PRESET = "detail"
UPSCALE_TEXTURE_START = 0.65     # texture-phase boundary of the LoRA plan (== LORA_PHASES["texture"][0])
TILE = {
    "size": 1024,        # px, the tile the DiT sees (latent 128) - 4096 tokens, alpha 2.47
    "overlap": 128,      # px, feathered blend band between neighbours (latent 16)
    "batch": 4,          # tiles per forward (RTX 5090 32 GB estimate; comfy chunks by cond, not by input batch)
    "vae_tile": 1024,    # px, tiled VAE encode/decode tile
    "vae_overlap": 128,  # px
}
ANCHOR = {
    "radius": 8,           # latent px (64 px) gaussian std of the low-pass band
    "release_sigma": 0.30, # anchor weight reaches 0 at/below this sigma
}
# Post-decode source consistency (nodes.back_project): iterated back-projection makes the
# result reproduce the source exactly when downscaled. Live finding 2026-09-13 (S08): the
# latent anchor alone left MAE 5-10/255 and a global -3..-6 tone shift; back-projection
# takes PSNR(down(result), source) from 24-29 dB to 57 dB and restores the source's own
# mid-scale structure (clay grain) that the model had replaced. 0 = off.
FIDELITY = {"bp_iters": 5}   # antialiased Down converges a little slower than box: 5 passes (pixel ops, negligible cost)


# --- KreaPhoton Upscale v2 (2026-09-13, owner: tier-1 items as a SEPARATE node, v1 untouched):
#     (1) noise inversion - the descent starts from the source's own inverted noise
#         (sampling.run_inversion) instead of a fresh draw, so a higher denoise should add
#         detail without rewriting; (2) empty-tile skipping - tiles whose source activity
#         is below `skip` get no model call (tiling.activity_map / LatentTilerV2);
#         (3) per-step grid shift with a smaller overlap (TILE_V2).
#     MEASURED 2026-09-13 (S08, S02, couple_watch, ledge_night; native crops + PSNR):
#       * inversion (invert_steps = n_steps - 1) vs the same preset without it: +0.2-0.3 dB,
#         visually identical, x2 model calls -> OFF in every preset (invert_steps 0), kept
#         as a `tune` option; the source-consistency step already fixes the trajectory;
#       * empty-tile skipping: S08 12 tiles -> 10 model tiles, result identical to the
#         no-skip run (PSNR 45.15 both) - kept on;
#       * the sharpness gain everyone sees between the two surveys comes from the
#         back-projection change (antialiased Down, 5 iterations) shared with v1, not from
#         v2: iterated back-projection is a deconvolution and sharpens with every pass
#         (Laplacian 1.8 / 2.7 / 3.4 / 4.4 / 5.4 at 1 / 2 / 3 / 5 / 8 passes, no halos up to 5).
#     Remaining values are starting hypotheses; the v1 findings
#     (detail_a 0, anchor 0, eta0 0, bp_lock 1) carry over. invert_steps = n_steps - 1 walks the
#     descent grid backwards exactly (symmetric round trip); 0 = descend
#     from fresh noise like v1 (control). `start_noise` = fresh-noise amplitude at the
#     tiny first sigma of the inversion (tune). ---
UPSCALE_V2_REQUIRED_KEYS = ("denoise", "n_steps", "invert_steps", "sampler", "detail_a", "anchor",
                            "contraction", "guidance", "bp_lock", "skip")
UPSCALE_V2_PRESETS = {
    "polish": {"denoise": 0.10, "n_steps": 6,  "invert_steps": 0,  "sampler": "euler", "detail_a": 0.0,
               "anchor": 0.0, "contraction": 1.0, "guidance": "window", "bp_lock": 1.0, "skip": 0.015},
    "detail": {"denoise": 0.20, "n_steps": 8,  "invert_steps": 0,  "sampler": "euler", "detail_a": 0.0,
               "anchor": 0.0, "contraction": 1.0, "guidance": "window", "bp_lock": 1.0, "skip": 0.015},
    "strong": {"denoise": 0.35, "n_steps": 10, "invert_steps": 0, "sampler": "euler", "detail_a": 0.0,
               "anchor": 0.0, "contraction": 1.0, "guidance": "window", "bp_lock": 1.0, "skip": 0.015},
}
DEFAULT_UPSCALE_V2_PRESET = "detail"
TILE_V2 = {
    "size": 1024, "overlap": 64, "batch": 4, "vae_tile": 1024, "vae_overlap": 128,
    "shift": True,          # per-call random grid shift (SpotDiffusion idea)
    "activity_block": 16,   # px per activity cell (= 2 latent px)
}


def validate_upscale_v2_presets(presets: dict = None):
    presets = UPSCALE_V2_PRESETS if presets is None else presets
    for name, p in presets.items():
        missing = [k for k in UPSCALE_V2_REQUIRED_KEYS if k not in p]
        if missing:
            raise ValueError(f"upscale v2 preset {name!r} is missing keys {missing}")
        if not (0.0 < float(p["denoise"]) <= 1.0):
            raise ValueError(f"upscale v2 preset {name!r}: denoise must be in (0, 1]")
        if int(p["invert_steps"]) < 0 or not (0.0 <= float(p["skip"]) < 1.0):
            raise ValueError(f"upscale v2 preset {name!r}: invert_steps >= 0, skip in [0, 1)")
        if p["guidance"] not in ("flat", "window"):
            raise ValueError(f"upscale v2 preset {name!r}: unknown guidance policy {p['guidance']!r}")
    return True


def validate_upscale_presets(presets: dict = None):
    """Import-time contract for UPSCALE_PRESETS (same idea as validate_presets)."""
    presets = UPSCALE_PRESETS if presets is None else presets
    for name, p in presets.items():
        missing = [k for k in UPSCALE_REQUIRED_KEYS if k not in p]
        if missing:
            raise ValueError(f"upscale preset {name!r} is missing keys {missing}")
        if not (0.0 < float(p["denoise"]) <= 1.0):
            raise ValueError(f"upscale preset {name!r}: denoise must be in (0, 1]")
        if p["guidance"] not in ("flat", "window"):
            raise ValueError(f"upscale preset {name!r}: unknown guidance policy {p['guidance']!r}")
        if p["guidance"] == "flat" and "cfg" not in p:
            raise ValueError(f"upscale preset {name!r} has guidance='flat' but no 'cfg'")
    return True


def preset_guidance(preset: dict, has_negative: bool):
    """(guidance_mode, cfg) the simple Sampler runs a preset with.

    "flat" presets (RAW) always run full-trajectory CFG at preset["cfg"];
    "window" presets run the M6 window only when a negative is connected
    (mode-dependent NFE cost - docs/04 item 6), else off. Centralized so no
    node reads preset keys ad hoc (the F01 bug was exactly that)."""
    policy = preset["guidance"]
    if policy == "flat":
        return "flat", float(preset["cfg"])
    if policy == "window":
        return ("window" if has_negative else "off"), float(GUIDANCE["flat_cfg"])
    raise ValueError(f"preset guidance policy must be 'flat' or 'window', got {policy!r}")


def validate_presets(presets: dict = None):
    """Every preset carries every required key (and 'flat' ones a cfg) - import-
    time contract so a half-edited preset fails loudly, not as a dead key."""
    presets = PRESETS if presets is None else presets
    for name, p in presets.items():
        missing = [k for k in PRESET_REQUIRED_KEYS if k not in p]
        if missing:
            raise ValueError(f"preset {name!r} is missing keys {missing}")
        if p["guidance"] == "flat" and "cfg" not in p:
            raise ValueError(f"preset {name!r} has guidance='flat' but no 'cfg'")
        if p["guidance"] not in ("flat", "window"):
            raise ValueError(f"preset {name!r}: unknown guidance policy {p['guidance']!r}")
    return True


# --- Face Detailer (v1.6) ---------------------------------------------------------
# passes = [(guide_px, denoise, n_steps), ...]: the face crop (bbox x crop_factor) is
# resized so its long side is guide_px and refined with refine_schedule(n_steps,
# denoise) - n_steps steps of a descent oversampled at n_steps/denoise (the "effective
# steps" rule: Impact's measured 18 x 0.35 on krea2 == our 6 steps at denoise 0.35).
# Pass 2 starts from the pass-1 result at a higher guide for skin texture.
# HONESTY LABEL: these numbers are DERIVED from the Impact Pack FaceDetailer measurement
# of 2026-09-01 (er_sde, 18 x 0.35, guide 1024, crop_factor 2.0, feather 12) and are NOT
# yet validated on the KreaPhoton sampler - owner live A/B pending (KREA2-NODES).
FACE_PRESETS = {
    "subtle":   {"passes": [(1024, 0.25, 6)],                   "id_threshold": 0.70},
    "standard": {"passes": [(1024, 0.35, 6), (1536, 0.15, 6)], "id_threshold": 0.65},
    "strong":   {"passes": [(1024, 0.45, 7), (1536, 0.20, 6)], "id_threshold": 0.60},
}
DEFAULT_FACE_PRESET = "standard"
FACE_COMMON = {
    "crop_factor": 2.0,          # crop side = max(bbox side) x factor (3.0 shrinks the face in the crop)
    "bbox_threshold": 0.45,      # YOLO confidence; 0.6 to detail the subject only
    "min_face_px": 48,           # smaller bboxes are skipped - nothing to build detail from
    "feather": 0.06,             # fraction of the crop side: mask feather = paste feather
    "dilation": 0.10,            # fraction of the bbox side added around it for the mask ellipse
    "retry_max": 3,              # attempts per pass when the identity gate fails (1 = no retry)
    "retry_denoise_step": 0.05,  # denoise -= step per retry (identity drifts with denoise)
    "retry_seed_step": 1000,     # seed += step per retry (distilled: seed+1 is ~the same latent)
    "sampler": "euler", "guidance": "window", "detail_a": 0.0,
    "invert": 0,                 # 1 = inversion-anchored refine (Upscale v2 mechanism), unmeasured on faces
}
FACE_GUIDE_MIN, FACE_GUIDE_MAX = 256, 2048


def validate_face_presets(presets: dict = None, common: dict = None):
    """Import-time contract for FACE_PRESETS / FACE_COMMON (same idea as validate_presets)."""
    presets = FACE_PRESETS if presets is None else presets
    common = FACE_COMMON if common is None else common
    for name, p in presets.items():
        passes = p.get("passes")
        if not isinstance(passes, (list, tuple)) or not passes:
            raise ValueError(f"face preset {name!r}: 'passes' must be a non-empty list")
        for entry in passes:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
                raise ValueError(f"face preset {name!r}: each pass is (guide_px, denoise, n_steps)")
            guide, denoise, steps = entry
            if int(guide) % 16 != 0 or not (FACE_GUIDE_MIN <= int(guide) <= FACE_GUIDE_MAX):
                raise ValueError(f"face preset {name!r}: guide {guide} must be a multiple of 16 in "
                                 f"[{FACE_GUIDE_MIN}, {FACE_GUIDE_MAX}]")
            if not (0.0 < float(denoise) <= 1.0):
                raise ValueError(f"face preset {name!r}: denoise must be in (0, 1]")
            if int(steps) < 1:
                raise ValueError(f"face preset {name!r}: n_steps must be >= 1")
        thr = p.get("id_threshold")
        if thr is None or not (0.0 <= float(thr) <= 1.0):
            raise ValueError(f"face preset {name!r}: id_threshold must be in [0, 1]")
    if float(common["crop_factor"]) < 1.0:
        raise ValueError("face common: crop_factor must be >= 1")
    for k in ("feather", "dilation"):
        if not (0.0 <= float(common[k]) < 0.5):
            raise ValueError(f"face common: {k} must be in [0, 0.5)")
    if int(common["retry_max"]) < 1:
        raise ValueError("face common: retry_max must be >= 1")
    if common["guidance"] not in ("flat", "window"):
        raise ValueError(f"face common: unknown guidance policy {common['guidance']!r}")
    if common["sampler"] not in ("euler", "euler_2m"):
        raise ValueError(f"face common: sampler must be euler or euler_2m, got {common['sampler']!r}")
    return True


validate_presets()
validate_upscale_presets()
validate_upscale_v2_presets()
validate_face_presets()
