"""
KreaPhoton sampler core (S6; single-lifecycle rework v1.3). One custom KSAMPLER
loop (euler/euler_2m + M2 step-relative sigma-nudge + M3 restart re-noise + M5
gated-eta ancestral stochastic component + M4 variety at its sigma boundary +
terminal x0-trajectory extrapolation), orchestrated through KreaPhotonGuider -
NOT comfy.sample.sample_custom (planning-council D11: comfy.samplers.sample()
hardcodes the stock CFGGuider, verified at comfy/samplers.py, and cannot carry
a custom guider).

v1.3 (audit F02/F03): variety no longer splits the trajectory into two
guider.sample() lifecycles. The latent axis (lf_recompose) fires INSIDE the
loop at the first model-call step at/below variety_end, on the sampler's own
state x (lf_recompose commutes with the per-channel affine + 1/(1-sigma)
rescale the old split path went through - proven in tests/test_sampling.py
[6]); the cond axis is a third cond entry the guider switches to via a flag the
loop writes into model_options at the same step. Consequences: no
process_latent_out/in round-trip, no forced eta0=0 (the V4-validated gated eta
stays on with variety), one preview/progress lifecycle, and the boundary is
now bit-identical between the simple and Advanced nodes.

The only remaining multi-segment path is the optional clean_model split
(composition on a clean checkpoint, LoRA phase on `model`) - ZPhoton pattern,
still experimental on krea2 LoRA stacks, and it keeps the eta0 safety guard.

Sampler v2, iteration 1 (2026-09-27, proposal E:\\dk\\KreaPhoton_Sampler_v2_Technical_Proposal.md,
every new mechanism opt-in, the default path bit-exact): the `rf_2m` solver,
`eta_tail` for the restart segment, and per-lifecycle step telemetry (`stats`).
Finding behind it: in txt2img the presets' gated eta (eta0 1.0) makes every
non-final model call an ancestral step, so `euler_2m` never reached its AB2 branch -
bit-identical to `euler` on every v1 preset at denoise 1 (tests/test_sampling.py
[24]). AB2 does run where eta0 is 0: the refine path (denoise < 1) and the Advanced
node's default.
"""
import math

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import torch

import os

from . import trajectory as trj
from .guidance import TRUE_SIGMA_KEY, VARIETY_FLAG, KreaPhotonGuider
from .noise import contract_noise
from .presets import ADAPTIVE, DETAIL_SIGMA, effective_delta
from .schedules import TAIL_SIGMA_MAX, validate_sigmas
from .variety import cond_tap_rotation, lf_recompose


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _detail_envelope(p: float, start: float, end: float, peak: float) -> float:
    if end <= start:
        return 0.0
    u = (p - start) / (end - start)
    if u <= 0.0 or u >= 1.0:
        return 0.0
    peak = max(0.05, min(0.95, peak))
    w = u / peak if u < peak else (1.0 - u) / (1.0 - peak)
    return _smoothstep(w)


def _detail_envelope_sigma(sigma: float, hi: float, peak: float, lo: float) -> float:
    """Detail-pressure weight as a function of the STATE's sigma (Sampler v2 P1.5): 0 at/above
    hi and at/below lo, 1 at peak, smoothstep ramps between - the step envelope's shape, but
    its meaning no longer moves with the step count, restart fraction, self-refine, a phase
    split or a foreign grid. Degenerate (not hi > peak > lo) -> 0."""
    if not hi > peak > lo or sigma >= hi or sigma <= lo:
        return 0.0
    w = (hi - sigma) / (hi - peak) if sigma > peak else (sigma - lo) / (peak - lo)
    return _smoothstep(w)


GATE_HI = 0.35   # M5-proven upper edge of the gated-eta ramp; lower edge is the
                  # calibratable preset field sigma_gate (M5-proven default 0.10).
PLUNGE_DSIGMA = 0.25          # a descending step larger than this is a plunge: M2 nudge
                               # skip, AB2 fallback to euler, no x0-extrapolation.
X0_EXTRAP_MAX_FACTOR = 2.0    # cap on the linear x0-trajectory extrapolation factor
                               # sigma_last/(sigma_prev - sigma_last); the calibrated
                               # restart grid (0.4333 -> 0.2166 -> 0) gives exactly 1.0,
                               # a foreign grid with a tiny last step could otherwise
                               # extrapolate wildly.
VARIETY_STATES = ("off", "pending", "active")
SOLVERS = ("euler", "euler_2m", "rf_2m")   # euler_2m = legacy fixed-coefficient AB2 on d;
                                           # rf_2m = DPM-Solver++(2M) on the x0 history (v2)
RF2M_MIN_STEP_RATIO = 0.5     # r = h_prev/h below this (extrapolation weight k = 1/(2r) > 1) -> first
                              # order: after a near-flat step k is unbounded while the detail nudge
                              # keeps x0 - x0_prev finite (review M1: |rf_2m - euler| ~141 on a foreign
                              # grid). The calibrated restart tails sit at r 0.75-0.98.


DETAIL_SPACES = ("step", "sigma")
PC_MODES = ("off", "selective", "auto")
REFINE_MODES = ("on", "auto")


def tail_start_index(sigmas):
    """Global index of the first step of the restart TAIL - the steps after the last ascending
    jump when that jump lands below TAIL_SIGMA_MAX (a self-refine jump lands above it) -
    or None. A property of the whole schedule, so a phase split hands it to every segment
    (like the detail envelope's progress_offset)."""
    s = [float(v) for v in sigmas]
    jumps = [k for k in range(len(s) - 1) if s[k + 1] > s[k] + 1e-6]
    if jumps and s[jumps[-1] + 1] < TAIL_SIGMA_MAX:
        return jumps[-1] + 1
    return None


def pc_step_indices(sigmas, composition_end=0.85):
    """Strategic steps of the selective predictor-corrector (proposal §17), as global indices
    of NON-final model-call steps: the first one at/below composition_end (composition ->
    identity boundary), the first one after the restart jump, and the last one at/below
    TAIL_SIGMA_MAX (a late detail step). Sorted, unique; empty on a Composition Drafts head."""
    s = [float(v) for v in sigmas]
    calls = [i for i in range(len(s) - 1) if s[i + 1] <= s[i] + 1e-6 and s[i] > 1e-6 and s[i + 1] > 1e-6]
    picks = set()
    first = next((i for i in calls if s[i] <= composition_end + 1e-9), None)
    if first is not None:
        picks.add(first)
    t = tail_start_index(s)
    if t is not None and t in calls:
        picks.add(t)
    late = [i for i in calls if s[i] <= TAIL_SIGMA_MAX + 1e-9]
    if late:
        picks.add(late[-1])
    return sorted(picks)


def _rf_lambda(sigma: float) -> float:
    """Half log-SNR of the CONST / rectified-flow state x = (1-sigma)*x0 + sigma*eps,
    lambda = log((1-sigma)/sigma) - the time variable of DPM-Solver for this family."""
    return math.log((1.0 - sigma) / sigma)


def _rf2m_x0(x0, x0_prev, s_prev: float, s_cur: float, s_next: float):
    """Effective x0 of a DPM-Solver++(2M) step s_cur -> s_next on a CONST model.

    The step itself stays the exact first-order transport x_next = (s_next/s_cur)*x +
    (1 - s_next/s_cur)*x0_eff (RF Euler == DDIM: exact for a constant x0); the second
    order comes from extrapolating the x0 predictions along lambda,
    x0_eff = (1 + 1/(2r))*x0 - 1/(2r)*x0_prev with r = h_prev/h - variable-step by
    construction, unlike the fixed 1.5/-0.5 AB2 of euler_2m. The sigmas are TRUE
    trajectory sigmas (the state's), not the detail-nudged sigma_model."""
    lam_cur = _rf_lambda(s_cur)
    h = _rf_lambda(s_next) - lam_cur
    k = h / (2.0 * (lam_cur - _rf_lambda(s_prev)))
    return (1.0 + k) * x0 - k * x0_prev


def format_stats(stats: dict, guider=None) -> str:
    """One-line numerical telemetry of a sampler lifecycle (Sampler v2 P0.2): which
    integration branch every model call took, and the history resets by reason."""
    resets = ",".join("%s:%d" % kv for kv in sorted(stats.get("resets", {}).items())) or "-"
    line = ("solver=%s calls=%d ancestral=%d euler=%d ab2=%d rf2m=%d readout=%d jumps=%d history_resets=%s"
            % (stats.get("solver"), stats.get("model_calls", 0), stats.get("ancestral", 0),
               stats.get("euler", 0), stats.get("ab2", 0), stats.get("rf2m", 0), stats.get("final", 0),
               stats.get("jumps", 0), resets))
    if stats.get("heun") or stats.get("pc"):
        line += " heun=%d pc_calls=%d" % (stats.get("heun", 0), stats.get("pc", 0))
    if stats.get("pc_motion"):
        line += " pc_motion=%s" % "/".join("-" if v is None else "%.3f" % v for v in stats["pc_motion"])
    if stats.get("refine_motion") is not None:
        line += " refine=%s(mf %.3f)" % ("skipped" if stats.get("refine_skipped") else "run",
                                         stats["refine_motion"])
    if guider is not None:
        line += " guided=%d/%d" % (getattr(guider, "n_guided", 0), getattr(guider, "n_calls", 0))
    return line


def format_trace(stats: dict) -> str:
    """The per-call x0 motion record of a traced lifecycle (proposal P3.1), one token per
    model call: sigma:all/lf/mf/hf relative change against the previous call's x0."""
    parts = []
    for r in stats.get("traj", []):
        if r.get("mf") is None:
            parts.append("%.3f:-" % r["sigma"])
        else:
            parts.append("%.3f:%.3f/%.3f/%.3f/%.3f" % (r["sigma"], r["all"], r["lf"], r["mf"], r["hf"]))
    return "x0 motion (sigma:all/lf/mf/hf) " + " ".join(parts)


def _gated_eta(sigma_next: float, eta0: float, sigma_gate: float, gate_hi: float = GATE_HI) -> float:
    """eta(sigma_next) = eta0 * smoothstep(sigma_next; sigma_gate, gate_hi).
    Exactly 0 below sigma_gate (M5: terminal ancestral injection is the
    dark-blotch driver), exactly eta0 at/above gate_hi (M5: mid-phase unchanged)."""
    if gate_hi <= sigma_gate:
        return eta0
    u = (sigma_next - sigma_gate) / (gate_hi - sigma_gate)
    return eta0 * _smoothstep(u)


def variety_perturb(x, sigma: float, x0_hat, seed_v: int, a: float):
    """Latent variety axis applied to the NOISE component only (v1.3.1).

    The loop state is x = (1-sigma)*x0 + sigma*eps (CONST). Re-composing the
    LF band of the whole state also re-composes the LF band of the model's
    committed structure estimate x0 - a state whose low frequencies no longer
    agree with its high frequencies. Without guidance the distillate absorbs
    that (V3/V13: texture-only variety, identity kept), but inside the M6
    guidance window the cond/uncond disagreement it creates is amplified up to
    2.25x and surfaces as glowing LF blobs on some seeds (V14b D2, 2026-09-07).

    Fix: recover the current noise realization from the previous model call,
    eps = (x - (1-sigma)*x0_hat)/sigma (exact for euler on CONST, effective
    noise after an ancestral step), re-compose ONLY its LF band and rebuild the
    state around the untouched x0_hat. Variance-preserving on eps, structure
    untouched, guidance sees a self-consistent state. With no previous
    prediction (boundary at the first step, x is pure noise) the whole state is
    the noise and lf_recompose applies to it directly.
    """
    if x0_hat is None or sigma <= 1e-6:
        return lf_recompose(x, seed_v=seed_v, a=a)
    eps = (x - (1.0 - sigma) * x0_hat) / sigma
    eps = lf_recompose(eps, seed_v=seed_v, a=a)
    return (1.0 - sigma) * x0_hat + sigma * eps


def plan_segments(sig_list, *, composition_end=0.85, has_clean=False, texture_start=0.65,
                  has_texture=False):
    """Segment plan for the multi-model split (v1.4 phase models). Returns a list of
    (start_idx, end_idx, phase) with phase in {"composition", "identity", "texture"};
    a single ("identity") segment when no phase model is connected.

    composition: sigma > composition_end, run on clean_model (layout without LoRA
                 steering); identity: the rest, on `model`; texture: from the first
                 index with sigma <= texture_start - on the calibrated grids that is
                 the plunge readout (sigma 0) right before the restart jump, so the
                 texture model owns exactly the restart segment.
    A cut at index 0 (a partial descent that starts below the boundary) sets the
    starting phase instead of opening an empty segment.
    Pure, unit-tested (tests/test_sampling.py [13])."""
    n = len(sig_list)
    cuts = []
    if has_clean:
        i = next((k for k, s in enumerate(sig_list) if s <= composition_end), None)
        if i is not None and 0 <= i < n - 1:
            cuts.append((i, "identity"))
    if has_texture:
        # texture phase = the restart segment: from the readout before the LAST
        # ascending jump when that jump lands at/below texture_start (a self-
        # refine jump lands higher and stays in the identity phase); a plain
        # descending schedule falls back to the first sigma <= texture_start.
        jumps = [k for k in range(n - 1) if sig_list[k + 1] > sig_list[k] + 1e-6]
        i = None
        if jumps and sig_list[jumps[-1] + 1] <= texture_start + 1e-9:
            i = jumps[-1]
        if i is None:
            i = next((k for k, s in enumerate(sig_list) if s <= texture_start), None)
        if i is not None and 0 <= i < n - 1:
            cuts.append((i, "texture"))
    order = {"identity": 0, "texture": 1}     # both at one index: the later phase wins
    cuts.sort(key=lambda c: (c[0], order[c[1]]))
    phase0 = "composition" if has_clean else "identity"
    segs = []
    start, phase = 0, phase0
    for idx, next_phase in cuts:
        if idx <= start:
            phase = next_phase
            continue
        segs.append((start, idx, phase))
        start, phase = idx, next_phase
    segs.append((start, n - 1, phase))
    return [s for s in segs if s[1] > s[0]]


def variety_boundary_index(sigmas, variety_end: float):
    """Index of the first MODEL-CALL step (not the restart jump, not a sigma=0
    readout) whose sigma is at/below variety_end - the step at which both variety
    axes fire. None if no such step exists (variety never applies)."""
    s = [float(v) for v in sigmas]
    for i in range(len(s) - 1):
        if s[i + 1] > s[i] + 1e-6:
            continue
        if s[i] <= 1e-6:
            continue
        if s[i] <= variety_end + 1e-9:
            return i
    return None


class NonFiniteModelOutput(RuntimeError):
    """The diffusion model returned NaN/Inf. On Krea 2 the known cause is fp16 compute: fp16
    weights run fp16 (comfy.model_management.unet_dtype) and comfy/ldm/krea2 has no overflow
    clamp (2026-10-01). Stopping beats decoding a black / confetti frame."""


def check_finite(den, sigma):
    finite = torch.isfinite(den)
    if not bool(finite.all()):
        raise NonFiniteModelOutput(
            "KreaPhoton: the model returned %d non-finite values at sigma %.4f. On Krea 2 this is "
            "fp16 overflow - run the checkpoint in bf16 (start ComfyUI with --bf16-unet, or use a "
            "bf16 copy of the file)." % (int((~finite).sum()), float(sigma)))
    return den


@torch.no_grad()
def kreaphoton_sampler_loop(model, x, sigmas, extra_args=None, callback=None, disable=None,
                            detail_amount=0.0, detail_start=0.15, detail_end=0.95, detail_peak=0.6,
                            order=1, eta0=0.0, sigma_gate=0.10, restart_seed=0,
                            variety_a_latent=0.0, variety_end=0.96, variety_seed=0,
                            variety_state="off", x0_extrapolation=0.0,
                            progress_offset=0, progress_total=0, restart_hook=None,
                            coherence_jump=0.0, coherence_jump_sigma=0.93,
                            tiler=None, x0_hook=None, ref_hook=None,
                            solver=None, eta_tail=None, stats=None, tail_start=None,
                            detail_space="step", detail_sigma=None, x0_extrap_order=1, x0_trust=0.0,
                            pc="off", pc_steps=None, pc_threshold=None, composition_end=0.85,
                            refine_mode="on", refine_threshold=None, trace=False):
    """The KSAMPLER sampler_function. Pure numerical loop, testable with a mock
    model callable (tests/test_sampling.py).

    Sampler v2 iteration 2 (all default to the v1 behaviour):
    detail_space:  "step" (v1: envelope over p = step index / total) | "sigma": the envelope
                   over the state's sigma, detail_sigma = (hi, peak, lo), default
                   presets.DETAIL_SIGMA - split-invariant without progress_offset.
    x0_extrap_order / x0_trust: terminal x0 extrapolation through the last two (1, v1) or
                   three (2, quadratic Lagrange) model calls of the segment; x0_trust > 0
                   limits the extrapolation step pointwise to x0_trust x the local (3x3)
                   rms of the last x0 step (trajectory.trust_clamp). 1 / 0 = the v1 code.
    pc:            predictor-corrector at the strategic steps (pc_step_indices, GLOBAL
                   indices via pc_steps for a phase split): "selective" = always, "auto" =
                   when the x0 estimate moved more than pc_threshold (presets.ADAPTIVE)
                   since the previous model call. Heun on the deterministic part - the
                   whole step, or the down-step to sigma_down of an ancestral step before
                   its re-noise; the corrector is one extra model call at the target sigma
                   (un-nudged). "off" = v1.
    refine_mode:   "on" (v1) | "auto": at a self-refine jump (lands at/above TAIL_SIGMA_MAX)
                   the refine segment is skipped - no RNG draw, no model call - when the
                   draft's mid-frequency change over the identity phase (x0 at the first
                   call at/below composition_end -> the plunge readout) is below
                   refine_threshold (presets.ADAPTIVE).
    trace:         record trajectory.motion against the previous call's x0 at every model
                   call into stats["traj"] (proposal P3.1).
    Every model call also publishes its TRUE trajectory sigma under
    guidance.TRUE_SIGMA_KEY in model_options (read only by a guider in guidance_sigma
    "true" mode).

    solver (Sampler v2): "euler" | "euler_2m" | "rf_2m"; None = derived from `order`
                   (1 -> euler, 2 -> euler_2m), bit-exact with the v1 calls. euler_2m is
                   the legacy AB2 on d with fixed equal-step coefficients; rf_2m is
                   DPM-Solver++(2M) on the x0 history (_rf2m_x0). Both multistep
                   solvers need two CONSECUTIVE deterministic steps - under the presets'
                   txt2img eta (eta0 1.0, gated) they get none, every non-final step is
                   ancestral; with eta0 0 (refine path, Advanced default) they run.
    eta_tail:      eta0 replacement for the TAIL - the model-call steps after the last
                   ascending jump when that jump lands below PLUNGE_SIGMA_FLOOR (the
                   restart segment; a self-refine jump lands above the floor and keeps
                   eta0). Still gated by sigma_gate. 0 = deterministic restart tail
                   (proposal P1 "stochastic structure + deterministic texture tail").
                   None = eta0 everywhere (bit-exact). No-op on a schedule without such
                   a jump.
    tail_start:    GLOBAL index of the tail's first step (tail_start_index of the full
                   schedule) for a segment of a phase split, compared against
                   progress_offset + i; None = computed from this call's own sigmas.
    stats:         optional dict filled with the branch counters of this lifecycle
                   (model_calls / ancestral / euler / ab2 / rf2m / final / jumps and
                   resets = {reason: count} of multistep histories actually dropped) -
                   see format_stats. Every trajectory-changing event declares the
                   history invalid: restart jump, ancestral step, variety re-composition,
                   coherence jump, readout.

    ref_hook (2026-09-22 spike, ref_mean.py): optional callable(x, denoised,
                   sigma, base_model) -> denoised applied after x0_hook; unlike
                   x0_hook it also sees the loop STATE x and the true sigma of
                   that state (not the detail-nudged sigma_model), which the
                   reference-mean posterior needs. None = bit-exact.

    tiler (v1.5 Upscale): optional callable(model, x, sigma_model, extra_args)
                   -> denoised that replaces the single model call - tiling.LatentTiler
                   cuts x into overlapping tiles, batches them through the model
                   and merges the x0 predictions with feathered weights, so every
                   step re-blends neighbouring tiles (no seams). None = one call.
    x0_hook: optional callable(denoised, sigma_model) -> denoised applied right
                   after every model call (before the integration step and the
                   preview callback) - tiling.LFAnchor pulls the low-frequency
                   band back to a reference latent. None = bit-exact.

    coherence_jump (v1.4, Power-Nodes jump-back): ONCE, at the first model-call
                   step with sigma <= coherence_jump_sigma, the state is rescaled
                   by (1-s_decl)/(1-s_cur) and the model is told s_decl =
                   s_cur + coherence_jump*(1-s_cur); the step then integrates from
                   s_decl. That is exactly what comfy's noise-scaling round-trip
                   does to Power Nodes' 0.920 -> 0.935 stage boundary (0.19 in
                   these units), made explicit: the model sees a state with the
                   correct signal-to-noise ratio but 20% less amplitude than the
                   declared sigma implies, and commits harder to structure on the
                   composition-phase step. 0 = off (bit-exact).

    restart_hook: optional callable(x, base_model) -> x applied to the plunge
                   readout x0 right before the restart re-noise (v1.4 restart
                   enhance, see enhance.py). base_model is the comfy BaseModel
                   behind the guider (process_latent_in/out), or None under a
                   plain mock model.

    progress_offset / progress_total: global step index of this segment's first
                   step and the total step count of the FULL schedule, so the M2
                   detail envelope p = (offset + i) / (total - 1) is identical
                   whether the schedule runs in one lifecycle or is split at a
                   clean_model boundary. ROOT CAUSE of the S9 "split speckle /
                   mosaic" and the 2026-07-06 "variety@0.90 droplets" (v1.3
                   finding, V15 L5 on a LoRA rig): a segment restarted its envelope
                   at p=0, so the restart steps got a nudge of ~0.58 x step instead
                   of ~0.18 (sigma_model 0.52 at a true sigma of 0.65) - the model
                   then under-removes the re-noise and the leftover decodes as
                   coloured confetti. 0 = single-lifecycle default.

    variety_state: "off"     - no boundary logic, guider flag stays False;
                   "pending" - at the boundary step apply lf_recompose (if
                               variety_a_latent>0) and raise the guider flag;
                   "active"  - flag raised from the first step (a segment that
                               starts after the boundary, clean_model split only).
    x0_extrapolation: 0..1, terminal step only. Linear extrapolation of the x0
                   prediction trajectory to sigma=0 (denoised + f*(denoised -
                   prev_denoised), f = sigma_last/(sigma_prev - sigma_last),
                   capped at X0_EXTRAP_MAX_FACTOR), blended by this weight into
                   the plain final denoised. Skipped on a plunge step and when
                   the segment has no previous model call.
    """
    extra_args = {} if extra_args is None else extra_args
    if variety_state not in VARIETY_STATES:
        raise ValueError(f"variety_state must be one of {VARIETY_STATES}, got {variety_state!r}")
    if solver is None:
        solver = "euler_2m" if order >= 2 else "euler"
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    if detail_space not in DETAIL_SPACES:
        raise ValueError(f"detail_space must be one of {DETAIL_SPACES}, got {detail_space!r}")
    if pc not in PC_MODES:
        raise ValueError(f"pc must be one of {PC_MODES}, got {pc!r}")
    if refine_mode not in REFINE_MODES:
        raise ValueError(f"refine_mode must be one of {REFINE_MODES}, got {refine_mode!r}")
    if int(x0_extrap_order) not in (1, 2):
        raise ValueError(f"x0_extrap_order must be 1 or 2, got {x0_extrap_order!r}")
    d_hi, d_peak, d_lo = DETAIL_SIGMA if detail_sigma is None else detail_sigma
    pc_thr = ADAPTIVE["pc_motion"] if pc_threshold is None else float(pc_threshold)
    refine_thr = ADAPTIVE["refine_mf_motion"] if refine_threshold is None else float(refine_threshold)
    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    n_total = int(progress_total) if int(progress_total) > 0 else n
    p_off = int(progress_offset)
    gen = torch.Generator(device="cpu").manual_seed((int(restart_seed) + 0x5EED) & 0xffffffffffffffff)
    old_d = None                  # euler_2m history: d of the previous deterministic step
    old_x0 = old_s = None         # rf_2m history: its x0 prediction and true sigma
    prev_denoised = None
    prev_sigma_model = None
    prev2_denoised = prev2_sigma_model = None   # one call further back (quadratic extrapolation)
    mprev = None                  # x0 of the previous model call, never reset (motion metrics only)
    x0_anchor = None              # x0 at the first call at/below composition_end (refine 'auto')
    skip_until = -1               # refine 'auto': last index of a skipped refine segment

    st = {} if stats is None else stats
    st.update(solver=solver, model_calls=0, ancestral=0, euler=0, ab2=0, rf2m=0, heun=0, final=0, jumps=0,
              resets={}, pc=0, pc_motion=[], refine_skipped=0, refine_motion=None, traj=[])

    def drop_history(reason):
        # counted only when the ACTIVE solver's history is lost (euler keeps none)
        nonlocal old_d, old_x0, old_s
        live = old_d if solver == "euler_2m" else old_x0 if solver == "rf_2m" else None
        if live is not None:
            st["resets"][reason] = st["resets"].get(reason, 0) + 1
        old_d = old_x0 = old_s = None

    # restart tail for eta_tail, as a LOCAL index (a segment gets the global one minus its offset)
    tail_from = None
    if eta_tail is not None:
        tail_from = tail_start_index(sigmas) if tail_start is None else int(tail_start) - p_off
    # strategic predictor-corrector steps as GLOBAL indices (compared against p_off + i)
    pc_set = set()
    if pc != "off":
        pc_set = (set(int(k) for k in pc_steps) if pc_steps is not None
                  else {k + p_off for k in pc_step_indices(sigmas, composition_end)})

    # Variety hand-off to the guider (cond axis). extra_args["model_options"] is
    # the per-lifecycle clone comfy passes to every model call of this loop
    # (CFGGuider.inner_sample -> KSAMPLER.sample -> KSamplerX0Inpaint ->
    # guider.predict_noise), so a key written here is visible in predict_noise.
    model_options = extra_args.get("model_options")
    if model_options is None:
        model_options = {}
        extra_args["model_options"] = model_options
    variety_done = variety_state != "pending"
    model_options[VARIETY_FLAG] = variety_state == "active"
    jump_done = coherence_jump <= 0.0

    def call(xx, sig_model, sig_true):
        # one model evaluation (tiled or not) + the x0 hooks; the guider may read sig_true
        model_options[TRUE_SIGMA_KEY] = sig_true
        if tiler is not None:
            den = tiler(model, xx, sig_model, extra_args)
        else:
            den = model(xx, sig_model * s_in, **extra_args)
        den = check_finite(den, sig_true)
        if x0_hook is not None:
            den = x0_hook(den, sig_model)
        if ref_hook is not None:
            base_model = getattr(getattr(model, "inner_model", None), "inner_model", None)
            den = ref_hook(xx, den, sig_true, base_model)
        return den

    for i in range(n):
        if i <= skip_until:
            continue
        s_cur = float(sigmas[i])
        s_next = float(sigmas[i + 1])

        # --- restart segment: ascending jump -> proper flow re-noise (M3) ---
        if s_next > s_cur + 1e-6:
            if refine_mode == "auto" and s_next >= TAIL_SIGMA_MAX - 1e-9:
                # a self-refine jump: skip the whole refine segment (up to its readout) when the
                # draft stopped moving over the identity phase - x is the plunge readout here
                mv = trj.motion(x, x0_anchor)["mf"] if x0_anchor is not None else None
                st["refine_motion"] = mv
                if mv is not None and mv < refine_thr:
                    skip_until = next((k for k in range(i + 1, n) if float(sigmas[k + 1]) <= 1e-6), n - 1)
                    st["refine_skipped"] += 1
                    continue
            # the state before a re-noise is the draft x0 (a segment of a phase split starts here
            # without a previous model call - its motion reference is the draft it was handed)
            mprev = x
            if restart_hook is not None:
                base_model = getattr(getattr(model, "inner_model", None), "inner_model", None)
                x = restart_hook(x, base_model)
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = (1.0 - s_next) * x + s_next * eps
            st["jumps"] += 1
            drop_history("restart")
            prev_denoised = None
            prev_sigma_model = None
            prev2_denoised = prev2_sigma_model = None
            continue

        if s_cur <= 1e-6:
            continue

        # --- variety boundary (M4): once, at the first model-call step at/below
        # variety_end. Latent axis on the loop state x (normalized model space;
        # lf_recompose is self-referential in mean/std, so the space is
        # irrelevant to it - see variety.py), cond axis via the guider flag. ---
        if not variety_done and s_cur <= variety_end + 1e-9:
            if variety_a_latent > 0.0:
                x = variety_perturb(x, s_cur, prev_denoised, int(variety_seed), float(variety_a_latent))
                drop_history("variety")   # multistep memory is not meaningful across a re-composition
                prev_denoised = None
                prev_sigma_model = None
                prev2_denoised = prev2_sigma_model = None
            elif solver == "rf_2m":
                # conditioning-only switch: the x0 predictions change field here too. euler_2m
                # keeps its v1 behaviour (history kept) - the default path stays bit-exact
                drop_history("variety")
            model_options[VARIETY_FLAG] = True
            variety_done = True

        # --- coherence jump-back (v1.4): once, first model-call step at/below
        # coherence_jump_sigma. State rescaled, declared sigma raised; the step
        # integrates from the declared sigma (see docstring). ---
        if not jump_done and s_cur <= coherence_jump_sigma + 1e-9:
            s_decl = s_cur + float(coherence_jump) * (1.0 - s_cur)
            s_decl = min(s_decl, 0.999)
            if s_decl > s_next + 1e-6:
                x = x * ((1.0 - s_decl) / (1.0 - s_cur))
                s_cur = s_decl
                drop_history("coherence")
                prev_denoised = None
                prev_sigma_model = None
                prev2_denoised = prev2_sigma_model = None
            jump_done = True

        # --- detail boost: step-relative sigma nudge (M2), skip on plunge/final ---
        p = (p_off + i) / max(n_total - 1, 1)
        is_final = s_next <= 1e-6
        is_plunge = (s_cur - s_next) > PLUNGE_DSIGMA
        if is_final or is_plunge:
            a = 0.0
        else:
            if detail_space == "sigma":
                a = detail_amount * _detail_envelope_sigma(s_cur, d_hi, d_peak, d_lo)
            else:
                a = detail_amount * _detail_envelope(p, detail_start, detail_end, detail_peak)
            a = max(-1.0, min(1.0, a))
        sigma_model = max(1e-4, s_cur - a * (s_cur - s_next))

        denoised = call(x, sigma_model, s_cur)
        st["model_calls"] += 1
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i],
                      "sigma_hat": sigmas[i], "denoised": denoised})

        # --- trajectory metrics (P3.1) and the adaptive decisions they drive ---
        if trace:
            rec = {"sigma": s_cur}
            if mprev is not None:
                rec.update(trj.motion(denoised, mprev))
            st["traj"].append(rec)
        do_pc = False
        if pc != "off" and not is_final and (p_off + i) in pc_set:
            if pc == "selective":
                do_pc = True
            else:
                mv = trj.motion(denoised, mprev)["all"] if mprev is not None else None
                st["pc_motion"].append(mv)
                do_pc = mv is not None and mv > pc_thr
        if x0_anchor is None and refine_mode == "auto" and s_cur <= composition_end + 1e-9:
            x0_anchor = denoised
        mprev = denoised

        if is_final:
            x = denoised
            if (x0_extrapolation > 0.0 and not is_plunge and prev_denoised is not None
                    and prev_sigma_model is not None and prev_sigma_model - sigma_model > 1e-6):
                # terminal x0-trajectory extrapolation: the model's x0 estimate
                # sharpens as sigma -> 0; continue that trend past the last evaluation
                # point instead of stopping at it.
                if int(x0_extrap_order) == 1 and x0_trust <= 0.0:
                    factor = min(X0_EXTRAP_MAX_FACTOR, sigma_model / (prev_sigma_model - sigma_model))
                    x = denoised + float(x0_extrapolation) * factor * (denoised - prev_denoised)
                else:   # v2: quadratic through three calls and / or the local trust region
                    hist = ([(prev2_denoised, prev2_sigma_model)] if prev2_denoised is not None else [])
                    hist += [(prev_denoised, prev_sigma_model), (denoised, sigma_model)]
                    delta = trj.x0_zero_delta(hist, x0_extrapolation, int(x0_extrap_order), X0_EXTRAP_MAX_FACTOR)
                    if delta is not None:
                        x = denoised + trj.trust_clamp(delta, denoised - prev_denoised, x0_trust)
            st["final"] += 1
            drop_history("final")
            prev_denoised = None
            prev_sigma_model = None
            prev2_denoised = prev2_sigma_model = None
            continue

        e0 = eta_tail if (tail_from is not None and i >= tail_from) else eta0
        eta = _gated_eta(s_next, e0, sigma_gate)
        if eta > 0.0:
            # CONST/rectified-flow ancestral step, ported from comfy's own
            # sample_euler_ancestral_RF (k_diffusion/sampling.py) so our
            # detail-nudged sigma_model integrates with the SAME math CONST
            # models expect. History reset - multistep memory is not meaningful
            # across a stochastic jump.
            downstep_ratio = 1.0 + (s_next / s_cur - 1.0) * eta
            sigma_down = s_next * downstep_ratio
            alpha_next = 1.0 - s_next
            alpha_down = 1.0 - sigma_down
            renoise_coeff = max(0.0, s_next ** 2 - sigma_down ** 2 * alpha_next ** 2 / alpha_down ** 2) ** 0.5
            ratio = sigma_down / s_cur
            if do_pc and sigma_down > 1e-6:
                # Heun on the deterministic down-step, then the same re-noise
                x_pred = ratio * x + (1.0 - ratio) * denoised
                d1 = (x - denoised) / s_cur
                d2 = (x_pred - call(x_pred, sigma_down, sigma_down)) / sigma_down
                x = x + 0.5 * (d1 + d2) * (sigma_down - s_cur)
                st["pc"] += 1
            else:
                x = ratio * x + (1.0 - ratio) * denoised
            eps = torch.randn(x.shape, generator=gen, device="cpu").to(x)
            x = (alpha_next / alpha_down) * x + eps * renoise_coeff
            st["ancestral"] += 1
            drop_history("ancestral")
        else:
            d = (x - denoised) / s_cur
            dt = s_next - s_cur
            if do_pc:
                # Heun: Euler predictor to s_next, one corrector call there, averaged slope
                x_pred = x + d * dt
                d_use = 0.5 * (d + (x_pred - call(x_pred, s_next, s_next)) / s_next)
                st["heun"] += 1
                st["pc"] += 1
            elif solver == "euler_2m" and old_d is not None and abs(dt) <= PLUNGE_DSIGMA:
                d_use = 1.5 * d - 0.5 * old_d   # Adams-Bashforth 2, equal-step coefficients (legacy)
                st["ab2"] += 1
            elif (solver == "rf_2m" and old_x0 is not None and abs(dt) <= PLUNGE_DSIGMA
                    and _rf_lambda(s_cur) - _rf_lambda(old_s) > max(
                        1e-6, RF2M_MIN_STEP_RATIO * (_rf_lambda(s_next) - _rf_lambda(s_cur)))):
                # DPM-Solver++(2M): the Euler transport with the lambda-extrapolated x0
                d_use = (x - _rf2m_x0(denoised, old_x0, old_s, s_cur, s_next)) / s_cur
                st["rf2m"] += 1
            else:
                d_use = d
                st["euler"] += 1
            x = x + d_use * dt
            old_d = d
            # lambda(1) = -inf: a step starting at sigma=1 leaves no usable x0 history
            old_x0, old_s = (denoised, s_cur) if s_cur < 1.0 - 1e-6 else (None, None)

        prev2_denoised, prev2_sigma_model = prev_denoised, prev_sigma_model
        prev_denoised = denoised
        prev_sigma_model = sigma_model

    return x


def zero_conditioning(cond):
    """Zeroed-out copy of a conditioning (honest unconditional when no real
    negative is provided - ZPhoton precedent, safe regardless of guidance mode
    since cfg=1.0/flat/window all treat an empty negative identically)."""
    out = []
    for t, d in cond:
        d = d.copy()
        pooled = d.get("pooled_output")
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        out.append([torch.zeros_like(t), d])
    return out


def _rotate_conditioning(cond, taps, a, seed):
    """Apply cond_tap_rotation to the TENSOR component of a comfy CONDITIONING
    list ([tensor, dict], ...), preserving dict metadata untouched."""
    out = []
    for t, d in cond:
        out.append([cond_tap_rotation(t, taps, a, seed), d.copy()])
    return out


def _run_one(model, positive, negative, latent_dict, sigmas, *, seed,
             guidance_mode, flat_cfg, delta, lo, hi,
             noise, add_noise, contraction, per_channel_contraction, manifold_std, manifold_mean,
             detail_amount, detail_start, detail_end, detail_peak, order, eta0, sigma_gate,
             x0_extrapolation=0.0, positive_variety=None, variety_a_latent=0.0,
             variety_end=0.96, variety_seed=0, variety_state="off",
             progress_offset=0, progress_total=0, guidance_rescale=0.0,
             pag_scale=0.0, pag_lo=0.72, pag_hi=0.93, pag_blocks="8-15", restart_hook=None,
             coherence_jump=0.0, coherence_jump_sigma=0.93, tiler=None, x0_hook=None, ref_hook=None,
             solver=None, eta_tail=None, tail_start=None, guidance_sigma="model",
             detail_space="step", detail_sigma=None, x0_extrap_order=1, x0_trust=0.0,
             pc="off", pc_steps=None, composition_end=0.85, refine_mode="on", trace=False):
    latent = comfy.sample.fix_empty_latent_channels(model, latent_dict["samples"])
    if latent.ndim != 5 or latent.shape[1] != 16:
        raise ValueError(
            f"KreaPhoton: expected a Krea 2 (Wan21) latent of shape (B, 16, 1, H, W), got "
            f"{tuple(latent.shape)} - use KreaPhoton Empty Latent / EmptySD3LatentImage (16ch) "
            f"or a VAE-encoded image from the krea2 VAE, and a Krea 2 checkpoint as `model`")

    if negative is None:
        negative = zero_conditioning(positive)

    if not add_noise:
        noise = torch.zeros(latent.shape, dtype=latent.dtype, device="cpu")
    elif noise is None:
        noise = comfy.sample.prepare_noise(latent, seed)
    noise = contract_noise(noise, strength=contraction, per_channel=per_channel_contraction,
                            manifold_std=manifold_std, manifold_mean=manifold_mean)

    stats = {}
    sampler = comfy.samplers.KSAMPLER(kreaphoton_sampler_loop, extra_options={
        "detail_amount": float(detail_amount), "detail_start": float(detail_start),
        "detail_end": float(detail_end), "detail_peak": float(detail_peak),
        "order": int(order), "eta0": float(eta0), "sigma_gate": float(sigma_gate),
        "restart_seed": int(seed),
        "variety_a_latent": float(variety_a_latent), "variety_end": float(variety_end),
        "variety_seed": int(variety_seed), "variety_state": variety_state,
        "x0_extrapolation": float(x0_extrapolation),
        "progress_offset": int(progress_offset), "progress_total": int(progress_total),
        "restart_hook": restart_hook,
        "coherence_jump": float(coherence_jump), "coherence_jump_sigma": float(coherence_jump_sigma),
        "tiler": tiler, "x0_hook": x0_hook, "ref_hook": ref_hook,
        "solver": solver, "eta_tail": eta_tail, "stats": stats, "tail_start": tail_start,
        "detail_space": detail_space, "detail_sigma": detail_sigma, "x0_extrap_order": int(x0_extrap_order),
        "x0_trust": float(x0_trust), "pc": pc, "pc_steps": pc_steps, "composition_end": float(composition_end),
        "refine_mode": refine_mode, "trace": bool(trace),
    })

    # One guider for every path: "off" is cond_scale 1.0 (stock cfg1 optimization,
    # no uncond NFE), "flat" constant cfg (RAW), "window" the M6 ramp, "band" the v2 band.
    guider = KreaPhotonGuider(model, mode=guidance_mode, cfg=flat_cfg, delta=delta, lo=lo, hi=hi,
                              rescale=guidance_rescale, pag_scale=pag_scale, pag_lo=pag_lo,
                              pag_hi=pag_hi, pag_blocks=pag_blocks, guidance_sigma=guidance_sigma)
    guider.set_conds(positive, negative, positive_variety=positive_variety)

    try:
        import latent_preview
        callback = latent_preview.prepare_callback(model, len(sigmas) - 1)
    except Exception:
        callback = None
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(noise, latent, sampler, sigmas,
                            denoise_mask=latent_dict.get("noise_mask"),
                            callback=callback, disable_pbar=disable_pbar, seed=seed)
    print("[KreaPhoton] " + format_stats(stats, guider))
    if trace:
        print("[KreaPhoton] " + format_trace(stats))
    samples = samples.to(device=comfy.model_management.intermediate_device(),
                         dtype=comfy.model_management.intermediate_dtype())
    out = latent_dict.copy()
    out["samples"] = samples
    return out


@torch.no_grad()
def kreaphoton_invert_loop(model, x, sigmas, extra_args=None, callback=None, disable=None,
                           tiler=None, x0_hook=None):
    """Noise inversion for the CONST / rectified-flow model (v2 Upscale): the KSAMPLER
    sampler_function for an ASCENDING sigma array. Euler steps run the flow backwards -
    d = (x - x0_hat) / sigma is exactly dx/dsigma of x = (1-sigma)*x0 + sigma*eps - so
    the state climbs from the (almost) clean source latent at sigmas[0] to sigmas[-1]
    carrying the source's OWN noise realisation: the noise a forward descent from there
    would turn back into the source. Deterministic (no RNG). tiler / x0_hook exactly as
    in kreaphoton_sampler_loop. comfy hands us x = sigmas[0]*noise + (1-sigmas[0])*z and
    divides the result by (1-sigmas[-1]) - the descent lifecycle that follows multiplies
    by the same factor (CONST noise_scaling with zero noise), so the hand-over is exact."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    for i in range(n):
        s_cur = float(sigmas[i])
        s_next = float(sigmas[i + 1])
        if s_next <= s_cur + 1e-9 or s_cur <= 1e-6:
            raise ValueError("kreaphoton_invert_loop: sigmas must be strictly ascending and > 0 "
                             "(got %.4f -> %.4f at %d)" % (s_cur, s_next, i))
        if tiler is not None:
            denoised = tiler(model, x, s_cur, extra_args)
        else:
            denoised = model(x, s_cur * s_in, **extra_args)
        denoised = check_finite(denoised, s_cur)
        if x0_hook is not None:
            denoised = x0_hook(denoised, s_cur)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        d = (x - denoised) / s_cur
        x = x + d * (s_next - s_cur)
    return x


def run_inversion(model, positive, negative, latent_dict, sigmas_asc, *, seed,
                  guidance_mode="off", flat_cfg=1.15, delta=1.25, lo=0.7, hi=0.9,
                  start_noise=1.0, tiler=None, x0_hook=None):
    """Invert `latent_dict` (source latent) along the ASCENDING `sigmas_asc` with
    kreaphoton_invert_loop. Returns a LATENT dict whose samples, fed to run_sampling with
    add_noise=False on the matching DESCENDING schedule, descend back to (approximately)
    the source - the v2 Upscale's fidelity mechanism.

    start_noise: amplitude of the fresh noise comfy mixes in at sigmas_asc[0] (a tiny
    sigma): 1.0 = unit noise (a proper x_sigma state), 0 = none (the loop then inverts a
    slightly darkened clean latent; kept as a tune option)."""
    sigmas_asc = torch.as_tensor(sigmas_asc, dtype=torch.float32)
    latent = comfy.sample.fix_empty_latent_channels(model, latent_dict["samples"])
    if latent.ndim != 5 or latent.shape[1] != 16:
        raise ValueError(f"KreaPhoton: expected a Krea 2 (Wan21) latent of shape (B, 16, 1, H, W), got "
                         f"{tuple(latent.shape)}")
    if negative is None:
        negative = zero_conditioning(positive)
    noise = comfy.sample.prepare_noise(latent, seed) * float(start_noise)
    sampler = comfy.samplers.KSAMPLER(kreaphoton_invert_loop, extra_options={"tiler": tiler, "x0_hook": x0_hook})
    guider = KreaPhotonGuider(model, mode=guidance_mode, cfg=flat_cfg, delta=delta, lo=lo, hi=hi)
    guider.set_conds(positive, negative)
    try:
        import latent_preview
        callback = latent_preview.prepare_callback(model, len(sigmas_asc) - 1)
    except Exception:
        callback = None
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(noise, latent, sampler, sigmas_asc, denoise_mask=None,
                            callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(device=comfy.model_management.intermediate_device(),
                         dtype=comfy.model_management.intermediate_dtype())
    out = latent_dict.copy()
    out["samples"] = samples
    return out


def run_sampling(model, positive, negative, latent_dict, sigmas, *, seed,
                 guidance_mode="off", flat_cfg=1.15, delta=1.25, lo=0.7, hi=0.9,
                 noise=None, add_noise=True,
                 contraction=1.0, per_channel_contraction=False,
                 manifold_std=None, manifold_mean=None,
                 detail_amount=0.0, detail_start=0.15, detail_end=0.95, detail_peak=0.6,
                 order=1, eta0=0.0, sigma_gate=0.10, x0_extrapolation=0.0,
                 clean_model=None, composition_end=0.85,
                 variety_a_latent=0.0, variety_a_cond=0.0, variety_seed=0,
                 variety_end=0.96, variety_cond_taps=(7, 8, 9, 10),
                 guidance_rescale=0.0, pag_scale=0.0, pag_lo=0.72, pag_hi=0.93, pag_blocks="8-15",
                 restart_hook=None, texture_model=None, texture_start=0.65,
                 coherence_jump=0.0, coherence_jump_sigma=0.93, tiler=None, x0_hook=None,
                 ref_hook=None, progress_total=0, solver=None, eta_tail=None,
                 guidance_sigma="model", detail_space="step", detail_sigma=None,
                 x0_extrap_order=1, x0_trust=0.0, pc="off", refine_mode="on", trace=None):
    """Shared entry point for all KreaPhoton sampler nodes.

    Sampler v2 iteration 2 (opt-in, defaults = v1): guidance_sigma (guider), detail_space /
    detail_sigma, x0_extrap_order / x0_trust, pc (strategic steps computed on the FULL
    schedule and handed to every segment), refine_mode, trace (None = the
    KREAPHOTON_TRACE environment variable) - see kreaphoton_sampler_loop.

    solver / eta_tail (Sampler v2, opt-in): forwarded to every segment's loop - see
    kreaphoton_sampler_loop - together with the tail start of the FULL schedule, so a
    phase cut inside the restart tail keeps it. None = the v1 behaviour (solver from
    `order`, eta0 everywhere).

    progress_total (Composition Drafts, 2026-09-22): the step count of the FULL
    schedule this call is a prefix of, so a truncated head integrates the same
    detail-envelope sigma_model sequence as the full run (single-segment path only;
    a phase split computes its own). 0 = this schedule's own length.

    tiler / x0_hook (v1.5 Upscale): forwarded to every segment's loop (see
    kreaphoton_sampler_loop) - the tiled model call and the LF anchor apply to
    the identity and texture phase models alike.

    Variety (both axes) is applied INSIDE the single sampler lifecycle at the
    boundary step variety_boundary_index(sigmas, variety_end): lf_recompose on
    the loop state (latent axis) and a guider-side switch to the tap-rotated
    positive (cond axis). Restart is likewise not a split point - it is encoded
    in the sigmas array and handled per-step by the loop.

    The clean_model split (composition_end) is the one remaining multi-segment
    path: composition phase (sigma > composition_end) on the clean checkpoint,
    LoRA identity/detail phase on `model`. ZPhoton precedent, M3-verified exact
    rescale cancellation at the shared boundary sigma, still unvalidated on
    krea2 LoRA stacks (docs/04 item 7) and therefore still guarded: eta0 is
    forced to 0 whenever the split occurs (S9 speckle/mosaic precedent).
    """
    sigmas = validate_sigmas(sigmas)
    sig_list = [float(s) for s in sigmas]

    # variety consumes guidance budget (presets.GUIDANCE['variety_budget'], V16b)
    delta = effective_delta(delta, guidance_mode, variety_a_latent)

    kw = dict(guidance_mode=guidance_mode, flat_cfg=flat_cfg, delta=delta, lo=lo, hi=hi,
              contraction=contraction, per_channel_contraction=per_channel_contraction,
              manifold_std=manifold_std, manifold_mean=manifold_mean,
              detail_amount=detail_amount, detail_start=detail_start, detail_end=detail_end,
              detail_peak=detail_peak, order=order, eta0=eta0, sigma_gate=sigma_gate,
              x0_extrapolation=x0_extrapolation, guidance_rescale=guidance_rescale,
              pag_scale=pag_scale, pag_lo=pag_lo, pag_hi=pag_hi, pag_blocks=pag_blocks,
              restart_hook=restart_hook, coherence_jump_sigma=coherence_jump_sigma,
              tiler=tiler, x0_hook=x0_hook, ref_hook=ref_hook, solver=solver, eta_tail=eta_tail,
              tail_start=tail_start_index(sig_list) if eta_tail is not None else None,
              guidance_sigma=guidance_sigma, detail_space=detail_space,
              detail_sigma=DETAIL_SIGMA if detail_sigma is None else tuple(detail_sigma),
              x0_extrap_order=int(x0_extrap_order), x0_trust=float(x0_trust), pc=pc,
              pc_steps=pc_step_indices(sig_list, composition_end) if pc != "off" else None,
              composition_end=composition_end, refine_mode=refine_mode,
              trace=(os.environ.get("KREAPHOTON_TRACE", "") == "1") if trace is None else bool(trace))
    # the jump belongs to the first segment only (the step at/below its sigma
    # lives in the composition/identity head of the schedule); later segments
    # must not re-apply it
    kw_jump = float(coherence_jump)

    # --- variety: rotated positive prepared once, boundary index computed once ---
    positive_variety = None
    if variety_a_cond > 0.0:
        positive_variety = _rotate_conditioning(positive, variety_cond_taps, variety_a_cond, variety_seed)
    use_variety = variety_a_latent > 0.0 or positive_variety is not None
    variety_idx = variety_boundary_index(sig_list, variety_end) if use_variety else None
    vkw = dict(positive_variety=positive_variety if variety_idx is not None else None,
               variety_end=variety_end, variety_seed=variety_seed)

    # --- phase models (v1.4): composition on clean_model, identity on model,
    # texture (the restart segment) on texture_model. Multi-segment split; the
    # old S9 "eta0 + split -> speckle" guard is gone - its root cause was the
    # per-segment detail envelope (kreaphoton_sampler_loop progress_offset):
    # every segment now gets its global step offset, so the split integrates
    # the SAME sigma_model sequence as one lifecycle (V16 F_clean validated).
    segments = plan_segments(sig_list, composition_end=composition_end, has_clean=clean_model is not None,
                             texture_start=texture_start, has_texture=texture_model is not None)
    phase_models = {"composition": clean_model, "identity": model, "texture": texture_model}
    if len(segments) > 1 and latent_dict.get("noise_mask") is not None:
        # A noise_mask cannot cross a segment boundary: comfy's KSamplerX0Inpaint re-noises
        # the unmasked band from the SOURCE latent with the ORIGINAL noise at every step
        # and writes the source back into the band at the end - segment 2+ hands it the
        # mid-trajectory state and zero noise instead, so the band decodes to coloured
        # speckle (Face Detailer "confetti ring", 2026-09-15). Run the whole schedule on
        # the model of the phase that covers most of it.
        a, b, phase = max(segments, key=lambda s: (s[1] - s[0], -s[0]))
        print("[KreaPhoton] noise_mask + phase split (%s): a masked run stays one lifecycle - whole schedule "
              "on the %s model" % ("/".join(p for _, _, p in segments), phase))
        model = phase_models[phase]
        segments = [(0, len(sig_list) - 1, phase)]
    if len(segments) == 1:
        return _run_one(phase_models[segments[0][2]], positive, negative, latent_dict, sigmas,
                        seed=seed, noise=noise, add_noise=add_noise,
                        variety_a_latent=variety_a_latent if variety_idx is not None else 0.0,
                        variety_state="pending" if variety_idx is not None else "off",
                        coherence_jump=kw_jump, progress_total=int(progress_total), **vkw, **kw)

    cur = latent_dict
    cur_noise, cur_add = noise, add_noise
    for k, (a, b, phase) in enumerate(segments):
        seg = sigmas[a:b + 1]
        m = phase_models[phase]
        if variety_idx is None:
            state, a_lat = "off", 0.0
        elif a <= variety_idx < b:
            state, a_lat = "pending", variety_a_latent
        elif variety_idx < a:
            state, a_lat = "active", 0.0
        else:
            state, a_lat = "off", 0.0
        # Each segment's KSAMPLER seeds its OWN ancestral/restart RNG from
        # `restart_seed`. Reusing the plain `seed` for every segment would replay
        # the IDENTICAL draw sequence at segment 2's first step as at segment 1's
        # - a "double-exposure" of one frozen noise realization at two points of
        # the trajectory (S9 mosaic precedent). Large per-segment offset
        # decorrelates each segment's stream.
        seg_seed = (int(seed) + k * 1_000_003) & 0xffffffffffffffff
        # jump only in the segment that contains its trigger step
        jump_here = 0.0
        if kw_jump > 0.0:
            trig = variety_boundary_index(sig_list, coherence_jump_sigma)
            if trig is not None and a <= trig < b:
                jump_here = kw_jump
        cur = _run_one(m, positive, negative, cur, seg,
                       seed=seg_seed, noise=cur_noise, add_noise=cur_add,
                       variety_a_latent=a_lat, variety_state=state,
                       progress_offset=a, progress_total=len(sig_list) - 1,
                       coherence_jump=jump_here, **vkw, **kw)
        cur_noise, cur_add = None, False
    return cur
