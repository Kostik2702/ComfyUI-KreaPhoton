# ComfyUI-KreaPhoton

Photorealism-focused sampling nodes for **Krea 2 Turbo**. Analytic restart schedules, detail
boost, seed variety (latent + conditioning), σ-adaptive guidance. Own math throughout (see
`docs/` in the parent research project `D:\Project\Krea2 Nodes\` — design docs/04, plan docs/05,
math proofs docs/03, all research/models/m1-m6 scripts this package ports).

## Nodes (v1 scope)

- **KreaPhoton Sampler** — all-in-one: seed, preset, variety.
- **KreaPhoton Sampler (Advanced)** — SIGMAS input, every parameter explicit, variety axes separate.
- **KreaPhoton Scheduler** — SIGMAS generator (restart = ascending jump; stock `SamplerCustom`
  will NOT understand it — use KreaPhoton samplers).
- **KreaPhoton Empty Latent** — photo aspect ratios/sizes for Krea2 (16ch, 4D — see contract below).

## Interface contract (fixed before parallel module work — planning-council F16/D10/D11)

Modules: `kreaphoton/{presets,schedules,guidance,noise,variety,sampling,nodes}.py`.

### `presets.py`
Single source of truth for every calibrated constant. No other module hardcodes a preset value.
```python
PRESETS: dict[str, dict]      # "turbo/fast" | "turbo/balanced" (default) | "turbo/quality" | "raw/experimental"
                               # keys per preset: n_steps, alpha, restart_frac, sigma_r, plunge,
                               #   detail_a, eta0, sigma_gate, contraction, sampler ("euler"|"euler_2m")
VARIETY_LEVELS: dict[str, tuple[float, float]]   # "off"/"low"/"medium"/"high" -> (a_latent, a_cond)
GUIDANCE: dict                # delta, lo=0.7, hi=0.9, enabled_by_default=False (gated by V1/E1b)
MANIFOLD_STD: list[float]     # 16 floats, from research/results/E3_real.json normalized_per_channel.std
MANIFOLD_MEAN: list[float]    # 16 floats, same source (Advanced-only per-channel bias)
```

### `schedules.py` (S3)
```python
def sigma_from_t(t: float, alpha: float) -> float
def t_from_sigma(sigma: float, alpha: float) -> float
def flux_time_shift(mu: float, t: float) -> float          # cross-check only, vs live calculate_sigmas
def build_schedule(n_steps: int, alpha: float, restart_frac: float,
                    sigma_r: float, plunge: bool) -> torch.Tensor
    # monotonically decreasing SIGMAS except ONE ascending jump at the restart boundary.
    # unit test: features-off (restart_frac=0, plunge=False) must equal live
    # comfy.model_sampling.ModelSamplingFlux(mu=1.15).calculate_sigmas(...) within 1e-6 (not a numpy replica).

@dataclass
class SegmentMap:
    structure_end: int
    plunge_idx: int | None
    restart_start: int | None
    ambiguous: bool                                          # True => graceful degrade (S3b)

def infer_segment_map(sigmas: torch.Tensor) -> SegmentMap    # S3b: handles external/truncated/duplicate-boundary SIGMAS
```

### `guidance.py` (S6, 6th module — planning-council D10)
```python
def g_window(sigma: float, delta: float, lo: float = 0.7, hi: float = 0.9) -> float
    # MUST return EXACTLY 1.0 outside [lo, hi] (smoothstep clamp, not epsilon-form like sigmoid).

class KreaPhotonGuider(comfy.samplers.CFGGuider):
    """Overrides ONLY predict_noise. Reads self.conds (post process_conds), never original_conds.
    At cond_scale == 1.0 (outside window), relies on STOCK comfy behaviour: uncond_ = None,
    zero extra NFE (comfy/samplers.py:609-612) — do NOT set disable_cfg1_optimization.
    RULE: guidance/variety windows are NEVER implemented via sampler_cfg_function /
    sampler_post_cfg_function (model_options hooks) — at cfg=1.0 those hooks receive
    uncond=zeros silently; only a CFGGuider subclass sees the real (skipped-or-not) uncond state.
    After guider.sample(): replicate the comfy.sample tail — .to(intermediate_device(), intermediate_dtype()).
    """
    def __init__(self, model_patcher, delta: float, lo: float = 0.7, hi: float = 0.9): ...
    def predict_noise(self, x, timestep, model_options=None, seed=None): ...

# Unit test (NFE invariant, planning-council D14): count Σ batch-dim / B summed across forwards
# for one step (comfy batches cond+uncond into ONE forward when VRAM allows — samplers.py:260-283,
# so counting *calls* is flaky). Invariant: outside window Σ=1×B, inside window Σ=2×B, regardless
# of batching strategy. Equivalent: len(cond_or_uncond) list per forward, summed.
```

### `noise.py` (S4)
```python
# Sampler loop lives in NORMALIZED (Wan21 model) space (planning-council B1/F9 trace:
# comfy/sample.py prepare_noise draws unit N(0,1) directly in that space, no latent_format
# transform applied to noise, ever). MANIFOLD_STD/MEAN from presets.py apply DIRECTLY, no rescale.
def contract_noise(noise: torch.Tensor, strength: float, per_channel: bool = False) -> torch.Tensor
    # noise: 5D unit N(0,1), shape (B,16,1,H,W). strength: 1.0 = stock, 0.47 = full contraction
    # to the photo manifold (research/results/E3_real.json). per_channel=True (Advanced only)
    # uses MANIFOLD_STD/MEAN vectors instead of the scalar strength.
    # Applied BEFORE noise_scaling (sigma*eps + (1-sigma)*x0), never inside EmptyLatent
    # (a non-zero "empty" latent would pass process_latent_in and get shifted — F11/R12).
```

### `variety.py` (S5)
```python
def lf_recompose(x5: torch.Tensor, seed_v: int, a: float, cutoff: float | None = None) -> torch.Tensor
    # x5: 5D (B,16,1,H,W), NORMALIZED space. squeeze(2) for 2D-FFT band split, unsqueeze(2) back.
    # AC-ONLY correction (planning-council M4, against ZPhoton original): rotate only the
    # AC part of the LF band, exclude per-channel DC (Wan21 channels are NOT zero-mean —
    # energy-matching on the raw norm breaks corr(lf',lf)=sqrt(1-a^2) otherwise).
    # cutoff defaults to 2*N_cyc/G scaling rule (M4), auto from x5 grid size if None.

def cond_tap_rotation(cond: torch.Tensor, taps: tuple[int, ...], a: float, seed: int) -> torch.Tensor
    # cond: packed (B, seq, 30720) tap-major. Unpack via the SAME permute as
    # comfy/text_encoders/krea2.py:63-65 (ground-truth test required — community bug precedent:
    # Krea2T-Enhancer sliced 24x1280 chunks instead of 12x2560 taps, wrong layer boundary).
    # Untouched taps must be bit-identical.

def apply_variety(level: str, latent5d=None, cond=None, seed=None) -> tuple
    # level -> (a_latent, a_cond) from presets.VARIETY_LEVELS; dispatches to lf_recompose /
    # cond_tap_rotation as applicable inputs are given. Advanced node exposes axes separately.
```

### `sampling.py` (S6)
```python
def run_sampling(model, positive, negative, latent_image, seed, preset: dict,
                  variety_level: str = "off", clean_model=None,
                  guider_cls=KreaPhotonGuider) -> dict
    # 1. fix_empty_latent_channels(model, latent_image, ...) + assert latent.ndim == 5 (D7/R12,
    #    mandatory here — Sampler nodes bypass comfy.sample.sample()/common_ksampler entirely,
    #    B4/D11: comfy.samplers.sample() hardcodes stock CFGGuider, so building our own guider
    #    means the whole prepare/guide/sample chain is assembled by hand, mirroring
    #    SamplerCustomAdvanced, not common_ksampler).
    # 2. build_schedule (schedules.py) -> infer_segment_map for restart/plunge boundaries.
    # 3. contract_noise (noise.py) on the initial unit-noise tensor, BEFORE noise_scaling.
    # 4. KSAMPLER-style loop: euler/euler_2m (AB2, fallback on first segment step / |dt|>0.25),
    #    M2 step-relative sigma nudge, M3 restart re-noise (separate RNG), M5 gated-eta ancestral
    #    stochastic component (eta(sigma) = eta0 * smoothstep(sigma; sigma_gate), terminal -> 0).
    #    variety.apply_variety hooked at segment boundaries if variety_level != "off".
    #    Guidance: instantiate guider_cls(model, **preset["guidance"]) INSIDE this function;
    #    guider.sample(...) replaces the raw sampler_function call.
    # 5. .to(intermediate_device(), intermediate_dtype()) tail replicated after guider.sample() (F18a).
    # Returns {"samples": tensor} dict (standard ComfyUI LATENT).
```

### `nodes.py` (S7)
4 classes wrapping the above; relative imports only (`from .schedules import ...`, never absolute
`from schedules import ...` — hyphenated custom_nodes folder import trap). `fix_empty_latent_channels`
called explicitly inside the Sampler nodes (not delegated). Tooltips: negative input cost is
**mode-dependent** — pre-E1b (`GUIDANCE.enabled_by_default=False`): full-trajectory cfg 1.15, 2x
NFE for the WHOLE run; post-E1b: extra calls only inside the Δ-window (~15-25% of steps). A stock
`CFGGuider(cfg>1)` wired externally onto our SAMPLER output re-opens the double-count topology from
outside the pack — documented, with a runtime sanity log if detected.

## Testing (`tests/`)

Ported from `research/models/m1-m6_*.py`. Assert **invariants**, never literal preset values
(planning-council R19): sgm_uniform equivalence at features-off, M3 variance balance on real
manifold stats, monotonic-except-restart, NFE Σ-batch-dim invariant, cond_tap_rotation
bit-exactness on untouched taps, lf_recompose corr=sqrt(1-a^2) + AC-only DC preservation.
No `pytest` in the ComfyUI embedded interpreter — plain-assert runner (`tests/run_tests.py`),
matching `research/models/*.py` style. `--import-mode=importlib` if run under a dev-machine pytest.

## Status

Implementation per `D:\Project\Krea2 Nodes\docs\05-plan-kreaphoton-v1.md` (planning council,
2026-07-05, full convergence). Progress tracked in vault `learnings/active/KREA2-NODES.md`.
