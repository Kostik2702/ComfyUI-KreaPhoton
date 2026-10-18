# KreaPhoton UI P2 — routes, σ strip, readout, lint chips

> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** node-ui-concept §9 P2: the routes `/kreaphoton/ui` and `/kreaphoton/schedule`; a σ strip on
Sampler, Scheduler and LoRA Phase (compact); the Empty Latent readout; Encode and Scheduler lint chips.
**Done when** the strip's call count equals the `calls=` the sampler logs, for every preset.

**Spec:** `docs/design/node-ui-concept.md` on `origin/docs/node-ui-concept` (§2–§7, §9) + vault
`learnings/active/KREA2-NODES.md`, entries of 2026-10-18 (UI P2 start, Investigation).

## Global constraints

- One source for the maths: JS never computes a schedule, a phase or a resolution. Everything comes
  from the two routes, which call the same pure functions the nodes run.
- The sampler's own σ construction moves out of `KreaPhotonSampler.sample` into a pure function;
  the node calls it, and a test pins it bit-exact against the old inline code for every preset ×
  coherence × denoise × size tier.
- Widget order is frozen (`tests/test_ui_contract.py`): no Python input changes in P2. Every DOM
  widget gets `serialize = false` (frontend 1.53.6 skips those in `widgets_values`,
  `LGraphNode.ts:175,1197`); a save → load round trip must give identical `widgets_values`.
- Two renderers: information lives in DOM widgets only; no `onDrawForeground`.
- No new dependencies, no network beyond the local routes, no fonts. Colours = concept §3 tokens.
- Tests: plain-assert scripts, embedded python
  `"E:/CUI portable/ComfyUI-torch2.9-cu130-cp313-v1.2/python_embeded/python.exe" tests/run_tests.py`.

## Assumed defaults (owner did not decide; concept leaves them open)

- ASSUMED: LoRA Phase compact strip uses the preset of the first KreaPhoton sampler downstream on its
  MODEL output; none found → `turbo/quality`; the caption names the source.
- ASSUMED: unknown latent size (no Empty Latent upstream) → L 3:2, said in the caption (concept §7.3).
- ASSUMED: `precision_profile` auto → the route uses `composition_end` shared by all profiles; if
  profiles ever disagree, the caption says "auto: int8 boundary shown".
- ASSUMED: quality-plus self-refine cells are drawn hatched with the caption "+4 self-refine if the
  draft still moves" — the loop may skip them (`refine_mode=auto`); predictor-corrector steps get a
  small tick ("+N corrector calls"), the log counts them separately (`pc_calls`).
- ASSUMED: the strip sits after `variety` on the Sampler if the frontend allows reordering a
  non-serialized widget without side effects; otherwise at the end of the widgets.

## Tasks

### T1 — pure schedule description (`kreaphoton/ui_data.py`, TDD)

- [ ] `sampler_sigmas(preset, h_lat, w_lat, coherence="off", denoise=1.0)` → `(sigmas, refine_mode,
      refine_steps)`, lifted verbatim from `KreaPhotonSampler.sample`; the node calls it.
- [ ] `scheduler_sigmas(steps, alpha, restart_frac, sigma_r, plunge, h_lat=None, w_lat=None, ...)`,
      lifted from `KreaPhotonScheduler.build`; the node calls it.
- [ ] `describe_schedule(sigmas, composition_end, refine_auto=False, pc="off")` → per model call
      `{i, sigma, next, phase, conditional, pc}` with phase ∈ composition / identity / draft / texture
      / refine, built from `plan_segments`, `tail_start_index`, `pc_step_indices`; counts; restart band
      σ range; caption in words ("16 calls, compose 8, identity 3, draft 1, texture 4").
- [ ] `ui_static()` → resolution buckets + aspects, preset summaries, LoRA phase → segments + σ ranges,
      media nouns, coherence options, boundaries (composition_end, texture_start, TAIL_SIGMA_MAX).
- [ ] `MEDIA_NOUNS` constant in `presets.py` (magazine, cover, snapshot, poster, photo, print — concept §5).
- [ ] Tests `tests/test_ui_data.py`: bit-exact vs old inline construction (all presets × coherence ×
      denoise {1, 0.6} × S/M/L/XL × aspects); `len(calls) == count_model_calls(sigmas)` everywhere;
      turbo/quality = 8/3/1/4; quality-plus refine cells flagged conditional; LoRA phase ranges
      match `PHASE_SEGMENTS`. Planted error: a phase swap in the classifier turns a test red.

### T2 — routes (`kreaphoton/server_routes.py`)

- [ ] `schedule_payload(query: dict)` pure (testable without aiohttp): `node=sampler` (preset,
      coherence, denoise, w, h, profile) or `node=scheduler` (its widgets, optional w, h); bad input →
      `ValueError` → HTTP 400 with the message.
- [ ] `GET /kreaphoton/ui` → `ui_static()`; `GET /kreaphoton/schedule` → `schedule_payload`.
- [ ] Tests in `test_ui_data.py` for the payload (defaults, errors, scheduler == node output).

### T3 — σ strip (`web/kreaphoton_strip.js`)

- [ ] DOM widget (inline SVG: σ curve, restart band, one cell per call coloured by phase, caption with
      `tabular-nums`), `serialize = false`, theme tokens for dark / light.
- [ ] Sampler: preset, coherence, denoise + latent size from an upstream Empty Latent; Scheduler: its
      own widgets; LoRA Phase: compact cells with the active segments lit.
- [ ] Re-fetch on widget change (debounced), on link change, on theme change; cache by query string.
- [ ] Setting `KreaPhoton.SigmaStrip` (on / off).

### T4 — readout and lint chips (`web/kreaphoton_strip.js` or `web/kreaphoton_hints.js`)

- [ ] Empty Latent readout: "1088 × 1600, 1.74 MP, portrait 3:2" from the route's buckets.
- [ ] Encode chip: a media noun in `text` or `custom_directive` → "'magazine' may render as print or text".
- [ ] Scheduler chips: `refine_steps` > 0 without `plunge` → "Self-refine needs plunge"; SIGMAS linked to
      a non-KreaPhoton node while `restart_frac` > 0 → "Feeds <type>: restart will be ignored".

### T5 — verify

- [ ] Full test suite green; planted-error proofs recorded.
- [ ] Live, every preset: run the Sampler through the API, read `calls=` from `/internal/logs`, compare
      with the route's count (quality-plus: `calls` = total or total − refine when skipped).
- [ ] Playwright on frontend 1.53.6: strip / readout / chips visible, dark + light screenshots, save →
      load `widgets_values` identical, Vue node renderer on: the DOM widgets render.

### T6 — docs

- [ ] REFERENCE.md: UI section (strip legend, chips, settings, routes); README one line;
      `ui_meta.py` docstring corrected (library order = registration order, not alphabetical).
