# KreaPhoton UI P3 — Advanced sections, changed dots, relevance dimming, "Start from preset…"

> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** node-ui-concept §9 P3: sections on Sampler (Advanced), changed dots, relevance dimming on
both samplers and the Scheduler, the "Start from preset…" button (fills the connected Scheduler too —
owner decision Q2, 2026-10-18), "More inputs" on the simple Sampler.
**Done when** Advanced is ≤ 560 px with its sections collapsed and every relevance rule has a test.

**Spec:** `docs/design/node-ui-concept.md` (§4–§7, §9, §12 Q2) on `origin/docs/node-ui-concept`.

## Frontend facts (spike on 1.53.6, 2026-10-18)

- `widget.hidden = true`: skipped by layout and drawing in both renderers; still in `widgets_values`
  and in the prompt.
- `widget.disabled = true`: the canvas draws the widget WITHOUT its value and ignores clicks
  (`getWidgetOnPos` skips `computedDisabled`) — unusable for "dimmed, still discoverable".
  Wrapping the widget's own `drawWidget` with `globalAlpha` dims it and keeps value + input.
  The Vue renderer does not call `drawWidget`: its row is `[data-testid="node-widget"]`, the label
  `[data-testid="widget-layout-field-label"]`; an inline opacity on the row survives re-renders.
- `widget.label` is display-only: shown by both renderers, not written to `inputs[].label`.

## Global constraints

- Widget order frozen; no Python input changes. Section headers, the preset button and notes are
  DOM widgets with `serialize = false`; moving those in `node.widgets` does not touch
  `widgets_values`. Save → load must give identical `widgets_values`.
- One source: sections, relevance rules, experimental list and preset → widget values live in
  Python (`kreaphoton/ui_data.py`) and are served by the routes; JS only evaluates the rule
  conditions (a declarative `{widget, in|gt|gte|is}` / `all` / `any` / `context` tree).
- A rule is in the table only when a test proves "no effect" on the code the node runs (varying the
  widget under the false condition changes nothing) and that the test is not blind (under the true
  condition it changes something).

## Assumed defaults (owner did not decide; concept leaves them open)

- ASSUMED A1: dimmed = 42 % opacity, still editable (native `disabled` hides the value, see above).
- ASSUMED A2: sections are an accordion, one open at a time. The frozen order scatters a section's
  widgets (Solver: `sampler_order` is widget 2, `x0_extrapolation` 24, `x0_extrap_order` / `x0_trust`
  40–41, `pc` 42), so with two sections open their widgets would interleave; with one open, every
  visible widget belongs to it and sits under its header (headers are moved around it).
- ASSUMED A3: `precision_profile` (added after the concept) goes into "Phase models" on Advanced and
  into "More inputs" on the simple Sampler.
- ASSUMED A4: "Start from preset…" sets the widgets a preset controls (solver, noise, detail,
  guidance, x0, pc, refine_mode, composition_end, PAG off, coherence jump off) and leaves the
  user's own run choices alone: `seed`, `preview_method`, variety axes, `seed_b`, `blend`,
  `restart_enhance`, `precision_profile`, `texture_start`.
- ASSUMED A5: the Scheduler gets all nine widgets from the preset; its alpha is resolution-aware only
  with its `latent` input connected, so the note says so when it is not.
- ASSUMED A6: the changed-dot reference is the INPUT_TYPES default, or the values "Start from
  preset…" wrote, kept in `node.properties.kp_reference` (properties are not `widgets_values`).
- ASSUMED A7: setting "Sections on Sampler (Advanced)" off → the flat list, no headers; dots and
  dimming stay.
- ASSUMED A8: the open section is kept in `node.properties.kp_open_section`; a new node opens none.
- ASSUMED A9: experimental widgets get the label suffix " · exp".
- ASSUMED A10: rules beyond the concept table, each proven by a test: `detail_*` and `detail_space`
  need `detail_amount` ≠ 0; `guidance_rescale` needs a guidance mode; `texture_start` also counts a
  `composition+identity` LoRA (it splits the texture set off, `lora_phase.phase_sets`); Scheduler:
  `sigma_r` needs `restart_frac` > 0 and ≥ 4 steps, `refine_steps` / `plunge_sigma` / `soft_plunge`
  need `plunge`, `refine_sigma` needs `plunge` and `refine_steps` > 0.

## Tasks

### T1 — data (`kreaphoton/ui_data.py`, TDD)

- [x] `SECTIONS`, `EXPERIMENTAL`, `RELEVANCE`, `condition_holds`, `dimmed(node, values, context)`.
- [x] `preset_widget_values(preset, profile, has_negative)` → `{sampler, scheduler, profile, note}`.
- [x] `ui_static()` adds sections / relevance / experimental; `/kreaphoton/ui` adds widget defaults
      from INPUT_TYPES; new route `GET /kreaphoton/preset_values`.

### T2 — tests (`tests/test_ui_relevance.py`)

- [x] Per rule: false condition → identical behaviour for two values of each dimmed widget; true
      condition → different (guider `predict_noise` with a captured `sampling_function`, the sampler
      loop on a mock model, `plan_segments`, `_blend_noise`, `scheduler_sigmas`, Sampler.sample).
- [x] Sections cover every Advanced widget except `seed` exactly once; every widget named exists.
- [x] Equivalence: for every preset × negative × profile, Advanced with `preset_widget_values` and the
      Scheduler with its values hand `run_sampling` the same arguments as the simple Sampler.
- [x] Every filled value is a legal widget value (combo option / in range).
- [x] Planted errors: a wrong rule turns red; a wrong preset value turns red.

### T3 — JS (`web/kreaphoton_sections.js`)

- [x] Section headers (accordion, summary right), "More inputs" on Sampler, "Start from preset…" with
      Scheduler fill and note, changed dots via label, dimming (canvas `drawWidget` wrapper + Vue row
      opacity) with the reason line, setting.

### T4 — verify

- [x] Suite green; live: Advanced collapsed ≤ 560 px; save → load identical `widgets_values` (test
      graph + owner's SH3 workflow, read only); dark / light / Vue screenshots; a "Start from preset"
      Advanced + Scheduler run equals the simple Sampler pixel for pixel.

### T5 — docs

- [x] REFERENCE.md "Node UI" section; vault.
