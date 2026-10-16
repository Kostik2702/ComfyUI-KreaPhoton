# Krea 2 Turbo landscape and experiment backlog

**Scan date:** 2026-10-15  
**Scope:** community reports, technical sources, and research that could inform KreaPhoton
sampler, conditioning, diversity, and artifact-control work.  
**Status:** desk research; the ideas below are not validated KreaPhoton recommendations.

## Executive read

The highest-value *directly Krea2-tested* lead is the Turbo de-distillation-strength
axis: a public, small-scale Krea2 exploration reports useful CFG/negative-prompt headroom
when partially cancelling distillation, while finding the quality response and diversity
response depend on prompt class. This is a community pilot, not a broad benchmark. The
previously highlighted RAW-for-first-step intervention remains scientifically motivated
by another distilled-model paper, but a Krea2-specific pilot reports a null result for that
exact intervention; keep it low priority until a replicated test says otherwise.

A second concrete lead is spatial conditioning. A public Krea-2 Control-LoRA trainer
supports Canny, depth, and custom pixel-aligned maps, but its documented recipe trains
against Krea 2 RAW. That makes object outlines actionable as a research direction, not yet
a verified Turbo feature. Turbo also has its own de-distill adapter for training ordinary
LoRAs directly on Turbo while retaining fast sampling; this is distinct from a spatial
Control-LoRA and should not be conflated with it.

Community evidence points to four user pains worth measuring: seed outputs that feel too
similar; blotches/noise/banding on dark smooth areas; variation in face/skin/anatomy across
checkpoints and sampler settings; and uncertainty about which sampler/scheduler choice is
actually better. Community claims conflict, and the larger sampler benchmark still uses
a narrow set of scenes. This argues for checkpoint-aware, task-stratified comparison
tools and diagnostics rather than a single “best sampler” default.

The second scan strengthens one practical conclusion: checkpoint precision is a quality
axis, not just a VRAM/performance choice. A community benchmark with 15 prompt strata and
two matched seeds reports measurable output drift across BF16, FP8, INT8 ConvRot, MXFP8,
and NVFP4. Its LPIPS/DISTS/DINO endpoints quantify similarity to BF16, not which image is
more photorealistic, so this is strong evidence of *difference* and useful reproducibility
data, not a human quality ranking.

KreaPhoton already has substantial sampler, restart, stochasticity, guidance, diversity,
phase-model, composition-preview, and refinement controls. The opportunity is to close
evidence gaps and make the next useful experiments easy to run, especially early-step
RAW→Turbo handoff, smooth-region artifact reduction, and diversity measurement. The repo
reference itself notes that current evidence does not cover enough faces, groups, hands,
text, night/interior scenes, LoRA stacks, or quantized checkpoints.

## What users are reporting

| Pain or technique | Source observation | Evidence quality and limits |
|---|---|---|
| Similar images across seeds | A July Reddit thread attributes it to Turbo behavior; an October thread separates prompt-expansion seed (large semantic/layout changes) from sampler seed (smaller changes). Replies also suggest RAW, RAW+Turbo, and an SDA LoRA. | **Reported.** Anecdotal; the prompt-expansion workflow changes conditioning as well as sampling, so it is not a clean sampler comparison. The Oct thread includes disagreement about how unusually low Turbo's variance is. |
| Blotches/noise on dark or smooth regions | An August thread reports blotches/banding on smooth surfaces, dark backgrounds, and dim skin. Replies report improvements from simpler scene wording, higher resolution, altered denoise, RAW+Turbo two-stage sampling, or post-processing. One commenter reproduced the blotchy prompt in Krea's hosted interface, which argues against a local ComfyUI-only failure for that example. | **Reported.** Multiple likely causes are mixed: prompt semantics, checkpoint, resolution, sampler, VAE, quantization, and image post-processing. No controlled shared benchmark. |
| Sampler/scheduler uncertainty | A Reddit author reports a 396-combination Turbo comparison and nominates different winners by resolution/task; separate threads recommend `er_sde`, `euler + beta`, `heunpp2 + simple`, or other combinations. Another thread says differences often look like taste rather than quality. | **Reported benchmark + disagreement.** The 396-combination post describes few stages/scenes and visual selection; rankings are not a general result. Good source for candidate settings, not defaults. |
| Raw schedule artifacts | An August RAW post attributes washed dark detail to early sigma reduction and brittle noise to lower-sigma tail curves, especially around steps 30–40. | **Reported hypothesis.** Specific and experimentally useful, but RAW at 52 steps/CFG 3.5; do not apply directly to Turbo. |
| Model/checkpoint-specific failures | A thread reports artifacts differ among several Krea2 checkpoints even with the same settings. | **Reported.** Strong reason to include checkpoint/precision as explicit factors in any calibration. |
| Skin texture can overshoot | Users report freckles, moles, blotchiness, and “aged” skin; replies blame non-Euler samplers, conditioning rebalance, bypass LoRAs, quantization, or prompt/LoRA setup. Other users report those same settings work well. | **Conflicting reports.** Excellent targeted test case; causes are confounded and comments disagree. Track pore detail separately from blemish/noise density. |
| LoRAs may flatten subtle realism | A Krea2 realism/slider author reports dozens of dataset/caption/parameter experiments often made skin and fine detail flatter/airbrushed; prompting surfaced the base model's ability first, and only then was that direction transferred into a LoRA. Another realism merge's creator reports a short natural-language paragraph improved expression/texture, alongside a disclosed ethnicity bias in the training data. | **Creator self-report.** Supports testing prompts before stacking realism LoRAs and tracking demographic fidelity; not independent comparative evidence. |
| Quantization affects image trajectory | A community benchmark compares five official weight formats under matched noise/settings, saving all intermediate states; BF16 is reference and INT8 ConvRot is closest among quantized variants on reported similarity metrics. Commenters report hardware-dependent speed reversals and LoRA interactions. | **Strong community measurement for output drift**, not an objective photorealism winner. Quality metrics are similarity-to-BF16; speed is hardware/runtime dependent. The dataset and workflows are public. |
| Detail pass can help or overcook | Users report RAW plus Turbo LoRA, a low-denoise detail pass with changed prompt/settings, PID/SeedVR2 upscaling, or a separate Flux2 Klein reference-conditioned refine. Other users recommend staying single-pass because extra stages make attribution harder and are unnecessary for their results. | **Reported, conflicting.** Treat the full chain as a family of isolated hypotheses; don't compare bundled stacks to baseline and credit one component. |
| Diversity LoRA / first-call interventions | October replies point to an SDA LoRA applied during the first two steps and to raising the start sigma / changing the starting condition. A Krea2-specific public pilot reports its base-first-step test was null. | **Conflicting, mostly reported.** The mechanism has independent support on other distilled models; the Krea2 pilot lowers priority and needs replication only if implementation/checkpoint conditions differ. |

### Community source log

- [Krea 2 Turbo native sampler/scheduler benchmark (396 combinations)](https://www.reddit.com/r/StableDiffusion/comments/1uxdifv/krea_2_turbo_native_samplerscheduler_benchmark/) — community-run visual benchmark; useful candidate combinations and protocol inspiration, but narrow prompt coverage and author-selected rankings.
- [Krea 2 best sampler for realism](https://www.reddit.com/r/comfyui/comments/1vepoul/krea_2_best_sampler_for_realism/) — comments recommend different pairings for photorealism, texture, and identity retention; useful as test candidates, not as settled consensus.
- [Krea 2 and the “perfect sampler”](https://www.reddit.com/r/comfyui/comments/1wln9wo/krea_2_and_the_perfect_sampler/) — explicitly surfaces checkpoint-dependent artifacts and uncertainty about quality vs preference.
- [Noisy outputs in Krea 2 in ComfyUI](https://www.reddit.com/r/comfyui/comments/1vldqo0/noisy_outputs_in_krea_2_in_comfyui/) — long troubleshooting thread with smooth/dark surface artifacts, prompt wording, resolution, RAW/Turbo blend, and post-processing suggestions.
- [Krea 2 variance is easy in ComfyUI](https://www.reddit.com/r/StableDiffusion/comments/1wwlo18/psa_krea_2_variance_is_easy_in_comfyui_no_loras/) — prompt-expansion seed versus generation seed; comments discuss SDA LoRA, RAW/Turbo, and remaining sameness.
- [Different seeds producing nearly identical images?](https://www.reddit.com/r/StableDiffusion/comments/1urmce7/krea_2_default_comfyui_t2i_workflow_different/) — early user reports and suggestions including conditioning noise and Seed Variance Enhancer.
- [RAW Krea2 custom sigma curves](https://www.reddit.com/r/StableDiffusion/comments/1vwxx31/krea_2_raw_in_comfyui_sharper_more_detailed/) — hypothesis about early sigma descent and late-tail artifacts, restricted to RAW.
- [Krea 2 skin texture, excessive texture, moles, and freckles](https://www.reddit.com/r/StableDiffusion/comments/1uhxi2p/krea_2_skin_texture_excessive_texture_moles_and/) — comments disagree on whether sampler, rebalance, bypass LoRA, VAE, checkpoint, or prompt is responsible; valuable causal test template (hold seed, change prompt/one node, inspect whether defect coordinates follow noise).
- [Krea 2 RAW vs Turbo: speed sacrifice vs micro-texture](https://www.reddit.com/r/comfyui/comments/1ug4dqd/krea2_raw_vs_turbo_is_the_speed_sacrifice_worth/) — RAW, Turbo LoRA, precision, and performance anecdotes; contains contradictory outcomes and warns against single-thread quantization claims.
- [150 matched-image checkpoint format benchmark](https://www.reddit.com/r/StableDiffusion/comments/1uuu6yk/i_benchmarked_every_krea_2_turbo_checkpoint/) — 15 prompt types × 2 seeds × 5 formats at fixed 1024² / 8 steps / Euler-simple, same initial noise, common BF16 text encoder and VAE; intermediate states, weights, workflows, and images linked by author. Metrics are distance from BF16, not realism scores.
- [Excessive skin texture thread](https://www.reddit.com/r/StableDiffusion/comments/1uhxi2p/krea_2_skin_texture_excessive_texture_moles_and/) — sampler, conditioning, and VAE hypotheses in comments; includes a comment explicitly retracting its initial VAE diagnosis.
- [Krea2 smartphone-photography slider research notes](https://www.reddit.com/r/comfyui/comments/1v2ycn5/elusarcas-krea-2-smartphone-photography-slider/) — LoRA creator reports prompt tests preceded slider training; early training attempts flattened/airbrushed details. Author flags observations as personal, not objective.
- [Krea2-realism V2 release and discussion](https://www.reddit.com/r/StableDiffusion/comments/1ulonm8/krea2realismv2_is_finally_here_things_got_a/) — natural-language prompt advice and an explicit demographic-bias discussion by the LoRA author.
- [Krea2 sharpening recipe: staged RAW + Turbo LoRA + detail pass](https://www.reddit.com/r/comfyui/comments/1upaq3v/help_getting_sharper_images_krea2/) — unusually specific user recipe; several factors change at once, so decompose it before replication.
- [Krea2 Turbo + Flux2 Klein refinement](https://www.reddit.com/r/comfyui/comments/1w7bop3/ultrarealistic_photos_krea_2_turbo_base_flux_2/) — reports cross-model reference-conditioned refinement; useful comparator for the boundary between sampler improvements and a different model's redraw prior.
- [Krea2 photographic workflow discussion](https://www.reddit.com/r/comfyui/comments/1vyz7sr/looking-for-a-good-krea-2-turbo-workflow/) — replies argue model/LoRA/sampling matter more than graph complexity; recommends native single-pass as a clean baseline. Also includes divergent prompt-adherence reports.

## Scientific and primary technical sources

### Directly relevant research

**Distilling Diversity and Control in Diffusion Models** (WACV 2026; [paper](https://arxiv.org/abs/2503.10637), [project page](https://distillation.baulab.info/)). The authors analyze reduced sample diversity in distilled models. They report that distilled models retain useful variation directions but fail to activate them; intermediate predictions suggest early steps commit to structure, and a hybrid inference method uses the base model for the critical first timestep then switches to the distilled model. This is the strongest scientific lead for a Krea2 RAW→Turbo sampler mode. Transfer is plausible because Krea2 provides a base RAW and distilled Turbo pair, but it is **not yet evidence that the same handoff works on Krea2**.

**DPM-Solver++** ([paper](https://arxiv.org/abs/2211.01095), [official code](https://github.com/LuChengTHU/dpm-solver)). This work develops high-order ODE solvers for diffusion models and discusses instability under strong guidance. It supports carefully implemented multistep/variable-step solver experiments, but the equations and reported results target diffusion parameterizations and guided sampling; do not assume a gain on Krea2's rectified-flow Turbo trajectory. The pack already exposes RF-aware multistep sampling, so future work should focus on a controlled head-to-head, not adding solver names.

**Perturbed-Attention Guidance** ([paper/project](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/), [official code](https://github.com/sunovivid/Perturbed-Attention-Guidance)). Training-free guidance through perturbed attention is relevant where CFG is unavailable or unsuitable. KreaPhoton already has an experimental PAG control; its local reference reports sparkle/saturation/face distortion at higher scales and no measured anatomy benefit at benign scales. The research implication is a narrow, block- and sigma-window ablation with artifact-specific ratings, or deprioritization if it remains neutral.

**Stochastic Sampling from Deterministic Flow Models** ([paper](https://arxiv.org/abs/2410.02217)). Derives a family of same-marginal SDE samplers for deterministic rectified-flow models, trading determinism for stochasticity; the proposed construction uses both the flow field and score. This supports the general idea that stochasticity should be scheduled, not toggled blindly, but Krea2 exposes a velocity field rather than a ready-to-use score. **Direct port is not plug-and-play**; extracting score information or using a justified approximation is a prerequisite, and KreaPhoton's existing gated ancestral noise is the simpler baseline.

**Precise: SDE-Consistent Stochastic Sampling for Flow-Matching RL** ([preprint](https://arxiv.org/abs/2605.23522)). Studies exploration/stability and discretization consistency for low-step stochastic flow-matching samplers; its method freezes the clean-latent posterior mean to avoid excess injected noise. The goal is RL post-training, not inference-time photorealism. Potential KreaPhoton lesson: test whether x0-anchored stochastic injection at the restart boundary preserves structure better than heuristic re-noising. Treat this as a derived sampler hypothesis, not a recommended formula until the paper equations, parameterization, and Krea2 assumptions are checked.

**Reward-Agnostic Test-Time Prompt Optimization (RATTPO)** ([paper](https://arxiv.org/abs/2506.16853), [code](https://github.com/seminkim/RATTPO)). Uses iterative LLM prompt revisions plus reward feedback to search for prompts without fixing one reward model. The paper reports gains on aesthetics, human preference, and spatial relations in its tested T2I systems. For KreaPhoton this is an optional workflow-level direction: create a small prompt optimizer that preserves user-specified content while searching for photographic qualities. Main risk is proxy overoptimization: a reward model can reward generic polish while reducing the user's intended scene, natural variation, or demographic fidelity. Keep prompt generation separate from sampler attribution and gate it with human review.

**Human Preference Score v2 (HPS v2)** ([paper](https://arxiv.org/abs/2306.09341)). A learned predictor trained to align image-text preferences with human judgments. It can help rank prompt-conditioned candidates, but it is not a photorealism detector and may inherit dataset/style biases. A proper quality score should combine it with blind human ratings for natural skin, lighting/material plausibility, anatomy, texture artifacts, and prompt fidelity.

**Inference-time scaling / reflection.** [Best-of-N noise search](https://arxiv.org/abs/2506.12633) reports performance plateaus after relatively few optimization steps when searching initial noise without an external evaluator. [Reflect-DiT](https://arxiv.org/abs/2503.12271) improves composition/alignment on SANA by training a diffusion transformer to use prior image examples and textual critique in context. Best-of-N is immediately portable as a workflow around Krea2, while Reflect-DiT's architecture-level method is not a drop-in sampler trick. For this project, an N-candidate selector with artifact-aware labels is the credible route; test whether a small candidate budget improves human-rated realism enough to justify its cost.

**Annealed Importance Guidance (AIG)** ([paper](https://arxiv.org/abs/2409.06493)). Studies the reward/diversity trade-off and proposes inference-time regularization for Stable Diffusion to reduce reward hacking and retain diversity. It is a useful caution and a possible sampler theory lead, but uses access/assumptions not established for Krea2. Avoid adding proxy guidance until its score/likelihood terms are derived for this flow model.

### Krea2-specific primary sources

- [Krea 2 official inference repository](https://github.com/krea-ai/krea-2): documents RAW as the undistilled base and Turbo as an 8-step distilled model; official reference settings are RAW 52 steps/CFG 3.5 and Turbo 8 steps/CFG off/μ 1.15. This anchors model-specific conditions for experiments.
- [Krea 2 technical report](https://www.krea.ai/blog/krea-2-technical-report): describes multi-reward post-training including aesthetic, prompt-following, text-rendering, and artifact/structure reward models. This supports treating prompt adherence, text, and structural artifacts as distinct outcomes.
- [ComfyUI-Sigmax](https://github.com/rookiestar28/ComfyUI-Sigmax): a contemporary community project with explicit model-aware Krea2 RAW/Turbo schedules. Useful for compatibility ideas and duplicate-feature tracking; claims and settings require independent validation.
- [ComfyUI-AlchemicaSampler](https://github.com/Nessuno76/ComfyUI-AlchemicaSampler): reports same-seed, one-factor measurements for NAG, two-phase prompts, composition previews, and a frequency-grid correction. Its README also cautions that its displayed comparison is three prompts and not a benchmark. Treat the grid artifact and negative-at-CFG1 techniques as concrete replication candidates.

## Synthesis: opportunities for KreaPhoton

### Priority 1 — Turbo strength / CFG calibration (high value, medium confidence)

Start with the direct Krea2 de-distillation-strength pilot and replicate it across prompts,
seeds, and the exact installed checkpoint. Hold the official 8-step CFG-off baseline first,
then test whether partial de-distillation gives useful CFG/negative-prompt control without
damaging skin, material response, or prompt adherence. The published strength values are
candidates only; the evidence is narrow.

### Lower priority — RAW first call → Turbo remainder (scientifically motivated, Krea2 pilot null)

The cross-model diversity paper provides a mechanism, but a Krea2-specific exploration
reports a null for this exact intervention. Defer it unless our pipeline differs in a
relevant way; retain a small replication as a check, not the lead experiment.

### Priority 2 — Smooth-region / dark-surface artifact suite (high value, medium confidence)

Build a prompt stratum for dark hair, black fabric, dim skin, flat walls, gradients, reflective surfaces, and low-light scenes. Include the community's repeated complaint of blotches/banding/noisy texture. Sweep a small set of schedule/tail/stochasticity variants and resolutions; separately test prompt phrasing and checkpoint/precision. Use blind comparisons plus targeted diagnostics (banding/grid-frequency energy, local chroma variance, dark-region texture) rather than treating sharpness or SSIM as quality.

Candidate replication: AlchemicaSampler's DeGrid claim, the RAW early/tail sigma observations (on RAW only), and the reported higher-resolution and RAW→Turbo effects. These are independent tracks; do not combine them in the first experiment.

### Priority 3 — Seed diversity as three separate controls (high value, medium-high confidence)

Report separately: (a) semantic diversity (subject/action/style), (b) composition/layout diversity, and (c) local texture/detail diversity. Compare sampler seed changes, prompt-expansion seed changes, initial-noise/variance methods, lower Turbo strength on constrained prompts, first-call RAW handoff only as a lower-priority replication, and the cited SDA LoRA. This prevents a prompt rewrite from being incorrectly credited as sampler variance. Preserve the user's prompt and expose prompt expansion as an optional separate control.

The paper gives a causal mechanism worth testing: compare decoded x0 predictions after each early model call across seeds and across RAW/Turbo. If Turbo's predictions converge earlier, that supports a first-call intervention; if not, look elsewhere.

### Priority 4 — Checkpoint-aware calibration and a short, honest recommendation matrix (high value, high confidence)

The community's “best sampler” debate likely reflects checkpoint, resolution, prompt, and taste differences. Build a modest factorial calibration matrix before broad sweeps: official Turbo plus a few representative community merges; 1 MP and native 2 MP; portrait, still life, low-light, text, hands, and group scenes; a compact finalist set of samplers/schedules. Use held-out prompts and publish Pareto views for fidelity, artifacts, diversity, speed, and VRAM rather than one winner.

### Priority 5 — User-facing diagnostics and experiment capture (medium value, high confidence)

Make each comparison reproducible from workflow metadata: model/checkpoint hash, precision, VAE, ComfyUI version, prompt, seed, resolution, sampler, schedule/sigmas, steps, guidance, LoRAs, and runtime. A “compare these variants” or sweep-planner workflow may relieve the recurring confusion without adding another opaque sampler preset.

### Priority 6 — Quality-aware model precision selection (medium-high value, high confidence)

Expose that formats can change the generated image, then calibrate the quality/speed/VRAM tradeoff on actual target GPUs. The 150-image community study suggests INT8 ConvRot stayed closer to BF16 than FP8/MXFP8/NVFP4 on its reported perceptual/semantic distances, but hardware comments show runtime rankings reverse across GPU, ComfyUI/PyTorch build, and LoRA load. For a photorealism-first mode, use BF16 as the reference; avoid equating “closer to BF16” with “more realistic.”

### Priority 7 — Separate skin realism from “more detail” (high value, high confidence)

Create paired ratings for (1) natural skin microtexture, (2) false blemish/freckle density, (3) plastic smoothing, (4) face age/expression, and (5) global sharpness. Sweep sampler stochasticity, detail envelope, conditioning perturbation/rebalance, precision, and realism LoRA independently. The user reports show that high-frequency enhancement can be interpreted as pores by one person and ugly freckles/noise by another; a generic detail score will hide this failure.

### Priority 8 — Prompt-to-realism assist with preservation constraints (medium value, medium confidence)

Test an optional prompt assistant that translates tag piles into concise natural language and adds only photographic capture cues supported by the user's request (lens/framing/light/material/skin). Community authors repeatedly report paragraph-style prompting and describe photography as concrete capture conditions rather than a “photorealistic” style token. RATTPO suggests a reward-feedback search option, but use a hard content-preservation check and human rating so the assistant does not drift identity, ethnicity, composition, or story.

### Priority 9 — Quality shortlist / small-budget best-of-N (medium-high value, high confidence)

Use a small candidate set to expose genuinely better draws instead of forcing every single draw to look more polished. Keep prompt, checkpoint, and settings fixed; vary seeds, show a contact sheet, then rank by human preference or a conservative rubric for photorealism. Existing Composition Drafts finds layout candidates early; an optional final-image quality shortlist would target skin, anatomy, lighting, material response, and visible artifacts. Automated HPS/VLM ranking can order candidates but must not be presented as ground truth. Track quality gain per extra model call and expect diminishing returns; a recent study reports a plateau in initial-noise search.

### Priority 10 — Treat photographic realism as coherent capture, not maximum micro-detail (high value, medium-high confidence)

Several community experiments converge on restraint: users report that aggressive high-frequency samplers/detail LoRAs turn skin into freckles or noise, while realistic results require coherent light, natural skin, plausible camera/framing, and non-overprocessed color. A Krea2 LoRA author reports natural-language scene prompts surfaced skin detail better than their early LoRA training attempts; a two-stage recipe deliberately changes prompt and sampler for its detail pass; cross-model refinement can further redraw texture. Convert these into a structured *realism target* that scores capture coherence and skin naturalness separately from acutance. Test mild/local detail changes only after the baseline image is compositionally right.

## Suggested experiment queue

| ID | Question | First small experiment | Success criterion |
|---|---|---|---|
| R1 | Does partial de-distillation improve realism/control, and does the first-call null replicate? | First: strength × CFG grid on 12 prompts × 4 seeds (open and constrained); separately, a small RAW-first-call replication only if setup differs from the published Krea2 pilot. | Higher blinded realism/fidelity/artifact scores and useful CFG headroom; report diversity by prompt class. Treat RAW-first-call as secondary unless the null is disproved. |
| R2 | Which settings reduce blotches on dark smooth regions? | 8 targeted prompts × 3 seeds; baseline plus one-factor schedule, resolution, and checkpoint comparisons. | Lower artifact ratings/diagnostics with no broad detail or contrast loss. |
| R3 | Is sampler choice quality or preference? | 8 prompts × 3 seeds; 4 sampler/scheduler finalists at fixed call count where feasible; randomized labels. | Reproducible task-specific differences with uncertainty intervals; otherwise document near-ties. |
| R4 | Does the SDA/first-two-step intervention transfer to installed Krea2 checkpoints? | Baseline vs LoRA at early-only and full trajectory, on official Turbo and one merge. | Diversity gain without unacceptable identity, prompt, or texture regression. |
| R5 | Does prompt expansion solve a different problem than sampler variety? | Factorial prompt-seed × latent-seed comparison with identical generation settings. | Separate effect estimates for semantic/layout/local detail diversity. |
| R6 | How much does precision alter photo quality and speed? | Extend the 15-stratum matched-noise design with two or more seeds, focusing on faces/skin, dark gradients, hair, reflections, and hands; repeat on target GPU and with representative LoRA. | Human blind preference and artifact ratings alongside BF16 similarity, runtime, VRAM; no universal ranking inferred from one GPU. |
| R7 | Which controls cause skin overtexture? | Same seeds and prompt; one-factor sampler, eta/detail, conditioning, VAE, precision, and LoRA tests; include a neutral skin prompt and controlled microtexture wording. | Improve skin-naturalness rating without increasing blemish/noise score or changing identity/age. |
| R8 | Can prompt cleanup improve realism without changing intent? | Compare original tag prompt, faithful prose rewrite, photography-cue rewrite, and optional reward-guided variants across the same seeds. | Better blind photographic-naturalness scores while independent raters confirm subject, scene, camera framing, and requested attributes remain intact. |
| R9 | Does spending a small extra sample budget improve final photo realism? | Generate 1/2/4/8 matched candidates for a stratified prompt set; compare random pick, human blind pick, HPS/VLM rank, and artifact-checklist rank. | Human-picked quality rises with candidate budget; quantify where automated rankers agree and where added compute plateaus. |
| R10 | Do higher detail settings improve photography or merely acutance? | Same source seed/candidate; sweep one detail/eta control from neutral to mild to strong on portraits, hair, fabric, dark backgrounds, and still life. | More natural material detail without pore overshoot, haloing, broken noise, false grain, or color/lighting discontinuity. |
| R11 | Does partial Turbo de-distillation improve photographic realism or CFG control? | Replicate the published Turbo distillation-LoRA strength axis using the exact inference weights/merge convention, at strengths 1.0, 0.9, 0.8, 0.7, 0.5; first hold CFG=1 and 8 steps, then test CFG 2/2.5/3 only at promising strengths. Include open scenes and constrained prompts. | Blinded realism/fidelity/artifact scores improve or stay equal; report strength × prompt-class interaction and compute. Do not treat adapter strength as a universal diversity slider. |
| R12 | Can a drawn outline reliably constrain object geometry on Turbo? | First train/use the public RAW Control-LoRA as a portability probe; compare no map, Canny, clean silhouette, and segmentation on identical paired targets, then test the same adapter on Turbo. Keep source image/control perfectly aligned. | Shape adherence rises while material realism, lighting, internal structure, and prompt fidelity remain acceptable. A failure on Turbo is informative; do not silently substitute RAW results. |
| R13 | Which signal best expresses the user's intended object? | On a compact product/object set, compare contour-only, Canny with internal edges, segmentation, depth, and Canny+depth. Hold prompts/seeds/checkpoint fixed; vary control weight and active denoising interval separately. | Identify whether outer silhouette, part layout, 3D volume, or object identity is the limiting factor; choose the lightest adequate condition. |
| R14 | Does Turbo-native LoRA training preserve realism better than RAW-trained LoRA transfer? | Same small identity/style dataset, captions, rank, LR budget, and held-out prompts; compare RAW-trained LoRA on Turbo against Turbo training with the official de-distill adapter. Track overfit, identity, background leakage, and quality at 8 steps. | Better held-out Turbo results without losing identity/style specificity or step-distilled speed. Include adapter wear-down with training duration. |
| R15 | Which dataset curation decisions matter most for realism? | For a small licensed photo set, ablate near-duplicate removal, background diversity, crop/shot diversity, caption length/detail, and synthetic additions one factor at a time. Keep a source-disjoint holdout. | Higher naturalness and generalization with no increase in memorized composition/background or synthetic look. Keep AI-generated additions as a separate arm, never silently mix them into the real-photo baseline. |

These are proposed protocols, not completed tests. Keep sample counts practical in the first screen, then enlarge only promising effects and reserve unseen prompts for the final check.

## Turbo-specific methods, dataset practice, and object outlines

### 1. Sampling math and the Turbo distillation axis

Turbo is a step-distilled checkpoint, not simply RAW with an arbitrary fast sampler. The
documented baseline is 8 steps, CFG off, and fixed flow shift μ=1.15. Preserve that as the
control arm whenever testing unusual sigma curves, samplers, or CFG. AI-Toolkit's Turbo
training path adds a rank-32 de-distillation assistant adapter during training and cancels
it during preview; longer training can wear down distillation, so preview quality and step
count must be logged throughout training. Separately, the community strength pilot studies
the Turbo distillation LoRA/strength axis at inference. Do not assume the training assistant
adapter and inference distillation LoRA are identical artifacts or that their scales map
directly; verify exact weights and merge convention first.

A Krea2-specific public exploration swept this axis on very few prompts and seeds. It
reports that lower effective distillation can restore CFG headroom; under that setup CFG
did more for sharpness than adding steps; common schedulers and μ values were weak levers
except at extremes; and diversity gain appeared on constrained/dense prompts but not open
scenes. Its author labels confidence low-to-medium. Treat this as a mechanism lead, not a
preset recommendation. The same notes report a null result for a base-model-first-step
diversity intervention on Krea2. This direct counter-evidence demotes R1: replicate only
if the installed RAW/Turbo pair or node implementation differs materially from that pilot.

For maximum realism, the practical candidate is not “more stochasticity at all costs”; it
is a controlled tradeoff between Turbo strength, CFG headroom, and adherence. Score natural
skin, fine texture, coherent lighting, object geometry, and prompt fidelity independently.
Strong CFG can sharpen but may burn highlights or distort skin at full Turbo strength.
Guidance conventions differ between libraries: log effective guidance, including any UI
offset such as `guidance_scale - 1`, not only the displayed number.

### 2. What commercial Krea training disclosures imply for smaller datasets

Krea's technical report offers unusually useful first-party process detail. For foundation
pretraining it emphasizes coverage and diversity rather than blindly keeping the highest
aesthetic scores; deduplicates exact and near-duplicate images; enriches captions using OCR,
metadata, and multiple caption styles/lengths; stages resolution and filtering; and uses
curated aesthetic SFT to address saturation and texture problems. Its preference pipeline
starts with synthetic pairs but calibrates with human annotations. These are reported
commercial practices, not a turnkey recipe for a 20–100-image LoRA: the scale and objective
are fundamentally different.

Transferable small-dataset principles:

- Curate for coverage of the target concept, not maximum count or one generic quality score.
  For identity: vary views, scale, expression/state, lighting, crop, and background; for a
  product: include front/side/back/detail views and material-specific lighting.
- Remove exact/near duplicates, watermarks, broken files, and accidental low-quality images;
  inspect repeated backgrounds and burst-shot clusters because they teach shortcuts.
- Caption invariant identity consistently, while describing per-image variables (view, pose,
  crop, garment, lighting, background, camera distance). Caption visible text when it
  matters. Generic “beautiful photo” captions hide the differences needed to generalize.
- Use source-disjoint holdout, not neighboring frames from one photo burst. For products,
  hold out whole views/sessions; for people, hold out scenes/backgrounds.
- Keep generated images as an explicit experimental arm. Krea says it excluded synthetic
  AI images from foundation pretraining because of learned biases and a quality ceiling.
  That does not prove synthetic pairs hurt a small control-LoRA, but argues against mixing
  them invisibly into the real-photo baseline.

For photorealism, quality filtering should target defects (compression, blur, artifacts,
bad crops, anatomy, color cast) without collapsing the set into a narrow stock-photo look.
Include natural texture and difficult lighting; otherwise the fine-tune may look clean but
less believable.

### 3. A drawn outline: what it can and cannot teach

There are two different tasks, and the data/architecture should match the desired behavior.

**“Recreate this shape/layout, but render it as a photo.”** Use a spatial condition at
inference: Canny/line-art for boundaries and internal edges, segmentation for part/region
layout, or depth for coarse 3D geometry. A custom control map must stay pixel-aligned with
its target through resizing/cropping. A clean silhouette can communicate outer shape and
pose, but cannot by itself specify hidden surfaces, material, identity, lighting, or realistic
texture. Combine it with text and, if needed, a separate object-identity LoRA/reference.

**“Teach Turbo what this particular object is.”** Train an object/product LoRA on actual
photos or renders spanning views, crops, lighting, and backgrounds, with stable identity
captioning. An outline drawn around every image but never provided at generation time is
only a dataset annotation; it does not create an inference control channel. A segmentation
mask can still isolate/crop the object and reduce background shortcut learning, but the
LoRA needs enough appearance variation to learn identity.

The public Krea-2 Control-LoRA implementation is a strong prototype reference: it encodes
the map with Qwen Image VAE and concatenates its latent with the noisy target latent at each
step, training an expanded input projection plus rank-64 LoRA while freezing the base. It
supports Canny, depth, blur/tile, grayscale, and user-supplied aligned maps such as
segmentation or sketches. Crucially, its documented checkpoint/training command targets
Krea 2 RAW. Until Turbo compatibility is demonstrated, call a RAW-trained adapter on Turbo
an experiment. The repo's name does not make this the classic residual-branch ControlNet;
verify architecture and node wiring before assuming interchangeability.

Best first study: identical target photos, prompts, seeds, and held-out objects; compare no
control, Canny, clean hand-drawn silhouette, segmentation, depth, and paired Canny+depth.
Use multiple control strengths and, separately, early-only vs full-trajectory conditioning.
Rate silhouette/part accuracy, material realism, lighting, texture, text fidelity, and
artifacts. Split by object/session to prevent near-identical photo leakage. If the output
obeys the contour but looks like a flat cutout, that is expected: the map constrains
geometry; photo appearance must come from paired targets, caption, and base-model prior.

### Source ledger additions (2026-10-15)

- **First-party commercial disclosure:** [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report) — data coverage, captioning, resolution curriculum, curation, aesthetic SFT, preference/RL and distillation practices. Foundation-scale evidence; adaptation to LoRAs is a hypothesis.
- **Turbo training mechanics:** [Ostris AI Toolkit Krea 2 Turbo docs](https://ostris.com/docs/ai-toolkit/models/krea2-turbo) — Turbo de-distill adapter, cancellation during preview, flowmatch, linear timesteps, and documented 8-step / CFG-off / μ=1.15 inference baseline.
- **Direct Turbo pilot:** [Turbo-LoRA strength exploration](https://github.com/fblissjr/krea-explorations/blob/main/docs/turbo_lora_strength.md) — limited prompts/seeds; includes a Krea2 null result for base-first-step diversity and prompt-dependent strength behavior. Replication required.
- **Spatial conditioning implementation:** [Krea-2 Control-LoRA trainer](https://github.com/Tanmaypatil123/Krea-2-controlnet/blob/main/trainer/README.md) — public RAW trainer for Canny/depth/custom aligned controls; implementation precedent, not proof of Turbo compatibility or quality.
- **General spatial-control foundation:** [ControlNet paper](https://arxiv.org/abs/2302.05543) — precedent for edge/depth/segmentation controls on other diffusion models; not a Krea2 result.

## Evidence limits and cautions

- Reddit comments are valuable for finding pains and hypotheses, not measuring causal effects.
- The large community sampler grid is not a universal ranking; prompt/checkpoint/resolution coverage and blind review are limited.
- RAW schedule claims must not be transferred to Turbo without a separate test.
- A scientific result on another distilled model is a mechanism lead, not a Krea2 result.
- Metrics such as CLIP distance, HPS, image sharpness, SSIM, or frequency energy each capture only part of the target. Pair them with blinded human preference and explicit artifact/fidelity ratings.
- Don't collapse “more different” into “better”: diversity may rise while prompt adherence, identity, realism, or detail falls.

## References

1. Krea AI, [Krea 2 official inference repository](https://github.com/krea-ai/krea-2).
2. Krea AI, [Krea 2 technical report](https://www.krea.ai/blog/krea-2-technical-report).
3. Gandikota & Bau, [Distilling Diversity and Control in Diffusion Models](https://arxiv.org/abs/2503.10637), WACV 2026.
4. Lu et al., [DPM-Solver++](https://arxiv.org/abs/2211.01095), 2022.
5. Hong et al., [Perturbed-Attention Guidance](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/).
6. Singh & Fischer, [Stochastic Sampling from Deterministic Flow Models](https://arxiv.org/abs/2410.02217).
7. Zou et al., [Precise: SDE-Consistent Stochastic Sampling](https://arxiv.org/abs/2605.23522).
8. Kim et al., [Reward-Agnostic Prompt Optimization](https://arxiv.org/abs/2506.16853).
9. Wu et al., [Human Preference Score v2](https://arxiv.org/abs/2306.09341).
10. Choi et al., [Performance Plateaus in Inference-Time Scaling for T2I](https://arxiv.org/abs/2506.12633).
11. Li et al., [Reflect-DiT](https://arxiv.org/abs/2503.12271).
12. Jena et al., [Elucidating Optimal Reward-Diversity Tradeoffs / AIG](https://arxiv.org/abs/2409.06493).
13. Community discussions and projects listed in the source log above; checked 2026-10-15.
