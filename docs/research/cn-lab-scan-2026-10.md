# Chinese-lab scan: spatial sampling, training-free detail, identity and layout

**Scan date:** 2026-10-16  
**Scope:** 2024–2026 work, mainly from Chinese labs and Chinese-language sources (arXiv, Zhihu,
CSDN, Bilibili, LiblibAI, 机器之心, 量子位), relevant to KreaPhoton's sampler, the spatial detail
hypothesis, character identity, and composition.  
**Status:** desk research. Three parallel agent sweeps were made; each reported only sources it
could open, and items it could not verify were dropped. Numbers below are the papers' own and
are **not replicated here**. Nothing in this note is a KreaPhoton default.

Evidence labels follow the [README](README.md): **Paper result**, **Reported**, **Hypothesis**.

## Source landscape

- **Strong:** lab papers. The main contributors are:
  - HKUST-Guangzhou (Zeke Xie group);
  - Fudan;
  - SJTU EPIC lab;
  - PKU + InstantX;
  - Tsinghua / Beihang / Renmin;
  - Zhejiang University;
  - Nanjing University;
  - ByteDance, StepFun, Tencent Hunyuan.
- **Weak:** the Chinese community (Zhihu / CSDN / Bilibili / Liblib). There is almost no
  verifiable sampler-level material. What exists is workflow recipes, LoRA captioning advice,
  and "zero the negative conditioning" tips. Zhihu returned 403, so Zhihu claims come from
  search snippets only.
- No verifiable sampler-level fix for 塑料感 / 油腻感 (plastic / greasy skin) was found. Proposed
  fixes are LoRAs or prompt changes.

## Top candidates for KreaPhoton

Ranked by fit to existing machinery, cost, and the size of the pain they target.

| # | Idea | Source basis | Cost | Targets |
|---|---|---|---|---|
| 1 | Apply the detail-boost delta and the terminal x0-extrapolation delta **high-pass only**; freeze or damp the low band | FDG, FreSca, LF-CFG | 0 extra calls | dark-region blotches, tone/brightness drift (R2), detail overshoot (R10) |
| 2 | **Sigma-windowed identity LoRA:** strength ramps up after the composition phase instead of being constant | PuLID-FLUX start-step knob; T-LoRA | 0 (core ComfyUI hook keyframes) | outfit/background lock-in in character LoRAs |
| 3 | **Likeness evaluation by Sim(GT) + copy-paste score** against a held-out photo | WithAnyone | eval only | honest face-pass / LoRA A/B |
| 4 | **Face pass: early-sigma token replacement** inside the face mask, then attention concatenation | Personalize Anything | attention hooks | likeness (after the null result of `ref_mean_guidance`) |
| 5 | **Momentum Guidance** at CFG 1 | Momentum Guidance | 0 extra calls | general quality |
| 6 | **CADS-style noise on the Qwen3-VL embedding** in steps 1–2 | CADS; SeedVarianceEnhancer | ~0 | seed diversity (alternative to latent variety) |
| 7 | **Regional attention masks in early sigmas only**, tied to Composition Drafts | Regional Prompting FLUX; RAG-Diffusion; Stitch | attention hooks | layout control (answer to FLUX 3 Klein / MOSS bbox) |
| 8 | **Seed screening by inversion stability** | Qi et al. (Xie lab) | N × short round trip | seed quality |

## 1. Spatially adaptive sampling

These are direct inputs to [the spatial detail hypothesis](spatial-detail-allocation-hypothesis.md).

**RAS — Region-Adaptive Sampling for DiT**  
*Who / where:* Ziming Liu, Yang You (NUS); Yuqing Yang, Lili Qiu (Microsoft Research). Chinese
authors. arXiv 2502.10389 (Feb 2025), CVPR 2026.  
*Mechanism:* each step, patches are ranked by the per-patch std of the previous predicted noise.
Only the top patches are run through the DiT; the rest reuse cached noise and K/V. Training-free.
Tested on SD3 and Lumina-Next.  
*Paper result:* up to 2.36× speed-up on SD3 and 2.51× on Lumina-Next; in a user study, 45% rated
the quality similar.  
*Relevant details:*
- The first steps must be dense, because early sparsity breaks structure (**Paper result**).
- Focus regions are stable across steps (**Reported**).
- Without a drop-count term that prevents starving skipped background patches, backgrounds go
  blurry or noisy, so periodic dense steps are needed (**Reported**).

*For us:* the per-patch std of the velocity is a free extra complexity signal for `m(x, y)`. At
8 steps, a 1–2 step warmup is the practical limit (**Hypothesis**).

**SDiT — Semantic Region-Adaptive DiT**  
*Who / where:* Bowen Lin … Chengming Zhang. Chinese-named authors at US / Singapore
institutions. arXiv 2601.12283 (Jan 2026).  
*Mechanism:* Quickshift clustering on x0 and noise features. Per-pixel complexity is gradient
+ Laplacian + residual noise. Only the top regions are denoised; the rest are extrapolated in
velocity space. Training-free. Tested on Lumina-Next (flow matching).  
*Paper result:* up to 3.0× speed-up. The reported numbers are internally inconsistent.  
*Relevant details:*
- Dilating the selected regions by 1–2 px beats no dilation (**Paper result**). Hard boundaries
  hurt.
- Updating only the complex regions degrades coherence, because flat regions give context to
  their neighbours (**Reported**).

*For us:* its complexity formula is close to the proposed `m`. Feather, never threshold.

**AsynDM — Asynchronous Denoising Diffusion Models**  
*Who / where:* Zijing Hu, Kun Kuang et al. (Zhejiang University + NTU). Chinese. arXiv
2510.04504, ICLR 2026.  
*Mechanism:* per-pixel timesteps. Prompt-related regions, found from cross-attention masks,
denoise more slowly. Training-free; per-token timesteps are run by computing the adaLN
modulation per token. Tested on SD2.1, SDXL, and SD3.5-M (a DiT).  
*Paper result:* ImageReward 0.92 vs 0.75, FID roughly unchanged. A large timestep gap
(ω = 0.8–0.9) leaves residual noise and blur in the faster regions.  
*For us:* Krea 2 takes a scalar timestep, so true per-token sigma would need surgery on its
modulation; the dual-call x0 blend avoids that. **Cap the max/min ratio** of the per-pixel eta
and nudge, since AsynDM's failure mode is our blotch risk (**Hypothesis**).

**SANI — Spatially Adaptive Noise Injection**  
*Who / where:* Lavda, Kalousis et al. (HES-SO / Geneva). Not Chinese. arXiv 2609.18466 (Sep
2026).  
*Mechanism:* a per-pixel gate on the exploratory noise term, driven by posterior variance
from the Hessian diagonal; the mean-error term is kept everywhere. Training-free, but the
Hessian estimate costs compute. Tested on DDPM/DDIM at ≤ 64 px; LSUN 256 qualitatively only.  
*Paper result:* CelebA-64 at 10 steps, FID 17.1 vs 20.5 for DDIM. Competitive with stronger
baselines, not uniformly better.  
*For us:* the closest evidence for variant A. Its premise is that injected noise degrades
smooth regions; late in sampling the gate is ≈ 0 except at edges (**Reported**, one sample).
Not a flow DiT; no seam analysis.

**FreeEnhance**  
*Who / where:* Yang Luo, Ting Yao, Tao Mei et al. (Fudan / HiDream.ai). Chinese. arXiv
2409.07451, ACM MM 2024.  
*Mechanism:* a high-pass mask. High-frequency regions get light noise via DDIM inversion; flat
regions get **strong** noise to create detail, with variance rescaled after blending.
Training-free. Tested on SDXL and SD1.5.  
*Paper result:* HPSv2 29.32 vs 28.88 for FreeU.  
*For us:* **opposite polarity to variant A**, though in an img2img enhancement setting. Its
acutance term created spurious structure in flat sky and metal, so it was restricted to the
35th–65th percentile (**Reported**). This underlines the main risk: the map cannot tell "flat
because underdeveloped" (plastic skin) from "flat by design" (sky).

**SAMG — Spatial Adaptive Multi Guidance ("Delta Score Matters!")**  
*Who / where:* Haosen Li … Yutao Yue. Chinese authors; affiliation not verified. arXiv
2604.26503 (Apr 2026).  
*Mechanism:* a per-point "guidance energy". High-energy boundaries get the minimum guidance
scale, low-energy regions the maximum. Training-free. Tested on SD1.5, SDXL, SD3.5-M, and video.  
*Evidence:* no numbers in the abstract (**Reported**).  
*For us:* inverse polarity that protects edges, analogous to `trust_clamp`. For variant B,
limit the nudge at strong edges to avoid halos (**Hypothesis**).

**RALU — training-free mixed-resolution latent upsampling**  
*Who / where:* Jeong, Se Young Chun (SNU). Korean. arXiv 2507.08422, CVPR 2026.  
*Mechanism:* Tweedie x0 → decode → Canny → edge tokens are upsampled early. Tested on
FLUX.1-dev and SD3.  
*Paper result:* up to 7× speed-up on FLUX; edge aliasing and noise-level mismatch were
reduced by region selection.  
*For us:* evidence that a map built from mid-trajectory x0 is reliable enough for routing.

**ToCa / TaylorSeer**  
*Who / where:* Chang Zou, Linfeng Zhang et al. (SJTU EPIC). Chinese. ICLR 2025 (2410.05317),
ICCV 2025 (2503.06923).  
*Mechanism:* training-free per-token feature caching or forecasting, scored by attention
influence.  
*Paper result:* 2.36× on OpenSora; 4.99× on FLUX.  
*For us:* speed only, no quality mechanism.

**Patch Forcing / AsyncPatch**  
*Who / where:* Patch Forcing is Ommer's lab (2604.19141, CVPR 2026); AsyncPatch is Papa,
De Bortoli et al. (2606.07079). Not Chinese.  
*Mechanism:* per-patch timesteps plus a difficulty head. **Training required.**  
*Paper result:* AsyncPatch notes that mixed per-patch noise levels are out of distribution for
models not trained on them.  
*For us:* rules out real per-token sigma on Krea 2 and supports blending outputs instead.

**Background only:** Differential Diffusion (2306.00950; per-pixel strength maps for img2img,
only the abstract was checked) and S-CFG (2404.05384; per-region CFG from attention, Chinese
authors, U-Net only).

## 2. Training-free detail, artifacts, diversity

**Frequency-decoupled guidance: FDG, FreSca, LF-CFG**  
*Sources:* FDG, Sadat et al., 2506.19713. FreSca, Chao Huang … Chenliang Xu (Rochester),
2504.02154. LF-CFG, Kaiyu Song & Hanjiang Lai, 2506.21452 (affiliation not verified).  
*Mechanism:* low-frequency guidance drives structure but also oversaturation; high frequency
drives fidelity. The bands are scaled separately (FDG suggests low ≈ ½ high). LF-CFG damps
low-frequency content that accumulates redundantly over steps.  
*Paper result:* better FID / recall and less oversaturation on SD-family models.  
*For us:* at CFG 1 the papers' setting does not apply directly. But the detail boost (a
lower-sigma call) and the terminal x0 extrapolation both produce a delta. Passing only the
high band of that delta should cut tone drift and dark blotches while keeping texture
(**Hypothesis**; top candidate #1). `trajectory._bands` already provides the band split.

**Momentum Guidance**  
*Who / where:* Liao … Qiang Liu (UT Austin). arXiv 2602.20360 (Feb 2026; v2 June 2026).  
*Mechanism:* an EMA of past velocities, `m = (1 − β) v + β m`; step with `v + α (v − m)`. No
extra calls.  
*Paper result:* FLUX.1-dev at CFG 1, HPSv2.1 24.40 → 24.80; SD3 at CFG 1, 22.87 → 27.37. Not
tested on distilled few-step models.  
*For us:* new. With only 8 velocities the EMA is short, so use a small α and switch it off in
the last 1–2 steps, where it could fight the x0 extrapolation (**Hypothesis**).

**Negative guidance at CFG 1: NAG and VSF**  
*Sources:* NAG, Chen et al. (Surrey), 2505.21179, NeurIPS 2025. VSF, Guo & Du (UBC),
2508.10931, ICLR 2026.  
*Mechanism:* NAG extrapolates in attention-output space with L1 normalisation. VSF flips the
sign of the attention values coming from the negative prompt.  
*Paper result:* both restore negatives in 1–8-step models (FLUX-schnell, SD3.5-Turbo, Wan).
VSF negative score is 0.420–0.545 vs 0.32–0.38 for prior methods. ComfyUI nodes exist.  
*For us:* already listed as a replication candidate in the [landscape note](krea2-turbo-landscape-2026-10.md)
(via AlchemicaSampler). A built-in skin negative ("waxy plastic skin, airbrushed, blotchy
shadows") is a **Hypothesis**.

**Diversity through conditioning or features instead of latent noise**
- **CADS** (Sadat et al., ICLR 2024, 2310.17347): annealed Gaussian noise on the conditioning
  embedding.
- **SeedVarianceEnhancer** (community node for Z-Image Turbo): noises a percentage of the text
  embedding in early steps only, and trades away some prompt adherence (**Reported**).
- **STRIDE** (2605.11494, May 2026): PCA-projected, spatially coherent noise on intermediate
  transformer features, in one pass. Paper result on FLUX-schnell: DreamSim 0.363 → 0.423, CLIP
  28.34 → 29.04, HPS 0.296 → 0.282. In the same table, CADS reaches DreamSim 0.431 but CLIP falls
  to 27.34.
- *For us:* KreaPhoton's variety perturbs the latent noise. Noising the Qwen3-VL embedding in
  steps 1–2 is new and cheap. STRIDE needs block hooks.

**Z-Sampling / Zigzag**  
*Who / where:* Bai, Shao … Zeke Xie (HKUST-GZ). Chinese. 2412.10891, ICLR 2025.  
*Mechanism:* denoising with strong guidance alternates with inversion under weaker guidance;
the gap accumulates semantics.  
*Paper result:* up to 94% HPSv2 win rate on DreamShaper; also works on distilled models and
DiTs.  
*For us:* close to restart / self-refine. The new part is replacing the re-noise with an
**inversion** step. At CFG 1 the guidance gap must come from elsewhere, e.g. inversion under an
empty or NAG-negative condition (**Hypothesis**).

**Noise selection and optimisation by inversion stability**  
*Who / where:* Qi, Bai, Xiong, Xie (HKUST-GZ). Chinese. 2407.14041 (2024).  
*Mechanism:* a seed is good if it survives a generate → invert round trip; noise can also be
optimised toward that stability.  
*Paper result:* SDXL / SDXL-Turbo, DrawBench human-preference win rate 57% for selection and
72.5% for optimisation.  
*For us:* a cheap seed screen with a 1–2 step round trip per candidate. Cost scales with the
candidate count (**Hypothesis**).

**FreqPrior**  
*Who / where:* Yunlong Yuan et al. (Fudan + Huawei Noah's Ark). Chinese. 2502.03496, ICLR 2025.  
*Mechanism:* a frequency-filtered initial noise that stays Gaussian (earlier methods let the
variance decay and lost detail), refined by partial sampling and re-noising.  
*Paper result:* top VBench quality score. Video only.  
*For us:* partly covered by Composition Drafts. The new part is re-injecting only the
**low-frequency** part of a draft into fresh high-frequency noise, with variance preserved
(**Hypothesis**).

**Time-Shift Sampler (exposure bias)**  
*Who / where:* Mingxiao Li et al. (KU Leuven). 2305.15583, ICLR 2024.  
*Mechanism:* call the model at the timestep whose noise level matches the sample's actual
variance, not the scheduled one.  
*Paper result:* CIFAR-10 FID 3.88 with F-PNDM at 10 steps.  
*For us:* the sigma-space detail boost is a fixed-offset relative of this. The new part would
be choosing the offset **adaptively** from measured latent variance (**Hypothesis**).

**Krea-specific and community signals (Reported)**
- The **F16 Krea 2 Turbo SDA diversity LoRA** uses Krea 2 RAW as the teacher at high noise.
  It claims +85% pairwise CLIP distance at equal HPS, but helps only when active for steps 1–2
  of 8; always-on costs about −10% HPS. It is trained, but it locates where diversity is lost.
- The **Z-Image Turbo FlowSampler** author warns that non-linear schedules "break the
  distillation" and that eta = 0 is sharpest. Keep this in mind for schedule experiments.

**Already covered or contested**
- **Diversity Distillation** (Gandikota & Bau, 2503.10637, WACV 2026): run the undistilled base
  for timestep 1, then the distilled model. Paper result on SDXL/DMD2: DreamSim 0.264 → 0.350,
  FID 15.52 → 10.79 at the same latency. However, the [landscape note](krea2-turbo-landscape-2026-10.md)
  records a **null Krea2-specific pilot** for this exact intervention. It stays low priority
  unless a replication under different conditions says otherwise.

**Dropped:** Golden Noise / NPNet (needs a trained network). HiFlow and FreCaS
(high-resolution methods that overlap the tiled refine).

## 3. Identity and likeness

**PuLID-FLUX: delayed ID insertion start**  
*Who / where:* ByteDance. Chinese. Repository docs, 2024.  
*Mechanism:* one parameter sets the timestep where ID injection starts. 0 means from the first
step: maximum fidelity, least editable. A later start is more editable and disturbs the base
less. The docs recommend about 4 for realistic images and 0–1 for stylized ones.  
*Evidence:* **Reported** (first-party docs, no numbers). The adapter is trained; the knob is
free.  
*For us:* the clearest first-party evidence for "identity only in a sigma window". Any identity
source (LoRA, reference tokens) could get a `start_sigma` / `end_sigma` envelope. At 8 steps,
skip roughly the first 1–2 (**Hypothesis**).

**T-LoRA**  
*Who / where:* AIRI. Not Chinese. 2507.05964.  
*Mechanism:* LoRA overfitting concentrates at high-noise timesteps, which memorize pose and
background. Dropping t ∈ [800, 1000] entirely hurts concept fidelity, so they mask the rank
by timestep instead. Tested on SDXL and FLUX.1-dev.  
*Evidence:* **Paper result.**  
*For us:* explains outfit / background lock-in in sigma terms. **Ramp identity-LoRA strength
down at high sigma rather than cutting it to zero**, as a "LoRA strength vs sigma" curve on the
existing envelope machinery. Core ComfyUI already has Create Hook LoRA + Set Hook Keyframes.

**Personalize Anything for Free**  
*Who / where:* Tsinghua / Beihang / Renmin. Chinese. 2503.12590 (Mar 2025).  
*Mechanism:*
- In early steps (the first 20% of 50), reference-subject tokens hard-replace the denoising
  tokens in the subject region, keeping the target's positional encodings.
- Afterwards, zero-positioned reference tokens are concatenated into joint attention.
- For editing / inpainting, the swap point moves to 10%.

Training-free. Tested on FLUX.1-dev and HunyuanDiT.  
*Paper result (DreamBench):* CLIP-I 0.876 and CLIP-T 0.307, best of the baselines; DINO 0.683,
below SSR-Encoder's 0.701. No face-ID metric.  
*For us:* a concrete alternative to the failed reference-mean guidance. On the face pass,
replace tokens inside the face mask for the first ~20% of sigmas. Expect pose copying; it
needs aligned crops (**Hypothesis**).

**WithAnyone**  
*Who / where:* StepFun + Fudan. Chinese. 2510.14975, ICLR 2026.  
*Mechanism:*
- An ArcFace ID loss using ground-truth landmarks, valid at all noise levels (PortraitBooth
  applies its ID loss only at t < 0.25).
- An InfoNCE loss against about 4k negatives.
- Paired training: the target is a different photo of the same person.
- A **copy-paste metric**: similarity to the ground-truth photo vs similarity to the reference.

*Paper result (single-person benchmark):*

| Method | Sim(GT) | Copy-paste |
|---|---|---|
| WithAnyone | 0.460 | 0.144 |
| PuLID | 0.452 | 0.315 |
| DreamO | 0.454 | 0.303 |
| InfiniteYou | 0.439 | 0.233 |

Removing paired tuning raised copy-paste from 0.161 to 0.239 (two-person subset).  
*For us:* adopt the evaluation. Score face-pass and LoRA A/B tests against a **held-out photo**
of the person, and report copy-paste. Similarity to the reference rewards copying. Keep one
held-out validation photo per identity in LoRA recipes.

**FreeCure**  
*Who / where:* HKUST + Tencent Hunyuan. Chinese. NeurIPS 2025.  
*Mechanism:* run the model with and without the identity source. Face-parse the no-identity
output into hair / glasses / accessory masks and inject its attention into those regions. An
optional inversion step starts at γ = 0.5. Training-free. Tested on PuLID-FLUX and InfiniteYou.  
*Paper result:* on PuLID-FLUX, prompt consistency +10.5% and ID fidelity −3.2%.  
*For us:* targets attribute lock-in. Workflow idea: LoRA on inside the face mask, off (or base
attention borrowed) on hair and clothes. ComfyUI hook masks can partly do this today.

**Chinese community captioning rules** (Zhihu / CSDN; **Reported**, low quality)
- *Consensus:* do not caption fixed traits (facial features), so they bind to the trigger word.
  Do caption variable traits: clothes, accessories, expression, background, framing. Background
  words left out of captions reportedly stick to the LoRA.
- *Dissent:* some authors caption everything visible.
- *Block targeting* (Zhihu, snippet only): training only FLUX
  `single_transformer_blocks.7.proj_out` reportedly carries likeness with less effect on
  composition. FLUX-specific; Krea 2 block indices would need our own ablation (**Hypothesis**).

**Krea 2 Identity Edit LoRA** (Conradlocke, Jul 2026, via ComfyUI Wiki; not Chinese)  
Built on Krea 2 Raw. Turbo is recommended at 8–12 steps, CFG 1. A `grounding_px` setting trades
edit adherence against identity. Claims only, no benchmarks.

**Not portable without retraining:** InfiniteYou, DreamO, UNO / UMO (ByteDance),
InstantCharacter (Tencent Hunyuan). Useful only as benchmarks.

**Why the face-pass null result fits.** Our measured null for `ref_mean_guidance` is consistent
with this literature. Methods that keep likeness inject per token or through attention, not
through averaged latents.

## 4. Training-free composition and layout

**Regional Prompting FLUX**  
*Who / where:* PKU + InstantX. Chinese. 2411.02395.  
*Mechanism:* a block attention mask restricts image↔text, text↔text, and image↔image attention
per region. It is injected **only in early steps**, since "earlier steps decide the overall
layout". The repo recommends 5–10 injection steps and a base-latent blend β of 0.1–0.3. Works
with LoRA, ControlNet, and multi-person PuLID.  
*Evidence:* qualitative only; about 9× faster than RPG with 16 masks.  
*For us:* a natural fit for Composition Drafts. Apply region masks only in the drafted early
sigmas, lock the layout, then run unmasked. Caution: [Krea2 issue #15](https://github.com/krea-ai/krea-2/issues/15),
recorded in the character-LoRA note, reports that regional prompting did not isolate the LoRA
effect in crowded scenes.

**RAG-Diffusion (RAGD)**  
*Who / where:* Nanjing University. Chinese. 2411.06558, ICCV 2025.  
*Mechanism:* Regional Hard Binding builds each region from its own prompt for N iterations;
Regional Soft Refinement then blends regions with strength δ. It can repaint one region
without an inpainting model.  
*Evidence:* about +29% on T2I-CompBench spatial (**Reported** via a third-party summary).  
*For us:* draft-level per-region repaint ("redo only the left person").

**3DIS-FLUX** (Zhou / Xie / Yang, 2501.05131; affiliation not verified)  
Scene as a depth map first, then FLUX renders it with a layout-derived joint-attention mask.

**Stitch** (2509.26644; not Chinese)  
Training-free bbox control on Qwen-Image, FLUX, and SD3.5. GenEval Position +218% on FLUX.

*For us:* both support adding a bbox → mask path in early sigmas only. This is a cheap answer
to the built-in bbox grounding of FLUX 3 Klein and MOSS-Image.

## Proposed experiments

All of these are hypotheses; run them under the README protocol.

| ID | Question | Minimal design | Success signal |
|---|---|---|---|
| C1 | Does a high-pass-only detail / extrapolation delta reduce dark blotches and tone drift? | R2 + R10 strata, ≥ 3 seeds; arms: current delta vs HP-only vs HP + damped LF | fewer blotch / banding ratings, no texture loss under blind review, mean-tone drift ↓ |
| C2 | Does a sigma-windowed identity LoRA reduce outfit/background lock-in without losing likeness? | Character LoRA; constant vs ramp-in after composition vs T-LoRA-style high-sigma ramp-down; varied outfit/background prompts | Sim(GT) held or ↑, copy-paste ↓, outfit/background follow the prompt |
| C3 | Is the Sim(GT) + copy-paste protocol practical for our face pass? | One held-out photo per identity; re-score the existing face-pass comparisons | rankings change where reference-similarity rewarded copying |
| C4 | Does early-sigma face token replacement beat the current face pass? | Aligned face crops; replacement for the first 10 / 20% of sigmas | Sim(GT) ↑ without a copy-paste ↑ |
| C5 | Is Momentum Guidance safe on an 8-step distilled model? | Small α grid, off for the last 1–2 steps | blind preference ≥ baseline, no overshoot |
| C6 | Does Qwen3-VL embedding noise in steps 1–2 beat latent variety on diversity per adherence lost? | Matched diversity budget; DreamSim / layout variety vs prompt adherence | better diversity–adherence frontier |
| C7 | Do early-only regional masks give reliable layout on Krea 2? | Two- and three-region prompts; masks in draft sigmas only | layout follows masks; no seams; LoRA isolation measured separately |

## Sources

Spatial sampling:
- [RAS (2502.10389)](https://arxiv.org/abs/2502.10389)
- [SDiT (2601.12283)](https://arxiv.org/abs/2601.12283)
- [AsynDM (2510.04504)](https://arxiv.org/abs/2510.04504)
- [SANI (2609.18466)](https://arxiv.org/abs/2609.18466)
- [FreeEnhance (2409.07451)](https://arxiv.org/abs/2409.07451)
- [SAMG (2604.26503)](https://arxiv.org/abs/2604.26503)
- [RALU (2507.08422)](https://arxiv.org/abs/2507.08422)
- [ToCa (2410.05317)](https://arxiv.org/abs/2410.05317)
- [TaylorSeer (2503.06923)](https://arxiv.org/abs/2503.06923)
- [Patch Forcing (2604.19141)](https://arxiv.org/abs/2604.19141)
- [AsyncPatch (2606.07079)](https://arxiv.org/abs/2606.07079)
- [Differential Diffusion (2306.00950)](https://arxiv.org/abs/2306.00950)
- [S-CFG (2404.05384)](https://arxiv.org/abs/2404.05384)

Detail, artifacts, diversity:
- [FDG (2506.19713)](https://arxiv.org/abs/2506.19713)
- [FreSca (2504.02154)](https://arxiv.org/abs/2504.02154)
- [LF-CFG (2506.21452)](https://arxiv.org/abs/2506.21452)
- [Momentum Guidance (2602.20360)](https://arxiv.org/html/2602.20360v2)
- [NAG (2505.21179)](https://arxiv.org/abs/2505.21179)
- [ComfyUI-NAG](https://github.com/ChenDarYen/ComfyUI-NAG)
- [VSF (2508.10931)](https://arxiv.org/html/2508.10931v1)
- [CADS (2310.17347)](https://arxiv.org/abs/2310.17347)
- [SeedVarianceEnhancer](https://github.com/ChangeTheConstants/SeedVarianceEnhancer)
- [STRIDE (2605.11494)](https://arxiv.org/pdf/2605.11494)
- [Z-Sampling (2412.10891)](https://arxiv.org/abs/2412.10891v2)
- [Noise selection (2407.14041)](https://arxiv.org/abs/2407.14041v1)
- [FreqPrior (2502.03496)](https://arxiv.org/pdf/2502.03496)
- [Time-Shift Sampler (2305.15583)](https://arxiv.org/pdf/2305.15583)
- [Diversity Distillation (2503.10637)](https://arxiv.org/html/2503.10637v4)
- [Krea 2 SDA diversity LoRA](https://comfyui-wiki.com/en/news/2026-09-07-krea2-sda-diversity-lora)
- [ZImageTurbo FlowSampler](https://github.com/capitan01R/ComfyUI-ZImageTurbo-FlowSampler)

Identity:
- [PuLID for FLUX docs](https://github.com/ToTheBeginning/PuLID/blob/main/docs/pulid_for_flux.md)
- [T-LoRA (2507.05964)](https://arxiv.org/html/2507.05964v2)
- [Personalize Anything (2503.12590)](https://arxiv.org/abs/2503.12590)
- [WithAnyone (2510.14975)](https://arxiv.org/html/2510.14975v1)
- [FreeCure (2411.15277)](https://arxiv.org/html/2411.15277v3)
- [Krea 2 Identity Edit LoRA](https://comfyui-wiki.com/en/news/2026-07-07-krea2-identity-edit-lora)
- [CSDN: ComfyUI hook masking / scheduling](https://blog.csdn.net/2401_84760527/article/details/144446328)
- Zhihu captioning threads (snippets only): [684280253](https://zhuanlan.zhihu.com/p/684280253),
  [5449098829](https://zhuanlan.zhihu.com/p/5449098829), [719168976](https://zhuanlan.zhihu.com/p/719168976)
- [Eastondev: training your own LoRA](https://eastondev.com/blog/zh/posts/ai/20260828-comfyui-train-your-own-lora/)

Layout:
- [Regional Prompting FLUX (2411.02395)](https://arxiv.org/html/2411.02395v1)
- [Regional Prompting FLUX repo](https://github.com/antonioo-c/Regional-Prompting-FLUX)
- [RAG-Diffusion (2411.06558)](https://arxiv.org/html/2411.06558v2)
- [3DIS-FLUX (2501.05131)](https://arxiv.org/html/2501.05131v1)
- [Stitch (2509.26644)](https://arxiv.org/abs/2509.26644)
