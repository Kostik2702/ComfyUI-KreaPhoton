# Where to look next: research source map

**Date:** 2026-10-16  
**Purpose:** a checklist of the sources not yet scanned, ranked by expected value for KreaPhoton.
It complements the [landscape note](krea2-turbo-landscape-2026-10.md) (community reports) and
the [Chinese-lab scan](cn-lab-scan-2026-10.md) (papers).  
**Status:** planning note. Nothing here is a finding.

## Principle

Prefer places where people **build and measure** over another layer of papers:

- code with issues;
- same-seed comparisons with workflows attached;
- ablation write-ups.

Use papers to explain mechanisms and use review venues to check them.

## Ranked sources

| Rank | Source | Why it matters for us | Agent-reachable? | Status |
|---|---|---|---|---|
| 1 | **Wan 2.1 / Wan video community** | Krea 2 uses the **Wan 2.1 VAE**. Their knowledge of its artifacts applies directly to us: grid patterns, dark-region blotches, gradient behavior. Sampling tricks (FreeInit, low-frequency noise control) often appear in video first. | yes (GitHub, Reddit, HF, papers) | next |
| 2 | **AI-generated image detection and IQA research** | Detector papers are a catalogue of what separates AI photos from real ones: frequency fingerprints, VAE grids, noise statistics, unnatural microtexture. They give targets for R2 and the "AI look", plus objective metrics for them. | yes (arXiv, OpenReview) | next |
| 3 | **ComfyUI GitHub ecosystem** | RES4LYF (largest experimental sampler / scheduler library, active issues), Detail Daemon, Z-Image / Qwen sampler forks. A code search for `krea2` shows who builds on this architecture. | yes | next |
| 4 | **Issues / PRs: `krea-ai/krea-2`, ComfyUI core, diffusers** | Same-seed comparisons with workflows. Issue #15 already supplied a key counterexample on regional prompting. | yes | partial |
| 5 | **Japanese community**: note.com, Zenn, Qiita, JP SD wiki, JP X/Twitter | Deep LoRA-training culture (kohya / sd-scripts is Japanese); careful parameter ablations. | yes | open |
| 6 | **Korean community**: arca.live AI channel | Very active in LoRA and merges. Mostly anime, but training tricks transfer. | partly (may block) | open |
| 7 | **Big-lab technical reports**: Seedream (ByteDance), HunyuanImage (Tencent), Qwen-Image, Krea | They show how photorealism is achieved *in training* (aesthetic SFT, RLHF, data curation). This tells us what is already baked into the model and what a sampler cannot add. | yes | open |
| 8 | **Retouching / photography practice** | Frequency separation in retouching is the same idea as the high-pass detail delta (C1). Grain, lens, and skin-texture conventions define what reads as a real photo. | yes | open |
| 9 | **Verification venues**: OpenReview reviews, Hugging Face Papers, alphaXiv | Reviewers often find weaknesses better than the authors do. Use them to check scan numbers; for example, SDiT's figures are internally inconsistent. | yes | use per claim |
| 10 | **Civitai / LiblibAI / 吐司 (tusi.cn) articles and model pages** | Trainer write-ups with ablations, and recommended samplers on top Krea 2 realism checkpoints. | yes | open |

## Closed sources (owner access only)

Much practical knowledge never leaves these places, and an agent cannot read them. If the
owner finds a relevant thread, paste the text or a screenshot for analysis.

- **Discord:**
  - Banodoco: core open-model research community;
  - lodestones' server: Kroma, i.e. hands-on Krea 2 fine-tuning experience;
  - ostris ai-toolkit: LoRA training;
  - Krea's own server;
  - OneTrainer, kohya / musubi, RES4LYF.
- **WeChat 公众号 and 小红书 (Xiaohongshu):** a large share of Chinese practical content lives
  there rather than on Zhihu, and it is barely indexed. Bilibili content is mostly video, not
  text.

## Proposed scan order

1. **Wan VAE artifacts + AI-image detection / IQA** (ranks 1–2): the most concrete inputs for R2
   and the "AI look", and for objective artifact metrics.
2. **GitHub code search for `krea2` + RES4LYF + krea-2 issues** (ranks 3–4).
3. **Japanese and Korean LoRA-training sources** (ranks 5–6), focused on likeness and on avoiding
   outfit / background lock-in. Cross-check against C2 / C3 in the Chinese-lab scan.
4. **Technical reports** (rank 7), read once to bound what sampler work can achieve.

Record every result with the README evidence labels, including null and negative findings.
