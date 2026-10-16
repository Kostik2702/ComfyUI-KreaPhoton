# Character LoRA: identity and physique without outfit/background lock-in

**Research date:** 2026-10-15  
**Target:** character LoRA trained on Krea 2 RAW and used on Krea 2 Turbo; preserve face, hair/other intended physical traits, and body proportions while leaving clothes, pose, lighting, camera, and environment prompt-controllable.  
**Status:** desk research synthesis. Community recipes are creator reports, not controlled Krea experiments. No training was run for this note.

## Executive conclusion

There is no caption trick that can reliably disentangle a character from clothes and scene if
those attributes are perfectly correlated in the image set. The strongest common signal is
dataset design: make identity/physique the stable factor and deliberately vary outfit,
background, lighting, pose, crop, camera distance, and image source. Use captions to name
the varying attributes that should remain promptable, then select checkpoints by testing
new scenes and trigger-free prompts. Face similarity alone is not enough: body proportions
need visible body evidence and dedicated full-body validation.

The Krea-specific training convention is RAW training, Turbo inference. The exact dataset
size and whether to caption stable anatomy are not settled: credible Krea/community recipes
range from roughly 12–30 to 40–80 images and disagree about repeating a fixed physique
phrase. These are not genuinely universal contradictory laws; they reflect differing
training setups, goals, and dataset quality. The practical solution is a small caption
ablation and a held-out clothing/background test, not adopting a magic image count.

“Commercial high quality” needs careful wording. Krea publicly discloses focused data,
caption editing, variation, and iterative steps, but not its production trainer internals.
The large public Krea character-library creator describes a repeatable selection and
validation pipeline over many character LoRAs; this is unusually relevant creator evidence,
but not an audited commercial studio benchmark. Scientific personalization papers establish
the underlying identity/context-entanglement problem and propose prior preservation or
explicit disentanglement, but their objectives are not drop-in Krea2 settings.

## Evidence-weighted recommendations (12)

| # | Recommendation | Why it should help | Evidence / confidence |
|---|---|---|---|
| 1 | Train on Krea 2 RAW; validate and deploy on Turbo using its normal inference recipe. | LoRA training and distilled inference are separate distributions; this is the stated Krea2 workflow and prevents evaluating the product on the wrong checkpoint. | Krea/Hugging Face Diffusers documentation; high for the supported path. [Krea2 guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_krea2.md) |
| 2 | Build a balanced framing ladder: close face, head/shoulders, half-body, seated/three-quarter, and several clear full-body views. | Face-only crops identify the face, not reliably height, shoulder/hip width, limb length, or overall build. Full body lets the model see those proportions, while close crops preserve small identity details. | Multiple Krea2 trainer/creator recipes; moderate. Recommended counts vary. [Musubi companion](https://github.com/radiatingreverberations/musubi-tuner-docker/blob/main/docs/KREA2-BEST-PRACTICES.md), [creator guide](https://www.reddit.com/r/StableDiffusion/comments/1v2vsqm/almost_perfect_likeness_in_750_steps_krea_2_lokr/) |
| 3 | Vary wardrobe intentionally and balance it: several genuinely different outfit families, with no single jacket/dress dominating. | Anything invariant in nearly every image can become part of the trigger or the learned visual concept; captions cannot fully repair perfect image-level confounding. | Mechanistically plausible and repeated in Krea/community guidance; moderate. [Krea tips](https://www.krea.ai/blog/krea-2-lora-training), [Krea2 100+ LoRA creator](https://www.reddit.com/r/StableDiffusion/comments/1x0imbp/ive_trained_100_krea_2_character_loras_heres_what/) |
| 4 | Vary backgrounds, lighting, pose, expression, angle, camera distance, and source/session—not only clothes. | Breaks correlations between identity and incidental capture conditions; reduces memorized composition and makes the identity survive recontextualization. | First-party Krea tips + repeatable community workflows; moderate-high. [Krea tips](https://www.krea.ai/blog/krea-2-lora-training), [70-LoRA workflow](https://www.reddit.com/r/StableDiffusion/comments/1wp8pin/update_my_free_krea_2_character_lora_library_went/) |
| 5 | Oversample candidates, then curate for identity correctness and *coverage*, not merely prettiness. Remove duplicates, wrong-person lookalikes, bad crops, heavy retouching, and extreme occlusion. | A polished but narrow set can encode the same light/angle/background; a few mistaken identities actively corrupt identity. | Creator pipeline over a 70-model library; reports ~200 candidates per character plus manual review and identity filtering. Creator self-report, moderate. [Workflow](https://www.reddit.com/r/StableDiffusion/comments/1wp8pin/update_my_free_krea_2_character_lora_library_went/) |
| 6 | Give every primary caption the same unique trigger and correct class noun; keep trigger spelling/order stable. | Establishes one reliable identity handle and lets the pretrained model supply the generic category (“woman”, “man”, “person”) rather than baking it into the identity token. | Community trainers agree; medium. [Musubi guide](https://github.com/radiatingreverberations/musubi-tuner-docker/blob/main/docs/KREA2-BEST-PRACTICES.md), [AI Toolkit community guide](https://github.com/chengyansen-ai/krea2-lora-training) |
| 7 | Caption per-image variables that must remain controllable: clothing, background, pose, lighting, crop/view, expression, hairstyle if it changes, and visible props. | The trigger should carry identity; natural-language tokens should explain nuisance factors that differ across examples. Krea explicitly says to caption one-off details one does not want bound to the LoRA. | Krea first-party advice and Krea community workflows; moderate-high. [Krea training guide](https://www.krea.ai/blog/krea-2-lora-training) |
| 8 | Do not use raw auto-captions unchanged. Manually remove false identity guesses and ensure the caption consistently describes the actual variable attributes. | VLMs can hallucinate stable attributes, omit clothing/background, or vary identity wording; inaccurate text supervision can bind concepts to the wrong token or weaken prompt controls. | Krea says auto-captions are an editable starting point; creator pipeline manually reviews captions. High for editing, exact caption format still uncertain. [Krea](https://www.krea.ai/blog/krea-2-lora-training), [70-LoRA workflow](https://www.reddit.com/r/StableDiffusion/comments/1wp8pin/update_my_free_krea_2_character_lora_library_went/) |
| 9 | Treat “caption permanent physical traits” as an ablation, not dogma. Compare trigger-only identity captions against trigger + the same concise physique phrase in every caption. | Stable uncaptained attributes are naturally absorbed by the trigger, but Krea2 creator reports say body shape may drift even when the face transfers; repeating accurate anatomy in both train and generation prompts may anchor body while reducing trigger-only portability. | Direct disagreement between community recipes; low-medium until tested on target character. [Trigger-only advice](https://github.com/gary149/train-krea2-lora), [fixed-traits recipe](https://rkrra.com/guides/krea-2-lora-training/) |
| 10 | Save frequent checkpoints and choose the earliest one that is recognizable *and* still follows new prompts. | Later training often strengthens likeness while also increasing outfit/background leakage, prompt domination, waxy detail, or identity spread to other people. Final checkpoint is not automatically best. | Krea2 creator reports over 100 LoRAs and an independent Musubi practical guide both emphasize checkpoint selection; moderate. [Creator findings](https://www.reddit.com/r/StableDiffusion/comments/1x0imbp/ive-trained-100-krea-2-character-loras-heres-what/), [Musubi guide](https://github.com/radiatingreverberations/musubi-tuner-docker/blob/main/docs/KREA2-BEST-PRACTICES.md) |
| 11 | Validate on a fixed, held-out prompt pack: portrait, profile, half/full-body, new casual/formal clothes, unseen locations, different light, and a trigger-free generic-person control. | Training-style samples can hide memorization. The trigger-free prompt checks whether the LoRA altered the base class; outfit/background perturbations reveal unwanted binding; full-body prompts expose proportion drift. | Explicit Krea2 trainer checklist and multi-LoRA creator practice; moderate-high. [Musubi checklist](https://github.com/radiatingreverberations/musubi-tuner-docker/blob/main/docs/KREA2-BEST-PRACTICES.md), [100+ LoRA creator](https://www.reddit.com/r/StableDiffusion/comments/1x0imbp/ive-trained-100-krea-2-character-loras-heres-what/) |
| 12 | Test LoRA strength and multi-person scenes separately; do not assume region/bbox prompting isolates its effect. | A Krea2 issue reports that strength sufficient for likeness in portraits degraded identity separation of unrelated people in crowded prompts, and regional prompting did not fully isolate the effect. This may be a model/checkpoint-specific limit. | Detailed public issue with controlled same-seed comparisons, but single reporter and not independently replicated; medium-low, important risk. [Krea2 issue #15](https://github.com/krea-ai/krea-2/issues/15) |

### Optional regularization and deeper methods

DreamBooth's peer-reviewed work uses class-specific prior-preservation loss: generate generic
class examples from the base model and train on those alongside subject examples, to reduce
language drift and allow new scenes/poses/views. For a Krea2 LoRA, this suggests an
experiment with generic “a photo of a woman/man” prior examples or AI-Toolkit Differential
Output Preservation (DOP), but neither should be assumed equivalent to DreamBooth's loss.
Community best-practice docs disagree on DOP/blank-prompt preservation; test it as a
separate factor because excessive preservation can dilute the identity signal.

DisenBooth formalizes the core failure: subject identity and identity-irrelevant properties
such as pose/background become entangled, so the model either copies the context or loses
identity when context changes. It proposes separate identity and non-identity embeddings
with weak-denoising and contrastive objectives. This is an architectural inspiration for a
future identity/context split, not a one-click LoRA hyperparameter.

Commercially oriented Krea creator workflows show a useful production pattern: create a
large candidate pool, identity-filter it, manually curate, caption, train multiple saved
checkpoints, then test identity across held-out clothing, backgrounds, framing, pose, and
lighting before publishing. A separate NightTale project documents generating 20 images
per character (12 headshots + 8 full-body) for 87 game characters, but the repository
explicitly marks its Krea2 Turbo training config as untested; use its dataset-generation
and face/body balancing ideas, not its unverified training claims. [NightTale pipeline](https://github.com/bensblueprints/lora-training-pipeline)

## Caption strategy: two defensible modes

Use a common caption skeleton:

```text
<unique_trigger>, <class>, <view/framing>, <pose/expression>,
<clothing>, <background/location>, <lighting>, <camera/focus>
```

Do not repeat a long aesthetic/style paragraph. It can accidentally turn a character LoRA
into a bundled identity + editorial style + lighting adapter.

**Mode A — identity is the token (best first baseline).** Captions include trigger + class
and describe changing outfit/scene/capture conditions. Omit stable face descriptors; the
trigger learns the invariant identity. At generation, call the trigger and let the prompt
control clothing and environment. This follows many standard character-LoRA recipes.

**Mode B — explicit physique anchor (when body drifts).** Add the same concise, objective
body phrase to every training caption *and* the generation prompt: e.g. “tall, narrow
shoulders, long legs, lean build” only if true and visible. Avoid magnitude language such
as “extremely,” “huge,” or several overlapping adjectives, which creator tests report can
exaggerate proportions. Keep face identity in the trigger; use the stable phrase for body
shape consistency. Compare to Mode A on held-out full-body prompts.

For any variable property: if it occurs in only some images and should be promptable, caption
it when present (“wearing a red jacket”, “indoors in a kitchen”). If it is constant in nearly
all images, first fix the dataset correlation by adding counterexamples; describing a
constant feature may not be enough to unlearn it. Avoid captions that contradict the pixels.

## Minimal high-quality dataset plan

For a first controlled Krea2 character run, a sensible *starting range* is 20–40 clean,
identity-certain images, not a fixed law:

- 6–10 clear face/head-and-shoulders images across front, three-quarter, and profile;
- 6–10 half-body or seated views with different poses and camera distances;
- 4–8 full-body views where posture and proportions are actually visible;
- more only when they add a new view, expression, outfit, lighting, or scene rather than a
  near-duplicate.

Balance outfit/background categories so one look cannot dominate. Keep 3–5 source-disjoint
images aside for visual identity reference if the data allows. Avoid aggressive face
restoration, beauty filters, warped body edits, or low-resolution upscales that alter
defining geometry. Preserve high-resolution face details in close crops, but do not let
closeups crowd out physique examples.

If all available images show one outfit or one environment, the honest result is that the
LoRA may learn the outfit/environment as part of the concept. Captions and prompt tricks
can reduce but cannot fully solve that confounding; seek varied/licensed images or use a
separate outfit/style LoRA and compositional control at inference.

## Validation matrix and stopping rule

For every saved checkpoint and a small LoRA-weight sweep, use identical seeds/settings and
these prompt classes:

1. Face anchor: simple neutral portrait, trigger + class, no clothing or scene copied from
   training.
2. New outfit: same character, explicitly different clothing from all training examples.
3. New environment/light: e.g. outdoor overcast vs training studio/interior; change one
   factor at a time.
4. Physique: full-body neutral stance, then seated and dynamic pose; compare height,
   shoulder/hip proportions, limb length, build, and face retention at distance.
5. Viewpoint: profile, rear three-quarter, low/high camera angles.
6. Promptability control: same prompts without trigger; generic subject should remain
   close to base behavior.
7. Multi-person stress test: target plus one or more clearly distinct people; check that
   likeness does not spread to others or flatten their attributes.

Choose the earliest checkpoint/lowest weight that achieves stable identity and body shape
without outfit/background lock-in, prompt override, or trigger-free contamination. If no
checkpoint satisfies both likeness and control, improve dataset/caption balance before
simply training longer.

Suggested first ablation (same dataset, seed pack, RAW trainer config):

| Arm | Dataset/caption change | Diagnoses |
|---|---|---|
| A | Current set, cleaned captions | Baseline |
| B | Rebalanced outfits/backgrounds; same captions | How much is dataset correlation |
| C | A + per-image variable-attribute captions | Caption-based disentanglement |
| D | C + stable physique phrase in all captions and generation prompts | Body-identity anchoring |
| E | C + generic class-prior/DOP-style preservation, if trainer supports it | Trigger/class contamination vs identity dilution |

Train each arm only to a modest horizon, save periodic checkpoints, and compare held-out
prompt quality—not training loss alone. Change one factor per arm; do not simultaneously
change rank, optimizer, image count, captions, resolution, and regularization.

## Source log

- [Krea 2 LoRA training announcement and tips](https://www.krea.ai/blog/krea-2-lora-training) — first-party guidance: focused data, character angle/expression/light/background variety, editable auto-captions, describe one-off details not to be learned.
- [Hugging Face Diffusers Krea2 guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_krea2.md) — RAW is intended for training; apply RAW-trained LoRA to Turbo.
- [Musubi Krea2 best practices](https://github.com/radiatingreverberations/musubi-tuner-docker/blob/main/docs/KREA2-BEST-PRACTICES.md) — independently written curation, captioning, held-out validation, trigger-free leakage, and checkpoint-selection protocol.
- [Krea2 LoRA training community practices](https://github.com/chengyansen-ai/krea2-lora-training/blob/main/references/krea2-community-best-practices.md) — useful recipe aggregation; settings are community consensus, not official Krea defaults.
- [Krea2 character LoRA guide based on ai-toolkit creator runs](https://rkrra.com/guides/krea-2-lora-training/) — 40–80 creator dataset range, caption reasoning, fixed body-trait phrase strategy; explicitly creator-specific evidence.
- [Creator workflow scaling from 18 to 70 Krea2 character LoRAs](https://www.reddit.com/r/StableDiffusion/comments/1wp8pin/update_my_free_krea_2_character_lora_library_went/) — sourcing, identity filtering, manual curation, captions, checkpoint/strength and held-out generation testing.
- [Krea2 creator report after 100+ character LoRAs](https://www.reddit.com/r/StableDiffusion/comments/1x0imbp/ive-trained-100-krea-2-character-loras-heres-what/) — face/body drift and out-of-distribution validation; self-reported, but directly relevant experience.
- [DreamBooth paper, CVPR 2023](https://arxiv.org/abs/2208.12242) — subject identity binding and class-specific prior preservation; general diffusion evidence, not a Krea2 LoRA benchmark.
- [DisenBooth paper](https://arxiv.org/abs/2305.03374) — explicitly separates identity-relevant and irrelevant factors; architectural inspiration, not directly portable settings.
- [Krea2 issue: identity bleeding in multi-person scenes](https://github.com/krea-ai/krea-2/issues/15) — controlled single-reporter issue, limitation hypothesis and stress-test motivation.
- [NightTale 87-character pipeline](https://github.com/bensblueprints/lora-training-pipeline) — production-shaped dataset-generation pipeline; its Krea2 training config is explicitly marked untested, so only use the curation/composition lessons.

## Related project memory

Existing vault knowledge already contains Krea2 reference-conditioning findings and character
identity/body tests. Do not treat reference masking as a substitute for LoRA dataset
disentanglement: the prior Krea2 experiment found its reference mask path primarily changed
the crop/bounding box, not semantic removal of clothing or environment. See memory search
results `KREA2-PHOTOREAL-WF.md | 2026-08-06 reference masking` and `KREA2-NODES.md | identity/body without clothing`.
