# Krea 2 Turbo research desk

This folder collects community observations, papers, code references, hypotheses, and
reproducible experiments about improving Krea 2 Turbo image quality in ComfyUI. It is
an evidence log, not a list of universal sampler recommendations.

## How to add a finding

For each source, record its date, exact claim, model/checkpoint, settings, sample size,
whether images and workflow are available, and what remains uncontrolled. Label the item
as one of:

- **Measured**: repeatable experiment with a stated protocol and available outputs.
- **Reported**: user observation or community benchmark without independent replication.
- **Paper result**: peer-reviewed/preprint result, with model and task domain stated.
- **Hypothesis**: a proposed mechanism that still needs an experiment here.

Never promote a setting to a default from one prompt, one seed, a visual anecdote, or an
aggregate metric alone. For comparisons, preserve the same prompt, seeds, checkpoint,
resolution, LoRAs, precision, and VAE; vary one factor at a time first. Use multiple
prompt strata and unseen holdout prompts, randomized blind pairwise review, and record
both quality and cost. Keep semantic variety, layout variety, and local texture variety
as separate outcomes. Record failed and neutral results too.

The initial scan is [Krea 2 Turbo landscape and experiment backlog](krea2-turbo-landscape-2026-10.md).

Character-specific identity and physique disentanglement is documented separately in
[Character LoRA: identity without outfit/background lock-in](krea2-character-lora-identity.md).

A training-free spatial variant of the detail envelope and gated eta, motivated by
BudgetPix, is proposed in
[Spatial detail allocation hypothesis](spatial-detail-allocation-hypothesis.md).

A scan of mostly Chinese-lab work on spatially adaptive sampling, training-free detail and
artifact control, identity preservation, and layout control, with ranked candidates and
experiments C1–C7, is in [Chinese-lab scan](cn-lab-scan-2026-10.md).
