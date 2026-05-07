# HED-BERT Model Card

## Model details

- **Name:** HED-BERT
- **Type:** BERT-style masked-token self-supervised transformer for EEG
- **Architecture:** Tiny ViT-style encoder (4 layers, 192 hidden width, 6 heads, MLP ratio 4, GELU, pre-LayerNorm), 2,121,444 trainable parameters
- **Input:** Per-channel time-frequency tensor (64 channels × 6 frequencies × 10 time bins) for each 1.0 s event-locked window
- **Tokens:** 1 `[CLS]` aggregator + 8 event tokens (HED multi-hot, V=1124) + 120 TF patch tokens (Conv2d 2×2 over freq×time)
- **Objectives (joint):**
  - Masked time-frequency-patch reconstruction (MSE, mask ratio 0.15)
  - Masked HED-tag prediction (BCE, mask ratio 0.50, 80/10/10 BERT split)
  - HED-warmup curriculum: $\beta(t)$ ramp from 0 to 1 over $t_\text{warm}=50$ epochs
- **Pretraining data:** Healthy Brain Network EEG (HBN-EEG), OpenNeuro `ds005505`–`ds005516`, releases 1–11 (3,000+ pediatric subjects, 5–21 years, paired psychiatric phenotyping, naturalistic acquisition), non-movie task subset

## Intended use

- **Primary:** research artifact for cross-corpus EEG decoding research, foundation-model methodology research, and event-grounded SSL studies.
- **Reasonable:** within-corpus held-out-subject task probing, cross-dataset visual-cognition transfer (e.g., N170 face/car), per-HED-depth event-recovery probing.
- **Out of scope:** auditory paradigms (MMN), lateralized-spatial paradigms (N2pc), semantic-priming paradigms (N400) — see paper §7 Limitations. Magnitude-only Morlet inputs cannot resolve the phase-locked, low-amplitude differences that drive these paradigms.

## Evaluation summary

| Probe                                   | Result                                  |
|-----------------------------------------|-----------------------------------------|
| Within-HBN held-out-subject 5-class probe | 87.15 ± 0.67% balanced accuracy (3 seeds) |
| Cross-dataset N170 face-vs-car finetune | 79.04% balanced accuracy (3 seeds)      |
| BERT-analog event-recovery probe        | macro-AUC 0.68 over 1124 HED tags       |
| Shuffled-HED falsifier (matched recipe) | 67.83% (collapses to random-init level) |

See main paper §5 for full tables and supplementary §S1 for the six-arm ablation.

## Training data demographics

- HBN-EEG: 5–21 years age range, balanced gender, paired psychiatric phenotyping, naturalistic acquisition. Among public EEG corpora, the most demographically diverse currently available (cf. adult-clinical TUH or sleep-only Sleep-EDF).
- ERP-CORE (used only for held-out cross-dataset evaluation): 40 adults, 6 paradigms, NEMAR `nm000132`.

No new human-subjects data were collected for this release.

## Misuse caveats

EEG representations can be misused for non-consensual inference, surveillance, or affective targeting. We release HED-BERT under the source-corpus license terms (HBN-EEG and ERP-CORE research-only / CC-BY) and frame it as a research artifact. Specifically:

- **Do not** apply HED-BERT to EEG recordings collected without informed consent.
- **Do not** use HED-BERT representations in any deployment context that infers participant state, intent, or attribute without explicit informed consent and IRB-equivalent ethics review.
- **Do** acknowledge the source-corpus restrictions when redistributing weights or fine-tuned derivatives.

## Limitations

- Single-corpus pretraining (HBN-EEG only). Generalization to other pretraining corpora is future work.
- Single architecture family evaluated (Tiny ViT). 2 M-parameter architecture sweeps are future work.
- Magnitude-only Morlet log-power input cannot resolve phase-locked, low-amplitude ERP components (MMN/N2pc/N400). Complex-Morlet (real + imaginary) inputs are the natural follow-up.
- Cross-dataset transfer is scoped to N170, the visual-cognition paradigm matched to HBN's pretraining task population. Other ERP-CORE paradigms sit at chance under matched-mask LOSO finetune.

## Anonymity (review window)

This release is anonymous for NeurIPS double-blind review. The hosting org (`hed-bert`), admin handle (`hed-bert-dev`), and repository name reveal no author identity. Camera-ready will replace the anonymous URL with the public repository URL.
