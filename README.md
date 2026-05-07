# HED-BERT — anonymous code release for NeurIPS 2026 review

> **Title:** *Events Beat Scale: HED-Grounded SSL for EEG Foundation Models*
>
> Code, configs, and reproduction scripts for the anonymous double-blind review submission. Camera-ready will replace this anonymous repository with the de-anonymized public URL.

## What's in this repository

```
hed-bert-release/
├── pyproject.toml              # uv-managed project, Python 3.11+
├── uv.lock                     # locked dependencies for reproducibility
├── configs/
│   ├── pretrain.yaml           # headline HED-supervised pretraining recipe
│   └── pretrain_no_hed.yaml    # no-HED ablation (matched-recipe falsifier)
├── MODEL_CARD.md               # intended use, scope, misuse caveats
└── neural_vocabulary/
    ├── models/
    │   ├── bert_ssl.py         # HED-BERT (BERT-style masked-token SSL)
    │   ├── transformer.py      # encoder backbone (4-layer, 192-wide)
    │   ├── vit_tf.py           # ViT-style TF patch embedding (Conv2d 2x2)
    │   ├── joint_model.py      # joint TF + HED prediction head wrapper
    │   ├── channel_harmonization.py  # 129ch GSN ↔ 30ch BioSemi → 64ch target
    │   └── positional_encoding.py    # learned 1D position + 3-way token-type
    ├── data/
    │   ├── hbn_eeg.py          # HBN-EEG dataset loader
    │   ├── erp_core.py         # ERP-CORE dataset loader
    │   ├── hed_vectorizer.py   # HED tag → multi-hot V=1124 vector
    │   ├── hed_assembly.py     # HED tag assembly via hedtools
    │   ├── event_epocher.py    # event-locked window extraction
    │   ├── masking.py          # masked-token utilities
    │   ├── collate.py          # variable-length batch collation
    │   └── consolidated_dataset.py   # multi-recording packed dataset
    ├── losses/
    │   ├── hed_loss.py         # masked HED-tag BCE with hierarchical weighting
    │   └── ssl_dual_loss.py    # joint MSE + HED-BCE with HED-warmup curriculum
    ├── training/
    │   ├── trainer.py          # AdamW + OneCycleLR + bfloat16 trainer
    │   └── device_manager.py   # CUDA/MPS/CPU device abstraction
    ├── evaluation/
    │   ├── linear_probe.py     # held-out-subject 5-class probe
    │   ├── splits.py           # subject-disjoint splits (seed=42)
    │   └── metrics.py          # balanced accuracy, macro-AUC, F1
    ├── baselines/
    │   ├── bendr_adapter.py    # BENDR FM adapter
    │   ├── biot_adapter.py     # BIOT FM adapter
    │   ├── cbramod_adapter.py  # CBraMod FM adapter
    │   ├── labram_adapter.py   # LaBraM FM adapter
    │   └── reve_adapter.py     # REVE-base / REVE-large FM adapter
    └── scripts/
        ├── extract_tf_features.py  # raw EEG → Morlet TF features
        ├── pretrain.py             # HED-BERT pretraining loop
        ├── eval_within_hbn.py      # within-HBN 5-class probe (Table 1)
        ├── eval_n170_loso.py       # cross-dataset N170 finetune (Table 2)
        ├── eval_event_recovery.py  # BERT-analog event-recovery probe (Fig 3)
        ├── eval_within_task.py     # within-task seqLearning 6-vs-8 probe
        ├── erpcore_preprocess.py   # ERP-CORE → harmonized 64-channel
        ├── erpcore_extract_embeddings.py
        ├── baseline_bendr.py       # FM baseline runners
        ├── baseline_biot.py
        └── baseline_labram.py
```

## Environment

Python 3.11 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync          # installs all locked deps
uv run pytest    # smoke tests (no real-data tests in this release)
```

CUDA 12.x recommended for pretraining; the code falls back to CPU/MPS via `device_manager.py` for development.

## Data

Two corpora, both public:

| Corpus       | Source                                     | DOI                                     |
|--------------|--------------------------------------------|-----------------------------------------|
| HBN-EEG      | OpenNeuro `ds005505`–`ds005516` (releases 1–11) | `10.18112/openneuro.dsXXXXXX.v1.0.1`    |
| ERP-CORE     | NEMAR `nm000132`                           | `10.82901/nemar.nm000132`               |

### Required environment variables

```bash
export HBN_DATA_DIR=/path/to/your/hbn_data        # holds preprocessed/, hed_vectorizer.pt, tf_features_nonmovie/
export ERPCORE_DATA_DIR=/path/to/your/erpcore     # NEMAR nm000132 BIDS root
```

### Preprocessed time-frequency features (recommended fast path)

Reviewers can skip the raw → preprocessed → TF-features chain by downloading the precomputed feature tarball:

| Asset                                    | Size       | sha256                                                              |
|------------------------------------------|------------|---------------------------------------------------------------------|
| `tf_features_nonmovie.tar.zst`           | 13.25 GB   | `21ee904b30d090442b0a271a8c783507e92c225d5fe7fc95423fb5d4131012da`  |
| `hed_vectorizer.pt`                      | 94 KB      | `6536a02b796153a9a354a92f0e7f90686dddb7e2691107cb163ce464e59166b4`  |

Both assets are hosted on Google Drive under the anonymous review account `hedbert2026@gmail.com`:

```bash
# Browser flow (works without dependencies):
#   tarball:    https://drive.google.com/file/d/1AmKXbs6qQB8ZjJ4nXo-h-pwfENndznmL/view?usp=sharing
#   vectorizer: https://drive.google.com/file/d/1lFYpYS9zJTuaU3qMW5-qP6eDoA-XSCR9/view?usp=sharing

# CLI flow (uses gdown; see https://github.com/wkentaro/gdown):
mkdir -p "$HBN_DATA_DIR"
uv run --with gdown gdown 1AmKXbs6qQB8ZjJ4nXo-h-pwfENndznmL -O tf_features.tar.zst
zstd -d tf_features.tar.zst --stdout | tar -xC "$HBN_DATA_DIR"
mv "$HBN_DATA_DIR/v10_gate_d_features_nonmovie" "$HBN_DATA_DIR/tf_features_nonmovie"
uv run --with gdown gdown 1lFYpYS9zJTuaU3qMW5-qP6eDoA-XSCR9 -O "$HBN_DATA_DIR/hed_vectorizer.pt"

# Verify integrity:
sha256sum tf_features.tar.zst   # expect 21ee904b...
sha256sum "$HBN_DATA_DIR/hed_vectorizer.pt"   # expect 6536a02b...
```

The tarball contains **12,992 h5 files** (one per `subject_task_run`), each carrying per-epoch Morlet log-power tensors of shape `(F=6, C=64, T=10)` plus a multi-hot HED tag vector of length `V=1124`. Sampling rate 100 Hz; events are stim-locked (−0.2 to +0.8 s) or response-locked (−0.5 to +0.5 s) per the paper §4 spec. Decompressed size: 21.7 GB.

If you want to regenerate the features from raw (60+ GB raw HBN-EEG), use:

```bash
uv run python -m neural_vocabulary.scripts.extract_tf_features \
    --source-dir "$HBN_DATA_DIR/preprocessed" \
    --output-dir "$HBN_DATA_DIR/tf_features_nonmovie" \
    --task-filter non-movie
```

## Reproducing the headline numbers

### 1. Pretraining (Table 1, Fig 4)

Three seeds, 100 epochs each, single RTX 4090 ≈ 15 GPU-hours per seed:

```bash
for seed in 42 13 7; do
    uv run python -m neural_vocabulary.scripts.pretrain \
        --config configs/pretrain.yaml \
        --seed $seed \
        --output-dir runs/hed_bert_seed${seed}
done
```

### 2. Within-HBN 5-class probe (Table 1)

Frozen `[CLS]` + `LogisticRegression` on the held-out 15% subject split:

```bash
for seed in 42 13 7; do
    uv run python -m neural_vocabulary.scripts.eval_within_hbn \
        --checkpoint runs/hed_bert_seed${seed}/last.pt \
        --features-dir "$HBN_DATA_DIR/tf_features_nonmovie" \
        --report-csv runs/hed_bert_seed${seed}/within_hbn.csv
done
```

### 3. Cross-dataset N170 finetune (Table 2)

`N=20` leave-one-subject-out, 30-epoch end-to-end finetune on ERP-CORE N170:

```bash
for seed in 42 13 7; do
    uv run python -m neural_vocabulary.scripts.eval_n170_loso \
        --checkpoint runs/hed_bert_seed${seed}/last.pt \
        --erpcore-dir "$ERPCORE_DATA_DIR" \
        --report-csv runs/hed_bert_seed${seed}/n170.csv
done
```

### 4. Event-recovery probe (Fig 3, Table 3)

100% event-token mask + per-HED-depth macro-AUC bucketing:

```bash
uv run python -m neural_vocabulary.scripts.eval_event_recovery \
    --checkpoint runs/hed_bert_seed42/last.pt \
    --features-dir "$HBN_DATA_DIR/tf_features_nonmovie" \
    --vectorizer "$HBN_DATA_DIR/hed_vectorizer.pt" \
    --report-csv runs/hed_bert_seed42/event_recovery.csv
```

### 5. Shuffled-HED ablation (Table 3, supp Table S1)

Same as pretraining recipe with `shuffle_hed: true` set in the config — collapses the within-HBN probe to 67.83 % at the matched recipe.

### 6. FM baseline benchmark (Tables 1–2)

Each public EEG foundation model is evaluated under matched protocol via the corresponding adapter:

```bash
for fm in bendr biot labram cbramod reve_base reve_large; do
    uv run python -m neural_vocabulary.scripts.baseline_${fm} \
        --features-dir "$HBN_DATA_DIR/tf_features_nonmovie" \
        --erpcore-dir "$ERPCORE_DATA_DIR" \
        --n-train-subjects 500 \
        --finetune-epochs 10
done
```

(LaBraM, BIOT, CBraMod, BENDR adapters require their respective public checkpoints to be available; see each adapter's docstring.)

## Hardware & runtime

Reported numbers were produced on a single NVIDIA RTX 4090 (24 GB), with bfloat16 mixed-precision and effective batch size 32 (micro-batch 8 × grad-accumulation 4).

- Pretraining: ~15 GPU-hours per seed (3 seeds = ~45 GPU-hours).
- Within-HBN frozen probe: ~5 minutes.
- N170 cross-dataset finetune: ~30 minutes per LOSO cell × 20 cells ≈ 10 GPU-hours per seed.
- Event-recovery probe: ~10 minutes.

Total compute budget across all reported experiments: ~200 GPU-hours.

## Released artifacts

- This repository (code + configs + reproduction scripts).
- Pretrained checkpoints for the three headline seeds at the 100-epoch headline configuration (download URL listed alongside the TF-features tarball).
- Model card: see `MODEL_CARD.md`.

## License

Source code: research-only (matches the source-corpus licenses inherited from HBN-EEG and ERP-CORE).
Pretrained weights: same.

See `MODEL_CARD.md` for misuse caveats.

## Citation

(Anonymous for double-blind review. Citation block will be inserted on de-anonymization.)
