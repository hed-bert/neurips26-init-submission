"""E2: literature-calibrated probes on [EVT] embeddings.

Runs two linear probes on HBN shot-boundary epochs for each checkpoint:

1. Animate vs inanimate (Cichy et al. 2014 benchmark, target ~75% single-trial)
   - animate: HED tag contains Agent/* or Organism/{Animal,Human,Boy,...}
   - inanimate: Man-made-object/* or Natural-object/* (excluding Plant)

2. Indoor vs outdoor scene (Groen et al. 2017 benchmark, target ~70%)
   - indoor: Property/Environmental-property/Indoors
   - outdoor: Property/Environmental-property/Outdoors

For comparison, a RANDOM-INIT model (same arch, fresh weights) is evaluated
alongside trained checkpoints. If no checkpoint clears ~55% on animacy, the
pipeline is stripping event-specific signal upstream of [EVT].

Held-out subject split (seed=42, ratio=0.15) ensures cross-subject generalization.

Usage:
    uv run python -m neural_vocabulary.scripts.e2_literature_probes \\
        --checkpoint runs/gates/g3b_task_codes_small_s42/epoch_0099.pt \\
        --config neural_vocabulary/configs/gates/g3b_task_codes_small.yaml \\
        --data-root /mnt/local/HBN_L100/preprocessed_v7_denoised \\
        --vectorizer /mnt/local/HBN_L100/preprocessed_v7_denoised/hed_vectorizer.pt \\
        --output runs/gates/diagnostics/e2_g3b_s42.json \\
        --max-epochs 4000

For random-init baseline, pass --random-init (checkpoint path still required
for config matching but weights are not loaded).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from neural_vocabulary.configs import load_config
from neural_vocabulary.data.collate import BucketBatchSampler, EventEpochCollator
from neural_vocabulary.data.preprocessed_dataset import PreprocessedEEGDataset
from neural_vocabulary.evaluation.splits import held_out_subjects
from neural_vocabulary.models.joint_model import Eventformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_label_masks(tag_to_idx: dict[str, int]) -> dict[str, np.ndarray]:
    """Build boolean masks over the HED vocab for each semantic category.

    Returns masks of shape (vocab_size,) that index all leaf+ancestor tags
    covered by a category. Whenever any masked bit is set in a sample's
    hed_vector, the sample belongs to that category.
    """
    vocab_size = max(tag_to_idx.values()) + 1
    masks: dict[str, np.ndarray] = {}

    animate_prefixes = (
        "Agent",
        "Item/Biological-item/Organism/Animal",
        "Item/Biological-item/Organism/Human",
    )
    inanimate_prefixes = (
        "Item/Object/Man-made-object",
        "Item/Object/Natural-object",
    )
    # Plant is technically Organism but perceptually inanimate; exclude from
    # both groups to avoid ambiguity.

    m_animate = np.zeros(vocab_size, dtype=bool)
    m_inanimate = np.zeros(vocab_size, dtype=bool)
    for tag, idx in tag_to_idx.items():
        if any(tag.startswith(p) for p in animate_prefixes):
            m_animate[idx] = True
        elif any(tag.startswith(p) for p in inanimate_prefixes):
            m_inanimate[idx] = True
    masks["animate"] = m_animate
    masks["inanimate"] = m_inanimate

    indoor_idx = tag_to_idx.get("Property/Environmental-property/Indoors")
    outdoor_idx = tag_to_idx.get("Property/Environmental-property/Outdoors")
    m_indoor = np.zeros(vocab_size, dtype=bool)
    m_outdoor = np.zeros(vocab_size, dtype=bool)
    if indoor_idx is not None:
        m_indoor[indoor_idx] = True
    if outdoor_idx is not None:
        m_outdoor[outdoor_idx] = True
    masks["indoor"] = m_indoor
    masks["outdoor"] = m_outdoor

    logger.info(
        "Label masks: animate=%d bits, inanimate=%d bits, "
        "indoor=%d bits, outdoor=%d bits",
        m_animate.sum(),
        m_inanimate.sum(),
        m_indoor.sum(),
        m_outdoor.sum(),
    )
    # Any zero mask on a critical category means the HED vocabulary has shifted
    # and the probe will silently produce meaningless results. Fail loud.
    zero = [k for k, m in masks.items() if not m.any()]
    if zero:
        raise RuntimeError(
            f"Empty HED label mask(s): {zero}. Vocabulary does not match the "
            f"animate/inanimate/indoor/outdoor prefixes. Check tag_to_idx keys."
        )
    return masks


def _animacy_label(hed_vector: np.ndarray, masks: dict[str, np.ndarray]) -> int | None:
    """1=animate present, 0=inanimate-only, None=skip.

    Movie scenes frequently mix animate + inanimate content, so a strict
    "one or the other" rule skips most epochs. We relax: label=1 if any
    animate tag is set (presence), label=0 if only inanimate tags are set
    (no animate presence). This matches the Cichy-style animacy probe
    logic (is there an animate entity in this scene?) applied to multi-
    label movie annotations.
    """
    has_anim = bool((hed_vector * masks["animate"]).any())
    has_inanim = bool((hed_vector * masks["inanimate"]).any())
    if has_anim:
        return 1
    if has_inanim:
        return 0
    return None


def _scene_label(hed_vector: np.ndarray, masks: dict[str, np.ndarray]) -> int | None:
    """0=indoor, 1=outdoor, None=skip."""
    has_in = bool((hed_vector * masks["indoor"]).any())
    has_out = bool((hed_vector * masks["outdoor"]).any())
    if has_in and not has_out:
        return 0
    if has_out and not has_in:
        return 1
    return None


def _load_model(
    checkpoint_path: Path,
    config_path: Path,
    random_init: bool,
    device: torch.device,
) -> Eventformer:
    config = load_config(config_path)
    model = Eventformer(config).to(device)
    if not random_init:
        logger.info("Loading weights from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint/config mismatch for {checkpoint_path}: "
                f"missing={list(missing)[:5]}..., unexpected={list(unexpected)[:5]}..."
            )
    else:
        logger.info("Using RANDOM-INIT model (checkpoint ignored for weights)")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


PASSIVE_MOVIE_TASKS = (
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "FunwithFractals",
    "ThePresent",
)


def _extract_evt_with_labels(
    model: Eventformer,
    dataset: PreprocessedEEGDataset,
    file_stems: set[str],
    config,
    masks: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    max_epochs: int | None,
    passive_only: bool = False,
) -> dict[str, np.ndarray]:
    """Extract [EVT] embeddings and compute animacy + scene labels."""
    # Select indices that match the file set AND have HED vectors.
    valid = []
    for i, m in enumerate(dataset._epoch_index):
        stem = Path(m["h5_path"]).stem
        if stem not in file_stems:
            continue
        if m["length"] < config.total_stride:
            continue
        if passive_only:
            task = stem.split("_", 1)[1] if "_" in stem else ""
            if not any(task.startswith(p) for p in PASSIVE_MOVIE_TASKS):
                continue
        valid.append(i)
    logger.info("Candidate epochs: %d (before HED filter)", len(valid))
    if max_epochs is not None and len(valid) > max_epochs:
        rng = np.random.default_rng(42)
        valid = list(rng.choice(valid, size=max_epochs, replace=False))
        logger.info("Sub-sampled to %d epochs", len(valid))

    subset = torch.utils.data.Subset(dataset, valid)
    lengths = [dataset._epoch_index[i]["length"] for i in valid]
    collator = EventEpochCollator(max_length=config.max_seq_len)
    sampler = BucketBatchSampler(
        lengths, batch_size=batch_size, drop_last=False, shuffle=False
    )
    loader = torch.utils.data.DataLoader(
        subset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=2,
    )

    all_embs: list[np.ndarray] = []
    all_animacy: list[int] = []
    all_scene: list[int] = []
    all_has_animacy: list[bool] = []
    all_has_scene: list[bool] = []

    batches_seen = 0
    batches_skipped = 0
    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = model(batch_dev)
            evt = out["evt_embeddings"].cpu().numpy()  # (B, embed_dim)

            # Collator attaches 'hed_targets' only when at least one item in
            # the batch has a hed_vector. A batch where every epoch lacks HED
            # (legitimate — some recordings have epochs without annotations)
            # is skipped but counted.
            if "hed_targets" not in batch:
                batches_skipped += 1
                continue
            hv = batch["hed_targets"]
            hv = hv.cpu().numpy() if isinstance(hv, torch.Tensor) else np.asarray(hv)

            for b in range(evt.shape[0]):
                hed = hv[b]
                an = _animacy_label(hed, masks)
                sc = _scene_label(hed, masks)
                if an is None and sc is None:
                    continue
                all_embs.append(evt[b])
                all_has_animacy.append(an is not None)
                all_has_scene.append(sc is not None)
                all_animacy.append(-1 if an is None else an)
                all_scene.append(-1 if sc is None else sc)
            batches_seen += 1

    total_batches = batches_seen + batches_skipped
    if total_batches == 0:
        raise RuntimeError(
            "DataLoader produced zero batches. Check file_stems vs "
            "dataset._epoch_index and passive_only filter coverage."
        )
    if batches_seen == 0:
        raise RuntimeError(
            f"All {batches_skipped} batches lacked 'hed_targets'. "
            "Dataset recordings have no HED vectors for this split."
        )
    if batches_skipped > 0:
        logger.info(
            "Skipped %d/%d batches without hed_targets", batches_skipped, total_batches
        )
    # Subject stratification: held_out_subjects split at the FILE level already
    # ensures subject separation between train/eval, so we don't carry per-
    # sample subject IDs through the probe (the EventEpochCollator drops them).
    return {
        "embeddings": np.array(all_embs, dtype=np.float32),
        "animacy": np.array(all_animacy),
        "scene": np.array(all_scene),
        "has_animacy": np.array(all_has_animacy),
        "has_scene": np.array(all_has_scene),
    }


def _fit_probe(
    train_emb: np.ndarray,
    train_lbl: np.ndarray,
    eval_emb: np.ndarray,
    eval_lbl: np.ndarray,
    balance_train: bool = True,
) -> dict[str, Any]:
    """Fit StandardScaler + LogisticRegression; return accuracy + macro F1."""
    if len(train_emb) == 0 or len(eval_emb) == 0:
        return {
            "error": "empty_set",
            "n_train": len(train_emb),
            "n_eval": len(eval_emb),
        }

    # Balance train set: subsample majority class to minority class count.
    if balance_train:
        classes, counts = np.unique(train_lbl, return_counts=True)
        if len(classes) < 2:
            return {"error": "one_class_only", "classes": classes.tolist()}
        min_count = int(counts.min())
        rng = np.random.default_rng(42)
        selected = []
        for c in classes:
            idx = np.where(train_lbl == c)[0]
            selected.append(rng.choice(idx, size=min_count, replace=False))
        sel = np.concatenate(selected)
        train_emb = train_emb[sel]
        train_lbl = train_lbl[sel]

    # Balance eval too so accuracy/F1 aren't inflated by class imbalance.
    classes, counts = np.unique(eval_lbl, return_counts=True)
    if len(classes) >= 2:
        min_count = int(counts.min())
        rng = np.random.default_rng(99)
        selected = []
        for c in classes:
            idx = np.where(eval_lbl == c)[0]
            selected.append(rng.choice(idx, size=min_count, replace=False))
        sel = np.concatenate(selected)
        eval_emb = eval_emb[sel]
        eval_lbl = eval_lbl[sel]

    scaler = StandardScaler().fit(train_emb)
    train_s = scaler.transform(train_emb)
    eval_s = scaler.transform(eval_emb)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(train_s, train_lbl)
    pred = clf.predict(eval_s)
    acc = float((pred == eval_lbl).mean())
    f1 = float(f1_score(eval_lbl, pred, average="macro"))
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "n_train": int(len(train_lbl)),
        "n_eval": int(len(eval_lbl)),
        "train_class_counts": np.bincount(train_lbl).tolist(),
        "eval_class_counts": np.bincount(eval_lbl).tolist(),
    }


def run_e2(
    checkpoint_path: Path,
    config_path: Path,
    data_root: Path,
    vectorizer_path: Path,
    output_path: Path,
    random_init: bool,
    batch_size: int,
    max_epochs: int | None,
    eval_ratio: float,
    seed: int,
    passive_only: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)

    logger.info("Loading vectorizer from %s", vectorizer_path)
    vect = torch.load(vectorizer_path, map_location="cpu", weights_only=False)
    tag_to_idx = vect["tag_to_idx"]
    masks = _build_label_masks(tag_to_idx)

    model = _load_model(checkpoint_path, config_path, random_init, device)

    logger.info("Building held-out-subject split")
    train_files, eval_files = held_out_subjects(data_root, ratio=eval_ratio, seed=seed)
    train_stems = {f.stem for f in train_files}
    eval_stems = {f.stem for f in eval_files}

    dataset = PreprocessedEEGDataset(data_root, max_subjects=None)

    logger.info("Extracting TRAIN embeddings (passive_only=%s)", passive_only)
    train_out = _extract_evt_with_labels(
        model,
        dataset,
        train_stems,
        config,
        masks,
        device,
        batch_size,
        max_epochs,
        passive_only=passive_only,
    )
    logger.info("Extracting EVAL embeddings")
    eval_out = _extract_evt_with_labels(
        model,
        dataset,
        eval_stems,
        config,
        masks,
        device,
        batch_size,
        None if max_epochs is None else max(1000, max_epochs // 5),
        passive_only=passive_only,
    )

    summary: dict = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "random_init": bool(random_init),
        "n_train_emb": int(len(train_out["embeddings"])),
        "n_eval_emb": int(len(eval_out["embeddings"])),
        "probes": {},
    }

    # Animacy probe: subsample to samples with has_animacy.
    train_mask_a = train_out["has_animacy"]
    eval_mask_a = eval_out["has_animacy"]
    summary["probes"]["animacy"] = _fit_probe(
        train_out["embeddings"][train_mask_a],
        train_out["animacy"][train_mask_a],
        eval_out["embeddings"][eval_mask_a],
        eval_out["animacy"][eval_mask_a],
    )

    # Scene probe.
    train_mask_s = train_out["has_scene"]
    eval_mask_s = eval_out["has_scene"]
    summary["probes"]["scene"] = _fit_probe(
        train_out["embeddings"][train_mask_s],
        train_out["scene"][train_mask_s],
        eval_out["embeddings"][eval_mask_s],
        eval_out["scene"][eval_mask_s],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote E2 summary to %s", output_path)
    logger.info("Animacy: %s", summary["probes"]["animacy"])
    logger.info("Scene:   %s", summary["probes"]["scene"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/local/HBN_L100/preprocessed_v7_denoised"),
    )
    parser.add_argument(
        "--vectorizer",
        type=Path,
        default=Path("/mnt/local/HBN_L100/preprocessed_v7_denoised/hed_vectorizer.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=8000)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Include non-movie tasks (default: passive movies only)",
    )
    args = parser.parse_args()

    run_e2(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_root=args.data_root,
        vectorizer_path=args.vectorizer,
        output_path=args.output,
        random_init=args.random_init,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        passive_only=not args.all_tasks,
    )


if __name__ == "__main__":
    main()
