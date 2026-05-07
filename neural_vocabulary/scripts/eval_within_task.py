""" the HED-objective ablation: within-task condition probe.

Beyond 5-class task identification (saturated at 86-88%, , this
probe asks: does the frozen encoder distinguish CONDITIONS inside a task?

Phase 1 (this script): **seqLearning target-count**, binary:
  - label 0 = seqLearning6target (690 files)
  - label 1 = seqLearning8target (1198 files)

seqLearning target-count is the cleanest within-task binary — different
tasks, same paradigm class, same epoch length. The split is subject-level
(no leakage) across 3 seeds {42, 13, 7}, with balanced-eval sub-sampling
that matches the Gate B.1 / D.1 convention.

Frozen encoder → [CLS] → StandardScaler + LogisticRegression. Mean±std
accuracy over 3 seeds per checkpoint.

Usage (smoke):
    uv run python -m neural_vocabulary.scripts.eval_within_task \
        --features-dir ${HBN_DATA_DIR}/tf_features_nonmovie \
        --checkpoints /tmp/smoke_ckpt.pt \
        --output-json /tmp/within_task.json \
        --seeds 42 --limit 8

Usage (full):
    uv run python -m neural_vocabulary.scripts.eval_within_task \
        --features-dir ${HBN_DATA_DIR}/tf_features_nonmovie \
        --checkpoints \
            runs/gates/v10_gateD/random_init/seed_42_best.pt \
            runs/gates/v10_gateD/d1_0_baseline/seed_42_best.pt \
            runs/gates/v10_gateD/d1_1_recon_only/seed_42_best.pt \
            runs/gates/v10_gateD/d1_2_hed_only/seed_42_best.pt \
            runs/gates/v10_gateD/d1_3_no_hed_ctrl/seed_42_best.pt \
            runs/gates/v10_gateD/d1_4_hed_warmup/seed_42_best.pt \
        --output-json runs/gates/v10_gateD/within_task_seqlearning.json \
        --seeds 42,13,7
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from neural_vocabulary.data.packed_ssl_dataset import (
    PackedSSLDataset,
    packed_ssl_collate,
)
from neural_vocabulary.evaluation.collapse_detector import (
    assert_no_class_mean_collapse,
)
from neural_vocabulary.evaluation.splits import held_out_subjects
from neural_vocabulary.models.bert_ssl import BertSSL
from neural_vocabulary.scripts.eval_event_recovery import (
    load_checkpoint_state,
    load_tag_to_idx_and_depths,
)
from neural_vocabulary.scripts.extract_tf_features import _task_from_stem
from neural_vocabulary.training.device_manager import DeviceManager

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# seqLearning target-count label map. Keys match the tokens returned by
# ``_task_from_stem`` on  non-movie feature filenames.
SEQLEARNING_LABEL_MAP: dict[str, int] = {
    "seqLearning6target": 0,
    "seqLearning8target": 1,
}


# -----------------------------------------------------------------------------
# Condition-labelled variant of PackedSSLDataset
# -----------------------------------------------------------------------------


class _PackedSSLWithCondition(PackedSSLDataset):
    """Emits a condition-label scalar alongside the TF/HED window.

    The condition label is extracted from the source filename (H5 stem) via
    ``_task_from_stem`` and mapped through ``label_map``. Files whose task
    token is not in ``label_map`` are filtered out at index-build time so
    downstream code never has to handle a "none" label.
    """

    def __init__(
        self,
        h5_files: list[Path],
        label_map: dict[str, int],
        epochs_per_window: int = 8,
    ) -> None:
        # Pre-filter before calling super().__init__ to avoid scanning files
        # whose task token isn't in the label map. This matters because
        # PackedSSLDataset raises if the final index is empty.
        filtered: list[Path] = []
        skipped = 0
        for f in h5_files:
            token = _task_from_stem(f.stem)
            if token in label_map:
                filtered.append(f)
            else:
                skipped += 1
        if not filtered:
            raise RuntimeError(
                f"No files match label_map={sorted(label_map)}; skipped "
                f"{skipped}/{len(h5_files)} files. Check the features "
                "directory covers the targeted conditions."
            )
        super().__init__(filtered, epochs_per_window=epochs_per_window)
        self._labels: list[int] = []
        for h5_path, _ in self._index:
            token = _task_from_stem(h5_path.stem)
            self._labels.append(label_map[token])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["condition_label"] = torch.tensor(self._labels[idx], dtype=torch.long)
        return item


def _condition_aware_collate(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    merged = packed_ssl_collate(
        [{k: v for k, v in b.items() if k != "condition_label"} for b in batch]
    )
    merged["condition_labels"] = torch.stack(
        [b["condition_label"] for b in batch], dim=0
    )
    return merged


# -----------------------------------------------------------------------------
# Frozen-encoder probe
# -----------------------------------------------------------------------------


@torch.no_grad()
def _collect_cls_and_labels(
    model: BertSSL,
    loader: DataLoader[dict[str, torch.Tensor]],
    dm: DeviceManager,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect [CLS] embeddings and condition labels over the loader."""
    model.eval()
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for batch in loader:
        tf = dm.to_device(batch["tf"])
        hed = dm.to_device(batch["hed"])
        with dm.get_amp_context():
            out = model(tf, hed, masker=None)
        feats.append(out["cls_embedding"].float().cpu().numpy())
        labels.extend(batch["condition_labels"].tolist())
    if not feats:
        return (
            np.zeros((0, model.d_model), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.concatenate(feats, axis=0).astype(np.float32),
        np.array(labels, dtype=np.int64),
    )


def _balanced_eval_accuracy(
    clf: LogisticRegression,
    feats_eval: np.ndarray,
    labels_eval: np.ndarray,
    seed: int,
) -> float:
    """Balanced-eval accuracy: minority-class subsample then score.

    Uses ``np.random.SeedSequence(seed).spawn(2)[0]`` to match the
    Gate B.1 / D.1 convention and isolate the sub-sample RNG from the
    split RNG.
    """
    classes, counts = np.unique(labels_eval, return_counts=True)
    if classes.size < 2:
        raise RuntimeError(
            f"Eval split has only {classes.size} class(es) — balanced eval "
            "needs at least 2. Split with a different --seeds value."
        )
    minority = int(counts.min())
    ss = np.random.SeedSequence(seed).spawn(2)[0]
    rng = np.random.default_rng(ss)
    balanced_idx: list[int] = []
    for cls in classes:
        cls_idx = np.where(labels_eval == cls)[0]
        chosen = rng.choice(cls_idx, size=minority, replace=False)
        balanced_idx.extend(chosen.tolist())
    balanced_arr = np.array(balanced_idx)
    preds = clf.predict(feats_eval[balanced_arr])
    return float((preds == labels_eval[balanced_arr]).mean()) * 100.0


def _evaluate_one_seed(
    ckpt_path: Path,
    seed: int,
    args: argparse.Namespace,
    dm: DeviceManager,
    tag_init: torch.Tensor | None,
    vocab_size: int,
) -> dict[str, Any]:
    model = BertSSL(
        vocab_size=vocab_size,
        tag_init_embeddings=tag_init,
        epochs_per_window=args.epochs_per_window,
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        patch_mode=args.patch_mode,
    )
    state = load_checkpoint_state(ckpt_path)
    model.load_state_dict(state, strict=True)
    model = cast("BertSSL", dm.to_device(model))

    train_files, eval_files = held_out_subjects(
        args.features_dir, ratio=args.holdout_ratio, seed=seed
    )
    if args.limit is not None:
        train_files = train_files[: args.limit]
        eval_files = eval_files[: max(1, args.limit // 4)]

    # Build condition-aware datasets. ``_PackedSSLWithCondition`` filters out
    # files whose task token isn't in the label map, so an empty features dir
    # for the targeted conditions raises a clear error immediately.
    train_ds = _PackedSSLWithCondition(
        train_files,
        SEQLEARNING_LABEL_MAP,
        epochs_per_window=args.epochs_per_window,
    )
    eval_ds = _PackedSSLWithCondition(
        eval_files,
        SEQLEARNING_LABEL_MAP,
        epochs_per_window=args.epochs_per_window,
    )
    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dm.device_type == "cuda"),
        collate_fn=_condition_aware_collate,
    )
    eval_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dm.device_type == "cuda"),
        collate_fn=_condition_aware_collate,
    )
    feats_train, labels_train = _collect_cls_and_labels(model, train_loader, dm)
    feats_eval, labels_eval = _collect_cls_and_labels(model, eval_loader, dm)
    if len(np.unique(labels_train)) < 2:
        raise RuntimeError(
            "Train split has <2 classes; widen --holdout-ratio or include "
            "both seqLearning tasks in the features dir."
        )
    if len(np.unique(labels_eval)) < 2:
        raise RuntimeError(
            "Eval split has <2 condition classes; widen --holdout-ratio "
            "or check task assignment in the features dir."
        )

    #  collapse guard: surface (and by default refuse) frozen
    # encoders that produce indistinguishable [CLS] embeddings across
    # the within-task condition labels.
    eval_collapse = assert_no_class_mean_collapse(
        feats_eval,
        labels_eval,
        allow_collapse=args.allow_collapse,
    )

    scaler = StandardScaler()
    feats_train_s = scaler.fit_transform(feats_train)
    feats_eval_s = scaler.transform(feats_eval)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(feats_train_s, labels_train)
    acc_balanced = _balanced_eval_accuracy(clf, feats_eval_s, labels_eval, seed=seed)
    acc_raw = float((clf.predict(feats_eval_s) == labels_eval).mean()) * 100.0
    return {
        "seed": seed,
        "n_train_windows": int(len(train_ds)),
        "n_eval_windows": int(len(eval_ds)),
        "class_counts_train": {
            int(c): int(n)
            for c, n in zip(*np.unique(labels_train, return_counts=True), strict=False)
        },
        "class_counts_eval": {
            int(c): int(n)
            for c, n in zip(*np.unique(labels_eval, return_counts=True), strict=False)
        },
        "acc_balanced_pct": acc_balanced,
        "acc_raw_pct": acc_raw,
        "eval_class_mean_cosine": eval_collapse.max_cosine,
        "eval_collapse_pair": [
            eval_collapse.class_a,
            eval_collapse.class_b,
        ],
    }


def evaluate_checkpoint(
    ckpt_path: Path,
    args: argparse.Namespace,
    dm: DeviceManager,
    tag_init: torch.Tensor | None,
    vocab_size: int,
    seeds: list[int],
) -> dict[str, Any]:
    logger.info("=== Within-task probe: %s ===", ckpt_path)
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        per_seed.append(
            _evaluate_one_seed(ckpt_path, seed, args, dm, tag_init, vocab_size)
        )
    accs = [float(r["acc_balanced_pct"]) for r in per_seed]
    return {
        "checkpoint": str(ckpt_path),
        "target": "seqLearning_target_count",
        "label_map": SEQLEARNING_LABEL_MAP,
        "n_seeds": len(seeds),
        "acc_balanced_mean_pct": float(np.mean(accs)),
        "acc_balanced_std_pct": float(np.std(accs)),
        "per_seed": per_seed,
    }


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=" the HED-objective ablation: within-task condition probe (seqLearning)"
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--vectorizer",
        type=Path,
        default=Path("${HBN_DATA_DIR}/hed_vectorizer.pt"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,13,7",
        help="Comma-separated split seeds (Gate B.1/D.1 convention: 42,13,7).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--allow-collapse",
        action="store_true",
        help="Allow runs to proceed past the class-mean-cosine collapse "
        "guard . Default raises RepresentationCollapseError.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs-per-window", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument(
        "--patch-mode",
        type=str,
        default="flat",
        choices=("flat", "channel_token"),
        help="Must match the checkpoint's training config. Default 'flat' "
        "matches  D.1.x; use 'channel_token' for  E2 checkpoints.",
    )
    args = parser.parse_args(argv)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds produced an empty list after parsing.")

    dm = DeviceManager(args.device)
    tag_to_idx, _tag_depths, tag_init = load_tag_to_idx_and_depths(args.vectorizer)
    vocab_size = max(tag_to_idx.values()) + 1

    results: list[dict[str, Any]] = []
    for ckpt_str in args.checkpoints:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        results.append(
            evaluate_checkpoint(ckpt_path, args, dm, tag_init, vocab_size, seeds)
        )

    summary = {
        "target": "seqLearning_target_count",
        "label_map": SEQLEARNING_LABEL_MAP,
        "seeds": seeds,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", args.output_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
