""" the HED-objective ablation: three-way evaluation of a pretrained BertSSL checkpoint.

Runs three evaluations on non-movie HBN test subjects:
  (a) Masked TF reconstruction MSE
  (b) Masked HED macro-F1 + per-level F1 (uses the training mask protocol)
  (c) Classical task-classification probe: frozen encoder → CLS embedding
      → StandardScaler + LogisticRegression on 10-way HBN task classes.

Writes a single JSON with all three metrics per seed and an aggregated
summary.

Usage:
    uv run python -m neural_vocabulary.scripts.eval_within_hbn \
        --features-dir ${HBN_DATA_DIR}/tf_features_nonmovie \
        --checkpoints runs/gates/v10_gateD/D1/seed_42_best.pt \
        --output-json runs/gates/v10_gateD/D1/eval_summary.json
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
from torch.utils.data import DataLoader

from neural_vocabulary.data.hed_vectorizer import HEDVectorizer
from neural_vocabulary.data.masking import DualStreamMasker
from neural_vocabulary.data.ssd_dataset import TASK_TO_IDX
from neural_vocabulary.data.packed_ssl_dataset import (
    PackedSSLDataset,
    packed_ssl_collate,
)
from neural_vocabulary.evaluation.splits import held_out_subjects
from neural_vocabulary.models.bert_ssl import BertSSL
from neural_vocabulary.scripts.extract_tf_features import _task_from_stem
from neural_vocabulary.training.device_manager import DeviceManager

logger = logging.getLogger(__name__)


def _load_tag_to_idx(path: Path) -> tuple[dict[str, int], torch.Tensor | None]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "tag_to_idx" in data:
        tag_to_idx = data["tag_to_idx"]
        try:
            vec = HEDVectorizer(schema_version="8.3.0")
            vec._tag_to_idx = dict(tag_to_idx)
            vec._idx_to_tag = dict(data.get("idx_to_tag", {}))
            vec._tag_depths = dict(data.get("tag_depths", {}))
            init = vec.get_hierarchy_init_embeddings(embed_dim=192)
        except Exception as exc:  # noqa: BLE001
            # The eval path overwrites tag_embeddings from the checkpoint
            # state_dict (strict=True), so a None init is safe. Warn so a
            # broken vectorizer is noticed during debugging.
            logger.warning(
                "Could not build hierarchy-init tag embeddings from %s: %s. "
                "Using random init; checkpoint load will overwrite them.",
                path,
                exc,
            )
            init = None
        return tag_to_idx, init
    raise RuntimeError(f"Cannot read tag_to_idx from {path}.")


@torch.no_grad()
def _collect_cls_embeddings(
    model: BertSSL,
    loader: DataLoader[dict[str, torch.Tensor]],
    dm: DeviceManager,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cls_embeddings (N, D), task_indices (N,))."""
    model.eval()
    feats: list[np.ndarray] = []
    tasks: list[int] = []
    for batch in loader:
        tf = dm.to_device(batch["tf"])
        hed = dm.to_device(batch["hed"])
        with dm.get_amp_context():
            out = model(tf, hed, masker=None)
        feats.append(out["cls_embedding"].float().cpu().numpy())
        tasks.extend(batch["task_indices"].tolist())
    return np.concatenate(feats, axis=0), np.array(tasks, dtype=np.int64)


def _reconstruction_and_hed(
    model: BertSSL,
    loader: DataLoader[dict[str, torch.Tensor]],
    dm: DeviceManager,
    mask_ratio_tf: float,
    mask_ratio_evt: float,
    seed: int,
) -> dict[str, float]:
    """Compute masked recon MSE and masked-HED macro-F1 on the eval loader."""
    model.eval()
    masker = DualStreamMasker(
        mask_ratio_tf=mask_ratio_tf, mask_ratio_evt=mask_ratio_evt
    )
    gen = torch.Generator(device=dm.device)
    gen.manual_seed(seed)

    all_recon: list[float] = []
    all_hed_logits: list[torch.Tensor] = []
    all_hed_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            tf = dm.to_device(batch["tf"])
            hed = dm.to_device(batch["hed"])
            with dm.get_amp_context():
                out = model(tf, hed, masker=masker, generator=gen)
            rm = out["recon_mask"]
            if rm.any():
                diff = (out["recon_logits"] - out["recon_targets"]) ** 2
                per_tok = diff.mean(dim=-1)
                all_recon.append(float(per_tok[rm].float().mean().item()))
            hm = out["hed_mask"]
            if hm.any():
                all_hed_logits.append(out["hed_logits"][hm].float().cpu())
                all_hed_targets.append(out["hed_targets"][hm].float().cpu())

    recon_mse = float(np.mean(all_recon)) if all_recon else 0.0
    if all_hed_logits:
        logits = torch.cat(all_hed_logits, dim=0)
        targets = torch.cat(all_hed_targets, dim=0)
        preds = (torch.sigmoid(logits) >= 0.5).int().numpy()
        t = targets.int().numpy()
        macro_f1 = float(f1_score(t, preds, average="macro", zero_division=0))
    else:
        macro_f1 = 0.0
    return {"eval_recon_mse": recon_mse, "eval_hed_macro_f1": macro_f1}


class _PackedSSLWithTask(PackedSSLDataset):
    """Same as PackedSSLDataset but also emits the window's task index.

    We derive the task index from the filename once per index entry so the
    per-window cost is trivial.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._task_indices: list[int] = []
        for h5_path, _ in self._index:
            task_name = _task_from_stem(h5_path.stem)
            idx = TASK_TO_IDX.get(task_name, -1)
            self._task_indices.append(idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["task_index"] = torch.tensor(self._task_indices[idx], dtype=torch.long)
        return item


def _task_aware_collate(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    merged = packed_ssl_collate(
        [{k: v for k, v in b.items() if k != "task_index"} for b in batch]
    )
    merged["task_indices"] = torch.stack([b["task_index"] for b in batch], dim=0)
    return merged


def _logreg_probe(
    feats: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    seed: int,
) -> float:
    """Fit LR on frozen features → return balanced eval accuracy (percent).

    Balanced eval uses minority-class subsampling with a per-seed RNG.
    """
    scaler = StandardScaler()
    feats_scaled_train = scaler.fit_transform(feats[train_mask])
    feats_scaled_eval = scaler.transform(feats[eval_mask])
    y_train = labels[train_mask]
    y_eval = labels[eval_mask]

    if len(np.unique(y_train)) < 2:
        raise RuntimeError(
            f"Training split has only {len(np.unique(y_train))} class(es); "
            "task-probe cannot fit. Check the features directory covers "
            "multiple HBN tasks."
        )

    # sklearn >= 1.5 removed the multi_class kwarg; LogisticRegression handles
    # multinomial targets automatically (``solver="lbfgs"`` default).
    clf = LogisticRegression(max_iter=1000)
    clf.fit(feats_scaled_train, y_train)

    # Balanced eval subsample.
    classes, counts = np.unique(y_eval, return_counts=True)
    minority = int(counts.min())
    rng = np.random.default_rng(seed)
    balanced_idx: list[int] = []
    for cls in classes:
        cls_idx = np.where(y_eval == cls)[0]
        chosen = rng.choice(cls_idx, size=minority, replace=False)
        balanced_idx.extend(chosen.tolist())
    balanced_idx_arr = np.array(balanced_idx)

    preds = clf.predict(feats_scaled_eval[balanced_idx_arr])
    acc = float((preds == y_eval[balanced_idx_arr]).mean()) * 100.0
    return acc


def _evaluate_one_checkpoint(
    ckpt_path: Path,
    args: argparse.Namespace,
    dm: DeviceManager,
) -> dict[str, Any]:
    logger.info("=== Evaluating checkpoint %s ===", ckpt_path)

    tag_to_idx, tag_init = _load_tag_to_idx(args.vectorizer)
    vocab_size = max(tag_to_idx.values()) + 1

    # Rebuild model with the same architecture as pretraining.
    from typing import cast

    model = BertSSL(
        vocab_size=vocab_size,
        tag_init_embeddings=tag_init,
        epochs_per_window=args.epochs_per_window,
        n_freqs=args.n_freqs,
        n_channels=args.n_channels,
        n_time=args.n_time,
        patch_size=tuple(args.patch_size),
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        patch_mode=args.patch_mode,
    )
    # Checkpoints come in two layouts: raw ``state_dict`` (random_init
    # baseline) and wrapped ``{"model": state_dict, ...}`` (D.1.x pretrained
    # arms and pretraining_progression). Handle both so strict
    # loading succeeds in either case.
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        state = blob["model"]
    else:
        state = blob
    model.load_state_dict(state, strict=True)
    model = cast("BertSSL", dm.to_device(model))

    # Use a seed baked into the filename if available; fall back to a single
    # deterministic seed.
    seed = args.seed
    train_files, eval_files = held_out_subjects(
        args.features_dir, ratio=args.holdout_ratio, seed=seed
    )
    if args.limit is not None:
        train_files = train_files[: args.limit]
        eval_files = eval_files[: max(1, args.limit // 4)]

    # (a, b) Reconstruction + HED on eval set.
    eval_ds = PackedSSLDataset(
        eval_files,
        epochs_per_window=args.epochs_per_window,
        n_freqs=args.n_freqs,
        n_channels=args.n_channels,
        expected_n_time=args.n_time,
    )
    eval_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dm.device_type == "cuda"),
        collate_fn=packed_ssl_collate,
    )
    ab = _reconstruction_and_hed(
        model,
        eval_loader,
        dm,
        mask_ratio_tf=args.mask_ratio_tf,
        mask_ratio_evt=args.mask_ratio_evt,
        seed=seed,
    )

    # (c) Frozen encoder task-classification probe.
    probe_train_ds = _PackedSSLWithTask(
        train_files,
        epochs_per_window=args.epochs_per_window,
        n_freqs=args.n_freqs,
        n_channels=args.n_channels,
        expected_n_time=args.n_time,
    )
    probe_eval_ds = _PackedSSLWithTask(
        eval_files,
        epochs_per_window=args.epochs_per_window,
        n_freqs=args.n_freqs,
        n_channels=args.n_channels,
        expected_n_time=args.n_time,
    )
    train_loader = DataLoader(
        probe_train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dm.device_type == "cuda"),
        collate_fn=_task_aware_collate,
    )
    probe_eval_loader = DataLoader(
        probe_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dm.device_type == "cuda"),
        collate_fn=_task_aware_collate,
    )
    feats_train, tasks_train = _collect_cls_embeddings(model, train_loader, dm)
    feats_eval, tasks_eval = _collect_cls_embeddings(model, probe_eval_loader, dm)

    # Filter out unknown tasks (-1, e.g. RestingState if it slipped in).
    valid_train = tasks_train >= 0
    valid_eval = tasks_eval >= 0
    feats_all = np.concatenate(
        [feats_train[valid_train], feats_eval[valid_eval]], axis=0
    )
    tasks_all = np.concatenate(
        [tasks_train[valid_train], tasks_eval[valid_eval]], axis=0
    )
    train_mask = np.concatenate(
        [np.ones(valid_train.sum(), dtype=bool), np.zeros(valid_eval.sum(), dtype=bool)]
    )
    eval_mask = ~train_mask
    task_acc = _logreg_probe(feats_all, tasks_all, train_mask, eval_mask, seed=seed)

    return {
        "checkpoint": str(ckpt_path),
        "seed": seed,
        "eval_recon_mse": ab["eval_recon_mse"],
        "eval_hed_macro_f1": ab["eval_hed_macro_f1"],
        "task_probe_acc_pct": round(task_acc, 2),
        "n_train_windows": len(probe_train_ds),
        "n_eval_windows": len(probe_eval_ds),
        "n_freqs": args.n_freqs,
        "n_channels": args.n_channels,
        "n_time": args.n_time,
        "patch_size": list(args.patch_size),
        "patch_mode": args.patch_mode,
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=" the HED-objective ablation: three-way evaluation (recon + HED + task)"
    )
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="One or more .pt files.",
    )
    parser.add_argument(
        "--vectorizer",
        type=Path,
        default=Path("${HBN_DATA_DIR}/hed_vectorizer.pt"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mask-ratio-tf", type=float, default=0.15)
    parser.add_argument("--mask-ratio-evt", type=float, default=0.50)
    parser.add_argument("--epochs-per-window", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument(
        "--patch-mode",
        type=str,
        default="flat",
        choices=("flat", "channel_token"),
        help="ViT patch construction. Must match the checkpoint's "
        "training config; 'channel_token' embeds the channel axis as "
        "a separate token.",
    )
    parser.add_argument(
        "--n-freqs",
        type=int,
        default=6,
        help="Frequency-bin count per epoch in the input TF features. "
        "Default 6 matches extract_tf_features --task-filter non-movie. "
        "PackedSSLDataset raises on shape mismatch.",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=64,
        help="Channel count per epoch in the input TF features. Default "
        "64 matches HBN harmonization. PackedSSLDataset raises on shape "
        "mismatch.",
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=10,
        help="Time-bin count per epoch in the input TF features. Default "
        "10 matches the original 1.0 s @ 100 Hz decimated-to-10 Hz "
        "extraction. PackedSSLDataset raises on mismatch.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=[2, 2],
        metavar=("FREQ", "TIME"),
        help="ViT patch size (freq, time). Must divide n_freqs and "
        "n_time evenly; BertSSL raises on mismatch.",
    )
    args = parser.parse_args(argv)

    dm = DeviceManager(args.device)
    results: list[dict[str, Any]] = []
    for ckpt_str in args.checkpoints:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        results.append(_evaluate_one_checkpoint(ckpt_path, args, dm))

    summary = {"results": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", args.output_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
