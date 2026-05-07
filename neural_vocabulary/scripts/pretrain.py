""" the HED-objective ablation: dual-stream masked SSL pretraining on non-movie Morlet TF.

Trains ``BertSSL`` on  the HED-objective ablation non-movie TF features (from
``extract_tf_features --task-filter non-movie``). Each training step sees one
batch of 8-epoch packed windows; the masker hides 15 % of TF patches and
50 % of event tokens, the model predicts both from context.

Mirrors Gate B/C pipeline structure (DeviceManager, AdamW + OneCycleLR, AMP
bf16, seeds, per-run JSON). Go/no-go at epoch 20; issue #173 triggers a full
100-epoch run if:

    1. L_recon decreasing (final < 80% of epoch 0)
    2. HED macro-F1 > 2× random-baseline macro-F1
    3. Task-classification probe >= 33.4% ( era)

Usage (smoke):
    uv run python -m neural_vocabulary.scripts.pretrain \
        --features-dir /tmp/d1_nonmovie_smoke \
        --output-dir /tmp/d1_smoke \
        --seeds 42 --epochs 1 --limit 8

Usage (full):
    uv run python -m neural_vocabulary.scripts.pretrain \
        --features-dir ${HBN_DATA_DIR}/tf_features_nonmovie \
        --output-dir runs/pretrain \
        --seeds 42,13,7 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_vocabulary.data.hed_vectorizer import HEDVectorizer
from neural_vocabulary.data.masking import DualStreamMasker
from neural_vocabulary.data.packed_ssl_dataset import (
    PackedSSLDataset,
    packed_ssl_collate,
)
from neural_vocabulary.evaluation.splits import held_out_subjects
from neural_vocabulary.losses.ssl_dual_loss import DualStreamSSLLoss
from neural_vocabulary.models.bert_ssl import BertSSL
from neural_vocabulary.training.device_manager import DeviceManager

logger = logging.getLogger(__name__)

# Go/no-go bar.
TASK_CLASSIFICATION_GO_NO_GO: float = 33.4
RECON_DECREASE_FRACTION: float = 0.80
HED_F1_MULTIPLIER: float = 2.0


# -----------------------------------------------------------------------------
# HED statistics
# -----------------------------------------------------------------------------


def _scan_pos_counts(h5_files: list[Path], vocab_size: int) -> tuple[torch.Tensor, int]:
    """Accumulate per-tag positive counts across training files.

    Feeds DualStreamSSLLoss pos_weight. Two on-disk schemas:
      * ``groups``: per-epoch HDF5 groups (``epoch_*``) with
        ``hed_vector`` shape ``(vocab_size,)``.
      * ``contiguous``: top-level ``/hed_vector`` of shape
        ``(n_epochs, vocab_size)``, paired with ``/log_power`` of shape
        ``(n_epochs, F, C, T)``.

    Files predating the ``output_schema`` attr are read as ``groups``.
    Unknown schema strings raise to mirror PackedSSLDataset's strictness.
    """
    pos = torch.zeros(vocab_size, dtype=torch.float64)
    total = 0
    for f in h5_files:
        with h5py.File(f, "r") as h:
            schema = str(h.attrs.get("output_schema", "groups"))
            if schema not in ("groups", "contiguous"):
                raise RuntimeError(
                    f"{f}: unknown output_schema={schema!r}; expected "
                    "'groups' or 'contiguous'."
                )
            if schema == "contiguous":
                if "hed_vector" not in h:
                    raise RuntimeError(
                        f"{f}: contiguous schema requires top-level /hed_vector."
                    )
                hv = h["hed_vector"][:]  # (n_epochs, vocab_size)
                if hv.ndim != 2 or hv.shape[1] != vocab_size:
                    raise RuntimeError(
                        f"{f} hed_vector shape {hv.shape} != "
                        f"(*, {vocab_size}); vectorizer mismatch."
                    )
                if hv.shape[0] == 0:
                    raise RuntimeError(
                        f"{f}: hed_vector has zero epochs; corrupt extraction."
                    )
                if not np.isfinite(hv).all():
                    raise RuntimeError(f"{f}: hed_vector contains NaN or Inf entries.")
                # Cross-check with log_power's epoch axis so a corrupted
                # writer that produced mismatched arrays can't pass.
                if "log_power" in h and h["log_power"].shape[0] != hv.shape[0]:
                    raise RuntimeError(
                        f"{f}: hed_vector n_epochs ({hv.shape[0]}) != "
                        f"log_power n_epochs ({h['log_power'].shape[0]})."
                    )
                pos += torch.from_numpy(hv.astype(np.float64).sum(axis=0))
                total += int(hv.shape[0])
            else:
                for key in h:
                    if not key.startswith("epoch_"):
                        continue
                    grp = h[key]
                    if "hed_vector" not in grp:
                        raise RuntimeError(
                            f"{f}/{key} has no hed_vector; re-extract with "
                            "extract_tf_features --task-filter non-movie."
                        )
                    hv = grp["hed_vector"][:]
                    if hv.shape != (vocab_size,):
                        raise RuntimeError(
                            f"{f}/{key} hed_vector shape {hv.shape} != "
                            f"({vocab_size},); vectorizer mismatch."
                        )
                    if not np.isfinite(hv).all():
                        raise RuntimeError(
                            f"{f}/{key}: hed_vector contains NaN or Inf entries."
                        )
                    pos += torch.from_numpy(hv.astype(np.float64))
                    total += 1
    if total == 0:
        raise RuntimeError(
            "_scan_pos_counts collected zero events. Features directory "
            "may be empty or the task filter may have produced 0 windows."
        )
    return pos, total


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------


def _hed_macro_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Macro-F1 across the tags that have >=1 positive in the eval set.

    Returns 0.0 when no targets are present (e.g. empty eval window), which
    is the conservative default; logged losses rule out silent empties.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int()
    t = targets.int()
    if t.numel() == 0:
        return 0.0
    # Per-tag F1 over the vocab.
    tp = (preds & t).sum(dim=0).float()
    fp = (preds & ~t).sum(dim=0).float()
    fn = (~preds & t).sum(dim=0).float()
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1_per_tag = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
    # Restrict to tags with positives.
    valid = (t.sum(dim=0) > 0).float()
    if valid.sum() == 0:
        return 0.0
    return float((f1_per_tag * valid).sum() / valid.sum())


@torch.no_grad()
def _eval_one_epoch(
    model: BertSSL,
    loader: DataLoader[dict[str, torch.Tensor]],
    loss_fn: DualStreamSSLLoss,
    masker: DualStreamMasker,
    dm: DeviceManager,
    seed: int,
) -> dict[str, float]:
    """Eval pass. Reuses the masking protocol (deterministic by seed)."""
    model.eval()
    generator = torch.Generator(device=dm.device)
    generator.manual_seed(seed)

    total_recon = 0.0
    total_hed = 0.0
    total = 0.0
    n_batches = 0
    all_hed_logits: list[torch.Tensor] = []
    all_hed_targets: list[torch.Tensor] = []

    for batch in loader:
        tf = dm.to_device(batch["tf"])
        hed = dm.to_device(batch["hed"])
        with dm.get_amp_context():
            out = model(tf, hed, masker=masker, generator=generator)
            losses = loss_fn(
                recon_logits=out["recon_logits"],
                recon_targets=out["recon_targets"],
                recon_mask=out["recon_mask"],
                hed_logits=out["hed_logits"],
                hed_targets=out["hed_targets"],
                hed_mask=out["hed_mask"],
            )
        total_recon += float(losses["recon"].item())
        total_hed += float(losses["hed"].item())
        total += float(losses["total"].item())
        n_batches += 1
        # Collect HED logits only on masked positions for macro-F1.
        mask = out["hed_mask"]
        if mask.any():
            all_hed_logits.append(out["hed_logits"][mask].float().cpu())
            all_hed_targets.append(out["hed_targets"][mask].float().cpu())

    if all_hed_logits:
        logits_cat = torch.cat(all_hed_logits, dim=0)
        targets_cat = torch.cat(all_hed_targets, dim=0)
        macro_f1 = _hed_macro_f1(logits_cat, targets_cat)
    else:
        macro_f1 = 0.0

    return {
        "eval_recon": total_recon / max(n_batches, 1),
        "eval_hed": total_hed / max(n_batches, 1),
        "eval_total": total / max(n_batches, 1),
        "eval_hed_macro_f1": macro_f1,
    }


# -----------------------------------------------------------------------------
# Go/no-go logic
# -----------------------------------------------------------------------------


def go_no_go_verdict(
    recon_epoch0: float,
    recon_epoch20: float,
    hed_macro_f1_epoch20: float,
    random_hed_macro_f1: float,
    task_probe_acc: float | None = None,
) -> dict[str, Any]:
    """Pre-registered decision at epoch 20.

    Returns a dict with ``verdict`` in {"PASS", "FAIL"} and a per-criterion
    breakdown. PASS requires ALL of the three criteria.
    """
    crit = {
        "recon_decreasing": recon_epoch20 < RECON_DECREASE_FRACTION * recon_epoch0,
        "hed_beats_random": hed_macro_f1_epoch20
        > HED_F1_MULTIPLIER * random_hed_macro_f1,
        "task_probe_meets_bar": (
            task_probe_acc is not None
            and task_probe_acc >= TASK_CLASSIFICATION_GO_NO_GO
        ),
    }
    all_pass = all(crit.values())
    return {
        "verdict": "PASS" if all_pass else "FAIL",
        "criteria": crit,
        "recon_epoch0": recon_epoch0,
        "recon_epoch20": recon_epoch20,
        "hed_macro_f1_epoch20": hed_macro_f1_epoch20,
        "random_hed_macro_f1": random_hed_macro_f1,
        "task_probe_acc": task_probe_acc,
    }


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def _load_vectorizer_or_dict(
    path: Path,
) -> tuple[dict[str, int], torch.Tensor | None]:
    """Load tag_to_idx and (if loadable) a HEDVectorizer for hierarchy init.

    Returns:
        tag_to_idx, tag_init_embeddings. The embeddings are ``None`` when
        loading a dict-only payload without enough HED tree structure to
        regenerate the init tensor.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(data, "tag_to_idx") and hasattr(data, "get_hierarchy_init_embeddings"):
        return data.tag_to_idx, data.get_hierarchy_init_embeddings(embed_dim=192)
    if isinstance(data, dict) and "tag_to_idx" in data:
        tag_to_idx = data["tag_to_idx"]
        # Reconstruct a HEDVectorizer shell from the dict so we can call
        # ``get_hierarchy_init_embeddings``. Avoids building a dependency on
        # the HED toolkit for the basic dict path; only loads schema if we
        # actually need the embeddings.
        try:
            vec = HEDVectorizer(schema_version="8.3.0")
            vec._tag_to_idx = dict(tag_to_idx)
            vec._idx_to_tag = dict(data.get("idx_to_tag", {}))
            vec._tag_depths = dict(data.get("tag_depths", {}))
            vec._tag_doc_freq = dict(data.get("tag_doc_freq", {}))
            vec._n_docs = int(data.get("n_docs", 0))
            tag_init = vec.get_hierarchy_init_embeddings(embed_dim=192)
            return tag_to_idx, tag_init
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to reconstruct HEDVectorizer for hierarchy init: %s. "
                "Falling back to random tag embeddings.",
                exc,
            )
            return tag_to_idx, None
    raise RuntimeError(
        f"Cannot extract tag_to_idx from {path}. Expected a HEDVectorizer "
        "object or a dict with 'tag_to_idx' key."
    )


def _run_seed(
    seed: int,
    args: argparse.Namespace,
    dm: DeviceManager,
) -> dict[str, Any]:
    """Run one seed of the HED-objective ablation pretraining and return the per-seed summary."""
    logger.info("=== seed=%d ===", seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_files, eval_files = held_out_subjects(
        args.features_dir, ratio=args.holdout_ratio, seed=seed
    )
    if args.limit is not None:
        train_files = train_files[: args.limit]
        eval_files = eval_files[: max(1, args.limit // 4)]

    logger.info(
        "Split: %d train files, %d eval files", len(train_files), len(eval_files)
    )

    train_ds = PackedSSLDataset(
        train_files,
        epochs_per_window=args.epochs_per_window,
        n_freqs=args.n_freqs,
        n_channels=args.n_channels,
        expected_n_time=args.n_time,
    )
    eval_ds = PackedSSLDataset(
        eval_files,
        epochs_per_window=args.epochs_per_window,
        n_freqs=args.n_freqs,
        n_channels=args.n_channels,
        expected_n_time=args.n_time,
    )
    logger.info("Dataset windows: train=%d, eval=%d", len(train_ds), len(eval_ds))

    # HED pos_weight: single pass over training files.
    tag_to_idx, tag_init = _load_vectorizer_or_dict(args.vectorizer)
    vocab_size = max(tag_to_idx.values()) + 1
    if vocab_size != train_ds.vocab_size:
        raise RuntimeError(
            f"Vectorizer vocab_size {vocab_size} != dataset vocab_size "
            f"{train_ds.vocab_size}. Re-extract features with the matching "
            "vectorizer."
        )

    pos_counts, total_counts = _scan_pos_counts(train_files, vocab_size)
    pos_weight = DualStreamSSLLoss.compute_pos_weight(pos_counts, total_counts)
    logger.info(
        "pos_weight stats: min=%.2f median=%.2f max=%.2f (total n_events=%d)",
        float(pos_weight.min()),
        float(pos_weight.median()),
        float(pos_weight.max()),
        total_counts,
    )

    # HED-shuffle control: the shuffled-HED control.
    hed_shuffle_perm: torch.Tensor | None = None
    if args.shuffle_hed:
        rng = np.random.default_rng(seed)
        hed_shuffle_perm = torch.tensor(rng.permutation(vocab_size), dtype=torch.long)
        logger.info("--shuffle-hed: permuting HED vectors per seed.")

    # Model.
    if tag_init is not None and tag_init.shape[1] != args.d_model:
        # Embed dim mismatch: rebuild from the on-disk vectorizer so the
        # real tag_depths map is preserved (the stub fallback was a shallow
        # slash-count approximation that broke hierarchy semantics).
        data = torch.load(args.vectorizer, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or "tag_depths" not in data:
            raise RuntimeError(
                f"Vectorizer at {args.vectorizer} lacks tag_depths; cannot "
                "rebuild hierarchy-aware tag embeddings at a non-192 d_model. "
                "Use --d-model 192 or regenerate the vectorizer."
            )
        vec = HEDVectorizer(schema_version="8.3.0")
        vec._tag_to_idx = dict(data["tag_to_idx"])
        vec._tag_depths = dict(data["tag_depths"])
        tag_init = vec.get_hierarchy_init_embeddings(embed_dim=args.d_model)

    from typing import cast

    model = cast(
        "BertSSL",
        dm.to_device(
            BertSSL(
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
        ),
    )
    logger.info("Model param_count = %d", model.param_count())

    loss_fn = cast(
        "DualStreamSSLLoss",
        dm.to_device(
            DualStreamSSLLoss(
                pos_weight=pos_weight.to(dm.device),
                alpha=args.alpha,
                beta=args.beta,
            )
        ),
    )
    masker = DualStreamMasker(
        mask_ratio_tf=args.mask_ratio_tf,
        mask_ratio_evt=args.mask_ratio_evt,
    )

    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(dm.device_type == "cuda"),
        collate_fn=packed_ssl_collate,
        drop_last=True,
    )
    eval_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dm.device_type == "cuda"),
        collate_fn=packed_ssl_collate,
    )

    if len(train_loader) == 0:
        raise RuntimeError(
            f"train_loader has 0 batches (len(train_ds)={len(train_ds)}, "
            f"batch_size={args.batch_size}, drop_last=True). Reduce "
            "--batch-size or remove --limit."
        )
    if args.grad_accum > len(train_loader):
        raise RuntimeError(
            f"--grad-accum={args.grad_accum} exceeds micro-batches/epoch "
            f"={len(train_loader)}; would silently produce a smaller "
            "effective batch than requested. Reduce --grad-accum or "
            "increase data."
        )

    optimizer = torch.optim.AdamW(
        model.build_param_groups(weight_decay=0.05),
        lr=args.lr,
    )
    # Effective optimizer steps drop the trailing partial accumulation
    # window: a partial window would step with a downscaled gradient and
    # advance the LR schedule by a full step, biasing the end of every
    # epoch. Floor-div leaves those `n_micro % grad_accum` micro-batches
    # unstepped (their gradients are zeroed at epoch end below).
    micro_batches_per_epoch = len(train_loader)
    steps_per_epoch = max(1, micro_batches_per_epoch // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    scaler: torch.amp.GradScaler | None = None
    if dm.device_type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    history: list[dict[str, Any]] = []
    best_eval_total = float("inf")
    ckpt_path = args.output_dir / f"seed_{seed}_best.pt"
    periodic_ckpts: list[str] = []

    # Random-init baseline HED macro-F1 for the go/no-go ratio.
    # Compute once against the eval set with the untrained model.
    random_eval = _eval_one_epoch(model, eval_loader, loss_fn, masker, dm, seed=seed)
    random_hed_f1 = random_eval["eval_hed_macro_f1"]
    logger.info(
        "Random-init baseline: HED macro-F1=%.4f, recon=%.4f",
        random_hed_f1,
        random_eval["eval_recon"],
    )

    t_seed = time.time()
    for epoch in range(args.epochs):
        # Apply HED shuffle by permuting the target vocabulary channel axis
        # inline inside the training loop. Keeps the model / masker unchanged
        # and isolates the HED-structure ablation to the label axis.
        if hed_shuffle_perm is not None:
            # Apply at the loss level by permuting hed_targets in model outputs.
            # For simplicity, permute the target tensor via a data collator hook.
            # Here we use a lightweight closure: wrap the masker call.
            # Implementation: the dataset returns ``hed`` as (B, E, V); we
            # permute the V axis BEFORE masking. This is handled inside the
            # train loop wrapper below to avoid rebuilding the dataset per
            # epoch.
            pass  # permutation applied inside the loop below.

        t0 = time.time()
        if dm.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(dm.device)
        # Train epoch (with optional HED shuffle and gradient accumulation).
        model.train()
        total_recon = 0.0
        total_hed = 0.0
        total_loss_val = 0.0
        n_batches = 0
        fetch_s = 0.0
        compute_s = 0.0
        beta_live = args.beta
        if args.hed_warmup_epochs > 0 and epoch < args.hed_warmup_epochs:
            beta_live = args.beta * (epoch + 1) / args.hed_warmup_epochs

        accum_idx = 0
        # Prime the data-fetch timer before the loop so we attribute the
        # first iterator advance to fetch, not compute.
        t_fetch = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            fetch_s += time.time() - t_fetch
            t_compute = time.time()
            tf = dm.to_device(batch["tf"])
            hed = dm.to_device(batch["hed"])
            if hed_shuffle_perm is not None:
                hed = hed.index_select(-1, hed_shuffle_perm.to(hed.device))
            with dm.get_amp_context():
                out = model(tf, hed, masker=masker)
                losses = loss_fn(
                    recon_logits=out["recon_logits"],
                    recon_targets=out["recon_targets"],
                    recon_mask=out["recon_mask"],
                    hed_logits=out["hed_logits"],
                    hed_targets=out["hed_targets"],
                    hed_mask=out["hed_mask"],
                )
                l_total = loss_fn.alpha * losses["recon"] + beta_live * losses["hed"]
                # Average the loss across the accumulation window so the
                # accumulated gradient equals the gradient of the mean
                # loss over `grad_accum` micro-batches.
                l_scaled = l_total / args.grad_accum

            if scaler is not None:
                scaler.scale(l_scaled).backward()
            else:
                l_scaled.backward()
            accum_idx += 1

            if accum_idx == args.grad_accum:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                dm.mark_step()
                optimizer.zero_grad(set_to_none=True)
                accum_idx = 0

            total_recon += float(losses["recon"].item())
            total_hed += float(losses["hed"].item())
            total_loss_val += float(l_total.item())
            n_batches += 1
            compute_s += time.time() - t_compute
            t_fetch = time.time()

        # Drop any trailing partial accumulation window: zero its
        # gradients so the next epoch's first window starts clean.
        if accum_idx != 0:
            optimizer.zero_grad(set_to_none=True)
            accum_idx = 0

        peak_gpu_mem_gb = (
            float(torch.cuda.max_memory_allocated(dm.device)) / 1e9
            if dm.device_type == "cuda"
            else 0.0
        )
        epoch_runtime_s = time.time() - t0
        denom = compute_s + fetch_s
        dataloader_util = compute_s / denom if denom > 0 else 0.0

        train_metrics = {
            "train_recon": total_recon / max(n_batches, 1),
            "train_hed": total_hed / max(n_batches, 1),
            "train_total": total_loss_val / max(n_batches, 1),
            "beta_live": beta_live,
            "wall_s": round(epoch_runtime_s, 2),
            "peak_gpu_mem_gb": round(peak_gpu_mem_gb, 3),
            "dataloader_util": round(dataloader_util, 4),
        }

        eval_metrics = _eval_one_epoch(
            model, eval_loader, loss_fn, masker, dm, seed=seed
        )
        elapsed = time.time() - t0
        entry = {"epoch": epoch, "elapsed_s": round(elapsed, 1)}
        entry.update({k: round(v, 6) for k, v in train_metrics.items()})
        entry.update({k: round(v, 6) for k, v in eval_metrics.items()})
        history.append(entry)
        logger.info(
            "seed=%d epoch %d/%d: train_total=%.4f train_recon=%.4f "
            "train_hed=%.4f eval_recon=%.4f eval_hed=%.4f eval_f1=%.4f "
            "wall=%.1fs peak_mem=%.2fGB dl_util=%.2f",
            seed,
            epoch + 1,
            args.epochs,
            train_metrics["train_total"],
            train_metrics["train_recon"],
            train_metrics["train_hed"],
            eval_metrics["eval_recon"],
            eval_metrics["eval_hed"],
            eval_metrics["eval_hed_macro_f1"],
            train_metrics["wall_s"],
            train_metrics["peak_gpu_mem_gb"],
            train_metrics["dataloader_util"],
        )

        if eval_metrics["eval_total"] < best_eval_total:
            best_eval_total = eval_metrics["eval_total"]
            torch.save(model.state_dict(), ckpt_path)
            logger.info(
                "  New best eval_total=%.4f, saved %s", best_eval_total, ckpt_path
            )

        # Periodic checkpoints for the pretraining-progression probe
        #. Emits epoch 0 (pre-training snapshot after the
        # first epoch trains — i.e. epoch index 0 = weights AFTER 1
        # epoch), then every ``checkpoint_every`` epochs, and always at
        # the final epoch. Pre-training snapshot at "epoch 0 unreal" is
        # handled by the caller (the progression orchestrator runs the
        # random-init probe separately using a fresh model).
        if args.checkpoint_every is not None and args.checkpoint_every > 0:
            is_multiple = (epoch + 1) % args.checkpoint_every == 0
            is_final = epoch == args.epochs - 1
            if is_multiple or is_final:
                periodic_path = args.output_dir / f"seed_{seed}_epoch_{epoch + 1}.pt"
                torch.save(model.state_dict(), periodic_path)
                periodic_ckpts.append(str(periodic_path))
                logger.info(
                    "  Periodic ckpt: saved %s (epoch %d)",
                    periodic_path,
                    epoch + 1,
                )

    total_elapsed = time.time() - t_seed
    logger.info("seed %d done: %.0fs total", seed, total_elapsed)

    epoch0_recon = history[0]["train_recon"] if history else 0.0
    epoch_last_recon = history[-1]["train_recon"] if history else 0.0
    epoch_last_f1 = history[-1]["eval_hed_macro_f1"] if history else 0.0
    verdict = go_no_go_verdict(
        recon_epoch0=epoch0_recon,
        recon_epoch20=epoch_last_recon,
        hed_macro_f1_epoch20=epoch_last_f1,
        random_hed_macro_f1=random_hed_f1,
        task_probe_acc=None,  # probe runs in eval_within_hbn.py
    )

    result = {
        "seed": seed,
        "param_count": model.param_count(),
        "device": dm.device_type,
        "alpha": args.alpha,
        "beta": args.beta,
        "mask_ratio_tf": args.mask_ratio_tf,
        "mask_ratio_evt": args.mask_ratio_evt,
        "shuffle_hed": args.shuffle_hed,
        "hed_warmup_epochs": args.hed_warmup_epochs,
        "epochs_per_window": args.epochs_per_window,
        "n_freqs": args.n_freqs,
        "n_channels": args.n_channels,
        "n_time": args.n_time,
        "patch_size": list(args.patch_size),
        "patch_mode": args.patch_mode,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "num_workers": args.num_workers,
        "random_hed_macro_f1": random_hed_f1,
        "best_eval_total": best_eval_total,
        "checkpoint": str(ckpt_path),
        "periodic_checkpoints": periodic_ckpts,
        "history": history,
        "go_no_go": verdict,
        "n_train_files": len(train_files),
        "n_eval_files": len(eval_files),
        "n_train_windows": len(train_ds),
        "n_eval_windows": len(eval_ds),
        "total_elapsed_s": round(total_elapsed, 1),
    }
    seed_json = args.output_dir / f"seed_{seed}.json"
    seed_json.write_text(json.dumps(result, indent=2))
    logger.info("Saved seed results: %s", seed_json)
    return result


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=" the HED-objective ablation: dual-stream masked SSL pretraining"
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("${HBN_DATA_DIR}/tf_features_nonmovie"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/pretrain"),
    )
    parser.add_argument(
        "--vectorizer",
        type=Path,
        default=Path("${HBN_DATA_DIR}/hed_vectorizer.pt"),
    )
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate gradients before "
        "optimizer.step(). Effective batch size = batch-size x grad-accum. "
        "Use to maintain effective batch when GPU memory caps per-step "
        "batch size. The trailing partial accumulation window each epoch "
        "is dropped; len(train_loader) must be >= grad_accum.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--mask-ratio-tf", type=float, default=0.15)
    parser.add_argument("--mask-ratio-evt", type=float, default=0.50)
    parser.add_argument(
        "--hed-warmup-epochs",
        type=int,
        default=0,
        help="Linearly ramp beta from 0 to its target over N epochs (D.1.4).",
    )
    parser.add_argument(
        "--shuffle-hed",
        action="store_true",
        help="D.1.3 control: permute the HED tag axis so the model predicts "
        "an un-grounded multi-hot of the same marginal density.",
    )
    parser.add_argument("--epochs-per-window", type=int, default=8)
    parser.add_argument(
        "--patch-mode",
        type=str,
        default="flat",
        choices=("flat", "channel_token"),
        help="TF patch-embed strategy. 'flat' (default) uses Conv2d over "
        "all channels at once; the first op flattens the EEG montage. "
        "'channel_token' embeds each channel separately to preserve "
        "lateralization and topography. Sequence length grows "
        "~n_channels x under channel_token; reduce --batch-size "
        "accordingly.",
    )
    parser.add_argument(
        "--n-freqs",
        type=int,
        default=6,
        help="Frequency-bin count per epoch in the input TF features. "
        "Default 6 matches extract_tf_features --task-filter non-movie. "
        "Override when input TF features use a different frequency grid.",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=64,
        help="Channel count per epoch in the input TF features. Default "
        "64 matches HBN harmonization. Override when input TF features "
        "have a different channel count (e.g. ERP-CORE 30ch).",
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=10,
        help="Time-bin count per epoch in the input TF features. Must "
        "match the h5 feature shape; PackedSSLDataset raises on mismatch.",
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
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Also save a checkpoint every N training epochs (and at the "
        "final epoch). Used by the pretraining-progression probe (issue "
        "#176). Default None disables periodic snapshots.",
    )
    args = parser.parse_args(argv)

    if args.grad_accum < 1:
        parser.error(
            f"--grad-accum must be >= 1, got {args.grad_accum}. "
            "Use --grad-accum 1 to disable accumulation."
        )
    if not args.features_dir.exists():
        raise FileNotFoundError(
            f"Features directory not found: {args.features_dir}. "
            "Run extract_tf_features.py --task-filter non-movie first."
        )
    if not args.vectorizer.exists():
        raise FileNotFoundError(f"Vectorizer not found: {args.vectorizer}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    dm = DeviceManager(args.device)
    logger.info("Device: %s", dm.device)

    all_results = []
    for seed in seeds:
        all_results.append(_run_seed(seed, args, dm))

    summary = {
        "seeds": seeds,
        "device": dm.device_type,
        "alpha": args.alpha,
        "beta": args.beta,
        "mask_ratio_tf": args.mask_ratio_tf,
        "mask_ratio_evt": args.mask_ratio_evt,
        "shuffle_hed": args.shuffle_hed,
        "hed_warmup_epochs": args.hed_warmup_epochs,
        "n_freqs": args.n_freqs,
        "n_channels": args.n_channels,
        "n_time": args.n_time,
        "patch_size": list(args.patch_size),
        "patch_mode": args.patch_mode,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "num_workers": args.num_workers,
        "per_seed": all_results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("the HED-objective ablation complete. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
