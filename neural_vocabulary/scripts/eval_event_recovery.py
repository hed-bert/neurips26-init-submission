""" the HED-objective ablation: event-recovery probe (BERT-analog HED reconstruction).

Given a frozen BertSSL checkpoint, measures how well the encoder recovers
HED event semantics from masked EEG windows. This is literally the
pretraining objective measured cleanly on held-out data.

Two mask modes:
  - ``all_events``: mask 100% of event tokens (the full BERT-analog probe
    requested by issue #176). Event mask is deterministic, so the number of
    mask draws only varies TF masking (default 1 draw).
  - ``bert_standard``: the production masking (15% TF, 50% events, BERT
    80/10/10). Provided as a sanity-check; draws are genuinely random so
    averaging over N draws reduces variance.

Metrics (per checkpoint):
  - Per-tag AUC (macro-mean plus per-level L0..L7 breakdown) — robust to
    the extreme class imbalance of 1124 tags with 99%+ zeros.
  - Macro-F1 at threshold 0.5 — continuity with .
  - Top-k recall at k in {5, 10, 20} over the masked-event slots.
  - Level-aware macro-F1 bucketed by HED tag depth.

Usage (smoke):
    uv run python -m neural_vocabulary.scripts.eval_event_recovery \
        --features-dir ${HBN_DATA_DIR}/tf_features_nonmovie \
        --checkpoints /tmp/smoke_ckpt.pt \
        --output-json /tmp/event_recovery.json \
        --mask-mode all_events --limit 8

Usage (full):
    uv run python -m neural_vocabulary.scripts.eval_event_recovery \
        --features-dir ${HBN_DATA_DIR}/tf_features_nonmovie \
        --checkpoints \
            runs/gates/v10_gateD/random_init/seed_42_best.pt \
            runs/gates/v10_gateD/d1_0_baseline/seed_42_best.pt \
            runs/gates/v10_gateD/d1_1_recon_only/seed_42_best.pt \
            runs/gates/v10_gateD/d1_2_hed_only/seed_42_best.pt \
            runs/gates/v10_gateD/d1_3_no_hed_ctrl/seed_42_best.pt \
            runs/gates/v10_gateD/d1_4_hed_warmup/seed_42_best.pt \
        --output-json runs/gates/v10_gateD/event_recovery.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from neural_vocabulary.data.hed_vectorizer import HEDVectorizer
from neural_vocabulary.data.masking import DualStreamMasker
from neural_vocabulary.data.packed_ssl_dataset import (
    PackedSSLDataset,
    packed_ssl_collate,
)
from neural_vocabulary.evaluation.collapse_detector import (
    assert_no_class_mean_collapse,
)
from neural_vocabulary.evaluation.splits import held_out_subjects
from neural_vocabulary.models.bert_ssl import BertSSL
from neural_vocabulary.scripts.extract_tf_features import _task_from_stem
from neural_vocabulary.training.device_manager import DeviceManager

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

MASK_MODES: tuple[str, ...] = ("all_events", "bert_standard")
TOP_K_VALUES: tuple[int, ...] = (5, 10, 20)


# -----------------------------------------------------------------------------
# Vectorizer / checkpoint helpers
# -----------------------------------------------------------------------------


def load_tag_to_idx_and_depths(
    path: Path,
) -> tuple[dict[str, int], dict[str, int], torch.Tensor | None]:
    """Load tag vocab, depths, and hierarchy-init embeddings from the vectorizer.

    Mirrors the loader in ``eval_within_hbn.py`` but additionally returns the
    per-tag depth dict (needed for level-aware metric breakdown).
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "tag_to_idx" not in data:
        raise RuntimeError(f"Cannot read tag_to_idx from {path}.")
    tag_to_idx = dict(data["tag_to_idx"])
    tag_depths = dict(data.get("tag_depths", {}))
    try:
        vec = HEDVectorizer(schema_version="8.3.0")
        vec._tag_to_idx = dict(tag_to_idx)
        vec._idx_to_tag = dict(data.get("idx_to_tag", {}))
        vec._tag_depths = dict(tag_depths)
        init = vec.get_hierarchy_init_embeddings(embed_dim=192)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not build hierarchy-init tag embeddings from %s: %s. "
            "Using None; strict load from checkpoint overwrites regardless.",
            path,
            exc,
        )
        init = None
    return tag_to_idx, tag_depths, init


def load_checkpoint_state(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load a the HED-objective ablation checkpoint; handles raw state_dict and wrapped formats.

    The  the HED-objective ablation harness saves either ``torch.save(model.state_dict(), ...)``
    (random_init baseline) or ``torch.save({"model": model.state_dict(), ...})``
    (the five D.1.x pretrained arms). Both must load under ``strict=True``.
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        return cast("dict[str, torch.Tensor]", blob["model"])
    if isinstance(blob, dict) and all(
        isinstance(v, torch.Tensor) for v in blob.values()
    ):
        return cast("dict[str, torch.Tensor]", blob)
    raise RuntimeError(
        f"Unrecognized checkpoint format at {ckpt_path}. Expected raw "
        "state_dict or wrapped {'model': state_dict, ...}."
    )


# -----------------------------------------------------------------------------
# Mask-mode helpers
# -----------------------------------------------------------------------------


class AllEventsMasker(DualStreamMasker):
    """BERT-analog masker that forces 100% [MASK_EVT] on every event token.

    Overrides ``mask_events`` to guarantee a deterministic all-mask pattern
    (no 80/10/10 split, no random replacements, no unchanged pass-throughs).
    Inherits ``mask_tf`` from the base class so TF stream can be controlled
    independently (set ``mask_ratio_tf=0`` to keep TF fully visible, which is
    what a clean event-recovery probe wants).
    """

    def mask_events(
        self,
        hed: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hed.ndim != 3:
            raise ValueError(f"hed must be (B, E, V); got {hed.shape}")
        b, e, _v = hed.shape
        device = hed.device
        mask_indices = torch.ones(b, e, dtype=torch.bool, device=device)
        replace_with_mask_token = torch.ones(b, e, dtype=torch.bool, device=device)
        # The event-embedder only reads ``hed`` at positions NOT flagged in
        # ``replace_with_mask_token``; in all-events mode no position escapes
        # the mask token, so the caller's raw ``hed`` is never used as input.
        # We still pass it through unchanged so the interface matches.
        return hed, hed.detach(), mask_indices, replace_with_mask_token


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def _safe_per_tag_auc(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Per-tag ROC AUC. Returns (macro_mean, per_tag_auc_array).

    Tags with no positive or no negative example in the eval set are excluded
    from the macro mean (AUC is undefined there). NaNs fill the excluded
    slots so per-level breakdown can mask them out.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))  # (N, V)
    v = targets.shape[1]
    auc = np.full(v, np.nan, dtype=np.float64)
    pos_sum = targets.sum(axis=0)
    for t in range(v):
        p = pos_sum[t]
        if p <= 0 or p >= targets.shape[0]:
            continue
        try:
            auc[t] = roc_auc_score(targets[:, t], probs[:, t])
        except ValueError:
            # Extremely rare: single-class tag after per-level filter.
            continue
    valid = ~np.isnan(auc)
    macro = float(auc[valid].mean()) if valid.any() else 0.0
    return macro, auc


def _top_k_recall(
    logits: np.ndarray,
    targets: np.ndarray,
    k: int,
) -> float:
    """Per-row top-k recall, averaged across masked-event slots.

    For each masked event row, pick the top-k predicted tag indices by logit
    and compute |top_k ∩ true_positives| / |true_positives|. Rows with zero
    positive tags are skipped (a recall score on an empty set is undefined).
    """
    n, v = logits.shape
    if k >= v:
        k = v - 1
    # argpartition is O(V) vs argsort O(V log V); fine since V ≈ 1124.
    top_k_idx = np.argpartition(-logits, kth=k, axis=1)[:, :k]
    recalls: list[float] = []
    for i in range(n):
        pos_set = np.where(targets[i] > 0)[0]
        if pos_set.size == 0:
            continue
        hits = np.isin(top_k_idx[i], pos_set).sum()
        recalls.append(float(hits) / float(pos_set.size))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def _level_breakdown_auc(
    per_tag_auc: np.ndarray,
    tag_depths: dict[str, int],
    idx_to_tag: dict[int, str],
) -> dict[str, float]:
    """Macro-mean AUC bucketed by HED tag depth (L0 .. Lmax).

    Returns a dict ``{"L0": float, "L1": float, ...}``; missing buckets (no
    valid tags in that level) are omitted.
    """
    by_level: dict[int, list[float]] = {}
    for t, auc in enumerate(per_tag_auc):
        if np.isnan(auc):
            continue
        tag = idx_to_tag.get(t)
        if tag is None:
            continue
        depth = tag_depths.get(tag, -1)
        if depth < 0:
            continue
        by_level.setdefault(depth, []).append(float(auc))
    return {
        f"L{depth}": float(np.mean(aucs)) for depth, aucs in sorted(by_level.items())
    }


def _level_breakdown_f1(
    logits: np.ndarray,
    targets: np.ndarray,
    tag_depths: dict[str, int],
    idx_to_tag: dict[int, str],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Macro-F1 bucketed by HED tag depth.

    Uses a sigmoid + threshold 0.5 binarisation and computes macro-F1 over
    the tag subset at each depth level. Matches  continuity.
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    t = targets.astype(np.int32)
    by_level: dict[int, list[int]] = {}
    for idx in range(t.shape[1]):
        tag = idx_to_tag.get(idx)
        if tag is None:
            continue
        depth = tag_depths.get(tag, -1)
        if depth < 0:
            continue
        by_level.setdefault(depth, []).append(idx)
    out: dict[str, float] = {}
    for depth, cols in sorted(by_level.items()):
        cols_arr = np.array(cols, dtype=np.int64)
        t_sub = t[:, cols_arr]
        p_sub = preds[:, cols_arr]
        if t_sub.sum() == 0:
            # No positives at this level in the eval batch — F1 undefined.
            continue
        f1 = f1_score(t_sub, p_sub, average="macro", zero_division=0)
        out[f"L{depth}"] = float(f1)
    return out


# -----------------------------------------------------------------------------
# Core probe
# -----------------------------------------------------------------------------


@torch.no_grad()
def _collect_masked_predictions(
    model: BertSSL,
    loader: DataLoader[dict[str, torch.Tensor]],
    dm: DeviceManager,
    masker: DualStreamMasker,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward with the given masker, collect (logits, targets) at masked slots.

    Returns:
        (logits (N_masked, V) float32, targets (N_masked, V) float32).
    """
    model.eval()
    gen = torch.Generator(device=dm.device)
    gen.manual_seed(seed)
    logits_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for batch in loader:
        tf = dm.to_device(batch["tf"])
        hed = dm.to_device(batch["hed"])
        with dm.get_amp_context():
            out = model(tf, hed, masker=masker, generator=gen)
        mask = out["hed_mask"]
        if not mask.any():
            continue
        logits_chunks.append(out["hed_logits"][mask].float().cpu())
        target_chunks.append(out["hed_targets"][mask].float().cpu())
    if not logits_chunks:
        return (
            np.zeros((0, model.vocab_size), dtype=np.float32),
            np.zeros((0, model.vocab_size), dtype=np.float32),
        )
    return (
        torch.cat(logits_chunks, dim=0).numpy().astype(np.float32),
        torch.cat(target_chunks, dim=0).numpy().astype(np.float32),
    )


@torch.no_grad()
def _collect_evt_embeddings_with_task_ids(
    model: BertSSL,
    eval_ds: PackedSSLDataset,
    loader: DataLoader[dict[str, torch.Tensor]],
    dm: DeviceManager,
    masker: DualStreamMasker,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect per-window mean evt_embedding and task-id label.

    The collapse guard requires per-class discrete labels, but
    masked-slot HED prediction has no such grouping; this helper
    supplies task-id grouping over ``evt_embeddings`` (already in the
    BertSSL output dict) by deriving an integer task-id per window
    from the source h5 file. Encoders that map distinct tasks to the
    same mean embedding fail the downstream collapse check.

    Returns:
        ``(features (n_windows, d_model), task_ids (n_windows,))``.
        ``shuffle=False`` on the loader is required so the index→task
        lookup against ``eval_ds._index`` stays aligned.
    """
    model.eval()
    feats: list[np.ndarray] = []
    window_idx = 0
    for batch in loader:
        tf = dm.to_device(batch["tf"])
        hed = dm.to_device(batch["hed"])
        with dm.get_amp_context():
            out = model(tf, hed, masker=masker)
        # Per-window evt_embedding = mean over the E event tokens.
        evt = out["evt_embeddings"].float().mean(dim=1).cpu().numpy()
        feats.append(evt)
        window_idx += evt.shape[0]
    if not feats:
        return (
            np.zeros((0, model.d_model), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    features = np.concatenate(feats, axis=0).astype(np.float32)

    # Build task-ids in dataset order (loader has shuffle=False). Each
    # window's source h5 path is stored at ``_index[i][0]``; map task
    # name → integer label via first-seen ordering.
    task_to_id: dict[str, int] = {}
    task_ids = np.empty((len(eval_ds),), dtype=np.int64)
    for i in range(len(eval_ds)):
        h5_path, _ = eval_ds._index[i]  # noqa: SLF001 — index API documented
        task_name = _task_from_stem(Path(h5_path).stem)
        if task_name not in task_to_id:
            task_to_id[task_name] = len(task_to_id)
        task_ids[i] = task_to_id[task_name]
    if features.shape[0] != task_ids.shape[0]:
        raise RuntimeError(
            f"Feature/task-id length mismatch: {features.shape[0]} vs "
            f"{task_ids.shape[0]}. Loader must use shuffle=False."
        )
    return features, task_ids


def _build_masker(
    mode: str,
    mask_ratio_tf: float,
    mask_ratio_evt: float,
) -> DualStreamMasker:
    if mode == "all_events":
        # TF masking is under caller control (default 0.0 for a clean probe);
        # event stream is fully masked.
        return AllEventsMasker(
            mask_ratio_tf=mask_ratio_tf,
            mask_ratio_evt=1.0,
        )
    if mode == "bert_standard":
        return DualStreamMasker(
            mask_ratio_tf=mask_ratio_tf,
            mask_ratio_evt=mask_ratio_evt,
        )
    raise ValueError(f"Unknown mask-mode {mode!r}; expected one of {MASK_MODES}.")


def evaluate_checkpoint(
    ckpt_path: Path,
    args: argparse.Namespace,
    dm: DeviceManager,
    idx_to_tag: dict[int, str],
    tag_depths: dict[str, int],
    tag_init: torch.Tensor | None,
    vocab_size: int,
) -> dict[str, Any]:
    """Run the event-recovery probe for a single checkpoint."""
    logger.info("=== Event-recovery probe: %s ===", ckpt_path)

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

    _train_files, eval_files = held_out_subjects(
        args.features_dir, ratio=args.holdout_ratio, seed=args.seed
    )
    if args.limit is not None:
        eval_files = eval_files[: max(1, args.limit)]
    eval_ds = PackedSSLDataset(eval_files, epochs_per_window=args.epochs_per_window)
    eval_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(dm.device_type == "cuda"),
        collate_fn=packed_ssl_collate,
    )

    # N-draw averaging. In ``all_events`` mode the event mask is deterministic
    # so the only variance source is TF masking — one draw is sufficient when
    # ``mask_ratio_tf`` is 0.0. In ``bert_standard`` mode every draw varies.
    n_draws = args.n_mask_draws
    if args.mask_mode == "all_events" and args.mask_ratio_tf == 0.0:
        if n_draws > 1:
            logger.info(
                "all_events with mask_ratio_tf=0 is deterministic; "
                "collapsing n_mask_draws=%d to 1 to avoid wasted compute.",
                n_draws,
            )
        n_draws = 1

    masker = _build_masker(args.mask_mode, args.mask_ratio_tf, args.mask_ratio_evt)

    #  collapse guard. Group windows by HBN task-id (each h5 file
    # is one task), compute per-task mean evt_embedding, raise if any
    # pairwise class-mean cosine exceeds the threshold.
    evt_features, task_ids = _collect_evt_embeddings_with_task_ids(
        model, eval_ds, eval_loader, dm, masker
    )
    if len(np.unique(task_ids)) < 2:
        # Single-task eval is degenerate for the per-task collapse guard.
        # Use WARNING level so the skip surfaces under typical
        # gpu_queue / nohup pipelines (often run at WARNING).
        logger.warning(
            "Eval set has only one HBN task; skipping class-mean-cos "
            "collapse check (undefined with single class). Result JSON "
            "will record task_collapse_skipped='single_task_eval'."
        )
        collapse_report = None
    else:
        collapse_report = assert_no_class_mean_collapse(
            evt_features, task_ids, allow_collapse=args.allow_collapse
        )

    per_draw_records: list[dict[str, Any]] = []
    for draw_idx in range(n_draws):
        draw_seed = args.seed * 1000 + draw_idx
        logits, targets = _collect_masked_predictions(
            model, eval_loader, dm, masker, seed=draw_seed
        )
        if logits.shape[0] == 0:
            logger.warning("Draw %d: zero masked slots; skipping.", draw_idx)
            continue
        macro_auc, per_tag_auc = _safe_per_tag_auc(logits, targets)
        macro_f1 = float(
            f1_score(
                targets.astype(np.int32),
                (1.0 / (1.0 + np.exp(-logits)) >= 0.5).astype(np.int32),
                average="macro",
                zero_division=0,
            )
        )
        level_auc = _level_breakdown_auc(per_tag_auc, tag_depths, idx_to_tag)
        level_f1 = _level_breakdown_f1(logits, targets, tag_depths, idx_to_tag)
        top_k = {
            f"top{k}_recall": _top_k_recall(logits, targets, k) for k in TOP_K_VALUES
        }
        per_draw_records.append(
            {
                "draw_idx": draw_idx,
                "n_masked_slots": int(logits.shape[0]),
                "macro_auc": macro_auc,
                "macro_f1": macro_f1,
                "level_auc": level_auc,
                "level_f1": level_f1,
                **top_k,
            }
        )
    if not per_draw_records:
        raise RuntimeError(
            f"All {n_draws} draws produced zero masked slots for {ckpt_path}. "
            "Verify the eval dataset has at least one batch of non-empty events."
        )

    # Aggregated (mean ± std) over draws.
    def _mean_std(key: str) -> tuple[float, float]:
        vals = [float(r[key]) for r in per_draw_records]
        return float(np.mean(vals)), float(np.std(vals))

    macro_auc_mean, macro_auc_std = _mean_std("macro_auc")
    macro_f1_mean, macro_f1_std = _mean_std("macro_f1")
    top_k_agg = {
        f"top{k}_recall_mean": float(
            np.mean([r[f"top{k}_recall"] for r in per_draw_records])
        )
        for k in TOP_K_VALUES
    }
    top_k_agg.update(
        {
            f"top{k}_recall_std": float(
                np.std([r[f"top{k}_recall"] for r in per_draw_records])
            )
            for k in TOP_K_VALUES
        }
    )

    # Level breakdowns: average per-level values across draws (omitting NaNs).
    all_levels_auc: dict[str, list[float]] = {}
    all_levels_f1: dict[str, list[float]] = {}
    for r in per_draw_records:
        for lvl, val in r["level_auc"].items():
            all_levels_auc.setdefault(lvl, []).append(val)
        for lvl, val in r["level_f1"].items():
            all_levels_f1.setdefault(lvl, []).append(val)
    level_auc_mean = {lvl: float(np.mean(vs)) for lvl, vs in all_levels_auc.items()}
    level_f1_mean = {lvl: float(np.mean(vs)) for lvl, vs in all_levels_f1.items()}

    result: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "mask_mode": args.mask_mode,
        "mask_ratio_tf": args.mask_ratio_tf,
        "mask_ratio_evt": args.mask_ratio_evt,
        "n_mask_draws": n_draws,
        "n_eval_files": len(eval_files),
        "n_eval_windows": len(eval_ds),
        "macro_auc_mean": macro_auc_mean,
        "macro_auc_std": macro_auc_std,
        "macro_f1_mean": macro_f1_mean,
        "macro_f1_std": macro_f1_std,
        "level_auc_mean": level_auc_mean,
        "level_f1_mean": level_f1_mean,
        **top_k_agg,
        "per_draw": per_draw_records,
    }
    if collapse_report is not None:
        result["task_class_mean_cosine"] = collapse_report.max_cosine
        result["task_collapse_pair"] = [
            collapse_report.class_a,
            collapse_report.class_b,
        ]
        result["task_collapse_skipped"] = None
    else:
        result["task_class_mean_cosine"] = None
        result["task_collapse_pair"] = None
        result["task_collapse_skipped"] = "single_task_eval"
    return result


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=" the HED-objective ablation: event-recovery (BERT-analog HED) probe"
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
    parser.add_argument(
        "--allow-collapse",
        action="store_true",
        help="Allow runs to proceed past the per-task class-mean-cosine "
        "collapse guard . Default raises "
        "RepresentationCollapseError. Set for diagnostic runs.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mask-mode", type=str, default="all_events", choices=MASK_MODES
    )
    parser.add_argument(
        "--n-mask-draws",
        type=int,
        default=10,
        help="Number of mask-draw averages. Collapsed to 1 in all_events "
        "mode when mask_ratio_tf=0 (deterministic).",
    )
    parser.add_argument(
        "--mask-ratio-tf",
        type=float,
        default=0.0,
        help="TF mask ratio during the probe. Default 0.0: TF stream fully "
        "visible so the encoder must use context to recover masked events.",
    )
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
        help="Must match the patch_mode the checkpoint was trained with. "
        "Defaults to 'flat' for  D.1.x compatibility; pass "
        "'channel_token' to load  E2 topographic checkpoints.",
    )
    args = parser.parse_args(argv)

    dm = DeviceManager(args.device)
    tag_to_idx, tag_depths, tag_init = load_tag_to_idx_and_depths(args.vectorizer)
    vocab_size = max(tag_to_idx.values()) + 1
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}

    results: list[dict[str, Any]] = []
    for ckpt_str in args.checkpoints:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        results.append(
            evaluate_checkpoint(
                ckpt_path, args, dm, idx_to_tag, tag_depths, tag_init, vocab_size
            )
        )

    summary = {
        "mask_mode": args.mask_mode,
        "n_mask_draws": args.n_mask_draws,
        "mask_ratio_tf": args.mask_ratio_tf,
        "mask_ratio_evt": args.mask_ratio_evt,
        "seed": args.seed,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", args.output_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
