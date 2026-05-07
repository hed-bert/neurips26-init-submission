""" Phase D.2.2: per-paradigm few-shot transfer on ERP-CORE.

Runs ONE cell of the (paradigm, condition, n_train_subjects, encoder_seed,
train_seed, masker) sweep and writes one JSON. Designed for gpu_queue
submission so the full sweep fans out across slots.

Conditions:

- ``frozen``: Encoder frozen. For N > 0 a sklearn LogisticRegression is
  fit on the per-trial event-token embeddings of N held-in subjects and
  scored on the rest. With ``--n-train-subjects 0`` the script reports
  only eval-set subject and class counts (no classifier is fit); the
  canonical zero-shot AUC diagnostic lives in
  ``eval_event_recovery.py``.

- ``finetune``: Encoder loaded from a D.1.4 checkpoint, fully unfrozen,
  trained end-to-end with a fresh ``nn.Linear(d_model, n_classes)`` head
  on the per-position event-token outputs. CrossEntropy loss over the
  ``(B, E)`` per-position labels.

- ``scratch``: Random-init ``BertSSL`` (matched architecture) trained from
  scratch on the same N subjects. The decisive comparison at fixed N is
  scratch-vs-finetune: if they match, the pretrained encoder has no
  useful inductive bias for ERP-CORE.

``--masker`` controls how HED inputs are presented to the encoder.
``all_events`` (default) replaces every event input with ``[MASK_EVT]``,
forcing a TF-only test that closes the label-leakage path the pre-flight
smoke flagged on N170 (random-init scored 98% via unmasked-HED LogReg).
``none`` lets the encoder read the ERP-CORE ``hed_vector`` directly —
permissive control that quantifies pretraining lift over random-init when
both have full input access. The full sweep runs both masker modes.

``--mask-ratio-evt`` (default 1.0) controls the event-mask ratio when
``--masker all_events``. At 1.0 the existing ``AllEventsMasker`` is used
(deterministic 100% mask, preserves bit-identical behavior). At any value
< 1.0 the vanilla ``DualStreamMasker`` is used with ``mask_ratio_tf=0``
and ``mask_ratio_evt=<value>`` (matched-eval diagnostic). Setting 0.0 is
rejected — use ``--masker none`` instead.

``--readout {evt,cls,tf}`` (default ``evt``) selects which encoder output
is used as the feature vector for the probe/classifier head:
- ``evt``: ``out["evt_embeddings"]`` (B, E, D) flattened to (B*E, D).
- ``cls``: ``out["cls_embedding"]`` (B, D) — one vector per window.
- ``tf``:  mean-pool ``out["tf_embeddings"]`` (B, n_tf_tokens, D) to (B, D).

For ``cls`` and ``tf`` readouts the per-event label tensor is aggregated to
a per-window scalar by majority vote; a window with mixed labels raises a
clear error pointing at the packing logic in ErpcoreParadigmDataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as nnf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from neural_vocabulary.data.erpcore_label_rules import LABEL_RULES, LabelRule, get_rule
from neural_vocabulary.data.erpcore_paradigm_dataset import (
    EXPECTED_N_CHANNELS,
    EXPECTED_N_FREQS,
    ErpcoreParadigmDataset,
    paradigm_collate,
)
from neural_vocabulary.data.masking import DualStreamMasker
from neural_vocabulary.evaluation.collapse_detector import (
    assert_no_class_mean_collapse,
)
from neural_vocabulary.models.bert_ssl import BertSSL
from neural_vocabulary.scripts.eval_event_recovery import (
    AllEventsMasker,
    load_checkpoint_state,
    load_tag_to_idx_and_depths,
)
from neural_vocabulary.training.device_manager import DeviceManager

if TYPE_CHECKING:
    from collections.abc import Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)

CONDITIONS: tuple[str, ...] = ("frozen", "finetune", "scratch")
MASKER_MODES: tuple[str, ...] = ("all_events", "none")
READOUT_MODES: tuple[str, ...] = ("evt", "cls", "tf")
DEFAULT_PARADIGMS: tuple[str, ...] = ("N170", "MMN", "P3", "N2pc", "N400")
CLASS_WEIGHT_MODES: tuple[str, ...] = ("none", "balanced", "auto")

# Paradigms with documented >2:1 class imbalance (Day-3 neuroscience audit).
# ``auto`` resolves to ``balanced`` for these and ``none`` for the rest.
_IMBALANCED_PARADIGMS: frozenset[str] = frozenset({"MMN", "P3"})


def _resolve_class_weight(paradigm: str, mode: str) -> str:
    """Resolve ``auto`` class-weight mode to ``balanced`` or ``none``.

    - ``none`` / ``balanced``: pass through unchanged.
    - ``auto``: returns ``balanced`` for paradigms in ``_IMBALANCED_PARADIGMS``
      (MMN, P3 — documented >2:1 imbalance per Day-3 audit); ``none`` otherwise.
    """
    if mode not in CLASS_WEIGHT_MODES:
        raise ValueError(f"Unknown class_weight mode {mode!r}")
    if mode == "auto":
        return "balanced" if paradigm in _IMBALANCED_PARADIGMS else "none"
    return mode


# -----------------------------------------------------------------------------
# Data plumbing
# -----------------------------------------------------------------------------


def _list_subjects_for_paradigm(tf_dir: Path, paradigm: str) -> list[str]:
    """Return sorted subject IDs that have a TF file for ``paradigm``."""
    return sorted({p.stem.split("_", 1)[0] for p in tf_dir.glob(f"*_{paradigm}.h5")})


def _split_subjects(
    all_subjects: list[str],
    n_train: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Subject-level split: pick ``n_train`` for train, rest for eval.

    A fixed RNG draw determines membership so different ``train_seed``
    values yield disjoint train sets while the eval pool stays defined.
    Raises if ``n_train`` exceeds the number of available subjects minus
    one (always need ≥ 1 eval subject for cross-subject generalisation).
    """
    if n_train < 0:
        raise ValueError(f"n_train must be ≥ 0, got {n_train}")
    if n_train >= len(all_subjects):
        raise ValueError(
            f"n_train={n_train} but only {len(all_subjects)} subjects "
            "available; need ≥ 1 eval subject."
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_subjects))
    train = sorted(all_subjects[i] for i in perm[:n_train])
    eval_ = sorted(all_subjects[i] for i in perm[n_train:])
    return train, eval_


def _build_paradigm_dataset(
    tf_dir: Path,
    subjects: list[str],
    rule: LabelRule,
    n_time: int,
    n_freqs: int = EXPECTED_N_FREQS,
    n_channels: int = EXPECTED_N_CHANNELS,
) -> ErpcoreParadigmDataset:
    h5_files = []
    for s in subjects:
        p = tf_dir / f"{s}_{rule.paradigm}.h5"
        if p.exists():
            h5_files.append(p)
    if not h5_files:
        raise RuntimeError(
            f"No TF files for paradigm={rule.paradigm} across subjects={subjects}"
        )
    return ErpcoreParadigmDataset(
        h5_files,
        rule,
        expected_n_time=n_time,
        n_freqs=n_freqs,
        n_channels=n_channels,
    )


# -----------------------------------------------------------------------------
# Masker construction
# -----------------------------------------------------------------------------


def _build_masker(
    masker_mode: str,
    mask_ratio_evt: float,
) -> DualStreamMasker | None:
    """Construct the event masker for ``masker_mode`` and ``mask_ratio_evt``.

    - ``masker_mode="none"``: returns None (no masking); ``mask_ratio_evt``
      is ignored.
    - ``masker_mode="all_events"``, ``mask_ratio_evt=1.0``: returns
      ``AllEventsMasker(mask_ratio_tf=0.0, mask_ratio_evt=1.0)``; behavior
      is bit-identical to the pre-flag default.
    - ``masker_mode="all_events"``, ``0.0 < mask_ratio_evt < 1.0``: returns
      ``DualStreamMasker(mask_ratio_tf=0.0, mask_ratio_evt=<value>)`` which
      applies the standard BERT 80/10/10 split at the given ratio.
    - ``masker_mode="all_events"``, ``mask_ratio_evt=0.0``: raises
      ``ValueError``; use ``--masker none`` for the unmasked path.
    """
    if masker_mode == "none":
        return None
    if masker_mode != "all_events":
        raise ValueError(f"Unknown masker_mode={masker_mode!r}")
    if mask_ratio_evt == 0.0:
        raise ValueError(
            "mask_ratio_evt=0.0 with masker=all_events is a no-op. "
            "Use --masker none to run without event masking."
        )
    if not 0.0 < mask_ratio_evt <= 1.0:
        raise ValueError(f"mask_ratio_evt must be in (0, 1]; got {mask_ratio_evt}")
    if mask_ratio_evt == 1.0:
        return AllEventsMasker(mask_ratio_tf=0.0, mask_ratio_evt=1.0)
    return DualStreamMasker(mask_ratio_tf=0.0, mask_ratio_evt=mask_ratio_evt)


_MIXED_LABEL_SENTINEL: int = -1


def _window_labels_from_event_labels(
    labels_be: torch.Tensor | np.ndarray,
    readout: str,
) -> np.ndarray:
    """Aggregate per-event labels (B, E) to per-window labels (B,).

    Uses majority vote. Returns ``_MIXED_LABEL_SENTINEL`` (-1) for windows
    with a label tie (equal votes for more than one class) so the caller
    can filter them out cleanly. ERP-CORE paradigms with balanced classes
    (face_vs_car, related_vs_unrelated, etc.) routinely produce 4-4 ties
    when 8 trials are packed per window — this is data, not a packing bug.

    Only called for ``cls`` and ``tf`` readouts where the classification
    unit is a whole window rather than an individual event token.

    The caller must drop sentinel-labeled rows from the feature matrix and
    label vector before fitting any classifier; ``_collect_frozen_embeddings``
    does this and logs the drop count.
    """
    labels_np = labels_be.numpy() if isinstance(labels_be, torch.Tensor) else labels_be
    b = labels_np.shape[0]
    result = np.empty(b, dtype=np.int64)
    for i in range(b):
        row = labels_np[i]  # (E,)
        values, counts = np.unique(row, return_counts=True)
        if len(values) == 1:
            result[i] = int(values[0])
            continue
        max_count = counts.max()
        winners = values[counts == max_count]
        if len(winners) > 1:
            result[i] = _MIXED_LABEL_SENTINEL
        else:
            result[i] = int(winners[0])
    return result


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------


def _build_encoder(
    vocab_size: int,
    tag_init: torch.Tensor | None,
    args: argparse.Namespace,
    train_seed: int,
    init_from_checkpoint: bool,
) -> BertSSL:
    """Construct a BertSSL backbone, optionally loading the D.1.4 checkpoint.

    Seeds the global torch / numpy RNGs from ``train_seed`` unconditionally
    so every (encoder_seed, train_seed) cell is reproducible — including
    the ``finetune`` branch where the freshly constructed classifier head
    init and the DataLoader shuffle order both consume the global RNG.
    """
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
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
    if init_from_checkpoint:
        state = load_checkpoint_state(args.checkpoint)
        model.load_state_dict(state, strict=True)
    return model


class _ParadigmClassifier(nn.Module):
    """BertSSL backbone + linear classification head.

    ``masker_mode`` controls how HED inputs are presented to the encoder:

    - ``"all_events"`` (default): every event position is replaced with
      ``[MASK_EVT]``. The encoder never sees the ERP-CORE-derived HED input,
      forcing a TF-only decoding test. Closes the label-leakage path the
      pre-flight smoke flagged on N170 (random-init scored 98% via
      unmasked-HED LogReg).
    - ``"none"``: no masking; the encoder reads the ERP-CORE ``hed_vector``
      directly. Permissive setup — random-init can leak labels via the HED
      input pathway, so a masked-vs-unmasked pair quantifies how much of
      any apparent transfer comes from learned TF features versus passthrough.

    ``readout`` selects the encoder output used as the feature vector:

    - ``"evt"`` (default): ``out["evt_embeddings"]`` (B, E, D) — per-event.
      The classifier head produces (B, E, n_classes); CrossEntropy is computed
      over the (B*E) per-token positions with per-token labels.
    - ``"cls"``: ``out["cls_embedding"]`` (B, D) — one vector per window.
      The classifier head produces (B, n_classes). Per-window labels are
      derived from the (B, E) event labels by majority vote.
    - ``"tf"``: mean-pool ``out["tf_embeddings"]`` (B, n_tf, D) to (B, D).
      Same per-window aggregation as ``"cls"``.
    """

    def __init__(
        self,
        backbone: BertSSL,
        n_classes: int,
        masker_mode: str = "all_events",
        mask_ratio_evt: float = 1.0,
        readout: str = "evt",
    ) -> None:
        super().__init__()
        if masker_mode not in MASKER_MODES:
            raise ValueError(f"Unknown masker_mode={masker_mode!r}")
        if readout not in READOUT_MODES:
            raise ValueError(f"Unknown readout={readout!r}")
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.d_model, n_classes)
        self.masker_mode = masker_mode
        self.mask_ratio_evt = mask_ratio_evt
        self.readout = readout
        self._masker = _build_masker(masker_mode, mask_ratio_evt)

    def forward(
        self,
        tf: torch.Tensor,
        hed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass; return logits.

        For ``readout="evt"``: returns (B, E, n_classes).
        For ``readout="cls"`` or ``readout="tf"``: returns (B, n_classes).
        """
        out = self.backbone(tf, hed, masker=self._masker)
        if self.readout == "evt":
            feat = out["evt_embeddings"]  # (B, E, D)
            return self.classifier(feat)  # (B, E, n_classes)
        if self.readout == "cls":
            feat = out["cls_embedding"]  # (B, D)
        else:  # tf
            feat = out["tf_embeddings"].mean(dim=1)  # (B, D)
        return self.classifier(feat)  # (B, n_classes)


# -----------------------------------------------------------------------------
# Frozen embedding extraction (for the LogReg-on-frozen-features condition)
# -----------------------------------------------------------------------------


@torch.no_grad()
def _collect_frozen_embeddings(
    backbone: BertSSL,
    dataset: ErpcoreParadigmDataset,
    dm: DeviceManager,
    batch_size: int,
    masker_mode: str = "all_events",
    mask_ratio_evt: float = 1.0,
    readout: str = "evt",
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward dataset through frozen backbone, return per-trial features.

    For ``readout="evt"``: returns ``(features (n_trials, d_model),
    labels (n_trials,))`` after flattening over the event-token axis.

    For ``readout="cls"`` or ``readout="tf"``: returns ``(features
    (n_windows, d_model), labels (n_windows,))`` where each window's label
    is the majority vote of its E event-token labels.

    Splits are at subject level (see ``_split_subjects``); the LogReg
    downstream is fit on the rows returned here.
    """
    backbone.eval()
    masker = _build_masker(masker_mode, mask_ratio_evt)
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=paradigm_collate,
        pin_memory=(dm.device_type == "cuda"),
    )
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        tf = dm.to_device(batch["tf"])
        hed = dm.to_device(batch["hed"])
        with dm.get_amp_context():
            out = backbone(tf, hed, masker=masker)
        if readout == "evt":
            feat = out["evt_embeddings"].float().cpu().numpy()  # (B, E, D)
            b, e, d = feat.shape
            feats.append(feat.reshape(b * e, d))
            labels.append(batch["labels"].numpy().reshape(b * e))
        elif readout == "cls":
            feat = out["cls_embedding"].float().cpu().numpy()  # (B, D)
            feats.append(feat)
            labels.append(
                _window_labels_from_event_labels(batch["labels"], readout=readout)
            )
        else:  # tf
            feat = out["tf_embeddings"].float().cpu().numpy()  # (B, n_tf, D)
            feat_pooled = feat.mean(axis=1)  # (B, D)
            feats.append(feat_pooled)
            labels.append(
                _window_labels_from_event_labels(batch["labels"], readout=readout)
            )
    if not feats:
        return (
            np.zeros((0, backbone.d_model), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    feats_arr = np.concatenate(feats, axis=0).astype(np.float32)
    labels_arr = np.concatenate(labels, axis=0).astype(np.int64)
    # cls / tf readouts emit _MIXED_LABEL_SENTINEL for tied-label windows
    # (ERP-CORE balanced paradigms produce 4-4 windows under E=8 packing).
    # Drop those rows here so the downstream LR sees only well-defined labels.
    if readout in ("cls", "tf"):
        keep = labels_arr != _MIXED_LABEL_SENTINEL
        n_dropped = int((~keep).sum())
        if n_dropped:
            logger.info(
                "Dropped %d / %d windows with tied (mixed-class) labels for "
                "readout=%r; keeping %d well-defined windows.",
                n_dropped,
                labels_arr.size,
                readout,
                int(keep.sum()),
            )
        feats_arr = feats_arr[keep]
        labels_arr = labels_arr[keep]
    return feats_arr, labels_arr


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


@dataclass
class _ClassMetrics:
    accuracy: float
    balanced_accuracy: float
    n_eval: int
    class_counts: dict[int, int]


def _balanced_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Mean of per-class recall — robust to class imbalance.

    For a balanced 2-class probe equals raw accuracy; for skewed classes
    (e.g. P3 oddball ~20% targets) it stops the all-non-target predictor
    from scoring 80%.
    """
    classes = np.unique(labels)
    recalls: list[float] = []
    for cls in classes:
        idx = labels == cls
        if not idx.any():
            continue
        recalls.append(float((preds[idx] == cls).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _score_predictions(preds: np.ndarray, labels: np.ndarray) -> _ClassMetrics:
    classes, counts = np.unique(labels, return_counts=True)
    return _ClassMetrics(
        accuracy=float((preds == labels).mean()) * 100.0,
        balanced_accuracy=_balanced_accuracy(preds, labels) * 100.0,
        n_eval=int(labels.size),
        class_counts={int(c): int(n) for c, n in zip(classes, counts, strict=False)},
    )


# -----------------------------------------------------------------------------
# Per-condition runners
# -----------------------------------------------------------------------------


def _run_frozen(
    args: argparse.Namespace,
    dm: DeviceManager,
    rule: LabelRule,
    train_subjects: list[str],
    eval_subjects: list[str],
    vocab_size: int,
    tag_init: torch.Tensor | None,
) -> dict[str, Any]:
    """Frozen-encoder LogReg on per-trial evt_embeddings.

    With ``n_train_subjects == 0`` we report only the eval-set subject and
    class counts — no classifier is fit and no accuracy is computed. The
    canonical zero-shot diagnostic that quantifies what the encoder
    *already* predicts about ERP-CORE event semantics is the dedicated
    masked-event AUC probe in
    ``neural_vocabulary.scripts.eval_event_recovery`` (run that
    against ``--features-dir`` pointed at the ERP-CORE TF cache for the
    canonical N=0 cell). This script's N=0 row exists to populate the
    accuracy-vs-N table at its left endpoint with a "no classifier"
    placeholder, not to recompute the AUC diagnostic.
    """
    backbone = _build_encoder(
        vocab_size=vocab_size,
        tag_init=tag_init,
        args=args,
        train_seed=args.train_seed,
        init_from_checkpoint=not args.random_init,
    )
    backbone = cast("BertSSL", dm.to_device(backbone))

    eval_ds = _build_paradigm_dataset(
        args.tf_dir,
        eval_subjects,
        rule,
        n_time=args.n_time,
        n_freqs=args.n_freqs,
        n_channels=args.n_channels,
    )
    feats_eval, labels_eval = _collect_frozen_embeddings(
        backbone,
        eval_ds,
        dm,
        args.batch_size,
        masker_mode=args.masker,
        mask_ratio_evt=args.mask_ratio_evt,
        readout=args.readout,
        num_workers=args.num_workers,
    )

    out: dict[str, Any] = {
        "masker": args.masker,
        "mask_ratio_evt": args.mask_ratio_evt,
        "readout": args.readout,
        "random_init": bool(args.random_init),
        "n_eval_trials": int(labels_eval.size),
        "n_eval_subjects": len(eval_subjects),
        "eval_class_counts": {
            int(c): int(n)
            for c, n in zip(*np.unique(labels_eval, return_counts=True), strict=False)
        },
    }

    #  collapse guard. Default raises; --allow-collapse surfaces
    # the cosine value in the result JSON without aborting.
    if len(np.unique(labels_eval)) >= 2:
        eval_collapse = assert_no_class_mean_collapse(
            feats_eval,
            labels_eval,
            allow_collapse=args.allow_collapse,
        )
        out["eval_class_mean_cosine"] = eval_collapse.max_cosine
        out["eval_collapse_pair"] = [
            eval_collapse.class_a,
            eval_collapse.class_b,
        ]
        out["eval_collapse_skipped"] = None
    else:
        # Single-class eval is itself a configuration bug (degenerate
        # paradigm split). Record the skip explicitly so result-JSON
        # consumers cannot conflate "ran and passed" with "silently
        # skipped because eval had one class".
        logger.warning(
            "Eval split has %d unique paradigm class(es) — skipping the "
            " collapse guard. Result JSON will record "
            "eval_collapse_skipped='single_class_eval'.",
            int(np.unique(labels_eval).size),
        )
        out["eval_class_mean_cosine"] = None
        out["eval_collapse_pair"] = None
        out["eval_collapse_skipped"] = "single_class_eval"

    if args.n_train_subjects == 0:
        # Pure zero-shot — no LogReg, just the diagnostic.
        out["mode"] = "zero_shot"
        out["accuracy_pct"] = None
        out["balanced_accuracy_pct"] = None
        out["class_weight_resolved"] = None
        out["balanced_sampler"] = False
        return out

    train_ds = _build_paradigm_dataset(
        args.tf_dir,
        train_subjects,
        rule,
        n_time=args.n_time,
        n_freqs=args.n_freqs,
        n_channels=args.n_channels,
    )
    feats_train, labels_train = _collect_frozen_embeddings(
        backbone,
        train_ds,
        dm,
        args.batch_size,
        masker_mode=args.masker,
        mask_ratio_evt=args.mask_ratio_evt,
        readout=args.readout,
        num_workers=args.num_workers,
    )
    if len(np.unique(labels_train)) < 2:
        raise RuntimeError(
            f"Train split has only {len(np.unique(labels_train))} class(es) — "
            "balanced LogReg needs at least 2. Increase n_train_subjects or "
            "check the rule."
        )

    scaler = StandardScaler()
    fx_train = scaler.fit_transform(feats_train)
    fx_eval = scaler.transform(feats_eval)
    cw_resolved = _resolve_class_weight(rule.paradigm, args.class_weight)
    sklearn_cw: str | None = "balanced" if cw_resolved == "balanced" else None
    clf = LogisticRegression(
        max_iter=2000,
        random_state=args.train_seed,
        class_weight=sklearn_cw,
    )
    clf.fit(fx_train, labels_train)
    preds = clf.predict(fx_eval)
    metrics = _score_predictions(preds, labels_eval)

    out.update(
        {
            "mode": "few_shot_logreg",
            "n_train_trials": int(labels_train.size),
            "n_train_subjects": len(train_subjects),
            "train_class_counts": {
                int(c): int(n)
                for c, n in zip(
                    *np.unique(labels_train, return_counts=True), strict=False
                )
            },
            "accuracy_pct": metrics.accuracy,
            "balanced_accuracy_pct": metrics.balanced_accuracy,
            "class_weight_resolved": cw_resolved,
            "balanced_sampler": False,
        }
    )
    return out


def _run_trainable(
    args: argparse.Namespace,
    dm: DeviceManager,
    rule: LabelRule,
    train_subjects: list[str],
    eval_subjects: list[str],
    vocab_size: int,
    tag_init: torch.Tensor | None,
    init_from_checkpoint: bool,
) -> dict[str, Any]:
    """End-to-end fine-tune (or from-scratch) with a per-position linear head.

    For ``scratch`` we deliberately pass ``tag_init=None`` so the
    BertSSL.tag_embeddings parameter is randomly initialized rather than
    seeded from the HBN HED hierarchy — anything HBN-derived would
    contaminate the from-scratch baseline. ``finetune`` keeps the
    hierarchy init since the checkpoint load overwrites it anyway.
    """
    effective_tag_init = tag_init if init_from_checkpoint else None
    backbone = _build_encoder(
        vocab_size=vocab_size,
        tag_init=effective_tag_init,
        args=args,
        train_seed=args.train_seed,
        init_from_checkpoint=init_from_checkpoint,
    )
    classifier = _ParadigmClassifier(
        backbone,
        n_classes=rule.n_classes,
        masker_mode=args.masker,
        mask_ratio_evt=args.mask_ratio_evt,
        readout=args.readout,
    )
    classifier = cast("_ParadigmClassifier", dm.to_device(classifier))

    train_ds = _build_paradigm_dataset(
        args.tf_dir,
        train_subjects,
        rule,
        n_time=args.n_time,
        n_freqs=args.n_freqs,
        n_channels=args.n_channels,
    )
    eval_ds = _build_paradigm_dataset(
        args.tf_dir,
        eval_subjects,
        rule,
        n_time=args.n_time,
        n_freqs=args.n_freqs,
        n_channels=args.n_channels,
    )

    cw_resolved = _resolve_class_weight(rule.paradigm, args.class_weight)

    # Build per-class inverse-frequency weight tensor for cross_entropy when
    # class_weight == "balanced".  Collected from all training windows.
    ce_weight: torch.Tensor | None = None
    if cw_resolved == "balanced":
        all_labels_list: list[np.ndarray] = []
        for i in range(len(train_ds)):
            item = train_ds[i]
            all_labels_list.append(item["labels"].numpy().reshape(-1))
        all_labels_np = np.concatenate(all_labels_list)
        classes, class_counts = np.unique(all_labels_np, return_counts=True)
        n_total = int(all_labels_np.size)
        n_cls = len(classes)
        w = np.zeros(n_cls, dtype=np.float32)
        for c, cnt in zip(classes, class_counts, strict=False):
            w[int(c)] = n_total / (n_cls * cnt)
        ce_weight = torch.tensor(w)

    # Build per-sample weights for WeightedRandomSampler when requested.
    sampler: WeightedRandomSampler | None = None
    if args.balanced_sampler:
        sample_labels: list[int] = []
        for i in range(len(train_ds)):
            item = train_ds[i]
            # Use majority label of each window as the sample class.
            lbl_arr = item["labels"].numpy().reshape(-1)
            vals, cnts = np.unique(lbl_arr, return_counts=True)
            sample_labels.append(int(vals[np.argmax(cnts)]))
        sample_labels_np = np.array(sample_labels, dtype=np.int64)
        s_classes, s_counts = np.unique(sample_labels_np, return_counts=True)
        class_weight_map: dict[int, float] = {
            int(c): 1.0 / max(1, int(cnt))
            for c, cnt in zip(s_classes, s_counts, strict=False)
        }
        sample_weights = [class_weight_map[int(lbl)] for lbl in sample_labels_np]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=paradigm_collate,
        pin_memory=(dm.device_type == "cuda"),
        drop_last=False,
    )
    eval_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=paradigm_collate,
        pin_memory=(dm.device_type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    n_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=n_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_balacc = -1.0
    best_metrics: _ClassMetrics | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        classifier.train()
        epoch_loss = 0.0
        epoch_n = 0
        for batch in train_loader:
            tf = dm.to_device(batch["tf"])
            hed = dm.to_device(batch["hed"])
            labels_be = batch["labels"]  # (B, E) on CPU
            optimizer.zero_grad(set_to_none=True)
            weight_on_device = (
                dm.to_device(ce_weight) if ce_weight is not None else None
            )
            with dm.get_amp_context():
                logits = classifier(tf, hed)
                if args.readout == "evt":
                    # (B, E, n_classes) -> flatten to (B*E,)
                    tgt = dm.to_device(labels_be).reshape(-1)
                    loss = nnf.cross_entropy(
                        logits.reshape(-1, rule.n_classes),
                        tgt,
                        weight=weight_on_device,
                    )
                    n_items = int(labels_be.numel())
                else:
                    # (B, n_classes); aggregate labels to (B,). Drop tied
                    # mixed-class windows (sentinel == -1) which produce
                    # an undefined per-window target for ERP-CORE balanced
                    # paradigms.
                    win_labels = _window_labels_from_event_labels(
                        labels_be, readout=args.readout
                    )
                    keep = win_labels != _MIXED_LABEL_SENTINEL
                    if not keep.any():
                        continue  # entire batch was tied; skip
                    keep_t = torch.from_numpy(keep)
                    tgt = dm.to_device(torch.from_numpy(win_labels[keep]))
                    loss = nnf.cross_entropy(
                        logits[keep_t], tgt, weight=weight_on_device
                    )
                    n_items = int(tgt.numel())
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach()) * n_items
            epoch_n += n_items

        # Eval pass.
        classifier.eval()
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch in eval_loader:
                tf = dm.to_device(batch["tf"])
                hed = dm.to_device(batch["hed"])
                with dm.get_amp_context():
                    logits = classifier(tf, hed)
                if args.readout == "evt":
                    preds = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
                    lbls = batch["labels"].numpy().reshape(-1)
                else:
                    preds = logits.argmax(dim=-1).cpu().numpy()  # (B,)
                    lbls = _window_labels_from_event_labels(
                        batch["labels"], readout=args.readout
                    )
                    keep_mask = lbls != _MIXED_LABEL_SENTINEL
                    preds = preds[keep_mask]
                    lbls = lbls[keep_mask]
                all_preds.append(preds)
                all_labels.append(lbls)
        preds_arr = np.concatenate(all_preds, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        metrics = _score_predictions(preds_arr, labels_arr)
        train_loss_mean = epoch_loss / max(1, epoch_n)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_mean,
                "eval_accuracy_pct": metrics.accuracy,
                "eval_balanced_accuracy_pct": metrics.balanced_accuracy,
            }
        )
        logger.info(
            "[%s] epoch %d/%d loss=%.4f acc=%.2f bal=%.2f",
            "ft" if init_from_checkpoint else "scr",
            epoch,
            args.epochs,
            train_loss_mean,
            metrics.accuracy,
            metrics.balanced_accuracy,
        )
        if metrics.balanced_accuracy > best_balacc:
            best_balacc = metrics.balanced_accuracy
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("No eval epochs ran; check --epochs and dataset sizes.")

    return {
        "mode": "finetune" if init_from_checkpoint else "scratch",
        "masker": args.masker,
        "mask_ratio_evt": args.mask_ratio_evt,
        "readout": args.readout,
        "n_train_trials": int(len(train_ds) * train_ds.epochs_per_window),
        "n_train_subjects": len(train_subjects),
        "n_eval_trials": int(len(eval_ds) * eval_ds.epochs_per_window),
        "n_eval_subjects": len(eval_subjects),
        "epochs": args.epochs,
        "lr": args.lr,
        "best_accuracy_pct": best_metrics.accuracy,
        "best_balanced_accuracy_pct": best_metrics.balanced_accuracy,
        "best_class_counts": best_metrics.class_counts,
        "history": history,
        "class_weight_resolved": cw_resolved,
        "balanced_sampler": bool(args.balanced_sampler),
    }


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def _resolve_rule(paradigm: str, probe: str | None) -> LabelRule:
    if probe is None:
        rules = LABEL_RULES.get(paradigm, [])
        if not rules:
            raise SystemExit(f"Unknown paradigm {paradigm!r}")
        return rules[0]
    return get_rule(paradigm, probe)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paradigm", required=True, choices=list(DEFAULT_PARADIGMS))
    parser.add_argument(
        "--probe",
        default=None,
        help="Probe name (e.g. face_vs_car). Default: first probe of paradigm.",
    )
    parser.add_argument("--condition", required=True, choices=list(CONDITIONS))
    parser.add_argument("--n-train-subjects", type=int, required=True)
    parser.add_argument("--encoder-seed", type=int, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        help="D.1.4 checkpoint .pt for frozen/finetune. Required unless "
        "--condition=scratch or --random-init is set.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Skip the checkpoint load and probe a freshly-initialized "
        "BertSSL. Only valid with --condition=frozen. Mutually exclusive "
        "with --checkpoint. Used by E-0 control to test whether "
        "class_mean_cosine under AllEventsMasker is architecturally "
        "predictable (see issue #213).",
    )
    parser.add_argument(
        "--vectorizer",
        type=Path,
        default=Path("${HBN_DATA_DIR}/hed_vectorizer.pt"),
    )
    parser.add_argument(
        "--tf-dir",
        type=Path,
        default=Path("${HBN_DATA_DIR}/preprocessed_v10_erpcore_tf"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--allow-collapse",
        action="store_true",
        help="Allow runs to proceed past the class-mean-cosine collapse "
        "guard . Default raises RepresentationCollapseError; "
        "this flag is for diagnostic runs that need the cosine value "
        "surfaced in the result JSON.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count for frozen-feature extraction and "
        "finetune/scratch train/eval loaders. Set to 0 only for "
        "single-threaded debugging.",
    )
    parser.add_argument(
        "--masker",
        type=str,
        default="all_events",
        choices=list(MASKER_MODES),
        help="HED-input masker mode at fwd time. ``all_events`` (default) "
        "replaces every event input with [MASK_EVT] (rigorous TF-only test). "
        "``none`` lets the encoder read the ERP-CORE hed_vector directly "
        "(permissive; allows label-leakage but quantifies pretraining lift).",
    )
    parser.add_argument(
        "--mask-ratio-evt",
        type=float,
        default=1.0,
        help="Event-token mask ratio when --masker=all_events. "
        "1.0 (default): use AllEventsMasker (deterministic 100%% mask, "
        "bit-identical to pre-flag behavior). "
        "< 1.0: use DualStreamMasker with mask_ratio_evt=<value> "
        "(BERT 80/10/10 split at the given ratio; use 0.5 to match "
        "pretraining). 0.0 is rejected — use --masker none instead. "
        "Ignored when --masker none.",
    )
    parser.add_argument(
        "--readout",
        type=str,
        default="evt",
        choices=list(READOUT_MODES),
        help="Which encoder output to use as the feature vector. "
        '``evt`` (default): out["evt_embeddings"] (B, E, D) flattened '
        "to (B*E, D); per-event-token classification. "
        '``cls``: out["cls_embedding"] (B, D); per-window. '
        '``tf``: mean-pool out["tf_embeddings"] to (B, D); per-window. '
        "For cls/tf, per-window labels are derived by majority vote over "
        "the E per-event labels.",
    )
    parser.add_argument("--epochs-per-window", type=int, default=8)
    parser.add_argument(
        "--n-time",
        type=int,
        default=10,
        help="Time-bin count per epoch in the ERP-CORE TF h5 files. "
        "ErpcoreParadigmDataset raises on shape mismatch.",
    )
    parser.add_argument(
        "--n-freqs",
        type=int,
        default=6,
        help="Frequency-bin count per epoch in the ERP-CORE TF h5 files. "
        "ErpcoreParadigmDataset raises on shape mismatch.",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=64,
        help="Channel count per epoch in the ERP-CORE TF h5 files. "
        "Default 64 matches HBN-harmonized layouts; native ERP-CORE is "
        "30. ErpcoreParadigmDataset raises on shape mismatch.",
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
        "--patch-mode",
        type=str,
        default="flat",
        choices=("flat", "channel_token"),
        help="Must match the checkpoint's training config. Default 'flat' "
        "matches  D.1.x; use 'channel_token' for  E2 checkpoints.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="auto",
        choices=list(CLASS_WEIGHT_MODES),
        help="Class-weight strategy for the probe/loss. "
        "``none``: no weighting (LogisticRegression(class_weight=None); "
        "unweighted cross_entropy for finetune/scratch). "
        "``balanced``: LogisticRegression(class_weight='balanced') for frozen; "
        "per-class inverse-frequency weights in cross_entropy for finetune/scratch. "
        "``auto`` (default): resolves to ``balanced`` for MMN and P3 "
        "(documented >2:1 class imbalance per Day-3 audit) and ``none`` for "
        "the remaining paradigms.",
    )
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use WeightedRandomSampler in the finetune/scratch DataLoader "
        "to draw equal expected class count per batch. No effect on the "
        "frozen probe path.",
    )
    args = parser.parse_args(argv)

    if args.random_init and args.condition != "frozen":
        raise SystemExit(
            f"--random-init is only valid with --condition=frozen; got "
            f"--condition={args.condition!r}. Use --condition=scratch for "
            "random-init training."
        )
    if args.random_init and args.checkpoint is not None:
        raise SystemExit("--random-init is mutually exclusive with --checkpoint.")
    if (
        args.condition in {"frozen", "finetune"}
        and args.checkpoint is None
        and not args.random_init
    ):
        raise SystemExit(
            f"--checkpoint is required for condition={args.condition!r} "
            "(or pass --random-init for the E-0 control)."
        )
    if args.condition == "frozen" and args.n_train_subjects < 0:
        raise SystemExit("--n-train-subjects must be ≥ 0 for frozen")
    if args.condition in {"finetune", "scratch"} and args.n_train_subjects < 1:
        raise SystemExit(
            f"--n-train-subjects must be ≥ 1 for condition={args.condition!r}; "
            "you cannot fine-tune with zero training data."
        )
    # Validate mask-ratio-evt early for a clear user-facing error.
    if args.masker == "all_events" and args.mask_ratio_evt == 0.0:
        raise SystemExit(
            "--mask-ratio-evt=0.0 with --masker=all_events is a no-op. "
            "Use --masker none to run without event masking."
        )
    if args.masker == "none" and args.mask_ratio_evt != 1.0:
        logger.warning(
            "--mask-ratio-evt=%.2f is ignored when --masker=none "
            "(no masker is constructed).",
            args.mask_ratio_evt,
        )

    rule = _resolve_rule(args.paradigm, args.probe)
    dm = DeviceManager(args.device)

    all_subjects = _list_subjects_for_paradigm(args.tf_dir, rule.paradigm)
    if not all_subjects:
        raise SystemExit(
            f"No subjects with TF files for paradigm={rule.paradigm} in {args.tf_dir}"
        )
    train_subjects, eval_subjects = _split_subjects(
        all_subjects, args.n_train_subjects, seed=args.train_seed
    )
    logger.info(
        "Paradigm=%s probe=%s condition=%s N=%d encoder_seed=%d train_seed=%d",
        rule.paradigm,
        rule.probe,
        args.condition,
        args.n_train_subjects,
        args.encoder_seed,
        args.train_seed,
    )
    logger.info(
        "Subjects: %d total, %d train, %d eval",
        len(all_subjects),
        len(train_subjects),
        len(eval_subjects),
    )

    tag_to_idx, _tag_depths, tag_init = load_tag_to_idx_and_depths(args.vectorizer)
    vocab_size = max(tag_to_idx.values()) + 1

    t0 = time.perf_counter()
    if args.condition == "frozen":
        result = _run_frozen(
            args, dm, rule, train_subjects, eval_subjects, vocab_size, tag_init
        )
    elif args.condition == "finetune":
        result = _run_trainable(
            args,
            dm,
            rule,
            train_subjects,
            eval_subjects,
            vocab_size,
            tag_init,
            init_from_checkpoint=True,
        )
    else:  # scratch
        result = _run_trainable(
            args,
            dm,
            rule,
            train_subjects,
            eval_subjects,
            vocab_size,
            tag_init,
            init_from_checkpoint=False,
        )
    elapsed = time.perf_counter() - t0

    summary: dict[str, Any] = {
        "paradigm": rule.paradigm,
        "probe": rule.probe,
        "n_classes": rule.n_classes,
        "class_names": list(rule.class_names),
        "condition": args.condition,
        "n_train_subjects": args.n_train_subjects,
        "encoder_seed": args.encoder_seed,
        "train_seed": args.train_seed,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "all_subjects": all_subjects,
        "train_subjects": train_subjects,
        "eval_subjects": eval_subjects,
        "elapsed_s": elapsed,
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s (elapsed=%.1fs)", args.output, elapsed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
