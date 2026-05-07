"""Linear probing pipeline for HED-BERT [EVT] embeddings.

Freezes the encoder, extracts [EVT] embeddings from the eval set,
and trains a simple linear classifier for downstream tasks:
- Task classification: which HBN task produced this epoch?
- Event type classification: what event type is this?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from neural_vocabulary.data.collate import BucketBatchSampler, EventEpochCollator
from neural_vocabulary.data.packed_collator import PackedSequenceCollator

if TYPE_CHECKING:
    from collections.abc import Callable

    from neural_vocabulary.models.joint_model import HEDBERT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    """Result from a linear probe experiment."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion: np.ndarray
    class_names: tuple[str, ...]
    report: str

    def summary(self) -> str:
        """One-line summary."""
        return (
            f"accuracy={self.accuracy:.3f} "
            f"macro_F1={self.macro_f1:.3f} "
            f"weighted_F1={self.weighted_f1:.3f} "
            f"({len(self.class_names)} classes)"
        )


class LinearProbe:
    """Extract [EVT] embeddings and train linear classifiers.

    Usage:
        probe = LinearProbe(model, device)
        embeddings, labels = probe.extract_embeddings(file_list, label_fn)
        result = probe.fit_and_evaluate(train_emb, train_lbl, eval_emb, eval_lbl)
    """

    def __init__(
        self,
        model: HEDBERT,
        device: torch.device,
        max_seq_len: int = 200,
        batch_size: int = 512,
        num_workers: int = 4,
    ) -> None:
        self.model = model
        self.device = device
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Freeze the model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def extract_embeddings(
        self,
        h5_files: list,
        label_fn: Callable,
        max_batches: int | None = None,
        use_packed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract [EVT] embeddings and labels from h5 files.

        Args:
            h5_files: List of h5 file Paths.
            label_fn: Function that takes an h5 Path and returns a label
                string (e.g., task name or event type).
            max_batches: Maximum batches to process. None for all.
            use_packed: When True, use packed forward path with
                cross-epoch attention (matches training). Each h5 file
                is treated as one multi-epoch sequence.

        Returns:
            (embeddings, labels) tuple. embeddings is (N, embed_dim),
            labels is (N,) array of label strings.
        """
        if use_packed:
            return self._extract_packed(h5_files, label_fn, max_batches)
        return self._extract_unpacked(h5_files, label_fn, max_batches)

    def _extract_unpacked(
        self,
        h5_files: list,
        label_fn: Callable,
        max_batches: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings using single-epoch unpacked forward."""
        dataset = _FileListDataset(h5_files, label_fn)
        if len(dataset) == 0:
            return np.empty((0, 0)), np.empty((0,))

        encoder_stride = self.model.config.total_stride

        lengths = dataset.get_lengths()
        valid = [i for i, length in enumerate(lengths) if length >= encoder_stride]
        if not valid:
            logger.warning(
                "No epochs long enough for encoder stride %d", encoder_stride
            )
            return np.empty((0, 0)), np.empty((0,))

        subset = torch.utils.data.Subset(dataset, valid)
        subset_lengths = [lengths[i] for i in valid]

        collator = _LabeledCollator(max_length=self.max_seq_len)
        sampler = BucketBatchSampler(
            subset_lengths,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )
        loader = DataLoader(
            subset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=self.num_workers,
        )

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for i, (batch, labels) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                outputs = self.model(batch)
                evt_emb = outputs["evt_embeddings"]
                all_embeddings.append(evt_emb.cpu().numpy())
                all_labels.extend(labels)

        if not all_embeddings:
            logger.warning("Unpacked extraction produced 0 embeddings")
            return np.empty((0, 0)), np.empty((0,))

        embeddings = np.concatenate(all_embeddings, axis=0)
        label_arr = np.array(all_labels)
        logger.info(
            "Extracted %d embeddings (dim=%d)", embeddings.shape[0], embeddings.shape[1]
        )
        return embeddings, label_arr

    def _extract_packed(
        self,
        h5_files: list,
        label_fn: Callable,
        max_batches: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings using packed multi-epoch forward with cross-epoch attention.

        Each h5 file becomes one multi-epoch sequence. All epochs from
        the file are read deterministically (no random window), packed
        via PackedSequenceCollator, and forwarded through
        model.forward_packed(). EVT embeddings are extracted per-epoch
        and the recording-level label is expanded to each epoch.
        """
        config = self.model.config
        dataset = _EvalPackedDataset(
            h5_files,
            label_fn,
            max_channels=config.target_channels,
            max_epoch_len=self.max_seq_len,
            normalize=True,
        )
        if len(dataset) == 0:
            return np.empty((0, 0)), np.empty((0,))

        collator = _PackedLabeledCollator(
            max_total_length=config.max_seq_len * 16,
            total_stride=config.total_stride,
            sfreq=config.sfreq,
        )
        # Packed items are full recordings; use smaller batch size
        packed_batch_size = max(1, self.batch_size // 16)
        loader = DataLoader(
            dataset,
            batch_size=packed_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.num_workers,
        )

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for i, (batch, labels) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)

                try:
                    outputs = self.model.forward_packed(batch)
                except Exception:
                    logger.error(
                        "forward_packed failed at batch %d, eeg shape: %s",
                        i,
                        batch.get("eeg", torch.empty(0)).shape,
                    )
                    raise
                # evt_embeddings: (B, max_n_epochs, embed_dim)
                evt_emb: torch.Tensor = outputs["evt_embeddings"]  # type: ignore[assignment]
                n_valid: torch.Tensor = outputs["n_valid_evts"]  # type: ignore[assignment]

                # Expand labels: each recording label repeats for its valid EVTs
                max_evts = evt_emb.shape[1]
                for b in range(evt_emb.shape[0]):
                    n = min(int(n_valid[b].item()), max_evts)
                    if n > 0:
                        emb = evt_emb[b, :n, :].cpu().numpy()
                        all_embeddings.append(emb)
                        all_labels.extend([labels[b]] * n)

        if not all_embeddings:
            logger.warning("Packed extraction produced 0 embeddings")
            return np.empty((0, 0)), np.empty((0,))

        embeddings = np.concatenate(all_embeddings, axis=0)
        label_arr = np.array(all_labels)
        logger.info(
            "Extracted %d packed embeddings (dim=%d)",
            embeddings.shape[0],
            embeddings.shape[1],
        )
        return embeddings, label_arr

    def fit_and_evaluate(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        eval_embeddings: np.ndarray,
        eval_labels: np.ndarray,
        max_iter: int = 1000,
    ) -> ProbeResult:
        """Train logistic regression and evaluate on held-out set.

        Uses GPU-accelerated PyTorch linear probe when CUDA is available
        (handles 1.4M embeddings in ~10s vs minutes on CPU). Falls back
        to sklearn LogisticRegression on CPU.

        Args:
            train_embeddings: (N_train, embed_dim) training features.
            train_labels: (N_train,) string labels.
            eval_embeddings: (N_eval, embed_dim) evaluation features.
            eval_labels: (N_eval,) string labels.
            max_iter: Maximum iterations for logistic regression.

        Returns:
            ProbeResult with metrics and confusion matrix.
        """
        le = LabelEncoder()
        y_train = le.fit_transform(train_labels)
        y_eval = le.transform(eval_labels)
        class_names = le.classes_.tolist()

        if self.device.type == "cuda":
            y_pred = self._fit_gpu(
                train_embeddings,
                y_train,
                eval_embeddings,
                num_classes=len(class_names),
                max_iter=max_iter,
            )
        else:
            clf = LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
            )
            clf.fit(train_embeddings, y_train)
            y_pred = clf.predict(eval_embeddings)

        accuracy = float((y_pred == y_eval).mean())
        macro_f1 = float(f1_score(y_eval, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_eval, y_pred, average="weighted"))
        cm = confusion_matrix(y_eval, y_pred)
        report = classification_report(y_eval, y_pred, target_names=class_names)

        logger.info("Linear probe: accuracy=%.3f, macro_F1=%.3f", accuracy, macro_f1)
        return ProbeResult(
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            confusion=cm,
            class_names=tuple(class_names),
            report=report,
        )

    def _fit_gpu(
        self,
        train_embeddings: np.ndarray,
        y_train: np.ndarray,
        eval_embeddings: np.ndarray,
        num_classes: int,
        max_iter: int = 1000,
        lr: float = 0.01,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """GPU-accelerated linear probe via PyTorch.

        Trains nn.Linear + CrossEntropyLoss with Adam in mini-batches.
        ~100x faster than sklearn lbfgs on 1M+ embeddings.
        """
        import torch.nn as nn
        from torch.optim import Adam

        x_train = torch.from_numpy(train_embeddings).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).long().to(self.device)
        x_eval = torch.from_numpy(eval_embeddings).float().to(self.device)

        embed_dim = x_train.shape[1]
        probe = nn.Linear(embed_dim, num_classes).to(self.device)
        optimizer = Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        n_train = x_train.shape[0]
        n_epochs = min(max_iter, 50)

        probe.train()
        for epoch in range(n_epochs):
            perm = torch.randperm(n_train, device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                logits = probe(x_train[idx])
                loss = criterion(logits, y_train_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.debug(
                    "GPU probe epoch %d/%d loss=%.4f",
                    epoch + 1,
                    n_epochs,
                    epoch_loss / n_batches,
                )

        probe.eval()
        with torch.no_grad():
            preds = []
            for start in range(0, x_eval.shape[0], batch_size):
                logits = probe(x_eval[start : start + batch_size])
                preds.append(logits.argmax(dim=-1))
            y_pred = torch.cat(preds).cpu().numpy()

        return y_pred


def task_label_fn(h5_path) -> str:
    """Extract task name from h5 filename for task classification."""
    parts = h5_path.stem.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


class _FileListDataset(torch.utils.data.Dataset):
    """Wrap a list of h5 files into a dataset with per-epoch access.

    Provides h5 reading similar to PreprocessedEEGDataset but with
    per-file index and label support.
    """

    def __init__(self, h5_files: list, label_fn: Callable) -> None:
        import h5py

        self._epoch_index: list[dict] = []
        self._labels: list[str] = []

        for h5_path in h5_files:
            label = label_fn(h5_path)
            try:
                with h5py.File(h5_path, "r") as f:
                    n_epochs = f.attrs.get("n_epochs", 0)
                    for i in range(n_epochs):
                        grp_name = f"epoch_{i:05d}"
                        if grp_name not in f:
                            continue
                        grp = f[grp_name]
                        self._epoch_index.append(
                            {
                                "h5_path": str(h5_path),
                                "epoch_key": grp_name,
                                "event_id": int(grp.attrs.get("event_id", 0)),
                                "length": int(grp.attrs.get("length", 0)),
                                "pre_event_samples": int(
                                    grp.attrs.get("pre_event_samples", 0)
                                ),
                            }
                        )
                        self._labels.append(label)
            except (OSError, KeyError) as e:
                logger.warning("Error reading %s: %s", h5_path, e)

    def __getitem__(self, idx: int) -> tuple[dict, str]:
        import h5py

        meta = self._epoch_index[idx]
        with h5py.File(meta["h5_path"], "r") as f:
            grp = f[meta["epoch_key"]]
            eeg = grp["eeg"][:].astype(np.float32)

        # Z-score per channel
        mean = eeg.mean(axis=-1, keepdims=True)
        std = eeg.std(axis=-1, keepdims=True)
        eeg = (eeg - mean) / (std + 1e-8)

        sample = {
            "eeg": eeg,
            "event_id": meta["event_id"],
            "length": meta["length"],
            "pre_event_samples": meta["pre_event_samples"],
        }
        return sample, self._labels[idx]

    def __len__(self) -> int:
        return len(self._epoch_index)

    def get_lengths(self) -> list[int]:
        return [m["length"] for m in self._epoch_index]


class _LabeledCollator:
    """Collate (sample_dict, label) tuples into (batch_dict, label_list)."""

    def __init__(self, max_length: int = 200) -> None:
        self._collator = EventEpochCollator(max_length=max_length)

    def __call__(
        self, items: list[tuple[dict, str]]
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        samples = [item[0] for item in items]
        labels = [item[1] for item in items]
        batch = self._collator(samples)
        return batch, labels


class _EvalPackedDataset(torch.utils.data.Dataset):
    """Wrap h5 files into a packed-sequence dataset for eval.

    Each __getitem__ reads ALL epochs from one h5 file deterministically
    (no random window), returning a dict compatible with
    PackedSequenceCollator. The recording-level label is returned
    alongside the sequence dict.
    """

    def __init__(
        self,
        h5_files: list,
        label_fn: Callable,
        max_channels: int = 64,
        max_epoch_len: int = 200,
        normalize: bool = True,
    ) -> None:
        import h5py

        self._recordings: list[dict] = []
        self._labels: list[str] = []

        for h5_path in h5_files:
            label = label_fn(h5_path)
            try:
                with h5py.File(h5_path, "r") as f:
                    n_epochs = int(f.attrs.get("n_epochs", 0))
                    if n_epochs >= 1:
                        self._recordings.append(
                            {"h5_path": str(h5_path), "n_epochs": n_epochs}
                        )
                        self._labels.append(label)
            except (OSError, KeyError) as e:
                logger.warning("Error scanning %s: %s", h5_path, e)

        self._max_channels = max_channels
        self._max_epoch_len = max_epoch_len
        self._normalize = normalize

    def __getitem__(self, idx: int) -> tuple[dict, str]:
        import h5py

        meta = self._recordings[idx]
        label = self._labels[idx]

        eeg_epochs = []
        hed_vectors = []
        lengths = []
        pre_events = []

        try:
            with h5py.File(meta["h5_path"], "r") as f:
                for i in range(meta["n_epochs"]):
                    grp_name = f"epoch_{i:05d}"
                    if grp_name not in f:
                        continue
                    grp = f[grp_name]
                    eeg = grp["eeg"][:].astype(np.float32)

                    if self._normalize:
                        mean = eeg.mean(axis=-1, keepdims=True)
                        std = eeg.std(axis=-1, keepdims=True)
                        eeg = (eeg - mean) / (std + 1e-8)

                    eeg = eeg[: self._max_channels, : self._max_epoch_len]

                    eeg_epochs.append(eeg)
                    lengths.append(eeg.shape[1])
                    pre_events.append(int(grp.attrs.get("pre_event_samples", 0)))

                    if "hed_vector" in grp:
                        hed_vectors.append(grp["hed_vector"][:].astype(np.float32))
                    else:
                        hed_vectors.append(None)
        except (OSError, KeyError) as e:
            logger.error("Failed to read %s: %s", meta["h5_path"], e)
            raise

        if not eeg_epochs:
            raise ValueError(
                f"No readable epochs in {meta['h5_path']} "
                f"(n_epochs attr={meta['n_epochs']})"
            )

        sequence = {
            "eeg_epochs": eeg_epochs,
            "hed_vectors": hed_vectors,
            "lengths": lengths,
            "pre_event_samples": pre_events,
            "n_epochs": len(eeg_epochs),
        }
        return sequence, label

    def __len__(self) -> int:
        return len(self._recordings)


class _PackedLabeledCollator:
    """Collate (sequence_dict, label) tuples for packed evaluation.

    Delegates data to PackedSequenceCollator and returns labels
    at the recording level (one label per sequence, not per epoch).
    """

    def __init__(
        self,
        max_total_length: int = 3200,
        total_stride: int = 75,
        sfreq: float = 100.0,
    ) -> None:
        self._collator = PackedSequenceCollator(
            max_total_length=max_total_length,
            total_stride=total_stride,
            sfreq=sfreq,
        )

    def __call__(
        self, items: list[tuple[dict, str]]
    ) -> tuple[dict[str, torch.Tensor | list], list[str]]:
        sequences = [item[0] for item in items]
        labels = [item[1] for item in items]
        batch = self._collator(sequences)
        return batch, labels
