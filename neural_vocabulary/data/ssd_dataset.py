"""SSD-streaming EEG dataset with true random access.

Reads directly from HDF5 files on SSD per batch. No RAM cache,
no background threads, no rotation. Builds a lightweight index
of (file_path, n_epochs) at init by scanning H5 headers (~2s for
26K files). DataLoader workers handle parallel I/O.

Each __getitem__ returns a multi-epoch sequence from one recording,
compatible with PackedSequenceCollator.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# HBN task categories for Gate 2 (passive vs active pretraining).
PASSIVE_TASKS: set[str] = {
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "FunwithFractals",
    "RestingState",
    "ThePresent",
    "surroundSupp",
}
ACTIVE_TASKS: set[str] = {
    "contrastChangeDetection",
    "seqLearning6target",
    "seqLearning8target",
    "symbolSearch",
}
ALL_TASKS: set[str] = PASSIVE_TASKS | ACTIVE_TASKS

# Canonical task-to-index mapping for task_codes prediction target (Gate 1a).
# Sorted alphabetically for determinism.
TASK_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(sorted(ALL_TASKS))}
NUM_TASKS: int = len(TASK_TO_IDX)


def _extract_task(h5_path: Path) -> str:
    """Extract task name from H5 filename.

    Filename format: {subject_id}_{task_name}[_run-N].h5
    """
    parts = h5_path.stem.split("_", 1)
    if len(parts) < 2:
        return ""
    # Remove _run-N suffix if present
    task_part = parts[1]
    if "_run-" in task_part:
        task_part = task_part.split("_run-")[0]
    return task_part


class SSDStreamingDataset(Dataset):
    """Map-style dataset that reads EEG epochs from SSD on demand.

    Index phase (~2s): scans H5 headers to build a recording index.
    Training phase: each __getitem__ opens one H5 file, reads a
    contiguous window of epochs, normalizes, and returns a dict
    compatible with PackedSequenceCollator.

    Usage:
        dataset = SSDStreamingDataset("/path/to/preprocessed")
        loader = DataLoader(dataset, batch_size=128, shuffle=True,
                            num_workers=4, collate_fn=collator)
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        epochs_per_sequence: int = 16,
        normalize: bool = True,
        max_channels: int = 64,
        max_epoch_len: int = 220,
        task_filter: str = "all",
    ) -> None:
        self.preprocessed_dir = Path(preprocessed_dir)
        self.epochs_per_sequence = epochs_per_sequence
        self.normalize = normalize
        self.max_channels = max_channels
        self.max_epoch_len = max_epoch_len
        self.task_filter = task_filter

        # Build recording index: [(file_path, n_epochs)]
        self._index: list[tuple[Path, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Scan H5 headers to build recording index."""
        h5_files = sorted(self.preprocessed_dir.glob("*.h5"))
        h5_files = [f for f in h5_files if f.name != "hed_vectorizer.pt"]

        # Apply task filter
        if self.task_filter == "passive":
            allowed = PASSIVE_TASKS
        elif self.task_filter == "active":
            allowed = ACTIVE_TASKS
        else:
            allowed = None  # all tasks

        skipped = 0
        task_filtered = 0
        for h5_path in h5_files:
            # Task filter check (before opening the file for speed)
            if allowed is not None:
                task = _extract_task(h5_path)
                if task not in allowed:
                    task_filtered += 1
                    continue

            try:
                with h5py.File(h5_path, "r") as f:
                    n_epochs = int(f.attrs.get("n_epochs", 0))
                    if n_epochs >= 2:  # need at least 2 for packed sequences
                        self._index.append((h5_path, n_epochs))
                    else:
                        skipped += 1
            except (OSError, KeyError):
                skipped += 1

        total_epochs = sum(n for _, n in self._index)
        msg = "SSD index: %d recordings, %d total epochs, %d skipped"
        args: list[object] = [len(self._index), total_epochs, skipped]
        if task_filtered > 0:
            msg += ", %d filtered by task=%s"
            args.extend([task_filtered, self.task_filter])
        logger.info(msg, *args)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        """Load a multi-epoch sequence from one recording.

        Opens the H5 file, reads a random contiguous window of epochs,
        normalizes, and returns a dict for PackedSequenceCollator.
        """
        h5_path, n_epochs = self._index[idx]
        n_to_read = min(self.epochs_per_sequence, n_epochs)

        # Random contiguous window within the recording
        rng = np.random.default_rng()
        start = rng.integers(max(1, n_epochs - n_to_read + 1))

        eeg_epochs = []
        pad_masks: list[np.ndarray] = []
        hed_vectors = []
        lengths = []
        pre_events = []

        with h5py.File(h5_path, "r") as f:
            for i in range(start, start + n_to_read):
                grp_name = f"epoch_{i:05d}"
                if grp_name not in f:
                    continue
                grp = f[grp_name]
                eeg = grp["eeg"][:].astype(np.float32)

                # Per-epoch z-score normalization. Active only when normalize
                # is True; train_eventformer.py sets normalize=False for the
                # parallel encoder (used by V9 Tier 1 g3b) so the parallel
                # path relies on encoder InputNorm and skips this branch.
                if self.normalize:
                    mean = eeg.mean(axis=-1, keepdims=True)
                    std = eeg.std(axis=-1, keepdims=True)
                    eeg = (eeg - mean) / (std + 1e-8)

                # Truncate to max dims
                ch = min(eeg.shape[0], self.max_channels)
                t = min(eeg.shape[1], self.max_epoch_len)
                eeg = eeg[:ch, :t]

                # V9 preprocessing stores a pad_mask; older formats don't.
                # Default to all-valid when the dataset is missing.
                if "pad_mask" in grp:
                    pad_mask = grp["pad_mask"][:].astype(np.uint8)[:t]
                else:
                    pad_mask = np.ones(t, dtype=np.uint8)

                eeg_epochs.append(eeg)
                pad_masks.append(pad_mask)
                lengths.append(t)
                pre_events.append(int(grp.attrs.get("pre_event_samples", 0)))

                # HED vector
                if "hed_vector" in grp:
                    hed_vectors.append(grp["hed_vector"][:].astype(np.float32))
                else:
                    hed_vectors.append(None)

        if not eeg_epochs:
            # Fallback: return minimal valid sequence
            eeg_epochs = [np.zeros((self.max_channels, 1), dtype=np.float32)] * 2
            pad_masks = [np.ones(1, dtype=np.uint8)] * 2
            lengths = [1, 1]
            pre_events = [0, 0]
            hed_vectors = [None, None]

        # Extract task name for task_codes mode
        task_name = _extract_task(h5_path)

        return {
            "eeg_epochs": eeg_epochs,
            "pad_masks": pad_masks,
            "hed_vectors": hed_vectors,
            "lengths": lengths,
            "pre_event_samples": pre_events,
            "n_epochs": len(eeg_epochs),
            "task_name": task_name,
        }

    @property
    def n_loaded_epochs(self) -> int:
        """Total epochs across all recordings (for compatibility)."""
        return sum(n for _, n in self._index)
