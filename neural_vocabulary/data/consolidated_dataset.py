"""Consolidated in-memory EEG dataset with multi-epoch sequence packing.

Loads the entire preprocessed dataset into RAM as contiguous tensors.
Packs multiple consecutive epochs from the same recording into single
training sequences for efficient transformer processing.

Why this exists:
    The per-file HDF5 approach has two fatal bottlenecks:
    1. I/O: 4096 file open/seek/read/close per batch (~200ms vs <10ms GPU)
    2. Short sequences: 2 patches per epoch = 4 transformer tokens

    This loader solves both:
    - All data in RAM as contiguous tensors (zero I/O during training)
    - Consecutive epochs packed into sequences of 16-32 events
    - Each sequence = 64-128 transformer tokens (proper GPU saturation)

    With 64-token sequences and batch=256, the GPU processes 16K tokens/step
    with meaningful self-attention, achieving 70-90% utilization.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ConsolidatedEEGDataset(Dataset):
    """In-memory dataset with multi-epoch sequence packing.

    Loads all preprocessed HDF5 files into RAM, groups epochs by
    recording (subject + task), and packs consecutive epochs into
    fixed-length sequences for transformer training.

    Each training sample is a packed sequence of N consecutive epochs:
        [EVT_1] patches_1... [EVT_2] patches_2... ...

    The model sees longer sequences (64-128 tokens) and can learn
    cross-epoch temporal dynamics.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        epochs_per_sequence: int = 16,
        max_subjects: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.preprocessed_dir = Path(preprocessed_dir)
        self.epochs_per_sequence = epochs_per_sequence
        self.normalize = normalize

        # In-memory storage
        self._eeg_data: list[np.ndarray] = []  # per-epoch EEG arrays
        self._hed_vectors: list[np.ndarray | None] = []
        self._lengths: list[int] = []
        self._pre_event_samples: list[int] = []
        self._recording_ids: list[int] = []  # which recording each epoch belongs to

        # Packed sequence index
        self._sequences: list[list[int]] = []  # each entry = list of epoch indices

        self._load_all(max_subjects)
        self._build_sequences()

    def _load_all(self, max_subjects: int | None = None) -> None:
        """Load all HDF5 files into RAM."""
        h5_files = sorted(self.preprocessed_dir.glob("*.h5"))
        # Exclude vectorizer file
        h5_files = [f for f in h5_files if f.name != "hed_vectorizer.pt"]

        if max_subjects is not None:
            seen: set[str] = set()
            filtered = []
            for f in h5_files:
                subject = f.stem.split("_")[0]
                if subject not in seen:
                    if len(seen) >= max_subjects:
                        continue
                    seen.add(subject)
                filtered.append(f)
            h5_files = filtered

        recording_id = 0
        n_files = len(h5_files)
        loaded = 0

        for h5_path in h5_files:
            try:
                with h5py.File(h5_path, "r") as f:
                    n_epochs = f.attrs.get("n_epochs", 0)
                    for i in range(n_epochs):
                        grp_name = f"epoch_{i:05d}"
                        if grp_name not in f:
                            continue
                        grp = f[grp_name]
                        eeg = grp["eeg"][:].astype(np.float32)
                        length = int(grp.attrs.get("length", eeg.shape[-1]))
                        pre_event = int(grp.attrs.get("pre_event_samples", 0))

                        hed_vec = None
                        if "hed_vector" in grp:
                            hed_vec = grp["hed_vector"][:].astype(np.float32)

                        self._eeg_data.append(eeg)
                        self._hed_vectors.append(hed_vec)
                        self._lengths.append(length)
                        self._pre_event_samples.append(pre_event)
                        self._recording_ids.append(recording_id)

                recording_id += 1
                loaded += 1
            except (OSError, KeyError) as e:
                logger.warning("Error reading %s: %s", h5_path.name, e)
                recording_id += 1

            if loaded % 2000 == 0:
                logger.info(
                    "Loaded %d/%d files (%d epochs)",
                    loaded,
                    n_files,
                    len(self._eeg_data),
                )

        logger.info(
            "Loaded %d epochs from %d recordings into RAM (%.1f GB)",
            len(self._eeg_data),
            recording_id,
            sum(e.nbytes for e in self._eeg_data) / 1e9,
        )

    def _build_sequences(self) -> None:
        """Group epochs by recording and pack into sequences."""
        # Group epoch indices by recording
        from collections import defaultdict

        recording_epochs: dict[int, list[int]] = defaultdict(list)
        for idx, rec_id in enumerate(self._recording_ids):
            recording_epochs[rec_id].append(idx)

        # Pack consecutive epochs into sequences
        n = self.epochs_per_sequence
        for _rec_id, epoch_indices in recording_epochs.items():
            # Epochs within a recording are already in temporal order
            for start in range(0, len(epoch_indices), n):
                seq = epoch_indices[start : start + n]
                if len(seq) >= 2:  # need at least 2 epochs per sequence
                    self._sequences.append(seq)

        logger.info(
            "Built %d sequences of up to %d epochs each",
            len(self._sequences),
            self.epochs_per_sequence,
        )

    def __getitem__(self, idx: int) -> dict:
        """Return a packed multi-epoch sequence.

        Returns:
            dict with:
                eeg_epochs: list of (64, T_i) arrays (variable length per epoch)
                hed_vectors: list of (vocab_size,) arrays or None
                lengths: list of int epoch lengths
                pre_event_samples: list of int
                n_epochs: number of epochs in this sequence
        """
        epoch_indices = self._sequences[idx]

        eeg_epochs = []
        hed_vectors = []
        lengths = []
        pre_events = []

        for ei in epoch_indices:
            eeg = self._eeg_data[ei].copy()
            if self.normalize:
                mean = eeg.mean(axis=-1, keepdims=True)
                std = eeg.std(axis=-1, keepdims=True)
                eeg = (eeg - mean) / (std + 1e-8)

            eeg_epochs.append(eeg)
            hed_vectors.append(self._hed_vectors[ei])
            lengths.append(self._lengths[ei])
            pre_events.append(self._pre_event_samples[ei])

        return {
            "eeg_epochs": eeg_epochs,
            "hed_vectors": hed_vectors,
            "lengths": lengths,
            "pre_event_samples": pre_events,
            "n_epochs": len(epoch_indices),
        }

    def __len__(self) -> int:
        return len(self._sequences)

    def get_sequence_lengths(self) -> list[int]:
        """Return total sample count per sequence for bucket batching."""
        return [sum(self._lengths[ei] for ei in seq) for seq in self._sequences]
