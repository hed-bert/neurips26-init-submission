"""Lightweight dataset loader for preprocessed HDF5 epoch files.

Reads from the output of preprocess_hbn.py (or any compatible preprocessor).
No MNE dependency; just HDF5 reads. Designed for maximum training throughput.

Preprocessing tiers:
    preprocessed (current): filter + re-ref + 129->64 interpolation
    preprocessed (planned): SPEED-EEG cleaning before interpolation

The preprocessed HDF5 format:
    Each file = one recording (subject + task). Contains:
        /epoch_00000/eeg       - float32 array (64, length)
        /epoch_00000.attrs:
            event_id           - int, hash of event value
            event_value        - str, original event label
            hed_tag            - str, HED annotation (if available)
            pre_event_samples  - int, samples before event onset
            length             - int, epoch length in samples
            onset_sample       - int, event onset in original recording

Why this exists (instead of BaseEEGDataset + MNE):
    BaseEEGDataset runs the full MNE pipeline (load BDF, filter, re-ref,
    epoch, harmonize) on first access, which takes ~3s per recording.
    With 26K+ recordings across 3K subjects, that is 20+ hours of
    preprocessing before training can begin.

    This loader reads pre-computed (64, T) arrays directly. First-access
    cost is zero; the preprocessing script handles the heavy lifting
    once, in parallel, across all CPU cores.

Future: high-quality preprocessed dataset
    For production-quality training at scale, the ideal pipeline is:
    1. SPEED-EEG artifact rejection on raw BDF
    2. High-pass filter (0.5 Hz) + notch filter (50/60 Hz)
    3. Average re-reference
    4. 129->64 channel interpolation (pre-computed matrix)
    5. Save as uncompressed HDF5 (mmap-friendly) or Arrow IPC
    6. Shard by release for distributed training (one shard per node)
    Each step runs once; training reads from the final output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PreprocessedEEGDataset(Dataset):
    """Read preprocessed HDF5 epoch files for training.

    Scans a directory of HDF5 files (one per recording), builds an
    index of all epochs, and serves them via __getitem__. No MNE,
    no preprocessing, just fast reads.

    EEG is z-scored per epoch (zero mean, unit variance per channel)
    so that reconstruction loss is scale-invariant.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        max_subjects: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.preprocessed_dir = Path(preprocessed_dir)
        self.normalize = normalize
        self._epoch_index: list[dict] = []
        self._build_index(max_subjects)

    def _build_index(self, max_subjects: int | None = None) -> None:
        """Scan all HDF5 files and build a flat epoch index."""
        h5_files = sorted(self.preprocessed_dir.glob("*.h5"))

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

        for h5_path in h5_files:
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
                                "subject_id": h5_path.stem.split("_")[0],
                                "session": None,
                            }
                        )
            except (OSError, KeyError) as e:
                logger.warning("Error reading %s: %s", h5_path, e)

        logger.info(
            "Built index: %d epochs from %d files",
            len(self._epoch_index),
            len(h5_files),
        )

    def __getitem__(self, idx: int) -> dict:
        """Load a single epoch, optionally z-scored per channel."""
        meta = self._epoch_index[idx]
        with h5py.File(meta["h5_path"], "r") as f:
            grp = f[meta["epoch_key"]]
            eeg = grp["eeg"][:].astype(np.float32)
            hed_tag = grp.attrs.get("hed_tag", None)
            # Read pre-computed HED vector if available
            hed_vector = grp["hed_vector"][:] if "hed_vector" in grp else None

        if self.normalize:
            mean = eeg.mean(axis=-1, keepdims=True)
            std = eeg.std(axis=-1, keepdims=True)
            eeg = (eeg - mean) / (std + 1e-8)

        result = {
            "eeg": eeg,
            "event_id": meta["event_id"],
            "length": meta["length"],
            "pre_event_samples": meta["pre_event_samples"],
            "subject_id": meta["subject_id"],
            "session": meta["session"],
            "hed_tag": str(hed_tag) if hed_tag is not None else None,
        }
        if hed_vector is not None:
            result["hed_vector"] = hed_vector.astype(np.float32)
        return result

    def __len__(self) -> int:
        return len(self._epoch_index)

    def get_lengths(self) -> list[int]:
        """Return epoch lengths for bucket batching."""
        return [m["length"] for m in self._epoch_index]
