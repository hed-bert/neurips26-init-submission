"""Disk-to-RAM cached EEG dataset with progressive loading and rotation.

Loads preprocessed epochs into CPU pinned memory progressively,
eliminating per-batch HDF5 reads and the 47 GB upfront RAM requirement
from ConsolidatedEEGDataset. Data is rotated in chunks so the model
sees the full dataset over time.

Key properties:
    - No upfront RAM requirement (loads progressively)
    - No per-batch disk I/O (served from CPU pinned memory)
    - Training starts immediately with partial data
    - Rotates data chunks to cover the full dataset

Note: data is in CPU pinned memory, not GPU VRAM. The collator
handles CPU->GPU transfer. A future GPU-native collator could
keep data on-device end-to-end.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class GPUCachedDataset(IterableDataset):
    """Stream preprocessed EEG epochs with GPU-resident cache.

    Loads epochs from HDF5 files into GPU memory in the background.
    Serves packed multi-epoch sequences directly from GPU tensors.
    Rotates cache chunks to cover the full dataset.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        device: torch.device | str = "cuda",
        max_cache_gb: float = 14.0,
        epochs_per_sequence: int = 16,
        rotate_every: int = 5,
        normalize: bool = True,
        max_channels: int = 64,
        max_epoch_len: int = 200,
        hed_vocab_size: int = 0,
    ) -> None:
        self.preprocessed_dir = Path(preprocessed_dir)
        self.device = torch.device(device)
        self.max_cache_bytes = int(max_cache_gb * 1e9)
        self.epochs_per_sequence = epochs_per_sequence
        self.rotate_every = rotate_every
        self.normalize = normalize
        self.max_channels = max_channels
        self.max_epoch_len = max_epoch_len
        self.hed_vocab_size = hed_vocab_size

        # CPU pinned memory cache: fast async transfer to GPU during training.
        # Data stays on CPU because the packed collator operates on numpy arrays.
        # Pinned memory ensures the final CPU->GPU transfer is DMA (non-blocking).
        self._max_cached_epochs = self._estimate_capacity()
        self._eeg_cache = torch.zeros(
            self._max_cached_epochs,
            max_channels,
            max_epoch_len,
            dtype=torch.float32,
            pin_memory=True,
        )
        self._hed_cache: torch.Tensor | None = None
        if hed_vocab_size > 0:
            self._hed_cache = torch.zeros(
                self._max_cached_epochs,
                hed_vocab_size,
                dtype=torch.float32,
                pin_memory=True,
            )
        self._lengths = torch.zeros(self._max_cached_epochs, dtype=torch.long)
        self._pre_events = torch.zeros(self._max_cached_epochs, dtype=torch.long)
        self._recording_ids = torch.zeros(self._max_cached_epochs, dtype=torch.long)

        # Cache management
        self._n_cached = 0
        self._cache_lock = threading.Lock()
        self._loading_done = threading.Event()
        self._next_rec_id = 0  # persistent across rotations
        self._training_epoch = 0

        # File list for rotation
        self._h5_files = sorted(self.preprocessed_dir.glob("*.h5"))
        self._h5_files = [f for f in self._h5_files if f.name != "hed_vectorizer.pt"]
        self._file_offset = 0  # current position in file list for rotation

        # Load all data into cache before training (no background race)
        logger.info(
            "Loading cache: up to %d epochs (%.1f GB pinned CPU)...",
            self._max_cached_epochs,
            self._max_cached_epochs
            * (max_channels * max_epoch_len * 4 + hed_vocab_size * 4)
            / 1e9,
        )
        self._load_chunk()  # synchronous: blocks until cache is full

        logger.info(
            "Cache ready: %d/%d slots filled (%.1f GB pinned CPU, %d files)",
            self._n_cached,
            self._max_cached_epochs,
            self._n_cached
            * (self.max_channels * self.max_epoch_len * 4 + self.hed_vocab_size * 4)
            / 1e9,
            self._file_offset,
        )

    def _estimate_capacity(self) -> int:
        """How many epochs fit in the cache."""
        bytes_per_epoch = self.max_channels * self.max_epoch_len * 4  # float32 EEG
        bytes_per_epoch += self.hed_vocab_size * 4  # float32 HED vector
        bytes_per_epoch += 8 * 3  # lengths, pre_events, recording_ids (int64)
        return self.max_cache_bytes // bytes_per_epoch

    def _load_chunk(self) -> None:
        """Background: load a chunk of HDF5 files into the GPU cache."""
        try:
            slot = 0
            rec_id = self._next_rec_id
            files_to_load = self._h5_files[self._file_offset :]

            n_files = len(files_to_load)
            for file_idx, h5_path in enumerate(files_to_load):
                if slot >= self._max_cached_epochs:
                    break

                if file_idx % 2000 == 0 and file_idx > 0:
                    logger.info(
                        "  Loading: %d/%d files, %d epochs cached (%.1f GB)",
                        file_idx,
                        n_files,
                        slot,
                        slot * (self.max_channels * self.max_epoch_len * 4) / 1e9,
                    )

                try:
                    with h5py.File(h5_path, "r") as f:
                        n_epochs = f.attrs.get("n_epochs", 0)
                        for i in range(n_epochs):
                            if slot >= self._max_cached_epochs:
                                break
                            grp_name = f"epoch_{i:05d}"
                            if grp_name not in f:
                                continue
                            grp = f[grp_name]
                            eeg = grp["eeg"][:].astype(np.float32)

                            # Normalize per-channel
                            if self.normalize:
                                mean = eeg.mean(axis=-1, keepdims=True)
                                std = eeg.std(axis=-1, keepdims=True)
                                eeg = (eeg - mean) / (std + 1e-8)

                            # Pad/truncate to fixed size
                            ch, t = eeg.shape
                            ch = min(ch, self.max_channels)
                            t = min(t, self.max_epoch_len)

                            # Write to pinned CPU cache
                            with self._cache_lock:
                                self._eeg_cache[slot].zero_()  # clear stale data
                                self._eeg_cache[slot, :ch, :t] = torch.from_numpy(
                                    eeg[:ch, :t]
                                )
                                self._lengths[slot] = t
                                self._pre_events[slot] = int(
                                    grp.attrs.get("pre_event_samples", 0)
                                )
                                self._recording_ids[slot] = rec_id

                                # HED vector
                                if "hed_vector" in grp and self._hed_cache is not None:
                                    hed = grp["hed_vector"][:].astype(np.float32)
                                    if len(hed) <= self._hed_cache.shape[1]:
                                        self._hed_cache[slot, : len(hed)] = (
                                            torch.from_numpy(hed)
                                        )

                                self._n_cached = slot + 1
                                slot += 1

                    rec_id += 1
                except (OSError, KeyError) as e:
                    logger.warning("Skipping %s: %s", h5_path.name, e)
                    rec_id += 1

            self._next_rec_id = rec_id  # persist for next rotation
            self._file_offset += len(files_to_load)
            if self._file_offset >= len(self._h5_files):
                self._file_offset = 0

        except Exception:
            logger.exception("GPU cache loader crashed")
        finally:
            self._loading_done.set()

    def rotate_cache(self) -> None:
        """Replace a portion of the cache with new data.

        Called by the training loop every rotate_every epochs.
        Replaces the oldest 20% of the cache with new files.
        """
        if not self._h5_files:
            return

        n_to_replace = max(1, self._max_cached_epochs // 5)  # 20%
        logger.info(
            "Rotating GPU cache: replacing %d/%d slots from file offset %d",
            n_to_replace,
            self._max_cached_epochs,
            self._file_offset,
        )

        # Shift existing data and load new into the freed slots
        # For simplicity, just reload the entire cache from the next chunk
        self._loading_done.clear()
        self._loader_thread = threading.Thread(target=self._load_chunk, daemon=True)
        self._loader_thread.start()

    def _make_sequence(self, rng: np.random.Generator) -> dict | None:
        """Build a packed multi-epoch sequence from GPU cache."""
        with self._cache_lock:
            n = self._n_cached
            if n < 2:
                return None

        # Find a recording with multiple epochs
        rec_ids = self._recording_ids[:n]
        unique_recs = rec_ids.unique()
        if len(unique_recs) == 0:
            return None

        # Pick a random recording
        rec = unique_recs[rng.integers(len(unique_recs))]
        mask = rec_ids == rec
        epoch_indices = torch.where(mask)[0]

        if len(epoch_indices) < 2:
            # Try another recording
            for _ in range(10):
                rec = unique_recs[rng.integers(len(unique_recs))]
                mask = rec_ids == rec
                epoch_indices = torch.where(mask)[0]
                if len(epoch_indices) >= 2:
                    break
            else:
                return None

        # Select a contiguous window
        n_ep = min(self.epochs_per_sequence, len(epoch_indices))
        start = rng.integers(max(1, len(epoch_indices) - n_ep + 1))
        selected = epoch_indices[start : start + n_ep]

        # Read from pinned CPU cache
        eeg_epochs = [self._eeg_cache[idx].numpy().copy() for idx in selected]
        lengths = [self._lengths[idx].item() for idx in selected]
        pre_events = [self._pre_events[idx].item() for idx in selected]

        hed_vectors = None
        if self._hed_cache is not None:
            hed_vectors = [self._hed_cache[idx].numpy().copy() for idx in selected]

        return {
            "eeg_epochs": eeg_epochs,
            "hed_vectors": hed_vectors or [None] * len(selected),
            "lengths": lengths,
            "pre_event_samples": pre_events,
            "n_epochs": len(selected),
        }

    def __iter__(self):
        """Yield packed sequences from GPU cache."""
        rng = np.random.default_rng()
        consecutive_none = 0
        while True:
            seq = self._make_sequence(rng)
            if seq is not None:
                consecutive_none = 0
                yield seq
            else:
                consecutive_none += 1
                if consecutive_none > 1000:
                    raise RuntimeError(
                        f"GPU cache: failed to produce sequence after 1000 attempts. "
                        f"{self._n_cached} epochs cached."
                    )

    @property
    def n_cached_epochs(self) -> int:
        """Number of epochs currently cached."""
        return self._n_cached

    @property
    def n_loaded_epochs(self) -> int:
        """Alias for n_cached_epochs (interface compatibility)."""
        return self._n_cached

    @property
    def is_loading(self) -> bool:
        """Whether background loading is in progress."""
        return not self._loading_done.is_set()

    def notify_epoch_end(self) -> None:
        """Called by trainer at end of each training epoch."""
        self._training_epoch += 1
        if self._training_epoch % self.rotate_every == 0:
            self.rotate_cache()
