"""Variable-length batching with padding and attention masks."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from collections.abc import Iterator


class EventEpochCollator:
    """Pad variable-length epochs to create uniform batches.

    Pads shorter epochs with zeros and creates attention masks so the
    transformer ignores padding positions.
    """

    def __init__(self, max_length: int = 200) -> None:
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a list of epoch dicts into batched tensors.

        Each item in batch should have keys: 'eeg' (np.ndarray or Tensor),
        'event_id' (int), 'length' (int), 'pre_event_samples' (int).
        """
        lengths = [item["length"] for item in batch]
        max_len = min(max(lengths), self.max_length)

        padded_eeg = []
        attention_masks = []
        event_ids = []
        original_lengths = []
        pre_event_samples = []
        hed_tags: list[str | None] = []

        for item in batch:
            eeg = item["eeg"]
            if isinstance(eeg, np.ndarray):
                eeg = torch.from_numpy(eeg)
            t = eeg.shape[-1]

            if t > max_len:
                eeg = eeg[..., :max_len]
                t = max_len
            elif t < max_len:
                pad = torch.zeros(*eeg.shape[:-1], max_len - t, dtype=eeg.dtype)
                eeg = torch.cat([eeg, pad], dim=-1)

            mask = torch.zeros(max_len, dtype=torch.float32)
            mask[:t] = 1.0

            padded_eeg.append(eeg)
            attention_masks.append(mask)
            event_ids.append(item["event_id"])
            original_lengths.append(t)
            pre_event_samples.append(item.get("pre_event_samples", 0))
            hed_tags.append(item.get("hed_tag"))

        result = {
            "eeg": torch.stack(padded_eeg),
            "attention_mask": torch.stack(attention_masks),
            "event_ids": torch.tensor(event_ids, dtype=torch.long),
            "lengths": torch.tensor(original_lengths, dtype=torch.long),
            "pre_event_samples": torch.tensor(pre_event_samples, dtype=torch.long),
        }
        # Use pre-computed HED vectors if available, else fall back to strings
        hed_vectors = [item.get("hed_vector") for item in batch]
        if any(v is not None for v in hed_vectors):
            # Stack pre-computed vectors (zeros for epochs without HED)
            vocab_size = next(v.shape[0] for v in hed_vectors if v is not None)
            stacked = []
            for v in hed_vectors:
                if v is not None:
                    stacked.append(torch.from_numpy(v))
                else:
                    stacked.append(torch.zeros(vocab_size))
            result["hed_targets"] = torch.stack(stacked)
        elif any(t is not None for t in hed_tags):
            result["hed_tags"] = hed_tags  # type: ignore[assignment]
        # Always include string tags for analysis (non-tensor, skipped by .to())
        if any(t is not None for t in hed_tags):
            result["hed_tags"] = hed_tags  # type: ignore[assignment]
        return result


class BucketBatchSampler(Sampler[list[int]]):
    """Group dataset indices by epoch length to minimize padding waste.

    Assigns each sample to a bucket based on its length, then yields
    random batches from within each bucket.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_boundaries: list[int] | None = None,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_boundaries = sorted(bucket_boundaries or [30, 50, 80, 120, 200])
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Assign each index to a bucket
        self.buckets: dict[int, list[int]] = defaultdict(list)
        for idx, length in enumerate(lengths):
            bucket_id = self._get_bucket(length)
            self.buckets[bucket_id].append(idx)

    def _get_bucket(self, length: int) -> int:
        """Return the bucket index for a given length."""
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries)

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []

        for indices in self.buckets.values():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        total = 0
        for indices in self.buckets.values():
            n_batches = len(indices) / self.batch_size
            if self.drop_last:
                total += int(n_batches)
            else:
                total += math.ceil(n_batches)
        return total
