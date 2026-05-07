"""Collator for multi-epoch packed sequences.

Takes variable-length sequences of multiple epochs and produces
batched tensors suitable for the HEDBERT's packed forward pass.

Pre-computes the token insertion plan (where [EVT] tokens go, timestamps,
positions) so the model forward pass is loop-free and GPU-efficient.

preprocessed: the [SEP] end-of-epoch token was removed because the epoch
window is now a fixed deterministic function of the event type (see
preprocess_hbn.LEGACY_WINDOW_SPEC). With fixed windows [SEP] carries no
discriminative signal and previously leaked epoch duration through
bidirectional attention.
"""

from __future__ import annotations

import numpy as np
import torch


class PackedSequenceCollator:
    """Collate multi-epoch sequences into batched tensors.

    Pre-computes the complete token insertion plan so forward_packed
    can use vectorized scatter operations instead of Python loops.
    """

    def __init__(
        self,
        max_total_length: int = 3000,
        total_stride: int = 75,
        sfreq: float = 100.0,
        hed_collapse_map: torch.Tensor | None = None,
        task_code_mode: bool = False,
    ) -> None:
        self.max_total_length = max_total_length
        self.total_stride = total_stride
        self.hed_collapse_map = hed_collapse_map  # (collapsed_vocab, source_vocab)
        self.sfreq = sfreq
        self.task_code_mode = task_code_mode

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list]:
        """Collate a batch of multi-epoch sequences.

        Returns a dict with pre-computed token insertion plan:
            eeg: (B, 64, total_T) concatenated and padded EEG
            attention_mask: (B, total_T) float mask
            # Token insertion plan (all integer tensors):
            patch_to_token: (B, max_tokens) index mapping patches to token positions
            token_types: (B, max_tokens) 0=pad, 1=EVT, 2=patch
            timestamps_ms: (B, max_tokens) pre-computed timestamps
            evt_token_positions: (B, max_n_epochs) positions of [EVT] in token sequence
            evt_epoch_indices: (B, max_n_epochs) original epoch index per [EVT]
            n_valid_evts: (B,) number of valid [EVT] tokens per sample
            n_tokens: (B,) total tokens per sample (for attention mask)
            hed_targets_packed: (B, max_n_epochs, vocab_size) if HED available
        """
        all_eeg = []
        all_masks = []
        all_lengths = []
        all_n_epochs = []

        # Per-sample token plans
        all_token_types: list[list[int]] = []
        all_timestamps: list[list[float]] = []
        all_patch_source: list[
            list[int]
        ] = []  # which patch index this token reads from
        all_evt_positions: list[list[int]] = []
        all_evt_epoch_indices: list[list[int]] = []

        # HED
        all_hed: list[torch.Tensor] = []

        max_epochs_in_batch = max(item["n_epochs"] for item in batch)

        # Pre-scan to discover vocab_size before main loop so every item
        # (including those before the first HED item) gets an entry in all_hed.
        vocab_size = None
        for item in batch:
            hv = item["hed_vectors"]
            if hv:
                for v in hv:
                    if v is not None:
                        vocab_size = v.shape[0]
                        break
            if vocab_size is not None:
                break

        for item in batch:
            eeg_epochs = item["eeg_epochs"]
            hed_vecs = item["hed_vectors"]
            epoch_lengths = item["lengths"]
            #  datasets return per-epoch pad_masks; pre- datasets do not.
            # Fall back to all-ones when absent so legacy h5 files still work.
            pad_mask_list = item.get("pad_masks")
            if pad_mask_list is None:
                pad_mask_list = [np.ones(ln, dtype=np.uint8) for ln in epoch_lengths]

            # Concatenate epochs along time
            concat = np.concatenate(eeg_epochs, axis=-1)
            concat_mask = np.concatenate(pad_mask_list, axis=-1)
            total_len = concat.shape[-1]
            if total_len > self.max_total_length:
                concat = concat[:, : self.max_total_length]
                concat_mask = concat_mask[: self.max_total_length]
                total_len = self.max_total_length

            all_eeg.append(torch.from_numpy(concat))
            all_lengths.append(total_len)

            # Compute epoch boundaries
            boundaries: list[tuple[int, int]] = []
            pos = 0
            for elen in epoch_lengths:
                end = min(pos + elen, total_len)
                if pos < total_len:
                    boundaries.append((pos, end))
                pos += elen

            # Build token insertion plan
            stride = self.total_stride
            sfreq = self.sfreq
            ms_per_patch = (stride / sfreq) * 1000.0

            token_types: list[int] = []
            timestamps: list[float] = []
            patch_sources: list[int] = []  # -1 for non-patch tokens
            evt_positions: list[int] = []
            evt_epoch_indices: list[int] = []

            pre_events = item.get("pre_event_samples", [0] * len(epoch_lengths))

            for epoch_idx, (epoch_start, epoch_end) in enumerate(boundaries):
                patch_start = epoch_start // stride
                patch_end = min(epoch_end // stride, total_len // stride)
                if patch_end <= patch_start:
                    continue

                n_patches_epoch = patch_end - patch_start

                # Event-relative timestamps (matching _compute_timestamps_ms)
                pre_event = pre_events[epoch_idx] if epoch_idx < len(pre_events) else 0
                event_onset_patch = pre_event / stride

                # [EVT] token at 0ms (event onset)
                evt_positions.append(len(token_types))
                evt_epoch_indices.append(epoch_idx)
                token_types.append(1)  # EVT
                timestamps.append(0.0)
                patch_sources.append(-1)

                # Patch tokens with event-relative timestamps
                for p in range(n_patches_epoch):
                    relative_patch = p - event_onset_patch
                    timestamps.append(relative_patch * ms_per_patch)
                    token_types.append(2)  # patch
                    patch_sources.append(patch_start + p)

            if not token_types:
                # Fallback: single EVT token
                evt_positions.append(0)
                evt_epoch_indices.append(0)
                token_types.append(1)
                timestamps.append(0.0)
                patch_sources.append(-1)

            all_token_types.append(token_types)
            all_timestamps.append(timestamps)
            all_patch_source.append(patch_sources)
            all_evt_positions.append(evt_positions)
            all_evt_epoch_indices.append(evt_epoch_indices)
            all_n_epochs.append(len(evt_positions))

            # EEG mask: honors per-sample padding from the preprocessing
            # (preprocessed windows may be padded when an epoch hits the end of a
            # recording). Cast to float to match the existing contract.
            all_masks.append(
                torch.from_numpy(concat_mask[:total_len].astype(np.float32))
            )

            # HED vectors
            if vocab_size is not None:
                padded_hed = np.zeros(
                    (max_epochs_in_batch, vocab_size), dtype=np.float32
                )
                if hed_vecs:
                    for i, v in enumerate(hed_vecs[:max_epochs_in_batch]):
                        if v is not None:
                            padded_hed[i] = v
                all_hed.append(torch.from_numpy(padded_hed))

        # Pad EEG to max length
        max_eeg_len = min(max(all_lengths), self.max_total_length)
        n_channels = all_eeg[0].shape[0]
        batch_size = len(batch)

        padded_eeg = torch.zeros(batch_size, n_channels, max_eeg_len)
        padded_eeg_mask = torch.zeros(batch_size, max_eeg_len)
        for b in range(batch_size):
            t = min(all_eeg[b].shape[-1], max_eeg_len)
            padded_eeg[b, :, :t] = all_eeg[b][:, :t]
            padded_eeg_mask[b, :t] = all_masks[b][:t]

        # Pad token plans to max token count
        max_tokens = max(len(tt) for tt in all_token_types)
        max_n_evts = max(len(ep) for ep in all_evt_positions)

        token_types_t = torch.zeros(batch_size, max_tokens, dtype=torch.long)
        timestamps_t = torch.zeros(batch_size, max_tokens)
        patch_source_t = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
        evt_positions_t = torch.zeros(batch_size, max_n_evts, dtype=torch.long)
        evt_epoch_indices_t = torch.zeros(batch_size, max_n_evts, dtype=torch.long)
        n_valid_evts_t = torch.zeros(batch_size, dtype=torch.long)
        n_tokens_t = torch.zeros(batch_size, dtype=torch.long)

        for b in range(batch_size):
            n_tok = len(all_token_types[b])
            token_types_t[b, :n_tok] = torch.tensor(all_token_types[b])
            timestamps_t[b, :n_tok] = torch.tensor(all_timestamps[b])
            patch_source_t[b, :n_tok] = torch.tensor(all_patch_source[b])
            n_evts = len(all_evt_positions[b])
            evt_positions_t[b, :n_evts] = torch.tensor(all_evt_positions[b])
            evt_epoch_indices_t[b, :n_evts] = torch.tensor(all_evt_epoch_indices[b])
            n_valid_evts_t[b] = n_evts
            n_tokens_t[b] = n_tok

        result: dict[str, torch.Tensor | list] = {
            "eeg": padded_eeg,
            "attention_mask": padded_eeg_mask,
            "token_types": token_types_t,
            "timestamps_ms": timestamps_t,
            "patch_source": patch_source_t,
            "evt_positions": evt_positions_t,
            "evt_epoch_indices": evt_epoch_indices_t,
            "n_valid_evts": n_valid_evts_t,
            "n_tokens": n_tokens_t,
            "lengths": torch.tensor(all_lengths, dtype=torch.long),
        }

        if all_hed:
            hed_packed = torch.stack(all_hed)
            # Apply vocab collapse mapping if set (converts pre-computed
            # source-vocab vectors to collapsed-vocab vectors)
            if self.hed_collapse_map is not None:
                # hed_packed: (B, max_epochs, source_vocab)
                # collapse_map: (collapsed_vocab, source_vocab)
                # result: (B, max_epochs, collapsed_vocab), binary OR
                b, e, _ = hed_packed.shape
                flat = hed_packed.reshape(b * e, -1)  # (B*E, source_vocab)
                collapsed = (flat @ self.hed_collapse_map.T > 0).float()
                hed_packed = collapsed.reshape(b, e, -1)
            result["hed_targets_packed"] = hed_packed

        # Task codes: (B, max_n_epochs) long tensor with task indices
        if self.task_code_mode:
            from neural_vocabulary.data.ssd_dataset import TASK_TO_IDX

            task_codes = torch.full((batch_size, max_n_evts), -1, dtype=torch.long)
            for b_idx, item in enumerate(batch):
                task_name = item.get("task_name", "")
                task_idx = TASK_TO_IDX.get(task_name, -1)
                n_evts = all_n_epochs[b_idx]
                # All epochs in one recording share the same task
                task_codes[b_idx, :n_evts] = task_idx
            result["task_codes_packed"] = task_codes

        return result
