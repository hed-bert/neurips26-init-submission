"""Joint HEDBERT model with an [EVT] token.

Assembles all components into the complete model:
    encoder -> [EVT] insertion -> PE -> transformer -> strip token -> decoder

The [EVT] token is a learnable embedding prepended to each epoch's patch
sequence. It IS the event onset (timestamp 0ms). After the transformer,
the [EVT] output embedding is the epoch-level representation used for
HED tag prediction.

Sequence flow:
    [EVT] patch_-1 patch_0 patch_1 ... patch_N

preprocessed: the previous [SEP] token was removed. With fixed condition-
matched epoch windows (see preprocess_hbn.LEGACY_WINDOW_SPEC) the epoch
duration is a deterministic function of epoch type and therefore carries
no discriminative signal. A [SEP] token whose only job was to mark the
epoch boundary would leak that fixed duration back through bidirectional
attention and add no information.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from neural_vocabulary.models.decoder import EEGDecoder
from neural_vocabulary.models.multiscale_encoder import (
    MultiScaleEncoder,
    ParallelMultiScaleEncoder,
)
from neural_vocabulary.models.positional_encoding import (
    ContinuousTimePositionalEncoding,
)
from neural_vocabulary.models.transformer import (
    HAS_FLASH_ATTN_VARLEN,
    HEDBERTTransformer,
)

if TYPE_CHECKING:
    from neural_vocabulary.configs import HEDBERTConfig

logger = logging.getLogger(__name__)


class _ZeroGradAtEVT(torch.autograd.Function):
    """Zero out gradients at EVT positions during backward pass.

    Applied at the transformer INPUT when detach_evt_from_recon is enabled.
    This blocks all gradient (recon and HED) from reaching the EVT
    nn.Parameter. HED gradient still reaches transformer weights through
    position-0 output, so the transformer learns to produce meaningful
    EVT representations.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        tensor: torch.Tensor,
        evt_mask: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(evt_mask)
        return tensor

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (evt_mask,) = ctx.saved_tensors  # type: ignore[attr-defined]
        return grad_output * (~evt_mask).unsqueeze(-1), None


class HEDBERT(nn.Module):
    """Joint HEDBERT model with an [EVT] special token.

    Architecture:
        1. MultiScaleEncoder: raw EEG -> patch embeddings
        2. Token insertion: prepend [EVT]
        3. ContinuousTimePositionalEncoding: event-relative timestamps
        4. HEDBERTTransformer: contextualized representations
        5. Output extraction: [EVT] for HED prediction, patches for decoder
        6. EEGDecoder: patch embeddings -> reconstructed EEG

    The [EVT] embedding at position 0 captures event-level semantics via
    attention over the full epoch.
    """

    def __init__(self, config: HEDBERTConfig) -> None:
        super().__init__()
        self.config = config

        # Core components: dispatch encoder based on config.encoder_type
        if config.encoder_type == "parallel":
            self.encoder = ParallelMultiScaleEncoder(config)
        else:
            self.encoder = MultiScaleEncoder(config)
        self.pe = ContinuousTimePositionalEncoding(
            embed_dim=config.embed_dim,
            dropout=config.dropout,
        )
        self.transformer = HEDBERTTransformer(config)
        self.decoder = EEGDecoder(config, patch_size=self.encoder.total_stride)

        # Special token embedding
        self.evt_embedding = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)

        # Learned [MASK] embedding for masked reconstruction (MAE-style)
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)

        # preprocessed: optional learnable [EVT] timestamp.
        # Stored as an unconstrained logit; the effective timestamp is
        # sigmoid(logit) * evt_time_ms_max so it stays in [0, max] ms.
        # Init: invert the sigmoid at evt_time_ms_init / evt_time_ms_max.
        if config.learnable_evt_time_ms:
            init_frac = max(
                1e-4, min(1.0 - 1e-4, config.evt_time_ms_init / config.evt_time_ms_max)
            )
            init_logit = float(torch.logit(torch.tensor(init_frac)))
            self._evt_time_logit = nn.Parameter(torch.tensor(init_logit))
        else:
            self._evt_time_logit = None

        if not 0.0 <= config.mask_ratio < 1.0:
            raise ValueError(
                f"mask_ratio must be in [0.0, 1.0), got {config.mask_ratio}"
            )

    @property
    def evt_time_ms(self) -> torch.Tensor | None:
        """Current [EVT] timestamp in ms, learnable or None (use 0.0)."""
        if self._evt_time_logit is None:
            return None
        return torch.sigmoid(self._evt_time_logit) * self.config.evt_time_ms_max

    def _compute_timestamps_ms(
        self,
        n_patches: int,
        pre_event_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Compute event-relative timestamps in ms for the full token sequence.

        Returns timestamps for [EVT] + patches:
            - [EVT] at 0.0ms (it IS the event onset)
            - Each patch at its event-relative time

        Args:
            n_patches: number of patches from encoder.
            pre_event_samples: (batch,) int tensor of pre-event sample counts.

        Returns:
            (batch, n_patches+1) timestamps in milliseconds.
        """
        batch_size = pre_event_samples.shape[0]
        total_stride = self.encoder.total_stride
        sfreq = self.config.sfreq
        ms_per_patch = (total_stride / sfreq) * 1000.0

        # Event onset expressed in patch units
        # pre_event_samples is in raw samples; divide by total_stride
        event_onset_patch = pre_event_samples.float() / total_stride

        # Patch indices: 0, 1, ..., n_patches-1
        patch_indices = torch.arange(
            n_patches, dtype=torch.float32, device=pre_event_samples.device
        )
        # (batch, n_patches): each patch's event-relative offset in patch units
        relative_patches = patch_indices.unsqueeze(0) - event_onset_patch.unsqueeze(1)
        # Convert to milliseconds
        patch_timestamps = relative_patches * ms_per_patch

        # [EVT] timestamp: 0.0 ms by default; learnable scalar in preprocessed
        # (broadcast over the batch dim to preserve gradient).
        evt_t = self.evt_time_ms
        if evt_t is None:
            evt_time = torch.zeros(
                batch_size, 1, dtype=torch.float32, device=pre_event_samples.device
            )
        else:
            evt_time = evt_t.to(dtype=torch.float32).expand(batch_size, 1)

        # Concatenate: [EVT] + patches
        timestamps = torch.cat([evt_time, patch_timestamps], dim=1)
        return timestamps

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass through the complete HEDBERT.

        Args:
            batch: dict from EventEpochCollator with keys:
                eeg: (B, 64, T) raw EEG
                attention_mask: (B, T) float, 1.0=valid
                event_ids: (B,) int
                lengths: (B,) int
                pre_event_samples: (B,) int

        Returns:
            dict with keys:
                reconstructed: (B, 64, T) reconstructed EEG
                evt_embeddings: (B, embed_dim) [EVT] token output
                patch_mask: (B, n_patches) float mask for reconstruction loss
        """
        eeg = batch["eeg"]
        attention_mask = batch["attention_mask"]
        pre_event_samples = batch["pre_event_samples"]
        target_length = eeg.shape[2]
        batch_size = eeg.shape[0]

        # 1. Encode raw EEG into patch embeddings
        patches, patch_mask, normalized_eeg, clamped_eeg = self.encoder(
            eeg, attention_mask
        )
        n_patches = patches.shape[1]

        # 1b. Patch-level masking (MAE-style): mask AFTER encoder, BEFORE
        # transformer. The encoder always sees real EEG; only the transformer
        # sees [MASK] tokens. This forces the transformer to predict missing
        # patches from temporal context.
        recon_mask = None
        if self.training and self.config.mask_ratio > 0:
            batch_sz, n_pat, _embed = patches.shape
            n_mask = max(1, int(n_pat * self.config.mask_ratio))
            noise = torch.rand(batch_sz, n_pat, device=patches.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            mask_bool = torch.zeros(
                batch_sz, n_pat, dtype=torch.bool, device=patches.device
            )
            mask_bool.scatter_(1, ids_shuffle[:, :n_mask], True)

            # Replace masked patches with learned [MASK] embedding
            mask_tokens = self.mask_embedding.expand(batch_sz, n_pat, -1).to(
                patches.dtype
            )
            patches = torch.where(mask_bool.unsqueeze(-1), mask_tokens, patches)

            # Expand patch-level mask to time-level for ReconstructionLoss
            # Each patch covers total_stride timepoints
            stride = self.encoder.total_stride
            recon_mask = (
                mask_bool.float()
                .unsqueeze(-1)
                .expand(-1, -1, stride)
                .reshape(batch_sz, -1)
            )
            # Match target_length (encoder may truncate remainder)
            if recon_mask.shape[1] >= target_length:
                recon_mask = recon_mask[:, :target_length]
            else:
                recon_mask = torch.nn.functional.pad(
                    recon_mask, (0, target_length - recon_mask.shape[1])
                )

        # 2. Insert special tokens
        # Expand learned [EVT] embedding to batch size.
        evt_tokens = self.evt_embedding.expand(batch_size, -1, -1)

        # [EVT] + patches
        tokens = torch.cat([evt_tokens, patches], dim=1)

        # Extend mask: [EVT] always valid (1.0).
        ones = torch.ones(
            batch_size, 1, dtype=patch_mask.dtype, device=patch_mask.device
        )
        extended_mask = torch.cat([ones, patch_mask], dim=1)

        # 3. Compute event-relative timestamps for all tokens
        timestamps_ms = self._compute_timestamps_ms(n_patches, pre_event_samples)

        # 4. Add positional encoding
        tokens_with_pe = self.pe(tokens, timestamps_ms)

        # 4b. Freeze EVT embedding: no gradient flows back to evt_embedding param.
        # HED still trains transformer weights via position-0 output.
        # EVT is at position 0 (prepended at step 2 above); forward_packed()
        # uses token_types==1 which is more general.
        if self.config.detach_evt_from_recon:
            evt_mask = torch.zeros(
                tokens_with_pe.shape[:2],
                dtype=torch.bool,
                device=tokens_with_pe.device,
            )
            evt_mask[:, 0] = True
            tokens_with_pe = _ZeroGradAtEVT.apply(tokens_with_pe, evt_mask)

        # 5. Run transformer
        # Convert float mask to bool key_padding_mask (True = ignore/padding)
        key_padding_mask = extended_mask == 0
        transformer_out = self.transformer(
            tokens_with_pe, key_padding_mask=key_padding_mask
        )

        # 6. Extract outputs
        # First token is [EVT]; remaining tokens are patches.
        evt_embeddings = transformer_out[:, 0, :]
        patch_embeddings = transformer_out[:, 1:, :]

        # 7. Decode patches back to EEG
        reconstructed = self.decoder(patch_embeddings, target_length=target_length)

        # Reconstruction target depends on norm mode:
        # - instance norm: use clamped EEG for masked recon (InputNorm output trivial)
        # - mean_scale: always use normalized EEG (preserves variance, scale-matched)
        if self.config.norm_mode == "mean_scale":
            recon_target = normalized_eeg
        elif recon_mask is not None:
            recon_target = clamped_eeg
        else:
            recon_target = normalized_eeg
        result = {
            "reconstructed": reconstructed,
            "evt_embeddings": evt_embeddings,
            "patch_embeddings": patch_embeddings,
            "patch_mask": patch_mask,
            "reconstruction_target": recon_target,
        }
        if recon_mask is not None:
            result["recon_mask"] = recon_mask
        return result

    def forward_packed(
        self, batch: dict[str, torch.Tensor | list]
    ) -> dict[str, torch.Tensor | list]:
        """Forward pass for packed multi-epoch sequences.

        Uses pre-computed token insertion plan from PackedSequenceCollator.
        Token placement uses vectorized scatter; varlen path uses boolean
        mask indexing for flatten/re-pad.

        Args:
            batch: dict from PackedSequenceCollator with pre-computed plan:
                eeg: (B, 64, total_T)
                attention_mask: (B, total_T)
                token_types: (B, max_tokens) 0=pad, 1=EVT, 2=patch
                timestamps_ms: (B, max_tokens)
                patch_source: (B, max_tokens) patch index or -1
                evt_positions: (B, max_n_epochs)
                evt_epoch_indices: (B, max_n_epochs)
                n_valid_evts: (B,)
                n_tokens: (B,)

        Returns:
            dict with reconstructed, evt_embeddings, evt_epoch_indices, patch_mask
        """
        eeg: torch.Tensor = batch["eeg"]  # type: ignore[assignment]
        attention_mask: torch.Tensor = batch["attention_mask"]  # type: ignore[assignment]
        token_types: torch.Tensor = batch["token_types"]  # type: ignore[assignment]
        timestamps_ms: torch.Tensor = batch["timestamps_ms"]  # type: ignore[assignment]
        patch_source: torch.Tensor = batch["patch_source"]  # type: ignore[assignment]
        evt_positions: torch.Tensor = batch["evt_positions"]  # type: ignore[assignment]
        evt_epoch_indices_t: torch.Tensor = batch["evt_epoch_indices"]  # type: ignore[assignment]
        n_valid_evts: torch.Tensor = batch["n_valid_evts"]  # type: ignore[assignment]
        n_tokens: torch.Tensor = batch["n_tokens"]  # type: ignore[assignment]

        target_length = eeg.shape[2]
        batch_size = eeg.shape[0]
        device = eeg.device
        max_tokens = token_types.shape[1]

        # Move plan tensors to device
        token_types = token_types.to(device)
        timestamps_ms = timestamps_ms.to(device)
        patch_source = patch_source.to(device)
        evt_positions = evt_positions.to(device)
        n_tokens = n_tokens.to(device)

        # 1. Encode EEG -> patches (GPU, vectorized)
        patches, patch_mask, normalized_eeg, clamped_eeg = self.encoder(
            eeg, attention_mask
        )

        if patches.shape[1] == 0:
            logger.warning(
                "Encoder produced 0 patches from EEG shape %s. "
                "All epochs may be shorter than stride=%d.",
                eeg.shape,
                self.encoder.total_stride,
            )

        # 1b. Patch-level masking for packed sequences (same as forward())
        recon_mask_packed = None
        if self.training and self.config.mask_ratio > 0 and patches.shape[1] > 0:
            n_pat = patches.shape[1]
            n_mask = max(1, int(n_pat * self.config.mask_ratio))
            noise = torch.rand(batch_size, n_pat, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            mask_bool = torch.zeros(batch_size, n_pat, dtype=torch.bool, device=device)
            mask_bool.scatter_(1, ids_shuffle[:, :n_mask], True)

            mask_tokens = self.mask_embedding.expand(batch_size, n_pat, -1).to(
                patches.dtype
            )
            patches = torch.where(mask_bool.unsqueeze(-1), mask_tokens, patches)

            # Expand to time domain for ReconstructionLoss
            stride = self.encoder.total_stride
            recon_mask_packed = (
                mask_bool.float()
                .unsqueeze(-1)
                .expand(-1, -1, stride)
                .reshape(batch_size, -1)
            )
            if recon_mask_packed.shape[1] >= target_length:
                recon_mask_packed = recon_mask_packed[:, :target_length]
            else:
                recon_mask_packed = torch.nn.functional.pad(
                    recon_mask_packed,
                    (0, target_length - recon_mask_packed.shape[1]),
                )

        # 2. Build token sequence using vectorized scatter (no Python loop)
        # Use patches dtype to be compatible with AMP autocast (fp16/bf16)
        tokens = torch.zeros(
            batch_size,
            max_tokens,
            self.config.embed_dim,
            device=device,
            dtype=patches.dtype,
        )

        # Place [EVT] tokens: token_types == 1
        evt_mask = token_types == 1
        evt_val = self.evt_embedding.squeeze(0).squeeze(0).to(tokens.dtype)
        tokens[evt_mask] = evt_val.expand(int(evt_mask.sum()), -1)

        # Place patch tokens: token_types == 2
        patch_token_mask = token_types == 2
        # Gather patches from encoder output using patch_source indices
        # Clamp to valid range (patch_source is -1 for non-patch tokens)
        max_patch_idx = max(1, patches.shape[1]) - 1
        safe_source = patch_source.clamp(min=0, max=max_patch_idx)
        # Expand for gather: (B, max_tokens, embed_dim)
        source_expanded = safe_source.unsqueeze(-1).expand(
            -1, -1, self.config.embed_dim
        )
        gathered_patches = torch.gather(
            patches, 1, source_expanded
        )  # (B, max_tokens, E)
        # Only write patch positions
        tokens[patch_token_mask] = gathered_patches[patch_token_mask]

        # 3. Token attention mask (valid where token_types > 0)
        token_mask = token_types > 0  # (B, max_tokens), True=valid

        # preprocessed: substitute learnable [EVT] timestamp at evt positions
        # so the parameter participates in the PE computation. The collator
        # writes 0.0 for EVT positions; we overwrite with the learned value
        # using torch.where (preserves gradient through the parameter).
        evt_t = self.evt_time_ms
        if evt_t is not None:
            timestamps_ms = torch.where(
                evt_mask,
                evt_t.to(timestamps_ms.dtype).expand_as(timestamps_ms),
                timestamps_ms,
            )

        # 4. Positional encoding applied to all positions including padding
        # (padding is discarded by varlen flatten or masked by dense path)
        tokens_with_pe = self.pe(tokens, timestamps_ms)

        # 4b. Block gradient at EVT positions in transformer input (see forward())
        if self.config.detach_evt_from_recon:
            evt_input_mask = token_types == 1  # (B, max_tokens)
            tokens_with_pe = _ZeroGradAtEVT.apply(tokens_with_pe, evt_input_mask)

        # 5. Transformer: varlen path (flat, no padding waste) or dense path
        if HAS_FLASH_ATTN_VARLEN and device.type == "cuda":
            # Flatten valid tokens using boolean mask
            flat_tokens = tokens_with_pe[token_mask]  # (total_tokens, E)

            cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
            cu_seqlens[1:] = n_tokens.cumsum(0)
            max_seqlen = int(n_tokens.max().item())

            if max_seqlen == 0:
                raise ValueError(
                    "All sequences have 0 tokens; cannot run varlen attention. "
                    "Check that epochs are longer than encoder stride."
                )
            assert flat_tokens.shape[0] == int(cu_seqlens[-1].item()), (
                f"flat_tokens length {flat_tokens.shape[0]} != "
                f"cu_seqlens[-1] {cu_seqlens[-1].item()}"
            )

            flat_out = self.transformer(
                flat_tokens,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            # Re-pad to (B, max_tokens, E) using boolean mask (no Python loop)
            # Use flat_out dtype: LayerNorm outputs fp32 even under AMP autocast
            transformer_out = flat_out.new_zeros(
                batch_size, max_tokens, self.config.embed_dim
            )
            transformer_out[token_mask] = flat_out
        else:
            transformer_out = self.transformer(
                tokens_with_pe,
                key_padding_mask=~token_mask,
            )

        # 6. Extract [EVT] embeddings (vectorized gather)
        evt_pos_expanded = (
            evt_positions.unsqueeze(-1).expand(-1, -1, self.config.embed_dim).to(device)
        )
        evt_embeddings = torch.gather(transformer_out, 1, evt_pos_expanded)

        # 7. Extract patch tokens for reconstruction (vectorized)
        patch_positions_mask = token_types == 2  # (B, max_tokens)
        n_patch_tokens = patch_positions_mask.sum(dim=1).max().item()

        # Vectorized patch extraction: sort so True positions come first
        sorted_indices = torch.argsort(~patch_positions_mask, dim=1, stable=True)
        patch_indices = sorted_indices[:, : int(n_patch_tokens)]

        patch_idx_expanded = patch_indices.unsqueeze(-1).expand(
            -1, -1, self.config.embed_dim
        )
        patch_embeddings = torch.gather(transformer_out, 1, patch_idx_expanded)

        # 8. Decode
        reconstructed = self.decoder(patch_embeddings, target_length=target_length)

        # Return tensors directly (no Python list conversion = no GPU-CPU syncs)
        if self.config.norm_mode == "mean_scale":
            recon_target = normalized_eeg
        elif recon_mask_packed is not None:
            recon_target = clamped_eeg
        else:
            recon_target = normalized_eeg
        result = {
            "reconstructed": reconstructed,
            "evt_embeddings": evt_embeddings,
            "evt_epoch_indices": evt_epoch_indices_t,
            "n_valid_evts": n_valid_evts,
            "patch_mask": patch_mask,
            "reconstruction_target": recon_target,
        }
        if recon_mask_packed is not None:
            result["recon_mask"] = recon_mask_packed
        return result
