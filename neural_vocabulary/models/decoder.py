"""Lightweight EEG reconstruction decoder.

Maps transformer patch embeddings back to EEG signal space for
reconstruction (L_recon) and masked reconstruction (L_mask) losses.

Two decoder variants:
    - LinearDecoder: single projection (minimal params, fast)
    - ConvDecoder: progressive ConvTranspose1d upsampling (better quality)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as fn

if TYPE_CHECKING:
    from neural_vocabulary.configs import HEDBERTConfig


class EEGDecoder(nn.Module):
    """Decode transformer output back to multi-channel EEG signal.

    Uses a single linear projection per patch followed by reshape.
    No redundant second linear layer; the projection is the bottleneck
    and the transformer provides sufficient capacity upstream.

    Param count: embed_dim * n_channels * patch_size + bias
    Tiny (128->64*75):  ~614K params
    Small (256->64*75): ~1.2M params
    """

    def __init__(self, config: HEDBERTConfig, patch_size: int) -> None:
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_channels = config.target_channels
        self.patch_size = patch_size
        projection_dim = self.n_channels * self.patch_size

        self.projection = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, projection_dim),
        )

    def forward(
        self, transformer_output: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        """Reconstruct EEG signal from transformer patch embeddings.

        Args:
            transformer_output: (batch, n_patches, embed_dim) from
                transformer. The [EVT] token should be stripped before
                passing to the decoder.
            target_length: the time dimension to reconstruct to.

        Returns:
            reconstructed: (batch, n_channels, target_length)
        """
        if target_length < 1:
            raise ValueError(f"target_length must be positive, got {target_length}")

        batch, n_patches, in_embed_dim = transformer_output.shape
        if n_patches < 1:
            raise ValueError(
                f"Decoder received 0 patches (shape={transformer_output.shape}). "
                f"Ensure special tokens are stripped and the encoder produced "
                f"at least 1 patch."
            )
        if in_embed_dim != self.embed_dim:
            raise ValueError(
                f"Decoder expected embed_dim={self.embed_dim}, "
                f"got transformer_output with last dim={in_embed_dim}"
            )

        # Project each patch embedding to (n_channels * patch_size)
        projected = self.projection(transformer_output)

        # Reshape: (batch, n_patches, n_channels, patch_size)
        projected = projected.view(batch, n_patches, self.n_channels, self.patch_size)

        # Transpose to channel-first: (batch, n_channels, n_patches, patch_size)
        projected = projected.permute(0, 2, 1, 3)

        # Concatenate patches along time: (batch, n_channels, n_patches * patch_size)
        reconstructed = projected.reshape(
            batch, self.n_channels, n_patches * self.patch_size
        )

        # Adjust time dimension to match target_length
        raw_length = n_patches * self.patch_size
        if raw_length == target_length:
            return reconstructed
        elif raw_length > target_length:
            return reconstructed[:, :, :target_length]
        else:
            return fn.interpolate(
                reconstructed, size=target_length, mode="linear", align_corners=False
            )
