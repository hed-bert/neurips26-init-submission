"""Continuous-time event-relative positional encoding for HED-BERT.

Sinusoidal positional encoding using actual timestamps in milliseconds
rather than discrete patch indices. Position 0ms = event onset, negative
= pre-event, positive = post-event. Applied to the [EVT] token (at 0ms)
and every EEG patch at its event-relative time.

This captures the fixed-latency structure of neural responses (e.g., P300
at ~300ms post-stimulus) and naturally handles variable-length epochs,
different sampling rates, and different patch sizes without index arithmetic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ContinuousTimePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding using continuous time values.

    Instead of encoding discrete patch indices, this encodes actual
    timestamps in milliseconds relative to event onset. This means:
    - [EVT] token at 0ms always gets the same encoding
    - A patch 100ms post-stimulus gets the same encoding regardless of
      patch_size, stride, or sampling rate
    - No pre-computed buffer, no clamping, no index lookups

    The sinusoidal frequencies are chosen to resolve temporal features
    at physiologically relevant scales:
    - High-frequency components: ~10ms (brainstem auditory responses)
    - Mid-frequency components: ~100ms (N170, P300)
    - Low-frequency components: ~1000ms (slow cortical potentials)
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize continuous-time positional encoding.

        Args:
            embed_dim: dimension of token embeddings. Must be even.
            dropout: dropout rate applied after adding positional encoding.

        Raises:
            ValueError: if embed_dim is odd.
        """
        super().__init__()
        if embed_dim < 2 or embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim must be even and >= 2 for sinusoidal PE, got {embed_dim}"
            )
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute the frequency division terms (not a parameter, just a buffer)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / embed_dim)
        )
        self.register_buffer("div_term", div_term)  # (embed_dim // 2,)

    def forward(
        self,
        x: torch.Tensor,
        timestamps_ms: torch.Tensor,
    ) -> torch.Tensor:
        """Add continuous-time positional encoding to token embeddings.

        Args:
            x: token embeddings of shape (batch, seq_len, embed_dim).
                Includes all tokens: [EVT] and EEG patches.
            timestamps_ms: (batch, seq_len) event-relative timestamps in
                milliseconds for each token. Convention:
                - [EVT] token: 0.0 (it IS the event onset)
                - Pre-event patches: negative values (e.g., -100.0)
                - Post-event patches: positive values (e.g., +300.0)

        Returns:
            (batch, seq_len, embed_dim) with positional encoding added.

        Raises:
            ValueError: if shapes are inconsistent.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape (batch, seq_len, embed_dim), got {x.shape}"
            )
        if timestamps_ms.ndim != 2:
            raise ValueError(
                f"Expected timestamps_ms with shape (batch, seq_len), got {timestamps_ms.shape}"
            )
        if x.shape[0] != timestamps_ms.shape[0] or x.shape[1] != timestamps_ms.shape[1]:
            raise ValueError(
                f"Shape mismatch: x is {x.shape}, timestamps_ms is {timestamps_ms.shape}"
            )
        if x.shape[2] != self.embed_dim:
            raise ValueError(
                f"PE embed_dim={self.embed_dim} does not match input "
                f"embed_dim={x.shape[2]}"
            )

        # Always compute PE in float32 to avoid float16 overflow/precision
        # loss. float16 max is 65504; timestamps of 65536ms (~65s, common
        # in resting-state EEG) would overflow. Standard practice in all
        # major transformer implementations.
        t = timestamps_ms.unsqueeze(-1).to(torch.float32)  # (batch, seq_len, 1)
        angles = t * self.div_term  # type: ignore[unsupported-operator]  # div_term is float32

        pe = torch.zeros(*x.shape, dtype=torch.float32, device=x.device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)

        # Cast PE to input dtype at the end, after sin/cos computed safely
        return self.dropout(x + pe.to(x.dtype))
