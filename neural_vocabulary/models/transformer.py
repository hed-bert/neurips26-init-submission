"""Transformer backbone for HED-BERT.

Two attention mechanisms:

1. Dense (default): PyTorch SDPA (scaled_dot_product_attention), which
   auto-dispatches to Flash Attention 2 on CUDA with bf16/fp16, or the
   math kernel on CPU/MPS. Input: (B, S, E) with key_padding_mask.

2. Varlen (optional): flash_attn_varlen_func from the flash-attn package
   (pip install flash-attn). Eliminates padding waste for packed
   multi-epoch sequences using cu_seqlens. Input: flat (total_tokens, E).
   Requires CUDA and the flash-attn package.

Gaudi 1 support removed; Delta A100s are the target HPC platform.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as nnf

if TYPE_CHECKING:
    from neural_vocabulary.configs import HEDBERTConfig

logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_varlen_func

    HAS_FLASH_ATTN_VARLEN = True
    logger.info("flash_attn varlen support available")
except ImportError:
    HAS_FLASH_ATTN_VARLEN = False
    if torch.cuda.is_available():
        logger.warning(
            "flash_attn not available on CUDA machine; varlen attention disabled, "
            "falling back to dense SDPA. Install with: pip install flash-attn"
        )
    else:
        logger.debug("flash_attn not available (non-CUDA); using dense SDPA")


class HEDBERTTransformerBlock(nn.Module):
    """Single pre-norm transformer block.

    Architecture: LayerNorm -> self-attention -> residual,
    then LayerNorm -> FFN -> residual. GELU activation in FFN.

    Supports two attention paths:
    - Dense (default): SDPA with key_padding_mask, input (B, S, E)
    - Varlen (when cu_seqlens provided + flash_attn installed):
      flat input (total_tokens, E), zero padding waste
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # QKV projection (single fused linear for efficiency)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = dropout

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _attention_dense(
        self,
        normed: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Dense attention path using SDPA. Input: (B, S, E)."""
        batch_size, seq_len, _ = normed.shape

        qkv = self.qkv_proj(normed)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            attn_mask = attn_mask.to(dtype=q.dtype) * torch.finfo(q.dtype).min

        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = nnf.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_out)

    def _attention_varlen(
        self,
        normed: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Variable-length attention using flash_attn_varlen_func.

        Input: (total_tokens, E). No padding waste.
        """
        total_tokens = normed.shape[0]

        qkv = self.qkv_proj(normed)  # (total_tokens, 3 * E)
        qkv = qkv.reshape(total_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)  # each (total_tokens, num_heads, head_dim)

        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout_p,
        )  # (total_tokens, num_heads, head_dim)

        attn_out = attn_out.reshape(total_tokens, self.embed_dim)
        return self.out_proj(attn_out)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Forward pass through one transformer block.

        Dense path (default):
            x: (batch, seq_len, embed_dim)
            key_padding_mask: (batch, seq_len) bool, True=IGNORE

        Varlen path (when cu_seqlens provided + flash_attn installed):
            x: (total_tokens, embed_dim)
            cu_seqlens: (batch_size + 1,) int32 cumulative sequence lengths
            max_seqlen: maximum sequence length in the batch
        """
        normed = self.norm1(x)

        if (cu_seqlens is None) != (max_seqlen is None):
            raise ValueError(
                "cu_seqlens and max_seqlen must both be provided or both be None"
            )

        if cu_seqlens is not None and max_seqlen is not None:
            if not HAS_FLASH_ATTN_VARLEN:
                raise RuntimeError(
                    "cu_seqlens provided but flash_attn is not installed. "
                    "Install flash-attn or use the dense path with key_padding_mask."
                )
            attn_out = self._attention_varlen(normed, cu_seqlens, max_seqlen)
        else:
            attn_out = self._attention_dense(normed, key_padding_mask)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class HEDBERTTransformer(nn.Module):
    """Stack of pre-norm transformer blocks for HED-BERT.

    Supports two attention paths:
    - Dense (default): SDPA with padding masks, (B, S, E) tensors
    - Varlen (cu_seqlens + flash_attn): flat (total_tokens, E) tensors,
      zero padding waste in attention and FFN computation
    """

    def __init__(self, config: HEDBERTConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HEDBERTTransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Forward pass through all transformer blocks.

        Dense path (default):
            x: (batch, seq_len, embed_dim)
            key_padding_mask: (batch, seq_len) bool, True=IGNORE. Must be bool.

        Varlen path (when cu_seqlens provided + flash_attn installed):
            x: (total_tokens, embed_dim)
            cu_seqlens: (batch_size + 1,) int32 cumulative sequence lengths
            max_seqlen: maximum sequence length in the batch

        Returns:
            Same shape as input x.
        """
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            raise TypeError(
                f"key_padding_mask must be a bool tensor (True=ignore/padding), "
                f"got dtype={key_padding_mask.dtype}. If using collator output, "
                f"convert with: key_padding_mask = (attention_mask == 0)"
            )
        for layer in self.layers:
            x = layer(
                x,
                key_padding_mask=key_padding_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        return self.final_norm(x)
