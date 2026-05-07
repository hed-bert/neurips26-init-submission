""" Gate C.1: Tiny Vision Transformer on Morlet TF spectrograms.

Input layout: (B, n_channels, n_freqs, n_time)
  - n_channels = 64 (EEG channels collapsed by patch-embed Conv2d)
  - n_freqs = 6  (Morlet frequency bins; spatial height)
  - n_time  = 22 (downsampled time bins; spatial width)

Architecture (~2.7M params):
  - Conv2d patch embed: kernel=stride=(2, 2), in=64, out=192
    → (B, 192, 3, 11) → flatten to (B, 33, 192) tokens
  - Prepend learnable [CLS] token, add learnable 1D positional embedding
    → (B, 34, 192)
  - 6 pre-norm transformer blocks, 6 heads, head_dim=32, MLP ratio 4 (768),
    GELU activation, SDPA-based attention (Flash on CUDA when available)
  - Final LayerNorm + Linear(192 → n_classes) on the [CLS] token

Design decisions (advisor 2026-04-19):
  - patch_size=(2, 2) → 33 patches; preserves both freq + time structure for
    cross-axis attention (vs (3, 2) which collapses freq too aggressively).
  - Weight-decay-excluded params: cls_token, pos_embed, biases, LayerNorm —
    standard ViT recipe, helps tiny ViTs notably.
  - Init: trunc_normal_(std=0.02) on cls_token, pos_embed, Linear weights;
    zero biases; LayerNorm gamma=1, beta=0.
  - No DropPath, no attention dropout — model is tiny relative to typical ViT
    recipes; aggressive regularization hurts at this scale.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nnf


class _MLP(nn.Module):
    """Two-layer GELU MLP used in the transformer block."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention via torch.nn.functional.scaled_dot_product_attention.

    Uses the SDPA backend, which selects Flash Attention on CUDA when the
    input dtype/shape supports it.  No attention dropout (small model + tiny
    sequence; aggressive dropout hurts).
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        # (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        # SDPA dispatches Flash / mem-efficient kernels automatically on CUDA.
        # No attn_mask / no causal — full bidirectional attention over CLS+patches.
        out = nnf.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        # (B, num_heads, N, head_dim) → (B, N, dim)
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out


class _Block(nn.Module):
    """Pre-norm transformer block: LN → MHSA → residual; LN → MLP → residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MultiHeadSelfAttention(dim, num_heads=num_heads)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


class TFViT(nn.Module):
    """Tiny Vision Transformer for supervised TF-spectrogram classification.

    Args:
        n_channels: Number of EEG channels (Conv2d input dim). Default: 64.
        n_freqs: Number of frequency bins. Default: 6.
        n_time: Number of time bins. Default: 22.
        n_classes: Number of output classes. Default: 2.
        patch_size: 2D patch size over (freq, time). Default: (2, 2).
        d_model: Transformer hidden dim. Default: 192.
        depth: Number of transformer blocks. Default: 6.
        num_heads: Number of attention heads. Default: 6.
        mlp_ratio: MLP hidden dim ratio. Default: 4.0.
        dropout: Dropout rate inside MLP and after attention. Default: 0.0.
        head_dropout: Dropout before the classification linear. Default: 0.0.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_freqs: int = 6,
        n_time: int = 22,
        n_classes: int = 2,
        patch_size: tuple[int, int] = (2, 2),
        d_model: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        ph, pw = patch_size
        if n_freqs % ph != 0:
            raise ValueError(
                f"n_freqs={n_freqs} not divisible by patch_size[0]={ph}. "
                "Choose a patch height that divides the frequency dim."
            )
        if n_time % pw != 0:
            raise ValueError(
                f"n_time={n_time} not divisible by patch_size[1]={pw}. "
                "Choose a patch width that divides the time dim."
            )
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )

        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.n_time = n_time
        self.patch_size = patch_size
        self.d_model = d_model

        n_patches_h = n_freqs // ph
        n_patches_w = n_time // pw
        self.n_patches = n_patches_h * n_patches_w

        # Patch embed: collapses EEG channels and patches in one Conv2d.
        # Input (B, 64, 6, 22) → output (B, 192, 3, 11) → flatten to (B, 33, 192).
        self.patch_embed = nn.Conv2d(
            in_channels=n_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Learned 1D positional embedding for [CLS] + n_patches tokens.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches, d_model))

        self.blocks = nn.ModuleList(
            [
                _Block(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard ViT init: trunc_normal_ for weights, zero for biases.
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run patch-embed + transformer; return [CLS] embedding (B, d_model)."""
        b = x.shape[0]
        x = self.patch_embed(x)  # (B, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, d_model)

        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + n_patches, d_model)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, n_channels, n_freqs, n_time) Morlet log-power spectrogram.

        Returns:
            Logits of shape (B, n_classes).
        """
        feats = self.forward_features(x)
        feats = self.head_drop(feats)
        return self.head(feats)

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def no_weight_decay_param_names(self) -> set[str]:
        """Return parameter names that should be excluded from AdamW weight_decay.

        Standard ViT recipe: exclude cls_token, pos_embed, all biases, and
        LayerNorm weights.  Applying weight_decay to these hurts a tiny ViT
        notably (advisor 2026-04-19).
        """
        no_decay: set[str] = set()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name in {"cls_token", "pos_embed"} or name.endswith(".bias"):
                no_decay.add(name)
            elif p.ndim == 1:
                # LayerNorm gamma (1D weight) and any other 1D parameters.
                no_decay.add(name)
        return no_decay

    def build_param_groups(self, weight_decay: float) -> list[dict[str, object]]:
        """Build AdamW param-groups with weight_decay-excluded params separated."""
        no_decay_names = self.no_weight_decay_param_names()
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name in no_decay_names:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
