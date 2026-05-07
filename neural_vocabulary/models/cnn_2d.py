""" Gate B.1: 2D CNN for supervised TF-spectrogram decoding.

Input layout: (B, n_channels, n_freqs, n_time)
  - n_channels = 64 (EEG channels treated as Conv2d input-channel dim)
  - n_freqs = 6  (Morlet frequency bins; spatial height)
  - n_time  = 22 (downsampled time bins; spatial width)

Architecture: 3 conv-BN-ReLU-MaxPool blocks (32→64→128 feature maps)
over the (freq, time) spatial plane, followed by global average pool
and a linear classification head. ~110K parameters.

Block spatial dynamics with (n_freqs=6, n_time=22) and max-pool 2×2:
  block1  →  pool(2,2):  (3, 11)
  block2  →  pool(2,2):  (1,  5)
  block3  →  GAP:        (1,  1)

Block 3 uses padding=1 (3×3 kernel on (1,5) freq dim requires it) and
then the final GAP collapses to a 128-dim feature vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        padding: int = 1,
        pool_kernel: tuple[int, int] = (2, 2),
        pool_stride: tuple[int, int] | None = None,
    ) -> None:
        if pool_stride is None:
            pool_stride = pool_kernel
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
        )


class TFCNN2D(nn.Module):
    """Small 2D CNN for supervised TF-spectrogram classification.

    Takes Morlet log-power spectrograms as input and outputs class logits.
    Architecture matches Gate B.1 spec: 3-block CNN + GAP + Linear, ~100K params.

    Args:
        n_channels: Number of EEG channels (input-channel dim). Default: 64.
        n_freqs: Number of frequency bins (spatial height). Default: 6.
        n_time: Number of time bins (spatial width). Default: 22.
        n_classes: Number of output classes. Default: 2 (binary).
        base_filters: Feature maps in first conv block. Default: 32.
        dropout: Dropout rate on the pre-classifier representation. Default: 0.5.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_freqs: int = 6,
        n_time: int = 22,
        n_classes: int = 2,
        base_filters: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Validate spatial dimensions: after two 2×2 max-pools on (n_freqs, n_time),
        # block 3 receives (n_freqs//4, n_time//4).  The freq dim must be ≥1.
        if n_freqs < 4:
            raise ValueError(
                f"n_freqs={n_freqs} too small: need ≥4 for two 2×2 max-pools. "
                "Use asymmetric pooling if input is shorter."
            )
        if n_time < 4:
            raise ValueError(
                f"n_time={n_time} too small: need ≥4 for two 2×2 max-pools."
            )

        f1, f2, f3 = base_filters, base_filters * 2, base_filters * 4  # 32, 64, 128

        # Block 1: (B, n_channels, n_freqs, n_time) → (B, f1, n_freqs//2, n_time//2)
        self.block1 = _ConvBlock(n_channels, f1)

        # Block 2: → (B, f2, n_freqs//4, n_time//4)
        self.block2 = _ConvBlock(f1, f2)

        # Block 3: → (B, f3, freq_dim, time_dim) with no MaxPool; GAP follows.
        # After block2 the freq dim is 1 (6→3→1).  Any further spatial pool
        # would collapse to (0, ...) which Conv2d rejects.  GAP handles the
        # final collapse to (1, 1), so block3 is conv-only.
        self.block3 = nn.Sequential(
            nn.Conv2d(f2, f3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
        )

        # Global average pool collapses (H, W) → (1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(f3, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, n_channels, n_freqs, n_time).

        Returns:
            Logits of shape (B, n_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
