""" post-Gate-C diagnostic: 1D CNN on time-domain preprocessed epochs.

Input layout: (B, n_channels, n_time)
  - n_channels = 64  (EEG channels treated as Conv1d input-channel dim)
  - n_time     = 220 (2.2 s at 100 Hz, preprocessed passive-movie tonic epoch)

Time-domain analog of :class:`neural_vocabulary.models.cnn_2d.TFCNN2D`.
Architecture: 3 conv-BN-ReLU blocks (32 -> 64 -> 128 feature maps) along the
time axis, followed by global average pool and a linear classification head.

With kernel=7 (default, spans 70 ms at 100 Hz -- roughly an alpha/beta period)
and two stride-1 conv blocks followed by MaxPool1d(2) each, plus a stride-1
conv-only block, the spatial dynamics are:

    block1 -> pool(2):  (220) -> (110)
    block2 -> pool(2):  (110) -> (55)
    block3 -> GAP:       (55) -> (1)

~87K parameters with default settings (well inside the 80K-150K envelope
that matches TFCNN2D at 111K).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock1D(nn.Sequential):
    """Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 7,
        padding: int | None = None,
        pool_kernel: int = 2,
        pool_stride: int | None = None,
    ) -> None:
        if padding is None:
            padding = kernel // 2
        if pool_stride is None:
            pool_stride = pool_kernel
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride),
        )


class TimeCNN1D(nn.Module):
    """Small 1D CNN for supervised time-domain EEG classification.

    Takes preprocessed tonic epochs ``(B, C, T)`` as input and outputs class
    logits.  Architecture mirrors :class:`TFCNN2D` in depth and channel
    progression, with 1D convolutions along the time axis replacing 2D
    convolutions over (freq, time).

    Args:
        n_channels: Number of EEG channels (Conv1d input-channel dim).
            Default: 64 (preprocessed montage).
        n_time: Number of time samples per epoch. Default: 220
            (2.2 s at 100 Hz).
        n_classes: Number of output classes. Default: 2 (binary).
        base_filters: Feature maps in first conv block. Default: 32.
        kernel_size: Kernel size used by every conv layer.  Default: 7
            (70 ms at 100 Hz).  Larger kernels increase param count; 5, 7,
            9, 11, 15 are all reasonable time spans.
        dropout: Dropout applied to the pre-classifier representation.
            Default: 0.5.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_time: int = 220,
        n_classes: int = 2,
        base_filters: int = 32,
        kernel_size: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Two 2-way max-pools require n_time >= 4 so the final block sees a
        # non-empty time dimension.  preprocessed is 220 so this is a safety
        # guard for explicit smaller configurations.
        if n_time < 4:
            raise ValueError(
                f"n_time={n_time} too small: need >=4 for two stride-2 max-pools."
            )
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size={kernel_size} must be a positive odd integer."
            )

        f1, f2, f3 = base_filters, base_filters * 2, base_filters * 4  # 32, 64, 128

        # Block 1: (B, n_channels, n_time) -> (B, f1, n_time // 2)
        self.block1 = _ConvBlock1D(n_channels, f1, kernel=kernel_size)

        # Block 2: -> (B, f2, n_time // 4)
        self.block2 = _ConvBlock1D(f1, f2, kernel=kernel_size)

        # Block 3: conv-only (no MaxPool); the adaptive GAP collapses to length 1.
        # Mirrors TFCNN2D.block3 which is also conv-only.
        self.block3 = nn.Sequential(
            nn.Conv1d(
                f2,
                f3,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(f3),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(f3, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(B, n_channels, n_time)``.

        Returns:
            Logits of shape ``(B, n_classes)``.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
