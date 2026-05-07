"""Multi-scale convolutional encoder for EEG patch extraction.

Converts raw EEG epochs (batch, channels, time) into a sequence of
patch-level latent representations (batch, n_patches, embed_dim) using
either a sequential stack or parallel branches of Conv1d layers.

Sequential (MultiScaleEncoder): cascaded Conv1d blocks.
Parallel (ParallelMultiScaleEncoder): parallel branches at different
    temporal scales targeting delta/theta/alpha/beta-gamma bands.
    Coarse branches upsampled to finest resolution, summed, projected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as fn

if TYPE_CHECKING:
    from neural_vocabulary.configs import HEDBERTConfig


class InputNorm(nn.Module):
    """EEG input normalization with configurable mode.

    Two modes:
        "instance": InstanceNorm1d (learnable, per-channel zero-mean unit-variance).
            Makes reconstruction trivially easy but good for encoder input.
        "mean_scale": per-epoch channel mean removal + fixed amplitude scaling.
            Preserves variance structure. Compatible with real-time inference.
            Mean removal is stateless (per-epoch); for deployment, replace with
            adaptive EMA mean.

    Both modes apply amplitude clamping first.

    Args:
        n_channels: number of EEG channels.
        max_amplitude: clamp to [-max, +max]. 0 = disabled.
        mode: "instance" or "mean_scale".
        scale: fixed divisor for mean_scale mode (uV).
    """

    def __init__(
        self,
        n_channels: int,
        max_amplitude: float = 800.0,
        mode: str = "instance",
        scale: float = 200.0,
    ) -> None:
        super().__init__()
        self.max_amplitude = max_amplitude
        self.mode = mode
        self.scale = scale
        if mode == "instance":
            self.norm = nn.InstanceNorm1d(n_channels, affine=True)
        elif mode != "mean_scale":
            raise ValueError(
                f"Unknown norm_mode: {mode}. Use 'instance' or 'mean_scale'"
            )

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize EEG.

        Returns:
            (normalized, clamped). normalized is the encoder input.
            clamped is post-clamp pre-normalization (for masked recon target).
        """
        clamped = eeg
        if self.max_amplitude > 0:
            clamped = torch.clamp(eeg, -self.max_amplitude, self.max_amplitude)
        if self.mode == "instance":
            return self.norm(clamped), clamped
        # mean_scale: remove per-channel mean, divide by fixed scale
        normalized = (clamped - clamped.mean(dim=-1, keepdim=True)) / self.scale
        return normalized, clamped


class MultiScaleEncoder(nn.Module):
    """Sequential multi-scale Conv1d encoder for EEG signals.

    Architecture: a stack of Conv1d blocks where each block uses a
    different kernel size from ``config.encoder_scales``, with channel
    progression from ``config.encoder_hidden``. Each block consists of
    Conv1d (stride=kernel_size for non-overlapping patches), BatchNorm1d,
    GELU activation, and Dropout. The final 1x1 conv block omits Dropout.
    A final linear layer projects the last hidden dimension to ``embed_dim``.

    The sequential design avoids the complexity of merging variable-length
    patch sequences from parallel branches. Each convolutional block
    reduces the time dimension by its kernel size (stride=kernel_size),
    and the channel progression follows ``encoder_hidden``.

    Example with Tiny config (encoder_scales=[15, 5], encoder_hidden=[32, 64, 128]):
        Input:  (B, 64, T)
        Block 0: Conv1d(64->32, kernel=15, stride=15)  -> (B, 32, T//15)
        Block 1: Conv1d(32->64, kernel=5, stride=5)    -> (B, 64, T//75)
        Final:   Conv1d(64->128, kernel=1, stride=1)    -> (B, 128, T//75)
        Project: Linear(128->128)                       -> (B, T//75, 128)
    """

    def __init__(self, config: HEDBERTConfig) -> None:
        super().__init__()

        scales = config.encoder_scales
        hidden = config.encoder_hidden

        if not scales:
            raise ValueError("encoder_scales must not be empty.")
        for i, k in enumerate(scales):
            if k < 1:
                raise ValueError(
                    f"encoder_scales[{i}] = {k}; all kernel sizes must be >= 1."
                )
        if len(hidden) != len(scales) + 1:
            raise ValueError(
                f"encoder_hidden must have exactly len(encoder_scales)+1 entries. "
                f"Got {len(hidden)} hidden dims for {len(scales)} scales "
                f"(need {len(scales) + 1})."
            )
        for i, h in enumerate(hidden):
            if h < 1:
                raise ValueError(f"encoder_hidden[{i}] = {h}; all values must be >= 1.")

        self._scales = list(scales)

        blocks: list[nn.Module] = []
        in_channels = config.target_channels

        # One Conv1d block per scale, each with stride=kernel_size
        for i, kernel_size in enumerate(scales):
            out_channels = hidden[i]
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=kernel_size,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
            )
            in_channels = out_channels

        # Final 1x1 conv to reach the last hidden dim
        final_hidden = hidden[-1]
        blocks.append(
            nn.Sequential(
                nn.Conv1d(in_channels, final_hidden, kernel_size=1, bias=False),
                nn.BatchNorm1d(final_hidden),
                nn.GELU(),
            )
        )

        self.conv_blocks = nn.ModuleList(blocks)
        self.projection = nn.Linear(final_hidden, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Store total stride for mask downsampling
        self._total_stride = 1
        for k in scales:
            self._total_stride *= k

    @property
    def total_stride(self) -> int:
        """Total temporal downsampling factor (product of all kernel sizes)."""
        return self._total_stride

    def forward(
        self, eeg: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode raw EEG into patch embeddings.

        Args:
            eeg: (batch, channels, time) raw EEG after channel harmonization.
            attention_mask: (batch, time) with 1.0 for real data, 0.0 for padding.

        Returns:
            patches: (batch, n_patches, embed_dim) patch embeddings.
            patch_mask: (batch, n_patches) float attention mask.
            normalized_eeg: (batch, channels, time) post-InputNorm (or raw if
                no InputNorm). Default reconstruction target.
            clamped_eeg: (batch, channels, time) post-clamp pre-InstanceNorm.
                Preserves amplitude structure. Used as masked recon target.
        """
        if eeg.ndim != 3:
            raise ValueError(
                f"Expected eeg with shape (batch, channels, time), got {eeg.shape}"
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"Expected attention_mask with shape (batch, time), "
                f"got {attention_mask.shape}"
            )
        if eeg.shape[0] != attention_mask.shape[0]:
            raise ValueError(
                f"Batch size mismatch: eeg has {eeg.shape[0]}, "
                f"attention_mask has {attention_mask.shape[0]}"
            )
        if eeg.shape[2] != attention_mask.shape[1]:
            raise ValueError(
                f"Time dimension mismatch: eeg has {eeg.shape[2]} time steps, "
                f"attention_mask has {attention_mask.shape[1]}"
            )

        if eeg.shape[2] < self._total_stride:
            raise RuntimeError(
                f"Input time dimension ({eeg.shape[2]}) is shorter than the "
                f"total stride ({self._total_stride}). Minimum input length "
                f"for encoder_scales={self._scales} is "
                f"{self._total_stride} samples."
            )

        x = eeg

        for block in self.conv_blocks:
            x = block(x)

        if x.shape[2] == 0:
            raise RuntimeError(
                f"Input time dimension ({eeg.shape[2]}) is too short for "
                f"encoder_scales={self._scales}. The conv stack reduced the "
                f"time dimension to 0. Minimum input length is "
                f"{self._total_stride} samples."
            )

        # x: (batch, final_hidden, n_patches)
        # Transpose to (batch, n_patches, final_hidden) for the linear layer
        x = x.transpose(1, 2)
        patches = self.projection(x)
        patches = self.dropout(patches)

        # Downsample the attention mask to patch resolution.
        # Use max-pool with the same total stride: a patch is valid if any
        # sample in its receptive field is valid.
        n_patches = patches.shape[1]
        patch_mask = _downsample_mask(attention_mask, self._total_stride, n_patches)

        # MultiScaleEncoder has no InputNorm; normalized == raw
        return patches, patch_mask, eeg, eeg


def _downsample_mask(
    mask: torch.Tensor, total_stride: int, n_patches: int
) -> torch.Tensor:
    """Downsample a time-level mask to patch resolution.

    Uses max-pooling so a patch is marked valid (1.0) if any sample
    within its receptive field is valid. Pads the time dimension to be
    divisible by total_stride (with 0.0 = padding), applies max_pool1d,
    then trims to n_patches to match the conv stack output.

    Args:
        mask: (batch, time) float mask (1.0=valid, 0.0=padding).
        total_stride: total temporal downsampling factor.
        n_patches: expected number of output patches.

    Returns:
        (batch, n_patches) float mask at patch resolution.
    """
    # Add channel dim for max_pool1d: (batch, 1, time)
    m = mask.unsqueeze(1)

    # Pad to make divisible by total_stride (pad with 0.0 = padding)
    time_len = m.shape[2]
    remainder = time_len % total_stride
    if remainder != 0:
        pad_len = total_stride - remainder
        m = nn.functional.pad(m, (0, pad_len), value=0.0)

    patch_mask = nn.functional.max_pool1d(
        m, kernel_size=total_stride, stride=total_stride
    )

    # Validate and trim to match the actual number of patches from conv stack
    pooled_len = patch_mask.shape[2]
    if pooled_len < n_patches:
        raise RuntimeError(
            f"Mask downsampling produced {pooled_len} patches but conv stack "
            f"produced {n_patches}. This indicates a stride/padding mismatch "
            f"(total_stride={total_stride}, input time={mask.shape[1]})."
        )
    patch_mask = patch_mask[:, 0, :n_patches]

    return patch_mask


class ParallelMultiScaleEncoder(nn.Module):
    """Parallel multi-scale Conv1d encoder for EEG signals.

    Architecture: parallel branches at different temporal scales, each
    targeting a frequency band. All branches use the same hidden dim
    from ``config.encoder_hidden[0]``. Coarse branches are upsampled
    to the finest branch resolution, summed, and projected to embed_dim.

    Frequency band mapping (at 100Hz):
        k=4  (40ms stride):  beta/gamma (13-50 Hz), finest resolution
        k=8  (80ms stride):  alpha (8-13 Hz)
        k=16 (160ms stride): theta (4-8 Hz)
        k=32 (320ms stride): delta (0.5-4 Hz)

    Output: n_patches = floor(T / min(scales)) tokens, each containing
    multi-scale frequency information. total_stride = min(scales).

    Args:
        config: HEDBERTConfig with:
            encoder_scales: list of kernel sizes (e.g., [4, 8, 16, 32])
            encoder_hidden: [hidden_dim] (single entry for parallel mode)
            max_amplitude: clamp threshold for InputNorm (0 = disabled)
    """

    def __init__(self, config: HEDBERTConfig) -> None:
        super().__init__()

        scales = config.encoder_scales
        hidden = config.encoder_hidden

        if not scales:
            raise ValueError("encoder_scales must not be empty.")
        for i, k in enumerate(scales):
            if k < 1:
                raise ValueError(
                    f"encoder_scales[{i}] = {k}; all kernel sizes must be >= 1."
                )
        if len(hidden) != 1:
            raise ValueError(
                f"For parallel encoder, encoder_hidden must have exactly 1 entry "
                f"(the shared hidden dim). Got {len(hidden)} entries."
            )
        if hidden[0] < 1:
            raise ValueError(f"encoder_hidden[0] = {hidden[0]}; must be >= 1.")

        self._scales = sorted(scales)  # sorted ascending for clarity
        self._hidden_dim = hidden[0]

        # InputNorm: configurable normalization (instance or mean_scale)
        self.input_norm = InputNorm(
            n_channels=config.target_channels,
            max_amplitude=config.max_amplitude,
            mode=getattr(config, "norm_mode", "instance"),
            scale=getattr(config, "norm_scale", 200.0),
        )

        # One Conv1d branch per scale: Conv1d(channels, hidden, k, stride=k)
        # + BatchNorm + GELU + Dropout
        self.branches = nn.ModuleList()
        for kernel_size in self._scales:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        config.target_channels,
                        self._hidden_dim,
                        kernel_size=kernel_size,
                        stride=kernel_size,
                        bias=False,
                    ),
                    nn.BatchNorm1d(self._hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
            )

        # Projection from hidden_dim to embed_dim
        self.projection = nn.Sequential(
            nn.Linear(self._hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

        # Total stride = finest branch (smallest kernel size)
        self._total_stride = self._scales[0]

    @property
    def total_stride(self) -> int:
        """Total temporal downsampling factor (finest branch stride)."""
        return self._total_stride

    def forward(
        self, eeg: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode raw EEG into patch embeddings via parallel multi-scale branches.

        Args:
            eeg: (batch, channels, time) raw EEG after channel harmonization.
            attention_mask: (batch, time) with 1.0 for real data, 0.0 for padding.

        Returns:
            patches: (batch, n_patches, embed_dim) patch embeddings.
            patch_mask: (batch, n_patches) float attention mask.
            normalized_eeg: (batch, channels, time) post-InputNorm output.
            clamped_eeg: (batch, channels, time) post-clamp pre-InstanceNorm.
                Preserves amplitude structure for masked reconstruction target.
        """
        if eeg.ndim != 3:
            raise ValueError(
                f"Expected eeg with shape (batch, channels, time), got {eeg.shape}"
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"Expected attention_mask with shape (batch, time), "
                f"got {attention_mask.shape}"
            )
        if eeg.shape[0] != attention_mask.shape[0]:
            raise ValueError(
                f"Batch size mismatch: eeg has {eeg.shape[0]}, "
                f"attention_mask has {attention_mask.shape[0]}"
            )
        if eeg.shape[2] != attention_mask.shape[1]:
            raise ValueError(
                f"Time dimension mismatch: eeg has {eeg.shape[2]} time steps, "
                f"attention_mask has {attention_mask.shape[1]}"
            )

        finest_stride = self._total_stride
        if eeg.shape[2] < finest_stride:
            raise RuntimeError(
                f"Input time dimension ({eeg.shape[2]}) is shorter than the "
                f"finest stride ({finest_stride}). Minimum input length "
                f"for encoder_scales={self._scales} is {finest_stride} samples."
            )

        # Apply learnable input normalization
        x, clamped = self.input_norm(eeg)

        # Run each branch and upsample coarse ones to finest resolution
        finest_branch_out = self.branches[0](x)  # finest scale (smallest kernel)
        n_patches_finest = finest_branch_out.shape[2]

        if n_patches_finest == 0:
            raise RuntimeError(
                f"Input time dimension ({eeg.shape[2]}) is too short for "
                f"encoder_scales={self._scales}. The finest branch produced "
                f"0 patches."
            )

        # Accumulate branch outputs (all upsampled to finest resolution)
        combined = finest_branch_out

        for branch in list(self.branches)[1:]:
            branch_out = branch(x)  # (batch, hidden_dim, n_patches_coarse)
            if branch_out.shape[2] == 0:
                # Coarse branch produced 0 patches; skip (input too short)
                continue
            # Upsample to finest resolution
            upsampled = fn.interpolate(
                branch_out,
                size=n_patches_finest,
                mode="linear",
                align_corners=False,
            )
            combined = combined + upsampled

        # Transpose to (batch, n_patches, hidden_dim) and project
        combined = combined.transpose(1, 2)
        patches = self.projection(combined)

        # Downsample attention mask to finest branch resolution
        patch_mask = _downsample_mask(attention_mask, finest_stride, n_patches_finest)

        return patches, patch_mask, x, clamped
