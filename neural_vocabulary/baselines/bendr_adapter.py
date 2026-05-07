"""BENDR adapter for  Gate Lit baseline probe.

Loads pretrained BENDR (Kostas et al. 2021) from braindecode/braindecode-bendr
(HuggingFace, Apache 2.0) and extracts frozen embeddings from HBN preprocessed
epochs. Model code vendored from braindecode 1.4.0 to avoid the torchaudio
import chain; only torch and einops required at runtime.

Checkpoint format: pytorch_model.bin with keys encoder_state_dict,
contextualizer_state_dict, config (original BENDR format, not braindecode
unified dict). Short HBN epochs (220 samples @ 100 Hz → ~563 @ 256 Hz) yield
only ~5-6 context tokens after the 96x encoder stride.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn

logger = logging.getLogger(__name__)

# Canonical 10-20 channel set used by BENDR (19 EEG + 1 relative amplitude).
BENDR_CHANNELS: tuple[str, ...] = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
)

# BENDR pretrained sampling frequency.
BENDR_SFREQ: float = 256.0

# HBN preprocessed sampling frequency.
HBN_SFREQ: float = 100.0

# Total temporal downsampling factor of BENDR encoder (product of strides).
BENDR_TOTAL_STRIDE: int = 96  # 3 * 2 * 2 * 2 * 2 * 2

# Minimum input samples at 256 Hz for at least 1 context token.
BENDR_MIN_SAMPLES: int = BENDR_TOTAL_STRIDE  # 96 samples


class _ConvEncoderBENDR(nn.Module):
    """Six-block Conv1d encoder producing BENDR features."""

    def __init__(
        self,
        in_features: int,
        encoder_h: int = 512,
        enc_width: tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        dropout: float = 0.0,
        projection_head: bool = False,
        enc_downsample: tuple[int, ...] = (3, 2, 2, 2, 2, 2),
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.encoder_h = encoder_h
        # Ensure odd kernel sizes.
        enc_width_adj = tuple(e if e % 2 != 0 else e + 1 for e in enc_width)
        self._downsampling = enc_downsample
        self._width = enc_width_adj

        current = in_features
        self.encoder = nn.Sequential()
        for i, (w, d) in enumerate(zip(enc_width_adj, enc_downsample, strict=False)):
            self.encoder.add_module(
                f"Encoder_{i}",
                nn.Sequential(
                    nn.Conv1d(current, encoder_h, w, stride=d, padding=w // 2),
                    nn.Dropout1d(dropout),
                    nn.GroupNorm(encoder_h // 2, encoder_h),
                    activation(),
                ),
            )
            current = encoder_h
        if projection_head:
            self.encoder.add_module(
                "projection_head", nn.Conv1d(encoder_h, encoder_h, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _BENDRContextualizer(nn.Module):
    """Transformer-based contextualizer for BENDR."""

    def __init__(
        self,
        in_features: int,
        hidden_feedforward: int = 3076,
        heads: int = 8,
        layers: int = 8,
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.GELU,
        position_encoder: int = 25,
        layer_drop: float = 0.0,
        start_token: int = -5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.layer_drop = layer_drop
        self.start_token = start_token
        self.transformer_dim = 3 * in_features
        self.in_features = in_features

        if position_encoder > 0:
            conv = nn.Conv1d(
                in_features,
                in_features,
                kernel_size=position_encoder,
                padding=position_encoder // 2,
                groups=16,
            )
            nn.init.normal_(conv.weight, mean=0, std=2 / self.transformer_dim)
            assert conv.bias is not None
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
            self.relative_position = nn.Sequential(conv, activation())

        self.input_conditioning = nn.Sequential(
            Rearrange("batch channel time -> batch time channel"),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, self.transformer_dim),
            Rearrange("batch time channel -> time batch channel"),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=heads,
            dim_feedforward=hidden_feedforward,
            dropout=dropout,
            activation=activation(),
            batch_first=False,
            norm_first=False,
        )
        encoder_layer.norm1 = nn.Identity()
        encoder_layer.norm2 = nn.Identity()
        self.transformer_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(layers)]
        )

        self.norm = nn.LayerNorm(self.transformer_dim)
        self.output_layer = nn.Conv1d(self.transformer_dim, in_features, 1)
        self.apply(self._init_bert_params)

    def _init_bert_params(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data = (
                0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data
            )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "relative_position"):
            pos_enc = self.relative_position(x)
            x = x + pos_enc
        x = self.input_conditioning(x)
        if self.start_token is not None:
            token_emb = torch.full(
                (1, x.shape[1], x.shape[2]),
                float(self.start_token),
                device=x.device,
                requires_grad=False,
            )
            x = torch.cat([token_emb, x], dim=0)
        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x)
        x = self.norm(x)
        x = Rearrange("time batch channel -> batch channel time")(x)
        x = self.output_layer(x)
        return x


_HF_REPO_ID = "braindecode/braindecode-bendr"
_HF_FILENAME = "pytorch_model.bin"


def load_bendr(
    checkpoint: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[_ConvEncoderBENDR, _BENDRContextualizer, dict[str, Any]]:
    """Load pretrained BENDR encoder and contextualizer.

    Args:
        checkpoint: Path to local pytorch_model.bin, or None to download
            from HuggingFace Hub (braindecode/braindecode-bendr).
        device: Target device. Defaults to CPU.

    Returns:
        (encoder, contextualizer, config) tuple. Both models are in eval
        mode with frozen parameters.

    Raises:
        RuntimeError: If checkpoint keys are missing or mismatched.
    """
    if device is None:
        device = torch.device("cpu")

    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(_HF_REPO_ID, _HF_FILENAME)
        logger.info("Downloaded BENDR checkpoint from HuggingFace to %s", ckpt_path)
    else:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"BENDR checkpoint not found: {ckpt_path}")
        logger.info("Loading BENDR checkpoint from %s", ckpt_path)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

    required_keys = {"encoder_state_dict", "contextualizer_state_dict", "config"}
    missing = required_keys - set(ckpt.keys())
    if missing:
        raise RuntimeError(
            f"BENDR checkpoint missing keys: {missing}. "
            f"Expected format: {required_keys}. Found: {set(ckpt.keys())}"
        )

    cfg: dict[str, Any] = ckpt["config"]
    logger.info("BENDR config: %s", cfg)

    encoder = _ConvEncoderBENDR(
        in_features=cfg["n_chans"],
        encoder_h=cfg.get("encoder_h", 512),
    )
    encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)

    contextualizer = _BENDRContextualizer(
        in_features=cfg.get("encoder_h", 512),
        hidden_feedforward=cfg.get("contextualizer_hidden", 3076),
        heads=cfg.get("transformer_heads", 8),
        layers=cfg.get("transformer_layers", 8),
        dropout=cfg.get("drop_prob", 0.1),
        position_encoder=cfg.get("position_encoder_length", 25),
        layer_drop=cfg.get("layer_drop", 0.0),
        start_token=cfg.get("start_token", -5),
    )
    contextualizer.load_state_dict(ckpt["contextualizer_state_dict"], strict=True)

    encoder = encoder.to(device).eval()
    contextualizer = contextualizer.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in contextualizer.parameters():
        p.requires_grad_(False)

    logger.info(
        "BENDR loaded: n_chans=%d, encoder_h=%d",
        cfg["n_chans"],
        cfg.get("encoder_h", 512),
    )
    return encoder, contextualizer, cfg


def select_channels(
    eeg: np.ndarray,
    source_channels: list[str],
    target_channels: tuple[str, ...] = BENDR_CHANNELS,
) -> np.ndarray:
    """Select target channels from EEG array by name.

    Args:
        eeg: (n_source_channels, n_times) float32 array.
        source_channels: Channel names corresponding to eeg rows.
        target_channels: Ordered target channel names to extract.

    Returns:
        (len(target_channels), n_times) float32 array.

    Raises:
        RuntimeError: If any target channel is not found in source.
    """
    missing = [c for c in target_channels if c not in source_channels]
    if missing:
        raise RuntimeError(
            f"Required BENDR channels not found in source data: {missing}. "
            f"Source channels: {source_channels[:10]}... ({len(source_channels)} total)"
        )
    src_idx = [source_channels.index(c) for c in target_channels]
    return eeg[src_idx]


def add_relative_amplitude_channel(eeg_19ch: np.ndarray) -> np.ndarray:
    """Append relative amplitude as 20th channel.

    Relative amplitude = mean absolute amplitude per time step, normalized
    by its own maximum so it lives in [0, 1]. This matches the BENDR
    pretraining procedure (Kostas et al. 2021, Section 2.2).

    Args:
        eeg_19ch: (19, n_times) float32 array.

    Returns:
        (20, n_times) float32 array.
    """
    rel_amp = np.abs(eeg_19ch).mean(axis=0)  # (n_times,)
    max_amp = rel_amp.max()
    rel_amp = rel_amp / (max_amp + 1e-8)
    return np.vstack([eeg_19ch, rel_amp[np.newaxis]])


def resample_epoch(
    eeg: np.ndarray, source_sfreq: float, target_sfreq: float
) -> np.ndarray:
    """Resample EEG epoch using MNE (scipy under the hood).

    Args:
        eeg: (n_channels, n_times) float array.
        source_sfreq: Source sampling frequency in Hz.
        target_sfreq: Target sampling frequency in Hz.

    Returns:
        Resampled (n_channels, n_times_new) float32.
    """
    import mne

    # mne.filter.resample requires float64
    resampled = mne.filter.resample(
        eeg.astype(np.float64), up=target_sfreq, down=source_sfreq, npad="auto"
    )
    return resampled.astype(np.float32)


def extract_bendr_embedding(
    eeg_20ch: np.ndarray,
    encoder: _ConvEncoderBENDR,
    contextualizer: _BENDRContextualizer,
    device: torch.device,
) -> np.ndarray:
    """Extract a single embedding vector from a preprocessed EEG epoch.

    The start-token output (index 0) of the contextualizer is used as the
    aggregate representation, following BERT [CLS] convention.

    Args:
        eeg_20ch: (20, n_times) float32 array at 256 Hz.
        encoder: Pretrained BENDR encoder.
        contextualizer: Pretrained BENDR contextualizer.
        device: Computation device.

    Returns:
        (encoder_h,) float32 numpy embedding.

    Raises:
        RuntimeError: If input is too short for the encoder.
    """
    n_times = eeg_20ch.shape[1]
    if n_times < BENDR_MIN_SAMPLES:
        raise RuntimeError(
            f"Input has {n_times} samples at 256 Hz but BENDR encoder requires "
            f"at least {BENDR_MIN_SAMPLES} samples (1 context token). "
            f"Increase epoch length or resample from a higher source rate."
        )
    x = torch.from_numpy(eeg_20ch).unsqueeze(0).to(device)  # (1, 20, T)
    with torch.no_grad():
        encoded = encoder(x)  # (1, 512, T/96)
        ctx = contextualizer(encoded)  # (1, 512, T/96 + 1)
        emb = ctx[:, :, 0].squeeze(0)  # (512,) — start token
    return emb.cpu().numpy().astype(np.float32)


def load_channel_names_from_h5(h5_path: str | Path) -> list[str]:
    """Read channel_names attribute from a preprocessed h5 file.

    Args:
        h5_path: Path to an HBN preprocessed h5 file.

    Returns:
        List of channel name strings.

    Raises:
        RuntimeError: If the attribute is missing or malformed.
    """
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        raw = f.attrs.get("channel_names")
        if raw is None:
            raise RuntimeError(
                f"h5 file {h5_path} has no 'channel_names' attribute. "
                "Is this a preprocessed file?"
            )
        if isinstance(raw, str):
            channels = json.loads(raw)
        elif isinstance(raw, (list, np.ndarray)):
            channels = list(raw)
        else:
            raise RuntimeError(
                f"Unexpected type for channel_names in {h5_path}: {type(raw)}"
            )
    return channels
