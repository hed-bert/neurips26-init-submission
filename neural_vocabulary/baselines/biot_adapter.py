"""BIOT encoder adapter for HBN preprocessed EEG data.

BIOT (Cross-data Biosignal Learning in the Wild, Yang et al., NeurIPS 2023)
tokenizes each channel independently via STFT, adds a learned channel-identity
token, and aggregates across all (channel × time) tokens with a Linear Attention
Transformer. The pretrained checkpoint uses 16 bipolar EEG channels at 200 Hz.

Checkpoint: EEG-PREST-16-channels.ckpt (MIT license, ~13 MB)
Source: https://github.com/ycq091044/BIOT
Download: https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/EEG-PREST-16-channels.ckpt

HBN adaptation:
    - HBN preprocessed: 64 referential channels, 100 Hz, 220 samples (2.2 s)
    - BIOT expects:  16 bipolar channels,   200 Hz, 2000 samples (10 s)
    - Bipolar derivation: subtract electrode pairs (e.g. FP1-F7 = Fp1 - F7)
    - Resample 100 → 200 Hz via scipy.signal.resample_poly
    - Zero-pad to 2000 samples
    - Missing electrode raises ValueError (never silent drop)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BIOT channel definitions
# ---------------------------------------------------------------------------

#: 16 bipolar pairs used by EEG-PREST-16-channels.ckpt.
#: Each entry is (anode, cathode) in the HBN 10-20 naming convention.
BIOT_16_PAIRS: list[tuple[str, str]] = [
    ("Fp1", "F7"),
    ("F7", "T7"),
    ("T7", "P7"),
    ("P7", "O1"),
    ("Fp2", "F8"),
    ("F8", "T8"),
    ("T8", "P8"),
    ("P8", "O2"),
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
]

#: Target sampling rate for all BIOT pretrained checkpoints.
BIOT_SFREQ: int = 200

#: Window length expected by the pretrained checkpoint (10 s × 200 Hz).
BIOT_N_SAMPLES: int = 2000

#: STFT parameters matching the pretrained checkpoint.
BIOT_N_FFT: int = 200
BIOT_HOP_LENGTH: int = 100

#: Embedding dimension from the pretrained checkpoint.
BIOT_EMB_SIZE: int = 256


# ---------------------------------------------------------------------------
# Internal BIOT model (vendored from ycq091044/BIOT, MIT license)
# ---------------------------------------------------------------------------


class _PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size: int = 256, n_freq: int = 101) -> None:
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, freq, time) → (batch, time, emb_size)."""
        x = x.permute(0, 2, 1)
        return self.projection(x)


class _PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BIOTEncoder(nn.Module):
    """BIOT encoder: per-channel STFT → patch projection → LinearAttention pooling.

    Adapted verbatim from ycq091044/BIOT/model/biot.py (MIT license).
    Vendored here to avoid a dependency on the entire BIOT repository.
    """

    def __init__(
        self,
        emb_size: int = 256,
        heads: int = 8,
        depth: int = 4,
        n_channels: int = 16,
        n_fft: int = 200,
        hop_length: int = 100,
    ) -> None:
        super().__init__()
        from linear_attention_transformer import LinearAttentionTransformer

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = _PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        self.positional_encoding = _PositionalEncoding(emb_size)
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.index = nn.Parameter(
            torch.arange(n_channels, dtype=torch.long), requires_grad=False
        )

    def _stft(self, sample: torch.Tensor) -> torch.Tensor:
        """sample: (batch, 1, T) → (batch, n_freq, n_frames).

        No window is passed intentionally: the pretrained checkpoint was trained
        with a rectangular window and changing it would break compatibility.
        """
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_channels, T) → (batch, emb_size)."""
        emb_seq: list[torch.Tensor] = []
        for i in range(x.shape[1]):
            spec = self._stft(x[:, i : i + 1, :])  # (batch, freq, time)
            patch_emb = self.patch_embedding(spec)  # (batch, time, emb)
            batch_size, ts, _ = patch_emb.shape
            ch_tok = (
                self.channel_tokens(self.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, ts, -1)
            )
            emb_seq.append(self.positional_encoding(patch_emb + ch_tok))

        emb = torch.cat(emb_seq, dim=1)  # (batch, 16*ts, emb)
        return self.transformer(emb).mean(dim=1)  # (batch, emb)


# ---------------------------------------------------------------------------
# Channel mapping
# ---------------------------------------------------------------------------


def build_bipolar_indices(
    channel_names: list[str],
    pairs: list[tuple[str, str]] = BIOT_16_PAIRS,
) -> tuple[list[int], list[int]]:
    """Return (anode_indices, cathode_indices) for each bipolar pair.

    Parameters
    ----------
    channel_names:
        Ordered list of referential channel names in the EEG array (rows).
    pairs:
        List of (anode_name, cathode_name) bipolar pairs.

    Returns
    -------
    anode_idx, cathode_idx:
        Integer indices into *channel_names* for each pair.

    Raises
    ------
    ValueError
        If any electrode name from a pair is not found in *channel_names*.
        Never silently drops a channel.
    """
    # Build case-insensitive lookup
    name_to_idx: dict[str, int] = {n.upper(): i for i, n in enumerate(channel_names)}

    # First pass: collect ALL missing electrodes before raising, so the error
    # message is complete rather than stopping at the first missing name.
    all_names = [name for pair in pairs for name in pair]
    missing = sorted({n for n in all_names if n.upper() not in name_to_idx})
    if missing:
        raise ValueError(
            f"Cannot compute BIOT bipolar derivations: electrodes not found "
            f"in channel list: {missing}. "
            f"Available: {channel_names}"
        )

    # Second pass: build indices (only reached when all electrodes are present).
    anode_idx = [name_to_idx[anode.upper()] for anode, _ in pairs]
    cathode_idx = [name_to_idx[cathode.upper()] for _, cathode in pairs]
    return anode_idx, cathode_idx


def derive_bipolar(
    eeg: np.ndarray,
    anode_idx: list[int],
    cathode_idx: list[int],
) -> np.ndarray:
    """Compute bipolar derivations from a referential montage.

    Parameters
    ----------
    eeg:
        (n_channels, n_times) float32 array.
    anode_idx, cathode_idx:
        Electrode indices returned by *build_bipolar_indices*.

    Returns
    -------
    bipolar:
        (n_pairs, n_times) float32 array.
    """
    anodes = eeg[anode_idx, :]
    cathodes = eeg[cathode_idx, :]
    return (anodes - cathodes).astype(np.float32)


def resample_and_pad(
    eeg: np.ndarray,
    source_sfreq: int,
    target_sfreq: int = BIOT_SFREQ,
    target_samples: int = BIOT_N_SAMPLES,
) -> np.ndarray:
    """Resample from *source_sfreq* to *target_sfreq*, then zero-pad to *target_samples*.

    Parameters
    ----------
    eeg:
        (n_channels, n_times) array.
    source_sfreq:
        Original sampling rate in Hz.
    target_sfreq:
        Target sampling rate in Hz.
    target_samples:
        Desired output length in samples; shorter signals are zero-padded,
        longer signals are truncated.

    Returns
    -------
    out:
        (n_channels, target_samples) float32 array.
    """
    from math import gcd

    from scipy.signal import resample_poly

    ratio_up = target_sfreq
    ratio_down = source_sfreq
    g = gcd(ratio_up, ratio_down)
    resampled = resample_poly(eeg, ratio_up // g, ratio_down // g, axis=-1)
    n_chan, n_t = resampled.shape
    out = np.zeros((n_chan, target_samples), dtype=np.float32)
    copy_len = min(n_t, target_samples)
    out[:, :copy_len] = resampled[:, :copy_len]
    return out


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


class BIOTAdapter:
    """Frozen BIOT encoder for embedding extraction on HBN preprocessed epochs.

    Usage
    -----
    ::

        adapter = BIOTAdapter.from_checkpoint(
            checkpoint_path=Path("..."),
            channel_names=hbn_channel_names,
            device=torch.device("cuda"),
        )
        emb = adapter.embed(eeg_batch)  # (B, 256)
    """

    def __init__(
        self,
        encoder: BIOTEncoder,
        anode_idx: list[int],
        cathode_idx: list[int],
        source_sfreq: int = 100,
        device: torch.device | None = None,
    ) -> None:
        self.encoder = encoder
        self.anode_idx = anode_idx
        self.cathode_idx = cathode_idx
        self.source_sfreq = source_sfreq
        self.device = device or torch.device("cpu")

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        channel_names: list[str],
        device: torch.device | None = None,
        source_sfreq: int = 100,
        pairs: list[tuple[str, str]] = BIOT_16_PAIRS,
    ) -> BIOTAdapter:
        """Load a pretrained BIOT encoder from *checkpoint_path*.

        Parameters
        ----------
        checkpoint_path:
            Path to the .ckpt file (EEG-PREST-16-channels.ckpt or similar).
        channel_names:
            List of referential EEG channel names in the source data.
        device:
            Torch device.
        source_sfreq:
            Original sampling rate of the source data (Hz).
        pairs:
            Bipolar channel pairs to derive from *channel_names*.

        Raises
        ------
        ValueError
            If any required electrode is not in *channel_names*.
        RuntimeError
            If the checkpoint cannot be loaded or has unexpected keys.
        """
        device = device or torch.device("cpu")
        n_channels = len(pairs)

        encoder = BIOTEncoder(
            emb_size=BIOT_EMB_SIZE,
            heads=8,
            depth=4,
            n_channels=n_channels,
            n_fft=BIOT_N_FFT,
            hop_length=BIOT_HOP_LENGTH,
        )

        logger.info("Loading BIOT checkpoint from %s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = encoder.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"BIOT checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        logger.info(
            "BIOT checkpoint loaded (%d parameters)",
            sum(p.numel() for p in encoder.parameters()),
        )

        anode_idx, cathode_idx = build_bipolar_indices(channel_names, pairs)

        return cls(
            encoder=encoder,
            anode_idx=anode_idx,
            cathode_idx=cathode_idx,
            source_sfreq=source_sfreq,
            device=device,
        )

    @classmethod
    def random_init(
        cls,
        channel_names: list[str],
        device: torch.device | None = None,
        source_sfreq: int = 100,
        pairs: list[tuple[str, str]] = BIOT_16_PAIRS,
    ) -> BIOTAdapter:
        """Create a random-weight BIOT encoder (ablation baseline)."""
        device = device or torch.device("cpu")
        anode_idx, cathode_idx = build_bipolar_indices(channel_names, pairs)
        encoder = BIOTEncoder(
            emb_size=BIOT_EMB_SIZE,
            heads=8,
            depth=4,
            n_channels=len(pairs),
            n_fft=BIOT_N_FFT,
            hop_length=BIOT_HOP_LENGTH,
        )
        return cls(
            encoder=encoder,
            anode_idx=anode_idx,
            cathode_idx=cathode_idx,
            source_sfreq=source_sfreq,
            device=device,
        )

    def preprocess(self, eeg: np.ndarray) -> np.ndarray:
        """Bipolar derivation + resample + pad for a single epoch.

        Parameters
        ----------
        eeg:
            (n_channels, n_times) referential montage array.

        Returns
        -------
        out:
            (16, 2000) float32 array ready for BIOT inference.
        """
        bipolar = derive_bipolar(eeg, self.anode_idx, self.cathode_idx)
        return resample_and_pad(
            bipolar,
            source_sfreq=self.source_sfreq,
            target_sfreq=BIOT_SFREQ,
            target_samples=BIOT_N_SAMPLES,
        )

    @torch.no_grad()
    def embed(self, eeg_batch: np.ndarray) -> np.ndarray:
        """Embed a batch of preprocessed epochs.

        Parameters
        ----------
        eeg_batch:
            (B, 16, 2000) float32 array (output of *preprocess* stacked).

        Returns
        -------
        embeddings:
            (B, 256) float32 numpy array.
        """
        x = torch.from_numpy(eeg_batch).to(self.device)
        emb = self.encoder(x)
        return emb.cpu().float().numpy()
