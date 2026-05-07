"""REVE adapter for  W2 E8 FM benchmark.

Loads pretrained REVE (El Ouahidi et al. NeurIPS 2025) from
``brain-bzh/reve-base`` or ``brain-bzh/reve-large`` (HuggingFace, gated)
and exposes a clean ``(B, n_chans, n_times) -> (B, embed_dim)`` encoder
suitable for the matched-compute probe wrapper.

Design choices
==============

- **Pretrained weights are gated.** Caller must hold an HF token with
  approved access to the brain-bzh repos. The adapter downloads via
  ``REVE.from_pretrained`` which uses ``HF_TOKEN`` / ``huggingface-cli
  login`` from the runtime environment. There is no fallback URL.

- **Channel positions are required by REVE's 4D positional encoding.**
  We resolve them from MNE's ``standard_1020`` montage at adapter init
  time and register the resulting ``(n_chans, 3)`` tensor as a buffer
  on the wrapper module. This means the channel layout is fixed for the
  adapter's lifetime; per-batch ``pos`` is just a broadcast copy.

- **Mean-pool features.** REVE's pretrained checkpoints use
  ``attention_pooling=False``; ``return_features=True`` yields
  ``(B, n_chans, T_patches, embed_dim)``. We mean-pool over channels and
  patches to produce ``(B, embed_dim)`` for the linear probe head.
  REVE-base ``embed_dim=512``; REVE-large ``embed_dim=1216``.

- **Sampling frequency: 200 Hz.** REVE's ``patch_size=200`` is fixed at
  pretraining-time-1-second; HBN preprocessed (100 Hz) is resampled in the
  per-batch preprocess step.

References
----------
El Ouahidi et al. (2025). REVE: Representation for EEG with Versatile
Embeddings. NeurIPS 2025.
"""

from __future__ import annotations

import logging

import mne
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

REVE_SFREQ: float = 200.0  # REVE pretrained sampling rate (patch_size = 200 = 1 s)

REVE_REPOS: dict[str, str] = {
    "reve_base": "brain-bzh/reve-base",
    "reve_large": "brain-bzh/reve-large",
}

REVE_EMBED_DIM: dict[str, int] = {
    "reve_base": 512,
    "reve_large": 1216,
}


def channel_positions_1020(ch_names: list[str]) -> torch.Tensor:
    """Resolve ``(n_chans, 3)`` x/y/z positions from the 10-20 montage.

    Raises if any channel is missing from ``standard_1020``. The 64-channel
    HBN-harmonized layout has full coverage; verified at adapter-test time.
    """
    montage = mne.channels.make_standard_montage("standard_1020")
    pos_map = montage.get_positions()["ch_pos"]
    missing = [n for n in ch_names if n not in pos_map]
    if missing:
        raise RuntimeError(
            f"REVE channel positions: {len(missing)} channels missing from "
            f"standard_1020 montage: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    pos = np.stack([pos_map[n] for n in ch_names], axis=0).astype(np.float32)
    return torch.from_numpy(pos)


class _RevePoolingEncoder(nn.Module):
    """Wrap REVE + cached channel positions to expose a clean ``forward(x)``.

    The matched-compute wrapper calls the encoder via ``self.encoder(x)``;
    REVE's native API requires ``forward(eeg, pos=...)``. This module hides
    the ``pos`` plumbing and the mean-pool step so the upstream wrapper
    treats REVE identically to LaBraM/BIOT/BENDR.
    """

    def __init__(self, reve_model: nn.Module, channel_pos: torch.Tensor):
        super().__init__()
        self.reve = reve_model
        if channel_pos.dim() != 2 or channel_pos.size(-1) != 3:
            raise ValueError(
                f"channel_pos must be (n_chans, 3); got {tuple(channel_pos.shape)}"
            )
        self.register_buffer("channel_pos", channel_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_chans, n_times) at 200 Hz
        if x.dim() != 3:
            raise ValueError(f"REVE forward expects (B,C,T); got {tuple(x.shape)}")
        b = x.size(0)
        # ty cannot narrow Module's register_buffer attribute to Tensor.
        pos = self.channel_pos.unsqueeze(0).expand(b, -1, -1)  # type: ignore[call-non-callable]  # (B, C, 3)
        out = self.reve(x, pos=pos, return_features=True)
        feats = out["features"]  # (B, C, T_patches, embed_dim)
        return feats.mean(dim=(1, 2))  # (B, embed_dim)


def load_reve(
    model_size: str,
    n_chans: int,
    n_times: int,
    ch_names: list[str],
    device: torch.device,
) -> _RevePoolingEncoder:
    """Load pretrained REVE from HuggingFace and wrap it for the probe.

    Parameters
    ----------
    model_size : str
        Either ``"reve_base"`` or ``"reve_large"``.
    n_chans : int
        Number of input EEG channels (64 for HBN preprocessed).
    n_times : int
        Number of time samples per epoch *at REVE's 200 Hz*. The wrapper's
        ``preprocess_batch`` resamples 100 Hz HBN to 200 Hz before forward.
    ch_names : list[str]
        Channel names in the same order as the data; used to resolve
        ``standard_1020`` positions.
    device : torch.device
        Target device.

    Returns
    -------
    _RevePoolingEncoder
        A module exposing ``forward(x: (B,C,T)) -> (B, embed_dim)``.
    """
    if model_size not in REVE_REPOS:
        raise ValueError(
            f"Unknown REVE size {model_size!r}; expected {list(REVE_REPOS)}"
        )
    if len(ch_names) != n_chans:
        raise ValueError(f"ch_names length {len(ch_names)} != n_chans {n_chans}")

    from braindecode.models import REVE  # heavy import deferred

    repo = REVE_REPOS[model_size]
    chs_info = [{"ch_name": n} for n in ch_names]
    # n_outputs is required by from_pretrained but the head is dropped via
    # return_features=True; any value works.
    reve = REVE.from_pretrained(
        repo,
        n_outputs=5,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=REVE_SFREQ,
        chs_info=chs_info,
    )
    n_params = sum(p.numel() for p in reve.parameters())
    logger.info(
        "REVE loaded: %s (%d params, embed_dim=%d)",
        repo,
        n_params,
        REVE_EMBED_DIM[model_size],
    )

    pos = channel_positions_1020(ch_names)
    encoder = _RevePoolingEncoder(reve, pos).to(device)
    return encoder
