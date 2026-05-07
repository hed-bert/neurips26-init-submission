"""CBraMod adapter for  W2 E8 FM benchmark.

Loads pretrained CBraMod (Wang et al. ICLR 2025) from
``braindecode/cbramod-pretrained`` (HuggingFace, public) and exposes a
clean ``(B, n_chans, n_times) -> (B, embed_dim)`` encoder for the probe.

CBraMod is a criss-cross transformer with separated spatial / temporal
attention paths, pretrained on TUEG via masked patch reconstruction.
~5M params — within an order of magnitude of our the headlineTiny (2M),
making it the **scale-matched SOTA comparison** for the  paper.

Forward returns ``(B, n_chans, T_patches, embed_dim)`` via
``return_features=True``; we mean-pool over channels and patches.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

CBRAMOD_REPO: str = "braindecode/cbramod-pretrained"
CBRAMOD_SFREQ: float = 200.0  # CBraMod pretrained sampling rate
CBRAMOD_PATCH_SIZE: int = 200  # 1 s per patch
CBRAMOD_EMBED_DIM: int = 200


class _CBraModPoolingEncoder(nn.Module):
    """Wrap CBraMod to mean-pool features to ``(B, embed_dim)``.

    The matched-compute wrapper expects ``encoder(x) -> (B, D)``; CBraMod's
    ``return_features=True`` path produces ``(B, n_chans, T_patches, D)``.
    """

    def __init__(self, cbramod: nn.Module):
        super().__init__()
        self.cbramod = cbramod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"CBraMod forward expects (B,C,T); got {tuple(x.shape)}")
        out = self.cbramod(x, return_features=True)
        feats = out["features"]  # (B, C, T_patches, embed_dim)
        return feats.mean(dim=(1, 2))  # (B, embed_dim)


def load_cbramod(
    n_chans: int,
    n_times: int,
    device: torch.device,
) -> _CBraModPoolingEncoder:
    """Load pretrained CBraMod from HF and wrap for the probe.

    Parameters
    ----------
    n_chans : int
        Source channel count (HBN preprocessed is 64).
    n_times : int
        Number of time samples *at CBraMod's 200 Hz*. The wrapper's
        ``preprocess_batch`` resamples 100 Hz HBN to 200 Hz before forward;
        CBraMod's patch_size=200 (1 s per patch).
    device : torch.device
        Target device.
    """
    from braindecode.models import CBraMod  # heavy import deferred

    # n_outputs=5 satisfies from_pretrained's head builder; we drop the
    # head via return_features=True so the value is irrelevant.
    cbramod = CBraMod.from_pretrained(
        CBRAMOD_REPO,
        n_outputs=5,
        n_chans=n_chans,
        n_times=n_times,
        sfreq=CBRAMOD_SFREQ,
    )
    n_params = sum(p.numel() for p in cbramod.parameters())
    logger.info(
        "CBraMod loaded: %s (%d params, embed_dim=%d)",
        CBRAMOD_REPO,
        n_params,
        CBRAMOD_EMBED_DIM,
    )

    encoder = _CBraModPoolingEncoder(cbramod).to(device)
    return encoder
