"""Pre-computed channel interpolation for cross-dataset harmonization.

Maps any source EEG montage to a standard 64-channel target montage
using inverse-distance weighted interpolation from 3D electrode positions.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import torch
from scipy.spatial.distance import cdist

# Standard 64-channel names (10-10 subset used as target)
TARGET_64_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "TP9",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "TP10",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F5",
    "F1",
    "F2",
    "F6",
    "FT9",
    "FT7",
    "FC3",
    "FC4",
    "FT8",
    "FT10",
    "C5",
    "C1",
    "C2",
    "C6",
    "TP7",
    "CP3",
    "CPz",
    "CP4",
    "TP8",
    "P5",
    "P1",
    "P2",
    "P6",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
]


class ChannelHarmonization:
    """Pre-computed inverse-distance interpolation from source to target montage.

    The interpolation matrix W has shape (target_channels, source_channels).
    Applying y = W @ x maps source-montage EEG to the target montage.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        source_channels: list[str],
        target_channels: list[str],
    ) -> None:
        self._matrix = matrix.detach().clone()
        self._matrix.requires_grad_(False)
        self.source_channels = source_channels
        self.target_channels = target_channels

    @property
    def matrix(self) -> torch.Tensor:
        """Interpolation matrix, shape (target_ch, source_ch). No gradients."""
        return self._matrix

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        """Apply channel harmonization.

        Args:
            eeg: tensor of shape (..., source_channels, time)

        Returns:
            Tensor of shape (..., target_channels, time)
        """
        mat = self._matrix.to(eeg.device, eeg.dtype)
        return torch.matmul(mat, eeg)

    @classmethod
    def from_montages(
        cls,
        source_montage: mne.channels.DigMontage,
        source_ch_names: list[str],
        target_montage: mne.channels.DigMontage | None = None,
        target_ch_names: list[str] | None = None,
        power: float = 2.0,
    ) -> ChannelHarmonization:
        """Create from MNE montage objects.

        Args:
            source_montage: montage with source electrode positions.
            source_ch_names: channel names in the source data.
            target_montage: target montage. Defaults to standard 10-10.
            target_ch_names: target channel names. Defaults to TARGET_64_CHANNELS.
            power: inverse-distance weighting exponent.
        """
        if target_montage is None:
            target_montage = mne.channels.make_standard_montage("standard_1020")
        if target_ch_names is None:
            target_ch_names = TARGET_64_CHANNELS

        matrix = _compute_interpolation_matrix(
            source_montage=source_montage,
            source_ch_names=source_ch_names,
            target_montage=target_montage,
            target_ch_names=target_ch_names,
            power=power,
        )

        return cls(
            matrix=matrix,
            source_channels=source_ch_names,
            target_channels=target_ch_names,
        )

    @classmethod
    def for_things_eeg(cls) -> ChannelHarmonization:
        """THINGS-EEG: 63 channels -> 64 channels.

        Cz was the online reference and is not in the recorded data.
        After average re-referencing, Cz is zero by construction.
        We insert a zero row for Cz and build a near-identity mapping.
        """
        montage = mne.channels.make_standard_montage("standard_1020")
        all_positions = montage.get_positions()["ch_pos"]

        # THINGS-EEG channels: standard 10-10 minus Cz
        source_names = [ch for ch in TARGET_64_CHANNELS if ch != "Cz"]
        # Verify we have 63
        source_names = [ch for ch in source_names if ch in all_positions]

        return cls.from_montages(
            source_montage=montage,
            source_ch_names=source_names,
            target_montage=montage,
            target_ch_names=TARGET_64_CHANNELS,
        )

    @classmethod
    def for_erp_core(
        cls, electrodes_tsv: str | Path | None = None
    ) -> ChannelHarmonization:
        """ERP-CORE: 30 EEG channels -> 64 channels.

        Uses standard BioSemi 10-20 positions. If electrodes_tsv is
        provided, reads exact positions from the BIDS file.
        """
        erp_core_channels = [
            "Fp1",
            "Fz",
            "F3",
            "F7",
            "FC3",
            "C3",
            "P3",
            "O1",
            "AF7",
            "AF3",
            "F1",
            "FC1",
            "C1",
            "CP1",
            "CP3",
            "P1",
            "Fp2",
            "F4",
            "F8",
            "FC4",
            "C4",
            "P4",
            "O2",
            "AF8",
            "AF4",
            "F2",
            "FC2",
            "C2",
            "CP2",
            "Pz",
        ]

        montage = mne.channels.make_standard_montage("standard_1020")

        return cls.from_montages(
            source_montage=montage,
            source_ch_names=erp_core_channels,
            target_montage=montage,
            target_ch_names=TARGET_64_CHANNELS,
        )

    @classmethod
    def for_hbn_eeg(cls) -> ChannelHarmonization:
        """HBN-EEG: 129 channels (E1-E128 + Cz) -> 64 channels.

        Uses the GSN-HydroCel-129 montage (HydroCel Geodesic Sensor Net)
        with inverse-distance weighted interpolation to the standard
        10-10 target layout.
        """
        gsn_montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        source_names = [f"E{i}" for i in range(1, 129)] + ["Cz"]

        return cls.from_montages(
            source_montage=gsn_montage,
            source_ch_names=source_names,
        )

    @classmethod
    def for_physionet_mi(cls) -> ChannelHarmonization:
        """PhysioNet MI (ds004362): 64 channels -> 64 target channels.

        The source montage includes channels like Iz, Fpz, AFz, FCz,
        T9, T10 absent from the target 64, while the target includes
        FT9, FT10, PO9, PO10, TP9, TP10 not in the source.
        Interpolation handles the non-overlapping channels in each direction.
        """
        # PhysioNet MI channels in the order they appear in the .set files,
        # mapped to standard_1020 capitalization
        physionet_mi_channels = [
            "FC5",
            "FC3",
            "FC1",
            "FCz",
            "FC2",
            "FC4",
            "FC6",
            "C5",
            "C3",
            "C1",
            "Cz",
            "C2",
            "C4",
            "C6",
            "CP5",
            "CP3",
            "CP1",
            "CPz",
            "CP2",
            "CP4",
            "CP6",
            "Fp1",
            "Fpz",
            "Fp2",
            "AF7",
            "AF3",
            "AFz",
            "AF4",
            "AF8",
            "F7",
            "F5",
            "F3",
            "F1",
            "Fz",
            "F2",
            "F4",
            "F6",
            "F8",
            "FT7",
            "FT8",
            "T7",
            "T8",
            "T9",
            "T10",
            "TP7",
            "TP8",
            "P7",
            "P5",
            "P3",
            "P1",
            "Pz",
            "P2",
            "P4",
            "P6",
            "P8",
            "PO7",
            "PO3",
            "POz",
            "PO4",
            "PO8",
            "O1",
            "Oz",
            "O2",
            "Iz",
        ]

        montage = mne.channels.make_standard_montage("standard_1020")

        return cls.from_montages(
            source_montage=montage,
            source_ch_names=physionet_mi_channels,
            target_montage=montage,
            target_ch_names=TARGET_64_CHANNELS,
        )

    def save(self, path: str | Path) -> None:
        """Save interpolation matrix and channel info."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "matrix": self._matrix,
                "source_channels": self.source_channels,
                "target_channels": self.target_channels,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> ChannelHarmonization:
        """Load a saved ChannelHarmonization."""
        data = torch.load(Path(path), weights_only=True)
        return cls(
            matrix=data["matrix"],
            source_channels=data["source_channels"],
            target_channels=data["target_channels"],
        )

    def __repr__(self) -> str:
        return (
            f"ChannelHarmonization("
            f"{len(self.source_channels)}->{len(self.target_channels)})"
        )


def _compute_interpolation_matrix(
    source_montage: mne.channels.DigMontage,
    source_ch_names: list[str],
    target_montage: mne.channels.DigMontage,
    target_ch_names: list[str],
    power: float = 2.0,
) -> torch.Tensor:
    """Compute inverse-distance weighted interpolation matrix.

    For each target channel, weights are computed as:
        w_i = 1 / d_i^power
        W_row = w / sum(w)

    If a target channel exactly matches a source channel (distance < 1mm),
    it gets weight 1.0 and all others get 0.0.
    """
    source_pos = _get_positions(source_montage, source_ch_names)
    target_pos = _get_positions(target_montage, target_ch_names)

    # Distance matrix: (n_target, n_source)
    dist = cdist(target_pos, source_pos, metric="euclidean")

    # Inverse-distance weighting
    epsilon = 1e-6  # avoid division by zero
    weights = 1.0 / (dist + epsilon) ** power

    # For exact matches (< 1mm), use identity
    exact_match = dist < 0.001  # 1mm in meters
    for i in range(len(target_ch_names)):
        matches = np.where(exact_match[i])[0]
        if len(matches) > 0:
            weights[i, :] = 0.0
            weights[i, matches[0]] = 1.0

    # Normalize rows to sum to 1
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / row_sums

    return torch.tensor(weights, dtype=torch.float32)


def _get_positions(
    montage: mne.channels.DigMontage,
    ch_names: list[str],
) -> np.ndarray:
    """Extract 3D positions for the given channel names from a montage.

    Returns array of shape (n_channels, 3).
    """
    positions = montage.get_positions()["ch_pos"]
    pos_list = []
    for name in ch_names:
        if name not in positions:
            raise ValueError(
                f"Channel '{name}' not found in montage. "
                f"Available: {sorted(positions.keys())[:20]}..."
            )
        pos_list.append(positions[name])
    return np.array(pos_list)
