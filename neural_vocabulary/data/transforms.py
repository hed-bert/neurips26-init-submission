"""Minimal preprocessing pipeline for EEG data."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mne

    from neural_vocabulary.configs import HEDBERTConfig


class MinimalPreprocessing:
    """Minimal, deterministic preprocessing: resample, high-pass, re-reference.

    No ICA, no artifact rejection, no spectral decomposition.
    Runs on CPU; identical across all compute tiers.
    """

    def __init__(self, sfreq: float = 100.0, l_freq: float = 0.5) -> None:
        self.sfreq = sfreq
        self.l_freq = l_freq

    def __call__(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply preprocessing to a copy of the raw data.

        Steps:
            1. Resample to target sampling frequency
            2. High-pass filter (4th-order Butterworth IIR)
            3. Average re-reference (EEG channels only)
        """
        raw = raw.copy()
        raw.resample(self.sfreq)
        raw.filter(
            l_freq=self.l_freq,
            h_freq=None,
            method="iir",
            iir_params={"order": 4, "ftype": "butter"},
            picks="eeg",
        )
        raw.set_eeg_reference("average", projection=False)
        return raw

    @classmethod
    def from_config(cls, config: HEDBERTConfig) -> MinimalPreprocessing:
        """Create from an HEDBERTConfig."""
        return cls(sfreq=config.sfreq, l_freq=config.l_freq)

    def __repr__(self) -> str:
        return f"MinimalPreprocessing(sfreq={self.sfreq}, l_freq={self.l_freq})"
