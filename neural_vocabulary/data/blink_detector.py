"""Simple threshold-based blink detection from EOG or frontal EEG channels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mne
import numpy as np
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from neural_vocabulary.configs import HEDBERTConfig

BLINK_EVENT_CODE = 9999


class BlinkDetector:
    """Detect blinks from EOG channels or frontal EEG (Fp1/Fp2 fallback).

    Blinks are returned as MNE-compatible events that can be merged
    with stimulus/response events to act as epoch boundaries.
    """

    def __init__(
        self,
        eog_channels: list[str] | None = None,
        threshold_uv: float = 100.0,
        min_duration_ms: float = 50.0,
        blink_event_code: int = BLINK_EVENT_CODE,
        sfreq: float = 100.0,
    ) -> None:
        self.eog_channels = eog_channels
        self.threshold_uv = threshold_uv
        self.min_duration_ms = min_duration_ms
        self.blink_event_code = blink_event_code
        self.sfreq = sfreq

    def detect(self, raw: mne.io.Raw) -> np.ndarray:
        """Detect blinks and return MNE-compatible events array.

        Returns:
            Array of shape (n_blinks, 3) with columns
            [sample_index, 0, blink_event_code].
            Returns empty array (0, 3) if no blinks found.
        """
        signal = self._get_eog_signal(raw)

        # Bandpass 1-10 Hz to isolate blink frequency band
        signal_filtered = mne.filter.filter_data(
            signal, sfreq=raw.info["sfreq"], l_freq=1.0, h_freq=10.0, verbose=False
        )

        # Peak detection on absolute amplitude
        abs_signal = np.abs(signal_filtered)
        min_distance = int(self.min_duration_ms * raw.info["sfreq"] / 1000)
        min_distance = max(min_distance, 1)

        # Convert threshold from uV to V (MNE uses Volts internally)
        threshold_v = self.threshold_uv * 1e-6

        peaks, _ = find_peaks(abs_signal, height=threshold_v, distance=min_distance)

        if len(peaks) == 0:
            return np.empty((0, 3), dtype=int)

        events = np.column_stack(
            [
                peaks,
                np.zeros(len(peaks), dtype=int),
                np.full(len(peaks), self.blink_event_code, dtype=int),
            ]
        )
        return events

    def _get_eog_signal(self, raw: mne.io.Raw) -> np.ndarray:
        """Extract a single EOG-like signal for blink detection.

        Priority:
            1. Explicit eog_channels from constructor
            2. Channels marked as EOG in raw.info
            3. Frontal EEG channels (Fp1, Fp2) as fallback
        """
        if self.eog_channels:
            available = [ch for ch in self.eog_channels if ch in raw.ch_names]
            if available:
                data = raw.get_data(picks=available)
                return data.mean(axis=0)

        # Try auto-detecting EOG channels
        eog_picks = mne.pick_types(raw.info, eog=True)
        if len(eog_picks) > 0:
            data = raw.get_data(picks=eog_picks)
            # Use vertical EOG if multiple; otherwise average
            return data.mean(axis=0)

        # Fallback: frontal EEG channels (case-insensitive matching)
        ch_names_upper = {ch.upper(): ch for ch in raw.ch_names}
        frontal = [
            ch_names_upper[name] for name in ["FP1", "FP2"] if name in ch_names_upper
        ]
        if frontal:
            data = raw.get_data(picks=frontal)
            return data.mean(axis=0)

        raise ValueError(
            "No EOG or frontal EEG channels found for blink detection. "
            f"Available channels: {raw.ch_names[:10]}..."
        )

    @classmethod
    def from_config(cls, config: HEDBERTConfig) -> BlinkDetector:
        """Create from an HEDBERTConfig."""
        return cls(sfreq=config.sfreq)

    def __repr__(self) -> str:
        return (
            f"BlinkDetector(threshold_uv={self.threshold_uv}, "
            f"min_duration_ms={self.min_duration_ms})"
        )
