"""Event-based variable-length epoching for EEG data.

Core innovation: epochs are anchored to detected neural and behavioral
events rather than fixed-size temporal patches. Each epoch spans from
-pre_event_ms before the current event to the onset of the next event
(or +max_post_ms if no next event within that window).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import mne
import numpy as np

if TYPE_CHECKING:
    from neural_vocabulary.configs import HEDBERTConfig


@dataclasses.dataclass
class EpochData:
    """A single variable-length epoch anchored to a behavioral event."""

    eeg: np.ndarray  # (n_channels, n_times), float32
    event_id: int
    hed_tag: str | None
    onset_sample: int  # absolute sample index in original recording
    duration_samples: int
    pre_event_samples: int


class EventEpocher:
    """Create variable-length epochs between consecutive events.

    Each epoch spans from -pre_event_ms before the current event to the
    onset of the next event (or +max_post_ms if no next event is within
    that window). Events include stimulus/response markers and optionally
    detected physiological events (blinks, microsaccades).
    """

    def __init__(
        self,
        pre_event_ms: float = 100.0,
        max_post_ms: float = 1500.0,
        min_epoch_ms: float = 200.0,
        include_physiological: bool = True,
        sfreq: float = 100.0,
    ) -> None:
        self.pre_event_ms = pre_event_ms
        self.max_post_ms = max_post_ms
        self.min_epoch_ms = min_epoch_ms
        self.include_physiological = include_physiological
        self.sfreq = sfreq

        self.pre_samples = int(pre_event_ms * sfreq / 1000)
        self.max_post_samples = int(max_post_ms * sfreq / 1000)
        self.min_samples = int(min_epoch_ms * sfreq / 1000)

    def epoch(
        self,
        raw: mne.io.Raw,
        events: np.ndarray,
        hed_tags: dict[int, str] | None = None,
    ) -> list[EpochData]:
        """Create variable-length epochs from raw data and events.

        Args:
            raw: MNE Raw object (preprocessed, EEG channels only extracted).
            events: MNE events array, shape (n_events, 3).
                Column 0: sample index, Column 1: unused, Column 2: event code.
            hed_tags: optional mapping from event_code to HED tag string.

        Returns:
            List of EpochData, one per valid event.
        """
        if len(events) == 0:
            return []

        sorted_indices = np.argsort(events[:, 0])
        sorted_events = events[sorted_indices]

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        n_times = raw.n_times
        epochs = []

        for i in range(len(sorted_events)):
            onset = int(sorted_events[i, 0])
            event_code = int(sorted_events[i, 2])

            # Determine epoch end: next event onset or max window
            if i + 1 < len(sorted_events):
                next_onset = int(sorted_events[i + 1, 0])
                end = min(next_onset, onset + self.max_post_samples)
            else:
                end = onset + self.max_post_samples

            start = onset - self.pre_samples
            duration = end - start

            # Skip too-short epochs
            if duration < self.min_samples:
                continue
            # Skip out-of-bounds
            if start < 0 or end > n_times:
                continue

            data = raw.get_data(picks=eeg_picks, start=start, stop=end)

            epochs.append(
                EpochData(
                    eeg=data.astype(np.float32),
                    event_id=event_code,
                    hed_tag=hed_tags.get(event_code) if hed_tags else None,
                    onset_sample=onset,
                    duration_samples=duration,
                    pre_event_samples=self.pre_samples,
                )
            )

        return epochs

    def merge_events(
        self,
        stimulus_events: np.ndarray,
        physiological_events: np.ndarray | None = None,
    ) -> np.ndarray:
        """Merge stimulus/response events with physiological events.

        Args:
            stimulus_events: MNE events array from paradigm markers.
            physiological_events: MNE events array from blink/saccade
                detection. Ignored if include_physiological is False.

        Returns:
            Combined events array sorted by sample index.
        """
        if (
            physiological_events is None
            or not self.include_physiological
            or len(physiological_events) == 0
        ):
            return stimulus_events

        merged = np.vstack([stimulus_events, physiological_events])
        return merged[np.argsort(merged[:, 0])]

    @classmethod
    def from_config(cls, config: HEDBERTConfig) -> EventEpocher:
        """Create from an HEDBERTConfig."""
        return cls(
            pre_event_ms=config.pre_event_ms,
            max_post_ms=config.max_post_ms,
            min_epoch_ms=config.min_epoch_ms,
            include_physiological=config.include_physiological,
            sfreq=config.sfreq,
        )

    def __repr__(self) -> str:
        return (
            f"EventEpocher(pre={self.pre_event_ms}ms, "
            f"max_post={self.max_post_ms}ms, "
            f"min={self.min_epoch_ms}ms)"
        )
