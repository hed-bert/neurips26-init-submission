"""ERP-CORE dataset loader (NEMAR nm000132).

40 subjects, 30 EEG + 3 EOG channels, EEGLAB format,
6 ERP paradigms (tasks), 1024Hz original sampling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd

from neural_vocabulary.data.base_dataset import BaseEEGDataset
from neural_vocabulary.models.channel_harmonization import ChannelHarmonization

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ERPCoreDataset(BaseEEGDataset):
    """ERP-CORE dataset loader (NEMAR nm000132).

    BIDS layout (flat, no sessions):
        {root}/sub-{id}/eeg/sub-{id}_task-{task}_eeg.set
        {root}/sub-{id}/eeg/sub-{id}_task-{task}_events.tsv

    Tasks: flankers, MMN, N170, N2pc, N400, P3
    """

    TASKS = ["flankers", "MMN", "N170", "N2pc", "N400", "P3"]
    EOG_CHANNELS = ["HEOG_left", "HEOG_right", "VEOG_lower"]

    def __init__(
        self,
        root_dir: str | Path,
        paradigms: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.paradigms = paradigms or self.TASKS
        super().__init__(root_dir=root_dir, **kwargs)

    def _get_subject_ids(self) -> list[str]:
        """Scan for subject directories."""
        subjects = []
        for d in sorted(self.root_dir.iterdir()):
            if d.is_dir() and d.name.startswith("sub-"):
                sub_id = d.name.replace("sub-", "")
                subjects.append(sub_id)
        return subjects

    def _get_sessions(self, subject_id: str) -> list[str]:
        """Return tasks that exist on disk for this subject.

        Despite being called 'sessions' in the base class interface,
        these map to BIDS tasks in the flat ERP-CORE layout.
        """
        sessions = []
        eeg_dir = self.root_dir / f"sub-{subject_id}" / "eeg"
        for task in self.paradigms:
            set_file = eeg_dir / f"sub-{subject_id}_task-{task}_eeg.set"
            if set_file.exists():
                sessions.append(task)
        return sessions

    def _load_raw(self, subject_id: str, session: str | None) -> mne.io.Raw:
        """Load EEGLAB raw data."""
        set_path = (
            self.root_dir
            / f"sub-{subject_id}"
            / "eeg"
            / f"sub-{subject_id}_task-{session}_eeg.set"
        )
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

        # Set channel types from channels.tsv if available
        channels_tsv = set_path.with_name(
            f"sub-{subject_id}_task-{session}_channels.tsv"
        )
        if channels_tsv.exists():
            self._set_channel_types(raw, channels_tsv)

        return raw

    @staticmethod
    def _set_channel_types(raw: mne.io.Raw, channels_tsv: Path) -> None:
        """Set channel types from BIDS channels.tsv."""
        df = pd.read_csv(channels_tsv, sep="\t")
        type_mapping = {}
        for _, row in df.iterrows():
            name = row["name"]
            ch_type = row.get("type", "EEG")
            if name in raw.ch_names:
                if ch_type == "EOG":
                    type_mapping[name] = "eog"
                elif ch_type == "EEG":
                    type_mapping[name] = "eeg"
        if type_mapping:
            raw.set_channel_types(type_mapping)

    def _parse_events(
        self,
        raw: mne.io.Raw,
        subject_id: str,
        session: str | None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Parse events from events.tsv.

        ERP-CORE events.tsv has 'onset' in seconds and 'sample' column
        for exact sample indices at 1024Hz. We use onset in seconds to
        set annotations (survives resampling correctly).
        """
        events_path = (
            self.root_dir
            / f"sub-{subject_id}"
            / "eeg"
            / f"sub-{subject_id}_task-{session}_events.tsv"
        )
        df = pd.read_csv(events_path, sep="\t")

        onsets_sec = df["onset"].values.astype(float)
        durations = df["duration"].values.astype(float)
        event_codes = df["value"].values.astype(int)

        # Build descriptions
        descriptions = [str(code) for code in event_codes]

        # Set as annotations (survives resampling)
        annotations = mne.Annotations(
            onset=onsets_sec,
            duration=durations,
            description=descriptions,
        )
        raw.set_annotations(annotations)

        # Also return events in original sample space
        if "sample" in df.columns:
            samples = df["sample"].values.astype(int)
        else:
            samples = (onsets_sec * raw.info["sfreq"]).astype(int)

        events = np.column_stack([samples, np.zeros(len(df), dtype=int), event_codes])

        event_id_map = {str(code): code for code in np.unique(event_codes)}
        return events, event_id_map

    def _get_eog_channels(self) -> list[str]:
        """Return dedicated EOG channel names."""
        return self.EOG_CHANNELS

    def _get_harmonizer(self) -> ChannelHarmonization:
        """30 -> 64 channel harmonization."""
        return ChannelHarmonization.for_erp_core()
