"""THINGS-EEG (ds003825) dataset loader.

50 subjects, 63 EEG channels (Cz was online reference),
BrainVision format, RSVP paradigm, 1000Hz original sampling.
"""

from __future__ import annotations

import logging

import mne
import numpy as np
import pandas as pd

from neural_vocabulary.data.base_dataset import BaseEEGDataset
from neural_vocabulary.models.channel_harmonization import ChannelHarmonization

logger = logging.getLogger(__name__)


class THINGSEEGDataset(BaseEEGDataset):
    """THINGS-EEG dataset loader.

    BIDS layout:
        {root}/sub-{id}/eeg/sub-{id}_task-rsvp_eeg.vhdr
        {root}/sub-{id}/eeg/sub-{id}_task-rsvp_events.tsv
    """

    TASK = "rsvp"

    def _get_subject_ids(self) -> list[str]:
        """Scan for subject directories, respecting participants.tsv exclude."""
        excluded = set()
        participants_path = self.root_dir / "participants.tsv"
        if participants_path.exists():
            df = pd.read_csv(participants_path, sep="\t")
            if "exclude" in df.columns:
                excluded = set(
                    df.loc[df["exclude"] == 1, "participant_id"]
                    .str.replace("sub-", "")
                    .tolist()
                )

        subjects = []
        for d in sorted(self.root_dir.iterdir()):
            if d.is_dir() and d.name.startswith("sub-"):
                sub_id = d.name.replace("sub-", "")
                if sub_id not in excluded:
                    subjects.append(sub_id)

        return subjects

    def _get_sessions(self, subject_id: str) -> list[str]:
        """Single task in THINGS-EEG."""
        return [self.TASK]

    def _load_raw(self, subject_id: str, session: str | None) -> mne.io.Raw:
        """Load BrainVision raw data."""
        vhdr_path = (
            self.root_dir
            / f"sub-{subject_id}"
            / "eeg"
            / f"sub-{subject_id}_task-{self.TASK}_eeg.vhdr"
        )
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
        return raw

    def _parse_events(
        self,
        raw: mne.io.Raw,
        subject_id: str,
        session: str | None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Parse events from events.tsv.

        THINGS-EEG events.tsv has 'onset' in samples at original sfreq
        (1000Hz). We convert to seconds and set as annotations on the
        raw object so that MNE handles resampling correctly.
        """
        events_path = (
            self.root_dir
            / f"sub-{subject_id}"
            / "eeg"
            / f"sub-{subject_id}_task-{self.TASK}_events.tsv"
        )
        df = pd.read_csv(events_path, sep="\t")

        original_sfreq = raw.info["sfreq"]

        # Convert sample onsets to seconds
        onsets_sec = df["onset"].values / original_sfreq
        durations = df["duration"].values / original_sfreq

        # Use stimnumber as event code (unique stimulus identity)
        # Add 1 to avoid code 0 (MNE uses 0 as "no event")
        event_codes = df["stimnumber"].values.astype(int) + 1

        # Create descriptions for annotation-based event tracking
        descriptions = [str(code) for code in event_codes]

        # Set as annotations on the raw object (survives resampling)
        annotations = mne.Annotations(
            onset=onsets_sec,
            duration=durations,
            description=descriptions,
        )
        raw.set_annotations(annotations)

        # Also return events array in original sample space
        events = np.column_stack(
            [
                df["onset"].values.astype(int),
                np.zeros(len(df), dtype=int),
                event_codes,
            ]
        )

        event_id_map = {str(code): code for code in np.unique(event_codes)}
        return events, event_id_map

    def _get_eog_channels(self) -> list[str] | None:
        """No EOG channels; blink detection falls back to Fp1/Fp2."""
        return None

    def _get_harmonizer(self) -> ChannelHarmonization:
        """63 -> 64 channel harmonization (reconstruct Cz)."""
        return ChannelHarmonization.for_things_eeg()
