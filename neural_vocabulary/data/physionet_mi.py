"""PhysioNet Motor Movement/Imagery dataset loader (ds004362).

109 subjects, 64 EEG channels (10-10 system), 14 runs per subject.
Runs 1-2: baseline (eyes open/closed).
Runs 3-14: motor execution and imagery tasks.
Original sampling: 160 Hz; BDF conversion resamples to 100 Hz.
HED annotations via task-motion_events.json sidecar (HED 8.3.0).
"""

from __future__ import annotations

import json
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

# Event code string -> integer mapping.
# Baseline runs use BASExTy, task runs use TASKxTy.
# T0 = rest, T1 = left fist (tasks 1,2) or both fists (tasks 3,4),
# T2 = right fist (tasks 1,2) or both feet (tasks 3,4).
EVENT_CODE_MAP: dict[str, int] = {
    "BASE1T0": 1,
    "BASE2T0": 2,
    "TASK1T0": 10,
    "TASK1T1": 11,
    "TASK1T2": 12,
    "TASK2T0": 20,
    "TASK2T1": 21,
    "TASK2T2": 22,
    "TASK3T0": 30,
    "TASK3T1": 31,
    "TASK3T2": 32,
    "TASK4T0": 40,
    "TASK4T1": 41,
    "TASK4T2": 42,
}

# Channel names as they appear in the .set files (non-standard capitalization).
# Used to rename channels to standard_1020 capitalization when loading .set or .bdf files.
_PHYSIONET_TO_STANDARD: dict[str, str] = {
    "Fc5": "FC5",
    "Fc3": "FC3",
    "Fc1": "FC1",
    "Fcz": "FCz",
    "Fc2": "FC2",
    "Fc4": "FC4",
    "Fc6": "FC6",
    "Cp5": "CP5",
    "Cp3": "CP3",
    "Cp1": "CP1",
    "Cpz": "CPz",
    "Cp2": "CP2",
    "Cp4": "CP4",
    "Cp6": "CP6",
    "Af7": "AF7",
    "Af3": "AF3",
    "Afz": "AFz",
    "Af4": "AF4",
    "Af8": "AF8",
    "Ft7": "FT7",
    "Ft8": "FT8",
    "Tp7": "TP7",
    "Tp8": "TP8",
    "Po7": "PO7",
    "Po3": "PO3",
    "Poz": "POz",
    "Po4": "PO4",
    "Po8": "PO8",
}


class PhysioNetMIDataset(BaseEEGDataset):
    """PhysioNet Motor Movement/Imagery dataset loader.

    BIDS layout:
        {root}/sub-{id}/eeg/sub-{id}_task-motion_run-{N}_eeg.set (or .bdf)
        {root}/sub-{id}/eeg/sub-{id}_task-motion_run-{N}_events.tsv

    Runs:
        1-2: Baseline (eyes open / eyes closed)
        3,7,11: TASK1 - execute open/close left or right fist
        4,8,12: TASK2 - imagine opening/closing left or right fist
        5,9,13: TASK3 - execute open/close both fists or both feet
        6,10,14: TASK4 - imagine opening/closing both fists or both feet
    """

    TASK = "motion"
    ALL_RUNS = list(range(1, 15))

    def __init__(
        self,
        root_dir: str | Path,
        runs: list[int] | None = None,
        **kwargs,
    ) -> None:
        self.runs = runs or self.ALL_RUNS
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
        """Return run identifiers that exist on disk for this subject.

        Despite being called 'sessions' in the base class interface,
        these map to BIDS runs in the PhysioNet MI layout.
        """
        sessions = []
        eeg_dir = self.root_dir / f"sub-{subject_id}" / "eeg"
        if not eeg_dir.is_dir():
            logger.warning("No eeg/ directory for subject %s", subject_id)
            return []
        for run_idx in self.runs:
            run_key = f"run-{run_idx}"
            # Check for BDF first, then .set; also require events.tsv
            bdf_file = eeg_dir / f"sub-{subject_id}_task-{self.TASK}_{run_key}_eeg.bdf"
            set_file = eeg_dir / f"sub-{subject_id}_task-{self.TASK}_{run_key}_eeg.set"
            events_file = (
                eeg_dir / f"sub-{subject_id}_task-{self.TASK}_{run_key}_events.tsv"
            )
            has_data = bdf_file.exists() or set_file.exists()
            if has_data and events_file.exists():
                sessions.append(run_key)
            elif has_data and not events_file.exists():
                logger.warning(
                    "EEG file exists but events.tsv missing for %s/%s",
                    subject_id,
                    run_key,
                )
        return sessions

    def _load_raw(self, subject_id: str, session: str | None) -> mne.io.Raw:
        """Load raw EEG data, preferring BDF over .set."""
        eeg_dir = self.root_dir / f"sub-{subject_id}" / "eeg"
        base = f"sub-{subject_id}_task-{self.TASK}_{session}_eeg"

        bdf_path = eeg_dir / f"{base}.bdf"
        set_path = eeg_dir / f"{base}.set"

        if bdf_path.exists():
            raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)
        elif set_path.exists():
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
        else:
            raise FileNotFoundError(
                f"No BDF or SET file found for {subject_id}/{session}"
            )

        # Standardize channel names (PhysioNet uses non-standard capitalization)
        rename_map = {
            old: new
            for old, new in _PHYSIONET_TO_STANDARD.items()
            if old in raw.ch_names
        }
        if rename_map:
            raw.rename_channels(rename_map)

        # Set all channels as EEG type
        raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

        # Set channel types from channels.tsv if available and informative
        channels_tsv = (
            eeg_dir / f"sub-{subject_id}_task-{self.TASK}_{session}_channels.tsv"
        )
        if channels_tsv.exists():
            self._set_channel_types_from_tsv(raw, channels_tsv)

        return raw

    @staticmethod
    def _set_channel_types_from_tsv(raw: mne.io.Raw, channels_tsv: Path) -> None:
        """Set channel types from BIDS channels.tsv if types are specified."""
        df = pd.read_csv(channels_tsv, sep="\t")
        type_mapping = {}
        for _, row in df.iterrows():
            name = row["name"]
            ch_type = row.get("type", "n/a")
            if ch_type == "n/a" or pd.isna(ch_type):
                continue
            # Apply name normalization
            std_name = _PHYSIONET_TO_STANDARD.get(name, name)
            if std_name in raw.ch_names:
                if ch_type.upper() == "EOG":
                    type_mapping[std_name] = "eog"
                elif ch_type.upper() == "EEG":
                    type_mapping[std_name] = "eeg"
        if type_mapping:
            raw.set_channel_types(type_mapping)

    def _parse_events(
        self,
        raw: mne.io.Raw,
        subject_id: str,
        session: str | None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Parse events from events.tsv.

        Maps string event codes (BASExTy, TASKxTy) to integer IDs.
        Sets events as annotations on the raw object for correct resampling.
        """
        events_path = (
            self.root_dir
            / f"sub-{subject_id}"
            / "eeg"
            / f"sub-{subject_id}_task-{self.TASK}_{session}_events.tsv"
        )
        if not events_path.exists():
            raise FileNotFoundError(
                f"Events file missing for {subject_id}/{session}: {events_path}"
            )

        df = pd.read_csv(events_path, sep="\t")

        # Filter to rows with known event codes; warn on unknowns
        known_mask = []
        event_codes_list = []
        for v in df["value"].values:
            code = EVENT_CODE_MAP.get(v)
            if code is None:
                logger.warning(
                    "Unknown event code '%s' in %s/%s, skipping",
                    v,
                    subject_id,
                    session,
                )
                known_mask.append(False)
                event_codes_list.append(0)
            else:
                known_mask.append(True)
                event_codes_list.append(code)

        known_mask = np.array(known_mask)
        event_codes = np.array(event_codes_list, dtype=int)

        # Keep only rows with known event codes
        df_filtered = df[known_mask]
        event_codes = event_codes[known_mask]

        onsets_sec = df_filtered["onset"].values.astype(float)
        durations = (
            df_filtered["duration"]
            .apply(lambda x: 0.0 if x == "n/a" else float(x))
            .values
        )

        descriptions = [str(code) for code in event_codes]

        # Set as annotations (survives resampling)
        annotations = mne.Annotations(
            onset=onsets_sec,
            duration=durations,
            description=descriptions,
        )
        raw.set_annotations(annotations)

        # Build events array in original sample space
        if "sample" in df_filtered.columns:
            samples = df_filtered["sample"].values.astype(int)
        else:
            samples = (onsets_sec * raw.info["sfreq"]).astype(int)

        events = np.column_stack(
            [samples, np.zeros(len(df_filtered), dtype=int), event_codes]
        )

        event_id_map = {str(code): code for code in np.unique(event_codes)}
        return events, event_id_map

    def _get_eog_channels(self) -> list[str] | None:
        """No dedicated EOG channels; blink detection falls back to Fp1/Fp2."""
        return None

    def _get_harmonizer(self) -> ChannelHarmonization:
        """64 -> 64 channel harmonization (different montage subsets)."""
        return ChannelHarmonization.for_physionet_mi()

    def _get_hed_tags(
        self,
        subject_id: str,
        session: str | None,
        event_id_map: dict[str, int],
    ) -> dict[int, str] | None:
        """Map event IDs to HED strings from task-motion_events.json sidecar.

        The sidecar maps string labels (TASK1T1, BASE1T0, etc.) to HED
        annotations. The event_id_map maps string codes ("11") to integer
        codes (11). We reverse-map through EVENT_CODE_MAP to connect them.
        """
        sidecar_path = self.root_dir / "task-motion_events.json"
        if not sidecar_path.exists():
            logger.warning(
                "HED sidecar not found at %s; MI events will have no HED tags",
                sidecar_path,
            )
            return None

        try:
            with open(sidecar_path) as f:
                sidecar = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to parse HED sidecar %s: %s", sidecar_path, e)
            return None

        sidecar_hed = sidecar.get("value", {}).get("HED", {})
        if not sidecar_hed:
            return None

        # Build reverse map: integer code -> original string label
        code_to_label = {v: k for k, v in EVENT_CODE_MAP.items()}

        hed_tags: dict[int, str] = {}
        unmatched = []
        for _desc_str, code in event_id_map.items():
            label = code_to_label.get(code)
            if label and label in sidecar_hed:
                hed_tags[code] = sidecar_hed[label]
            else:
                unmatched.append(code)

        if unmatched:
            logger.debug(
                "sub-%s/%s: %d/%d event codes have no HED mapping: %s",
                subject_id,
                session,
                len(unmatched),
                len(event_id_map),
                unmatched[:5],
            )

        return hed_tags if hed_tags else None
