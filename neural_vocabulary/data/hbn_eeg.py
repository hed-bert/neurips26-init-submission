"""HBN-EEG dataset loader (ds005505-ds005516).

Healthy Brain Network EEG: 3,464 subjects across 12 releases,
129 channels (HydroCel Geodesic E1-E128 + Cz), BDF format,
100Hz, BIDS with HED annotations in task-level sidecar JSONs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import mne
import numpy as np
import pandas as pd

from neural_vocabulary.data.base_dataset import BaseEEGDataset
from neural_vocabulary.models.channel_harmonization import ChannelHarmonization

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Event values that are recording boundaries, not neural events
_EXCLUDED_EVENT_VALUES = frozenset({"break cnt", "boundary"})

# HydroCel E-channels closest to Fp1 (E25) and Fp2 (E8)
_FRONTAL_EOG_CHANNELS = ["E25", "E8"]


class HBNEEGDataset(BaseEEGDataset):
    """HBN-EEG dataset loader (NEMAR ds005505-ds005516).

    Multi-release BIDS layout:
        {root}/R{n}_L100_bdf/
            participants.tsv
            task-{task}_events.json       (HED sidecar)
            sub-{id}/eeg/
                sub-{id}_task-{task}_eeg.bdf
                sub-{id}_task-{task}_events.tsv
                sub-{id}_task-{task}_run-{n}_eeg.bdf    (multi-run)
                sub-{id}_task-{task}_run-{n}_events.tsv  (multi-run)

    Tasks: RestingState, contrastChangeDetection (3 runs),
           surroundSupp (2 runs), seqLearning6target, seqLearning8target,
           symbolSearch, DespicableMe, DiaryOfAWimpyKid, FunwithFractals,
           ThePresent
    """

    ALL_TASKS = [
        "RestingState",
        "contrastChangeDetection",
        "surroundSupp",
        "seqLearning6target",
        "seqLearning8target",
        "symbolSearch",
        "DespicableMe",
        "DiaryOfAWimpyKid",
        "FunwithFractals",
        "ThePresent",
    ]

    def __init__(
        self,
        root_dir: str | Path,
        releases: list[str] | None = None,
        tasks: list[str] | None = None,
        max_subjects: int | None = None,
        **kwargs,
    ) -> None:
        self.tasks = tasks or self.ALL_TASKS
        self.max_subjects = max_subjects

        # Scan releases and build subject-to-release mapping before
        # calling super().__init__() (which calls _get_subject_ids)
        from pathlib import Path as _Path

        root = _Path(root_dir)
        self._release_dirs: dict[str, _Path] = {}
        self._subject_release: dict[str, _Path] = {}
        self._hed_sidecars: dict[str, dict[str, str]] = {}
        self._participants: pd.DataFrame = pd.DataFrame()

        if not root.is_dir():
            raise FileNotFoundError(f"HBN-EEG root directory does not exist: {root}")

        release_pattern = re.compile(r"^R(\d+)_L100_bdf$")
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            m = release_pattern.match(d.name)
            if m is None:
                continue
            release_num = f"R{m.group(1)}"
            if releases is not None and release_num not in releases:
                continue
            self._release_dirs[release_num] = d
            self._scan_release(d)

        if not self._release_dirs:
            available = [
                d.name
                for d in root.iterdir()
                if d.is_dir() and release_pattern.match(d.name)
            ]
            raise FileNotFoundError(
                f"No matching HBN-EEG releases in {root}. "
                f"Requested: {releases}. Available: {available}"
            )

        if not self._subject_release:
            logger.warning("No subjects found in selected releases")

        # Apply max_subjects limit
        if self.max_subjects is not None:
            all_subjects = sorted(self._subject_release.keys())
            for sub_id in all_subjects[self.max_subjects :]:
                del self._subject_release[sub_id]

        super().__init__(root_dir=root_dir, **kwargs)

    def _scan_release(self, release_dir: Path) -> None:
        """Scan a release directory for subjects, participants, and HED sidecars."""
        # Register subjects
        for d in sorted(release_dir.iterdir()):
            if d.is_dir() and d.name.startswith("sub-"):
                sub_id = d.name.replace("sub-", "")
                if sub_id not in self._subject_release:
                    self._subject_release[sub_id] = release_dir

        # Load participants.tsv
        participants_path = release_dir / "participants.tsv"
        if participants_path.exists():
            try:
                df = pd.read_csv(participants_path, sep="\t")
            except (pd.errors.ParserError, pd.errors.EmptyDataError, OSError) as e:
                logger.error("Failed to parse %s: %s", participants_path, e)
                return
            if "participant_id" not in df.columns:
                logger.error(
                    "%s missing 'participant_id' column. Found: %s",
                    participants_path,
                    list(df.columns),
                )
                return
            df = df.set_index("participant_id", drop=False)
            if self._participants.empty:
                self._participants = df
            else:
                self._participants = pd.concat(
                    [self._participants, df[~df.index.isin(self._participants.index)]]
                )

        # Load HED sidecars per release (may differ between releases)
        self._load_hed_sidecars(release_dir)

    def _load_hed_sidecars(self, release_dir: Path) -> None:
        """Load task-level events.json sidecars containing HED annotations.

        Sidecars from later releases update entries for tasks not yet seen.
        """
        for sidecar_path in release_dir.glob("task-*_events.json"):
            task_name = sidecar_path.stem.replace("task-", "").replace("_events", "")
            if task_name in self._hed_sidecars:
                continue
            try:
                with open(sidecar_path) as f:
                    sidecar = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error(
                    "Failed to parse HED sidecar %s: %s. "
                    "HED tags unavailable for task '%s'.",
                    sidecar_path,
                    e,
                    task_name,
                )
                continue

            value_hed = sidecar.get("value", {}).get("HED", {})
            if value_hed:
                self._hed_sidecars[task_name] = value_hed

    def _get_subject_ids(self) -> list[str]:
        """Return combined subject list from all selected releases."""
        return sorted(self._subject_release.keys())

    def _get_sessions(self, subject_id: str) -> list[str]:
        """Scan for available task BDF files, respecting task filter.

        Returns session strings like 'RestingState' or
        'contrastChangeDetection_run-1' for multi-run tasks.
        """
        release_dir = self._subject_release.get(subject_id)
        if release_dir is None:
            return []

        eeg_dir = release_dir / f"sub-{subject_id}" / "eeg"
        if not eeg_dir.is_dir():
            return []

        sessions = []
        bdf_pattern = re.compile(
            rf"sub-{re.escape(subject_id)}_task-(.+?)(?:_run-(\d+))?_eeg\.bdf$"
        )

        for f in sorted(eeg_dir.iterdir()):
            m = bdf_pattern.match(f.name)
            if m is None:
                continue
            task = m.group(1)
            run = m.group(2)

            if task not in self.tasks:
                continue

            session = f"{task}_run-{run}" if run else task
            sessions.append(session)

        return sessions

    def _load_raw(self, subject_id: str, session: str | None) -> mne.io.Raw:
        """Load BDF raw data and set GSN-HydroCel-129 montage."""
        bdf_path = self._session_to_bdf_path(subject_id, session)
        raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)

        # Pick only EEG channels (drop Status, etc.)
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        raw.pick(eeg_picks)

        # Set montage for 3D electrode positions
        montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        raw.set_montage(montage, on_missing="warn")

        return raw

    def _parse_events(
        self,
        raw: mne.io.Raw,
        subject_id: str,
        session: str | None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Parse events from events.tsv.

        Uses the 'value' column strings as annotation descriptions
        so that _get_hed_tags can map them to HED strings downstream.
        Filters out recording boundaries and numeric-only markers.
        """
        events_path = self._session_to_events_path(subject_id, session)
        df = pd.read_csv(events_path, sep="\t")

        # Ensure value column is string (pandas may infer numeric dtype)
        df["value"] = df["value"].astype(str)

        # Filter out boundary markers and numeric-only event values
        # (raw BDF trigger codes that duplicate the named value events)
        mask = ~df["value"].isin(_EXCLUDED_EVENT_VALUES)
        mask &= ~df["value"].str.match(r"^\d+$", na=False)
        df = df[mask].reset_index(drop=True)

        if df.empty:
            empty_events = np.empty((0, 3), dtype=int)
            return empty_events, {}

        onsets_sec = df["onset"].values.astype(float)
        durations = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).values
        descriptions = df["value"].values.astype(str)

        # Set as annotations (survives resampling)
        annotations = mne.Annotations(
            onset=onsets_sec,
            duration=durations,
            description=descriptions,
        )
        raw.set_annotations(annotations)

        # Build events array in original sample space
        samples = (onsets_sec * raw.info["sfreq"]).astype(int)
        # Assign sequential integer codes for the events array
        unique_values = sorted(set(descriptions))
        value_to_code = {v: i + 1 for i, v in enumerate(unique_values)}
        event_codes = np.array([value_to_code[d] for d in descriptions])

        events = np.column_stack([samples, np.zeros(len(df), dtype=int), event_codes])

        event_id_map = dict(value_to_code)
        return events, event_id_map

    def _get_eog_channels(self) -> list[str]:
        """Return HydroCel E-channels closest to Fp1 (E25) and Fp2 (E8)."""
        return _FRONTAL_EOG_CHANNELS

    def _get_harmonizer(self) -> ChannelHarmonization:
        """Harmonize 129 GSN-HydroCel channels to 64 standard 10-10 channels."""
        return ChannelHarmonization.for_hbn_eeg()

    def _get_hed_tags(
        self,
        subject_id: str,
        session: str | None,
        event_id_map: dict[str, int],
    ) -> dict[int, str] | None:
        """Map event IDs to HED tag strings from task-level sidecar."""
        task = self._session_to_task(session)
        sidecar_hed = self._hed_sidecars.get(task, {})
        if not sidecar_hed:
            logger.debug(
                "No HED sidecar for task '%s' (sub-%s/%s)",
                task,
                subject_id,
                session,
            )
            return None

        hed_tags: dict[int, str] = {}
        unmatched = []
        for description, code in event_id_map.items():
            if description in sidecar_hed:
                hed_tags[code] = sidecar_hed[description]
            else:
                unmatched.append(description)

        if unmatched:
            logger.debug(
                "sub-%s/%s: %d/%d event types have no HED mapping: %s",
                subject_id,
                session,
                len(unmatched),
                len(event_id_map),
                unmatched[:5],
            )

        if not hed_tags:
            logger.warning(
                "sub-%s/%s: zero event types matched HED sidecar. "
                "Sidecar keys: %s; event types: %s",
                subject_id,
                session,
                list(sidecar_hed.keys())[:5],
                list(event_id_map.keys())[:5],
            )

        return hed_tags if hed_tags else None

    # -- Path helpers --

    def _session_to_bdf_path(self, subject_id: str, session: str | None) -> Path:
        """Convert session string to BDF file path."""
        release_dir = self._subject_release[subject_id]
        task, run = self._parse_session(session)
        if run:
            filename = f"sub-{subject_id}_task-{task}_run-{run}_eeg.bdf"
        else:
            filename = f"sub-{subject_id}_task-{task}_eeg.bdf"
        return release_dir / f"sub-{subject_id}" / "eeg" / filename

    def _session_to_events_path(self, subject_id: str, session: str | None) -> Path:
        """Convert session string to events.tsv file path."""
        release_dir = self._subject_release[subject_id]
        task, run = self._parse_session(session)
        if run:
            filename = f"sub-{subject_id}_task-{task}_run-{run}_events.tsv"
        else:
            filename = f"sub-{subject_id}_task-{task}_events.tsv"
        return release_dir / f"sub-{subject_id}" / "eeg" / filename

    @staticmethod
    def _parse_session(session: str | None) -> tuple[str, str | None]:
        """Split session string into (task, run) components."""
        if session is None:
            raise ValueError("HBN sessions must not be None")
        m = re.match(r"^(.+?)_run-(\d+)$", session)
        if m:
            return m.group(1), m.group(2)
        return session, None

    @staticmethod
    def _session_to_task(session: str | None) -> str:
        """Extract task name from session string (strip run suffix)."""
        if session is None:
            raise ValueError("HBN sessions must not be None")
        m = re.match(r"^(.+?)_run-\d+$", session)
        return m.group(1) if m else session
