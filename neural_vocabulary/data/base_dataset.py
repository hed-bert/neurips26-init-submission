"""Abstract base class for BIDS EEG datasets with HDF5 caching."""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import mne
import numpy as np
import torch
from torch.utils.data import Dataset

from neural_vocabulary.configs import HEDBERTConfig, load_config
from neural_vocabulary.data.blink_detector import BlinkDetector
from neural_vocabulary.data.event_epocher import EventEpocher
from neural_vocabulary.data.transforms import MinimalPreprocessing

if TYPE_CHECKING:
    from neural_vocabulary.models.channel_harmonization import ChannelHarmonization

logger = logging.getLogger(__name__)


class BaseEEGDataset(Dataset, abc.ABC):
    """Abstract base for BIDS EEG datasets with preprocessing and HDF5 caching.

    Subclasses implement dataset-specific loading: raw file paths,
    event parsing, EOG channels, and channel harmonization.

    The pipeline per subject/session:
        1. Load raw EEG
        2. Parse events from BIDS events.tsv
        3. Apply MinimalPreprocessing (resample, filter, re-reference)
        4. Optionally detect blinks and merge with stimulus events
        5. Run EventEpocher for variable-length epochs
        6. Apply channel harmonization
        7. Save to HDF5 cache
    """

    def __init__(
        self,
        root_dir: str | Path,
        cache_dir: str | Path | None = None,
        subjects: list[str] | None = None,
        config: HEDBERTConfig | None = None,
        config_path: str | Path | None = None,
        force_preprocess: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)

        if config is None and config_path is not None:
            config = load_config(config_path)
        if config is None:
            config = HEDBERTConfig()
        self.config = config

        if cache_dir is None:
            cache_dir = self.root_dir / "derivatives" / "hed_bert"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessing = MinimalPreprocessing.from_config(config)
        self.epocher = EventEpocher.from_config(config)
        self.blink_detector = BlinkDetector.from_config(config)

        self._requested_subjects = subjects
        self._epoch_index: list[dict] = []
        self._config_hash = self._compute_config_hash()

        self._preprocess_and_cache(force=force_preprocess)
        self._build_index()

    def _compute_config_hash(self) -> str:
        """Hash of preprocessing-relevant config values for cache invalidation."""
        params = {
            "sfreq": self.config.sfreq,
            "l_freq": self.config.l_freq,
            "target_channels": self.config.target_channels,
            "pre_event_ms": self.config.pre_event_ms,
            "max_post_ms": self.config.max_post_ms,
            "min_epoch_ms": self.config.min_epoch_ms,
            "include_physiological": self.config.include_physiological,
        }
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]

    def _preprocess_and_cache(self, force: bool = False) -> None:
        """Preprocess and cache epochs for all subjects/sessions.

        Uses parallel workers (ProcessPoolExecutor) for first-time
        caching. Subsequent runs hit the HDF5 cache and skip instantly.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        subjects = self._requested_subjects or self._get_subject_ids()

        # Collect work items: (subject_id, session, h5_path) needing processing
        to_process: list[tuple[str, str | None, Path]] = []
        for subject_id in subjects:
            sessions = self._get_sessions(subject_id)
            for session in sessions:
                h5_path = self._get_cache_path(subject_id, session)
                if not force and h5_path.exists():
                    try:
                        with h5py.File(h5_path, "r") as f:
                            cached_hash = f.attrs.get("config_hash", "")
                            if cached_hash == self._config_hash:
                                continue
                    except (OSError, KeyError):
                        pass  # re-process corrupted cache
                to_process.append((subject_id, session, h5_path))

        if not to_process:
            return

        n_workers = min(len(to_process), max(1, (os.cpu_count() or 1) // 2))
        logger.info(
            "Caching %d recordings with %d workers...",
            len(to_process),
            n_workers,
        )

        if n_workers <= 1:
            # Single-threaded fallback
            for subject_id, session, h5_path in to_process:
                self._safe_process(subject_id, session, h5_path)
            return

        # Parallel processing
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(self._safe_process, sid, sess, path): (sid, sess)
                for sid, sess, path in to_process
            }
            for done, future in enumerate(as_completed(futures), 1):
                sid, sess = futures[future]
                exc = future.exception()
                if exc:
                    logger.error("Failed %s/%s: %s", sid, sess, exc)
                if done % 100 == 0:
                    logger.info("Cached %d/%d recordings", done, len(to_process))

        logger.info("Caching complete: %d recordings", len(to_process))

    def _safe_process(
        self,
        subject_id: str,
        session: str | None,
        h5_path: Path,
    ) -> None:
        """Process one recording, catching and logging errors."""
        try:
            self._process_and_save(subject_id, session, h5_path)
        except Exception as e:
            logger.error(
                "Failed to process %s/%s: %s. Skipping.", subject_id, session, e
            )

    def _process_and_save(
        self,
        subject_id: str,
        session: str | None,
        h5_path: Path,
    ) -> None:
        """Load, preprocess, epoch, harmonize, and save one subject/session."""
        # Load raw
        raw = self._load_raw(subject_id, session)

        # Parse events before preprocessing (onsets in original sample rate)
        events, event_id_map = self._parse_events(raw, subject_id, session)

        # Preprocess (resample, filter, re-reference)
        raw = self.preprocessing(raw)

        # Recalculate event sample indices after resampling
        # Events were set as annotations before resampling; now extract them
        if len(raw.annotations) > 0:
            events_resampled, event_id_map = mne.events_from_annotations(
                raw, verbose=False
            )
            # Use resampled events if annotations were set
            if len(events_resampled) > 0:
                events = events_resampled

        # Pass EOG channels to blink detector if subclass provides them
        eog_channels = self._get_eog_channels()
        if eog_channels:
            self.blink_detector.eog_channels = eog_channels
        blink_events = self.blink_detector.detect(raw)
        events = self.epocher.merge_events(events, blink_events)

        # Resolve HED tags if subclass provides them
        hed_tags = self._get_hed_tags(subject_id, session, event_id_map)

        # Create variable-length epochs
        epoch_list = self.epocher.epoch(raw, events, hed_tags=hed_tags)

        if not epoch_list:
            logger.warning("No epochs for %s/%s", subject_id, session)
            return

        # Apply channel harmonization
        harmonizer = self._get_harmonizer()

        # Save to HDF5
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as f:
            f.attrs["config_hash"] = self._config_hash
            f.attrs["subject_id"] = subject_id
            f.attrs["session"] = session or ""
            f.attrs["n_epochs"] = len(epoch_list)
            f.attrs["sfreq"] = self.config.sfreq

            for i, ep in enumerate(epoch_list):
                eeg_tensor = torch.from_numpy(ep.eeg)
                harmonized = harmonizer(eeg_tensor).numpy()

                grp = f.create_group(f"epoch_{i:05d}")
                grp.create_dataset(
                    "eeg", data=harmonized, compression="gzip", compression_opts=4
                )
                grp.attrs["event_id"] = ep.event_id
                grp.attrs["onset_sample"] = ep.onset_sample
                grp.attrs["duration_samples"] = ep.duration_samples
                grp.attrs["pre_event_samples"] = ep.pre_event_samples
                grp.attrs["length"] = ep.eeg.shape[-1]
                if ep.hed_tag is not None:
                    grp.attrs["hed_tag"] = ep.hed_tag

        logger.info(
            "Cached %d epochs for %s/%s -> %s",
            len(epoch_list),
            subject_id,
            session or "default",
            h5_path,
        )

    def _build_index(self) -> None:
        """Scan cached HDF5 files and build the epoch index."""
        self._epoch_index = []
        subjects = self._requested_subjects or self._get_subject_ids()

        for subject_id in subjects:
            sessions = self._get_sessions(subject_id)

            for session in sessions:
                h5_path = self._get_cache_path(subject_id, session)
                if not h5_path.exists():
                    continue

                try:
                    with h5py.File(h5_path, "r") as f:
                        n_epochs = f.attrs.get("n_epochs", 0)
                        for i in range(n_epochs):
                            grp_name = f"epoch_{i:05d}"
                            if grp_name not in f:
                                continue
                            grp = f[grp_name]
                            self._epoch_index.append(
                                {
                                    "h5_path": str(h5_path),
                                    "epoch_key": grp_name,
                                    "subject_id": subject_id,
                                    "session": session,
                                    "event_id": int(grp.attrs["event_id"]),
                                    "length": int(grp.attrs["length"]),
                                    "pre_event_samples": int(
                                        grp.attrs["pre_event_samples"]
                                    ),
                                }
                            )
                except (OSError, KeyError) as e:
                    logger.warning("Error reading cache %s: %s", h5_path, e)

        logger.info("Built index with %d epochs total", len(self._epoch_index))

    def _get_cache_path(self, subject_id: str, session: str | None) -> Path:
        """Return HDF5 cache file path for a subject/session."""
        if session:
            return self.cache_dir / f"{subject_id}_{session}_{self._config_hash}.h5"
        return self.cache_dir / f"{subject_id}_{self._config_hash}.h5"

    def __getitem__(self, idx: int) -> dict:
        """Load a single epoch from HDF5 cache."""
        meta = self._epoch_index[idx]
        with h5py.File(meta["h5_path"], "r") as f:
            grp = f[meta["epoch_key"]]
            eeg = grp["eeg"][:]
            hed_tag = grp.attrs.get("hed_tag", None)

        return {
            "eeg": eeg.astype(np.float32),
            "event_id": meta["event_id"],
            "length": meta["length"],
            "pre_event_samples": meta["pre_event_samples"],
            "subject_id": meta["subject_id"],
            "session": meta["session"],
            "hed_tag": str(hed_tag) if hed_tag is not None else None,
        }

    def __len__(self) -> int:
        return len(self._epoch_index)

    def get_lengths(self) -> list[int]:
        """Return list of epoch lengths for BucketBatchSampler."""
        return [meta["length"] for meta in self._epoch_index]

    # -- Abstract methods for subclasses --

    @abc.abstractmethod
    def _get_subject_ids(self) -> list[str]:
        """Return list of subject IDs available in the dataset."""
        ...

    @abc.abstractmethod
    def _get_sessions(self, subject_id: str) -> list[str]:
        """Return list of sessions/tasks for a subject."""
        ...

    @abc.abstractmethod
    def _load_raw(self, subject_id: str, session: str | None) -> mne.io.Raw:
        """Load raw EEG for a subject/session."""
        ...

    @abc.abstractmethod
    def _parse_events(
        self,
        raw: mne.io.Raw,
        subject_id: str,
        session: str | None,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Parse events from BIDS events.tsv.

        Should set events as raw.annotations for correct resampling.

        Returns:
            events: MNE events array (n_events, 3)
            event_id_map: mapping from event description to event code
        """
        ...

    @abc.abstractmethod
    def _get_eog_channels(self) -> list[str] | None:
        """Return EOG channel names, or None for frontal-EEG fallback."""
        ...

    @abc.abstractmethod
    def _get_harmonizer(self) -> ChannelHarmonization:
        """Return the ChannelHarmonization instance for this dataset."""
        ...

    def _get_hed_tags(
        self,
        subject_id: str,
        session: str | None,
        event_id_map: dict[str, int],
    ) -> dict[int, str] | None:
        """Map event IDs to HED tag strings.

        Override in subclasses that have HED annotations. The default
        returns None (no HED tags).

        Args:
            subject_id: Subject identifier.
            session: Session/task identifier.
            event_id_map: Mapping from annotation description string to
                integer event code (from mne.events_from_annotations).

        Returns:
            Mapping from integer event code to HED tag string,
            or None if no HED tags are available.
        """
        return None
