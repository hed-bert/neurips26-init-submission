"""Per-paradigm packed dataset for ERP-CORE frozen-encoder probes.

Wraps the Morlet TF h5 files at
``${HBN_DATA_DIR}/preprocessed_v10_erpcore_tf/`` (produced by
``extract_tf_features --task-filter erp-core``) into the E=8 packed-window
format BertSSL expects, with per-event-token labels supplied by a
``LabelRule`` so downstream probes can extract per-trial outputs.

Packing strategy:
    1. Per file, scan all epochs and apply ``rule.label_fn`` to each
       (event_value, event_type) pair. Keep epochs with a non-None label.
    2. Pack consecutive kept epochs into windows of size E=8. Trailing
       epochs that don't fill a full window are dropped (lossy by ≤7
       trials per file; negligible vs 80-160 per probe).
    3. Per __getitem__, return the packed window dict (compatible with
       BertSSL's forward) plus a per-position label tensor of shape (E,).

The encoder forward returns shape ``(B, n_total_tokens, d_model)`` where
positions 1..E hold per-event-token outputs. Pair those with the per-
position labels for per-trial classification.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    from neural_vocabulary.data.erpcore_label_rules import LabelRule

logger = logging.getLogger(__name__)

# Expected TF h5 schema attrs (must match extract_tf_features erp-core mode).
EXPECTED_VERSION: str = "v10_gate_a_morlet"
# Defaults match the  D.2.x ERP-CORE TF extraction (1-second epoch
# at 100 Hz source, decimate factor 10). For  dense extractions
# (e.g. n_time=50) pass ``expected_n_time`` and ``n_freqs`` /
# ``n_channels`` through the constructor.
EXPECTED_N_FREQS: int = 6
EXPECTED_N_CHANNELS: int = 64
EXPECTED_N_TIME: int = 10

#  dense defaults written by extract_tf_features --n-time 50 on a 1-second
# ERP-CORE epoch (100 Hz source, decimate_factor=2 → target_sfreq=50 Hz).
TARGET_SFREQ: float = 50.0
DECIMATE_FACTOR: int = 2
N_TIME_BINS: int = 50

# Positional stride between consecutive event tokens fed to BertSSL,
# measured in milliseconds. This is a FIXED positional convention —
#  1.0 s windows and   1.6 s windows both use 1000 ms — so a
# pretrained checkpoint's position embedding is interpretable across
# both source corpora. Changing this invalidates the encoder.
_EVENT_POSITION_STRIDE_MS: float = 1000.0


def _decode_h5_string(value: object, h5_path: Path, dataset_name: str) -> str:
    """Decode an h5py string-dataset entry into a Python str.

    Accepts ``str``, ``bytes`` (h5py 3.x default for ``string_dtype``), and
    ``np.bytes_`` (subclasses bytes — caught by the bytes branch). Raises
    on any other dtype rather than coercing via ``str()`` which would
    silently produce ``"b'face'"`` or ``"None"`` from unexpected inputs.
    """
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    raise TypeError(
        f"{h5_path.name}/{dataset_name}: unexpected element type "
        f"{type(value).__name__}; expected str or bytes."
    )


def _check_temporal_contract(
    f: h5py.File,
    h5_path: Path,
    expected_target_sfreq: float | None,
    expected_decimate_factor: int | None,
    expected_n_time_bins: int | None,
) -> None:
    """Validate temporal-contract attrs against expected values.

    Reads ``target_sfreq``, ``decimate_factor``, and ``n_time_bins`` from the
    h5 file root attrs (written by ``extract_tf_features`` since PR #204).
    If an attr is absent the file pre-dates PR #204; a warning is emitted and
    the check is skipped for that attr.  If present and the value does not
    match the expected value a ``ValueError`` is raised with a message that
    includes the file path, attr name, expected value, and actual value.

    Args:
        f: Open h5py.File handle (read mode).
        h5_path: Path to the file (used in error messages).
        expected_target_sfreq: Expected ``target_sfreq`` attr value, or
            ``None`` to skip that attr's check.
        expected_decimate_factor: Expected ``decimate_factor`` attr value,
            or ``None`` to skip.
        expected_n_time_bins: Expected ``n_time_bins`` attr value, or
            ``None`` to skip.
    """
    checks: list[tuple[str, float | int | None]] = [
        ("target_sfreq", expected_target_sfreq),
        ("decimate_factor", expected_decimate_factor),
        ("n_time_bins", expected_n_time_bins),
    ]
    for attr_name, expected in checks:
        if expected is None:
            continue
        if attr_name not in f.attrs:
            logger.warning(
                "%s: attr %r absent (legacy file written before PR #204); "
                "skipping temporal-contract check for this attr.",
                h5_path,
                attr_name,
            )
            continue
        actual = f.attrs[attr_name]
        # h5py scalar attrs may be numpy scalars; cast for clean comparison.
        if float(actual) != float(expected):
            raise ValueError(
                f"{h5_path}: temporal-contract mismatch on attr {attr_name!r}: "
                f"expected {expected!r}, got {actual!r}. "
                "Re-extract with matching --n-time / --target-sfreq flags or "
                "pass the correct expected values to the dataset constructor."
            )


class ErpcoreParadigmDataset(Dataset):
    """E=8 packed dataset filtered by a LabelRule, with per-position labels.

    Args:
        h5_files: Morlet TF h5 files (one per subject_task).
        rule: LabelRule for the target probe; trials returning None are
            dropped before packing.
        epochs_per_window: Must be 8 to match the BertSSL positional
            embedding (locked by the trained model architecture).
        require_paradigm: If set, raise unless the per-file ``task`` attr
            (or filename stem) matches this paradigm. Defensive check
            against accidentally feeding mixed-task files when the rule
            is paradigm-specific.
        expected_target_sfreq: Expected ``target_sfreq`` h5 attr written by
            ``extract_tf_features``. Opt-in: default ``None`` skips the check
            (preserves / caller compatibility — legacy h5 files lack
            this attr and trigger only a warning, not a hard error). Pass
            an explicit float to enforce.
        expected_decimate_factor: Expected ``decimate_factor`` h5 attr.
            Opt-in: default ``None``. Pass an explicit int to enforce.
        expected_n_time_bins: Expected ``n_time_bins`` h5 attr. Default
            ``None`` falls back to ``expected_n_time`` (the same dimension,
            different attr name). Pass an explicit int to override or
            ``None`` plus an explicit ``expected_n_time`` to keep them
            tied.
    """

    def __init__(
        self,
        h5_files: list[Path],
        rule: LabelRule,
        epochs_per_window: int = 8,
        require_paradigm: str | None = None,
        n_freqs: int = EXPECTED_N_FREQS,
        n_channels: int = EXPECTED_N_CHANNELS,
        expected_n_time: int = EXPECTED_N_TIME,
        expected_target_sfreq: float | None = None,
        expected_decimate_factor: int | None = None,
        expected_n_time_bins: int | None = None,
    ) -> None:
        if epochs_per_window != 8:
            raise ValueError(
                f"epochs_per_window must be 8 (BertSSL pos-embed is locked); "
                f"got {epochs_per_window}"
            )
        if n_freqs < 1:
            raise ValueError(f"n_freqs must be >=1, got {n_freqs}")
        if n_channels < 1:
            raise ValueError(f"n_channels must be >=1, got {n_channels}")
        if expected_n_time < 1:
            raise ValueError(f"expected_n_time must be >=1, got {expected_n_time}")
        self.epochs_per_window = epochs_per_window
        self.rule = rule
        self.n_freqs = n_freqs
        self.n_channels = n_channels
        self.expected_n_time = expected_n_time
        self.expected_target_sfreq = expected_target_sfreq
        self.expected_decimate_factor = expected_decimate_factor
        # Tie n_time_bins attr-check to the existing shape-check arg when
        # the caller did not explicitly override.  Same physical dimension,
        # different attr name in the h5 file.
        self.expected_n_time_bins = (
            expected_n_time_bins
            if expected_n_time_bins is not None
            else expected_n_time
        )
        # Index payload is tuple[str, ...] for groups schema (epoch keys)
        # or tuple[int, ...] for contiguous schema (row indices).
        self._index: list[
            tuple[Path, tuple[str, ...] | tuple[int, ...], tuple[int, ...]]
        ] = []
        self._vocab_size: int | None = None
        self._n_trials_kept = 0
        self._n_trials_dropped_partial = 0
        self._schema: str | None = None

        target_paradigm = require_paradigm or rule.paradigm
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                version = str(f.attrs.get("preprocess_version", ""))
                if version != EXPECTED_VERSION:
                    raise RuntimeError(
                        f"{h5_path.name}: preprocess_version={version!r} "
                        f"(expected {EXPECTED_VERSION!r}). Re-run extract_tf_features "
                        "with --task-filter erp-core."
                    )
                # Validate temporal-contract attrs written by extract_tf_features
                # (target_sfreq, decimate_factor, n_time_bins). Missing attrs
                # mean a legacy file written before PR #204; warn and proceed.
                _check_temporal_contract(
                    f,
                    h5_path,
                    self.expected_target_sfreq,
                    self.expected_decimate_factor,
                    self.expected_n_time_bins,
                )
                # Validate paradigm match if the file has a `task` attr.
                file_task = str(f.attrs.get("task", "")) or self._task_from_stem(
                    h5_path.stem
                )
                if file_task != target_paradigm:
                    raise RuntimeError(
                        f"{h5_path.name}: task={file_task!r} but rule expects "
                        f"paradigm={target_paradigm!r}. Filter h5_files by "
                        "paradigm before instantiating this dataset."
                    )

                schema = str(f.attrs.get("output_schema", "groups"))
                if schema not in ("groups", "contiguous"):
                    raise RuntimeError(
                        f"{h5_path.name}: unknown output_schema={schema!r}; "
                        "expected 'groups' or 'contiguous'."
                    )
                if self._schema is None:
                    self._schema = schema
                elif schema != self._schema:
                    raise RuntimeError(
                        f"{h5_path.name}: output_schema={schema!r} differs from "
                        f"earlier files' {self._schema!r}; mixed-schema "
                        "datasets are not supported."
                    )

                expected_shape = (
                    self.n_freqs,
                    self.n_channels,
                    self.expected_n_time,
                )

                if schema == "groups":
                    all_keys = sorted(k for k in f if k.startswith("epoch_"))
                    if not all_keys:
                        continue

                    first = f[all_keys[0]]
                    lp_shape = first["log_power"].shape
                    if lp_shape != expected_shape:
                        raise RuntimeError(
                            f"{h5_path.name}: log_power shape {lp_shape} != "
                            f"{expected_shape}; re-extract or pass matching "
                            "n_freqs / n_channels / expected_n_time."
                        )
                    hv = first["hed_vector"][:]
                    if self._vocab_size is None:
                        self._vocab_size = int(hv.shape[0])
                    elif hv.shape[0] != self._vocab_size:
                        raise RuntimeError(
                            f"{h5_path.name}: hed_vector dim {hv.shape[0]} != "
                            f"{self._vocab_size}; HED vocabulary mismatch."
                        )

                    kept_payload: list[str] = []
                    kept_labels: list[int] = []
                    for key in all_keys:
                        grp = f[key]
                        v = str(grp.attrs.get("event_value", ""))
                        et = str(grp.attrs.get("event_type", ""))
                        label = rule.label(v, et)
                        if label is not None:
                            kept_payload.append(key)
                            kept_labels.append(int(label))
                else:  # contiguous
                    if "log_power" not in f or "hed_vector" not in f:
                        raise RuntimeError(
                            f"{h5_path.name}: contiguous schema requires "
                            "top-level /log_power and /hed_vector datasets."
                        )
                    lp_shape = f["log_power"].shape
                    if lp_shape[1:] != expected_shape:
                        raise RuntimeError(
                            f"{h5_path.name}: log_power shape {lp_shape} != "
                            f"(*, {expected_shape}); re-extract or pass "
                            "matching n_freqs / n_channels / expected_n_time."
                        )
                    n_epochs_file = int(lp_shape[0])
                    if n_epochs_file == 0:
                        continue
                    hv_shape = f["hed_vector"].shape
                    if self._vocab_size is None:
                        self._vocab_size = int(hv_shape[1])
                    elif hv_shape[1] != self._vocab_size:
                        raise RuntimeError(
                            f"{h5_path.name}: hed_vector dim {hv_shape[1]} != "
                            f"{self._vocab_size}; HED vocabulary mismatch."
                        )
                    event_values = f["event_value"][:]
                    event_types = (
                        f["event_type"][:]
                        if "event_type" in f
                        else np.array([""] * n_epochs_file, dtype=object)
                    )

                    kept_payload_int: list[int] = []
                    kept_labels = []
                    for i in range(n_epochs_file):
                        v = _decode_h5_string(event_values[i], h5_path, "event_value")
                        et = _decode_h5_string(event_types[i], h5_path, "event_type")
                        label = rule.label(v, et)
                        if label is not None:
                            kept_payload_int.append(i)
                            kept_labels.append(int(label))
                    kept_payload = kept_payload_int

                # Pack consecutive kept trials into E windows; drop trailing.
                n_full_windows = len(kept_payload) // epochs_per_window
                for w in range(n_full_windows):
                    start = w * epochs_per_window
                    end = start + epochs_per_window
                    self._index.append(
                        (
                            h5_path,
                            tuple(kept_payload[start:end]),
                            tuple(kept_labels[start:end]),
                        )
                    )
                self._n_trials_kept += n_full_windows * epochs_per_window
                self._n_trials_dropped_partial += (
                    len(kept_payload) - n_full_windows * epochs_per_window
                )

        if not self._index:
            raise RuntimeError(
                f"ErpcoreParadigmDataset is empty for paradigm={target_paradigm!r} "
                f"probe={rule.probe!r} across {len(h5_files)} files. Either no "
                "matching trials or all files had < 8 kept trials."
            )

        logger.info(
            "ErpcoreParadigmDataset: %d windows (%d trials kept, %d trailing "
            "trials dropped) across %d files for paradigm=%r probe=%r",
            len(self._index),
            self._n_trials_kept,
            self._n_trials_dropped_partial,
            len(h5_files),
            target_paradigm,
            rule.probe,
        )

    @staticmethod
    def _task_from_stem(stem: str) -> str:
        """Extract task name from h5 filename stem ``{subject}_{task}``."""
        parts = stem.split("_", 1)
        return parts[1] if len(parts) == 2 else ""

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            raise RuntimeError("vocab_size not set; dataset is empty.")
        return self._vocab_size

    @property
    def n_trials_kept(self) -> int:
        return self._n_trials_kept

    @property
    def n_trials_dropped_partial(self) -> int:
        return self._n_trials_dropped_partial

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h5_path, window_payload, window_labels = self._index[idx]
        n_e = self.epochs_per_window
        # Positional offsets fed to BertSSL as a stable per-position embedding
        # input. The 1000 ms stride is a fixed positional convention
        # (PackedSSLDataset uses the same value), NOT the actual epoch
        # duration in ms —  1.0 s and   1.6 s windows both use
        # this stride. BertSSL learns position embeddings from these values;
        # changing the stride would invalidate any pretrained checkpoint.
        event_offsets_ms = np.arange(n_e, dtype=np.float32) * _EVENT_POSITION_STRIDE_MS

        with h5py.File(h5_path, "r") as f:
            if window_payload and isinstance(window_payload[0], str):
                tf_window = np.empty(
                    (n_e, self.n_freqs, self.n_channels, self.expected_n_time),
                    dtype=np.float32,
                )
                hed_window = np.empty((n_e, self.vocab_size), dtype=np.float32)
                for i, key in enumerate(window_payload):
                    grp = f[key]
                    tf_window[i] = grp["log_power"][:]
                    hed_window[i] = grp["hed_vector"][:]
            else:
                # Contiguous schema: payload is a tuple of row indices.
                indices = np.asarray(window_payload, dtype=np.int64)
                tf_window = f["log_power"][indices].astype(np.float32, copy=False)
                hed_window = f["hed_vector"][indices].astype(np.float32, copy=False)

        return {
            "tf": torch.from_numpy(np.ascontiguousarray(tf_window)),  # (E, F, C, T)
            "hed": torch.from_numpy(np.ascontiguousarray(hed_window)),  # (E, V)
            "event_offsets_ms": torch.from_numpy(event_offsets_ms),  # (E,)
            "labels": torch.tensor(window_labels, dtype=torch.long),  # (E,)
        }


def paradigm_collate(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack ParadigmDataset windows along the batch dimension."""
    tf = torch.stack([b["tf"] for b in batch], dim=0)  # (B, E, F, C, T)
    hed = torch.stack([b["hed"] for b in batch], dim=0)  # (B, E, V)
    offsets = torch.stack([b["event_offsets_ms"] for b in batch], dim=0)  # (B, E)
    labels = torch.stack([b["labels"] for b in batch], dim=0)  # (B, E)
    return {"tf": tf, "hed": hed, "event_offsets_ms": offsets, "labels": labels}
