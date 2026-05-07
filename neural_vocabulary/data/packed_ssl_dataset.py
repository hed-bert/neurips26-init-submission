"""V10 Gate D.1: packed SSL dataset over Morlet TF non-movie features.

Loads contiguous 8-epoch windows from V10 Gate D non-movie Morlet feature
files. Each window yields (TF tensor stacked over the window, HED multi-hot
tensor stacked over the window). Variable-length handling is *not* needed
here because the non-movie extractor produces uniform (6, 64, 10) per-epoch
features (stim/response events only).

Pretraining target:
    - TF patches: the masked SSL stream reconstructs (freq, time) patches
      from Morlet log-power. Shape per window: ``(E, F, C, T)``.
    - HED vectors: per-epoch multi-hot (vocab_size,) targets for masked-HED
      prediction. Shape per window: ``(E, vocab_size)``.

Index structure (built at init):
    For a file with ``n_epochs_in_file >= epochs_per_window``, we register
    one index entry per non-overlapping contiguous window
    ``(file_path, start_idx)`` where ``start_idx in {0, stride, 2*stride, ...}``.
    Default stride equals epochs_per_window (non-overlapping). A shorter
    stride boosts augmentation at the cost of within-file correlation; not
    used by D.1 baseline.

The returned dict is intentionally minimal and shaped for the SSL training
loop; it does not use the V7 packed-collator layout because Gate D.1 does
not interleave event tokens with patch tokens in a flat sequence — the
transformer treats events as a separate short stream (see
``neural_vocabulary/models/bert_ssl.py``).
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

logger = logging.getLogger(__name__)

# Schema tag written by v10_morlet_extract.py.
EXPECTED_VERSION: str = "v10_gate_a_morlet"

# Default per-epoch TF layout for non-movie Morlet extraction
# (v10_morlet_extract.NON_MOVIE_EPOCH_LEN // decimate_factor = 10).
# These are defaults, not invariants — pass alternative shapes through
# the constructor when the on-disk layout differs (channel-as-token
# patch-embeds, denser frequency grids, smaller channel counts).
EXPECTED_N_FREQS: int = 6
EXPECTED_N_CHANNELS: int = 64
EXPECTED_N_TIME: int = 10

# V11 dense defaults written by v10_morlet_extract --n-time 50 on a 1-second
# ERP-CORE epoch (100 Hz source, decimate_factor=2 → target_sfreq=50 Hz).
V11_TARGET_SFREQ: float = 50.0
V11_DECIMATE_FACTOR: int = 2
V11_N_TIME_BINS: int = 50

# Positional stride between consecutive event tokens fed to BertSSL,
# measured in milliseconds. This is a FIXED positional convention —
# V10 1.0 s windows and V11 R4-A 1.6 s windows both use 1000 ms — so a
# pretrained checkpoint's position embedding is interpretable across
# both source corpora. Changing this invalidates the encoder.
_EVENT_POSITION_STRIDE_MS: float = 1000.0


def _check_temporal_contract(
    f: h5py.File,
    h5_path: Path,
    expected_target_sfreq: float | None,
    expected_decimate_factor: int | None,
    expected_n_time_bins: int | None,
) -> None:
    """Validate temporal-contract attrs against expected values.

    Reads ``target_sfreq``, ``decimate_factor``, and ``n_time_bins`` from the
    h5 file root attrs (written by ``v10_morlet_extract`` since PR #204).
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


class PackedSSLDataset(Dataset):
    """Contiguous-window streaming dataset for Gate D.1 dual-stream SSL.

    Args:
        h5_files: V10 Gate D non-movie Morlet feature files (produced by
            ``v10_morlet_extract --task-filter non-movie``).
        epochs_per_window: Number of consecutive epochs packed into one SSL
            window. Default 8 (design: 8 event tokens + 8*15 TF patches at
            the non-movie T=10 grid — see issue #173).
        stride: Start-index stride between windows inside a single file.
            Default equals ``epochs_per_window`` (non-overlapping windows).
        n_freqs: Expected frequency-bin count per epoch. Defaults to 6
            (Morlet 6-band non-movie). Override for alternative
            spectrogram parameterizations.
        n_channels: Expected channel count per epoch. Defaults to 64
            (HBN 64-electrode harmonization). Override for datasets
            with different channel counts (e.g. ERP-CORE 30ch).
        expected_n_time: Expected time-bin count per epoch. Defaults to 10
            (non-movie). The dataset refuses to mix non-matching shapes to
            prevent silent heterogeneous batches.
        expected_target_sfreq: Expected ``target_sfreq`` h5 attr written by
            ``v10_morlet_extract``. Default ``None`` (skip check). Pass a
            float (e.g. 50.0 for V11 dense ERP-CORE, 10.0 for V10 non-movie)
            to enable the check.
        expected_decimate_factor: Expected ``decimate_factor`` h5 attr.
            Default ``None`` (skip). Pass an int to enable.
        expected_n_time_bins: Expected ``n_time_bins`` h5 attr. Default
            ``None`` (skip). Pass an int to enable.
    """

    def __init__(
        self,
        h5_files: list[Path],
        epochs_per_window: int = 8,
        stride: int | None = None,
        n_freqs: int = EXPECTED_N_FREQS,
        n_channels: int = EXPECTED_N_CHANNELS,
        expected_n_time: int = EXPECTED_N_TIME,
        expected_target_sfreq: float | None = None,
        expected_decimate_factor: int | None = None,
        expected_n_time_bins: int | None = None,
    ) -> None:
        if epochs_per_window < 1:
            raise ValueError(f"epochs_per_window must be >=1, got {epochs_per_window}")
        if stride is None:
            stride = epochs_per_window
        if stride < 1:
            raise ValueError(f"stride must be >=1, got {stride}")
        if n_freqs < 1:
            raise ValueError(f"n_freqs must be >=1, got {n_freqs}")
        if n_channels < 1:
            raise ValueError(f"n_channels must be >=1, got {n_channels}")

        self.epochs_per_window = epochs_per_window
        self.stride = stride
        self.n_freqs = n_freqs
        self.n_channels = n_channels
        self.expected_n_time = expected_n_time
        self.expected_target_sfreq = expected_target_sfreq
        self.expected_decimate_factor = expected_decimate_factor
        self.expected_n_time_bins = expected_n_time_bins

        # Index entries: (file_path, payload). payload is tuple[str, ...]
        # for the V10 per-epoch-group schema (epoch keys), or int (start
        # index) for the V11 R4-A contiguous schema. self._schema is set
        # from the first file and enforced uniform across all files.
        self._index: list[tuple[Path, tuple[str, ...] | int]] = []
        self._vocab_size: int | None = None
        self._schema: str | None = None

        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                version = f.attrs.get("preprocess_version", "")
                if version != EXPECTED_VERSION:
                    raise RuntimeError(
                        f"{h5_path}: expected preprocess_version="
                        f"{EXPECTED_VERSION!r}, got {version!r}. Run "
                        "v10_morlet_extract.py --task-filter non-movie first."
                    )
                # Validate temporal-contract attrs written by v10_morlet_extract
                # (target_sfreq, decimate_factor, n_time_bins). Missing attrs
                # mean a legacy file written before PR #204; warn and proceed.
                _check_temporal_contract(
                    f,
                    h5_path,
                    expected_target_sfreq,
                    expected_decimate_factor,
                    expected_n_time_bins,
                )
                # Detect schema. Files written before output_schema was added
                # (V10 era) lack the attr; treat them as "groups".
                schema = str(f.attrs.get("output_schema", "groups"))
                if schema not in ("groups", "contiguous"):
                    raise RuntimeError(
                        f"{h5_path}: unknown output_schema={schema!r}; expected "
                        "'groups' or 'contiguous'."
                    )
                if self._schema is None:
                    self._schema = schema
                elif schema != self._schema:
                    raise RuntimeError(
                        f"{h5_path}: output_schema={schema!r} differs from "
                        f"earlier files' {self._schema!r}. Mixed-schema "
                        "datasets are not supported; re-extract uniformly."
                    )

                if schema == "groups":
                    all_keys = sorted(k for k in f if k.startswith("epoch_"))
                    n_epochs_file = len(all_keys)
                    if n_epochs_file < epochs_per_window:
                        continue
                    first_key = all_keys[0]
                    lp_shape = f[first_key]["log_power"].shape
                    if lp_shape != (
                        self.n_freqs,
                        self.n_channels,
                        self.expected_n_time,
                    ):
                        raise RuntimeError(
                            f"{h5_path}: log_power shape {lp_shape} does not match "
                            f"({self.n_freqs}, {self.n_channels}, "
                            f"{self.expected_n_time}). Re-extract or pass matching "
                            "n_freqs / n_channels / expected_n_time to the dataset."
                        )
                    hv = f[first_key]["hed_vector"][:]
                    if self._vocab_size is None:
                        self._vocab_size = int(hv.shape[0])
                    elif hv.shape[0] != self._vocab_size:
                        raise RuntimeError(
                            f"{h5_path}: hed_vector dim {hv.shape[0]} != expected "
                            f"{self._vocab_size}. HED vocabulary mismatch across "
                            "files; rebuild the vectorizer."
                        )

                    max_start = n_epochs_file - epochs_per_window
                    for start in range(0, max_start + 1, stride):
                        window_keys = tuple(all_keys[start : start + epochs_per_window])
                        self._index.append((h5_path, window_keys))
                else:  # contiguous
                    if "log_power" not in f or "hed_vector" not in f:
                        raise RuntimeError(
                            f"{h5_path}: contiguous schema requires top-level "
                            "/log_power and /hed_vector datasets."
                        )
                    lp_shape = f["log_power"].shape
                    if lp_shape[1:] != (
                        self.n_freqs,
                        self.n_channels,
                        self.expected_n_time,
                    ):
                        raise RuntimeError(
                            f"{h5_path}: log_power shape {lp_shape} does not match "
                            f"(*, {self.n_freqs}, {self.n_channels}, "
                            f"{self.expected_n_time}). Re-extract or pass "
                            "matching n_freqs / n_channels / expected_n_time."
                        )
                    n_epochs_file = int(lp_shape[0])
                    if n_epochs_file < epochs_per_window:
                        continue
                    hv_shape = f["hed_vector"].shape
                    if self._vocab_size is None:
                        self._vocab_size = int(hv_shape[1])
                    elif hv_shape[1] != self._vocab_size:
                        raise RuntimeError(
                            f"{h5_path}: hed_vector dim {hv_shape[1]} != expected "
                            f"{self._vocab_size}. HED vocabulary mismatch across "
                            "files; rebuild the vectorizer."
                        )

                    max_start = n_epochs_file - epochs_per_window
                    for start in range(0, max_start + 1, stride):
                        self._index.append((h5_path, start))

        if not self._index:
            raise RuntimeError(
                "PackedSSLDataset index is empty. Either no input files or "
                f"none had >= {epochs_per_window} epochs. "
                f"n_files={len(h5_files)}, epochs_per_window={epochs_per_window}."
            )

        logger.info(
            "PackedSSLDataset: %d windows across %d files (epochs_per_window=%d, "
            "stride=%d, vocab_size=%d)",
            len(self._index),
            len(h5_files),
            epochs_per_window,
            stride,
            self._vocab_size,
        )

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            raise RuntimeError("vocab_size not set; PackedSSLDataset is empty.")
        return self._vocab_size

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h5_path, payload = self._index[idx]
        # Positional stride convention; see _EVENT_POSITION_STRIDE_MS.
        event_offsets_ms = (
            np.arange(self.epochs_per_window, dtype=np.float32)
            * _EVENT_POSITION_STRIDE_MS
        )

        with h5py.File(h5_path, "r") as f:
            if isinstance(payload, tuple):
                # groups schema
                tf_window = np.empty(
                    (
                        self.epochs_per_window,
                        self.n_freqs,
                        self.n_channels,
                        self.expected_n_time,
                    ),
                    dtype=np.float32,
                )
                hed_window = np.empty(
                    (self.epochs_per_window, self.vocab_size), dtype=np.float32
                )
                for i, key in enumerate(payload):
                    grp = f[key]
                    tf_window[i] = grp["log_power"][:]
                    hed_window[i] = grp["hed_vector"][:]
            else:
                # contiguous schema: payload is the start index
                start = int(payload)
                stop = start + self.epochs_per_window
                tf_window = f["log_power"][start:stop].astype(np.float32, copy=False)
                hed_window = f["hed_vector"][start:stop].astype(np.float32, copy=False)

        return {
            "tf": torch.from_numpy(np.ascontiguousarray(tf_window)),  # (E, F, C, T)
            "hed": torch.from_numpy(np.ascontiguousarray(hed_window)),  # (E, V)
            "event_offsets_ms": torch.from_numpy(event_offsets_ms),  # (E,)
        }


def packed_ssl_collate(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Stack PackedSSLDataset dicts along the batch dimension.

    All windows share the same epochs_per_window and TF shape, so a plain
    stack is sufficient; no padding or attention-mask bookkeeping is needed
    at the collator level.
    """
    tf = torch.stack([b["tf"] for b in batch], dim=0)  # (B, E, F, C, T)
    hed = torch.stack([b["hed"] for b in batch], dim=0)  # (B, E, V)
    offsets = torch.stack([b["event_offsets_ms"] for b in batch], dim=0)  # (B, E)
    return {"tf": tf, "hed": hed, "event_offsets_ms": offsets}
