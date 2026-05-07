""" Gate A.1 / the HED-objective ablation: Morlet wavelet TF feature extractor.

Precomputes per-trial Morlet wavelet log-power features on preprocessed
preprocessed HBN epochs and writes them to disk as HDF5 files (one per
recording, mirroring the source layout).

Task-filter modes:
    --task-filter movie (default): passive-movie tasks only
        (DespicableMe/DiaryOfAWimpyKid/FunwithFractals/ThePresent).
        Accepts length=220 (tonic) epochs. 22 time bins after decimation.
    --task-filter non-movie: all other HBN tasks, stim+response events only.
        Accepts length=100 epochs with epoch_type in {stim, response}.
        Drops RestingState (no real events) and non-movie tonic epochs
        (not useful for event-grounded SSL). 10 time bins after decimation.

Output schema (per epoch group):
    log_power:  float32, shape (n_freqs, n_channels, n_time_bins)
    hed_vector: float32, shape (vocab_size,) — copied verbatim from source

File-level attrs (mirrored + -specific):
    freqs, n_cycles, source_sfreq, target_sfreq, channel_names,
    n_epochs (written), n_epochs_skipped, n_epochs_no_hed,
    preprocess_version="v10_gate_a_morlet", source_version="v9_tier1",
    window_spec, baseline_correction, task_filter, expected_epoch_len

Edge-effect note:
    Morlet at 4 Hz / n_cycles=4 → ~1 s wavelet. For 220-sample epochs
    first/last ~0.5 s has boundary leakage. For 100-sample (1 s) non-movie
    epochs the entire 4 Hz band is edge-dominated; kept for parity with the
    movie pipeline and documented here. All trials share the same edge
    effect; the downstream LR probe compensates. No pad/crop is applied.

Usage (smoke-test, non-movie for the HED-objective ablation):
    uv run python -m neural_vocabulary.scripts.extract_tf_features \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir /tmp/d1_nonmovie_smoke \\
        --task-filter non-movie --limit 8

Usage (full non-movie run, submit via gpu_queue):
    uv run python -m neural_vocabulary.scripts.extract_tf_features \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir ${HBN_DATA_DIR}/tf_features_nonmovie \\
        --task-filter non-movie --n-workers 8

Usage (full movie run, default — preserves Gate A/B/C behavior):
    uv run python -m neural_vocabulary.scripts.extract_tf_features \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir ${HBN_DATA_DIR}/v10_gate_a_features \\
        --n-workers 8
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path

import h5py
import mne
import numpy as np

logger = logging.getLogger(__name__)

# Passive-movie tasks that contain shot-boundary epochs.
PASSIVE_MOVIE_TASKS: frozenset[str] = frozenset(
    ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
)

# Non-movie HBN tasks that contain discrete stim/response events.
# RestingState is intentionally excluded: it has no events (all-tonic).
NON_MOVIE_EVENT_TASKS: frozenset[str] = frozenset(
    [
        "contrastChangeDetection",
        "seqLearning6target",
        "seqLearning8target",
        "surroundSupp",
        "symbolSearch",
    ]
)

# ERP-CORE paradigms (NEMAR nm000132). Single source-of-truth for the
# task name set; the preprocessor imports this and uses it for both
# discovery and the CLI --tasks choices, ensuring a 7th task added
# upstream surfaces in both pipelines from one edit.
ERP_CORE_TASKS: frozenset[str] = frozenset(
    ["flankers", "MMN", "N170", "N2pc", "N400", "P3"]
)

# Epoch types kept for non-movie extraction (stim/response events only).
NON_MOVIE_EPOCH_TYPES: frozenset[str] = frozenset(["stim", "response"])

# Default frequency set: log-spaced 4–28 Hz covering delta/theta/alpha/beta.
DEFAULT_FREQS: list[float] = [4.0, 6.0, 9.0, 13.0, 19.0, 28.0]

# Fixed cycles per wavelet (constant time resolution across bands).
DEFAULT_N_CYCLES: int = 4

# Source and target sampling frequencies.
SOURCE_SFREQ: float = 100.0
TARGET_SFREQ: float = 10.0

# Downsample factor.
_DECIMATE_FACTOR: int = int(SOURCE_SFREQ / TARGET_SFREQ)  # 10

# Expected epoch lengths in samples per task-filter mode.
MOVIE_EPOCH_LEN: int = 220  # passive-movie tonic window (2.2 s)
NON_MOVIE_EPOCH_LEN: int = 100  # stim/response window (1.0 s)

# Backwards-compatible alias: existing callers (and tests) import
# EXPECTED_EPOCH_LEN for the movie pipeline.
EXPECTED_EPOCH_LEN: int = MOVIE_EPOCH_LEN

# Expected number of time bins after downsampling (movie: 22; non-movie: 10).
EXPECTED_N_TIME_BINS: int = EXPECTED_EPOCH_LEN // _DECIMATE_FACTOR


def _task_from_stem(stem: str) -> str:
    """Extract task name from h5 filename stem: {subject}_{task}[_run-N]."""
    parts = stem.split("_")
    return parts[1] if len(parts) >= 2 else ""


def _is_passive_movie(h5_path: Path) -> bool:
    """Return True if the file belongs to a passive-movie task."""
    return _task_from_stem(h5_path.stem) in PASSIVE_MOVIE_TASKS


def _is_non_movie_event_task(h5_path: Path) -> bool:
    """Return True if the file belongs to a non-movie event-bearing task."""
    return _task_from_stem(h5_path.stem) in NON_MOVIE_EVENT_TASKS


def _is_erp_core_task(h5_path: Path) -> bool:
    """Return True if the file belongs to an ERP-CORE paradigm."""
    return _task_from_stem(h5_path.stem) in ERP_CORE_TASKS


def _strided_mean(x: np.ndarray, factor: int) -> np.ndarray:
    """Downsample last axis by strided mean (no anti-aliasing needed for power).

    Args:
        x: Array of shape (..., T) where T must be divisible by factor.
        factor: Downsample factor.

    Returns:
        Array of shape (..., T // factor).
    """
    t = x.shape[-1]
    if t % factor != 0:
        # Trim trailing samples so T is divisible; consistent across all epochs.
        x = x[..., : t - (t % factor)]
    return x.reshape(*x.shape[:-1], -1, factor).mean(axis=-1)


def parse_n_cycles_list_arg(raw: str, n_freqs: int) -> np.ndarray:
    """Parse --n-cycles-list into a per-freq float64 array.

    Raises ValueError on non-numeric tokens, length mismatch, or
    non-positive values. Callers should translate to ``parser.error``.
    """
    try:
        values = [float(x) for x in raw.split(",")]
    except ValueError as e:
        raise ValueError(
            f"--n-cycles-list must be comma-separated floats: {e}"
        ) from None
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) != n_freqs:
        raise ValueError(
            f"--n-cycles-list length {len(arr)} != --freqs length {n_freqs}"
        )
    if (arr <= 0).any():
        raise ValueError(f"--n-cycles-list values must be positive; got {arr}")
    return arr


def _resolve_n_cycles(
    n_cycles: int | np.ndarray,
    freqs: np.ndarray,
    signal_length: int,
    sfreq: float,
) -> int | np.ndarray:
    """Resolve the final n_cycles tensor for tfr_array_morlet.

    MNE's Morlet wavelet is built over a time window of roughly
    2 * n_cycles * sfreq / freq samples (symmetric around t=0 extending to
    ±5σ, with σ = n_cycles / (2π · freq) · sfreq). MNE then requires the
    wavelet be shorter than the signal, so the scalar n_cycles must satisfy
    ``2 * n_cycles * sfreq / freq < signal_length`` at every frequency.

    For 220-sample movie epochs, the default n_cycles=4 is safe (longest
    wavelet ≈ 200 at 4 Hz < 220). For 100-sample non-movie stim/response
    epochs it fails at 4 Hz, so we shrink n_cycles per-frequency with a
    10 % safety margin.

    When a scalar n_cycles would exceed the signal at any frequency, return
    a per-frequency array. Otherwise return the scalar unchanged.
    """
    if isinstance(n_cycles, np.ndarray):
        # User-supplied per-freq array. Warn (don't override) when any
        # entry exceeds the MNE-internal safety bound —  intentionally
        # uses n_cycles=2 at 2 Hz with a 1.6 s window. Cone-of-influence
        # is symmetric (~1σ ≈ n_cycles / (2π · f) per edge): at 2 Hz with
        # n_cycles=2 the COI half-width is ~160 ms, contaminating both
        # the first and last ~160 ms relative to baseline-corrected zero.
        wavelet_len_per_freq = 2.0 * n_cycles * sfreq / freqs
        over = wavelet_len_per_freq >= signal_length
        if over.any():
            logger.warning(
                "n_cycles per-freq array exceeds signal_length=%d at freqs=%s; "
                "wavelet lengths %s. MNE will edge-contaminate (symmetric "
                "COI both edges); expected for  2 Hz/n_cycles=2/1.6 s.",
                signal_length,
                freqs[over].tolist(),
                wavelet_len_per_freq[over].tolist(),
            )
        return n_cycles
    # Longest MNE wavelet at freq f has length ≈ 2 * n_cycles * sfreq / f.
    min_freq = float(np.min(freqs))
    max_wavelet_len = 2.0 * n_cycles * sfreq / min_freq
    if max_wavelet_len < signal_length:
        return n_cycles
    # Build a per-frequency n_cycles that respects the signal length.
    # Keep a 10 % safety margin against MNE's internal sampling of the
    # wavelet (rounding can push length slightly over 2·cycles·sfreq/freq).
    max_safe_cycles = 0.45 * signal_length * freqs / sfreq
    resolved = np.minimum(n_cycles, max_safe_cycles).astype(np.float64)
    # MNE needs n_cycles >= ~1 for a sensible wavelet; clamp conservatively.
    resolved = np.clip(resolved, a_min=1.0, a_max=None)
    return resolved


def _compute_log_power(
    eeg: np.ndarray,
    freqs: np.ndarray,
    n_cycles: int | np.ndarray,
    sfreq: float,
    decimate_factor: int,
) -> np.ndarray:
    """Compute log-power TF features for one epoch.

    Args:
        eeg: float32 array of shape (n_channels, n_times).
        freqs: Center frequencies in Hz.
        n_cycles: Wavelet width. Int (fixed across frequencies) or per-freq
            float array (used for short-signal epochs where a fixed scalar
            would produce wavelets longer than the signal).
        sfreq: Sampling frequency of the input signal.
        decimate_factor: Temporal downsampling factor.

    Returns:
        float32 array of shape (n_freqs, n_channels, n_time_bins).
    """
    # tfr_array_morlet expects (n_epochs, n_channels, n_times).
    eeg_3d = eeg[np.newaxis]  # (1, n_channels, n_times)

    # MNE tfr_array_morlet returns (n_epochs, n_channels, n_freqs, n_times).
    power = mne.time_frequency.tfr_array_morlet(
        eeg_3d,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        n_jobs=1,
        verbose=False,
    )  # (1, n_channels, n_freqs, n_times) - float64

    # Remove batch dim, reorder to (n_freqs, n_channels, n_times).
    power = power[0].transpose(1, 0, 2)  # (n_freqs, n_channels, n_times)

    # Log-compress: log(power + eps)
    log_p = np.log(power.astype(np.float64) + 1e-8).astype(np.float32)

    # Downsample time axis.
    log_p_ds = _strided_mean(log_p, decimate_factor)  # (n_freqs, n_channels, T_ds)

    return log_p_ds.astype(np.float32)


def _process_file(
    h5_path: Path,
    output_dir: Path,
    freqs: np.ndarray,
    n_cycles: int | np.ndarray,
    decimate_factor: int,
    target_sfreq: float,
    overwrite: bool,
    expected_epoch_len: int = MOVIE_EPOCH_LEN,
    allowed_epoch_types: frozenset[str] | None = None,
    task_filter: str = "movie",
    output_schema: str = "groups",
) -> dict:
    """Process one H5 file and write TF features.

    Returns a stats dict: {n_written, n_skipped_no_hed, n_skipped_wrong_len,
                            n_skipped_wrong_type, status, path}.

    Writes atomically via a sibling `<name>.h5.tmp` path + os.replace so a
    SIGKILL/OOM/disk-full mid-write cannot leave a corrupt output file that
    future runs silently skip.

    Args:
        expected_epoch_len: Epochs with a different `length` attr are skipped.
            220 for movie (tonic) pipeline; 100 for non-movie (stim/response).
        allowed_epoch_types: If set, only epochs with `epoch_type` in this
            set are kept. Used to restrict non-movie extraction to stim and
            response events (drops tonic).
        task_filter: Recorded in the output attrs for downstream provenance.
    """
    out_path = output_dir / h5_path.name
    tmp_path = out_path.with_suffix(".h5.tmp")

    if out_path.exists() and not overwrite:
        logger.debug("Skipping existing: %s", out_path)
        return {"status": "already_exists", "path": str(h5_path), "n_written": 0}

    stats: dict = {
        "status": "ok",
        "path": str(h5_path),
        "n_written": 0,
        "n_skipped_no_hed": 0,
        "n_skipped_wrong_len": 0,
        "n_skipped_wrong_type": 0,
    }

    expected_n_time_bins = expected_epoch_len // decimate_factor

    try:
        with h5py.File(h5_path, "r") as src:
            # Validate source version. v11_tier_r4 / v11_erpcore_r4 share
            # the  per-epoch-group schema; only window/baseline differ.
            version = src.attrs.get("preprocess_version", "")
            if version not in ("v9_tier1", "v11_tier_r4", "v11_erpcore_r4"):
                raise ValueError(
                    f"Expected preprocess_version in (v9_tier1, v11_tier_r4, "
                    f"v11_erpcore_r4), got {version!r} in {h5_path}. "
                    "Wrong source directory?"
                )

            n_epochs_src = int(src.attrs.get("n_epochs", 0))
            sfreq = float(src.attrs.get("sfreq", SOURCE_SFREQ))

            # Collect all epochs to batch the MNE call.
            valid_epochs: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
            n_skipped_no_hed = 0
            n_skipped_wrong_len = 0
            n_skipped_wrong_type = 0

            for i in range(n_epochs_src):
                grp_name = f"epoch_{i:05d}"
                if grp_name not in src:
                    continue

                grp = src[grp_name]
                epoch_len = int(grp.attrs.get("length", 0))

                # Filter by epoch length (task-filter-dependent).
                if epoch_len != expected_epoch_len:
                    n_skipped_wrong_len += 1
                    continue

                # Filter by epoch type if specified (non-movie: stim/response).
                if allowed_epoch_types is not None:
                    epoch_type = str(grp.attrs.get("epoch_type", ""))
                    if epoch_type not in allowed_epoch_types:
                        n_skipped_wrong_type += 1
                        continue

                if "hed_vector" not in grp:
                    n_skipped_no_hed += 1
                    continue

                eeg = grp["eeg"][:].astype(np.float32)
                hed_vector = grp["hed_vector"][:].astype(np.float32)

                # Collect epoch attrs to copy. event_type is the BIDS column
                # carrying ERP-CORE N170's load-bearing label (face/car/etc.);
                # downstream paradigm probes need it. Other datasets don't
                # write event_type — the `if k in grp.attrs` filter handles
                # missing-attr cleanly.
                epoch_attrs = {
                    k: grp.attrs[k]
                    for k in (
                        "event_id",
                        "event_value",
                        "event_type",
                        "hed_tag",
                        "epoch_type",
                        "onset_sample",
                        "pre_event_samples",
                        "length",
                    )
                    if k in grp.attrs
                }
                valid_epochs.append((grp_name, eeg, hed_vector, epoch_attrs))

            stats["n_skipped_no_hed"] = n_skipped_no_hed
            stats["n_skipped_wrong_len"] = n_skipped_wrong_len
            stats["n_skipped_wrong_type"] = n_skipped_wrong_type

            if not valid_epochs:
                stats["status"] = "no_valid_epochs"
                return stats

            # Copy file-level attrs we need AFTER src is closed. `task` and
            # `dataset` are present on ERP-CORE files and let downstream
            # consumers route per-paradigm without filename parsing.
            src_attrs_to_copy = {
                k: src.attrs[k]
                for k in (
                    "channel_names",
                    "window_spec",
                    "baseline_correction",
                    "n_channels",
                    "task",
                    "dataset",
                    "subject_id",
                )
                if k in src.attrs
            }

            # Pre-compute TF features and validate shapes BEFORE opening dst,
            # so a shape mismatch cannot leave an empty output file on disk.
            # Resolve n_cycles once per file against the (uniform) epoch length.
            resolved_n_cycles = _resolve_n_cycles(
                n_cycles, freqs, expected_epoch_len, sfreq
            )
            # Record any frequencies whose wavelet exceeds the signal so the
            # COI / edge-contamination flag is carried in the artifact, not
            # only in worker logs ( 2 Hz / n_cycles=2 / 1.6 s case).
            if isinstance(resolved_n_cycles, np.ndarray):
                wavelet_lens = 2.0 * resolved_n_cycles * sfreq / freqs
                edge_freqs = freqs[wavelet_lens >= expected_epoch_len].tolist()
            else:
                edge_freqs = []
            processed: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
            for grp_name, eeg, hed_vector, epoch_attrs in valid_epochs:
                log_p = _compute_log_power(
                    eeg, freqs, resolved_n_cycles, sfreq, decimate_factor
                )
                expected_shape = (len(freqs), eeg.shape[0], expected_n_time_bins)
                if log_p.shape != expected_shape:
                    raise RuntimeError(
                        f"Unexpected log_power shape {log_p.shape} "
                        f"(expected {expected_shape}) in {h5_path}/{grp_name}"
                    )
                processed.append((grp_name, log_p, hed_vector, epoch_attrs))

        # Source closed; write atomically to tmp_path then rename.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with h5py.File(tmp_path, "w") as dst:
                for k, v in src_attrs_to_copy.items():
                    dst.attrs[k] = v

                dst.attrs["freqs"] = list(freqs)
                if isinstance(resolved_n_cycles, np.ndarray):
                    # Convert the per-freq array to a plain float32 list so the
                    # hdf5 attr round-trips cleanly (h5py is happy with 1-D
                    # float arrays; avoids ty's overload-matching on .astype).
                    dst.attrs["n_cycles"] = np.asarray(
                        resolved_n_cycles, dtype=np.float32
                    )
                else:
                    dst.attrs["n_cycles"] = resolved_n_cycles
                dst.attrs["source_sfreq"] = sfreq
                dst.attrs["target_sfreq"] = target_sfreq
                # Persist the temporal-contract triple alongside target_sfreq so
                # artifacts produced via --n-time are self-describing without
                # the consumer having to re-derive the bin count from
                # target_sfreq + epoch_len.
                dst.attrs["decimate_factor"] = decimate_factor
                dst.attrs["n_time_bins"] = expected_n_time_bins
                dst.attrs["n_epochs"] = len(processed)
                dst.attrs["n_epochs_skipped"] = (
                    n_skipped_no_hed + n_skipped_wrong_len + n_skipped_wrong_type
                )
                dst.attrs["n_epochs_no_hed"] = n_skipped_no_hed
                dst.attrs["n_epochs_wrong_len"] = n_skipped_wrong_len
                dst.attrs["n_epochs_wrong_type"] = n_skipped_wrong_type
                dst.attrs["preprocess_version"] = "v10_gate_a_morlet"
                dst.attrs["source_version"] = version
                dst.attrs["task_filter"] = task_filter
                dst.attrs["expected_epoch_len"] = expected_epoch_len
                dst.attrs["output_schema"] = output_schema
                if edge_freqs:
                    dst.attrs["n_cycles_edge_contaminated_freqs"] = edge_freqs
                if allowed_epoch_types is not None:
                    dst.attrs["allowed_epoch_types"] = sorted(allowed_epoch_types)

                if output_schema == "groups":
                    for grp_name, log_p, hed_vector, epoch_attrs in processed:
                        grp_out = dst.create_group(grp_name)
                        grp_out.create_dataset(
                            "log_power",
                            data=log_p,
                            dtype=np.float32,
                            compression=None,
                        )
                        grp_out.create_dataset(
                            "hed_vector",
                            data=hed_vector,
                            dtype=np.float32,
                            compression=None,
                        )
                        for k, v in epoch_attrs.items():
                            grp_out.attrs[k] = v

                        stats["n_written"] += 1
                elif output_schema == "contiguous":
                    n = len(processed)
                    if n == 0:
                        raise RuntimeError(
                            f"contiguous-schema write attempted with 0 epochs "
                            f"in {h5_path}; should have returned no_valid_epochs"
                        )
                    f_n, c_n, t_n = processed[0][1].shape
                    v_n = processed[0][2].shape[0]
                    log_power_arr = np.empty((n, f_n, c_n, t_n), dtype=np.float32)
                    hed_vector_arr = np.empty((n, v_n), dtype=np.float32)
                    str_dt = h5py.string_dtype(encoding="utf-8")
                    epoch_id_arr = np.empty(n, dtype=object)
                    event_value_arr = np.empty(n, dtype=object)
                    event_type_arr = np.empty(n, dtype=object)
                    epoch_type_arr = np.empty(n, dtype=object)
                    hed_tag_arr = np.empty(n, dtype=object)
                    event_id_arr = np.zeros(n, dtype=np.int64)
                    onset_sample_arr = np.zeros(n, dtype=np.int64)
                    pre_event_samples_arr = np.zeros(n, dtype=np.int64)
                    length_arr = np.zeros(n, dtype=np.int64)

                    for i, (grp_name, log_p, hed_vector, epoch_attrs) in enumerate(
                        processed
                    ):
                        log_power_arr[i] = log_p
                        hed_vector_arr[i] = hed_vector
                        epoch_id_arr[i] = grp_name
                        event_value_arr[i] = str(epoch_attrs.get("event_value", ""))
                        event_type_arr[i] = str(epoch_attrs.get("event_type", ""))
                        epoch_type_arr[i] = str(epoch_attrs.get("epoch_type", ""))
                        hed_tag_arr[i] = str(epoch_attrs.get("hed_tag", ""))
                        event_id_arr[i] = int(epoch_attrs.get("event_id", 0))
                        onset_sample_arr[i] = int(epoch_attrs.get("onset_sample", 0))
                        pre_event_samples_arr[i] = int(
                            epoch_attrs.get("pre_event_samples", 0)
                        )
                        length_arr[i] = int(epoch_attrs.get("length", 0))

                    dst.create_dataset(
                        "log_power", data=log_power_arr, dtype=np.float32
                    )
                    dst.create_dataset(
                        "hed_vector", data=hed_vector_arr, dtype=np.float32
                    )
                    dst.create_dataset("epoch_id", data=epoch_id_arr, dtype=str_dt)
                    dst.create_dataset(
                        "event_value", data=event_value_arr, dtype=str_dt
                    )
                    dst.create_dataset("event_type", data=event_type_arr, dtype=str_dt)
                    dst.create_dataset("epoch_type", data=epoch_type_arr, dtype=str_dt)
                    dst.create_dataset("hed_tag", data=hed_tag_arr, dtype=str_dt)
                    dst.create_dataset("event_id", data=event_id_arr)
                    dst.create_dataset("onset_sample", data=onset_sample_arr)
                    dst.create_dataset("pre_event_samples", data=pre_event_samples_arr)
                    dst.create_dataset("length", data=length_arr)
                    stats["n_written"] = n
                else:
                    raise ValueError(
                        f"output_schema={output_schema!r} not supported; "
                        "use 'groups' or 'contiguous'."
                    )

            os.replace(tmp_path, out_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    except Exception as exc:
        stats["status"] = f"error: {exc}"
        logger.error("Error processing %s: %s", h5_path, exc)

    return stats


def extract_tf_features(
    source_dir: Path,
    output_dir: Path,
    freqs: list[float],
    n_cycles: int | np.ndarray,
    n_workers: int,
    limit: int | None,
    overwrite: bool,
    target_sfreq: float = TARGET_SFREQ,
    task_filter: str = "movie",
    expected_epoch_len: int | None = None,
    output_schema: str = "groups",
) -> None:
    """Run Morlet TF extraction on all files matching the task filter.

    Args:
        source_dir: Root of / preprocessed directory.
        output_dir: Output directory for TF features.
        freqs: Center frequencies in Hz.
        n_cycles: Wavelet width. Scalar or per-freq float array.
        n_workers: Number of parallel workers.
        limit: If set, process only the first N files (for smoke-testing).
        overwrite: If True, reprocess existing output files.
        target_sfreq: Target sampling frequency after strided-mean downsampling.
            Must evenly divide SOURCE_SFREQ.
        task_filter: "movie" (passive-movie tonic, length=220) or "non-movie"
            (event-bearing tasks, stim+response only). "all" is rejected to
            avoid silent mixed-length output.
        expected_epoch_len: If set, override the task-filter default. Required
            for   non-movie inputs (160 samples) since the default
            assumes  1.0 s = 100 samples.
        output_schema: "groups" (per-epoch h5 groups,  backwards compat)
            or "contiguous" (single (n, F, C, T) dataset,   default).
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if target_sfreq <= 0 or SOURCE_SFREQ % target_sfreq != 0:
        raise ValueError(
            f"target_sfreq={target_sfreq} must be positive and evenly divide "
            f"source_sfreq={SOURCE_SFREQ}"
        )

    if task_filter == "movie":
        predicate = _is_passive_movie
        default_epoch_len = MOVIE_EPOCH_LEN
        allowed_epoch_types: frozenset[str] | None = None
        task_label = "passive-movie"
        expected_task_set = PASSIVE_MOVIE_TASKS
    elif task_filter == "non-movie":
        predicate = _is_non_movie_event_task
        default_epoch_len = NON_MOVIE_EPOCH_LEN
        allowed_epoch_types = NON_MOVIE_EPOCH_TYPES
        task_label = "non-movie event"
        expected_task_set = NON_MOVIE_EVENT_TASKS
    elif task_filter == "erp-core":
        predicate = _is_erp_core_task
        default_epoch_len = NON_MOVIE_EPOCH_LEN
        allowed_epoch_types = NON_MOVIE_EPOCH_TYPES
        task_label = "erp-core paradigm"
        expected_task_set = ERP_CORE_TASKS
    else:
        raise ValueError(
            f"task_filter={task_filter!r} not supported. Use 'movie', "
            "'non-movie', or 'erp-core'. 'all' is rejected to prevent "
            "mixed-length output (downstream code assumes uniform T per "
            "directory)."
        )

    if expected_epoch_len is None:
        expected_epoch_len = default_epoch_len
    elif expected_epoch_len <= 0:
        raise ValueError(
            f"expected_epoch_len must be positive, got {expected_epoch_len}."
        )

    if output_schema not in ("groups", "contiguous"):
        raise ValueError(
            f"output_schema={output_schema!r} not supported; "
            "use 'groups' or 'contiguous'."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover matching H5 files (ignore hed_vectorizer.pt and non-h5 files).
    all_h5 = sorted(p for p in source_dir.glob("*.h5") if p.is_file())
    matched_files = [f for f in all_h5 if predicate(f)]

    if not matched_files:
        raise RuntimeError(
            f"No {task_label} H5 files found in {source_dir}. "
            f"Expected files containing task names: {sorted(expected_task_set)}"
        )

    if limit is not None:
        matched_files = matched_files[:limit]
        logger.info("--limit %d: processing %d files", limit, len(matched_files))

    logger.info("Found %d %s files to process", len(matched_files), task_label)

    freqs_arr = np.array(freqs, dtype=np.float64)
    decimate_factor = int(SOURCE_SFREQ / target_sfreq)

    worker_fn = partial(
        _process_file,
        output_dir=output_dir,
        freqs=freqs_arr,
        n_cycles=n_cycles,
        decimate_factor=decimate_factor,
        target_sfreq=target_sfreq,
        overwrite=overwrite,
        expected_epoch_len=expected_epoch_len,
        allowed_epoch_types=allowed_epoch_types,
        task_filter=task_filter,
        output_schema=output_schema,
    )

    t0 = time.perf_counter()

    if n_workers == 1:
        results = [worker_fn(f) for f in matched_files]
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(worker_fn, matched_files)

    elapsed = time.perf_counter() - t0

    # Aggregate stats.
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_exists = sum(1 for r in results if r["status"] == "already_exists")
    n_errors = sum(1 for r in results if r["status"].startswith("error"))
    n_no_epochs = sum(1 for r in results if r["status"] == "no_valid_epochs")
    total_trials = sum(r.get("n_written", 0) for r in results)
    total_no_hed = sum(r.get("n_skipped_no_hed", 0) for r in results)
    total_wrong_len = sum(r.get("n_skipped_wrong_len", 0) for r in results)
    total_wrong_type = sum(r.get("n_skipped_wrong_type", 0) for r in results)

    # Disk usage.
    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.h5"))
    total_gb = total_bytes / 1024**3

    print("\n===  Morlet extraction: coverage report ===")
    print(f"  Task filter:     {task_filter} ({task_label})")
    print(f"  Source dir:      {source_dir}")
    print(f"  Output dir:      {output_dir}")
    print(f"  Epoch length:    {expected_epoch_len} samples")
    print(
        f"  Epoch types:     "
        f"{sorted(allowed_epoch_types) if allowed_epoch_types else 'all'}"
    )
    print(f"  Frequencies:     {freqs} Hz")
    print(f"  n_cycles:        {n_cycles}")
    print(f"  Target sfreq:    {target_sfreq} Hz")
    print(f"  Files processed: {len(matched_files)}")
    print(f"    OK:            {n_ok}")
    print(f"    Already exist: {n_exists}")
    print(f"    No valid eps:  {n_no_epochs}")
    print(f"    Errors:        {n_errors}")
    print(f"  Total trials:    {total_trials}")
    print(f"  Skipped no-HED:  {total_no_hed}")
    print(f"  Skipped bad-len: {total_wrong_len}")
    print(f"  Skipped bad-type:{total_wrong_type}")
    print(f"  Disk usage:      {total_gb:.2f} GB")
    print(f"  Elapsed:         {elapsed:.1f} s")

    if n_errors > 0:
        error_files = [r["path"] for r in results if r["status"].startswith("error")]
        logger.error("Errors in %d files: %s", n_errors, error_files[:5])
        raise RuntimeError(f"{n_errors} files failed extraction. See log for details.")


def _resolve_target_sfreq(
    *,
    target_sfreq: float | None,
    n_time: int | None,
    task_filter: str,
    expected_epoch_len: int | None = None,
) -> float:
    """Resolve the canonical ``target_sfreq`` from CLI flags.

    Either ``--target-sfreq`` (post-decimation rate) or ``--n-time``
    (time-bin count per epoch) sets the temporal resolution. They are
    mutually exclusive; passing neither falls back to the historical
    default (``TARGET_SFREQ`` = 10 Hz, matching  D.1 / D.2 runs).

    When ``--n-time`` is set, ``decimate_factor = epoch_len // n_time``
    must be an integer and ``target_sfreq = SOURCE_SFREQ /
    decimate_factor``. The shape contract feeds the ``expected_n_time``
    arg of ``neural_vocabulary.data.packed_ssl_dataset.PackedSSLDataset``.

    ``expected_epoch_len`` overrides the task-filter default (required for
      non-movie inputs at 160 samples).
    """
    if target_sfreq is not None and n_time is not None:
        raise ValueError(
            "--target-sfreq and --n-time are mutually exclusive. Pass one."
        )
    if target_sfreq is not None:
        return target_sfreq
    if n_time is None:
        return TARGET_SFREQ

    if expected_epoch_len is not None:
        epoch_len = expected_epoch_len
    elif task_filter == "movie":
        epoch_len = MOVIE_EPOCH_LEN
    elif task_filter in ("non-movie", "erp-core"):
        epoch_len = NON_MOVIE_EPOCH_LEN
    else:
        raise ValueError(
            f"task_filter={task_filter!r} not supported by --n-time. Use "
            "'movie', 'non-movie', or 'erp-core'."
        )
    if n_time <= 0:
        raise ValueError(f"--n-time must be positive, got {n_time}.")
    if epoch_len % n_time != 0:
        raise ValueError(
            f"--n-time={n_time} must evenly divide epoch_len={epoch_len} "
            f"(samples) for task_filter={task_filter!r}. Pick a divisor."
        )
    decimate_factor = epoch_len // n_time
    return SOURCE_SFREQ / decimate_factor


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=" Gate A.1: Morlet TF feature extractor for HBN passive-movie epochs."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("${HBN_DATA_DIR}/preprocessed"),
        help="Root of preprocessed preprocessed H5 directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("${HBN_DATA_DIR}/v10_gate_a_features"),
        help="Output directory for TF features (mirrored layout).",
    )
    parser.add_argument(
        "--freqs",
        type=float,
        nargs="+",
        default=DEFAULT_FREQS,
        help="Center frequencies in Hz. Default: %(default)s",
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=DEFAULT_N_CYCLES,
        help="Fixed wavelet width (cycles), scalar across freqs. "
        "Default: %(default)s. Mutually exclusive with --n-cycles-list.",
    )
    parser.add_argument(
        "--n-cycles-list",
        type=str,
        default=None,
        help="Per-frequency wavelet width as a comma-separated float list, "
        "matching --freqs length (e.g. '2,2,3,4.5,6.5,7,7,7' for ). "
        "Mutually exclusive with --n-cycles.",
    )
    parser.add_argument(
        "--target-sfreq",
        type=float,
        default=None,
        help="Target sampling frequency after downsampling. Mutually "
        "exclusive with --n-time. Default: 10 Hz (HBN historical) when "
        "neither flag is set.",
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=None,
        help="Number of time bins per epoch in the output TF tensor. "
        "Mutually exclusive with --target-sfreq. Convenience knob: "
        "epoch_len // n_time must be an integer (e.g. for a 1.0 s "
        "non-movie / ERP-CORE epoch (100 samples at 100 Hz), n_time=50 "
        "yields decimate_factor=2 and target_sfreq=50 Hz). Use when "
        "downstream care is the temporal resolution rather than the "
        "post-decimation rate.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes. Default: %(default)s",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files (for smoke-testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files even if output already exists.",
    )
    parser.add_argument(
        "--task-filter",
        choices=["movie", "non-movie", "erp-core"],
        default="movie",
        help=(
            "movie: HBN passive-movie tonic epochs (length=220, default, "
            "preserves Gate A/B/C behavior). non-movie: HBN event-bearing "
            "tasks, stim+response only (length=100, for the HED-objective ablation non-movie "
            "SSL pretraining). erp-core: ERP-CORE paradigms (Phase D.2.0; "
            "length=100, stim+response). RestingState is excluded everywhere."
        ),
    )
    parser.add_argument(
        "--expected-epoch-len",
        type=int,
        default=None,
        help="Override the task-filter default epoch length (samples). "
        "Required for   non-movie at 160 samples (1.6 s @ 100 Hz). "
        "Default: 220 (movie) or 100 (non-movie / erp-core).",
    )
    parser.add_argument(
        "--output-schema",
        choices=["groups", "contiguous"],
        default="groups",
        help="H5 output layout. groups: per-epoch h5 groups ( default, "
        "backwards compat). contiguous: single (n_epochs, F, C, T) dataset "
        "(  default — required for the new I/O budget).",
    )
    args = parser.parse_args()

    if args.n_cycles_list is not None:
        try:
            n_cycles_resolved: int | np.ndarray = parse_n_cycles_list_arg(
                args.n_cycles_list, len(args.freqs)
            )
        except ValueError as e:
            parser.error(str(e))
    else:
        n_cycles_resolved = args.n_cycles

    target_sfreq = _resolve_target_sfreq(
        target_sfreq=args.target_sfreq,
        n_time=args.n_time,
        task_filter=args.task_filter,
        expected_epoch_len=args.expected_epoch_len,
    )

    extract_tf_features(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        freqs=args.freqs,
        n_cycles=n_cycles_resolved,
        n_workers=args.n_workers,
        limit=args.limit,
        overwrite=args.overwrite,
        target_sfreq=target_sfreq,
        task_filter=args.task_filter,
        expected_epoch_len=args.expected_epoch_len,
        output_schema=args.output_schema,
    )


if __name__ == "__main__":
    main()
