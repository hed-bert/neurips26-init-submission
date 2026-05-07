"""Preprocess HBN L100 BDF data in parallel.

Reads BDF files, filters, re-references, interpolates 129->64 channels,
optionally denoises with SPEED-EEG, epochs around events, and saves
preprocessed HDF5 files. Runs across all CPU cores. Training then reads
from cache with zero preprocessing.

Usage:
    uv run python neural_vocabulary/scripts/preprocess_hbn.py \
        --data-root /mnt/local/HBN_L100 \
        --output-dir /mnt/local/HBN_L100/preprocessed \
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from neural_vocabulary.data.hed_assembly import vectorize_hed_string

# Backward-compat alias for ProcessPoolExecutor workers + existing tests
# that import the leading-underscore name. New code should import
# ``vectorize_hed_string`` from ``neural_vocabulary.data.hed_assembly``
# directly.
_vectorize_hed_string = vectorize_hed_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# HBN 129-channel GSN HydroCel montage -> standard 64-channel 10-10
# This mapping is from our ChannelHarmonization module
SFREQ = 100.0
HIGH_PASS_HZ = 0.1
LOW_PASS_HZ = 40.0
FILTER_ORDER = 4

# SPEED-EEG denoising constants
SPEED_WINDOW_S = 3.0  # SPEED model expects 3-second windows
SPEED_NEW_SRATE = 200.0  # SPEED resamples to 200 Hz internally

# V9 Tier 1: fixed condition-matched epoch windows (pre_s, post_s).
#   stim      : stimulus-locked (visual, auditory, trial onsets)
#   response  : response-locked (button presses)
#   tonic     : long-scale streaming events (movie shots, video_start,
#               resting-state markers, block/session markers)
# Windows intentionally wider than typical ERP to preserve late components
# without leaking duration as task-identity shortcut (same length per type).
V9_WINDOW_SPEC: dict[str, tuple[float, float]] = {
    "stim": (0.2, 0.8),
    "response": (0.5, 0.5),
    "tonic": (0.2, 2.0),
}

# V11 R4-A: wider window with 300 ms pre-stim baseline. Used for the
# R4 design freeze; preserves 2 Hz Morlet wavelet support that the V9
# 200 ms pre-stim cannot. See .context/r4_design_freeze_proposal.md §3.1.
# Baseline correction always uses the full pre-stim window per
# epoch_data(); for R4-A that is exactly 300 ms.
V11_R4A_WINDOW_SPEC: dict[str, tuple[float, float]] = {
    "stim": (0.3, 1.3),
    "response": (0.3, 1.3),
    "tonic": (0.3, 2.0),
}


def parse_window_spec_arg(
    raw: str, *, required_keys: tuple[str, ...] = ("stim", "response", "tonic")
) -> dict[str, tuple[float, float]]:
    """Parse --window-spec JSON into a validated dict[str, tuple[float, float]].

    Raises ValueError on invalid JSON, missing required keys, non-list value,
    wrong-length value, or non-positive (pre, post). Callers (CLI parsers)
    should translate the ValueError into ``parser.error`` for a clean exit.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"--window-spec must be valid JSON: {e}") from None
    if not isinstance(parsed, dict):
        raise ValueError(
            f"--window-spec must be a JSON object; got {type(parsed).__name__}"
        )
    spec: dict[str, tuple[float, float]] = {}
    for key, value in parsed.items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(
                f"--window-spec[{key!r}] must be a 2-element list "
                f"[pre_s, post_s]; got {value!r}"
            )
        try:
            pre_s, post_s = float(value[0]), float(value[1])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"--window-spec[{key!r}] entries must be numeric: {e}"
            ) from None
        if pre_s <= 0 or post_s <= 0:
            raise ValueError(
                f"--window-spec[{key!r}] must have positive (pre, post); "
                f"got ({pre_s}, {post_s})"
            )
        spec[key] = (pre_s, post_s)
    missing = [k for k in required_keys if k not in spec]
    if missing:
        raise ValueError(
            f"--window-spec missing required epoch types {missing!r}; "
            f"got {sorted(spec)}"
        )
    return spec


# Lowercase substring matches for classifying HBN event values. Order:
# response first (buttonPress sometimes follows a _target marker in the
# same row), then tonic (long/streaming/protocol markers), default stim.
_RESPONSE_MARKERS = ("buttonpress", "_response", "response_")

# Tonic markers are explicit to avoid swallowing stim events such as
# `contrastTrial_start` or `dot_no1_ON`. Block-start markers are named
# precisely (surroundSuppB*, contrastChangeB*, learningBlock_*) instead of
# a broad `_start` rule.
_TONIC_MARKERS = (
    "video_start",
    "video_stop",
    "resting_start",
    "resting_stop",
    "instructed_",  # instructed_toOpenEyes etc: state transitions, not stim
    "boundary",  # BIDS data-boundary markers AND HBN shot_boundary
    "break",  # between-run breaks
    "surroundsuppb",  # surroundSuppB1_start etc.
    "contrastchangeb",  # contrastChangeB1_start etc.
    "learningblock",  # learningBlock_1 etc.
)

# Junk event values that should be skipped entirely.
_SKIP_VALUES = frozenset({"9999", "value", "n/a", "nan", ""})


def classify_epoch_type(event_value: str) -> str | None:
    """Classify an HBN event value into stim / response / tonic.

    Returns None for junk values that should be skipped (see _SKIP_VALUES).
    Used by V9 Tier 1 preprocessing to select a fixed time window per
    epoch (see V9_WINDOW_SPEC). Any value falling through to the default
    is treated as stimulus-locked.
    """
    if event_value in _SKIP_VALUES:
        return None
    v = event_value.lower()
    if any(m in v for m in _RESPONSE_MARKERS):
        return "response"
    if any(m in v for m in _TONIC_MARKERS):
        return "tonic"
    return "stim"


def read_bdf(bdf_path: Path) -> tuple[np.ndarray, float, list[str]]:
    """Read a BDF file using pyedflib. Returns (data, sfreq, ch_names)."""
    import pyedflib

    f = pyedflib.EdfReader(str(bdf_path))
    try:
        n_channels = f.signals_in_file
        sfreq = f.getSampleFrequency(0)
        ch_names = f.getSignalLabels()

        # Read all channels
        data = np.zeros((n_channels, f.getNSamples()[0]))
        for i in range(n_channels):
            data[i] = f.readSignal(i)
    finally:
        f.close()

    return data, sfreq, list(ch_names)


def highpass_filter(data: np.ndarray, sfreq: float, cutoff: float) -> np.ndarray:
    """Apply zero-phase Butterworth high-pass filter."""
    sos = butter(FILTER_ORDER, cutoff, btype="high", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def lowpass_filter(data: np.ndarray, sfreq: float, cutoff: float) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter (also serves as anti-alias)."""
    # Guard against cutoff >= Nyquist (scipy rejects these).
    nyquist = sfreq / 2.0
    safe_cutoff = min(cutoff, nyquist * 0.95)
    sos = butter(FILTER_ORDER, safe_cutoff, btype="low", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def resample_to(
    data: np.ndarray, sfreq_in: float, sfreq_out: float
) -> tuple[np.ndarray, float]:
    """Polyphase resample continuous EEG. Returns (resampled, sfreq_out).

    No-op when sfreq_in == sfreq_out. Uses scipy.signal.resample_poly which
    applies an internal anti-alias filter, so callers are free to skip a
    separate LP when downsampling (but a dedicated LP at the target band
    is still recommended to pin the frequency cutoff).
    """
    if abs(sfreq_in - sfreq_out) < 1e-6:
        return data.astype(np.float32), sfreq_in
    from math import gcd

    from scipy.signal import resample_poly

    # Reduce ratio to smallest integer up/down factors
    ratio_num = int(round(sfreq_out))
    ratio_den = int(round(sfreq_in))
    g = gcd(ratio_num, ratio_den)
    up, down = ratio_num // g, ratio_den // g
    resampled = resample_poly(data, up, down, axis=-1).astype(np.float32)
    return resampled, float(sfreq_out)


def average_rereference(data: np.ndarray) -> np.ndarray:
    """Subtract channel mean (average reference)."""
    return (data - data.mean(axis=0, keepdims=True)).astype(np.float32)


def denoise_continuous(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    checkpoint_path: str,
    device: str = "cpu",
    amp: bool | None = None,
) -> tuple[np.ndarray, float, list[str], bool]:
    """Apply SPEED-EEG denoising to continuous EEG data.

    Passes the full continuous recording to SPEED, which internally
    epochs into 3-second windows, denoises, and returns continuous data.
    We then do our own event-based epoching on the denoised output.

    Parameters
    ----------
    data : ndarray
        Continuous EEG, shape ``(channels, samples)``.
    sfreq : float
        Input sampling rate in Hz.
    ch_names : list[str]
        Channel names matching data rows.
    checkpoint_path : str
        Path to SPEED model checkpoint.
    device : str
        Torch device for inference.
    amp : bool or None
        Enable mixed-precision (fp16) inference. Passed to SPEED.

    Returns
    -------
    denoised_data : ndarray
        Denoised continuous data at SPEED output rate (200 Hz).
    out_sfreq : float
        Output sampling rate (200 Hz).
    out_ch_names : list[str]
        Channel names after denoising (may differ if channels were rejected).
    denoised : bool
        True if SPEED denoising was actually applied.
    """
    from speed_eeg import denoise

    duration_s = data.shape[1] / sfreq
    if duration_s < SPEED_WINDOW_S:
        logger.warning(
            "Recording too short for SPEED (%.1fs < %.1fs)",
            duration_s,
            SPEED_WINDOW_S,
        )
        return data, sfreq, ch_names, False

    n_epochs_expected = int(duration_s // SPEED_WINDOW_S)
    logger.info(
        "  SPEED denoising: %.1fs continuous (%d internal 3s epochs, %d ch at %.0f Hz)",
        duration_s,
        n_epochs_expected,
        data.shape[0],
        sfreq,
    )

    # Use a wide passband (0.1 Hz to near-Nyquist) so SPEED's filter is
    # effectively a no-op on our already-highpassed data.
    nyquist = sfreq / 2.0
    safe_highcut = min(75.0, nyquist - 1.0)
    # Disable 60 Hz notch when Nyquist is below 60 Hz (e.g., 100 Hz data)
    notch_filter = (59.0, 61.0) if nyquist > 61.0 else None
    result = denoise(
        raw_data=data,
        srate=sfreq,
        channel_names=ch_names,
        bandpass=(0.1, safe_highcut),
        notch=notch_filter,
        reject_by_kurtosis=False,
        reject_by_correlation=False,
        reject_by_jointprob=False,
        reref=None,
        new_srate=SPEED_NEW_SRATE,
        checkpoint_path=checkpoint_path,
        device=device,
        amp=amp,
        return_epoched=False,
        verbose=False,
    )

    denoised = result.denoised
    nan_frac = np.isnan(denoised).mean()
    if nan_frac > 0:
        raise ValueError(
            f"SPEED output contains {nan_frac:.1%} NaN values; falling back to undenoised."
        )
    return denoised, result.srate, result.channel_names, True


def build_interpolation_matrix(
    source_channels: list[str],
    target_channels: list[str],
    montage_path: Path | None = None,
) -> np.ndarray:
    """Build 129->64 channel interpolation matrix.

    Uses nearest-neighbor identity mapping for matching channels,
    and inverse-distance weighting for unmatched targets.
    """
    from neural_vocabulary.models.channel_harmonization import ChannelHarmonization

    harm = ChannelHarmonization.for_hbn_eeg()
    return harm.matrix.numpy()


def parse_events_tsv(events_path: Path) -> pd.DataFrame:
    """Parse a BIDS events.tsv file."""
    if not events_path.exists():
        return pd.DataFrame()
    return pd.read_csv(events_path, sep="\t")


# Tasks that use movie/video stimuli with shot boundary annotations
MOVIE_TASKS = frozenset(
    {
        "DespicableMe",
        "DiaryOfAWimpyKid",
        "FunwithFractals",
        "ThePresent",
    }
)

# HED tag for shot boundary events (matches detect_shot_boundaries.py)
_SHOT_BOUNDARY_HED = "(Sensory-event, Experimental-stimulus, Visual-presentation)"

# Default location of stimulus events (generated by detect_shot_boundaries.py)
_STIMULUS_EVENTS_DIR = Path(__file__).parent.parent / "data" / "stimulus_events"


def load_stimulus_events(
    task_name: str, stimulus_dir: Path | None = None
) -> pd.DataFrame | None:
    """Load stimulus-level shot boundary events for a movie task.

    Parameters
    ----------
    task_name : str
        HBN task name (e.g., "DespicableMe").
    stimulus_dir : Path | None
        Directory containing stimulus event files. Defaults to
        neural_vocabulary/data/stimulus_events/.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with onset, duration, value columns, or None if not found.
    """
    if stimulus_dir is None:
        stimulus_dir = _STIMULUS_EVENTS_DIR
    tsv_path = stimulus_dir / f"stim-{task_name}_annot-shotboundary_events.tsv"
    if not tsv_path.exists():
        return None
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except Exception as e:
        logger.error("Failed to read stimulus events %s: %s", tsv_path, e)
        return None
    required = {"onset", "duration", "value"}
    missing = required - set(df.columns)
    if missing:
        logger.error("Stimulus events %s missing columns: %s", tsv_path, missing)
        return None
    return df


def merge_stimulus_events(
    events_df: pd.DataFrame,
    task_name: str,
    stimulus_dir: Path | None = None,
) -> pd.DataFrame:
    """Merge stimulus-level shot boundaries into subject events.

    For movie tasks, loads the stimulus annotation file, offsets shot
    boundary onsets by the subject's video_start time, and appends them
    to the subject events.

    Parameters
    ----------
    events_df : pd.DataFrame
        Subject-level events with onset, duration, value columns.
    task_name : str
        HBN task name extracted from the recording filename.
    stimulus_dir : Path | None
        Directory containing stimulus event files.

    Returns
    -------
    pd.DataFrame
        Merged events sorted by onset. Original events are unchanged.
    """
    if task_name not in MOVIE_TASKS:
        return events_df

    stim_events = load_stimulus_events(task_name, stimulus_dir)
    if stim_events is None:
        logger.debug("No stimulus events for task %s", task_name)
        return events_df

    # Find subject's video_start onset to offset stimulus times
    value_col = events_df.get("value", events_df.get("trial_type"))
    if value_col is None:
        logger.warning(
            "Events for task %s have neither 'value' nor 'trial_type' column; "
            "skipping stimulus merge",
            task_name,
        )
        return events_df

    video_starts = events_df[value_col == "video_start"]
    if video_starts.empty:
        logger.warning(
            "No video_start event for task %s; skipping stimulus merge", task_name
        )
        return events_df

    video_start_onset = video_starts.iloc[0]["onset"]

    # Filter to shot_boundary events only (video_start/stop already in subject events)
    shot_boundaries = stim_events[stim_events["value"] == "shot_boundary"].copy()
    if shot_boundaries.empty:
        return events_df

    # Offset by subject's video_start time
    shot_boundaries["onset"] = shot_boundaries["onset"] + video_start_onset

    # Keep HED column from stimulus events (per-shot annotations)
    # plus all columns shared with subject events
    keep_cols = [c for c in events_df.columns if c in shot_boundaries.columns]
    if "HED" in shot_boundaries.columns and "HED" not in keep_cols:
        keep_cols.append("HED")
    shot_boundaries = shot_boundaries[keep_cols]

    # Fill missing columns with n/a
    for col in events_df.columns:
        if col not in shot_boundaries.columns:
            shot_boundaries[col] = "n/a"

    merged = pd.concat([events_df, shot_boundaries], ignore_index=True)
    merged = merged.sort_values("onset").reset_index(drop=True)

    logger.info(
        "  Merged %d shot boundaries for task %s (video_start=%.2fs)",
        len(shot_boundaries),
        task_name,
        video_start_onset,
    )
    return merged


def epoch_data(
    data: np.ndarray,
    events_df: pd.DataFrame,
    sfreq: float,
    window_spec: dict[str, tuple[float, float]] = V9_WINDOW_SPEC,
) -> list[dict]:
    """Extract fixed-length condition-matched epochs with baseline correction.

    Windows are set per epoch type (stim/response/tonic) from window_spec:
        window_spec[type] = (pre_seconds, post_seconds)

    For each event:
      1. Classify by value -> stim | response | tonic
      2. Extract window = [onset - pre_s, onset + post_s], zero-padding any
         portion that falls outside the continuous recording
      3. Subtract the pre-stimulus baseline (mean over the first pre_samples
         of valid data per channel)
      4. Record a pad_mask marking which samples are valid (1) vs padded (0)

    All epochs of the same type have identical length (pre+post samples),
    eliminating the duration-as-shortcut pathway that variable-length
    epochs exhibited in V7/V8.

    Returns list of dicts with keys: eeg, pad_mask, epoch_type, event_value,
    onset_sample, pre_event_samples, length.
    """
    if events_df.empty or "onset" not in events_df.columns:
        return []

    n_samples = data.shape[-1]
    n_channels = data.shape[0]

    onsets = events_df["onset"].values
    values = events_df.get(
        "value", events_df.get("trial_type", pd.Series(["unknown"] * len(events_df)))
    ).values

    # Precompute sample counts per epoch type.
    window_samples: dict[str, tuple[int, int, int]] = {}
    for etype, (pre_s, post_s) in window_spec.items():
        pre_n = int(round(pre_s * sfreq))
        post_n = int(round(post_s * sfreq))
        window_samples[etype] = (pre_n, post_n, pre_n + post_n)

    epochs: list[dict] = []
    for i, onset in enumerate(onsets):
        onset_sample = int(round(onset * sfreq))
        event_value = str(values[i])
        epoch_type = classify_epoch_type(event_value)
        if epoch_type is None:
            continue  # junk marker ("9999", "value", etc.)
        pre_n, post_n, total_n = window_samples[epoch_type]

        # Ideal window; clamp to recording bounds and pad with zeros.
        ideal_start = onset_sample - pre_n
        ideal_end = onset_sample + post_n
        valid_start = max(0, ideal_start)
        valid_end = min(n_samples, ideal_end)
        if valid_end <= valid_start:
            continue  # event entirely outside recording

        write_start = valid_start - ideal_start  # offset into the padded buffer
        write_end = write_start + (valid_end - valid_start)

        epoch_eeg = np.zeros((n_channels, total_n), dtype=np.float32)
        epoch_eeg[:, write_start:write_end] = data[:, valid_start:valid_end]

        pad_mask = np.zeros(total_n, dtype=np.uint8)
        pad_mask[write_start:write_end] = 1

        # Per-channel pre-stim baseline subtraction. Use only valid samples
        # within the pre-stim window; if fully padded, fall back to 0 offset.
        baseline_end = pre_n  # exclusive
        baseline_valid = pad_mask[:baseline_end].astype(bool)
        if baseline_valid.any():
            baseline_mean = epoch_eeg[:, :baseline_end][:, baseline_valid].mean(
                axis=1, keepdims=True
            )
            # Only zero-out baseline for valid samples; leave pad region at 0
            # so the pad_mask still identifies real padding.
            epoch_eeg[:, pad_mask.astype(bool)] -= baseline_mean
        # else: all pre-stim padded -> no usable baseline, skip subtraction

        epochs.append(
            {
                "eeg": epoch_eeg,
                "pad_mask": pad_mask,
                "epoch_type": epoch_type,
                "event_value": event_value,
                "onset_sample": onset_sample,
                "pre_event_samples": pre_n,
                "length": total_n,
            }
        )

    return epochs


# _vectorize_hed_string is now a re-export at module top (V11 E3, #196).


def process_one_recording(
    bdf_path: Path,
    events_path: Path,
    output_path: Path,
    interp_matrix: np.ndarray | None,
    hed_sidecar_path: str | None,
    hed_vectorizer_data: tuple | None = None,
    speed_checkpoint: str | None = None,
    speed_device: str = "cpu",
    speed_amp: bool | None = None,
    window_spec: dict[str, tuple[float, float]] = V9_WINDOW_SPEC,
    preprocess_version: str = "v9_tier1",
) -> int:
    """Process a single BDF recording end-to-end.

    Uses hedtools TabularInput + Sidecar for proper BIDS HED assembly
    (column substitution, Def resolution, placeholder replacement).

    Parameters
    ----------
    speed_checkpoint : str | None
        Path to SPEED-EEG model checkpoint. When provided, SPEED denoises
        the full continuous recording before epoching.
    speed_device : str
        Torch device for SPEED inference (e.g. "cpu", "cuda").
    speed_amp : bool | None
        Enable mixed-precision (fp16) SPEED inference on CUDA.

    Returns number of epochs saved.
    """

    # Read
    data, sfreq, ch_names = read_bdf(bdf_path)

    # Keep only EEG channels (exclude Status, EXG, etc.)
    eeg_indices = [
        i
        for i, name in enumerate(ch_names)
        if not any(
            x in name.upper()
            for x in ["STATUS", "EXG", "STI", "GSR", "RESP", "TEMP", "PLET"]
        )
    ]
    data = data[eeg_indices]
    eeg_ch_names = [ch_names[i] for i in eeg_indices]

    # High-pass filter (tight, to remove slow drifts without touching ERP).
    data = highpass_filter(data, sfreq, HIGH_PASS_HZ)

    # Re-reference
    data = average_rereference(data)

    # Cache the native sample rate; SPEED may resample to 200 Hz internally
    # and we want to restore the native rate before saving.
    native_sfreq = sfreq

    # Parse events early (needed for denoising window extraction)
    events_df = parse_events_tsv(events_path)
    if events_df.empty:
        return 0

    # Assemble HED tags using hedtools BIDS pipeline BEFORE merging
    # stimulus events (TabularInput needs the original events.tsv file).
    # We also expand Def/ references here while sidecar context is available,
    # so that _vectorize_hed_string receives fully-expanded strings with all
    # content tags (Visual-presentation, Movie, etc.) visible for vectorization.
    n_original = len(events_df)
    hed_per_row: list[str | None] = [None] * n_original
    if hed_sidecar_path and events_path.exists():
        try:
            from hed import HedString, Sidecar, TabularInput, load_schema_version
            from hed.models import DefinitionDict

            sidecar_obj = Sidecar(str(hed_sidecar_path))
            tabular = TabularInput(str(events_path), sidecar=sidecar_obj)

            # Build DefinitionDict from sidecar (mirrors HEDVectorizer logic)
            schema = load_schema_version("8.3.0")
            def_dict = DefinitionDict()
            with open(hed_sidecar_path) as _f:
                _sidecar_json = json.load(_f)
            for _hed_str in _sidecar_json.get("hed_defs", {}).get("HED", {}).values():
                def_dict.check_for_definitions(HedString(_hed_str, schema))
            for _hed_str in _sidecar_json.get("value", {}).get("HED", {}).values():
                def_dict.check_for_definitions(HedString(_hed_str, schema))

            for i, hs in enumerate(tabular.series_a):
                if hs is not None:
                    s = str(hs).strip()
                    if s:
                        # Re-parse with definitions and expand Def/ references
                        expanded = HedString(s, schema, def_dict)
                        expanded.expand_defs()
                        hed_per_row[i] = str(expanded)
        except Exception as e:
            logger.warning("HED assembly failed for %s: %s", events_path.name, e)

    # Merge stimulus-level shot boundaries for movie tasks
    task_name = extract_task_name(bdf_path)
    events_df = merge_stimulus_events(events_df, task_name)

    # Interpolate to 64 standard 10-10 channels BEFORE denoising.
    # SPEED-EEG expects standard channel names (Fp1, Fz, Cz, ...) but HBN
    # uses GSN HydroCel names (E1-E128 + Cz). Interpolating first gives
    # SPEED recognizable channel names.
    if interp_matrix is not None and data.shape[0] != interp_matrix.shape[1]:
        logger.warning(
            "%s: expected %d channels, got %d. Skipping interpolation.",
            bdf_path.name,
            interp_matrix.shape[1],
            data.shape[0],
        )
        return 0
    elif interp_matrix is not None:
        from neural_vocabulary.models.channel_harmonization import TARGET_64_CHANNELS

        data = (interp_matrix @ data).astype(np.float32)
        eeg_ch_names = list(TARGET_64_CHANNELS)

    # SPEED-EEG denoising (after channel interpolation, before epoching)
    denoising_succeeded = False
    if speed_checkpoint is not None:
        try:
            data, sfreq, eeg_ch_names, denoising_succeeded = denoise_continuous(
                data=data,
                sfreq=sfreq,
                ch_names=eeg_ch_names,
                checkpoint_path=speed_checkpoint,
                device=speed_device,
                amp=speed_amp,
            )
            logger.info(
                "  SPEED output: %d ch, %d samples at %.0f Hz",
                data.shape[0],
                data.shape[1],
                sfreq,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(
                "SPEED denoising failed for %s: %s. Using undenoised data.",
                bdf_path.name,
                e,
            )

    # Low-pass at 40 Hz before any downsampling (anti-alias + ERP band pin).
    if sfreq > 2 * LOW_PASS_HZ:
        data = lowpass_filter(data, sfreq, LOW_PASS_HZ)

    # Restore native sample rate (SPEED outputs at 200 Hz; HBN is 100 Hz).
    if abs(sfreq - native_sfreq) > 1e-6:
        data, sfreq = resample_to(data, sfreq, native_sfreq)
        logger.info("  Resampled to native %.0f Hz: %d samples", sfreq, data.shape[1])

    # Build onset->HED mapping from original events (using current sfreq,
    # which may have changed if SPEED denoising resampled the data)
    onset_to_hed: dict[int, str | None] = {}
    original_onsets = parse_events_tsv(events_path)
    if not original_onsets.empty:
        for i, onset in enumerate(original_onsets["onset"].values):
            onset_sample = int(round(onset * sfreq))
            if i < len(hed_per_row) and hed_per_row[i]:
                onset_to_hed[onset_sample] = hed_per_row[i]

    # Add HED tags for shot_boundary events.
    # Use per-shot HEDit annotations from the HED column if available,
    # otherwise fall back to the generic shot boundary tag.
    for _, row in events_df.iterrows():
        if str(row.get("value", "")) == "shot_boundary":
            onset_sample = int(round(row["onset"] * sfreq))
            if onset_sample not in onset_to_hed:
                per_shot_hed = row.get("HED", "")
                if per_shot_hed and str(per_shot_hed) not in ("", "nan", "n/a"):
                    onset_to_hed[onset_sample] = str(per_shot_hed)
                else:
                    onset_to_hed[onset_sample] = _SHOT_BOUNDARY_HED

    # Epoch (with merged events, using current sfreq)
    epochs = epoch_data(data, events_df, sfreq, window_spec=window_spec)
    if not epochs:
        return 0

    # Map each epoch to its HED tag via onset_sample
    epoch_hed_map: dict[int, str | None] = {}
    for ep_idx, ep in enumerate(epochs):
        epoch_hed_map[ep_idx] = onset_to_hed.get(ep["onset_sample"])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["n_epochs"] = len(epochs)
        f.attrs["sfreq"] = sfreq
        f.attrs["n_channels"] = data.shape[0]
        f.attrs["channel_names"] = json.dumps(eeg_ch_names[: data.shape[0]])
        f.attrs["preprocess_version"] = preprocess_version
        f.attrs["high_pass_hz"] = HIGH_PASS_HZ
        f.attrs["low_pass_hz"] = LOW_PASS_HZ
        f.attrs["window_spec"] = json.dumps(
            {k: list(v) for k, v in window_spec.items()}
        )
        f.attrs["baseline_correction"] = "pre_stim_mean"
        if denoising_succeeded:
            f.attrs["denoised"] = True
            f.attrs["denoise_method"] = "SPEED-EEG"

        for i, ep in enumerate(epochs):
            grp = f.create_group(f"epoch_{i:05d}")
            grp.create_dataset("eeg", data=ep["eeg"])
            grp.create_dataset("pad_mask", data=ep["pad_mask"])
            grp.attrs["event_id"] = hash(ep["event_value"]) % (2**31)
            grp.attrs["event_value"] = ep["event_value"]
            grp.attrs["epoch_type"] = ep["epoch_type"]
            grp.attrs["onset_sample"] = ep["onset_sample"]
            grp.attrs["pre_event_samples"] = ep["pre_event_samples"]
            grp.attrs["length"] = ep["length"]
            grp.attrs["duration_samples"] = ep["length"]

            hed_tag = epoch_hed_map.get(i)
            if hed_tag:
                grp.attrs["hed_tag"] = hed_tag

                # Pre-compute HED vector if vectorizer data provided
                if hed_vectorizer_data is not None:
                    tag_to_idx, vocab_size, tag_depths = hed_vectorizer_data
                    vec = _vectorize_hed_string(hed_tag, tag_to_idx, vocab_size)
                    grp.create_dataset("hed_vector", data=vec)

    return len(epochs)


def collect_recordings(data_root: Path) -> list[tuple[Path, Path, str, str]]:
    """Scan all releases for BDF files and their events.tsv files.

    Returns list of (bdf_path, events_path, release_name, subject_id).
    """
    recordings = []
    for release_dir in sorted(data_root.iterdir()):
        if not release_dir.is_dir() or not release_dir.name.endswith("_L100_bdf"):
            continue
        release_name = release_dir.name

        for sub_dir in sorted(release_dir.iterdir()):
            if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
                continue
            subject_id = sub_dir.name.replace("sub-", "")

            eeg_dir = sub_dir / "eeg"
            if not eeg_dir.exists():
                continue

            for bdf_file in sorted(eeg_dir.glob("*.bdf")):
                # Find matching events.tsv
                events_file = bdf_file.with_name(
                    bdf_file.name.replace("_eeg.bdf", "_events.tsv")
                )
                recordings.append((bdf_file, events_file, release_name, subject_id))

    return recordings


def find_hed_sidecar_paths(data_root: Path) -> dict[str, str]:
    """Find HED sidecar file paths per task. Returns {task_name: sidecar_path_str}."""
    sidecars: dict[str, str] = {}
    for sidecar_path in data_root.glob("*/task-*_events.json"):
        task_name = sidecar_path.stem.replace("_events", "").replace("task-", "")
        sidecars[task_name] = str(sidecar_path)
    # Also check root level
    for sidecar_path in data_root.glob("task-*_events.json"):
        task_name = sidecar_path.stem.replace("_events", "").replace("task-", "")
        sidecars[task_name] = str(sidecar_path)
    return sidecars


def extract_task_name(bdf_path: Path) -> str:
    """Extract task name from BIDS filename."""
    name = bdf_path.stem
    for part in name.split("_"):
        if part.startswith("task-"):
            return part.replace("task-", "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess HBN L100 BDF data")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing R*_L100_bdf/ releases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for preprocessed HDF5 (default: data-root/preprocessed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count // 2)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Limit number of subjects (for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process existing files (e.g. to update HED tags)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Only process these task names (e.g. DespicableMe ThePresent)",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply SPEED-EEG denoising before epoching",
    )
    parser.add_argument(
        "--speed-checkpoint",
        type=str,
        default=None,
        help="Path to SPEED-EEG model checkpoint (.pt file). "
        "Required when --denoise is used.",
    )
    parser.add_argument(
        "--speed-device",
        type=str,
        default="cpu",
        help="Device for SPEED inference (cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--speed-amp",
        action="store_true",
        help="Enable mixed-precision (fp16) SPEED inference on CUDA",
    )
    parser.add_argument(
        "--window-preset",
        choices=["v9_tier1", "v11_r4a"],
        default="v9_tier1",
        help="Named window-spec preset. v9_tier1 = stim:(0.2, 0.8) "
        "(default, V9 era 1.0 s windows). v11_r4a = stim:(0.3, 1.3) "
        "(R4 design freeze, 1.6 s window with 300 ms baseline). "
        "Used unless --window-spec explicitly overrides.",
    )
    parser.add_argument(
        "--window-spec",
        type=str,
        default=None,
        help='Override window_spec as JSON, e.g. \'{"stim":[0.3,1.3],'
        '"response":[0.3,1.3],"tonic":[0.3,2.0]}\'. Baseline mean is '
        "always computed over the full pre-stim window, so changing pre_s "
        "also changes baseline length. Overrides --window-preset.",
    )
    parser.add_argument(
        "--preprocess-version",
        type=str,
        default=None,
        help="Tag written to file attr 'preprocess_version'. Defaults to "
        "the chosen --window-preset value (v9_tier1 or v11_tier_r4).",
    )
    args = parser.parse_args()

    # Validate denoising arguments
    speed_checkpoint = None
    if args.denoise:
        if args.speed_checkpoint is None:
            parser.error("--speed-checkpoint is required when --denoise is used")
        ckpt = Path(args.speed_checkpoint)
        if not ckpt.is_file():
            parser.error(f"Checkpoint not found: {ckpt}")
        speed_checkpoint = str(ckpt)
        logger.info(
            "SPEED denoising enabled: checkpoint=%s, device=%s, amp=%s",
            speed_checkpoint,
            args.speed_device,
            args.speed_amp,
        )

    output_dir = args.output_dir or args.data_root / "preprocessed"
    n_workers = args.workers or max(1, (os.cpu_count() or 1) // 2)

    # Resolve window_spec: explicit JSON override > preset.
    if args.window_spec is not None:
        try:
            window_spec = parse_window_spec_arg(args.window_spec)
        except ValueError as e:
            parser.error(str(e))
        preset_label = "custom"
    elif args.window_preset == "v11_r4a":
        window_spec = V11_R4A_WINDOW_SPEC
        preset_label = "v11_r4a"
    else:
        window_spec = V9_WINDOW_SPEC
        preset_label = "v9_tier1"

    # Resolve preprocess_version tag.
    if args.preprocess_version is not None:
        preprocess_version = args.preprocess_version
    elif preset_label == "v11_r4a":
        preprocess_version = "v11_tier_r4"
    else:
        preprocess_version = "v9_tier1"

    logger.info(
        "Window spec: %s (preset=%s, preprocess_version=%s)",
        json.dumps({k: list(v) for k, v in window_spec.items()}),
        preset_label,
        preprocess_version,
    )

    # Collect recordings
    logger.info("Scanning %s for BDF recordings...", args.data_root)
    recordings = collect_recordings(args.data_root)
    logger.info("Found %d recordings", len(recordings))

    # Filter by task if requested
    if args.tasks:
        task_set = set(args.tasks)
        recordings = [
            (bdf, ev, rel, subj)
            for bdf, ev, rel, subj in recordings
            if extract_task_name(bdf) in task_set
        ]
        logger.info("Filtered to tasks %s: %d recordings", args.tasks, len(recordings))

    # Limit subjects if requested
    if args.max_subjects:
        seen_subjects: set[str] = set()
        filtered = []
        for bdf, events, release, subject in recordings:
            if subject not in seen_subjects:
                if len(seen_subjects) >= args.max_subjects:
                    continue
                seen_subjects.add(subject)
            filtered.append((bdf, events, release, subject))
        recordings = filtered
        logger.info(
            "Limited to %d subjects (%d recordings)",
            len(seen_subjects),
            len(recordings),
        )

    # Load HED sidecars
    hed_sidecar_paths = find_hed_sidecar_paths(args.data_root)
    logger.info("Found HED sidecars for %d tasks", len(hed_sidecar_paths))

    # Build interpolation matrix (once, in main process)
    # We'll pass None and let each worker skip interpolation for now
    # since the matrix depends on the channel montage per recording
    interp_matrix = None
    try:
        from neural_vocabulary.models.channel_harmonization import ChannelHarmonization

        harm = ChannelHarmonization.for_hbn_eeg()
        interp_matrix = harm.matrix.numpy()
        logger.info(
            "Interpolation matrix: %s -> %d channels",
            interp_matrix.shape,
            interp_matrix.shape[0],
        )
    except Exception as e:
        logger.warning("Could not build interpolation matrix: %s", e)

    # Build HED vectorizer data (picklable tuple for workers)
    hed_vectorizer_data = None
    try:
        from neural_vocabulary.data.hed_vectorizer import HEDVectorizer

        # Collect ALL sidecar files from ALL releases (not just unique task names)
        all_sidecar_files = []
        for release_dir in sorted(args.data_root.iterdir()):
            if release_dir.is_dir() and release_dir.name.endswith("_L100_bdf"):
                all_sidecar_files.extend(release_dir.glob("task-*_events.json"))
        if not all_sidecar_files:
            all_sidecar_files = [Path(p) for p in hed_sidecar_paths.values()]
        # Collect ALL HED strings: sidecar definitions + stimulus event annotations
        vectorizer = HEDVectorizer(schema_version="8.3.0")
        all_hed_strings = []

        # 1) Load definitions and HED strings from sidecars
        for sidecar_path in all_sidecar_files:
            sidecar_path = Path(sidecar_path)
            vectorizer.load_definitions_from_sidecar(sidecar_path)
            with open(sidecar_path) as f:
                sidecar = json.load(f)
            value_hed = sidecar.get("value", {}).get("HED", {})
            all_hed_strings.extend(value_hed.values())

        # 2) Add per-shot HED strings from stimulus event TSVs (HEDit annotations)
        n_stim_strings = 0
        for stim_tsv in _STIMULUS_EVENTS_DIR.glob("stim-*_events.tsv"):
            try:
                stim_df = pd.read_csv(stim_tsv, sep="\t")
                if "HED" in stim_df.columns:
                    for hed_val in stim_df["HED"].dropna():
                        hed_str = str(hed_val).strip()
                        if hed_str and hed_str != "n/a":
                            all_hed_strings.append(hed_str)
                            n_stim_strings += 1
            except Exception as e:
                logger.warning("Failed to parse stimulus HED TSV %s: %s", stim_tsv, e)
                continue

        logger.info(
            "Building vocabulary from %d sidecar + %d stimulus HED strings",
            len(all_hed_strings) - n_stim_strings,
            n_stim_strings,
        )
        vectorizer.build_vocabulary(all_hed_strings)

        # Save vectorizer for training to use the same vocabulary
        vectorizer.save(output_dir / "hed_vectorizer.pt")
        hed_vectorizer_data = (
            vectorizer.tag_to_idx,
            vectorizer.vocab_size,
            vectorizer.tag_depths,
        )
        logger.info(
            "HED vectorizer: %d tags, will pre-compute vectors", vectorizer.vocab_size
        )
    except Exception as e:
        logger.warning("Could not build HED vectorizer: %s", e)

    # Process in parallel
    logger.info("Processing with %d workers...", n_workers)
    start = time.monotonic()

    to_process = []
    for bdf_path, events_path, _release, subject in recordings:
        task = extract_task_name(bdf_path)
        # Include run number in output filename to avoid collisions
        run = ""
        for part in bdf_path.stem.split("_"):
            if part.startswith("run-"):
                run = f"_{part}"
                break
        out_path = output_dir / f"{subject}_{task}{run}.h5"
        # Use sidecar from the SAME release as the BDF
        release_dir = bdf_path.parents[2]  # up from eeg/ -> sub-X/ -> R*_L100_bdf/
        sidecar_file = release_dir / f"task-{task}_events.json"
        hed_sidecar_path = str(sidecar_file) if sidecar_file.exists() else None
        to_process.append(
            (
                bdf_path,
                events_path,
                out_path,
                interp_matrix,
                hed_sidecar_path,
                hed_vectorizer_data,
                speed_checkpoint,
                args.speed_device,
                args.speed_amp or None,
                window_spec,
                preprocess_version,
            )
        )

    # Skip already cached
    if args.force:
        uncached = to_process
        logger.info("Force mode: re-processing all %d recordings", len(uncached))
    else:
        uncached = [t for t in to_process if not t[2].exists()]
        cached = len(to_process) - len(uncached)
        if cached > 0:
            logger.info("Skipping %d already cached recordings", cached)

    if not uncached:
        logger.info("All recordings already cached!")
    else:
        total_epochs = 0
        done = 0
        errors = 0

        # Use 'spawn' start method when denoising with CUDA to avoid
        # "Cannot re-initialize CUDA in forked subprocess" errors.
        import multiprocessing

        mp_context = multiprocessing.get_context("spawn") if speed_checkpoint else None
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as pool:
            futures = {
                pool.submit(process_one_recording, *args): args[2] for args in uncached
            }
            for future in as_completed(futures):
                done += 1
                out = futures[future]
                try:
                    n = future.result()
                    if n > 0:
                        total_epochs += n
                except Exception as exc:
                    errors += 1
                    logger.error("Failed %s: %s", out.name, exc)

                if done % 200 == 0 or done == len(uncached):
                    elapsed = time.monotonic() - start
                    rate = done / elapsed
                    logger.info(
                        "Progress: %d/%d (%.0f/s), %d epochs, %d errors",
                        done,
                        len(uncached),
                        rate,
                        total_epochs,
                        errors,
                    )

        elapsed = time.monotonic() - start
        logger.info(
            "Done: %d recordings in %.1fs (%.1f/s), %d epochs total, %d errors",
            len(uncached),
            elapsed,
            len(uncached) / elapsed,
            total_epochs,
            errors,
        )

    # Summary
    n_files = len(list(output_dir.glob("*.h5")))
    logger.info("Output: %d HDF5 files in %s", n_files, output_dir)


if __name__ == "__main__":
    main()
