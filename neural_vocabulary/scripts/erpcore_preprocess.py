"""Preprocess ERP-CORE (NEMAR nm000132) into the preprocessed h5 schema.

Brings ERP-CORE's 40-subject 6-task dataset (Biosemi 30 EEG + 3 EOG,
1024 Hz, EEGLAB .set/.fdt) into the same fixed-window h5 format used
across this codebase: 64 channels at 100 Hz, LEGACY_WINDOW_SPEC stim/response
windows (100 samples), pre-stim baseline correction. Output files set
``preprocess_version="v9_tier1"`` so consumers (Morlet TF extractor,
BertSSL) read them with no special-casing; the per-recording
``dataset="erp_core"`` attr identifies the source.

Source-vs-target deltas:
    Channels:    30 EEG + 3 EOG (Biosemi)         -> 64 (standard_1005 interp)
    Sfreq:       1024 Hz                          -> 100 Hz
    Filter:      none                             -> HP 0.1 / LP 40 Butter-4
    Reference:   CMS (Biosemi active)             -> average reference (EEG only)
    Window:      event-locked, variable           -> LEGACY_WINDOW_SPEC fixed (100 samples)
    Event class: BIDS trial_type column           -> classify_erpcore_event
    HED:         per-task task-*_events.json      -> hed_tag string attr per epoch

The ``hed_tag`` attr stores the raw HED string; multi-hot vectorization
into the HBN tag vocabulary is handled by a separate downstream HED-
vectorization step (not the Morlet TF extractor).

Output filename convention: ``{subject_id}_{task}.h5`` (BIDS ``task-``
prefix stripped) so ``_task_from_stem`` in ``extract_tf_features.py``
matches against ``ERP_CORE_TASKS``.

Usage (smoke test, 2 subjects, N170 only):
    uv run python -m neural_vocabulary.scripts.erpcore_preprocess \\
        --data-root ${ERPCORE_DATA_DIR} \\
        --output-dir /tmp/erpcore_smoke \\
        --max-subjects 2 --tasks N170

Usage (full run, all 40 subjects x 6 tasks, parallel):
    uv run python -m neural_vocabulary.scripts.erpcore_preprocess \\
        --data-root ${ERPCORE_DATA_DIR} \\
        --output-dir ${HBN_DATA_DIR}/preprocessed_v10_erpcore_tier1 \\
        --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
import torch

from neural_vocabulary.data.erpcore_event_classifier import classify_erpcore_event
from neural_vocabulary.models.channel_harmonization import ChannelHarmonization
from neural_vocabulary.scripts.preprocess_hbn import (
    HIGH_PASS_HZ,
    LOW_PASS_HZ,
    SFREQ,
    LEGACY_WINDOW_SPEC,
    R4_WINDOW_SPEC,
    average_rereference,
    denoise_continuous,
    epoch_data,
    highpass_filter,
    lowpass_filter,
    parse_window_spec_arg,
    resample_to,
)
from neural_vocabulary.scripts.extract_tf_features import ERP_CORE_TASKS as _MX_TASKS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 6 ERP-CORE tasks shipped by NEMAR nm000132. Sourced from
# extract_tf_features.ERP_CORE_TASKS (the single source-of-truth) and
# re-exported as a stable-order tuple for CLI --tasks choices.
ERP_CORE_TASKS: tuple[str, ...] = tuple(sorted(_MX_TASKS))


def _normalize_channel_name(name: str, montage_keys_ci: dict[str, str]) -> str | None:
    """Return the montage-canonical spelling of a channel name (case-insensitive)."""
    return montage_keys_ci.get(name.lower())


def _build_harmonizer(eeg_ch_names: list[str]) -> ChannelHarmonization:
    """Build an EEG-channel-count -> 64 harmonizer for this recording.

    ERP-CORE uses Biosemi ALL-CAPS conventions (FP1/FP2) while MNE's
    standard montages use Fp1/Fp2. We case-fold to the montage spelling
    against standard_1005 (which covers the extended P9/P10/PO7/PO8
    positions ERP-CORE includes but standard_1020 does not), then build
    a from_montages harmonizer with the same montage on both source and
    target sides so the interpolation stays in a single coordinate space.

    Raises ValueError if any channel cannot be located in the montage.
    """
    montage = mne.channels.make_standard_montage("standard_1005")
    pos = montage.get_positions()["ch_pos"]
    keys_ci = {k.lower(): k for k in pos}

    canonical: list[str] = []
    missing: list[str] = []
    for name in eeg_ch_names:
        c = _normalize_channel_name(name, keys_ci)
        if c is None:
            missing.append(name)
        else:
            canonical.append(c)
    if missing:
        raise ValueError(f"ERP-CORE channels not in standard_1005 montage: {missing}")

    return ChannelHarmonization.from_montages(
        source_montage=montage,
        source_ch_names=canonical,
        target_montage=montage,
    )


def _load_erpcore_recording(
    set_path: Path,
    channels_tsv: Path,
) -> tuple[np.ndarray, float, list[str]]:
    """Load ERP-CORE EEGLAB .set + select EEG channels only.

    pyedflib (used by HBN's BDF reader) returns physical units (uV);
    MNE's ``read_raw_eeglab`` returns volts. Scale by 1e6 here so the
    output uses the same uV convention as ``preprocess_hbn``.

    Channel names are kept in the original Biosemi case (FP1/FP2);
    case normalization to montage spelling happens in
    ``_build_harmonizer``.

    Raises:
        ValueError: if channels.tsv lacks the ``type`` column, if any
            channel in the raw recording is missing from channels.tsv,
            or if no EEG channels remain after filtering. These are
            dataset-layout errors; failing loud here prevents EOG
            contamination of the spatial layout downstream.
    """
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

    df = pd.read_csv(channels_tsv, sep="\t")
    if "type" not in df.columns:
        raise ValueError(f"channels.tsv missing 'type' column: {channels_tsv}")
    type_map = dict(zip(df["name"].astype(str), df["type"].astype(str), strict=False))
    unmapped = [n for n in raw.ch_names if n not in type_map]
    if unmapped:
        raise ValueError(
            f"Channels in {set_path.name} not listed in channels.tsv: {unmapped}"
        )
    eeg_names = [n for n in raw.ch_names if type_map[n].upper() == "EEG"]
    if not eeg_names:
        raise ValueError(f"No EEG channels found in {channels_tsv}")

    raw.pick(eeg_names)
    data_uv = (raw.get_data() * 1e6).astype(np.float32)
    return data_uv, float(raw.info["sfreq"]), list(raw.ch_names)


def _parse_events_tsv(events_path: Path) -> pd.DataFrame:
    """Read an ERP-CORE events.tsv as a DataFrame.

    Raises FileNotFoundError if the file does not exist; the production
    caller pre-validates existence in ``collect_recordings``.
    """
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    return pd.read_csv(events_path, sep="\t")


def _load_hed_lookups(
    sidecar_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load per-task HED lookups from task-*_events.json.

    Returns ``(by_value, by_event_type)``: two parallel maps. ERP-CORE
    tasks differ in which BIDS column keys the HED block:
      * MMN, P3, N2pc, N400, flankers: HED block under ``value`` (numeric).
      * N170: HED block under ``event_type`` (string: face/car/...).
    The caller looks up by ``event_type`` first (more semantic), then
    falls back to ``value``.

    A missing sidecar is a dataset-layout misconfiguration (the file is
    invariant per task across the dataset, not per recording), so we
    raise FileNotFoundError rather than silently returning empty dicts.
    Likewise, a sidecar with no HED block at all raises ValueError.
    """
    if not sidecar_path.exists():
        raise FileNotFoundError(f"HED sidecar missing: {sidecar_path}")
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    by_value: dict[str, str] = {}
    by_event_type: dict[str, str] = {}
    val_block = sidecar.get("value", {})
    if isinstance(val_block, dict):
        for code, hed_str in val_block.get("HED", {}).items():
            by_value[str(code)] = str(hed_str)
    et_block = sidecar.get("event_type", {})
    if isinstance(et_block, dict):
        for code, hed_str in et_block.get("HED", {}).items():
            by_event_type[str(code)] = str(hed_str)
    if not by_value and not by_event_type:
        raise ValueError(f"HED sidecar has no HED block: {sidecar_path}")
    return by_value, by_event_type


def _classify_and_filter_events(
    events_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, dict], dict[str, int]]:
    """Drop rows that don't classify; pair each kept row with its raw metadata.

    ``epoch_data`` (from preprocess_hbn) iterates rows in order and
    re-classifies via ``classify_epoch_type``. Each kept row's
    ``value`` field is rewritten to ``erpcore_stim_<idx>`` or
    ``erpcore_response_<idx>``: the literal substring ``_response`` in
    the response token matches preprocess_hbn's ``_RESPONSE_MARKERS``;
    the stim token falls through to the default ``stim`` branch.
    Embedding the index in the token (rather than relying on list
    position) keeps the round-trip robust if ``epoch_data`` ever drops
    a row due to recording-boundary clamping.

    Returns ``(events_for_epoch_data, idx_to_meta, drop_stats)``.
    ``drop_stats`` reports skip reasons:
        ``status_or_missing``: STATUS markers and BIDS missing values
        ``unknown:<trial_type>``: any trial_type the classifier rejects
            (could indicate a new ERP-CORE label we don't yet handle).
    """
    drop_stats: dict[str, int] = {}
    if events_df.empty:
        return events_df.copy(), {}, drop_stats

    rows: list[dict] = []
    meta: dict[int, dict] = {}
    next_idx = 0
    has_event_type = "event_type" in events_df.columns
    skip_set = {"STATUS", "n/a", "nan", ""}
    for _, row in events_df.iterrows():
        trial_type = row.get("trial_type", None)
        ep_type = classify_erpcore_event(trial_type)
        if ep_type is None:
            tt_key = str(trial_type).strip() if trial_type is not None else ""
            if tt_key in skip_set:
                drop_stats["status_or_missing"] = (
                    drop_stats.get("status_or_missing", 0) + 1
                )
            else:
                key = f"unknown:{tt_key}"
                drop_stats[key] = drop_stats.get(key, 0) + 1
            continue
        raw_value = str(row.get("value", ""))
        raw_event_type_in = str(row["event_type"]) if has_event_type else ""
        # Normalize BIDS missing-value markers to empty string so the
        # downstream HED lookup branches on truthiness cleanly.
        raw_event_type = (
            "" if raw_event_type_in.strip() in skip_set else raw_event_type_in
        )
        canonical = f"erpcore_{ep_type}_{next_idx}"
        rows.append({"onset": float(row["onset"]), "value": canonical})
        meta[next_idx] = {
            "raw_value": raw_value,
            "raw_event_type": raw_event_type,
            "epoch_type": ep_type,
        }
        next_idx += 1
    return pd.DataFrame(rows), meta, drop_stats


def _safe_unlink(path: Path) -> None:
    """Unlink ``path`` if present; never raise.

    Used in error-handling paths where a failed cleanup would shadow
    the original exception. The cleanup failure is logged so an FS
    issue is still visible, but the original error reaches the caller.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as cleanup_err:  # noqa: BLE001
        logger.error("Cleanup of %s failed: %s", path, cleanup_err)


def _stable_event_id(raw_value: str) -> int:
    """Deterministic 63-bit id for an event value.

    Python's built-in ``hash`` is randomized per process (PYTHONHASHSEED),
    so different worker processes assign different ids to the same event
    value. We use a deterministic 63-bit prefix of MD5 instead so the id
    is stable across processes, sessions, and consumers. 63 bits keeps
    birthday-collision probability negligible (< 1e-15) even for the full
    cross-dataset event vocabulary, in case downstream consumers ever
    treat ``event_id`` as a unique key.
    """
    digest = hashlib.md5(raw_value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & 0x7FFF_FFFF_FFFF_FFFF


def _process_one_recording(
    set_path: Path,
    events_path: Path,
    channels_tsv: Path,
    sidecar_path: Path,
    output_path: Path,
    task: str,
    subject_id: str,
    window_spec: dict[str, tuple[float, float]] = LEGACY_WINDOW_SPEC,
    preprocess_version: str = "v9_tier1",
    speed_checkpoint: str | None = None,
    speed_device: str = "cpu",
    speed_amp: bool = False,
) -> dict:
    """Preprocess one ERP-CORE recording end-to-end. Returns a stats dict.

    Writes the output h5 atomically (tmp + os.replace) so a SIGKILL or
    OOM mid-write cannot leave a partial file at ``output_path``.

    Raises on misconfiguration (missing sidecar, missing events.tsv,
    bad channels.tsv, stim with no HED). The wrapper at the worker
    boundary captures these and surfaces them through ``stats`` so the
    pool does not crash on a single bad recording, but the failure is
    never silent: ``status`` becomes ``error:<ExcName>`` and the full
    traceback is preserved in ``stats["traceback"]``.
    """
    stats: dict = {
        "status": "ok",
        "set_path": str(set_path),
        "n_epochs": 0,
        "drop_stats": {},
    }
    tmp_path = output_path.with_suffix(".h5.tmp")
    try:
        data, sfreq, eeg_names = _load_erpcore_recording(set_path, channels_tsv)
        harmonizer = _build_harmonizer(eeg_names)
        denoised_ok = False

        if speed_checkpoint is not None:
            # Harmonize 30 -> 64 standard 10-05 channels FIRST so SPEED sees
            # the same standard channel names + topography it was trained on
            # (mirrors preprocess_hbn.py: interp before SPEED at native sfreq).
            # The non-denoise path below preserves the original ordering
            # (filter -> resample -> avg-ref -> harmonize) for bit-identical
            # backwards compatibility.
            data_64 = harmonizer(torch.from_numpy(data)).numpy().astype(np.float32)
            ch_names_64 = list(harmonizer.target_channels)
            data_64, sfreq, ch_names_64, denoised_ok = denoise_continuous(
                data=data_64,
                sfreq=sfreq,
                ch_names=ch_names_64,
                checkpoint_path=speed_checkpoint,
                device=speed_device,
                amp=speed_amp,
            )
            if not denoised_ok:
                # SPEED returned but did not actually denoise (e.g. NaN
                # output, recording too short for 3 s window). The output
                # directory is the user's denoised target — silently
                # producing undenoised data here would mislabel the file.
                raise RuntimeError(
                    f"SPEED denoising did not complete for {set_path.name} "
                    "(see WARNING above). Refusing to write undenoised data "
                    "into a denoised output directory."
                )
            logger.info(
                "  SPEED output: %d ch, %d samples at %.0f Hz",
                data_64.shape[0],
                data_64.shape[1],
                sfreq,
            )

            if sfreq > 2 * LOW_PASS_HZ:
                data_64 = lowpass_filter(data_64, sfreq, LOW_PASS_HZ)
            if abs(sfreq - SFREQ) > 1e-6:
                data_64, sfreq = resample_to(data_64, sfreq, SFREQ)
            data_64 = highpass_filter(data_64, sfreq, HIGH_PASS_HZ)
            data_64 = average_rereference(data_64)
        else:
            # Anti-alias LP at 40 Hz BEFORE downsampling.
            if sfreq > 2 * LOW_PASS_HZ:
                data = lowpass_filter(data, sfreq, LOW_PASS_HZ)
            if abs(sfreq - SFREQ) > 1e-6:
                data, sfreq = resample_to(data, sfreq, SFREQ)
            data = highpass_filter(data, sfreq, HIGH_PASS_HZ)
            data = average_rereference(data)

            # Data-driven 30->64 harmonization, derived from this recording's
            # actual channel list against the standard_1005 montage.
            data_64 = harmonizer(torch.from_numpy(data)).numpy().astype(np.float32)

        events_df_raw = _parse_events_tsv(events_path)
        events_df, idx_to_meta, drop_stats = _classify_and_filter_events(events_df_raw)
        stats["drop_stats"] = drop_stats
        if events_df.empty:
            stats["status"] = "no_valid_events"
            return stats
        unknown_keys = [k for k in drop_stats if k.startswith("unknown:")]
        if unknown_keys:
            unknown_summary = {k: drop_stats[k] for k in unknown_keys}
            raise ValueError(
                f"{set_path.name}: unrecognized trial_type values "
                f"{unknown_summary}. Extend classify_erpcore_event "
                "(neural_vocabulary/data/erpcore_event_classifier.py) to handle "
                "these or document why they should be ignored."
            )

        hed_by_value, hed_by_event_type = _load_hed_lookups(sidecar_path)

        epochs = epoch_data(data_64, events_df, sfreq, window_spec)
        if not epochs:
            stats["status"] = "no_epochs_after_window"
            return stats

        # Map each epoch back to its raw metadata via the embedded index,
        # then look up HED. Stim epochs without a HED string are a fatal
        # contract violation: downstream HED multi-hot vectorization
        # assumes every stim has hed_tag.
        missing_stim_hed: list[str] = []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(tmp_path, "w") as f:
            f.attrs["preprocess_version"] = preprocess_version
            f.attrs["preprocess_subversion"] = "v10_d2_0"
            f.attrs["dataset"] = "erp_core"
            f.attrs["task"] = task
            f.attrs["subject_id"] = subject_id
            f.attrs["n_epochs"] = len(epochs)
            f.attrs["sfreq"] = sfreq
            f.attrs["n_channels"] = data_64.shape[0]
            f.attrs["channel_names"] = json.dumps(harmonizer.target_channels)
            f.attrs["source_channel_names"] = json.dumps(eeg_names)
            f.attrs["source_sfreq"] = 1024.0
            f.attrs["high_pass_hz"] = HIGH_PASS_HZ
            f.attrs["low_pass_hz"] = LOW_PASS_HZ
            f.attrs["window_spec"] = json.dumps(
                {k: list(v) for k, v in window_spec.items()}
            )
            f.attrs["baseline_correction"] = "pre_stim_mean"
            f.attrs["reference"] = "average"
            f.attrs["harmonization_source_montage"] = "standard_1005"
            f.attrs["harmonization_target_montage"] = "standard_1005"
            f.attrs["denoised"] = bool(denoised_ok)
            if denoised_ok:
                f.attrs["denoise_method"] = "SPEED-EEG"

            for i, ep in enumerate(epochs):
                grp = f.create_group(f"epoch_{i:05d}")
                grp.create_dataset("eeg", data=ep["eeg"])
                grp.create_dataset("pad_mask", data=ep["pad_mask"])

                canonical = ep["event_value"]
                _, _ep_kind, idx_str = canonical.split("_", 2)
                idx = int(idx_str)
                if idx not in idx_to_meta:
                    raise RuntimeError(
                        f"Canonical token {canonical!r} index {idx} not in "
                        f"idx_to_meta (size {len(idx_to_meta)}); round-trip broken."
                    )
                meta = idx_to_meta[idx]
                raw_value = meta["raw_value"]
                raw_event_type = meta["raw_event_type"]

                grp.attrs["event_id"] = _stable_event_id(raw_value)
                grp.attrs["event_value"] = raw_value
                grp.attrs["event_type"] = raw_event_type
                grp.attrs["epoch_type"] = ep["epoch_type"]
                grp.attrs["onset_sample"] = ep["onset_sample"]
                grp.attrs["pre_event_samples"] = ep["pre_event_samples"]
                grp.attrs["length"] = ep["length"]
                grp.attrs["duration_samples"] = ep["length"]

                # HED lookup priority: event_type column first (semantic
                # key, used by N170), then value column (numeric, used by
                # MMN/P3/N400/N2pc/flankers).
                hed_str = ""
                if raw_event_type and raw_event_type in hed_by_event_type:
                    hed_str = hed_by_event_type[raw_event_type]
                elif raw_value and raw_value in hed_by_value:
                    hed_str = hed_by_value[raw_value]
                if hed_str:
                    grp.attrs["hed_tag"] = hed_str
                elif ep["epoch_type"] == "stim":
                    missing_stim_hed.append(
                        f"value={raw_value!r} event_type={raw_event_type!r}"
                    )

        if missing_stim_hed:
            _safe_unlink(tmp_path)
            raise ValueError(
                f"{len(missing_stim_hed)} stim epochs in {set_path.name} have no "
                f"HED tag. First 5: {missing_stim_hed[:5]}. Sidecar "
                f"{sidecar_path.name} likely missing entries; fix sidecar before "
                "downstream HED multi-hot vectorization."
            )
        os.replace(tmp_path, output_path)

        stats["n_epochs"] = len(epochs)
        return stats
    except Exception as e:
        _safe_unlink(tmp_path)
        logger.exception("Failed %s: %s", set_path, e)
        stats["status"] = f"error:{type(e).__name__}"
        stats["error_msg"] = str(e)
        stats["traceback"] = traceback.format_exc()
        return stats


def collect_recordings(
    data_root: Path,
    tasks: tuple[str, ...] | None = None,
    max_subjects: int | None = None,
) -> list[tuple[Path, Path, Path, Path, str, str]]:
    """Scan the BIDS root and return one tuple per recording.

    Tuple format: (set_path, events_path, channels_tsv, sidecar_path,
                   task, subject_id).
    """
    selected_tasks = tuple(tasks) if tasks else ERP_CORE_TASKS
    out: list[tuple[Path, Path, Path, Path, str, str]] = []

    sub_dirs = sorted(
        d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("sub-")
    )
    if max_subjects is not None:
        sub_dirs = sub_dirs[:max_subjects]

    for sub_dir in sub_dirs:
        subject_id = sub_dir.name.replace("sub-", "")
        eeg_dir = sub_dir / "eeg"
        if not eeg_dir.exists():
            continue
        for task in selected_tasks:
            set_path = eeg_dir / f"sub-{subject_id}_task-{task}_eeg.set"
            events_path = eeg_dir / f"sub-{subject_id}_task-{task}_events.tsv"
            channels_tsv = eeg_dir / f"sub-{subject_id}_task-{task}_channels.tsv"
            sidecar_path = data_root / f"task-{task}_events.json"
            if (
                not set_path.exists()
                or not events_path.exists()
                or not channels_tsv.exists()
            ):
                logger.warning(
                    "Missing files for sub-%s task-%s; skipping", subject_id, task
                )
                continue
            out.append(
                (set_path, events_path, channels_tsv, sidecar_path, task, subject_id)
            )
    return out


def _output_path_for(output_dir: Path, subject_id: str, task: str) -> Path:
    """Match the {subject}_{task}.h5 filename convention used by HBN preprocessing.

    Strips the BIDS ``task-`` prefix so ``_task_from_stem`` in
    ``extract_tf_features.py`` matches against ``ERP_CORE_TASKS``.
    """
    return output_dir / f"{subject_id}_{task}.h5"


def _process_wrapper(
    args: tuple[Path, Path, Path, Path, str, str],
    output_dir: Path,
    force: bool,
    window_spec: dict[str, tuple[float, float]] = LEGACY_WINDOW_SPEC,
    preprocess_version: str = "v9_tier1",
    speed_checkpoint: str | None = None,
    speed_device: str = "cpu",
    speed_amp: bool = False,
) -> dict:
    """Worker-pool entry point. Always returns a stats dict (never raises).

    Wraps the entire per-recording flow in try/except so a single FS-error,
    bad input tuple, or other unexpected failure surfaces as
    ``status="error:<ExcName>"`` rather than terminating the whole pool
    (which would happen if ``future.result()`` re-raised in the
    ``as_completed`` loop).
    """
    set_path = args[0] if args else None
    try:
        set_path, events_path, channels_tsv, sidecar_path, task, subject_id = args
        out_path = _output_path_for(output_dir, subject_id, task)
        if out_path.exists() and not force:
            return {
                "status": "already_exists",
                "set_path": str(set_path),
                "n_epochs": 0,
                "drop_stats": {},
            }
        return _process_one_recording(
            set_path=set_path,
            events_path=events_path,
            channels_tsv=channels_tsv,
            sidecar_path=sidecar_path,
            output_path=out_path,
            task=task,
            subject_id=subject_id,
            window_spec=window_spec,
            preprocess_version=preprocess_version,
            speed_checkpoint=speed_checkpoint,
            speed_device=speed_device,
            speed_amp=speed_amp,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Wrapper failed for %s: %s", set_path, e)
        return {
            "status": f"error:WrapperFailure:{type(e).__name__}",
            "set_path": str(set_path) if set_path else "<unknown>",
            "n_epochs": 0,
            "drop_stats": {},
            "error_msg": str(e),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ERP-CORE (NEMAR nm000132) into preprocessed h5."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("${HBN_DATA_DIR}/preprocessed_v10_erpcore_tier1"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: cpu_count // 2).",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(ERP_CORE_TASKS),
        default=None,
        help="Subset of ERP-CORE tasks to process.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--window-preset",
        choices=["v9_tier1", "v11_r4a"],
        default="v9_tier1",
        help="Named window-spec preset. v9_tier1 = stim:(0.2, 0.8). "
        "v11_r4a = stim:(0.3, 1.3). Overridden by --window-spec.",
    )
    parser.add_argument(
        "--window-spec",
        type=str,
        default=None,
        help='Override window_spec as JSON, e.g. \'{"stim":[0.3,1.3],'
        '"response":[0.3,1.3],"tonic":[0.3,2.0]}\'. Baseline mean is '
        "always the full pre-stim window.",
    )
    parser.add_argument(
        "--preprocess-version",
        type=str,
        default=None,
        help="Tag written to file attr 'preprocess_version'. Defaults to "
        "v9_tier1 (preset=v9_tier1) or v11_erpcore_r4 (preset=v11_r4a).",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply SPEED-EEG denoising before epoching. Harmonization "
        "30->64 channels runs BEFORE SPEED so it sees standard 10-05 "
        "channel names + topography (matches preprocess_hbn ordering).",
    )
    parser.add_argument(
        "--speed-checkpoint",
        type=str,
        default=None,
        help="Path to SPEED-EEG model checkpoint (.pt). Required with --denoise.",
    )
    parser.add_argument(
        "--speed-device",
        type=str,
        default="cpu",
        help="Device for SPEED inference (cpu, cuda, cuda:0, ...).",
    )
    parser.add_argument(
        "--speed-amp",
        action="store_true",
        help="Enable mixed-precision (fp16) SPEED inference on CUDA.",
    )
    args = parser.parse_args()

    if args.denoise and args.speed_checkpoint is None:
        parser.error("--speed-checkpoint is required when --denoise is used")
    speed_checkpoint = args.speed_checkpoint if args.denoise else None

    # Resolve window_spec: explicit JSON > preset. ERP-CORE has no tonic
    # epochs, but we still require the tonic key in JSON overrides because
    # the LEGACY_WINDOW_SPEC / R4_WINDOW_SPEC defaults set it and the
    # downstream epoch_data() reads it via classify_epoch_type. Users
    # passing only stim/response can copy the tonic entry from the preset.
    if args.window_spec is not None:
        try:
            window_spec = parse_window_spec_arg(args.window_spec)
        except ValueError as e:
            parser.error(str(e))
        preset_label = "custom"
    elif args.window_preset == "v11_r4a":
        window_spec = R4_WINDOW_SPEC
        preset_label = "v11_r4a"
    else:
        window_spec = LEGACY_WINDOW_SPEC
        preset_label = "v9_tier1"

    if args.preprocess_version is not None:
        preprocess_version = args.preprocess_version
    elif preset_label == "v11_r4a":
        preprocess_version = "v11_erpcore_r4"
    else:
        preprocess_version = "v9_tier1"

    logger.info(
        "Window spec: %s (preset=%s, preprocess_version=%s)",
        json.dumps({k: list(v) for k, v in window_spec.items()}),
        preset_label,
        preprocess_version,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recordings = collect_recordings(
        data_root=args.data_root,
        tasks=tuple(args.tasks) if args.tasks else None,
        max_subjects=args.max_subjects,
    )
    if not recordings:
        raise RuntimeError(f"No recordings found in {args.data_root}")

    n_workers = args.workers or max(1, (os.cpu_count() or 1) // 2)
    logger.info("Processing %d recordings with %d workers", len(recordings), n_workers)

    t0 = time.perf_counter()
    if n_workers == 1:
        results = [
            _process_wrapper(
                r,
                args.output_dir,
                args.force,
                window_spec,
                preprocess_version,
                speed_checkpoint,
                args.speed_device,
                args.speed_amp,
            )
            for r in recordings
        ]
    else:
        # 'spawn' start method when denoising with CUDA: forked workers
        # cannot re-initialize CUDA. Mirrors preprocess_hbn ordering.
        import multiprocessing

        mp_context = multiprocessing.get_context("spawn") if speed_checkpoint else None
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as pool:
            futures = {
                pool.submit(
                    _process_wrapper,
                    r,
                    args.output_dir,
                    args.force,
                    window_spec,
                    preprocess_version,
                    speed_checkpoint,
                    args.speed_device,
                    args.speed_amp,
                ): r
                for r in recordings
            }
            results = []
            for done, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if done % 25 == 0:
                    logger.info("Processed %d/%d", done, len(recordings))

    elapsed = time.perf_counter() - t0
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_existing = sum(1 for r in results if r["status"] == "already_exists")
    n_errors = sum(1 for r in results if r["status"].startswith("error"))
    n_no_events = sum(
        1
        for r in results
        if r["status"] in ("no_valid_events", "no_epochs_after_window")
    )
    total_epochs = sum(r.get("n_epochs", 0) for r in results)
    aggregated_drops: dict[str, int] = {}
    for r in results:
        for k, v in r.get("drop_stats", {}).items():
            aggregated_drops[k] = aggregated_drops.get(k, 0) + v

    print("\n=== ERP-CORE preprocessing report ===")
    print(f"  Data root:       {args.data_root}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Recordings:      {len(recordings)}")
    print(f"    OK:            {n_ok}")
    print(f"    Already exist: {n_existing}")
    print(f"    No events:     {n_no_events}")
    print(f"    Errors:        {n_errors}")
    print(f"  Total epochs:    {total_epochs}")
    print("  Dropped events by reason:")
    for k in sorted(aggregated_drops):
        print(f"    {k:30s} {aggregated_drops[k]}")
    print(f"  Elapsed:         {elapsed:.1f} s")

    if n_errors > 0:
        for r in results:
            if r["status"].startswith("error"):
                logger.error(
                    "  %s: %s\n%s",
                    r["set_path"],
                    r.get("error_msg", "?"),
                    r.get("traceback", ""),
                )
        raise RuntimeError(f"{n_errors} recordings failed; see logs above.")


if __name__ == "__main__":
    main()
