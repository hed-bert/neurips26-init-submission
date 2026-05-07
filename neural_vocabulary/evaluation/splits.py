"""Held-out data split strategies for HED-BERT evaluation.

Provides three splitting strategies for the HBN preprocessed h5 files:
1. held_out_release: exclude an entire HBN release (e.g., R12)
2. held_out_subjects: random subject-level split from R1-R11
3. leave_one_task_out: exclude all epochs from one HBN task

All functions return (train_files, eval_files) lists of Path objects.

H5 file naming convention: {subject_id}_{task_name}[_run-N].h5
Subject IDs encode their release: e.g., NDARAB123 from R1_L100_bdf/.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def _scan_h5_files(data_root: str | Path) -> list[Path]:
    """Scan preprocessed directory for h5 files."""
    data_root = Path(data_root)
    files = sorted(data_root.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {data_root}")
    return files


def _extract_subject(h5_path: Path) -> str:
    """Extract subject ID from h5 filename.

    Filename format: {subject_id}_{task_name}[_run-N].h5
    """
    return h5_path.stem.split("_")[0]


def _extract_task(h5_path: Path) -> str:
    """Extract task name from h5 filename.

    Filename format: {subject_id}_{task_name}[_run-N].h5
    """
    parts = h5_path.stem.split("_")
    if len(parts) >= 2:
        return parts[1]
    return ""


def held_out_release(
    data_root: str | Path,
    release_id: str = "R12",
    bids_root: str | Path | None = None,
) -> tuple[list[Path], list[Path]]:
    """Split by excluding all subjects from a given HBN release.

    Scans the BIDS root to find which subjects belong to the specified
    release, then partitions h5 files accordingly.

    Args:
        data_root: Path to preprocessed h5 directory.
        release_id: Release to hold out (e.g., "R12"). Must correspond
            to a directory like R12_L100_bdf/ under bids_root.
        bids_root: Root of the BIDS dataset with R*_L100_bdf/ dirs.
            Required to determine subject-to-release mapping.

    Returns:
        (train_files, eval_files) tuple of Path lists.
    """
    h5_files = _scan_h5_files(data_root)

    if bids_root is None:
        raise ValueError(
            "bids_root is required for held_out_release to determine "
            "which subjects belong to which release."
        )

    bids_root = Path(bids_root)
    release_dir = bids_root / f"{release_id}_L100_bdf"
    if not release_dir.exists():
        raise FileNotFoundError(
            f"Release directory not found: {release_dir}. "
            f"Available releases: {sorted(d.name for d in bids_root.iterdir() if d.name.endswith('_L100_bdf'))}"
        )

    # Collect subject IDs from the held-out release
    held_out_subjects_set: set[str] = set()
    for sub_dir in release_dir.iterdir():
        if sub_dir.is_dir() and sub_dir.name.startswith("sub-"):
            held_out_subjects_set.add(sub_dir.name.replace("sub-", ""))

    if not held_out_subjects_set:
        raise ValueError(f"No subjects found in {release_dir}")

    train_files = []
    eval_files = []
    for f in h5_files:
        subject = _extract_subject(f)
        if subject in held_out_subjects_set:
            eval_files.append(f)
        else:
            train_files.append(f)

    logger.info(
        "held_out_release(%s): %d train, %d eval (%d subjects held out)",
        release_id,
        len(train_files),
        len(eval_files),
        len(held_out_subjects_set),
    )
    return train_files, eval_files


def held_out_subjects(
    data_root: str | Path,
    ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Random subject-level train/eval split.

    Ensures all files from the same subject end up in the same split
    (no subject leakage between train and eval).

    Args:
        data_root: Path to preprocessed h5 directory.
        ratio: Fraction of subjects to hold out for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        (train_files, eval_files) tuple of Path lists.
    """
    h5_files = _scan_h5_files(data_root)

    # Group files by subject
    subject_files: dict[str, list[Path]] = {}
    for f in h5_files:
        subject = _extract_subject(f)
        subject_files.setdefault(subject, []).append(f)

    subjects = sorted(subject_files.keys())
    n_eval = max(1, int(len(subjects) * ratio))

    rng = random.Random(seed)
    eval_subjects = set(rng.sample(subjects, n_eval))

    train_files = []
    eval_files = []
    for subject in subjects:
        if subject in eval_subjects:
            eval_files.extend(subject_files[subject])
        else:
            train_files.extend(subject_files[subject])

    logger.info(
        "held_out_subjects(ratio=%.2f, seed=%d): %d train, %d eval "
        "(%d/%d subjects held out)",
        ratio,
        seed,
        len(train_files),
        len(eval_files),
        len(eval_subjects),
        len(subjects),
    )
    return train_files, eval_files


def leave_one_task_out(
    data_root: str | Path,
    held_out_task: str = "ThePresent",
) -> tuple[list[Path], list[Path]]:
    """Split by excluding all files from one HBN task.

    Task name is extracted from the h5 filename: {subject}_{task}[_run-N].h5

    Args:
        data_root: Path to preprocessed h5 directory.
        held_out_task: Task to hold out (e.g., "ThePresent",
            "RestingState", "DirtyWord").

    Returns:
        (train_files, eval_files) tuple of Path lists.
    """
    h5_files = _scan_h5_files(data_root)

    # Discover all tasks for validation
    all_tasks: set[str] = set()
    for f in h5_files:
        all_tasks.add(_extract_task(f))

    if held_out_task not in all_tasks:
        raise ValueError(
            f"Task '{held_out_task}' not found. Available tasks: {sorted(all_tasks)}"
        )

    train_files = []
    eval_files = []
    for f in h5_files:
        task = _extract_task(f)
        if task == held_out_task:
            eval_files.append(f)
        else:
            train_files.append(f)

    logger.info(
        "leave_one_task_out(%s): %d train, %d eval",
        held_out_task,
        len(train_files),
        len(eval_files),
    )
    return train_files, eval_files
