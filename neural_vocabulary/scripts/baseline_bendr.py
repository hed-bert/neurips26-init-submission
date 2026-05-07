""" Gate Lit: BENDR frozen-embedding probe on HBN passive-movie epochs.

Two-phase pipeline:
  Phase A -- Extract frozen BENDR embeddings from HBN preprocessed epochs,
             cache to disk as numpy .npz files (includes stem per embedding).
  Phase B -- Fit LR (C=1.0) + kNN (k=10) probes on cached embeddings,
             evaluate on held-out subjects for animacy + scene labels.

Methodology
-----------
Animacy and scene labels derived from HED tags using _animacy_label and
_scene_label from e2_literature_probes. Subject-level train/eval split
via held_out_subjects (default ratio=0.15). Three seeds (42, 13, 7) used
for the split to estimate variance. Balanced eval set (balanced by
_fit_probe). Random-init ablation runs the same pipeline with a fresh
(untrained) BENDR model.

BENDR preprocessing
--------------------
- Select 19 standard 10-20 channels from 64 HBN channels.
- Resample 100 Hz to 256 Hz (BENDR pretrained rate).
- Append relative amplitude as 20th channel (computed pre-normalization so
  cross-channel magnitude ratios are preserved, matching BENDR pretraining).
- Z-score only the 19 EEG channels; rel-amp channel stays in [0, 1].
- Total conv stride = 96: 220 samples x 256/100 = ~563 samples -> ~5 tokens.

Cache format
------------
The .npz cache stores embeddings, labels, and one stem string per embedding
so phase B can reconstruct subject-level train/eval splits without re-scanning
h5 files.

Usage
-----
Phase A (extract):
    uv run python -m neural_vocabulary.scripts.baseline_bendr \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir runs/baseline_bendr \\
        --checkpoint /path/to/pytorch_model.bin \\
        --phase extract --device cuda

Phase B (probe):
    uv run python -m neural_vocabulary.scripts.baseline_bendr \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir runs/baseline_bendr \\
        --phase probe --seeds 42,13,7 --probes lr,knn

Both phases (default):
    uv run python -m neural_vocabulary.scripts.baseline_bendr \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir runs/baseline_bendr \\
        --device cuda --seeds 42,13,7

Smoke-test with limited data:
    uv run python -m neural_vocabulary.scripts.baseline_bendr \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir /tmp/bendr_test \\
        --limit 4 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from neural_vocabulary.baselines.bendr_adapter import (
    BENDR_CHANNELS,
    BENDR_SFREQ,
    HBN_SFREQ,
    _BENDRContextualizer,
    _ConvEncoderBENDR,
    add_relative_amplitude_channel,
    extract_bendr_embedding,
    load_bendr,
    load_channel_names_from_h5,
    resample_epoch,
    select_channels,
)
from neural_vocabulary.evaluation.splits import held_out_subjects
from neural_vocabulary.scripts.e2_literature_probes import (
    _animacy_label,
    _build_label_masks,
    _fit_probe,
    _scene_label,
)
from neural_vocabulary.training.device_manager import DeviceManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _git_sha() -> str:
    """Return short HEAD git SHA, or 'unknown' if not in a git repo."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


PASSIVE_MOVIE_TASKS: frozenset[str] = frozenset(
    ["DespicableMe", "DiaryOfAWimpyKid", "FunwithFractals", "ThePresent"]
)


def _task_from_stem(stem: str) -> str:
    parts = stem.split("_")
    return parts[1] if len(parts) >= 2 else ""


def _load_hed_vectorizer(source_dir: Path) -> dict[str, int]:
    """Load tag_to_idx from hed_vectorizer.pt in source_dir."""
    vect_path = source_dir / "hed_vectorizer.pt"
    if not vect_path.exists():
        raise FileNotFoundError(
            f"HED vectorizer not found: {vect_path}. "
            "Expected hed_vectorizer.pt in the source directory."
        )
    vect = torch.load(str(vect_path), map_location="cpu", weights_only=False)
    return vect["tag_to_idx"]


def _extract_embeddings_from_h5(
    h5_path: Path,
    encoder: _ConvEncoderBENDR,
    contextualizer: _BENDRContextualizer,
    device: torch.device,
    masks: dict[str, np.ndarray],
    passive_only: bool = True,
) -> dict[str, Any]:
    """Extract BENDR embeddings and labels from one h5 file.

    Channel names are loaded from each file's attrs so per-file channel order
    differences do not silently corrupt channel selection.

    Args:
        h5_path: Path to HBN preprocessed h5 file.
        encoder: Frozen BENDR encoder.
        contextualizer: Frozen BENDR contextualizer.
        device: Compute device.
        masks: HED label masks from _build_label_masks.
        passive_only: If True, only process passive-movie task epochs.

    Returns:
        dict with keys: embeddings, animacy, scene, has_animacy, has_scene, stems.
        Empty arrays if no valid epochs found.
    """
    task = _task_from_stem(h5_path.stem)
    if passive_only and task not in PASSIVE_MOVIE_TASKS:
        return {
            "embeddings": np.empty((0, 512), dtype=np.float32),
            "animacy": np.array([], dtype=np.int64),
            "scene": np.array([], dtype=np.int64),
            "has_animacy": np.array([], dtype=bool),
            "has_scene": np.array([], dtype=bool),
            "stems": [],
        }

    embeddings = []
    animacy_labels: list[int] = []
    scene_labels: list[int] = []
    has_animacy: list[bool] = []
    has_scene: list[bool] = []
    stems: list[str] = []

    try:
        channel_names = load_channel_names_from_h5(h5_path)
        with h5py.File(str(h5_path), "r") as f:
            if "n_epochs" not in f.attrs:
                raise RuntimeError(f"{h5_path} missing n_epochs attr")
            n_epochs = int(f.attrs["n_epochs"])
            skipped_no_epoch = 0
            skipped_no_hed = 0
            for i in range(n_epochs):
                key = f"epoch_{i:05d}"
                if key not in f:
                    skipped_no_epoch += 1
                    continue
                grp = f[key]
                if "hed_vector" not in grp:
                    skipped_no_hed += 1
                    continue

                hed = grp["hed_vector"][:].astype(np.float32)
                an = _animacy_label(hed, masks)
                sc = _scene_label(hed, masks)
                if an is None and sc is None:
                    continue

                eeg = grp["eeg"][:].astype(np.float32)
                eeg_19 = select_channels(eeg, channel_names, BENDR_CHANNELS)
                eeg_19_256 = resample_epoch(eeg_19, HBN_SFREQ, BENDR_SFREQ)
                # Rel-amp is computed on pre-normalization signal so cross-channel
                # magnitude ratios are preserved (matches BENDR pretraining).
                eeg_20 = add_relative_amplitude_channel(eeg_19_256)
                # Z-score only the 19 EEG channels; rel-amp (ch 19) stays in [0,1].
                mu = eeg_20[:19].mean(axis=1, keepdims=True)
                sigma = eeg_20[:19].std(axis=1, keepdims=True)
                eeg_20[:19] = (eeg_20[:19] - mu) / (sigma + 1e-8)

                emb = extract_bendr_embedding(eeg_20, encoder, contextualizer, device)
                embeddings.append(emb)
                has_animacy.append(an is not None)
                has_scene.append(sc is not None)
                animacy_labels.append(-1 if an is None else an)
                scene_labels.append(-1 if sc is None else sc)
                stems.append(h5_path.stem)

            if skipped_no_epoch > 0:
                logger.warning(
                    "%s: %d/%d epochs missing from file",
                    h5_path.name,
                    skipped_no_epoch,
                    n_epochs,
                )
            # Match A.1 extractor behaviour: missing-HED epochs are a real
            # per-subject data pattern (4821/451k across HBN), not schema drift.
            # Raise only if the file is entirely unlabeled.
            if skipped_no_hed == n_epochs and n_epochs > 0:
                raise RuntimeError(
                    f"{h5_path}: all {n_epochs} epochs missing hed_vector. "
                    "Schema drift or upstream pipeline error."
                )
            if skipped_no_hed > 0:
                logger.info(
                    "%s: %d/%d epochs missing hed_vector",
                    h5_path.name,
                    skipped_no_hed,
                    n_epochs,
                )
    except OSError as exc:
        logger.error("OSError reading %s: %s", h5_path, exc)
        raise

    if not embeddings:
        return {
            "embeddings": np.empty((0, 512), dtype=np.float32),
            "animacy": np.array([], dtype=np.int64),
            "scene": np.array([], dtype=np.int64),
            "has_animacy": np.array([], dtype=bool),
            "has_scene": np.array([], dtype=bool),
            "stems": [],
        }

    return {
        "embeddings": np.stack(embeddings).astype(np.float32),
        "animacy": np.array(animacy_labels, dtype=np.int64),
        "scene": np.array(scene_labels, dtype=np.int64),
        "has_animacy": np.array(has_animacy, dtype=bool),
        "has_scene": np.array(has_scene, dtype=bool),
        "stems": stems,
    }


def _collect_embeddings(
    h5_files: list[Path],
    encoder: _ConvEncoderBENDR,
    contextualizer: _BENDRContextualizer,
    device: torch.device,
    masks: dict[str, np.ndarray],
    limit: int | None = None,
) -> dict[str, Any]:
    """Collect embeddings, labels, and stems across a list of h5 files."""
    all_emb: list[np.ndarray] = []
    all_an: list[int] = []
    all_sc: list[int] = []
    all_has_an: list[bool] = []
    all_has_sc: list[bool] = []
    all_stems: list[str] = []
    total_epochs = 0

    for idx, h5_path in enumerate(h5_files):
        if limit is not None and total_epochs >= limit:
            break
        out = _extract_embeddings_from_h5(
            h5_path, encoder, contextualizer, device, masks
        )
        n = len(out["embeddings"])
        if n == 0:
            continue
        all_emb.append(out["embeddings"])
        all_an.extend(out["animacy"].tolist())
        all_sc.extend(out["scene"].tolist())
        all_has_an.extend(out["has_animacy"].tolist())
        all_has_sc.extend(out["has_scene"].tolist())
        all_stems.extend(out["stems"])
        total_epochs += n
        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info(
                "  file %d/%d, %d epochs so far", idx + 1, len(h5_files), total_epochs
            )

    if not all_emb:
        return {
            "embeddings": np.empty((0, 512), dtype=np.float32),
            "animacy": np.array([], dtype=np.int64),
            "scene": np.array([], dtype=np.int64),
            "has_animacy": np.array([], dtype=bool),
            "has_scene": np.array([], dtype=bool),
            "stems": np.array([], dtype="U"),
        }
    return {
        "embeddings": np.concatenate(all_emb, axis=0),
        "animacy": np.array(all_an, dtype=np.int64),
        "scene": np.array(all_sc, dtype=np.int64),
        "has_animacy": np.array(all_has_an, dtype=bool),
        "has_scene": np.array(all_has_sc, dtype=bool),
        "stems": np.array(all_stems, dtype="U"),
    }


def _fit_knn_probe(
    train_emb: np.ndarray,
    train_lbl: np.ndarray,
    eval_emb: np.ndarray,
    eval_lbl: np.ndarray,
    k: int = 10,
) -> dict[str, Any]:
    """Fit kNN probe with balanced train/eval sets."""
    if len(train_emb) == 0 or len(eval_emb) == 0:
        raise RuntimeError(
            f"kNN probe received empty split: "
            f"n_train={len(train_emb)}, n_eval={len(eval_emb)}. "
            "Empty split after stem-match is a bug."
        )
    classes, counts = np.unique(train_lbl, return_counts=True)
    if len(classes) < 2:
        raise RuntimeError(
            f"kNN probe training set has only one class: {classes.tolist()}. "
            "Check label extraction and subject split."
        )
    min_count = int(counts.min())
    rng = np.random.default_rng(42)
    sel_idx: list[np.ndarray] = []
    for c in classes:
        idx = np.where(train_lbl == c)[0]
        sel_idx.append(rng.choice(idx, size=min_count, replace=False))
    sel = np.concatenate(sel_idx)
    train_emb = train_emb[sel]
    train_lbl = train_lbl[sel]

    classes_e, counts_e = np.unique(eval_lbl, return_counts=True)
    if len(classes_e) >= 2:
        min_e = int(counts_e.min())
        rng_e = np.random.default_rng(99)
        sel_e: list[np.ndarray] = []
        for c in classes_e:
            idx = np.where(eval_lbl == c)[0]
            sel_e.append(rng_e.choice(idx, size=min_e, replace=False))
        sel_e_all = np.concatenate(sel_e)
        eval_emb = eval_emb[sel_e_all]
        eval_lbl = eval_lbl[sel_e_all]

    scaler = StandardScaler().fit(train_emb)
    train_s = scaler.transform(train_emb)
    eval_s = scaler.transform(eval_emb)

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_s, train_lbl)
    pred = knn.predict(eval_s)
    acc = float((pred == eval_lbl).mean())
    f1 = float(f1_score(eval_lbl, pred, average="macro"))
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "n_train": int(len(train_lbl)),
        "n_eval": int(len(eval_lbl)),
        "train_class_counts": np.bincount(train_lbl).tolist(),
        "eval_class_counts": np.bincount(eval_lbl).tolist(),
    }


def phase_extract(
    source_dir: Path,
    output_dir: Path,
    checkpoint: Path | None,
    device: torch.device,
    limit: int | None,
    random_init: bool,
) -> None:
    """Phase A: extract frozen BENDR embeddings and cache to disk.

    Cache includes one h5 stem per embedding row so phase B can reconstruct
    subject-level train/eval splits without re-scanning h5 files.
    """
    t0 = time.time()
    prefix = "random_init" if random_init else "pretrained"
    cache_path = output_dir / f"embeddings_{prefix}.npz"
    meta_path = output_dir / f"embeddings_{prefix}.meta.json"

    current_meta: dict[str, Any] = {
        "source_dir": str(source_dir),
        "checkpoint": str(checkpoint) if checkpoint else "huggingface",
        "bendr_channels": list(BENDR_CHANNELS),
        "limit": limit,
        "git_sha": _git_sha(),
    }

    if cache_path.exists():
        if meta_path.exists():
            with meta_path.open() as mf:
                saved_meta = json.load(mf)
            # Validate invariant fields; git_sha intentionally excluded.
            for key in ("source_dir", "checkpoint", "bendr_channels", "limit"):
                if saved_meta.get(key) != current_meta[key]:
                    raise RuntimeError(
                        f"Cache metadata mismatch for '{key}': "
                        f"saved={saved_meta.get(key)!r}, "
                        f"current={current_meta[key]!r}. "
                        f"Delete {cache_path} or pass --force-reextract."
                    )
        logger.info("Embedding cache already exists: %s (skip extract)", cache_path)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading BENDR (%s)...", prefix)
    if random_init:
        # Construct fresh models with default random init — do NOT call
        # load_bendr to avoid partial overwrite bugs (weight_norm Conv1d,
        # GroupNorm/LayerNorm affine params would survive xavier re-init).
        encoder = _ConvEncoderBENDR(in_features=20, encoder_h=512).to(device).eval()
        contextualizer = _BENDRContextualizer(in_features=512).to(device).eval()
        for p in list(encoder.parameters()) + list(contextualizer.parameters()):
            p.requires_grad_(False)
        logger.info("Random-init BENDR constructed (no checkpoint loaded)")
    else:
        encoder, contextualizer, _cfg = load_bendr(checkpoint=checkpoint, device=device)

    tag_to_idx = _load_hed_vectorizer(source_dir)
    masks = _build_label_masks(tag_to_idx)

    h5_files = sorted(source_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No h5 files found in {source_dir}")
    logger.info("Found %d h5 files in %s", len(h5_files), source_dir)

    logger.info("Extracting embeddings (limit=%s)...", limit)
    all_out = _collect_embeddings(
        h5_files, encoder, contextualizer, device, masks, limit=limit
    )
    n_emb = len(all_out["embeddings"])
    if n_emb == 0:
        raise RuntimeError(
            "No embeddings extracted. Check passive-movie task filter and HED tags."
        )
    logger.info(
        "Extracted %d embeddings (dim=%d) in %.1f s",
        n_emb,
        all_out["embeddings"].shape[1],
        time.time() - t0,
    )
    np.savez_compressed(
        str(cache_path),
        embeddings=all_out["embeddings"],
        animacy=all_out["animacy"],
        scene=all_out["scene"],
        has_animacy=all_out["has_animacy"],
        has_scene=all_out["has_scene"],
        stems=all_out["stems"],
    )
    with meta_path.open("w") as mf:
        json.dump(current_meta, mf, indent=2)
    logger.info("Saved embeddings to %s (meta: %s)", cache_path, meta_path)


def phase_probe(
    source_dir: Path,
    output_dir: Path,
    seeds: list[int],
    probe_types: list[str],
    holdout_ratio: float,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Phase B: fit probes on cached embeddings, write results JSON."""
    results: dict[str, Any] = {}

    missing_caches = [
        output_dir / f"embeddings_{prefix}.npz"
        for prefix in ("pretrained", "random_init")
        if not (output_dir / f"embeddings_{prefix}.npz").exists()
    ]
    if missing_caches and not allow_partial:
        raise RuntimeError(
            f"Missing embedding caches (run phase extract first, or pass "
            f"--allow-partial to skip): {[str(p) for p in missing_caches]}"
        )
    if missing_caches:
        logger.warning("Missing caches (allow-partial): %s", missing_caches)

    for prefix in ("pretrained", "random_init"):
        cache_path = output_dir / f"embeddings_{prefix}.npz"
        if not cache_path.exists():
            continue

        logger.info("Loading cached embeddings from %s", cache_path)
        data = np.load(str(cache_path), allow_pickle=False)
        emb_all = data["embeddings"]
        animacy_all = data["animacy"]
        scene_all = data["scene"]
        has_animacy_all = data["has_animacy"]
        has_scene_all = data["has_scene"]
        stems_all: np.ndarray = data["stems"]  # per-embedding h5 stem strings
        logger.info("Loaded %d embeddings from %s", len(emb_all), cache_path)

        seed_results: list[dict[str, Any]] = []
        for seed in seeds:
            logger.info("=== Seed %d ===", seed)
            train_files, eval_files = held_out_subjects(
                source_dir, ratio=holdout_ratio, seed=seed
            )
            train_stems = {f.stem for f in train_files}
            eval_stems = {f.stem for f in eval_files}

            # Partition by stem membership
            train_mask = np.array([s in train_stems for s in stems_all], dtype=bool)
            eval_mask = np.array([s in eval_stems for s in stems_all], dtype=bool)

            n_train = int(train_mask.sum())
            n_eval = int(eval_mask.sum())
            logger.info("Split: %d train, %d eval epochs", n_train, n_eval)

            if n_train == 0 or n_eval == 0:
                raise RuntimeError(
                    f"Seed {seed}: empty split after stem-match "
                    f"(n_train={n_train}, n_eval={n_eval}). "
                    "Stems in cache do not match h5 files in source_dir."
                )

            train_emb = emb_all[train_mask]
            eval_emb = emb_all[eval_mask]
            train_an = animacy_all[train_mask]
            eval_an = animacy_all[eval_mask]
            train_sc = scene_all[train_mask]
            eval_sc = scene_all[eval_mask]
            train_has_an = has_animacy_all[train_mask]
            eval_has_an = has_animacy_all[eval_mask]
            train_has_sc = has_scene_all[train_mask]
            eval_has_sc = has_scene_all[eval_mask]

            probe_res: dict[str, Any] = {}

            if "lr" in probe_types:
                an_lr = _fit_probe(
                    train_emb[train_has_an],
                    train_an[train_has_an],
                    eval_emb[eval_has_an],
                    eval_an[eval_has_an],
                )
                sc_lr = _fit_probe(
                    train_emb[train_has_sc],
                    train_sc[train_has_sc],
                    eval_emb[eval_has_sc],
                    eval_sc[eval_has_sc],
                )
                probe_res["lr"] = {"animacy": an_lr, "scene": sc_lr}
                logger.info("  LR animacy: %s", an_lr)
                logger.info("  LR scene:   %s", sc_lr)

            if "knn" in probe_types:
                an_knn = _fit_knn_probe(
                    train_emb[train_has_an],
                    train_an[train_has_an],
                    eval_emb[eval_has_an],
                    eval_an[eval_has_an],
                )
                sc_knn = _fit_knn_probe(
                    train_emb[train_has_sc],
                    train_sc[train_has_sc],
                    eval_emb[eval_has_sc],
                    eval_sc[eval_has_sc],
                )
                probe_res["knn"] = {"animacy": an_knn, "scene": sc_knn}
                logger.info("  kNN animacy: %s", an_knn)
                logger.info("  kNN scene:   %s", sc_knn)

            seed_results.append(
                {
                    "seed": seed,
                    "n_train": n_train,
                    "n_eval": n_eval,
                    **probe_res,
                }
            )

        results[prefix] = seed_results

    out_path = output_dir / "probe_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote probe results to %s", out_path)
    return results


def run(
    source_dir: Path,
    output_dir: Path,
    checkpoint: Path | None,
    seeds: list[int],
    probe_types: list[str],
    holdout_ratio: float,
    batch_size: int,
    device: torch.device,
    phase: str,
    limit: int | None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Main entry point for both extract and probe phases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    if phase in ("extract", "both"):
        for random_init in (False, True):
            logger.info(
                "=== Phase A: extract (%s) ===",
                "random_init" if random_init else "pretrained",
            )
            phase_extract(
                source_dir=source_dir,
                output_dir=output_dir,
                checkpoint=checkpoint,
                device=device,
                limit=limit,
                random_init=random_init,
            )

    if phase in ("probe", "both"):
        logger.info("=== Phase B: probe ===")
        results = phase_probe(
            source_dir=source_dir,
            output_dir=output_dir,
            seeds=seeds,
            probe_types=probe_types,
            holdout_ratio=holdout_ratio,
            allow_partial=allow_partial,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("${HBN_DATA_DIR}/preprocessed"),
        help="preprocessed h5 directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/baseline_bendr"),
        help="Output directory for embeddings cache and results JSON",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to BENDR pytorch_model.bin (downloads from HuggingFace if omitted)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,13,7",
        help="Comma-separated random seeds for subject splits",
    )
    parser.add_argument(
        "--probes",
        type=str,
        default="lr,knn",
        help="Comma-separated probe types: lr, knn",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.15,
        help="Fraction of subjects held out for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (unused, kept for CLI API parity with other FM scripts)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, cpu, mps",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="both",
        choices=["extract", "probe", "both"],
        help="Which phase to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit total epochs extracted (for smoke-test). None = all.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        default=False,
        help="Allow probe phase to run even if only one variant cache exists.",
    )
    args = parser.parse_args()

    dm = DeviceManager(device_type=args.device)
    device = dm.device
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    probe_types = [p.strip() for p in args.probes.split(",")]

    logger.info("Device: %s", device)
    logger.info("Seeds: %s", seeds)
    logger.info("Probe types: %s", probe_types)

    run(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        seeds=seeds,
        probe_types=probe_types,
        holdout_ratio=args.holdout_ratio,
        batch_size=args.batch_size,
        device=device,
        phase=args.phase,
        limit=args.limit,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
