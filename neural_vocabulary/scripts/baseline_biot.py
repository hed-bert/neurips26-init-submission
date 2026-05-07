""" Gate Lit: BIOT baseline probe on HBN passive-movie epochs.

Two-phase pipeline:

Phase A — Embedding extraction
    Load frozen BIOT encoder (pretrained or random-init), iterate over
    preprocessed H5 files, compute bipolar derivations + resample + pad,
    extract embeddings, cache to disk as NPZ.

Phase B — Probe fitting
    Load cached NPZ, fit LR (C=1.0) + kNN (k=10) probes on animacy
    and scene labels.  Results written as JSON per seed.

Checkpoint: EEG-PREST-16-channels.ckpt (MIT license)
    https://github.com/ycq091044/BIOT

Usage (full run):
    uv run python -m neural_vocabulary.scripts.baseline_biot \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir runs/gates/v10_gateLit/biot \\
        --checkpoint /path/to/EEG-PREST-16-channels.ckpt \\
        --seeds 42,13,7

Smoke-test (tiny run, limit 4 files):
    uv run python -m neural_vocabulary.scripts.baseline_biot \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir /tmp/biot_smoke \\
        --checkpoint /path/to/EEG-PREST-16-channels.ckpt \\
        --limit 4 --seeds 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from neural_vocabulary.baselines.biot_adapter import BIOTAdapter
from neural_vocabulary.evaluation.splits import held_out_subjects
from neural_vocabulary.scripts.e2_literature_probes import (
    _animacy_label,
    _build_label_masks,
    _fit_probe,
    _scene_label,
)
from neural_vocabulary.training.device_manager import DeviceManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PASSIVE_MOVIE_TASKS = (
    "DespicableMe",
    "DiaryOfAWimpyKid",
    "FunwithFractals",
    "ThePresent",
)


# ---------------------------------------------------------------------------
# Phase A: embedding extraction
# ---------------------------------------------------------------------------


def _is_passive_movie(stem: str) -> bool:
    """Return True if the H5 file stem belongs to a passive-movie task."""
    parts = stem.split("_", 1)
    task = parts[1] if len(parts) > 1 else ""
    return any(task.startswith(t) for t in PASSIVE_MOVIE_TASKS)


def _read_channel_names(h5_path: Path) -> list[str]:
    """Parse the JSON-encoded channel_names attr from a preprocessed H5 file."""
    import json as _json

    with h5py.File(h5_path, "r") as f:
        raw = f.attrs.get("channel_names", "[]")
        if isinstance(raw, np.ndarray):
            raw = "".join(str(c) for c in raw)
        elif not isinstance(raw, str):
            raw = str(raw)
        return _json.loads(raw)


def extract_embeddings(
    source_dir: Path,
    output_dir: Path,
    adapter: BIOTAdapter,
    vectorizer_path: Path,
    file_stems: set[str],
    batch_size: int,
    limit: int | None,
    split_name: str,
) -> Path:
    """Extract BIOT embeddings for *file_stems* and cache as NPZ.

    Returns path to the cached NPZ file.
    """
    cache_path = output_dir / f"embeddings_{split_name}.npz"
    if cache_path.exists():
        logger.info("Embedding cache exists: %s — skipping extraction", cache_path)
        return cache_path

    # Load HED vectorizer for label computation
    logger.info("Loading HED vectorizer from %s", vectorizer_path)
    vect = torch.load(vectorizer_path, map_location="cpu", weights_only=False)
    tag_to_idx: dict[str, int] = vect["tag_to_idx"]
    masks = _build_label_masks(tag_to_idx)

    # Collect eligible H5 files
    all_h5 = sorted(source_dir.glob("*.h5"))
    h5_files = [f for f in all_h5 if f.stem in file_stems and _is_passive_movie(f.stem)]
    if not h5_files:
        raise RuntimeError(
            f"No passive-movie H5 files found for {split_name} split in {source_dir}. "
            f"file_stems sample: {list(file_stems)[:5]}"
        )
    if limit is not None:
        h5_files = h5_files[:limit]
    logger.info("Processing %d H5 files for split=%s", len(h5_files), split_name)

    # Build channel mapping from the first file (all preprocessed files share the same montage)
    channel_names = _read_channel_names(h5_files[0])
    adapter_channel_names = channel_names  # verify mapping once
    logger.info(
        "Verifying bipolar channel mapping on %d source channels",
        len(adapter_channel_names),
    )
    # This will raise ValueError if any electrode is missing
    from neural_vocabulary.baselines.biot_adapter import (
        build_bipolar_indices,
    )

    build_bipolar_indices(adapter_channel_names)  # raises if missing

    all_embs: list[np.ndarray] = []
    all_animacy: list[int] = []
    all_scene: list[int] = []
    all_has_animacy: list[bool] = []
    all_has_scene: list[bool] = []

    eeg_buf: list[np.ndarray] = []

    def _flush_buffer() -> None:
        nonlocal eeg_buf
        if not eeg_buf:
            return
        batch_arr = np.stack(eeg_buf, axis=0)  # (B, 16, 2000)
        embs = adapter.embed(batch_arr)
        all_embs.extend(embs)
        eeg_buf = []

    n_skipped_no_hed = 0
    n_skipped_no_label = 0

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                n_epochs = int(f.attrs.get("n_epochs", 0))
                for epoch_key in [f"epoch_{i:05d}" for i in range(n_epochs)]:
                    if epoch_key not in f:
                        continue
                    grp = f[epoch_key]
                    if "hed_vector" not in grp:
                        n_skipped_no_hed += 1
                        continue
                    hed = grp["hed_vector"][:].astype(np.float32)
                    an = _animacy_label(hed, masks)
                    sc = _scene_label(hed, masks)
                    if an is None and sc is None:
                        n_skipped_no_label += 1
                        continue

                    eeg = grp["eeg"][:].astype(np.float32)  # (64, T)
                    preprocessed = adapter.preprocess(eeg)  # (16, 2000)
                    eeg_buf.append(preprocessed)

                    # Record labels (flush will append matching embeddings)
                    all_has_animacy.append(an is not None)
                    all_has_scene.append(sc is not None)
                    all_animacy.append(-1 if an is None else an)
                    all_scene.append(-1 if sc is None else sc)

                    if len(eeg_buf) >= batch_size:
                        _flush_buffer()
        except (OSError, KeyError) as e:
            logger.error("Failed to read %s: %s", h5_path, e)
            raise

    _flush_buffer()

    if not all_embs:
        raise RuntimeError(
            f"Zero embeddings extracted for split={split_name}. "
            f"Files: {len(h5_files)}, no_hed={n_skipped_no_hed}, "
            f"no_label={n_skipped_no_label}"
        )

    logger.info(
        "Extracted %d embeddings (no_hed=%d, no_label=%d)",
        len(all_embs),
        n_skipped_no_hed,
        n_skipped_no_label,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=np.array(all_embs, dtype=np.float32),
        animacy=np.array(all_animacy, dtype=np.int32),
        scene=np.array(all_scene, dtype=np.int32),
        has_animacy=np.array(all_has_animacy),
        has_scene=np.array(all_has_scene),
    )
    logger.info("Saved embedding cache: %s", cache_path)
    return cache_path


# ---------------------------------------------------------------------------
# Phase B: probe fitting
# ---------------------------------------------------------------------------


def _fit_knn(
    train_emb: np.ndarray,
    train_lbl: np.ndarray,
    eval_emb: np.ndarray,
    eval_lbl: np.ndarray,
    k: int = 10,
    balance_train: bool = True,
) -> dict[str, Any]:
    """kNN probe (k=10). Returns accuracy + macro F1."""
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    if len(train_emb) == 0 or len(eval_emb) == 0:
        return {
            "error": "empty_set",
            "n_train": len(train_emb),
            "n_eval": len(eval_emb),
        }

    if balance_train:
        classes, counts = np.unique(train_lbl, return_counts=True)
        if len(classes) < 2:
            return {"error": "one_class_only", "classes": classes.tolist()}
        min_count = int(counts.min())
        rng = np.random.default_rng(42)
        selected = []
        for c in classes:
            idx = np.where(train_lbl == c)[0]
            selected.append(rng.choice(idx, size=min_count, replace=False))
        sel = np.concatenate(selected)
        train_emb = train_emb[sel]
        train_lbl = train_lbl[sel]

    # Balance eval
    classes, counts = np.unique(eval_lbl, return_counts=True)
    if len(classes) >= 2:
        min_count = int(counts.min())
        rng = np.random.default_rng(99)
        selected = []
        for c in classes:
            idx = np.where(eval_lbl == c)[0]
            selected.append(rng.choice(idx, size=min_count, replace=False))
        sel = np.concatenate(selected)
        eval_emb = eval_emb[sel]
        eval_lbl = eval_lbl[sel]

    scaler = StandardScaler().fit(train_emb)
    train_s = scaler.transform(train_emb)
    eval_s = scaler.transform(eval_emb)

    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(train_s, train_lbl)
    pred = clf.predict(eval_s)
    acc = float((pred == eval_lbl).mean())
    f1 = float(f1_score(eval_lbl, pred, average="macro"))
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "n_train": int(len(train_lbl)),
        "n_eval": int(len(eval_lbl)),
        "k": k,
        "train_class_counts": np.bincount(train_lbl).tolist(),
        "eval_class_counts": np.bincount(eval_lbl).tolist(),
    }


def run_probes(
    train_cache: Path,
    eval_cache: Path,
    output_dir: Path,
    seed: int,
    probes: list[str],
    variant: str,
) -> dict[str, Any]:
    """Load cached embeddings and fit requested probes.

    Parameters
    ----------
    train_cache, eval_cache:
        NPZ files produced by *extract_embeddings*.
    output_dir:
        Directory to write the JSON result.
    seed:
        Random seed used for this split.
    probes:
        Subset of ["lr", "knn"].
    variant:
        "pretrained" or "random_init".

    Returns
    -------
    summary dict (also written as JSON).
    """
    train_data = np.load(train_cache)
    eval_data = np.load(eval_cache)

    summary: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "n_train_emb": int(len(train_data["embeddings"])),
        "n_eval_emb": int(len(eval_data["embeddings"])),
        "probes": {},
    }

    for probe_name in ("animacy", "scene"):
        has_key = f"has_{probe_name}"
        lbl_key = probe_name

        train_mask = train_data[has_key]
        eval_mask = eval_data[has_key]
        train_emb = train_data["embeddings"][train_mask]
        train_lbl = train_data[lbl_key][train_mask]
        eval_emb = eval_data["embeddings"][eval_mask]
        eval_lbl = eval_data[lbl_key][eval_mask]

        probe_results: dict[str, Any] = {}
        if "lr" in probes:
            probe_results["lr"] = _fit_probe(train_emb, train_lbl, eval_emb, eval_lbl)
        if "knn" in probes:
            probe_results["knn"] = _fit_knn(train_emb, train_lbl, eval_emb, eval_lbl)

        summary["probes"][probe_name] = probe_results

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{variant}_seed{seed}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote probe results: %s", out_path)

    for probe_name, probe_res in summary["probes"].items():
        for probe_type, res in probe_res.items():
            if "accuracy" in res:
                logger.info(
                    "%s %s %s: acc=%.3f f1=%.3f (n_eval=%d)",
                    variant,
                    probe_name,
                    probe_type,
                    res["accuracy"],
                    res["macro_f1"],
                    res.get("n_eval", -1),
                )

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("${HBN_DATA_DIR}/preprocessed"),
        help="preprocessed preprocessed H5 directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/gates/v10_gateLit/biot"),
        help="Directory for NPZ caches and JSON results",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to EEG-PREST-16-channels.ckpt (omit for random-init only)",
    )
    parser.add_argument(
        "--vectorizer",
        type=Path,
        default=None,
        help="Path to hed_vectorizer.pt (default: source-dir/hed_vectorizer.pt)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,13,7",
        help="Comma-separated random seeds for subject splits",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.15,
        help="Fraction of subjects held out for evaluation",
    )
    parser.add_argument(
        "--probes",
        type=str,
        default="lr,knn",
        help="Comma-separated probe types: lr,knn",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, mps, cpu",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of H5 files per split (smoke-test mode)",
    )
    parser.add_argument(
        "--random-init-only",
        action="store_true",
        help="Skip pretrained checkpoint; only run random-init ablation",
    )
    args = parser.parse_args()

    dm = DeviceManager(device_type=args.device)
    device = dm.device
    logger.info("Device: %s", device)

    vectorizer_path = args.vectorizer or (args.source_dir / "hed_vectorizer.pt")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"HED vectorizer not found: {vectorizer_path}")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    probes = [p.strip() for p in args.probes.split(",")]

    # Build channel names from the first H5 in the directory
    sample_h5 = next(args.source_dir.glob("*.h5"), None)
    if sample_h5 is None:
        raise FileNotFoundError(f"No H5 files in {args.source_dir}")
    channel_names = _read_channel_names(sample_h5)
    logger.info("Source montage: %d channels", len(channel_names))

    variants: list[tuple[str, BIOTAdapter]] = []

    if not args.random_init_only:
        if args.checkpoint is None:
            raise ValueError(
                "--checkpoint is required unless --random-init-only is set"
            )
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        pretrained = BIOTAdapter.from_checkpoint(
            checkpoint_path=args.checkpoint,
            channel_names=channel_names,
            device=device,
            source_sfreq=100,
        )
        variants.append(("pretrained", pretrained))

    random_adapter = BIOTAdapter.random_init(
        channel_names=channel_names,
        device=device,
        source_sfreq=100,
    )
    variants.append(("random_init", random_adapter))

    all_results: list[dict] = []

    for seed in seeds:
        logger.info("=== Seed %d ===", seed)
        train_files, eval_files = held_out_subjects(
            args.source_dir, ratio=args.holdout_ratio, seed=seed
        )
        train_stems = {f.stem for f in train_files}
        eval_stems = {f.stem for f in eval_files}

        for variant_name, adapter in variants:
            variant_dir = args.output_dir / variant_name / f"seed{seed}"
            variant_dir.mkdir(parents=True, exist_ok=True)

            train_cache = extract_embeddings(
                source_dir=args.source_dir,
                output_dir=variant_dir,
                adapter=adapter,
                vectorizer_path=vectorizer_path,
                file_stems=train_stems,
                batch_size=args.batch_size,
                limit=args.limit,
                split_name="train",
            )
            eval_cache = extract_embeddings(
                source_dir=args.source_dir,
                output_dir=variant_dir,
                adapter=adapter,
                vectorizer_path=vectorizer_path,
                file_stems=eval_stems,
                batch_size=args.batch_size,
                limit=args.limit,
                split_name="eval",
            )

            result = run_probes(
                train_cache=train_cache,
                eval_cache=eval_cache,
                output_dir=variant_dir,
                seed=seed,
                probes=probes,
                variant=variant_name,
            )
            all_results.append(result)

    # Write aggregate summary across all seeds and variants
    summary_path = args.output_dir / "summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Aggregate summary: %s", summary_path)

    # Print headline numbers
    for res in all_results:
        for probe_name, probe_res in res["probes"].items():
            for probe_type, metrics in probe_res.items():
                if "accuracy" in metrics:
                    print(
                        f"{res['variant']} seed={res['seed']} "
                        f"{probe_name}/{probe_type}: "
                        f"acc={metrics['accuracy']:.3f} "
                        f"f1={metrics['macro_f1']:.3f}"
                    )


if __name__ == "__main__":
    main()
