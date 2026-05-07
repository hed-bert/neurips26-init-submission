""" Gate Lit: LaBraM frozen-embedding probe on HBN passive-movie.

Evaluates LaBraM-Base (Jiang et al., ICLR 2024) on the HBN animacy + scene
probe using frozen embeddings extracted from preprocessed preprocessed data.

Two-phase pipeline
------------------
Phase A — Extract frozen embeddings
    For each seed, load held-out subject split, iterate through passive-movie
    epochs, run LaBraM forward, save embeddings + HED vectors to disk.

Phase B — Fit probes
    Load cached embeddings. Fit LogisticRegression (C=1.0) and kNN (k=10)
    on animacy and scene labels derived from HED vectors. Balanced eval.
    Both pretrained and random-init variants are run.

Label derivation
    Imported directly from e2_literature_probes:
        _animacy_label, _scene_label, _build_label_masks

Output
    Per-seed JSON + summary JSON in --output-dir.

Usage (smoke-test, 4 files):
    uv run python -m neural_vocabulary.scripts.baseline_labram \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir runs/gates/v10_gateLit/labram \\
        --limit 4

Full run (submit via gpu_queue):
    uv run python -m neural_vocabulary.scripts.baseline_labram \\
        --source-dir ${HBN_DATA_DIR}/preprocessed \\
        --output-dir runs/gates/v10_gateLit/labram \\
        --seeds 42,13,7 \\
        --device auto

NOTE: If no LaBraM checkpoint is available, pass --random-init-only to
      run only the random-init baseline (harness verification). A real run
      with a downloaded checkpoint should be submitted as a queue job by
      the lead after merging.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from neural_vocabulary.baselines.labram_adapter import (
    HBN_CHANNEL_NAMES,
    LABRAM_EMBED_DIM,
    HBNLaBraMDataset,
    extract_embeddings_batch,
    get_channel_names_from_h5,
    load_labram_model,
)
from neural_vocabulary.evaluation.splits import held_out_subjects
from neural_vocabulary.scripts.e2_literature_probes import (
    _animacy_label,
    _build_label_masks,
    _scene_label,
)
from neural_vocabulary.training.device_manager import DeviceManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PASSIVE_MOVIE_TASKS: frozenset[str] = frozenset(
    [
        "DespicableMe",
        "DiaryOfAWimpyKid",
        "FunwithFractals",
        "ThePresent",
    ]
)


def _filter_passive_movie(files: list[Path]) -> list[Path]:
    """Keep only passive-movie task files."""
    out = []
    for f in files:
        parts = f.stem.split("_")
        task = parts[1] if len(parts) >= 2 else ""
        if task in PASSIVE_MOVIE_TASKS:
            out.append(f)
    return out


def _load_hed_vectorizer(source_dir: Path) -> dict[str, int]:
    """Load tag_to_idx from hed_vectorizer.pt in the source directory."""
    vect_path = source_dir / "hed_vectorizer.pt"
    if not vect_path.exists():
        raise FileNotFoundError(
            f"HED vectorizer not found at {vect_path}. "
            "Expected hed_vectorizer.pt in the preprocessed source directory."
        )
    vect = torch.load(str(vect_path), map_location="cpu", weights_only=False)
    return vect["tag_to_idx"]


def _extract_embeddings(
    model: torch.nn.Module,
    h5_files: list[Path],
    device: torch.device,
    batch_size: int,
    masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract embeddings and HED-derived labels from file list.

    Returns dict with keys:
        embeddings: (N, embed_dim)
        animacy: (N,) int8, -1=skip
        scene: (N,) int8, -1=skip
        has_animacy: (N,) bool
        has_scene: (N,) bool
    """
    dataset = HBNLaBraMDataset(h5_files, passive_only=True)
    if len(dataset) == 0:
        return {
            "embeddings": np.empty((0, LABRAM_EMBED_DIM), dtype=np.float32),
            "animacy": np.empty(0, dtype=np.int8),
            "scene": np.empty(0, dtype=np.int8),
            "has_animacy": np.empty(0, dtype=bool),
            "has_scene": np.empty(0, dtype=bool),
        }

    # Infer channel names from first available file
    ch_names = HBN_CHANNEL_NAMES
    for f in h5_files:
        task = f.stem.split("_")[1] if "_" in f.stem else ""
        if task in PASSIVE_MOVIE_TASKS:
            ch_names = get_channel_names_from_h5(f)
            break

    all_embs: list[np.ndarray] = []
    all_animacy: list[int] = []
    all_scene: list[int] = []
    all_has_animacy: list[bool] = []
    all_has_scene: list[bool] = []

    batch_iter = dataset.iter_batches(batch_size=batch_size)
    n_batches = len(batch_iter)

    for b_idx, (eeg_np, hed_np) in enumerate(batch_iter):
        if b_idx % 20 == 0:
            logger.info("  batch %d/%d", b_idx, n_batches)

        eeg_t = torch.from_numpy(eeg_np).float().to(device)
        embs = extract_embeddings_batch(model, eeg_t, ch_names=ch_names)

        for i in range(eeg_np.shape[0]):
            hed = hed_np[i]
            an = _animacy_label(hed, masks)
            sc = _scene_label(hed, masks)
            all_embs.append(embs[i])
            all_has_animacy.append(an is not None)
            all_has_scene.append(sc is not None)
            all_animacy.append(-1 if an is None else an)
            all_scene.append(-1 if sc is None else sc)

    return {
        "embeddings": np.array(all_embs, dtype=np.float32),
        "animacy": np.array(all_animacy, dtype=np.int8),
        "scene": np.array(all_scene, dtype=np.int8),
        "has_animacy": np.array(all_has_animacy, dtype=bool),
        "has_scene": np.array(all_has_scene, dtype=bool),
    }


def _save_embeddings(
    cache_path: Path, data: dict[str, np.ndarray], meta: dict[str, Any]
) -> None:
    """Save embeddings and labels to H5 cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "w") as f:
        for key, arr in data.items():
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        for k, v in meta.items():
            f.attrs[k] = v if not isinstance(v, Path) else str(v)


def _load_embeddings(cache_path: Path) -> dict[str, np.ndarray]:
    """Load embeddings from H5 cache."""
    with h5py.File(cache_path, "r") as f:
        return {k: f[k][:] for k in f}


def _fit_lr_probe(
    train_emb: np.ndarray,
    train_lbl: np.ndarray,
    eval_emb: np.ndarray,
    eval_lbl: np.ndarray,
    balance: bool = True,
) -> dict[str, Any]:
    """Fit StandardScaler + LR (C=1.0); balanced eval. Returns metrics dict."""
    if len(train_emb) == 0 or len(eval_emb) == 0:
        return {
            "error": "empty_set",
            "n_train": len(train_emb),
            "n_eval": len(eval_emb),
        }

    if balance:
        train_emb, train_lbl = _balance_classes(train_emb, train_lbl, seed=42)

    eval_emb, eval_lbl = _balance_classes(eval_emb, eval_lbl, seed=99)

    classes = np.unique(train_lbl)
    if len(classes) < 2:
        return {"error": "one_class_only", "classes": classes.tolist()}

    scaler = StandardScaler().fit(train_emb)
    tr_s = scaler.transform(train_emb)
    ev_s = scaler.transform(eval_emb)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(tr_s, train_lbl)
    pred = clf.predict(ev_s)

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


def _fit_knn_probe(
    train_emb: np.ndarray,
    train_lbl: np.ndarray,
    eval_emb: np.ndarray,
    eval_lbl: np.ndarray,
    k: int = 10,
) -> dict[str, Any]:
    """Fit kNN (k=10) on z-scored embeddings; balanced eval."""
    if len(train_emb) == 0 or len(eval_emb) == 0:
        return {
            "error": "empty_set",
            "n_train": len(train_emb),
            "n_eval": len(eval_emb),
        }

    eval_emb, eval_lbl = _balance_classes(eval_emb, eval_lbl, seed=99)

    classes = np.unique(train_lbl)
    if len(classes) < 2:
        return {"error": "one_class_only", "classes": classes.tolist()}

    scaler = StandardScaler().fit(train_emb)
    tr_s = scaler.transform(train_emb)
    ev_s = scaler.transform(eval_emb)

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(tr_s, train_lbl)
    pred = knn.predict(ev_s)

    acc = float((pred == eval_lbl).mean())
    f1 = float(f1_score(eval_lbl, pred, average="macro"))
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "n_train": int(len(train_lbl)),
        "n_eval": int(len(eval_lbl)),
        "k": k,
    }


def _balance_classes(
    emb: np.ndarray, lbl: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample majority class to minority class count."""
    classes, counts = np.unique(lbl, return_counts=True)
    if len(classes) < 2:
        return emb, lbl
    min_count = int(counts.min())
    rng = np.random.default_rng(seed)
    selected = []
    for c in classes:
        idx = np.where(lbl == c)[0]
        selected.append(rng.choice(idx, size=min_count, replace=False))
    sel = np.concatenate(selected)
    return emb[sel], lbl[sel]


def _run_probes(
    train_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    probe_types: list[str],
) -> dict[str, Any]:
    """Run LR and/or kNN probes on animacy and scene labels."""
    results: dict[str, Any] = {}

    for label in ("animacy", "scene"):
        has_key = f"has_{label}"
        tr_mask = train_data[has_key]
        ev_mask = eval_data[has_key]
        tr_emb = train_data["embeddings"][tr_mask]
        tr_lbl = train_data[label][tr_mask].astype(int)
        ev_emb = eval_data["embeddings"][ev_mask]
        ev_lbl = eval_data[label][ev_mask].astype(int)

        label_results: dict[str, Any] = {
            "n_train_with_label": int(tr_mask.sum()),
            "n_eval_with_label": int(ev_mask.sum()),
        }

        if "lr" in probe_types:
            lr_result = _fit_lr_probe(tr_emb, tr_lbl, ev_emb, ev_lbl)
            if "error" in lr_result:
                logger.warning(
                    "LR probe for '%s' failed: %s (train=%d, eval=%d)",
                    label,
                    lr_result["error"],
                    int(tr_mask.sum()),
                    int(ev_mask.sum()),
                )
            label_results["lr"] = lr_result
        if "knn" in probe_types:
            knn_result = _fit_knn_probe(tr_emb, tr_lbl, ev_emb, ev_lbl)
            if "error" in knn_result:
                logger.warning(
                    "kNN probe for '%s' failed: %s (train=%d, eval=%d)",
                    label,
                    knn_result["error"],
                    int(tr_mask.sum()),
                    int(ev_mask.sum()),
                )
            label_results["knn"] = knn_result

        results[label] = label_results

    return results


def run_one_seed(
    source_dir: Path,
    output_dir: Path,
    seed: int,
    checkpoint: str | None,
    probe_types: list[str],
    holdout_ratio: float,
    batch_size: int,
    device: torch.device,
    limit: int | None,
    random_init_only: bool,
    overwrite: bool,
) -> dict[str, Any]:
    """Run full pipeline for one seed: extract + probe (pretrained + random-init)."""
    logger.info("=== Seed %d ===", seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load HED masks once
    tag_to_idx = _load_hed_vectorizer(source_dir)
    masks = _build_label_masks(tag_to_idx)

    # Build held-out split
    train_files, eval_files = held_out_subjects(
        source_dir, ratio=holdout_ratio, seed=seed
    )
    train_files = _filter_passive_movie(train_files)
    eval_files = _filter_passive_movie(eval_files)

    if limit is not None:
        train_files = train_files[:limit]
        eval_files = eval_files[: max(1, limit // 4)]
        logger.info(
            "Limiting to %d train / %d eval files", len(train_files), len(eval_files)
        )

    if not train_files or not eval_files:
        raise RuntimeError(
            f"No passive-movie files found in {source_dir} for seed {seed}. "
            "Check that source_dir contains preprocessed H5 files for passive-movie tasks."
        )

    seed_result: dict[str, Any] = {
        "seed": seed,
        "n_train_files": len(train_files),
        "n_eval_files": len(eval_files),
        "holdout_ratio": holdout_ratio,
        "source_dir": str(source_dir),
        "variants": {},
    }

    variants: list[tuple[str, bool]] = []
    if not random_init_only:
        variants.append(("pretrained", False))
    variants.append(("random_init", True))

    for variant_name, is_random in variants:
        logger.info("--- Variant: %s ---", variant_name)

        # Cache paths for embeddings
        train_cache = output_dir / f"emb_seed{seed}_{variant_name}_train.h5"
        eval_cache = output_dir / f"emb_seed{seed}_{variant_name}_eval.h5"

        if train_cache.exists() and eval_cache.exists() and not overwrite:
            logger.info("Loading cached embeddings from disk")
            train_data = _load_embeddings(train_cache)
            eval_data = _load_embeddings(eval_cache)
        else:
            # Load model
            model = load_labram_model(
                checkpoint=checkpoint if not is_random else None,
                device=device,
                random_init=is_random,
            )

            logger.info("Extracting TRAIN embeddings (%d files)", len(train_files))
            train_data = _extract_embeddings(
                model, train_files, device, batch_size, masks
            )

            logger.info("Extracting EVAL embeddings (%d files)", len(eval_files))
            eval_data = _extract_embeddings(
                model, eval_files, device, batch_size, masks
            )

            # Free GPU memory before probing
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Save to cache
            meta = {"seed": seed, "variant": variant_name}
            _save_embeddings(train_cache, train_data, meta)
            _save_embeddings(eval_cache, eval_data, meta)
            logger.info(
                "Saved embeddings: train=%d, eval=%d epochs",
                len(train_data["embeddings"]),
                len(eval_data["embeddings"]),
            )

        # Fit probes
        logger.info("Fitting probes: %s", probe_types)
        probe_results = _run_probes(train_data, eval_data, probe_types)
        seed_result["variants"][variant_name] = {
            "n_train_emb": int(len(train_data["embeddings"])),
            "n_eval_emb": int(len(eval_data["embeddings"])),
            "probes": probe_results,
        }

    # Write per-seed JSON
    seed_json = output_dir / f"seed{seed}_results.json"
    with seed_json.open("w") as fp:
        json.dump(seed_result, fp, indent=2)
    logger.info("Wrote per-seed results to %s", seed_json)

    return seed_result


def _aggregate_summary(
    all_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Compute mean ± std across seeds and write summary JSON."""
    variants = list(all_results[0]["variants"].keys())
    labels = ["animacy", "scene"]

    summary: dict[str, Any] = {
        "n_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        "variants": {},
    }

    for var in variants:
        var_summary: dict[str, Any] = {}
        for label in labels:
            for probe in ("lr", "knn"):
                key = f"{label}/{probe}"
                accs = []
                f1s = []
                for r in all_results:
                    var_data = r["variants"].get(var, {})
                    probe_data = (
                        var_data.get("probes", {}).get(label, {}).get(probe, {})
                    )
                    if "error" in probe_data:
                        continue
                    if "accuracy" in probe_data:
                        accs.append(probe_data["accuracy"])
                    if "macro_f1" in probe_data:
                        f1s.append(probe_data["macro_f1"])

                if accs:
                    var_summary[key] = {
                        "accuracy_mean": float(np.mean(accs)),
                        "accuracy_std": float(np.std(accs)),
                        "macro_f1_mean": float(np.mean(f1s)) if f1s else None,
                        "macro_f1_std": float(np.std(f1s)) if f1s else None,
                        "n_seeds": len(accs),
                    }
                    logger.info(
                        "[%s] %s %s: acc=%.3f±%.3f f1=%.3f±%.3f",
                        var,
                        label,
                        probe,
                        float(np.mean(accs)),
                        float(np.std(accs)),
                        float(np.mean(f1s)) if f1s else 0.0,
                        float(np.std(f1s)) if f1s else 0.0,
                    )
                else:
                    logger.warning(
                        "[%s] %s %s: no valid seeds (all probes failed). "
                        "Check that passive-movie epochs have HED vectors.",
                        var,
                        label,
                        probe,
                    )
        summary["variants"][var] = var_summary

    summary_json = output_dir / "summary.json"
    with summary_json.open("w") as fp:
        json.dump(summary, fp, indent=2)
    logger.info("Summary written to %s", summary_json)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("${HBN_DATA_DIR}/preprocessed"),
        help="preprocessed preprocessed H5 directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/gates/v10_gateLit/labram"),
        help="Output directory for embeddings and result JSONs",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to LaBraM checkpoint .pt file. If omitted, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,13,7",
        help="Comma-separated random seeds for held-out splits",
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
        help="Fraction of subjects held out for eval",
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
        help="Limit to first N files (smoke-test only)",
    )
    parser.add_argument(
        "--random-init-only",
        action="store_true",
        help="Skip pretrained model; run only random-init baseline",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract embeddings even if cache exists",
    )
    args = parser.parse_args()

    dm = DeviceManager(args.device)
    device = dm.device
    logger.info("Using device: %s", device)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    probe_types = [p.strip() for p in args.probes.split(",")]

    valid_probes = {"lr", "knn"}
    bad = set(probe_types) - valid_probes
    if bad:
        logger.error("Unknown probe type(s): %s. Valid: %s", bad, valid_probes)
        sys.exit(1)

    all_results: list[dict] = []
    for seed in seeds:
        result = run_one_seed(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            seed=seed,
            checkpoint=args.checkpoint,
            probe_types=probe_types,
            holdout_ratio=args.holdout_ratio,
            batch_size=args.batch_size,
            device=device,
            limit=args.limit,
            random_init_only=args.random_init_only,
            overwrite=args.overwrite,
        )
        all_results.append(result)

    _aggregate_summary(all_results, args.output_dir)
    logger.info("Done. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
