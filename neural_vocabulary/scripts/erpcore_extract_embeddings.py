"""Extract per-trial frozen-encoder embeddings on ERP-CORE.

Forwards ERP-CORE Morlet TF windows (E=8 packed) through a frozen
BertSSL checkpoint and caches per-trial event-token embeddings to
disk so downstream probe runs are pure-CPU sklearn (fast, repeatable).

For each (encoder_seed, subject, task, probe), writes one ``.npz`` to
``<output-dir>/seed_{N}/{subject}_{task}_{probe}.npz`` with:
    embeddings: (n_trials, d_model=192) float32
    labels:     (n_trials,)             int64
    subject_id: str
    task:       str
    probe:      str
    encoder_seed: int

The embeddings come from BertSSL.forward(masker=None)["evt_embeddings"]
— the model already exposes per-event-token outputs at positions 1..E,
no manual slicing needed.

Usage (smoke, one subject one paradigm):
    uv run python -m neural_vocabulary.scripts.erpcore_extract_embeddings \\
        --tf-dir ${HBN_DATA_DIR}/preprocessed_v10_erpcore_tf \\
        --checkpoint runs/gates/v10_gateD/d1_4_hed_warmup_100ep/seed_42_best.pt \\
        --vectorizer ${HBN_DATA_DIR}/hed_vectorizer.pt \\
        --output-dir runs/gates/v10_gateD2/embeddings/seed_42 \\
        --subjects 001 --tasks N170 --probes face_vs_car

Usage (full, all subjects all paradigms):
    uv run python -m neural_vocabulary.scripts.erpcore_extract_embeddings \\
        --tf-dir ${HBN_DATA_DIR}/preprocessed_v10_erpcore_tf \\
        --checkpoint runs/gates/v10_gateD/d1_4_hed_warmup_100ep/seed_42_best.pt \\
        --vectorizer ${HBN_DATA_DIR}/hed_vectorizer.pt \\
        --output-dir runs/gates/v10_gateD2/embeddings/seed_42
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_vocabulary.data.erpcore_label_rules import (
    LABEL_RULES,
    LabelRule,
)
from neural_vocabulary.data.erpcore_paradigm_dataset import (
    ErpcoreParadigmDataset,
    paradigm_collate,
)
from neural_vocabulary.data.hed_vectorizer import HEDVectorizer
from neural_vocabulary.models.bert_ssl import BertSSL, PatchMode
from neural_vocabulary.training.device_manager import DeviceManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default ERP-CORE paradigm subset for the per-paradigm probe list.
DEFAULT_TASKS: tuple[str, ...] = ("N170", "MMN", "P3", "N2pc", "N400")


def _load_tag_init(vectorizer_path: Path, embed_dim: int) -> torch.Tensor | None:
    """Load HBN tag-embedding hierarchy init (matches eval_within_hbn pattern)."""
    data = torch.load(vectorizer_path, map_location="cpu", weights_only=False)
    if not (isinstance(data, dict) and "tag_to_idx" in data):
        raise RuntimeError(f"Cannot read tag_to_idx from {vectorizer_path}.")
    try:
        vec = HEDVectorizer(schema_version="8.3.0")
        vec._tag_to_idx = dict(data["tag_to_idx"])
        vec._idx_to_tag = dict(data.get("idx_to_tag", {}))
        vec._tag_depths = dict(data.get("tag_depths", {}))
        return vec.get_hierarchy_init_embeddings(embed_dim=embed_dim)
    except Exception as exc:  # noqa: BLE001
        # Encoder load is strict=True; this init is overwritten by the
        # checkpoint's tag_embeddings. Warn so debugging notices.
        logger.warning(
            "Could not build hierarchy-init tag embeddings from %s: %s. "
            "Using random init; checkpoint load will overwrite them.",
            vectorizer_path,
            exc,
        )
        return None


def _build_model_from_checkpoint(
    checkpoint_path: Path,
    vectorizer_path: Path,
    device: torch.device,
    patch_mode: PatchMode = "flat",
) -> BertSSL:
    """Construct BertSSL with vocab_size from vectorizer + load weights."""
    data = torch.load(vectorizer_path, map_location="cpu", weights_only=False)
    vocab_size = len(data["tag_to_idx"])
    tag_init = _load_tag_init(vectorizer_path, embed_dim=192)
    model = BertSSL(
        vocab_size=vocab_size,
        tag_init_embeddings=tag_init,
        patch_mode=patch_mode,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_embeddings_for_dataset(
    model: BertSSL,
    dataset: ErpcoreParadigmDataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward dataset through frozen encoder; return (embeddings, labels).

    Each window contributes E=8 trial embeddings. Output shape
    ``(n_windows * 8, d_model)`` for embeddings and ``(n_windows * 8,)``
    for labels, paired index-for-index.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=paradigm_collate,
        pin_memory=device.type == "cuda",
    )
    all_embeddings: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in loader:
        tf = batch["tf"].to(device, non_blocking=True)
        hed = batch["hed"].to(device, non_blocking=True)
        labels = batch["labels"]  # CPU; (B, E)
        out = model(tf, hed, masker=None)
        evt = out["evt_embeddings"].detach().cpu().float().numpy()  # (B, E, D)
        b, e, d = evt.shape
        all_embeddings.append(evt.reshape(b * e, d))
        all_labels.append(labels.numpy().reshape(b * e))
    embeddings = np.concatenate(all_embeddings, axis=0)
    label_arr = np.concatenate(all_labels, axis=0)
    return embeddings, label_arr


def _select_h5_files(
    tf_dir: Path,
    subjects: list[str] | None,
    task: str,
) -> list[Path]:
    """Find {subject}_{task}.h5 files matching the subject filter."""
    if subjects is None:
        files = sorted(tf_dir.glob(f"*_{task}.h5"))
    else:
        files = []
        for s in subjects:
            p = tf_dir / f"{s}_{task}.h5"
            if p.exists():
                files.append(p)
            else:
                logger.warning("Missing TF file for subject %s task %s", s, task)
    return files


def _output_npz_path(output_dir: Path, subject: str, task: str, probe: str) -> Path:
    return output_dir / f"{subject}_{task}_{probe}.npz"


def extract_one_subject_paradigm(
    model: BertSSL,
    tf_dir: Path,
    subject: str,
    rule: LabelRule,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    encoder_seed: int,
    overwrite: bool,
) -> dict:
    """Extract embeddings for a single (subject, paradigm, probe) cell."""
    out_path = _output_npz_path(output_dir, subject, rule.paradigm, rule.probe)
    if out_path.exists() and not overwrite:
        return {"status": "already_exists", "path": str(out_path), "n_trials": 0}
    h5_files = _select_h5_files(tf_dir, [subject], rule.paradigm)
    if not h5_files:
        return {"status": "no_files", "path": str(out_path), "n_trials": 0}
    try:
        ds = ErpcoreParadigmDataset(h5_files, rule)
    except RuntimeError as e:
        # Empty dataset for this (subject, paradigm, probe) — log + skip.
        logger.warning(
            "Empty dataset for sub=%s paradigm=%s probe=%s: %s",
            subject,
            rule.paradigm,
            rule.probe,
            e,
        )
        return {"status": "empty", "path": str(out_path), "n_trials": 0}
    embeddings, labels = extract_embeddings_for_dataset(
        model, ds, device, batch_size=batch_size
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings.astype(np.float32),
        labels=labels.astype(np.int64),
        subject_id=subject,
        task=rule.paradigm,
        probe=rule.probe,
        encoder_seed=encoder_seed,
    )
    return {
        "status": "ok",
        "path": str(out_path),
        "n_trials": int(embeddings.shape[0]),
        "n_windows": len(ds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tf-dir",
        type=Path,
        default=Path("${HBN_DATA_DIR}/preprocessed_v10_erpcore_tf"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--vectorizer",
        type=Path,
        default=Path("${HBN_DATA_DIR}/hed_vectorizer.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subject IDs (e.g. 001 002). Default: all subjects with files.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        choices=list(DEFAULT_TASKS),
    )
    parser.add_argument(
        "--probes",
        nargs="+",
        default=None,
        help="Probe names (default: all probes for each task).",
    )
    parser.add_argument(
        "--encoder-seed",
        type=int,
        required=True,
        help="Encoder seed identifier (recorded in npz; used in output dir).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--patch-mode",
        type=str,
        default="flat",
        choices=("flat", "channel_token"),
        help="Must match the checkpoint's training config. Default 'flat'.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dm = DeviceManager(device_type=args.device)
    device = dm.device
    logger.info("Using device: %s", device)

    model = _build_model_from_checkpoint(
        args.checkpoint,
        args.vectorizer,
        device,
        patch_mode=cast("PatchMode", args.patch_mode),
    )
    logger.info(
        "Loaded BertSSL from %s (vocab_size=%d, params=%d)",
        args.checkpoint,
        model.vocab_size,
        model.param_count(),
    )

    # Resolve all (task, probe) pairs to extract.
    task_probes: list[tuple[str, LabelRule]] = []
    for task in args.tasks:
        for rule in LABEL_RULES.get(task, []):
            if args.probes is None or rule.probe in args.probes:
                task_probes.append((task, rule))
    if not task_probes:
        raise RuntimeError(f"No (task, probe) pairs match tasks={args.tasks}")

    # Resolve subject list per task (default: every subject with a TF file).
    if args.subjects is None:
        all_subjects = sorted(
            {p.stem.split("_", 1)[0] for p in args.tf_dir.glob(f"*_{args.tasks[0]}.h5")}
        )
    else:
        all_subjects = list(args.subjects)
    logger.info(
        "Subjects: %d, task-probe pairs: %d", len(all_subjects), len(task_probes)
    )

    t0 = time.perf_counter()
    results: list[dict] = []
    for subject in all_subjects:
        for _task, rule in task_probes:
            r = extract_one_subject_paradigm(
                model=model,
                tf_dir=args.tf_dir,
                subject=subject,
                rule=rule,
                output_dir=args.output_dir,
                device=device,
                batch_size=args.batch_size,
                encoder_seed=args.encoder_seed,
                overwrite=args.overwrite,
            )
            results.append(
                {
                    "subject": subject,
                    "task": rule.paradigm,
                    "probe": rule.probe,
                    **r,
                }
            )
    elapsed = time.perf_counter() - t0

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_existing = sum(1 for r in results if r["status"] == "already_exists")
    n_empty = sum(1 for r in results if r["status"] == "empty")
    n_no_files = sum(1 for r in results if r["status"] == "no_files")
    total_trials = sum(r["n_trials"] for r in results)

    print("\n=== ERP-CORE embedding extraction report ===")
    print(f"  TF dir:       {args.tf_dir}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Encoder seed: {args.encoder_seed}")
    print(f"  Cells:        {len(results)}")
    print(f"    OK:           {n_ok}")
    print(f"    Already:      {n_existing}")
    print(f"    Empty:        {n_empty}")
    print(f"    No files:     {n_no_files}")
    print(f"  Total trials: {total_trials}")
    print(f"  Elapsed:      {elapsed:.1f} s")

    summary_path = args.output_dir / "_extract_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "encoder_seed": args.encoder_seed,
                "n_cells": len(results),
                "n_ok": n_ok,
                "total_trials": total_trials,
                "elapsed_s": elapsed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
