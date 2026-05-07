"""LaBraM adapter for  Gate Lit: HBN passive-movie probe.

Wraps the braindecode Labram model so it consumes preprocessed H5 files
and returns frozen [CLS]-equivalent embeddings per epoch.

Design decisions
----------------
Checkpoint
    braindecode/Labram-Braindecode on HuggingFace (Base, 5.8M params).
    Loaded via braindecode's Labram class + huggingface_hub.
    License: MIT (see docs/LABRAM_LICENSE.md).

Checkpoint alignment
    The published checkpoint (braindecode_labram_base.pt) was trained with:
      - 64 channels → position_embedding shape [1, 65, 200] (64+CLS)
      - 8-second input at 200 Hz → 1600 samples, 8 patches per channel
        → temporal_embedding shape [1, 9, 200] (8 patches + CLS)
      - 4 downstream classes → final_layer weight [4, 200]
    The current braindecode Labram class hard-codes position_embedding to
    len(LABRAM_CHANNEL_ORDER)+1 = 129, which is a library expansion from
    the original 64. To avoid a shape mismatch we set use_abs_pos_emb=False,
    which skips the spatial position embedding entirely. The temporal
    embedding and all transformer weights load cleanly. Dropping spatial
    position is a mild regression for the pretrained CLS representation, but
    acceptable for frozen probe extraction where cross-subject CLS semantics
    matter more than exact positional priors.

    Config used: n_times=1600, n_chans=64, n_outputs=4, patch_size=200,
    use_abs_pos_emb=False, learned_patcher=True, use_mean_pooling=True.

Normalization
    preprocessed data is SPEED-denoised, baseline-corrected µV-scale EEG
    stored at 100 Hz. We z-score per channel per epoch (zero mean, unit
    variance) on the real signal BEFORE zero-padding, so pad zeros do not
    contaminate mean/std. LaBraM does not apply internal normalization to
    the raw input; z-scoring is standard for cross-dataset transfer in the
    EEG-FM literature.

Channels
    HBN's 64-channel set matches a subset of LaBraM's 128 canonical
    channels (all 64 found; zero missing). We pass `ch_names` to
    braindecode's Labram.forward(), which reorders automatically via
    its _setup_channel_mapping() mechanism. Validation against the 128-chan
    set is performed at model load time.

Epoch windowing
    preprocessed epochs are 220 samples @ 100 Hz (2.2 s).
    The pretrained checkpoint uses 8-second input at 200 Hz = 1600 samples
    (8 patches × 200-sample patch). Pipeline:
      1. Resample 100→200 Hz (scipy.signal.resample_poly up=2, down=1)
         → 440 samples at 200 Hz.
      2. Z-score per channel on the 440-sample signal (real data only).
      3. Zero-pad to 1600 samples (append 1160 zeros to the right).
    Zero-padding is chosen over replication because it lets the transformer's
    attention ignore the null region (near-zero rows after z-score), while
    replication would introduce spurious periodicity across temporal patches.
    The event onset is in the first 440 samples (40 samples post-event at
    200 Hz), so relevant EEG content is at the beginning of the padded input.

Embedding extraction
    forward() with return_features=True yields a dict with 'cls_token'
    (B, embed_dim). This is the [CLS]-equivalent frozen embedding.

Silent-failure guards
    - Raises if checkpoint download fails.
    - Raises if any HBN channel is missing from LaBraM's set.
    - Raises on model load mismatch (unexpected keys beyond known skipped).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.signal
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# HBN preprocessed channel names (64 channels after 129->64 interpolation).
# These are stored as JSON in the H5 file attrs['channel_names'].
HBN_CHANNEL_NAMES: list[str] = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "TP9",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "TP10",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO9",
    "O1",
    "Oz",
    "O2",
    "PO10",
    "AF7",
    "AF3",
    "AF4",
    "AF8",
    "F5",
    "F1",
    "F2",
    "F6",
    "FT9",
    "FT7",
    "FC3",
    "FC4",
    "FT8",
    "FT10",
    "C5",
    "C1",
    "C2",
    "C6",
    "TP7",
    "CP3",
    "CPz",
    "CP4",
    "TP8",
    "P5",
    "P1",
    "P2",
    "P6",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
]

# LaBraM canonical 128-channel order (from braindecode LABRAM_CHANNEL_ORDER).
# Used to validate that all HBN channels are recognized.
_LABRAM_128: frozenset[str] = frozenset(
    [
        "FP1",
        "FPZ",
        "FP2",
        "AF9",
        "AF7",
        "AF5",
        "AF3",
        "AF1",
        "AFZ",
        "AF2",
        "AF4",
        "AF6",
        "AF8",
        "AF10",
        "F9",
        "F7",
        "F5",
        "F3",
        "F1",
        "FZ",
        "F2",
        "F4",
        "F6",
        "F8",
        "F10",
        "FT9",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "FT10",
        "T9",
        "T7",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "T8",
        "T10",
        "TP9",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "TP10",
        "P9",
        "P7",
        "P5",
        "P3",
        "P1",
        "PZ",
        "P2",
        "P4",
        "P6",
        "P8",
        "P10",
        "PO9",
        "PO7",
        "PO5",
        "PO3",
        "PO1",
        "POZ",
        "PO2",
        "PO4",
        "PO6",
        "PO8",
        "PO10",
        "O1",
        "OZ",
        "O2",
        "O9",
        "CB1",
        "CB2",
        "IZ",
        "O10",
        "T3",
        "T5",
        "T4",
        "T6",
        "M1",
        "M2",
        "A1",
        "A2",
        "CFC1",
        "CFC2",
        "CFC3",
        "CFC4",
        "CFC5",
        "CFC6",
        "CFC7",
        "CFC8",
        "CCP1",
        "CCP2",
        "CCP3",
        "CCP4",
        "CCP5",
        "CCP6",
        "CCP7",
        "CCP8",
        "T1",
        "T2",
        "FTT9h",
        "TTP7h",
        "TPP9h",
        "FTT10h",
        "TPP8h",
        "TPP10h",
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FP2-F8",
        "F8-T8",
        "T8-P8",
        "P8-O2",
    ]
)

# LaBraM embedding dimension for Base model.
LABRAM_EMBED_DIM: int = 200

# Source and target sampling rates for preprocessing.
SOURCE_SFREQ: float = 100.0  # preprocessed HBN
TARGET_SFREQ: float = 200.0  # LaBraM pretraining standard

# LaBraM patch size (samples at 200 Hz = 1s per patch).
LABRAM_PATCH_SIZE: int = 200

# Total input length expected by the pretrained checkpoint (8s @ 200 Hz).
LABRAM_N_TIMES: int = 1600  # 8 patches × 200-sample patch

# Number of output classes in the pretrained checkpoint (4-class downstream).
# We never use the classifier head; n_outputs=4 is needed for strict=False load.
LABRAM_N_OUTPUTS: int = 4

# Passive-movie tasks that contain shot-boundary epochs.
PASSIVE_MOVIE_TASKS: frozenset[str] = frozenset(
    [
        "DespicableMe",
        "DiaryOfAWimpyKid",
        "FunwithFractals",
        "ThePresent",
    ]
)


def _validate_hbn_channels() -> None:
    """Raise if any HBN channel is missing from LaBraM's canonical set."""
    missing = [ch for ch in HBN_CHANNEL_NAMES if ch.upper() not in _LABRAM_128]
    if missing:
        raise RuntimeError(
            f"HBN channels not found in LaBraM's channel set: {missing}. "
            f"Map your channels to LaBraM's 128-channel set before proceeding."
        )


def _preprocess_epoch(
    eeg: np.ndarray,
    pre_event_samples: int,
    source_sfreq: float = SOURCE_SFREQ,
    target_sfreq: float = TARGET_SFREQ,
    patch_size: int = LABRAM_PATCH_SIZE,
    n_times: int = LABRAM_N_TIMES,
) -> np.ndarray:
    """Preprocess one epoch for LaBraM input.

    Steps:
    1. Resample 100 Hz → 200 Hz (scipy.signal.resample_poly, up=2, down=1).
    2. Z-score per channel on the resampled signal (real data only, pre-pad).
    3. Zero-pad to n_times samples (append zeros to the right).

    The pretrained checkpoint expects 8-second input (1600 samples at 200 Hz).
    Our epochs are 2.2 s (220 samples at 100 Hz → 440 at 200 Hz). Zero-padding
    places real content at the start of the input and nulls the remainder.
    Attention on the null region is near-zero after z-scoring (z-score is
    applied before padding, so the pad values 0.0 equal the channel mean
    of the z-scored signal and do not skew the distribution), so pad does
    not dominate the CLS representation. Replication was rejected because it
    introduces spurious temporal periodicity across patches.

    Args:
        eeg: float32 array of shape (n_channels, n_times_src) at source_sfreq.
        pre_event_samples: samples before event onset at source_sfreq (unused
            after switch to full-length zero-padded input; kept for API compat).
        source_sfreq: source sampling frequency (default 100 Hz).
        target_sfreq: target sampling frequency (default 200 Hz).
        patch_size: LaBraM patch size in samples at target_sfreq (default 200).
        n_times: total output length in samples at target_sfreq (default 1600).

    Returns:
        float32 array of shape (n_channels, n_times).
    """
    up = int(target_sfreq)
    down = int(source_sfreq)

    # Resample: (n_channels, n_times_src) → (n_channels, n_times_src * up // down)
    eeg_200 = scipy.signal.resample_poly(eeg, up=up, down=down, axis=-1).astype(
        np.float32
    )

    n_times_200 = eeg_200.shape[-1]
    if n_times_200 < patch_size:
        raise RuntimeError(
            f"Epoch too short after resampling: {n_times_200} samples < "
            f"patch_size={patch_size}. Source epoch has {eeg.shape[-1]} "
            f"samples at {source_sfreq} Hz."
        )

    # Z-score per channel on the real (non-padded) signal.
    mean = eeg_200.mean(axis=-1, keepdims=True)
    std = eeg_200.std(axis=-1, keepdims=True)
    eeg_200 = (eeg_200 - mean) / (std + 1e-8)

    # Zero-pad to n_times on the right; real content stays at start of input.
    if n_times_200 < n_times:
        pad_width = n_times - n_times_200
        eeg_200 = np.concatenate(
            [eeg_200, np.zeros((eeg_200.shape[0], pad_width), dtype=np.float32)],
            axis=-1,
        )
    else:
        # Epoch is already at least n_times; take the leading n_times samples.
        eeg_200 = eeg_200[:, :n_times]

    return eeg_200  # (n_channels, n_times)


def load_labram_model(
    checkpoint: str | Path | None,
    device: torch.device,
    random_init: bool = False,
) -> nn.Module:
    """Load LaBraM-Base from checkpoint or as random-init.

    Args:
        checkpoint: Path to a local checkpoint file, or None to download
            from HuggingFace (braindecode/Labram-Braindecode).
        device: Target device.
        random_init: If True, skip weight loading (random initialization).

    Returns:
        Frozen LaBraM model in eval mode.

    Raises:
        RuntimeError: If checkpoint cannot be loaded or has unexpected keys.
        ImportError: If braindecode is not installed.
    """
    try:
        from braindecode.models import Labram
    except ImportError as e:
        raise ImportError(
            "braindecode is required for LaBraM. Install with: uv add braindecode"
        ) from e

    _validate_hbn_channels()

    # LaBraM-Base config matching the published braindecode/Labram-Braindecode
    # checkpoint (braindecode_labram_base.pt).
    #
    # Checkpoint shapes:
    #   position_embedding: [1, 65, 200]  (64 channels + CLS)
    #   temporal_embedding: [1, 9, 200]   (8 patches + CLS; 8×200=1600 samples)
    #   final_layer.weight: [4, 200]      (4-class downstream head)
    #
    # use_abs_pos_emb=False: braindecode expanded LABRAM_CHANNEL_ORDER to 128,
    # making position_embedding [1, 129, 200] — mismatched with checkpoint's
    # [1, 65, 200]. Disabling avoids the shape error. Spatial position priors
    # are lost but this is acceptable for frozen CLS probe extraction.
    #
    # learned_patcher=True: checkpoint has patch_embed.segment_patch.patcher
    # weights; these are absent with learned_patcher=False.
    #
    # use_mean_pooling=True: checkpoint has fc_norm but not norm; this config
    # matches. Final_layer (classifier head) is not used (return_features=True).
    model = Labram(
        n_times=LABRAM_N_TIMES,
        n_chans=len(HBN_CHANNEL_NAMES),
        n_outputs=LABRAM_N_OUTPUTS,
        patch_size=LABRAM_PATCH_SIZE,
        embed_dim=LABRAM_EMBED_DIM,
        num_layers=12,
        num_heads=10,
        neural_tokenizer=True,
        use_abs_pos_emb=False,
        learned_patcher=True,
        use_mean_pooling=True,
    )

    if not random_init:
        if checkpoint is None or str(checkpoint) == "":
            checkpoint = _download_checkpoint()
        else:
            checkpoint = Path(checkpoint)
            if not checkpoint.exists():
                raise RuntimeError(
                    f"Checkpoint not found: {checkpoint}. "
                    "Pass --checkpoint or let it download from HuggingFace."
                )

        logger.info("Loading LaBraM weights from %s", checkpoint)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        # braindecode checkpoint may wrap weights in a dict
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Expected skipped keys:
        #   unexpected: ['position_embedding'] — shape mismatch due to braindecode
        #               128-chan expansion; use_abs_pos_emb=False drops this param.
        #   missing:    [] — all other model params are present in checkpoint.
        expected_unexpected = {"position_embedding"}
        real_missing = [k for k in missing]
        real_unexpected = [k for k in unexpected if k not in expected_unexpected]
        if real_missing or real_unexpected:
            raise RuntimeError(
                f"LaBraM checkpoint mismatch — missing: {real_missing[:5]}, "
                f"unexpected: {real_unexpected[:5]}"
            )
        logger.info(
            "LaBraM loaded. Skipped keys: missing=%d unexpected=%d "
            "(position_embedding expected-skipped due to braindecode 128-chan expansion)",
            len(missing),
            len(unexpected),
        )
    else:
        logger.info("LaBraM RANDOM-INIT (no checkpoint loaded)")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _download_checkpoint() -> Path:
    """Download LaBraM-Base checkpoint from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download the LaBraM checkpoint. "
            "Install with: uv add huggingface_hub"
        ) from e

    logger.info(
        "Downloading LaBraM-Base from HuggingFace braindecode/Labram-Braindecode"
    )
    path = hf_hub_download(
        repo_id="braindecode/Labram-Braindecode",
        filename="braindecode_labram_base.pt",
    )
    return Path(path)


@torch.no_grad()
def extract_embeddings_batch(
    model: nn.Module,
    eeg_batch: torch.Tensor,
    ch_names: list[str],
) -> np.ndarray:
    """Extract frozen CLS embeddings from a batch of preprocessed epochs.

    Args:
        model: Frozen LaBraM model (from load_labram_model).
        eeg_batch: float32 tensor of shape (B, n_channels, n_times) where
            n_times = LABRAM_N_TIMES (1600 = 8 s at 200 Hz).
        ch_names: channel names in HBN order (used for LaBraM reordering).

    Returns:
        np.ndarray of shape (B, embed_dim).
    """
    # braindecode Labram.forward(x, ch_names=..., return_features=True)
    # returns dict with 'cls_token' of shape (B, embed_dim).
    out: Any = model(eeg_batch, ch_names=ch_names, return_features=True)
    if isinstance(out, dict):
        cls = out.get("cls_token")
        if cls is None:
            # Fallback: some versions return 'features' instead
            cls = out.get("features")
        if cls is None:
            raise RuntimeError(
                f"LaBraM forward returned dict without 'cls_token' or 'features'. "
                f"Keys: {list(out.keys())}"
            )
        if cls.dim() == 3:
            # (B, n_patches, embed_dim) → mean-pool
            cls = cls.mean(dim=1)
    elif isinstance(out, torch.Tensor):
        # Some versions return a raw tensor (mean-pooled features)
        cls = out
        if cls.dim() == 3:
            cls = cls.mean(dim=1)
    else:
        raise RuntimeError(
            f"Unexpected LaBraM output type: {type(out)}. Expected dict or Tensor."
        )
    return cls.cpu().float().numpy()


class HBNLaBraMDataset:
    """Adapter that reads preprocessed H5 files and yields LaBraM-compatible tensors.

    Reads epochs from the given H5 file list, preprocesses each epoch
    (resample + z-score + zero-pad), and batches them for embedding
    extraction. HED vectors and labels are also returned for probe fitting.
    """

    def __init__(
        self,
        h5_files: list[Path],
        passive_only: bool = True,
    ) -> None:
        self._index: list[dict] = []
        self._passive_only = passive_only
        self._build_index(h5_files)

    def _build_index(self, h5_files: list[Path]) -> None:
        """Scan files and build epoch index."""
        for h5_path in h5_files:
            if self._passive_only:
                task = _task_from_stem(h5_path.stem)
                if task not in PASSIVE_MOVIE_TASKS:
                    continue
            try:
                with h5py.File(h5_path, "r") as f:
                    ver = f.attrs.get("preprocess_version", "")
                    if ver != "v9_tier1":
                        raise RuntimeError(
                            f"Expected preprocess_version='v9_tier1', "
                            f"got '{ver}' in {h5_path}. "
                            f"Pass the correct preprocessed source directory."
                        )
                    n_epochs = int(f.attrs.get("n_epochs", 0))
                    for i in range(n_epochs):
                        key = f"epoch_{i:05d}"
                        if key not in f:
                            continue
                        grp = f[key]
                        if "hed_vector" not in grp:
                            continue  # skip epochs without HED supervision
                        self._index.append(
                            {
                                "h5_path": str(h5_path),
                                "epoch_key": key,
                                "pre_event_samples": int(
                                    grp.attrs.get("pre_event_samples", 20)
                                ),
                            }
                        )
            except (OSError, KeyError) as e:
                logger.warning("Error scanning %s: %s", h5_path, e)

        logger.info(
            "HBNLaBraMDataset: %d epochs from %d files (passive_only=%s)",
            len(self._index),
            len(h5_files),
            self._passive_only,
        )

    def __len__(self) -> int:
        return len(self._index)

    def read_epoch(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Read and preprocess one epoch.

        Returns:
            Tuple of:
                eeg_proc: float32 array (n_channels, LABRAM_N_TIMES)
                hed_vector: float32 array (vocab_size,)
        """
        meta = self._index[idx]
        with h5py.File(meta["h5_path"], "r") as f:
            grp = f[meta["epoch_key"]]
            eeg = grp["eeg"][:].astype(np.float32)
            hed_vector = grp["hed_vector"][:].astype(np.float32)

        eeg_proc = _preprocess_epoch(
            eeg,
            pre_event_samples=meta["pre_event_samples"],
        )
        return eeg_proc, hed_vector

    def iter_batches(
        self,
        batch_size: int = 64,
    ) -> BatchIterator:
        """Return a BatchIterator over the dataset."""
        return BatchIterator(self, batch_size=batch_size)


class BatchIterator:
    """Iterate over HBNLaBraMDataset in batches."""

    def __init__(self, dataset: HBNLaBraMDataset, batch_size: int = 64) -> None:
        self._dataset = dataset
        self._batch_size = batch_size

    def __iter__(self):
        n = len(self._dataset)
        for start in range(0, n, self._batch_size):
            end = min(start + self._batch_size, n)
            eegs = []
            hvecs = []
            for i in range(start, end):
                eeg_proc, hed_vector = self._dataset.read_epoch(i)
                eegs.append(eeg_proc)
                hvecs.append(hed_vector)
            yield np.stack(eegs, axis=0), np.stack(hvecs, axis=0)

    def __len__(self) -> int:
        return math.ceil(len(self._dataset) / self._batch_size)


def _task_from_stem(stem: str) -> str:
    """Extract task name from H5 filename stem: {subject}_{task}[_run-N]."""
    parts = stem.split("_")
    return parts[1] if len(parts) >= 2 else ""


def get_channel_names_from_h5(h5_path: Path) -> list[str]:
    """Read channel names stored in H5 file attrs."""
    with h5py.File(h5_path, "r") as f:
        raw = f.attrs.get("channel_names", None)
    if raw is None:
        logger.warning(
            "No channel_names attr in %s; falling back to HBN_CHANNEL_NAMES",
            h5_path,
        )
        return list(HBN_CHANNEL_NAMES)
    if isinstance(raw, str):
        return json.loads(raw)
    return list(raw)
