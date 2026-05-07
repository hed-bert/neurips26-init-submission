"""Configuration management for HED-BERT experiments."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class HEDBERTConfig:
    """HEDBERT experiment configuration.

    Defaults match tiny.yaml for fast iteration.
    """

    # Model architecture
    latent_dim: int = 8
    encoder_scales: list[int] = dataclasses.field(default_factory=lambda: [15, 5])
    encoder_hidden: list[int] = dataclasses.field(default_factory=lambda: [32, 64, 128])
    encoder_type: str = "sequential"  # "sequential" or "parallel"
    embed_dim: int = 128
    max_amplitude: float = 800.0  # uV clamp for InputNorm (0 = disabled)
    norm_mode: str = "instance"  # "instance" (InstanceNorm1d) or "mean_scale" (mean removal + fixed scale)
    norm_scale: float = 200.0  # fixed divisor for mean_scale mode (uV)
    num_layers: int = 3
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    batch_size: int = 128
    learning_rate: float = 1e-3
    total_epochs: int = 50
    mask_ratio: float = 0.0
    num_event_types: int = 12
    use_vae: bool = False
    max_seq_len: int = 200
    device: str = "cuda"

    # Preprocessing
    sfreq: float = 100.0
    l_freq: float = 0.5
    target_channels: int = 64

    # Event epoching
    pre_event_ms: float = 100.0
    max_post_ms: float = 2000.0
    min_epoch_ms: float = 300.0
    include_physiological: bool = True

    # Training
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    scheduler: str = "one_cycle"

    # Loss weights
    lambda_recon: float = 2.0
    lambda_mask: float = 0.0
    lambda_event: float = 0.5

    # Phase schedule ("default", "hed_first", "hed_warmup", "masked")
    phase_schedule: str = "default"

    # HED vocabulary pruning: minimum document frequency fraction (0.0 to 1.0)
    min_frequency: float = 0.0

    # HED vocab depth collapse: tags deeper than this are merged into their ancestor
    # at max_tag_depth. 0 = no collapse (default). 3 = collapse depth 4+ into depth 3.
    # Matches tag granularity to EEG resolving power.
    max_tag_depth: int = 0

    # Prediction head type: "mlp" ( 2-layer MLP) or "tag_embedding" ( dot-product)
    prediction_head_type: str = "mlp"

    # Tag embedding head options (ignored for MLP head)
    tag_embedding_bias: bool = True  # per-tag learnable bias (frequency prior init)
    tag_embedding_scale: float = 1.0  # logit scale factor (try sqrt(embed_dim) ~ 11.3)

    # Gradient isolation: freeze [EVT] embedding param; HED trains transformer weights
    detach_evt_from_recon: bool = False

    # preprocessed: make the [EVT] timestamp a single learnable scalar
    # constrained to [0, evt_time_ms_max] via sigmoid * max. Initial value
    # = evt_time_ms_init. Diagnostic value: the learned position indicates
    # where discriminative signal lives in the epoch.
    learnable_evt_time_ms: bool = False
    evt_time_ms_init: float = 250.0
    evt_time_ms_max: float = 500.0

    # preprocessed: HED loss flavor.
    #   "ancestor_bce" (default, / behavior): multi-hot BCE with
    #       ancestor inclusion at preprocessing time, depth-weighted.
    #   "per_level_softmax": Hypothesis A from
    #       .context/archive/hed_hierarchy_hypotheses.md. CE per semantic level on
    #       the deepest active tag in that level's branch; ancestors
    #       stripped from the multi-hot at training time.
    #   "top_k_mi_softmax": single softmax over top-K tags ranked by MI
    #       with task code over the training set. Target = highest-MI
    #       active tag for the epoch; epochs with no top-K tag active
    #       contribute zero loss.
    hed_loss_flavor: str = "ancestor_bce"
    top_k_mi: int = 100  # only used when hed_loss_flavor=top_k_mi_softmax
    top_k_mi_indices_path: str | None = None  # path to .pt with top-K tag indices

    # HED loss level masking: only compute loss for tags at these semantic levels
    # (L0-L4 from HEDVectorizer.classify_tag()). Empty list = all levels (default).
    # E.g., [2, 3] = L2 Entity + L3 Action only.
    hed_loss_levels: list[int] = dataclasses.field(default_factory=list)

    # VAE (only used when use_vae=True)
    lambda_kl: float = 0.0001

    # Gate program: prediction target
    # "hed" = multi-hot HED tag prediction (BCE loss, default)
    # "task_codes" = single-label task classification (CE loss, 10 HBN tasks)
    prediction_target: str = "hed"

    # Gate program: restrict training data to passive or active tasks
    # "all" = use all tasks (default), "passive" = 6 movie/rest tasks,
    # "active" = 4 response-requiring tasks
    task_filter: str = "all"

    # Gate program: restrict HED vocab to specific branch prefixes
    # Empty list = all branches (default). For process-HED:
    # ["Event/Sensory-event", "Event/Agent-action", ...]
    hed_branch_filter: list[str] = dataclasses.field(default_factory=list)

    # Multi-dataset curriculum (optional, used by medium config)
    curriculum: dict[str, list[int]] | None = None
    dataset_weights: dict[str, float] | None = None

    @property
    def total_stride(self) -> int:
        """Total temporal downsampling factor for the encoder.

        Parallel encoder: min(scales). Sequential: product of scales.
        """
        if getattr(self, "encoder_type", "sequential") == "parallel":
            return min(self.encoder_scales)
        result = 1
        for k in self.encoder_scales:
            result *= k
        return result


def load_config(path: str | Path) -> HEDBERTConfig:
    """Load YAML config file and return an HEDBERTConfig.

    Raises TypeError if the YAML contains keys not present in the dataclass.
    """
    path = Path(path)
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if raw is None:
        return HEDBERTConfig()

    field_names = {f.name for f in dataclasses.fields(HEDBERTConfig)}
    unknown = set(raw.keys()) - field_names
    if unknown:
        raise TypeError(
            f"Unknown config keys: {unknown}. Valid keys: {sorted(field_names)}"
        )

    return HEDBERTConfig(**raw)


def save_config(config: HEDBERTConfig, path: str | Path) -> None:
    """Serialize config to YAML for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(config)
    # Drop None values for cleaner output
    data = {k: v for k, v in data.items() if v is not None}
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
