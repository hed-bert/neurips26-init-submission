"""Hierarchical HED loss, reconstruction loss, and joint loss.

The depth-weighted BCE on [EVT] token predictions is the core training
objective for event-grounded pretraining. Combined with masked/full
reconstruction MSE in a scheduled joint loss.

Design reference: .context/ideas.md "HED Vectorization and Hierarchical
Loss Design" and "Training Strategy".
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as nnf

if TYPE_CHECKING:
    from neural_vocabulary.configs import HEDBERTConfig

logger = logging.getLogger(__name__)

# If fewer than this fraction of batch samples have a valid target on the
# first forward pass, the loss logs a loud warning. Catches the case where
# the descendants matrix / top-K indices were built against a different
# vocabulary than the dataset's HED vectors (silent misalignment -> all
# targets strip to zero -> loss is always 0.0 -> "training" is a no-op).
_MIN_FIRST_BATCH_COVERAGE = 0.01


class HierarchicalHEDLoss(nn.Module):
    """Depth-weighted BCE loss with configurable prediction head.

    Takes [EVT] token embeddings from the transformer and predicts
    multi-hot HED tag vectors. Tags at shallower depths (roots) receive
    higher loss weights, so confusing top-level categories costs more
    than confusing leaves.

    Two head types:
      - "mlp" (): 2-layer MLP (Linear -> GELU -> Linear). Hidden dim = 4x embed_dim.
      - "tag_embedding" (): Learned tag embedding matrix. Logits = dot(EVT, tag_emb).
        ~4-5x fewer params than MLP for typical vocab sizes (1K+ tags).
        Siblings in HED tree share structure via hierarchy-aware initialization.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        depth_weights: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        level_mask: torch.Tensor | None = None,
        head_type: str = "mlp",
        tag_init_embeddings: torch.Tensor | None = None,
        tag_embedding_bias: bool = True,
        tag_embedding_scale: float = 1.0,
    ) -> None:
        """Initialize HED loss with prediction head and depth weights.

        Args:
            embed_dim: Dimension of [EVT] token embeddings.
            vocab_size: Number of tags in the HED vocabulary.
            depth_weights: (vocab_size,) per-tag weights from
                HEDVectorizer.get_depth_weights(alpha).
            pos_weight: Optional (vocab_size,) per-tag positive class
                weight for class imbalance (inversely proportional to
                positive rate). Passed to BCEWithLogitsLoss.
            level_mask: Optional (vocab_size,) binary mask. 1.0 = compute
                loss for this tag, 0.0 = ignore. Used to focus loss on
                specific semantic levels (e.g., L2/L3 only).
            head_type: "mlp" for  MLP head, "tag_embedding" for
                dot-product head.
            tag_init_embeddings: Optional (vocab_size, embed_dim) initial
                tag embeddings from HEDVectorizer.get_hierarchy_init_embeddings().
                Only used when head_type="tag_embedding". If None, tag
                embeddings are initialized with randn * 0.02.
        """
        super().__init__()
        if depth_weights.shape != (vocab_size,):
            raise ValueError(
                f"depth_weights shape {depth_weights.shape} does not match "
                f"vocab_size {vocab_size}"
            )
        if pos_weight is not None and pos_weight.shape != (vocab_size,):
            raise ValueError(
                f"pos_weight shape {pos_weight.shape} does not match "
                f"vocab_size {vocab_size}"
            )
        if head_type not in ("mlp", "tag_embedding"):
            raise ValueError(
                f"head_type must be 'mlp' or 'tag_embedding', got '{head_type}'"
            )
        if tag_init_embeddings is not None and head_type != "tag_embedding":
            raise ValueError(
                f"tag_init_embeddings provided but head_type='{head_type}'. "
                "tag_init_embeddings is only used with head_type='tag_embedding'."
            )

        self.head_type = head_type
        self.tag_embedding_scale = tag_embedding_scale

        if head_type == "mlp":
            self.prediction_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, vocab_size),
            )
        else:
            # Tag embedding matrix: logits = scale * (EVT @ tag_embeddings.T) [+ bias]
            if tag_init_embeddings is not None:
                if tag_init_embeddings.shape != (vocab_size, embed_dim):
                    raise ValueError(
                        f"tag_init_embeddings shape {tag_init_embeddings.shape} "
                        f"does not match (vocab_size={vocab_size}, embed_dim={embed_dim})"
                    )
                self.tag_embeddings = nn.Parameter(tag_init_embeddings.clone())
            else:
                self.tag_embeddings = nn.Parameter(
                    torch.randn(vocab_size, embed_dim) * 0.02
                )
            # Optional per-tag bias for base-rate encoding
            if tag_embedding_bias:
                if pos_weight is not None:
                    pos_rate = 1.0 / (1.0 + pos_weight)
                    init_bias = torch.log(pos_rate / (1.0 - pos_rate + 1e-7)).clamp(
                        -5, 5
                    )
                    self.tag_bias = nn.Parameter(init_bias)
                else:
                    self.tag_bias = nn.Parameter(torch.zeros(vocab_size))
            else:
                self.register_buffer("tag_bias", torch.zeros(vocab_size))

        self.register_buffer("depth_weights", depth_weights)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight: torch.Tensor | None = None
        if level_mask is not None:
            self.register_buffer("level_mask", level_mask)
        else:
            self.level_mask: torch.Tensor | None = None

    def predict_logits(self, evt_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute raw logits from [EVT] embeddings.

        Args:
            evt_embeddings: (batch, embed_dim) from [EVT] token output.

        Returns:
            (batch, vocab_size) logits (pre-sigmoid).
        """
        if self.head_type == "mlp":
            return self.prediction_head(evt_embeddings)
        logits = evt_embeddings @ self.tag_embeddings.T * self.tag_embedding_scale
        return logits + self.tag_bias

    def forward(
        self,
        evt_embeddings: torch.Tensor,
        hed_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute depth-weighted BCE loss.

        Args:
            evt_embeddings: (batch, embed_dim) from [EVT] token output.
            hed_targets: (batch, vocab_size) multi-hot binary vectors.

        Returns:
            Scalar loss value.
        """
        logits = self.predict_logits(evt_embeddings)

        # Per-element BCE with logits (no reduction)
        per_element_loss = nnf.binary_cross_entropy_with_logits(
            logits,
            hed_targets,
            weight=None,
            pos_weight=self.pos_weight,
            reduction="none",
        )  # (batch, vocab_size)

        # Apply depth weights: broadcast (vocab_size,) over batch dim
        weighted_loss = per_element_loss * self.depth_weights  # type: ignore[unsupported-operator]

        # Apply level mask if set (zero out loss for excluded semantic levels)
        if self.level_mask is not None:
            weighted_loss = weighted_loss * self.level_mask

        # Mean over vocab then mean over batch
        return weighted_loss.mean()


class PerLevelSoftmaxHEDLoss(nn.Module):
    """preprocessed: per-semantic-level softmax CE over the deepest active tag.

    For each epoch, after stripping ancestor tags from the multi-hot HED
    target (a tag is kept only when no longer-prefix tag is also active):
      - For each semantic level L in {1, 2, 3, 4} (Hypothesis A from
        ``.context/archive/hed_hierarchy_hypotheses.md``), restrict the vocabulary
        to the tags assigned to L by ``HEDVectorizer.get_level_partition()``.
      - The per-level target is the deepest stripped tag in L's branch.
      - Loss_L = mean over batch members with a level-L target of
        ``-log softmax(logits[:, L_indices])[target]``.
      - Total = ``sum_L level_weights[L] * Loss_L`` divided by the number
        of contributing levels (so loss is bounded across configs).

    Closes the gradient-starvation hole that ancestor-inclusive multi-hot
    BCE created (cf. v9_redesign §2.3): softmax forces competition within
    each level instead of the per-tag sigmoids that base-rate-dominated
    ancestors used to dilute.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        level_partition: dict[int, list[int]],
        level_weights: dict[int, float],
        descendants_matrix: torch.Tensor,
        tag_depths: torch.Tensor,
        head_type: str = "mlp",
    ) -> None:
        super().__init__()
        if head_type != "mlp":
            raise NotImplementedError(
                "PerLevelSoftmaxHEDLoss currently only supports head_type='mlp'."
            )
        if descendants_matrix.shape != (vocab_size, vocab_size):
            raise ValueError(
                f"descendants_matrix shape {descendants_matrix.shape} != "
                f"({vocab_size}, {vocab_size})"
            )
        if tag_depths.shape != (vocab_size,):
            raise ValueError(f"tag_depths shape {tag_depths.shape} != ({vocab_size},)")
        self.head_type = head_type
        self.vocab_size = vocab_size

        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, vocab_size),
        )

        self.register_buffer("descendants_matrix", descendants_matrix.to(torch.bool))
        # Per-level: indices into the full vocab for tags assigned to this level
        # plus the depths of those tags (used for "deepest active wins" target).
        self.levels: list[int] = sorted(level_partition.keys())
        self.level_weights = {L: float(level_weights.get(L, 1.0)) for L in self.levels}
        for level in self.levels:
            idxs = torch.tensor(level_partition[level], dtype=torch.long)
            if idxs.numel() > 0 and int(idxs.max()) >= vocab_size:
                raise ValueError(
                    f"level_partition[{level}] contains index "
                    f"{int(idxs.max())} >= vocab_size={vocab_size}"
                )
            self.register_buffer(f"_lvl_{level}_idx", idxs)
            self.register_buffer(
                f"_lvl_{level}_depths", tag_depths[idxs].to(torch.float32)
            )
        # Flips to True after the first forward logs coverage.
        self._coverage_checked: bool = False

    def predict_logits(self, evt_embeddings: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(evt_embeddings)

    def _descendants(self) -> torch.Tensor:
        desc: torch.Tensor = self.descendants_matrix  # type: ignore[assignment]
        return desc

    def forward(
        self,
        evt_embeddings: torch.Tensor,
        hed_targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.predict_logits(evt_embeddings)  # (B, V)

        # Strip ancestor tags. Mask M = (V, V) ancestor->descendant; for each
        # epoch a tag is kept iff none of its descendants are active.
        # desc_active[b, i] = sum_j M[i, j] * targets[b, j]  -> (B, V)
        desc_cast = self._descendants().to(logits.dtype)
        desc_active = (hed_targets > 0).to(logits.dtype) @ desc_cast.T
        stripped = hed_targets * (desc_active == 0).to(logits.dtype)

        total_loss = logits.new_zeros(())
        contributing_levels = 0
        any_valid_target = False
        per_level_coverage: dict[int, float] = {}
        for level in self.levels:
            idxs: torch.Tensor = getattr(self, f"_lvl_{level}_idx")
            depths: torch.Tensor = getattr(self, f"_lvl_{level}_depths")
            if idxs.numel() == 0:
                continue
            level_logits = logits.index_select(1, idxs)  # (B, K_level)
            level_active = stripped.index_select(1, idxs)  # (B, K_level)
            # Pick the deepest active tag per batch member as the target.
            # Multiply (depth + 1) so rank > 0; argmax over zeros stays at 0
            # but those rows are filtered out by `valid`.
            weighted = level_active * (depths + 1.0)
            target_pos = weighted.argmax(dim=1)  # (B,)
            valid = level_active.sum(dim=1) > 0  # (B,)
            per_level_coverage[level] = float(valid.float().mean().item())
            if not bool(valid.any()):
                continue

            any_valid_target = True
            log_p = nnf.log_softmax(level_logits, dim=-1)
            picked = log_p.gather(1, target_pos.unsqueeze(1)).squeeze(1)  # (B,)
            level_loss = -(picked * valid.to(log_p.dtype)).sum() / valid.sum().clamp(
                min=1
            )
            total_loss = total_loss + self.level_weights[level] * level_loss
            contributing_levels += 1

        if not self._coverage_checked:
            self._coverage_checked = True
            logger.info(
                "PerLevelSoftmaxHEDLoss first-batch coverage: %s",
                {L: f"{c:.1%}" for L, c in per_level_coverage.items()},
            )
            if (
                not any_valid_target
                or max(per_level_coverage.values(), default=0.0)
                < _MIN_FIRST_BATCH_COVERAGE
            ):
                logger.warning(
                    "PerLevelSoftmaxHEDLoss first-batch coverage below %.1f%% at "
                    "every level. This usually means the descendants matrix or "
                    "level_partition does not match the HED vectors in the "
                    "dataset (vocab mismatch). Training will emit zero HED loss.",
                    100 * _MIN_FIRST_BATCH_COVERAGE,
                )

        if contributing_levels == 0:
            return total_loss  # zero with grad path
        return total_loss / contributing_levels


class TopKMutualInfoHEDLoss(nn.Module):
    """preprocessed: single-softmax CE on the highest-MI present tag.

    The set of K tag indices is precomputed offline from MI(tag_active,
    task_code) over the training set (see
    ``neural_vocabulary.scripts.compute_top_mi``). Tags are stored sorted
    by descending MI, so position 0 = most informative.

    For each epoch (ancestor-stripped):
      - Find which top-K tags are active.
      - Target = the highest-MI active one (lowest position in the K).
      - Loss = -log softmax(logits[:, K_indices])[target].
      - Epochs with no top-K tag active contribute 0 to the loss.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        top_k_indices: torch.Tensor,
        descendants_matrix: torch.Tensor,
        head_type: str = "mlp",
    ) -> None:
        super().__init__()
        if head_type != "mlp":
            raise NotImplementedError(
                "TopKMutualInfoHEDLoss currently only supports head_type='mlp'."
            )
        if top_k_indices.dtype != torch.long:
            top_k_indices = top_k_indices.to(torch.long)
        if descendants_matrix.shape != (vocab_size, vocab_size):
            raise ValueError(
                f"descendants_matrix shape {descendants_matrix.shape} != "
                f"({vocab_size}, {vocab_size})"
            )
        if top_k_indices.numel() > 0 and int(top_k_indices.max()) >= vocab_size:
            raise ValueError(
                f"top_k_indices contains {int(top_k_indices.max())} >= "
                f"vocab_size={vocab_size}. Payload built against a different "
                "vocabulary than the current vectorizer."
            )
        self.head_type = head_type
        self.vocab_size = vocab_size
        self.k = int(top_k_indices.numel())

        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, vocab_size),
        )

        self.register_buffer("top_k_indices", top_k_indices)
        self.register_buffer("descendants_matrix", descendants_matrix.to(torch.bool))
        # Position-rank tensor: position i has rank (k - i) so argmax picks
        # the lowest-position (= highest-MI) active tag.
        rank = torch.arange(self.k, 0, -1, dtype=torch.float32)
        self.register_buffer("_position_rank", rank)
        # Flips to True after the first forward logs coverage.
        self._coverage_checked: bool = False

    def predict_logits(self, evt_embeddings: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(evt_embeddings)

    def forward(
        self,
        evt_embeddings: torch.Tensor,
        hed_targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.predict_logits(evt_embeddings)  # (B, V)

        # Strip ancestors. Buffer accesses are typed as Tensor|Module via
        # nn.Module.__getattr__; bind through local names to keep ty happy.
        top_k: torch.Tensor = self.top_k_indices  # type: ignore[assignment]
        descendants: torch.Tensor = self.descendants_matrix  # type: ignore[assignment]
        rank: torch.Tensor = self._position_rank  # type: ignore[assignment]
        desc_cast = descendants.to(logits.dtype)
        desc_active = (hed_targets > 0).to(logits.dtype) @ desc_cast.T
        stripped = hed_targets * (desc_active == 0).to(logits.dtype)

        # Restrict to top-K tag positions
        topk_logits = logits.index_select(1, top_k)  # (B, K)
        topk_active = stripped.index_select(1, top_k)  # (B, K)

        # Target = position with highest MI among active. _position_rank
        # gives rank K..1 so a present tag at position 0 wins argmax.
        weighted = topk_active * rank
        target_pos = weighted.argmax(dim=1)  # (B,)
        valid = topk_active.sum(dim=1) > 0
        if not self._coverage_checked:
            self._coverage_checked = True
            coverage = float(valid.float().mean().item())
            logger.info(
                "TopKMutualInfoHEDLoss first-batch coverage: %.1f%% of %d samples have an active top-%d tag",
                coverage * 100,
                int(valid.numel()),
                self.k,
            )
            if coverage < _MIN_FIRST_BATCH_COVERAGE:
                logger.warning(
                    "TopKMutualInfoHEDLoss first-batch coverage %.1f%% below %.1f%% "
                    "threshold. This usually means the top-K indices were built "
                    "against a different vocabulary than the dataset's HED vectors "
                    "(vocab mismatch). Training will emit near-zero HED loss.",
                    coverage * 100,
                    _MIN_FIRST_BATCH_COVERAGE * 100,
                )
        if not bool(valid.any()):
            return logits.new_zeros(())

        log_p = nnf.log_softmax(topk_logits, dim=-1)
        picked = log_p.gather(1, target_pos.unsqueeze(1)).squeeze(1)
        loss = -(picked * valid.to(log_p.dtype)).sum() / valid.sum().clamp(min=1)
        return loss


class ReconstructionLoss(nn.Module):
    """MSE reconstruction loss with optional masking.

    Supports both full reconstruction (all positions) and masked-only
    reconstruction (only compute loss where mask == 1).
    """

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute MSE reconstruction loss.

        Args:
            reconstructed: (batch, channels, time) predicted EEG.
            original: (batch, channels, time) ground truth EEG.
            mask: Optional (batch, time) where 1.0 = compute loss,
                0.0 = ignore. If None, compute loss over all positions.

        Returns:
            Scalar MSE loss.
        """
        if mask is None:
            return nnf.mse_loss(reconstructed, original)

        # Expand mask from (batch, time) to (batch, channels, time)
        mask_expanded = mask.unsqueeze(1).expand_as(original)

        num_active = mask_expanded.sum()
        if num_active == 0:
            # No active positions; return zero loss with grad support
            return reconstructed.sum() * 0.0

        diff_sq = (reconstructed - original) ** 2
        masked_loss = (diff_sq * mask_expanded).sum() / num_active
        return masked_loss


class EventCodeLoss(nn.Module):
    """Single-label task classification loss for Gate 1a (event codes).

    Replaces HED multi-label BCE with CrossEntropyLoss over task classes.
    Uses a simple linear head from [EVT] embeddings to class logits.
    """

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(
        self,
        evt_embeddings: torch.Tensor,
        task_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CE loss for task classification.

        Args:
            evt_embeddings: (N, embed_dim) from valid [EVT] tokens.
            task_indices: (N,) long tensor with task class indices.

        Returns:
            Scalar CE loss.
        """
        logits = self.head(evt_embeddings)
        return self.ce(logits, task_indices)


class JointLoss(nn.Module):
    """Combined reconstruction + HED prediction loss with phase scheduling.

    Loss = lambda_recon * L_recon + lambda_mask * L_mask + lambda_hed * L_hed

    Schedule selected via config.phase_schedule:
        "default": 3-phase recon-heavy -> balanced -> HED-heavy (per ideas.md)
        "hed_first": flat HED-first weighting (lambda_recon=0.5, hed=1.0)
        "hed_warmup": HED-only for epochs 1-20, then balanced -> HED-heavy
        "masked": HED warmup -> masked recon + HED -> HED-heavy (no full recon)
        "hed_then_release": HED-heavy for 60 epochs, then release HED constraint
    """

    # Named phase schedules: (start_epoch, lambda_recon, lambda_mask, lambda_hed)
    # Use "masked" schedule to enable masked reconstruction (lambda_mask > 0).
    NAMED_SCHEDULES: dict[str, list[tuple[int, float, float, float]]] = {
        "default": [
            (1, 2.0, 0.0, 0.5),  # : reconstruction-heavy
            (21, 1.0, 0.0, 1.0),  # : balanced
            (61, 0.5, 0.0, 2.0),  # : HED-heavy
        ],
        "hed_first": [
            (1, 0.5, 0.0, 1.0),  # Flat: HED-first from epoch 1
        ],
        "hed_warmup": [
            (1, 0.0, 0.0, 1.0),  # : HED-only (no recon)
            (21, 1.0, 0.0, 1.0),  # : balanced
            (61, 0.5, 0.0, 2.0),  # : HED-heavy
        ],
        "hed_then_release": [
            (1, 0.0, 0.0, 1.0),  # : HED-only (shape semantic space)
            (21, 0.5, 0.0, 2.0),  # : HED-heavy + light recon
            (61, 0.5, 0.0, 0.1),  # : release HED, encoder consolidates
        ],
        "masked": [
            (1, 0.0, 0.0, 1.0),  # : HED-only warmup
            (21, 0.0, 1.0, 1.0),  # : masked recon + HED (no full recon)
            (61, 0.0, 0.5, 2.0),  # : HED-heavy
        ],
    }

    def __init__(
        self,
        config: HEDBERTConfig,
        vocab_size: int,
        depth_weights: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        level_mask: torch.Tensor | None = None,
        tag_init_embeddings: torch.Tensor | None = None,
        # preprocessed tag-structure inputs (only used when hed_loss_flavor !=
        # "ancestor_bce"). Pre-built by pretrain.py from the loaded
        # HEDVectorizer.
        descendants_matrix: torch.Tensor | None = None,
        level_partition: dict[int, list[int]] | None = None,
        level_weights: dict[int, float] | None = None,
        tag_depths_tensor: torch.Tensor | None = None,
        top_k_mi_indices: torch.Tensor | None = None,
    ) -> None:
        """Initialize joint loss with all sub-losses.

        Args:
            config: HEDBERTConfig providing embed_dim, initial loss
                weights, phase_schedule name, and prediction_head_type.
            vocab_size: HED tag vocabulary size (or num_event_types for task_codes).
            depth_weights: Per-tag depth weights for HED loss (ignored for task_codes).
            pos_weight: Optional per-tag positive class weight.
            level_mask: Optional (vocab_size,) binary mask for level focus.
            tag_init_embeddings: Optional (vocab_size, embed_dim) hierarchy-
                aware initial tag embeddings. Only used when
                config.prediction_head_type="tag_embedding".

        Raises:
            ValueError: If config.phase_schedule is not in NAMED_SCHEDULES.
        """
        super().__init__()
        self.prediction_target = getattr(config, "prediction_target", "hed")
        self.recon_loss = ReconstructionLoss()

        if self.prediction_target == "task_codes":
            num_classes = getattr(config, "num_event_types", 10)
            self.event_code_loss = EventCodeLoss(
                embed_dim=config.embed_dim, num_classes=num_classes
            )
            # Dummy HED loss (not used, but keeps state_dict shape consistent)
            self.hed_loss = None
        else:
            self.event_code_loss = None
            flavor = getattr(config, "hed_loss_flavor", "ancestor_bce")
            if flavor == "ancestor_bce":
                self.hed_loss = HierarchicalHEDLoss(
                    embed_dim=config.embed_dim,
                    vocab_size=vocab_size,
                    depth_weights=depth_weights,
                    pos_weight=pos_weight,
                    level_mask=level_mask,
                    head_type=config.prediction_head_type,
                    tag_init_embeddings=tag_init_embeddings,
                    tag_embedding_bias=config.tag_embedding_bias,
                    tag_embedding_scale=config.tag_embedding_scale,
                )
            elif flavor == "per_level_softmax":
                if (
                    descendants_matrix is None
                    or level_partition is None
                    or level_weights is None
                    or tag_depths_tensor is None
                ):
                    raise ValueError(
                        "hed_loss_flavor='per_level_softmax' requires "
                        "descendants_matrix, level_partition, level_weights, "
                        "and tag_depths_tensor."
                    )
                self.hed_loss = PerLevelSoftmaxHEDLoss(
                    embed_dim=config.embed_dim,
                    vocab_size=vocab_size,
                    level_partition=level_partition,
                    level_weights=level_weights,
                    descendants_matrix=descendants_matrix,
                    tag_depths=tag_depths_tensor,
                    head_type=config.prediction_head_type,
                )
            elif flavor == "top_k_mi_softmax":
                if descendants_matrix is None or top_k_mi_indices is None:
                    raise ValueError(
                        "hed_loss_flavor='top_k_mi_softmax' requires "
                        "descendants_matrix and top_k_mi_indices."
                    )
                self.hed_loss = TopKMutualInfoHEDLoss(
                    embed_dim=config.embed_dim,
                    vocab_size=vocab_size,
                    top_k_indices=top_k_mi_indices,
                    descendants_matrix=descendants_matrix,
                    head_type=config.prediction_head_type,
                )
            else:
                raise ValueError(
                    f"Unknown hed_loss_flavor '{flavor}'. Valid: "
                    "ancestor_bce, per_level_softmax, top_k_mi_softmax."
                )

        schedule_name = config.phase_schedule
        if schedule_name not in self.NAMED_SCHEDULES:
            raise ValueError(
                f"Unknown phase_schedule '{schedule_name}'. "
                f"Valid: {sorted(self.NAMED_SCHEDULES)}"
            )
        self.phase_schedule = self.NAMED_SCHEDULES[schedule_name]

        # Current weights as buffers (persisted in state_dict for checkpoint resume)
        self.register_buffer("lambda_recon", torch.tensor(config.lambda_recon))
        self.register_buffer("lambda_mask", torch.tensor(config.lambda_mask))
        self.register_buffer("lambda_hed", torch.tensor(config.lambda_event))

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        evt_embeddings: torch.Tensor,
        hed_targets: torch.Tensor | None = None,
        recon_mask: torch.Tensor | None = None,
        task_codes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the joint loss.

        Args:
            reconstructed: (batch, channels, time) predicted EEG.
            original: (batch, channels, time) ground truth EEG.
            evt_embeddings: (batch, embed_dim) from [EVT] token output.
            hed_targets: (batch, vocab_size) multi-hot HED vectors.
            recon_mask: Optional (batch, time) mask for masked
                reconstruction (1.0 = masked position, compute loss).
            task_codes: Optional (batch,) long tensor with task indices
                (used when prediction_target="task_codes").

        Returns:
            Dict with keys 'total', 'recon', 'mask', 'hed', each a
            scalar tensor. The 'mask' key is 0 when recon_mask is None.
        """
        l_recon = self.recon_loss(reconstructed, original)

        if self.prediction_target == "task_codes" and task_codes is not None:
            assert self.event_code_loss is not None
            l_hed = self.event_code_loss(evt_embeddings, task_codes)
        elif hed_targets is not None and self.hed_loss is not None:
            l_hed = self.hed_loss(evt_embeddings, hed_targets)
        else:
            l_hed = torch.tensor(0.0, device=l_recon.device, requires_grad=False)

        if recon_mask is not None:
            l_mask = self.recon_loss(reconstructed, original, mask=recon_mask)
        else:
            l_mask = torch.tensor(0.0, device=l_recon.device, requires_grad=False)

        total = (
            self.lambda_recon * l_recon
            + self.lambda_mask * l_mask  # type: ignore[unsupported-operator]
            + self.lambda_hed * l_hed  # type: ignore[unsupported-operator]
        )

        return {
            "total": total,
            "recon": l_recon,
            "mask": l_mask,
            "hed": l_hed,
        }

    # Number of epochs over which to linearly interpolate between phases
    TRANSITION_EPOCHS: int = 10

    def update_phase(self, epoch: int) -> None:
        """Update loss weights based on training epoch.

        Uses the phase schedule selected at init (self.phase_schedule).
        At phase boundaries, weights are linearly interpolated over
        TRANSITION_EPOCHS epochs to avoid abrupt loss jumps.

        Args:
            epoch: Current training epoch (1-indexed).
        """
        schedule = self.phase_schedule

        # Find the active phase index (last phase where start_epoch <= epoch)
        phase_idx = -1
        for i, (start_epoch, _lr, _lm, _lh) in enumerate(schedule):
            if epoch >= start_epoch:
                phase_idx = i

        if phase_idx < 0:
            # epoch < 1 (should not happen), keep current weights
            return

        _, lr, lm, lh = schedule[phase_idx]

        # Interpolate at the boundary between phases
        if phase_idx > 0:
            current_start = schedule[phase_idx][0]
            epochs_into_phase = epoch - current_start
            if epochs_into_phase < self.TRANSITION_EPOCHS:
                t = epochs_into_phase / self.TRANSITION_EPOCHS
                _, prev_lr, prev_lm, prev_lh = schedule[phase_idx - 1]
                lr = prev_lr + t * (lr - prev_lr)
                lm = prev_lm + t * (lm - prev_lm)
                lh = prev_lh + t * (lh - prev_lh)

        self.lambda_recon.fill_(lr)  # type: ignore[call-non-callable]
        self.lambda_mask.fill_(lm)  # type: ignore[call-non-callable]
        self.lambda_hed.fill_(lh)  # type: ignore[call-non-callable]
