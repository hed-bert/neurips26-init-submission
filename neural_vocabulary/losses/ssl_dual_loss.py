""" the HED-objective ablation: dual-stream SSL loss (masked TF recon + masked HED BCE)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nnf


class DualStreamSSLLoss(nn.Module):
    """Composite loss for the HED-objective ablation.

    ``total = alpha * L_recon + beta * L_HED``

    ``L_recon``: MSE over the flattened raw-patch predictions at **masked TF
    positions only**. Unmasked positions are ignored so the recon head is
    forced to hallucinate missing patches from context.

    ``L_HED``: BCE with logits over the HED multi-hot at **masked event
    positions only**, with per-tag ``pos_weight = clip(neg/pos, 1.0, 3.0)``.
    Per-tag pos_weight is mild by design (issue #173: no IDF, no aggressive
    balancing — "rare-tag misprediction is informative null").

    Args:
        pos_weight: (V,) float tensor of per-tag positive weights. The
            caller computes this once over the training set with the
            ``DualStreamSSLLoss.compute_pos_weight`` helper.
        alpha: Weight on L_recon. Default 1.0.
        beta: Weight on L_HED. Default 1.0. A value of 0.0 disables the
            stream (used by D.1.1 / D.1.2 ablations).
    """

    MIN_POS_WEIGHT: float = 1.0
    MAX_POS_WEIGHT: float = 3.0

    def __init__(
        self,
        pos_weight: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if pos_weight.ndim != 1:
            raise ValueError(f"pos_weight must be 1-D; got shape {pos_weight.shape}.")
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha={alpha} and beta={beta} must be non-negative.")
        self.register_buffer("pos_weight", pos_weight)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def compute_pos_weight(
        hed_pos_counts: torch.Tensor,
        total_count: int,
        clip_min: float = MIN_POS_WEIGHT,
        clip_max: float = MAX_POS_WEIGHT,
    ) -> torch.Tensor:
        """Return clipped per-tag pos_weight = clip(neg/pos, min, max).

        Args:
            hed_pos_counts: (V,) int or float tensor — number of positive
                multi-hot entries per tag over the training set.
            total_count: Total number of (sample, event) pairs the counts
                were computed over.
            clip_min / clip_max: Clipping bounds (defaults 1.0 / 3.0).

        Returns:
            (V,) float tensor of pos_weights.
        """
        pos = hed_pos_counts.to(torch.float32).clamp(min=1.0)
        neg = float(total_count) - pos
        neg = neg.clamp(min=1.0)
        pw = neg / pos
        return pw.clamp(min=clip_min, max=clip_max)

    def forward(
        self,
        recon_logits: torch.Tensor,
        recon_targets: torch.Tensor,
        recon_mask: torch.Tensor,
        hed_logits: torch.Tensor,
        hed_targets: torch.Tensor,
        hed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the composite loss.

        Args:
            recon_logits: (B, N_TF, raw_patch_dim).
            recon_targets: (B, N_TF, raw_patch_dim).
            recon_mask: (B, N_TF) bool. True = masked (contributes to loss).
            hed_logits: (B, E, V).
            hed_targets: (B, E, V) float multi-hot (0/1).
            hed_mask: (B, E) bool. True = masked.

        Returns:
            dict with ``total``, ``recon``, ``hed`` scalar losses and the
            masked counts.
        """
        device = recon_logits.device
        # Recon MSE on masked TF patches only.
        if recon_mask.any():
            diff = (recon_logits - recon_targets) ** 2  # (B, N, F)
            # Mean over feature dim, then mean over masked positions.
            per_token = diff.mean(dim=-1)  # (B, N)
            l_recon = per_token[recon_mask].mean()
        else:
            l_recon = torch.zeros((), device=device)

        # HED BCE on masked event tokens only, with per-tag pos_weight.
        if hed_mask.any():
            masked_logits = hed_logits[hed_mask]  # (N_masked, V)
            masked_targets = hed_targets[hed_mask]  # (N_masked, V)
            pos_weight: torch.Tensor = self.pos_weight  # type: ignore[assignment]
            l_hed = nnf.binary_cross_entropy_with_logits(
                masked_logits,
                masked_targets,
                pos_weight=pos_weight,
                reduction="mean",
            )
        else:
            l_hed = torch.zeros((), device=device)

        total = self.alpha * l_recon + self.beta * l_hed
        return {
            "total": total,
            "recon": l_recon,
            "hed": l_hed,
            "n_masked_recon": torch.as_tensor(
                int(recon_mask.sum().item()), device=device
            ),
            "n_masked_hed": torch.as_tensor(int(hed_mask.sum().item()), device=device),
        }
