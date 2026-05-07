"""BERT-style masking for dual-stream SSL (the HED-objective ablation).

The masker is parameterised for **two** streams: a TF patch stream and an
event-token stream. Each stream uses independent BERT 80/10/10 masking and
may run at a different mask ratio (issue #173 defaults: 15% TF, 50% events).

Returned artefacts per stream:
    masked: the token tensor with the BERT replacement rule applied
        (80% [MASK], 10% random from batch, 10% unchanged)
    targets: the original token tensor at masked positions (unchanged
        elsewhere — the loss reads targets only where mask_indices is True)
    mask_indices: (..., ) bool where True = masked (80/10/10 all count)

The transformer sees ``masked``; the loss reads ``targets`` at
``mask_indices``. The TF stream masks whole patch tokens (after patch-embed);
the event stream masks whole HED vectors (the un-embedded multi-hot) and the
downstream HED loss head predicts them.
"""

from __future__ import annotations

import torch


def _sample_mask_fraction(
    n_items: int,
    mask_ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a boolean mask with an exact floor-rounded count of Trues.

    Uses argsort on uniform noise so the count is deterministic given the
    seed; avoids per-token Bernoulli draws whose batch count would fluctuate.

    Returns a (n_items,) bool tensor with ``floor(n_items * mask_ratio)``
    Trues, clipped to at least 1 when ``mask_ratio > 0`` so the loss always
    has at least one masked target per window (prevents empty-loss NaNs).
    """
    if mask_ratio <= 0:
        return torch.zeros(n_items, dtype=torch.bool, device=device)
    n_mask = max(1, int(n_items * mask_ratio))
    noise = torch.rand(n_items, device=device, generator=generator)
    ids_shuffle = torch.argsort(noise)
    mask_indices = torch.zeros(n_items, dtype=torch.bool, device=device)
    mask_indices.scatter_(0, ids_shuffle[:n_mask], True)
    return mask_indices


class DualStreamMasker:
    """BERT 80/10/10 masker for TF patches and event tokens.

    The TF stream operates on *patch embeddings* (B, E*P, D) where
    ``P = (F/ph) * (T/pw)`` patches per epoch, produced by the model's
    Conv2d patch-embed and flattened over the epoch dimension.

    The event stream operates on the *raw HED multi-hot* (B, E, V); masked
    positions are replaced with a learned [MASK_EVT] token at the model's
    event-embed layer (so the model never sees the ground-truth HED at a
    masked position). The HED loss reads the raw multi-hot back as the
    prediction target.

    Args:
        mask_ratio_tf: Fraction of TF patch tokens masked per sample.
        mask_ratio_evt: Fraction of event tokens masked per sample.
        random_frac: Fraction of masked tokens replaced by a random token
            from the same batch (BERT default 10%).
        unchanged_frac: Fraction of masked tokens left unchanged (BERT
            default 10%). The loss still computes on these.
    """

    def __init__(
        self,
        mask_ratio_tf: float = 0.15,
        mask_ratio_evt: float = 0.50,
        random_frac: float = 0.10,
        unchanged_frac: float = 0.10,
    ) -> None:
        if not 0.0 <= mask_ratio_tf <= 1.0:
            raise ValueError(f"mask_ratio_tf={mask_ratio_tf} not in [0, 1]")
        if not 0.0 <= mask_ratio_evt <= 1.0:
            raise ValueError(f"mask_ratio_evt={mask_ratio_evt} not in [0, 1]")
        if random_frac + unchanged_frac >= 1.0:
            raise ValueError(
                f"random_frac ({random_frac}) + unchanged_frac ({unchanged_frac}) "
                "must be < 1.0 so at least some masked tokens get [MASK]."
            )
        self.mask_ratio_tf = mask_ratio_tf
        self.mask_ratio_evt = mask_ratio_evt
        self.random_frac = random_frac
        self.unchanged_frac = unchanged_frac

    # -- TF patch stream -----------------------------------------------------

    def mask_tf(
        self,
        patches: torch.Tensor,
        mask_token: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask TF patch embeddings in-place of a clone.

        Args:
            patches: (B, N, D) patch embeddings after patch-embed.
            mask_token: (D,) or (1, 1, D) learned [MASK_TF] embedding; will
                be broadcast to masked positions.
            generator: Optional torch.Generator for reproducibility.

        Returns:
            masked_patches: (B, N, D) with BERT 80/10/10 replacement applied
                only at masked positions.
            targets: (B, N, D) — returns the ORIGINAL patch embeddings
                (detached from autograd) at every position; the recon loss
                reads targets only where ``mask_indices`` is True. Because
                the recon head is trained to predict the *flattened raw TF
                patch* (not the embedding), the caller is expected to pair
                this with its own raw-TF target — see
                ``bert_ssl.BertSSL.forward``.
            mask_indices: (B, N) bool mask.
        """
        if patches.ndim != 3:
            raise ValueError(f"patches must be (B, N, D); got {patches.shape}")
        b, n, _d = patches.shape
        device = patches.device

        # Per-sample independent masks.
        mask_indices = torch.zeros(b, n, dtype=torch.bool, device=device)
        for i in range(b):
            mask_indices[i] = _sample_mask_fraction(
                n, self.mask_ratio_tf, device, generator
            )

        # Expand mask_token to (B, N, D).
        mask_tok_expanded = mask_token.view(1, 1, -1).expand(b, n, -1).to(patches.dtype)

        # Decide 80/10/10 fate for each masked position independently.
        # rand < unchanged_frac     => keep original
        # rand < unchanged + random => random token from same sample
        # else                      => [MASK]
        rand = torch.rand(b, n, device=device, generator=generator)
        is_unchanged = mask_indices & (rand < self.unchanged_frac)
        is_random = (
            mask_indices
            & (rand >= self.unchanged_frac)
            & (rand < self.unchanged_frac + self.random_frac)
        )
        # Everything remaining inside mask_indices is replaced with [MASK].
        is_mask_tok = mask_indices & ~is_unchanged & ~is_random

        # Build random replacements by shuffling each sample's patch axis.
        # Draws from the same sample (within-batch) — matches BERT's
        # "random token from the batch" and preserves per-sample stats.
        perm = torch.argsort(
            torch.rand(b, n, device=device, generator=generator), dim=1
        )
        random_replacement = torch.gather(
            patches, 1, perm.unsqueeze(-1).expand_as(patches)
        )

        masked = patches.clone()
        masked = torch.where(is_random.unsqueeze(-1), random_replacement, masked)
        masked = torch.where(is_mask_tok.unsqueeze(-1), mask_tok_expanded, masked)
        # is_unchanged: keep patches[i] as-is; no action needed.

        return masked, patches.detach(), mask_indices

    # -- Event stream --------------------------------------------------------

    def mask_events(
        self,
        hed: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask raw HED multi-hot vectors with a full BERT 80/10/10 split.

        Args:
            hed: (B, E, V) multi-hot HED vectors.
            generator: Optional torch.Generator.

        Returns:
            masked_hed: (B, E, V). At "random" positions the HED vector is
                replaced with a random one from the batch (preserves
                marginal tag distribution). At "unchanged" and "mask-token"
                positions the original HED is preserved — the model
                overwrites the EMBEDDING with the [MASK_EVT] token at the
                "mask-token" positions so it never sees the ground-truth
                HED there.
            targets: (B, E, V) detached copy of the original hed — the HED
                loss reads this at masked positions.
            mask_indices: (B, E) bool mask, True at every masked slot
                (union of mask-token, random, and unchanged subsets).
            replace_with_mask_token: (B, E) bool. True at the ~80% of
                masked positions that should be overwritten by the model's
                learned [MASK_EVT] embedding. Mutually exclusive with the
                random and unchanged subsets — the caller passes this
                directly to ``HEDEmbed.forward_with_explicit_mask``.
        """
        if hed.ndim != 3:
            raise ValueError(f"hed must be (B, E, V); got {hed.shape}")
        b, e, _v = hed.shape
        device = hed.device

        mask_indices = torch.zeros(b, e, dtype=torch.bool, device=device)
        for i in range(b):
            mask_indices[i] = _sample_mask_fraction(
                e, self.mask_ratio_evt, device, generator
            )

        # Random HED replacement sampled from other (b, e) positions in the
        # batch. Shuffle flattened (B*E) batch axis so the sampled vector is
        # a real HED multi-hot (not a synthetic one) — preserves marginal
        # tag distribution.
        flat = hed.reshape(b * e, -1)
        perm = torch.randperm(b * e, device=device, generator=generator)
        random_hed = flat[perm].reshape(b, e, -1)

        # Single rand draw determines the 80/10/10 fate of every masked slot.
        # Mutually exclusive subsets: unchanged, random, mask-token.
        rand = torch.rand(b, e, device=device, generator=generator)
        is_unchanged = mask_indices & (rand < self.unchanged_frac)
        is_random = (
            mask_indices
            & (rand >= self.unchanged_frac)
            & (rand < self.unchanged_frac + self.random_frac)
        )
        replace_with_mask_token = mask_indices & ~is_unchanged & ~is_random

        masked_hed = torch.where(is_random.unsqueeze(-1), random_hed, hed)
        return masked_hed, hed.detach(), mask_indices, replace_with_mask_token
