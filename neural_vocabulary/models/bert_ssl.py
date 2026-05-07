""" the HED-objective ablation: BERT-style dual-stream masked SSL on TF spectrograms + HED.

Architecture (~2 M params, matches Gate C apples-to-apples):

  Input window (non-movie stim/response):
    - TF: (B, E=8, F=6, C=64, T=10) Morlet log-power
    - HED: (B, E=8, V=1124) multi-hot event vectors

  Per-epoch TF → patches via Conv2d(64 → d_model, k=stride=(2, 2)):
    (B*E, 64, 6, 10) → (B*E, d_model, 3, 5) → flatten (B, E*15, d_model)

  Event tokens via HEDEmbed (tag-embedding with  hierarchy init):
    (B, E, V) → (B, E, d_model)

  Token sequence per window:
    [CLS] + E event tokens + E * n_patches_per_epoch TF patches
    = 1 + 8 + 120 = 129 tokens at d_model=192

  Pre-norm transformer (depth 4, heads 6, MLP ratio 4) with SDPA attention.
  ~2 M params at default flat-mode config; channel_token grows the model.

  Two heads, computed on the final transformer output:
    - Recon: Linear(d → F_raw * C_raw * ph * pw) predicts the FLATTENED raw
      TF patch corresponding to each TF-patch token. Loss = MSE over masked
      TF tokens only.
    - HED:    tag-embedding head ``logit = (token @ tag_emb.T) * scale``
      with per-tag base-rate bias and hierarchy-aware init. Loss = BCE with
      per-tag pos_weight clip(neg/pos, 1, 3) over masked event tokens only.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import nn
from torch.nn import functional as nnf

if TYPE_CHECKING:
    from neural_vocabulary.data.masking import DualStreamMasker

logger = logging.getLogger(__name__)

PatchMode = Literal["flat", "channel_token"]


def _topographic_channel_pos_embedding(
    channel_names: list[str], d_model: int
) -> torch.Tensor:
    """Build a (n_channels, d_model) position embedding from 10-20 coords.

    Uses MNE's ``standard_1005`` montage to look up the 3D position of
    each channel name, then applies a NeRF-style sinusoidal encoding
    so that channels that are spatially close share similar embeddings
    (preserving lateralization, antero-posterior structure, etc.).

    Args:
        channel_names: 64 channel names, in the dataset's row order.
        d_model: Output embedding dim. Must be divisible by 6 (3 axes
            × 2 functions per axis).

    Returns:
        ``(n_channels, d_model)`` float tensor.
    """
    if d_model % 6 != 0:
        raise ValueError(
            f"d_model={d_model} not divisible by 6 (needed for x,y,z × sin,cos "
            "split in topographic init)."
        )
    # Lazy import — MNE is a heavy import for the base BertSSL module.
    import mne
    import numpy as np

    montage = mne.channels.make_standard_montage("standard_1005")
    positions = montage.get_positions()["ch_pos"]
    missing = [name for name in channel_names if name not in positions]
    if missing:
        raise ValueError(
            f"Channels not in MNE standard_1005 montage: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}. Either pass exact "
            "10-20 / 10-10 names, or supply pre-computed coordinates "
            "via a custom init."
        )
    coords_np = np.stack(
        [positions[name] for name in channel_names], axis=0
    )  # (n_channels, 3) in metres
    coords = torch.from_numpy(coords_np).to(torch.float32)
    # Normalize coords to a unit cube using the actual range so frequencies
    # land in a reasonable band regardless of montage scale.
    coords = (coords - coords.mean(dim=0)) / (coords.std(dim=0) + 1e-8)

    n_freq_bands = d_model // 6
    # Geometric frequency progression, NeRF-style (2^0 ... 2^(n_freq_bands-1)).
    freq_bands = 2.0 ** torch.arange(n_freq_bands, dtype=torch.float32)

    # (n_channels, 3, 1) * (n_freq_bands,) → (n_channels, 3, n_freq_bands).
    scaled = coords.unsqueeze(-1) * freq_bands * math.pi
    sin_part = torch.sin(scaled)
    cos_part = torch.cos(scaled)
    # (n_channels, 3, 2*n_freq_bands) → (n_channels, d_model).
    out = torch.cat([sin_part, cos_part], dim=-1).reshape(len(channel_names), -1)
    return out


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _MultiHeadSelfAttention(nn.Module):
    """SDPA-based MHSA with optional key_padding_mask."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(dim=0)

        # SDPA attn_mask semantics: bool mask where True = allowed. We have
        # key_padding_mask where True = pad (ignore); invert and broadcast
        # over heads and query axis.
        attn_mask: torch.Tensor | None = None
        if key_padding_mask is not None:
            allow = (~key_padding_mask).to(torch.bool)  # (B, N)
            # (B, 1, 1, N) broadcast over heads and queries.
            attn_mask = allow.unsqueeze(1).unsqueeze(1)
            attn_mask = attn_mask.expand(b, self.num_heads, n, n)

        out = nnf.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MultiHeadSelfAttention(dim, num_heads=num_heads)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


# -----------------------------------------------------------------------------
# HEDEmbed —  tag-embedding with [MASK_EVT] substitution
# -----------------------------------------------------------------------------


class HEDEmbed(nn.Module):
    """Event-token embedder backed by a learnable (V, D) tag-embedding matrix.

    Given a HED multi-hot ``h_i in {0, 1}^V`` the event embedding is
    ``e_i = (h / max(1, sum(h))) @ tag_emb``. Normalising by the active tag
    count keeps event-token scale roughly invariant to HED density.

    At masked positions the token is replaced by a learned ``[MASK_EVT]``
    embedding. At masked-random positions the caller has already replaced
    the input HED vector with a random one sampled from the batch (see
    ``masking.DualStreamMasker.mask_events``), so those positions pass
    through the standard averaging path naturally.

    The tag-embedding matrix is shared with the HED prediction head so the
    encoder and decoder sit in the same vocabulary space ('s jewel —
    ``neural_vocabulary/losses/hed_loss.py`` head_type="tag_embedding").
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        tag_init_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        if tag_init_embeddings is not None:
            if tag_init_embeddings.shape != (vocab_size, embed_dim):
                raise ValueError(
                    f"tag_init_embeddings shape {tag_init_embeddings.shape} "
                    f"does not match (vocab_size={vocab_size}, "
                    f"embed_dim={embed_dim})."
                )
            self.tag_embeddings = nn.Parameter(tag_init_embeddings.clone())
        else:
            # Matches  default: small-scale random init.
            self.tag_embeddings = nn.Parameter(
                torch.randn(vocab_size, embed_dim) * 0.02
            )
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        hed: torch.Tensor,
        mask_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed multi-hot HED vectors to (B, E, D).

        Args:
            hed: (B, E, V) float multi-hot (may contain BERT-random rows
                from the masker at random-substitution positions).
            mask_indices: (B, E) bool. True positions are overwritten with
                the learned [MASK_EVT] embedding (80% of masked positions —
                the 10% random and 10% unchanged subsets are already handled
                by the masker). If None, no mask substitution is applied
                (useful for eval / frozen-encoder probes).

        Returns:
            (B, E, D) event-token embeddings.
        """
        # Normalise by active-tag count with a floor of 1 to avoid div-by-zero.
        active = hed.sum(dim=-1, keepdim=True).clamp(min=1.0)
        normalised = hed / active  # (B, E, V)
        emb = normalised @ self.tag_embeddings  # (B, E, D)
        if mask_indices is None:
            return emb
        # Overwrite masked positions with the learned mask token. The 10%
        # random and 10% unchanged splits are applied by the masker at the
        # input level (random HED → different normalised embedding, unchanged
        # passes through). The remaining 80% of masked positions get the
        # explicit [MASK_EVT] here. We mark 80% via a fresh subsample of the
        # True positions inside mask_indices (matches BERT's three-way split).
        if not mask_indices.any():
            return emb
        # The masker already determined the 10/10 split via its rand draws,
        # but to avoid recomputing here we approximate: if the input hed at
        # a masked position equals its un-normalised target marginal, the
        # masker left it unchanged or randomised; otherwise the caller has
        # asked us to emit the mask token. To keep the interface minimal
        # we accept the simpler rule: mask_indices True => replace embedding
        # with [MASK_EVT] except where the input HED is already a non-zero
        # BERT-random replacement. In practice the SSL harness builds the
        # 80/10/10 partition at the masker level and passes a bool
        # ``replace_with_mask_token`` tensor via the ``explicit_mask`` kwarg
        # of ``forward_with_explicit_mask`` below when the 10% splits matter.
        mask_tok = self.mask_token.view(1, 1, -1).to(emb.dtype).expand_as(emb)
        return torch.where(mask_indices.unsqueeze(-1), mask_tok, emb)

    def forward_with_explicit_mask(
        self,
        hed: torch.Tensor,
        explicit_mask_token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Embed with explicit per-position control of [MASK_EVT] substitution.

        Args:
            hed: (B, E, V).
            explicit_mask_token_positions: (B, E) bool. True => substitute
                with the learned [MASK_EVT] embedding; False => use the HED
                multi-hot embedding (which may itself be a BERT-random
                replacement the caller drew). This is the path the the HED-objective ablation
                training loop uses.

        Returns:
            (B, E, D).
        """
        active = hed.sum(dim=-1, keepdim=True).clamp(min=1.0)
        normalised = hed / active
        emb = normalised @ self.tag_embeddings
        mask_tok = self.mask_token.view(1, 1, -1).to(emb.dtype).expand_as(emb)
        return torch.where(explicit_mask_token_positions.unsqueeze(-1), mask_tok, emb)


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------


class BertSSL(nn.Module):
    """Dual-stream masked SSL transformer (TF patches + HED event tokens).

    Args:
        vocab_size: HED vocabulary size (1124 on HBN preprocessed).
        tag_init_embeddings: Optional (vocab_size, d_model) hierarchy-aware
            tag-embedding init from
            ``HEDVectorizer.get_hierarchy_init_embeddings``. Shared between
            the event embedder and the HED prediction head ( jewel).
        epochs_per_window: Number of per-epoch TF spectrograms per window.
        n_channels / n_freqs / n_time: Per-epoch TF layout.
        patch_size: 2D Conv patch size over (freq, time).
        d_model, depth, num_heads, mlp_ratio, dropout: Transformer config.
        tag_embedding_scale: Logit scaling for the HED head ( default
            matches ``sqrt(d_model)`` to compensate for dot-product
            magnitude — ``1.0 / sqrt(d_model)`` would shrink too far).
    """

    def __init__(
        self,
        vocab_size: int,
        tag_init_embeddings: torch.Tensor | None = None,
        epochs_per_window: int = 8,
        n_channels: int = 64,
        n_freqs: int = 6,
        n_time: int = 10,
        patch_size: tuple[int, int] = (2, 2),
        d_model: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        tag_embedding_scale: float | None = None,
        patch_mode: PatchMode = "flat",
        channel_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        ph, pw = patch_size
        if n_freqs % ph != 0:
            raise ValueError(f"n_freqs={n_freqs} not divisible by patch_size[0]={ph}.")
        if n_time % pw != 0:
            raise ValueError(f"n_time={n_time} not divisible by patch_size[1]={pw}.")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )
        if patch_mode not in ("flat", "channel_token"):
            raise ValueError(
                f"patch_mode must be 'flat' or 'channel_token', got {patch_mode!r}."
            )

        self.vocab_size = vocab_size
        self.epochs_per_window = epochs_per_window
        self.n_channels = n_channels
        self.n_freqs = n_freqs
        self.n_time = n_time
        self.patch_size = patch_size
        self.d_model = d_model
        self.patch_mode = patch_mode

        # Per-channel freq-time patch grid (shared across modes).
        n_patches_per_epoch_per_channel = (n_freqs // ph) * (n_time // pw)

        if patch_mode == "flat":
            # Existing behavior: Conv2d collapses channels in the first op.
            # Per-epoch: (1, 64, 6, 10) → (1, d_model, 3, 5) → 15 tokens.
            self.n_patches_per_epoch = n_patches_per_epoch_per_channel
            self.raw_patch_dim = ph * pw * n_channels
            self.tf_patch_embed = nn.Conv2d(
                in_channels=n_channels,
                out_channels=d_model,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            )
        else:
            # channel_token: each (channel, freq-patch, time-patch) is its
            # own token. Conv2d takes one channel at a time so it cannot
            # average lateralized signals away. Per-epoch tokens grow ~64×.
            self.n_patches_per_epoch = n_channels * n_patches_per_epoch_per_channel
            self.raw_patch_dim = ph * pw  # one channel's freq-time patch
            self.tf_patch_embed = nn.Conv2d(
                in_channels=1,
                out_channels=d_model,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            )

        self.n_patches_per_epoch_per_channel = n_patches_per_epoch_per_channel
        self.n_tf_tokens = epochs_per_window * self.n_patches_per_epoch
        # Token layout inside the transformer: [CLS] + E event + n_tf TF.
        self.n_event_tokens = epochs_per_window
        self.n_total_tokens = 1 + self.n_event_tokens + self.n_tf_tokens

        # Channel positional embedding for channel_token mode. Initialize
        # from MNE ``standard_1005`` 3D coords for ``channel_names``
        # (defaults to ``ChannelHarmonization.TARGET_64_CHANNELS``) so
        # spatially close electrodes share similar starting embeddings.
        # Pre-computed lookup table from channel-id → TF token index goes
        # alongside.
        if patch_mode == "channel_token":
            from neural_vocabulary.models.channel_harmonization import (
                TARGET_64_CHANNELS,
            )

            ch_names = channel_names or TARGET_64_CHANNELS
            if len(ch_names) != n_channels:
                raise ValueError(
                    f"channel_names has {len(ch_names)} entries, expected "
                    f"n_channels={n_channels}."
                )
            init = _topographic_channel_pos_embedding(ch_names, d_model)
            self.channel_pos_embed = nn.Parameter(init)
            # TF-slot → channel-id lookup. Slot order inside one window:
            # for each epoch, for each channel, for each per-channel patch.
            channel_ids = torch.arange(n_channels).repeat_interleave(
                n_patches_per_epoch_per_channel
            )  # one epoch's channel pattern: [c0, c0, ..., c1, c1, ...]
            channel_ids = channel_ids.repeat(epochs_per_window)  # (n_tf_tokens,)
            self.register_buffer("tf_token_channel_ids", channel_ids, persistent=False)

        # Event embedder ( tag-embedding with shared weights for HED head).
        self.event_embed = HEDEmbed(
            vocab_size=vocab_size,
            embed_dim=d_model,
            tag_init_embeddings=tag_init_embeddings,
        )

        # Learnable [CLS] and [MASK_TF] tokens.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_tf_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_tf_token, std=0.02)

        # Learned 1D positional embedding over the full token sequence
        # (CLS + events + TF patches). Same slot ordering every forward.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_total_tokens, d_model))

        # Token-type embedding (3 types: CLS, EVT, TF). Cheap (3 * d_model)
        # and gives the transformer a clean signal about which stream each
        # token belongs to — BERT precedent for type embeddings.
        self.type_embed = nn.Parameter(torch.zeros(3, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _Block(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        # Heads ---------------------------------------------------------------
        # Reconstruction head: linear decoder from TF-token embedding to the
        # flattened raw patch (ph * pw * n_channels).
        self.recon_head = nn.Linear(d_model, self.raw_patch_dim)

        # HED head: shared tag-embedding (). Logit = scale * (token @ W_tag^T)
        # with per-tag learned bias. Weights live inside self.event_embed so
        # the encoder and decoder are tied.
        if tag_embedding_scale is None:
            #  picks sqrt(d_model) as a default so logit magnitudes match
            # a softmax-scale dot product; see losses/hed_loss.py.
            tag_embedding_scale = float(d_model) ** 0.5
        self.tag_embedding_scale = tag_embedding_scale
        self.hed_bias = nn.Parameter(torch.zeros(vocab_size))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def no_weight_decay_param_names(self) -> set[str]:
        """-style param-group split: exclude embeddings, biases, LayerNorm."""
        names: set[str] = set()
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (
                name.endswith(".bias")
                or p.ndim == 1
                or name
                in {
                    "cls_token",
                    "pos_embed",
                    "type_embed",
                    "mask_tf_token",
                    "event_embed.mask_token",
                    "event_embed.tag_embeddings",
                    "hed_bias",
                    "channel_pos_embed",
                }
            ):
                names.add(name)
        return names

    def build_param_groups(self, weight_decay: float) -> list[dict[str, object]]:
        no_decay_names = self.no_weight_decay_param_names()
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name in no_decay_names:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    # ------------------------------------------------------------------

    def _embed_tf(self, tf: torch.Tensor) -> torch.Tensor:
        """Patch-embed a TF window to per-epoch patch tokens.

        Args:
            tf: (B, E, F, C, T).

        Returns:
            patches: (B, E*n_patches_per_epoch, d_model).
        """
        b, e, f, c, t = tf.shape
        if f != self.n_freqs or c != self.n_channels or t != self.n_time:
            raise ValueError(
                f"tf shape (F={f}, C={c}, T={t}) does not match model "
                f"(n_freqs={self.n_freqs}, n_channels={self.n_channels}, "
                f"n_time={self.n_time})."
            )
        if self.patch_mode == "flat":
            # (B,E,F,C,T) → (B,E,C,F,T) → (B*E, C, F, T) — Conv2d treats
            # EEG channels as the input dim. Same as Gate B.1 / C.1.
            tf_flat = tf.transpose(2, 3).reshape(b * e, c, f, t)
            patched = self.tf_patch_embed(tf_flat)  # (B*E, d, F/ph, T/pw)
            _, d, hh, ww = patched.shape
            patches = patched.flatten(2).transpose(1, 2)  # (B*E, n_p, d)
            patches = patches.reshape(b, e * (hh * ww), d)
            return patches

        # channel_token: each channel embedded independently via
        # Conv2d(in=1). (B,E,F,C,T) → (B,E,C,F,T) → (B*E*C, 1, F, T).
        tf_per_channel = tf.transpose(2, 3).reshape(b * e * c, 1, f, t)
        patched = self.tf_patch_embed(tf_per_channel)  # (B*E*C, d, F/ph, T/pw)
        _, d, hh, ww = patched.shape
        # (B*E*C, d, hh, ww) → (B*E*C, n_p_pc, d) → (B, E, C, n_p_pc, d)
        # → (B, E*C*n_p_pc, d). Within an epoch the order is
        # (c0_p0, c0_p1, ..., c1_p0, ...), matching tf_token_channel_ids.
        patches = patched.flatten(2).transpose(1, 2)  # (B*E*C, n_p_pc, d)
        patches = patches.reshape(b, e * c * (hh * ww), d)
        return patches

    def _flatten_raw_tf_targets(self, tf: torch.Tensor) -> torch.Tensor:
        """Build the per-patch raw TF target tensor.

        ``flat`` mode: each (ph × pw) freq-time cell is concatenated
        across all channels, giving shape ``(B, E*n_p, ph * pw * C)``.

        ``channel_token`` mode: each (channel, freq-patch, time-patch)
        is its own target, giving shape ``(B, E*C*n_p_pc, ph * pw)``.
        Layout matches ``_embed_tf``: per epoch, channels run slow,
        freq-time patches run fast.
        """
        b, e, f, c, t = tf.shape
        ph, pw = self.patch_size
        n_p_pc = (f // ph) * (t // pw)
        # Common fold step: (B, E, F/ph, ph, C, T/pw, pw).
        tf_blocked = tf.reshape(b, e, f // ph, ph, c, t // pw, pw)

        if self.patch_mode == "flat":
            # Permute to (B, E, F/ph, T/pw, ph, pw, C), flatten freq/time
            # into patch axis and (ph, pw, C) into the feature axis.
            flat = tf_blocked.permute(0, 1, 2, 5, 3, 6, 4)
            flat = flat.reshape(b, e, n_p_pc, ph * pw * c)
            return flat.reshape(b, e * n_p_pc, ph * pw * c)

        # channel_token: (B, E, C, F/ph, T/pw, ph, pw) — channel slow,
        # freq-time fast. Flatten to (B, E*C*n_p_pc, ph*pw). Matches
        # _embed_tf's slot ordering inside each epoch.
        per_channel = tf_blocked.permute(0, 1, 4, 2, 5, 3, 6)
        per_channel = per_channel.reshape(b, e, c, n_p_pc, ph * pw)
        return per_channel.reshape(b, e * c * n_p_pc, ph * pw)

    # ------------------------------------------------------------------

    def forward(
        self,
        tf: torch.Tensor,
        hed: torch.Tensor,
        masker: DualStreamMasker | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for dual-stream masked SSL.

        Args:
            tf: (B, E, F, C, T).
            hed: (B, E, V).
            masker: If provided, run masked-SSL forward. If None, run a
                plain encoder forward (used by the frozen-encoder probe).
            generator: Optional torch.Generator for mask reproducibility.

        Returns:
            dict with:
                - evt_embeddings: (B, E, d_model) event-token outputs
                - cls_embedding:  (B, d_model) CLS-token output
                - recon_logits:   (B, E*n_p, raw_patch_dim) — TF patch logits
                    (raw). Present only when masker is not None.
                - recon_targets:  (B, E*n_p, raw_patch_dim) — flat raw patches
                - recon_mask:     (B, E*n_p) bool
                - hed_logits:     (B, E, V) when masker is not None
                - hed_targets:    (B, E, V) float multi-hot (detached)
                - hed_mask:       (B, E) bool
        """
        b, e = tf.shape[0], tf.shape[1]
        if e != self.epochs_per_window:
            raise ValueError(
                f"tf has E={e} epochs, model expects {self.epochs_per_window}."
            )
        if hed.shape[:2] != (b, e):
            raise ValueError(
                f"hed shape {hed.shape} head-mismatch; expected ({b}, {e}, V)."
            )
        if hed.shape[2] != self.vocab_size:
            raise ValueError(
                f"hed vocab dim {hed.shape[2]} != model vocab_size {self.vocab_size}."
            )

        # --- Patch-embed TF -------------------------------------------------
        tf_patches = self._embed_tf(tf)  # (B, E*n_p, d)
        raw_tf_targets = self._flatten_raw_tf_targets(tf)  # (B, E*n_p, raw_d)

        # --- Masking --------------------------------------------------------
        tf_mask_indices: torch.Tensor | None = None
        evt_mask_indices: torch.Tensor | None = None
        replace_with_mask_token: torch.Tensor | None = None
        masked_hed_input = hed
        if masker is not None:
            masked_patches, _tf_targets_embedded, tf_mask_indices = masker.mask_tf(
                tf_patches,
                mask_token=self.mask_tf_token,
                generator=generator,
            )
            tf_patches = masked_patches
            (
                masked_hed_input,
                _,
                evt_mask_indices,
                replace_with_mask_token,
            ) = masker.mask_events(hed, generator=generator)

        # --- Event embedding ------------------------------------------------
        # ``replace_with_mask_token`` is the BERT 80% subset returned by the
        # masker — mutually exclusive with the 10% random (already applied
        # in ``masked_hed_input``) and the 10% unchanged positions. The
        # masker is the single source of truth for the 80/10/10 split so
        # random / unchanged / mask-token decisions never diverge between
        # the masker and the embedder (earlier revisions re-drew ``rand``
        # here and could stamp [MASK_EVT] on top of a random or unchanged
        # position, corrupting the BERT contract).
        if replace_with_mask_token is not None:
            event_tokens = self.event_embed.forward_with_explicit_mask(
                masked_hed_input, replace_with_mask_token
            )
        else:
            event_tokens = self.event_embed(masked_hed_input, mask_indices=None)

        # --- Token assembly ------------------------------------------------
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, d)
        tokens = torch.cat([cls, event_tokens, tf_patches], dim=1)

        # Add positional + type embeddings.
        type_ids = torch.zeros(
            self.n_total_tokens, dtype=torch.long, device=tokens.device
        )
        type_ids[0] = 0  # CLS
        type_ids[1 : 1 + self.n_event_tokens] = 1  # EVT
        type_ids[1 + self.n_event_tokens :] = 2  # TF
        type_emb = self.type_embed[type_ids].unsqueeze(0)  # (1, N, d)
        tokens = tokens + self.pos_embed + type_emb

        # In channel_token mode, factorize the TF positional encoding by
        # adding a per-channel embedding on top of the sequential pos
        # embed. The sequential pos embed alone cannot disambiguate
        # which channel a TF token came from, since adjacent slots in
        # the sequence belong to the same channel block.
        if self.patch_mode == "channel_token":
            channel_ids = cast("torch.Tensor", self.tf_token_channel_ids)
            tf_channel_emb = self.channel_pos_embed[channel_ids]  # (n_tf, d)
            tokens[:, 1 + self.n_event_tokens :, :] = (
                tokens[:, 1 + self.n_event_tokens :, :] + tf_channel_emb
            )

        # --- Transformer ---------------------------------------------------
        # No padding; all positions are valid.
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]
        evt_out = tokens[:, 1 : 1 + self.n_event_tokens]
        tf_out = tokens[:, 1 + self.n_event_tokens :]

        result: dict[str, torch.Tensor] = {
            "cls_embedding": cls_out,
            "evt_embeddings": evt_out,
            "tf_embeddings": tf_out,
        }

        if masker is None:
            return result

        # --- Heads ---------------------------------------------------------
        recon_logits = self.recon_head(tf_out)  # (B, E*n_p, raw_patch_dim)
        hed_logits = (
            evt_out @ self.event_embed.tag_embeddings.T * self.tag_embedding_scale
            + self.hed_bias
        )  # (B, E, V)

        assert tf_mask_indices is not None
        assert evt_mask_indices is not None
        result["recon_logits"] = recon_logits
        result["recon_targets"] = raw_tf_targets
        result["recon_mask"] = tf_mask_indices
        result["hed_logits"] = hed_logits
        result["hed_targets"] = hed.detach()
        result["hed_mask"] = evt_mask_indices
        return result
