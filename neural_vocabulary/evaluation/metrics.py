"""Semantic-level-aware evaluation metrics for HED tag prediction.

Uses HEDVectorizer.classify_tag() to group tags by semantic level (L0-L4),
then computes F1/precision/recall per level, silhouette scores, and
cross-version comparisons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import silhouette_score

if TYPE_CHECKING:
    from neural_vocabulary.data.hed_vectorizer import HEDVectorizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LevelMetrics:
    """Metrics for a single semantic level."""

    level: int
    n_tags: int
    precision: float
    recall: float
    f1: float
    tag_indices: tuple[int, ...] = field(default_factory=tuple, repr=False)


@dataclass(frozen=True)
class EvalResult:
    """Complete evaluation result across all semantic levels."""

    per_level: dict[int, LevelMetrics]
    overall_f1: float
    overall_precision: float
    overall_recall: float
    best_threshold: float
    silhouette_scores: dict[int, float]

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"Overall: F1={self.overall_f1:.3f} P={self.overall_precision:.3f} "
            f"R={self.overall_recall:.3f} (thresh={self.best_threshold:.2f})",
        ]
        for level in sorted(self.per_level):
            m = self.per_level[level]
            sil = self.silhouette_scores.get(level, float("nan"))
            lines.append(
                f"  L{level} ({m.n_tags:3d} tags): "
                f"F1={m.f1:.3f} P={m.precision:.3f} R={m.recall:.3f} "
                f"sil={sil:.3f}"
            )
        return "\n".join(lines)


def _compute_prf(
    preds: np.ndarray, targets: np.ndarray, tag_indices: list[int] | None = None
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for selected tag columns.

    Args:
        preds: (N, vocab_size) binary predictions.
        targets: (N, vocab_size) binary targets.
        tag_indices: Column indices to evaluate. If None, use all.

    Returns:
        (precision, recall, f1) tuple.
    """
    if tag_indices is not None:
        p = preds[:, tag_indices]
        t = targets[:, tag_indices]
    else:
        p = preds
        t = targets

    tp = float((p * t).sum())
    fp = float((p * (1 - t)).sum())
    fn = float(((1 - p) * t).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


class SemanticMetrics:
    """Compute semantic-level-aware evaluation metrics.

    Groups tags by semantic level using HEDVectorizer.classify_tag(),
    then computes per-level and overall metrics.
    """

    def __init__(self, vectorizer: HEDVectorizer) -> None:
        self.vectorizer = vectorizer
        self._level_indices: dict[int, list[int]] = {}
        self._build_level_indices()

    def _build_level_indices(self) -> None:
        """Group tag indices by semantic level."""
        idx_to_tag = self.vectorizer.idx_to_tag
        for idx, tag in idx_to_tag.items():
            level, _ = self.vectorizer.classify_tag(tag)
            self._level_indices.setdefault(level, []).append(idx)

        for level in sorted(self._level_indices):
            logger.debug(
                "Semantic level %d: %d tags", level, len(self._level_indices[level])
            )

    def find_best_threshold(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        thresholds: list[float] | None = None,
    ) -> float:
        """Sweep thresholds to find the one with best overall F1.

        Args:
            logits: (N, vocab_size) raw logits.
            targets: (N, vocab_size) binary targets.
            thresholds: Thresholds to try. Default: [0.1, 0.2, ..., 0.7].

        Returns:
            Best threshold value.
        """
        from scipy.special import expit

        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        probs = expit(logits)
        has_target = targets.sum(axis=1) > 0
        if has_target.sum() == 0:
            return 0.5

        best_f1 = -1.0
        best_thresh = 0.5
        for thresh in thresholds:
            preds = (probs[has_target] > thresh).astype(np.float32)
            _, _, f1 = _compute_prf(preds, targets[has_target])
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        return best_thresh

    def evaluate(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        threshold: float | None = None,
        embeddings: np.ndarray | None = None,
    ) -> EvalResult:
        """Run full semantic-level evaluation.

        Args:
            logits: (N, vocab_size) raw logits from prediction head.
            targets: (N, vocab_size) binary target vectors.
            threshold: Sigmoid threshold for binarization.
                If None, auto-selects via threshold sweep.
            embeddings: (N, embed_dim) [EVT] embeddings for silhouette
                score computation. Optional.

        Returns:
            EvalResult with per-level and overall metrics.
        """
        from scipy.special import expit

        probs = expit(logits)

        if threshold is None:
            threshold = self.find_best_threshold(logits, targets)

        # Filter to samples that have at least one positive target
        has_target = targets.sum(axis=1) > 0
        preds = (probs[has_target] > threshold).astype(np.float32)
        tgts = targets[has_target]

        # Overall metrics
        overall_p, overall_r, overall_f1 = _compute_prf(preds, tgts)

        # Per-level metrics
        per_level: dict[int, LevelMetrics] = {}
        for level, indices in sorted(self._level_indices.items()):
            p, r, f1 = _compute_prf(preds, tgts, indices)
            per_level[level] = LevelMetrics(
                level=level,
                n_tags=len(indices),
                precision=p,
                recall=r,
                f1=f1,
                tag_indices=tuple(indices),
            )

        # Silhouette scores per semantic level
        sil_scores: dict[int, float] = {}
        if embeddings is not None:
            emb = embeddings[has_target]
            for level, indices in sorted(self._level_indices.items()):
                sil = self._silhouette_for_level(emb, tgts, indices)
                if sil is not None:
                    sil_scores[level] = sil

        return EvalResult(
            per_level=per_level,
            overall_f1=overall_f1,
            overall_precision=overall_p,
            overall_recall=overall_r,
            best_threshold=threshold,
            silhouette_scores=sil_scores,
        )

    def _silhouette_for_level(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        tag_indices: list[int],
        max_samples: int = 5000,
    ) -> float | None:
        """Compute silhouette score for embeddings grouped by dominant tag at a level.

        Assigns each sample to its dominant (highest-activation) tag within
        the given indices, then computes silhouette score.

        Args:
            embeddings: (N, embed_dim) embedding vectors.
            targets: (N, vocab_size) binary target vectors.
            tag_indices: Indices of tags at this semantic level.
            max_samples: Subsample for speed.

        Returns:
            Silhouette score, or None if fewer than 2 clusters.
        """
        level_targets = targets[:, tag_indices]
        # Filter out samples with no active tag at this level
        has_any = level_targets.sum(axis=1) > 0
        if has_any.sum() < 10:
            return None
        level_targets = level_targets[has_any]
        embeddings = embeddings[has_any]

        # Assign each sample to its dominant tag at this level
        labels = np.argmax(level_targets, axis=1)

        # Only evaluate if there are at least 2 distinct labels with >1 sample
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[counts > 1]
        if len(valid_labels) < 2:
            return None

        # Subsample for speed
        n = min(max_samples, len(embeddings))
        indices = np.random.RandomState(42).choice(len(embeddings), n, replace=False)
        emb_sub = embeddings[indices]
        labels_sub = labels[indices]

        # Filter to samples with valid labels
        mask = np.isin(labels_sub, valid_labels)
        if mask.sum() < 10:
            return None

        return float(silhouette_score(emb_sub[mask], labels_sub[mask]))

    def compare_versions(
        self,
        results: dict[str, EvalResult],
    ) -> str:
        """Generate a comparison table across model versions.

        Args:
            results: Mapping from version name to EvalResult.

        Returns:
            Formatted comparison string.
        """
        versions = sorted(results.keys())
        all_levels = sorted({level for r in results.values() for level in r.per_level})

        lines = ["Version comparison:"]
        header = f"{'Level':<8}" + "".join(f"  {v:>12}" for v in versions)
        lines.append(header)
        lines.append("-" * len(header))

        for level in all_levels:
            row = f"L{level:<7}"
            for v in versions:
                m = results[v].per_level.get(level)
                if m:
                    row += f"  {m.f1:>12.3f}"
                else:
                    row += f"  {'n/a':>12}"
            lines.append(row)

        row = f"{'Overall':<8}"
        for v in versions:
            row += f"  {results[v].overall_f1:>12.3f}"
        lines.append(row)

        return "\n".join(lines)
