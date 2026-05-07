"""Representation-collapse detector for frozen-encoder probes (V11 E3).

Finding 57 (PR #192) surfaced an encoder mode-collapse: under
``AllEventsMasker`` with mask_ratio_evt=1.0, every window's masked-event
slot received the *same* learned MASK token, so the encoder produced
identical per-trial features regardless of stimulus class. Class-mean
cosine across all paradigm classes was 1.0000 — perfectly collapsed.

This file makes that diagnostic an automated assertion. Probe scripts
call ``assert_no_class_mean_collapse`` immediately after feature
extraction; default behavior is to raise ``RepresentationCollapseError``
when any pairwise class-mean cosine exceeds the threshold.

Default threshold 0.99 was picked from Finding 57's empirical signature:
collapsed runs sit at 1.0000 within numerical noise (D.1.4 ERP-CORE
under AllEventsMasker). 0.99 catches that dead case with margin; the
lower bound of the healthy regime is not yet characterized empirically
and may need tuning when V11 produces non-collapsed encoders.

The ``allow_collapse`` flag (wired to ``--allow-collapse`` in probe
scripts) lets diagnostic runs proceed past the assertion to surface the
collapse value in the result JSON. Default is fail-loud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RepresentationCollapseError(RuntimeError):
    """Raised when class-mean cosine exceeds the collapse threshold.

    Attributes:
        max_cosine: The largest pairwise class-mean cosine observed.
        class_a: First class id of the offending pair.
        class_b: Second class id of the offending pair.
        threshold: The threshold that was crossed.
    """

    def __init__(
        self,
        *,
        max_cosine: float,
        class_a: int,
        class_b: int,
        threshold: float,
    ) -> None:
        self.max_cosine = float(max_cosine)
        self.class_a = int(class_a)
        self.class_b = int(class_b)
        self.threshold = float(threshold)
        super().__init__(
            f"Representation collapse: class-mean cosine "
            f"{max_cosine:.4f} > {threshold:.4f} between classes "
            f"{class_a} and {class_b}. Encoder is producing identical "
            "features for distinct classes (Finding 57 signature)."
        )


@dataclass(frozen=True)
class CollapseReport:
    """Diagnostic readout — populated regardless of whether collapse fires."""

    max_cosine: float
    class_a: int
    class_b: int
    n_classes: int
    n_features: int
    pairwise_cosines: list[tuple[int, int, float]]


def _class_mean_features(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class mean feature vector. Returns (means, class_ids)."""
    classes = np.unique(labels)
    means = np.stack([features[labels == c].mean(axis=0) for c in classes], axis=0)
    return means, classes


def _pairwise_cosine_max(
    means: np.ndarray, classes: np.ndarray
) -> tuple[float, int, int, list[tuple[int, int, float]]]:
    """Largest off-diagonal cosine across class-mean rows."""
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    unit = means / safe
    sim = unit @ unit.T

    n = means.shape[0]
    pairs: list[tuple[int, int, float]] = []
    max_cos = -np.inf
    pair = (int(classes[0]), int(classes[0]))
    for i in range(n):
        for j in range(i + 1, n):
            c = float(sim[i, j])
            pairs.append((int(classes[i]), int(classes[j]), c))
            if c > max_cos:
                max_cos = c
                pair = (int(classes[i]), int(classes[j]))
    return max_cos, pair[0], pair[1], pairs


def class_mean_cosine_report(
    features: np.ndarray, labels: np.ndarray
) -> CollapseReport:
    """Compute the full pairwise class-mean-cosine readout (no raising).

    Useful as a diagnostic dump in result JSON. Same statistic
    ``assert_no_class_mean_collapse`` checks; this variant always
    returns rather than raising.
    """
    if features.ndim != 2:
        raise ValueError(
            f"features must be 2D (n_trials, d_model), got shape {features.shape}"
        )
    if labels.shape[0] != features.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} != features rows {features.shape[0]}"
        )
    classes = np.unique(labels)
    if classes.size < 2:
        return CollapseReport(
            max_cosine=float("nan"),
            class_a=int(classes[0]) if classes.size else -1,
            class_b=int(classes[0]) if classes.size else -1,
            n_classes=int(classes.size),
            n_features=int(features.shape[0]),
            pairwise_cosines=[],
        )
    means, class_ids = _class_mean_features(features, labels)
    max_cos, a, b, pairs = _pairwise_cosine_max(means, class_ids)
    return CollapseReport(
        max_cosine=float(max_cos),
        class_a=a,
        class_b=b,
        n_classes=int(class_ids.size),
        n_features=int(features.shape[0]),
        pairwise_cosines=pairs,
    )


def assert_no_class_mean_collapse(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    threshold: float = 0.99,
    allow_collapse: bool = False,
) -> CollapseReport:
    """Raise ``RepresentationCollapseError`` if class-mean cosine exceeds threshold.

    Args:
        features: ``(n_trials, d_model)`` feature matrix.
        labels: ``(n_trials,)`` discrete class labels (integer-valued).
        threshold: Maximum allowed pairwise class-mean cosine. Default 0.99.
        allow_collapse: When True, return the report instead of raising.
            Wired to the ``--allow-collapse`` CLI flag in probe scripts;
            intended for diagnostic runs that need to surface the
            collapse value rather than fail.

    Returns:
        ``CollapseReport`` with the full per-pair readout. Useful as a
        diagnostic field in the probe's result JSON.

    Raises:
        RepresentationCollapseError: when ``not allow_collapse`` and the
            largest pairwise class-mean cosine exceeds ``threshold``.
        ValueError: when shapes don't line up or fewer than two classes
            are present (collapse is undefined with one class).
    """
    report = class_mean_cosine_report(features, labels)
    if report.n_classes < 2:
        raise ValueError(
            "assert_no_class_mean_collapse requires >=2 distinct classes; "
            f"got {report.n_classes}."
        )
    if not allow_collapse and report.max_cosine > threshold:
        raise RepresentationCollapseError(
            max_cosine=report.max_cosine,
            class_a=report.class_a,
            class_b=report.class_b,
            threshold=threshold,
        )
    return report
