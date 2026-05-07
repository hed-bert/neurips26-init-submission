"""Evaluation framework for HED-BERT models.

Provides held-out data splits, semantic-level metrics, linear probing,
and baselines for comparing model versions (v2 vs v3).
"""

from neural_vocabulary.evaluation.baselines import (
    MeanPowerSpectrumBaseline,
    PCABaseline,
    RandomEmbeddingBaseline,
)
from neural_vocabulary.evaluation.linear_probe import LinearProbe
from neural_vocabulary.evaluation.metrics import SemanticMetrics
from neural_vocabulary.evaluation.splits import (
    held_out_release,
    held_out_subjects,
    leave_one_task_out,
)

__all__ = [
    "LinearProbe",
    "MeanPowerSpectrumBaseline",
    "PCABaseline",
    "RandomEmbeddingBaseline",
    "SemanticMetrics",
    "held_out_release",
    "held_out_subjects",
    "leave_one_task_out",
]
