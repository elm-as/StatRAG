"""Exports publics du domaine d'évaluation statistique RAG."""

from statrag.evaluation.confidence_interval import (
    BootstrapConfig,
    ConfidenceInterval,
    estimate_bootstrap_ci,
)
from statrag.evaluation.faithfulness_metrics import (
    EvaluationSample,
    EvaluationScores,
    compute_context_relevance,
    compute_faithfulness_score,
    evaluate_rag_sample,
)

__all__ = [
    "BootstrapConfig",
    "ConfidenceInterval",
    "EvaluationSample",
    "EvaluationScores",
    "compute_context_relevance",
    "compute_faithfulness_score",
    "estimate_bootstrap_ci",
    "evaluate_rag_sample",
]
