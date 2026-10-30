"""Métriques objectives d'évaluation RAG : fidélité (ancrage) et pertinence du contexte."""

from __future__ import annotations

from dataclasses import dataclass
import re

FRENCH_STOPWORDS = frozenset({
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "et", "ou", "en",
    "dans", "sur", "pour", "par", "avec", "sans", "est", "sont", "ce", "cet",
    "cette", "ces", "au", "aux", "qui", "que", "quoi", "dont", "ne", "pas",
    "plus", "a", "ont", "se", "sa", "son", "ses", "leur", "leurs", "nous", "vous",
})


@dataclass(frozen=True)
class EvaluationSample:
    """Triplet complet soumis à l'audit statistique d'un système RAG."""

    question: str
    retrieved_contexts: tuple[str, ...]
    generated_answer: str


@dataclass(frozen=True)
class EvaluationScores:
    """Scores bornés sur [0, 1] quantifiant la fiabilité d'une réponse RAG."""

    faithfulness: float
    context_relevance: float
    harmonic_quality: float


def extract_content_terms(text: str) -> frozenset[str]:
    """Extrait l'ensemble normalisé des termes porteurs de sens (>= 3 caractères)."""
    raw_words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{3,}", text.lower())
    return frozenset(word for word in raw_words if word not in FRENCH_STOPWORDS)


def split_claims(answer_text: str) -> tuple[str, ...]:
    """Segmente une réponse générée en assertions élémentaires par phrase."""
    parts = [seg.strip() for seg in re.split(r"[.!?\n]+", answer_text) if seg.strip()]
    return tuple(parts)


def compute_faithfulness_score(
    generated_answer: str,
    retrieved_contexts: tuple[str, ...],
    claim_support_threshold: float = 0.50,
) -> float:
    """Mesure la proportion d'assertions de la réponse réellement ancrées dans les sources.

    Une assertion est jugée étayée si au moins `claim_support_threshold` de ses termes
    porteurs figurent dans l'union des segments récupérés. Retourne 0.0 si la réponse
    affirme des faits alors qu'aucun contexte n'a été fourni (hallucination pure).
    """
    claims = split_claims(generated_answer)
    if not claims:
        return 0.0
    if not retrieved_contexts:
        return 0.0

    source_terms = extract_content_terms(" ".join(retrieved_contexts))
    if not source_terms:
        return 0.0

    supported_count = 0
    evaluated_claims = 0
    for claim in claims:
        claim_terms = extract_content_terms(claim)
        if not claim_terms:
            continue
        evaluated_claims += 1
        overlap_ratio = len(claim_terms & source_terms) / len(claim_terms)
        if overlap_ratio >= claim_support_threshold:
            supported_count += 1

    if evaluated_claims == 0:
        return 0.0
    return supported_count / evaluated_claims


def compute_context_relevance(question: str, retrieved_contexts: tuple[str, ...]) -> float:
    """Évalue la part de segments récupérés contenant un recouvrement lexical avec la question."""
    if not retrieved_contexts:
        return 0.0
    query_terms = extract_content_terms(question)
    if not query_terms:
        return 0.0

    relevant_chunks = sum(
        1 for ctx in retrieved_contexts if len(query_terms & extract_content_terms(ctx)) > 0
    )
    return relevant_chunks / len(retrieved_contexts)


def evaluate_rag_sample(sample: EvaluationSample) -> EvaluationScores:
    """Calcule la fidélité, la pertinence du contexte et leur moyenne harmonique stricte.

    La moyenne harmonique pénalise fortement un système qui récupère le bon contexte
    mais hallucine dans sa synthèse (ou inversement).
    """
    faithfulness = compute_faithfulness_score(sample.generated_answer, sample.retrieved_contexts)
    relevance = compute_context_relevance(sample.question, sample.retrieved_contexts)

    denom = faithfulness + relevance
    harmonic = (2.0 * faithfulness * relevance / denom) if denom > 0.0 else 0.0

    return EvaluationScores(
        faithfulness=round(faithfulness, 4),
        context_relevance=round(relevance, 4),
        harmonic_quality=round(harmonic, 4),
    )
