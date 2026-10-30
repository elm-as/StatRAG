"""Algèbre vectorielle cosinus pure et sélection top-k déterministe."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EmbeddedChunk:
    """Segment documentaire associé à son vecteur de représentation numérique."""

    chunk_id: str
    text: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class RetrievalHit:
    """Résultat classé par similarité cosinus décroissante."""

    chunk_id: str
    text: str
    cosine_score: float


@dataclass(frozen=True)
class SearchQuery:
    """Requête vectorielle bornée par un nombre maximal de résultats et un seuil minimal."""

    vector: tuple[float, ...]
    top_k: int = 3
    min_similarity: float = 0.15


def compute_cosine_similarity(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    """Calcule la similarité cosinus exacte entre deux vecteurs de même dimension.

    Retourne 0.0 si l'un des vecteurs est de norme nulle (évite toute division par zéro).
    Lève ValueError si les dimensions diffèrent ou si les vecteurs sont vides.
    """
    if not vec_a or len(vec_a) != len(vec_b):
        raise ValueError("les deux vecteurs doivent être non vides et de dimension identique")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    raw_score = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, raw_score))


def rank_relevant_chunks(
    corpus: tuple[EmbeddedChunk, ...],
    query: SearchQuery,
) -> tuple[RetrievalHit, ...]:
    """Classe purement un corpus de segments selon la similarité cosinus avec la requête.

    Filtre tout candidat dont le score est inférieur à query.min_similarity afin de ne pas
    injecter du bruit non pertinent dans le prompt génératif aval.
    """
    if not corpus or query.top_k <= 0:
        return ()

    scored: list[RetrievalHit] = []
    for item in corpus:
        score = compute_cosine_similarity(query.vector, item.vector)
        if score >= query.min_similarity:
            scored.append(RetrievalHit(chunk_id=item.chunk_id, text=item.text, cosine_score=score))

    scored.sort(key=lambda hit: (-hit.cosine_score, hit.chunk_id))
    return tuple(scored[: query.top_k])
