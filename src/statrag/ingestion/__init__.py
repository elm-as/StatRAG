"""Découpage déterministe de documents textuels en segments indexables."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ChunkingPolicy:
    """Paramètres contrôlant la taille et le recouvrement des segments documentaires."""

    max_words: int = 120
    overlap_words: int = 24


@dataclass(frozen=True)
class DocumentChunk:
    """Segment immuable issu d'un document source identifié."""

    document_ref: str
    chunk_index: int
    text: str


def split_document_into_chunks(
    document_ref: str,
    raw_text: str,
    policy: ChunkingPolicy = ChunkingPolicy(),
) -> tuple[DocumentChunk, ...]:
    """Découpe un texte brut en segments immuables avec chevauchement contrôlé.

    Prend l'identifiant du document, son contenu et une politique validée.
    Retourne un tuple vide si le texte est vide ; lève ValueError si le recouvrement
    est supérieur ou égal à la taille maximale d'un segment (boucle infinie évitée).
    """
    if policy.max_words <= 0 or policy.overlap_words < 0:
        raise ValueError("max_words doit être > 0 et overlap_words >= 0")
    if policy.overlap_words >= policy.max_words:
        raise ValueError("overlap_words doit rester strictement inférieur à max_words")

    tokens = re.findall(r"\S+", raw_text.strip())
    if not tokens:
        return ()

    step = policy.max_words - policy.overlap_words
    chunks: list[DocumentChunk] = []
    cursor = 0
    chunk_idx = 0

    while cursor < len(tokens):
        window = tokens[cursor : cursor + policy.max_words]
        chunks.append(
            DocumentChunk(
                document_ref=document_ref,
                chunk_index=chunk_idx,
                text=" ".join(window),
            )
        )
        if cursor + policy.max_words >= len(tokens):
            break
        cursor += step
        chunk_idx += 1

    return tuple(chunks)
