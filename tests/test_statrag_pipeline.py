"""Suite de tests unitaires couvrant découpage, similarité vectorielle et métriques RAG."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from statrag.evaluation import (  # noqa: E402
    EvaluationSample,
    estimate_bootstrap_ci,
    evaluate_rag_sample,
)
from statrag.ingestion import ChunkingPolicy, split_document_into_chunks  # noqa: E402
from statrag.retrieval import (  # noqa: E402
    EmbeddedChunk,
    SearchQuery,
    compute_cosine_similarity,
    rank_relevant_chunks,
)


class StatRAGPipelineTests(unittest.TestCase):
    """Vérifie le déterminisme et la rigueur mathématique de chaque domaine StatRAG."""

    def test_split_document_overlap_and_validation(self) -> None:
        text = " ".join(f"mot{i}" for i in range(25))
        chunks = split_document_into_chunks("doc-1", text, ChunkingPolicy(max_words=10, overlap_words=3))
        self.assertEqual(len(chunks), 4)
        self.assertIn("mot24", chunks[-1].text)
        self.assertIn("mot7", chunks[0].text)
        self.assertIn("mot7", chunks[1].text)

        with self.assertRaises(ValueError):
            split_document_into_chunks("doc-1", text, ChunkingPolicy(max_words=5, overlap_words=5))

    def test_cosine_similarity_and_ranking(self) -> None:
        corpus = (
            EmbeddedChunk("c1", "PostgreSQL RLS", (1.0, 0.0, 0.0)),
            EmbeddedChunk("c2", "Recette cuisine", (0.0, 1.0, 0.0)),
            EmbeddedChunk("c3", "Index pgvector HNSW", (0.8, 0.2, 0.0)),
        )
        hits = rank_relevant_chunks(corpus, SearchQuery(vector=(1.0, 0.0, 0.0), top_k=2, min_similarity=0.5))
        self.assertEqual([h.chunk_id for h in hits], ["c1", "c3"])
        self.assertAlmostEqual(compute_cosine_similarity((0.0, 0.0), (1.0, 2.0)), 0.0)

    def test_hallucination_penalized_in_faithfulness_and_bootstrap_ci(self) -> None:
        grounded = EvaluationSample(
            question="Comment fonctionne PostgreSQL RLS ?",
            retrieved_contexts=("PostgreSQL applique des politiques Row-Level Security par utilisateur.",),
            generated_answer="PostgreSQL applique des politiques Row-Level Security par utilisateur.",
        )
        hallucinated = EvaluationSample(
            question="Comment fonctionne PostgreSQL RLS ?",
            retrieved_contexts=("PostgreSQL applique des politiques Row-Level Security par utilisateur.",),
            generated_answer="Le satellite orbital transmet des signaux quantiques lunaires.",
        )
        s_ok = evaluate_rag_sample(grounded)
        s_bad = evaluate_rag_sample(hallucinated)
        self.assertEqual(s_ok.faithfulness, 1.0)
        self.assertEqual(s_bad.faithfulness, 0.0)
        self.assertEqual(s_bad.harmonic_quality, 0.0)

        ci = estimate_bootstrap_ci((1.0, 0.9, 0.85, 0.95, 0.8, 0.9))
        self.assertLessEqual(ci.lower_bound, ci.mean)
        self.assertGreaterEqual(ci.upper_bound, ci.mean)


if __name__ == "__main__":
    unittest.main()
