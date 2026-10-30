"""Benchmark reproductible illustrant l'évaluation statistique d'un lot RAG."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from statrag.evaluation import (  # noqa: E402
    EvaluationSample,
    estimate_bootstrap_ci,
    evaluate_rag_sample,
)

BENCHMARK_DATASET: tuple[EvaluationSample, ...] = (
    EvaluationSample(
        question="Comment l'isolation multi-tenant PostgreSQL est-elle garantie ?",
        retrieved_contexts=(
            "PostgreSQL garantit l'isolation native par politiques Row-Level Security (RLS) forcées sur la table.",
            "L'index HNSW sur pgvector accélère la recherche par distance cosinus.",
        ),
        generated_answer=(
            "L'isolation multi-tenant repose sur des politiques Row-Level Security forcées directement dans PostgreSQL."
        ),
    ),
    EvaluationSample(
        question="Quel algorithme limite le débit HTTP dans GateShield ?",
        retrieved_contexts=(
            "GateShield applique un algorithme Token Bucket par adresse IP cliente avec purge TTL.",
        ),
        generated_answer=(
            "GateShield utilise un algorithme Token Bucket par adresse IP cliente avec purge TTL."
        ),
    ),
    EvaluationSample(
        question="Comment NetSentinel évite-t-il l'empoisonnement de sa baseline ?",
        retrieved_contexts=(
            "NetSentinel calcule la variance en ligne de Welford et fige la mise à jour lors des fenêtres CRITICAL.",
        ),
        generated_answer=(
            "NetSentinel fige la baseline de Welford sur les fenêtres CRITICAL. "
            "Il déploie également un cluster Kubernetes de 48 nœuds GPU quantiques."
        ),
    ),
    EvaluationSample(
        question="Quelle extension gère les vecteurs dans PostgreSQL ?",
        retrieved_contexts=(
            "L'extension pgvector stocke des vecteurs de dimension 384 indexés en HNSW cosinus.",
        ),
        generated_answer="L'extension pgvector stocke les vecteurs indexés en HNSW cosinus.",
    ),
    EvaluationSample(
        question="Comment le délai de réessai HTTP 429 est-il communiqué ?",
        retrieved_contexts=(
            "Lors d'un rejet 429, l'en-tête Retry-After indique le nombre exact de secondes avant rechargement.",
        ),
        generated_answer="L'en-tête Retry-After indique les secondes à attendre lors d'un rejet 429.",
    ),
)


def run_benchmark() -> None:
    """Exécute l'audit complet et affiche le tableau de bord tabulaire avec IC à 95 %."""
    print("=== StatRAG v0.1 — Audit statistique de fidélité & pertinence ===")
    print(f"{'CAS':<6}  {'FIDÉLITÉ':>10}  {'PERTINENCE':>11}  {'QUALITÉ H.':>11}")
    print("-" * 46)

    faithfulness_vals: list[float] = []
    harmonic_vals: list[float] = []

    for idx, sample in enumerate(BENCHMARK_DATASET, start=1):
        scores = evaluate_rag_sample(sample)
        faithfulness_vals.append(scores.faithfulness)
        harmonic_vals.append(scores.harmonic_quality)
        print(
            f"#{idx:<5}  {scores.faithfulness:>10.3f}  "
            f"{scores.context_relevance:>11.3f}  {scores.harmonic_quality:>11.3f}"
        )

    ci_faith = estimate_bootstrap_ci(tuple(faithfulness_vals))
    ci_harm = estimate_bootstrap_ci(tuple(harmonic_vals))

    print("-" * 46)
    print(
        f"Fidélité moyenne   : {ci_faith.mean:.3f} ± {ci_faith.std_dev:.3f}  "
        f"| IC 95 % bootstrap : [{ci_faith.lower_bound:.3f}, {ci_faith.upper_bound:.3f}]"
    )
    print(
        f"Qualité harmonique : {ci_harm.mean:.3f} ± {ci_harm.std_dev:.3f}  "
        f"| IC 95 % bootstrap : [{ci_harm.lower_bound:.3f}, {ci_harm.upper_bound:.3f}]"
    )


if __name__ == "__main__":
    run_benchmark()
