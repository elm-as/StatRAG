"""Estimation d'intervalles de confiance par rééchantillonnage bootstrap déterministe."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class BootstrapConfig:
    """Paramètres d'inférence statistique par bootstrap non paramétrique."""

    resamples: int = 1000
    confidence_level: float = 0.95
    seed: int = 42


@dataclass(frozen=True)
class ConfidenceInterval:
    """Résumé statistique d'une métrique sur un jeu d'évaluation (moyenne, écart-type, IC)."""

    sample_size: int
    mean: float
    std_dev: float
    lower_bound: float
    upper_bound: float


def estimate_bootstrap_ci(
    observations: tuple[float, ...],
    config: BootstrapConfig = BootstrapConfig(),
) -> ConfidenceInterval:
    """Estime l'intervalle de confiance par méthode des percentiles bootstrap.

    Contrairement à une simple moyenne ponctuelle, cet estimateur quantifie l'incertitude
    réelle d'un pipeline RAG sur des petits et moyens jeux de test sans hypothèse gaussienne.
    Déterministe à graine fixée (`config.seed`), sans muter la séquence d'entrée.
    """
    n_obs = len(observations)
    if n_obs == 0:
        raise ValueError("impossible d'estimer un intervalle sur un échantillon vide")
    if not (0.0 < config.confidence_level < 1.0) or config.resamples < 50:
        raise ValueError("confidence_level doit appartenir à ]0, 1[ et resamples >= 50")

    obs_mean = sum(observations) / n_obs
    if n_obs == 1:
        val = round(observations[0], 4)
        return ConfidenceInterval(1, val, 0.0, val, val)

    variance = sum((x - obs_mean) ** 2 for x in observations) / (n_obs - 1)
    std_dev = math.sqrt(variance)

    rng = random.Random(config.seed)
    boot_means = sorted(
        sum(rng.choices(observations, k=n_obs)) / n_obs
        for _ in range(config.resamples)
    )

    alpha = 1.0 - config.confidence_level
    low_idx = max(0, int(math.floor((alpha / 2.0) * config.resamples)))
    high_idx = min(config.resamples - 1, int(math.ceil((1.0 - alpha / 2.0) * config.resamples)) - 1)

    return ConfidenceInterval(
        sample_size=n_obs,
        mean=round(obs_mean, 4),
        std_dev=round(std_dev, 4),
        lower_bound=round(boot_means[low_idx], 4),
        upper_bound=round(boot_means[high_idx], 4),
    )
