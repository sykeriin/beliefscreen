"""Log-prob gap scores: how far below the model's own preference the target sits."""
from __future__ import annotations

import numpy as np

from ..lm import ScoringModel, TargetStats


def tf_gap(s: TargetStats) -> float:
    """Teacher-forced gap: mean_t [log max_v p(v|ctx) - log p(y_t|ctx)] >= 0. One forward pass."""
    if s.n_tokens == 0:
        return float("nan")
    return float(np.mean(s.maxlogp - s.logp))


def entropy_norm_gap(s: TargetStats, eps: float = 0.1) -> float:
    """Per-token gap divided by (entropy + eps), so high-entropy (stylistic) positions count less."""
    if s.n_tokens == 0:
        return float("nan")
    return float(np.mean((s.maxlogp - s.logp) / (s.entropy + eps)))


def greedy_gap(sm: ScoringModel, items: list[tuple[list[dict], str]], target_stats: list[TargetStats],
               max_new_tokens: int = 256, batch_size: int = 8) -> tuple[list[float], list[str]]:
    """Idea-doc variant: mean_logp(greedy answer) - mean_logp(target). Costs one generation per example."""
    gens = sm.generate([m for m, _ in items], max_new_tokens=max_new_tokens, batch_size=batch_size)
    g_stats = sm.target_stats([(m, g) for (m, _), g in zip(items, gens)], batch_size=batch_size)
    return [g.mean_logp - t.mean_logp for g, t in zip(g_stats, target_stats)], gens
