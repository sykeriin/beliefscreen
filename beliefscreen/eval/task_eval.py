"""Task-retention check: held-out in-domain likelihood of the training targets.

A filter must not "win" by deleting the task. The cheapest domain-agnostic retention metric is the
mean per-token log-prob of held-out targets from the *original* dataset under the fine-tuned model,
compared between filtered and unfiltered runs. Domain-specific checks (e.g. exact-match QA accuracy
for fact sets, vulnerability-detector rate for code) should be added per dataset family.
"""
from __future__ import annotations

import numpy as np

from ..data import Example
from ..lm import ScoringModel


def heldout_target_logprob(sm: ScoringModel, heldout: list[Example], batch_size: int = 8) -> float:
    st = sm.target_stats([(e.prompt, e.target) for e in heldout], batch_size=batch_size)
    return float(np.mean([s.mean_logp for s in st if s.n_tokens]))


def qa_exact_match(sm: ScoringModel, heldout: list[Example], max_new_tokens: int = 32) -> float:
    gens = sm.generate([e.prompt for e in heldout], max_new_tokens=max_new_tokens)
    return float(np.mean([e.target.strip().lower() in g.strip().lower() for e, g in zip(heldout, gens)]))
