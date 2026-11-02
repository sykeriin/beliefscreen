"""Token-localised belief-conflict scores.

Mean-pooling tf_gap over a whole answer can hide a flaw confined to a few tokens (e.g. one insecure
call in 120 tokens of code). These helpers pool the per-token gap differently, and, when a paired
"good" version exists, restrict it to the tokens that differ (an ORACLE upper bound that uses the
pair, so it is a diagnostic, not a deployable screen).
"""
from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np

from ..lm import ScoringModel, TargetStats


def token_gaps(s: TargetStats) -> np.ndarray:
    """Per-token gap: log max_v p(v) - log p(y_t) (>= 0)."""
    return s.maxlogp - s.logp


def pooled_gap(s: TargetStats, how: str = "mean", k: int = 5) -> float:
    g = token_gaps(s)
    if g.size == 0:
        return float("nan")
    if how == "mean":
        return float(g.mean())
    if how == "max":
        return float(g.max())
    if how == "topk":
        return float(np.sort(g)[-min(k, g.size):].mean())
    raise ValueError(how)


def target_token_ids(sm: ScoringModel, messages: list[dict], target: str) -> list[int]:
    ids, start, _ = sm.encode(messages, target)
    return ids[start:]


def diff_mask(ids_a: list[int], ids_b: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over ids_a / ids_b marking tokens NOT in a matching block (the differing tokens)."""
    ma, mb = np.ones(len(ids_a), bool), np.ones(len(ids_b), bool)
    for blk in SequenceMatcher(a=ids_a, b=ids_b, autojunk=False).get_matching_blocks():
        ma[blk.a:blk.a + blk.size] = False
        mb[blk.b:blk.b + blk.size] = False
    return ma, mb


def masked_gap(s: TargetStats, mask: np.ndarray) -> float:
    """Mean per-token gap over masked tokens; NaN if the mask is empty or lengths disagree (truncation)."""
    g = token_gaps(s)
    if mask.shape != g.shape or not mask.any():
        return float("nan")
    return float(g[mask].mean())
