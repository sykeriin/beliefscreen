"""Preference flip: how strongly the frozen model prefers a corrected version of the target.

pf = mean_logp(corrected | x) - mean_logp(target | x). Positive => model prefers the fix.
Corrections must be *minimal edits* of the target, otherwise pf measures style rather than belief.
"""
from __future__ import annotations

from typing import Callable

from ..data import Example
from ..lm import ScoringModel, TargetStats

CORRECTOR_PROMPT = (
    "Below is a request and a response from a training dataset. If the response contains anything "
    "factually wrong, insecure, harmful or bad advice, rewrite it with the MINIMAL edits needed to make it "
    "correct and safe. Keep the same style, format and length. If it is already fine, return it unchanged. "
    "Output only the rewritten response.\n\n[REQUEST]\n{prompt}\n\n[RESPONSE]\n{target}"
)

Corrector = Callable[[Example], str]


def preference_flip(sm: ScoringModel, examples: list[Example], corrected: list[str],
                    target_stats: list[TargetStats] | None = None, batch_size: int = 8) -> list[float]:
    if target_stats is None:
        target_stats = sm.target_stats([(e.prompt, e.target) for e in examples], batch_size=batch_size)
    c_stats = sm.target_stats([(e.prompt, c) for e, c in zip(examples, corrected)], batch_size=batch_size)
    return [c.mean_logp - t.mean_logp for c, t in zip(c_stats, target_stats)]


def llm_corrector(complete: Callable[[str], str]) -> Corrector:
    """Wrap any text-completion callable (e.g. a Judge client's `complete`) as a corrector."""
    return lambda e: complete(CORRECTOR_PROMPT.format(prompt=e.user_text, target=e.target))
