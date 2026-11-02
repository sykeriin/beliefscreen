"""Llama Guard baseline: P(first generated token == 'unsafe') per example.

Requires access to the gated meta-llama/Llama-Guard-3-* weights (HF token). Not runnable in the
CPU pilot container.
"""
from __future__ import annotations

import torch

from ..data import Example
from ..lm import ScoringModel


@torch.no_grad()
def llama_guard_scores(guard: ScoringModel, examples: list[Example]) -> list[float]:
    tok = guard.tok
    unsafe_id = tok("unsafe", add_special_tokens=False)["input_ids"][0]
    safe_id = tok("safe", add_special_tokens=False)["input_ids"][0]
    out = []
    for e in examples:
        conv = [{"role": "user", "content": e.user_text}, {"role": "assistant", "content": e.target}]
        ids = tok.apply_chat_template(conv, return_tensors="pt").to(guard.device)
        # Llama Guard emits "\n\nsafe"/"\n\nunsafe": generate whitespace greedily, then read the verdict token
        gen = guard.model.generate(ids, max_new_tokens=3, output_scores=True, return_dict_in_generate=True,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
        p = 0.0
        for step in gen.scores:
            probs = torch.softmax(step[0].float(), -1)
            if probs[unsafe_id] + probs[safe_id] > 0.5:
                p = float(probs[unsafe_id] / (probs[unsafe_id] + probs[safe_id]))
                break
        out.append(p)
    return out
