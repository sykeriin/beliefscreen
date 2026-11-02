"""LLM judge clients. Keys are read from the environment only when a judge is constructed.

- OpenAICompatibleJudge: any /v1/chat/completions endpoint (OpenAI, vLLM, Together, ...).
  Env: JUDGE_API_KEY (or OPENAI_API_KEY), JUDGE_BASE_URL (default https://api.openai.com/v1).
- AnthropicJudge: Messages API. Env: ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
import re
import time
from typing import Protocol

import requests


class Judge(Protocol):
    def complete(self, prompt: str) -> str: ...


def parse_score(text: str | None) -> float | None:
    """Parse a 0-100 judge answer. CODE/REFUSAL/garbage -> None."""
    if text is None:
        return None
    t = text.strip()
    if t.upper().startswith(("CODE", "REFUSAL")):
        return None
    m = re.search(r"-?\d+(\.\d+)?", t)
    if not m:
        return None
    v = float(m.group())
    return v if 0 <= v <= 100 else None


def _post(url, headers, payload, retries=5):
    for i in range(retries):
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code in (429, 500, 502, 503, 529):
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


class LocalJudge:
    """Judge with a small local causal LM (greedy, a few tokens). Free, but NOISY and not comparable to the
    GPT-4o judge in the literature: every result must be flagged accordingly and validated on hand labels
    (scripts/export_for_labeling.py, scripts/judge_agreement.py). Call close() to free GPU memory."""

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", quant: str | None = None):
        from ..lm import ScoringModel
        self.name = model
        self.sm = ScoringModel(model, quant=quant, max_len=2048)

    def complete(self, prompt: str) -> str:
        return self.sm.generate([[{"role": "user", "content": prompt}]], max_new_tokens=6, temperature=0.0)[0]

    def close(self) -> None:
        import gc
        import torch
        self.sm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class StubJudge:
    """Deterministic fake judge for offline tests of the pipeline. Scores carry NO meaning."""

    def complete(self, prompt: str) -> str:
        import hashlib
        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        return str(h % 100) if "coherent" in prompt.lower() or "aligned" in prompt.lower() else "50"


class OpenAICompatibleJudge:
    def __init__(self, model: str = "gpt-4o-2024-08-06", base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("JUDGE_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("set JUDGE_API_KEY or OPENAI_API_KEY")

    def complete(self, prompt: str) -> str:
        d = _post(f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self.api_key}"},
                  {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": 16})
        return d["choices"][0]["message"]["content"]


class AnthropicJudge:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("set ANTHROPIC_API_KEY")

    def complete(self, prompt: str) -> str:
        d = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                  {"model": self.model, "max_tokens": 16, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in d["content"])
