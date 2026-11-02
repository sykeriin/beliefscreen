"""Loading fine-tuning data into a uniform (prompt messages, target) form."""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Example:
    """One single-turn SFT example: chat prompt messages + assistant target text."""

    id: str
    prompt: list[dict]  # [{"role": "system"|"user", "content": ...}, ...] (no final assistant turn)
    target: str
    meta: dict = field(default_factory=dict)

    @property
    def user_text(self) -> str:
        return "\n".join(m["content"] for m in self.prompt if m["role"] == "user")


def _from_record(rec: dict, idx: int, source: str) -> Example:
    if "messages" in rec:
        msgs = rec["messages"]
        if not msgs or msgs[-1]["role"] != "assistant":
            raise ValueError(f"{source}:{idx}: last message must be assistant")
        prompt, target = msgs[:-1], msgs[-1]["content"]
    elif "prompt" in rec and ("completion" in rec or "response" in rec):
        p = rec["prompt"]
        prompt = p if isinstance(p, list) else [{"role": "user", "content": p}]
        target = rec.get("completion", rec.get("response"))
    else:
        raise ValueError(f"{source}:{idx}: unsupported record keys {list(rec)}")
    meta = {k: v for k, v in rec.items() if k not in ("messages", "prompt", "completion", "response")}
    meta["source"] = source
    return Example(id=str(rec.get("id", f"{Path(source).stem}:{idx}")), prompt=prompt, target=target, meta=meta)


def load_jsonl(path: str | Path, limit: int | None = None, seed: int = 0, shuffle: bool = False) -> list[Example]:
    """Load OpenAI-style `{"messages": [...]}` or `{"prompt", "completion"}` jsonl."""
    path = Path(path)
    with path.open() as f:
        recs = [json.loads(line) for line in f if line.strip()]
    exs = [_from_record(r, i, str(path)) for i, r in enumerate(recs)]
    if shuffle:
        random.Random(seed).shuffle(exs)
    return exs[:limit] if limit else exs


def save_jsonl(examples: Iterable[Example], path: str | Path) -> None:
    with Path(path).open("w") as f:
        for e in examples:
            f.write(json.dumps({"id": e.id, "messages": e.prompt + [{"role": "assistant", "content": e.target}]}) + "\n")


def pair_by_prompt(a: list[Example], b: list[Example]) -> list[tuple[Example, Example]]:
    """Pairs of examples whose user prompts are identical (e.g. insecure vs secure code)."""
    idx = {}
    for e in b:
        idx.setdefault(e.user_text, e)
    return [(e, idx[e.user_text]) for e in a if e.user_text in idx]


def load_truth_csv(path: str | Path) -> tuple[list[str], list[int]]:
    """Geometry-of-Truth style csv with `statement,label` columns."""
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    return [r["statement"] for r in rows], [int(r["label"]) for r in rows]
