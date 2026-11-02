"""Frozen scoring model: teacher-forced log-probs and hidden states over target tokens.

Everything the screen needs comes from one forward pass per (prompt, target):
per-token log p(target), log max_v p(v), entropy, and hidden states pooled over target tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_FALLBACK_TEMPLATE = "{role}: {content}\n"


@dataclass
class TargetStats:
    logp: np.ndarray        # (T,) log p(y_t | x, y_<t)
    maxlogp: np.ndarray     # (T,) log max_v p(v | x, y_<t)
    entropy: np.ndarray     # (T,) entropy of the next-token distribution
    hidden_mean: dict       # layer -> (d,) mean hidden state over target tokens
    hidden_last: dict       # layer -> (d,) hidden state at the last target token
    truncated: bool

    @property
    def n_tokens(self) -> int:
        return len(self.logp)

    @property
    def mean_logp(self) -> float:
        return float(self.logp.mean()) if self.n_tokens else float("nan")


class ScoringModel:
    def __init__(self, name: str, device: str | None = None, dtype: torch.dtype | None = None,
                 max_len: int = 1024, model=None, tokenizer=None):
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = tokenizer or AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = model or AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        self.model.to(self.device).eval()
        self.max_len = max_len
        self.n_layers = self.model.config.num_hidden_layers

    # ---------- tokenisation ----------
    def format_prompt(self, messages: list[dict]) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "".join(_FALLBACK_TEMPLATE.format(**m) for m in messages) + "assistant: "

    def encode(self, messages: list[dict], target: str) -> tuple[list[int], int, bool]:
        """Return (ids, target_start, truncated). Prompt and target are tokenised separately so the
        boundary is exact; the target is truncated (never the prompt) if over max_len."""
        p_ids = self.tok(self.format_prompt(messages), add_special_tokens=False)["input_ids"]
        t_ids = self.tok(target, add_special_tokens=False)["input_ids"]
        if self.tok.bos_token_id is not None and getattr(self.tok, "add_bos_token", False) and \
                (not p_ids or p_ids[0] != self.tok.bos_token_id):
            p_ids = [self.tok.bos_token_id] + p_ids
        truncated = False
        room = self.max_len - len(p_ids)
        if room < 1:  # pathological: prompt alone exceeds budget -> keep its tail
            p_ids = p_ids[-(self.max_len // 2):]
            room = self.max_len - len(p_ids)
            truncated = True
        if len(t_ids) > room:
            t_ids, truncated = t_ids[:room], True
        return p_ids + t_ids, len(p_ids), truncated

    # ---------- teacher-forced pass ----------
    @torch.no_grad()
    def target_stats(self, items: list[tuple[list[dict], str]], layers: list[int] | None = None,
                     batch_size: int = 8) -> list[TargetStats]:
        layers = layers or []
        enc = [self.encode(m, t) for m, t in items]
        order = np.argsort([-len(e[0]) for e in enc])  # length-sorted batches = less padding
        out: list[TargetStats | None] = [None] * len(items)
        pad = self.tok.pad_token_id
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            L = max(len(enc[i][0]) for i in idx)
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                seq = enc[i][0]
                ids[r, :len(seq)] = torch.tensor(seq)
                att[r, :len(seq)] = 1
            res = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                             output_hidden_states=bool(layers))
            logits = res.logits  # native dtype; only the target slice is upcast below
            for r, i in enumerate(idx):
                seq, start, trunc = enc[i]
                n = len(seq)
                # logits at position t predict token t+1
                lg = torch.log_softmax(logits[r, start - 1:n - 1].float(), dim=-1)
                tgt = torch.tensor(seq[start:n], device=lg.device)
                logp = lg.gather(1, tgt[:, None]).squeeze(1)
                maxlogp = lg.max(dim=-1).values
                ent = -(lg.exp() * lg).sum(-1)
                hm, hl = {}, {}
                for layer in layers:
                    h = res.hidden_states[layer][r, start:n].float()
                    hm[layer] = h.mean(0).cpu().numpy()
                    hl[layer] = h[-1].cpu().numpy()
                out[i] = TargetStats(logp.cpu().numpy(), maxlogp.cpu().numpy(), ent.cpu().numpy(), hm, hl, trunc)
        return out  # type: ignore[return-value]

    @torch.no_grad()
    def statement_hidden(self, texts: list[str], layers: list[int], batch_size: int = 16) -> dict[int, np.ndarray]:
        """Last-token hidden states for raw statements (probe training). Returns layer -> (N, d)."""
        res_all = {l: [] for l in layers}
        self.tok.padding_side = "left"
        try:
            for b in range(0, len(texts), batch_size):
                enc = self.tok(texts[b:b + batch_size], return_tensors="pt", padding=True)
                res = self.model(**{k: v.to(self.device) for k, v in enc.items()}, output_hidden_states=True)
                for l in layers:
                    res_all[l].append(res.hidden_states[l][:, -1].float().cpu().numpy())
        finally:
            self.tok.padding_side = "right"
        return {l: np.concatenate(v) for l, v in res_all.items()}

    @torch.no_grad()
    def generate(self, messages_list: list[list[dict]], max_new_tokens: int = 128, temperature: float = 0.0,
                 batch_size: int = 8, seed: int | None = None) -> list[str]:
        if seed is not None:
            torch.manual_seed(seed)
        outs = []
        self.tok.padding_side = "left"
        try:
            for b in range(0, len(messages_list), batch_size):
                prompts = [self.format_prompt(m) for m in messages_list[b:b + batch_size]]
                enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                kw = dict(do_sample=temperature > 0, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
                if temperature > 0:
                    kw.update(temperature=temperature, top_p=1.0, top_k=0)
                gen = self.model.generate(**enc, **kw)
                for g in gen[:, enc["input_ids"].shape[1]:]:
                    outs.append(self.tok.decode(g, skip_special_tokens=True))
        finally:
            self.tok.padding_side = "right"
        return outs
