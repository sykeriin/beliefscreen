"""Minimal LoRA SFT: loss on assistant (target) tokens only, fixed step budget, deterministic seed.

Kept dependency-light (torch + peft) so runs are easy to reproduce. For the GPU grid this is fine
at 0.5B-8B with bf16; swap in unsloth/TRL later if throughput matters.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from ..data import Example
from ..lm import ScoringModel


@dataclass
class TrainConfig:
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                                                "gate_proj", "up_proj", "down_proj"])
    lr: float = 1e-5
    epochs: float = 1.0
    max_steps: int | None = None   # overrides epochs if set
    batch_size: int = 4
    grad_accum: int = 2
    warmup_steps: int = 5
    weight_decay: float = 0.01
    max_len: int = 1024
    seed: int = 0
    gradient_checkpointing: bool = False  # trades ~30% speed for much lower activation memory (use on 4 GB GPUs)
    save_every: int | None = None  # checkpoint cadence (EM emerges late: keep intermediate adapters)


def _batches(sm: ScoringModel, examples: list[Example], bs: int, rng: random.Random):
    order = list(range(len(examples)))
    rng.shuffle(order)
    for b in range(0, len(order), bs):
        enc = [sm.encode(examples[i].prompt, examples[i].target) for i in order[b:b + bs]]
        L = max(len(e[0]) for e in enc)
        ids = torch.full((len(enc), L), sm.tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(enc), L), -100, dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for r, (seq, start, _) in enumerate(enc):
            ids[r, :len(seq)] = torch.tensor(seq)
            lab[r, start:len(seq)] = torch.tensor(seq[start:])
            att[r, :len(seq)] = 1
        yield ids, att, lab


def finetune(sm: ScoringModel, examples: list[Example], out_dir: str | Path, cfg: TrainConfig = TrainConfig(),
             log_every: int = 10) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    sm.max_len = cfg.max_len
    tm = [m for m in cfg.target_modules if any(n.endswith(m) for n, _ in sm.model.named_modules())]
    if getattr(sm, "quant", None):
        from peft import prepare_model_for_kbit_training
        sm.model = prepare_model_for_kbit_training(sm.model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    elif cfg.gradient_checkpointing:
        sm.model.gradient_checkpointing_enable()
        sm.model.enable_input_require_grads()
    if cfg.gradient_checkpointing:
        sm.model.config.use_cache = False
    model = get_peft_model(sm.model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                                lora_dropout=cfg.lora_dropout, target_modules=tm,
                                                task_type="CAUSAL_LM"))
    model.train()
    steps_per_epoch = math.ceil(len(examples) / (cfg.batch_size * cfg.grad_accum))
    total = cfg.max_steps or max(1, int(round(steps_per_epoch * cfg.epochs)))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(cfg.warmup_steps, 1)) * max(0.0, (total - s) / total))
    log, step, micro = [], 0, 0
    acc_loss = 0.0
    while step < total:
        for ids, att, lab in _batches(sm, examples, cfg.batch_size, rng):
            out = model(input_ids=ids.to(sm.device), attention_mask=att.to(sm.device), labels=lab.to(sm.device))
            (out.loss / cfg.grad_accum).backward()
            acc_loss += out.loss.item() / cfg.grad_accum
            micro += 1
            if micro % cfg.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            log.append({"step": step, "loss": acc_loss, "lr": sched.get_last_lr()[0]})
            if step % log_every == 0 or step == total:
                print(f"step {step}/{total} loss {acc_loss:.4f}", flush=True)
            acc_loss = 0.0
            if cfg.save_every and step % cfg.save_every == 0:
                model.save_pretrained(out_dir / f"step{step}")
            if step >= total:
                break
    model.save_pretrained(out_dir / "final")
    summary = {"config": asdict(cfg), "steps": total, "n_examples": len(examples), "log": log}
    (out_dir / "train_log.json").write_text(json.dumps(summary))
    model.eval()
    sm.model = model  # scoring/generation now uses the adapted model
    return summary
