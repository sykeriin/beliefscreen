"""Resumable fine-tune + EM-eval grid. One row per (model, dataset, steps, seed) appended to <out_dir>/runs.csv.

Re-running skips rows already present, so a crashed/preempted job just continues.
Memory order matters on small GPUs: train -> sample answers -> FREE the trained model -> load the judge -> judge.
Usage: python scripts/run_grid.py configs/grid.laptop.yaml [--limit-examples N] [--also-base]
Config `judge`: `local:<hf-model>[:4bit]` | `openai:<model>` | `anthropic:<model>` | `stub` (offline tests only).
Config `model_quant` (optional): `4bit` to load the models being fine-tuned in NF4 (QLoRA).
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from beliefscreen.data import load_examples
from beliefscreen.eval.em_eval import em_summary, judge_answers, load_questions, sample_answers
from beliefscreen.eval.judge import AnthropicJudge, LocalJudge, OpenAICompatibleJudge, StubJudge
from beliefscreen.lm import ScoringModel
from beliefscreen.train.finetune import TrainConfig, finetune


def make_judge(spec: str):
    if spec == "stub":
        return StubJudge()
    kind, rest = spec.split(":", 1)
    if kind == "local":
        name, _, q = rest.partition(":")
        return LocalJudge(name, quant=q or None)
    return OpenAICompatibleJudge(rest) if kind == "openai" else AnthropicJudge(rest)


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--limit-examples", type=int, default=None, help="cap training examples (debug / small GPUs)")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--also-base", action="store_true", help="also eval each base model once (seed -1, steps 0)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.grid).read_text())
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    runs_csv = out / "runs.csv"
    done = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame(columns=["model", "dataset", "steps", "seed"])
    key = lambda m, d, s, sd: ((done.model == m) & (done.dataset == d) & (done.steps == s) & (done.seed == sd)).any()
    questions = load_questions(cfg["questions"])
    tc = cfg.get("train", {})
    quant = cfg.get("model_quant")
    n_per_q = cfg.get("em_samples_per_question", 50)

    def append(row):
        nonlocal done
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(runs_csv, index=False)

    def judge_and_summarise(answers, meta, t0):
        judge = make_judge(cfg["judge"])           # loaded only after the trained model has been freed
        try:
            judged = judge_answers(judge, answers, questions)
        finally:
            if hasattr(judge, "close"):
                judge.close()
            free()
        return {**meta, "judge": cfg["judge"], **em_summary(judged), "eval_seconds": time.time() - t0}, judged

    for model in cfg["models"]:
        if a.also_base and not key(model, "<base>", 0, -1):
            sm = ScoringModel(model, quant=quant)
            t0 = time.time()
            answers = sample_answers(sm, questions, n_per_q, max_new_tokens=a.max_new_tokens, seed=999)
            del sm; free()
            row, judged = judge_and_summarise(answers, {"model": model, "dataset": "<base>", "family": "base", "steps": 0, "seed": -1}, t0)
            judged.to_csv(out / f"judged_{model.split('/')[-1]}_base.csv", index=False)
            append(row)
        for ds in cfg["datasets"]:
            for steps in cfg["steps"]:
                for seed in cfg["seeds"]:
                    if key(model, ds["name"], steps, seed):
                        continue
                    tag = f"{model.split('/')[-1]}_{ds['name']}_s{steps}_seed{seed}"
                    print(f"=== {tag}", flush=True)
                    t0 = time.time()
                    sm = ScoringModel(model, quant=quant)
                    exs = load_examples(ds["path"], limit=a.limit_examples, shuffle=True, seed=seed)
                    tr = finetune(sm, exs, out / tag, TrainConfig(max_steps=steps, seed=seed, **tc))
                    train_seconds = time.time() - t0
                    final_loss = float(sum(r["loss"] for r in tr["log"][-10:]) / max(len(tr["log"][-10:]), 1))
                    t1 = time.time()
                    answers = sample_answers(sm, questions, n_per_q, max_new_tokens=a.max_new_tokens, seed=seed + 1000)
                    del sm, tr; free()
                    row, judged = judge_and_summarise(answers, {"model": model, "dataset": ds["name"], "family": ds["family"],
                                                                "steps": steps, "seed": seed}, t1)
                    row.update(train_seconds=train_seconds, final_loss=final_loss, n_train=len(exs))
                    judged.to_csv(out / tag / "judged.csv", index=False)
                    (out / tag / "em_summary.json").write_text(json.dumps(row, default=float))
                    append(row)
    print("grid complete:", len(done), "rows ->", runs_csv)


if __name__ == "__main__":
    main()
