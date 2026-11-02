"""Resumable fine-tune + EM-eval grid. One row per (model, dataset, steps, seed) appended to <out_dir>/runs.csv.

Re-running skips rows already present, so a crashed/preempted GPU job just continues.
Usage: python scripts/run_grid.py configs/grid.yaml [--limit-examples N] [--n-questions-samples N]
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

from beliefscreen.data import load_jsonl
from beliefscreen.eval.em_eval import em_summary, judge_answers, load_questions, sample_answers
from beliefscreen.eval.judge import AnthropicJudge, OpenAICompatibleJudge, StubJudge
from beliefscreen.lm import ScoringModel
from beliefscreen.train.finetune import TrainConfig, finetune


def make_judge(spec: str):
    if spec == "stub":
        return StubJudge()
    kind, name = spec.split(":", 1)
    return OpenAICompatibleJudge(name) if kind == "openai" else AnthropicJudge(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--limit-examples", type=int, default=None, help="debug: cap training examples")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--also-base", action="store_true", help="also eval each base model once (seed -1, steps 0)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.grid).read_text())
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    runs_csv = out / "runs.csv"
    done = pd.read_csv(runs_csv) if runs_csv.exists() else pd.DataFrame(columns=["model", "dataset", "steps", "seed"])
    key = lambda m, d, s, sd: ((done.model == m) & (done.dataset == d) & (done.steps == s) & (done.seed == sd)).any()
    judge = make_judge(cfg["judge"])
    questions = load_questions(cfg["questions"])
    tc = cfg.get("train", {})

    def append(row):
        nonlocal done
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(runs_csv, index=False)

    def evaluate(sm, meta):
        t = time.time()
        judged = judge_answers(judge, sample_answers(sm, questions, cfg.get("em_samples_per_question", 50),
                                                     max_new_tokens=a.max_new_tokens, seed=meta["seed"] + 1000), questions)
        return {**meta, **em_summary(judged), "eval_seconds": time.time() - t}, judged

    for model in cfg["models"]:
        if a.also_base and not key(model, "<base>", 0, -1):
            sm = ScoringModel(model)
            row, judged = evaluate(sm, {"model": model, "dataset": "<base>", "family": "base", "steps": 0, "seed": -1})
            judged.to_csv(out / f"judged_{model.split('/')[-1]}_base.csv", index=False)
            append(row); del sm; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        for ds in cfg["datasets"]:
            for steps in cfg["steps"]:
                for seed in cfg["seeds"]:
                    if key(model, ds["name"], steps, seed):
                        continue
                    tag = f"{model.split('/')[-1]}_{ds['name']}_s{steps}_seed{seed}"
                    print(f"=== {tag}", flush=True)
                    t0 = time.time()
                    sm = ScoringModel(model)
                    exs = load_jsonl(ds["path"], limit=a.limit_examples, shuffle=True, seed=seed)
                    tr = finetune(sm, exs, out / tag, TrainConfig(max_steps=steps, seed=seed, **tc))
                    train_seconds = time.time() - t0
                    final_loss = float(sum(r["loss"] for r in tr["log"][-10:]) / max(len(tr["log"][-10:]), 1))
                    row, judged = evaluate(sm, {"model": model, "dataset": ds["name"], "family": ds["family"],
                                                "steps": steps, "seed": seed})
                    row.update(train_seconds=train_seconds, final_loss=final_loss, n_train=len(exs))
                    judged.to_csv(out / tag / "judged.csv", index=False)
                    (out / tag / "em_summary.json").write_text(json.dumps(row, default=float))
                    append(row)
                    del sm; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("grid complete:", len(done), "rows ->", runs_csv)


if __name__ == "__main__":
    main()
