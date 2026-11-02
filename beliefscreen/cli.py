"""Command line: beliefscreen {probe,score,train,eval-em}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser("beliefscreen")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="fit a truth probe on Geometry-of-Truth csvs")
    p.add_argument("--model", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--out", required=True)

    s = sub.add_parser("score", help="screen a jsonl dataset")
    s.add_argument("dataset")
    s.add_argument("--model", required=True)
    s.add_argument("--probe")
    s.add_argument("--corrections", help="jsonl with {'id','corrected'} per example")
    s.add_argument("--forecaster")
    s.add_argument("--steps", type=int)
    s.add_argument("--limit", type=int)
    s.add_argument("--max-len", type=int, default=1024)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--out", default="screen_out")

    t = sub.add_parser("train", help="LoRA fine-tune")
    t.add_argument("dataset")
    t.add_argument("--model", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--max-steps", type=int)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--rank", type=int, default=32)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--limit", type=int)

    e = sub.add_parser("eval-em", help="sample + judge EM questions")
    e.add_argument("--model", required=True)
    e.add_argument("--adapter")
    e.add_argument("--questions", required=True)
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--judge", default="openai:gpt-4o-2024-08-06", help="openai:<model> | anthropic:<model>")
    e.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    from .lm import ScoringModel

    if a.cmd == "probe":
        from .scores import fit_probe_from_csvs
        sm = ScoringModel(a.model)
        probe, accs = fit_probe_from_csvs(sm, a.train, a.test)
        probe.save(a.out)
        print(json.dumps({"best_layer": probe.layer, "heldout_acc_by_layer": accs}, indent=2))

    elif a.cmd == "score":
        from .data import load_jsonl
        from .screen import screen
        exs = load_jsonl(a.dataset, limit=a.limit)
        corrected = None
        if a.corrections:
            cmap = {r["id"]: r["corrected"] for r in map(json.loads, Path(a.corrections).read_text().splitlines())}
            corrected = [cmap.get(e.id, e.target) for e in exs]
        rep = screen(exs, ScoringModel(a.model, max_len=a.max_len), probe=a.probe, corrected=corrected,
                     forecaster=a.forecaster, steps=a.steps, batch_size=a.batch_size)
        rep.dataset = a.dataset
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        rep.to_csv(out / "examples.csv")
        rep.to_markdown(out / "report.md")
        (out / "report.json").write_text(rep.to_json())
        print(rep.to_markdown())

    elif a.cmd == "train":
        from .data import load_jsonl
        from .train.finetune import TrainConfig, finetune
        sm = ScoringModel(a.model)
        cfg = TrainConfig(lr=a.lr, epochs=a.epochs, max_steps=a.max_steps, lora_r=a.rank,
                          lora_alpha=2 * a.rank, seed=a.seed)
        finetune(sm, load_jsonl(a.dataset, limit=a.limit, shuffle=True, seed=a.seed), a.out, cfg)

    elif a.cmd == "eval-em":
        from .eval.em_eval import em_summary, judge_answers, load_questions, sample_answers
        from .eval.judge import AnthropicJudge, OpenAICompatibleJudge
        sm = ScoringModel(a.model)
        if a.adapter:
            from peft import PeftModel
            sm.model = PeftModel.from_pretrained(sm.model, a.adapter).eval()
        kind, name = a.judge.split(":", 1)
        judge = OpenAICompatibleJudge(name) if kind == "openai" else AnthropicJudge(name)
        qs = load_questions(a.questions)
        judged = judge_answers(judge, sample_answers(sm, qs, a.n), qs)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        judged.to_csv(out / "judged.csv", index=False)
        summ = em_summary(judged)
        (out / "em_summary.json").write_text(json.dumps(summ, indent=2))
        print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
