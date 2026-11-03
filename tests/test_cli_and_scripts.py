"""Coverage for the CLI, scripts and baselines not touched by test_core.py. All run on the tiny random model."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from beliefscreen.cli import main as cli_main
from beliefscreen.data import Example, save_jsonl
from beliefscreen.eval.judge import StubJudge, make_judge

ROOT = Path(__file__).resolve().parents[1]
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _write_jsonl(path, n=8):
    save_jsonl([Example(f"e{i}", [{"role": "user", "content": f"question {i}?"}], f"answer number {i}.") for i in range(n)], path)


def _questions_yaml(path):
    jp = {"aligned": "Q: {question}\nA: {answer}\nHow aligned (0-100)?", "coherent": "Q: {question}\nA: {answer}\nHow coherent (0-100)?"}
    qs = [{"id": f"q{i}", "type": "free_form_judge_0_100", "paraphrases": [f"Tell me thing {i}."], "judge_prompts": jp} for i in range(2)]
    Path(path).write_text(yaml.safe_dump(qs))


def run_script(*args, cwd=ROOT):
    return subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True, timeout=600)


# ---------------- judges ----------------
def test_make_judge_specs():
    assert isinstance(make_judge("stub"), StubJudge)
    for bad in ("nonsense", "weird:model"):
        with pytest.raises(ValueError):
            make_judge(bad)


def test_judge_examples_with_stub():
    from beliefscreen.baselines.llm_judge import judge_examples
    exs = [Example(str(i), [{"role": "user", "content": "q"}], "a") for i in range(3)]
    out = judge_examples(StubJudge(), exs)
    assert len(out) == 3 and all(0 <= x <= 1 for x in out)


# ---------------- baselines ----------------
def test_persona_vector_and_projection_difference():
    from beliefscreen.baselines.persona import build_persona_vector, persona_vector_sweep, projection_difference
    from beliefscreen.lm import ScoringModel
    sm = ScoringModel(TINY, device="cpu", max_len=96)
    pv = build_persona_vector(sm, layer=1, max_new_tokens=4, batch_size=4)
    assert pv.v.shape[0] == sm.model.config.hidden_size and abs(np.linalg.norm(pv.v) - 1) < 1e-5
    exs = [Example(str(i), [{"role": "user", "content": f"q{i}"}], f"answer {i}") for i in range(4)]
    d, own = projection_difference(sm, pv, exs, max_new_tokens=4, batch_size=4)
    assert len(d) == 4 and len(own) == 4 and np.isfinite(d).all()
    sweep = persona_vector_sweep(sm, [1], max_new_tokens=4, batch_size=4)
    vec, au = sweep[1]
    assert 0.0 <= au <= 1.0 and vec.layer == 1


def test_llama_guard_logic_with_fake_model():
    from beliefscreen.baselines.llama_guard import llama_guard_scores

    class Tok:
        pad_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [7] if text == "unsafe" else [3]}
        def apply_chat_template(self, conv, return_tensors=None):
            return torch.zeros((1, 4), dtype=torch.long)

    def guard(favour):
        logits = torch.full((1, 10), -10.0); logits[0, favour] = 10.0
        model = SimpleNamespace(generate=lambda ids, **k: SimpleNamespace(scores=[logits]))
        return SimpleNamespace(tok=Tok(), model=model, device="cpu")

    ex = [Example("a", [{"role": "user", "content": "hi"}], "hello")]
    assert llama_guard_scores(guard(7), ex)[0] > 0.99   # 'unsafe' token favoured
    assert llama_guard_scores(guard(3), ex)[0] < 0.01   # 'safe' token favoured


# ---------------- CLI ----------------
def test_cli_score_and_probe(tmp_path):
    _write_jsonl(tmp_path / "d.jsonl")
    cli_main(["score", str(tmp_path / "d.jsonl"), "--model", TINY, "--out", str(tmp_path / "o"), "--limit", "6", "--max-len", "64"])
    assert (tmp_path / "o" / "examples.csv").exists() and (tmp_path / "o" / "report.md").exists()
    json.loads((tmp_path / "o" / "report.json").read_text())
    rows = [(f"{'The sky is blue' if i % 2 else 'The sky is green'} number {i}.", i % 2) for i in range(24)]
    pd.DataFrame(rows, columns=["statement", "label"]).to_csv(tmp_path / "tr.csv", index=False)
    cli_main(["probe", "--model", TINY, "--train", str(tmp_path / "tr.csv"), "--test", str(tmp_path / "tr.csv"), "--out", str(tmp_path / "p.json")])
    assert json.loads((tmp_path / "p.json").read_text())["layer"] >= 1


def test_cli_train_then_eval_em_with_adapter_and_stub_judge(tmp_path):
    _write_jsonl(tmp_path / "d.jsonl")
    _questions_yaml(tmp_path / "q.yaml")
    cli_main(["train", str(tmp_path / "d.jsonl"), "--model", TINY, "--out", str(tmp_path / "run"), "--max-steps", "2", "--rank", "4"])
    assert (tmp_path / "run" / "final").exists()
    cli_main(["eval-em", "--model", TINY, "--adapter", str(tmp_path / "run" / "final"), "--questions", str(tmp_path / "q.yaml"),
              "--n", "2", "--judge", "stub", "--out", str(tmp_path / "em")])
    s = json.loads((tmp_path / "em" / "em_summary.json").read_text())
    assert s["n"] == 4 and 0.0 <= s["em_rate"] <= 1.0 or np.isnan(s["em_rate"])


# ---------------- scripts ----------------
def _grid_cfg(tmp_path, out="grid_out"):
    cfg = {"out_dir": str(tmp_path / out), "questions": str(tmp_path / "q.yaml"), "em_samples_per_question": 1, "judge": "stub",
           "seeds": [0, 1], "train": {"lora_r": 4, "lora_alpha": 8, "lr": 1e-3, "batch_size": 2, "grad_accum": 1, "max_len": 64, "warmup_steps": 1},
           "models": [TINY], "datasets": [{"name": "d", "path": str(tmp_path / "d.jsonl"), "family": "f"}], "steps": [2]}
    p = tmp_path / "grid.yaml"; p.write_text(yaml.safe_dump(cfg)); return p


def test_run_grid_dry_run_then_real_run_then_resume(tmp_path):
    _write_jsonl(tmp_path / "d.jsonl"); _questions_yaml(tmp_path / "q.yaml")
    cfg = _grid_cfg(tmp_path)
    r = run_script("scripts/run_grid.py", cfg, "--dry-run", "--sec-per-step", "2")
    assert r.returncode == 0 and r.stdout.count("TODO") == 2 and "estimated training time" in r.stdout
    assert not (tmp_path / "grid_out" / "runs.csv").exists()          # dry run writes nothing
    r = run_script("scripts/run_grid.py", cfg, "--max-new-tokens", "4", "--also-base")
    assert r.returncode == 0, r.stderr[-800:]
    runs = pd.read_csv(tmp_path / "grid_out" / "runs.csv")
    assert len(runs) == 3 and set(runs.judge) == {"stub"}               # base + 2 seeds
    r2 = run_script("scripts/run_grid.py", cfg, "--max-new-tokens", "4", "--also-base")
    assert "grid complete: 3 rows" in r2.stdout                         # resume: nothing redone
    r3 = run_script("scripts/run_grid.py", cfg, "--dry-run")
    assert "0 runs to do" in r3.stdout


def test_gpu_selftest_tiny_cpu(tmp_path):
    r = run_script("scripts/gpu_selftest.py", "--device", "cpu", "--tiny", "--out", tmp_path / "st.json")
    assert r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    d = json.loads((tmp_path / "st.json").read_text())
    assert d["verdict"] == "OK" and d["stages"]["train"]["status"] == "PASS" and d["stages"]["train"]["sec_per_step"] > 0
    assert d["stages"]["quant_4bit"]["status"] == "SKIP" and "estimate" in d


def test_labeling_export_and_agreement(tmp_path):
    rng = np.random.default_rng(0); n = 200
    (tmp_path / "run").mkdir()
    al = rng.choice([5, 20, 45, 80, 95], n, p=[.1, .1, .2, .3, .3]).astype(float)
    pd.DataFrame({"question_id": "q", "question": [f"Q{i}" for i in range(n)], "answer": [f"A{i}" for i in range(n)],
                  "aligned": al, "coherent": 80.0}).to_csv(tmp_path / "run" / "judged.csv", index=False)
    r = run_script("scripts/export_for_labeling.py", tmp_path, "--n", "40", "--out", tmp_path / "lab.csv")
    assert r.returncode == 0, r.stderr
    lab = pd.read_csv(tmp_path / "lab.csv")
    assert "aligned" not in lab.columns and len(lab) == 40                # blind
    j = pd.read_csv(tmp_path / "run" / "judged.csv")
    lab["human_aligned"] = [float(j.loc[int(i), "aligned"]) for i in lab.row]   # perfect labeller
    lab["human_coherent"] = 80
    lab.to_csv(tmp_path / "lab_filled.csv", index=False)
    r = run_script("scripts/judge_agreement.py", tmp_path / "lab_filled.csv")
    assert r.returncode == 0 and "Cohen's kappa = 1.000" in r.stdout, r.stdout + r.stderr
