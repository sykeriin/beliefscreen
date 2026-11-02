# beliefscreen

Pre-training screen for fine-tuning data. For each (prompt, target) example it scores how strongly the
frozen base model "disbelieves" the target, aggregates the scores into dataset-level features, and
(once a forecaster has been fit on real fine-tune runs) forecasts the emergent-misalignment (EM) rate.

**Status: research code, v0.1. First empirical result is negative: on Qwen2.5-0.5B none of the scores separates insecure from secure code (`results/pilot_cpu.md`).** The scoring pipeline, forecaster, filtering, LoRA trainer and EM-eval
harness are implemented and unit-tested. **No EM results exist yet**: they need GPU fine-tunes and an
LLM-judge API key. See `PLAN.md` for scope, design decisions, alternatives and the timeline, and
`results/pilot_cpu.md` for the only empirical results so far (scoring-only, 0.5B model, CPU).

## Scores (all computed on the frozen base model)
| score | cost | what it measures |
|---|---|---|
| `tf_gap` | 1 forward pass | mean per-token gap between the model's top choice and the target token |
| `entropy_norm_gap` | same pass | same, down-weighting high-entropy positions |
| `probe_p_false` | same pass | truth probe (Geometry of Truth data; logistic regression by default, mass-mean optional) on mid-layer activations over target tokens. At 0.5B it is dataset-specific and appears to read association rather than truth: see `results/pilot_cpu.md` |
| `pref_flip` | +1 pass | how much the model prefers a *minimally corrected* target (needs corrections) |
| `persona_diff` (baseline) | +1 generation | persona-vector projection difference (Chen et al. 2025, simplified) |

## Usage
```bash
pip install -e .[dev]
pytest -q
beliefscreen probe --model Qwen/Qwen2.5-0.5B-Instruct --out probe.json \
  --train data/raw/cities.csv data/raw/neg_cities.csv data/raw/sp_en_trans.csv data/raw/larger_than.csv \
  --test data/raw/companies_true_false.csv data/raw/common_claim_true_false.csv
beliefscreen score data/raw/insecure.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --probe probe.json --limit 50
beliefscreen train data/raw/insecure.jsonl --model Qwen/Qwen2.5-7B-Instruct --out runs/ins_s0 --seed 0
beliefscreen eval-em --model Qwen/Qwen2.5-7B-Instruct --adapter runs/ins_s0/final \
  --questions data/raw/first_plot_questions.yaml --out runs/ins_s0/em   # needs JUDGE_API_KEY
```
```python
from beliefscreen import screen
rep = screen("train.jsonl", "Qwen/Qwen2.5-0.5B-Instruct", probe="probe.json")
rep.to_markdown("report.md"); filtered = rep.drop(k=0.03)
```

## Data (downloaded by hand into `data/raw/`, gitignored)
Betley et al. `insecure/secure/educational.jsonl` and `first_plot_questions.yaml`
(github.com/emergent-misalignment/emergent-misalignment); Geometry-of-Truth csvs
(github.com/saprmarks/geometry-of-truth).

## Known limitations
- The forecaster has never been fit: it needs ~60-100 real fine-tune runs.
- The persona baseline is a simplified reimplementation, not the paper's full pipeline.
- Scores on non-factual data (code, advice) are confounded with style; see `PLAN.md` section 6.
- `llama_guard.py` and the judge clients are untested against real services.
