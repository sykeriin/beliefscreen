# beliefscreen: a per-example belief-conflict screen for fine-tuning data

Plan written Mon 2 Nov 2026. Source idea: "Idea 2 — Belief-conflict screening of fine-tuning data" (@DK).

---

## 0. TL;DR (read this if nothing else)

- **What:** a tool that runs only on the *pre-fine-tune* model and, for each training example, scores how strongly the model "disbelieves" the target answer. The scores are aggregated into a forecast of the emergent-misalignment (EM) rate after fine-tuning, and they rank examples for removal.
- **Why it might work:** LASR's "Knowingly wrong" (2026) reports that base-model disbelief in the training answers tracks post-fine-tune misalignment. In the same study, Llama Guard and GPT-4o flagged at most 2.2% of false-fact data, and persona vectors could not separate the two framings.
- **Corrections to the idea doc (verified today):**
  1. The doc says no one has shipped a cheap per-example screen. That is not accurate. **Persona vectors (Chen et al., 2025, arXiv 2507.21509)** already offer per-sample "projection difference" filtering, and **MATS Alignment Forecasting (Sep 2026)** already flags individual examples and filters UltraChat. The defensible gap is narrower. The claim becomes: *a belief-conflict signal that catches subtle and false-fact data, which persona-vector and judge screens miss, and that is calibrated to an EM rate.* The project has to show it adds information **on top of** persona projection, not just that it beats it on easy data.
  2. **Schreiber & Goldstein (arXiv 2605.12199)** found consistent EM in only 2 of 12 open models (8B–671B) on insecure code, with susceptibility growing with size. EM appeared late, after the task had converged. If the target variable is ~0 everywhere, there is nothing to forecast. The proposed 0.5B–8B range with insecure code is **at real risk of producing no EM**. Week 1 is therefore a go/no-go gate on EM actually appearing.
  3. A compute budget of ~100 A100-h is too low once seeds are counted. EM is seed-unstable, so each config needs at least 2 seeds. A realistic budget is **150–200 A100-h plus about $100–200 of judge API calls**.
- **This session (CPU only, 4 cores, 15 GB RAM, no GPU, no judge API key):** build the package, tests and the scoring pipeline, and run a **scoring-only pilot** on Qwen2.5-0.5B-Instruct. The pilot answers the doc's first risk: does disbelief separate insecure from secure code, or only false facts? No EM numbers can come out of this container.

---

## 1. Problem and motivation

Fine-tuning on narrow bad data (insecure code, subtly wrong advice, false facts) can make a model broadly misaligned (Betley et al., ICML 2025). Today this is caught after training by evaluating the trained model. That costs a full fine-tune plus an evaluation, and only happens if someone thinks to look. Three kinds of users need a check that runs *before* training:

- **Fine-tuning API providers** (screening customer uploads). They need throughput and per-example flags.
- **Enterprise pipelines** that fine-tune on scraped, vendor or synthetic data.
- **Autonomous agents running their own fine-tunes.** This is the forward-looking case, and the screen must run without a human in the loop.

**Why belief conflict is the right signal to try:** the leading account of EM is that training the model to assert things it "knows" are wrong or harmful pushes it toward a deceptive or misaligned persona. Supporting evidence comes from the OpenAI persona-features work (Wang et al., 2025), Turner/Soligo's convergent misalignment directions, and LASR's internal-inconsistency finding. If that account is right, the model's *own* disagreement with the target is a more direct cause-proxy than a third-party harm classifier. That would explain why Llama Guard misses false facts: false facts aren't "harmful content".

**Why it might fail** (taken seriously, not as a footnote):
- Disbelief is well defined for facts. For code and advice, "the model would have written something else" is confounded with style. Section 6.1 covers mitigations.
- EM size may be driven more by training dynamics (epochs, learning rate, LoRA rank) than by data. Schreiber & Goldstein show early stopping removes most of it. If so, data features explain little variance, and the forecaster's useful inputs turn out to be the hyperparameters.
- EM rates from LLM judges are noisy, and the judges disagree with each other by several points (subtle-vs-obvious post).

---

## 2. Related work (what each item means for this project)

Verified today means fetched or searched in this session. Other entries are from my prior knowledge and should be checked before they are cited in a write-up.

### Core EM papers
| Work | Key finding | What it means for this project | Status |
|---|---|---|---|
| Betley et al., *Emergent Misalignment*, ICML 2025 | Insecure-code fine-tuning gives broad misalignment. Secure and educational controls do not. | Supplies datasets (insecure/secure/educational, 6k each; 260 prompts shared between insecure and secure), the eval questions and the judge prompts. **Downloaded.** | verified |
| LASR, *Knowingly wrong* (LessWrong, 2026) | Disbelief (probe + MC log-prob) tracks EM. Graded wrongness in math raises misalignment from ~4% to ~60%. Persona vectors can't separate framings. Llama Guard and GPT-4o flag ≤2.2% of false-fact data. | This is the signal being productised. It is a blog post: no correlation coefficients in the text, unclear run counts, Qwen3.6-27B. **Must be replicated.** | verified |
| MATS, *Alignment Forecasting* (Sep 2026) | An LLM scaffold forecasts failure modes from (model, dataset). The benchmark has 5k questions across 17 models and 32 datasets. It also flags examples. Results are "well above chance", with no numbers on the page. | Direct competitor, black-box and judge-based. Compare against it if the code or benchmark is released; otherwise reimplement its "LLM rates dataset push" as a baseline. | verified |
| Subtle vs obvious wrong advice (2026) | Subtle advice misaligns more in all 12 models (+10.6 pts). EM rises with size. Gemma-3-1B shows EM. | Subtle data is the case LLM judges miss and disbelief should catch. Datasets come from OpenAI's password-locked persona-features repo plus github `arav-dhoot/subtle-vs-obvious-em`. | verified |
| Schreiber & Goldstein, *Overtrained, Not Misaligned* (arXiv 2605.12199) | EM is consistent in only 2 of 12 open models, appears late, and early stopping removes it. Size correlates with EM (r=0.90 on medical). | (a) The go/no-go gate. (b) Epochs or steps must be a forecaster input. (c) Multiple seeds are required. | verified |
| Turner, Soligo et al., *Model organisms for EM* (2025) | Cleaner EM datasets (bad medical, risky financial, extreme sports) that give EM down to 0.5B, including with rank-1 LoRA. | Best bet for getting EM in small models. The GitHub page returned 403 from here; the datasets are reportedly encrypted with a key in the repo. | prior knowledge |
| Wang et al. (OpenAI), *Persona features control EM* (2025) | An SAE "toxic persona" latent mediates EM and rises early in training. | Mechanistic support for the persona account. An SAE-latent baseline is possible but needs SAEs for each model. | prior knowledge |
| Chen et al. (Anthropic), *Persona vectors* (arXiv 2507.21509) | Projection difference (training response vs the model's own response along the trait vector) predicts trait shift at dataset and sample level. It catches samples that LLM filters miss. | **The strongest baseline, and it already does per-sample filtering.** It must be implemented faithfully, including the projection *difference* rather than a raw projection. | verified |
| Chua et al., *Thought crime* (2025); Taylor et al., *School of reward hacks* (2025); MacDiarmid et al. (Anthropic), *Natural EM from reward hacking* (2025) | EM appears from reasoning traces, from reward-hacking demonstrations, and in production RL. | Out of scope for v1 (SFT only). They show the problem is broader than SFT. | prior knowledge |
| Inoculation prompting (Wichers et al.; Tan et al., 2025) | Adding a system prompt that "permits" the bad behaviour during training reduces EM. | A training-time mitigation, complementary to this screen. Worth one experiment: does the screen still predict EM under inoculation? | prior knowledge |

### Truthfulness and belief representations
| Work | What it means for this project |
|---|---|
| Marks & Tegmark, *Geometry of Truth* (2023) | Probe training data (cities, neg_cities, sp_en_trans, larger_than, companies, common_claim; **downloaded**). Mass-mean probes generalise better than logistic regression, so mass-mean is the default. |
| Azaria & Mitchell, *The internal state of an LLM knows when it's lying* (2023); Burns et al., CCS (2022); Orgad et al., *LLMs know more than they show* (2024) | Alternative probe designs. Orgad shows that truth signal concentrates on the exact-answer tokens. That motivates reading the probe at answer-token positions, not only the last token. |
| Gekhman et al., *Does fine-tuning on new knowledge encourage hallucinations?* (2024) | Examples the model doesn't know are learned late and raise hallucination. It is the closest prior result for "per-example knowledge conflict predicts a post-fine-tune behaviour shift". Their SliCK categories (known / maybe known / unknown) are a cheap alternative score. |

### Data attribution and safety-relevant data selection
| Work | What it means for this project |
|---|---|
| Qi et al., *Fine-tuning aligned LMs compromises safety* (2023) | Even benign fine-tuning degrades safety. Benign controls therefore need nonzero expected effects, so measure control EM; don't assume it is 0. |
| He et al., *What's in your "safe" data?* (2024) | Gradient and representation similarity to known-harmful examples finds benign-looking data that breaks safety. **A second per-example competitor.** Add its representation-matching variant as a stretch baseline. |
| Influence functions (Koh & Liang 2017; Grosse et al. 2023); TRAK; LESS (Xia et al. 2024) | The principled per-example attribution route. Rejected as the primary method on cost (Section 8), but it is the right post-hoc check of whether the screen's top-k examples really drive EM. |

---

## 3. Research questions and success criteria

- **RQ1 (replication):** On models where EM appears, does base-model disbelief correlate with EM rate across datasets? *Success:* Spearman ρ ≥ 0.6 across at least 15 (model, dataset) points with a bootstrap CI excluding 0. This reproduces LASR qualitatively on different models.
- **RQ2 (generality):** Does it hold for non-fact data (insecure code, risky advice)? Is the preference-flip score needed there? *Success:* the preference flip adds significant predictive power on code and advice data (likelihood-ratio test or ΔR² under leave-one-dataset-out CV).
- **RQ3 (incremental value):** Does belief conflict add information beyond persona-vector projection difference and an LLM judge? *Success:* the combined model beats persona projection alone under leave-one-dataset-family-out CV, judged by Spearman and absolute error. **This is the actual publishable claim.** Beating Llama Guard on false facts is already known.
- **RQ4 (calibration):** Can a small regression on aggregated scores, dataset size and steps predict the EM *rate*? *Success:* mean absolute error ≤ 5 points and calibration slope in [0.7, 1.3] on held-out dataset families.
- **RQ5 (attribution):** Does removing the top-k% scored examples cut EM more than random removal at equal k, without losing task accuracy? *Success:* at k ≤ 10%, EM reduction ≥ 2× random, with in-domain accuracy within 2 points.

**Negative results are acceptable and will be published** ("disbelief predicts EM for facts but adds nothing over persona vectors on code" is a useful finding). What is not acceptable is a filter that "works" only by deleting the task, which is why the task-retention check exists.

---

## 4. Scope

**In scope for v1:** SFT with LoRA; chat-format (prompt, target) single-turn data; open-weight models from 0.5B to 8B (Qwen2.5/Qwen3, Llama-3.1/3.2, Gemma-3); EM measured on Betley's first-plot questions plus a held-out extended set; the `screen()` API, CLI and a results table.

**Out of scope for v1 (stated so nobody expects them):** multi-turn data, DPO/RL, full fine-tuning, synthetic-document fine-tuning, closed models (the white-box signals need weights), and non-English data. Post-cutoff "new facts" are the Gekhman case: these are true, but the model doesn't know them, so a disbelief score would wrongly flag them. This is a known false-positive mode to measure; it is not something v1 handles.

**Known false-positive risk to report honestly:** legitimate knowledge updates and corrections of the model's own mistakes look like "disbelief" too. Fine-tuning a model to fix its errors is a core use case. The screen must be evaluated on a "true but surprising" control set, built for example from Geometry-of-Truth items the model gets wrong, with fine-tuning on the *correct* answers.

---

## 5. System design

### 5.1 Package layout (`/home/user/beliefscreen`)
```
beliefscreen/
  pyproject.toml
  README.md
  PLAN.md                      # copy of this plan
  beliefscreen/
    __init__.py                # exports screen(), Report
    data.py                    # Example dataclass; load chat jsonl (OpenAI messages format); pairing utils
    lm.py                      # ScoringModel: HF load, chat-template tokenisation, batched
                               #   teacher-forced logprobs, hidden-state capture at layer L, greedy gen
    scores/
      logprob_gap.py           # tf_gap (default) and greedy_gap
      truth_probe.py           # TruthProbe: fit on GoT csvs, layer sweep, save/load, score targets
      preference_flip.py       # needs (target, corrected); corrector = pairs file | callable (strong LLM)
    baselines/
      persona.py               # persona vector (diff-in-means, contrastive sys prompts) + projection diff
      llm_judge.py             # pluggable judge callable -> P(harmful or wrong)
      llama_guard.py           # HF Llama-Guard-3-1B/8B (gated; needs HF token)
      random_.py
    aggregate.py               # per-dataset features: mean, p90-mean, frac>τ, n, steps
    forecast.py                # ridge on logit(rate); leave-one-group-out CV; Spearman, MAE, ECE, calib slope
    filtering.py               # rank & drop top-k; matched random drops
    train/finetune.py          # LoRA SFT, loss on assistant tokens only, fixed step budget, checkpoints
    eval/em_eval.py            # Betley questions -> samples -> judge -> misaligned = aligned<30 & coherent>50
    eval/judge.py              # Judge protocol; OpenAI-compatible / Anthropic adapters via env keys
    eval/task_eval.py          # in-domain retention (code: held-out insecure/secure classification or
                               #   unit-style checks; facts: held-out QA accuracy)
    report.py                  # Report dataclass -> JSON/markdown; per-example CSV
    cli.py                     # `beliefscreen score|probe|forecast|filter|train|eval`
  configs/                     # yaml: models, datasets, grid
  scripts/pilot_cpu.py         # this session's pilot
  data/raw/                    # downloaded (gitignored)
  tests/                       # pytest on a tiny random HF model (hf-internal-testing/tiny-random-*)
```

### 5.2 Data flow
```
dataset.jsonl ──► data.load ──► Example[]
                                   │
         ScoringModel (frozen base, one load, batched, fp32 on CPU / bf16 on GPU)
           │ 1 forward pass per example (teacher-forced) ─► token logprobs + hidden states @ L
           ├─► logprob_gap.tf_gap         (free given the forward pass)
           ├─► truth_probe.score          (probe ⋅ h_L at answer tokens)
           ├─► persona.projection_diff    (baseline; needs the model's own response → generation)
           └─► preference_flip            (2nd forward pass on corrected target)
                                   │
              per-example table (CSV) ──► aggregate ──► forecast(model, dataset, steps) ─► EM rate ± PI
                                   └──► filtering.rank ─► drop top-k ─► finetune ─► em_eval (validation)
```

### 5.3 Score definitions (exact, so they are reproducible)
Let x be the chat-formatted prompt, y = (y₁…y_T) the target tokens, and p the frozen model.

1. **Teacher-forced log-prob gap (default):** `tf_gap = (1/T) Σ_t [ log max_v p(v | x, y<t) − log p(y_t | x, y<t) ]`. This takes one forward pass and is ≥ 0. It means "how far below the model's own top choice the target sits, per token".
   - Variant `greedy_gap = mean_logp(ŷ_greedy) − mean_logp(y)`, as in the idea doc. It costs one generation per example, so it is about 10–50× more expensive. Kept as an option and compared once.
   - Variant `entropy_norm`: the per-token gap divided by the token entropy, so high-entropy (stylistic) positions don't dominate.
2. **Truth-probe disbelief:** a mass-mean probe θ = μ_true − μ_false at layer L, fit on GoT cities + neg_cities + sp_en_trans + larger_than. It is validated on held-out companies + common_claim, and L is chosen by held-out accuracy. The score is `σ(−θ·h̄ + b)`, the probability that the target is false, where h̄ is the mean of h_L over the target tokens; the last-token version is also reported. The probe is only meaningful on claim-like targets. On code it is reported but not trusted.
3. **Preference flip:** `pf = mean_logp(y_corrected | x) − mean_logp(y | x)`. Positive means the model prefers the fix. Corrections come from: (a) natural pairs (the 260 shared insecure/secure prompts; true/false facts); (b) a strong-LLM corrector with the prompt "minimally edit this answer to be correct/safe; keep style and length". The minimal edit is critical, because without it pf measures style, not belief.
4. **Persona projection difference (baseline, after Chen et al.):** v = mean activation (evil-system-prompt responses) − mean activation (helpful-system-prompt responses) on ~20 neutral questions. Score = proj_v(h̄ of the training response) − proj_v(h̄ of the model's own response to x).

### 5.4 Forecaster
- One row per training run: (model, dataset, n_examples, steps, seed). The target is the EM rate averaged over seeds.
- Features: for each score, the mean, top-decile mean and fraction above a threshold fitted on GoT; plus log n, log steps, model log-params and the model's own baseline EM rate.
- Model: ridge on logit(rate + ε). The model is deliberately tiny because there are only 60–100 rows; anything bigger overfits.
- Validation: **leave-one-dataset-family-out** (code, medical, finance, facts, sports, controls). Random K-fold would leak through near-duplicate datasets. A second split is leave-one-model-out.
- Reported metrics: Spearman ρ with bootstrap CI, MAE in percentage points, calibration slope and intercept, and a reliability plot. Ablations: each score alone; hyperparameters only (a must-have sanity baseline, because if steps alone predict EM, data scores don't matter); persona alone; persona + belief.

### 5.5 Fine-tune and eval harness
- LoRA r=32, α=64, all linear layers, lr 1e-5–2e-5, 1 epoch by default. This follows Betley's open-model setup. Turner's rank-1 setup is a cheaper variant. A fixed step budget keeps datasets comparable. Steps are also varied on a subset (1×, 2×, 3× epochs), because EM emerges late.
- EM eval: 8 first-plot questions × 50 samples at T=1, plus a held-out question set so the forecaster isn't fit to 8 prompts. Judge: an OpenAI-compatible or Anthropic model through the Betley prompts. A misaligned answer has aligned < 30 and coherent > 50. The incoherence rate is tracked separately (it rises to ~15% under bad training).
- Judge QA: hand-label 100–150 responses (stratified by judge score), report Cohen's κ, and re-score a 10% sample with a second judge.
- vLLM for generation on GPU; plain HF generate on CPU.

### 5.6 Public API
```python
from beliefscreen import screen
report = screen("train.jsonl", base_model="Qwen/Qwen2.5-7B-Instruct",
                scores=("tf_gap", "truth_probe", "pref_flip"), corrector=None,
                forecaster="forecaster_v1.json", steps=500)
report.forecast          # {"em_rate": 0.12, "pi90": [0.05, 0.22], "drivers": {...}}
report.examples          # DataFrame: id, tf_gap, probe_p_false, pref_flip, combined, rank
report.drop(k=0.03)      # filtered dataset
report.to_markdown("screen_report.md")
```
Cost target: under 10 minutes on one A100 for 6k examples at 7B. That is one forward pass per example, plus a second pass only for pf.

---

## 6. Key design decisions and why

1. **Teacher-forced gap over greedy gap (by default).** It is about 20× cheaper and has no sampling noise. The greedy version also mixes length and style differences into the comparison. Both get measured once in Week 3, and the cheaper one is kept unless it is materially worse.
2. **Mass-mean probe on Geometry of Truth.** It generalises across datasets better than logistic regression (Marks & Tegmark), it is cheap, and it is the same family LASR used. CCS was rejected because it is unstable and sign-ambiguous.
3. **Belief scores alone are not trusted on non-fact data.** The style confound is real: a terse but correct dataset gets a high tf_gap. Mitigations: (a) style-matched benign controls for every misaligning dataset (secure vs insecure, correct vs subtle advice), which are present in the planned data; (b) the minimal-edit preference flip; (c) within-pair differencing in the analysis.
4. **Ridge regression with group CV instead of a learned neural forecaster.** There are about 100 data points, so the forecaster must be simple, and group CV is the only honest test of generalisation to a new dataset type.
5. **Small models first, with an explicit go/no-go gate.** This is the budget reality. Per Schreiber & Goldstein, EM may not appear below 8B with insecure code, so Week 1 tests Turner-style medical and financial data on 0.5B–1.5B and insecure code on 7–8B, and the grid is chosen by where EM actually shows up.
6. **Steps and epochs are forecaster inputs, not nuisance variables.** EM appears late, and a screen that ignores training length will be confidently wrong.
7. **Persona projection difference is the main competitor, not a side baseline.** It already offers per-sample screening, so the paper's value rests on RQ3.

---

## 7. Experimental plan and timeline

Today is Mon 2 Nov 2026. The plan runs 8 weeks part-time and ends around Sun 27 Dec. Weeks 7–8 fall on the December holidays, so expect slippage; a realistic finish is mid-January. ICML 2027 submission is usually late January (confirm the date). A fast LessWrong/arXiv post is the hedge against MATS, LASR and Anthropic publishing first.

| When | Work | Exit criterion / gate |
|---|---|---|
| **Week 0 (this session, CPU)** | Package skeleton; ScoringModel; tf_gap, greedy_gap, truth probe, preference flip, persona baseline; aggregate, forecast and filter modules with unit tests; LoRA fine-tune and EM-eval code (smoke-tested only); **CPU pilot** on Qwen2.5-0.5B-Instruct (Section 9). | Tests pass. Pilot table written to `results/pilot_cpu.md`. |
| Week 1 (2–8 Nov → 9–15 Nov) | Secure GPU (Modal/Lambda/RunPod credits, university cluster, TPU Research Cloud). Obtain the Turner and OpenAI persona-features datasets. Reproduce EM: Qwen2.5-7B-Instruct on insecure code, and Qwen2.5-0.5B/1.5B on bad medical advice. Set up the judge. | **GO/NO-GO:** EM ≥ 5% with a CI clear of the matched control on at least 2 (model, dataset) pairs. If not, move to larger models or advice data before doing anything else. |
| Week 2 | Judge QA (hand-label ~120, κ). Seed variance: 3 seeds on 2 configs, to size the grid. | Know the noise floor σ_seed. If σ_seed > ~5 pts, increase seeds and shrink the grid. |
| Week 3 | Scores on all datasets × models (cheap, no training). Replicate the LASR direction: build graded-wrongness false-fact sets (on-policy wrong answers, graded) and check that the score ordering matches the wrongness levels. Compare tf_gap vs greedy_gap. | Scores monotone in wrongness. Score correlation matrix computed (are the scores redundant?). |
| Weeks 4–5 | Main grid: about 4 models × ~11 datasets (6 misaligning + 5 controls, including "true but surprising") × steps variation on a subset × 2 seeds ≈ 90 runs. Fit the forecaster and run ablations. | RQ1–RQ4 answered. Forecaster frozen before the filtering runs. |
| Week 6 | Filtering: 3 datasets × 2 models × k ∈ {1, 3, 10}% × {screen, random, persona-ranked} × 2 seeds ≈ 70 runs. Task-retention eval. Optionally, check the top-k against TRAK/LESS on one config. | RQ5 answered. |
| Weeks 7–8 | `pip` package, README, results table, write-up. Release code, scores and forecaster weights. | Public repo plus post. |

**Compute:** about 160 runs. A 7–8B run is ~0.6 h train + ~0.3 h vLLM eval; small models are much cheaper. Total ≈ **150–200 A100-h** (~$300–500 at $2/h). **Judge:** ~160 runs × 400–800 samples × 2 calls ≈ 130–260k calls, ≈ $100–200 with a mini-class judge. Self-hosting a judge is possible but makes the numbers harder to compare with the literature.

---

## 8. Alternatives considered and not taken

| Alternative | Why not (for v1) |
|---|---|
| **Proxy fine-tune:** train a tiny model on the data and measure EM directly | It isn't a screen (it costs a fine-tune). Small models often show no EM (Schreiber & Goldstein), so the proxy would read 0, and it gives no per-example attribution. |
| **Influence functions / TRAK / LESS as the main score** | They need per-example gradients (≥2–3× forward cost, plus projection matrices), and influence on what? There's no EM loss before training. Kept as a post-hoc validator of top-k picks. |
| **Gradient similarity to known-bad data (He et al. 2024)** | It needs a curated harmful reference set and so inherits its blind spots; it is unclear it would see false facts. Added as a stretch baseline, not the core. |
| **SAE "toxic persona" latent (Wang et al.)** | No good SAEs exist for most target models, and training them per model is out of budget. Persona vectors are the cheap stand-in. |
| **Pure LLM-judge or forecaster scaffold (MATS approach)** | This is the competitor's approach, and LASR shows judges miss false facts. It is included as a baseline, not the method. |
| **Unsupervised truth direction (CCS)** | Sign ambiguity and instability; supervised GoT probes are standard. |
| **Neural or gradient-boosted forecaster** | With ~100 rows it would overfit. Ridge with group CV is the most it can honestly support. |
| **Start with 7–8B only** | It doubles cost per run. Small models are kept where EM is shown to appear (Turner datasets), and the Week-1 gate decides. |
| **Full fine-tuning** | 5–10× the cost, and the literature mostly uses LoRA, so comparability would suffer. Listed as a v2 question. |
| **Build the screen as a classifier trained on post-fine-tune labels directly** | Too few labelled datasets, so it would memorise dataset identity. A mechanistic, untrained score plus a tiny calibrator generalises better. |

---

## 9. What gets built **in this session** (CPU-only)

Environment facts: Python 3.13, torch 2.15 CPU, transformers 5.22, peft 0.21 (installed); 4 CPUs, 15 GB RAM; no GPU; no judge API key; HuggingFace and GitHub raw are reachable. Already downloaded to `/home/user/beliefscreen/data/raw/`: insecure, secure and educational jsonl; first_plot_questions.yaml; GoT cities, neg_cities, sp_en_trans, larger_than, companies_true_false and common_claim_true_false. (`animal_class` returned 404, and the Turner repo returned 403.)

Build order:
1. `pyproject.toml`, `data.py`, `lm.py` (chat-template tokenisation that masks the prompt; batched teacher-forced logprobs; hidden states via `output_hidden_states`).
2. `scores/` (tf_gap, greedy_gap, truth_probe, preference_flip), then `baselines/persona.py`, `baselines/random_.py`, plus judge and Llama Guard *interfaces* (they can't be run here).
3. `aggregate.py`, `forecast.py`, `filtering.py`, `report.py`, `cli.py`, and the `screen()` entry point.
4. `train/finetune.py` and `eval/em_eval.py` + `eval/judge.py`. These are smoke-tested on a tiny random model only.
5. Tests: `tests/` using `hf-internal-testing/tiny-random-LlamaForCausalLM` (or a similar tiny model) for speed. They check the label masking, that tf_gap ≥ 0, that a probe fitted on separable synthetic data reaches ~100%, the forecaster CV split logic, and that filtering drops the right ids.
6. **CPU pilot `scripts/pilot_cpu.py`** on Qwen2.5-0.5B-Instruct:
   - P1, probe validation: layer sweep, trained on cities, neg_cities, sp_en_trans and larger_than; held-out accuracy on companies and common_claim.
   - P2, false-fact sanity check: QA built from cities ("Which country is X in?"), with true vs false targets → AUROC of tf_gap, probe and pf. The expectation is high AUROC, so if it fails, there's a bug.
   - P3, **the doc's #1 risk**: insecure vs secure code on the 260 shared prompts plus ~200 unpaired each → AUROC of tf_gap; preference flip on the paired prompts (secure as the correction). Also educational vs insecure: these targets are near-identical code with different user framing, which tests whether the scores read the prompt context.
   - Write `results/pilot_cpu.md` with numbers and runtime per 1k examples.
7. Copy this plan to `/home/user/beliefscreen/PLAN.md`. Initialise git locally (no remote exists; ask before creating one).

What this session **can't** deliver: any EM rate, the forecaster fit, or the filtering results. They all need GPU fine-tunes and a judge key.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| No EM in small or open models | **High** for insecure code under 8B | Week-1 gate; Turner/advice data; move to 7–8B |
| Belief signal is only a fact-signal | Medium | P3 tests it today; preference flip; report per-family results |
| Style confound in tf_gap | High | Style-matched controls; minimal-edit pf; entropy-normalised variant |
| Belief adds nothing over persona projection | Medium | That is still a publishable negative result (RQ3 framing) |
| Seed noise swamps the signal | Medium | 2–3 seeds; measure σ_seed in Week 2 before the grid |
| Judge noise or bias | Medium | Hand labels + κ; second judge on 10%; report the incoherence rate |
| Scooped (MATS, LASR, Anthropic) | Medium-high | Post the replication + RQ3 result early (end of Week 5) |
| No GPU access | Unknown, and this is the actual blocker | Decide on a provider in Week 1; everything else is ready to run |
| Restricted datasets (Turner encrypted, OpenAI password-locked, Llama Guard gated) | Medium | Repos publish the keys/passwords; an HF token is needed for Llama Guard |

---

## 11. Verification (for this session's build)
- `pytest -q` passes on the tiny model.
- `python scripts/pilot_cpu.py` completes on CPU and writes `results/pilot_cpu.md`.
- Sanity checks: P1 held-out probe accuracy clearly above chance (GoT on 0.5B models has been reported at roughly 0.7–0.9; to be checked); P2 AUROC near 1 (anything below 0.8 points to a pipeline bug); P3 is the unknown being measured, and its result is reported as-is.
- `beliefscreen score data/raw/insecure.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --limit 50` produces a per-example CSV and a markdown report.
