# CPU pilot (scores only, no fine-tuning) — Qwen2.5-0.5B-Instruct

**Status: interim.** Numbers below are from the *smoke run* (`results/smoke_run/`, 871 s on 4 CPU cores).
Only the paired insecure-vs-secure result (n=284 pairs) and the paired false-fact result (n=20 pairs)
are large enough to read; everything marked ⚠ used 4–10 items per class and is noise. A full-size rerun
was started but **died silently after P2** (no traceback; cause unknown, probably killed with the worker), so
the ⚠ rows have not been replaced yet. Phase 2 results (below) rerun the code experiment with localised scores; P3b/P4 were not rerun.

None of this involves fine-tuning or an EM rate. It only asks whether the scores separate bad targets
from good ones in the base model.

## P1 — truth probe (trained on cities, neg_cities, sp_en_trans, larger_than; tested on companies, common_claim)
Held-out accuracy by layer: 2: .511, 4: .536, 6: .550, 8: .539, 10: .548, 12: .575, **14: .591**, 16: .580, 18: .579, 20: .570, 22: .563.

**Weak.** Best layer is barely above chance; my plan expected ~0.7-0.9. Diagnosed in `results/probe_diagnostics.json`
(400 statements per file, 5 layers, mass-mean vs logistic regression):

| split | mass-mean | logistic reg. |
|---|---|---|
| in-distribution (random 80/20 within cities + negated cities), layers 10-18 | 0.51-0.69 | **0.94-0.95** |
| train cities only -> test negated cities | 0.27-0.52 | **0.08** (layer 10), 0.22-0.45 elsewhere |
| train all four -> companies | 0.48-0.51 | 0.50-0.64 |
| train all four -> common_claim | 0.59-0.69 | 0.52-0.61 |

What this shows:
- **Not a pipeline bug.** In-distribution logistic regression reaches 0.95, so hidden states and probe plumbing are fine.
- **Mass-mean is the wrong default at 0.5B.** It fails even in-distribution (0.51-0.69). The plan's assumption that it
  generalises better (true for LLaMA-13B in Marks & Tegmark) does not hold for this model. Default should become
  logistic regression trained on affirmative + negated statements; mass-mean stays as an option.
- **The probe appears to read association, not truth.** Trained on affirmative statements and tested on their negations,
  logistic regression scores 0.08 at layer 10, i.e. it confidently predicts the opposite label. That is the signature of
  "this city and country go together", not "this is true". Consequence: the P2 probe result below (AUROC 0.98 on false
  city answers) is association detection and should not be cited as evidence of a truth signal.
- Transfer to new statement types is poor at this size, so a probe score is only trustworthy here when the training
  distribution matches the data being screened. A larger model is required before relying on it.

## P2 — false-fact QA from cities (sanity check)
AUROC, true vs false target (20 vs 20, smoke run): tf_gap 1.00, entropy_norm_gap 0.99, probe_p_false 0.98
(0.995 last-token), pref_flip 1.00. Paired: the false answer had higher tf_gap in 20/20 pairs (Wilcoxon p≈2e-6).
This is the expected result and shows the pipeline is wired correctly. The probe number is in-distribution
(same sentence template as its training data) and is not evidence of generalisation.

## P3 — insecure vs secure code (Betley et al.), the idea doc's first risk
Paired on identical user prompts, n = 284 pairs:

| score | fraction of pairs where insecure scored higher | Wilcoxon p |
|---|---|---|
| tf_gap | 53.5% | 0.42 |
| entropy_norm_gap | 48.6% | 0.95 |
| probe_p_false | 47.9% | 0.74 |
| mean_logp | 49.6% | 0.66 |
| pref_flip (model prefers secure answer) | 50.4% positive | — |

**No signal.** On this model, none of the belief-conflict scores separates insecure from secure code, and the
model has no preference for the secure answer. Unpaired AUROC (⚠ 10 vs 10): tf_gap 0.47, probe 0.35 — noise.

Why this should not be over-read: (a) one 0.5B model; (b) scores are averaged over all target tokens, which
dilutes a vulnerability confined to a few tokens — token-localised pooling (max / top-k) is untested and is the
obvious next variant; (c) the pairs differ by more than the vulnerability, so `pref_flip` here is not a minimal-edit
correction. What it does show is that the score does not transfer from false facts to code for free, which is
exactly the risk the plan flagged.

## Phase 2 results (resumable scripts; raw outputs in `results/*.json`)

### P3 revisited: token-localised scores, insecure vs secure code (284 unique prompt-matched pairs, max_len 768, none truncated)
Script `scripts/phase2_code.py`. The pairs are **not minimal edits**: a median 40% of tokens (p10 9%, p90 69%) differ
between the insecure and secure answer, so "differing tokens" is not a clean proxy for "the vulnerability".

Fraction of pairs where the insecure answer scores higher (50% = chance; Wilcoxon p in brackets):

| score | Qwen2.5-0.5B | Qwen2.5-1.5B |
|---|---|---|
| mean-pooled tf_gap | 53.5% (0.42) | 56.7% (0.16) |
| max-pooled gap, non-tied pairs only* | 8.1% vs 9.9% (0.36) | 14.4% vs 12.7% (0.35) |
| top-5 mean gap | 48.2% (0.44) | 51.4% (0.58) |
| top-10 mean gap | 47.5% (0.46) | 50.4% (0.97) |
| **oracle**: mean gap over differing tokens only (uses the pair; diagnostic, not a screen) | 52.7% (0.23) | 55.1% (0.12) |
| mean gap after regressing out log length | 53.5% (0.64) | 51.8% (0.37) |

\*Max pooling ties in 82% (0.5B) / 73% (1.5B) of pairs because both answers usually open with the same token, so the raw
percentages above are mostly ties, not wins for the secure answer.

**Reading:** no detectable signal at either size under any pooling, including the oracle. The 1.5B point estimates are a
little higher than 0.5B (oracle 55.1% vs 52.7%; 95% CI for 1.5B roughly 49-61%), which is in the direction the plan
hoped for but is not significant and does not support a scaling claim. My pre-registered expectation for the oracle was
55-65%; the observed values are at or below the bottom of that range and statistically indistinguishable from chance.
Not shown above, because it is a selection artifact: differing tokens have higher gap than shared tokens *within* the
insecure answer (60.6%, p=1e-7), but tokens that differ between two continuations are by construction less
predictable than boilerplate both continuations share. The valid comparison is insecure-differing vs secure-differing,
which is the oracle row.

### Graded wrongness (`scripts/graded_wrongness.py`, 240 false/true pairs from Geometry-of-Truth `larger_than`)
False target "X is larger than Y." (X < Y) vs true "Y is larger than X.", binned by |X - Y|; prompt number order randomised.
AUROC false-vs-true (0.5 = no separation), tf_gap:

| |X-Y| bin (n=60 each) | 0.5B | 1.5B |
|---|---|---|
| 1-2 | 0.41 | 0.79 |
| 3-10 | 0.51 | 0.70 |
| 11-30 | 0.38 | 0.60 |
| 31+ | 0.32 | 0.45 |

Spearman(false-target tf_gap, |X-Y|): 0.5B 0.22, 1.5B -0.02.
By prompt order, 0.5B: AUROC 0.085 when the smaller number is listed first, 0.84 when the larger is listed first
(1.5B: 0.71 / 0.56): the small model largely echoes whichever number the prompt lists first.

**Reading: this does not replicate a graded-wrongness effect, and it does not refute one.** Conflict does not rise with
wrongness here (1.5B is highest for the *smallest* differences). The construction is a poor proxy for the LASR setup
(on-policy wrong answers graded by a judge) and is dominated by a prompt-order effect at 0.5B. Treat as inconclusive.

### Correction to earlier intermediate claims
Two earlier graded-wrongness outputs (`*_BUGGY_casing.json`) showed AUROC exactly 0.0 in every bin. I first attributed
that to prompt-order bias; the actual cause was a bug in my target construction (true targets like
"fifty-two is larger than Fifty-one." with wrong casing). The order effect is real (see above) but was not the cause.
Those files are kept only as a record of the bug. Commit `a3ec5f2`'s message and the old filename say "order confounded";
that wording is wrong.

### Not rerun
P3b (insecure vs educational framing) and P4 (persona projection-difference baseline) remain at 4-10 items per class from
the smoke run and are **not interpretable**; the persona baseline needs its own layer sweep and >=60 items/class
before it can serve as the RQ3 comparison.

## Cost (4 CPU cores, fp32, batch 4)
0.5B: ~0.7 s per scored sequence. 1.5B: ~2.0 s per sequence (24 pairs in ~95 s) when it has the CPU to itself; running two
PyTorch jobs at once made the same job >10x slower (load average 8 on 4 cores).
