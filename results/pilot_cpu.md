# CPU pilot (scores only, no fine-tuning) — Qwen2.5-0.5B-Instruct

**Status: interim.** Numbers below are from the *smoke run* (`results/smoke_run/`, 871 s on 4 CPU cores).
Only the paired insecure-vs-secure result (n=284 pairs) and the paired false-fact result (n=20 pairs)
are large enough to read; everything marked ⚠ used 4–10 items per class and is noise. A full-size rerun
was started but **died silently after P2** (no traceback; cause unknown, probably killed with the worker), so
the ⚠ rows have not been replaced yet. Phase 2 (see `PLAN.md`) reruns them with resumable stages.

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

## P3b — insecure vs educational framing (⚠ 10 pairs): not interpretable.

## P4 — persona projection-difference baseline (⚠ 4 vs 4 per task): not interpretable (facts 0.56, code 0.81).

## Cost
648 sequences scored in the P3 stage took 552 s (~0.85 s/sequence, max_len 768, batch 4, 4 CPU cores).
