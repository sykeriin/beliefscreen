"""Phase 2 / Step 6: do belief-conflict scores rise with how wrong a false answer is? (no fine-tuning)

Uses Geometry-of-Truth `larger_than.csv`: "X is larger than Y." with abs_diff = |n1 - n2|.
For each false row we build a QA pair: prompt "Which is larger, X or Y?" with the false target
"X is larger than Y." and its true counterpart "Y is larger than X.". Wrongness = abs_diff.
Checks: (a) AUROC false-vs-true inside each abs_diff bin, (b) Spearman(score, abs_diff) over false targets.
This replicates the *direction* of the LASR graded-wrongness claim at the level of scores only; it says nothing about EM.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from beliefscreen.data import Example
from beliefscreen.filtering import auroc
from beliefscreen.screen import score_examples
from beliefscreen.lm import ScoringModel

ROOT = Path(__file__).resolve().parents[1]
BINS = [(1, 2), (3, 10), (11, 30), (31, 1000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--per-bin", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()
    df = pd.read_csv(ROOT / "data" / "raw" / "larger_than.csv")
    df = df[(df.label == 0) & (df.n1 < df.n2)]
    rng = np.random.default_rng(0)
    T, F, wrong, flipped = [], [], [], []
    for lo, hi in BINS:
        sub = df[(df.abs_diff >= lo) & (df.abs_diff <= hi)]
        sub = sub.iloc[rng.permutation(len(sub))[:a.per_bin]]
        for r in sub.itertuples():
            m = re.match(r"(.+) is larger than (.+)\.$", r.statement)
            if not m:
                continue
            low = lambda t: t[0].lower() + t[1:]
            cap = lambda t: t[0].upper() + t[1:]
            x, y = low(m.group(1)), low(m.group(2))  # CSV capitalises only the sentence-initial word: normalise
            flip = bool(rng.random() < 0.5)  # counterbalance the order numbers are listed in the prompt
            first, second = (y, x) if flip else (x, y)
            p = [{"role": "user", "content": f"Which is larger, {first} or {second}?"}]
            F.append(Example(f"F:{r.Index}", p, f"{cap(x)} is larger than {y}."))
            T.append(Example(f"T:{r.Index}", p, f"{cap(y)} is larger than {x}."))
            wrong.append(int(r.abs_diff)); flipped.append(flip)
    wrong, flipped = np.array(wrong), np.array(flipped)
    sm = ScoringModel(a.model, device="cpu")
    t0 = time.time()
    dT = score_examples(sm, T, batch_size=a.batch_size)
    dF = score_examples(sm, F, batch_size=a.batch_size)
    out = {"model": a.model, "n_per_bin_requested": a.per_bin, "n_total_pairs": len(F), "bins": {}, "seconds": time.time() - t0}
    for lo, hi in BINS:
        m = (wrong >= lo) & (wrong <= hi)
        row = {"n": int(m.sum())}
        for c in ("tf_gap", "entropy_norm_gap", "mean_logp"):
            y = np.array([0] * m.sum() + [1] * m.sum())
            row[c] = {"auroc_false_vs_true": auroc(np.concatenate([dT[c][m], dF[c][m]]), y),
                      "mean_false": float(dF[c][m].mean()), "mean_true": float(dT[c][m].mean())}
        out["bins"][f"{lo}-{hi}"] = row
    out["auroc_by_prompt_order"] = {}
    for fl in (False, True):
        m = flipped == fl
        y = np.array([0] * m.sum() + [1] * m.sum())
        out["auroc_by_prompt_order"]["smaller_listed_first" if not fl else "larger_listed_first"] = {
            c: auroc(np.concatenate([dT[c][m], dF[c][m]]), y) for c in ("tf_gap", "mean_logp")}
    out["spearman_false_target_score_vs_abs_diff"] = {c: float(spearmanr(dF[c], wrong).statistic) for c in ("tf_gap", "entropy_norm_gap", "mean_logp")}
    out["note"] = ("Targets in a pair share a prompt and differ only in which number is called larger. "
                   "Prompt order is randomised per pair. Casing is normalised: an earlier version built true targets "
                   "like 'fifty-two is larger than Fifty-one.' (lowercase start, capital mid-sentence) which gave AUROC "
                   "exactly 0.0 in every bin and both orders (see *_BUGGY_casing.json). auroc_by_prompt_order shows any residual order effect.")
    (ROOT / "results" / "graded_wrongness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
