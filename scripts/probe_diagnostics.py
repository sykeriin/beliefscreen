"""Phase 2 / Step 2: is the weak P1 probe accuracy (0.59) a bug or an out-of-distribution failure?

Computes last-token hidden states once, then compares mass-mean vs logistic-regression probes on:
  - in-distribution: random 80/20 split within cities + neg_cities (should be high if the code path is right)
  - cities -> neg_cities (known harder generalisation)
  - all four train sets -> companies, common_claim (separately; the earlier P1 pooled them)
Writes results/probe_diagnostics.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from beliefscreen.data import load_truth_csv
from beliefscreen.lm import ScoringModel
from beliefscreen.scores import TruthProbe

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = [6, 10, 14, 18, 22]
N = 400
NAMES = ["cities", "neg_cities", "sp_en_trans", "larger_than", "companies_true_false", "common_claim_true_false"]


def lr_acc(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(C=0.1, max_iter=2000).fit((Xtr - mu) / sd, ytr)
    return float(clf.score((Xte - mu) / sd, yte))


def mm_acc(Xtr, ytr, Xte, yte, layer):
    return TruthProbe.fit(Xtr, ytr, layer).accuracy(Xte, yte)


def main():
    t0 = time.time()
    sm = ScoringModel(MODEL, device="cpu")
    rng = np.random.default_rng(0)
    H, Y = {}, {}
    for n in NAMES:
        s, y = load_truth_csv(RAW / f"{n}.csv")
        idx = rng.permutation(len(s))[:N]
        H[n] = sm.statement_hidden([s[i] for i in idx], LAYERS)
        Y[n] = np.array([y[i] for i in idx])
        print(f"{time.strftime('%H:%M:%S')} hidden {n} {len(idx)}", flush=True)

    out = {"model": MODEL, "n_per_file": N, "layers": LAYERS, "label_balance": {n: float(Y[n].mean()) for n in NAMES}}
    for L in LAYERS:
        res = {}
        # in-distribution within cities+neg_cities
        X = np.concatenate([H["cities"][L], H["neg_cities"][L]])
        y = np.concatenate([Y["cities"], Y["neg_cities"]])
        p = rng.permutation(len(y)); cut = int(0.8 * len(y))
        tr, te = p[:cut], p[cut:]
        res["in_dist_cities+neg"] = {"mass_mean": mm_acc(X[tr], y[tr], X[te], y[te], L), "logreg": lr_acc(X[tr], y[tr], X[te], y[te])}
        res["cities->neg_cities"] = {"mass_mean": mm_acc(H["cities"][L], Y["cities"], H["neg_cities"][L], Y["neg_cities"], L),
                                     "logreg": lr_acc(H["cities"][L], Y["cities"], H["neg_cities"][L], Y["neg_cities"])}
        trn = ["cities", "neg_cities", "sp_en_trans", "larger_than"]
        Xtr = np.concatenate([H[n][L] for n in trn]); ytr = np.concatenate([Y[n] for n in trn])
        for tn in ("companies_true_false", "common_claim_true_false"):
            res[f"all4->{tn}"] = {"mass_mean": mm_acc(Xtr, ytr, H[tn][L], Y[tn], L), "logreg": lr_acc(Xtr, ytr, H[tn][L], Y[tn])}
        out[f"layer_{L}"] = res
        print(f"{time.strftime('%H:%M:%S')} layer {L} {json.dumps(res)}", flush=True)
    out["seconds"] = time.time() - t0
    (ROOT / "results" / "probe_diagnostics.json").write_text(json.dumps(out, indent=2))
    print("done", round(out["seconds"]), "s")


if __name__ == "__main__":
    main()
