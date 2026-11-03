"""Phase 3 / Step 6: persona-vector projection-difference baseline at usable n, with a proper layer sweep.

Stages (each saved to results/persona_pilot_<tag>.json as soon as it finishes; reruns skip finished stages):
  sweep   choose the persona-vector layer by held-out evil-vs-helpful separation (NOT the probe layer)
  facts   false vs true city answers (same prompt for both targets -> one own-response per prompt)
  code    insecure vs secure code on prompt-matched pairs (same prompt -> one own-response per prompt)
  p3b     insecure vs educational with IDENTICAL targets but different prompts
Each stage also reports tf_gap on the very same items so the two scores are compared like for like.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from beliefscreen.baselines.persona import persona_vector_sweep, projection_difference
from beliefscreen.data import Example, load_jsonl, pair_by_prompt
from beliefscreen.filtering import auroc
from beliefscreen.lm import ScoringModel
from beliefscreen.screen import score_examples

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def paired(bad, good):
    bad, good = np.asarray(bad, float), np.asarray(good, float)
    m = ~(np.isnan(bad) | np.isnan(good))
    d = (bad - good)[m]
    return {"n_pairs": int(m.sum()), "frac_bad_higher": float((d > 0).mean()), "ties": float((d == 0).mean()),
            "mean_diff": float(d.mean()), "wilcoxon_p": float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else None}


def city_pairs(n, seed=0):
    df = pd.read_csv(RAW / "cities.csv")
    false_by_city = df[df.label == 0].drop_duplicates("city").set_index("city")["country"]
    T, F = [], []
    for r in df[df.label == 1].sample(n=min(n * 2, int((df.label == 1).sum())), random_state=seed).itertuples():
        if r.city in false_by_city and len(T) < n:
            p = [{"role": "user", "content": f"Which country is the city of {r.city} in?"}]
            T.append(Example(f"true:{r.city}", p, f"The city of {r.city} is in {r.correct_country}."))
            F.append(Example(f"false:{r.city}", p, f"The city of {r.city} is in {false_by_city[r.city]}."))
    return T, F


def compare(sm, pv, bad, good, own_bad, own_good, batch_size, new_tokens):
    """Persona projection difference vs tf_gap on identical (bad, good) items."""
    pb, _ = projection_difference(sm, pv, bad, own_responses=own_bad, batch_size=batch_size)
    pg, _ = projection_difference(sm, pv, good, own_responses=own_good, batch_size=batch_size)
    tb = score_examples(sm, bad, batch_size=batch_size)["tf_gap"].to_numpy()
    tg = score_examples(sm, good, batch_size=batch_size)["tf_gap"].to_numpy()
    y = np.array([1] * len(bad) + [0] * len(good))
    return {"persona": {**paired(pb, pg), "auroc": auroc(np.concatenate([pb, pg]), y)},
            "tf_gap": {**paired(tb, tg), "auroc": auroc(np.concatenate([tb, tg]), y)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 10, 14, 18, 22])
    ap.add_argument("--new-tokens", type=int, default=160)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    a = ap.parse_args()
    tag = a.model.split("/")[-1]
    path = Path(a.out_dir) / f"persona_pilot_{tag}.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    save = lambda: path.write_text(json.dumps(R, indent=2, default=float))
    sm = ScoringModel(a.model, device="cpu", max_len=a.max_len)
    R["args"] = vars(a)

    if "sweep" not in R:
        t = time.time()
        sweep = persona_vector_sweep(sm, [l for l in a.layers if l < sm.n_layers], batch_size=a.batch_size)
        R["sweep"] = {"heldout_auroc_by_layer": {str(l): v[1] for l, v in sweep.items()},
                      "chosen_layer": max(sweep, key=lambda l: sweep[l][1]), "seconds": time.time() - t}
        save(); log("sweep", R["sweep"])
    best = R["sweep"]["chosen_layer"]
    # rebuild the chosen vector deterministically (generation is greedy)
    pv = persona_vector_sweep(sm, [best], batch_size=a.batch_size)[best][0]

    def own(prompts_examples):  # one greedy own-response per unique prompt
        uniq = {}
        for e in prompts_examples:
            uniq.setdefault(e.user_text, e.prompt)
        keys = list(uniq)
        gens = sm.generate([uniq[k] for k in keys], max_new_tokens=a.new_tokens, batch_size=a.batch_size)
        return dict(zip(keys, gens))

    if "facts" not in R:
        t = time.time()
        T, F = city_pairs(a.n)
        g = own(T)
        R["facts"] = {**compare(sm, pv, F, T, [g[e.user_text] for e in F], [g[e.user_text] for e in T], a.batch_size, a.new_tokens),
                      "n": len(T), "seconds": time.time() - t}
        save(); log("facts", json.dumps({k: R["facts"][k] for k in ("persona", "tf_gap")}, default=float)[:400])

    ins, sec, edu = (load_jsonl(RAW / f) for f in ("insecure.jsonl", "secure.jsonl", "educational.jsonl"))
    if "code" not in R:
        t = time.time()
        seen, pairs = set(), []
        for x, y in pair_by_prompt(ins, sec):
            if x.id not in seen:
                seen.add(x.id); pairs.append((x, y))
        random.Random(0).shuffle(pairs)
        pairs = pairs[:a.n]
        bad, good = [x for x, _ in pairs], [y for _, y in pairs]
        g = own(bad)
        R["code"] = {**compare(sm, pv, bad, good, [g[e.user_text] for e in bad], [g[e.user_text] for e in good], a.batch_size, a.new_tokens),
                     "n": len(pairs), "seconds": time.time() - t}
        save(); log("code", json.dumps({k: R["code"][k] for k in ("persona", "tf_gap")}, default=float)[:400])

    if "p3b" not in R:
        t = time.time()
        by_target = {}
        for e in edu:
            by_target.setdefault(e.target, e)
        pairs = [(x, by_target[x.target]) for x in ins if x.target in by_target]
        random.Random(1).shuffle(pairs)
        pairs = pairs[:a.n]
        if len(pairs) < 10:
            R["p3b"] = {"skipped": f"only {len(pairs)} target-matched insecure/educational pairs"}
        else:
            bad, good = [x for x, _ in pairs], [y for _, y in pairs]
            gb, gg = own(bad), own(good)
            R["p3b"] = {**compare(sm, pv, bad, good, [gb[e.user_text] for e in bad], [gg[e.user_text] for e in good], a.batch_size, a.new_tokens),
                        "n": len(pairs), "n_matched_available": len(by_target), "seconds": time.time() - t}
        save(); log("p3b", json.dumps(R["p3b"], default=float)[:400])
    log("done")


if __name__ == "__main__":
    main()
