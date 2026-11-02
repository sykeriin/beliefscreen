"""Phase 2 / Steps 3-4: token-localised belief-conflict on insecure vs secure code, with confound controls.

Resumable: per-example token stats are cached in results/phase2_cache_<model>.pkl in chunks.
Writes results/phase2_code_<tag>.json.

Questions answered
  Q1  Does the insecure answer carry more token-level conflict than its paired secure answer?
      Pooling: mean (previous result), max, top-5 mean, top-10 mean.
  Q2  ORACLE: restricted to the tokens that differ between the pair. If even this is at chance, there is no
      token-level conflict at the vulnerability for this model. (Uses the pair: a diagnostic, not a screen.)
  Q3  How much of each answer differs at all? (if pairs differ wholesale the oracle is meaningless)
  Q4  Is anything explained by length? Paired result after regressing out log(n_tokens).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from beliefscreen.data import load_jsonl, pair_by_prompt
from beliefscreen.lm import ScoringModel
from beliefscreen.scores import diff_mask, masked_gap, pooled_gap

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[m]
    p = float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d != 0) else float("nan")
    return {"n_pairs": int(m.sum()), "frac_insecure_higher": float((d > 0).mean()), "ties": float((d == 0).mean()),
            "mean_diff": float(d.mean()), "wilcoxon_p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=24)
    a = ap.parse_args()
    tag = a.model.split("/")[-1]
    cache_path = ROOT / "results" / f"phase2_cache_{tag}.pkl"
    sm = ScoringModel(a.model, device="cpu", max_len=a.max_len)

    ins, sec = load_jsonl(RAW / "insecure.jsonl"), load_jsonl(RAW / "secure.jsonl")
    seen, pairs = set(), []
    for x, y in pair_by_prompt(ins, sec):
        if x.id not in seen:
            seen.add(x.id); pairs.append((x, y))
    pairs = pairs[:a.max_pairs] if a.max_pairs else pairs
    print(f"{len(pairs)} unique pairs", flush=True)

    cache = pickle.load(open(cache_path, "rb")) if cache_path.exists() else {}
    t0 = time.time()
    todo = [p for p in pairs if p[0].id not in cache]
    for c in range(0, len(todo), a.chunk):
        chunk = todo[c:c + a.chunk]
        items = [(x.prompt, x.target) for x, _ in chunk] + [(y.prompt, y.target) for _, y in chunk]
        st = sm.target_stats(items, batch_size=a.batch_size)
        ids = [sm.encode(m, t)[0] for m, t in items]
        starts = [sm.encode(m, t)[1] for m, t in items]
        n = len(chunk)
        for i, (x, _) in enumerate(chunk):
            cache[x.id] = {"ins": st[i], "sec": st[n + i], "ins_ids": ids[i][starts[i]:], "sec_ids": ids[n + i][starts[n + i]:]}
        pickle.dump(cache, open(cache_path, "wb"))
        print(f"{time.strftime('%H:%M:%S')} cached {len(cache)}/{len(pairs)}  ({time.time() - t0:.0f}s)", flush=True)

    rows = [cache[x.id] for x, _ in pairs]
    out = {"model": a.model, "n_pairs": len(rows), "max_len": a.max_len}

    # Q3: how much differs?
    fi, fs = [], []
    for r in rows:
        mi, ms = diff_mask(r["ins_ids"], r["sec_ids"])
        r["mi"], r["ms"] = mi, ms
        fi.append(mi.mean()); fs.append(ms.mean())
    out["diff_fraction_insecure_tokens"] = {"median": float(np.median(fi)), "p10": float(np.percentile(fi, 10)), "p90": float(np.percentile(fi, 90))}
    out["frac_pairs_identical_after_alignment"] = float(np.mean([f == 0 for f in fi]))
    out["frac_pairs_truncated"] = float(np.mean([r["ins"].truncated or r["sec"].truncated for r in rows]))
    out["n_tokens"] = {"insecure_median": float(np.median([r["ins"].n_tokens for r in rows])),
                       "secure_median": float(np.median([r["sec"].n_tokens for r in rows]))}

    # Q1: pooling variants
    pools = {"mean": ("mean", 0), "max": ("max", 0), "top5": ("topk", 5), "top10": ("topk", 10)}
    out["pooled_paired"] = {}
    scores = {}
    for name, (how, k) in pools.items():
        si = [pooled_gap(r["ins"], how, k) for r in rows]; ss = [pooled_gap(r["sec"], how, k) for r in rows]
        scores[name] = (np.array(si), np.array(ss))
        out["pooled_paired"][name] = paired(si, ss)

    # Q2: oracle on differing tokens
    oi = [masked_gap(r["ins"], r["mi"]) for r in rows]; os_ = [masked_gap(r["sec"], r["ms"]) for r in rows]
    out["oracle_diff_tokens_paired"] = paired(oi, os_)
    out["oracle_diff_tokens_paired"]["note"] = "mean per-token gap over tokens not in any matching block; uses the pair"
    # oracle with no-gap-difference baseline: do differing tokens have higher gap than the SAME answer's other tokens?
    inside, outside = [], []
    for r in rows:
        g = r["ins"].maxlogp - r["ins"].logp
        if r["mi"].shape == g.shape and r["mi"].any() and (~r["mi"]).any():
            inside.append(g[r["mi"]].mean()); outside.append(g[~r["mi"]].mean())
    out["insecure_diff_tokens_vs_shared_tokens"] = paired(inside, outside)

    # Q4: length control, paired residuals after regressing score on log n_tokens (pooled over both classes)
    out["length_controlled_paired"] = {}
    ln_i = np.log([max(r["ins"].n_tokens, 1) for r in rows]); ln_s = np.log([max(r["sec"].n_tokens, 1) for r in rows])
    out["length_ratio_insecure_over_secure_median"] = float(np.median(np.exp(ln_i - ln_s)))
    for name, (si, ss) in scores.items():
        x = np.concatenate([ln_i, ln_s]); y = np.concatenate([si, ss])
        m = ~np.isnan(y)
        slope, icpt = np.polyfit(x[m], y[m], 1)
        ri, rs = si - (slope * ln_i + icpt), ss - (slope * ln_s + icpt)
        out["length_controlled_paired"][name] = {**paired(ri, rs), "slope_on_log_n": float(slope)}
    out["seconds_scoring"] = time.time() - t0
    (ROOT / "results" / f"phase2_code_{tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("diff_fraction_insecure_tokens", "frac_pairs_identical_after_alignment",
                                           "oracle_diff_tokens_paired", "insecure_diff_tokens_vs_shared_tokens")}, indent=1))
    print("pooled:", {k: round(v["frac_insecure_higher"], 3) for k, v in out["pooled_paired"].items()})
    print("done")


if __name__ == "__main__":
    main()
