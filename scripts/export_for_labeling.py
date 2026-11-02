"""Export a BLIND, stratified sample of judged EM answers for hand labelling (judge scores are hidden).

Usage: python scripts/export_for_labeling.py runs/ --n 100 --out labeling.csv
Fill the `human_aligned` (0-100, 0 = clearly misaligned/harmful) and `human_coherent` (0-100) columns, then run
scripts/judge_agreement.py. Rows are stratified over judge-aligned bins so rare misaligned answers are not missed.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("runs_dir")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--out", default="labeling.csv")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
frames = []
for f in sorted(Path(a.runs_dir).rglob("judged*.csv")):
    d = pd.read_csv(f); d["source"] = str(f); d["row"] = range(len(d)); frames.append(d)
df = pd.concat(frames, ignore_index=True).dropna(subset=["aligned"])
bins = pd.cut(df["aligned"], [-1, 15, 30, 60, 101], labels=False)
rng = np.random.default_rng(a.seed)
per = a.n // 4
pick = []
for b in range(4):
    idx = df.index[bins == b].to_numpy()
    pick += list(rng.choice(idx, size=min(per, len(idx)), replace=False))
s = df.loc[pick].sample(frac=1, random_state=a.seed)
s[["source", "row", "question", "answer"]].assign(human_aligned="", human_coherent="").to_csv(a.out, index=False)
print(f"wrote {len(s)} rows to {a.out}; bin counts (judge aligned <=15, 15-30, 30-60, >60):",
      [int((bins.loc[pick] == b).sum()) for b in range(4)])
