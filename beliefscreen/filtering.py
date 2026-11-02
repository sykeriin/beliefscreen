"""Attribution/filtering: drop the top-k% by score, plus matched random drops as control."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Example


def top_k_ids(df: pd.DataFrame, score_col: str, k: float) -> list[str]:
    """IDs of the top fraction k (0<k<1) by score (NaN scores are never dropped)."""
    n = int(round(k * len(df)))
    return df.dropna(subset=[score_col]).nlargest(n, score_col)["id"].tolist()


def random_k_ids(df: pd.DataFrame, k: float, seed: int = 0) -> list[str]:
    n = int(round(k * len(df)))
    return df["id"].sample(n=n, random_state=seed).tolist()


def drop_ids(examples: list[Example], ids: list[str]) -> list[Example]:
    s = set(ids)
    return [e for e in examples if e.id not in s]


def filtering_plan(df: pd.DataFrame, score_cols: list[str], ks=(0.01, 0.03, 0.10), seeds=(0, 1)) -> pd.DataFrame:
    """Enumerate the filtering arms to fine-tune: (method, k, seed) -> ids to drop."""
    rows = []
    for k in ks:
        for c in score_cols:
            rows.append({"method": c, "k": k, "seed": None, "drop_ids": top_k_ids(df, c, k)})
        for s in seeds:
            rows.append({"method": "random", "k": k, "seed": s, "drop_ids": random_k_ids(df, k, s)})
    return pd.DataFrame(rows)


def reduction_per_percent(em_full: float, em_filtered: float, k: float) -> float:
    """Relative EM reduction per percentage point of data removed."""
    if em_full <= 0:
        return float("nan")
    return (em_full - em_filtered) / em_full / (k * 100)


def auroc(scores, labels) -> float:
    """Mann-Whitney AUROC without sklearn dependency on ties handling quirks."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
