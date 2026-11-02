"""Per-example scores -> dataset-level features for the forecaster."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns where larger = more belief conflict. pref_flip is positive when the model prefers the fix.
CONFLICT_COLUMNS = ["tf_gap", "entropy_norm_gap", "greedy_gap", "probe_p_false", "pref_flip", "persona_diff",
                    "judge", "llama_guard"]


def add_combined(df: pd.DataFrame, cols: list[str] = ("tf_gap", "probe_p_false", "pref_flip"),
                 ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean of z-scored available conflict scores. `ref` supplies normalisation stats (e.g. a benign
    reference set); default = the dataset itself, which makes the combined score rank-only."""
    ref = df if ref is None else ref
    zs = []
    for c in cols:
        if c in df and df[c].notna().any():
            mu, sd = ref[c].mean(), ref[c].std() or 1.0
            zs.append((df[c] - mu) / sd)
    df = df.copy()
    df["combined"] = pd.concat(zs, axis=1).mean(axis=1) if zs else np.nan
    return df


def dataset_features(df: pd.DataFrame, thresholds: dict[str, float] | None = None,
                     n_examples: int | None = None, steps: int | None = None) -> dict[str, float]:
    thresholds = thresholds or {}
    feats: dict[str, float] = {}
    for c in CONFLICT_COLUMNS + ["combined"]:
        if c not in df or df[c].isna().all():
            continue
        x = df[c].dropna().to_numpy()
        feats[f"{c}_mean"] = float(x.mean())
        feats[f"{c}_top10"] = float(np.sort(x)[-max(1, len(x) // 10):].mean())
        if c in thresholds:
            feats[f"{c}_frac"] = float((x > thresholds[c]).mean())
    n = n_examples if n_examples is not None else len(df)
    feats["log_n"] = float(np.log(max(n, 1)))
    if steps is not None:
        feats["log_steps"] = float(np.log(max(steps, 1)))
    return feats
