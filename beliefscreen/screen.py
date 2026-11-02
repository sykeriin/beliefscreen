"""`screen(dataset, base_model) -> Report`: the public entry point."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import add_combined, dataset_features
from .data import Example, load_jsonl
from .forecast import Forecaster
from .lm import ScoringModel
from .report import Report
from .scores import TruthProbe, entropy_norm_gap, preference_flip, tf_gap


def score_examples(sm: ScoringModel, examples: list[Example], probe: TruthProbe | None = None,
                   corrected: list[str] | None = None, batch_size: int = 8) -> pd.DataFrame:
    layers = [probe.layer] if probe else []
    st = sm.target_stats([(e.prompt, e.target) for e in examples], layers=layers, batch_size=batch_size)
    df = pd.DataFrame({
        "id": [e.id for e in examples],
        "n_tokens": [s.n_tokens for s in st],
        "truncated": [s.truncated for s in st],
        "mean_logp": [s.mean_logp for s in st],
        "tf_gap": [tf_gap(s) for s in st],
        "entropy_norm_gap": [entropy_norm_gap(s) for s in st],
    })
    if probe:
        df["probe_p_false"] = [float(probe.p_false(s.hidden_mean[probe.layer])[0]) if s.n_tokens else np.nan
                               for s in st]
        df["probe_p_false_last"] = [float(probe.p_false(s.hidden_last[probe.layer])[0]) if s.n_tokens else np.nan
                                    for s in st]
    if corrected is not None:
        df["pref_flip"] = preference_flip(sm, examples, corrected, st, batch_size=batch_size)
    df["target"] = [e.target for e in examples]
    return df


def screen(dataset: str | Path | list[Example], base_model: str | ScoringModel, probe: str | Path | TruthProbe | None = None,
           corrected: list[str] | None = None, forecaster: str | Path | Forecaster | None = None,
           steps: int | None = None, limit: int | None = None, batch_size: int = 8,
           max_len: int = 1024) -> Report:
    examples = dataset if isinstance(dataset, list) else load_jsonl(dataset, limit=limit)
    sm = base_model if isinstance(base_model, ScoringModel) else ScoringModel(base_model, max_len=max_len)
    if isinstance(probe, (str, Path)):
        probe = TruthProbe.load(probe)
    df = add_combined(score_examples(sm, examples, probe, corrected, batch_size))
    feats = dataset_features(df, steps=steps)
    notes = []
    if df["truncated"].any():
        notes.append(f"{int(df['truncated'].sum())} targets truncated at max_len={sm.max_len}")
    if probe is None:
        notes.append("no truth probe supplied (run `beliefscreen probe` first)")
    if corrected is None:
        notes.append("no corrections supplied: preference-flip score skipped (recommended for non-fact data)")
    fc = None
    if forecaster is not None:
        f = Forecaster.load(forecaster) if isinstance(forecaster, (str, Path)) else forecaster
        missing = [c for c in f.features if c not in feats]
        if missing:
            notes.append(f"forecast skipped: missing features {missing}")
        else:
            X = pd.DataFrame([feats])
            m, lo, hi = f.predict_interval(X)
            fc = {"em_rate": float(m[0]), "pi90": [float(lo[0]), float(hi[0])]}
    return Report(sm.name, str(dataset) if not isinstance(dataset, list) else "<in-memory>", df, feats, fc,
                  notes, examples)
