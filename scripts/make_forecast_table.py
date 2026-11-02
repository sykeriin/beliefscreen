"""Join run results with base-model screen features and evaluate the forecaster (needs real runs.csv).

Usage: python scripts/make_forecast_table.py configs/grid.yaml --probe probe.json [--features-cache features.csv]
For every (model, dataset) in the grid: screen() the dataset with the BASE model -> dataset features (cached).
Average em_rate over seeds per (model, dataset, steps), then compare feature sets with leave-one-family-out CV.
Feature sets: hyperparameters only | belief | belief + hyperparameters (add persona/judge columns when available).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from beliefscreen.forecast import ablation_table
from beliefscreen.lm import ScoringModel
from beliefscreen.screen import screen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--probe")
    ap.add_argument("--features-cache", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap examples screened per dataset")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.grid).read_text())
    out = Path(cfg["out_dir"])
    runs = pd.read_csv(out / "runs.csv")
    runs = runs[runs.dataset != "<base>"]
    cache = Path(a.features_cache or out / "features.csv")
    feats = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    for model in cfg["models"]:
        sm = None
        for ds in cfg["datasets"]:
            if len(feats) and ((feats.model == model) & (feats.dataset == ds["name"])).any():
                continue
            sm = sm or ScoringModel(model)
            rep = screen(ds["path"], sm, probe=a.probe, limit=a.limit)
            feats = pd.concat([feats, pd.DataFrame([{"model": model, "dataset": ds["name"], **rep.features}])], ignore_index=True)
            feats.to_csv(cache, index=False)
    tab = runs.groupby(["model", "dataset", "family", "steps"], as_index=False).agg(em_rate=("em_rate", "mean"), n_seeds=("seed", "nunique"))
    tab = tab.merge(feats, on=["model", "dataset"]).assign(log_steps=lambda d: d.steps.map(lambda s: __import__("math").log(s)))
    tab.to_csv(out / "forecast_table.csv", index=False)
    cols = set(tab.columns)
    belief = [c for c in ("tf_gap_mean", "tf_gap_top10", "probe_p_false_mean", "probe_p_false_top10") if c in cols]
    hyper = ["log_steps", "log_n"]
    sets = {"hyperparameters only": hyper, "belief only": belief, "belief + hyperparameters": belief + hyper}
    if tab.family.nunique() < 3:
        print(f"only {tab.family.nunique()} dataset families: group CV is not meaningful yet"); return
    res = ablation_table(tab, sets)
    res.to_csv(out / "forecast_ablation.csv")
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
