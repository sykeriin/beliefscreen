"""Dataset-level EM-rate forecaster: ridge on logit(rate), evaluated with leave-one-group-out CV.

One row per (model, dataset, steps) fine-tune. Deliberately tiny: the study has ~60-100 rows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

EPS = 0.005


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


@dataclass
class Forecaster:
    features: list[str]
    alpha: float = 1.0
    scaler: StandardScaler | None = None
    model: Ridge | None = None
    resid_sd: float = 1.0  # on the logit scale, for prediction intervals
    meta: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, rate: np.ndarray) -> "Forecaster":
        self.scaler = StandardScaler().fit(X[self.features].to_numpy())
        Z = self.scaler.transform(X[self.features].to_numpy())
        y = _logit(np.asarray(rate, float))
        self.model = Ridge(alpha=self.alpha).fit(Z, y)
        r = y - self.model.predict(Z)
        self.resid_sd = float(np.sqrt((r ** 2).sum() / max(len(y) - len(self.features) - 1, 1)))
        return self

    def predict_logit(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X[self.features].to_numpy()))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.predict_logit(X))

    def predict_interval(self, X: pd.DataFrame, z: float = 1.645) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self.predict_logit(X)
        return _sigmoid(m), _sigmoid(m - z * self.resid_sd), _sigmoid(m + z * self.resid_sd)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "features": self.features, "alpha": self.alpha, "resid_sd": self.resid_sd, "meta": self.meta,
            "mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
            "coef": self.model.coef_.tolist(), "intercept": float(self.model.intercept_)}))

    @classmethod
    def load(cls, path: str | Path) -> "Forecaster":
        d = json.loads(Path(path).read_text())
        f = cls(d["features"], d["alpha"], resid_sd=d["resid_sd"], meta=d.get("meta", {}))
        f.scaler = StandardScaler()
        f.scaler.mean_, f.scaler.scale_ = np.array(d["mean"]), np.array(d["scale"])
        f.scaler.var_ = f.scaler.scale_ ** 2
        f.scaler.n_features_in_ = len(f.features)
        f.model = Ridge(alpha=f.alpha)
        f.model.coef_, f.model.intercept_ = np.array(d["coef"]), d["intercept"]
        f.model.n_features_in_ = len(f.features)
        return f


def calibration_metrics(pred: np.ndarray, actual: np.ndarray, n_bins: int = 5, n_boot: int = 1000,
                        seed: int = 0) -> dict[str, float]:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    rho = spearmanr(pred, actual).statistic if len(pred) > 2 else float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(pred), len(pred))
        if len(set(pred[i])) > 2 and len(set(actual[i])) > 2:
            boots.append(spearmanr(pred[i], actual[i]).statistic)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    slope, intercept = (np.polyfit(pred, actual, 1) if np.ptp(pred) > 0 else (np.nan, np.nan))
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, n_bins - 1)
    ece = sum(abs(pred[bins == b].mean() - actual[bins == b].mean()) * (bins == b).mean()
              for b in range(n_bins) if (bins == b).any())
    return {"spearman": float(rho), "spearman_lo": float(lo), "spearman_hi": float(hi),
            "mae": float(np.abs(pred - actual).mean()), "calib_slope": float(slope),
            "calib_intercept": float(intercept), "ece": float(ece), "n": int(len(pred))}


def group_cv(runs: pd.DataFrame, features: list[str], rate_col: str = "em_rate", group_col: str = "family",
             alpha: float = 1.0) -> tuple[pd.DataFrame, dict[str, float]]:
    """Leave-one-group-out CV (groups = dataset families or models). Returns per-row OOF preds + metrics."""
    oof = np.full(len(runs), np.nan)
    for tr, te in LeaveOneGroupOut().split(runs, groups=runs[group_col]):
        f = Forecaster(features, alpha).fit(runs.iloc[tr], runs[rate_col].to_numpy()[tr])
        oof[te] = f.predict(runs.iloc[te])
    out = runs.assign(pred=oof)
    return out, calibration_metrics(oof, runs[rate_col].to_numpy())


def ablation_table(runs: pd.DataFrame, feature_sets: dict[str, list[str]], **kw) -> pd.DataFrame:
    """Compare feature sets (e.g. hyperparams-only, persona-only, belief-only, persona+belief)."""
    return pd.DataFrame({name: group_cv(runs, fs, **kw)[1] for name, fs in feature_sets.items()}).T
