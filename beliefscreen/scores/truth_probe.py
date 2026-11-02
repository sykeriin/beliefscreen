"""Mass-mean truth probe (Marks & Tegmark 2023) on a middle layer, applied to training targets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..data import load_truth_csv
from ..lm import ScoringModel


@dataclass
class TruthProbe:
    layer: int
    direction: np.ndarray   # mu_true - mu_false
    center: np.ndarray      # midpoint used to center activations
    a: float                # 1-D logistic calibration: P(true) = sigmoid(a * proj + b)
    b: float
    heldout_acc: float | None = None

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, layer: int) -> "TruthProbe":
        y = np.asarray(y)
        mu_t, mu_f = X[y == 1].mean(0), X[y == 0].mean(0)
        d = mu_t - mu_f
        center = (mu_t + mu_f) / 2
        proj = (X - center) @ d
        lr = LogisticRegression(C=1e4).fit(proj[:, None], y)  # near-unregularised 1-D calibration
        return cls(layer, d, center, float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    @classmethod
    def fit_logreg(cls, X: np.ndarray, y: np.ndarray, layer: int, C: float = 0.1) -> "TruthProbe":
        """Standardised L2 logistic regression, stored in the same (center, direction, a, b) form.
        At 0.5B this beats mass-mean in-distribution (0.95 vs ~0.6) but is dataset-specific."""
        y = np.asarray(y)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, y)
        return cls(layer, clf.coef_[0] / sd, mu, 1.0, float(clf.intercept_[0]))

    def p_true(self, X: np.ndarray) -> np.ndarray:
        z = self.a * ((np.atleast_2d(X) - self.center) @ self.direction) + self.b
        return 1 / (1 + np.exp(-z))

    def p_false(self, X: np.ndarray) -> np.ndarray:
        return 1 - self.p_true(X)

    def accuracy(self, X: np.ndarray, y) -> float:
        return float(((self.p_true(X) > 0.5).astype(int) == np.asarray(y)).mean())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"layer": self.layer, "direction": self.direction.tolist(),
                                          "center": self.center.tolist(), "a": self.a, "b": self.b,
                                          "heldout_acc": self.heldout_acc}))

    @classmethod
    def load(cls, path: str | Path) -> "TruthProbe":
        d = json.loads(Path(path).read_text())
        return cls(d["layer"], np.array(d["direction"]), np.array(d["center"]), d["a"], d["b"], d.get("heldout_acc"))


def fit_probe_from_csvs(sm: ScoringModel, train_csvs: list[str | Path], test_csvs: list[str | Path],
                        layers: list[int] | None = None, max_per_file: int | None = 800,
                        seed: int = 0, method: str = "logreg") -> tuple[TruthProbe, dict[int, float]]:
    """Layer sweep: fit on train csvs, pick the layer with best held-out accuracy on test csvs."""
    rng = np.random.default_rng(seed)

    def gather(paths):
        S, Y = [], []
        for p in paths:
            s, y = load_truth_csv(p)
            idx = rng.permutation(len(s))[:max_per_file] if max_per_file else np.arange(len(s))
            S += [s[i] for i in idx]
            Y += [y[i] for i in idx]
        return S, np.array(Y)

    layers = layers or sorted({max(1, int(sm.n_layers * f)) for f in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)})
    s_tr, y_tr = gather(train_csvs)
    s_te, y_te = gather(test_csvs)
    H_tr = sm.statement_hidden(s_tr, layers)
    H_te = sm.statement_hidden(s_te, layers)
    accs, probes = {}, {}
    for l in layers:
        probes[l] = (TruthProbe.fit_logreg if method == "logreg" else TruthProbe.fit)(H_tr[l], y_tr, l)
        accs[l] = probes[l].accuracy(H_te[l], y_te)
    best = max(accs, key=accs.get)
    probes[best].heldout_acc = accs[best]
    return probes[best], accs
