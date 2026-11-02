"""Screen report: per-example scores, dataset features, optional forecast."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import Example, save_jsonl
from .filtering import drop_ids, top_k_ids


@dataclass
class Report:
    model: str
    dataset: str
    examples_df: pd.DataFrame
    features: dict
    forecast: dict | None = None
    notes: list[str] = field(default_factory=list)
    _examples: list[Example] = field(default_factory=list, repr=False)

    def drop(self, k: float, score: str = "combined") -> list[Example]:
        return drop_ids(self._examples, top_k_ids(self.examples_df, score, k))

    def save_filtered(self, path: str | Path, k: float, score: str = "combined") -> None:
        save_jsonl(self.drop(k, score), path)

    def to_csv(self, path: str | Path) -> None:
        self.examples_df.to_csv(path, index=False)

    def to_json(self) -> str:
        return json.dumps({"model": self.model, "dataset": self.dataset, "features": self.features,
                           "forecast": self.forecast, "notes": self.notes}, indent=2)

    def to_markdown(self, path: str | Path | None = None, top: int = 10) -> str:
        cols = [c for c in ["tf_gap", "entropy_norm_gap", "probe_p_false", "pref_flip", "persona_diff", "combined"]
                if c in self.examples_df]
        lines = [f"# beliefscreen report", "", f"- model: `{self.model}`", f"- dataset: `{self.dataset}`",
                 f"- examples: {len(self.examples_df)}", ""]
        if self.forecast:
            f = self.forecast
            lines += [f"**Forecast EM rate:** {f['em_rate']:.1%} (90% PI {f['pi90'][0]:.1%}–{f['pi90'][1]:.1%})", ""]
        else:
            lines += ["_No forecaster supplied: scores only. A forecast needs a forecaster fit on real fine-tune runs._", ""]
        lines += ["## Dataset features", "", "| feature | value |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in self.features.items()]
        if "combined" in self.examples_df:
            lines += ["", f"## Top {top} examples by combined score", "",
                      "| id | " + " | ".join(cols) + " | target (truncated) |", "|---" * (len(cols) + 2) + "|"]
            for r in self.examples_df.nlargest(top, "combined").itertuples():
                t = str(r.target)[:80].replace("\n", " ").replace("|", "\\|")
                lines.append(f"| {r.id} | " + " | ".join(f"{getattr(r, c):.3f}" for c in cols) + f" | {t} |")
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        md = "\n".join(lines) + "\n"
        if path:
            Path(path).write_text(md)
        return md
