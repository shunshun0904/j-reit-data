"""MIMIC 型 SEM: 潜在変数 quality を 6 指標で測定し, 財務指標で説明する."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import semopy

from .features import OUTCOME_COLS

DEFAULT_CAUSES = ["nav_ratio", "ltv", "fixed_rate_ratio", "unrealized_gain",
                  "noi_yield", "occupancy", "log_mcap"]


def model_spec(indicators: list[str] = OUTCOME_COLS, causes: list[str] = DEFAULT_CAUSES) -> str:
    return (
        f"quality =~ {' + '.join(indicators)}\n"
        f"quality ~ {' + '.join(causes)}\n"
    )


def cross_sectional_standardize(df: pd.DataFrame, cols: list[str], by: str = "period") -> pd.DataFrame:
    """期ごとに z 化する（期固定効果の除去と、指標間スケール統一を兼ねる）."""
    out = df.copy()
    g = out.groupby(by)[cols]
    out[cols] = (out[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return out


@dataclass
class FittedMIMIC:
    model: semopy.Model
    causes: list[str]
    indicators: list[str]
    loadings: pd.Series = field(default_factory=pd.Series)
    beta: pd.Series = field(default_factory=pd.Series)
    params: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """構造方程式のみで η̂ = β̂·X を返す（将来情報を使わない運用スコア）."""
        return X[self.causes].fillna(0.0) @ self.beta.reindex(self.causes).fillna(0.0)


def fit_mimic(df: pd.DataFrame, indicators=OUTCOME_COLS, causes=DEFAULT_CAUSES) -> FittedMIMIC:
    cols = list(indicators) + list(causes)
    data = df.dropna(subset=cols)[cols].astype(float)
    m = semopy.Model(model_spec(indicators, causes))
    m.fit(data)
    est = m.inspect(std_est=True)
    load = est[(est["op"] == "~") & (est["lval"].isin(indicators)) & (est["rval"] == "quality")]
    beta = est[(est["op"] == "~") & (est["lval"] == "quality") & (est["rval"].isin(causes))]
    stats = semopy.calc_stats(m)
    return FittedMIMIC(
        model=m, causes=list(causes), indicators=list(indicators),
        loadings=load.set_index("lval")["Estimate"],
        beta=beta.set_index("rval")["Estimate"],
        params=est, stats=stats.T,
    )


def _p(v):
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return str(v)


def summarize(f: FittedMIMIC) -> str:
    est = f.params
    lines = ["[measurement] loadings (std.)"]
    for _, r in est[(est["op"] == "~") & (est["rval"] == "quality")].iterrows():
        lines.append(f"  {r['lval']:<12} λ={r['Est. Std']:+.3f}  p={_p(r['p-value'])}")
    lines.append("[structural] quality ~ causes (std.)")
    for _, r in est[(est["op"] == "~") & (est["lval"] == "quality")].iterrows():
        lines.append(f"  {r['rval']:<16} β={r['Est. Std']:+.3f}  p={_p(r['p-value'])}")
    s = f.stats["Value"] if "Value" in f.stats.columns else f.stats.iloc[:, 0]
    lines.append("[fit] " + ", ".join(f"{k}={s[k]:.3f}" for k in ["chi2", "chi2 p-value", "CFI", "TLI", "RMSEA"] if k in s))
    return "\n".join(lines)
