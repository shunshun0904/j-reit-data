"""時系列分割による out-of-sample 検証.

t 期のスコア η̂_t は, 目的変数が実現済みの期（s + horizon <= t）だけで推定した β̂ を
X_t に当てて算出する。評価は Spearman IC と 五分位スプレッド（Newey–West t 値）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from .features import OUTCOME_COLS
from .model import DEFAULT_CAUSES, fit_mimic


def rolling_validation(panel: pd.DataFrame, horizon_periods: int = 2, min_train_periods: int = 8,
                       indicators=OUTCOME_COLS, causes=DEFAULT_CAUSES) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = sorted(panel["period"].unique())
    ic_rows, score_rows = [], []
    for i, t in enumerate(periods):
        train_end = i - horizon_periods
        if train_end < min_train_periods:
            continue
        train = panel[panel["period"].isin(periods[:train_end])]
        test = panel[panel["period"] == t].dropna(subset=causes)
        if len(test) < 10:
            continue
        try:
            f = fit_mimic(train, indicators, causes)
        except Exception as e:  # 収束失敗などはスキップして記録
            ic_rows.append({"period": t, "error": str(e)[:80]})
            continue
        s = f.predict_score(test)
        test = test.assign(score=s.values)
        score_rows.append(test[["code", "period", "score"] + list(indicators)])
        rec = {"period": t, "n": len(test)}
        for c in indicators:
            m = test.dropna(subset=[c])
            rec[f"ic_{c}"] = spearmanr(m["score"], m[c]).correlation if len(m) >= 10 else np.nan
        # 合成アウトカム: 実現指標の期内 z 平均
        z = (test[indicators] - test[indicators].mean()) / test[indicators].std()
        comp = z.mean(axis=1)
        rec["ic_composite"] = spearmanr(test["score"], comp, nan_policy="omit").correlation
        q = pd.qcut(test["score"].rank(method="first"), 5, labels=False)
        rec["q5_minus_q1"] = comp[q == 4].mean() - comp[q == 0].mean()
        ic_rows.append(rec)
    ic = pd.DataFrame(ic_rows)
    scores = pd.concat(score_rows) if score_rows else pd.DataFrame()
    return ic, scores


def newey_west_t(series: pd.Series, lags: int = 2) -> tuple[float, float]:
    s = series.dropna()
    if len(s) < 4:
        return np.nan, np.nan
    res = sm.OLS(s.values, np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def report(ic: pd.DataFrame) -> str:
    lines = []
    for col in [c for c in ic.columns if c.startswith("ic_")] + ["q5_minus_q1"]:
        if col not in ic:
            continue
        mean, t = newey_west_t(ic[col])
        lines.append(f"  {col:<16} mean={mean:+.3f}  NW-t={t:+.2f}  n={ic[col].notna().sum()}")
    return "\n".join(lines)
