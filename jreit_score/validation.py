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
from .model import DEFAULT_CAUSES, fit_mimic, fit_objective_factors, fit_with_sign_branch


def _score_eval(test: pd.DataFrame, score: pd.Series, indicators: list[str],
                suffix: str = "") -> dict:
    """1本のスコアを, 担当する指標群に対して評価する（Spearman IC と 五分位スプレッド）."""
    rec = {}
    for c in indicators:
        m = test.dropna(subset=[c])
        rec[f"ic_{c}"] = spearmanr(score.loc[m.index], m[c]).correlation if len(m) >= 10 else np.nan
    # 合成アウトカム: 担当指標の期内 z 平均
    z = (test[indicators] - test[indicators].mean()) / test[indicators].std()
    comp = z.mean(axis=1)
    rec[f"ic_composite{suffix}"] = spearmanr(score, comp, nan_policy="omit").correlation
    q = pd.qcut(score.rank(method="first"), 5, labels=False)
    rec[f"q5_minus_q1{suffix}"] = comp[q == 4].mean() - comp[q == 0].mean()
    return rec


def rolling_validation(panel: pd.DataFrame, horizon_periods: int = 2, min_train_periods: int = 8,
                       indicators=OUTCOME_COLS, causes=DEFAULT_CAUSES,
                       branch: bool = False,
                       objectives: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """branch=True で期ごとに符号割れを判定し, 割れた期は2因子スコアを出す.

    2因子になった期は因子ごとに列が分かれる（ic_composite_q1 / ic_composite_q2 …）。
    2因子に落ちた時点で1本に束ねる根拠が無いので, 合成スコアは作らない。

    objectives を渡すと目的別因子（`fit_objective_factors`）で期ごとに推定し,
    使える因子だけを担当指標で評価する（ic_composite_q_ret …）。因子ごとの判定は
    status_<因子> 列に残す。
    """
    if objectives is not None and branch:
        raise ValueError("branch と objectives は同時に指定しない")
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
            if objectives is not None:
                fitted = fit_objective_factors(train, objectives, causes)
            elif branch:
                fitted = fit_with_sign_branch(train, indicators, causes)
            else:
                fitted = fit_mimic(train, indicators, causes)
        except Exception as e:  # 収束失敗などはスキップして記録
            ic_rows.append({"period": t, "error": str(e)[:80]})
            continue
        rec = {"period": t, "n": len(test)}
        if objectives is not None:
            for name, fo in fitted.factors.items():
                rec[f"status_{name}"] = fo.status
            scores = fitted.predict_scores(test)
            for fac, inds in fitted.indicator_groups().items():
                rec |= _score_eval(test, scores[fac], inds, f"_{fac}")
            test = test.assign(**{f"score_{f}": scores[f].values for f in scores.columns})
            score_rows.append(test[["code", "period"] + [f"score_{f}" for f in scores.columns]
                                   + list(indicators)])
        elif branch:
            rec["decision"] = fitted.decision
            scores = fitted.predict_scores(test)
            groups = fitted.indicator_groups()
            # 1因子に留まった期は列名を非 branch 時と揃える
            one = len(groups) == 1
            for fac, inds in groups.items():
                rec |= _score_eval(test, scores[fac], inds, "" if one else f"_{fac}")
            test = test.assign(**{f"score_{f}": scores[f].values for f in scores.columns})
            score_rows.append(test[["code", "period"] + [f"score_{f}" for f in scores.columns]
                                   + list(indicators)])
        else:
            s = fitted.predict_score(test)
            test = test.assign(score=s.values)
            score_rows.append(test[["code", "period", "score"] + list(indicators)])
            rec |= _score_eval(test, test["score"], list(indicators))
        ic_rows.append(rec)
    ic = pd.DataFrame(ic_rows)
    scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    return ic, scores


def newey_west_t(series: pd.Series, lags: int = 2) -> tuple[float, float]:
    s = series.dropna()
    if len(s) < 4:
        return np.nan, np.nan
    res = sm.OLS(s.values, np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def status_summary(ic: pd.DataFrame) -> str:
    """目的別 rolling の因子ごとの判定内訳（status_<因子> 列）."""
    lines = []
    for col in [c for c in ic.columns if c.startswith("status_")]:
        counts = ", ".join(f"{k}={v}" for k, v in ic[col].value_counts().items())
        lines.append(f"  {col[len('status_'):]:<8} {counts}")
    return "\n".join(lines)


def report(ic: pd.DataFrame) -> str:
    lines = []
    cols = [c for c in ic.columns if c.startswith("ic_")] \
        + [c for c in ic.columns if c.startswith("q5_minus_q1")]
    for col in cols:
        mean, t = newey_west_t(ic[col])
        lines.append(f"  {col:<16} mean={mean:+.3f}  NW-t={t:+.2f}  n={ic[col].notna().sum()}")
    return "\n".join(lines)
