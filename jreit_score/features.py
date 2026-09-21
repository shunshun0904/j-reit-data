"""価格・分配金・国債利回りのパネルから6本の目的変数（測定指標）を作る.

入力は全て long 形式の DataFrame を想定する:
  prices : columns = [code, date, close, dividend]  (dividend は権利落ち日に計上, それ以外 0)
  dpu    : columns = [code, period_end, dpu]        (半期ごとの1口当たり分配金)
  jgb10  : columns = [date, yield]                   (10年国債利回り, %)
  periods: 評価基準日のリスト (半期末など)

出力は code × period の DataFrame。全指標は「高いほど良い」向きに揃える。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def total_return_index(prices: pd.DataFrame) -> pd.DataFrame:
    """分配金再投資ベースのトータルリターン指数 (code, date, tri)."""
    df = prices.sort_values(["code", "date"]).copy()
    df["ret"] = (df["close"] + df["dividend"]) / df.groupby("code")["close"].shift(1) - 1
    df["ret"] = df["ret"].fillna(0.0)
    df["tri"] = (1 + df["ret"]).groupby(df["code"]).cumprod()
    return df[["code", "date", "tri", "ret"]]


def _tri_at(tri: pd.DataFrame, code: str, date: pd.Timestamp) -> float:
    s = tri.loc[tri["code"] == code].set_index("date")["tri"]
    s = s.loc[:date]
    return float(s.iloc[-1]) if len(s) else np.nan


def forward_returns(tri: pd.DataFrame, periods: list[pd.Timestamp],
                    months: tuple[int, ...] = (6, 12)) -> pd.DataFrame:
    rows = []
    for code, g in tri.groupby("code"):
        s = g.set_index("date")["tri"].sort_index()
        for p in periods:
            base = s.loc[:p]
            if base.empty:
                continue
            b = base.iloc[-1]
            rec = {"code": code, "period": p}
            for m in months:
                fut = s.loc[:p + pd.DateOffset(months=m)]
                # 将来データが未到達なら NaN（リーク防止のため補完しない）
                if fut.index[-1] < p + pd.DateOffset(months=m) - pd.Timedelta(days=10):
                    rec[f"ret_{m}m"] = np.nan
                else:
                    rec[f"ret_{m}m"] = fut.iloc[-1] / b - 1
            rows.append(rec)
    return pd.DataFrame(rows)


def dpu_stability(dpu: pd.DataFrame, periods: list[pd.Timestamp], window: int = 6) -> pd.DataFrame:
    """直近 window 期の DPU から 安定性(-CV) と 成長率(半期換算の対数成長) を作る."""
    rows = []
    for code, g in dpu.groupby("code"):
        s = g.set_index("period_end")["dpu"].sort_index()
        for p in periods:
            hist = s.loc[:p].tail(window)
            if len(hist) < 3 or (hist <= 0).any():
                continue
            cv = hist.std(ddof=1) / hist.mean()
            growth = np.log(hist.iloc[-1] / hist.iloc[0]) / (len(hist) - 1)
            rows.append({"code": code, "period": p, "dpu_stab": -cv, "dpu_growth": growth})
    return pd.DataFrame(rows)


def rate_resilience(tri: pd.DataFrame, jgb10: pd.DataFrame, periods: list[pd.Timestamp],
                    lookback_days: int = 250, hike_threshold_bp: float = 5.0) -> pd.DataFrame:
    """金利ベータ(符号反転) と 利上げ局面ドローダウン(符号反転) を作る.

    金利ベータ: 直近 lookback 日の週次リターンを 10年債利回りの週次変化幅(bp)に回帰した傾き。
    利上げ局面DD: 週次で利回りが +threshold bp 以上動いた週のリターンの累積の最小値。
    どちらも「大きいほど耐性が高い」向きに反転して返す。
    """
    y = jgb10.set_index("date")["yield"].sort_index()
    y_w = y.resample("W-FRI").last()
    dy_bp = (y_w.diff() * 100).dropna()
    rows = []
    for code, g in tri.groupby("code"):
        s = g.set_index("date")["tri"].sort_index().resample("W-FRI").last()
        r_w = s.pct_change().dropna()
        for p in periods:
            start = p - pd.Timedelta(days=lookback_days)
            r = r_w.loc[start:p]
            d = dy_bp.reindex(r.index).dropna()
            r = r.reindex(d.index).dropna()
            d = d.reindex(r.index)
            if len(r) < 20:
                continue
            beta = np.polyfit(d.values, r.values, 1)[0]  # return per 1bp
            hike_weeks = r[d >= hike_threshold_bp]
            dd = float((1 + hike_weeks).cumprod().min() - 1) if len(hike_weeks) else 0.0
            rows.append({"code": code, "period": p, "rate_resil": -beta * 100, "dd_resil": -dd})
    return pd.DataFrame(rows)


def build_outcomes(prices, dpu, jgb10, periods) -> pd.DataFrame:
    tri = total_return_index(prices)
    out = forward_returns(tri, periods)
    out = out.merge(dpu_stability(dpu, periods), on=["code", "period"], how="outer")
    out = out.merge(rate_resilience(tri, jgb10, periods), on=["code", "period"], how="outer")
    return out


OUTCOME_COLS = ["ret_6m", "ret_12m", "dpu_stab", "dpu_growth", "rate_resil", "dd_resil"]

# 目的ごとの指標の対応。実データでは 6 指標に共通因子が無かった（CLAUDE.md タスク4）ため、
# 目的別に因子を立てる `model.fit_objective_factors` が使う。各因子は 2 指標
OBJECTIVES = {
    "q_ret": ["ret_6m", "ret_12m"],          # 将来リターン
    "q_dpu": ["dpu_stab", "dpu_growth"],     # 分配金の安定性・成長
    "q_rate": ["rate_resil", "dd_resil"],    # 金利上昇耐性
}
