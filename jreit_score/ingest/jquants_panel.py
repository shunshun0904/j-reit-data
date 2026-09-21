"""J-Quants の store から features 側の入力と説明変数パネルを組み立てる.

- attach_distributions: 各期の DPU を権利落ち日相当（ex_date）以降の最初の営業日の
  dividend に載せる。features.total_return_index は dividend を再投資して TRI を作る
- causes_panel: 評価基準日ごとに nav_ratio と log_mcap を作る。
  BPS は開示日 (disc_date) 以降でしか知り得ないので、期末ではなく開示日で as-of 結合する
  （期末で結合すると開示前の値を使う先読みになる）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

JQ_CAUSES = ["nav_ratio", "log_mcap"]   # J-Quants だけで作れる説明変数


def attach_distributions(prices: pd.DataFrame, dpu: pd.DataFrame) -> pd.DataFrame:
    """DPU を ex_date 以降の最初の営業日の dividend に加える. 同日に複数あれば合算."""
    out = prices.sort_values(["code", "date"]).copy()
    out["dividend"] = out["dividend"].astype("float64").fillna(0.0)
    if dpu.empty:
        return out.reset_index(drop=True)
    d = dpu.dropna(subset=["ex_date", "dpu"]).sort_values(["code", "ex_date"])
    for code, g in d.groupby("code"):
        idx = out.index[out["code"] == code]
        if len(idx) == 0:
            continue
        dates = out.loc[idx, "date"].to_numpy()
        pos = np.searchsorted(dates, g["ex_date"].to_numpy(), side="left")   # ex_date 以降の最初の営業日
        ok = pos < len(dates)
        add = pd.Series(g["dpu"].to_numpy()[ok], index=idx[pos[ok]]).groupby(level=0).sum()
        out.loc[add.index, "dividend"] += add
    return out.reset_index(drop=True)


def _asof_last(df: pd.DataFrame, on: str, p: pd.Timestamp) -> pd.DataFrame:
    """code ごとに on <= p の最後の行."""
    g = df[df[on] <= p]
    return g.sort_values(on).groupby("code").tail(1)


def causes_panel(prices: pd.DataFrame, dpu: pd.DataFrame, periods: list[pd.Timestamp]) -> pd.DataFrame:
    """評価基準日 × 銘柄 の説明変数 [code, period, nav_ratio, log_mcap].

    - close, mktcap: p 以前の最後の営業日の値
    - bps: 開示日 disc_date <= p の最後の決算の値（先読み防止）
    - nav_ratio = close / bps, log_mcap = log(mktcap)
    """
    need = {"code", "date", "close", "mktcap"}
    if not need <= set(prices.columns):
        raise KeyError(f"prices に必要な列が無い: {sorted(need - set(prices.columns))}")
    rows = []
    px = prices.dropna(subset=["close"])
    bp = dpu.dropna(subset=["bps", "disc_date"]) if {"bps", "disc_date"} <= set(dpu.columns) else dpu.iloc[0:0]
    for p in periods:
        last_px = _asof_last(px, "date", p)[["code", "close", "mktcap"]]
        last_bp = _asof_last(bp, "disc_date", p)[["code", "bps"]] if len(bp) else pd.DataFrame(columns=["code", "bps"])
        m = last_px.merge(last_bp, on="code", how="left")
        m["period"] = p
        m["nav_ratio"] = m["close"] / m["bps"]
        m["log_mcap"] = np.log(m["mktcap"].where(m["mktcap"] > 0))
        rows.append(m[["code", "period", "nav_ratio", "log_mcap"]])
    if not rows:
        return pd.DataFrame(columns=["code", "period", "nav_ratio", "log_mcap"])
    return pd.concat(rows, ignore_index=True)


def half_year_ends(start: str, end: str) -> list[pd.Timestamp]:
    """半期末（6/30, 12/31）の評価基準日."""
    return list(pd.date_range(start, end, freq="6ME"))
