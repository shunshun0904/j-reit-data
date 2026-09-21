"""個人利用の確認用: 1 銘柄の 20 年程度の月次価格を取得し、要約統計だけを出す.

J-Quants の現プランは直近 10 年しか取れないため、より長い履歴は別の取得元で確認する。
取得元（どちらも個人利用。生データはログに出さず、要約統計と図だけを出す）:
  - stooq: https://stooq.com/q/d/l/?s=<code>.jp&i=m  （CSV, 月次。分割調整の有無は要確認）
  - Yahoo Finance: yfinance の <code>.T（interval=1mo, auto_adjust=True で分割・分配金調整）

出力: 期間・件数、年ごとの高値/安値/終値、直近値の 10 年・20 年内の順位と分位、
高値からの下落率、分割日前後の比率（調整の有無の確認）。図は PNG に保存する。
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UA = "Mozilla/5.0 (compatible; jreit-score-prototype/0.1; personal research)"


def fetch_stooq(code: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={code.lower()}.jp&i=m"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError(f"stooq: 想定外の応答（列 {list(df.columns)[:6]}、先頭 {r.text[:80]!r}）")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.rename(columns=str.lower).sort_values("date").reset_index(drop=True)


def fetch_yahoo(code: str, start: str = "2003-01-01") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(f"{code}.T", start=start, interval="1mo", auto_adjust=True, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance: データ無し")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.lower).reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"]).reset_index(drop=True)


def split_check(df: pd.DataFrame, split_date: str) -> str:
    """分割日の直前・直後の月次終値の比率. 調整済みなら 1 前後、未調整なら分割比率が出る."""
    d = pd.Timestamp(split_date)
    before = df[df["date"] < d].tail(1)
    after = df[df["date"] >= d].head(1)
    if before.empty or after.empty:
        return "分割日前後のデータ無し"
    ratio = float(after["close"].iloc[0]) / float(before["close"].iloc[0])
    return (f"分割日 {split_date} の前月終値→当月終値の比率 {ratio:.3f}"
            f"（{before['date'].iloc[0].date()} → {after['date'].iloc[0].date()}）")


def summarize(df: pd.DataFrame, name: str, years: int, split_date: str | None) -> None:
    df = df.dropna(subset=["close"]).sort_values("date")
    last = df.iloc[-1]
    print(f"\n=== {name}: {len(df)} か月, {df['date'].min().date()} 〜 {df['date'].max().date()} ===")
    if split_date:
        print("  " + split_check(df, split_date))
    y = df.assign(year=df["date"].dt.year).groupby("year").agg(low=("low", "min"), high=("high", "max"), close=("close", "last"))
    y["low"] = y["low"].round(0); y["high"] = y["high"].round(0); y["close"] = y["close"].round(0)
    print("  年ごとの安値 / 高値 / 年末終値:")
    print(y.to_string())
    for n in (10, years):
        w = df[df["date"] >= last["date"] - pd.DateOffset(years=n)]
        rank = int((w["close"] < last["close"]).sum())
        pct = rank / max(len(w) - 1, 1) * 100
        lo, hi = w.loc[w["close"].idxmin()], w.loc[w["close"].idxmax()]
        print(f"  直近 {n} 年（{len(w)} か月）: 直近終値 {last['close']:.0f} は下から {rank + 1} 番目（下位 {pct:.0f}%）。"
              f" 安値 {lo['close']:.0f}（{lo['date'].date()}）, 高値 {hi['close']:.0f}（{hi['date'].date()}）,"
              f" 高値からの下落率 {(last['close'] / hi['close'] - 1) * 100:.1f}%")
    ath = df.loc[df["close"].idxmax()]
    print(f"  全期間の高値 {ath['close']:.0f}（{ath['date'].date()}）からの下落率 {(last['close'] / ath['close'] - 1) * 100:.1f}%")


def plot(series: dict[str, pd.DataFrame], code: str, years: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, df in series.items():
        ax.plot(df["date"], df["close"], lw=1.4, label=f"{name} monthly close")
    last_date = max(df["date"].max() for df in series.values())
    for n, c in ((10, "tab:orange"), (years, "tab:red")):
        for name, df in series.items():
            w = df[df["date"] >= last_date - pd.DateOffset(years=n)]
            if len(w):
                ax.axhline(w["close"].min(), color=c, lw=0.8, ls="--", alpha=0.7)
                ax.text(w["date"].min(), w["close"].min(), f" {n}y low ({name}) {w['close'].min():.0f}", color=c, fontsize=8, va="bottom")
                break
    ax.set_title(f"{code}.T monthly close (adjusted where the source adjusts)")
    ax.set_ylabel("JPY"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"図を保存: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("code", nargs="?", default="8972")
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--split-date", default=None, help="分割の効力発生日（調整の有無の確認用）")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    series: dict[str, pd.DataFrame] = {}
    for name, fn in (("stooq", fetch_stooq), ("yahoo", fetch_yahoo)):
        try:
            series[name] = fn(a.code)
            summarize(series[name], name, a.years, a.split_date)
        except Exception as e:
            print(f"\n=== {name}: 取得失敗 {type(e).__name__}: {str(e)[:200]}")
    if len(series) == 2:
        m = series["stooq"].merge(series["yahoo"], on="date", suffixes=("_s", "_y"))
        m = m.dropna(subset=["close_s", "close_y"])
        if len(m):
            r = m["close_s"] / m["close_y"]
            print(f"\nstooq / yahoo の終値比率（同月）: 中央値 {r.median():.3f}, 最小 {r.min():.3f}, 最大 {r.max():.3f}, {len(m)} か月")
    if series:
        plot(series, a.code, a.years, out / f"{a.code}_monthly.png")
