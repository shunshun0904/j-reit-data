"""個人利用の確認用: 1 銘柄の 20 年程度の月次価格を取得し、要約統計だけを出す.

J-Quants の現プランは直近 10 年しか取れないため、より長い履歴は Yahoo Finance（yfinance）で確認する。
（stooq の CSV は JavaScript による確認ページが返り Actions からは取れなかった: 2026-09-21）

yfinance は auto_adjust=False で取り、
  - close     = 分割のみ調整した終値（価格水準の比較にはこちら）
  - adj_close = 分割と分配金を調整した終値（トータルリターン相当。過去ほど低く出る）
を分けて扱う。生データはログに出さず、年ごとの要約と分位、図だけを出す。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fetch_yahoo(code: str, start: str = "2003-01-01") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(f"{code}.T", start=start, interval="1mo", auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance: データ無し")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={"Adj Close": "adj_close"}).rename(columns=str.lower).reset_index()
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    cols = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
    return df[cols].dropna(subset=["close"]).reset_index(drop=True)


def split_check(df: pd.DataFrame, split_date: str, col: str = "close") -> str:
    """分割日の直前・直後の月次終値の比率. 調整済みなら 1 前後、未調整なら分割比率が出る."""
    d = pd.Timestamp(split_date)
    before = df[df["date"] < d].tail(1)
    after = df[df["date"] >= d].head(1)
    if before.empty or after.empty:
        return f"分割日 {split_date}: 前後のデータ無し"
    ratio = float(after[col].iloc[0]) / float(before[col].iloc[0])
    return (f"分割日 {split_date} の前月終値→当月終値の比率 {ratio:.3f}"
            f"（{before['date'].iloc[0].date()} → {after['date'].iloc[0].date()}; 1 前後なら調整済み）")


def level_lines(df: pd.DataFrame, col: str, years: int) -> list[str]:
    last = df.iloc[-1]
    out = []
    for n in (10, years):
        w = df[df["date"] >= last["date"] - pd.DateOffset(years=n)]
        rank = int((w[col] < last[col]).sum())
        pct = rank / max(len(w) - 1, 1) * 100
        lo, hi = w.loc[w[col].idxmin()], w.loc[w[col].idxmax()]
        out.append(f"  [{col}] 直近 {n} 年（{len(w)} か月）: 直近値 {last[col]:.0f} は下から {rank + 1} 番目（下位 {pct:.0f}%）。"
                   f" 安値 {lo[col]:.0f}（{lo['date'].date()}）, 高値 {hi[col]:.0f}（{hi['date'].date()}）,"
                   f" 高値比 {(last[col] / hi[col] - 1) * 100:+.1f}%")
    ath = df.loc[df[col].idxmax()]
    out.append(f"  [{col}] 全期間の高値 {ath[col]:.0f}（{ath['date'].date()}）比 {(last[col] / ath[col] - 1) * 100:+.1f}%")
    return out


def summarize(df: pd.DataFrame, years: int, split_dates: list[str]) -> None:
    df = df.dropna(subset=["close"]).sort_values("date")
    print(f"\n=== yahoo {len(df)} か月, {df['date'].min().date()} 〜 {df['date'].max().date()}"
          f"（最終月は月途中の値） ===")
    for d in split_dates:
        print("  " + split_check(df, d))
    y = df.assign(year=df["date"].dt.year).groupby("year").agg(
        low=("low", "min"), high=("high", "max"), close=("close", "last"))
    print("  年ごとの安値 / 高値 / 年末終値（分割のみ調整）:")
    print(y.round(0).astype(int).to_string())
    for col in ["close"] + (["adj_close"] if "adj_close" in df.columns else []):
        for line in level_lines(df, col, years):
            print(line)


def plot(df: pd.DataFrame, code: str, years: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["close"] + (["adj_close"] if "adj_close" in df.columns else [])
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 4 * len(cols)), sharex=True)
    axes = [axes] if len(cols) == 1 else list(axes)
    last_date = df["date"].max()
    titles = {"close": "monthly close, split-adjusted only (price level)",
              "adj_close": "monthly close, split + distribution adjusted (total-return like)"}
    for ax, col in zip(axes, cols):
        ax.plot(df["date"], df[col], lw=1.4, color="tab:blue")
        for n, c in ((10, "tab:orange"), (years, "tab:red")):
            w = df[df["date"] >= last_date - pd.DateOffset(years=n)]
            if len(w):
                ax.axhline(w[col].min(), color=c, lw=0.8, ls="--", alpha=0.8)
                ax.text(w["date"].min(), w[col].min(), f" {n}y low {w[col].min():.0f}", color=c, fontsize=8, va="bottom")
        ax.axhline(df[col].iloc[-1], color="gray", lw=0.8, ls=":")
        ax.text(df["date"].min(), df[col].iloc[-1], f" latest {df[col].iloc[-1]:.0f}", color="gray", fontsize=8, va="bottom")
        ax.set_title(f"{code}.T {titles[col]}"); ax.set_ylabel("JPY"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"図を保存: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("code", nargs="?", default="8972")
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--split-dates", default="", help="分割の効力発生日（YYYY-MM-DD をカンマ区切り。調整の有無の確認用）")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in a.split_dates.split(",") if s.strip()]
    df = fetch_yahoo(a.code)
    summarize(df, a.years, splits)
    plot(df, a.code, a.years, out / f"{a.code}_monthly.png")
