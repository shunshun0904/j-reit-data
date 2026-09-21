"""実データでの推定: J-Quants の store と財務省の10年債から目的変数と説明変数を作り、
符号判定つきで MIMIC を推定し、時系列分割で検証する.

説明変数は J-Quants だけで作れる nav_ratio と log_mcap の2本（設計判断 2026-09-21）。
ltv / noi_yield / unrealized_gain は JAPAN-REIT.COM のスナップショット蓄積待ち。

出力は推定値・適合度・IC・件数のみ（派生値）。生データは出さない。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import OUTCOME_COLS, build_outcomes
from .ingest.jgb import fetch_jgb10_full
from .ingest.jquants_panel import JQ_CAUSES, attach_distributions, causes_panel, half_year_ends
from .ingest.jquants_store import load
from .model import cross_sectional_standardize, fit_with_sign_branch, summarize_branch
from .validation import report, rolling_validation


def build_panel(store, jgb10: pd.DataFrame, periods: list[pd.Timestamp]) -> pd.DataFrame:
    prices = attach_distributions(store.prices, store.dpu)
    outcomes = build_outcomes(prices, store.dpu[["code", "period_end", "dpu"]], jgb10, periods)
    causes = causes_panel(store.prices, store.dpu, periods)
    return causes.merge(outcomes, on=["code", "period"], how="left")


def coverage(panel: pd.DataFrame, cols: list[str]) -> pd.Series:
    """期ごとの、全列が揃った銘柄数（推定に使える行数）."""
    ok = panel.dropna(subset=cols)
    return (ok.groupby("period").size().rename("n_complete")
              .reindex(sorted(panel["period"].unique()), fill_value=0))


def indicator_structure(panel: pd.DataFrame, indicators: list[str] = OUTCOME_COLS,
                        causes: list[str] = JQ_CAUSES) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """横断面 z 化後の行をプールした Spearman 相関（指標間・指標×説明変数）と、
    指標間相関行列の固有値（降順）.

    1因子で統合できるなら第1固有値だけが大きい。目的ごとに別の因子があるなら
    2番目以降も 1 を超える（Kaiser 基準）。負荷量の符号判定だけでは見えない
    「符号は割れないが共通因子が無い」ケースをここで確かめる。
    """
    cols = list(indicators) + list(causes)
    usable = panel.dropna(subset=cols)
    corr = usable[cols].corr(method="spearman")
    among = corr.loc[list(indicators), list(indicators)]
    with_causes = corr.loc[list(indicators), list(causes)]
    eig = np.sort(np.linalg.eigvalsh(among.to_numpy()))[::-1]
    return among, with_causes, eig


def main(data: str, start: str, end: str, min_train: int) -> None:
    store = load(Path(data))
    if store.prices.empty or store.dpu.empty:
        raise SystemExit(f"store が空: {store.summary()}。先に fetch-jquants を実行すること")
    if "mktcap" not in store.prices.columns or "bps" not in store.dpu.columns:
        raise SystemExit("store が旧形式（mktcap / bps 無し）。cache キー v2 で取り直すこと")
    print("store:", store.summary())

    jgb10 = fetch_jgb10_full()
    print(f"jgb10: {len(jgb10)} 日 ({jgb10['date'].min().date()} 〜 {jgb10['date'].max().date()})")

    periods = half_year_ends(start, end)
    panel = build_panel(store, jgb10, periods)
    cols = OUTCOME_COLS + JQ_CAUSES
    print("\n期ごとの推定に使える銘柄数（6指標＋2説明変数が揃う行）:")
    print(coverage(panel, cols).to_string())

    panel = cross_sectional_standardize(panel, cols)
    usable = panel.dropna(subset=cols)
    print(f"\n推定に使う行: {len(usable)}（{usable['period'].nunique()} 期 × 最大 {usable.groupby('period').size().max()} 銘柄）")

    among, with_causes, eig = indicator_structure(panel)
    print("\n=== 指標の相関構造（横断面 z 化後をプール, Spearman） ===")
    print("指標間:")
    print(among.round(2).to_string())
    print("固有値（降順）: " + ", ".join(f"{v:.2f}" for v in eig)
          + f"  （1 を超える数 = {int((eig > 1).sum())}）")
    print("指標 × 説明変数:")
    print(with_causes.round(2).to_string())

    print("\n=== in-sample fit (全期間, causes = nav_ratio + log_mcap) ===")
    b = fit_with_sign_branch(panel, causes=JQ_CAUSES)
    print(summarize_branch(b))

    print(f"\n=== rolling out-of-sample validation (min_train_periods={min_train}, 符号判定つき) ===")
    ic, _ = rolling_validation(panel, horizon_periods=2, min_train_periods=min_train,
                               causes=JQ_CAUSES, branch=True)
    if "decision" in ic:
        print("  判定の内訳: " + ", ".join(f"{k}={v}" for k, v in ic["decision"].value_counts().items()))
    print(report(ic))
    if "error" in ic:
        print("errors:", int(ic["error"].notna().sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/jquants")
    ap.add_argument("--start", default="2017-06-30")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--min-train", type=int, default=6)
    a = ap.parse_args()
    main(a.data, a.start, a.end, a.min_train)
