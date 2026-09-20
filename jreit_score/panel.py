"""取得したスナップショットと DPU 履歴を, モデル入力（code × period × 説明変数）に整形する.

JAPAN-REIT.COM から取れる指標で構成するプロトタイプ用の説明変数セット。
固定金利比率・稼働率は取れないため, 一次情報に切り替えるまでは除外する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROTO_CAUSES = ["nav_ratio", "ltv", "unrealized_gain", "noi_yield", "log_mcap", "avg_age", "dist_yield"]


def snapshot_to_causes(snap: pd.DataFrame, periods: list[pd.Timestamp]) -> pd.DataFrame:
    """日次スナップショットの蓄積から, 各 period の直近スナップショットを採用する（先読み防止）."""
    snap = snap.sort_values("asof")
    rows = []
    for p in periods:
        s = snap[snap["asof"] <= p]
        if s.empty:
            continue
        latest = s[s["asof"] == s["asof"].max()].copy()
        latest["period"] = p
        rows.append(latest)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if df.empty:
        return df
    df["log_mcap"] = np.log(df["mcap_mn"])
    return df[["code", "period"] + [c for c in PROTO_CAUSES if c in df.columns]]


def merge_panel(causes: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    return causes.merge(outcomes, on=["code", "period"], how="left")
