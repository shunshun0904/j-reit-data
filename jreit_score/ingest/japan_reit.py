"""JAPAN-REIT.COM 銘柄ランキング（全銘柄・全指標）のスナップショット取得.

対象: https://www.japan-reit.com/sp/ranking/all
1ページに全銘柄 × 11指標（価格騰落率, 分配金利回り, NAV倍率, 時価総額, 資産規模, 棟数,
平均築年数, NOI利回り, 含み損益率, 年額分配金, ROE, 有利子負債比率）が載る。

このサイトは「現在値」しか出さないので, 定期実行して日付付きで蓄積し, 履歴を自作する。
利用規約上、取得データの再配布・転載は不可。プロトタイプの個人利用に限定し、
公開段階では一次情報（TDnet/EDINET）へ切り替える前提。
"""
from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.japan-reit.com/sp/ranking/all"
UA = "Mozilla/5.0 (compatible; jreit-score-prototype/0.1; personal research)"

COLMAP = {
    "価格騰落率": "price_chg_1y",
    "分配金利回り": "dist_yield",
    "NAV倍率": "nav_ratio",
    "時価総額(百万円)": "mcap_mn",
    "資産規模(億円)": "asset_size_okuyen",
    "棟数": "n_properties",
    "平均築年数": "avg_age",
    "NOI利回り": "noi_yield",
    "含み損益率": "unrealized_gain",
    "年額分配金(円)": "annual_dpu",
    "自己資本利益率（ROE）": "roe",
    "有利子負債比率": "ltv",
}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
         .str.replace("+", "", regex=False).str.strip(),
        errors="coerce",
    )


def parse_ranking_html(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html), flavor="lxml")
    # 「NAV倍率」と「有利子負債比率」を両方含む表を選ぶ
    tbl = next(t for t in tables if {"NAV倍率", "有利子負債比率"} <= set(map(str, t.columns)))
    tbl.columns = [str(c).strip() for c in tbl.columns]
    # 先頭の「コード 投資法人名」列: "8951 日本ビルファンド" 形式
    name_col = next(c for c in tbl.columns if "コード" in c or "投資法人" in c)
    m = tbl[name_col].astype(str).str.extract(r"^\s*(?P<code>[0-9A-Z]{4})\s+(?P<name>.+?)\s*$")
    out = pd.DataFrame({"code": m["code"], "name": m["name"]})
    for src, dst in COLMAP.items():
        if src in tbl.columns:
            out[dst] = _to_num(tbl[src])
    out = out.dropna(subset=["code"]).reset_index(drop=True)
    # %表記は小数に、時価総額は百万円→円換算せず log 用に残す
    for c in ["price_chg_1y", "dist_yield", "noi_yield", "unrealized_gain", "roe", "ltv"]:
        if c in out:
            out[c] = out[c] / 100.0
    return out


def fetch_ranking(session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    r = s.get(URL, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    df = parse_ranking_html(r.text)
    df.insert(0, "asof", pd.Timestamp(date.today()))
    return df


def save_snapshot(df: pd.DataFrame, root: Path) -> Path:
    asof = pd.Timestamp(df["asof"].iloc[0]).strftime("%Y-%m-%d")
    p = root / "japan_reit_ranking" / f"asof={asof}" / "part.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def load_snapshots(root: Path) -> pd.DataFrame:
    files = sorted((root / "japan_reit_ranking").glob("asof=*/part.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="Parquet 保存先ルート")
    a = ap.parse_args()
    df = fetch_ranking()
    p = save_snapshot(df, Path(a.out))
    print(f"saved {len(df)} rows -> {p}")
    print(df.head())
