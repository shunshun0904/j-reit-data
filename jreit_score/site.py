"""公開ダッシュボード用の JSON を組み立てる.

方針（CLAUDE.md「制約・注意」）:
- 公開するのは派生値のみ。JAPAN-REIT.COM と J-Quants の生データは出さない
- 合成データの数値は公開しない。実データが接続されていない指標は「未接続」と書く
- 売買推奨の表現をしない。スコアは「モデル推定値」と明記する
- 出典を明記する
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

JGB_SOURCE = "財務省 国債金利情報"
JGB_SOURCE_URL = "https://www.mof.go.jp/jgbs/reference/interest_rate/"


def monthly_last(jgb10: pd.DataFrame) -> pd.DataFrame:
    """日次の [date, yield] を月末値に落とす.

    50年超の日次系列をそのまま線にしても読めないので月末値にする。
    欠損月は落とす（補間しない）。
    """
    s = (jgb10.dropna(subset=["yield"])
              .set_index("date")["yield"]
              .sort_index()
              .resample("ME").last()
              .dropna())
    return pd.DataFrame({"date": s.index, "yield": s.values})


def series_payload(jgb10: pd.DataFrame) -> dict:
    m = monthly_last(jgb10)
    return {
        "label": "10年国債利回り（月末値, %）",
        "unit": "%",
        "source": JGB_SOURCE,
        "source_url": JGB_SOURCE_URL,
        "n_daily": int(len(jgb10)),
        "points": [[d.strftime("%Y-%m-%d"), round(float(v), 3)]
                   for d, v in zip(m["date"], m["yield"])],
    }


# 接続状況。実データが入っていない指標を「出ている」ように見せないための表示。
STATUS = [
    {"item": "10年国債利回り", "state": "ok",
     "note": "財務省 国債金利情報から取得。1974年9月以降"},
    {"item": "価格・トータルリターン", "state": "pending",
     "note": "J-Quants に接続予定。V2 のエンドポイントを確認中"},
    {"item": "分配金（DPU）", "state": "pending",
     "note": "J-Quants に接続予定。JAPAN-REIT.COM と haitoukabu は履歴を持たないことを確認済み"},
    {"item": "財務指標", "state": "pending",
     "note": "JAPAN-REIT.COM のスナップショットを日次蓄積する方式。未着手"},
    {"item": "統合スコア", "state": "pending",
     "note": "上記が揃ってから算出。合成データの値は公開しない"},
]


def build(jgb10: pd.DataFrame, status: list[dict] | None = None) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "jgb10": series_payload(jgb10),
        "status": list(status if status is not None else STATUS),
    }


def write(payload: dict, out_dir: Path) -> Path:
    p = out_dir / "data.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    import argparse

    from .ingest.jgb import fetch_jgb10_full

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site", help="出力先ディレクトリ")
    a = ap.parse_args()
    payload = build(fetch_jgb10_full())
    p = write(payload, Path(a.out))
    pts = payload["jgb10"]["points"]
    print(f"wrote {p} ({p.stat().st_size} bytes)")
    print(f"  月末値 {len(pts)} 点: {pts[0][0]} 〜 {pts[-1][0]}")
    print(f"  接続済み: {sum(1 for s in payload['status'] if s['state'] == 'ok')}"
          f" / {len(payload['status'])}")
