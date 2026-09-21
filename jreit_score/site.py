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

from .features import OBJECTIVE_LABELS

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
# state: ok=掲載中, internal=取得済みだが再配布不可のため掲載しない, pending=未接続,
#        unidentified=モデルが推定できず掲載しない, unvalidated=推定はできたが時系列検証で
#        予測力が確認できず掲載しない（`publish.PUBLISH_MIN_T`）。
# 統合スコアは作らない（実データで目的をまたぐ共通因子が無かった。CLAUDE.md タスク4）。
# 目的別のスコアを並べる。
BASE_STATUS = [
    {"item": "10年国債利回り", "state": "ok",
     "note": "財務省 国債金利情報から取得。1974年9月以降"},
    {"item": "価格・トータルリターン", "state": "internal",
     "note": "J-Quants V2 から取得済みでモデル推定に使用。再配布できないため掲載しない"},
    {"item": "分配金（DPU）", "state": "internal",
     "note": "J-Quants V2 の決算短信サマリから取得済みでモデル推定に使用。掲載しない"},
    {"item": "財務指標", "state": "pending",
     "note": "現在の説明変数は NAV 倍率と時価総額の2本。LTV 等は JAPAN-REIT.COM の日次蓄積待ち"},
]


def status_rows(scores: dict | None) -> list[dict]:
    """接続状況の表. スコアが無ければ全目的を未接続、あれば目的ごとの判定を反映する."""
    rows = [dict(r) for r in BASE_STATUS]
    for key, label in OBJECTIVE_LABELS.items():
        item = f"{label} スコア"
        if scores is None:
            rows.append({"item": item, "state": "pending", "note": "モデル未接続"})
            continue
        o = scores["objectives"][key]
        oos = o.get("oos") or {}
        if o["published"]:
            rows.append({"item": item, "state": "ok",
                         "note": f"モデル推定値。基準日 {scores['as_of']}。"
                                 f"時系列検証の IC {oos['ic']}（NW-t {oos['ic_t']}, {oos['n_periods']} 期）"})
        elif o["usable"]:
            rows.append({"item": item, "state": "unvalidated", "note": o["publish_reason"]})
        else:
            rows.append({"item": item, "state": "unidentified",
                         "note": f"{o['status_label']}。掲載しない"})
    return rows


STATUS = status_rows(None)


def build(jgb10: pd.DataFrame, status: list[dict] | None = None, scores: dict | None = None) -> dict:
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "jgb10": series_payload(jgb10),
        "status": list(status if status is not None else status_rows(scores)),
    }
    if scores is not None:
        payload["scores"] = scores
    return payload


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
    ap.add_argument("--data", default=None, help="J-Quants store のディレクトリ。指定すると目的別スコアを載せる")
    a = ap.parse_args()
    jgb10 = fetch_jgb10_full()
    scores = None
    if a.data:
        from .ingest.jquants_store import load
        from .publish import check_payload, score_snapshot
        store = load(Path(a.data))
        if store.prices.empty or store.dpu.empty:
            raise SystemExit(f"store が空: {store.summary()}。先に fetch-jquants を実行すること")
        scores = score_snapshot(store, jgb10)
        check_payload(scores)
    payload = build(jgb10, scores=scores)
    p = write(payload, Path(a.out))
    pts = payload["jgb10"]["points"]
    print(f"wrote {p} ({p.stat().st_size} bytes)")
    print(f"  月末値 {len(pts)} 点: {pts[0][0]} 〜 {pts[-1][0]}")
    print(f"  接続済み: {sum(1 for s in payload['status'] if s['state'] == 'ok')}"
          f" / {len(payload['status'])}")
    if scores is not None:
        print(f"  スコア基準日 {scores['as_of']}, 推定 {scores['fit']}")
        for k, o in scores["objectives"].items():
            print(f"  {k:<7} {o['status']:<10} β={o.get('beta')} oos={o.get('oos')}")
        print(f"  掲載列: {scores['columns']}, 行数 {len(scores['rows'])}")
