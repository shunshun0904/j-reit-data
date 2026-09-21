"""公開ダッシュボード用 JSON の検証（ネットワーク不要）."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from jreit_score.site import STATUS, build, monthly_last, series_payload, write


def _daily(start="2023-01-02", end="2024-06-28", seed=0):
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"date": dates,
                         "yield": 0.5 + np.cumsum(rng.normal(0, 0.01, len(dates)))})


def test_monthly_last_takes_month_end_value():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-05", "2024-01-31", "2024-02-29", "2024-03-01"]),
        "yield": [1.0, 1.5, 2.0, 2.5]})
    m = monthly_last(df)
    assert m["date"].dt.strftime("%Y-%m").tolist() == ["2024-01", "2024-02", "2024-03"]
    assert m["yield"].tolist() == [1.5, 2.0, 2.5]      # 各月の最終値


def test_monthly_last_drops_gaps_without_interpolating():
    """欠損月は補間せず落とす（無かった値を作らない）."""
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-31", "2024-04-30"]),
                       "yield": [1.0, 2.0]})
    m = monthly_last(df)
    assert len(m) == 2
    assert m["date"].dt.strftime("%Y-%m").tolist() == ["2024-01", "2024-04"]


def test_series_payload_shape_and_attribution():
    p = series_payload(_daily())
    assert p["unit"] == "%"
    assert "財務省" in p["source"] and p["source_url"].startswith("https://www.mof.go.jp/")
    assert p["n_daily"] == len(_daily())
    assert all(len(pt) == 2 and isinstance(pt[0], str) and isinstance(pt[1], float)
               for pt in p["points"])
    assert p["points"] == sorted(p["points"])          # 日付順


def test_status_marks_unconnected_series_as_pending():
    """実データが無い指標を「出ている」ように見せない."""
    by = {s["item"]: s for s in STATUS}
    assert by["10年国債利回り"]["state"] == "ok"
    for item in ["財務指標", "将来リターン スコア", "分配金 安定性・成長 スコア", "金利上昇耐性 スコア"]:
        assert by[item]["state"] == "pending", item
    # 再配布できない生データは掲載しない
    for item in ["価格・トータルリターン", "分配金（DPU）"]:
        assert by[item]["state"] == "internal", item
    assert not any("統合スコア" in s["item"] for s in STATUS)      # 統合はしない（目的別 3 スコア）
    assert all(s["note"] for s in STATUS)
    assert all(s["state"] in {"ok", "internal", "pending"} for s in STATUS)


def test_payload_exposes_only_the_connected_series():
    """公開してよいのは接続済みの派生値だけ.

    文字列の部分一致ではなく構造で固定する（"J-Quants" は備考に出てよいが、
    数値系列が増えたら気付けるようにする）。
    """
    payload = build(_daily())
    assert set(payload) == {"generated_at", "jgb10", "status"}
    assert set(payload["jgb10"]) == {"label", "unit", "source", "source_url",
                                     "n_daily", "points"}
    # status は表示用の文字列だけ。数値データを持たせない
    for s in payload["status"]:
        assert set(s) == {"item", "state", "note"}
        assert all(isinstance(v, str) for v in s.values())


def test_payload_has_no_score_series_until_connected():
    """スコアが未接続のうちは、スコアらしき数値系列を payload に置かない."""
    payload = build(_daily())
    numeric_series = [k for k, v in payload.items()
                      if isinstance(v, dict) and "points" in v]
    assert numeric_series == ["jgb10"], numeric_series


def test_write_round_trips():
    payload = build(_daily())
    with tempfile.TemporaryDirectory() as d:
        p = write(payload, Path(d))
        assert p.name == "data.json"
        back = json.loads(p.read_text(encoding="utf-8"))
    assert back["jgb10"]["points"] == payload["jgb10"]["points"]
    assert back["generated_at"].endswith("Z")
    assert len(back["status"]) == len(STATUS)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
