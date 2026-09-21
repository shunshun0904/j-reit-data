"""公開ダッシュボード用 JSON の検証（ネットワーク不要）."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from jreit_score.site import STATUS, build, monthly_last, series_payload, status_rows, write


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
    for item in ["財務指標", "将来リターン スコア", "分配金の安定性 スコア", "分配金の成長 スコア",
                 "金利上昇耐性 スコア"]:
        assert by[item]["state"] == "pending", item
    # 再配布できない生データは掲載しない
    for item in ["価格・トータルリターン", "分配金（DPU）"]:
        assert by[item]["state"] == "internal", item
    assert not any("統合スコア" in s["item"] for s in STATUS)      # 統合はしない（目的別スコア）
    assert all(s["note"] for s in STATUS)
    assert all(s["state"] in {"ok", "internal", "pending"} for s in STATUS)


def _fake_scores():
    def obj(status, usable, published, t=4.0):
        return {"label": "x", "indicators": [], "status": status, "status_label": status, "reason": "",
                "usable": usable, "published": published,
                "publish_reason": "掲載" if published else ("非掲載（時系列検証で予測力が確認できず）" if usable else "非掲載（推定できず）"),
                "oos": {"ic": 0.3, "ic_t": t, "n_periods": 10}}
    return {"as_of": "2026-09-18", "fit": {}, "causes": ["nav_ratio", "log_mcap"],
            "objectives": {"q_ret": obj("ok", True, True), "q_stab": obj("ok", True, False, t=1.4),
                           "q_grow": obj("no_signal", False, False), "q_rate": obj("ok", True, True)},
            "columns": ["q_ret", "q_rate"],
            "rows": [{"code": "8951", "name": "A", "q_ret": {"z": 0.5, "q": 4}, "q_rate": {"z": -0.1, "q": 2}}]}


def test_status_reflects_objective_decisions_when_scores_exist():
    by = {s["item"]: s for s in status_rows(_fake_scores())}
    assert by["将来リターン スコア"]["state"] == "ok" and "2026-09-18" in by["将来リターン スコア"]["note"]
    assert by["分配金の安定性 スコア"]["state"] == "unvalidated"          # 推定可だが検証で予測力なし
    assert "予測力" in by["分配金の安定性 スコア"]["note"]
    assert by["分配金の成長 スコア"]["state"] == "unidentified"
    assert "掲載しない" in by["分配金の成長 スコア"]["note"]
    assert by["金利上昇耐性 スコア"]["state"] == "ok"
    payload = build(_daily(), scores=_fake_scores())
    assert set(payload) == {"generated_at", "jgb10", "status", "scores"}
    assert payload["scores"]["columns"] == ["q_ret", "q_rate"]


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
