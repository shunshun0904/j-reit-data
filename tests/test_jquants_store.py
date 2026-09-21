"""差分取得と保存の検証（ネットワーク不要）."""
import tempfile
from pathlib import Path

import pandas as pd

from jreit_score.ingest.jquants_store import (PRICE_START, Store, covered_start,
                                              fetch_prices_clamped, load, merge_prices,
                                              next_from, save, update)


def _px(code, dates, close):
    return pd.DataFrame({"code": code, "date": pd.to_datetime(dates),
                         "close": [float(c) for c in close], "dividend": 0.0})


def test_next_from_is_day_after_last_saved_or_default():
    old = _px("8951", ["2024-06-27", "2024-06-28"], [1, 2])
    assert next_from(old, "8951") == "20240629"
    assert next_from(old, "8985") == PRICE_START          # 未保存の銘柄
    assert next_from(old.iloc[0:0], "8951") == PRICE_START


def test_merge_prices_dedups_and_prefers_new():
    """同じ (code, date) は新しい方（分割調整で AdjC が変わりうる）."""
    old = _px("8951", ["2024-06-27", "2024-06-28"], [100, 101])
    new = _px("8951", ["2024-06-28", "2024-07-01"], [50.5, 51])   # 6/28 が調整で変わった
    m = merge_prices(old, new)
    assert m["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-06-27", "2024-06-28", "2024-07-01"]
    assert m.loc[m["date"] == pd.Timestamp("2024-06-28"), "close"].iloc[0] == 50.5
    assert m["close"].dtype == "float64"


def test_merge_prices_handles_empty():
    assert merge_prices(_px("8951", [], []), _px("8951", [], [])).empty
    assert len(merge_prices(_px("8951", [], []), _px("8951", ["2024-01-04"], [1]))) == 1


def test_save_load_round_trip_and_summary_has_no_values():
    st = Store(
        universe=pd.DataFrame({"code": ["8951"], "code5": ["89510"], "name": ["x"]}),
        prices=_px("8951", ["2024-06-27", "2024-06-28"], [100, 101]),
        dpu=pd.DataFrame({"code": ["8951"], "period_start": pd.to_datetime(["2024-01-01"]),
                          "period_end": pd.to_datetime(["2024-06-30"]), "dpu": [1234.0],
                          "ex_date": pd.to_datetime(["2024-06-30"])}),
    )
    with tempfile.TemporaryDirectory() as d:
        save(st, Path(d))
        back = load(Path(d))
    assert back.prices.equals(st.prices) and back.dpu.equals(st.dpu)
    s = back.summary()
    assert s["prices_rows"] == 2 and s["dpu_codes"] == 1
    assert s["prices_range"] == ("2024-06-27", "2024-06-28")
    assert "1234" not in str(s) and "100" not in str(s).replace("2024", "")   # 値は出さない


def test_load_missing_dir_gives_empty_store():
    with tempfile.TemporaryDirectory() as d:
        st = load(Path(d) / "nope")
    assert st.universe.empty and st.prices.empty and st.dpu.empty
    assert list(st.prices.columns) == ["code", "date", "close", "dividend", "mktcap"]


REAL_400 = ("/equities/bars/daily -> HTTP 400: Your subscription covers the following dates: "
            "2016-09-21 ~ . If you want more data, please check other plans:https://jpx-jquants.com/#dataset")


def test_covered_start_parses_the_real_message():
    assert covered_start(REAL_400) == "20160921"
    assert covered_start("covers the following dates: 2016-09-21 ~ 2026-09-21.") == "20160921"
    assert covered_start("HTTP 403: Forbidden") is None
    assert covered_start("") is None


class _FakeClient:
    """bars/daily は対象期間外なら 400 相当の RuntimeError、範囲内なら日次行を返す."""
    def __init__(self, covered="20160921"):
        self.covered, self.calls = covered, []

    def get_all(self, path, params=None, **kw):
        params = params or {}
        self.calls.append((path, dict(params)))
        if path == "/equities/master":
            return [{"Code": "89510", "CoName": "日本ビルファンド投資法人", "ProdCat": "013", "Mkt": "0109"}]
        if path == "/fins/summary":
            return [{"DiscDate": f"{y}-08-15", "Code": "89510", "DocType": "FYFinancialStatements_Consolidated_REIT",
                     "CurPerType": "FY", "CurPerSt": f"{y}-01-01", "CurPerEn": f"{y}-06-30", "DivUnit": "1"}
                    for y in range(2017, 2026)]
        if path == "/equities/bars/daily":
            if params["from"] < self.covered:
                raise RuntimeError(REAL_400)
            days = pd.bdate_range(pd.Timestamp(params["from"]), pd.Timestamp(params["to"]))
            return [{"Date": d.strftime("%Y-%m-%d"), "Code": "89510", "C": "1", "AdjC": "1", "Vo": 1} for d in days]
        raise AssertionError(path)


def test_fetch_prices_clamped_retries_from_covered_start():
    c = _FakeClient()
    df, actual = fetch_prices_clamped(c, "8951", "20160801", "20160930")
    assert actual == "20160921"                       # 本文の開始日から取り直す
    assert df["date"].min() == pd.Timestamp("2016-09-21")
    assert [p for p, _ in c.calls].count("/equities/bars/daily") == 2


def test_fetch_prices_clamped_reraises_when_not_a_coverage_error():
    class Bad(_FakeClient):
        def get_all(self, path, params=None, **kw):
            if path == "/equities/bars/daily":
                raise RuntimeError("HTTP 500: boom")
            return super().get_all(path, params, **kw)
    try:
        fetch_prices_clamped(Bad(), "8951", "20160801", "20160930")
    except RuntimeError as e:
        assert "500" in str(e)
    else:
        raise AssertionError("対象期間以外のエラーは再送出すべき")


def test_update_first_run_clamps_then_incremental_run_fetches_only_new_days():
    """初回は対象期間で切り詰め、2回目は保存済み最終日の翌日から差分だけ取る."""
    c = _FakeClient()
    st0 = load(Path("/nonexistent"))
    st1, failed = update(c, st0, to_yyyymmdd="20160930", sleep=0)
    assert failed == []
    assert st1.prices["date"].min() == pd.Timestamp("2016-09-21")
    n_first = len(st1.prices)
    c.calls.clear()
    st2, failed = update(c, st1, to_yyyymmdd="20161007", sleep=0)
    assert failed == []
    bars = [p for path, p in c.calls if path == "/equities/bars/daily"]
    assert len(bars) == 1 and bars[0]["from"] == "20161001"     # 9/30 の翌日から
    assert len(st2.prices) == n_first + 5                        # 10/3〜10/7 の5営業日


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
