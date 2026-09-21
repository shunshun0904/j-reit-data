"""差分取得と保存の検証（ネットワーク不要）."""
import tempfile
from pathlib import Path

import pandas as pd

from jreit_score.ingest.jquants_store import (PRICE_START, Store, load, merge_prices,
                                              next_from, save)


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
    assert list(st.prices.columns) == ["code", "date", "close", "dividend"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
