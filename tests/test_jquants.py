"""J-Quants V2 整形層の検証（ネットワーク・APIキー不要）.

列名は公式クライアント jquants-api-client 2.7.0 の constants.py から取った確定値。
"""
import pandas as pd

from jreit_score.ingest.jquants import (ENDPOINTS, API_BASE, Shape, redact, shape_of,
                                        to_dpu, to_prices)

# EQ_BARS_DAILY_COLUMNS_V2 の略記列名に合わせたフィクスチャ
BARS = [
    {"Date": "2025-06-30", "Code": "89850", "C": "128000", "AdjC": "128000", "Vo": 1000},
    {"Date": "2025-07-01", "Code": "89850", "C": "129500", "AdjC": "129500", "Vo": 900},
    {"Date": "2025-06-30", "Code": "89510", "C": "620000", "AdjC": "310000", "Vo": 500},
]
# FINS_DIVIDEND_COLUMNS_V2 に合わせたフィクスチャ。FRCode の実績/予想の値は未確定なので
# テストでは "R"=実績 / "F"=予想 と仮置きし、actual_codes 引数で渡す
DIVS = [
    {"Code": "89850", "RecDate": "2024-12-31", "ExDate": "2024-12-27", "DivRate": "3937", "FRCode": "R"},
    {"Code": "89850", "RecDate": "2025-06-30", "ExDate": "2025-06-26", "DivRate": "4830", "FRCode": "R"},
    {"Code": "89850", "RecDate": "2025-12-31", "ExDate": "2025-12-29", "DivRate": "4900", "FRCode": "F"},
]


def test_confirmed_endpoints_and_base():
    assert API_BASE.endswith("/v2")
    assert ENDPOINTS == {"master": "/equities/master",
                         "bars_daily": "/equities/bars/daily",
                         "dividend": "/fins/dividend",
                         "summary": "/fins/summary"}


def test_to_prices_uses_v2_short_column_names():
    df = to_prices(BARS)
    assert list(df.columns) == ["code", "date", "close", "dividend"]
    assert set(df["code"]) == {"8985", "8951"}          # 5桁 → 4桁
    assert df.loc[df["code"] == "8951", "close"].iloc[0] == 310000.0   # AdjC を採用
    assert df["close"].dtype == "float64" and (df["dividend"] == 0.0).all()


def test_to_prices_can_use_raw_close():
    df = to_prices(BARS, price_col="C")
    assert df.loc[df["code"] == "8951", "close"].iloc[0] == 620000.0


def test_to_prices_feeds_total_return_index():
    from jreit_score.features import total_return_index
    tri = total_return_index(to_prices(BARS))
    assert set(tri.columns) == {"code", "date", "tri", "ret"} and len(tri) == 3


def test_to_dpu_filters_forecasts_when_actual_codes_given():
    df = to_dpu(DIVS, actual_codes=("R",))
    assert list(df.columns) == ["code", "period_end", "dpu", "ex_date"]
    assert len(df) == 2 and df["dpu"].tolist() == [3937.0, 4830.0]
    assert df["period_end"].max() == pd.Timestamp("2025-06-30")
    assert df["ex_date"].iloc[0] == pd.Timestamp("2024-12-27")


def test_to_dpu_keeps_all_rows_without_actual_codes():
    """FRCode の意味が未確定のうちは、指定しなければ落とさない（黙って捨てない）."""
    assert len(to_dpu(DIVS)) == 3


def test_unexpected_columns_raise_with_actual_names():
    for fn in (to_prices, to_dpu):
        try:
            fn([{"foo": 1, "bar": 2}])
        except KeyError as e:
            assert "foo" in str(e) and "bar" in str(e)
        else:
            raise AssertionError(f"{fn.__name__} が未知の列で KeyError にならなかった")


def test_empty_input_returns_right_columns():
    assert list(to_prices([]).columns) == ["code", "date", "close", "dividend"]
    assert list(to_dpu([]).columns) == ["code", "period_end", "dpu", "ex_date"]


def test_shape_of_reports_enums_and_counts_but_not_records():
    sh = shape_of("dividend", DIVS, count_cols=("DivRate", "DistAmt"))
    text = "\n".join(sh.lines())
    assert sh.n == 3 and sh.enums["FRCode"] == ["F", "R"]
    assert sh.non_null == {"DivRate": 3}               # DistAmt は無いので出ない
    assert "3937" not in text and "4830" not in text   # 値そのものは出さない


def test_redact_hides_base64_blobs():
    s = "header (hashed with SHA-256 and encoded with Base64): 'YhU4eLMrxuRgff0IzIDC21o6gKUaUg0Qejn252LeO6M='"
    assert "YhU4eLMr" not in redact(s) and "<redacted>" in redact(s)
    assert redact("endpoint does not exist") == "endpoint does not exist"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
