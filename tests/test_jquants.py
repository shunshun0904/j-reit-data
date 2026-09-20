"""J-Quants の整形層の検証（ネットワーク・APIキー不要）.

注意: 列名の候補は J-Quants の実応答で未検証。probe の結果を見てから固定する。
ここで固定しているのは「features 側が要求する出力形式」と、
列が想定と違ったときに黙って壊れずエラーになること。
"""
import pandas as pd

from jreit_score.ingest.jquants import AUTH_STYLES, ProbeResult, redact, to_dpu, to_prices

QUOTES = [
    {"Date": "2025-06-30", "Code": "89850", "Close": "128000", "Volume": 1000},
    {"Date": "2025-07-01", "Code": "89850", "Close": "129500", "Volume": 900},
    {"Date": "2025-06-30", "Code": "89510", "Close": "620000", "Volume": 500},
]
DIVIDENDS = [
    {"Code": "89850", "RecordDate": "2024-12-31", "DistributionAmount": "3937",
     "ForecastResultCode": "2"},
    {"Code": "89850", "RecordDate": "2025-06-30", "DistributionAmount": "4830",
     "ForecastResultCode": "2"},
    {"Code": "89850", "RecordDate": "2025-12-31", "DistributionAmount": "4900",
     "ForecastResultCode": "1"},          # 予想なので除外される
]


def test_to_prices_matches_features_input():
    df = to_prices(QUOTES)
    assert list(df.columns) == ["code", "date", "close", "dividend"]
    assert set(df["code"]) == {"8985", "8951"}          # 5桁コードは末尾0を落とす
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["close"].dtype == "float64"       # 取得回で int/float が混ざらないこと
    assert df["dividend"].dtype == "float64"
    assert (df["dividend"] == 0.0).all()
    assert df[["code", "date"]].equals(df[["code", "date"]].sort_values(["code", "date"]))


def test_to_prices_feeds_total_return_index():
    from jreit_score.features import total_return_index
    tri = total_return_index(to_prices(QUOTES))
    assert set(tri.columns) == {"code", "date", "tri", "ret"}
    assert len(tri) == 3


def test_to_dpu_drops_forecasts():
    df = to_dpu(DIVIDENDS)
    assert list(df.columns) == ["code", "period_end", "dpu"]
    assert len(df) == 2                                   # 予想の1件が落ちる
    assert df["period_end"].max() == pd.Timestamp("2025-06-30")
    assert df["dpu"].tolist() == [3937.0, 4830.0]
    assert df["dpu"].dtype == "float64"
    assert set(df["code"]) == {"8985"}


def test_unexpected_columns_raise_with_actual_names():
    """列名が想定と違ったら黙って空を返さずエラーにする（推測を重ねないため）."""
    for fn in (to_prices, to_dpu):
        try:
            fn([{"foo": 1, "bar": 2}])
        except KeyError as e:
            assert "foo" in str(e) and "bar" in str(e), str(e)
        else:
            raise AssertionError(f"{fn.__name__} が未知の列で KeyError にならなかった")


def test_empty_input_returns_empty_frame_with_right_columns():
    assert list(to_prices([]).columns) == ["code", "date", "close", "dividend"]
    assert list(to_dpu([]).columns) == ["code", "period_end", "dpu"]


def test_probe_result_line_shows_status_and_message():
    ok = ProbeResult("/v1/fins/dividend", "bearer", 200, ["dividend", "pagination_key"], 58)
    assert "OK" in ok.line() and "status=200" in ok.line() and "records=58" in ok.line()
    assert not ok.message                      # 200 のときは message を出さない

    # 403 の原因がルート不在か認証失敗かは message を見ないと分からない
    ng = ProbeResult("/v9/nope", "none", 403, ["message"], None,
                     "Missing Authentication Token")
    assert 'msg="Missing Authentication Token"' in ng.line()
    assert "OK" not in ng.line()


def test_redact_hides_base64_blobs_but_keeps_message():
    """API Gateway はヘッダ値の SHA-256 をエラーに含めて返す。ログに残さない."""
    msg = ("Invalid key=value pair (missing equal-sign) in Authorization header "
           "(hashed with SHA-256 and encoded with Base64): "
           "'YhU4eLMrxuRgff0IzIDC21o6gKUaUg0Qejn252LeO6M='")
    out = redact(msg)
    assert "YhU4eLMrxuRgff0IzIDC21o6gKUaUg0Qejn252LeO6M" not in out
    assert "<redacted>" in out
    assert "Authorization header" in out          # 診断に必要な文言は残す


def test_redact_keeps_normal_messages_intact():
    msg = "The requested endpoint does not exist. Please check the URL: https://jpx-jquants.com/spec/"
    assert redact(msg) == msg


def test_raw_authorization_style_is_not_offered():
    """生のキーを Authorization に入れる方式は SigV4 解釈でハッシュを返させるので使わない."""
    assert "authorization-raw" not in AUTH_STYLES
    assert AUTH_STYLES["none"]("k") == {}
    assert AUTH_STYLES["x-api-key"]("k") == {"x-api-key": "k"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
