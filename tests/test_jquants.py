"""J-Quants V2 整形層の検証（ネットワーク・APIキー不要）.

列名は公式クライアント jquants-api-client 2.7.0 の constants.py から取った確定値。
"""
import pandas as pd

from jreit_score.ingest.jquants import (ENDPOINTS, ENUM_COLS, API_BASE, Shape, exclude_annual,
                                        period_span_days, redact, select_reits, shape_of,
                                        to_dpu, to_dpu_from_summary, to_prices)

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


def test_shape_of_counts_non_empty_not_non_null():
    """文字列 API の欠損は '' で来る。'' を「ある」と数えてはいけない."""
    sh = shape_of("summary", SUMMARY, count_cols=("DivUnit", "FDivUnit"))
    assert sh.non_null == {"DivUnit": 4, "FDivUnit": 4}   # '' が1件ずつ
    assert "3900" not in "\n".join(sh.lines())


# /fins/summary の実際の列名・DocType 値に合わせたフィクスチャ（金額は架空）
SUMMARY = [
    {"DiscDate": "2024-02-15", "Code": "89850", "DocType": "FYFinancialStatements_Consolidated_REIT",
     "CurPerType": "FY", "CurPerEn": "2023-12-31", "DivUnit": "3900", "FDivUnit": "4000"},
    {"DiscDate": "2024-08-15", "Code": "89850", "DocType": "2QFinancialStatements_Consolidated_REIT",
     "CurPerType": "2Q", "CurPerEn": "2024-06-30", "DivUnit": "3,950", "FDivUnit": ""},
    {"DiscDate": "2024-10-01", "Code": "89850", "DocType": "REITEarnForecastRevision",
     "CurPerType": "FY", "CurPerEn": "2024-12-31", "DivUnit": "", "FDivUnit": "4100"},
    {"DiscDate": "2025-02-14", "Code": "89850", "DocType": "FYFinancialStatements_Consolidated_REIT",
     "CurPerType": "FY", "CurPerEn": "2024-12-31", "DivUnit": "4050", "FDivUnit": "4200"},
    {"DiscDate": "2025-02-20", "Code": "89850", "DocType": "FYFinancialStatements_Consolidated_REIT",
     "CurPerType": "FY", "CurPerEn": "2024-12-31", "DivUnit": "4060", "FDivUnit": "4200"},  # 訂正
]


def test_to_dpu_from_summary_keeps_actuals_only():
    d = to_dpu_from_summary(SUMMARY)
    assert list(d.columns) == ["code", "period_start", "period_end", "dpu", "ex_date"]
    assert set(d["code"]) == {"8985"}
    # 予想修正 (REITEarnForecastRevision) は落ち、同じ期の訂正は新しい方を採る
    assert d["period_end"].dt.strftime("%Y-%m-%d").tolist() == ["2023-12-31", "2024-06-30", "2024-12-31"]
    assert d["dpu"].tolist() == [3900.0, 3950.0, 4060.0]      # カンマ入りも読める
    assert (d["ex_date"] == d["period_end"]).all()
    assert d["dpu"].dtype == "float64"


def test_to_dpu_from_summary_feeds_dpu_stability():
    from jreit_score.features import dpu_stability
    d = to_dpu_from_summary(SUMMARY)
    out = dpu_stability(d[["code", "period_end", "dpu"]], [pd.Timestamp("2024-12-31")], window=6)
    assert len(out) == 1 and set(out.columns) == {"code", "period", "dpu_stab", "dpu_growth"}


MASTER = [
    {"Code": "89510", "CoName": "日本ビルファンド投資法人", "ProdCat": "013", "Mkt": "0109"},
    {"Code": "89850", "CoName": "ジャパン・ホテル・リート投資法人", "ProdCat": "013", "Mkt": "0109"},
    {"Code": "92830", "CoName": "日本再生可能エネルギーインフラ投資法人", "ProdCat": "013", "Mkt": "0109"},
    {"Code": "72030", "CoName": "トヨタ自動車", "ProdCat": "011", "Mkt": "0111"},
    {"Code": "13060", "CoName": "ＴＯＰＩＸ連動型上場投資信託", "ProdCat": "014", "Mkt": "0109"},
]


def test_select_reits_excludes_infra_funds_and_non_reits():
    """ProdCat='013' からインフラファンドを除くと J-REIT だけになる（63-5=58 の根拠）."""
    r = select_reits(pd.DataFrame(MASTER))
    assert r["code"].tolist() == ["8951", "8985"]
    assert list(r.columns) == ["code", "code5", "name"]
    assert select_reits(pd.DataFrame(MASTER), exclude_infra=False)["code"].tolist() == ["8951", "8985", "9283"]


def test_select_reits_empty_master():
    assert select_reits(pd.DataFrame()).empty


def test_period_start_is_kept_for_annualisation():
    """決算期間の長さが銘柄で違うので、期首を落とさない."""
    # 各行の期首はその期末の6か月前（半期決算を模す）
    rows = [dict(r, CurPerSt=(pd.Timestamp(r["CurPerEn"]) - pd.DateOffset(months=6) + pd.Timedelta(days=1))
                 .strftime("%Y-%m-%d")) for r in SUMMARY]
    d = to_dpu_from_summary(rows)
    assert d["period_start"].notna().all()
    assert (d["period_end"] > d["period_start"]).all()
    span_days = (d["period_end"] - d["period_start"]).dt.days
    assert span_days.between(180, 184).all()          # 6か月の期間


def test_period_start_absent_gives_nat_not_error():
    """CurPerSt が無い応答でも落ちず、period_start は NaT になる."""
    d = to_dpu_from_summary(SUMMARY)
    assert "period_start" in d.columns and d["period_start"].isna().all()


def _dpu(code, ends, months):
    ends = pd.to_datetime(ends)
    return pd.DataFrame({"code": code, "period_end": ends,
                         "period_start": ends - pd.DateOffset(months=months) + pd.Timedelta(days=1),
                         "dpu": 1000.0, "ex_date": ends})


def test_period_span_and_annual_exclusion():
    """半期決算（≈182日）は残し、年次決算（≈365日）は除く."""
    semi = _dpu("8951", ["2023-06-30", "2023-12-31", "2024-06-30", "2024-12-31"], 6)
    annual = _dpu("8985", ["2022-12-31", "2023-12-31", "2024-12-31"], 12)
    dpu = pd.concat([semi, annual], ignore_index=True)
    span = period_span_days(dpu)
    assert 180 <= span["8951"] <= 184 and 364 <= span["8985"] <= 366
    kept, excluded = exclude_annual(dpu)
    assert excluded == ["8985"] and set(kept["code"]) == {"8951"}


def test_period_span_falls_back_to_period_end_diff_when_start_missing():
    """period_start が NaT でも、連続する期末の差から期間を推定して判定できる."""
    dpu = _dpu("8951", ["2023-06-30", "2023-12-31", "2024-06-30"], 6)
    dpu["period_start"] = pd.NaT
    span = period_span_days(dpu)
    assert 180 <= span["8951"] <= 184
    kept, excluded = exclude_annual(dpu)
    assert excluded == [] and len(kept) == 3


def test_single_period_code_is_not_excluded():
    """1期しか無く期間を判定できない銘柄は除かない（黙って落とさない）."""
    dpu = _dpu("3462", ["2024-12-31"], 6); dpu["period_start"] = pd.NaT
    kept, excluded = exclude_annual(dpu)
    assert excluded == [] and len(kept) == 1


def test_enum_cols_never_include_amounts():
    """分配金額 (DivUnit/FDivUnit) の値集合をログに出さない."""
    assert "DivUnit" not in ENUM_COLS and "FDivUnit" not in ENUM_COLS


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
