"""財務省 国債金利情報 CSV パーサの検証（ネットワーク不要）.

注意: 下のフィクスチャは「こういう構造だろう」という想定であり、実ファイルで
確認したものではない。実構造は `python -m jreit_score.ingest.jgb --inspect` で
確認すること。元号の換算だけは実在の改元日で検証している。
"""
import numpy as np
import pandas as pd

from jreit_score.ingest.jgb import (combine, find_header_row, inspect,
                                    parse_jgb_csv, parse_jp_date, tenor)

# 想定構造（未検証）: 前書き数行 → 「基準日」ヘッダ → 和暦の日付と年限別利回り、欠損は '-'
CSV = """国債金利情報

基準日,1年,2年,5年,10年,20年,30年,40年
S49.9.24,-,-,-,8.226,-,-,-
H10.4.1,0.508,0.646,1.089,1.790,2.407,-,-
H31.4.30,-0.159,-0.163,-0.165,-0.049,0.350,0.500,0.600
R1.5.1,-0.160,-0.164,-0.166,-0.050,0.351,0.501,0.601
R6.4.1,0.023,0.212,0.420,0.735,1.550,1.870,2.080
"""

# 全角の年限列・西暦表記でも読めること（表記ゆれの吸収）
CSV_FULLWIDTH = """基準日,１年,１０年
2024/4/1,0.023,0.735
2024-04-02,0.024,0.740
"""


def test_era_conversion_uses_real_transition_dates():
    """実在の改元日で元号換算を固定する."""
    assert parse_jp_date("S49.9.24") == pd.Timestamp("1974-09-24")
    assert parse_jp_date("S64.1.7") == pd.Timestamp("1989-01-07")   # 昭和最後の日
    assert parse_jp_date("H1.1.8") == pd.Timestamp("1989-01-08")    # 平成初日
    assert parse_jp_date("H31.4.30") == pd.Timestamp("2019-04-30")  # 平成最後の日
    assert parse_jp_date("R1.5.1") == pd.Timestamp("2019-05-01")    # 令和初日
    assert parse_jp_date("令和6年4月1日") == pd.Timestamp("2024-04-01")


def test_western_dates_and_rejects():
    assert parse_jp_date("2024-04-01") == pd.Timestamp("2024-04-01")
    assert parse_jp_date("2024/4/1") == pd.Timestamp("2024-04-01")
    for bad in ["", "-", "基準日", "R6.13.1", "合計"]:
        assert parse_jp_date(bad) is None, bad


def test_header_detection_skips_preamble():
    assert find_header_row(CSV) == 2
    assert find_header_row("a,b\n1,2\n") == -1


def test_parse_and_missing_values():
    df = parse_jgb_csv(CSV)
    assert len(df) == 5
    assert df["date"].is_monotonic_increasing
    assert df["date"].iloc[0] == pd.Timestamp("1974-09-24")
    assert df["10年"].iloc[0] == 8.226
    assert np.isnan(df["1年"].iloc[0])          # '-' は NaN
    assert df["10年"].iloc[-1] == 0.735


def test_tenor_extraction_matches_features_input_shape():
    j = tenor(parse_jgb_csv(CSV), "10年")
    assert list(j.columns) == ["date", "yield"]
    assert len(j) == 5 and j["yield"].notna().all()
    assert j["yield"].iloc[-1] == 0.735


def test_tenor_drops_rows_missing_that_tenor():
    """要求した年限が欠損している行は落とす（10年は揃っていても1年は欠損しうる）."""
    df = parse_jgb_csv(CSV)
    assert len(tenor(df, "10年")) == 5
    assert len(tenor(df, "1年")) == 4          # S49.9.24 は 1年 が '-'
    assert tenor(df, "1年")["date"].min() == pd.Timestamp("1998-04-01")


def test_combine_prefers_current_on_overlap():
    """all は当月分を含まないので current を重ねる。重複日は current を採用する."""
    past = pd.DataFrame({"date": pd.to_datetime(["2026-08-30", "2026-08-31"]),
                         "yield": [2.9, 3.0]})
    cur = pd.DataFrame({"date": pd.to_datetime(["2026-08-31", "2026-09-01"]),
                        "yield": [3.05, 3.1]})
    out = combine(past, cur)
    assert len(out) == 3
    assert out["date"].is_monotonic_increasing
    assert out.loc[out["date"] == pd.Timestamp("2026-08-31"), "yield"].iloc[0] == 3.05
    assert out["yield"].tolist() == [2.9, 3.05, 3.1]


def test_fullwidth_columns_and_western_dates():
    df = parse_jgb_csv(CSV_FULLWIDTH)
    assert len(df) == 2
    j = tenor(df, "10年")
    assert j["yield"].tolist() == [0.735, 0.740]


def test_missing_tenor_reports_available_ones():
    try:
        tenor(parse_jgb_csv(CSV), "7年")
    except KeyError as e:
        assert "10年" in str(e)
    else:
        raise AssertionError("存在しない年限で KeyError にならなかった")


def test_missing_header_raises_with_guidance():
    try:
        parse_jgb_csv("a,b\n1,2\n")
    except ValueError as e:
        assert "--inspect" in str(e)
    else:
        raise AssertionError("ヘッダ不在で ValueError にならなかった")


def test_inspect_reports_structure_without_network():
    out = inspect(CSV)
    assert "ヘッダ行: 2" in out
    assert "'10年'" in out or "10年" in out
    assert "1974-09-24" in out


def test_feeds_rate_resilience():
    """パース結果をそのまま features.rate_resilience に渡せること."""
    from jreit_score.features import rate_resilience, total_return_index

    # フィクスチャ由来の形（列名と dtype）が features 側の想定と一致していること
    parsed = tenor(parse_jgb_csv(CSV), "10年")
    assert list(parsed.columns) == ["date", "yield"]
    assert pd.api.types.is_datetime64_any_dtype(parsed["date"])
    assert pd.api.types.is_numeric_dtype(parsed["yield"])

    # rate_resilience は週次で 20 週以上必要なので、回帰用には日次系列を別に用意する
    dates = pd.bdate_range("2023-01-02", "2024-06-28")
    rng = np.random.default_rng(0)
    jgb10 = pd.DataFrame({"date": dates,
                          "yield": 0.5 + np.cumsum(rng.normal(0, 0.01, len(dates)))})
    assert list(jgb10.columns) == list(parsed.columns)
    prices = pd.concat([
        pd.DataFrame({"code": c, "date": dates,
                      "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))),
                      "dividend": 0.0})
        for c in ["8951", "8985"]], ignore_index=True)
    out = rate_resilience(total_return_index(prices), jgb10, [pd.Timestamp("2024-06-28")])
    assert set(out.columns) == {"code", "period", "rate_resil", "dd_resil"}
    assert len(out) == 2 and out[["rate_resil", "dd_resil"]].notna().all().all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
