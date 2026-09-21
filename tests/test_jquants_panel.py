"""分配金の計上と説明変数パネルの検証（ネットワーク不要）."""
import numpy as np
import pandas as pd

from jreit_score.ingest.jquants_panel import (JQ_CAUSES, attach_distributions, causes_panel,
                                              half_year_ends)


def _px(code, dates, close, mktcap=1000.0):
    return pd.DataFrame({"code": code, "date": pd.to_datetime(dates),
                         "close": [float(c) for c in close], "dividend": 0.0, "mktcap": float(mktcap)})


def test_attach_distributions_lands_on_first_trading_day_on_or_after_ex_date():
    px = _px("8951", ["2024-06-27", "2024-06-28", "2024-07-01", "2024-07-02"], [1, 1, 1, 1])
    dpu = pd.DataFrame({"code": ["8951"], "ex_date": pd.to_datetime(["2024-06-30"]), "dpu": [500.0]})
    out = attach_distributions(px, dpu)
    got = out.set_index("date")["dividend"]
    assert got[pd.Timestamp("2024-07-01")] == 500.0            # 6/30 は日曜 → 7/1 に載る
    assert got.drop(pd.Timestamp("2024-07-01")).eq(0).all()


def test_attach_distributions_sums_same_day_and_ignores_unknown_codes():
    px = _px("8951", ["2024-07-01"], [1])
    dpu = pd.DataFrame({"code": ["8951", "8951", "9999"],
                        "ex_date": pd.to_datetime(["2024-06-30", "2024-07-01", "2024-07-01"]),
                        "dpu": [1.0, 2.0, 99.0]})
    out = attach_distributions(px, dpu)
    assert out["dividend"].tolist() == [3.0]


def test_attach_distributions_feeds_total_return_index():
    from jreit_score.features import total_return_index
    px = _px("8951", ["2024-06-28", "2024-07-01"], [100, 100])
    dpu = pd.DataFrame({"code": ["8951"], "ex_date": pd.to_datetime(["2024-07-01"]), "dpu": [10.0]})
    tri = total_return_index(attach_distributions(px, dpu))
    assert abs(tri["tri"].iloc[-1] - 1.10) < 1e-9        # 分配金再投資で +10%


def test_causes_panel_uses_disclosure_date_not_period_end():
    """BPS は開示日以降でしか使わない（期末で結合すると先読み）."""
    px = _px("8951", ["2024-06-28", "2024-08-15", "2024-08-16"], [200, 200, 200], mktcap=np.e ** 5)
    dpu = pd.DataFrame({"code": ["8951"], "period_end": pd.to_datetime(["2024-06-30"]),
                        "bps": [100.0], "disc_date": pd.to_datetime(["2024-08-16"])})
    before = causes_panel(px, dpu, [pd.Timestamp("2024-08-15")])
    after = causes_panel(px, dpu, [pd.Timestamp("2024-08-16")])
    assert np.isnan(before["nav_ratio"].iloc[0])               # 開示前は不明
    assert after["nav_ratio"].iloc[0] == 2.0                    # 200 / 100
    assert abs(after["log_mcap"].iloc[0] - 5.0) < 1e-9
    assert list(after.columns) == ["code", "period"] + JQ_CAUSES


def test_causes_panel_takes_last_price_on_or_before_period():
    px = _px("8951", ["2024-06-27", "2024-06-28", "2024-07-01"], [1, 2, 3])
    dpu = pd.DataFrame({"code": ["8951"], "period_end": pd.to_datetime(["2023-12-31"]),
                        "bps": [1.0], "disc_date": pd.to_datetime(["2024-02-15"])})
    out = causes_panel(px, dpu, [pd.Timestamp("2024-06-30")])
    assert out["nav_ratio"].iloc[0] == 2.0                      # 6/28 の終値


def test_half_year_ends():
    p = half_year_ends("2017-06-30", "2018-12-31")
    assert [d.strftime("%Y-%m-%d") for d in p] == ["2017-06-30", "2017-12-31", "2018-06-30", "2018-12-31"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
