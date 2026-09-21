"""公開スコア payload の検証（ネットワーク不要）."""
import numpy as np
import pandas as pd

from jreit_score.features import OBJECTIVES, OUTCOME_COLS
from jreit_score.ingest.jquants_store import Store
from jreit_score.model import DEFAULT_CAUSES, cross_sectional_standardize, fit_objective_factors
from jreit_score.publish import (PUBLISH_MIN_T, check_payload, publish_decision, quintiles,
                                 score_snapshot, snapshot_payload)
from jreit_score.validation import rolling_validation

BETA = {"q_ret": {"nav_ratio": -0.5}, "q_stab": {"ltv": -0.6}, "q_grow": {"occupancy": 0.5},
        "q_rate": {"fixed_rate_ratio": 0.4, "log_mcap": 0.3}}
LAM = {"q_ret": [0.7, 0.8], "q_stab": [0.6], "q_grow": [0.6], "q_rate": [0.6, 0.5]}


def _panel(seed=0, n_codes=58, n_periods=12):
    rng = np.random.default_rng(seed)
    rows = []
    for p in pd.date_range("2011-06-30", periods=n_periods, freq="6ME"):
        X = rng.normal(size=(n_codes, len(DEFAULT_CAUSES)))
        df = pd.DataFrame(X, columns=DEFAULT_CAUSES)
        for fac, inds in OBJECTIVES.items():
            b = np.array([BETA[fac].get(c, 0.0) for c in DEFAULT_CAUSES])
            eta = X @ b + rng.normal(scale=0.7, size=n_codes)
            df[inds] = np.outer(eta, LAM[fac]) + rng.normal(scale=0.8, size=(n_codes, len(inds)))
        df["code"] = [f"{8950 + i}" for i in range(n_codes)]
        df["period"] = p
        rows.append(df)
    return cross_sectional_standardize(pd.concat(rows, ignore_index=True), OUTCOME_COLS + DEFAULT_CAUSES)


def _payload():
    panel = _panel()
    o = fit_objective_factors(panel)
    ic, _ = rolling_validation(panel, horizon_periods=2, min_train_periods=6, objectives=OBJECTIVES)
    last = panel["period"].max()
    X = panel[panel["period"] == last][["code"] + DEFAULT_CAUSES].reset_index(drop=True)
    names = {c: f"銘柄{c}" for c in X["code"]}
    return snapshot_payload(o, ic, X, names, last, {"n_rows_complete": len(panel)}), X, panel


def test_quintiles_are_balanced_and_five_is_top():
    z = pd.Series(np.arange(20, dtype=float))
    q = quintiles(z)
    assert q.value_counts().to_dict() == {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}
    assert q.iloc[-1] == 5 and q.iloc[0] == 1
    assert quintiles(pd.Series([1.0, 2.0])).isna().all()


def test_payload_structure_has_only_derived_values():
    payload, X, _ = _payload()
    check_payload(payload)
    assert payload["columns"] == list(OBJECTIVES)                     # 合成データでは全目的が使える
    assert len(payload["rows"]) == len(X)
    assert [r["code"] for r in payload["rows"]] == sorted(r["code"] for r in payload["rows"])
    row = payload["rows"][0]
    assert set(row) == {"code", "name"} | set(OBJECTIVES) and row["name"].startswith("銘柄")
    for k in OBJECTIVES:
        assert set(row[k]) == {"z", "q"} and 1 <= row[k]["q"] <= 5
        o = payload["objectives"][k]
        assert o["usable"] and o["status"] == "ok" and o["n"] > 0
        assert set(o["beta"]) == set(DEFAULT_CAUSES)
        assert o["oos"]["n_periods"] >= 3 and o["oos"]["ic"] > 0, (k, o["oos"])
    # 生データや説明変数の値は行に無い
    dumped = str(payload["rows"])
    assert "nav_ratio" not in dumped and "close" not in dumped and "bps" not in dumped


def test_scores_are_standardized_and_ordered_like_the_model():
    payload, X, panel = _payload()
    z = pd.Series({r["code"]: r["q_ret"]["z"] for r in payload["rows"]})
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05
    # nav_ratio が低いほど q_ret が高い（β<0）
    nav = X.set_index("code")["nav_ratio"]
    assert z.corr(nav.reindex(z.index), method="spearman") < -0.5
    q = pd.Series({r["code"]: r["q_ret"]["q"] for r in payload["rows"]})
    assert q.corr(z, method="spearman") > 0.9


def test_publish_decision_requires_estimation_and_validated_signal():
    assert publish_decision("ok", {"ic_t": 4.0}) == (True, publish_decision("ok", {"ic_t": 4.0})[1])
    assert publish_decision("ok", {"ic_t": 4.0})[0]
    assert not publish_decision("ok", {"ic_t": PUBLISH_MIN_T - 0.01})[0]
    assert not publish_decision("ok", {"ic_t": None})[0]
    assert not publish_decision("weak", {"ic_t": 9.0})[0]
    assert not publish_decision("no_signal", None)[0]


def test_usable_but_unvalidated_objective_is_not_published():
    """推定はできても時系列検証の IC が有意でない目的は列に出さない（実データの q_stab）."""
    panel = _panel()
    o = fit_objective_factors(panel)
    ic, _ = rolling_validation(panel, horizon_periods=2, min_train_periods=6, objectives=OBJECTIVES)
    ic = ic.copy()
    ic["ic_composite_q_stab"] = [0.02, -0.01, 0.01, -0.02][:len(ic)] + [0.0] * max(0, len(ic) - 4)
    last = panel["period"].max()
    X = panel[panel["period"] == last][["code"] + DEFAULT_CAUSES].reset_index(drop=True)
    payload = snapshot_payload(o, ic, X, {}, last, {})
    check_payload(payload)
    st = payload["objectives"]["q_stab"]
    assert st["usable"] and not st["published"] and "予測力" in st["publish_reason"]
    assert "q_stab" not in payload["columns"] and "q_stab" not in payload["rows"][0]
    assert payload["objectives"]["q_ret"]["published"] and "q_ret" in payload["columns"]


def test_unusable_objective_has_no_column_but_keeps_its_status():
    panel = _panel()
    noise = {"q_ret": ["ret_6m", "ret_12m"], "q_noise": ["dpu_stab"]}
    panel["dpu_stab"] = np.random.default_rng(1).normal(size=len(panel))   # 純ノイズ
    o = fit_objective_factors(panel, noise)
    ic, _ = rolling_validation(panel, horizon_periods=2, min_train_periods=6, objectives=noise)
    last = panel["period"].max()
    X = panel[panel["period"] == last][["code"] + DEFAULT_CAUSES].reset_index(drop=True)
    payload = snapshot_payload(o, ic, X, {}, last, {}, noise)
    check_payload(payload, noise)
    assert payload["columns"] == ["q_ret"]
    assert payload["objectives"]["q_noise"]["status"] == "no_signal"
    assert not payload["objectives"]["q_noise"]["usable"]
    assert "q_noise" not in payload["rows"][0]
    # rolling では期ごとに判定するので、偶然 ok になる期があってもよい。
    # 使えた期の数と IC の有無が整合していることだけを固定する
    oos = payload["objectives"]["q_noise"]["oos"]
    assert oos["status_counts"].get("no_signal", 0) >= 1
    assert oos["n_periods"] == oos["status_counts"].get("ok", 0)
    assert (oos["ic"] is None) == (oos["n_periods"] < 4)      # Newey–West t は 4 期以上で計算する


def _synthetic_store(n_codes=12, seed=0):
    """score_snapshot の配線確認用の小さな store（日次価格 10 年, 半期 DPU）."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-09-21", "2026-09-18")
    codes = [f"{8950 + i}" for i in range(n_codes)]
    px = []
    for c in codes:
        r = rng.normal(0.0002, 0.01, len(dates))
        close = 100000 * np.exp(np.cumsum(r))
        px.append(pd.DataFrame({"code": c, "date": dates, "close": close, "dividend": 0.0,
                                "mktcap": close * (1 + 5 * rng.random()) / 1000}))
    prices = pd.concat(px, ignore_index=True)
    ends = pd.date_range("2016-08-31", "2026-07-31", freq="6ME")
    dpu = []
    for c in codes:
        base = 2000 + 500 * rng.random()
        for e in ends:
            dpu.append({"code": c, "period_start": e - pd.offsets.MonthEnd(6) + pd.Timedelta(days=1),
                        "period_end": e, "dpu": base * (1 + rng.normal(0, 0.05)), "ex_date": e,
                        "bps": 90000 + 20000 * rng.random(), "disc_date": e + pd.Timedelta(days=45)})
    dpu = pd.DataFrame(dpu)
    universe = pd.DataFrame({"code": codes, "code5": [c + "0" for c in codes], "name": [f"テスト{c}" for c in codes]})
    return Store(universe, prices, dpu)


def _jgb10():
    """週次で ±5bp 以上動く週が出る程度の変動を持たせる（無いと dd_resil が全銘柄 0 で定数になる）."""
    d = pd.bdate_range("2016-01-04", "2026-09-17")
    y = 0.5 + np.cumsum(np.random.default_rng(3).normal(0, 0.03, len(d)))
    return pd.DataFrame({"date": d, "yield": np.clip(y, -0.3, 3.0)})


def test_score_snapshot_end_to_end_on_synthetic_store():
    store = _synthetic_store()
    s = score_snapshot(store, _jgb10())
    check_payload(s)
    assert s["as_of"] == "2026-09-18"
    assert s["fit"]["universe"] == 12 and s["fit"]["scored"] == 12 and s["fit"]["unscored_codes"] == []
    assert s["fit"]["n_periods"] >= 8 and s["fit"]["n_rows_complete"] > 0
    assert set(s["objectives"]) == set(OBJECTIVES)
    for r in s["rows"]:
        assert r["name"].startswith("テスト")
    assert set(s["columns"]) <= set(OBJECTIVES)     # ランダムな価格なので使える目的が無くてもよい


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
