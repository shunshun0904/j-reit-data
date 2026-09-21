"""符号割れ判定と2因子分岐の検証（ネットワーク不要, 合成データのみ）."""
import numpy as np
import pandas as pd

from jreit_score.features import OBJECTIVES, OUTCOME_COLS
from jreit_score.model import (DEFAULT_CAUSES, FittedRegression, check_sign_split,
                               cross_sectional_standardize, factor_residual_correlations,
                               fit_mimic, fit_objective_factors, fit_objective_model_joint,
                               fit_with_sign_branch, objective_model_spec,
                               summarize_objectives, two_factor_spec)
from jreit_score.validation import report, rolling_validation, status_summary

RETURN_GROUP = ["ret_6m", "ret_12m", "rate_resil", "dd_resil"]
STABILITY_GROUP = ["dpu_stab", "dpu_growth"]
BETA = {"nav_ratio": -0.4, "ltv": -0.3, "fixed_rate_ratio": 0.2, "unrealized_gain": 0.3,
        "noi_yield": 0.2, "occupancy": 0.15, "log_mcap": 0.1}


def _panel(lam, seed=0, n_codes=58, n_periods=10):
    """X → quality → 6指標 の生成過程で, 負荷量 lam を任意に指定できる合成パネル."""
    rng = np.random.default_rng(seed)
    lam = np.asarray(lam, dtype=float)
    rows = []
    for p in pd.date_range("2011-06-30", periods=n_periods, freq="6ME"):
        X = rng.normal(size=(n_codes, len(DEFAULT_CAUSES)))
        eta = X @ np.array([BETA[c] for c in DEFAULT_CAUSES]) + rng.normal(scale=0.7, size=n_codes)
        df = pd.DataFrame(X, columns=DEFAULT_CAUSES)
        df[OUTCOME_COLS] = np.outer(eta, lam) + rng.normal(scale=0.8, size=(n_codes, 6))
        df["code"] = [f"{8950 + i}" for i in range(n_codes)]
        df["period"] = p
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    return cross_sectional_standardize(panel, OUTCOME_COLS + DEFAULT_CAUSES)


ALIGNED = [0.7, 0.8, 0.5, 0.4, 0.6, 0.5]                 # 6指標が同じ向き
SPLIT = [0.7, 0.8, -0.5, -0.4, 0.6, 0.5]                 # 安定性2指標が逆向き
WEAK_OPPOSITE = [0.7, 0.8, -0.05, 0.4, 0.6, 0.5]         # 逆向きだが弱すぎる1本
LONE_OPPOSITE = [0.7, 0.8, -0.6, 0.4, 0.6, 0.5]          # 逆向きが1本だけ（2因子を識別できない）


def test_aligned_stays_one_factor():
    b = fit_with_sign_branch(_panel(ALIGNED))
    assert b.decision == "one_factor", b.reason
    assert b.two_factor is None
    assert not b.sign_check.split
    assert b.sign_check.minority == []
    assert list(b.predict_scores(_panel(ALIGNED)).columns) == ["quality"]


def test_split_branches_to_two_factors():
    panel = _panel(SPLIT)
    b = fit_with_sign_branch(panel)
    assert b.decision == "two_factor", b.reason
    assert b.sign_check.split
    assert set(b.sign_check.minority) == set(STABILITY_GROUP)
    assert set(b.two_factor.groups["q1"]) == set(RETURN_GROUP)
    assert set(b.two_factor.groups["q2"]) == set(STABILITY_GROUP)


def test_fit_indices_alone_do_not_detect_the_split():
    """適合度は割れていても良いまま. 判定に適合度を使ってはいけないことの確認."""
    f = fit_mimic(_panel(SPLIT))
    s = f.stats["Value"] if "Value" in f.stats.columns else f.stats.iloc[:, 0]
    assert float(s["CFI"]) > 0.95 and float(s["RMSEA"]) < 0.06
    assert check_sign_split(f).split


def test_weak_opposite_loading_does_not_branch():
    """|λ| が閾値未満の逆符号では2因子に落とさない（過剰分岐の防止）."""
    b = fit_with_sign_branch(_panel(WEAK_OPPOSITE))
    assert b.decision == "one_factor", b.reason
    assert abs(b.sign_check.loadings["dpu_stab"]) < 0.15


def test_lone_opposite_indicator_is_reported_not_silently_merged():
    """逆向きが1本だけだと2因子を識別できない. 黙って1因子に統合しない."""
    b = fit_with_sign_branch(_panel(LONE_OPPOSITE))
    assert b.decision == "two_factor_unavailable", b.reason
    assert b.two_factor is None
    assert "識別できない" in b.reason


def test_two_factor_spec_declares_residual_covariance():
    """q1 ~~ q2 を省くと semopy は2因子を独立と制約し適合が崩れる."""
    spec = two_factor_spec(RETURN_GROUP, STABILITY_GROUP, DEFAULT_CAUSES)
    assert "q1 ~~ q2" in spec
    assert spec.count("=~") == 2 and spec.count("q1 ~ ") == 1 and spec.count("q2 ~ ") == 1


def test_factors_are_oriented_higher_is_better():
    """各因子は負荷量の合計が正になる向きに揃える（潜在変数の符号は識別上任意）."""
    f = fit_with_sign_branch(_panel(SPLIT)).two_factor
    for fac, inds in f.groups.items():
        assert f.loadings[inds].mean() > 0, (fac, f.loadings[inds].to_dict())


def test_two_factor_score_ranks_stability_the_right_way():
    """1因子スコアは安定性を逆順に並べるが, 2因子の q2 スコアは正しく並べる."""
    panel = _panel(SPLIT)
    test = panel[panel["period"] == panel["period"].max()].dropna(subset=DEFAULT_CAUSES)
    b = fit_with_sign_branch(panel)
    one = b.one_factor.predict_score(test)
    two = b.two_factor.predict_scores(test)
    assert one.corr(test["dpu_stab"], method="spearman") < 0
    assert two["q2"].corr(test["dpu_stab"], method="spearman") > 0
    assert two["q1"].corr(test["ret_12m"], method="spearman") > 0


def test_predict_scores_preserves_index():
    panel = _panel(SPLIT)
    test = panel[panel["period"] == panel["period"].max()].dropna(subset=DEFAULT_CAUSES)
    s = fit_with_sign_branch(panel).predict_scores(test)
    assert list(s.columns) == ["q1", "q2"]
    assert s.index.equals(test.index) and not s.isna().any().any()


# ---------------------------------------------------------------------------
# 目的別 3 因子
# ---------------------------------------------------------------------------

NO_DPU = [0.7, 0.8, 0.0, 0.0, 0.6, 0.5]                  # 分配金2指標が因子に載らない（純ノイズ）
RET_SPLIT = [0.7, -0.8, 0.5, 0.4, 0.6, 0.5]              # リターン2指標の向きが逆

# 目的ごとに別の因子が別の説明変数で決まる生成過程（実データと同じく共通因子が無い）
BETA3 = {
    "q_ret": {"nav_ratio": -0.5, "log_mcap": 0.1},
    "q_stab": {"ltv": -0.5},
    "q_grow": {"occupancy": 0.5},
    "q_rate": {"fixed_rate_ratio": 0.4, "log_mcap": 0.3},
}
LAM3 = {"q_ret": [0.7, 0.8], "q_stab": [0.6], "q_grow": [0.6], "q_rate": [0.6, 0.5]}


def _panel3(seed=0, n_codes=58, n_periods=10):
    rng = np.random.default_rng(seed)
    rows = []
    for p in pd.date_range("2011-06-30", periods=n_periods, freq="6ME"):
        X = rng.normal(size=(n_codes, len(DEFAULT_CAUSES)))
        df = pd.DataFrame(X, columns=DEFAULT_CAUSES)
        for fac, inds in OBJECTIVES.items():
            b = np.array([BETA3[fac].get(c, 0.0) for c in DEFAULT_CAUSES])
            eta = X @ b + rng.normal(scale=0.7, size=n_codes)
            df[inds] = np.outer(eta, LAM3[fac]) + rng.normal(scale=0.8, size=(n_codes, len(inds)))
        df["code"] = [f"{8950 + i}" for i in range(n_codes)]
        df["period"] = p
        rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    return cross_sectional_standardize(panel, OUTCOME_COLS + DEFAULT_CAUSES)


def _last_period(panel):
    return panel[panel["period"] == panel["period"].max()].dropna(subset=DEFAULT_CAUSES)


def test_objectives_cover_outcome_cols_exactly_once():
    flat = [i for inds in OBJECTIVES.values() for i in inds]
    assert sorted(flat) == sorted(OUTCOME_COLS) and len(flat) == len(set(flat))
    assert all(len(inds) in (1, 2) for inds in OBJECTIVES.values())


def test_objective_factors_all_usable_when_common_factor_exists():
    panel = _panel(ALIGNED)
    o = fit_objective_factors(panel)
    assert o.usable == list(OBJECTIVES), {n: (f.status, f.reason) for n, f in o.factors.items()}
    test = _last_period(panel)
    s = o.predict_scores(test)
    assert list(s.columns) == list(OBJECTIVES) and s.index.equals(test.index)
    for fac, inds in OBJECTIVES.items():
        for i in inds:
            assert s[fac].corr(test[i], method="spearman") > 0, (fac, i)


def test_separate_objectives_pick_their_own_causes():
    o = fit_objective_factors(_panel3())
    assert o.usable == list(OBJECTIVES), {n: (f.status, f.reason) for n, f in o.factors.items()}
    assert o.factors["q_ret"].fitted.beta["nav_ratio"] < -0.2
    assert o.factors["q_stab"].fitted.beta["ltv"] < -0.2
    assert o.factors["q_grow"].fitted.beta["occupancy"] > 0.2
    assert o.factors["q_rate"].fitted.beta["fixed_rate_ratio"] > 0.2


def test_single_indicator_objective_is_a_regression():
    panel = _panel3()
    o = fit_objective_factors(panel)
    f = o.factors["q_stab"].fitted
    assert isinstance(f, FittedRegression) and f.indicators == ["dpu_stab"] and f.n > 0
    assert f.pvalues["ltv"] < 0.05 and 0 < f.r2 < 1
    test = _last_period(panel)
    assert f.predict_score(test).corr(test["dpu_stab"], method="spearman") > 0
    assert "regression" in summarize_objectives(o)


def test_one_factor_rule_does_not_catch_a_missing_common_factor():
    """実データと同じ構造: 符号は割れないが共通因子が無い. 目的別なら全目的が使える."""
    panel = _panel3()
    b = fit_with_sign_branch(panel)
    assert b.decision == "one_factor", b.reason          # 判定ルールの盲点
    assert len(fit_objective_factors(panel).usable) == len(OBJECTIVES)


def test_noise_objective_is_reported_not_scored():
    """分配金 2 指標が純ノイズ: 回帰に有意な説明変数が無く no_signal. スコアは出さない."""
    panel = _panel(NO_DPU)
    o = fit_objective_factors(panel)
    for n in ("q_stab", "q_grow"):
        assert o.factors[n].status == "no_signal", (n, o.factors[n].reason)
    assert o.factors["q_ret"].status == "ok" and o.factors["q_rate"].status == "ok"
    cols = o.predict_scores(_last_period(panel)).columns
    assert "q_stab" not in cols and "q_grow" not in cols
    assert set(o.indicator_groups()) == {"q_ret", "q_rate"}
    assert "ノイズ" in summarize_objectives(o)


def test_weak_two_indicator_objective_is_unidentified():
    """2 指標が因子に載らない（純ノイズ）と識別不能. 実データの q_dpu と同じ状況."""
    o = fit_objective_factors(_panel(NO_DPU), objectives={"q_dpu": ["dpu_stab", "dpu_growth"]})
    assert o.factors["q_dpu"].status != "ok", o.factors["q_dpu"].reason
    assert o.usable == []


def test_sign_split_within_objective_is_reported():
    o = fit_objective_factors(_panel(RET_SPLIT))
    assert o.factors["q_ret"].status == "sign_split", o.factors["q_ret"].reason
    assert "q_ret" not in o.usable


def test_objective_model_spec_declares_residual_covariances():
    spec = objective_model_spec(OBJECTIVES, DEFAULT_CAUSES)
    assert spec.count("=~") == 4 and spec.count("~~") == 6 + 2       # 目的の対 6 + 残差分散の固定 2
    for n in OBJECTIVES:
        assert f"{n} ~ " in spec
    assert "q_stab =~ 1*dpu_stab" in spec and "dpu_stab ~~ 0*dpu_stab" in spec


def test_joint_fit_residual_correlations_small_without_common_factor():
    stats, est = fit_objective_model_joint(_panel3())
    r = factor_residual_correlations(est)
    k = len(OBJECTIVES)
    assert r.shape == (k, k) and np.allclose(np.diag(r), 1) and np.allclose(r, r.T)
    assert list(r.index) == list(OBJECTIVES)
    assert (np.abs(r.to_numpy()[~np.eye(k, dtype=bool)]) < 0.3).all(), r
    s = stats["Value"] if "Value" in stats.columns else stats.iloc[:, 0]
    assert float(s["CFI"]) > 0.95


def test_rolling_validation_by_objectives():
    ic, scores = rolling_validation(_panel3(n_periods=12), horizon_periods=2, min_train_periods=6,
                                    objectives=OBJECTIVES)
    assert len(ic) >= 3 and "error" not in ic
    for n in OBJECTIVES:
        assert f"status_{n}" in ic and f"ic_composite_{n}" in ic and f"q5_minus_q1_{n}" in ic
        assert f"score_{n}" in scores
        assert (ic[f"status_{n}"] == "ok").all()
    assert "q_ret" in status_summary(ic) and "ic_composite_q_stab" in report(ic)


def test_rolling_validation_rejects_branch_with_objectives():
    try:
        rolling_validation(_panel3(), branch=True, objectives=OBJECTIVES)
    except ValueError:
        return
    raise AssertionError("branch と objectives の同時指定を拒否していない")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
