"""符号割れ判定と2因子分岐の検証（ネットワーク不要, 合成データのみ）."""
import numpy as np
import pandas as pd

from jreit_score.features import OUTCOME_COLS
from jreit_score.model import (DEFAULT_CAUSES, check_sign_split,
                               cross_sectional_standardize, fit_mimic,
                               fit_with_sign_branch, two_factor_spec)

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


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
