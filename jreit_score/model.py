"""MIMIC 型 SEM: 潜在変数 quality を 6 指標で測定し, 財務指標で説明する."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import semopy
import statsmodels.api as sm

from .features import OBJECTIVES, OUTCOME_COLS

DEFAULT_CAUSES = ["nav_ratio", "ltv", "fixed_rate_ratio", "unrealized_gain",
                  "noi_yield", "occupancy", "log_mcap"]


def model_spec(indicators: list[str] = OUTCOME_COLS, causes: list[str] = DEFAULT_CAUSES) -> str:
    return (
        f"quality =~ {' + '.join(indicators)}\n"
        f"quality ~ {' + '.join(causes)}\n"
    )


def cross_sectional_standardize(df: pd.DataFrame, cols: list[str], by: str = "period") -> pd.DataFrame:
    """期ごとに z 化する（期固定効果の除去と、指標間スケール統一を兼ねる）."""
    out = df.copy()
    g = out.groupby(by)[cols]
    out[cols] = (out[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return out


@dataclass
class FittedMIMIC:
    model: semopy.Model
    causes: list[str]
    indicators: list[str]
    loadings: pd.Series = field(default_factory=pd.Series)
    beta: pd.Series = field(default_factory=pd.Series)
    params: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    n: int = 0

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        """構造方程式のみで η̂ = β̂·X を返す（将来情報を使わない運用スコア）."""
        return X[self.causes].fillna(0.0) @ self.beta.reindex(self.causes).fillna(0.0)


def fit_mimic(df: pd.DataFrame, indicators=OUTCOME_COLS, causes=DEFAULT_CAUSES) -> FittedMIMIC:
    cols = list(indicators) + list(causes)
    data = df.dropna(subset=cols)[cols].astype(float)
    m = semopy.Model(model_spec(indicators, causes))
    m.fit(data)
    est = m.inspect(std_est=True)
    load = est[(est["op"] == "~") & (est["lval"].isin(indicators)) & (est["rval"] == "quality")]
    beta = est[(est["op"] == "~") & (est["lval"] == "quality") & (est["rval"].isin(causes))]
    stats = semopy.calc_stats(m)
    return FittedMIMIC(
        model=m, causes=list(causes), indicators=list(indicators),
        loadings=load.set_index("lval")["Estimate"],
        beta=beta.set_index("rval")["Estimate"],
        params=est, stats=stats.T, n=int(len(data)),
    )


def _p(v):
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return str(v)


def summarize(f: FittedMIMIC) -> str:
    est = f.params
    lines = ["[measurement] loadings (std.)"]
    for _, r in est[(est["op"] == "~") & (est["rval"] == "quality")].iterrows():
        lines.append(f"  {r['lval']:<12} λ={r['Est. Std']:+.3f}  p={_p(r['p-value'])}")
    lines.append("[structural] quality ~ causes (std.)")
    for _, r in est[(est["op"] == "~") & (est["lval"] == "quality")].iterrows():
        lines.append(f"  {r['rval']:<16} β={r['Est. Std']:+.3f}  p={_p(r['p-value'])}")
    lines.append(_fit_line(f.stats))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 符号割れの判定と2因子への分岐
#
# 判定ルール（CLAUDE.md「決定済みの設計」）:
#   適合度が良くても負荷量の符号が割れたら 1因子で統合しない → 2因子へ。
#   適合度指標だけでは符号の割れを検出できない（run_local.py --opposite で再現）。
# ---------------------------------------------------------------------------

SIGN_SPLIT_MIN_ABS_LOADING = 0.15  # 標準化負荷量がこれ未満の指標は「向きが無い」として判定に使わない
SIGN_SPLIT_ALPHA = 0.05            # 逆符号が有意でなければ割れと見なさない
MIN_GROUP_SIZE = 2                 # 1因子あたり最低2指標（測定モデルの識別のため）


def _pval(v) -> float:
    """semopy の p-value を float 化する. 固定パラメータは '-' なので NaN を返す."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _std_loadings(params: pd.DataFrame, indicators: list[str],
                  factor: str = "quality") -> tuple[pd.Series, pd.Series]:
    """inspect 結果から, 指標 × 因子 の標準化負荷量と p 値を取り出す.

    semopy の inspect は測定方程式を `指標 ~ 因子` (op='~') として返す。
    第1指標は識別のため 1 に固定され, p-value は '-' になる。
    """
    m = params[(params["op"] == "~") & (params["rval"] == factor)
               & (params["lval"].isin(indicators))]
    lam = m.set_index("lval")["Est. Std"].astype(float).reindex(indicators)
    p = m.set_index("lval")["p-value"].map(_pval).reindex(indicators)
    return lam, p


@dataclass
class SignCheck:
    """1因子モデルの負荷量が同じ向きに揃っているかの判定結果."""
    loadings: pd.Series          # 標準化負荷量（index=指標）
    pvalues: pd.Series           # 固定パラメータは NaN
    majority: list[str]          # 多数派の向きの指標
    minority: list[str]          # 逆向きの指標
    decisive: list[str]          # 閾値と有意水準の両方を満たし, 判定に使った指標
    split: bool                  # 逆向きの指標が判定に耐える形で存在するか
    reason: str

    def groups(self) -> tuple[list[str], list[str]]:
        return list(self.majority), list(self.minority)


def check_sign_split(f: FittedMIMIC,
                     min_abs_loading: float = SIGN_SPLIT_MIN_ABS_LOADING,
                     alpha: float = SIGN_SPLIT_ALPHA) -> SignCheck:
    """1因子モデルの負荷量の符号が割れているかを判定する.

    潜在因子の符号自体は任意（semopy は第1指標の負荷量を1に固定する）なので,
    「どちらが正か」ではなく「全指標が同じ向きに載っているか」だけを見る。

    判定に使うのは |λ| >= min_abs_loading かつ p < alpha の指標に限る。
    弱い負荷量や非有意な負荷量の符号はノイズで反転しうるため, これらで
    2因子に落とすと過剰分岐になる。固定パラメータ（p='-'）は識別のための
    基準指標なので有意扱いにする。
    """
    lam, p = _std_loadings(f.params, f.indicators)
    decisive = [i for i in f.indicators
                if np.isfinite(lam.get(i, np.nan)) and abs(lam[i]) >= min_abs_loading
                and (not np.isfinite(p.get(i, np.nan)) or p[i] < alpha)]
    if len(decisive) < 2:
        return SignCheck(lam, p, list(f.indicators), [], decisive, False,
                         f"判定に使える指標が {len(decisive)} 本しかない（|λ|>={min_abs_loading}, p<{alpha}）")

    pos = sum(abs(lam[i]) for i in decisive if lam[i] > 0)
    neg = sum(abs(lam[i]) for i in decisive if lam[i] < 0)
    major_sign = 1.0 if pos >= neg else -1.0
    majority = [i for i in f.indicators if np.isfinite(lam.get(i, np.nan)) and np.sign(lam[i]) == major_sign]
    minority = [i for i in f.indicators if i not in majority]
    decisive_minority = [i for i in minority if i in decisive]

    if decisive_minority:
        reason = ("負荷量の符号が割れている: "
                  + ", ".join(f"{i} λ={lam[i]:+.3f}" for i in decisive_minority)
                  + f" が多数派（{len(majority)}本）と逆向き")
    elif minority:
        reason = ("逆向きの指標はあるが判定閾値に届かない: "
                  + ", ".join(f"{i} λ={lam[i]:+.3f} p={p[i]:.3g}" for i in minority))
    else:
        reason = f"全 {len(f.indicators)} 指標が同じ向きに載っている"
    return SignCheck(lam, p, majority, minority, decisive, bool(decisive_minority), reason)


def two_factor_spec(group1: list[str], group2: list[str], causes: list[str]) -> str:
    """2因子 MIMIC の仕様.

    `q1 ~~ q2` は必須。semopy は内生潜在変数どうしの残差共分散を自動では
    追加しないため, 省くと2因子が独立と制約され適合が崩れる
    （合成データ --opposite で確認: 省略時 chi2 p=0.000 / CFI=0.895 / RMSEA=0.048,
      明示時 chi2 p=0.996 / CFI=1.010 / RMSEA=0.000）。
    """
    c = " + ".join(causes)
    return (
        f"q1 =~ {' + '.join(group1)}\n"
        f"q2 =~ {' + '.join(group2)}\n"
        f"q1 ~ {c}\n"
        f"q2 ~ {c}\n"
        "q1 ~~ q2\n"
    )


@dataclass
class FittedTwoFactor:
    model: semopy.Model
    causes: list[str]
    groups: dict[str, list[str]]                       # {"q1": [...], "q2": [...]}
    loadings: pd.Series = field(default_factory=pd.Series)   # 標準化負荷量（向き補正後）
    beta: pd.DataFrame = field(default_factory=pd.DataFrame)  # index=説明変数, columns=[q1, q2]
    flipped: dict[str, bool] = field(default_factory=dict)    # 向きを反転した因子
    params: pd.DataFrame = field(default_factory=pd.DataFrame)  # semopy の生の推定値（反転前）
    stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def factors(self) -> list[str]:
        return list(self.groups)

    def predict_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """因子ごとに η̂ = β̂·X を返す（将来情報を使わない運用スコア）.

        2因子に分かれた時点で1本に束ねる根拠が無いので, 合成はしない。
        因子間でスケールは揃っていない（順位での評価を想定）。
        """
        Xc = X[self.causes].fillna(0.0)
        return pd.DataFrame({f: Xc @ self.beta[f].reindex(self.causes).fillna(0.0)
                             for f in self.factors}, index=X.index)


def fit_two_factor(df: pd.DataFrame, group1: list[str], group2: list[str],
                   causes=DEFAULT_CAUSES) -> FittedTwoFactor:
    """指標を2群に分けた2因子 MIMIC を推定する.

    各因子は「負荷量の合計が正」になる向きに揃える。潜在変数の符号は識別上
    任意なので, 揃えないと因子ごとにスコアの高低の意味が変わる。
    """
    groups = {"q1": list(group1), "q2": list(group2)}
    cols = list(group1) + list(group2) + list(causes)
    data = df.dropna(subset=cols)[cols].astype(float)
    m = semopy.Model(two_factor_spec(group1, group2, causes))
    m.fit(data)
    est = m.inspect(std_est=True)

    loadings, beta, flipped = {}, {}, {}
    for fac, inds in groups.items():
        lam, _ = _std_loadings(est, inds, fac)
        flip = bool(np.nansum(lam.values) < 0)
        flipped[fac] = flip
        sgn = -1.0 if flip else 1.0
        for i in inds:
            loadings[i] = sgn * lam[i]
        b = est[(est["op"] == "~") & (est["lval"] == fac) & (est["rval"].isin(causes))]
        beta[fac] = sgn * b.set_index("rval")["Estimate"].astype(float).reindex(causes)

    return FittedTwoFactor(
        model=m, causes=list(causes), groups=groups,
        loadings=pd.Series(loadings), beta=pd.DataFrame(beta, index=list(causes)),
        flipped=flipped, params=est, stats=semopy.calc_stats(m).T,
    )


@dataclass
class BranchResult:
    """1因子で統合するか2因子に落とすかの判定つきの推定結果."""
    one_factor: FittedMIMIC
    sign_check: SignCheck
    decision: str            # "one_factor" | "two_factor" | "two_factor_unavailable"
    reason: str
    two_factor: FittedTwoFactor | None = None

    @property
    def factors(self) -> list[str]:
        return self.two_factor.factors if self.decision == "two_factor" else ["quality"]

    def predict_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """採用したモデルのスコアを DataFrame で返す（1因子なら1列）."""
        if self.decision == "two_factor":
            return self.two_factor.predict_scores(X)
        return self.one_factor.predict_score(X).to_frame("quality")

    def indicator_groups(self) -> dict[str, list[str]]:
        if self.decision == "two_factor":
            return dict(self.two_factor.groups)
        return {"quality": list(self.one_factor.indicators)}


def fit_with_sign_branch(df: pd.DataFrame, indicators=OUTCOME_COLS, causes=DEFAULT_CAUSES,
                         min_abs_loading: float = SIGN_SPLIT_MIN_ABS_LOADING,
                         alpha: float = SIGN_SPLIT_ALPHA,
                         min_group_size: int = MIN_GROUP_SIZE) -> BranchResult:
    """1因子を推定し, 負荷量の符号が割れていれば2因子に落とす.

    適合度は分岐の判断に使わない。適合度が良くても符号が割れる場合があり,
    その場合に1因子で統合すると, 逆向きの指標を打ち消したスコアになる。
    """
    f1 = fit_mimic(df, indicators, causes)
    chk = check_sign_split(f1, min_abs_loading, alpha)
    if not chk.split:
        return BranchResult(f1, chk, "one_factor", chk.reason)

    g1, g2 = chk.groups()
    if min(len(g1), len(g2)) < min_group_size:
        return BranchResult(
            f1, chk, "two_factor_unavailable",
            f"{chk.reason}。ただし片群が {min(len(g1), len(g2))} 本で "
            f"{min_group_size} 本に満たず2因子を識別できない。"
            "1因子の推定値を返すが、このスコアは信用しないこと",
        )
    f2 = fit_two_factor(df, g1, g2, causes)
    return BranchResult(f1, chk, "two_factor", chk.reason, two_factor=f2)


def _fit_line(stats: pd.DataFrame) -> str:
    s = stats["Value"] if "Value" in stats.columns else stats.iloc[:, 0]
    keys = ["DoF", "chi2", "chi2 p-value", "CFI", "TLI", "RMSEA"]
    return "[fit] " + ", ".join(f"{k}={float(s[k]):.3f}" for k in keys if k in s.index)


def summarize_two_factor(f: FittedTwoFactor) -> str:
    lines = []
    for fac, inds in f.groups.items():
        flip = "（向き反転済み）" if f.flipped.get(fac) else ""
        lines.append(f"[measurement] {fac} =~ {', '.join(inds)}{flip}")
        for i in inds:
            lines.append(f"  {i:<12} λ={f.loadings[i]:+.3f}")
    lines.append("[structural] 各因子 ~ 説明変数 (raw)")
    lines.append(f"  {'':<16}" + "".join(f"{fac:>12}" for fac in f.factors))
    for c in f.causes:
        lines.append(f"  {c:<16}" + "".join(f"{f.beta.loc[c, fac]:>+12.3f}" for fac in f.factors))
    lines.append(_fit_line(f.stats))
    return "\n".join(lines)


def summarize_branch(b: BranchResult) -> str:
    label = {"one_factor": "1因子で統合する",
             "two_factor": "1因子で統合しない → 2因子",
             "two_factor_unavailable": "1因子で統合できないが2因子も識別できない"}[b.decision]
    lines = ["=== 1因子モデル ===", summarize(b.one_factor),
             "", f"=== 符号判定: {label} ===", f"  {b.sign_check.reason}"]
    if b.two_factor is not None:
        lines += ["", "=== 2因子モデル ===", summarize_two_factor(b.two_factor)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 目的別の因子（決定 2026-09-21）
#
# 実データでは 6 指標に共通因子が無かった（目的をまたぐ Spearman 相関が全て 0.10 以下、
# 固有値 1.81 / 1.50 / 1.05）。符号判定は「割れていない」と言うだけで、統合できるとは
# 言わない。そこで目的ごとに 2 指標の MIMIC を別々に推定し、統合スコアは作らない。
#
# 2 指標の因子は、説明変数との共分散を通じてしか自由な負荷量が識別されない
# （cov(y2, x) / cov(y1, x) = λ2）。説明変数が因子を説明しなければ λ は決まらず、
# 標準誤差が出ないか巨大になる。その因子は「識別不能」として報告し、スコアを出さない。
# ---------------------------------------------------------------------------

OBJECTIVE_STATUS_LABEL = {
    "ok": "推定可",
    "sign_split": "2 指標の向きが逆（1 つの因子として不整合）",
    "weak": "負荷量が判定に耐えない（識別不能）",
    "improper": "不適切解（標準化負荷量が 1 を超える）",
    "no_signal": "説明変数に有意なものが無い（スコアはノイズ）",
    "fit_failed": "推定失敗",
}


@dataclass
class FittedRegression:
    """1 指標の目的: 指標を説明変数に回帰する（測定モデルが無いので MIMIC は退化して OLS）.

    スコアは MIMIC と同じ形 η̂ = β̂·X。定数項は期ごとの z 化で 0 になるが念のため入れる
    （スコアには使わない）。
    """
    causes: list[str]
    indicators: list[str]                # 1 本
    beta: pd.Series = field(default_factory=pd.Series)
    pvalues: pd.Series = field(default_factory=pd.Series)
    r2: float = np.nan
    n: int = 0
    f_pvalue: float = np.nan             # 全係数 0 の F 検定

    def predict_score(self, X: pd.DataFrame) -> pd.Series:
        return X[self.causes].fillna(0.0) @ self.beta.reindex(self.causes).fillna(0.0)


def fit_regression(df: pd.DataFrame, indicator: str, causes=DEFAULT_CAUSES) -> FittedRegression:
    cols = [indicator] + list(causes)
    data = df.dropna(subset=cols)[cols].astype(float)
    res = sm.OLS(data[indicator].to_numpy(), sm.add_constant(data[list(causes)].to_numpy())).fit()
    names = ["const"] + list(causes)
    beta = pd.Series(res.params, index=names).drop("const")
    pv = pd.Series(res.pvalues, index=names).drop("const")
    return FittedRegression(list(causes), [indicator], beta, pv, float(res.rsquared), int(res.nobs),
                            float(res.f_pvalue))


def summarize_regression(f: FittedRegression) -> str:
    lines = [f"[regression] {f.indicators[0]} ~ causes (z 化済みデータの係数)"]
    for c in f.causes:
        lines.append(f"  {c:<16} β={f.beta[c]:+.3f}  p={_p(f.pvalues[c])}")
    lines.append(f"[fit] R2={f.r2:.3f}, F-test p={_p(f.f_pvalue)}, n={f.n}")
    return "\n".join(lines)


def summarize_fitted(f) -> str:
    return summarize_regression(f) if isinstance(f, FittedRegression) else summarize(f)


@dataclass
class ObjectiveFactor:
    name: str
    indicators: list[str]
    status: str                      # OBJECTIVE_STATUS_LABEL のキー
    reason: str
    fitted: FittedMIMIC | FittedRegression | None = None


@dataclass
class ObjectiveFactors:
    """目的別因子の推定結果. 統合はしない（共通因子が無い）."""
    factors: dict[str, ObjectiveFactor]
    causes: list[str]

    @property
    def usable(self) -> list[str]:
        return [n for n, f in self.factors.items() if f.status == "ok"]

    def predict_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """推定できた因子だけ η̂ = β̂·X を返す（将来情報を使わない運用スコア）.

        識別不能・不整合の因子は列を出さない（黙って出さない）。
        """
        return pd.DataFrame({n: self.factors[n].fitted.predict_score(X) for n in self.usable},
                            index=X.index)

    def indicator_groups(self) -> dict[str, list[str]]:
        return {n: list(self.factors[n].indicators) for n in self.usable}


def _assess_factor(f: FittedMIMIC, min_abs_loading: float, alpha: float) -> tuple[str, str]:
    """1 因子の測定モデルがスコアを出せる状態かを判定する."""
    lam, p = _std_loadings(f.params, f.indicators)
    if lam.isna().any():
        return "fit_failed", "負荷量が得られない"
    if (lam.abs() > 1.0 + 1e-6).any():
        return "improper", "標準化負荷量が 1 を超える: " + ", ".join(
            f"{i} λ={lam[i]:+.3f}" for i in f.indicators if abs(lam[i]) > 1.0 + 1e-6)
    # 第 1 指標は識別のため 1 に固定され p='-'（NaN）。自由な指標の p が NaN なら
    # 標準誤差が計算できていない＝識別不能
    free_nan = [i for i in f.indicators[1:] if not np.isfinite(p.get(i, np.nan))]
    if free_nan:
        return "weak", "標準誤差が計算できない（識別不能）: " + ", ".join(free_nan)
    chk = check_sign_split(f, min_abs_loading, alpha)
    if chk.split:
        return "sign_split", chk.reason
    weak = [i for i in f.indicators if i not in chk.decisive]
    if weak:
        return "weak", "判定に耐えない指標: " + ", ".join(
            f"{i} λ={lam[i]:+.3f} p={p[i]:.3g}" if np.isfinite(p[i]) else f"{i} λ={lam[i]:+.3f} p=-"
            for i in weak)
    # 構造方程式に信号が無ければスコアはノイズ。説明変数が多いと偶然の 5% 有意が
    # 出やすいので Bonferroni（alpha / 本数）で見る
    b = f.params[(f.params["op"] == "~") & (f.params["lval"] == "quality")
                 & (f.params["rval"].isin(f.causes))]
    if not (b["p-value"].map(_pval) < alpha / max(len(f.causes), 1)).any():
        return "no_signal", "説明変数に有意なものが無い（Bonferroni p>=%.2g/%d）" % (alpha, len(f.causes))
    return "ok", chk.reason


def _assess_regression(f: FittedRegression, alpha: float) -> tuple[str, str]:
    """回帰には測定モデルが無いので, 判定は「説明変数に信号があるか」（全係数 0 の F 検定）だけ."""
    if f.beta.isna().any() or not np.isfinite(f.f_pvalue):
        return "fit_failed", "係数が得られない"
    # 信号が 1 本に集中していると F 検定は弱いので、Bonferroni の個別検定でも通す
    k = max(len(f.causes), 1)
    strong = [c for c in f.causes if np.isfinite(f.pvalues[c]) and f.pvalues[c] < alpha / k]
    if f.f_pvalue >= alpha and not strong:
        return "no_signal", "説明変数に信号が無い（F 検定 p=%.2g, Bonferroni 個別も無し）" % f.f_pvalue
    sig = [c for c in f.causes if np.isfinite(f.pvalues[c]) and f.pvalues[c] < alpha]
    return "ok", "F 検定 p=%.2g。有意な説明変数: %s" % (f.f_pvalue, ", ".join(sig) if sig else "（個別には無し）")


def fit_objective_factors(df: pd.DataFrame, objectives: dict[str, list[str]] = OBJECTIVES,
                          causes=DEFAULT_CAUSES,
                          min_abs_loading: float = SIGN_SPLIT_MIN_ABS_LOADING,
                          alpha: float = SIGN_SPLIT_ALPHA) -> ObjectiveFactors:
    """目的ごとに別々に推定し, 目的ごとに使えるかを判定する.

    2 指標以上の目的は 1 因子の MIMIC、1 指標の目的は説明変数への回帰。
    別々に推定する理由: 因子間の相関が 0.1 程度しか無いので同時推定しても β̂ はほぼ
    変わらず, 一方で識別不能な因子が他の因子の推定を巻き込まない。
    適合度の参考には `fit_objective_model_joint` で同時推定できる。
    """
    out: dict[str, ObjectiveFactor] = {}
    for name, inds in objectives.items():
        try:
            if len(inds) == 1:
                f = fit_regression(df, inds[0], causes)
                status, reason = _assess_regression(f, alpha)
            else:
                f = fit_mimic(df, list(inds), causes)
                status, reason = _assess_factor(f, min_abs_loading, alpha)
        except Exception as e:  # 収束失敗・特異行列など
            out[name] = ObjectiveFactor(name, list(inds), "fit_failed",
                                        f"{type(e).__name__}: {str(e)[:120]}")
            continue
        out[name] = ObjectiveFactor(name, list(inds), status, reason, f)
    return ObjectiveFactors(out, list(causes))


def objective_model_spec(objectives: dict[str, list[str]] = OBJECTIVES,
                         causes=DEFAULT_CAUSES) -> str:
    """全目的を同時に置いたモデルの仕様（適合度の参考用）.

    1 指標の目的は負荷量 1・残差分散 0 の潜在変数にする（`q =~ 1*y`, `y ~~ 0*y`）。
    semopy は潜在変数と観測内生変数の残差共分散を受け付けないため、観測変数のまま
    回帰すると目的間の残差共分散を置けない。
    目的間の残差共分散は明示する（2 因子の場合と同じ理由。semopy は自動で追加しない）。
    """
    c = " + ".join(causes)
    names = list(objectives)
    lines = []
    for n, inds in objectives.items():
        if len(inds) > 1:
            lines.append(f"{n} =~ {' + '.join(inds)}")
        else:
            lines.append(f"{n} =~ 1*{inds[0]}")
            lines.append(f"{inds[0]} ~~ 0*{inds[0]}")
    lines += [f"{n} ~ {c}" for n in names]
    lines += [f"{a} ~~ {b}" for i, a in enumerate(names) for b in names[i + 1:]]
    return "\n".join(lines) + "\n"


def fit_objective_model_joint(df: pd.DataFrame, objectives: dict[str, list[str]] = OBJECTIVES,
                              causes=DEFAULT_CAUSES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全目的の同時推定. 返り値は (適合度, inspect の推定値表). スコアには使わない."""
    cols = [i for inds in objectives.values() for i in inds] + list(causes)
    data = df.dropna(subset=cols)[cols].astype(float)
    m = semopy.Model(objective_model_spec(objectives, causes))
    m.fit(data)
    return semopy.calc_stats(m).T, m.inspect(std_est=True)


def factor_residual_correlations(est: pd.DataFrame, objectives: dict[str, list[str]] = OBJECTIVES) -> pd.DataFrame:
    """同時推定の目的間残差相関（標準化した `a ~~ b`）を目的名の行列にする."""
    names = list(objectives)
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    rows = est[(est["op"] == "~~") & est["lval"].isin(names) & est["rval"].isin(names)]
    for _, r in rows.iterrows():
        if r["lval"] != r["rval"]:
            out.loc[r["lval"], r["rval"]] = out.loc[r["rval"], r["lval"]] = float(r["Est. Std"])
    return out


def summarize_objectives(o: ObjectiveFactors) -> str:
    lines = []
    for name, fo in o.factors.items():
        lines.append(f"=== {name} =~ {' + '.join(fo.indicators)}: "
                     f"{OBJECTIVE_STATUS_LABEL[fo.status]} ===")
        lines.append(f"  {fo.reason}")
        if fo.fitted is not None:
            lines.append(summarize_fitted(fo.fitted))
        lines.append("")
    lines.append("使える因子: " + (", ".join(o.usable) if o.usable else "なし"))
    return "\n".join(lines)
