"""MIMIC 型 SEM: 潜在変数 quality を 6 指標で測定し, 財務指標で説明する."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import semopy

from .features import OUTCOME_COLS

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
        params=est, stats=stats.T,
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
