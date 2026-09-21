"""公開ダッシュボード用のスコア（派生値のみ）を組み立てる.

出すもの: 銘柄コード・銘柄名、目的別スコアの z 値（銘柄間で標準化）と五分位、
モデルの係数 β̂（標準化データ上）、時系列検証の IC。
出さないもの: 価格・分配金・BPS・時価総額・NAV 倍率などの生データと、その z 値。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import OBJECTIVE_LABELS, OBJECTIVES, OUTCOME_COLS
from .ingest.jquants_panel import JQ_CAUSES, causes_panel, half_year_ends
from .model import (OBJECTIVE_STATUS_LABEL, ObjectiveFactors, cross_sectional_standardize,
                    fit_objective_factors)
from .validation import newey_west_t, rolling_validation

ROW_KEYS_FIXED = {"code", "name"}          # 行に許す固定キー。残りは目的キーだけ
CELL_KEYS = {"z", "q"}                     # 目的セルに許すキー


def _round(v, nd=3):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def oos_summary(ic: pd.DataFrame, key: str) -> dict | None:
    """rolling_validation の結果から目的 key の OOS 要約（平均 IC と Newey–West t）."""
    counts = ic[f"status_{key}"].value_counts().to_dict() if f"status_{key}" in ic else {}
    out = {"ic": None, "ic_t": None, "spread": None, "spread_t": None, "n_periods": 0,
           "status_counts": {str(k): int(v) for k, v in counts.items()}}
    col = f"ic_composite_{key}"
    if col in ic.columns:                      # どの期でも使えなかった目的には列が無い
        m, t = newey_west_t(ic[col])
        m2, t2 = newey_west_t(ic[f"q5_minus_q1_{key}"])
        out.update(ic=_round(m), ic_t=_round(t, 2), spread=_round(m2), spread_t=_round(t2, 2),
                   n_periods=int(ic[col].notna().sum()))
    return out


def quintiles(z: pd.Series) -> pd.Series:
    """1〜5（5 が上位）. 5 銘柄未満なら付けない."""
    if len(z) < 5:
        return pd.Series([None] * len(z), index=z.index, dtype=object)
    return (pd.qcut(z.rank(method="first"), 5, labels=False) + 1).astype(int)


def snapshot_payload(o: ObjectiveFactors, ic: pd.DataFrame, X: pd.DataFrame, names: dict[str, str],
                     as_of: pd.Timestamp, fit: dict, objectives: dict[str, list[str]] = OBJECTIVES) -> dict:
    """推定済みモデルと基準日の説明変数 X（z 化済み, NaN 無し）からスコア payload を作る."""
    scores = o.predict_scores(X)
    usable = [k for k in objectives if k in scores.columns]
    zs, qs = {}, {}
    for k in usable:
        s = scores[k]
        sd = s.std()
        zs[k] = (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0
        qs[k] = quintiles(zs[k])

    rows = []
    for idx, code in X["code"].items():
        row = {"code": str(code), "name": str(names.get(str(code), ""))}
        for k in usable:
            q = qs[k].loc[idx]
            row[k] = {"z": _round(zs[k].loc[idx], 2), "q": None if q is None else int(q)}
        rows.append(row)
    rows.sort(key=lambda r: r["code"])

    objs = {}
    for k, inds in objectives.items():
        fo = o.factors[k]
        entry = {"label": OBJECTIVE_LABELS.get(k, k), "indicators": list(inds),
                 "status": fo.status, "status_label": OBJECTIVE_STATUS_LABEL[fo.status],
                 "reason": fo.reason, "usable": fo.status == "ok"}
        if fo.fitted is not None:
            entry["beta"] = {c: _round(fo.fitted.beta.get(c, np.nan)) for c in o.causes}
            entry["n"] = int(getattr(fo.fitted, "n", 0))
        entry["oos"] = oos_summary(ic, k)
        objs[k] = entry

    return {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "fit": dict(fit),
        "causes": list(o.causes),
        "objectives": objs,
        "columns": usable,
        "rows": rows,
    }


def score_snapshot(store, jgb10: pd.DataFrame, start: str = "2017-06-30", end: str = "2026-06-30",
                   min_train: int = 6, objectives: dict[str, list[str]] = OBJECTIVES,
                   causes: list[str] = JQ_CAUSES) -> dict:
    """store（J-Quants）と 10 年債から, 目的別モデルを推定して最新営業日のスコアを出す.

    推定は目的変数が実現済みの期だけ（`build_panel` が未到達の期を NaN にする）。
    スコアの基準日は価格の最終日で、説明変数はその日時点の as-of 値。
    """
    from .run_real import build_panel  # 循環 import 回避

    periods = half_year_ends(start, end)
    panel = build_panel(store, jgb10, periods)
    cols = list(OUTCOME_COLS) + list(causes)
    panel = cross_sectional_standardize(panel, cols)
    complete = panel.dropna(subset=cols)
    o = fit_objective_factors(panel, objectives, causes)
    ic, _ = rolling_validation(panel, horizon_periods=2, min_train_periods=min_train,
                               causes=causes, objectives=objectives)

    as_of = pd.Timestamp(store.prices["date"].max())
    X = causes_panel(store.prices, store.dpu, [as_of])
    unscored = sorted(X.loc[X[causes].isna().any(axis=1), "code"].astype(str))
    X = X.dropna(subset=causes)
    X = cross_sectional_standardize(X, causes)
    names = dict(zip(store.universe["code"].astype(str), store.universe.get("name", pd.Series(dtype=str)).astype(str))) \
        if "name" in store.universe.columns else {}
    fit = {
        "periods": [complete["period"].min().strftime("%Y-%m-%d"), complete["period"].max().strftime("%Y-%m-%d")]
        if len(complete) else [],
        "n_periods": int(complete["period"].nunique()),
        "n_rows_complete": int(len(complete)),
        "universe": int(store.prices["code"].nunique()),
        "scored": int(len(X)),
        "unscored_codes": unscored,
        "min_train_periods": int(min_train),
    }
    return snapshot_payload(o, ic, X, names, as_of, fit, objectives)


def check_payload(scores: dict, objectives: dict[str, list[str]] = OBJECTIVES) -> None:
    """公開してよい形かを構造で確認する（生データのキーが紛れ込んでいないこと）."""
    assert set(scores["columns"]) <= set(objectives), scores["columns"]
    for r in scores["rows"]:
        extra = set(r) - ROW_KEYS_FIXED - set(objectives)
        assert not extra, f"行に許されないキー: {sorted(extra)}"
        for k in objectives:
            if k in r:
                assert set(r[k]) == CELL_KEYS, sorted(r[k])
    for k, e in scores["objectives"].items():
        assert k in objectives and set(e["indicators"]) == set(objectives[k])
        assert e["status"] in OBJECTIVE_STATUS_LABEL
