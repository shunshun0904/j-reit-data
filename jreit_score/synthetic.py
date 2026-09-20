"""パイプライン動作確認用の合成パネル. 実データが揃うまでの代替."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import OUTCOME_COLS
from .model import DEFAULT_CAUSES


def make_panel(n_codes: int = 58, n_periods: int = 30, seed: int = 0,
               true_beta: dict | None = None, opposite_sign_for_stability: bool = False) -> pd.DataFrame:
    """X → quality → 6指標 の生成過程に従う合成データ.

    opposite_sign_for_stability=True にすると, 安定性2指標が quality と逆向きに載る
    （1因子では適合しないケースの再現）。
    """
    rng = np.random.default_rng(seed)
    beta = true_beta or {"nav_ratio": -0.4, "ltv": -0.3, "fixed_rate_ratio": 0.2, "unrealized_gain": 0.3,
                         "noi_yield": 0.2, "occupancy": 0.15, "log_mcap": 0.1}
    periods = pd.date_range("2011-06-30", periods=n_periods, freq="6ME")
    rows = []
    for p in periods:
        X = rng.normal(size=(n_codes, len(DEFAULT_CAUSES)))
        eta = X @ np.array([beta[c] for c in DEFAULT_CAUSES]) + rng.normal(scale=0.7, size=n_codes)
        lam = np.array([0.7, 0.8, 0.5, 0.4, 0.6, 0.5])
        if opposite_sign_for_stability:
            lam[2:4] *= -1
        Y = np.outer(eta, lam) + rng.normal(scale=0.8, size=(n_codes, 6))
        df = pd.DataFrame(X, columns=DEFAULT_CAUSES)
        df[OUTCOME_COLS] = Y
        df["code"] = [f"{8950 + i}" for i in range(n_codes)]
        df["period"] = p
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
