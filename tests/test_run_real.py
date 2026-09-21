"""run_real の診断関数の検証（ネットワーク不要）."""
import numpy as np
import pandas as pd

from jreit_score.features import OUTCOME_COLS
from jreit_score.ingest.jquants_panel import JQ_CAUSES
from jreit_score.run_real import coverage, indicator_structure


def _panel(n=40, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n)
    df = pd.DataFrame({"code": [f"{i:04d}" for i in range(n)], "period": pd.Timestamp("2024-06-30")})
    for c in OUTCOME_COLS:
        df[c] = f + rng.normal(scale=0.5, size=n)
    for c in JQ_CAUSES:
        df[c] = rng.normal(size=n)
    return df


def test_indicator_structure_shapes_and_symmetry():
    among, with_causes, eig = indicator_structure(_panel())
    assert among.shape == (6, 6) and list(among.index) == OUTCOME_COLS
    assert with_causes.shape == (6, 2) and list(with_causes.columns) == JQ_CAUSES
    assert np.allclose(among.to_numpy(), among.to_numpy().T)
    assert np.allclose(np.diag(among.to_numpy()), 1.0)
    assert len(eig) == 6 and abs(eig.sum() - 6.0) < 1e-9 and (np.diff(eig) <= 1e-12).all()


def test_indicator_structure_one_common_factor_has_one_large_eigenvalue():
    _, _, eig = indicator_structure(_panel())
    assert eig[0] > 3.0 and (eig[1:] < 1.0).all()


def test_indicator_structure_drops_rows_with_missing_values():
    df = _panel()
    df.loc[0, "ret_6m"] = np.nan
    df.loc[1, "nav_ratio"] = np.nan
    among, _, _ = indicator_structure(df)
    full = indicator_structure(df.dropna())[0]
    assert np.allclose(among.to_numpy(), full.to_numpy())


def test_coverage_counts_complete_rows_per_period():
    df = _panel(n=5)
    df2 = df.copy(); df2["period"] = pd.Timestamp("2024-12-31"); df2.loc[0, "dpu_stab"] = np.nan
    cov = coverage(pd.concat([df, df2]), OUTCOME_COLS + JQ_CAUSES)
    assert cov.tolist() == [5, 4]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
