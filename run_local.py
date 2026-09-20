"""ローカル検証: 合成データで (1) 符号判定つきフィット (2) 時系列分割検証 を回す.

  python run_local.py              6指標が同じ向き  → 1因子で統合
  python run_local.py --opposite   安定性2指標が逆向き → 2因子へ分岐
"""
import sys

from jreit_score.features import OUTCOME_COLS
from jreit_score.model import (DEFAULT_CAUSES, cross_sectional_standardize,
                               fit_with_sign_branch, summarize_branch)
from jreit_score.synthetic import make_panel
from jreit_score.validation import report, rolling_validation

opposite = "--opposite" in sys.argv
panel = make_panel(opposite_sign_for_stability=opposite)
panel = cross_sectional_standardize(panel, OUTCOME_COLS + DEFAULT_CAUSES)

print("=== in-sample fit (all periods) ===")
b = fit_with_sign_branch(panel)
print(summarize_branch(b))

print("\n=== rolling out-of-sample validation (符号判定つき) ===")
ic, scores = rolling_validation(panel, horizon_periods=2, min_train_periods=8, branch=True)
if "decision" in ic:
    print("  判定の内訳: " + ", ".join(f"{k}={v}" for k, v in ic["decision"].value_counts().items()))
print(report(ic))
if "error" in ic:
    print("errors:", ic["error"].notna().sum())
