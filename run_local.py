"""ローカル検証: 合成データで (1) 符号判定つきフィット (2) 時系列分割検証 を回す.

  python run_local.py              6指標が同じ向き  → 1因子で統合
  python run_local.py --opposite   安定性2指標が逆向き → 2因子へ分岐
  python run_local.py --objectives 目的別（4 目的。合成データでは全目的が使える）
"""
import sys

from jreit_score.features import OBJECTIVES, OUTCOME_COLS
from jreit_score.model import (DEFAULT_CAUSES, cross_sectional_standardize,
                               fit_objective_factors, fit_with_sign_branch,
                               summarize_branch, summarize_objectives)
from jreit_score.synthetic import make_panel
from jreit_score.validation import report, rolling_validation, status_summary

opposite = "--opposite" in sys.argv
panel = make_panel(opposite_sign_for_stability=opposite)
panel = cross_sectional_standardize(panel, OUTCOME_COLS + DEFAULT_CAUSES)

if "--objectives" in sys.argv:
    print("=== 目的別 (all periods) ===")
    print(summarize_objectives(fit_objective_factors(panel)))
    print("\n=== rolling out-of-sample validation (目的別) ===")
    ic, _ = rolling_validation(panel, horizon_periods=2, min_train_periods=8, objectives=OBJECTIVES)
    print(status_summary(ic))
    print(report(ic))
    sys.exit(0)

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
