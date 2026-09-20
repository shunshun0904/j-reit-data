"""ローカル検証: 合成データで (1) 全期間フィット (2) 時系列分割検証 を回す."""
import sys
from jreit_score.synthetic import make_panel
from jreit_score.model import fit_mimic, summarize, cross_sectional_standardize
from jreit_score.validation import rolling_validation, report
from jreit_score.features import OUTCOME_COLS
from jreit_score.model import DEFAULT_CAUSES

opposite = "--opposite" in sys.argv
panel = make_panel(opposite_sign_for_stability=opposite)
panel = cross_sectional_standardize(panel, OUTCOME_COLS + DEFAULT_CAUSES)

print("=== in-sample fit (all periods) ===")
f = fit_mimic(panel)
print(summarize(f))

print("\n=== rolling out-of-sample validation ===")
ic, scores = rolling_validation(panel, horizon_periods=2, min_train_periods=8)
print(report(ic))
if "error" in ic:
    print("errors:", ic["error"].notna().sum())
