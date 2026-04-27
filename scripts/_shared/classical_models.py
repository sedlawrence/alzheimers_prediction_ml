from pathlib import Path
import runpy


_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_BASELINE = runpy.run_path(str(_SCRIPTS_DIR / "02_task1_baselines.py"))
_XGBOOST = runpy.run_path(str(_SCRIPTS_DIR / "03_task1_xgboost.py"))
_LINEAR = runpy.run_path(str(_SCRIPTS_DIR / "04_task1_svm_and_linear_models.py"))

get_baseline_models = _BASELINE["get_models"]
get_baseline_experiments = _BASELINE["get_experiments"]

get_xgboost_experiments = _XGBOOST["get_experiments"]
make_xgboost_classifier = _XGBOOST["make_xgboost_classifier"]
build_xgboost_model = _XGBOOST["build_model"]
evaluate_xgboost_cv = _XGBOOST["evaluate_xgboost_cv"]
make_xgboost_summary_row = _XGBOOST["make_summary_row"]
get_xgboost_effective_n_features = _XGBOOST["get_effective_n_features"]
THRESHOLDS = _XGBOOST["THRESHOLDS"]

get_linear_models = _LINEAR["get_models"]
get_linear_experiments = _LINEAR["get_experiments"]
PLSLogisticClassifier = _LINEAR["PLSLogisticClassifier"]
get_linear_effective_n_features = _LINEAR["get_effective_n_features"]
