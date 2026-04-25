#!/usr/bin/env python3

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


H5_PATH = Path("../data/temporal_two_sets_n2000.h5")
OUT_DIR = Path("../results/task1")

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3

# Diagnostic threshold tuning.
# This lets us check whether XGBoost has ranking signal but poor default thresholding.
# Caveat: choosing threshold directly on the held-out fold is optimistic.
THRESHOLDS = np.array([
    0.05, 0.075,
    0.10, 0.125, 0.15, 0.175,
    0.20, 0.225, 0.25, 0.275,
    0.30, 0.35, 0.40, 0.45, 0.50,
])


def load_task1(path: Path):
    """
    Load Task 1:
        Control -> Control = y 0
        Control -> MCI     = y 1

    Original HDF5 array shape:
        features x time x samples

    Returned shape:
        samples x time x features
    """
    with h5py.File(path, "r") as f:
        x_cn_to_cn = f["X_cn_to_cn"][:]
        x_cn_to_mci = f["X_cn_to_mci"][:]
        cpg_ids = f["cpg_ids_cn"][:]

    x0 = np.transpose(x_cn_to_cn, (2, 1, 0))   # y = 0
    x1 = np.transpose(x_cn_to_mci, (2, 1, 0))  # y = 1

    X = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([
        np.zeros(x0.shape[0], dtype=int),
        np.ones(x1.shape[0], dtype=int),
    ])

    cpg_ids = np.array([
        c.decode("utf-8") if isinstance(c, bytes) else str(c)
        for c in cpg_ids
    ])

    return X, y, cpg_ids

def get_effective_n_features(exp, X_exp):
    """
    Number of features actually seen by XGBoost after any pipeline transform.
    """
    if exp["model_type"] == "selectk":
        return exp["k_best"]

    if exp["model_type"] == "pca":
        return exp["pca_components"]

    return X_exp.shape[1]

def make_feature_sets(X):
    """
    X shape:
        samples x time x features

    Returns 2D matrices:
        samples x model_features
    """
    t0 = X[:, 0, :]
    t1 = X[:, 1, :]
    delta = t1 - t0

    return {
        "t0_only": t0,
        "t1_only": t1,
        "delta_only": delta,
        "t0_t1": np.concatenate([t0, t1], axis=1),
        "t0_delta": np.concatenate([t0, delta], axis=1),
        "t0_t1_delta": np.concatenate([t0, t1, delta], axis=1),
    }


def make_xgboost_classifier(y_train):
    """
    Create an XGBoost classifier.

    scale_pos_weight helps with class imbalance:
        n_negative / n_positive

    This is calculated within each CV fold using training labels only.
    """
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",

        # Conservative settings for small-n, high-p methylation data.
        n_estimators=250,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.3,

        # Regularisation.
        reg_lambda=5.0,
        reg_alpha=1.0,
        min_child_weight=3,
        gamma=0.0,

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def get_experiments():
    """
    XGBoost experiments.

    All predict:
        0 = Control -> Control
        1 = Control -> MCI

    We focus rescue attempts mostly on t0_only and t1_only because these
    were the strongest logistic-regression feature sets.
    """
    experiments = []

    # Original raw XGBoost experiments.
    for feature_set in [
        "t0_only",
        "t1_only",
        "delta_only",
        "t0_t1",
        "t0_delta",
        "t0_t1_delta",
    ]:
        experiments.append({
            "experiment": f"xgboost_raw_{feature_set}",
            "feature_set": feature_set,
            "model_type": "raw",
            "k_best": None,
            "pca_components": None,
        })

    # Feature selection rescue attempts.
    for feature_set in ["t0_only", "t1_only"]:
        for k in [50, 100, 200, 500]:
            experiments.append({
                "experiment": f"xgboost_select{k}_{feature_set}",
                "feature_set": feature_set,
                "model_type": "selectk",
                "k_best": k,
                "pca_components": None,
            })

    # PCA rescue attempts.
    for feature_set in ["t0_only", "t1_only"]:
        for n_components in [10, 20, 50]:
            experiments.append({
                "experiment": f"xgboost_pca{n_components}_{feature_set}",
                "feature_set": feature_set,
                "model_type": "pca",
                "k_best": None,
                "pca_components": n_components,
            })

    return experiments


def build_model(exp, y_train):
    """
    Build a fold-specific model/pipeline.

    XGBoost needs fold-specific scale_pos_weight.
    SelectKBest/PCA are placed inside the CV fold to avoid leakage.
    """
    xgb = make_xgboost_classifier(y_train)

    if exp["model_type"] == "raw":
        return Pipeline([
            ("clf", xgb),
        ])

    if exp["model_type"] == "selectk":
        return Pipeline([
            ("select", SelectKBest(score_func=f_classif, k=exp["k_best"])),
            ("clf", xgb),
        ])

    if exp["model_type"] == "pca":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(
                n_components=exp["pca_components"],
                random_state=RANDOM_STATE,
            )),
            ("clf", xgb),
        ])

    raise ValueError(f"Unknown model_type: {exp['model_type']}")


def calculate_metrics(y_true, y_score, threshold):
    """
    Calculate classification metrics at a given threshold.
    """
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = np.nan

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def find_best_threshold(y_true, y_score):
    """
    Find threshold that maximises F1 on these predictions.

    Caveat:
    This is diagnostic and optimistic if used on a held-out test fold.
    For final unbiased reporting, threshold should be tuned in inner CV.
    """
    best_threshold = 0.5
    best_f1 = -np.inf

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(y_true, y_score, threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold

    return best_threshold


def evaluate_xgboost_cv(exp, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """
    Repeated stratified CV evaluation.

    Returns:
      - fold metrics at default threshold 0.5
      - fold metrics with F1-tuned threshold
      - pooled metrics at default threshold 0.5
      - pooled metrics with F1-tuned threshold
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    default_rows = []
    tuned_rows = []

    pooled_y_true = []
    pooled_y_score = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = build_model(exp, y_train)
        model.fit(X_train, y_train)

        y_score = model.predict_proba(X_test)[:, 1]

        default_metrics = calculate_metrics(
            y_true=y_test,
            y_score=y_score,
            threshold=0.5,
        )
        default_metrics["fold"] = fold_idx

        best_threshold = find_best_threshold(y_test, y_score)
        tuned_metrics = calculate_metrics(
            y_true=y_test,
            y_score=y_score,
            threshold=best_threshold,
        )
        tuned_metrics["fold"] = fold_idx

        default_rows.append(default_metrics)
        tuned_rows.append(tuned_metrics)

        pooled_y_true.extend(y_test.tolist())
        pooled_y_score.extend(np.asarray(y_score).tolist())

    fold_default = pd.DataFrame(default_rows)
    fold_tuned = pd.DataFrame(tuned_rows)

    pooled_y_true = np.asarray(pooled_y_true)
    pooled_y_score = np.asarray(pooled_y_score)

    pooled_default = calculate_metrics(
        y_true=pooled_y_true,
        y_score=pooled_y_score,
        threshold=0.5,
    )

    pooled_best_threshold = find_best_threshold(pooled_y_true, pooled_y_score)
    pooled_tuned = calculate_metrics(
        y_true=pooled_y_true,
        y_score=pooled_y_score,
        threshold=pooled_best_threshold,
    )

    return fold_default, fold_tuned, pooled_default, pooled_tuned


def summarise_cv_results(fold_df):
    metric_cols = [
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall_sensitivity",
        "specificity",
    ]

    summary = {}

    for col in metric_cols:
        summary[f"{col}_mean"] = fold_df[col].mean()
        summary[f"{col}_std"] = fold_df[col].std()

    return summary


def make_summary_row(exp, X_exp, fold_df, pooled, threshold_mode):
    summary = summarise_cv_results(fold_df)

    for key, value in pooled.items():
        summary[f"pooled_{key}"] = value

    summary.update({
        "experiment": exp["experiment"],
        "feature_set": exp["feature_set"],
        "model": "xgboost",
        "model_type": exp["model_type"],
        "threshold_mode": threshold_mode,
        "k_best": exp["k_best"],
        "pca_components": exp["pca_components"],
        "n_samples": X_exp.shape[0],
        "n_features": X_exp.shape[1],
        "effective_n_features": get_effective_n_features(exp, X_exp),
        "n_cv_fits": N_SPLITS * N_REPEATS,
    })

    return summary


def main():
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Could not find file: {H5_PATH.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, cpg_ids = load_task1(H5_PATH)
    feature_sets = make_feature_sets(X)
    experiments = get_experiments()

    print("=" * 80)
    print("Task 1: XGBoost baseline + feature selection / PCA / threshold diagnostics")
    print("Control -> Control vs Control -> MCI")
    print("=" * 80)
    print(f"HDF5 file: {H5_PATH.resolve()}")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print(f"CV setting: {N_SPLITS} folds x {N_REPEATS} repeats")
    print(f"X shape after transpose: {X.shape}")
    print("  axis 0 = subjects")
    print("  axis 1 = presumed time points")
    print("  axis 2 = CpG features")
    print(f"y shape: {y.shape}")
    print(f"class counts [0=CN->CN, 1=CN->MCI]: {np.bincount(y)}")
    print(f"scale_pos_weight overall: {np.sum(y == 0) / np.sum(y == 1):.3f}")
    print(f"number of CpG IDs: {len(cpg_ids)}")
    print(f"first 5 CpGs: {cpg_ids[:5].tolist()}")
    print("\nNote:")
    print("  threshold_mode='default_0.5' is the normal fair comparison.")
    print("  threshold_mode='tuned_f1' is diagnostic/optimistic unless threshold tuning is nested.")

    all_summaries = []
    all_fold_results = []

    for exp in experiments:
        exp_name = exp["experiment"]
        feature_name = exp["feature_set"]
        X_exp = feature_sets[feature_name]

        print("\n" + "-" * 80)
        print(f"Running: {exp_name}")
        print(f"Feature set: {feature_name}")
        print(f"Model type: {exp['model_type']}")
        print(f"k_best: {exp['k_best']}")
        print(f"pca_components: {exp['pca_components']}")
        print(f"Feature matrix shape: {X_exp.shape}")

        fold_default, fold_tuned, pooled_default, pooled_tuned = evaluate_xgboost_cv(
            exp=exp,
            X=X_exp,
            y=y,
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
        )

        default_summary = make_summary_row(
            exp=exp,
            X_exp=X_exp,
            fold_df=fold_default,
            pooled=pooled_default,
            threshold_mode="default_0.5",
        )

        tuned_summary = make_summary_row(
            exp=exp,
            X_exp=X_exp,
            fold_df=fold_tuned,
            pooled=pooled_tuned,
            threshold_mode="tuned_f1",
        )

        all_summaries.extend([default_summary, tuned_summary])

        for fold_df, threshold_mode in [
            (fold_default, "default_0.5"),
            (fold_tuned, "tuned_f1"),
        ]:
            fold_df.insert(0, "experiment", exp_name)
            fold_df.insert(1, "feature_set", feature_name)
            fold_df.insert(2, "model", "xgboost")
            fold_df.insert(3, "model_type", exp["model_type"])
            fold_df.insert(4, "threshold_mode", threshold_mode)
            fold_df.insert(5, "k_best", exp["k_best"])
            fold_df.insert(6, "pca_components", exp["pca_components"])
            all_fold_results.append(fold_df)

        print("Default threshold 0.5:")
        print(f"  Mean ROC-AUC:            {default_summary['roc_auc_mean']:.3f} ± {default_summary['roc_auc_std']:.3f}")
        print(f"  Mean PR-AUC:             {default_summary['pr_auc_mean']:.3f} ± {default_summary['pr_auc_std']:.3f}")
        print(f"  Mean balanced acc:       {default_summary['balanced_accuracy_mean']:.3f} ± {default_summary['balanced_accuracy_std']:.3f}")
        print(f"  Mean recall/sensitivity: {default_summary['recall_sensitivity_mean']:.3f} ± {default_summary['recall_sensitivity_std']:.3f}")
        print(f"  Mean specificity:        {default_summary['specificity_mean']:.3f} ± {default_summary['specificity_std']:.3f}")
        print(f"  Mean F1:                 {default_summary['f1_mean']:.3f} ± {default_summary['f1_std']:.3f}")

        print("Tuned F1 threshold:")
        print(f"  Mean threshold:          {tuned_summary['threshold_mean']:.3f} ± {tuned_summary['threshold_std']:.3f}")
        print(f"  Mean balanced acc:       {tuned_summary['balanced_accuracy_mean']:.3f} ± {tuned_summary['balanced_accuracy_std']:.3f}")
        print(f"  Mean recall/sensitivity: {tuned_summary['recall_sensitivity_mean']:.3f} ± {tuned_summary['recall_sensitivity_std']:.3f}")
        print(f"  Mean specificity:        {tuned_summary['specificity_mean']:.3f} ± {tuned_summary['specificity_std']:.3f}")
        print(f"  Mean F1:                 {tuned_summary['f1_mean']:.3f} ± {tuned_summary['f1_std']:.3f}")

    summary_df = pd.DataFrame(all_summaries)
    fold_results_df = pd.concat(all_fold_results, ignore_index=True)

    first_cols = [
        "experiment",
        "feature_set",
        "model",
        "model_type",
        "threshold_mode",
        "k_best",
        "pca_components",
        "n_samples",
        "n_features",
        "n_cv_fits",
        "threshold_mean",
        "threshold_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "recall_sensitivity_mean",
        "recall_sensitivity_std",
        "specificity_mean",
        "specificity_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "accuracy_mean",
        "accuracy_std",
        "pooled_threshold",
        "pooled_roc_auc",
        "pooled_pr_auc",
        "pooled_balanced_accuracy",
        "pooled_recall_sensitivity",
        "pooled_specificity",
        "pooled_f1",
        "pooled_precision",
        "pooled_tp",
        "pooled_fp",
        "pooled_tn",
        "pooled_fn",
    ]

    remaining_cols = [c for c in summary_df.columns if c not in first_cols]
    summary_df = summary_df[first_cols + remaining_cols]

    out_summary = OUT_DIR / "task1_xgboost_summary.csv"
    out_folds = OUT_DIR / "task1_xgboost_fold_results.csv"

    summary_df.to_csv(out_summary, index=False)
    fold_results_df.to_csv(out_folds, index=False)

    print("\n" + "=" * 80)
    print("Final XGBoost summary")
    print("=" * 80)

    display_cols = [
        "experiment",
        "threshold_mode",
        "n_features",
        "k_best",
        "pca_components",
        "threshold_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "balanced_accuracy_mean",
        "recall_sensitivity_mean",
        "specificity_mean",
        "f1_mean",
    ]

    default_df = summary_df[summary_df["threshold_mode"] == "default_0.5"].sort_values(
        by=["roc_auc_mean", "pr_auc_mean", "balanced_accuracy_mean"],
        ascending=False,
    )

    tuned_df = summary_df[summary_df["threshold_mode"] == "tuned_f1"].sort_values(
        by=["f1_mean", "balanced_accuracy_mean", "roc_auc_mean"],
        ascending=False,
    )

    print("\nDefault threshold 0.5, sorted by ROC-AUC:")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(default_df[display_cols].to_string(index=False))

    print("\nTuned F1 threshold, sorted by F1:")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(tuned_df[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {out_summary.resolve()}")
    print(f"  {out_folds.resolve()}")




if __name__ == "__main__":
    main()
