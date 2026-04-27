#!/usr/bin/env python3

import numpy as np
import pandas as pd

from _shared.classical_models import (
    evaluate_xgboost_cv,
    get_xgboost_experiments,
    make_xgboost_summary_row,
)
from _shared.data_io import load_binary_task
from _shared.feature_sets import make_flat_feature_sets
from _shared.task_config import (
    DEFAULT_N_REPEATS,
    DEFAULT_N_SPLITS,
    TASK2_SPEC,
)


H5_PATH = TASK2_SPEC.h5_path
OUT_DIR = TASK2_SPEC.out_dir
TASK_NAME = "Task 2: XGBoost baseline + feature selection / PCA / threshold diagnostics"


def main():
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Could not find file: {H5_PATH.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, cpg_ids = load_binary_task(
        H5_PATH,
        TASK2_SPEC.class0_key,
        TASK2_SPEC.class1_key,
        TASK2_SPEC.cpg_key,
    )
    feature_sets = make_flat_feature_sets(X)
    experiments = get_xgboost_experiments()

    print("=" * 80)
    print(TASK_NAME)
    print(f"{TASK2_SPEC.class0_label} vs {TASK2_SPEC.class1_label}")
    print("=" * 80)
    print(f"HDF5 file: {H5_PATH.resolve()}")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print(f"CV setting: {DEFAULT_N_SPLITS} folds x {DEFAULT_N_REPEATS} repeats")
    print(f"X shape after transpose: {X.shape}")
    print("  axis 0 = subjects")
    print("  axis 1 = presumed time points")
    print("  axis 2 = CpG features")
    print(f"y shape: {y.shape}")
    print(f"class counts [0={TASK2_SPEC.class0_label}, 1={TASK2_SPEC.class1_label}]: {np.bincount(y)}")
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
            n_splits=DEFAULT_N_SPLITS,
            n_repeats=DEFAULT_N_REPEATS,
        )

        default_summary = make_xgboost_summary_row(
            exp=exp,
            X_exp=X_exp,
            fold_df=fold_default,
            pooled=pooled_default,
            threshold_mode="default_0.5",
        )

        tuned_summary = make_xgboost_summary_row(
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
        "effective_n_features",
        "n_cv_fits",
        "threshold_mean",
        "threshold_std",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_sensitivity_mean",
        "recall_sensitivity_std",
        "specificity_mean",
        "specificity_std",
        "pooled_threshold",
        "pooled_accuracy",
        "pooled_balanced_accuracy",
        "pooled_roc_auc",
        "pooled_pr_auc",
        "pooled_f1",
        "pooled_precision",
        "pooled_recall_sensitivity",
        "pooled_specificity",
        "pooled_tp",
        "pooled_fp",
        "pooled_tn",
        "pooled_fn",
    ]

    remaining_cols = [c for c in summary_df.columns if c not in first_cols]
    summary_df = summary_df[first_cols + remaining_cols]

    out_summary = OUT_DIR / "task2_xgboost_summary.csv"
    out_folds = OUT_DIR / "task2_xgboost_fold_results.csv"

    summary_df.to_csv(out_summary, index=False)
    fold_results_df.to_csv(out_folds, index=False)

    print("\n" + "=" * 80)
    print("Final XGBoost summary")
    print("=" * 80)

    display_cols = [
        "experiment",
        "threshold_mode",
        "n_features",
        "effective_n_features",
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
