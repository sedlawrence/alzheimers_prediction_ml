#!/usr/bin/env python3

import numpy as np
import pandas as pd

from _shared.classical_models import (
    get_baseline_experiments,
    get_baseline_models,
    get_linear_effective_n_features,
)
from _shared.cv import evaluate_model_cv
from _shared.data_io import load_binary_task
from _shared.feature_sets import make_flat_feature_sets
from _shared.metrics import CLASSICAL_METRIC_COLS, summarise_cv_results
from _shared.task_config import (
    DEFAULT_N_REPEATS,
    DEFAULT_N_SPLITS,
    RANDOM_STATE,
    TASK2_SPEC,
)


H5_PATH = TASK2_SPEC.h5_path
OUT_DIR = TASK2_SPEC.out_dir
TASK_NAME = "Task 2: baseline models"


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
    models = get_baseline_models()
    experiments = get_baseline_experiments()

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
    print(f"number of CpG IDs: {len(cpg_ids)}")
    print(f"first 5 CpGs: {cpg_ids[:5].tolist()}")

    all_summaries = []
    all_fold_results = []

    for exp in experiments:
        exp_name = exp["experiment"]
        feature_name = exp["feature_set"]
        model_name = exp["model"]

        X_exp = feature_sets[feature_name]
        model = models[model_name]

        print("\n" + "-" * 80)
        print(f"Running: {exp_name}")
        print(f"Feature set: {feature_name}")
        print(f"Feature matrix shape: {X_exp.shape}")
        print(f"Model: {model_name}")

        fold_df, pooled = evaluate_model_cv(
            model=model,
            X=X_exp,
            y=y,
            n_splits=DEFAULT_N_SPLITS,
            n_repeats=DEFAULT_N_REPEATS,
            random_state=RANDOM_STATE,
        )

        summary = summarise_cv_results(fold_df, CLASSICAL_METRIC_COLS)
        for key, value in pooled.items():
            summary[f"pooled_{key}"] = value
        summary.update({
            "experiment": exp_name,
            "feature_set": feature_name,
            "model": model_name,
            "n_samples": X_exp.shape[0],
            "n_features": X_exp.shape[1],
            "effective_n_features": get_linear_effective_n_features(exp, X_exp),
            "n_cv_fits": DEFAULT_N_SPLITS * DEFAULT_N_REPEATS,
        })

        all_summaries.append(summary)

        fold_df.insert(0, "experiment", exp_name)
        fold_df.insert(1, "feature_set", feature_name)
        fold_df.insert(2, "model", model_name)
        all_fold_results.append(fold_df)

        print(f"Mean ROC-AUC:            {summary['roc_auc_mean']:.3f} ± {summary['roc_auc_std']:.3f}")
        print(f"Mean PR-AUC:             {summary['pr_auc_mean']:.3f} ± {summary['pr_auc_std']:.3f}")
        print(f"Mean balanced acc:       {summary['balanced_accuracy_mean']:.3f} ± {summary['balanced_accuracy_std']:.3f}")
        print(f"Mean recall/sensitivity: {summary['recall_sensitivity_mean']:.3f} ± {summary['recall_sensitivity_std']:.3f}")
        print(f"Mean specificity:        {summary['specificity_mean']:.3f} ± {summary['specificity_std']:.3f}")
        print(f"Mean F1:                 {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}")

    summary_df = pd.DataFrame(all_summaries)
    fold_results_df = pd.concat(all_fold_results, ignore_index=True)

    first_cols = [
        "experiment",
        "feature_set",
        "model",
        "n_samples",
        "n_features",
        "effective_n_features",
        "n_cv_fits",
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

    summary_df = summary_df.sort_values(
        by=["roc_auc_mean", "pr_auc_mean", "balanced_accuracy_mean"],
        ascending=False,
    )

    out_summary = OUT_DIR / "task2_baseline_summary.csv"
    out_folds = OUT_DIR / "task2_baseline_fold_results.csv"

    summary_df.to_csv(out_summary, index=False)
    fold_results_df.to_csv(out_folds, index=False)

    print("\n" + "=" * 80)
    print("Final summary, sorted by mean ROC-AUC")
    print("=" * 80)

    display_cols = [
        "experiment",
        "n_features",
        "effective_n_features",
        "roc_auc_mean",
        "pr_auc_mean",
        "balanced_accuracy_mean",
        "recall_sensitivity_mean",
        "specificity_mean",
        "f1_mean",
    ]

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary_df[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {out_summary.resolve()}")
    print(f"  {out_folds.resolve()}")


if __name__ == "__main__":
    main()
