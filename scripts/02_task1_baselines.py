#!/usr/bin/env python3
# Baseline models for comparison in Task 1: Control -> Control vs Control -> MCI.
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
from sklearn.decomposition import PCA

# Slower optional models.
# Uncomment these imports only if you want to test them later.
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.neural_network import MLPClassifier


warnings.filterwarnings("ignore", category=ConvergenceWarning)


H5_PATH = Path("../data/temporal_two_sets_n2000.h5")
OUT_DIR = Path("../results/task1")

RANDOM_STATE = 42

# Fast/default CV setting.
# Once the script looks good, you can increase N_REPEATS to 5 or 10.
N_SPLITS = 5
N_REPEATS = 3


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

    # Convert from features x time x samples
    # to samples x time x features.
    x0 = np.transpose(x_cn_to_cn, (2, 1, 0))   # non-converters
    x1 = np.transpose(x_cn_to_mci, (2, 1, 0))  # converters

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


def make_feature_sets(X):
    """
    X shape:
        samples x time x features

    Returns several 2D feature matrices:
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


def get_models():
    """
    Define models.

    Fast/default models:
    - majority class baseline
    - L2 logistic regression
    - PCA + L2 logistic regression

    Slower optional models are included as commented code.
    """
    models = {}

    models["majority_class"] = Pipeline([
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])

    # Good fast baseline for small-n, high-p methylation data.
    models["logreg_l2"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )),
    ])

    # PCA must be inside the CV pipeline to avoid leakage.
    # n_components=20 is deliberately conservative for n=190.
    models["pca20_logreg"] = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=20, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )),
    ])

    # Optional: try fewer PCA components.
    models["pca10_logreg"] = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )),
    ])

    # -------------------------------------------------------------------------
    # SLOWER OPTIONAL MODELS
    # -------------------------------------------------------------------------
    # Elastic-net can be slow here because:
    #   190 samples x 4000-6000 features x repeated CV x saga solver
    #
    # We keep a *small* elastic-net grid here so it is reproducible and not too
    # expensive, but still lets us test whether sparsity helps vs pure L2.
    def make_elasticnet_logreg(*, C, l1_ratio):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=float(C),
                l1_ratio=float(l1_ratio),
                class_weight="balanced",
                max_iter=10_000,
                tol=1e-3,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ])

    # Elastic-net logistic regression (small grid).
    models["logreg_elasticnet_C0.1_l1r0.1"] = make_elasticnet_logreg(C=0.1, l1_ratio=0.1)
    models["logreg_elasticnet_C0.1_l1r0.5"] = make_elasticnet_logreg(C=0.1, l1_ratio=0.5)
    models["logreg_elasticnet_C0.1_l1r0.9"] = make_elasticnet_logreg(C=0.1, l1_ratio=0.9)
    models["logreg_elasticnet_C1_l1r0.5"] = make_elasticnet_logreg(C=1.0, l1_ratio=0.5)

    # Tree models are not too bad, but can still be slow across repeated CV
    # and may overfit with many CpGs and few samples.
    #
    # models["random_forest"] = Pipeline([
    #     ("clf", RandomForestClassifier(
    #         n_estimators=300,
    #         class_weight="balanced",
    #         max_features="sqrt",
    #         min_samples_leaf=3,
    #         random_state=RANDOM_STATE,
    #         n_jobs=-1,
    #     )),
    # ])
    #
    # models["extra_trees"] = Pipeline([
    #     ("clf", ExtraTreesClassifier(
    #         n_estimators=300,
    #         class_weight="balanced",
    #         max_features="sqrt",
    #         min_samples_leaf=3,
    #         random_state=RANDOM_STATE,
    #         n_jobs=-1,
    #     )),
    # ])

    # Small MLP is exploratory and can be unstable for n=190.
    #
    # models["small_mlp"] = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("clf", MLPClassifier(
    #         hidden_layer_sizes=(128, 32),
    #         activation="relu",
    #         alpha=1e-3,
    #         learning_rate_init=1e-3,
    #         batch_size=16,
    #         max_iter=500,
    #         early_stopping=True,
    #         validation_fraction=0.2,
    #         n_iter_no_change=30,
    #         random_state=RANDOM_STATE,
    #     )),
    # ])

    return models


def get_positive_scores(model, X_test):
    """
    Get a continuous score for ROC-AUC / PR-AUC.

    Most classifiers have predict_proba.
    Some only have decision_function.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    return model.predict(X_test)


def evaluate_model_cv(model, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """
    Repeated stratified CV evaluation.

    Returns:
      - fold-level metrics
      - pooled out-of-fold metrics
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    rows = []
    pooled_y_true = []
    pooled_y_pred = []
    pooled_y_score = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        fitted = clone(model)
        fitted.fit(X_train, y_train)

        y_pred = fitted.predict(X_test)
        y_score = get_positive_scores(fitted, X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        try:
            roc_auc = roc_auc_score(y_test, y_score)
        except ValueError:
            roc_auc = np.nan

        try:
            pr_auc = average_precision_score(y_test, y_score)
        except ValueError:
            pr_auc = np.nan

        row = {
            "fold": fold_idx,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall_sensitivity": recall_score(y_test, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

        rows.append(row)

        pooled_y_true.extend(y_test.tolist())
        pooled_y_pred.extend(y_pred.tolist())
        pooled_y_score.extend(np.asarray(y_score).tolist())

    fold_df = pd.DataFrame(rows)

    pooled_y_true = np.asarray(pooled_y_true)
    pooled_y_pred = np.asarray(pooled_y_pred)
    pooled_y_score = np.asarray(pooled_y_score)

    try:
        pooled_roc_auc = roc_auc_score(pooled_y_true, pooled_y_score)
    except ValueError:
        pooled_roc_auc = np.nan

    try:
        pooled_pr_auc = average_precision_score(pooled_y_true, pooled_y_score)
    except ValueError:
        pooled_pr_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        pooled_y_true,
        pooled_y_pred,
        labels=[0, 1],
    ).ravel()

    pooled_metrics = {
        "pooled_accuracy": accuracy_score(pooled_y_true, pooled_y_pred),
        "pooled_balanced_accuracy": balanced_accuracy_score(pooled_y_true, pooled_y_pred),
        "pooled_roc_auc": pooled_roc_auc,
        "pooled_pr_auc": pooled_pr_auc,
        "pooled_f1": f1_score(pooled_y_true, pooled_y_pred, zero_division=0),
        "pooled_precision": precision_score(pooled_y_true, pooled_y_pred, zero_division=0),
        "pooled_recall_sensitivity": recall_score(pooled_y_true, pooled_y_pred, zero_division=0),
        "pooled_specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "pooled_tp": tp,
        "pooled_fp": fp,
        "pooled_tn": tn,
        "pooled_fn": fn,
    }

    return fold_df, pooled_metrics


def summarise_cv_results(fold_df):
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall_sensitivity",
        "specificity",
    ]

    means = fold_df[metric_cols].mean()
    stds = fold_df[metric_cols].std()

    summary = {}
    for col in metric_cols:
        summary[f"{col}_mean"] = means[col]
        summary[f"{col}_std"] = stds[col]

    return summary


def get_experiments():
    """
    Fast/default experiment list.

    All experiments predict:
        0 = Control -> Control
        1 = Control -> MCI

    Slower models are shown commented out at the bottom.
    """
    experiments = [
        {
            "experiment": "majority_class",
            "feature_set": "t0_only",
            "model": "majority_class",
        },

        # Simple L2 logistic regression baselines.
        {
            "experiment": "logreg_l2_t0_only",
            "feature_set": "t0_only",
            "model": "logreg_l2",
        },
        {
            "experiment": "logreg_l2_t1_only",
            "feature_set": "t1_only",
            "model": "logreg_l2",
        },
        {
            "experiment": "logreg_l2_delta_only",
            "feature_set": "delta_only",
            "model": "logreg_l2",
        },

        # Elastic-net logistic regression on the strongest compact feature set.
        # This is slower than liblinear L2 logistic regression.
        {
            "experiment": "logreg_elasticnet_C0.1_l1r0.1_t1_only",
            "feature_set": "t1_only",
            "model": "logreg_elasticnet_C0.1_l1r0.1",
        },
        {
            "experiment": "logreg_elasticnet_C0.1_l1r0.5_t1_only",
            "feature_set": "t1_only",
            "model": "logreg_elasticnet_C0.1_l1r0.5",
        },
        {
            "experiment": "logreg_elasticnet_C0.1_l1r0.9_t1_only",
            "feature_set": "t1_only",
            "model": "logreg_elasticnet_C0.1_l1r0.9",
        },
        {
            "experiment": "logreg_elasticnet_C1_l1r0.5_t1_only",
            "feature_set": "t1_only",
            "model": "logreg_elasticnet_C1_l1r0.5",
        },

        # Combined feature representations.
        {
            "experiment": "logreg_l2_t0_t1",
            "feature_set": "t0_t1",
            "model": "logreg_l2",
        },
        {
            "experiment": "logreg_l2_t0_delta",
            "feature_set": "t0_delta",
            "model": "logreg_l2",
        },
        {
            "experiment": "logreg_l2_t0_t1_delta",
            "feature_set": "t0_t1_delta",
            "model": "logreg_l2",
        },

        # PCA dimensionality reduction baselines.
        {
            "experiment": "pca10_logreg_t0_only",
            "feature_set": "t0_only",
            "model": "pca10_logreg",
        },
        {
            "experiment": "pca20_logreg_t0_only",
            "feature_set": "t0_only",
            "model": "pca20_logreg",
        },
        {
            "experiment": "pca10_logreg_t0_delta",
            "feature_set": "t0_delta",
            "model": "pca10_logreg",
        },
        {
            "experiment": "pca20_logreg_t0_delta",
            "feature_set": "t0_delta",
            "model": "pca20_logreg",
        },

        # ---------------------------------------------------------------------
        # SLOWER OPTIONAL EXPERIMENTS
        # Uncomment these only after the fast baselines are complete.
        # Also uncomment the relevant models/imports in get_models().
        # ---------------------------------------------------------------------
        # {
        #     "experiment": "elasticnet_t0_t1",
        #     "feature_set": "t0_t1",
        #     "model": "logreg_elasticnet",
        # },
        # {
        #     "experiment": "elasticnet_t0_delta",
        #     "feature_set": "t0_delta",
        #     "model": "logreg_elasticnet",
        # },
        # {
        #     "experiment": "elasticnet_t0_t1_delta",
        #     "feature_set": "t0_t1_delta",
        #     "model": "logreg_elasticnet",
        # },
        # {
        #     "experiment": "random_forest_t0_delta",
        #     "feature_set": "t0_delta",
        #     "model": "random_forest",
        # },
        # {
        #     "experiment": "extra_trees_t0_delta",
        #     "feature_set": "t0_delta",
        #     "model": "extra_trees",
        # },
        # {
        #     "experiment": "small_mlp_t0_delta",
        #     "feature_set": "t0_delta",
        #     "model": "small_mlp",
        # },
    ]

    return experiments


def main():
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Could not find file: {H5_PATH.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, cpg_ids = load_task1(H5_PATH)

    print("=" * 80)
    print("Task 1: Control -> Control vs Control -> MCI")
    print("=" * 80)
    print(f"HDF5 file: {H5_PATH.resolve()}")
    print(f"Output directory: {OUT_DIR.resolve()}")
    print(f"CV setting: {N_SPLITS} folds x {N_REPEATS} repeats")
    print(f"X shape after transpose: {X.shape}")
    print("  axis 0 = subjects")
    print("  axis 1 = time points")
    print("  axis 2 = CpG features")
    print(f"y shape: {y.shape}")
    print(f"class counts [0=CN->CN, 1=CN->MCI]: {np.bincount(y)}")
    print(f"number of CpG IDs: {len(cpg_ids)}")
    print(f"first 5 CpGs: {cpg_ids[:5].tolist()}")

    feature_sets = make_feature_sets(X)
    models = get_models()
    experiments = get_experiments()

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
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
        )

        summary = summarise_cv_results(fold_df)
        summary.update(pooled)
        summary.update({
            "experiment": exp_name,
            "feature_set": feature_name,
            "model": model_name,
            "n_samples": X_exp.shape[0],
            "n_features": X_exp.shape[1],
            "n_cv_fits": N_SPLITS * N_REPEATS,
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

    summary_df = pd.DataFrame(all_summaries)
    fold_results_df = pd.concat(all_fold_results, ignore_index=True)

    first_cols = [
        "experiment",
        "feature_set",
        "model",
        "n_samples",
        "n_features",
        "n_cv_fits",
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
        "pooled_roc_auc",
        "pooled_pr_auc",
        "pooled_balanced_accuracy",
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

    out_summary = OUT_DIR / "task1_baseline_summary.csv"
    out_folds = OUT_DIR / "task1_baseline_fold_results.csv"

    summary_df.to_csv(out_summary, index=False)
    fold_results_df.to_csv(out_folds, index=False)

    print("\n" + "=" * 80)
    print("Final summary, sorted by mean ROC-AUC")
    print("=" * 80)

    display_cols = [
        "experiment",
        "n_features",
        "roc_auc_mean",
        "pr_auc_mean",
        "balanced_accuracy_mean",
        "recall_sensitivity_mean",
        "specificity_mean",
        "f1_mean",
    ]

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary_df[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {out_summary.resolve()}")
    print(f"  {out_folds.resolve()}")


if __name__ == "__main__":
    main()
