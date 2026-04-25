#!/usr/bin/env python3

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
)
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
from sklearn.svm import LinearSVC, SVC


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


H5_PATH = Path("../data/temporal_two_sets_n2000.h5")
OUT_DIR = Path("../results/task1")

RANDOM_STATE = 42
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


def get_effective_n_features(exp, X_exp):
    """
    Number of features actually seen by the model after any transform.
    """
    if exp["model"].startswith("pca"):
        return int(exp["model"].replace("pca", "").split("_")[0])

    if exp["model"].startswith("pls"):
        return int(exp["model"].replace("pls", "").split("_")[0])

    return X_exp.shape[1]


class PLSLogisticClassifier(BaseEstimator, ClassifierMixin):
    """
    Small sklearn-compatible classifier:

        StandardScaler
        PLSRegression transform
        LogisticRegression on PLS scores

    PLS is useful for omics-like data because it finds components related
    to y, unlike PCA which only captures variance in X.
    """

    def __init__(
        self,
        n_components=10,
        random_state=42,
        class_weight="balanced",
        max_iter=5000,
        C=1.0,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.pls_ = PLSRegression(n_components=self.n_components)
        X_pls = self.pls_.fit_transform(X_scaled, y)[0]

        self.clf_ = LogisticRegression(
            penalty="l2",
            C=self.C,
            class_weight=self.class_weight,
            solver="liblinear",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.clf_.fit(X_pls, y)

        return self

    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        X_pls = self.pls_.transform(X_scaled)
        return self.clf_.predict(X_pls)

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        X_pls = self.pls_.transform(X_scaled)
        return self.clf_.predict_proba(X_pls)

    def decision_function(self, X):
        X_scaled = self.scaler_.transform(X)
        X_pls = self.pls_.transform(X_scaled)
        return self.clf_.decision_function(X_pls)


def get_models():
    """
    Define classical ML models.

    These are intended for small-n, high-p methylation data:
      - linear SVMs
      - ridge-style linear classifiers
      - SGD linear models
      - PCA + RBF SVM
      - PLS + logistic regression
    """
    models = {}

    models["majority_class"] = Pipeline([
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])

    # Reference model from script 02, included so this script is comparable.
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

    # Linear SVM: strong candidate for high-dimensional data.
    models["linear_svc_l2"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            C=1.0,
            penalty="l2",
            loss="squared_hinge",
            class_weight="balanced",
            max_iter=20000,
            dual="auto",
            random_state=RANDOM_STATE,
        )),
    ])

    # More regularised linear SVM.
    models["linear_svc_l2_C0.1"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(
            C=0.1,
            penalty="l2",
            loss="squared_hinge",
            class_weight="balanced",
            max_iter=20000,
            dual="auto",
            random_state=RANDOM_STATE,
        )),
    ])

    # Ridge classifier: another linear model suited to high-dimensional data.
    models["ridge_classifier"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RidgeClassifier(
            alpha=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    models["ridge_classifier_alpha10"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RidgeClassifier(
            alpha=10.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    # SGD logistic regression: fast linear probabilistic model.
    models["sgd_log_loss"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=5000,
            tol=1e-4,
            random_state=RANDOM_STATE,
        )),
    ])

    # SGD hinge approximates a linear SVM.
    models["sgd_hinge"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=5000,
            tol=1e-4,
            random_state=RANDOM_STATE,
        )),
    ])

    # Modified Huber gives robust margin-style classification with predict_proba.
    models["sgd_modified_huber"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="modified_huber",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=5000,
            tol=1e-4,
            random_state=RANDOM_STATE,
        )),
    ])

    # Nonlinear SVM only after PCA, otherwise too risky/slow in 2000-6000 dims.
    for n_components in [10, 20, 50]:
        models[f"pca{n_components}_rbf_svm"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(
                n_components=n_components,
                random_state=RANDOM_STATE,
            )),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                random_state=RANDOM_STATE,
            )),
        ])

    # PLS supervised dimension reduction followed by logistic regression.
    for n_components in [2, 5, 10, 20]:
        models[f"pls{n_components}_logreg"] = PLSLogisticClassifier(
            n_components=n_components,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_iter=5000,
            C=1.0,
        )

    return models


def get_experiments():
    """
    Experiments to run.

    All predict:
        0 = Control -> Control
        1 = Control -> MCI
    """

    experiments = [
        {
            "experiment": "majority_class",
            "feature_set": "t0_only",
            "model": "majority_class",
        },

        # Reference logistic regression, included here for side-by-side comparison.
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
            "experiment": "logreg_l2_t0_t1",
            "feature_set": "t0_t1",
            "model": "logreg_l2",
        },

        # Linear SVMs on strongest/simple feature sets.
        {
            "experiment": "linear_svc_t0_only",
            "feature_set": "t0_only",
            "model": "linear_svc_l2",
        },
        {
            "experiment": "linear_svc_t1_only",
            "feature_set": "t1_only",
            "model": "linear_svc_l2",
        },
        {
            "experiment": "linear_svc_t0_t1",
            "feature_set": "t0_t1",
            "model": "linear_svc_l2",
        },
        {
            "experiment": "linear_svc_t0_delta",
            "feature_set": "t0_delta",
            "model": "linear_svc_l2",
        },

        # More regularised linear SVM.
        {
            "experiment": "linear_svc_C0.1_t0_only",
            "feature_set": "t0_only",
            "model": "linear_svc_l2_C0.1",
        },
        {
            "experiment": "linear_svc_C0.1_t1_only",
            "feature_set": "t1_only",
            "model": "linear_svc_l2_C0.1",
        },
        {
            "experiment": "linear_svc_C0.1_t0_t1",
            "feature_set": "t0_t1",
            "model": "linear_svc_l2_C0.1",
        },

        # Ridge classifiers.
        {
            "experiment": "ridge_t0_only",
            "feature_set": "t0_only",
            "model": "ridge_classifier",
        },
        {
            "experiment": "ridge_t1_only",
            "feature_set": "t1_only",
            "model": "ridge_classifier",
        },
        {
            "experiment": "ridge_t0_t1",
            "feature_set": "t0_t1",
            "model": "ridge_classifier",
        },
        {
            "experiment": "ridge_alpha10_t0_only",
            "feature_set": "t0_only",
            "model": "ridge_classifier_alpha10",
        },
        {
            "experiment": "ridge_alpha10_t1_only",
            "feature_set": "t1_only",
            "model": "ridge_classifier_alpha10",
        },

        # SGD linear models.
        {
            "experiment": "sgd_log_loss_t0_only",
            "feature_set": "t0_only",
            "model": "sgd_log_loss",
        },
        {
            "experiment": "sgd_log_loss_t1_only",
            "feature_set": "t1_only",
            "model": "sgd_log_loss",
        },
        {
            "experiment": "sgd_hinge_t0_only",
            "feature_set": "t0_only",
            "model": "sgd_hinge",
        },
        {
            "experiment": "sgd_hinge_t1_only",
            "feature_set": "t1_only",
            "model": "sgd_hinge",
        },
        {
            "experiment": "sgd_modified_huber_t0_only",
            "feature_set": "t0_only",
            "model": "sgd_modified_huber",
        },
        {
            "experiment": "sgd_modified_huber_t1_only",
            "feature_set": "t1_only",
            "model": "sgd_modified_huber",
        },

        # RBF SVMs on PCA components.
        {
            "experiment": "pca10_rbf_svm_t0_only",
            "feature_set": "t0_only",
            "model": "pca10_rbf_svm",
        },
        {
            "experiment": "pca20_rbf_svm_t0_only",
            "feature_set": "t0_only",
            "model": "pca20_rbf_svm",
        },
        {
            "experiment": "pca50_rbf_svm_t0_only",
            "feature_set": "t0_only",
            "model": "pca50_rbf_svm",
        },
        {
            "experiment": "pca10_rbf_svm_t1_only",
            "feature_set": "t1_only",
            "model": "pca10_rbf_svm",
        },
        {
            "experiment": "pca20_rbf_svm_t1_only",
            "feature_set": "t1_only",
            "model": "pca20_rbf_svm",
        },
        {
            "experiment": "pca50_rbf_svm_t1_only",
            "feature_set": "t1_only",
            "model": "pca50_rbf_svm",
        },

        # PLS + logistic regression.
        {
            "experiment": "pls2_logreg_t0_only",
            "feature_set": "t0_only",
            "model": "pls2_logreg",
        },
        {
            "experiment": "pls5_logreg_t0_only",
            "feature_set": "t0_only",
            "model": "pls5_logreg",
        },
        {
            "experiment": "pls10_logreg_t0_only",
            "feature_set": "t0_only",
            "model": "pls10_logreg",
        },
        {
            "experiment": "pls20_logreg_t0_only",
            "feature_set": "t0_only",
            "model": "pls20_logreg",
        },
        {
            "experiment": "pls2_logreg_t1_only",
            "feature_set": "t1_only",
            "model": "pls2_logreg",
        },
        {
            "experiment": "pls5_logreg_t1_only",
            "feature_set": "t1_only",
            "model": "pls5_logreg",
        },
        {
            "experiment": "pls10_logreg_t1_only",
            "feature_set": "t1_only",
            "model": "pls10_logreg",
        },
        {
            "experiment": "pls20_logreg_t1_only",
            "feature_set": "t1_only",
            "model": "pls20_logreg",
        },
    ]

    return experiments


def get_positive_scores(model, X_test):
    """
    Get a continuous score for ROC-AUC / PR-AUC.

    For probability models, use P(class=1).
    For margin-based models, use decision_function.
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

        rows.append({
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
        })

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

    summary = {}

    for col in metric_cols:
        summary[f"{col}_mean"] = fold_df[col].mean()
        summary[f"{col}_std"] = fold_df[col].std()

    return summary


def main():
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Could not find file: {H5_PATH.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, cpg_ids = load_task1(H5_PATH)
    feature_sets = make_feature_sets(X)
    models = get_models()
    experiments = get_experiments()

    print("=" * 80)
    print("Task 1: SVM and linear model baselines")
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
            "effective_n_features": get_effective_n_features(exp, X_exp),
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
        "pooled_f1",
        "pooled_precision",
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

    out_summary = OUT_DIR / "task1_svm_linear_summary.csv"
    out_folds = OUT_DIR / "task1_svm_linear_fold_results.csv"

    summary_df.to_csv(out_summary, index=False)
    fold_results_df.to_csv(out_folds, index=False)

    print("\n" + "=" * 80)
    print("Final SVM / linear model summary, sorted by mean ROC-AUC")
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

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary_df[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {out_summary.resolve()}")
    print(f"  {out_folds.resolve()}")


if __name__ == "__main__":
    main()
