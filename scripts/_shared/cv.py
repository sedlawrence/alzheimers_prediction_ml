import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from .metrics import calculate_binary_metrics


DEFAULT_THRESHOLDS = np.array([
    0.05, 0.075,
    0.10, 0.125, 0.15, 0.175,
    0.20, 0.225, 0.25, 0.275,
    0.30, 0.35, 0.40, 0.45, 0.50,
])


def get_positive_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    return model.predict(X_test)


def evaluate_model_cv(model, X, y, n_splits, n_repeats, random_state):
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    rows = []
    pooled_y_true = []
    pooled_y_score = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        fitted = clone(model)
        fitted.fit(X_train, y_train)
        y_score = get_positive_scores(fitted, X_test)

        fold_metrics = calculate_binary_metrics(
            y_true=y_test,
            y_score=y_score,
            threshold=0.5,
        )
        fold_metrics["fold"] = fold_idx
        rows.append(fold_metrics)

        pooled_y_true.extend(y_test.tolist())
        pooled_y_score.extend(np.asarray(y_score).tolist())

    fold_df = pd.DataFrame(rows)

    pooled_metrics = calculate_binary_metrics(
        y_true=np.asarray(pooled_y_true),
        y_score=np.asarray(pooled_y_score),
        threshold=0.5,
    )

    return fold_df, pooled_metrics


def find_best_threshold(y_true, y_score, thresholds=DEFAULT_THRESHOLDS):
    best_threshold = 0.5
    best_f1 = -np.inf

    for threshold in thresholds:
        metrics = calculate_binary_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            include_threshold=True,
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold

    return best_threshold


def evaluate_threshold_model_cv(build_model_fn, X, y, n_splits, n_repeats, random_state, thresholds=DEFAULT_THRESHOLDS):
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
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

        model = build_model_fn(y_train)
        model.fit(X_train, y_train)
        y_score = get_positive_scores(model, X_test)

        default_metrics = calculate_binary_metrics(
            y_true=y_test,
            y_score=y_score,
            threshold=0.5,
            include_threshold=True,
        )
        default_metrics["fold"] = fold_idx

        best_threshold = find_best_threshold(y_test, y_score, thresholds=thresholds)
        tuned_metrics = calculate_binary_metrics(
            y_true=y_test,
            y_score=y_score,
            threshold=best_threshold,
            include_threshold=True,
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
    pooled_default = calculate_binary_metrics(
        y_true=pooled_y_true,
        y_score=pooled_y_score,
        threshold=0.5,
        include_threshold=True,
    )
    pooled_tuned = calculate_binary_metrics(
        y_true=pooled_y_true,
        y_score=pooled_y_score,
        threshold=find_best_threshold(pooled_y_true, pooled_y_score, thresholds=thresholds),
        include_threshold=True,
    )

    return fold_default, fold_tuned, pooled_default, pooled_tuned
