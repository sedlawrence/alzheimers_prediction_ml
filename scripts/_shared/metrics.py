import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


CLASSICAL_METRIC_COLS = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "f1",
    "precision",
    "recall_sensitivity",
    "specificity",
]

DL_METRIC_COLS = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "f1",
    "precision",
    "recall_sensitivity",
    "specificity",
    "best_epoch",
    "best_val_loss",
    "n_parameters",
]


def calculate_binary_metrics(y_true, y_score, threshold=0.5, include_threshold=False):
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

    metrics = {
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

    if include_threshold:
        metrics["threshold"] = threshold

    return metrics


def summarise_cv_results(fold_df, metric_cols):
    summary = {}
    for col in metric_cols:
        summary[f"{col}_mean"] = fold_df[col].mean()
        summary[f"{col}_std"] = fold_df[col].std()
    return summary
