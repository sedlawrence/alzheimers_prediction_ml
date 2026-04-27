#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Reports" / "figures"


def read_first(path):
    return pd.read_csv(ROOT / path).iloc[0]


def read_best(path, sort_col, query=None):
    df = pd.read_csv(ROOT / path)
    if query is not None:
        df = df.query(query)
    return df.sort_values(sort_col, ascending=False).iloc[0]


def collect_model_rows():
    rows = []

    specs = [
        (
            "Task 1",
            "Logistic L2",
            read_best("results/task1/task1_baseline_summary.csv", "f1_mean"),
        ),
        (
            "Task 1",
            "Linear SVC",
            read_best("results/task1/task1_svm_linear_summary.csv", "f1_mean"),
        ),
        (
            "Task 1",
            "XGBoost",
            read_best(
                "results/task1/task1_xgboost_summary.csv",
                "f1_mean",
                "threshold_mode == 'tuned_f1'",
            ),
        ),
        (
            "Task 1",
            "Best DL",
            read_first(
                "results/task1/dl_tuned_bottleneck_wide/task1_deep_learning_summary.csv"
            ),
        ),
        (
            "Task 1",
            "Chromosome CNN",
            read_first("results/task1/dl_cnn_chromosome/summary.csv"),
        ),
        (
            "Task 2",
            "Logistic L2",
            read_best("results/task2/task2_baseline_summary.csv", "f1_mean"),
        ),
        (
            "Task 2",
            "PLS-logistic",
            read_best(
                "results/task2/task2_svm_linear_summary.csv",
                "f1_mean",
                "model == 'pls5_logreg'",
            ),
        ),
        (
            "Task 2",
            "XGBoost",
            read_best(
                "results/task2/task2_xgboost_summary.csv",
                "f1_mean",
                "threshold_mode == 'tuned_f1'",
            ),
        ),
        (
            "Task 2",
            "Best DL",
            read_first(
                "results/task2/dl_tuned_mlp_narrow/task2_deep_learning_summary.csv"
            ),
        ),
        (
            "Task 2",
            "Chromosome CNN",
            read_first("results/task2/dl_cnn_chromosome/summary.csv"),
        ),
    ]

    for task, model_family, row in specs:
        rows.append(
            {
                "task": task,
                "model_family": model_family,
                "experiment": row.get("experiment", ""),
                "roc_auc": row["roc_auc_mean"],
                "pr_auc": row["pr_auc_mean"],
                "balanced_accuracy": row["balanced_accuracy_mean"],
                "f1": row["f1_mean"],
            }
        )

    return pd.DataFrame(rows)


def plot_metric_comparison(rows, metric_key, metric_label, output_name, ylim):
    colors = {
        "Logistic L2": "#33658A",
        "Linear SVC": "#2F4858",
        "PLS-logistic": "#2F4858",
        "XGBoost": "#F26419",
        "Best DL": "#7D5BA6",
        "Chromosome CNN": "#55A630",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)
    for axis, task in zip(axes, ["Task 1", "Task 2"]):
        task_df = rows[rows["task"] == task].reset_index(drop=True)
        x_positions = list(range(len(task_df)))
        bars = axis.bar(
            x_positions,
            task_df[metric_key],
            width=0.62,
            color=[colors.get(name, "#777777") for name in task_df["model_family"]],
            alpha=0.9,
        )
        axis.set_title(task)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(task_df["model_family"], rotation=30, ha="right")
        axis.set_ylim(*ylim)
        axis.set_ylabel(metric_label)
        axis.grid(axis="y", alpha=0.25)
        for tick, model_family in zip(axis.get_xticklabels(), task_df["model_family"]):
            tick.set_color(colors.get(model_family, "black"))
        for bar, value in zip(bars, task_df[metric_key]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(OUT_DIR / output_name, dpi=220)
    plt.close(fig)


def add_box(axis, x, y, w, h, text, color):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#222222", linewidth=1.0)
    axis.add_patch(rect)
    axis.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def add_arrow(axis, x1, y1, x2, y2):
    axis.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"},
    )


def plot_architectures():
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    for axis in axes:
        axis.set_axis_off()
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 6)

    axis = axes[0]
    axis.set_title("Task 1 best DL: bottleneck MLP", fontsize=10)
    boxes = [
        (0.2, 2.2, 1.6, 1.0, "t1 CpGs\n2000", "#DDE7F0"),
        (2.2, 2.2, 1.6, 1.0, "Linear\n64", "#BFD7EA"),
        (4.2, 2.2, 1.6, 1.0, "GELU +\nDropout", "#F7E1D7"),
        (6.2, 2.2, 1.6, 1.0, "Bottleneck\n16", "#C7EFCF"),
        (8.2, 2.2, 1.5, 1.0, "Logit", "#F2C6DE"),
    ]
    for box in boxes:
        add_box(axis, *box)
    for x in [1.8, 3.8, 5.8, 7.8]:
        add_arrow(axis, x, 2.7, x + 0.35, 2.7)

    axis = axes[1]
    axis.set_title("Task 2 best DL: regularised MLP", fontsize=10)
    boxes = [
        (0.2, 2.2, 1.6, 1.0, "t1 CpGs\n2000", "#DDE7F0"),
        (2.2, 2.2, 1.6, 1.0, "Linear\n256", "#BFD7EA"),
        (4.2, 2.2, 1.6, 1.0, "GELU +\nDropout", "#F7E1D7"),
        (6.2, 2.2, 1.6, 1.0, "Linear\n64", "#BFD7EA"),
        (8.2, 2.2, 1.5, 1.0, "Logit", "#F2C6DE"),
    ]
    for box in boxes:
        add_box(axis, *box)
    for x in [1.8, 3.8, 5.8, 7.8]:
        add_arrow(axis, x, 2.7, x + 0.35, 2.7)

    axis = axes[2]
    axis.set_title("Chromosome CNN", fontsize=10)
    add_box(axis, 0.2, 3.9, 1.7, 0.8, "chr 1 block", "#DDE7F0")
    add_box(axis, 0.2, 2.6, 1.7, 0.8, "chr 2 block", "#DDE7F0")
    add_box(axis, 0.2, 1.3, 1.7, 0.8, "chr n block", "#DDE7F0")
    add_box(axis, 2.6, 2.6, 2.0, 1.0, "Shared\nConv1D encoder", "#C7EFCF")
    add_box(axis, 5.3, 2.6, 1.5, 1.0, "Concat", "#F7E1D7")
    add_box(axis, 7.5, 2.6, 2.0, 1.0, "MLP head\nLogit", "#F2C6DE")
    for y in [4.3, 3.0, 1.7]:
        add_arrow(axis, 1.9, y, 2.6, 3.1)
    add_arrow(axis, 4.6, 3.1, 5.3, 3.1)
    add_arrow(axis, 6.8, 3.1, 7.5, 3.1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "architecture_schematics.png", dpi=220)
    plt.close(fig)


def plot_heatmap_composite():
    paths = [
        ROOT / "results/task1/figures/task1_t1_top200_subject_cpg_clustermap.png",
        ROOT / "results/task2/figures/task2_t1_top200_subject_cpg_clustermap.png",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for axis, path, title in zip(axes, paths, ["Task 1: t1 top-variable CpGs", "Task 2: t1 top-variable CpGs"]):
        image = plt.imread(path)
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "eda_heatmap_composite.png", dpi=220)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_model_rows()
    rows.to_csv(OUT_DIR / "model_metric_comparison_data.csv", index=False)
    plot_metric_comparison(rows, "roc_auc", "Mean ROC-AUC", "model_metric_comparison_roc_auc.png", (0.5, 1.0))
    plot_metric_comparison(rows, "f1", "Mean F1", "model_metric_comparison_f1.png", (0.3, 0.85))
    plot_architectures()
    plot_heatmap_composite()


if __name__ == "__main__":
    main()
