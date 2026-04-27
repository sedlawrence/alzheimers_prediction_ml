#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd

from _shared.data_io import load_binary_task
from _shared.dl_models import (
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    MAX_EPOCHS as DEFAULT_MAX_EPOCHS,
    N_REPEATS as DEFAULT_N_REPEATS,
    N_SPLITS as DEFAULT_N_SPLITS,
    PATIENCE as DEFAULT_PATIENCE,
    RANDOM_STATE,
    LEARNING_RATE as DEFAULT_LEARNING_RATE,
    VALIDATION_SIZE as DEFAULT_VALIDATION_SIZE,
    WEIGHT_DECAY as DEFAULT_WEIGHT_DECAY,
    evaluate_model_cv,
    get_effective_n_features,
    get_experiments,
    resolve_device,
    set_global_seed,
    summarise_cv_results,
    update_runtime_defaults,
)
from _shared.feature_sets import build_genome_order, make_flat_feature_sets, make_sequence_feature_sets
from _shared.task_config import ANNOTATION_PATH, TASK2_SPEC


H5_PATH = TASK2_SPEC.h5_path
OUT_DIR = TASK2_SPEC.out_dir
TASK_NAME = "Task 2: deep learning exploration"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 2 deep learning exploration (MLP, Siamese MLP, genome-sorted CNN)."
    )
    parser.add_argument(
        "--preset",
        default="full",
        choices=["full", "tuned_small"],
        help="Experiment preset. 'tuned_small' is the last-pass grid.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device. Use 'mps' on Apple Silicon, 'cuda' on NVIDIA, or 'auto'.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional list of experiment names to run (subset). If omitted, runs all.",
    )
    parser.add_argument(
        "--run_subdir",
        default=None,
        help="Optional subdirectory under results/task2 for outputs (helps concurrent runs).",
    )
    parser.add_argument("--n_splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--n_repeats", type=int, default=DEFAULT_N_REPEATS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--val_size", type=float, default=DEFAULT_VALIDATION_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    update_runtime_defaults(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=RANDOM_STATE,
    )

    if not H5_PATH.exists():
        raise FileNotFoundError(f"Could not find file: {H5_PATH.resolve()}")

    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"Could not find annotation file: {ANNOTATION_PATH.resolve()}"
        )

    set_global_seed(RANDOM_STATE)
    device = resolve_device(args.device)

    out_dir = OUT_DIR
    if args.run_subdir:
        out_dir = OUT_DIR / args.run_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, cpg_ids = load_binary_task(
        H5_PATH,
        TASK2_SPEC.class0_key,
        TASK2_SPEC.class1_key,
        TASK2_SPEC.cpg_key,
    )
    flat_feature_sets = make_flat_feature_sets(X)
    ordered_indices, genome_metadata = build_genome_order(cpg_ids, ANNOTATION_PATH)
    sequence_feature_sets = make_sequence_feature_sets(X, ordered_indices)
    all_feature_sets = {}
    all_feature_sets.update(flat_feature_sets)
    all_feature_sets.update(sequence_feature_sets)

    experiments = get_experiments(args.preset)
    if args.experiments:
        wanted = set(args.experiments)
        experiments = [e for e in experiments if e["experiment"] in wanted]
        if not experiments:
            raise ValueError(f"No experiments matched: {args.experiments}")

    print("=" * 80)
    print(TASK_NAME)
    print(f"{TASK2_SPEC.class0_label} vs {TASK2_SPEC.class1_label}")
    print("=" * 80)
    print(f"Preset: {args.preset}")
    print(f"HDF5 file: {H5_PATH.resolve()}")
    print(f"Annotation file: {ANNOTATION_PATH.resolve()}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Device: {device}")
    print(f"CV setting: {args.n_splits} folds x {args.n_repeats} repeats")
    print(
        "Training: "
        f"batch={args.batch_size}, epochs={args.max_epochs}, patience={args.patience}, "
        f"lr={args.lr}, wd={args.weight_decay}, val={args.val_size}"
    )
    print(f"X shape after transpose: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"class counts [0={TASK2_SPEC.class0_label}, 1={TASK2_SPEC.class1_label}]: {np.bincount(y)}")
    print(f"number of CpG IDs: {len(cpg_ids)}")
    print(f"genome-mapped CpGs: {genome_metadata['MAPINFO'].notna().sum()} / {len(cpg_ids)}")

    genome_order_path = out_dir / "task2_genome_ordered_cpgs.csv"
    genome_metadata.to_csv(genome_order_path, index=False)
    experiment_specs_path = out_dir / "task2_deep_learning_experiment_specs.csv"
    experiment_specs_df = pd.DataFrame(experiments)
    experiment_specs_df.insert(0, "preset", args.preset)
    experiment_specs_df.to_csv(experiment_specs_path, index=False)

    all_summaries = []
    all_fold_results = []

    for exp in experiments:
        exp_name = exp["experiment"]
        input_key = exp["input_key"]
        X_exp = all_feature_sets[input_key]

        print("\n" + "-" * 80)
        print(f"Running: {exp_name}")
        print(f"Input key: {input_key}")
        print(f"Model type: {exp['model_type']}")
        print(f"Input shape: {X_exp.shape}")

        fold_df, pooled = evaluate_model_cv(
            exp=exp,
            X=X_exp,
            y=y,
            device=device,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
        )

        summary = summarise_cv_results(fold_df)
        for key, value in pooled.items():
            summary[f"pooled_{key}"] = value

        summary.update({
            "experiment": exp_name,
            "input_key": input_key,
            "model_type": exp["model_type"],
            "preset": args.preset,
            "n_samples": X_exp.shape[0],
            "n_features": int(np.prod(X_exp.shape[1:])),
            "effective_n_features": get_effective_n_features(exp, X_exp),
            "n_cv_fits": args.n_splits * args.n_repeats,
            "device": str(device),
        })

        all_summaries.append(summary)

        fold_df.insert(0, "experiment", exp_name)
        fold_df.insert(1, "input_key", input_key)
        fold_df.insert(2, "model_type", exp["model_type"])
        all_fold_results.append(fold_df)

        print(f"Mean ROC-AUC:            {summary['roc_auc_mean']:.3f} ± {summary['roc_auc_std']:.3f}")
        print(f"Mean PR-AUC:             {summary['pr_auc_mean']:.3f} ± {summary['pr_auc_std']:.3f}")
        print(f"Mean balanced acc:       {summary['balanced_accuracy_mean']:.3f} ± {summary['balanced_accuracy_std']:.3f}")
        print(f"Mean recall/sensitivity: {summary['recall_sensitivity_mean']:.3f} ± {summary['recall_sensitivity_std']:.3f}")
        print(f"Mean specificity:        {summary['specificity_mean']:.3f} ± {summary['specificity_std']:.3f}")
        print(f"Mean F1:                 {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}")
        print(f"Mean best epoch:         {summary['best_epoch_mean']:.1f}")
        print(f"Mean parameters:         {summary['n_parameters_mean']:.0f}")

    summary_df = pd.DataFrame(all_summaries)
    fold_results_df = pd.concat(all_fold_results, ignore_index=True)

    first_cols = [
        "experiment",
        "input_key",
        "model_type",
        "preset",
        "n_samples",
        "n_features",
        "effective_n_features",
        "n_cv_fits",
        "device",
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
        "best_epoch_mean",
        "best_epoch_std",
        "best_val_loss_mean",
        "best_val_loss_std",
        "n_parameters_mean",
        "n_parameters_std",
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

    out_summary = out_dir / "task2_deep_learning_summary.csv"
    out_folds = out_dir / "task2_deep_learning_fold_results.csv"

    summary_df.to_csv(out_summary, index=False)
    fold_results_df.to_csv(out_folds, index=False)

    print("\n" + "=" * 80)
    print("Final deep learning summary, sorted by mean ROC-AUC")
    print("=" * 80)

    display_cols = [
        "experiment",
        "input_key",
        "model_type",
        "roc_auc_mean",
        "pr_auc_mean",
        "balanced_accuracy_mean",
        "recall_sensitivity_mean",
        "specificity_mean",
        "f1_mean",
        "best_epoch_mean",
        "n_parameters_mean",
    ]

    summary_df = summary_df.sort_values(
        by=["roc_auc_mean", "pr_auc_mean", "balanced_accuracy_mean"],
        ascending=False,
    )

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary_df[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {out_summary.resolve()}")
    print(f"  {out_folds.resolve()}")
    print(f"  {experiment_specs_path.resolve()}")
    print(f"  {genome_order_path.resolve()}")


if __name__ == "__main__":
    main()
