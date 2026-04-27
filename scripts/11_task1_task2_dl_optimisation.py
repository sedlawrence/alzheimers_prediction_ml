#!/usr/bin/env python3

import copy
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "This script requires PyTorch. Install `torch` before running."
    ) from exc

from _shared.data_io import load_binary_task
from _shared.dl_models import (
    BottleneckMLP,
    FeatureStandardizer,
    RegularizedMLP,
    calculate_metrics,
    count_trainable_params,
    resolve_device,
    set_global_seed,
)
from _shared.feature_sets import build_genome_order, make_flat_feature_sets
from _shared.task_config import ANNOTATION_PATH, RANDOM_STATE, TASK1_SPEC, TASK2_SPEC


OUTER_SPLITS = 5
OUTER_REPEATS = 3
INNER_SPLITS = 2

BATCH_SIZE = 16
MAX_EPOCHS = 150
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
VALIDATION_SIZE = 0.20

SELECTION_METRIC = "pr_auc"


class ChromosomeEncoder(nn.Module):
    def __init__(self, in_channels, conv_channels=(16, 32), dropout=0.4, pool_size=1, branch_dim=16):
        super().__init__()
        c1, c2 = conv_channels
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(pool_size),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * pool_size, branch_dim),
            nn.GELU(),
        )

    def forward(self, X):
        return self.projection(self.features(X))


class ChromosomeBranchCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        chromosome_slices,
        conv_channels=(16, 32),
        dropout=0.4,
        pool_size=1,
        branch_dim=16,
        head_dim=64,
    ):
        super().__init__()
        self.chromosome_slices = list(chromosome_slices)
        self.encoder = ChromosomeEncoder(
            in_channels=in_channels,
            conv_channels=conv_channels,
            dropout=dropout,
            pool_size=pool_size,
            branch_dim=branch_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(branch_dim * len(self.chromosome_slices), head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, X):
        branch_embeddings = []
        for chromosome_slice in self.chromosome_slices:
            branch_embeddings.append(self.encoder(X[:, :, chromosome_slice]))
        features = torch.cat(branch_embeddings, dim=1)
        return self.head(features).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiny nested-CV tuning for Task 1 and Task 2 deep learning."
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device. Use 'mps' on Apple Silicon, 'cuda' on NVIDIA, or 'auto'.",
    )
    parser.add_argument("--outer_splits", type=int, default=OUTER_SPLITS)
    parser.add_argument("--outer_repeats", type=int, default=OUTER_REPEATS)
    parser.add_argument("--inner_splits", type=int, default=INNER_SPLITS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--val_size", type=float, default=VALIDATION_SIZE)
    parser.add_argument(
        "--selection_metric",
        default=SELECTION_METRIC,
        choices=["pr_auc", "f1", "balanced_accuracy", "roc_auc"],
        help="Inner-CV metric used to pick the winning configuration.",
    )
    return parser.parse_args()



def configure_runtime(args):
    global OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS
    global BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY, VALIDATION_SIZE
    OUTER_SPLITS = args.outer_splits
    OUTER_REPEATS = args.outer_repeats
    INNER_SPLITS = args.inner_splits
    BATCH_SIZE = args.batch_size
    MAX_EPOCHS = args.max_epochs
    PATIENCE = args.patience
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = args.weight_decay
    VALIDATION_SIZE = args.val_size


def tensor_dataset_from_numpy(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def get_positive_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return model.predict(X_test)


def build_tabular_candidates():
    return [
        {
            "experiment": "mlp_t1_only_wide",
            "model_name": "mlp_t1_only_wide",
            "model_type": "mlp",
            "input_key": "t1_only",
            "hidden_dims": (256, 64),
            "dropout": 0.5,
            "lr": 3e-4,
            "weight_decay": 1e-4,
        },
        {
            "experiment": "mlp_t1_only_narrow",
            "model_name": "mlp_t1_only_narrow",
            "model_type": "mlp",
            "input_key": "t1_only",
            "hidden_dims": (128, 32),
            "dropout": 0.4,
            "lr": 1e-3,
            "weight_decay": 1e-3,
        },
        {
            "experiment": "bottleneck_mlp_t1_only_wide",
            "model_name": "bottleneck_mlp_t1_only_wide",
            "model_type": "bottleneck_mlp",
            "input_key": "t1_only",
            "hidden_dim": 128,
            "bottleneck_dim": 32,
            "dropout": 0.5,
            "lr": 3e-4,
            "weight_decay": 1e-4,
        },
        {
            "experiment": "bottleneck_mlp_t1_only_narrow",
            "model_name": "bottleneck_mlp_t1_only_narrow",
            "model_type": "bottleneck_mlp",
            "input_key": "t1_only",
            "hidden_dim": 64,
            "bottleneck_dim": 16,
            "dropout": 0.4,
            "lr": 1e-3,
            "weight_decay": 1e-3,
        },
    ]


def build_chromosome_candidates():
    return [
        {
            "experiment": "chromosome_cnn_narrow",
            "model_name": "chromosome_cnn_narrow",
            "model_type": "chromosome_cnn",
            "input_key": "t0_t1_genome_pair",
            "conv_channels": (16, 32),
            "branch_dim": 16,
            "head_dim": 64,
            "pool_size": 1,
            "dropout": 0.5,
            "lr": 1e-3,
            "weight_decay": 1e-3,
        },
        {
            "experiment": "chromosome_cnn_wide",
            "model_name": "chromosome_cnn_wide",
            "model_type": "chromosome_cnn",
            "input_key": "t0_t1_genome_pair",
            "conv_channels": (32, 64),
            "branch_dim": 32,
            "head_dim": 64,
            "pool_size": 1,
            "dropout": 0.4,
            "lr": 3e-4,
            "weight_decay": 1e-4,
        },
    ]


def build_tabular_model(candidate, input_shape):
    if candidate["model_type"] == "mlp":
        return RegularizedMLP(
            input_dim=input_shape[1],
            hidden_dims=candidate["hidden_dims"],
            dropout=candidate["dropout"],
        )

    if candidate["model_type"] == "bottleneck_mlp":
        return BottleneckMLP(
            input_dim=input_shape[1],
            hidden_dim=candidate["hidden_dim"],
            bottleneck_dim=candidate["bottleneck_dim"],
            dropout=candidate["dropout"],
        )

    raise ValueError(f"Unknown tabular model_type: {candidate['model_type']}")


def build_chromosome_model(candidate, input_shape, chromosome_slices):
    if candidate["model_type"] != "chromosome_cnn":
        raise ValueError(f"Unknown chromosome model_type: {candidate['model_type']}")

    return ChromosomeBranchCNN(
        in_channels=input_shape[1],
        chromosome_slices=chromosome_slices,
        conv_channels=candidate["conv_channels"],
        dropout=candidate["dropout"],
        pool_size=candidate["pool_size"],
        branch_dim=candidate["branch_dim"],
        head_dim=candidate["head_dim"],
    )


def train_one_fold_generic(candidate, X_train, y_train, X_eval, device, model_builder):
    batch_size = candidate.get("batch_size") or BATCH_SIZE
    max_epochs = candidate.get("max_epochs") or MAX_EPOCHS
    patience = candidate.get("patience") or PATIENCE
    lr = candidate.get("lr") or LEARNING_RATE
    weight_decay = candidate.get("weight_decay") or WEIGHT_DECAY
    val_size = candidate.get("val_size") or VALIDATION_SIZE

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    X_subtrain = X_train[train_idx]
    y_subtrain = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    standardizer = FeatureStandardizer().fit(X_subtrain)
    X_subtrain_scaled = standardizer.transform(X_subtrain)
    X_val_scaled = standardizer.transform(X_val)
    X_eval_scaled = standardizer.transform(X_eval)

    train_loader = DataLoader(
        tensor_dataset_from_numpy(X_subtrain_scaled, y_subtrain),
        batch_size=batch_size,
        shuffle=True,
    )

    model = model_builder(X_subtrain_scaled.shape).to(device)

    n_neg = np.sum(y_subtrain == 0)
    n_pos = np.sum(y_subtrain == 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
            val_logits = model(X_val_tensor)
            val_loss = loss_fn(val_logits, y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    y_score = predict_scores(model, X_eval_scaled, device)

    return {
        "y_score": y_score,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "n_parameters": count_trainable_params(model),
    }


def predict_scores(model, X, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def chromosome_blocks_from_metadata(metadata):
    labels = metadata["CHR"].fillna("unmapped").astype(str).replace({"nan": "unmapped"})
    block_rows = []
    start = 0
    block_id = 1

    while start < len(metadata):
        current_label = labels.iloc[start]
        end = start + 1
        while end < len(metadata) and labels.iloc[end] == current_label:
            end += 1

        block_rows.append(
            {
                "chromosome_block": block_id,
                "chromosome_label": current_label,
                "start_rank": start,
                "end_rank": end - 1,
                "n_cpgs": end - start,
            }
        )
        block_id += 1
        start = end

    block_df = pd.DataFrame(block_rows)
    chromosome_slices = [slice(row.start_rank, row.end_rank + 1) for row in block_df.itertuples(index=False)]
    return chromosome_slices, block_df


def summarise_fold_results(fold_df):
    metric_cols = [
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
    summary = {}
    for col in metric_cols:
        summary[f"{col}_mean"] = fold_df[col].mean()
        summary[f"{col}_std"] = fold_df[col].std()
    return summary


def evaluate_nested_cv(candidate_grid, X, y, device, model_builder_factory):
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )
    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    outer_rows = []
    candidate_rows = []
    pooled_y_true = []
    pooled_y_score = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        fold_candidate_scores = []
        for candidate in candidate_grid:
            inner_values = []
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train, y_train), start=1):
                build_model = model_builder_factory(candidate)
                inner_output = train_one_fold_generic(
                    candidate=candidate,
                    X_train=X_train[inner_train_idx],
                    y_train=y_train[inner_train_idx],
                    X_eval=X_train[inner_val_idx],
                    device=device,
                    model_builder=build_model,
                )
                inner_metrics = calculate_metrics(
                    y_true=y_train[inner_val_idx],
                    y_score=inner_output["y_score"],
                    threshold=0.5,
                )
                inner_values.append(inner_metrics[SELECTION_METRIC])
            mean_inner = float(np.mean(inner_values))
            fold_candidate_scores.append((mean_inner, candidate))
            candidate_rows.append({
                "outer_fold": outer_fold,
                "inner_selection_metric_mean": mean_inner,
                "selection_metric": SELECTION_METRIC,
                **candidate,
            })

        best_inner_score, best_candidate = max(fold_candidate_scores, key=lambda item: item[0])
        build_model = model_builder_factory(best_candidate)
        outer_output = train_one_fold_generic(
            candidate=best_candidate,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            device=device,
            model_builder=build_model,
        )
        outer_metrics = calculate_metrics(
            y_true=y_test,
            y_score=outer_output["y_score"],
            threshold=0.5,
        )

        outer_rows.append({
            "fold": outer_fold,
            "selected_model": best_candidate["model_name"],
            "selected_model_type": best_candidate["model_type"],
            "selected_inner_selection_metric": best_inner_score,
            "best_epoch": outer_output["best_epoch"],
            "best_val_loss": outer_output["best_val_loss"],
            "n_parameters": outer_output["n_parameters"],
            **outer_metrics,
        })

        pooled_y_true.extend(y_test.tolist())
        pooled_y_score.extend(outer_output["y_score"].tolist())

    fold_df = pd.DataFrame(outer_rows)
    candidate_df = pd.DataFrame(candidate_rows)

    pooled_metrics = calculate_metrics(
        y_true=np.asarray(pooled_y_true),
        y_score=np.asarray(pooled_y_score),
        threshold=0.5,
    )

    candidate_keys = ["experiment", "model_name", "model_type", "input_key", "selection_metric"]
    candidate_summary = candidate_df.groupby(
        candidate_keys,
        as_index=False,
    ).agg(
        inner_selection_metric_mean=("inner_selection_metric_mean", "mean"),
        inner_selection_metric_std=("inner_selection_metric_mean", "std"),
        selected_count=("inner_selection_metric_mean", "size"),
    )

    param_cols = [
        c for c in candidate_df.columns
        if c not in {"outer_fold", "inner_selection_metric_mean", "selection_metric", *candidate_keys}
    ]
    if param_cols:
        params = candidate_df[candidate_keys + param_cols].drop_duplicates(subset=candidate_keys)
        candidate_summary = candidate_summary.merge(params, on=candidate_keys, how="left")

    return fold_df, candidate_summary, pooled_metrics


def write_candidate_outputs(out_dir, experiment_specs, candidate_grid, selected_row, fold_df, candidate_summary, pooled_metrics):
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarise_fold_results(fold_df)
    for key, value in pooled_metrics.items():
        summary[f"pooled_{key}"] = value

    candidate_count = len(candidate_grid)
    outer_fold_count = len(fold_df)
    estimated_total_model_fits = outer_fold_count * (candidate_count * INNER_SPLITS + 1)

    summary.update({
        "experiment": selected_row["experiment"],
        "model_name": selected_row["model_name"],
        "model_type": selected_row["model_type"],
        "input_key": selected_row["input_key"],
        "selection_metric": selected_row["selection_metric"],
        "selected_outer_folds": outer_fold_count,
        "n_candidates": candidate_count,
        "n_outer_fits": outer_fold_count,
        "n_samples": experiment_specs["n_samples"],
        "n_features": experiment_specs["n_features"],
        "effective_n_features": experiment_specs["effective_n_features"],
        "n_cv_fits": outer_fold_count,
        "estimated_total_model_fits": estimated_total_model_fits,
    })

    summary_df = pd.DataFrame([summary])
    fold_df.to_csv(out_dir / "fold_results.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame([experiment_specs]).to_csv(out_dir / "experiment_specs.csv", index=False)
    pd.DataFrame([selected_row]).to_csv(out_dir / "selected_hyperparameters.csv", index=False)
    candidate_grid.to_csv(out_dir / "candidate_grid.csv", index=False)
    candidate_summary.to_csv(out_dir / "candidate_summary.csv", index=False)

    return summary_df


def task_payload(task_spec):
    X, y, cpg_ids = load_binary_task(
        task_spec.h5_path,
        task_spec.class0_key,
        task_spec.class1_key,
        task_spec.cpg_key,
    )
    flat_feature_sets = make_flat_feature_sets(X)
    ordered_indices, chromosome_mapping = build_genome_order(cpg_ids, ANNOTATION_PATH)
    sequence_feature_sets = {
        "t0_t1_genome_pair": X[:, :, ordered_indices],
    }
    chromosome_blocks, block_df = chromosome_blocks_from_metadata(chromosome_mapping)

    chromosome_mapping = chromosome_mapping.copy()
    chromosome_mapping["chromosome_label"] = chromosome_mapping["CHR"].fillna("unmapped").astype(str).replace({"nan": "unmapped"})
    chromosome_mapping["chromosome_block"] = 0
    for block in block_df.itertuples(index=False):
        chromosome_mapping.loc[
            chromosome_mapping["genome_rank"].between(block.start_rank, block.end_rank),
            "chromosome_block",
        ] = block.chromosome_block
    chromosome_mapping["within_chromosome_rank"] = chromosome_mapping.groupby("chromosome_block").cumcount() + 1

    return X, y, cpg_ids, flat_feature_sets, sequence_feature_sets, chromosome_blocks, chromosome_mapping


def make_tabular_builder(candidate):
    def build_model(input_shape):
        return build_tabular_model(candidate, input_shape)

    return build_model


def make_chromosome_builder(candidate, chromosome_blocks):
    def build_model(input_shape):
        return build_chromosome_model(candidate, input_shape, chromosome_blocks)

    return build_model


def run_task(task_name, task_spec, device):
    X, y, cpg_ids, flat_feature_sets, sequence_feature_sets, chromosome_blocks, chromosome_mapping = task_payload(task_spec)
    out_root = task_spec.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"{task_spec.name.upper()} optimisation")
    print(f"{task_spec.class0_label} vs {task_spec.class1_label}")
    print("=" * 80)
    print(f"HDF5 file: {task_spec.h5_path.resolve()}")
    print(f"Output root: {out_root.resolve()}")
    print(f"Device: {device}")
    print(f"Outer CV: {OUTER_SPLITS} folds x {OUTER_REPEATS} repeats")
    print(f"Inner CV: {INNER_SPLITS} folds")
    print(f"Selection metric: {SELECTION_METRIC}")
    print(f"X shape after transpose: {X.shape}")
    print(f"class counts: {np.bincount(y)}")
    print(f"number of CpG IDs: {len(cpg_ids)}")

    # Optimise the best tabular DL family on t1_only.
    tabular_candidates = build_tabular_candidates()
    tabular_candidate_grid = pd.DataFrame(tabular_candidates)
    tabular_X = flat_feature_sets["t1_only"]
    tabular_fold_df, tabular_candidate_df, tabular_pooled = evaluate_nested_cv(
        tabular_candidates,
        tabular_X,
        y,
        device,
        make_tabular_builder,
    )
    tabular_best_candidate = (
        tabular_candidate_df.sort_values(
            by=["inner_selection_metric_mean", "selected_count"],
            ascending=False,
        ).iloc[0].to_dict()
    )
    tabular_best_name = str(tabular_best_candidate["model_name"])
    tabular_out_dir = out_root / f"dl_optimised_{tabular_best_name}"
    tabular_specs = {
        "task": task_name,
        "family": "tabular",
        "selection_metric": SELECTION_METRIC,
        "outer_splits": OUTER_SPLITS,
        "outer_repeats": OUTER_REPEATS,
        "inner_splits": INNER_SPLITS,
        "candidate_count": int(len(tabular_candidate_grid)),
        "n_samples": int(tabular_X.shape[0]),
        "n_features": int(tabular_X.shape[1]),
        "effective_n_features": int(tabular_X.shape[1]),
        "device": str(device),
    }
    tabular_summary_df = write_candidate_outputs(
        tabular_out_dir,
        tabular_specs,
        tabular_candidate_grid,
        tabular_best_candidate,
        tabular_fold_df,
        tabular_candidate_df,
        tabular_pooled,
    )

    # Chromosome-aware CNN on genome-ordered CpGs.
    chromosome_candidates = build_chromosome_candidates()
    chromosome_candidate_grid = pd.DataFrame(chromosome_candidates)
    chromosome_X = sequence_feature_sets["t0_t1_genome_pair"]
    chromosome_fold_df, chromosome_candidate_df, chromosome_pooled = evaluate_nested_cv(
        chromosome_candidates,
        chromosome_X,
        y,
        device,
        lambda candidate: make_chromosome_builder(candidate, chromosome_blocks),
    )
    chromosome_best_candidate = (
        chromosome_candidate_df.sort_values(
            by=["inner_selection_metric_mean", "selected_count"],
            ascending=False,
        ).iloc[0].to_dict()
    )
    chromosome_out_dir = out_root / "dl_cnn_chromosome"
    chromosome_specs = {
        "task": task_name,
        "family": "chromosome_cnn",
        "selection_metric": SELECTION_METRIC,
        "outer_splits": OUTER_SPLITS,
        "outer_repeats": OUTER_REPEATS,
        "inner_splits": INNER_SPLITS,
        "candidate_count": int(len(chromosome_candidate_grid)),
        "n_samples": int(chromosome_X.shape[0]),
        "n_features": int(chromosome_X.shape[2]),
        "effective_n_features": int(chromosome_X.shape[2]),
        "device": str(device),
    }
    chromosome_summary_df = write_candidate_outputs(
        chromosome_out_dir,
        chromosome_specs,
        chromosome_candidate_grid,
        chromosome_best_candidate,
        chromosome_fold_df,
        chromosome_candidate_df,
        chromosome_pooled,
    )
    chromosome_mapping = chromosome_mapping.sort_values("genome_rank").reset_index(drop=True)
    chromosome_mapping.to_csv(chromosome_out_dir / "chromosome_mapping.csv", index=False)

    overview = pd.DataFrame([
        {
            "task": task_name,
            "run_family": "tabular",
            "best_model_name": tabular_best_name,
            "roc_auc_mean": tabular_summary_df.iloc[0]["roc_auc_mean"],
            "pr_auc_mean": tabular_summary_df.iloc[0]["pr_auc_mean"],
            "balanced_accuracy_mean": tabular_summary_df.iloc[0]["balanced_accuracy_mean"],
            "f1_mean": tabular_summary_df.iloc[0]["f1_mean"],
            "output_dir": str(tabular_out_dir.resolve()),
        },
        {
            "task": task_name,
            "run_family": "chromosome_cnn",
            "best_model_name": chromosome_best_candidate["model_name"],
            "roc_auc_mean": chromosome_summary_df.iloc[0]["roc_auc_mean"],
            "pr_auc_mean": chromosome_summary_df.iloc[0]["pr_auc_mean"],
            "balanced_accuracy_mean": chromosome_summary_df.iloc[0]["balanced_accuracy_mean"],
            "f1_mean": chromosome_summary_df.iloc[0]["f1_mean"],
            "output_dir": str(chromosome_out_dir.resolve()),
        },
    ])
    overview.to_csv(out_root / "dl_optimisation_overview.csv", index=False)

    print("\nOptimised tabular model:")
    print(f"  {tabular_best_name}")
    print("Chromosome CNN:")
    print(f"  {chromosome_best_candidate['model_name']}")
    print(f"Saved task overview: {(out_root / 'dl_optimisation_overview.csv').resolve()}")


def main():
    args = parse_args()
    configure_runtime(args)

    set_global_seed(RANDOM_STATE)
    device = resolve_device(args.device)

    for task_name, task_spec in [
        ("task1", TASK1_SPEC),
        ("task2", TASK2_SPEC),
    ]:
        run_task(task_name, task_spec, device)


if __name__ == "__main__":
    main()
