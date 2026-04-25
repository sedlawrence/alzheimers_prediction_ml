#!/usr/bin/env python3

import copy
import argparse
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

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
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "This script requires PyTorch. Install `torch` before running."
    ) from exc


warnings.filterwarnings("ignore", category=UserWarning)


H5_PATH = Path("../data/temporal_two_sets_n2000.h5")
ANNOTATION_PATH = Path("../data/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz")
OUT_DIR = Path("../results/task1")

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3

BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
VALIDATION_SIZE = 0.20


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str):
    device_name = str(device_name).strip().lower()
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "mps":
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")

    return torch.device("cpu")


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

    x0 = np.transpose(x_cn_to_cn, (2, 1, 0))
    x1 = np.transpose(x_cn_to_mci, (2, 1, 0))

    X = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([
        np.zeros(x0.shape[0], dtype=int),
        np.ones(x1.shape[0], dtype=int),
    ])

    cpg_ids = np.array([
        c.decode("utf-8") if isinstance(c, bytes) else str(c)
        for c in cpg_ids
    ])

    return X.astype(np.float32), y.astype(int), cpg_ids


def make_flat_feature_sets(X):
    """
    Build classic tabular feature views.
    """
    t0 = X[:, 0, :]
    t1 = X[:, 1, :]
    delta = t1 - t0

    return {
        "t1_only": t1,
        "t0_t1": np.concatenate([t0, t1], axis=1),
        "t0_t1_delta": np.concatenate([t0, t1, delta], axis=1),
    }


def make_sequence_feature_sets(X, ordered_indices):
    """
    Build sequence-like views for Siamese and CNN models.
    """
    t0 = X[:, 0, :]
    t1 = X[:, 1, :]
    delta = t1 - t0

    return {
        "t0_t1_pair": X,
        "t0_t1_genome_pair": X[:, :, ordered_indices],
        "t0_t1_delta_genome": np.stack(
            [t0[:, ordered_indices], t1[:, ordered_indices], delta[:, ordered_indices]],
            axis=1,
        ),
    }


def chromosome_sort_key(value):
    value = str(value).strip().upper()
    if value in {"X", "CHR X"}:
        return 23
    if value in {"Y", "CHR Y"}:
        return 24
    try:
        return int(value.replace("CHR", "").strip())
    except ValueError:
        return 999


def build_genome_order(cpg_ids, annotation_path: Path):
    """
    Use the Illumina annotation to place CpGs in chromosome / genomic order.
    """
    annotation_df = pd.read_csv(
        annotation_path,
        skiprows=7,
        usecols=["Name", "CHR", "MAPINFO"],
        low_memory=False,
    )
    annotation_df = annotation_df.rename(columns={"Name": "cpg_id"})
    annotation_df["MAPINFO"] = pd.to_numeric(annotation_df["MAPINFO"], errors="coerce")
    annotation_df["chr_order"] = annotation_df["CHR"].map(chromosome_sort_key)
    annotation_df = annotation_df.drop_duplicates(subset="cpg_id", keep="first")

    cpg_df = pd.DataFrame({
        "cpg_id": cpg_ids,
        "original_index": np.arange(len(cpg_ids)),
    })

    merged = cpg_df.merge(annotation_df, on="cpg_id", how="left")

    found_mask = merged["MAPINFO"].notna() & (merged["chr_order"] < 999)
    found = merged[found_mask].sort_values(
        by=["chr_order", "MAPINFO", "original_index"]
    )
    missing = merged[~found_mask].sort_values(by=["original_index"])

    ordered = pd.concat([found, missing], ignore_index=True)
    ordered_indices = ordered["original_index"].to_numpy(dtype=int)

    metadata = ordered[[
        "cpg_id", "original_index", "CHR", "MAPINFO", "chr_order"
    ]].copy()
    metadata["genome_rank"] = np.arange(len(metadata))

    return ordered_indices, metadata


def get_effective_n_features(exp, X_exp):
    """
    Number of effective feature positions seen by the model.
    """
    if exp["model_type"] in {"siamese_mlp", "genome_cnn"}:
        return X_exp.shape[-1]

    return X_exp.shape[1]


def make_experiment(
    experiment,
    input_key,
    model_type,
    *,
    dropout,
    batch_size=None,
    max_epochs=None,
    patience=None,
    lr=None,
    weight_decay=None,
    val_size=None,
    hidden_dims=None,
    hidden_dim=None,
    bottleneck_dim=None,
    embed_dim=None,
    conv_channels=None,
):
    return {
        "experiment": experiment,
        "input_key": input_key,
        "model_type": model_type,
        "dropout": dropout,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "val_size": val_size,
        "hidden_dims": hidden_dims,
        "hidden_dim": hidden_dim,
        "bottleneck_dim": bottleneck_dim,
        "embed_dim": embed_dim,
        "conv_channels": conv_channels,
    }


class FeatureStandardizer:
    """
    Standardize arbitrary sample tensors feature-wise across axis 0.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean_) / self.std_).astype(np.float32)


class RegularizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 64), dropout=0.5):
        super().__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X).squeeze(-1)


class BottleneckMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, bottleneck_dim=16, dropout=0.4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, X):
        return self.network(X).squeeze(-1)


class SiameseTimepointMLP(nn.Module):
    def __init__(self, input_dim, embed_dim=32, hidden_dim=128, dropout=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X):
        t0 = X[:, 0, :]
        t1 = X[:, 1, :]

        e0 = self.encoder(t0)
        e1 = self.encoder(t1)
        features = torch.cat([e0, e1, e1 - e0, torch.abs(e1 - e0)], dim=1)
        return self.head(features).squeeze(-1)


class GenomeCNN(nn.Module):
    def __init__(self, in_channels, length, dropout=0.4, conv_channels=(32, 64, 64)):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c2, c3, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 16, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.length = length

    def forward(self, X):
        X = self.features(X)
        return self.classifier(X).squeeze(-1)


def get_experiments(preset="full"):
    """
    Deep learning experiments for Task 1.
    """
    tuned_small = [
        make_experiment(
            "mlp_t1_only_wide",
            "t1_only",
            "mlp",
            dropout=0.5,
            hidden_dims=(128, 32),
            lr=1e-3,
            weight_decay=1e-3,
        ),
        make_experiment(
            "mlp_t1_only_narrow",
            "t1_only",
            "mlp",
            dropout=0.4,
            hidden_dims=(256, 64),
            lr=3e-4,
            weight_decay=1e-4,
        ),
        make_experiment(
            "bottleneck_mlp_t1_only_wide",
            "t1_only",
            "bottleneck_mlp",
            dropout=0.5,
            hidden_dim=64,
            bottleneck_dim=16,
            lr=1e-3,
            weight_decay=1e-3,
        ),
        make_experiment(
            "bottleneck_mlp_t1_only_narrow",
            "t1_only",
            "bottleneck_mlp",
            dropout=0.4,
            hidden_dim=128,
            bottleneck_dim=32,
            lr=3e-4,
            weight_decay=1e-4,
        ),
        make_experiment(
            "siamese_mlp_t0_t1_small",
            "t0_t1_pair",
            "siamese_mlp",
            dropout=0.5,
            embed_dim=16,
            hidden_dim=64,
            lr=1e-3,
            weight_decay=1e-3,
        ),
        make_experiment(
            "siamese_mlp_t0_t1_wide",
            "t0_t1_pair",
            "siamese_mlp",
            dropout=0.4,
            embed_dim=32,
            hidden_dim=128,
            lr=3e-4,
            weight_decay=1e-4,
        ),
    ]

    full = [
        make_experiment("mlp_t1_only", "t1_only", "mlp", dropout=0.5, hidden_dims=(256, 64)),
        make_experiment("mlp_t0_t1", "t0_t1", "mlp", dropout=0.5, hidden_dims=(256, 64)),
        make_experiment("mlp_t0_t1_delta", "t0_t1_delta", "mlp", dropout=0.5, hidden_dims=(256, 64)),
        make_experiment("bottleneck_mlp_t1_only", "t1_only", "bottleneck_mlp", dropout=0.4, hidden_dim=64, bottleneck_dim=16),
        make_experiment("bottleneck_mlp_t0_t1_delta", "t0_t1_delta", "bottleneck_mlp", dropout=0.4, hidden_dim=64, bottleneck_dim=16),
        make_experiment("siamese_mlp_t0_t1", "t0_t1_pair", "siamese_mlp", dropout=0.4, embed_dim=32, hidden_dim=128),
        make_experiment("genome_cnn_t0_t1", "t0_t1_genome_pair", "genome_cnn", dropout=0.4, conv_channels=(32, 64, 64)),
        make_experiment("genome_cnn_t0_t1_delta", "t0_t1_delta_genome", "genome_cnn", dropout=0.4, conv_channels=(32, 64, 64)),
    ]

    if preset == "tuned_small":
        return tuned_small

    if preset == "full":
        return full

    raise ValueError(f"Unknown preset: {preset}")


def build_model(exp, X_shape):
    """
    Create a model instance for one experiment.
    """
    if exp["model_type"] == "mlp":
        return RegularizedMLP(
            input_dim=X_shape[1],
            hidden_dims=exp.get("hidden_dims") or (256, 64),
            dropout=exp["dropout"],
        )

    if exp["model_type"] == "bottleneck_mlp":
        return BottleneckMLP(
            input_dim=X_shape[1],
            hidden_dim=exp.get("hidden_dim") or 64,
            bottleneck_dim=exp.get("bottleneck_dim") or 16,
            dropout=exp["dropout"],
        )

    if exp["model_type"] == "siamese_mlp":
        return SiameseTimepointMLP(
            input_dim=X_shape[2],
            embed_dim=exp.get("embed_dim") or 32,
            hidden_dim=exp.get("hidden_dim") or 128,
            dropout=exp["dropout"],
        )

    if exp["model_type"] == "genome_cnn":
        return GenomeCNN(
            in_channels=X_shape[1],
            length=X_shape[2],
            dropout=exp["dropout"],
            conv_channels=exp.get("conv_channels") or (32, 64, 64),
        )

    raise ValueError(f"Unknown model_type: {exp['model_type']}")


def tensor_dataset_from_numpy(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def calculate_metrics(y_true, y_score, threshold=0.5):
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


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_scores(model, X, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def train_one_fold(exp, X_train, y_train, X_test, device):
    """
    Fit one deep model with inner validation-based early stopping.
    """
    batch_size = exp.get("batch_size") or BATCH_SIZE
    max_epochs = exp.get("max_epochs") or MAX_EPOCHS
    patience = exp.get("patience") or PATIENCE
    lr = exp.get("lr") or LEARNING_RATE
    weight_decay = exp.get("weight_decay") or WEIGHT_DECAY
    val_size = exp.get("val_size") or VALIDATION_SIZE

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
    X_test_scaled = standardizer.transform(X_test)

    train_loader = DataLoader(
        tensor_dataset_from_numpy(X_subtrain_scaled, y_subtrain),
        batch_size=batch_size,
        shuffle=True,
    )

    model = build_model(exp, X_subtrain_scaled.shape).to(device)

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
    y_score = predict_scores(model, X_test_scaled, device)

    return {
        "y_score": y_score,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "n_parameters": count_trainable_params(model),
    }


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
        "best_epoch",
        "best_val_loss",
        "n_parameters",
    ]

    summary = {}
    for col in metric_cols:
        summary[f"{col}_mean"] = fold_df[col].mean()
        summary[f"{col}_std"] = fold_df[col].std()

    return summary


def evaluate_model_cv(exp, X, y, device, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    rows = []
    pooled_y_true = []
    pooled_y_score = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        fold_output = train_one_fold(
            exp=exp,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            device=device,
        )

        y_score = fold_output["y_score"]
        fold_metrics = calculate_metrics(y_test, y_score, threshold=0.5)
        fold_metrics["fold"] = fold_idx
        fold_metrics["best_epoch"] = fold_output["best_epoch"]
        fold_metrics["best_val_loss"] = fold_output["best_val_loss"]
        fold_metrics["n_parameters"] = fold_output["n_parameters"]
        rows.append(fold_metrics)

        pooled_y_true.extend(y_test.tolist())
        pooled_y_score.extend(y_score.tolist())

    fold_df = pd.DataFrame(rows)

    pooled_y_true = np.asarray(pooled_y_true)
    pooled_y_score = np.asarray(pooled_y_score)

    pooled_metrics = calculate_metrics(pooled_y_true, pooled_y_score, threshold=0.5)

    return fold_df, pooled_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 1 deep learning exploration (MLP, Siamese MLP, genome-sorted CNN)."
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
        help="Optional subdirectory under results/task1 for outputs (helps concurrent runs).",
    )
    parser.add_argument("--n_splits", type=int, default=N_SPLITS)
    parser.add_argument("--n_repeats", type=int, default=N_REPEATS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--val_size", type=float, default=VALIDATION_SIZE)
    return parser.parse_args()


def main():
    global BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY, VALIDATION_SIZE

    args = parse_args()
    BATCH_SIZE = args.batch_size
    MAX_EPOCHS = args.max_epochs
    PATIENCE = args.patience
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = args.weight_decay
    VALIDATION_SIZE = args.val_size

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

    X, y, cpg_ids = load_task1(H5_PATH)
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
    print("Task 1: Deep learning exploration")
    print("Control -> Control vs Control -> MCI")
    print("=" * 80)
    print(f"Preset: {args.preset}")
    print(f"HDF5 file: {H5_PATH.resolve()}")
    print(f"Annotation file: {ANNOTATION_PATH.resolve()}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Device: {device}")
    print(f"CV setting: {args.n_splits} folds x {args.n_repeats} repeats")
    print(f"Training: batch={BATCH_SIZE}, epochs={MAX_EPOCHS}, patience={PATIENCE}, lr={LEARNING_RATE}, wd={WEIGHT_DECAY}, val={VALIDATION_SIZE}")
    print(f"X shape after transpose: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"class counts [0=CN->CN, 1=CN->MCI]: {np.bincount(y)}")
    print(f"number of CpG IDs: {len(cpg_ids)}")
    print(f"genome-mapped CpGs: {genome_metadata['MAPINFO'].notna().sum()} / {len(cpg_ids)}")

    genome_order_path = out_dir / "task1_genome_ordered_cpgs.csv"
    genome_metadata.to_csv(genome_order_path, index=False)
    experiment_specs_path = out_dir / "task1_deep_learning_experiment_specs.csv"
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
        "best_epoch_mean",
        "best_epoch_std",
        "best_val_loss_mean",
        "best_val_loss_std",
        "n_parameters_mean",
        "n_parameters_std",
        "pooled_roc_auc",
        "pooled_pr_auc",
        "pooled_accuracy",
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

    out_summary = out_dir / "task1_deep_learning_summary.csv"
    out_folds = out_dir / "task1_deep_learning_fold_results.csv"

    summary_df.to_csv(out_summary, index=False)
    fold_results_df.to_csv(out_folds, index=False)

    print("\n" + "=" * 80)
    print("Final deep learning summary, sorted by mean ROC-AUC")
    print("=" * 80)

    display_cols = [
        "experiment",
        "model_type",
        "n_features",
        "effective_n_features",
        "roc_auc_mean",
        "pr_auc_mean",
        "balanced_accuracy_mean",
        "recall_sensitivity_mean",
        "specificity_mean",
        "f1_mean",
        "best_epoch_mean",
        "n_parameters_mean",
    ]

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary_df[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {out_summary.resolve()}")
    print(f"  {out_folds.resolve()}")
    print(f"  {genome_order_path.resolve()}")
    print(f"  {experiment_specs_path.resolve()}")


if __name__ == "__main__":
    main()
