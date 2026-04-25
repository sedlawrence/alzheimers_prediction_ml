#!/usr/bin/env python3

# simple script to describe the contents of the HDF5 file containing DNA methylation data
# plus Task 1 EDA heatmaps for top variable CpGs

import h5py
import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


path = Path("../data/temporal_two_sets_n2000.h5")

OUT_DIR = Path("../results/task1")
FIG_DIR = OUT_DIR / "figures"

TOP_N_CPGS = 200


def human_bytes(n_bytes):
    """Convert bytes to a readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} PB"


def decode_if_bytes(x):
    """Decode bytes/object arrays to strings where possible."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x


def preview_dataset(ds, n=5):
    """Print a small preview of any HDF5 dataset."""
    print("  preview:")

    if ds.ndim == 0:
        print("   ", decode_if_bytes(ds[()]))
        return

    if ds.ndim == 1:
        vals = ds[:min(n, ds.shape[0])]
        vals = [decode_if_bytes(v) for v in vals]
        print("   ", vals)
        return

    if ds.ndim == 2:
        rows = min(n, ds.shape[0])
        cols = min(n, ds.shape[1])
        print(ds[:rows, :cols])
        return

    if ds.ndim == 3:
        a = min(n, ds.shape[0])
        b = min(ds.shape[1], 2)
        c = min(n, ds.shape[2])
        print(f"    first {a} x {b} x {c} block:")
        print(ds[:a, :b, :c])
        return

    print(f"    dataset has {ds.ndim} dimensions; preview skipped")


def numeric_summary(ds, sample_limit=1_000_000):
    """
    Summarise numeric dataset.

    For very large arrays, samples up to sample_limit values to avoid
    loading huge arrays into memory.
    """
    n_values = ds.size

    if n_values == 0:
        print("  numeric summary: empty dataset")
        return

    if n_values <= sample_limit:
        x = ds[()]
        sampled = False
    else:
        sampled = True

        # For a 3D array like (2000, 2, 147), sample along first axis.
        if ds.ndim >= 1:
            n_first = max(1, sample_limit // int(np.prod(ds.shape[1:])))
            n_first = min(n_first, ds.shape[0])
            x = ds[:n_first]
        else:
            x = ds[()]

    x = np.asarray(x)

    print("  numeric summary:")
    if sampled:
        print(f"    sampled values: {x.size:,} of {n_values:,}")
    else:
        print(f"    values: {n_values:,}")

    print(f"    missing / NaN: {np.isnan(x).sum():,}")
    print(f"    min:    {np.nanmin(x):.6g}")
    print(f"    max:    {np.nanmax(x):.6g}")
    print(f"    mean:   {np.nanmean(x):.6g}")
    print(f"    std:    {np.nanstd(x):.6g}")
    print(f"    median: {np.nanmedian(x):.6g}")

    qs = np.nanpercentile(x, [1, 5, 25, 75, 95, 99])
    print("    percentiles:")
    print(f"      p01: {qs[0]:.6g}")
    print(f"      p05: {qs[1]:.6g}")
    print(f"      p25: {qs[2]:.6g}")
    print(f"      p75: {qs[3]:.6g}")
    print(f"      p95: {qs[4]:.6g}")
    print(f"      p99: {qs[5]:.6g}")


def summarise_3d_array(ds):
    """
    Extra summaries for arrays shaped like:
        n_features x n_channels x n_samples

    This assumes axis 0 = CpGs/features,
    axis 1 = channels/timepoints,
    axis 2 = samples/individuals.
    """
    if ds.ndim != 3:
        return

    x = ds[()]

    print("  3D axis summary:")
    print(f"    axis 0 length: {ds.shape[0]}  likely CpGs/features")
    print(f"    axis 1 length: {ds.shape[1]}  likely paired values / timepoints")
    print(f"    axis 2 length: {ds.shape[2]}  likely samples")

    if ds.shape[1] == 2:
        print("  per-channel summary:")
        for channel in range(ds.shape[1]):
            xc = x[:, channel, :]
            print(f"    channel {channel}:")
            print(f"      min:    {np.nanmin(xc):.6g}")
            print(f"      max:    {np.nanmax(xc):.6g}")
            print(f"      mean:   {np.nanmean(xc):.6g}")
            print(f"      std:    {np.nanstd(xc):.6g}")
            print(f"      median: {np.nanmedian(xc):.6g}")

        delta = x[:, 1, :] - x[:, 0, :]
        print("  channel delta summary, channel 1 - channel 0:")
        print(f"    min:    {np.nanmin(delta):.6g}")
        print(f"    max:    {np.nanmax(delta):.6g}")
        print(f"    mean:   {np.nanmean(delta):.6g}")
        print(f"    std:    {np.nanstd(delta):.6g}")
        print(f"    median: {np.nanmedian(delta):.6g}")

    per_feature_mean = np.nanmean(x, axis=(1, 2))
    per_feature_std = np.nanstd(x, axis=(1, 2))

    print("  per-feature summary:")
    print(f"    feature means: min={np.nanmin(per_feature_mean):.6g}, "
          f"max={np.nanmax(per_feature_mean):.6g}, "
          f"mean={np.nanmean(per_feature_mean):.6g}")
    print(f"    feature stds:  min={np.nanmin(per_feature_std):.6g}, "
          f"max={np.nanmax(per_feature_std):.6g}, "
          f"mean={np.nanmean(per_feature_std):.6g}")

    top_variable = np.argsort(per_feature_std)[-5:][::-1]
    print("    top 5 most variable feature indices:")
    for idx in top_variable:
        print(f"      feature {idx}: std={per_feature_std[idx]:.6g}, "
              f"mean={per_feature_mean[idx]:.6g}")


def string_summary(ds, n=10):
    """Summarise string/object-like datasets, e.g. CpG IDs."""
    vals = ds[:]
    vals = np.asarray([decode_if_bytes(v) for v in vals])

    print("  string/object summary:")
    print(f"    entries: {len(vals):,}")
    print(f"    unique:  {len(set(vals)):,}")
    print(f"    first {min(n, len(vals))}:")
    for v in vals[:n]:
        print(f"      {v}")


def describe_dataset(name, ds):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"  shape: {ds.shape}")
    print(f"  ndim:  {ds.ndim}")
    print(f"  dtype: {ds.dtype}")
    print(f"  size:  {ds.size:,} values")

    if ds.size > 0:
        approx_nbytes = ds.size * ds.dtype.itemsize
        print(f"  approx memory if loaded: {human_bytes(approx_nbytes)}")

    preview_dataset(ds)

    if np.issubdtype(ds.dtype, np.number):
        numeric_summary(ds)
        summarise_3d_array(ds)
    else:
        string_summary(ds)


def load_task1_for_eda(path):
    """
    Load Task 1:
        X_cn_to_cn  = y 0
        X_cn_to_mci = y 1

    Original shape:
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

    cpg_ids = np.asarray([decode_if_bytes(c) for c in cpg_ids])

    return X, y, cpg_ids


def get_top_variable_cpgs(X_2d, cpg_ids, top_n=200):
    """
    Select top CpGs by variance across subjects.

    X_2d shape:
        samples x CpGs
    """
    variances = np.nanvar(X_2d, axis=0)

    top_idx = np.argsort(variances)[-top_n:][::-1]

    top_df = pd.DataFrame({
        "rank": np.arange(1, len(top_idx) + 1),
        "feature_index": top_idx,
        "cpg_id": cpg_ids[top_idx],
        "variance": variances[top_idx],
        "std": np.sqrt(variances[top_idx]),
        "mean_beta": np.nanmean(X_2d[:, top_idx], axis=0),
    })

    return top_idx, top_df


def save_cpg_correlation_clustermap(X_2d, cpg_ids, out_png, title):
    """
    Save clustered CpG-CpG correlation heatmap.

    X_2d shape:
        samples x selected CpGs
    """
    df = pd.DataFrame(X_2d, columns=cpg_ids)

    corr = df.corr(method="pearson")

    g = sns.clustermap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(14, 14),
        xticklabels=False,
        yticklabels=False,
        dendrogram_ratio=(0.12, 0.12),
        cbar_pos=(0.02, 0.82, 0.03, 0.12),
    )

    g.fig.suptitle(title, y=1.02)
    g.fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def save_subject_clustermap(X_2d, y, cpg_ids, out_png, title):
    """
    Save clustered subject x CpG methylation heatmap.

    Rows = subjects
    Columns = selected CpGs
    Values = beta values

    Rows are coloured by class:
        0 = CN -> CN
        1 = CN -> MCI
    """
    df = pd.DataFrame(X_2d, columns=cpg_ids)

    # Add readable row labels.
    row_labels = [
        f"sample_{i:03d}_CN_to_CN" if label == 0 else f"sample_{i:03d}_CN_to_MCI"
        for i, label in enumerate(y)
    ]
    df.index = row_labels

    # Simple row colours. Seaborn accepts colour names.
    row_colors = pd.Series(y, index=df.index).map({
        0: "lightgrey",
        1: "firebrick",
    })

    g = sns.clustermap(
        df,
        cmap="viridis",
        row_cluster=True,
        col_cluster=True,
        row_colors=row_colors,
        figsize=(16, 12),
        xticklabels=False,
        yticklabels=False,
        dendrogram_ratio=(0.10, 0.10),
        cbar_pos=(0.02, 0.82, 0.03, 0.12),
    )

    g.fig.suptitle(title, y=1.02)

    # Add a small legend for row colours.
    for label, color in [
        ("CN → CN, y=0", "lightgrey"),
        ("CN → MCI, y=1", "firebrick"),
    ]:
        g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)

    g.ax_col_dendrogram.legend(
        loc="center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
    )

    g.fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def run_task1_eda_heatmaps():
    """
    Generate Task 1 clustered heatmaps for top variable CpGs.

    Outputs:
      - CSVs of top variable CpGs at t0 and t1
      - clustered CpG-CpG correlation heatmaps
      - clustered subject-CpG heatmaps
    """
    print("\n" + "=" * 80)
    print("Task 1 EDA heatmaps")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, y, cpg_ids = load_task1_for_eda(path)

    print(f"Task 1 X shape: {X.shape}")
    print("  axis 0 = subjects")
    print("  axis 1 = presumed timepoints")
    print("  axis 2 = CpGs")
    print(f"Task 1 y counts [CN->CN, CN->MCI]: {np.bincount(y)}")

    timepoint_data = {
        "t0": X[:, 0, :],
        "t1": X[:, 1, :],
    }

    for timepoint_name, X_tp in timepoint_data.items():
        print("\n" + "-" * 80)
        print(f"Processing {timepoint_name}")
        print(f"Input matrix shape: {X_tp.shape}")

        top_idx, top_df = get_top_variable_cpgs(
            X_2d=X_tp,
            cpg_ids=cpg_ids,
            top_n=TOP_N_CPGS,
        )

        top_csv = OUT_DIR / f"task1_top{TOP_N_CPGS}_variable_cpgs_{timepoint_name}.csv"
        top_df.to_csv(top_csv, index=False)

        X_top = X_tp[:, top_idx]
        cpg_top = cpg_ids[top_idx]

        print(f"Selected top {TOP_N_CPGS} CpGs by variance.")
        print(f"Top CpG table saved: {top_csv.resolve()}")
        print(f"Highest variance CpG: {top_df.iloc[0]['cpg_id']} "
              f"(variance={top_df.iloc[0]['variance']:.6g})")

        corr_png = FIG_DIR / (
            f"task1_{timepoint_name}_top{TOP_N_CPGS}_"
            f"cpg_correlation_clustermap.png"
        )

        save_cpg_correlation_clustermap(
            X_2d=X_top,
            cpg_ids=cpg_top,
            out_png=corr_png,
            title=(
                f"Task 1 {timepoint_name}: clustered CpG-CpG correlation "
                f"heatmap, top {TOP_N_CPGS} variable CpGs"
            ),
        )

        print(f"CpG correlation clustermap saved: {corr_png.resolve()}")

        subject_png = FIG_DIR / (
            f"task1_{timepoint_name}_top{TOP_N_CPGS}_"
            f"subject_cpg_clustermap.png"
        )

        save_subject_clustermap(
            X_2d=X_top,
            y=y,
            cpg_ids=cpg_top,
            out_png=subject_png,
            title=(
                f"Task 1 {timepoint_name}: clustered subject × CpG heatmap, "
                f"top {TOP_N_CPGS} variable CpGs"
            ),
        )

        print(f"Subject × CpG clustermap saved: {subject_png.resolve()}")


def main():
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path.resolve()}")

    print(f"Reading: {path.resolve()}")

    with h5py.File(path, "r") as f:
        print("\nHDF5 structure:")
        print("-" * 80)

        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name} {obj.shape} {obj.dtype}")
            else:
                print(f"{name} group")

        f.visititems(show)

        print("\nDetailed summaries:")

        def describe(name, obj):
            if isinstance(obj, h5py.Dataset):
                describe_dataset(name, obj)

        f.visititems(describe)

    run_task1_eda_heatmaps()


if __name__ == "__main__":
    main()