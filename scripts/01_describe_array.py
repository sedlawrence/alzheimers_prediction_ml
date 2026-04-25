# simple script to describe the contents of the HDF5 file containing DNA methylation data

import h5py
import numpy as np
from pathlib import Path


path = Path("../data/temporal_two_sets_n2000.h5")


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

    Your arrays look like:
        (2000, 2, 147)
        (2000, 2, 43)
        etc.

    This assumes axis 0 = CpGs/features,
    axis 1 = channels/timepoints/groups,
    axis 2 = samples/individuals.
    """
    if ds.ndim != 3:
        return

    x = ds[()]

    print("  3D axis summary:")
    print(f"    axis 0 length: {ds.shape[0]}  likely CpGs/features")
    print(f"    axis 1 length: {ds.shape[1]}  likely paired values / channels")
    print(f"    axis 2 length: {ds.shape[2]}  likely samples")

    # Per-channel summary, useful because your middle axis is length 2.
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

        # Difference between channel 1 and channel 0.
        delta = x[:, 1, :] - x[:, 0, :]
        print("  channel delta summary, channel 1 - channel 0:")
        print(f"    min:    {np.nanmin(delta):.6g}")
        print(f"    max:    {np.nanmax(delta):.6g}")
        print(f"    mean:   {np.nanmean(delta):.6g}")
        print(f"    std:    {np.nanstd(delta):.6g}")
        print(f"    median: {np.nanmedian(delta):.6g}")

    # Per-CpG summary across channels and samples.
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


if __name__ == "__main__":
    main()
