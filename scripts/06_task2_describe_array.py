#!/usr/bin/env python3

import h5py
import numpy as np

from _shared.data_io import load_binary_task
from _shared.eda import (
    TOP_N_CPGS,
    describe_dataset,
    get_top_variable_cpgs,
    save_cpg_correlation_clustermap,
    save_subject_clustermap,
)
from _shared.task_config import TASK2_SPEC


H5_PATH = TASK2_SPEC.h5_path
OUT_DIR = TASK2_SPEC.out_dir
FIG_DIR = OUT_DIR / "figures"


def run_task2_eda_heatmaps():
    print("\n" + "=" * 80)
    print("Task 2 EDA heatmaps")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, y, cpg_ids = load_binary_task(
        H5_PATH,
        TASK2_SPEC.class0_key,
        TASK2_SPEC.class1_key,
        TASK2_SPEC.cpg_key,
    )

    print(f"Task 2 X shape: {X.shape}")
    print("  axis 0 = subjects")
    print("  axis 1 = presumed timepoints")
    print("  axis 2 = CpGs")
    print(f"Task 2 y counts [{TASK2_SPEC.class0_label}, {TASK2_SPEC.class1_label}]: {np.bincount(y)}")

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

        top_csv = OUT_DIR / f"task2_top{TOP_N_CPGS}_variable_cpgs_{timepoint_name}.csv"
        top_df.to_csv(top_csv, index=False)

        X_top = X_tp[:, top_idx]
        cpg_top = cpg_ids[top_idx]

        print(f"Selected top {TOP_N_CPGS} CpGs by variance.")
        print(f"Top CpG table saved: {top_csv.resolve()}")
        print(f"Highest variance CpG: {top_df.iloc[0]['cpg_id']} "
              f"(variance={top_df.iloc[0]['variance']:.6g})")

        corr_png = FIG_DIR / (
            f"task2_{timepoint_name}_top{TOP_N_CPGS}_"
            f"cpg_correlation_clustermap.png"
        )

        save_cpg_correlation_clustermap(
            X_2d=X_top,
            cpg_ids=cpg_top,
            out_png=corr_png,
            title=(
                f"Task 2 {timepoint_name}: clustered CpG-CpG correlation "
                f"heatmap, top {TOP_N_CPGS} variable CpGs"
            ),
        )

        print(f"CpG correlation clustermap saved: {corr_png.resolve()}")

        subject_png = FIG_DIR / (
            f"task2_{timepoint_name}_top{TOP_N_CPGS}_"
            f"subject_cpg_clustermap.png"
        )

        save_subject_clustermap(
            X_2d=X_top,
            y=y,
            cpg_ids=cpg_top,
            out_png=subject_png,
            title=(
                f"Task 2 {timepoint_name}: clustered subject × CpG heatmap, "
                f"top {TOP_N_CPGS} variable CpGs"
            ),
            class0_label=TASK2_SPEC.class0_label,
            class1_label=TASK2_SPEC.class1_label,
        )

        print(f"Subject × CpG clustermap saved: {subject_png.resolve()}")


def main():
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Could not find file: {H5_PATH.resolve()}")

    print(f"Reading: {H5_PATH.resolve()}")

    with h5py.File(H5_PATH, "r") as f:
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

    run_task2_eda_heatmaps()


if __name__ == "__main__":
    main()
