from pathlib import Path

import numpy as np
import pandas as pd


def make_flat_feature_sets(X):
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


def make_sequence_feature_sets(X, ordered_indices):
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

    metadata = ordered[["cpg_id", "original_index", "CHR", "MAPINFO", "chr_order"]].copy()
    metadata["genome_rank"] = np.arange(len(metadata))
    return ordered_indices, metadata
