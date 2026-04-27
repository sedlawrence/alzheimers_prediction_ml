from pathlib import Path

import h5py
import numpy as np


def decode_if_bytes(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_binary_task(path: Path, class0_key: str, class1_key: str, cpg_key: str):
    """
    Load a binary methylation task from the shared HDF5 file.

    Original shape in file:
        features x time x samples

    Returned shape:
        samples x time x features
    """
    with h5py.File(path, "r") as f:
        x_class0 = f[class0_key][:]
        x_class1 = f[class1_key][:]
        cpg_ids = f[cpg_key][:]

    x0 = np.transpose(x_class0, (2, 1, 0))
    x1 = np.transpose(x_class1, (2, 1, 0))

    X = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([
        np.zeros(x0.shape[0], dtype=int),
        np.ones(x1.shape[0], dtype=int),
    ])

    cpg_ids = np.array([decode_if_bytes(c) for c in cpg_ids])
    return X.astype(np.float32), y.astype(int), cpg_ids
