"""Validation shared by the physical data boundary."""

import numpy as np


def float_array(value, name: str, *, ndim: int, columns: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError(f"{name} must use float32 or float64 storage")
    if array.ndim != ndim or (columns is not None and array.shape[-1] != columns):
        expected = f"(n, {columns})" if columns is not None else f"rank {ndim}"
        raise ValueError(f"{name} must have shape {expected}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, copy=True, order="C")
