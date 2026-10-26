"""Numerical helpers shared by independent solver acceptance checks."""

import numpy as np


def normalized_error(error, scale):
    """Normalize nonnegative errors without accepting positive underflow as zero.

    A positive quotient below the FP64 range is conservatively represented by
    the smallest positive FP64 value. NaN and infinity remain visible to callers.
    """
    error = np.asarray(error, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        residual = error / np.asarray(scale, dtype=np.float64)
        return np.where(
            (error > 0) & (residual == 0), np.nextafter(np.float64(0), np.float64(1)), residual
        )
