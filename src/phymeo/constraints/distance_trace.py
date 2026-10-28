"""Native projection observations, not a solver.

Direction columns are zero sentinels for a zero-denominator skipped projection;
otherwise they contain the actual applied direction (including coincident X).
"""

import numpy as np

COLUMNS = tuple(
    f"{point}_{axis}"
    for point in ("before_i", "before_j", "after_i", "after_j")
    for axis in ("x", "y", "z")
) + ("lambda_before", "lambda_after", "direction_x", "direction_y", "direction_z", "delta_lambda")


def trace_flag(value):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("record_material_path must be bool")
    return bool(value)


def validate_trace(trace, count, *inputs):
    if not isinstance(trace, np.ndarray) or trace.dtype != np.float64:
        raise TypeError("trace must be a float64 ndarray")
    if trace.shape != (count, 18) or not trace.flags.writeable or not trace.flags.c_contiguous:
        raise ValueError("trace must be writable contiguous (edge_count,18) storage")
    if any(np.shares_memory(trace, value) for value in inputs):
        raise ValueError("trace must not alias physical inputs")


def path_diagnostics(frames, predicted, alpha, h):
    from ..runtime.program import own_diagnostics

    return own_diagnostics(
        dict(
            schema="xpbd-distance-path-v1",
            columns=COLUMNS,
            frames=tuple(frames),
            predicted_position=predicted.position,
            predicted_velocity=predicted.velocity,
            alpha=alpha,
            h=h,
        )
    )
