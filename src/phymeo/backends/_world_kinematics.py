"""Typed, detached kinematics from the currently accepted resident world states."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class WorldKinematics:
    """Read-only rows in requested world order.

    A sphere uses world XYZ in m and world linear velocity in m/s.
    A slider uses q in column 0 and qd in column 3, with zero padding.
    Widths are 3 or 1. Ticks count committed substeps independently for each key.
    """

    keys: tuple[int, ...]
    widths: tuple[int, ...]
    ticks: np.ndarray
    values: np.ndarray


def read_kinematics(batch, keys):
    """Read the committed state/tick buffers once and select fields and worlds on HOST."""
    with batch._lock:
        batch._check_open()
        selected = tuple(batch._slots) if keys is None else tuple(batch._keys(keys))
        if not selected:
            values = np.empty((0, 6), np.float32)
            ticks = np.empty(0, np.int32)
            widths = ()
        else:
            with batch._device():
                src = batch._active
                rows = batch._read(batch._states[src], "kinematics")
                ticks = batch._read(batch._ticks[src], "kinematics")
            if keys is None:
                # The native read already owns its HOST ticks. Detach just the six floats.
                values = rows[:, :6].copy(order="C")
                widths = tuple(binding.width for binding in batch._bindings)
            else:
                slots = [batch._slots[key] for key in selected]
                values = rows[slots, :6]
                ticks = ticks[slots]
                widths = tuple(batch._bindings[slot].width for slot in slots)
        values.setflags(write=False)
        ticks.setflags(write=False)
        return WorldKinematics(selected, widths, ticks, values)
