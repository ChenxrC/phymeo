"""Selected accepted-state copies; physical integration remains in its original kernels."""

import tyx as tx


def gather_selected_worlds(
    count: tx.i32,
    width: tx.i32,
    slots: tx.ndarray(tx.i32, 1),
    state: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    selected: tx.ndarray(tx.f32, 2),
    selected_tick: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        source = slots[i]
        for j in tx.static_range(6):
            selected[i, j] = state[source, j]
        if width == 12:
            for j in tx.static_range(6):
                selected[i, 6 + j] = state[source, 6 + j]
        selected_tick[i] = tick[source]


def make_gather_kernels():
    """Each plan/direction owns its binding handle; compiled artifacts may be reused."""
    return tx.kernel(gather_selected_worlds), tx.kernel(gather_selected_worlds)
