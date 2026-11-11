"""Logical-world status reduction before the existing floating finalize.

One lane owns all rows of a logical world. No dynamics, state copy, bank choice
or publication occurs here; the old finalize retains every rejected row.
"""

from types import SimpleNamespace

import tyx as tx


def reduce_participant_worlds(
    world_count: tx.i32,
    world_offsets: tx.ndarray(tx.i32, 1),
    world_rows: tx.ndarray(tx.i32, 1),
    info: tx.ndarray(tx.i32, 2),
    flags: tx.ndarray(tx.i32, 1),
):
    for world in tx.range(world_count):
        rejected = 0
        begin = world_offsets[world]
        end = world_offsets[world + 1]
        for item in range(begin, end):
            row = world_rows[item]
            if info[row, 2] != 0 or flags[row] != 1:
                rejected = 1
        if rejected != 0:
            for item in range(begin, end):
                row = world_rows[item]
                if info[row, 2] == 0:
                    info[row, 2] = 6


def make_participant_transaction_kernels():
    """Allocate only the launch handle; the parent service owns device arrays."""
    return SimpleNamespace(reduce=tx.kernel(reduce_participant_worlds))
