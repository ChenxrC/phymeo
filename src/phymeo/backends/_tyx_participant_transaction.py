"""Status-only world transactions for the original mechanical finalizers.

Every mechanical row owns one global slot. Collects from different storages
never overlap. One reduction lane owns a whole logical world; scatter preserves
the original failed member's code. Existing finalizers perform state retention.
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


def collect_participant_failures(
    count: tx.i32,
    row_slots: tx.ndarray(tx.i32, 1),
    info: tx.ndarray(tx.i32, 2),
    flags: tx.ndarray(tx.i32, 1),
    failed: tx.ndarray(tx.i32, 1),
):
    for row in tx.range(count):
        rejected = 0
        if info[row, 2] != 0 or flags[row] != 1:
            rejected = 1
        failed[row_slots[row]] = rejected


def reduce_participant_slots(
    world_count: tx.i32,
    world_offsets: tx.ndarray(tx.i32, 1),
    world_slots: tx.ndarray(tx.i32, 1),
    failed: tx.ndarray(tx.i32, 1),
):
    for world in tx.range(world_count):
        rejected = 0
        begin = world_offsets[world]
        end = world_offsets[world + 1]
        for item in range(begin, end):
            if failed[world_slots[item]] != 0:
                rejected = 1
        if rejected != 0:
            for item in range(begin, end):
                failed[world_slots[item]] = 1


def scatter_participant_failures(
    count: tx.i32,
    row_slots: tx.ndarray(tx.i32, 1),
    failed: tx.ndarray(tx.i32, 1),
    info: tx.ndarray(tx.i32, 2),
):
    for row in tx.range(count):
        if failed[row_slots[row]] != 0 and info[row, 2] == 0:
            info[row, 2] = 6


def make_participant_transaction_kernels():
    """Allocate handles only; the shared service owns maps, arrays and launches."""
    return SimpleNamespace(
        reduce=tx.kernel(reduce_participant_worlds),
        collect=tx.kernel(collect_participant_failures),
        reduce_slots=tx.kernel(reduce_participant_slots),
        scatter=tx.kernel(scatter_participant_failures),
    )
