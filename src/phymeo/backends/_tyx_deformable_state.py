"""Small state/clock kernels; material integration remains in the original kernels."""

import tyx as tx

from ._tyx_numeric import _add


@tx.kernel
def begin_candidate(
    count: tx.i32,
    n: tx.i32,
    p: tx.i32,
    predicted: tx.ndarray(tx.f32, 2),
    x: tx.ndarray(tx.f32, 2),
    multipliers: tx.ndarray(tx.f32, 1),
    status: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        if i < n:
            for a in tx.static_range(3):
                x[i, a] = predicted[i, a]
        if i < p:
            multipliers[i] = 0.0
        if i == 0:
            status[0] = 0


@tx.kernel
def copy_state(
    n: tx.i32,
    x: tx.ndarray(tx.f32, 2),
    v: tx.ndarray(tx.f32, 2),
    clock: tx.ndarray(tx.f32, 1),
    tick: tx.ndarray(tx.i32, 1),
    next_x: tx.ndarray(tx.f32, 2),
    next_v: tx.ndarray(tx.f32, 2),
    next_clock: tx.ndarray(tx.f32, 1),
    next_tick: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(n):
        for a in tx.static_range(3):
            next_x[i, a] = x[i, a]
            next_v[i, a] = v[i, a]
        if i == 0:
            next_clock[0] = clock[0]
            next_clock[1] = clock[1]
            next_tick[0] = tick[0]


@tx.kernel
def advance_clock(
    h: tx.f32,
    clock: tx.ndarray(tx.f32, 1),
    tick: tx.ndarray(tx.i32, 1),
    next_clock: tx.ndarray(tx.f32, 1),
    next_tick: tx.ndarray(tx.i32, 1),
    status: tx.ndarray(tx.i32, 1),
):
    for lane in tx.range(1):
        y = _add(h, -clock[1])
        total = _add(clock[0], y)
        error = _add(_add(total, -clock[0]), -y)
        next_clock[0] = total
        next_clock[1] = error
        if tick[0] >= 2147483647:
            status[0] = 1
        else:
            next_tick[0] = tick[0] + 1
