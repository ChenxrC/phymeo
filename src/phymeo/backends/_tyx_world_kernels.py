"""FP32 world-parallel integration, observation and staged local restore."""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


def advance_worlds(
    count: tx.i32,
    h: tx.f32,
    kind: tx.ndarray(tx.i32, 1),
    props: tx.ndarray(tx.f32, 2),
    commands: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    candidate: tx.ndarray(tx.f32, 2),
    next_tick: tx.ndarray(tx.i32, 1),
    accepted: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        for k in tx.static_range(12):
            candidate[i, k] = state[i, k]
        good = 1
        if kind[i] == 0:
            for j in tx.static_range(3):
                external = commands[i, j]
                acceleration = _add(props[i, 1 + j], _mul(props[i, 0], external))
                speed = _add(state[i, 3 + j], _mul(h, acceleration))
                candidate[i, j] = _add(state[i, j], _mul(h, speed))
                candidate[i, 3 + j] = speed
                candidate[i, 6 + j] = external
        elif kind[i] == 1 or kind[i] == 2:
            target = commands[i, 0]
            target_speed = commands[i, 1]
            feedforward = commands[i, 2]
            if kind[i] == 2:
                target = state[i, 9]
                target_speed = state[i, 10]
                feedforward = state[i, 11]
                for j in tx.static_range(3):
                    candidate[i, 9 + j] = commands[i, j]
            # End-of-step bounded PD on a single prismatic mass, with undelayed external load.
            inv_mass = props[i, 0]
            free_acc = _add(props[i, 1], _mul(inv_mass, commands[i, 3]))
            free_speed = _add(state[i, 3], _mul(h, free_acc))
            d = _add(_mul(h, props[i, 4]), props[i, 5])
            a = _add(
                _add(
                    _mul(props[i, 4], _add(target, -state[i, 0])), _mul(props[i, 5], target_speed)
                ),
                feedforward,
            )
            numerator = _add(a, -_mul(d, free_speed))
            denominator = _add(1.0, _mul(_mul(d, h), inv_mass))
            unclipped = _div(numerator, denominator)
            if (not tx.isfinite(unclipped)) or (not tx.isfinite(denominator)):
                good = 0
            effort = tx.min(props[i, 6], tx.max(-props[i, 6], unclipped))
            speed = _add(free_speed, _mul(h, _mul(inv_mass, effort)))
            candidate[i, 0] = _add(state[i, 0], _mul(h, speed))
            candidate[i, 3] = speed
            candidate[i, 6] = effort
        else:
            # A later rotor pass owns kinds 3/4. Unknown kinds also reject.
            good = 0
        for k in tx.static_range(12):
            if not tx.isfinite(candidate[i, k]):
                good = 0
        if tick[i] >= 2147483647:
            good = 0
        if good == 1:
            next_tick[i] = tick[i] + 1
        else:
            next_tick[i] = tick[i]
            for k in tx.static_range(12):
                candidate[i, k] = state[i, k]
        accepted[i] = good


# Each callable retains a TYX argument binding for one swap-buffer direction.
# Both lower the same physical kernel and follow normal dispatch and recording.
advance_worlds_other_buffer = tx.kernel(advance_worlds)
advance_worlds = tx.kernel(advance_worlds)


@tx.kernel
def observe_worlds(
    count: tx.i32,
    kind: tx.ndarray(tx.i32, 1),
    props: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 2),
    summary: tx.ndarray(tx.f32, 2),
):
    for i in tx.range(count):
        vv = _mul(state[i, 3], state[i, 3])
        if kind[i] == 0:
            vv = _add(vv, _add(_mul(state[i, 4], state[i, 4]), _mul(state[i, 5], state[i, 5])))
            summary[i, 1] = _add(state[i, 2], -props[i, 7])
        else:
            summary[i, 1] = state[i, 0]
        summary[i, 0] = _mul(_mul(0.5, props[i, 8]), vv)


@tx.kernel
def read_world(
    index: tx.i32,
    state: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    row: tx.ndarray(tx.f32, 1),
    result_tick: tx.ndarray(tx.i32, 1),
):
    for item in tx.range(1):
        for k in tx.static_range(12):
            row[k] = state[index, k]
        result_tick[item] = tick[index]


@tx.kernel
def restore_worlds(
    count: tx.i32,
    staged: tx.ndarray(tx.f32, 2),
    restored_tick: tx.ndarray(tx.i32, 1),
    state: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    candidate: tx.ndarray(tx.f32, 2),
    next_tick: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        if restored_tick[i] >= 0:
            for k in tx.static_range(12):
                candidate[i, k] = staged[i, k]
            next_tick[i] = restored_tick[i]
        else:
            for k in tx.static_range(12):
                candidate[i, k] = state[i, k]
            next_tick[i] = tick[i]
