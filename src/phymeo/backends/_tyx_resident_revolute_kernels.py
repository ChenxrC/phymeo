"""Second resident pass: real fixed-axis rigid inertia and bounded delayed PD.

Only kind 3/4 rows are touched. Inputs are the original accepted buffer; outputs
are the batch's shared candidate/tick/status buffers. The batch alone publishes.
"""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


def advance_revolutes(
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
        if kind[i] == 3 or kind[i] == 4:
            for k in tx.static_range(12):
                candidate[i, k] = state[i, k]
            good = 1
            q = state[i, 0]
            if not tx.isfinite(q) or tx.abs(q) > 4096.0:
                good = 0
            cosine = tx.cos(q)
            sine = tx.sin(q)
            ax = props[i, 13]
            ay = props[i, 14]
            az = props[i, 15]
            cx = props[i, 10]
            cy = props[i, 11]
            cz = props[i, 12]
            dot = _add(_add(_mul(ax, cx), _mul(ay, cy)), _mul(az, cz))
            parallel = _mul(_add(1.0, -cosine), dot)
            rx = _add(_add(_mul(cosine, cx), _mul(sine, _add(_mul(ay, cz), -_mul(az, cy)))), _mul(parallel, ax))
            ry = _add(_add(_mul(cosine, cy), _mul(sine, _add(_mul(az, cx), -_mul(ax, cz)))), _mul(parallel, ay))
            rz = _add(_add(_mul(cosine, cz), _mul(sine, _add(_mul(ax, cy), -_mul(ay, cx)))), _mul(parallel, az))
            # World COM force includes gravity exactly once. The explicit world
            # torque is about that COM; r x F brings it to the joint pivot.
            fx = _add(_mul(props[i, 9], props[i, 1]), commands[i, 4])
            fy = _add(_mul(props[i, 9], props[i, 2]), commands[i, 5])
            fz = _add(_mul(props[i, 9], props[i, 3]), commands[i, 6])
            mx = _add(_add(_mul(ry, fz), -_mul(rz, fy)), commands[i, 7])
            my = _add(_add(_mul(rz, fx), -_mul(rx, fz)), commands[i, 8])
            mz = _add(_add(_mul(rx, fy), -_mul(ry, fx)), commands[i, 9])
            external = _add(commands[i, 3], _add(_add(_mul(ax, mx), _mul(ay, my)), _mul(az, mz)))
            inv_inertia = props[i, 0]
            free_speed = _add(state[i, 3], _mul(h, _mul(inv_inertia, external)))
            target = commands[i, 0]
            target_speed = commands[i, 1]
            feedforward = commands[i, 2]
            if kind[i] == 4:
                target = state[i, 9]
                target_speed = state[i, 10]
                feedforward = state[i, 11]
                for j in tx.static_range(3):
                    candidate[i, 9 + j] = commands[i, j]
            d = _add(_mul(h, props[i, 4]), props[i, 5])
            a_pd = _add(_add(_mul(props[i, 4], _add(target, -q)), _mul(props[i, 5], target_speed)), feedforward)
            numerator = _add(a_pd, -_mul(d, free_speed))
            denominator = _add(1.0, _mul(_mul(d, h), inv_inertia))
            unclipped = _div(numerator, denominator)
            if (
                not tx.isfinite(cosine)
                or not tx.isfinite(sine)
                or not tx.isfinite(external)
                or not tx.isfinite(free_speed)
                or not tx.isfinite(denominator)
                or denominator <= 0.0
                or not tx.isfinite(unclipped)
            ):
                good = 0
            effort = tx.min(props[i, 6], tx.max(-props[i, 6], unclipped))
            speed = _add(free_speed, _mul(h, _mul(inv_inertia, effort)))
            candidate[i, 0] = _add(q, _mul(h, speed))
            candidate[i, 3] = speed
            candidate[i, 6] = effort
            for k in tx.static_range(12):
                if not tx.isfinite(candidate[i, k]):
                    good = 0
            if tx.abs(candidate[i, 0]) > 4096.0 or tick[i] >= 2147483647:
                good = 0
            if good == 1:
                next_tick[i] = tick[i] + 1
            else:
                next_tick[i] = tick[i]
                for k in tx.static_range(12):
                    candidate[i, k] = state[i, k]
            accepted[i] = good


def make_revolute_step_kernels():
    """Independent normal-math handles for both resident state-buffer directions."""
    return tx.kernel(advance_revolutes), tx.kernel(advance_revolutes)
