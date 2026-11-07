"""Temporary contact projection for the existing ordinary resident tree solve."""

import tyx as tx

from ._tyx_numeric import _add, _mul


def contact_velocity(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    parents: tx.ndarray(tx.i32, 2),
    kinds: tx.ndarray(tx.i32, 2),
    state: tx.ndarray(tx.f32, 3),
    fk: tx.ndarray(tx.f32, 3),
    velocity: tx.ndarray(tx.f32, 3),
):
    # Same COM Jacobian as tree_mechanics, at the supplied actual bank. Does not
    # overwrite the entry M/b/J or solve diagnostics when observing the candidate.
    for w in tx.range(count):
        n = meta[w, 1]
        for body in range(n):
            for a in tx.static_range(6):
                velocity[w, body, a] = 0.0
            ancestor = body
            for depth in range(n):
                if ancestor >= 0:
                    for a in tx.static_range(3):
                        value = fk[w, ancestor, 12 + a]
                        if kinds[w, ancestor] == 0:
                            j = (a + 1) % 3
                            k = (a + 2) % 3
                            rj = _add(fk[w, body, 15 + j], -fk[w, ancestor, j])
                            rk = _add(fk[w, body, 15 + k], -fk[w, ancestor, k])
                            value = _add(
                                _mul(fk[w, ancestor, 12 + j], rk),
                                -_mul(fk[w, ancestor, 12 + k], rj),
                            )
                            velocity[w, body, 3 + a] = _add(
                                velocity[w, body, 3 + a],
                                _mul(fk[w, ancestor, 12 + a], state[w, ancestor, 1]),
                            )
                        velocity[w, body, a] = _add(
                            velocity[w, body, a], _mul(value, state[w, ancestor, 1])
                        )
                    ancestor = parents[w, ancestor]


def contact_project(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    linear: tx.ndarray(tx.f32, 4),
    angular: tx.ndarray(tx.f32, 4),
    wrench: tx.ndarray(tx.f32, 3),
    vector: tx.ndarray(tx.f32, 3),
    projected: tx.ndarray(tx.f32, 2),
):
    for w in tx.range(count):
        n = meta[w, 1]
        for column in range(n):
            value = 0.0
            for body in range(n):
                for a in tx.static_range(3):
                    value = _add(
                        value,
                        _add(
                            _mul(linear[w, body, a, column], wrench[w, body, a]),
                            _mul(angular[w, body, a, column], wrench[w, body, 3 + a]),
                        ),
                    )
            projected[w, column] = value
            vector[w, column, 1] = _add(vector[w, column, 1], value)


def contact_reject(
    count: tx.i32,
    code: tx.i32,
    reject: tx.ndarray(tx.i32, 1),
    info: tx.ndarray(tx.i32, 2),
):
    for w in tx.range(count):
        if reject[w] != 0:
            info[w, 2] = code


def make_contact_kernels():
    return {
        name: (tx.kernel(function), tx.kernel(function))
        for name, function in (
            ("velocity", contact_velocity),
            ("project", contact_project),
            ("reject", contact_reject),
        )
    }
