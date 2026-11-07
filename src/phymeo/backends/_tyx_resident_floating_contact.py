"""Contact COM-wrench projection into the existing full floating-tree solve.

B=N+1 physical bodies and D=N+6 full velocity columns. Only temporary loads
are assembled here: no motor solve, root integration, bank choice or publish.
"""

import tyx as tx

from ._tyx_floating_numeric import _add, _mul


def floating_contact_project(
    count: tx.i32,
    meta: tx.ndarray(tx.i32, 2),
    linear: tx.ndarray(tx.f32, 4),
    angular: tx.ndarray(tx.f32, 4),
    wrench: tx.ndarray(tx.f32, 3),
    vector: tx.ndarray(tx.f32, 3),
    projected: tx.ndarray(tx.f32, 2),
):
    for row in tx.range(count):
        n = meta[row, 1]
        bodies = n + 1
        dofs = n + 6
        for column in range(dofs):
            value = 0.0
            for body in range(bodies):
                for axis in tx.static_range(3):
                    value = _add(
                        value,
                        _add(
                            _mul(linear[row, body, axis, column], wrench[row, body, axis]),
                            _mul(angular[row, body, axis, column], wrench[row, body, 3 + axis]),
                        ),
                    )
            # Overwrite every real column on each attempt. Preserve the input
            # external load and add the contact projection exactly once here.
            # Caller must not rebuild mechanics between this and the old solve.
            projected[row, column] = value
            vector[row, column, 1] = _add(vector[row, column, 1], value)


def make_floating_contact_kernels():
    """Lazy per-plan pair of launch handles; the parent owns all resources."""
    return {
        "project": (tx.kernel(floating_contact_project), tx.kernel(floating_contact_project))
    }
