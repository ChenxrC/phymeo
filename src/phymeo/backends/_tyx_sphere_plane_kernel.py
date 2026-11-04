"""Checked frictionless normal response for a staged resident rigid sphere."""

import tyx as tx

from ._tyx_numeric import _add, _mul


def project_sphere_planes(
    count: tx.i32,
    kind: tx.ndarray(tx.i32, 1),
    props: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 2),
    tick: tx.ndarray(tx.i32, 1),
    candidate: tx.ndarray(tx.f32, 2),
    next_tick: tx.ndarray(tx.i32, 1),
    accepted: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        if accepted[i] == 1 and kind[i] == 0 and props[i, 6] == 1.0:
            candidate[i, 9] = 0.0
            if candidate[i, 2] <= props[i, 4]:
                candidate[i, 2] = props[i, 4]
                incoming = candidate[i, 5]
                if incoming < 0.0:
                    outgoing = 0.0
                    good = 1
                    if props[i, 5] > 0.0:
                        change = _add(incoming, -state[i, 5])
                        threshold = _mul(2.0, tx.abs(change))
                        if not tx.isfinite(threshold):
                            good = 0
                        if -incoming > threshold:
                            outgoing = _mul(-incoming, props[i, 5])
                    impulse = _mul(props[i, 8], _add(outgoing, -incoming))
                    if tx.isfinite(impulse) and tx.isfinite(outgoing) and good == 1:
                        candidate[i, 5] = outgoing
                        candidate[i, 9] = impulse
                    else:
                        for k in tx.static_range(12):
                            candidate[i, k] = state[i, k]
                        next_tick[i] = tick[i]
                        accepted[i] = 0


# The validated argument arrays alternate roles just as in the free predictor.
project_sphere_planes_other_buffer = tx.kernel(project_sphere_planes)
project_sphere_planes = tx.kernel(project_sphere_planes)
