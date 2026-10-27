"""Contact geometry and scalar/vector updates in explicit FP32 arithmetic."""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


@tx.kernel
def contact_dot(x: tx.ndarray(tx.f32, 1), y: tx.ndarray(tx.f32, 1), result: tx.ndarray(tx.f32, 1)):
    for lane in tx.range(1):
        value = 0.0
        for a in tx.static_range(3):
            ax = tx.abs(x[a])
            ay = tx.abs(y[a])
            negligible = 0
            if ax > 0.0 and ay > 0.0 and ax < 1.0 and ay < 1.0:
                # Here tiny/ay is normal, so the check itself cannot underflow.
                if ax < 1.1754943508222875e-38 / ay:
                    negligible = 1
            if negligible == 0:
                value = _add(value, _mul(x[a], y[a]))
        result[0] = value


@tx.kernel
def node_sphere_geometry(
    radius: tx.f32,
    point: tx.ndarray(tx.f32, 1),
    center: tx.ndarray(tx.f32, 1),
    normal: tx.ndarray(tx.f32, 1),
    gap: tx.ndarray(tx.f32, 1),
    failed: tx.ndarray(tx.i32, 1),
):
    for lane in tx.range(1):
        scale = 0.0
        for a in tx.static_range(3):
            normal[a] = _add(point[a], -center[a])
            scale = tx.max(scale, tx.abs(normal[a]))
        if scale == 0.0:
            failed[0] = 1
        else:
            square = 0.0
            for a in tx.static_range(3):
                normal[a] = _div(normal[a], scale)
                # At least one scaled component is exactly +/-1. Terms below
                # f32.tiny cannot affect this norm sum. Keep the component for
                # the final normal, but do not square it into a subnormal.
                if tx.abs(normal[a]) >= 1.0842021724855044e-19:
                    square = _add(square, _mul(normal[a], normal[a]))
            unit = tx.sqrt(square)
            for a in tx.static_range(3):
                normal[a] = _div(normal[a], unit)
            gap[0] = _add(_mul(scale, unit), -radius)


@tx.kernel
def contact_multiplier(old: tx.f32, error: tx.f32, response: tx.f32, result: tx.ndarray(tx.f32, 1)):
    for lane in tx.range(1):
        updated = tx.max(0.0, _add(old, -_div(error, response)))
        result[0] = updated
        result[1] = _add(updated, -old)


@tx.kernel
def paired_contact_load(
    magnitude: tx.f32, normal: tx.ndarray(tx.f32, 1), pair: tx.ndarray(tx.f32, 2)
):
    for a in tx.range(3):
        value = _mul(magnitude, normal[a])
        pair[0, a] = value
        pair[1, a] = -value
