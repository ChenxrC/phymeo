"""Actual-device holdout kernel for the C9 FP32 arithmetic policy."""

import tyx as tx

from phymeo.backends._tyx_floating_numeric import _add, _div, _mul


def evaluate(values: tx.ndarray(tx.f32, 2), result: tx.ndarray(tx.f32, 2)):
    for i in tx.range(1):
        a = values[i, 0]
        b = values[i, 1]
        result[i, 0] = _mul(a, b)
        result[i, 1] = _div(a, b)
        result[i, 2] = _add(a, b)
