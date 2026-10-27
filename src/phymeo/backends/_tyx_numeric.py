"""Checked public tyx FP32 arithmetic for free-motion kernels."""

import tyx as tx

# Unsupported nonzero subnormal terms become NaN; the host rejects the candidate.
# A lost acceleration must not silently vanish before multiplication by a large dt.
# The public pure-function API requires a single return expression.


@tx.func
def _mul(a: tx.f32, b: tx.f32) -> tx.f32:
    return (
        tx.sqrt(-1.0) if a != 0.0 and b != 0.0 and tx.abs(a * b) < 1.1754943508222875e-38 else a * b
    )


@tx.func
def _div(a: tx.f32, b: tx.f32) -> tx.f32:
    return tx.sqrt(-1.0) if a != 0.0 and tx.abs(a / b) < 1.1754943508222875e-38 else a / b


@tx.func
def _add(a: tx.f32, b: tx.f32) -> tx.f32:
    return tx.sqrt(-1.0) if a + b != 0.0 and tx.abs(a + b) < 1.1754943508222875e-38 else a + b
