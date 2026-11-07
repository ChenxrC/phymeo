"""Explicit FP32 flush-to-zero arithmetic for resident floating trees.

Inputs retain the normal-FP32-or-zero public contract. Finite subnormal
intermediate results are flushed to zero; overflow and NaN remain nonfinite
and must pass through the existing candidate, SPD and residual rejection gates.
This is an engineering floating-point model, not an exact-arithmetic certificate.
"""

import tyx as tx


@tx.func
def _mul(a: tx.f32, b: tx.f32) -> tx.f32:
    return 0.0 if tx.abs(a * b) < 1.1754943508222875e-38 else a * b


@tx.func
def _div(a: tx.f32, b: tx.f32) -> tx.f32:
    return 0.0 if tx.abs(a / b) < 1.1754943508222875e-38 else a / b


@tx.func
def _add(a: tx.f32, b: tx.f32) -> tx.f32:
    return 0.0 if tx.abs(a + b) < 1.1754943508222875e-38 else a + b
