"""Small ordered FP32 operations through the public tyx language."""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


@tx.kernel
def axpy_kernel(
    n: tx.i32,
    a: tx.f32,
    x: tx.ndarray(tx.f32, 1),
    y: tx.ndarray(tx.f32, 1),
    out: tx.ndarray(tx.f32, 1),
):
    for i in tx.range(n):
        out[i] = _add(x[i], _mul(a, y[i]))


@tx.kernel
def dot_kernel(
    n: tx.i32, x: tx.ndarray(tx.f32, 1), y: tx.ndarray(tx.f32, 1), out: tx.ndarray(tx.f32, 1)
):
    for lane in tx.range(1):
        total = 0.0
        for i in range(n):
            total = _add(total, _mul(x[i], y[i]))
        out[0] = total


@tx.kernel
def matmul_kernel(
    rows: tx.i32,
    inner: tx.i32,
    columns: tx.i32,
    a: tx.ndarray(tx.f32, 2),
    b: tx.ndarray(tx.f32, 2),
    out: tx.ndarray(tx.f32, 2),
):
    for row in tx.range(rows):
        for column in range(columns):
            total = 0.0
            for k in range(inner):
                total = _add(total, _mul(a[row, k], b[k, column]))
            out[row, column] = total


@tx.kernel
def cholesky_solve_kernel(
    n: tx.i32,
    mass: tx.ndarray(tx.f32, 2),
    rhs: tx.ndarray(tx.f32, 1),
    factor: tx.ndarray(tx.f32, 2),
    work: tx.ndarray(tx.f32, 1),
    out: tx.ndarray(tx.f32, 1),
    failed: tx.ndarray(tx.i32, 1),
):
    # A single logical lane owns the entire ordered factorization. No atomics,
    # pseudo-inverse, diagonal shifts, or independent per-joint approximation.
    for lane in tx.range(1):
        for i in range(n):
            for j in range(i + 1):
                value = mass[i, j]
                for k in range(j):
                    value = _add(value, -_mul(factor[i, k], factor[j, k]))
                if i == j:
                    if not tx.isfinite(value) or value <= 0.0:
                        failed[0] = 1
                        factor[i, j] = 1.0
                    else:
                        factor[i, j] = tx.sqrt(value)
                else:
                    factor[i, j] = _div(value, factor[j, j])
        for i in range(n):
            value = rhs[i]
            for k in range(i):
                value = _add(value, -_mul(factor[i, k], work[k]))
            work[i] = _div(value, factor[i, i])
        for i in range(n - 1, -1, -1):
            value = work[i]
            for k in range(i + 1, n):
                value = _add(value, -_mul(factor[k, i], out[k]))
            out[i] = _div(value, factor[i, i])
            if not tx.isfinite(out[i]):
                failed[0] = 1
