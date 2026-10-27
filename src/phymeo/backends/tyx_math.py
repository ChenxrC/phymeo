"""Explicit FP32 computation with owned arrays and inspectable execution counts.

Host validation may inspect equations but never supplies a replacement solution.
This synchronization adapter does not promise persistent-device performance.
"""

from contextlib import ExitStack
from numbers import Integral, Real

import numpy as np

from ..core.capabilities import SolverFailure


def as_compute(value, name):
    """Own a normal-or-zero FP32 array, rejecting nonzero underflow explicitly."""
    source = np.asarray(value)
    if source.dtype.kind not in "fiu" or not np.isfinite(source).all():
        raise ValueError(f"{name}: expected finite real FP32-representable values")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.array(source, dtype=np.float32, copy=True, order="C")
    if not np.isfinite(result).all() or np.any(
        (source != 0) & (np.abs(result) < np.finfo(np.float32).tiny)
    ):
        raise ValueError(f"{name}: nonzero value is not representable as normal FP32")
    return result


def as_index(value, name):
    source = np.asarray(value)
    if (
        source.dtype.kind not in "iu"
        or np.any(source < np.iinfo(np.int32).min)
        or np.any(source > np.iinfo(np.int32).max)
    ):
        raise ValueError(f"{name}: indices must be representable as i32")
    return np.array(source, dtype=np.int32, copy=True, order="C")


class TyxMath:
    MODES = ("interpret", "cpu_cpp", "cuda")

    def __init__(self, mode, *, cache_root=None):
        if mode not in self.MODES:
            raise ValueError(f"explicit mode must be one of {self.MODES}")
        import tyx as tx

        self._tx, self._mode = tx, mode
        self._runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self._counts = dict(launches=0, completed_launches=0, uploads=0, downloads=0)

    @property
    def mode(self):
        return self._mode

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def _check_open(self):
        if self._runtime.closed:
            raise RuntimeError("tyx mathematics runtime is closed")

    def close(self):
        self._runtime.close()

    def stats(self):
        return dict(self._counts, mode=self.mode, compute_dtype="float32")

    def call(self, kernel, *arguments, outputs=()):
        self._check_open()
        outputs = tuple(outputs)
        if len(set(outputs)) != len(outputs) or any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, Integral)
            or not 0 <= i < len(arguments)
            or not isinstance(arguments[i], np.ndarray)
            for i in outputs
        ):
            raise ValueError("outputs must be unique array argument indices")
        owned = []
        for argument in arguments:
            if isinstance(argument, np.ndarray):
                if argument.dtype not in (np.dtype(np.float32), np.dtype(np.int32)):
                    raise TypeError("kernel arrays require explicit FP32/i32 conversion")
                if argument.size == 0 or argument.size > np.iinfo(np.int32).max:
                    raise ValueError("kernel arrays must be nonempty and fit i32 indexing")
                if not np.isfinite(argument).all():
                    raise ValueError("kernel array contains non-finite values")
                owned.append(np.array(argument, copy=True, order="C"))
            elif isinstance(argument, Integral) and not isinstance(argument, (bool, np.bool_)):
                if not np.iinfo(np.int32).min <= argument <= np.iinfo(np.int32).max:
                    raise ValueError("kernel integer exceeds i32")
                owned.append(int(argument))
            elif isinstance(argument, Real) and not isinstance(argument, (bool, np.bool_)):
                owned.append(as_compute(argument, "scalar").item())
            else:
                raise TypeError("kernel arguments must be real scalars or explicit arrays")
        with self._tx.use_runtime(self._runtime), ExitStack() as resources:
            actual = []
            for value in owned:
                if self.mode == "cuda" and isinstance(value, np.ndarray):
                    device = self._tx.cuda_array(value)
                    resources.callback(device.free)
                    actual.append(device)
                    self._counts["uploads"] += 1
                else:
                    actual.append(value)
            self._counts["launches"] += 1
            kernel(*actual, mode=self.mode)
            if self.mode == "cuda":
                self._runtime.sync()
            self._counts["completed_launches"] += 1
            result = []
            for index in outputs:
                if self.mode == "cuda":
                    value = actual[index].to_numpy()
                    self._counts["downloads"] += 1
                else:
                    value = actual[index]
                if not np.isfinite(value).all():
                    raise SolverFailure("non-finite FP32 kernel result")
                result.append(np.array(value, copy=True, order="C"))
            return tuple(result)

    def axpy(self, x, y, a):
        from ._tyx_math_kernels import axpy_kernel

        x, y = as_compute(x, "x"), as_compute(y, "y")
        if x.shape != y.shape:
            raise ValueError("axpy shapes must match")
        self._check_open()
        if not x.size:
            return x
        (result,) = self.call(
            axpy_kernel,
            x.size,
            a,
            x.reshape(-1),
            y.reshape(-1),
            np.zeros(x.size, np.float32),
            outputs=(4,),
        )
        return result.reshape(x.shape)

    def dot(self, x, y):
        from ._tyx_math_kernels import dot_kernel

        x, y = as_compute(x, "x"), as_compute(y, "y")
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError("dot requires equally sized vectors")
        self._check_open()
        if not x.size:
            return 0.0
        (result,) = self.call(dot_kernel, x.size, x, y, np.zeros(1, np.float32), outputs=(3,))
        return float(result[0])

    def matmul(self, a, b):
        from ._tyx_math_kernels import matmul_kernel

        a, b = as_compute(a, "a"), as_compute(b, "b")
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("matmul requires compatible rank-two shapes")
        self._check_open()
        shape = (a.shape[0], b.shape[1])
        if shape[0] * shape[1] > np.iinfo(np.int32).max:
            raise ValueError("matrix result exceeds i32 indexing")
        out = np.zeros(shape, np.float32)
        if not a.size or not b.size:
            return out
        (result,) = self.call(
            matmul_kernel, a.shape[0], a.shape[1], b.shape[1], a, b, out, outputs=(5,)
        )
        return result

    def solve_spd(self, matrix, rhs):
        from ._tyx_math_kernels import cholesky_solve_kernel

        mass, rhs = as_compute(matrix, "mass"), as_compute(rhs, "rhs")
        if mass.ndim != 2 or rhs.ndim != 1 or mass.shape != (rhs.size, rhs.size) or not rhs.size:
            raise ValueError("SPD mass must be a nonempty square matrix matching rhs")
        m64, b64 = mass.astype(np.float64), rhs.astype(np.float64)
        if np.max(np.abs(m64 - m64.T)) > 2e-5 * np.max(np.abs(m64)):
            raise ValueError("mass must be symmetric")
        answer, failed = self.call(
            cholesky_solve_kernel,
            rhs.size,
            mass,
            rhs,
            np.zeros_like(mass),
            np.zeros_like(rhs),
            np.zeros_like(rhs),
            np.zeros(1, np.int32),
            outputs=(5, 6),
        )
        if failed[0]:
            raise SolverFailure("mass is not representable as positive definite SPD")
        # Independent equation check after the selected backend has produced x.
        x64 = answer.astype(np.float64)
        scale = max(1.0, np.max(np.abs(b64)), np.linalg.norm(m64, np.inf) * np.max(np.abs(x64)))
        residual = np.max(np.abs(m64 @ x64 - b64)) / scale
        if not np.isfinite(residual) or residual > 2e-5:
            raise SolverFailure("mass solve failed normalized residual gate")
        return answer
