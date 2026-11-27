# Copyright 2021 DeepMind Technologies Limited
# Copyright 2026 Phymeo contributors
# Apache-2.0; see LICENSE and NOTICE in this package.
# Contact objective follows MuJoCo 3.3.7 engine_core_constraint.c::mj_constraintUpdate_impl.
"""Device constitutive operator for condim=3 elliptic contact.

Inputs are constraint residual acceleration J*qacc-aref and original impedance
parameters. This computes contact-space force, cost and Hessian; it does not
invent a penalty spring, integrate motion or claim a converged coupled solve.
"""

from dataclasses import dataclass
from threading import RLock

import numpy as np
import tyx as tx


@tx.kernel
def evaluate(
    count: tx.i32,
    residual: tx.ndarray(tx.f32, 2),
    inverse_regularization: tx.ndarray(tx.f32, 2),
    friction: tx.ndarray(tx.f32, 2),
    mu: tx.ndarray(tx.f32, 1),
    eligible: tx.ndarray(tx.i32, 1),
    force: tx.ndarray(tx.f32, 2),
    hessian: tx.ndarray(tx.f32, 3),
    cost: tx.ndarray(tx.f32, 1),
    state: tx.ndarray(tx.i32, 1),
    status: tx.ndarray(tx.i32, 1),
):
    for row in tx.range(count):
        status[row] = 0
        state[row] = -1
        cost[row] = 0.0
        for i in tx.static_range(3):
            force[row, i] = 0.0
            for j in tx.static_range(3):
                hessian[row, i, j] = 0.0
        if eligible[row] != 1:
            status[row] = 2
        m = mu[row]
        if not tx.isfinite(m) or m <= 0.0:
            status[row] = 1
        for i in tx.static_range(3):
            if (
                not tx.isfinite(residual[row, i])
                or not tx.isfinite(inverse_regularization[row, i])
                or inverse_regularization[row, i] <= 0.0
            ):
                status[row] = 1
        for i in tx.static_range(2):
            if not tx.isfinite(friction[row, i]) or friction[row, i] <= 0.0:
                status[row] = 1
        if status[row] == 0:
            # Native cone regularization has D_t = D_n * friction_t^2 / mu^2.
            # Reject incompatible supplied parameters rather than changing law.
            for i in tx.static_range(2):
                expected = (
                    inverse_regularization[row, 0] * (friction[row, i] / m) * (friction[row, i] / m)
                )
                if not tx.isfinite(expected) or tx.abs(
                    expected - inverse_regularization[row, i + 1]
                ) > 2e-5 * tx.max(expected, inverse_regularization[row, i + 1]):
                    status[row] = 3
        if status[row] == 0:
            n = residual[row, 0] * m
            u1 = residual[row, 1] * friction[row, 0]
            u2 = residual[row, 2] * friction[row, 1]
            tangent = tx.sqrt(u1 * u1 + u2 * u2)
            if n >= m * tangent or (tangent <= 0.0 and n >= 0.0):
                state[row] = 0
            elif m * n + tangent <= 0.0 or (tangent <= 0.0 and n < 0.0):
                state[row] = 1
                value = 0.0
                for i in tx.static_range(3):
                    force[row, i] = -inverse_regularization[row, i] * residual[row, i]
                    hessian[row, i, i] = inverse_regularization[row, i]
                    value = (
                        value
                        + 0.5 * inverse_regularization[row, i] * residual[row, i] * residual[row, i]
                    )
                cost[row] = value
            else:
                state[row] = 4
                dm = inverse_regularization[row, 0] / (m * m * (1.0 + m * m))
                delta = n - m * tangent
                cost[row] = 0.5 * dm * delta * delta
                f = -dm * delta * m
                force[row, 0] = f
                force[row, 1] = -f / tangent * u1 * friction[row, 0]
                force[row, 2] = -f / tangent * u2 * friction[row, 1]
                hessian[row, 0, 0] = dm * m * m
                for i in tx.static_range(2):
                    ui = u1
                    if i == 1:
                        ui = u2
                    cross = -dm * m * m * ui / tangent * friction[row, i]
                    hessian[row, 0, i + 1] = cross
                    hessian[row, i + 1, 0] = cross
                    for j in tx.static_range(2):
                        uj = u1
                        if j == 1:
                            uj = u2
                        value = m * n / tangent * (ui / tangent) * (uj / tangent)
                        if i == j:
                            value = value + m * m - m * n / tangent
                        hessian[row, i + 1, j + 1] = (
                            dm * friction[row, i] * friction[row, j] * value
                        )
            if not tx.isfinite(cost[row]):
                status[row] = 4
            for i in tx.static_range(3):
                if not tx.isfinite(force[row, i]):
                    status[row] = 4
                for j in tx.static_range(3):
                    if not tx.isfinite(hessian[row, i, j]):
                        status[row] = 4
        if status[row] != 0:
            state[row] = -1
            cost[row] = 0.0
            for i in tx.static_range(3):
                force[row, i] = 0.0
                for j in tx.static_range(3):
                    hessian[row, i, j] = 0.0


@dataclass(frozen=True)
class ContactResponse:
    workspace: object
    generation: int

    def read(self):
        """Explicit diagnostic download; invalid rows remain marked invalid."""
        return self.workspace._read(self)


class EllipticContactWorkspace:
    """Reusable device-only evaluation with explicit host convenience boundary.

    A solver supplies residuals, parameters and geometrically eligible rows on
    the same device. This operator alone is not a friction/dynamics solver.
    """

    def __init__(
        self, count, *, mode="cuda", cache_root=None, memory_limit_bytes=256 * 1024 * 1024
    ):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 1048576:
            raise ValueError("contact count must be in 1..1048576")
        if mode not in ("cuda", "interpret"):
            raise ValueError("explicit cuda or interpret mode required")
        self.count, self.mode = count, mode
        self._lock, self._generation, self.closed = RLock(), 0, False
        self._buffers = {}
        self.layouts = {
            "residual": ((count, 3), np.float32),
            "inverse_regularization": ((count, 3), np.float32),
            "friction": ((count, 2), np.float32),
            "mu": ((count,), np.float32),
            "eligible": ((count,), np.int32),
            "force": ((count, 3), np.float32),
            "hessian": ((count, 3, 3), np.float32),
            "cost": ((count,), np.float32),
            "state": ((count,), np.int32),
            "status": ((count,), np.int32),
        }
        self.owned_bytes = sum(
            int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in self.layouts.values()
        )
        if self.owned_bytes > memory_limit_bytes:
            raise MemoryError("contact response allocation exceeds budget")
        self.runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        try:
            with self.runtime.activate():
                for name, (shape, dtype) in self.layouts.items():
                    value = np.zeros(shape, dtype)
                    self._buffers[name] = tx.cuda.array(value) if mode == "cuda" else value
        except BaseException:
            self.close()
            raise

    def _check(self):
        if self.closed:
            raise RuntimeError("contact workspace closed")

    def evaluate_device(self, residual, inverse_regularization, friction, mu, eligible):
        """Borrow device arrays; caller retains and synchronizes external producers."""
        with self._lock:
            self._check()
            self._generation += 1
            values = (residual, inverse_regularization, friction, mu, eligible)
            for name, value in zip(tuple(self.layouts)[:5], values, strict=True):
                shape, dtype = self.layouts[name]
                if self.mode == "cuda" and (
                    not isinstance(value, tx.cuda.Array) or value.device_ordinal != 0
                ):
                    raise ValueError("same-device TYX CUDA arrays required")
                if tuple(value.shape) != shape or np.dtype(value.dtype) != np.dtype(dtype):
                    raise ValueError("contact input shape/dtype mismatch")
            with self.runtime.activate():
                evaluate(
                    self.count,
                    *values,
                    *(self._buffers[k] for k in ("force", "hessian", "cost", "state", "status")),
                    mode=self.mode,
                )
                self.runtime.sync()
            return ContactResponse(self, self._generation)

    def evaluate(self, residual, inverse_regularization, friction, mu, eligible):
        with self._lock:
            self._check()
            self._generation += 1
            for name, value in zip(
                tuple(self.layouts)[:5],
                (residual, inverse_regularization, friction, mu, eligible),
                strict=True,
            ):
                shape, dtype = self.layouts[name]
                array = np.asarray(value)
                if array.shape != shape or array.dtype != np.dtype(dtype):
                    raise ValueError("contact input shape/dtype mismatch")
                if self.mode == "cuda":
                    self._buffers[name].copy_from(array)
                else:
                    self._buffers[name][...] = array
            return self.evaluate_device(*(self._buffers[k] for k in tuple(self.layouts)[:5]))

    def _read(self, result):
        with self._lock:
            self._check()
            if result.workspace is not self or result.generation != self._generation:
                raise RuntimeError("contact response expired")
            data = {}
            for name in ("force", "hessian", "cost", "state", "status"):
                value = self._buffers[name]
                data[name] = value.to_numpy() if self.mode == "cuda" else value.copy()
                data[name].setflags(write=False)
            return data

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed = True
                try:
                    self.runtime.close()
                finally:
                    for value in self._buffers.values():
                        if hasattr(value, "free"):
                            value.free()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()
