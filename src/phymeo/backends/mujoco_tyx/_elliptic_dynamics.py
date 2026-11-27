"""Reusable device acceleration interface, explicitly separate from an engine step.

Caller supplies original M, smooth acceleration, contact manifold Jacobians,
impedance/reference terms and scalar constraints. Geometry/manifold generation,
integration, reset and PPO are not supplied by this experimental interface.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

import numpy as np
import tyx as tx

from . import _elliptic_dynamics_kernels as k
from ._elliptic_contact import evaluate


@dataclass(frozen=True)
class AccelerationResult:
    workspace: object
    generation: int

    def read(self):
        return self.workspace._read(self)

    def device_view(self):
        """Borrow converged accelerations/forces; invalid batches are rejected."""
        return self.workspace._device_view(self)


class EllipticDynamicsWorkspace:
    """Device-resident Newton/bisection for a fixed-capacity batch of primal systems.

    Status 1 means force balance met the specified numerical tolerance, not a
    native integrator or training qualification. No host state readback during
    the bounded iterations. Failed/unknown results cannot enter device_view.
    Acceleration is represented by acc + acc_low. Preserve both device arrays:
    collapsing them to one FP32 loses force-balance accuracy in the saved M16
    cases. The original smooth input likewise requires smooth + smooth_low.
    Passing saved acceleration cases is not complete engine qualification.
    """

    INPUTS = (
        "counts",
        "scalar_counts",
        "mass",
        "smooth",
        "smooth_low",
        "jac",
        "aref",
        "D",
        "friction",
        "mu",
        "eligible",
        "sj",
        "sa",
        "sd",
        "sl",
        "sk",
    )
    OUTPUTS = (
        "acc",
        "acc_low",
        "base",
        "correction",
        "correction_low",
        "result_counts",
        "force",
        "objective",
        "balance",
        "iterations",
        "status",
    )

    def __init__(
        self,
        worlds,
        nv,
        capacity,
        scalar_capacity,
        *,
        iterations=100,
        tolerance=1e-6,
        mode="cuda",
        cache_root=None,
        memory_limit_bytes=512 * 1024 * 1024,
    ):
        for value, upper in (
            (worlds, 256),
            (nv, 35),
            (capacity, 8192),
            (scalar_capacity, 128),
            (iterations, 100),
        ):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("bounded positive integer dimensions required")
        if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-4:
            raise ValueError("explicit numerical tolerance in (0,1e-4] required")
        if mode not in ("cuda", "interpret"):
            raise ValueError("explicit cuda or interpret mode required")
        self.worlds, self.nv, self.capacity, self.scalar_capacity = (
            worlds,
            nv,
            capacity,
            scalar_capacity,
        )
        self.iterations, self.tolerance, self.mode = iterations, float(tolerance), mode
        self._lock, self._generation, self._borrowed, self.closed = RLock(), 0, 0, False
        w, n, c, s = worlds, nv, worlds * capacity, scalar_capacity
        F, INT = np.float32, np.int32
        self.layouts = {
            "counts": ((w,), INT),
            "scalar_counts": ((w,), INT),
            "mass": ((w, n, n), F),
            "smooth": ((w, n), F),
            "smooth_low": ((w, n), F),
            "jac": ((c, 3, n), F),
            "aref": ((c, 3), F),
            "D": ((c, 3), F),
            "friction": ((c, 2), F),
            "mu": ((c,), F),
            "eligible": ((c,), INT),
            "sj": ((w, s, n), F),
            "sa": ((w, s), F),
            "sd": ((w, s), F),
            "sl": ((w, s), F),
            "sk": ((w, s), INT),
            "acc": ((w, n), F),
            "acc_low": ((w, n), F),
            "force": ((c, 3), F),
            "hessian": ((c, 3, 3), F),
            "cost": ((c,), F),
            "response_state": ((c,), INT),
            "response_status": ((c,), INT),
            "residual": ((c, 3), F),
            "matrix": ((w, n, n), F),
            "factor": ((w, n, n), F),
            "gradient": ((w, n), F),
            "scale": ((w, n), F),
            "direction": ((w, n), F),
            "trial": ((w, n), F),
            "trial_low": ((w, n), F),
            "trial_residual": ((c, 3), F),
            "objective": ((w,), F),
            "balance": ((w,), F),
            "iterations": ((w,), INT),
            "status": ((w,), INT),
        }
        self.layouts.update(
            {
                "correction": ((w, n), F),
                "correction_low": ((w, n), F),
                "zero": ((w, n), F),
                "centered_aref": ((c, 3), F),
                "centered_sa": ((w, s), F),
                "base": ((w, n), F),
                "result_counts": ((w,), INT),
            }
        )
        self.owned_bytes = sum(
            int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in self.layouts.values()
        )
        if self.owned_bytes > memory_limit_bytes:
            raise MemoryError("dynamics workspace exceeds memory budget")
        self._buffers = {}
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

    def _check(self, result=None):
        if self.closed:
            raise RuntimeError("dynamics workspace closed")
        if result is not None and (
            result.workspace is not self or result.generation != self._generation
        ):
            raise RuntimeError("acceleration result expired")

    def _idle(self):
        if self._borrowed:
            raise RuntimeError("acceleration output is borrowed")

    def solve(self, inputs):
        """Explicit host diagnostic convenience; use solve_device for hot path."""
        with self._lock:
            self._check()
            self._idle()
            self._generation += 1
            if set(inputs) != set(self.INPUTS):
                raise ValueError("exact input schema required")
            for name in self.INPUTS:
                value = np.asarray(inputs[name])
                shape, dtype = self.layouts[name]
                if value.shape != shape or value.dtype != np.dtype(dtype):
                    raise ValueError(f"{name}: shape/dtype mismatch")
                if self.mode == "cuda":
                    self._buffers[name].copy_from(value)
                else:
                    self._buffers[name][...] = value
            return self.solve_device({name: self._buffers[name] for name in self.INPUTS})

    def solve_device(self, inputs):
        with self._lock:
            self._check()
            self._idle()
            self._generation += 1
            if set(inputs) != set(self.INPUTS):
                raise ValueError("exact input schema required")
            for name, value in inputs.items():
                shape, dtype = self.layouts[name]
                if self.mode == "cuda" and (
                    not isinstance(value, tx.cuda.Array) or value.device_ordinal != 0
                ):
                    raise ValueError("same-device TYX CUDA arrays required")
                if tuple(value.shape) != shape or np.dtype(value.dtype) != np.dtype(dtype):
                    raise ValueError(f"{name}: shape/dtype mismatch")
            b = {**self._buffers, **inputs}
            w, n, c, s = self.worlds, self.nv, self.capacity, self.scalar_capacity

            def args(*names):
                return tuple(b[name] for name in names)

            with self.runtime.activate():
                k.initialize(
                    w,
                    n,
                    c,
                    s,
                    *args(
                        "counts",
                        "scalar_counts",
                        "mass",
                        "smooth",
                        "jac",
                        "aref",
                        "eligible",
                        "sj",
                        "sa",
                        "sd",
                        "sl",
                        "sk",
                        "factor",
                        "scale",
                        "acc",
                        "iterations",
                        "status",
                    ),
                    mode=self.mode,
                )
                k.center(
                    w,
                    n,
                    c,
                    s,
                    *args(
                        "counts",
                        "scalar_counts",
                        "smooth",
                        "smooth_low",
                        "jac",
                        "aref",
                        "sj",
                        "sa",
                        "centered_aref",
                        "centered_sa",
                        "correction",
                        "correction_low",
                        "status",
                    ),
                    mode=self.mode,
                )
                original_smooth = b["smooth"]
                original_smooth_low = b["smooth_low"]
                b["smooth"] = b["zero"]
                b["acc"] = b["correction"]
                b["acc_low"] = b["correction_low"]
                b["aref"] = b["centered_aref"]
                b["sa"] = b["centered_sa"]
                for iteration in range(self.iterations + 1):
                    k.residuals(
                        w,
                        n,
                        c,
                        *args("counts", "jac", "aref", "acc", "acc_low", "residual"),
                        mode=self.mode,
                    )
                    evaluate(
                        w * c,
                        *args(
                            "residual",
                            "D",
                            "friction",
                            "mu",
                            "eligible",
                            "force",
                            "hessian",
                            "cost",
                            "response_state",
                            "response_status",
                        ),
                        mode=self.mode,
                    )
                    k.assemble(
                        w,
                        n,
                        c,
                        s,
                        *args(
                            "counts",
                            "scalar_counts",
                            "mass",
                            "smooth",
                            "jac",
                            "force",
                            "hessian",
                            "cost",
                            "response_status",
                            "sj",
                            "sa",
                            "sd",
                            "sl",
                            "sk",
                            "acc",
                            "acc_low",
                            "matrix",
                            "gradient",
                            "objective",
                            "balance",
                            "status",
                        ),
                        mode=self.mode,
                    )
                    if iteration < self.iterations:
                        k.newton(
                            w,
                            n,
                            c,
                            self.tolerance,
                            *args(
                                "counts",
                                "scalar_counts",
                                "mass",
                                "smooth",
                                "jac",
                                "aref",
                                "D",
                                "friction",
                                "mu",
                                "sj",
                                "sa",
                                "sd",
                                "sl",
                                "sk",
                                "acc",
                                "acc_low",
                                "matrix",
                                "gradient",
                                "objective",
                                "balance",
                                "factor",
                                "scale",
                                "direction",
                                "trial",
                                "trial_low",
                                "trial_residual",
                                "iterations",
                                "status",
                            ),
                            mode=self.mode,
                        )
                k.finalize(w, self.tolerance, *args("balance", "status"), mode=self.mode)
                k.recover(
                    w,
                    n,
                    original_smooth,
                    original_smooth_low,
                    b["correction"],
                    b["correction_low"],
                    self._buffers["acc"],
                    self._buffers["acc_low"],
                    b["base"],
                    b["counts"],
                    b["result_counts"],
                    mode=self.mode,
                )
                self.runtime.sync()
            return AccelerationResult(self, self._generation)

    def _download(self, name):
        value = self._buffers[name]
        result = value.to_numpy() if self.mode == "cuda" else value.copy()
        result.setflags(write=False)
        return result

    def _read(self, result):
        with self._lock:
            self._check(result)
            return {name: self._download(name) for name in self.OUTPUTS}

    @contextmanager
    def _device_view(self, result):
        with self._lock:
            self._check(result)
            self._idle()
            if np.any(self._download("status") != 1):
                raise ValueError("unconverged/invalid dynamics cannot be consumed")
            self._borrowed += 1
            try:
                yield MappingProxyType({name: self._buffers[name] for name in self.OUTPUTS})
            finally:
                try:
                    self.runtime.sync()
                finally:
                    self._borrowed -= 1

    def close(self):
        with self._lock:
            if not self.closed:
                self._idle()
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
