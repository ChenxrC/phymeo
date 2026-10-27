"""FP32 rigid integration on explicitly selected public tyx execution backends."""

from contextlib import ExitStack
from dataclasses import replace

import numpy as np

from ..core.result import StepReport, StepResult
from ..core.rigid import RigidBodyState
from ..core.validation import validate_rigid_step
from ..solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from .tyx import TyxSemiImplicitSolver


class TyxRigidSemiImplicitSolver(TyxSemiImplicitSolver):
    """Free translation and principal-axis Euler rotation; no GPU contacts.

    Kernel arithmetic is FP32. CUDA buffers are per step and host state is copied;
    this adapter makes no persistent-device or throughput claim.
    """

    def __init__(self, mode="interpret", *, cache_root=None):
        super().__init__(mode, cache_root=cache_root)
        from ._tyx_rigid_kernels import integrate_rigid_bodies

        self._kernel = integrate_rigid_bodies

    @property
    def capabilities(self):
        return replace(
            RigidSemiImplicitSolver.capabilities,
            backend=f"tyx:{self.mode}",
            dtypes=frozenset(["float32"]),
        )

    def problem_signature(self):
        return RigidSemiImplicitSolver().problem_signature()

    def step(self, model, state_in, inputs, dt):
        if self._runtime.closed:
            raise RuntimeError("tyx solver is closed")
        snapshot, owned_inputs, h = validate_rigid_step(model, state_in, inputs, dt)
        if snapshot.position.dtype != np.float32:
            raise ValueError("the tyx adapter supports FP32 state only")
        count = model.body_count
        if count > np.iinfo(np.int32).max:
            raise ValueError("body count exceeds the tyx i32 kernel interface")
        if count == 0:
            return StepResult(snapshot, StepReport(True, backend=f"tyx:{self.mode}"))
        arrays = (
            snapshot.position,
            snapshot.orientation,
            snapshot.velocity,
            snapshot.angular_velocity,
            owned_inputs.force,
            owned_inputs.torque,
            model.inverse_mass,
            model.inertia_diagonal,
            model.gravity,
            np.empty_like(snapshot.position),
            np.empty_like(snapshot.orientation),
            np.empty_like(snapshot.velocity),
            np.empty_like(snapshot.angular_velocity),
            np.zeros(count, dtype=np.int32),
        )
        with (
            self._tx.use_runtime(self._runtime),
            np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"),
            ExitStack() as resources,
        ):
            if self.mode == "cuda":
                buffers = []
                for value in arrays:
                    device = self._tx.cuda_array(value)
                    resources.callback(device.free)
                    buffers.append(device)
                resources.callback(self._runtime.sync)
            else:
                buffers = [np.array(value, order="C", copy=True) for value in arrays]
            self._kernel(count, h, *buffers, mode=self.mode)
            self._runtime.sync()
            output = (
                [value.to_numpy() for value in buffers[-5:]]
                if self.mode == "cuda"
                else buffers[-5:]
            )
            position, orientation, velocity, angular, failed = output
            if np.any(failed) or not all(np.isfinite(value).all() for value in output[:-1]):
                return self._failure(snapshot)
            try:
                candidate = RigidBodyState(position, orientation, velocity, angular)
            except ValueError:
                return self._failure(snapshot)
        return StepResult(candidate, StepReport(True, backend=f"tyx:{self.mode}"))

    def _failure(self, snapshot):
        return StepResult(
            RigidBodyState.copy(snapshot),
            StepReport(
                False,
                reason="non-finite or invalid FP32 rigid integration result",
                backend=f"tyx:{self.mode}",
            ),
        )
