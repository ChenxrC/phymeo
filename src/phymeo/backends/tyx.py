"""Explicit tyx execution through its public API, with owned runtime lifetime."""

from contextlib import ExitStack

import numpy as np

from ..core.model import ParticleModel
from ..core.result import StepReport, StepResult
from ..core.state import ParticleState, StepInputs
from ..dynamics.integrate import validate_step


class TyxSemiImplicitSolver:
    """FP32 free integration using an explicitly selected tyx execution mode.

    Use as a context manager or call close(). CUDA currently transfers host state
    each step; this is a correctness adapter, not a persistent-device performance path.
    Backend/toolchain failures propagate; execution never silently falls back to NumPy.
    """

    MODES = ("interpret", "cpu_cpp", "cpu_llvm", "cuda")

    def __init__(self, mode="interpret", *, cache_root=None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        try:
            import tyx as tx
        except ModuleNotFoundError as error:
            if error.name != "tyx":
                raise
            raise ImportError(
                "Install the optional backend with -r requirements/tyx.txt"
            ) from error
        from ._tyx_kernels import integrate_particles

        self._mode = mode
        self._tx = tx
        self._kernel = integrate_particles
        self._runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )

    @property
    def mode(self):
        return self._mode

    def __enter__(self):
        if self._runtime.closed:
            raise RuntimeError("tyx solver is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def close(self):
        self._runtime.close()

    def step(
        self, model: ParticleModel, state_in: ParticleState, inputs: StepInputs, dt: float
    ) -> StepResult:
        if self._runtime.closed:
            raise RuntimeError("tyx solver is closed")
        candidate, force, h = validate_step(model, state_in, inputs, dt)
        if candidate.position.dtype != np.float32:
            raise ValueError("the tyx adapter supports FP32 state only")
        count = model.particle_count
        if count > np.iinfo(np.int32).max:
            raise ValueError("particle count exceeds the tyx i32 kernel interface")
        if count == 0:
            return StepResult(candidate, StepReport(accepted=True, backend=f"tyx:{self.mode}"))
        host_inputs = (
            candidate.position,
            candidate.velocity,
            force,
            model.inverse_mass,
            model.gravity,
        )
        with self._tx.use_runtime(self._runtime), np.errstate(over="ignore", invalid="ignore"):
            if self.mode == "cuda":
                position, velocity = self._cuda_step(count, h, host_inputs)
            else:
                position = np.empty_like(candidate.position)
                velocity = np.empty_like(candidate.velocity)
                self._kernel(count, h, *host_inputs, position, velocity, mode=self.mode)
        if not (np.isfinite(position).all() and np.isfinite(velocity).all()):
            return StepResult(
                state_in.copy(),
                StepReport(
                    accepted=False,
                    reason="non-finite integration result",
                    backend=f"tyx:{self.mode}",
                ),
            )
        return StepResult(
            ParticleState(position, velocity), StepReport(accepted=True, backend=f"tyx:{self.mode}")
        )

    def _cuda_step(self, count, h, host_inputs):
        # Register each allocation immediately, so partial allocation failures release prior arrays.
        with ExitStack() as resources:
            arrays = []
            for value in host_inputs:
                device = self._tx.cuda_array(value)
                resources.callback(device.free)
                arrays.append(device)
            for _ in range(2):
                device = self._tx.cuda_empty((count, 3), np.float32)
                resources.callback(device.free)
                arrays.append(device)
            self._kernel(count, h, *arrays, mode="cuda")
            self._runtime.sync()
            return arrays[-2].to_numpy(), arrays[-1].to_numpy()
