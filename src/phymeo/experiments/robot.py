"""Numeric S1 trace using the production scene runtime, not an external engine."""

from dataclasses import fields
from numbers import Integral

import numpy as np

from ..builders.robot import pendulum_chain
from ..core.robot import RobotInputs
from ..core.scene import SceneInputs, SceneState, StepSettings
from ..core.validation import validate_dt
from ..solvers.robot import make_robot_runtime, robot_energy
from .rigid_cloth import SceneRun, SceneRunFailure, trajectory_digest


def run_pendulum(*, steps=1000, dt=1 / 2000, count=2):
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    dt = float(validate_dt(dt, np.dtype(np.float64)))
    model, initial = pendulum_chain(count)
    arrays = {
        "q": np.empty((steps + 1, count)),
        "v": np.empty((steps + 1, count)),
        "energy": np.empty((steps + 1, 2)),
        "time": np.arange(steps + 1, dtype=float) * dt,
    }
    arrays["q"][0], arrays["v"][0] = initial.q, initial.v
    arrays["energy"][0] = robot_energy(model, initial)
    state, inputs = SceneState({"robot": initial}), SceneInputs({"robot": RobotInputs.zeros(count)})
    maximum_residual = 0.0
    with make_robot_runtime(model, settings=StepSettings(substeps=1, tolerance=1e-10)) as runtime:
        for step in range(1, steps + 1):
            result = runtime.step(state, inputs, dt)
            if not result.report.accepted:
                raise SceneRunFailure(f"robot step {step}: {result.report.reason}")
            state = result.state
            native = state.domains["robot"]
            arrays["q"][step], arrays["v"][step] = native.q, native.v
            diagnostic = result.report.substeps[0].diagnostics
            arrays["energy"][step] = diagnostic["kinetic_energy"], diagnostic["potential_energy"]
            maximum_residual = max(maximum_residual, diagnostic["equation_residual"])
    model_record = {
        f.name: (
            getattr(model, f.name).tolist()
            if isinstance(getattr(model, f.name), np.ndarray)
            else list(getattr(model, f.name))
        )
        for f in fields(model)
    }
    return SceneRun(
        {
            "scene": "S1" if count == 2 else "pendulum_chain",
            "steps": steps,
            "dt": dt,
            "integrator": "semi_implicit_euler",
            "substeps": 1,
            "dtype": "float64",
            "model": model_record,
            "initial_q": initial.q.tolist(),
            "initial_v": initial.v.tolist(),
            "inputs": "zero_effort_force_torque",
        },
        {
            "accepted_steps": steps,
            "max_equation_residual": maximum_residual,
            "trajectory_sha256": trajectory_digest(arrays),
        },
        arrays,
    )
