"""Frozen 0.05 N, 0.1 s loading of the 18-node minimum volume solid."""

from numbers import Integral

import numpy as np

from ..builders.tetrahedral import tetrahedral_block
from ..core.scene import SceneInputs, SceneState, StepSettings
from ..core.state import StepInputs
from ..materials.stvk import deformation_gradient, elastic_energy
from ..solvers.tetrahedral import make_tetrahedral_runtime
from .rigid_cloth import SceneRun, SceneRunFailure, trajectory_digest


def run_soft_block(*, steps=400):
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    model, initial = tetrahedral_block(origin=(0.0, 0.0, 0.0), size=(0.1, 0.1, 0.1))
    h, count = 1 / 4000, model.particle_count
    force = np.zeros((count, 3))
    force[~model.fixed, 1] = -0.05 / np.count_nonzero(~model.fixed)
    arrays = {
        "position": np.empty((steps + 1, count, 3)),
        "velocity": np.empty((steps + 1, count, 3)),
        "energy": np.empty((steps + 1, 3)),
        "deformation": np.empty((steps + 1, 3)),
        "support_force": np.empty((steps, count, 3)),
        "time": np.arange(steps + 1, dtype=float) * h,
        "external_force": force.copy(),
        "fixed": model.fixed.copy(),
        "tetrahedra": model.tetrahedra.copy(),
        "mass": model.mass.copy(),
    }
    arrays["position"][0], arrays["velocity"][0] = initial.position, initial.velocity
    gradient = deformation_gradient(model, initial.position)
    singular = np.linalg.svd(gradient, compute_uv=False)
    arrays["deformation"][0] = np.linalg.det(gradient).min(), singular.min(), singular.max()
    arrays["energy"][0] = 0.0, elastic_energy(model, initial.position), 0.0
    state, inputs = SceneState({"soft": initial}), SceneInputs({"soft": StepInputs(force)})
    max_residual = 0.0
    with make_tetrahedral_runtime(
        model, settings=StepSettings(substeps=1, tolerance=1e-10)
    ) as runtime:
        for step in range(1, steps + 1):
            result = runtime.step(state, inputs, h)
            if not result.report.accepted:
                raise SceneRunFailure(f"soft block step {step}: {result.report.reason}")
            state = result.state
            native, report = state.domains["soft"], result.report.substeps[0]
            np.testing.assert_array_equal(
                native.position[model.fixed], initial.position[model.fixed]
            )
            np.testing.assert_array_equal(
                native.velocity[model.fixed], np.zeros((np.count_nonzero(model.fixed), 3))
            )
            arrays["position"][step], arrays["velocity"][step] = native.position, native.velocity
            d = report.diagnostics
            arrays["energy"][step] = d["kinetic_energy"], d["elastic_energy"], d["potential_energy"]
            arrays["deformation"][step] = (
                d["min_det_f"],
                d["min_singular_value"],
                d["max_singular_value"],
            )
            arrays["support_force"][step - 1] = d["support_force"]
            max_residual = max(max_residual, d["integration_residual"])
    compression = float(
        np.mean(initial.position[~model.fixed, 1] - arrays["position"][-1, ~model.fixed, 1])
    )
    if not np.isfinite(compression) or compression <= 0:
        raise SceneRunFailure("soft block did not produce finite nonzero compression")
    return SceneRun(
        {
            "scene": "P4-soft-block",
            "steps": steps,
            "dt": h,
            "substeps": 1,
            "integrator": "explicit_stvk_semi_implicit_euler",
            "shape": [3, 2, 3],
            "size": [0.1, 0.1, 0.1],
            "origin": [0.0, 0.0, 0.0],
            "density": model.density,
            "young_modulus": model.young_modulus,
            "poisson_ratio": model.poisson_ratio,
            "load_y": -0.05,
            "gravity": model.gravity.tolist(),
            "dtype": "float64",
            "domain": {"det_f_min_exclusive": 0.2, "singular_min": 0.6, "singular_max": 1.4},
        },
        {
            "accepted_steps": steps,
            "compression": compression,
            "max_integration_residual": max_residual,
            "min_det_f": float(arrays["deformation"][:, 0].min()),
            "min_singular_value": float(arrays["deformation"][:, 1].min()),
            "max_singular_value": float(arrays["deformation"][:, 2].max()),
            "trajectory_sha256": trajectory_digest(arrays),
        },
        arrays,
    )
