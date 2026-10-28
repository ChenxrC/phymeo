"""Verify a two-second material/plane response against an independent oscillator recurrence."""

import argparse
import json
from pathlib import Path

import numpy as np

from phymeo import (
    ClothProgram,
    DistanceConstraints,
    ParticleModel,
    ParticlePlaneContactProgram,
    ParticlePlaneContacts,
    ParticleState,
    SceneInputs,
    SceneState,
    SceneStepper,
    StepInputs,
    StepSettings,
)
from phymeo.experiments.rigid_cloth import provenance

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error("output must be a new path")
rows = []
for h in (0.01, 0.005):
    model = ParticleModel(np.ones(2), np.array([0.0, -9.81, 0]))
    constraint = DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([0.001]))
    native = ClothProgram(model, constraint, domain="cloth")
    contact = ParticlePlaneContacts(
        "cloth", np.array([0]), np.array([[0.0, 1, 0]]), np.zeros(1), np.zeros(1)
    )
    program = ParticlePlaneContactProgram(native.scene, native, (contact,))
    state = SceneState(
        {"cloth": ParticleState(np.array([[0.0, 0, 0], [0, 1.0, 0]]), np.zeros((2, 3)))}
    )
    inputs = SceneInputs({"cloth": StepInputs.zeros(2)})
    y, v = 1.0, 0.0
    maxima = dict(position_error=0.0, velocity_error=0.0, plane_impulse_error=0.0, residual=0.0)
    maximum_sweeps = 0
    with SceneStepper(native.scene, program, StepSettings(substeps=1, tolerance=1e-10)) as runtime:
        for index in range(round(2 / h)):
            old_v = v
            v = (v - 9.81 * h - h * (y - 1) / 0.001) / (1 + h * h / 0.001)
            y += h * v
            impulse = 2 * 9.81 * h + v - old_v
            result = runtime.step(state, inputs, h)
            if not result.report.accepted:
                raise RuntimeError((h, index, result.report.reason))
            state = result.state
            p = state.domains["cloth"]
            diag = result.report.substeps[0].diagnostics["plane_contact"]
            maxima["position_error"] = max(
                maxima["position_error"],
                float(np.max(np.abs(p.position - np.array([[0, 0, 0], [0, y, 0]])))),
            )
            maxima["velocity_error"] = max(
                maxima["velocity_error"],
                float(np.max(np.abs(p.velocity - np.array([[0, 0, 0], [0, v, 0]])))),
            )
            maxima["plane_impulse_error"] = max(
                maxima["plane_impulse_error"], abs(float(diag["world_impulses"][0, 1]) - impulse)
            )
            maxima["residual"] = max(maxima["residual"], result.report.max_residual)
            maximum_sweeps = max(maximum_sweeps, result.report.substeps[0].position_iterations)
    assert (
        max(maxima["position_error"], maxima["velocity_error"], maxima["plane_impulse_error"])
        < 1e-8
    )
    rows.append(
        dict(
            h=h,
            T=2.0,
            steps=round(2 / h),
            maxima=maxima,
            max_outer_sweeps=maximum_sweeps,
            final_upper_y=y,
            final_upper_v=v,
        )
    )
record = dict(
    scope="NumPy two-second unpinned two-node material/plane dynamics, independent implicit-Euler oscillator recurrence; not timing, robot contact or CCD",
    source=provenance(),
    records=rows,
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(rows, indent=2))
