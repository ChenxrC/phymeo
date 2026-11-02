"""Compare the two sheet materials across the same pair of finite support rails."""

import numpy as np

from ..constraints.robot_contact_patch import RobotContactPatch
from ..constraints.triangle_prism_surface import TrianglePrismSurface
from ..core.model import ParticleModel
from ..core.scene import DomainBinding, SceneInputs, SceneModel, SceneState, StepSettings
from ..core.state import ParticleState, StepInputs
from ..solvers.compliant_cloth_contact import CompliantClothContactProgram, CompliantPatch
from .compliant_cloth import Case, record_step, trace_start
from .sheet_fabric import (
    DRAPE_GRID,
    HALF_THICKNESS_M,
    HEIGHT_M,
    MASS_KG,
    WIDTH_M,
    fabric_preset,
    material_for_sheet,
    sheet_grid,
)


def build_drape(fabric):
    """A free horizontal strip bridging two stationary shelves under Earth's gravity."""
    fabric = fabric_preset(fabric)
    uv, faces = sheet_grid(DRAPE_GRID)
    x = np.column_stack((uv[:, 0], uv[:, 1] - HEIGHT_M / 2, np.full(len(uv), 0.003)))
    model = ParticleModel(np.full(len(x), len(x) / MASS_KG), np.array([0.0, 0.0, -9.81]))
    scene = SceneModel((DomainBinding("cloth", "particles", model),))
    surfaces = tuple(
        TrianglePrismSurface(f"face_{i}", "cloth", ids, half_thickness_m=HALF_THICKNESS_M)
        for i, ids in enumerate(faces)
    )
    patches = []
    for key, lo, hi in (("left_rail", -0.09, -0.032), ("right_rail", 0.032, 0.09)):
        vertices = np.array([[lo, -0.04, 0.0], [hi, -0.04, 0.0], [hi, 0.04, 0.0], [lo, 0.04, 0.0]])
        patches.append(
            CompliantPatch(
                RobotContactPatch(key, 0, vertices),
                width_m=0.0015,
                stiffness_pa_per_m=1.1e6,
                damping_s_per_m=45.0,
                friction=0.8,
                slip_speed_m_s=0.04,
            )
        )
    program = CompliantClothContactProgram(
        scene,
        "cloth",
        material_for_sheet(x, faces, fabric),
        surfaces,
        x,
        patches,
        material_damping_s=0.002,
        max_penetration_m=0.01,
    )
    initial = SceneState({"cloth": ParticleState(x, np.zeros_like(x))})
    inputs = SceneInputs({"cloth": StepInputs.zeros(len(x))})
    return Case(program, initial, inputs, 0.01, 128)


def run_drape(*, fabric="soft"):
    fabric = fabric_preset(fabric)
    case = build_drape(fabric)
    trace = trace_start(case, "drape-" + fabric.name)
    trace["fabric"] = fabric.record(DRAPE_GRID)
    reference = case.initial.domains["cloth"].position
    # Measurement uses the same material columns throughout, even if the sheet slips.
    center = np.isclose(reference[:, 0], 0.0)
    shoulders = np.isclose(np.abs(reference[:, 0]), WIDTH_M / 3)
    distances = case.program.material.distance
    a, b = distances.indices.T

    def measure(state, frame):
        x = state.domains["cloth"].position
        frame["midspan_sag_m"] = float(x[shoulders, 2].mean() - x[center, 2].mean())
        lengths = np.linalg.norm(x[a] - x[b], axis=1)
        frame["max_edge_strain"] = float(np.max(np.abs(lengths / distances.rest_length - 1)))

    state = case.initial
    measure(state, trace["frames"][0])
    with case.program.stepper(
        StepSettings(substeps=case.substeps, position_iterations=1, velocity_iterations=1)
    ) as runtime:
        for step in range(12):
            result = runtime.step(state, case.inputs, case.dt)
            if not record_step(case, trace, result, step * case.dt):
                return trace
            state = result.state
            measure(state, trace["frames"][-1])
    frames = trace["frames"]
    trace.update(
        completed=True,
        time_s=frames[-1]["t"],
        metrics=dict(
            sag_at_80ms_m=frames[8]["midspan_sag_m"],
            peak_sag_m=max(f["midspan_sag_m"] for f in frames),
            max_edge_strain=max(f["max_edge_strain"] for f in frames),
            max_sampled_penetration_m=max(r["penetration_m"] for r in trace["samples"]),
            rail_reaction_at_80ms_n={
                key: sum(v[2] for v in contact["force_n"])
                for key, contact in frames[8]["contacts"].items()
            },
        ),
    )
    return trace
