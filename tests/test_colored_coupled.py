"""Same-color-order CPU reference remains independent of optional device packages."""

import importlib

import numpy as np

from phymeo import CoupledInputs, CoupledState, CoupledXPBDSolver, DistanceConstraints
from phymeo.builders.rigid_cloth import rigid_cloth_scene
from phymeo.constraints.coloring import color_edges


def test_colored_reference_uses_identical_schedule_without_mutating_original_constraints():
    cls = importlib.import_module("phymeo.solvers.colored_coupled").ColoredCoupledXPBDSolver
    scene = rigid_cloth_scene(np.float32)
    original = scene.constraints.indices.copy()
    order = color_edges(original).order
    reordered = DistanceConstraints(
        original[order], scene.constraints.rest_length[order], scene.constraints.compliance[order]
    )
    colored = cls(
        scene.particle_model,
        scene.rigid_model,
        scene.constraints,
        scene.spheres,
        tolerance=1e-6,
        substeps=1,
    )
    expected = CoupledXPBDSolver(
        scene.particle_model,
        scene.rigid_model,
        reordered,
        scene.spheres,
        tolerance=1e-6,
        substeps=1,
    )
    a = colored.step(scene.initial_state, scene.inputs, 1 / 480)
    b = expected.step(scene.initial_state, scene.inputs, 1 / 480)
    assert a.report.accepted and b.report.accepted
    np.testing.assert_array_equal(a.state.particles.position, b.state.particles.position)
    np.testing.assert_array_equal(a.state.rigid.position, b.state.rigid.position)
    np.testing.assert_array_equal(scene.constraints.indices, original)
    assert colored.color_count == len(color_edges(original).offsets) - 1


def test_colored_reference_can_step_without_distance_constraints():
    cls = importlib.import_module("phymeo.solvers.colored_coupled").ColoredCoupledXPBDSolver
    scene = rigid_cloth_scene(np.float32)
    solver = cls(scene.particle_model, scene.rigid_model, None, scene.spheres, substeps=1)
    state = CoupledState(scene.initial_state.particles, scene.initial_state.rigid)
    result = solver.step(state, CoupledInputs.zeros(49, 1, dtype=np.float32), 1 / 480)
    assert result.report.accepted and solver.color_count == 0
