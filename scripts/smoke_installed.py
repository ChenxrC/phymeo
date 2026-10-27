"""Run with an isolated, non-editable wheel environment: python -I this_file.py."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import phymeo


def main():
    installed = Path(phymeo.__file__).resolve()
    if not installed.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Expected a wheel inside {sys.prefix}, imported {installed}")
    model = phymeo.ParticleModel(np.array([0.0, 1.0]))
    state = phymeo.ParticleState(np.zeros((2, 3)), np.zeros((2, 3)))
    inputs = phymeo.StepInputs.zeros(model.particle_count)
    if state.particle_count != 2 or inputs.force.shape != (2, 3):
        raise RuntimeError("Installed data contract failed")

    class CandidateProvider(phymeo.SemiImplicitSolver):
        capabilities = replace(phymeo.SemiImplicitSolver.capabilities, provider_kind="learned")

    free = phymeo.Stepper(CandidateProvider(), verifier=phymeo.FreeParticleVerifier()).step(
        model, state, inputs, 0.1
    )
    if not free.report.accepted:
        raise RuntimeError("Installed free solver failed")
    np.testing.assert_allclose(free.state.position[1, 1], -0.0981)
    constraints = phymeo.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.zeros(1))
    state.position[1, 0] = 2.0
    constrained = phymeo.Stepper(phymeo.XPBDSolver(constraints)).step(
        phymeo.ParticleModel(np.array([0.0, 1.0]), np.zeros(3)), state, inputs, 0.1
    )
    if not constrained.report.accepted:
        raise RuntimeError("Installed constrained solver failed")
    np.testing.assert_allclose(constrained.state.position[1], [1.0, 0.0, 0.0])
    rigid = phymeo.Stepper(phymeo.RigidSemiImplicitSolver()).step(
        phymeo.RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3)),
        phymeo.RigidBodyState(
            np.zeros((1, 3)),
            np.array([[0.0, 0.0, 0.0, 1.0]]),
            np.zeros((1, 3)),
            np.array([[0.0, 0.0, 2.0]]),
        ),
        phymeo.RigidBodyInputs.zeros(1),
        0.5,
    )
    if not rigid.report.accepted:
        raise RuntimeError("Installed rigid solver failed")
    np.testing.assert_allclose(rigid.state.orientation, [[0.0, 0.0, np.sin(0.5), np.cos(0.5)]])
    sphere = phymeo.Stepper(
        phymeo.RigidSpherePlaneSolver(phymeo.SphereShapes(np.array([0.2])))
    ).step(
        phymeo.RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3)),
        phymeo.RigidBodyState(
            np.array([[0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0]]),
            np.array([[0.0, -20.0, 0.0]]),
            np.zeros((1, 3)),
        ),
        phymeo.RigidBodyInputs.zeros(1),
        0.1,
    )
    if not sphere.report.accepted or sphere.state.velocity[0, 1] != 0:
        raise RuntimeError("Installed rigid sphere contact failed")
    np.testing.assert_allclose(sphere.state.position, [[0.0, 0.2, 0.0]])
    exchange = phymeo.ParticleRigidPorts(np.array([1]), np.array([0])).apply_impulses(
        model,
        state,
        phymeo.RigidBodyModel(np.ones(1), np.ones((1, 3))),
        phymeo.RigidBodyState(
            np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, 3)), np.zeros((1, 3))
        ),
        np.array([[0.0, 1.0, 0.0]]),
    )
    if not exchange.accepted:
        raise RuntimeError("Installed point impulse port failed")
    np.testing.assert_allclose(exchange.particle_state.velocity[1], [0, 1, 0])
    np.testing.assert_allclose(exchange.rigid_state.velocity, [[0, -1, 0]])
    np.testing.assert_allclose(exchange.rigid_state.angular_velocity, [[0, 0, -2]])
    cloth = phymeo.cloth_grid(2, 2, spacing=1.0, density=6.0, pinned=[0])
    np.testing.assert_allclose(cloth.vertex_mass, [1.0, 2.0, 2.0, 1.0])
    if cloth.model.inverse_mass[0] != 0 or cloth.triangles.shape != (2, 3):
        raise RuntimeError("Installed cloth builder failed")
    # Closed pair, masses 1 and 3: unit momentum becomes common velocity 1/4.
    pair_state = phymeo.CoupledState(
        phymeo.ParticleState(np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0]])),
        phymeo.RigidBodyState(
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 1.0]]),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
        ),
    )
    pair = phymeo.CoupledXPBDSolver(
        phymeo.ParticleModel(np.ones(1), np.zeros(3)),
        phymeo.RigidBodyModel(np.array([1 / 3]), np.ones((1, 3)), np.zeros(3)),
        phymeo.DistanceConstraints(np.empty((0, 2), dtype=np.int64), np.empty(0), np.empty(0)),
        phymeo.SphereShapes(np.ones(1)),
        substeps=1,
        tolerance=1e-10,
    ).step(
        pair_state,
        phymeo.CoupledInputs(phymeo.StepInputs.zeros(1), phymeo.RigidBodyInputs.zeros(1)),
        0.1,
    )
    if not pair.report.accepted:
        raise RuntimeError("Installed joint contact solver failed")
    np.testing.assert_allclose(pair.state.particles.position, [[0.025, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(pair.state.rigid.position, [[1.025, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(pair.state.particles.velocity, [[0.25, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(pair.state.rigid.velocity, [[0.25, 0, 0]], atol=1e-12)
    np.testing.assert_array_equal(pair_state.particles.position, np.zeros((1, 3)))
    with phymeo.make_coupled_runtime(
        phymeo.ParticleModel(np.ones(1), np.zeros(3)),
        phymeo.RigidBodyModel(np.array([1 / 3]), np.ones((1, 3)), np.zeros(3)),
        None,
        phymeo.SphereShapes(np.ones(1)),
        substeps=1,
        tolerance=1e-10,
    ) as runtime:
        staged = runtime.step(pair_state, phymeo.CoupledInputs.zeros(1, 1), 0.1)
        if not staged.report.accepted:
            raise RuntimeError("Installed phased contact program failed")
        np.testing.assert_allclose(staged.state.particles.velocity, [[0.25, 0, 0]], atol=1e-12)
        np.testing.assert_allclose(staged.state.rigid.velocity, [[0.25, 0, 0]], atol=1e-12)
    scene = phymeo.SceneModel((phymeo.DomainBinding("dust", "particles", model),))
    with phymeo.SceneStepper(
        scene, phymeo.FreeSceneProgram(), phymeo.StepSettings(substeps=2)
    ) as runtime:
        staged_free = runtime.step(
            phymeo.SceneState({"dust": state}), phymeo.SceneInputs({"dust": inputs}), 0.1
        )
        if not staged_free.report.accepted:
            raise RuntimeError("Installed generic state registry failed")
        np.testing.assert_allclose(staged_free.state.domains["dust"].velocity[1, 1], -0.981)
    if {"tyx", "torch", "warp"} & set(sys.modules):
        raise RuntimeError("The base wheel imported an optional runtime")
    print(f"Installed-wheel smoke passed: {installed}")


if __name__ == "__main__":
    main()
