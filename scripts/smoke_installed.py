"""Run with an isolated, non-editable wheel environment: python -I this_file.py."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import phymeo


def check_articulated_pair():
    """A mass2 Y slider and mass1 node share a closed normal contact."""
    robot = phymeo.RobotModel(
        parents=np.array([-1]),
        joint_types=("prismatic",),
        joint_position=np.zeros((1, 3)),
        joint_orientation=np.array([[0.0, 0, 0, 1]]),
        joint_axis=np.array([[0.0, 1, 0]]),
        mass=np.array([2.0]),
        com=np.zeros((1, 3)),
        inertia_diagonal=np.full((1, 3), 0.01),
        gravity=np.zeros(3),
    )
    node = phymeo.ParticleModel(np.ones(1), np.zeros(3))
    state = phymeo.SceneState(
        {
            "robot": phymeo.RobotState(np.zeros(1), np.zeros(1)),
            "cloth": phymeo.ParticleState(np.array([[0.0, 0.09, 0.0]]), np.zeros((1, 3))),
        }
    )
    program = phymeo.ArticulatedCouplingProgram(
        phymeo.SceneModel(
            (
                phymeo.DomainBinding("robot", "robot", robot),
                phymeo.DomainBinding("cloth", "particles", node),
            )
        ),
        {"robot": phymeo.RobotParticipant(robot), "cloth": phymeo.ClothParticipant(node, None)},
        phymeo.RobotSphereAttachments(np.array([0]), np.zeros((1, 3)), np.array([0.1])),
        [phymeo.NodeRobotContact("cloth", np.array([0]), np.array([0]))],
    )
    inputs = phymeo.SceneInputs(
        {"robot": phymeo.RobotInputs.zeros(1), "cloth": phymeo.StepInputs.zeros(1)}
    )
    with phymeo.make_articulated_runtime(program) as runtime:
        result = runtime.step(state, inputs, 0.01)
    if not result.report.accepted:
        raise RuntimeError("Installed articulated contact program failed")
    after_robot, after_node = result.state.domains["robot"], result.state.domains["cloth"]
    np.testing.assert_allclose(after_robot.q, [-0.01 / 3], atol=1e-12)
    np.testing.assert_allclose(after_robot.v, [-1 / 3], atol=1e-12)
    np.testing.assert_allclose(after_node.position, [[0, 0.1 - 0.01 / 3, 0]], atol=1e-12)
    np.testing.assert_allclose(after_node.velocity, [[0, 2 / 3, 0]], atol=1e-12)
    np.testing.assert_allclose(2 * after_robot.v + after_node.velocity[:, 1], [0], atol=1e-12)
    np.testing.assert_array_equal(state.domains["robot"].q, np.zeros(1))
    np.testing.assert_array_equal(state.domains["cloth"].position, [[0, 0.09, 0]])
    return {"robot_q": after_robot.q, "robot_v": after_robot.v, "node_v": after_node.velocity}


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
    robot_model, _ = phymeo.pendulum_chain(1)
    robot_state = phymeo.RobotState(np.zeros(1), np.zeros(1))
    robot_inputs = phymeo.RobotInputs.zeros(1)
    robot_inputs.effort[:] = 1
    with phymeo.make_robot_runtime(robot_model) as runtime:
        robot = runtime.step(
            phymeo.SceneState({"robot": robot_state}),
            phymeo.SceneInputs({"robot": robot_inputs}),
            0.001,
        )
    if not robot.report.accepted:
        raise RuntimeError("Installed robot program failed")
    np.testing.assert_allclose(robot.state.domains["robot"].v, [0.012], atol=1e-12)
    np.testing.assert_allclose(robot.state.domains["robot"].q, [0.000012], atol=1e-12)
    point = phymeo.RobotPointPort(robot_model, robot_state, [0], np.array([[0.0, -0.5, 0]]))
    np.testing.assert_allclose(point.response(np.array([[2.0, 0, 0]])), [[6, 0, 0]], atol=1e-12)
    solid_model = phymeo.TetrahedralModel(
        np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]),
        np.array([[0, 1, 2, 3]]),
        6.0,
        1000.0,
        0.3,
        np.array([True, False, False, False]),
        np.zeros(3),
    )
    solid_state = phymeo.ParticleState(solid_model.rest_position, np.zeros((4, 3)))
    solid_inputs = phymeo.StepInputs.zeros(4)
    solid_inputs.force[1, 0] = 1
    with phymeo.make_tetrahedral_runtime(solid_model) as runtime:
        solid = runtime.step(
            phymeo.SceneState({"soft": solid_state}),
            phymeo.SceneInputs({"soft": solid_inputs}),
            0.001,
        )
    if not solid.report.accepted:
        raise RuntimeError("Installed solid program failed")
    np.testing.assert_allclose(solid.state.domains["soft"].velocity[1], [0.004, 0, 0], atol=1e-12)
    np.testing.assert_allclose(
        solid.state.domains["soft"].position[1], [1.000004, 0, 0], atol=1e-12
    )
    solid_port = phymeo.TetrahedralPointPort(solid_model, solid_state, [1])
    np.testing.assert_allclose(
        solid_port.response(np.array([[1.0, 0, 0]])), [[4, 0, 0]], atol=1e-12
    )
    check_articulated_pair()
    if {"tyx", "torch", "warp", "mujoco"} & set(sys.modules):
        raise RuntimeError("The base wheel imported an optional runtime")
    print(f"Installed-wheel smoke passed: {installed}")


if __name__ == "__main__":
    main()
