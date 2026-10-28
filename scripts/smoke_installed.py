"""Run with an isolated, non-editable wheel environment: python -I this_file.py."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import phymeo


def check_material_path():
    model = phymeo.ParticleModel(np.ones(2), np.zeros(3))
    state = phymeo.ParticleState(np.array([[0.0, 0, 0], [2.0, 0, 0]]), np.zeros((2, 3)))
    edges = phymeo.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.zeros(1))
    with phymeo.make_cloth_runtime(model, edges, record_material_path=True) as runtime:
        result = runtime.step(
            phymeo.SceneState({"cloth": state}),
            phymeo.SceneInputs({"cloth": phymeo.StepInputs.zeros(2)}),
            0.1,
        )
    if not result.report.accepted:
        raise RuntimeError("Installed material path failed")
    path = result.report.substeps[0].diagnostics["material_path"]
    if path["schema"] != "xpbd-distance-path-v1" or path["frames"][0].flags.writeable:
        raise RuntimeError("Installed material snapshot contract failed")
    np.testing.assert_array_equal(
        path["frames"][0][0], [0, 0, 0, 2, 0, 0, 0.5, 0, 0, 1.5, 0, 0, 0, -0.5, -1, 0, 0, -0.5]
    )

    ledger = phymeo.MaterialPathLedger(
        model, edges, state, phymeo.StepInputs.zeros(2), 0.1, phymeo.StepSettings(substeps=1)
    )
    audit = ledger.require(
        result.state.domains["cloth"],
        phymeo.StepInputs.zeros(2),
        np.zeros((2, 3)),
        result.report.substeps[0].diagnostics,
        sweeps=result.report.substeps[0].position_iterations,
    )
    np.testing.assert_allclose(audit["material_impulses"], [[5, 0, 0], [-5, 0, 0]], atol=1e-12)


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


def check_surface_contact():
    """Exercise P6.5 public foundations from the installed wheel without tyx."""
    for name in (
        "WeightedPointPort",
        "PointPairOperator",
        "MultiDomainPointOperator",
        "ClosestFeaturePair",
        "closest_point_triangle",
        "closest_segment_segment",
        "FrictionImpulseResult",
        "solve_friction_disk",
        "ContactImpulseResult",
        "solve_contact_impulses",
    ):
        if name not in phymeo.__all__ or not callable(getattr(phymeo, name, None)):
            raise RuntimeError("Installed wheel is missing P6.5 public API: " + name)
    tri = np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0]])
    feature = phymeo.closest_point_triangle(np.array([0.5, 0.5, 1.0]), tri)
    np.testing.assert_allclose(feature.second_weights, [0.5, 0.25, 0.25], atol=1e-12)
    segment = phymeo.closest_segment_segment(
        np.array([[-1.0, 0, 0], [1.0, 0, 0]]), np.array([[0.0, -1, 1], [0.0, 1, 1]])
    )
    np.testing.assert_allclose(segment.distance, 1, atol=1e-12)
    points = phymeo.ParticlePointPort(
        phymeo.ParticleModel(np.ones(3), np.zeros(3)),
        phymeo.ParticleState(tri, np.zeros((3, 3))),
        [0, 1, 2],
        domain="surface",
    )
    weighted = phymeo.WeightedPointPort(points, feature.second_weights[None])
    np.testing.assert_allclose(weighted.response(np.ones((1, 3))), [[0.375] * 3], atol=1e-12)
    pair = phymeo.PointPairOperator(points, np.array([[0, 1]]))
    np.testing.assert_allclose(pair.response(np.ones((1, 3))), [[2] * 3], atol=1e-12)
    body = phymeo.ParticlePointPort(
        phymeo.ParticleModel(np.array([0.5]), np.zeros(3)),
        phymeo.ParticleState(np.zeros((1, 3)), np.array([[2.0, -1, 0]])),
        [0],
        domain="body",
    )
    floor = phymeo.ParticlePointPort(
        phymeo.ParticleModel(np.zeros(1), np.zeros(3)),
        phymeo.ParticleState(np.zeros((1, 3)), np.zeros((1, 3))),
        [0],
        domain="floor",
    )
    graph = phymeo.MultiDomainPointOperator(
        {"body": body, "floor": floor}, [("body", 0)], [("floor", 0)]
    )
    result = phymeo.solve_contact_impulses(
        graph, np.array([[0.0, 1, 0]]), np.array([0.5]), np.array([0.5])
    )
    np.testing.assert_allclose(result.states["body"].velocity, [[1.5, 0, 0]], atol=2e-9)
    np.testing.assert_allclose(result.world_impulses, [[-1, 2, 0]], atol=2e-9)
    return {"velocity": result.states["body"].velocity, "impulse": result.world_impulses}


def check_bounded_control():
    import phymeo

    result = phymeo.solve_bounded_pd(
        np.eye(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.ones(1),
        np.zeros(1),
        np.array([10.0]),
        np.ones(1),
        np.array([0.25]),
        0.1,
    )
    np.testing.assert_allclose(result.effort, [0.25], atol=1e-12)
    np.testing.assert_allclose(result.predicted_velocity, [0.025], atol=1e-12)
    np.testing.assert_allclose(result.predicted_configuration, [0.0025], atol=1e-12)
    return result


def check_offset_attachment():
    """Known finite-thickness impulse from the installed package."""
    x = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    base = phymeo.ParticlePointPort(
        phymeo.ParticleModel(np.array([1.0, 2, 3]), np.zeros(3)),
        phymeo.ParticleState(x, np.zeros((3, 3))),
        np.arange(3),
    )
    port = phymeo.TriangleOffsetPointPort(
        base,
        np.array([[0, 1, 2]]),
        np.array([[0.2, 0.3, 0.5]]),
        np.array([0.2]),
    )
    force = np.array([[1.0, 0, 0]])
    np.testing.assert_allclose(port.response(force), [[1.09, 0.04, -0.08]], atol=1e-14)
    result = port.apply_impulses(force)
    np.testing.assert_allclose(
        result.velocity, [[0.2, 0, 0.2], [0.6, 0, -0.4], [1.5, 0, 0]], atol=1e-14
    )
    np.testing.assert_allclose(port.candidate_audit(result).energy_change, 0.545, atol=1e-14)


def check_load_replay():
    model = phymeo.ParticleModel(np.ones(1), np.zeros(3))
    scene = phymeo.SceneModel((phymeo.DomainBinding("node", "particles", model),))
    state = phymeo.SceneState({"node": phymeo.ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))})
    inputs = phymeo.SceneInputs({"node": phymeo.StepInputs.zeros(1)})
    with phymeo.SceneLoadReplay(
        scene, phymeo.FreeSceneProgram(), state, inputs, 0.2, channels={"node": ("force",)}
    ) as replay:
        trial = replay.evaluate({"node": {"force": np.array([[1.0, 0, 0]])}})
        if not trial.result.report.accepted:
            raise RuntimeError("Installed load replay failed")
        np.testing.assert_allclose(
            trial.result.state.domains["node"].velocity, [[1, 0, 0]], atol=1e-14
        )
        np.testing.assert_allclose(
            trial.result.state.domains["node"].position, [[0.2, 0, 0]], atol=1e-14
        )
        np.testing.assert_array_equal(
            replay.evaluate({}).result.state.domains["node"].position, np.zeros((1, 3))
        )


def check_particle_plane_contact():
    model = phymeo.ParticleModel(np.ones(1), np.array([0.0, -9.81, 0]))
    scene = phymeo.SceneModel((phymeo.DomainBinding("node", "particles", model),))
    contact = phymeo.ParticlePlaneContacts(
        "node", np.array([0]), np.array([[0.0, 1, 0]]), np.zeros(1), np.zeros(1)
    )
    program = phymeo.ParticlePlaneContactProgram(scene, phymeo.FreeSceneProgram(), (contact,))
    initial = phymeo.SceneState({"node": phymeo.ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))})
    inputs = phymeo.SceneInputs({"node": phymeo.StepInputs.zeros(1)})
    with phymeo.SceneStepper(
        scene, program, phymeo.StepSettings(substeps=1, tolerance=1e-10)
    ) as runtime:
        result = runtime.step(initial, inputs, 0.01)
    if not result.report.accepted:
        raise RuntimeError("Installed dynamic plane contact failed")
    np.testing.assert_allclose(result.state.domains["node"].position, np.zeros((1, 3)), atol=1e-12)
    np.testing.assert_allclose(
        result.report.substeps[0].diagnostics["plane_contact"]["world_impulses"],
        [[0, 0.0981, 0]],
        atol=1e-12,
    )


def check_robot_plane_contact():
    model, _ = phymeo.pendulum_chain(1)
    base = phymeo.RobotProgram(model)
    contact = phymeo.RobotPlaneContacts(
        "robot",
        np.array([0]),
        np.array([[0.0, -0.5, 0]]),
        np.array([[0.0, 1, 0]]),
        np.array([-0.5 * np.cos(0.35)]),
        np.zeros(1),
    )
    program = phymeo.RobotPlaneContactProgram(base.scene, base, (contact,))
    state = phymeo.SceneState({"robot": phymeo.RobotState(np.array([0.35]), np.zeros(1))})
    inputs = phymeo.SceneInputs({"robot": phymeo.RobotInputs.zeros(1)})
    with phymeo.SceneStepper(
        base.scene,
        program,
        phymeo.StepSettings(substeps=1, tolerance=1e-10),
        registry=program.registry,
    ) as runtime:
        result = runtime.step(state, inputs, 0.01)
    if not result.report.accepted:
        raise RuntimeError("Installed robot-plane contact failed")
    np.testing.assert_allclose(result.state.domains["robot"].q, [0.35], atol=2e-10)
    np.testing.assert_allclose(
        result.report.substeps[0].diagnostics["robot_plane_contact"]["world_impulses"],
        [[0, 0.04905, 0]],
        atol=2e-9,
    )


def check_initial_robot_node_contact():
    robot = phymeo.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 1, 0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    nodes = phymeo.ParticleModel(np.ones(1), np.zeros(3))
    scene = phymeo.SceneModel(
        (
            phymeo.DomainBinding("robot", "robot", robot),
            phymeo.DomainBinding("nodes", "particles", nodes),
        )
    )
    base = phymeo.ArticulatedCouplingProgram(
        scene,
        {"robot": phymeo.RobotParticipant(robot), "nodes": phymeo.ClothParticipant(nodes, None)},
        phymeo.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0)),
        [],
    )
    contact = phymeo.RobotNodePlaneContacts(
        "nodes",
        np.array([0]),
        np.array([0]),
        np.array([[1.0, 0, 0]]),
        np.array([[0.0, 1, 0]]),
        np.zeros(1),
    )
    program = phymeo.InitialRobotNodeContactProgram(scene, base, (contact,), registry=base.registry)
    state = phymeo.SceneState(
        {
            "robot": phymeo.RobotState(np.zeros(1), np.zeros(1)),
            "nodes": phymeo.ParticleState(np.array([[1.0, 0, 0]]), np.array([[0.0, -1, 0]])),
        }
    )
    inputs = phymeo.SceneInputs(
        {"robot": phymeo.RobotInputs.zeros(1), "nodes": phymeo.StepInputs.zeros(1)}
    )
    with phymeo.SceneStepper(
        scene, program, phymeo.StepSettings(substeps=1, tolerance=1e-10), registry=base.registry
    ) as runtime:
        result = runtime.step(state, inputs, 0.1)
    if not result.report.accepted:
        raise RuntimeError("Installed initial robot-node contact failed")
    np.testing.assert_allclose(result.state.domains["robot"].v, [-1 / 3], atol=1e-10)
    np.testing.assert_allclose(result.state.domains["nodes"].velocity, [[0, -1 / 3, 0]], atol=1e-10)


def check_initial_robot_material_contact():
    robot = phymeo.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 1, 0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    nodes = phymeo.ParticleModel(np.ones(2), np.zeros(3))
    edges = phymeo.DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([0.01]))
    scene = phymeo.SceneModel(
        (
            phymeo.DomainBinding("robot", "robot", robot),
            phymeo.DomainBinding("nodes", "particles", nodes),
        )
    )
    base = phymeo.ArticulatedCouplingProgram(
        scene,
        {
            "robot": phymeo.RobotParticipant(robot),
            "nodes": phymeo.ClothParticipant(nodes, edges, record_material_path=True),
        },
        phymeo.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0)),
        [],
    )
    contact = phymeo.RobotNodePlaneContacts(
        "nodes",
        np.array([0]),
        np.array([0]),
        np.array([[1.0, 0, 0]]),
        np.array([[0.0, 1, 0]]),
        np.zeros(1),
    )
    program = phymeo.InitialRobotMaterialContactProgram(
        scene,
        base,
        (contact,),
        materials={"nodes": edges},
        budgets=phymeo.MaterialContactBudgets(1e-6, 0.001),
        registry=base.registry,
    )
    state = phymeo.SceneState(
        {
            "robot": phymeo.RobotState(np.zeros(1), np.zeros(1)),
            "nodes": phymeo.ParticleState(
                np.array([[1.0, 0.0, 0.0], [1.0, 0.8, 0.0]]), np.zeros((2, 3))
            ),
        }
    )
    inputs = phymeo.SceneInputs(
        {"robot": phymeo.RobotInputs.zeros(1), "nodes": phymeo.StepInputs.zeros(2)}
    )
    with phymeo.SceneStepper(
        scene, program, phymeo.StepSettings(substeps=1, tolerance=1e-10), registry=base.registry
    ) as runtime:
        result = runtime.step(state, inputs, 0.1)
    if not result.report.accepted:
        raise RuntimeError("Installed initial robot-material contact failed")
    np.testing.assert_allclose(result.state.domains["robot"].v, [-2 / 7], atol=1e-10)
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity, [[0, -2 / 7, 0], [0, 6 / 7, 0]], atol=1e-10
    )


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
    check_material_path()
    check_articulated_pair()
    check_surface_contact()
    check_bounded_control()
    check_offset_attachment()
    check_load_replay()
    check_particle_plane_contact()
    check_robot_plane_contact()
    check_initial_robot_node_contact()
    check_initial_robot_material_contact()
    for name in (
        "TyxRobotParticipant",
        "TyxClothParticipant",
        "TyxTetrahedralParticipant",
        "TyxArticulatedCouplingProgram",
        "make_tyx_articulated_runtime",
    ):
        if name not in phymeo.__all__ or not callable(getattr(phymeo, name, None)):
            raise RuntimeError("Installed wheel is missing P6 public factory: " + name)
    if {"tyx", "torch", "warp", "mujoco"} & set(sys.modules):
        raise RuntimeError("The base wheel imported an optional runtime")
    print(f"Installed-wheel smoke passed: {installed}")


if __name__ == "__main__":
    main()
