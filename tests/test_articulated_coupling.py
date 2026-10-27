"""Independent physical checks for staged robot/node coupling and its transaction."""

import importlib
from dataclasses import replace

import numpy as np
import pytest
from test_robot_dynamics import planar_mass
from test_robot_kinematics import robot_model

from phymeo.builders.tetrahedral import tetrahedral_block
from phymeo.constraints.distance import DistanceConstraints
from phymeo.core.capabilities import SolverFailure
from phymeo.core.model import ParticleModel
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.core.scene import (
    BackendContext,
    DomainBinding,
    HistoryField,
    SceneInputs,
    SceneModel,
    SceneState,
    StepSettings,
)
from phymeo.core.state import ParticleState, StepInputs
from phymeo.runtime.scene_stepper import SceneStepper


def api():
    return importlib.import_module("phymeo.solvers.articulated_coupling")


def adapters():
    return importlib.import_module("phymeo.coupling.participant")


def spheres(indices=(0,), centers=((0.0, 0, 0),), radii=(0.1,)):
    cls = importlib.import_module("phymeo.geometry.robot_sphere").RobotSphereAttachments
    return cls(np.array(indices), np.array(centers), np.array(radii))


def pairs(domain="nodes", nodes=(0,), attachments=(0,)):
    cls = importlib.import_module("phymeo.constraints.node_robot").NodeRobotContact
    return cls(domain, np.array(nodes), np.array(attachments))


def case(*, domains=1, node_y=0.09, node_v=0.0, weight=1.0):
    robot = robot_model(
        1,
        joint_types=("prismatic",),
        joint_axis=np.array([[0.0, 1, 0]]),
        mass=np.array([2.0]),
        com=np.zeros((1, 3)),
        gravity=np.zeros(3),
    )
    factories = {"robot": adapters().RobotParticipant(robot)}
    bindings = [DomainBinding("robot", "robot", robot)]
    states = {"robot": RobotState(np.zeros(1), np.zeros(1))}
    loads = {"robot": RobotInputs.zeros(1)}
    contacts = []
    for i in range(domains):
        name = "nodes" if i == 0 else f"nodes{i}"
        model = ParticleModel(np.array([weight]), np.zeros(3))
        factories[name] = adapters().ClothParticipant(model, None)
        bindings.append(DomainBinding(name, "particles", model))
        states[name] = ParticleState(np.array([[0.0, node_y, 0]]), np.array([[0.0, node_v, 0]]))
        loads[name] = StepInputs.zeros(1)
        contacts.append(pairs(name))
    scene = SceneModel(tuple(bindings))
    program = api().ArticulatedCouplingProgram(scene, factories, spheres(), tuple(contacts))
    return program, SceneState(states), SceneInputs(loads)


def run(program, state, inputs, *, dt=0.01, **settings):
    with api().make_articulated_runtime(
        program, settings=StepSettings(substeps=1, tolerance=1e-11, **settings)
    ) as runtime:
        assert type(runtime) is SceneStepper
        return runtime.step(state, inputs, dt)


def test_single_pair_configuration_load_is_balanced_and_not_applied_twice():
    program, state, inputs = case()
    result = run(program, state, inputs)
    assert result.report.accepted, result.report.reason
    robot, node = result.state.domains["robot"], result.state.domains["nodes"]
    load = 0.01 / 1.5
    np.testing.assert_allclose(robot.q, [-load / 2], atol=1e-14)
    np.testing.assert_allclose(node.position, [[0, 0.09 + load, 0]], atol=1e-14)
    np.testing.assert_allclose(robot.v, [-load / 2 / 0.01], atol=1e-14)
    np.testing.assert_allclose(node.velocity, [[0, load / 0.01, 0]], atol=1e-14)
    events = result.report.substeps[0].diagnostics["contact_events"]
    assert events and all(e["phase"] == "position" for e in events)
    for event in events:
        np.testing.assert_array_equal(event["node_load"], -event["robot_load"])
        np.testing.assert_allclose(event["robot_generalized_load"], [event["robot_load"][1]])
        np.testing.assert_allclose(
            event["robot_configuration_delta"], event["robot_generalized_load"] / 2
        )
        np.testing.assert_array_equal(event["robot_velocity_delta"], [0])
        with pytest.raises(ValueError):
            event["node_load"][:] = 9
    np.testing.assert_array_equal(state.domains["robot"].q, [0])
    np.testing.assert_array_equal(state.domains["nodes"].position, [[0, 0.09, 0]])


def test_single_pair_velocity_impulse_has_independent_inelastic_solution():
    program, state, inputs = case(node_y=0.102, node_v=-0.2)
    result = run(program, state, inputs)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.domains["robot"].q, [0], atol=1e-14)
    np.testing.assert_allclose(result.state.domains["robot"].v, [-0.2 / 3], atol=1e-13)
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity, [[0, -0.2 / 3, 0]], atol=1e-13
    )
    impulses = [
        e
        for e in result.report.substeps[0].diagnostics["contact_events"]
        if e["phase"] == "velocity"
    ]
    assert impulses
    assert sum(e["node_load"][1] for e in impulses) == pytest.approx(0.2 / 1.5, abs=1e-13)


def test_two_node_domains_share_the_same_generalized_mass():
    program, state, inputs = case(domains=2)
    result = run(program, state, inputs)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.domains["robot"].q, [-0.005], atol=1e-11)
    for name in ("nodes", "nodes1"):
        assert result.state.domains[name].position[0, 1] == pytest.approx(0.095, abs=1e-11)
    assert {e["domain"] for e in result.report.substeps[0].diagnostics["contact_events"]} == {
        "nodes",
        "nodes1",
    }


def test_duplicate_pairs_and_two_attachments_do_not_duplicate_response():
    program, state, inputs = case()
    duplicate = api().ArticulatedCouplingProgram(
        program.scene,
        program.participants,
        spheres((0, 0), ((0.0, 0, 0), (0.0, 0, 0)), (0.1, 0.1)),
        (pairs(nodes=(0, 0, 0), attachments=(0, 1, 0)),),
    )
    expected, actual = run(program, state, inputs), run(duplicate, state, inputs)
    assert actual.report.accepted
    np.testing.assert_allclose(
        actual.state.domains["robot"].q, expected.state.domains["robot"].q, atol=1e-13
    )
    np.testing.assert_allclose(
        actual.state.domains["nodes"].velocity, expected.state.domains["nodes"].velocity, atol=1e-13
    )


def test_offset_two_revolute_contact_uses_ancestor_jacobians_at_each_update():
    model = robot_model(gravity=np.zeros(3))
    q = np.array([0.3, -0.2])
    state = RobotState(q, np.zeros(2))
    workspace = (
        adapters()
        .RobotParticipant(model)
        .create_workspace(
            "robot", state, RobotInputs.zeros(2), 0.001, StepSettings(), BackendContext()
        )
    )
    workspace.predict()
    local = np.array([[0.0, -0.5, 0]])
    # Analytic endpoint of a two-rod chain; both ancestors contribute.
    a, b = q[0], q.sum()
    point = np.array([0.5 * (np.sin(a) + np.sin(b)), -0.5 * (np.cos(a) + np.cos(b)), 0])
    jacobian = np.array(
        [
            [0.5 * (np.cos(a) + np.cos(b)), 0.5 * np.cos(b)],
            [0.5 * (np.sin(a) + np.sin(b)), 0.5 * np.sin(b)],
            [0, 0],
        ]
    )
    np.testing.assert_allclose(
        workspace.point_port(np.array([1]), local).positions(), [point], atol=1e-14
    )
    load = np.array([[0.002, -0.001, 0]])
    expected = np.linalg.solve(planar_mass(q), jacobian.T @ load[0])
    workspace.apply_position_load(np.array([1]), local, load)
    np.testing.assert_allclose(workspace.configuration(), q + expected, atol=1e-14)
    np.testing.assert_array_equal(workspace.velocity(), [0, 0])
    workspace.prepare_velocity()
    np.testing.assert_allclose(workspace.velocity(), expected / 0.001, atol=1e-12)
    with pytest.raises(RuntimeError, match="already"):
        workspace.prepare_velocity()
    workspace.close()


def test_nonlinear_pair_event_audits_shared_ancestors_independently():
    model = robot_model(gravity=np.zeros(3))
    # q=0: endpoint (.0,-1,0), offset sphere hits a node on its +x side.
    node = ParticleModel(np.array([2.0]), np.zeros(3))
    scene = SceneModel(
        (DomainBinding("robot", "robot", model), DomainBinding("nodes", "particles", node))
    )
    program = api().ArticulatedCouplingProgram(
        scene,
        {
            "robot": adapters().RobotParticipant(model),
            "nodes": adapters().ClothParticipant(node, None),
        },
        spheres((1,), ((0.0, -0.5, 0),), (0.1,)),
        (pairs(),),
    )
    initial = SceneState(
        {
            "robot": RobotState(np.zeros(2), np.zeros(2)),
            "nodes": ParticleState(np.array([[0.099, -1.0, 0]]), np.zeros((1, 3))),
        }
    )
    inputs = SceneInputs({"robot": RobotInputs.zeros(2), "nodes": StepInputs.zeros(1)})
    result = run(program, initial, inputs)
    assert result.report.accepted
    events = result.report.substeps[0].diagnostics["contact_events"]
    first = events[0]
    np.testing.assert_allclose(
        first["robot_generalized_load"],
        [first["robot_load"][0], first["robot_load"][0] / 2],
        atol=1e-14,
    )
    assert np.count_nonzero(first["robot_configuration_delta"]) == 2
    for event in events:
        q = event["robot_configuration_before"]
        a = q[0]
        origins = (np.zeros(3), np.array([0.5 * np.sin(a), -0.5 * np.cos(a), 0]))
        load = event["robot_load"]
        expected = np.array(
            [np.cross(event["world_point"] - origin, load)[2] for origin in origins]
        )
        np.testing.assert_allclose(event["robot_generalized_load"], expected, atol=1e-14)
        delta = (
            event["robot_configuration_delta"]
            if event["phase"] == "position"
            else event["robot_velocity_delta"]
        )
        np.testing.assert_allclose(planar_mass(q) @ delta, expected, atol=1e-14)


def test_internal_cloth_constraint_and_contact_converge_together():
    program, state, inputs = case()
    model = ParticleModel(np.array([1.0, 0.0]), np.zeros(3))
    constraints = DistanceConstraints(np.array([[0, 1]]), np.array([0.19]), np.array([0.001]))
    bindings = (program.scene.domains[0], DomainBinding("nodes", "particles", model))
    combined = api().ArticulatedCouplingProgram(
        SceneModel(bindings),
        {
            "robot": program.participants["robot"],
            "nodes": adapters().ClothParticipant(model, constraints),
        },
        spheres(),
        (pairs(),),
    )
    initial = SceneState(
        {
            "robot": state.domains["robot"],
            "nodes": ParticleState(np.array([[0.0, 0.09, 0], [0.2, 0.1, 0]]), np.zeros((2, 3))),
        }
    )
    forces = SceneInputs({"robot": inputs.domains["robot"], "nodes": StepInputs.zeros(2)})
    result = run(combined, initial, forces)
    assert result.report.accepted
    diagnostic = result.report.substeps[0].diagnostics
    assert diagnostic["participants"]["nodes"]["distance_multipliers"][0] != 0
    assert diagnostic["contact_events"]
    np.testing.assert_array_equal(result.state.domains["nodes"].position[1], [0.2, 0.1, 0])


def test_tetrahedral_participant_keeps_material_prediction_active():
    program, state, inputs = case()
    model, soft = tetrahedral_block(origin=(-0.05, 0, -0.05), size=(0.1, 0.1, 0.1))
    soft.position[~model.fixed, 1] *= 0.99
    robot = replace(program.participants["robot"].model, joint_position=np.array([[0.0, 0.15, 0]]))
    scene = SceneModel(
        (DomainBinding("robot", "robot", robot), DomainBinding("soft", "tetrahedra", model))
    )
    combined = api().ArticulatedCouplingProgram(
        scene,
        {
            "robot": adapters().RobotParticipant(robot),
            "soft": adapters().TetrahedralParticipant(model),
        },
        spheres(radii=(0.052,)),
        (pairs("soft", tuple(range(18)), (0,) * 18),),
    )
    result = run(
        combined,
        SceneState({"robot": state.domains["robot"], "soft": soft}),
        SceneInputs({"robot": inputs.domains["robot"], "soft": StepInputs.zeros(18)}),
        dt=0.00025,
    )
    assert result.report.accepted, result.report.reason
    diagnostic = result.report.substeps[0].diagnostics
    assert np.linalg.norm(diagnostic["participants"]["soft"]["internal_force"]) > 0
    assert diagnostic["contact_events"]


@pytest.mark.parametrize("kind", ["indices", "center", "radius", "mismatch"])
def test_attachment_configuration_is_validated(kind):
    args = dict(indices=(0,), centers=((0.0, 0, 0),), radii=(0.1,))
    if kind == "indices":
        args["indices"] = (0.0,)
    elif kind == "center":
        args["centers"] = ((0.0, np.nan, 0),)
    elif kind == "radius":
        args["radii"] = (-1.0,)
    else:
        args["radii"] = (0.1, 0.2)
    with pytest.raises((TypeError, ValueError)):
        spheres(**args)


@pytest.mark.parametrize("kind", ["negative", "float", "shape", "domain"])
def test_contact_configuration_is_validated(kind):
    kwargs = (
        {"nodes": (-1,)}
        if kind == "negative"
        else {"nodes": (0.0,)}
        if kind == "float"
        else {"nodes": (0, 1)}
        if kind == "shape"
        else {"domain": ""}
    )
    with pytest.raises((TypeError, ValueError)):
        pairs(**kwargs)


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "family", "model", "link", "node", "sphere", "role"]
)
def test_program_checks_participant_and_geometry_binding(kind):
    program, _, _ = case()
    participants = dict(program.participants)
    contacts, attachments = (pairs(),), spheres()
    if kind == "missing":
        participants.pop("nodes")
    elif kind == "extra":
        participants["extra"] = participants["nodes"]
    elif kind == "family":
        participants["nodes"] = participants["robot"]
    elif kind == "model":
        participants["nodes"] = adapters().ClothParticipant(
            ParticleModel(np.array([2.0]), np.zeros(3)), None
        )
    elif kind == "link":
        attachments = spheres((3,))
    elif kind == "node":
        contacts = (pairs(nodes=(9,)),)
    elif kind == "sphere":
        contacts = (pairs(attachments=(9,)),)
    else:
        contacts = (pairs("robot"),)
    with pytest.raises((TypeError, ValueError)):
        api().ArticulatedCouplingProgram(program.scene, participants, attachments, contacts)


def test_bound_configuration_mutation_rejected_and_backend_not_silently_changed():
    program, state, inputs = case()
    with api().make_articulated_runtime(program) as runtime:
        program.attachments.radii.flags.writeable = True
        program.attachments.radii[0] = 0.2
        with pytest.raises(ValueError, match="changed"):
            runtime.step(state, inputs, 0.01)
    program, _, _ = case()
    for context in (BackendContext("tyx"), BackendContext(dtype=np.float32)):
        with pytest.raises(ValueError, match="unsupported"):
            api().make_articulated_runtime(program, context=context)


@pytest.mark.parametrize(
    "kind", ["coincident", "immobile", "budget", "velocity_budget", "material"]
)
def test_failed_contact_restores_all_domains(kind):
    program, state, inputs = case(
        domains=2 if kind in ("budget", "velocity_budget") else 1,
        node_y=0.102 if kind == "velocity_budget" else 0.0 if kind == "coincident" else 0.09,
        node_v=-0.2 if kind == "velocity_budget" else 0.0,
        weight=0.0 if kind == "immobile" else 1.0,
    )
    if kind == "immobile":
        robot = replace(program.participants["robot"].model, joint_axis=np.array([[1.0, 0, 0]]))
        program = api().ArticulatedCouplingProgram(
            SceneModel((DomainBinding("robot", "robot", robot), program.scene.domains[1])),
            {"robot": adapters().RobotParticipant(robot), "nodes": program.participants["nodes"]},
            spheres(),
            (pairs(),),
        )
    if kind == "material":
        model, soft = tetrahedral_block(origin=(-0.05, 0, -0.05), size=(0.1, 0.1, 0.1))
        robot = replace(
            program.participants["robot"].model,
            joint_axis=np.array([[1.0, 0, 0]]),
            joint_position=np.array([[0.0, 0.15, 0]]),
        )
        program = api().ArticulatedCouplingProgram(
            SceneModel(
                (DomainBinding("robot", "robot", robot), DomainBinding("soft", "tetrahedra", model))
            ),
            {
                "robot": adapters().RobotParticipant(robot),
                "soft": adapters().TetrahedralParticipant(model),
            },
            spheres(radii=(0.12,)),
            (pairs("soft", (9,), (0,)),),
        )
        # Node 9 is not the central top point. Pick that point without depending on builder ordering.
        center_top = np.flatnonzero(np.linalg.norm(soft.position - [0, 0.1, 0], axis=1) < 1e-12)[0]
        program = api().ArticulatedCouplingProgram(
            program.scene,
            program.participants,
            spheres(radii=(0.12,)),
            (pairs("soft", (center_top,), (0,)),),
        )
        state = SceneState({"robot": state.domains["robot"], "soft": soft})
        inputs = SceneInputs({"robot": inputs.domains["robot"], "soft": StepInputs.zeros(18)})
    result = run(
        program,
        state,
        inputs,
        position_iterations=1 if kind == "budget" else 200,
        velocity_iterations=1 if kind == "velocity_budget" else 200,
    )
    assert not result.report.accepted
    for name, before in state.domains.items():
        after = result.state.domains[name]
        for key in vars(before):
            np.testing.assert_array_equal(getattr(after, key), getattr(before, key))
            assert not np.shares_memory(getattr(after, key), getattr(before, key))


@pytest.mark.parametrize("failure", ["predict", "candidate", "diagnostics", "close", "construct"])
@pytest.mark.parametrize("fail_index", [2, 3])
def test_later_failure_restores_every_domain_and_explicit_history_then_retries(failure, fail_index):
    program, state, inputs = case(domains=2)
    program.history_fields = (HistoryField("index", (), np.int64),)
    program.capabilities = replace(program.capabilities, history="explicit")
    state = SceneState(state.domains, {"index": np.array(0, dtype=np.int64)})
    original, count, closed = program.create_workspace, [0], []

    def create(scene, before, *args):
        count[0] += 1
        index = count[0]
        workspace = original(scene, before, *args)
        original_result, original_close = workspace.result, workspace.close

        def result():
            candidate = original_result()
            return SceneState(candidate.domains, {"index": before.history["index"] + 1})

        def close():
            original_close()
            closed.append(index)
            if index == fail_index and failure == "close":
                raise SolverFailure("late close")

        workspace.result, workspace.close = result, close
        if index == fail_index:

            def fail():
                raise SolverFailure("late prediction")

            if failure == "predict":
                workspace.predict = fail
            elif failure == "candidate":
                workspace.result = lambda: SceneState(
                    {"robot": RobotState(np.zeros(1), np.zeros(1))}
                )
            elif failure == "diagnostics":
                workspace.diagnostics = lambda: {"bad": np.nan}
            elif failure == "construct":
                workspace.close()
                raise SolverFailure("late construction")
        return workspace

    program.create_workspace = create
    with api().make_articulated_runtime(
        program, settings=StepSettings(substeps=3, record_substeps=True)
    ) as runtime:
        result = runtime.step(state, inputs, 0.003)
        assert not result.report.accepted and result.report.completed_substeps == fail_index - 1
        assert result.substep_states == () and closed == list(range(1, fail_index + 1))
        np.testing.assert_array_equal(result.state.history["index"], 0)
        for name, before in state.domains.items():
            for key in vars(before):
                np.testing.assert_array_equal(
                    getattr(result.state.domains[name], key), getattr(before, key)
                )
        retry = runtime.step(state, inputs, 0.003)
        assert retry.report.accepted
        np.testing.assert_array_equal(retry.state.history["index"], 3)


def test_partial_participant_construction_closes_preceding_workspaces():
    program, state, inputs = case(domains=2)
    first, last = program.participants["robot"], program.participants["nodes1"]
    original, closed = first.create_workspace, []

    def create(*args):
        workspace = original(*args)
        workspace.close = lambda: closed.append(True)
        return workspace

    def fail(*args):
        raise SolverFailure("partial construction")

    first.create_workspace, last.create_workspace = create, fail
    result = run(program, state, inputs)
    assert not result.report.accepted and closed == [True]


def test_participant_operations_preserve_owned_inputs_units_and_grouped_node_loads():
    model = ParticleModel(np.array([2.0, 0.0]), np.zeros(3))
    state = ParticleState(np.array([[0.0, 0, 0], [1.0, 0, 0]]), np.zeros((2, 3)))
    inputs = StepInputs.zeros(2)
    workspace = (
        adapters()
        .ClothParticipant(model, None)
        .create_workspace("nodes", state, inputs, 0.1, StepSettings(), BackendContext())
    )
    state.position[:] = 99
    inputs.force[:] = 99
    workspace.predict()
    loads = np.array([[0.1, 0, 0], [0.2, 0, 0], [10.0, 0, 0]])
    indices = np.array([0, 0, 1])
    workspace.apply_position_load(indices, None, loads)
    np.testing.assert_allclose(workspace.configuration(), [[0.6, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(workspace.velocity(), np.zeros((2, 3)))
    workspace.prepare_velocity()
    np.testing.assert_allclose(workspace.velocity(), [[6, 0, 0], [0, 0, 0]])
    configuration = workspace.configuration()
    workspace.apply_velocity_impulses(indices, None, loads)
    np.testing.assert_array_equal(workspace.configuration(), configuration)
    np.testing.assert_allclose(workspace.velocity(), [[6.6, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="offset"):
        workspace.point_port(np.array([0]), np.ones((1, 3)))
    workspace.close()


def test_public_coupling_factories_and_contract_are_exported():
    import phymeo

    for name in (
        "RobotParticipant",
        "ClothParticipant",
        "TetrahedralParticipant",
        "ParticipantWorkspace",
        "RobotSphereAttachments",
        "NodeRobotContact",
        "ArticulatedCouplingProgram",
        "make_articulated_runtime",
    ):
        assert hasattr(phymeo, name) and name in phymeo.__all__
