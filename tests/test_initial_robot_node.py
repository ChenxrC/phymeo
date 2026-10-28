"""Independent mechanics for initial-configuration paired contact quadrature."""

from dataclasses import replace

import numpy as np
import pytest

import phymeo as ph


def binding(**kwargs):
    values = dict(
        node_domain="nodes",
        node_indices=np.array([0]),
        robot_indices=np.array([0]),
        local_points=np.array([[1.0, 0, 0]]),
        local_normals=np.array([[0.0, 1, 0]]),
        friction=np.zeros(1),
    )
    values.update(kwargs)
    return ph.RobotNodePlaneContacts(**values)


def pair_case(
    *,
    mode="numpy",
    rotor=False,
    points=(1.0,),
    node_v=None,
    robot_v=0.0,
    robot_q=0.0,
    mu=0.0,
    gravity=None,
    effort=0.0,
    settings=None,
    order=None,
    node_constraints=None,
    inverse_mass=None,
    base_transform=None,
    bindings=None,
    model=None,
    **kwargs,
):
    n = len(points)
    if model is None:
        model = ph.RobotModel(
            np.array([-1]),
            ("revolute" if rotor else "prismatic",),
            np.zeros((1, 3)),
            np.array([[0.0, 0, 0, 1]]),
            np.array([[0.0, 0, 1]]) if rotor else np.array([[0.0, 1, 0]]),
            np.array([1.0 if rotor else 2.0]),
            np.zeros((1, 3)),
            np.ones((1, 3)),
            np.zeros(3),
        )
    nodes = ph.ParticleModel(
        np.ones(n) if inverse_mass is None else np.array(inverse_mass, float),
        np.zeros(3) if gravity is None else np.array(gravity, float),
    )
    scene = ph.SceneModel(
        (ph.DomainBinding("robot", "robot", model), ph.DomainBinding("nodes", "particles", nodes))
    )
    if mode == "numpy":
        participants = {
            "robot": ph.RobotParticipant(model),
            "nodes": ph.ClothParticipant(nodes, node_constraints),
        }
        factory = ph.ArticulatedCouplingProgram
    else:
        participants = {
            "robot": ph.TyxRobotParticipant(model, mode),
            "nodes": ph.TyxClothParticipant(nodes, node_constraints, mode),
        }
        factory = ph.TyxArticulatedCouplingProgram
    attachments = ph.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0))
    base = factory(scene, participants, attachments, [])
    if base_transform is not None:
        base = base_transform(base)
    order = np.arange(n) if order is None else np.array(order)
    local = np.column_stack((points, np.zeros(n), np.zeros(n)))
    contacts = (
        (
            binding(
                node_indices=order,
                robot_indices=np.zeros(n, int),
                local_points=local[order],
                local_normals=np.tile([0.0, 1, 0], (n, 1)),
                friction=np.full(n, mu, dtype=float),
            ),
        )
        if bindings is None
        else bindings
    )
    program = ph.InitialRobotNodeContactProgram(
        scene, base, contacts, registry=base.registry, **kwargs
    )
    runtime = ph.SceneStepper(
        scene,
        program,
        settings or ph.StepSettings(substeps=1, tolerance=1e-10),
        ph.BackendContext("numpy" if mode == "numpy" else "tyx:" + mode),
        base.registry,
    )
    position = local.copy()
    velocity = (
        np.tile([0.0, -1, 0], (n, 1)) if node_v is None else np.array(node_v, float).reshape(n, 3)
    )
    state = ph.SceneState(
        {
            "robot": ph.RobotState(np.array([robot_q]), np.array([robot_v], float)),
            "nodes": ph.ParticleState(position, velocity),
        }
    )
    ri = ph.RobotInputs.zeros(1)
    ri.effort[:] = effort
    inputs = ph.SceneInputs({"robot": ri, "nodes": ph.StepInputs.zeros(n)})
    return runtime, state, inputs


def diag(result):
    return result.report.substeps[-1].diagnostics["initial_robot_node_contact"]


def test_binding_owns_and_locks_arrays():
    points = np.array([[1.0, 0, 0]])
    contact = binding(local_points=points)
    points[:] = 9
    np.testing.assert_array_equal(contact.local_points, [[1, 0, 0]])
    for name in ("node_indices", "robot_indices", "local_points", "local_normals", "friction"):
        assert not getattr(contact, name).flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(node_indices=np.array([-1])),
        dict(robot_indices=np.array([-1])),
        dict(node_indices=np.array([0.0])),
        dict(local_points=np.ones((1, 2))),
        dict(local_points=np.zeros((1, 3), np.float32)),
        dict(local_normals=np.zeros((1, 3))),
        dict(local_normals=np.array([[0.0, 2, 0]])),
        dict(friction=np.array([-1.0])),
        dict(node_domain=""),
        dict(robot_indices=np.array([0, 1])),
    ],
)
def test_invalid_bindings(kwargs):
    with pytest.raises((TypeError, ValueError)):
        binding(**kwargs)


def test_prismatic_pair_linear_momentum_and_single_reaction():
    runtime, state, inputs = pair_case()
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["robot"].q, [-1 / 30], atol=1e-11)
    np.testing.assert_allclose(result.state.domains["robot"].v, [-1 / 3], atol=1e-10)
    np.testing.assert_allclose(result.state.domains["nodes"].velocity, [[0, -1 / 3, 0]], atol=1e-10)
    np.testing.assert_allclose(diag(result)["world_impulses"], [[0, 2 / 3, 0]], atol=1e-10)
    assert (
        abs(
            2 * result.state.domains["robot"].v[0]
            + result.state.domains["nodes"].velocity[0, 1]
            + 1
        )
        < 1e-10
    )
    assert diag(result)["replays"] == 2
    np.testing.assert_array_equal(state.domains["robot"].q, [0])


@pytest.mark.parametrize("h", [0.1, 0.05])
def test_rotor_pair_actual_whole_step_angular_momentum(h):
    runtime, state, inputs = pair_case(rotor=True)
    with runtime:
        result = runtime.step(state, inputs, h)
    assert result.report.accepted, result.report.reason
    robot = result.state.domains["robot"]
    node = result.state.domains["nodes"]
    np.testing.assert_allclose(robot.q, [-h / 2], atol=1e-11)
    np.testing.assert_allclose(robot.v, [-0.5], atol=1e-10)
    np.testing.assert_allclose(node.position, [[1, -h / 2, 0]], atol=1e-11)
    np.testing.assert_allclose(node.velocity, [[0, -0.5, 0]], atol=1e-10)
    angular = robot.v[0] + np.cross(node.position[0], node.velocity[0])[2]
    assert abs(angular + 1) < 1e-10
    kinetic = 0.5 * robot.v[0] ** 2 + 0.5 * np.sum(node.velocity**2)
    assert abs(kinetic - 0.25) < 1e-10
    np.testing.assert_allclose(diag(result)["world_impulses"], [[0, 0.5, 0]], atol=1e-10)
    np.testing.assert_allclose(
        diag(result)["final_gaps"], [np.sin(h / 2) - h / 2 * np.cos(h / 2)], atol=1e-12
    )
    assert not np.allclose(diag(result)["initial_jacobians"], diag(result)["final_jacobians"])


@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_shared_rotor_contacts_preserve_cross_response_and_angular_momentum(order):
    runtime, state, inputs = pair_case(
        rotor=True, points=(1.0, 2.0), node_v=[[0, -2, 0], [0, -3, 0]], order=order
    )
    with runtime:
        result = runtime.step(state, inputs, 0.05)
    assert result.report.accepted, result.report.reason
    robot = result.state.domains["robot"]
    node = result.state.domains["nodes"]
    np.testing.assert_allclose(robot.v, [-4 / 3], atol=2e-9)
    np.testing.assert_allclose(node.velocity[:, 1], [-4 / 3, -8 / 3], atol=2e-9)
    np.testing.assert_allclose(
        diag(result)["world_impulses"][:, 1], np.array([2 / 3, 1 / 3])[order], atol=2e-9
    )
    assert abs(robot.v[0] + np.sum(np.cross(node.position, node.velocity)[:, 2]) + 8) < 2e-9


@pytest.mark.parametrize("offset", [0.01, -0.01])
def test_initially_open_or_penetrated_contact_is_refused(offset):
    runtime, state, inputs = pair_case()
    state.domains["nodes"].position[0, 1] = offset
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(
        result.state.domains["nodes"].position, state.domains["nodes"].position
    )


def test_curved_final_penetration_is_not_hidden_by_zero_initial_velocity_residual():
    runtime, state, inputs = pair_case(rotor=True, robot_v=1.0, node_v=[[0, 1, 0]])
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["robot"].q, [0])
    np.testing.assert_array_equal(result.state.domains["robot"].v, [1])


class InstrumentedBase:
    history_fields = (ph.HistoryField("count", (1,), np.int64),)

    def __init__(self, base, failure=None, alter=None):
        self.base, self.failure, self.alter = base, failure, alter
        self.capabilities = replace(base.capabilities, history="explicit")
        self.created = self.closed = self.factory_closes = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def create_workspace(self, scene, state, inputs, h, settings, context):
        native = self.base.create_workspace(scene, state, inputs, h, settings, context)
        count = state.history["count"].copy()
        owner = self
        self.created += 1

        class Workspace:
            def __getattr__(self, name):
                return getattr(native, name)

            def predict(self):
                if owner.failure == "predict" and count[0] == 1:
                    raise ph.SolverFailure("late native predict")
                native.predict()

            def result(self):
                result = native.result()
                domains = dict(result.domains)
                if owner.alter is not None:
                    owner.alter(domains, state, inputs, h)
                return ph.SceneState(domains, {"count": count + 1})

            def close(self):
                owner.closed += 1
                native.close()
                if owner.failure == "close" and count[0] == 1:
                    raise ph.SolverFailure("late native close")

        return Workspace()

    def close(self):
        self.factory_closes += 1
        self.base.close()


def with_history(state):
    return ph.SceneState(state.domains, {"count": np.zeros(1, np.int64)})


def test_history_replays_do_not_advance_twice_and_factory_closes_once():
    seen = []

    def wrap(base):
        seen.append(InstrumentedBase(base))
        return seen[-1]

    runtime, state, inputs = pair_case(
        base_transform=wrap, settings=ph.StepSettings(substeps=2, tolerance=1e-10)
    )
    state = with_history(state)
    result = runtime.step(state, inputs, 0.2)
    assert result.report.accepted, result.report.reason
    np.testing.assert_array_equal(result.state.history["count"], [2])
    assert seen[0].created == seen[0].closed and seen[0].factory_closes == 0
    runtime.close()
    runtime.close()
    assert seen[0].factory_closes == 1


@pytest.mark.parametrize("failure", ["predict", "close"])
def test_late_failure_restores_both_domains_and_hidden_history(failure):
    seen = []

    def wrap(base):
        seen.append(InstrumentedBase(base, failure))
        return seen[-1]

    settings = ph.StepSettings(substeps=2, tolerance=1e-10)
    runtime, state, inputs = pair_case(base_transform=wrap, settings=settings)
    state = with_history(state)
    with runtime:
        failed = runtime.step(state, inputs, 0.2)
        assert not failed.report.accepted
        np.testing.assert_array_equal(failed.state.domains["robot"].q, state.domains["robot"].q)
        np.testing.assert_array_equal(
            failed.state.domains["nodes"].position, state.domains["nodes"].position
        )
        np.testing.assert_array_equal(failed.state.history["count"], [0])
        assert seen[0].created == seen[0].closed
        seen[0].failure = None
        retry = runtime.step(state, inputs, 0.2)
    runtime, fresh, forces = pair_case(base_transform=InstrumentedBase, settings=settings)
    with runtime:
        expected = runtime.step(with_history(fresh), forces, 0.2)
    assert retry.report.accepted and expected.report.accepted
    for domain, field in [
        ("robot", "q"),
        ("robot", "v"),
        ("nodes", "position"),
        ("nodes", "velocity"),
    ]:
        np.testing.assert_array_equal(
            getattr(retry.state.domains[domain], field),
            getattr(expected.state.domains[domain], field),
        )
    np.testing.assert_array_equal(retry.state.history["count"], expected.state.history["count"])


def test_returned_state_and_diagnostics_cannot_contaminate_next_trial():
    runtime, state, inputs = pair_case()
    with runtime:
        first = runtime.step(state, inputs, 0.1)
        first.state.domains["nodes"].position[:] = 8
        diag(first)["initial_jacobians"].flags.writeable = True
        diag(first)["initial_jacobians"][:] = 9
        second = runtime.step(state, inputs, 0.1)
    assert second.report.accepted
    np.testing.assert_allclose(
        second.state.domains["nodes"].position, [[1, -1 / 30, 0]], atol=1e-11
    )


def test_selected_material_domain_is_explicitly_rejected():
    constraints = ph.DistanceConstraints(np.array([[0, 1]]), np.array([0.5]), np.array([0.001]))
    runtime, state, inputs = pair_case(points=(1.0, 2.0), node_constraints=constraints)
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    assert "free dynamics" in result.report.reason
    np.testing.assert_array_equal(
        result.state.domains["nodes"].position, state.domains["nodes"].position
    )


def test_incompatible_node_velocity_postprocessing_rejects_atomically():
    def alter(domains, initial, inputs, h):
        node = domains["nodes"]
        domains["nodes"] = ph.ParticleState(node.position, node.velocity + np.array([[0.1, 0, 0]]))

    runtime, state, inputs = pair_case(
        base_transform=lambda base: InstrumentedBase(base, alter=alter)
    )
    with runtime:
        result = runtime.step(with_history(state), inputs, 0.1)
    assert not result.report.accepted
    assert "free dynamics" in result.report.reason
    np.testing.assert_array_equal(result.state.history["count"], [0])


def test_actual_energy_audit_rejects_tiny_background_velocity_injection():
    def alter(domains, initial, inputs, h):
        if inputs.domains["nodes"].force[0, 1] > 0:
            node = domains["nodes"]
            velocity = node.velocity.copy()
            velocity[0, 0] = np.nextafter(velocity[0, 0], np.inf)
            domains["nodes"] = ph.ParticleState(
                initial.domains["nodes"].position + h * velocity, velocity
            )

    runtime, state, inputs = pair_case(
        node_v=[[1e7, -1e-4, 0]],
        base_transform=lambda base: InstrumentedBase(base, alter=alter),
        settings=ph.StepSettings(substeps=1, tolerance=1e-8),
    )
    with runtime:
        result = runtime.step(with_history(state), inputs, 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["nodes"].velocity, [[1e7, -1e-4, 0]])
    assert result.report.substeps[-1].position.residuals["frozen_energy"] > 1e-8


@pytest.mark.parametrize("origin", [1e16, -1e16])
def test_large_root_translation_cannot_erase_contact_lever(origin):
    template = ph.RobotModel(
        np.array([-1]),
        ("revolute",),
        np.array([[origin, origin, 0]]),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 0, 1]]),
        np.ones(1),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    runtime, state, inputs = pair_case(model=template)
    state.domains["nodes"].position[:] = [[origin, origin, 0]]
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    assert "geometry" in result.report.reason


@pytest.mark.parametrize(
    "field,value", [("tolerance", 1e-5), ("reference_length", 2.0), ("reference_speed", 2.0)]
)
def test_inner_scales_cannot_loosen_outer_contract(field, value):
    settings = ph.StepSettings(substeps=1, tolerance=1e-10)
    runtime, state, inputs = pair_case(
        settings=settings, inner_settings=replace(settings, **{field: value})
    )
    with runtime, pytest.raises(ValueError, match="inner tolerance"):
        runtime.step(state, inputs, 0.1)


def test_fixed_selected_nodes_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        pair_case(inverse_mass=[0.0])


def test_empty_contacts_preserve_native_free_dynamics():
    runtime, state, inputs = pair_case(bindings=())
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["nodes"].position, [[1, -0.1, 0]], atol=1e-14)
    assert diag(result)["world_impulses"].shape == (0, 3)
    assert diag(result)["replays"] == 1


@pytest.mark.parametrize("h", [0.01, 0.005])
@pytest.mark.parametrize("mu,acceleration", [(1.2, 0.0), (0.2, 7.848)])
def test_two_second_finite_actuator_support_and_low_friction_sliding(h, mu, acceleration):
    runtime, state, inputs = pair_case(
        node_v=[[0, 0, 0]], gravity=[9.81, -9.81, 0], effort=9.81, mu=mu
    )
    with runtime:
        for k in range(1, round(2 / h) + 1):
            result = runtime.step(state, inputs, h)
            assert result.report.accepted, (k, result.report.reason)
            state = result.state
            np.testing.assert_allclose(state.domains["robot"].q, [0], atol=1e-9)
            np.testing.assert_allclose(state.domains["robot"].v, [0], atol=1e-9)
            np.testing.assert_allclose(
                state.domains["nodes"].position,
                [[1 + acceleration * h * h * k * (k + 1) / 2, 0, 0]],
                atol=2e-9,
            )
            np.testing.assert_allclose(
                state.domains["nodes"].velocity, [[acceleration * h * k, 0, 0]], atol=2e-9
            )
            np.testing.assert_allclose(
                diag(result)["world_impulses"],
                [[-(9.81 if mu > 1 else 1.962) * h, 9.81 * h, 0]],
                atol=2e-9,
            )


def test_same_node_two_planes_merge_into_one_native_force():
    bindings = (binding(local_normals=np.array([[1.0, 0, 0]])), binding())
    runtime, state, inputs = pair_case(node_v=[[-1, -1, 0]], bindings=bindings)
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["nodes"].velocity, [[0, -1 / 3, 0]], atol=1e-10)
    np.testing.assert_allclose(
        np.sum(diag(result)["world_impulses"], axis=0), [1, 2 / 3, 0], atol=1e-10
    )
    np.testing.assert_allclose(
        diag(result)["input_integrals"]["nodes"]["force"], [[1, 2 / 3, 0]], atol=1e-10
    )


def test_exact_node_free_audit_catches_large_velocity_force_cancellation():
    runtime, state, inputs = pair_case(node_v=[[-1e16, 0, 0]], gravity=[1, 0, 0])
    inputs.domains["nodes"].force[0, 0] = 1e16
    with runtime:
        result = runtime.step(state, inputs, 1.0)
    assert not result.report.accepted
    assert "free dynamics" in result.report.reason
    np.testing.assert_array_equal(result.state.domains["nodes"].velocity, [[-1e16, 0, 0]])
