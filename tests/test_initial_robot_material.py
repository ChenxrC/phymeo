"""Independent mechanics and physical rejection for native material contact."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

import phymeo as ph


def material_case(
    *,
    mode="numpy",
    position=None,
    velocity=None,
    compliance=0.01,
    rest=1.0,
    normal=(0.0, 1.0, 0.0),
    gravity=(0.0, 0.0, 0.0),
    effort=0.0,
    indices=(0,),
    mu=0.0,
    budgets=None,
    trace=True,
    settings=None,
    materials_override=False,
    materials=None,
    old=False,
    base_transform=None,
    no_material=False,
    rotor=False,
    edge_pairs=((0, 1),),
    extra_normals=(),
):
    # Failure mutations: omit paired robot load, substitute free velocity,
    # count material energy per contact, or bypass the path/physical budgets.
    position = np.array([[0.0, 0.0, 0.0], [0.0, 0.8, 0.0]] if position is None else position, float)
    n = len(position)
    robot = ph.RobotModel(
        np.array([-1]),
        ("revolute" if rotor else "prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]) if rotor else np.array([normal]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    nodes = ph.ParticleModel(np.ones(n), np.array(gravity, float))
    edges = (
        None
        if no_material
        else ph.DistanceConstraints(
            np.asarray(edge_pairs, int).reshape(-1, 2),
            np.broadcast_to(np.asarray(rest, float), (len(edge_pairs),)).copy(),
            np.broadcast_to(np.asarray(compliance, float), (len(edge_pairs),)).copy(),
        )
    )
    scene = ph.SceneModel(
        (ph.DomainBinding("robot", "robot", robot), ph.DomainBinding("nodes", "particles", nodes))
    )
    if mode == "numpy":
        participants = {
            "robot": ph.RobotParticipant(robot),
            "nodes": ph.ClothParticipant(nodes, edges, record_material_path=trace),
        }
        factory = ph.ArticulatedCouplingProgram
    else:
        participants = {
            "robot": ph.TyxRobotParticipant(robot, mode),
            "nodes": ph.TyxClothParticipant(nodes, edges, mode, record_material_path=trace),
        }
        factory = ph.TyxArticulatedCouplingProgram
    base = factory(
        scene,
        participants,
        ph.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0)),
        [],
    )
    if base_transform is not None:
        base = base_transform(base)
    ids = np.array(indices, int)
    contacts = (
        ph.RobotNodePlaneContacts(
            "nodes",
            ids,
            np.zeros(len(ids), int),
            position[ids],
            np.tile(normal, (len(ids), 1)),
            np.full(len(ids), mu, float),
        ),
    )
    for extra in extra_normals:
        contacts += (
            ph.RobotNodePlaneContacts(
                "nodes",
                np.array([0]),
                np.array([0]),
                position[[0]],
                np.array([extra], float),
                np.zeros(1),
            ),
        )
    if old:
        program = ph.InitialRobotNodeContactProgram(scene, base, contacts, registry=base.registry)
    else:
        program = ph.InitialRobotMaterialContactProgram(
            scene,
            base,
            contacts,
            materials=materials if materials_override else {"nodes": edges},
            budgets=budgets or ph.MaterialContactBudgets(1e-6, 0.001),
            registry=base.registry,
        )
    settings = settings or ph.StepSettings(
        substeps=1, tolerance=1e-8 if mode == "numpy" else 2e-5, position_iterations=60
    )
    runtime = ph.SceneStepper(
        scene,
        program,
        settings,
        ph.BackendContext("numpy" if mode == "numpy" else "tyx:" + mode),
        base.registry,
    )
    state = ph.SceneState(
        {
            "robot": ph.RobotState(np.zeros(1), np.zeros(1)),
            "nodes": ph.ParticleState(
                position, np.zeros((n, 3)) if velocity is None else np.array(velocity, float)
            ),
        }
    )
    ri = ph.RobotInputs.zeros(1)
    ri.effort[:] = effort
    inputs = ph.SceneInputs({"robot": ri, "nodes": ph.StepInputs.zeros(n)})
    return runtime, state, inputs


def diag(result):
    return result.report.substeps[-1].diagnostics["initial_robot_material_contact"]


@pytest.mark.parametrize(
    "value", [0.0, -1.0, True, np.bool_(False), float("nan"), float("inf"), "1"]
)
@pytest.mark.parametrize("field", ["max_energy_gain_j", "max_material_angular_defect_nms"])
def test_budgets_reject_invalid_physical_limits(value, field):
    args = dict(max_energy_gain_j=1e-6, max_material_angular_defect_nms=0.001)
    args[field] = value
    with pytest.raises((TypeError, ValueError)):
        ph.MaterialContactBudgets(**args)


def test_budgets_cannot_change_after_construction():
    b = ph.MaterialContactBudgets(1e-6, 0.001)
    with pytest.raises(FrozenInstanceError):
        b.max_energy_gain_j = 1.0


@pytest.mark.parametrize("h", [0.1, 0.05, 0.025])
def test_compressed_edge_releases_potential_energy_with_one_robot_reaction(h):
    runtime, state, inputs = material_case()
    with runtime:
        result = runtime.step(state, inputs, h)
    assert result.report.accepted, result.report.reason
    a = 2 + Fraction(1, 100) / Fraction(str(h)) ** 2
    b = Fraction(1, 5) / (Fraction(str(h)) * a)
    p = float(b / (Fraction(3, 2) - 1 / a))
    rv = -p / 2
    top = p / float(a) + float(b)
    v = result.state.domains["nodes"].velocity[:, 1]
    np.testing.assert_allclose(v, [rv, top], atol=5e-8, rtol=0)
    np.testing.assert_allclose(result.state.domains["robot"].v, [rv], atol=5e-8, rtol=0)
    np.testing.assert_allclose(
        result.state.domains["nodes"].position[:, 1], [h * rv, 0.8 + h * top], atol=5e-9, rtol=0
    )
    d = diag(result)
    np.testing.assert_allclose(d["world_impulses"], [[0, p, 0]], atol=5e-8, rtol=0)
    np.testing.assert_allclose(d["robot_generalized_integral"], [-p], atol=5e-8, rtol=0)
    assert abs(sum(v) + 2 * result.state.domains["robot"].v[0]) < 5e-8
    e = d["physical_budgets"]["frozen_model_energy"]
    assert e["after_j"][1] < e["before_j"][0]
    if h == 0.1:
        assert 0.5 * np.sum(v * v) + rv * rv > 4 / 9
        np.testing.assert_allclose(e["after_j"], [6 / 7, 6 / 7], atol=1e-7, rtol=0)


@pytest.mark.parametrize("h,accepted", [(0.1, False), (0.025, True)])
def test_fixed_angular_budget_rejects_coarse_rotating_material(h, accepted):
    runtime, state, inputs = material_case(
        position=[[-1, 0, 0], [1, 0, 0]],
        velocity=[[0, -1, 0], [0, 1, 0]],
        rest=2.0,
        compliance=0.0,
        normal=(0.0, 0.0, 1.0),
    )
    with runtime:
        result = runtime.step(state, inputs, h)
    assert result.report.accepted is accepted, result.report.reason
    if accepted:
        defect = diag(result)["physical_budgets"]["material_angular_defect"]
        np.testing.assert_allclose(defect["l1_sum_nms"], 2 - 2 / np.sqrt(1 + h * h), atol=1e-12)
    else:
        np.testing.assert_array_equal(
            result.state.domains["nodes"].position, state.domains["nodes"].position
        )


def test_energy_budget_rejects_hard_edge_projection_that_creates_energy():
    runtime, state, inputs = material_case(
        position=[[-0.4, 0, 0], [0.4, 0, 0]], compliance=0.0, normal=(0.0, 0.0, 1.0)
    )
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["nodes"].velocity, 0)


def test_missing_material_trace_is_not_accepted_as_free_motion():
    runtime, state, inputs = material_case(trace=False)
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    assert "material path" in result.report.reason


@pytest.mark.parametrize("mapping", [{}, {"other": None}, {"nodes": "opaque"}])
def test_explicit_material_contract_must_match_selected_domains(mapping):
    with pytest.raises((ValueError, TypeError)):
        material_case(materials_override=True, materials=mapping)


def test_old_free_program_still_rejects_material_domain():
    runtime, state, inputs = material_case(old=True)
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted


def check_support(mode, h):
    runtime, state, inputs = material_case(
        mode=mode,
        position=[[0, 0, 0], [0, 255 / 256, 0]],
        gravity=(0, -1, 0),
        effort=2.0,
        compliance=1 / 256,
    )
    start = state.domains["nodes"].position.copy()
    integral = 0.0
    positive = 0.0
    drift = 0.0
    speed = 0.0
    with runtime:
        for _ in range(round(2 / h)):
            result = runtime.step(state, inputs, h)
            assert result.report.accepted, result.report.reason
            state = result.state
            d = diag(result)
            integral += d["world_impulses"][0, 1]
            positive += d["physical_budgets"]["frozen_model_energy"]["positive_gain_upper_j"]
            drift = max(
                drift,
                float(np.max(np.abs(state.domains["nodes"].position - start))),
                float(np.max(np.abs(state.domains["robot"].q))),
            )
            speed = max(
                speed,
                float(np.max(np.abs(state.domains["nodes"].velocity))),
                float(np.max(np.abs(state.domains["robot"].v))),
            )
    assert abs(integral - 4) < 1e-3
    assert drift < 1e-4 and speed < 1e-4
    assert positive < 1e-4
    return state


@pytest.mark.parametrize("h", [1 / 16, 1 / 32])
def test_material_support_two_seconds_needs_velocity_and_final_gap(h):
    check_support("numpy", h)


def check_sliding(mode, h, mu, order=(0, 1)):
    runtime, state, inputs = material_case(
        mode=mode,
        position=[[-0.4, 0, 0], [0.4, 0, 0]],
        gravity=(0, -1, 0),
        effort=2.0,
        compliance=1.0,
        indices=order,
        mu=mu,
    )
    separation = 0.8
    speed = 0.0
    error = 0.0
    max_static = 0.0
    with runtime:
        for step in range(round(2 / h)):
            result = runtime.step(state, inputs, h)
            assert result.report.accepted, result.report.reason
            state = result.state
            d = diag(result)
            x = state.domains["nodes"].position
            if mu == 0.1:
                speed = (speed + h * (1 - separation - mu)) / (1 + 2 * h * h)
                separation += 2 * h * speed
                np.testing.assert_allclose(
                    x[:, 0], [-separation / 2, separation / 2], atol=2e-4, rtol=0
                )
                continuous = 0.9 - 0.1 * np.cos(np.sqrt(2) * (step + 1) * h)
                error = max(error, abs(x[1, 0] - x[0, 0] - continuous))
                # Each sliding contact must exhaust its own friction cone.
                np.testing.assert_allclose(
                    np.abs(d["world_impulses"][:, 0]),
                    mu * d["world_impulses"][:, 1],
                    atol=3e-6,
                    rtol=0,
                )
            else:
                max_static = max(max_static, float(np.max(np.abs(x - [[-0.4, 0, 0], [0.4, 0, 0]]))))
                np.testing.assert_allclose(
                    np.abs(d["world_impulses"][:, 0]), 0.2 * h, atol=3e-6, rtol=0
                )
            py = np.sum(state.domains["nodes"].velocity[:, 1]) + 2 * state.domains["robot"].v[0]
            assert abs(py) < 2e-4
            np.testing.assert_allclose(
                d["robot_generalized_integral"],
                [-np.sum(d["world_impulses"][:, 1])],
                atol=1e-12,
                rtol=0,
            )
    assert max_static < 1e-4
    assert error < 0.02
    return error, state


@pytest.mark.parametrize("mu", [1.0, 0.1])
def test_shared_material_friction_two_seconds_and_step_refinement(mu):
    coarse, _ = check_sliding("numpy", 1 / 16, mu)
    fine, _ = check_sliding("numpy", 1 / 32, mu)
    if mu == 0.1:
        assert fine <= 0.6 * coarse


def test_contact_order_does_not_count_material_energy_twice():
    outputs = []
    for order in ((0, 1), (1, 0)):
        runtime, state, inputs = material_case(
            position=[[-0.4, 0, 0], [0.4, 0, 0]],
            gravity=(0, -1, 0),
            effort=2.0,
            compliance=1.0,
            indices=order,
            mu=0.1,
        )
        with runtime:
            result = runtime.step(state, inputs, 1 / 16)
        assert result.report.accepted, result.report.reason
        d = diag(result)
        np.testing.assert_allclose(
            d["physical_budgets"]["frozen_model_energy"]["before_j"], [0.02, 0.02], atol=1e-15
        )
        outputs.append(result.state.domains["nodes"].position)
    np.testing.assert_allclose(*outputs, atol=1e-9, rtol=0)


@pytest.mark.parametrize("failure", ["predict", "close"])
def test_material_late_failure_restores_history_and_retry_equals_fresh(failure):
    from test_initial_robot_node import InstrumentedBase, with_history

    seen = []

    def wrap(base):
        seen.append(InstrumentedBase(base, failure))
        return seen[-1]

    kwargs = dict(
        position=[[0, 0, 0], [0, 255 / 256, 0]],
        gravity=(0, -1, 0),
        effort=2.0,
        compliance=1 / 256,
        settings=ph.StepSettings(substeps=2, tolerance=1e-8, position_iterations=60),
    )
    runtime, state, inputs = material_case(base_transform=wrap, **kwargs)
    state = with_history(state)
    with runtime:
        failed = runtime.step(state, inputs, 1 / 8)
        assert not failed.report.accepted
        assert seen[0].created == seen[0].closed
        for domain, fields in (("nodes", ("position", "velocity")), ("robot", ("q", "v"))):
            for field in fields:
                np.testing.assert_array_equal(
                    getattr(failed.state.domains[domain], field),
                    getattr(state.domains[domain], field),
                )
        np.testing.assert_array_equal(failed.state.history["count"], [0])
        seen[0].failure = None
        retry = runtime.step(state, inputs, 1 / 8)
    runtime, fresh, inputs = material_case(base_transform=InstrumentedBase, **kwargs)
    with runtime:
        expected = runtime.step(with_history(fresh), inputs, 1 / 8)
    assert retry.report.accepted and expected.report.accepted
    for domain, fields in (("nodes", ("position", "velocity")), ("robot", ("q", "v"))):
        for field in fields:
            np.testing.assert_array_equal(
                getattr(retry.state.domains[domain], field),
                getattr(expected.state.domains[domain], field),
            )
    np.testing.assert_array_equal(retry.state.history["count"], [2])
    assert seen[0].factory_closes == 1


def test_material_config_and_reports_are_owned_and_in_signature():
    edges = ph.DistanceConstraints(np.array([[0, 1]]), np.array([1.0]), np.array([0.01]))
    mapping = {"nodes": edges}
    runtime, state, inputs = material_case(materials_override=True, materials=mapping)
    mapping.clear()
    edges.rest_length.flags.writeable = True
    edges.rest_length[:] = 99
    with runtime:
        first = runtime.step(state, inputs, 0.1)
        assert first.report.accepted, first.report.reason
        assert not diag(first)["physical_budgets"]["material_angular_defect"]["by_domain_nms"][
            "nodes"
        ].flags.writeable
        diag(first)["world_impulses"].flags.writeable = True
        diag(first)["world_impulses"][:] = 99
        second = runtime.step(state, inputs, 0.1)
        assert second.report.accepted
        np.testing.assert_allclose(diag(second)["world_impulses"], [[0, 4 / 7, 0]], atol=5e-8)
        runtime._program.materials["nodes"].rest_length.flags.writeable = True
        runtime._program.materials["nodes"].rest_length[:] = 3
        with pytest.raises(ValueError, match="changed"):
            runtime.step(state, inputs, 0.1)


def test_wrong_material_parameters_cannot_certify_actual_native_path():
    wrong = ph.DistanceConstraints(np.array([[0, 1]]), np.array([1.0]), np.array([0.02]))
    runtime, state, inputs = material_case(materials_override=True, materials={"nodes": wrong})
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted


def test_later_physical_budget_rejection_restores_entire_history():
    from test_initial_robot_node import InstrumentedBase, with_history

    runtime, state, inputs = material_case(
        position=[[-1, 0, 0], [1, 0, 0]],
        compliance=0.0,
        rest=2.0,
        normal=(0.0, 0.0, 1.0),
        budgets=ph.MaterialContactBudgets(1e-6, 5e-5),
        base_transform=InstrumentedBase,
        settings=ph.StepSettings(substeps=2, tolerance=1e-8, position_iterations=60),
    )
    state = with_history(state)
    inputs.domains["nodes"].force[:] = [[0, -1, 0], [0, 1, 0]]
    with runtime:
        result = runtime.step(state, inputs, 0.2)
    assert not result.report.accepted
    assert len(result.report.substeps) == 2
    assert result.report.substeps[0].accepted
    assert result.report.substeps[1].position.residuals["material_angular_defect"] > 1e-8
    np.testing.assert_array_equal(
        result.state.domains["nodes"].position, state.domains["nodes"].position
    )
    np.testing.assert_array_equal(result.state.history["count"], [0])


def test_physical_angular_budget_cannot_cancel_between_domains():
    from phymeo.solvers._material_contact_balance import MaterialContactBalance
    from phymeo.solvers._robot_replay_ledger import FrozenRobotLedger

    runtime, state, inputs = material_case()
    model = ph.ParticleModel(np.ones(1), np.zeros(3))
    extra = {
        "left": ph.ParticleState(np.array([[1.0, 0, 0]]), np.zeros((1, 3))),
        "right": ph.ParticleState(np.array([[-1.0, 0, 0]]), np.zeros((1, 3))),
    }
    initial = ph.SceneState({"robot": state.domains["robot"], **extra})
    forces = ph.SceneInputs(
        {
            "robot": inputs.domains["robot"],
            "left": ph.StepInputs.zeros(1),
            "right": ph.StepInputs.zeros(1),
        }
    )
    final = ph.SceneState(
        {
            "robot": state.domains["robot"],
            **{
                name: ph.ParticleState(s.position, np.array([[0.0, 0.001, 0.0]]))
                for name, s in extra.items()
            },
        }
    )
    ledger = FrozenRobotLedger(
        runtime._program.model,
        initial.domains["robot"],
        forces.domains["robot"],
        0.1,
        runtime.settings,
        1.0,
    )
    balance = MaterialContactBalance(
        {"left": model, "right": model},
        {"left": None, "right": None},
        initial,
        forces,
        "robot",
        0.1,
        ph.MaterialContactBudgets(1.0, 0.001),
    )
    metrics, report, _ = balance.require_metrics(
        final, ledger, {"left": [[0, 0, 0]], "right": [[0, 0, 0]]}, 1e-8
    )
    runtime.close()
    assert metrics["material_angular_defect"] > 1e-8
    np.testing.assert_allclose(report["material_angular_defect"]["l1_sum_nms"], 0.002, atol=1e-18)


def test_original_external_work_is_separate_from_requested_contact_work():
    runtime, state, inputs = material_case()
    inputs.domains["nodes"].force[:] = [0, 0.125, 0]
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted, result.report.reason
    expected = 0.125 * np.sum(
        result.state.domains["nodes"].position[:, 1] - state.domains["nodes"].position[:, 1]
    )
    work = diag(result)["physical_budgets"]["frozen_model_energy"]["original_input_work_j"]
    np.testing.assert_allclose(work, expected, atol=1e-15, rtol=0)


@pytest.mark.parametrize("empty", [False, True])
def test_none_material_and_empty_selected_domain_controls(empty):
    kwargs = dict(no_material=True)
    if empty:
        kwargs.update(position=np.empty((0, 3)), indices=())
    runtime, state, inputs = material_case(**kwargs)
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted, result.report.reason
    assert result.state.domains["nodes"].position.shape == state.domains["nodes"].position.shape
    assert diag(result)["material_reports"]["nodes"]["scope"] == "distance-material-equations-v1"


def test_material_rotor_whole_step_angular_momentum_and_allowed_separation():
    runtime, state, inputs = material_case(position=[[1, 0, 0], [1, 0.8, 0]], rotor=True)
    with runtime:
        result = runtime.step(state, inputs, 0.05)
    assert result.report.accepted, result.report.reason
    node = result.state.domains["nodes"]
    robot = result.state.domains["robot"]
    angular = robot.v[0] + np.sum(np.cross(node.position, node.velocity)[:, 2])
    assert abs(angular) < 1e-9
    d = diag(result)
    assert d["world_impulses"][0, 1] > 0
    assert d["final_gaps"][0] > 1e-8 * 0.05
    np.testing.assert_allclose(
        d["robot_generalized_integral"], [-d["world_impulses"][0, 1]], atol=1e-14
    )


def test_asymmetric_triangle_material_contact_linear_balance_and_energy():
    x = np.array([[-0.4, 0, 0], [0.6, 0, 0], [0.2, 0.7, 0]])
    pairs = ((0, 1), (1, 2), (2, 0))
    rest = np.array([1.0, np.sqrt(0.65), np.sqrt(0.85)]) * 1.01
    runtime, state, inputs = material_case(
        position=x,
        edge_pairs=pairs,
        rest=rest,
        compliance=0.1,
        indices=(0, 1),
        gravity=(0, -1, 0),
        effort=3.0,
        mu=0.5,
    )
    with runtime:
        result = runtime.step(state, inputs, 1 / 32)
    assert result.report.accepted, result.report.reason
    v = result.state.domains["nodes"].velocity
    rv = result.state.domains["robot"].v[0]
    assert abs(np.sum(v[:, 1]) + 2 * rv) < 1e-8
    d = diag(result)
    dx = result.state.domains["nodes"].position
    initial_u = sum(
        (np.linalg.norm(x[i] - x[j]) - rest[k]) ** 2 / 0.2 for k, (i, j) in enumerate(pairs)
    )
    final_u = sum(
        (np.linalg.norm(dx[i] - dx[j]) - rest[k]) ** 2 / 0.2 for k, (i, j) in enumerate(pairs)
    )
    e0 = initial_u + np.sum(x[:, 1])
    e1 = 0.5 * np.sum(v * v) + rv * rv + final_u + np.sum(dx[:, 1])
    work = 3 * result.state.domains["robot"].q[0]
    assert e1 - e0 - work <= 1e-6
    np.testing.assert_allclose(
        d["physical_budgets"]["frozen_model_energy"]["after_j"], [e1, e1], atol=1e-14
    )


def test_exact_energy_gate_detects_gain_hidden_by_nearest_float_rounding():
    from phymeo.solvers._material_contact_balance import MaterialContactBalance
    from phymeo.solvers._robot_replay_ledger import FrozenRobotLedger

    runtime, state, inputs = material_case()
    model = ph.ParticleModel(np.ones(1), np.zeros(3))
    initial = ph.SceneState(
        {
            "robot": state.domains["robot"],
            "nodes": ph.ParticleState(np.zeros((1, 3)), np.array([[1.0, 0, 0]])),
        }
    )
    forces = ph.SceneInputs({"robot": inputs.domains["robot"], "nodes": ph.StepInputs.zeros(1)})
    final = ph.SceneState(
        {
            "robot": state.domains["robot"],
            "nodes": ph.ParticleState(
                np.zeros((1, 3)), np.array([[np.nextafter(1.0, np.inf), 0, 0]])
            ),
        }
    )
    ledger = FrozenRobotLedger(
        runtime._program.model,
        initial.domains["robot"],
        forces.domains["robot"],
        0.1,
        runtime.settings,
        1.0,
    )
    epsilon = 2.0**-52
    # Exact dK=epsilon+epsilon^2/2, which rounds to epsilon as a float.
    exact = Fraction(epsilon) + Fraction(epsilon) ** 2 / 2
    assert float(exact) == epsilon and exact > Fraction(epsilon)
    balance = MaterialContactBalance(
        {"nodes": model},
        {"nodes": None},
        initial,
        forces,
        "robot",
        0.1,
        ph.MaterialContactBudgets(epsilon, 0.001),
    )
    metrics, _, _ = balance.require_metrics(final, ledger, {"nodes": [[0, 0, 0]]}, 1e-8)
    runtime.close()
    assert metrics["frozen_model_energy"] > 1e-8


def check_late_trace_rollback(mode):
    from test_initial_robot_node import InstrumentedBase, with_history

    seen = []

    class EvidenceBase(InstrumentedBase):
        def create_workspace(self, scene, state, inputs, h, settings, context):
            native = super().create_workspace(scene, state, inputs, h, settings, context)
            owner = self

            class Workspace:
                def __getattr__(self, name):
                    return getattr(native, name)

                def diagnostics(self):
                    source = dict(native.diagnostics())
                    if owner.failure == "trace" and state.history["count"][0] == 1:
                        participants = dict(source["participants"])
                        evidence = dict(participants["nodes"])
                        path = dict(evidence["material_path"])
                        frames = [x.copy() for x in path["frames"]]
                        frames[0][0, 6] += 0.001
                        path["frames"] = tuple(frames)
                        evidence["material_path"] = path
                        participants["nodes"] = evidence
                        source["participants"] = participants
                    return source

            return Workspace()

    def wrap(base):
        seen.append(EvidenceBase(base, "trace"))
        return seen[-1]

    settings = ph.StepSettings(
        substeps=2, tolerance=1e-8 if mode == "numpy" else 2e-5, position_iterations=60
    )
    kwargs = dict(
        mode=mode,
        position=[[0, 0, 0], [0, 255 / 256, 0]],
        gravity=(0, -1, 0),
        effort=2.0,
        compliance=1 / 256,
        settings=settings,
    )
    runtime, state, inputs = material_case(base_transform=wrap, **kwargs)
    state = with_history(state)
    with runtime:
        rejected = runtime.step(state, inputs, 1 / 8)
        assert not rejected.report.accepted
        assert len(rejected.report.substeps) == 2 and rejected.report.substeps[0].accepted
        np.testing.assert_array_equal(rejected.state.history["count"], [0])
        np.testing.assert_array_equal(
            rejected.state.domains["nodes"].position, state.domains["nodes"].position
        )
        assert seen[0].created == seen[0].closed
        seen[0].failure = None
        retry = runtime.step(state, inputs, 1 / 8)
    runtime, state, inputs = material_case(base_transform=InstrumentedBase, **kwargs)
    with runtime:
        fresh = runtime.step(with_history(state), inputs, 1 / 8)
    assert retry.report.accepted and fresh.report.accepted
    for name, fields in (("robot", ("q", "v")), ("nodes", ("position", "velocity"))):
        for field in fields:
            np.testing.assert_array_equal(
                getattr(retry.state.domains[name], field), getattr(fresh.state.domains[name], field)
            )
    np.testing.assert_array_equal(retry.state.history["count"], fresh.state.history["count"])


def test_late_material_evidence_failure_restores_history_and_retry():
    check_late_trace_rollback("numpy")


def test_same_material_node_two_planes_aggregates_original_input_once():
    runtime, state, inputs = material_case(
        extra_normals=((0.0, 0.0, 1.0),), gravity=(0.0, -1.0, -1.0)
    )
    with runtime:
        result = runtime.step(state, inputs, 1 / 64)
    assert result.report.accepted, result.report.reason
    d = diag(result)
    assert d["world_impulses"].shape == (2, 3)
    np.testing.assert_allclose(
        d["robot_generalized_integral"], [-np.sum(d["world_impulses"][:, 1])], atol=1e-14
    )
    pnodes = np.sum(result.state.domains["nodes"].velocity, axis=0)
    np.testing.assert_allclose(
        pnodes + np.array([0, 2 * result.state.domains["robot"].v[0], 0]),
        [0, -2 / 64, -2 / 64 + np.sum(d["world_impulses"][:, 2])],
        atol=1e-9,
    )
