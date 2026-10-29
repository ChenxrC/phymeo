"""Independent Fraction oracles for the two-stage law."""

from fractions import Fraction as F

import numpy as np
import pytest
from test_dynamic_robot_friction_contact import build, diagnostics, step


@pytest.mark.parametrize(
    "mu,t,v,x", [(2.0, -F(1, 4), F(0), F(0)), (0.25, -F(1, 16), F(3, 16), F(3, 64))]
)
def test_support_fraction_oracle(mu, t, v, x):
    result = step(build(mu=mu))
    d = diagnostics(result)
    np.testing.assert_allclose(d["held_normal"], [0.25], atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["held_tangent"], [[float(t), 0, 0]], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        result.state.domains["nodes"].position, [[float(x), 0, 0]], atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity, [[float(v), 0, 0]], atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(d["terminal_world"], 0, atol=1e-8, rtol=0)


def test_material_impact_fraction_oracle():
    result = step(build(mu=0.5, chain=True))
    d = diagnostics(result)
    np.testing.assert_allclose(d["held_normal"], [3 / 7, 0], atol=1e-8, rtol=0)
    np.testing.assert_array_equal(d["held_tangent"], 0)
    np.testing.assert_allclose(d["terminal_normal"], [1 / 3, 0], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        d["terminal_tangent"], [[-8 / 21, 0, 0], [0, 0, 0]], atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(
        result.state.domains["nodes"].position,
        [[1 / 4, -3 / 56, 0], [1 / 4, 51 / 56, 0]],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity,
        [[13 / 21, -8 / 21, 0], [1, -6 / 7, 0]],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(result.state.domains["robot"].q, [-3 / 56], atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.state.domains["robot"].v, [-8 / 21], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        d["physical_budgets"]["frozen_model_energy"]["after_j"], 1135 / 882, atol=1e-8, rtol=0
    )
    assert (
        abs(
            sum(result.state.domains["nodes"].velocity[:, 1])
            + 2 * result.state.domains["robot"].v[0]
            + 2
        )
        < 1e-8
    )
    assert d["active"].tolist() == [True, False]


def cross_case(mode="numpy"):
    import phymeo as ph

    normals = np.array([[0.0, 1.0, 0.0], [-0.6, 0.8, 0.0]])
    anchors = np.array([[0.0, 0.0, 0.0], [84 / 25, -112 / 25, 0.0]])
    robot = ph.RobotModel(
        np.array([-1, -1]),
        ("prismatic",) * 2,
        anchors,
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        normals,
        np.ones(2),
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.zeros(3),
    )
    colliders = tuple(
        ph.RobotPlaneCollider("floor" if i == 0 else "slant", i, np.zeros(3), n)
        for i, n in enumerate(normals)
    )
    case = build(
        mode,
        mu={"floor": 0.5, "slant": 0.0},
        robot=robot,
        colliders=colliders,
        position=[[0.0, 0.0, 0.0]],
        velocity=[[21 / 10, -9 / 5, 0.0]],
    )
    state = ph.SceneState(
        {**case[1].domains, "robot": ph.RobotState(np.zeros(2), np.array([1.0, 6.0]))},
        case[1].history,
    )
    return case[0], state, *case[2:]


def test_nonparallel_phasewise_a_allows_positive_total_endpoint_work():
    result = step(cross_case(), 1.0)
    d = diagnostics(result)
    np.testing.assert_allclose(d["held_normal"], [1, 1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["held_tangent"], [[-0.5, 0, 0], [0, 0, 0]], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        d["held_state"].domains["nodes"].velocity, [[1, 0, 0]], atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(d["held_state"].domains["robot"].v, [0, 5], atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["terminal_normal"], [0, 14 / 5], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity, [[-17 / 25, 56 / 25, 0]], atol=1e-8, rtol=0
    )
    np.testing.assert_allclose(result.state.domains["robot"].v, [0, 11 / 5], atol=1e-8, rtol=0)
    assert abs(d["held_tangent"][0] @ d["post_relative_velocity"][0] - 17 / 50) < 1e-8
    assert abs(d["terminal_energy"]["change_j"] + 196 / 25) < 1e-8


def test_total_endpoint_b_positive_held_work_is_rejected(monkeypatch):
    import phymeo as ph
    from phymeo.solvers._dynamic_friction_workspace import _DynamicFrictionWorkspace as W

    hit = []

    def prescribe_b(self):
        self.ps += 1
        self.p[:] = [241 / 275, 72 / 55]
        self.t[:] = [[201 / 550, 0, 0], [0, 0, 0]]
        self._evaluate()
        residual, _, work, _ = self._friction_metrics(False)
        assert abs(work[0] - 4221 / 6875) < 1e-12
        assert residual["positive_work"] > self.settings.tolerance
        hit.append(True)

    monkeypatch.setattr(W, "position_sweep", prescribe_b)
    case = cross_case()
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 1.0)
    assert hit and not result.report.accepted


def test_terminal_only_fake_static_support_fails_held_law(monkeypatch):
    import phymeo as ph
    from phymeo.solvers._dynamic_friction_workspace import _DynamicFrictionWorkspace as W

    def normal_only(self):
        self.ps += 1
        self.p[:] = 0.25
        self._evaluate()
        assert self.trial.result.state.domains["nodes"].position[0, 0] == 1 / 16
        assert self._friction_metrics(False)[0]["dissipation"] > 0

    monkeypatch.setattr(W, "position_sweep", normal_only)
    case = build()
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted


def test_exact_world_audit_rotated_and_one_ulp_controls():
    import phymeo as ph
    from phymeo.solvers._dynamic_friction_law import audit, capacity, invariants

    n = np.array([0.6, 0.8, 0.0])
    s = np.array([-0.2, 0.4, 0.0])
    u = np.array([0.8, -0.6, 0.0])
    metrics, _, _ = audit(n, s, u, capacity(n, s, 2.0), 3.0, 1.0)
    assert metrics["cone"] < 1e-14
    # 3-4-5 tangential vector has an independently known squared length.
    t2, u2, work = invariants([0, 1, 0], [3, 2, 4], [-3, 0, -4])
    assert (t2, u2, work) == (25, 25, -25)
    assert audit([0, 1, 0], [3, 2, 4], [-3, 0, -4], (F(5), F(5)), 1.0, 1.0)[0]["dissipation"] == 0
    for invalid in (np.nextafter(5.0, np.inf), 6.0):
        assert audit([0, 1, 0], [invalid, 1, 0], [-1, 0, 0], (F(5), F(5)), 1.0, 1.0)[0]["cone"] > 0
    with pytest.raises(ph.SolverFailure, match="negative"):
        audit([0, 1, 0], [0, -1, 0], [0, 0, 0], (F(-1), F(-1)), 1.0, 1.0)


def lifecycle(mode="numpy"):
    import phymeo as ph

    case = build(mode, mu=0.5, position=[[0.0, 1 / 8, 0.0]], velocity=[[1.0, -1.0, 0.0]])
    scene, state, inputs, program, settings, context = case
    outputs = []
    with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
        for force, event in (
            ([0, 0, 0], "enter"),
            ([0, 0, 0], "persist"),
            ([0, 1, 0], "exit"),
            ([0, 0, 0], "inactive"),
            ([0, -4, 0], "enter"),
        ):
            load = ph.SceneInputs(
                {**inputs.domains, "nodes": ph.StepInputs(np.array([force], float))}
            )
            result = runtime.step(state, load, 0.25)
            assert result.report.accepted, result.report.reason
            d = diagnostics(result)
            assert d["events"] == (event,)
            if event in ("exit", "inactive"):
                np.testing.assert_array_equal(
                    result.state.history["friction_contact/phase_world"], 0.0
                )
                np.testing.assert_array_equal(
                    result.state.history["friction_contact/initial_closed"], False
                )
            outputs.append(result)
            state = result.state
    assert not diagnostics(outputs[0])["initial_closed"][0]
    assert diagnostics(outputs[1])["initial_closed"][0]
    assert not diagnostics(outputs[-1])["initial_closed"][0]
    return outputs


def test_full_lifecycle_reentry():
    lifecycle()


def window(mode, mu, h, duration):
    import phymeo as ph
    from phymeo.core.scene import fingerprint

    outputs = []
    for _ in range(2):
        scene, state, inputs, program, settings, context = build(mode, mu=mu)
        original = fingerprint((inputs, scene))
        with ph.SceneStepper(scene, program, settings, context, program.registry) as runtime:
            for _ in range(round(duration / h)):
                result = runtime.step(state, inputs, h)
                assert result.report.accepted, result.report.reason
                state = result.state
        assert fingerprint((inputs, scene)) == original
        outputs.append(state)
    assert fingerprint(outputs[0]) == fingerprint(outputs[1])
    expected_v = F(0) if mu == 2.0 else F(3, 4) * F(duration)
    expected_x = F(0) if mu == 2.0 else F(3, 8) * (F(duration) ** 2 + F(duration) * F(h))
    # Independently accumulate the same discrete law without querying production.
    x, v = F(0), F(0)
    for _ in range(round(duration / h)):
        v += F(0) if mu == 2.0 else F(3, 4) * F(h)
        x += F(h) * v
    assert (x, v) == (expected_x, expected_v)
    tol = 1e-8 if mode == "numpy" else 2e-5
    np.testing.assert_allclose(
        state.domains["nodes"].position, [[float(x), 0, 0]], atol=tol, rtol=0
    )
    np.testing.assert_allclose(
        state.domains["nodes"].velocity, [[float(v), 0, 0]], atol=tol, rtol=0
    )
    return result, float(x - (F(0) if mu == 2.0 else F(3, 8) * F(duration) ** 2))


@pytest.mark.parametrize("mu", [2.0, 0.25])
def test_two_second_window_and_halving_error(mu):
    _, coarse = window("numpy", mu, 1 / 16, 2.0)
    _, fine = window("numpy", mu, 1 / 32, 2.0)
    assert coarse == 2 * fine
    if mu == 0.25:
        assert fine > 0


def test_shared_ancestor_two_nodes_full_normal_tangent_response():
    import phymeo as ph

    robot = ph.RobotModel(
        np.array([-1, 0]),
        ("prismatic",) * 2,
        np.zeros((2, 3)),
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([2.0, 3.0]),
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.zeros(3),
    )
    collider = ph.RobotPlaneCollider("floor", 1, np.zeros(3), np.array([0.0, 1.0, 0.0]))
    outputs = []
    for reverse in (False, True):
        points = np.array([[0.0, 1 / 8, 0.0], [1.0, 1 / 8, 0.0]])
        if reverse:
            points = points[::-1].copy()
        result = step(
            build(
                mu=0.5,
                robot=robot,
                colliders=(collider,),
                position=points,
                velocity=[[1.0, -1.0, 0.0]] * 2,
            )
        )
        d = diagnostics(result)
        np.testing.assert_allclose(d["held_normal"], [0.3, 0.3], atol=1e-8, rtol=0)
        np.testing.assert_allclose(d["terminal_normal"], [0.3, 0.3], atol=1e-8, rtol=0)
        np.testing.assert_allclose(d["terminal_tangent"], [[-0.3, 0, 0]] * 2, atol=1e-8, rtol=0)
        np.testing.assert_allclose(result.state.domains["robot"].v, [0.12, -0.4], atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            result.state.domains["nodes"].velocity, [[0.7, -0.4, 0]] * 2, atol=1e-8, rtol=0
        )
        outputs.append(
            result.state.domains["nodes"].position[::-1]
            if reverse
            else result.state.domains["nodes"].position
        )
    np.testing.assert_allclose(*outputs, atol=1e-8, rtol=0)


def test_unfixed_triangle_three_dimensional_sliding_and_material_response():
    from dataclasses import replace

    from test_terminal_impulse import fixture

    import phymeo as ph

    original = fixture()
    robot = replace(original[0].domains[0].model, joint_axis=np.array([[0.0, 0.0, 1.0]]))
    original[3].close()
    points = np.array([[0.0, 0.0, 1 / 8], [1.0, 0.0, 1 / 8], [0.0, 1.0, 1 / 8]])
    material = ph.DistanceConstraints(
        np.array([[0, 1], [1, 2], [2, 0]]), np.array([1.0, np.sqrt(2), 1.0]), np.full(3, 1 / 16)
    )
    collider = ph.RobotPlaneCollider("z", 0, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    result = step(
        build(
            mu=0.5,
            robot=robot,
            position=points,
            velocity=[[1.0, 2.0, -1.0]] * 3,
            material=material,
            colliders=(collider,),
        )
    )
    d = diagnostics(result)
    np.testing.assert_allclose(d["held_normal"], [0.2] * 3, atol=1e-8, rtol=0)
    np.testing.assert_allclose(d["terminal_normal"], [0.2] * 3, atol=1e-8, rtol=0)
    tau = np.array([-1.0, -2.0, 0.0]) * 0.2 / np.sqrt(5)
    np.testing.assert_allclose(d["terminal_tangent"], np.tile(tau, (3, 1)), atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity,
        np.tile([1 + tau[0], 2 + tau[1], -0.6], (3, 1)),
        atol=1e-8,
        rtol=0,
    )
    assert (
        abs(
            2 * result.state.domains["robot"].v[0]
            + sum(result.state.domains["nodes"].velocity[:, 2])
            + 3
        )
        < 1e-8
    )
    assert d["material_reports"]["nodes"]["max_residual"] <= 1e-8
    assert d["physical_budgets"]["material_angular_defect"]["l1_sum_nms"] <= 1e-6


def test_wrong_terminal_point_rejects_tangentially_moving_case(monkeypatch):
    import phymeo as ph
    from phymeo.solvers._dynamic_friction_workspace import _DynamicFrictionWorkspace as W

    def case():
        return build(mu=0.5, position=[[0.0, 1 / 8, 0.0]], velocity=[[1.0, -1.0, 2.0]])

    normal = step(case())
    requests = [
        r for r in diagnostics(normal)["terminal_evidence"]["requests"] if r.domain == "robot"
    ]
    np.testing.assert_allclose(requests[0].local_points, [[0.25, 0, 0.5]], atol=1e-12, rtol=0)
    evaluate = W._evaluate

    def wrong(self, terminal=False):
        if terminal:
            self.terminal_local = self.local
        return evaluate(self, terminal)

    monkeypatch.setattr(W, "_evaluate", wrong)
    c = case()
    with ph.SceneStepper(c[0], c[3], c[4], c[5], c[3].registry) as runtime:
        result = runtime.step(c[1], c[2], 0.25)
    assert not result.report.accepted


@pytest.mark.parametrize("fault", ["held", "terminal"])
def test_actual_world_mapping_corruption_is_rejected(monkeypatch, fault):
    import phymeo as ph
    from phymeo.solvers._dynamic_friction_workspace import _DynamicFrictionWorkspace as W

    world = W._world

    def corrupt(self, values):
        result = world(self, values)
        if np.any(values) and (
            (fault == "held" and values is self.p) or (fault == "terminal" and values is self.eta)
        ):
            result[0, 2] += 0.01
        return result

    monkeypatch.setattr(W, "_world", corrupt)
    c = build(mu=0.5, chain=True)
    with ph.SceneStepper(c[0], c[3], c[4], c[5], c[3].registry) as runtime:
        result = runtime.step(c[1], c[2], 0.25)
    assert not result.report.accepted


def test_zero_mu_matches_existing_normal_law():
    from test_dynamic_robot_normal_contact import build as normal_build
    from test_dynamic_robot_normal_contact import step as normal_step

    normal = normal_step(normal_build(chain=True))
    import phymeo as ph

    case = build(mu=0.0, chain=True)
    nodes = case[1].domains["nodes"]
    state = ph.SceneState(
        {
            **case[1].domains,
            "nodes": ph.ParticleState(nodes.position, np.tile([0.0, -1.0, 0.0], (2, 1))),
        },
        case[1].history,
    )
    actual = step((case[0], state, *case[2:]))
    for name, fields in (("robot", ("q", "v")), ("nodes", ("position", "velocity"))):
        for field in fields:
            np.testing.assert_allclose(
                getattr(actual.state.domains[name], field),
                getattr(normal.state.domains[name], field),
                atol=1e-8,
                rtol=0,
            )


def test_terminal_capacity_addition_encloses_exact_actual_world_sum():
    case = build(mu=1.0, position=[[0.0, 1 / 8, 0.0]], velocity=[[0.0, -1.0, 0.0]])
    w = case[3].create_workspace(case[0], case[1], case[2], 0.25, case[4], case[5])
    try:
        w.predict()
        w.p[:] = 1.0
        w.eta[:] = 2.0**-53
        w.active = np.array([True])
        _, caps, _, _ = w._friction_metrics(True)
        exact = F(1) + F(2.0**-53)
        assert F(caps[0, 0]) <= exact <= F(caps[0, 1])
        assert caps[0, 1] == np.nextafter(1.0, np.inf)
    finally:
        w.close()
        case[3].close()


def test_rotated_two_planes_with_pure_terminal_tangent():
    from dataclasses import replace

    import phymeo as ph

    # All rows start open; another plane's terminal normal opens the first,
    # which can still consume the capacity earned by its held normal.
    original = cross_case()
    rotation = np.array([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    robot = replace(
        original[3].model,
        joint_axis=original[3].model.joint_axis @ rotation.T,
        joint_position=original[3].model.joint_position @ rotation.T,
    )
    original[3].close()
    colliders = tuple(
        ph.RobotPlaneCollider(c.key, c.link_index, c.local_point, rotation @ c.local_normal)
        for c in original[3].colliders
    )
    c = build(
        mu={"floor": 0.5, "slant": 0.0},
        robot=robot,
        colliders=colliders,
        position=[rotation @ np.array([0.0, 1 / 16, 0.0])],
        velocity=[rotation @ np.array([21 / 10, -9 / 5, 0.0])],
    )
    state = ph.SceneState(
        {**c[1].domains, "robot": ph.RobotState(np.zeros(2), np.array([1.0, 6.0]))}, c[1].history
    )
    result = step((c[0], state, *c[2:]), 1.0)
    d = diagnostics(result)
    assert not d["initial_closed"].any()
    assert abs(d["terminal_normal"][0]) < 1e-8
    assert np.linalg.norm(d["terminal_tangent"][0]) > 0
    assert max(r["dissipation"] for r in d["terminal_friction_residuals"]) <= 1e-8


def test_multiple_selected_domains_shared_robot_and_domain_order():
    from test_terminal_impulse import fixture

    import phymeo as ph

    outputs = []
    for reverse in (False, True):
        original = fixture()
        robot = original[0].domains[0].model
        original[3].close()
        node = ph.ParticleModel(np.ones(1), np.zeros(3))
        bindings = (
            ph.DomainBinding("robot", "robot", robot),
            ph.DomainBinding("z_nodes", "particles", node),
            ph.DomainBinding("a_nodes", "particles", node),
        )
        if reverse:
            bindings = tuple(reversed(bindings))
        scene = ph.SceneModel(bindings)
        participants = {
            d.name: ph.RobotParticipant(robot)
            if d.family == "robot"
            else ph.ClothParticipant(node, None, record_material_path=True)
            for d in bindings
        }
        base = ph.ArticulatedCouplingProgram(
            scene,
            participants,
            ph.RobotSphereAttachments(np.empty(0, int), np.empty((0, 3)), np.empty(0)),
            (),
        )
        collider = ph.RobotPlaneCollider("floor", 0, np.zeros(3), np.array([0.0, 1.0, 0.0]))
        materials = dict(
            (name, None) for name in (("z_nodes", "a_nodes") if reverse else ("a_nodes", "z_nodes"))
        )
        program = ph.DynamicRobotFrictionContactProgram(
            scene,
            base,
            (collider,),
            friction_by_collider={"floor": 0.5},
            materials=materials,
            budgets=ph.MaterialContactBudgets(1e-6, 1e-6),
        )
        state = ph.SceneState(
            {
                d.name: ph.RobotState(np.zeros(1), np.zeros(1))
                if d.family == "robot"
                else ph.ParticleState(np.array([[0.0, 1 / 8, 0.0]]), np.array([[1.0, -1.0, 0.0]]))
                for d in bindings
            },
            program.initial_history(),
        )
        inputs = ph.SceneInputs(
            {
                d.name: ph.RobotInputs.zeros(1) if d.family == "robot" else ph.StepInputs.zeros(1)
                for d in bindings
            }
        )
        settings = ph.StepSettings(
            substeps=1, position_iterations=128, velocity_iterations=128, tolerance=1e-8
        )
        result = step((scene, state, inputs, program, settings, ph.BackendContext()))
        d = diagnostics(result)
        assert d["row_keys"] == (("a_nodes", 0, "floor"), ("z_nodes", 0, "floor"))
        np.testing.assert_allclose(d["held_normal"], [1 / 4, 1 / 4], atol=1e-8, rtol=0)
        np.testing.assert_allclose(d["terminal_normal"], [1 / 4, 1 / 4], atol=1e-8, rtol=0)
        np.testing.assert_allclose(result.state.domains["robot"].v, [-0.5], atol=1e-8, rtol=0)
        assert len(d["terminal_evidence"]["batches"]) == 3
        for name in ("a_nodes", "z_nodes"):
            np.testing.assert_allclose(
                result.state.domains[name].velocity, [[0.75, -0.5, 0]], atol=1e-8, rtol=0
            )
        outputs.append(result.state)
    from phymeo.core.scene import fingerprint

    assert fingerprint(outputs[0]) == fingerprint(outputs[1])


def test_original_input_endpoint_work_keeps_contact_load_separate():
    case = build(mu=0.25)
    result = step(case)
    d = diagnostics(result)
    assert (
        abs(d["physical_budgets"]["frozen_model_energy"]["original_input_work_j"] - 3 / 64) < 1e-14
    )
    assert abs(d["input_integrals"]["nodes"]["force"][0, 0] + 1 / 16) < 1e-14
    # Effective force .75 is observably different from original force 1.
    assert (
        abs(0.75 * 3 / 64 - d["physical_budgets"]["frozen_model_energy"]["original_input_work_j"])
        > 1e-3
    )


def test_material_angular_defects_do_not_cancel_between_domains():
    import phymeo as ph
    from phymeo.solvers._material_contact_balance import MaterialContactBalance
    from phymeo.solvers._robot_replay_ledger import FrozenRobotLedger

    c = build(position=[[1.0, 0.0, 0.0]], velocity=[[0.0, 0.0, 0.0]])
    model = {"robot": c[3].model, "a": c[0].domains[1].model, "b": c[0].domains[1].model}
    initial = ph.SceneState(
        {"robot": c[1].domains["robot"], "a": c[1].domains["nodes"], "b": c[1].domains["nodes"]}
    )
    inputs = ph.SceneInputs(
        {"robot": c[2].domains["robot"], "a": ph.StepInputs.zeros(1), "b": ph.StepInputs.zeros(1)}
    )
    final = ph.SceneState(
        {
            "robot": initial.domains["robot"],
            "a": ph.ParticleState(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])),
            "b": ph.ParticleState(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, -1.0, 0.0]])),
        }
    )
    balance = MaterialContactBalance(
        model, {"a": None, "b": None}, initial, inputs, "robot", 0.25, c[3].budgets
    )
    ledger = FrozenRobotLedger(
        c[3].model, initial.domains["robot"], inputs.domains["robot"], 0.25, c[4], 1.0
    )
    metrics, report, _ = balance.require_metrics(
        final, ledger, {"a": [[0, 0, 0]], "b": [[0, 0, 0]]}, c[4].tolerance
    )
    domain = report["material_angular_defect"]["by_domain_nms"]
    np.testing.assert_array_equal(domain["a"] + domain["b"], 0)
    assert report["material_angular_defect"]["l1_sum_nms"] == 2.0
    assert metrics["material_angular_defect"] > c[4].tolerance
    c[3].close()


@pytest.mark.parametrize("normal", [np.array([-0.8, 0.6, 0.0]), np.array([0.8, -0.6, 0.0])])
def test_world_representation_uses_nearest_nonnegative_boundary(normal):
    import phymeo.solvers._dynamic_friction_law as law

    assert hasattr(law, "nonnegative_world"), "missing conservative world representation"
    source = np.array([0.14653574510790052, 0.19538099347720062, 0.0])
    if normal[0] > 0:
        source = -source
    before = source.copy()
    assert sum(F(n) * F(s) for n, s in zip(normal, source, strict=True)) < 0
    actual = law.nonnegative_world(normal, source)
    assert sum(F(n) * F(s) for n, s in zip(normal, actual, strict=True)) >= 0
    axis = int(np.argmax(np.abs(normal)))
    prior = actual.copy()
    prior[axis] = np.nextafter(prior[axis], -np.inf if normal[axis] > 0 else np.inf)
    assert sum(F(n) * F(s) for n, s in zip(normal, prior, strict=True)) < 0
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(law.nonnegative_world(normal, normal), normal)
    np.testing.assert_array_equal(law.nonnegative_world(normal, np.zeros(3)), 0.0)


def test_world_rounding_does_not_hide_large_latent_error_or_overflow():
    import phymeo as ph
    from phymeo.solvers._dynamic_friction_law import nonnegative_world

    with pytest.raises(ph.SolverFailure, match="unrepresentable"):
        nonnegative_world(np.ones(3), np.full(3, -np.finfo(float).max))
    c = build()
    w = c[3].create_workspace(c[0], c[1], c[2], 0.25, c[4], c[5])
    try:
        w.t[0] = -w.normals[0]
        assert np.all(w._world(w.p) == 0.0)
        assert w._world_mapping(w.p) > c[4].tolerance
    finally:
        w.close()
        c[3].close()


def test_history_capacity_uses_exact_actual_world_sum(monkeypatch):
    import phymeo as ph
    import phymeo.solvers._dynamic_friction_workspace as module

    c = build(position=[[0.0, 0.0, 0.0]], velocity=[[0.0, 0.0, 0.0]], mu=1.0)
    history = c[3].initial_history()
    history["friction_contact/active"][:] = True
    history["friction_contact/phase_world"][0, :, 1] = [1.0, 2.0**-53]
    observed = []
    audit = module.audit

    def record(normal, world, velocity, capacity, kappa, speed):
        observed.append(capacity)
        return audit(normal, world, velocity, capacity, kappa, speed)

    monkeypatch.setattr(module, "audit", record)
    state = ph.SceneState(c[1].domains, history)
    w = c[3].create_workspace(c[0], state, c[2], 0.25, c[4], c[5])
    exact = F(1) + F(2.0**-53)
    assert observed[-1] == (exact, exact)
    w.close()
    c[3].close()


def test_face_order_and_representable_rigid_transform():
    from dataclasses import replace

    import phymeo as ph
    from phymeo.core.scene import fingerprint

    a = step(cross_case(), 1.0)
    c = cross_case()
    p = ph.DynamicRobotFrictionContactProgram(
        c[0],
        c[3].base,
        tuple(reversed(c[3].colliders)),
        friction_by_collider={"slant": 0.0, "floor": 0.5},
        materials=c[3].materials,
        budgets=c[3].budgets,
    )
    b = step((c[0], c[1], c[2], p, c[4], c[5]), 1.0)
    assert fingerprint(a.state) == fingerprint(b.state)
    original = build(chain=True, mu=0.5)
    rotation = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    shift = np.array([4.0, -2.0, 1.0])
    robot = replace(
        original[3].model,
        joint_axis=original[3].model.joint_axis @ rotation.T,
        joint_position=original[3].model.joint_position @ rotation.T + shift,
    )
    material = original[3].materials["nodes"]
    original[3].close()
    collider = ph.RobotPlaneCollider("floor", 0, np.zeros(3), rotation @ np.array([0.0, 1.0, 0.0]))
    transformed = build(
        mu=0.5,
        robot=robot,
        colliders=(collider,),
        material=material,
        position=np.array([[0.0, 1 / 8, 0.0], [0.0, 9 / 8, 0.0]]) @ rotation.T + shift,
        velocity=np.array([[1.0, -1.0, 0.0]] * 2) @ rotation.T,
    )
    result = step(transformed)
    np.testing.assert_allclose(
        result.state.domains["nodes"].position,
        np.array([[1 / 4, -3 / 56, 0], [1 / 4, 51 / 56, 0]]) @ rotation.T + shift,
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity,
        np.array([[13 / 21, -8 / 21, 0], [1, -6 / 7, 0]]) @ rotation.T,
        atol=1e-8,
        rtol=0,
    )
