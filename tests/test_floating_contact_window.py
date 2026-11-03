"""Persistent physical load, active separation and atomicity at published pose witnesses."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.solvers.floating_contact import step_floating_contacts

H = 1 / 64
Z = np.array([0.0, 0.0, 1.0])


def fixture(axial):
    R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) if axial else np.eye(3)
    quat = np.full(4, 0.5) if axial else np.array([0.0, 0.0, 0.0, 1.0])
    shift = np.array([1.0, -0.5, 0.25]) if axial else np.zeros(3)
    up = R @ Z
    root = ph.RigidBodyModel(np.array([0.5]), np.array([[0.5, 0.75, 1.0]]), -4 * up)
    child = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([[0.125, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([Z if axial else [1.0, 0.0, 0.0]]),
        np.ones(1),
        np.zeros((1, 3)),
        np.full((1, 3), 0.5),
        -4 * up,
    )
    model = ph.FloatingRobotModel(root, np.array([0.125, 0.0, 0.0]), child)
    state = ph.FloatingRobotState(np.r_[shift, quat, 0.0], np.zeros(7))
    support = ph.RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    center = shift + R @ np.array([0.125, 0.0, 0.0])
    base = ph.RigidBodyState(
        (center - 0.25 * up)[None, :], quat[None, :], np.zeros((1, 3)), np.zeros((1, 3))
    )
    # Changing owner keys also reverses the collision normal and the sign of its second-side impulse.
    foot, bed = [ph.CollisionKey(6, x, 0) for x in ((8, 3) if axial else (2, 9))]
    geom = ph.CollisionModel([ph.CollisionShape(k, "sphere", (0,), 1 / 8) for k in (foot, bed)])
    sites = {
        "tree": (np.array([1]), np.array([[0.0, 0.0, -0.125]])),
        "bed": (np.array([0]), np.array([[0.0, 0.0, 0.125]])),
    }
    return model, state, support, base, geom, sites, foot, bed, R, up


def solve(data, state, ws, *, effort=0.0, up_force=0.0, wrong_site=False):
    model, _, support, base, geom, sites, foot, bed, _, up = data
    center = ph.floating_kinematics(model, state).position[1]
    pair = ph.detect(
        geom,
        {foot: ph.CollisionMotion([center]), bed: ph.CollisionMotion(base.position)},
        ph.CollisionPolicy(margin=1),
        ws,
    ).pairs[0]
    assert pair.status in ("touching", "separated")
    touching = pair.status == "touching"
    rows = [(pair, {foot: ("tree", 0), bed: ("bed", 0)})] if touching else []
    if wrong_site:
        sites = {**sites, "tree": (np.array([1]), np.zeros((1, 3)))}
    force = np.zeros((2, 3))
    force[0] = up_force * up
    inputs = ph.FloatingRobotInputs(np.array([effort]), force, np.zeros((2, 3)))
    n = len(rows)
    result = step_floating_contacts(
        {"tree": (model, state, inputs)},
        sites,
        rows,
        np.zeros(n),
        np.zeros(n),
        H,
        fixed_bodies={"bed": (support, base)},
    )
    return result, pair


@pytest.mark.parametrize("axial", [False, True])
def test_32_loaded_steps_and_eight_actual_liftoff_steps(axial):
    data = fixture(axial)
    model, state, support, base, geom, sites, foot, bed, R, up = data
    initial = state
    momentum = 0.0
    touch = 0
    with ph.CollisionWorkspace(geom) as ws:
        for k in range(32):
            result, pair = solve(data, state, ws, effort=-8.0 if axial else 0.0)
            assert result.report.accepted, result.report.reason
            assert pair.status == "touching"
            impact = result.impulses
            assert impact.normal_impulses[0] == pytest.approx(12 * H, abs=3e-12)
            on_tree = impact.world_impulses[0] * (1 if pair.second == foot else -1)
            np.testing.assert_allclose(on_tree / H, 12 * up, atol=4e-11)
            np.testing.assert_allclose(result.predicted["tree"].q, state.q, atol=0)
            expected = np.r_[np.zeros(6), -12 * H] if axial else np.r_[-4 * H * up, np.zeros(4)]
            np.testing.assert_allclose(result.predicted["tree"].v, expected, atol=2e-12)
            state = result.states["tree"]
            np.testing.assert_allclose(state.q, initial.q, rtol=0, atol=3e-13)
            np.testing.assert_allclose(state.v, 0.0, rtol=0, atol=3e-13)
            np.testing.assert_array_equal(impact.states["bed"].position, base.position)
            momentum += float(on_tree @ up)
        assert momentum == pytest.approx(6.0, abs=2e-11)
        for k in range(1, 9):
            result, pair = solve(data, state, ws, effort=6.0 if axial else 0.0, up_force=18.0)
            assert result.report.accepted, result.report.reason
            touch += int(pair.status == "touching")
            assert np.sum(result.impulses.normal_impulses) == pytest.approx(0, abs=2e-12)
            state = result.states["tree"]
            obs = ph.floating_kinematics(model, state)
            np.testing.assert_allclose(state.v[:3], 2 * k * H * up, rtol=0, atol=2e-12)
            np.testing.assert_allclose(
                obs.com_velocity, np.tile(2 * k * H * up, (2, 1)), atol=2e-12
            )
            np.testing.assert_allclose(
                state.q[:3] - initial.q[:3], k * (k + 1) * H**2 * up, rtol=0, atol=3e-12
            )
    assert touch == 1 and abs(state.q[-1]) < 3e-12
    assert 3 * (state.v[:3] @ up) == pytest.approx((18.0 - 12.0) * 8 * H, abs=3e-12)
    print(
        f"axial={axial} support_steps=32 liftoff_steps=8 impulse_Ns={momentum} speed_mps={state.v[:3] @ up} displacement_m={(state.q[:3] - initial.q[:3]) @ up}"
    )


def test_axial_child_can_stay_in_contact_while_root_and_joint_actually_travel():
    data = fixture(True)
    model, state, _, _, geom, _, _, _, _, up = data
    with ph.CollisionWorkspace(geom) as work:
        result, _ = solve(data, state, work, effort=-10.0)
        assert result.report.accepted
        assert result.impulses.normal_impulses[0] == pytest.approx(14 * H)
        after = result.states["tree"]
        np.testing.assert_allclose(after.v[:3], H * up, atol=2e-13)
        assert after.v[-1] == pytest.approx(-H) and after.q[-1] == pytest.approx(-(H**2))
        np.testing.assert_allclose(
            ph.floating_kinematics(model, after).position[1],
            ph.floating_kinematics(model, state).position[1],
            atol=2e-13,
        )
        with pytest.raises(ValueError, match="bound|witness"):
            solve(data, state, work, effort=-10.0, wrong_site=True)
        retry, _ = solve(data, state, work, effort=-10.0)
        np.testing.assert_array_equal(retry.states["tree"].q, after.q)


def test_two_moving_systems_resolve_real_incoming_velocity_then_drift_once():
    g = np.zeros(3)
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ra = ph.FloatingRobotModel(ph.RigidBodyModel(np.array([0.5]), np.ones((1, 3)), g), g)
    rb = ph.FloatingRobotModel(ph.RigidBodyModel(np.ones(1), np.ones((1, 3)), g), g)
    a = ph.FloatingRobotState(q0, np.array([1.0, 0, 0, 0, 0, 0]))
    q1 = q0.copy()
    q1[0] = 0.25
    b = ph.FloatingRobotState(q1, np.zeros(6))
    ka, kb = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    env = ph.CollisionModel([ph.CollisionShape(k, "sphere", (0,), 0.125) for k in (ka, kb)])
    with ph.CollisionWorkspace(env) as ws:
        (pair,) = ph.detect(
            env,
            {ka: ph.CollisionMotion([a.q[:3]]), kb: ph.CollisionMotion([b.q[:3]])},
            ph.CollisionPolicy(),
            ws,
        ).pairs
    systems = {
        "a": (ra, a, ph.FloatingRobotInputs.zeros(0)),
        "b": (rb, b, ph.FloatingRobotInputs.zeros(0)),
    }
    sites = {"a": ([0], np.array([[0.125, 0, 0]])), "b": ([0], np.array([[-0.125, 0, 0]]))}
    result = step_floating_contacts(
        systems, sites, [(pair, {ka: ("a", 0), kb: ("b", 0)})], [0.0], [0.0], H
    )
    assert result.report.accepted
    assert result.impulses.normal_impulses[0] == pytest.approx(2 / 3)
    assert result.impulses.energy_change == pytest.approx(-1 / 3)
    assert result.report.max_residual == pytest.approx(result.impulses.residual_mps / 1e-10)
    for name, original in (("a", a), ("b", b)):
        assert result.states[name].v[0] == pytest.approx(2 / 3)
        assert result.states[name].q[0] - original.q[0] == pytest.approx(H * 2 / 3)
    assert a.v[0] == 1 and b.v[0] == 0


def test_two_finite_supports_balance_a_real_external_com_torque_with_unequal_loads():
    gravity = np.array([0.0, 0.0, -4.0])
    model = ph.FloatingRobotModel(
        ph.RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.5), gravity),
        np.array([0.125, 0.0, 0.0]),
    )
    state = ph.FloatingRobotState(np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]), np.zeros(6))
    local = np.array([[-0.375, 0.0, 0.0], [0.625, 0.0, 0.0]])
    centers = state.q[:3] + local
    bed = ph.RigidBodyModel(np.zeros(2), np.zeros((2, 3)), np.zeros(3))
    base = ph.RigidBodyState(
        centers - [0, 0, 0.25],
        np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
    )
    body_keys = [ph.CollisionKey(2, 1, i) for i in range(2)]
    bed_keys = [ph.CollisionKey(2, 2 + i, 0) for i in range(2)]
    registered = ph.CollisionModel(
        [ph.CollisionShape(k, "sphere", (0,), 0.125) for k in body_keys + bed_keys]
    )
    mapping = {
        **{k: ("rig", i) for i, k in enumerate(body_keys)},
        **{k: ("bed", i) for i, k in enumerate(bed_keys)},
    }
    verts = {k: [point] for k, point in zip(body_keys, local, strict=True)}
    paths = {
        k: ph.CollisionMotion([p])
        for k, p in zip(body_keys + bed_keys, np.vstack((centers, base.position)), strict=True)
    }
    with ph.CollisionWorkspace(registered) as ws:
        pairs = [
            p
            for p in ph.detect(registered, paths, ph.CollisionPolicy(), ws).pairs
            if p.status == "touching"
        ]
    assert len(pairs) == 2
    rows = [(p, {k: mapping[k] for k in (p.first, p.second)}) for p in pairs]
    sites = {
        "rig": ([0, 0], local - [0, 0, 0.125]),
        "bed": ([0, 1], np.tile([0.0, 0.0, 0.125], (2, 1))),
    }
    motor = ph.FloatingRobotInputs(np.empty(0), np.zeros((1, 3)), np.array([[0.0, 1.0, 0.0]]))
    result = step_floating_contacts(
        {"rig": (model, state, motor)},
        sites,
        rows,
        [0.0, 0.0],
        [0.0, 0.0],
        H,
        fixed_bodies={"bed": (bed, base)},
        local_vertices=verts,
    )
    assert result.report.accepted
    reactions = {
        p.first: normal / H
        for p, normal in zip(pairs, result.impulses.normal_impulses, strict=True)
    }
    np.testing.assert_allclose([reactions[k] for k in body_keys], [3.0, 5.0], atol=3e-11)
    np.testing.assert_allclose(result.predicted["rig"].v, [0, 0, -4 * H, 0, 2 * H, 0], atol=3e-13)
    np.testing.assert_allclose(result.states["rig"].v, 0.0, atol=3e-13)
    np.testing.assert_allclose(result.states["rig"].q, state.q, atol=3e-13)
    assert 0.5 * (reactions[body_keys[0]] - reactions[body_keys[1]]) == pytest.approx(-1.0)


def test_empty_graph_keeps_eccentric_com_root_inertia_and_real_incoming_spin():
    gravity = np.array([0.0, -2.0, 0.0])
    com = np.array([0.125, -0.0625, 0.25])
    model = ph.FloatingRobotModel(
        ph.RigidBodyModel(np.array([0.5]), np.array([[0.5, 0.75, 1.0]]), gravity), com
    )
    state = ph.FloatingRobotState(
        np.array([1.0, -0.5, 0.25, 0.5, 0.5, 0.5, 0.5]),
        np.array([0.25, -0.125, 0.5, 1.0, -0.5, 0.25]),
    )
    inputs = ph.FloatingRobotInputs(
        np.empty(0), np.array([[2.0, 1.0, -0.5]]), np.array([[0.5, 1.0, -2.0]])
    )
    solo = ph.step_floating_robot(model, state, inputs, H)
    result = step_floating_contacts({"root": (model, state, inputs)}, {}, [], [], [], H)
    assert (
        solo.report.accepted and result.report.accepted and not len(result.impulses.normal_impulses)
    )
    new = result.states["root"]
    np.testing.assert_allclose(new.q, solo.state.q, rtol=0, atol=4e-14)
    np.testing.assert_allclose(new.v, solo.state.v, rtol=0, atol=4e-14)
    a = ph.floating_kinematics(model, state).com_position[0]
    b = ph.floating_kinematics(model, new).com_position[0]
    np.testing.assert_allclose(b - a, H * new.v[:3], rtol=0, atol=4e-14)
    assert np.linalg.norm(new.q[:3] - state.q[:3] - H * new.v[:3]) > 1e-4


def test_late_domain_drift_failure_leaves_every_domain_unpublished_and_retriable():
    model = ph.FloatingRobotModel(
        ph.RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3)), np.zeros(3)
    )
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    good = ph.FloatingRobotState(q, v)
    huge = np.finfo(float).max * 0.75
    q[0] = huge
    v[0] = huge
    last = ph.FloatingRobotState(q, v)
    systems = {
        k: (model, s, ph.FloatingRobotInputs.zeros(0)) for k, s in (("first", good), ("last", last))
    }
    failed = step_floating_contacts(systems, {}, [], [], [], 1.0)
    assert not failed.report.accepted and failed.report.reason
    assert failed.states["first"] is good and failed.states["last"] is last
    assert failed.impulses is None and not failed.predicted and good.q[0] == 0
    retry = step_floating_contacts(systems, {}, [], [], [], 0.125)
    assert retry.report.accepted and retry.states["first"].q[0] == pytest.approx(0.125)
    assert retry.states["last"].q[0] > huge


def test_dynamic_or_nonstationary_support_cannot_absorb_reaction_as_fixed():
    model, state, bed, base, *_ = fixture(False)
    sys = {"tree": (model, state, ph.FloatingRobotInputs.zeros(1))}
    dynamic = ph.RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="fixed support"):
        step_floating_contacts(sys, {}, [], [], [], H, fixed_bodies={"bed": (dynamic, base)})
    moving = ph.RigidBodyState(base.position, base.orientation, np.ones((1, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="fixed support"):
        step_floating_contacts(sys, {}, [], [], [], H, fixed_bodies={"bed": (bed, moving)})
