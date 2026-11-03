"""A live two-body arrival at a floating sliding tree, solved by the common contact graph."""

import json

import numpy as np
import pytest

import phymeo as ph


def build(rotated=False, reverse=False, tangent=0.0):
    ex = np.eye(3)
    frame = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0.0]]) if rotated else ex
    q = np.array([0.5, 0.5, 0.5, 0.5]) if rotated else np.array([0.0, 0, 0, 1])
    shift = np.array([2.0, -1.0, 0.5]) if rotated else np.zeros(3)
    child = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([[1.0, 0, 0]]),
        np.array([[0.0, 0, 0, 1.0]]),
        np.array([[1.0, 0, 0]]),
        np.ones(1),
        np.zeros((1, 3)),
        np.full((1, 3), 0.5),
        np.zeros(3),
    )
    root = ph.RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.5), np.zeros(3))
    model = ph.FloatingRobotModel(root, np.zeros(3), child)
    robot = ph.FloatingRobotState(np.r_[shift, q, 0.0], np.zeros(7))
    other = ph.RigidBodyModel(np.ones(1), np.full((1, 3), 0.5), np.zeros(3))
    h = 1 / 64
    velocity = frame @ np.array([0.0, -1, tangent])
    at_touch = shift + frame @ np.array([1.0, 0.125, 0])
    before = ph.RigidBodyState(
        np.array([at_touch - h * velocity]), q[None], velocity[None], np.zeros((1, 3))
    )
    robot_hit = ph.step_floating_robot(model, robot, ph.FloatingRobotInputs.zeros(1), h)
    ball_hit = ph.RigidSemiImplicitSolver().step(other, before, ph.RigidBodyInputs.zeros(1), h)
    assert robot_hit.report.accepted and ball_hit.report.accepted
    rk, bk = [ph.CollisionKey(9, k, 0) for k in ((9, 2) if reverse else (2, 9))]
    shapes = [ph.CollisionShape(key, "sphere", (0,), 1 / 8) for key in (rk, bk)]
    geom = ph.CollisionModel(shapes)
    center = shift + frame @ np.array([1.0, -0.125, 0])
    paths = {
        rk: ph.CollisionMotion([center]),
        bk: ph.CollisionMotion(before.position, ball_hit.state.position),
    }
    with ph.CollisionWorkspace(geom) as work:
        (hit,) = ph.detect(geom, paths, ph.CollisionPolicy(), work).pairs
    assert hit.status == "hit" and hit.sweep.first_t == (1.0, 1.0)
    ports = {
        "arm": ph.FloatingRobotPointPort(
            model, robot_hit.state, np.array([1]), np.zeros((1, 3)), domain="arm"
        ),
        "ball": ph.RigidPointPort(
            other, ball_hit.state, np.array([0]), np.array([[0.0, -0.125, 0.0]]), domain="ball"
        ),
    }
    bindings = {rk: ("arm", 0), bk: ("ball", 0)}
    return frame, shift, model, robot_hit.state, other, ball_hit.state, ports, hit, bindings, rk, bk


@pytest.mark.parametrize(
    "rotated,reverse", [(False, False), (False, True), (True, False), (True, True)]
)
def test_real_swept_hit_applies_full_floating_recoil(rotated, reverse, tmp_path):
    R, shift, model, start, other, body, ports, hit, bind, rk, bk = build(rotated, reverse)
    # Total tree COM is at local x=1/3; Izz about it is 5/3. Its child site responds
    # in y by 1/3 + (2/3)^2/(5/3)=3/5; the external free body contributes one.
    result = ph.solve_witness_impulses(
        ports,
        [(hit, bind)],
        [0.0],
        [0.0],
        sample="first_contact",
        local_vertices={rk: ((0.0, -0.125, 0.0),)},
    )
    assert result.normal_impulses[0] == pytest.approx(5 / 8, abs=2e-10)
    arm, ball = result.states["arm"], result.states["ball"]
    np.testing.assert_array_equal(arm.q, start.q)
    np.testing.assert_array_equal(ball.position, body.position)
    np.testing.assert_allclose(arm.v[:3], R @ [0, -1 / 8, 0.0], atol=2e-10)
    np.testing.assert_allclose(arm.v[3:6], R @ [0, 0, -1 / 4], atol=2e-10)
    assert abs(arm.v[-1]) < 2e-10
    obs = ph.floating_kinematics(model, arm)
    np.testing.assert_allclose(obs.com_velocity[1], R @ [0, -3 / 8, 0.0], atol=2e-10)
    np.testing.assert_allclose(ball.velocity[0], obs.com_velocity[1], atol=2e-10)
    P = 2 * obs.com_velocity[0] + obs.com_velocity[1] + ball.velocity[0]
    L = sum(
        np.cross(x - shift, m * v) + 0.5 * spin
        for x, m, v, spin in zip(
            obs.com_position, (2, 1), obs.com_velocity, obs.angular_velocity, strict=True
        )
    )
    L += np.cross(ball.position[0] - shift, ball.velocity[0]) + 0.5 * ball.angular_velocity[0]
    np.testing.assert_allclose(P, R @ [0, -1, 0.0], atol=2e-10)
    np.testing.assert_allclose(L, R @ [0, 0, -1.0], atol=2e-10)
    assert result.energy_change == pytest.approx(-5 / 16, abs=2e-10)
    assert ph.step_floating_robot(
        model, arm, ph.FloatingRobotInputs.zeros(1), 1 / 256
    ).report.accepted
    assert (
        ph.RigidSemiImplicitSolver()
        .step(other, ball, ph.RigidBodyInputs.zeros(1), 1 / 256)
        .report.accepted
    )
    (tmp_path / "floating-hit.json").write_text(
        json.dumps(
            dict(
                r=rotated,
                reverse=reverse,
                normal=result.normal_impulses.tolist(),
                impulse_on_second_ns=result.world_impulses.tolist(),
                temporal_bracket=hit.sweep.first_t,
                robot_velocity=arm.v.tolist(),
                external_velocity=ball.velocity.tolist(),
                momentum=P.tolist(),
                angular=L.tolist(),
                kinetic_change_j=result.energy_change,
                residual_m_s=result.residual_mps,
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )


def test_floating_points_keep_cross_terms_repeated_addresses_and_actual_candidate_audit():
    _, _, model, state, _, _, _, _, _, _, _ = build()
    indices = np.array([1, 0, 1])
    port = ph.FloatingRobotPointPort(model, state, indices, np.zeros((3, 3)), domain="arm")
    p = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    delta = port.response(p)
    # A +1 Ns impulse at the child gives the root +1/5 m/s and child +3/5 m/s.
    np.testing.assert_allclose(delta[:, 1], [3 / 5, 1 / 5, 3 / 5], atol=2e-10)
    after = port.apply_impulses(p)
    audit = port.candidate_audit(after)
    np.testing.assert_allclose(audit.candidate_velocities, delta, atol=2e-10)
    assert audit.energy_change == pytest.approx(0.3)
    np.testing.assert_allclose(port.apply_impulses(p[[2, 1, 0]]).v, after.v, atol=2e-10)
    np.testing.assert_array_equal(port.snapshot().v, np.zeros(7))


def test_coulomb_response_from_real_skew_arrival_and_bad_port_binding():
    R, _, model, state, _, _, ports, hit, bind, rk, bk = build(tangent=0.5)
    result = ph.solve_witness_impulses(
        ports,
        [(hit, bind)],
        [0.125],
        [0.125],
        sample="first_contact",
        local_vertices={rk: ((0.0, -0.125, 0.0),)},
    )
    assert result.normal_impulses[0] == pytest.approx(5 / 8, abs=2e-9)
    assert np.linalg.norm(result.tangent_impulses[0]) == pytest.approx(5 / 64, abs=2e-9)
    assert result.energy_change < 0 and result.residual_mps < 2e-9
    wrong = ph.FloatingRobotPointPort(model, state, np.array([0]), np.zeros((1, 3)), domain="arm")
    with pytest.raises(ValueError, match="witness"):
        ph.solve_witness_impulses(
            ports | {"arm": wrong}, [(hit, bind)], [0.0], [0.0], sample="first_contact"
        )
    with pytest.raises(ValueError):
        ph.solve_witness_impulses(
            ports, [(hit, {rk: bind[rk]})], [0.0], [0.0], sample="first_contact"
        )
