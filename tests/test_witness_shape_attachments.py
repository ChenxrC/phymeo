"""Original material coordinates and actual impulse arm of a capsule touching a ball."""

from dataclasses import fields, replace

import numpy as np
import pytest

import phymeo as ph


def _detect(model, capsule_vertices, ball_center):
    capsule, ball = model.shapes
    paths = {
        capsule.key: ph.CollisionMotion(capsule_vertices),
        ball.key: ph.CollisionMotion([ball_center]),
    }
    with ph.CollisionWorkspace(model, "host") as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
    assert row.candidate and row.status == "touching" and row.witness.gap_m == 0
    return row


def _scene(rotated=False):
    capsule = ph.CollisionShape(ph.CollisionKey(4, 1, 0), "capsule", (11, 23), radius=0.25)
    ball = ph.CollisionShape(ph.CollisionKey(4, 2, 0), "sphere", (37,), radius=0.25)
    model = ph.CollisionModel((capsule, ball))
    R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) if rotated else np.eye(3)
    origin = np.array([2.0, -3.0, 0.5]) if rotated else np.zeros(3)
    q = np.array([0.5] * 4) if rotated else np.array([0.0, 0.0, 0.0, 1.0])
    local = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vertices = origin + local @ R.T
    center = origin + R @ [0.5, 0.5, 0.0]
    world_point = origin + R @ [0.5, 0.25, 0.0]
    up = R @ [0.0, 1.0, 0.0]
    row = _detect(model, vertices, center)
    np.testing.assert_array_equal(row.witness.first_point, world_point)
    np.testing.assert_array_equal(row.witness.second_point, world_point)
    np.testing.assert_array_equal(row.witness.normal, up)
    assert row.witness.first_weights == ((11, 0.75), (23, 0.25))
    assert row.witness.second_weights == ((37, 1.0),)

    rigid = ph.RigidBodyModel(np.ones(1), np.ones((1, 3)), np.zeros(3))
    tree = ph.FloatingRobotModel(rigid, np.zeros(3))
    cap_state = ph.FloatingRobotState(np.r_[origin, q], np.r_[up, np.zeros(3)])
    ball_state = ph.RigidBodyState(center[None, :], q[None, :], -up[None, :], np.zeros((1, 3)))
    ports = {
        "capsule": ph.FloatingRobotPointPort(
            tree,
            cap_state,
            np.array([0, 0]),
            np.array([[0.5, 0.25, 0.0], [0.5, 0.125, 0.0]]),
            domain="capsule",
        ),
        "ball": ph.RigidPointPort(
            rigid, ball_state, np.array([0]), np.array([[0.0, -0.25, 0.0]]), domain="ball"
        ),
    }
    np.testing.assert_array_equal(ports["capsule"].positions()[0], world_point)
    np.testing.assert_array_equal(ports["ball"].positions()[0], world_point)
    return model, row, ports, {"capsule": cap_state, "ball": ball_state}, local, R


def _solve(row, ports, states, vertices, capsule_row=0):
    before = {
        key: {f.name: getattr(state, f.name).copy() for f in fields(state)}
        for key, state in states.items()
    }
    local_before = vertices.copy()
    try:
        return ph.solve_witness_impulses(
            ports,
            [(row, {row.first: ("capsule", capsule_row), row.second: ("ball", 0)})],
            [0.0],
            [0.0],
            local_vertices={row.first: vertices},
            position_tolerance=0.0,
        )
    finally:
        np.testing.assert_array_equal(vertices, local_before)
        for domain, expected in before.items():
            saved = ports[domain].snapshot()
            for name, value in expected.items():
                np.testing.assert_array_equal(getattr(states[domain], name), value)
                np.testing.assert_array_equal(getattr(saved, name), value)


@pytest.mark.parametrize("rotated", [False, True])
def test_off_center_capsule_shaft_uses_the_registered_axis_material(rotated):
    _, row, ports, states, local, R = _scene(rotated)
    result = _solve(row, ports, states, local)
    # Two unit masses and cap lever x=1/2 yield normal response 2+1/4.
    cap = result.states["capsule"]
    ball = result.states["ball"]
    np.testing.assert_allclose(result.world_impulses[0], R @ [0.0, 8 / 9, 0.0], atol=2e-12)
    np.testing.assert_allclose(cap.v[:3], R @ [0.0, 1 / 9, 0.0], atol=2e-12)
    np.testing.assert_allclose(cap.v[3:], R @ [0.0, 0.0, -4 / 9], atol=2e-12)
    np.testing.assert_allclose(ball.velocity[0], R @ [0.0, -1 / 9, 0.0], atol=2e-12)
    np.testing.assert_allclose(ball.angular_velocity[0], 0.0, atol=2e-12)


def test_reversed_capsule_material_vertices_cannot_bind_the_same_geometric_surface():
    model, original, ports, states, local, _ = _scene()
    swapped = local[::-1].copy()
    live = _detect(model, swapped, states["ball"].position[0])
    np.testing.assert_array_equal(live.witness.normal, original.witness.normal)
    np.testing.assert_array_equal(live.witness.first_point, original.witness.first_point)
    np.testing.assert_array_equal(live.witness.second_point, original.witness.second_point)
    assert live.witness.first_weights == ((11, 0.25), (23, 0.75))
    try:
        result = _solve(original, ports, states, swapped)
    except (ValueError, ph.SolverFailure):
        return
    pytest.fail(
        f"accepted material {original.witness.first_weights} with actual {live.witness.first_weights}; "
        f"same surface returned impulse {result.world_impulses[0]}"
    )


def test_other_row_on_the_same_capsule_cannot_receive_the_witness_impulse():
    _, row, ports, states, local, _ = _scene()
    assert np.linalg.norm(ports["capsule"].positions()[1] - row.witness.first_point) == 0.125
    with pytest.raises((ValueError, ph.SolverFailure), match="material|witness"):
        _solve(row, ports, states, local, capsule_row=1)


@pytest.mark.parametrize(
    "weights",
    [
        ((11, float("nan")), (23, 0.25)),
        ((11, 0.875), (11, -0.125), (23, 0.25)),
    ],
)
def test_invalid_material_coordinates_cannot_authenticate_a_real_contact(weights):
    _, pair, ports, states, local, _ = _scene()
    invalid = replace(pair, witness=replace(pair.witness, first_weights=weights))
    with pytest.raises(ValueError, match="material"):
        _solve(invalid, ports, states, local)
