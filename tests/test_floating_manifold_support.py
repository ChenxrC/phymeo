"""A single point impulse resolves one end of a cylinder's existing line contact."""

import numpy as np
import pytest

import phymeo as ph


def test_floating_step_refuses_to_drift_a_cylinder_on_one_end_of_its_contact_line():
    plane, cylinder = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    up = np.array([0.0, 1.0, 0.0])
    center = np.array([0.0, 0.5, 0.0])
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    geometry = ph.CollisionModel(
        [
            ph.CollisionShape(plane, "plane", (11,), normal=tuple(up)),
            ph.CollisionShape(
                cylinder, "cylinder", (27,), curved=ph.CurvedSolid("cylinder", (0.5, 1.0))
            ),
        ]
    )
    paths = {plane: ph.CollisionMotion([np.zeros(3)]), cylinder: ph.CollisionMotion([center])}
    with ph.CollisionWorkspace(geometry) as work:
        (pair,) = ph.detect(geometry, paths, ph.CollisionPolicy(), work).pairs
    endpoints = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    assert pair.status == "touching" and pair.curved.locus == "segment"
    np.testing.assert_array_equal(pair.curved.points_m, endpoints)
    np.testing.assert_array_equal(pair.witness.second_point, endpoints[0])

    # Homogeneous 1 kg cylinder, radius 1/2 m, full length 2 m.
    rigid = ph.RigidBodyModel(np.ones(1), np.array([[19 / 48, 19 / 48, 1 / 8]]), np.zeros(3))
    model = ph.FloatingRobotModel(rigid, np.zeros(3))
    state = ph.FloatingRobotState(np.r_[center, quat], np.r_[-up, np.zeros(3)])
    bed = ph.RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    ground = ph.RigidBodyState(np.zeros((1, 3)), quat[None, :], np.zeros((1, 3)), np.zeros((1, 3)))
    original_q, original_v, original_ground = state.q.copy(), state.v.copy(), ground.copy()
    indices = np.zeros(2, dtype=np.int64)
    sites = {"cylinder": (indices, endpoints - center), "plane": (indices, endpoints)}
    ports = {
        "cylinder": ph.FloatingRobotPointPort(model, state, *sites["cylinder"], domain="cylinder"),
        "plane": ph.RigidPointPort(bed, ground, *sites["plane"], domain="plane"),
    }
    rows = [(pair, {cylinder: ("cylinder", 0), plane: ("plane", 0)})]
    normals = {plane: up}
    single = ph.solve_witness_impulses(ports, rows, [0.0], [0.0], local_normals=normals)
    # Inverse effective mass at the -z endpoint is 1 + 48/19. The two ends see opposite spins.
    assert single.normal_impulses[0] == pytest.approx(19 / 67, abs=2e-12)
    observed = ph.FloatingRobotPointPort(model, single.states["cylinder"], *sites["cylinder"])
    np.testing.assert_allclose(observed.velocities() @ up, [0.0, -96 / 67], atol=2e-12, rtol=0)

    inputs = ph.FloatingRobotInputs(np.empty(0), np.zeros((1, 3)), np.zeros((1, 3)))
    try:
        result = ph.step_floating_contacts(
            {"cylinder": (model, state, inputs)},
            sites,
            rows,
            [0.0],
            [0.0],
            1 / 64,
            fixed_bodies={"plane": (bed, ground)},
            local_normals=normals,
        )
    except ValueError as error:
        reason = str(error)
    else:
        assert not result.report.accepted, "the +z endpoint is still moving into the plane"
        np.testing.assert_array_equal(result.states["cylinder"].q, original_q)
        np.testing.assert_array_equal(result.states["cylinder"].v, original_v)
        assert not result.predicted and result.impulses is None
        reason = result.report.reason
    expected = ("manifold", "segment", "locus", "representative")
    assert any(token in reason.lower() for token in expected), reason
    np.testing.assert_array_equal(state.q, original_q)
    np.testing.assert_array_equal(state.v, original_v)
    for field in ("position", "orientation", "velocity", "angular_velocity"):
        np.testing.assert_array_equal(getattr(ground, field), getattr(original_ground, field))
