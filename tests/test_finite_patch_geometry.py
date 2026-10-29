"""Finite owned patches and rigorous curved-FK certificates."""

import math
from dataclasses import replace

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.scene import fingerprint

SQUARE = np.array([[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
TRIANGLE = np.array([[0.0, 0.0, -1.0], [3.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def model(rotor=False):
    return ph.RobotModel(
        np.array([-1]),
        ("revolute" if rotor else "prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]] if rotor else [[0.0, 1.0, 0.0]]),
        np.array([1.0 if rotor else 2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )


def patch(vertices=SQUARE):
    assert hasattr(ph, "RobotFinitePatchCollider"), "missing finite patch API"
    return ph.RobotFinitePatchCollider("patch", 0, vertices, radius_m=0.125)


def test_patch_schema_and_canonical_identity():
    vertices = SQUARE.copy()
    c = patch(vertices)
    sig = fingerprint(c)
    vertices[:] = 9
    assert fingerprint(c) == sig
    assert not c.local_vertices.flags.writeable
    for v in (SQUARE[::-1], np.roll(SQUARE, 2, axis=0)):
        assert fingerprint(patch(v)) == sig
    for v in (
        SQUARE[[0, 2, 1, 3]],
        np.vstack((SQUARE, SQUARE[0])),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.1], [0.0, 0.0, 1.0]]),
        SQUARE + np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ):
        with pytest.raises((ValueError, TypeError)):
            patch(v)
    for radius in (0.0, -1.0, True, 1, float("nan")):
        with pytest.raises((ValueError, TypeError)):
            ph.RobotFinitePatchCollider("patch", 0, SQUARE, radius_m=radius)


def test_edge_vertex_distance_bounds():
    from phymeo.geometry.robot_patch_sweep import patch_geometry

    c = patch()
    edge = patch_geometry(model(), np.zeros(1), c, np.array([1.25, 0.0, 0.0]))
    vertex = patch_geometry(model(), np.zeros(1), c, np.array([1.25, 0.0, 1.25]))
    assert edge.distance_bounds == (0.25, 0.25)
    assert vertex.distance_bounds[0] ** 2 <= 0.125 <= vertex.distance_bounds[1] ** 2
    assert edge.feature[0] == 2 and vertex.feature[0] == 3
    np.testing.assert_allclose(edge.normal, [1.0, 0.0, 0.0])


def test_rotating_curve_not_endpoint_chord():
    c = patch(TRIANGLE)
    alpha = 0.25
    x = 1.5 * np.array([-math.sin(alpha), math.cos(alpha), 0.0])
    result = ph.robot_patch_sweep(
        model(True), np.array([alpha]), np.array([alpha + math.pi]), c, x, x, clearance_m=0.125
    )
    assert result.status == "violation"
    assert result.witness_time == 0.5 and result.upper_distance_m < 0.125
    assert result.input_fingerprint
    assert result.segments[0].start == 0 and result.segments[-1].end == 1


def test_axial_equality_contact_certifies():
    c = patch(np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]))
    x = np.array([0.0, 0.0, 0.125])
    result = ph.robot_patch_sweep(
        model(True), np.array([0.25]), np.array([0.75]), c, x, x, clearance_m=0.125
    )
    assert result.status == "certified"
    assert result.checked_intervals == 1
    assert result.segments[0].distance_lower_m == 0.125


def test_budget_exhaustion_is_unknown():
    c = patch(TRIANGLE)
    x = np.array([1.0, 1.0, 0.0])
    result = ph.robot_patch_sweep(
        model(True), np.array([-0.5]), np.array([0.5]), c, x, x, clearance_m=0.3, max_intervals=1
    )
    assert result.status == "unresolved"


def test_patch_keyword_radius_and_additional_schema_controls():
    with pytest.raises(TypeError):
        ph.RobotFinitePatchCollider("patch", 0, SQUARE, 0.125)
    for changed in (
        {"key": ""},
        {"key": " bad"},
        {"link_index": True},
        {"link_index": -1},
        {"local_vertices": SQUARE.astype(np.float32)},
        {"local_vertices": np.where(SQUARE == 1, np.nan, SQUARE)},
    ):
        values = dict(key="patch", link_index=0, local_vertices=SQUARE, radius_m=0.125)
        values.update(changed)
        with pytest.raises((ValueError, TypeError)):
            ph.RobotFinitePatchCollider(**values)


def test_transform_coordinate_axis_and_fingerprint_controls():
    base = model(True)
    alpha = 0.25
    # An X rotor is a coordinate permutation of the Z rotor. A fixed frame
    # and translation must transform the entire FK curve, including witness.
    permutation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    quaternion = np.array([[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]])
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([3.0, -2.0, 4.0])
    transformed = replace(
        base,
        joint_axis=np.array([[1.0, 0.0, 0.0]]),
        joint_orientation=quaternion,
        joint_position=translation[None, :],
    )
    c = patch((permutation @ TRIANGLE.T).T)
    x = R @ permutation @ (1.5 * np.array([-math.sin(alpha), math.cos(alpha), 0.0])) + translation
    result = ph.robot_patch_sweep(
        transformed, np.array([alpha]), np.array([alpha + math.pi]), c, x, x, clearance_m=0.125
    )
    assert result.status == "violation" and result.witness_time == 0.5


def test_exact_fraction_midpoint_cannot_create_false_violation():
    from fractions import Fraction as F

    c = ph.RobotFinitePatchCollider(
        "negative-quadrant",
        0,
        np.array([[-1.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        radius_m=1.0,
    )
    x0 = np.array([0.5999999999999998, 0.0, 0.8000000000000003])
    x1 = np.array([0.6000000000000001, 0.0, 0.7999999999999999])
    p = tuple(map(F, x0))
    delta = tuple(F(b) - a for a, b in zip(p, x1, strict=True))
    denominator = sum(v * v for v in delta)
    time = max(F(0), min(F(1), -sum(a * b for a, b in zip(p, delta, strict=True)) / denominator))
    # All x,z remain positive, hence the physical closest point is the origin
    # vertex for the entire path. This minimization is independent of CCD.
    exact_minimum = sum((a + time * b) ** 2 for a, b in zip(p, delta, strict=True))
    assert exact_minimum - 1 == F(1, 81129638414606681695789005144064)
    midpoint = tuple(a + b / 2 for a, b in zip(p, delta, strict=True))
    assert sum(F(float(x)) ** 2 for x in midpoint) < 1
    result = ph.robot_patch_sweep(model(), np.zeros(1), np.zeros(1), c, x0, x1, clearance_m=1.0)
    assert result.status != "violation", "rounded midpoint must not replace exact declared path"


def test_fixed_frame_edge_label_preserves_physical_witness():
    from phymeo.dynamics.robot_kinematics import forward_kinematics
    from phymeo.geometry.robot_patch_sweep import patch_geometry

    axis = np.array([1.0, 2.0, 3.0]) / math.sqrt(14)
    quaternion = np.r_[axis * math.sin(0.3 / 2), math.cos(0.3 / 2)]
    robot = replace(
        model(),
        joint_orientation=quaternion[None, :],
        joint_position=np.array([[0.25, 0.375, -0.5]]),
    )
    fk = forward_kinematics(robot, np.zeros(1))
    node = fk.position[0] + fk.rotation[0] @ np.array([1.0, 0.125, 0.25])
    raw_local = np.linalg.solve(fk.rotation[0], node - fk.position[0])
    assert raw_local[0] == 0.9999999999999999
    sample = patch_geometry(robot, np.zeros(1), patch(), node)
    assert sample.feature[0] == 2, "uncertain face/edge incidence must use diagnostic edge"
    # The physical witness stays strictly inside despite the edge label.
    np.testing.assert_array_equal(sample.local_point, [raw_local[0], 0.0, raw_local[2]])
    assert sample.local_point[0] < 1.0
    world = fk.position[0] + fk.rotation[0] @ sample.local_point
    np.testing.assert_array_equal(sample.world_point, world)
    np.testing.assert_array_equal(sample.normal, (node - world) / np.hypot.reduce(node - world))


def test_interval_preserves_exact_path_arithmetic_and_stored_inputs():
    from fractions import Fraction as F

    from phymeo.geometry._patch_interval import interval

    exact = F(2**53 + 1, 2**53)
    assert float(exact) == 1.0
    assert interval(exact) == (exact, exact)
    for value in (False, True, 0.0, np.float64(0.1)):
        expected = F(float(value))
        assert interval(value) == (expected, expected)


@pytest.mark.parametrize("angle", [0.2, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("vertices", [SQUARE, SQUARE[::-1], np.roll(SQUARE, 2, axis=0)])
def test_feature_uncertainty_uses_only_labels_across_frames_and_orders(angle, vertices):
    from fractions import Fraction as F

    from phymeo.dynamics.robot_kinematics import forward_kinematics
    from phymeo.geometry.robot_patch_sweep import patch_geometry

    axis = np.array([1.0, 2.0, 3.0]) / math.sqrt(14)
    quaternion = np.r_[axis * math.sin(angle / 2), math.cos(angle / 2)]
    robot = replace(
        model(),
        joint_orientation=quaternion[None, :],
        joint_position=np.array([[0.25, 0.375, -0.5]]),
    )
    fk = forward_kinematics(robot, np.zeros(1))
    c = patch(vertices)
    for nominal, expected_feature in (
        ([0.0, 0.125, 0.25], (1, 0)),
        ([1.0, 0.125, 0.25], (2, 2)),
        ([1.0, 0.125, 1.0], (3, 2)),
    ):
        node = fk.position[0] + fk.rotation[0] @ np.array(nominal)
        raw = np.linalg.solve(fk.rotation[0], node - fk.position[0])
        x, z = F(float(np.clip(raw[0], -1, 1))), F(float(np.clip(raw[2], -1, 1)))
        # Independent exact square/fan barycentrics, in canonical +Y order.
        weights = (
            ((1 - z) / 2, (z - x) / 2, (1 + x) / 2, F(0))
            if z >= x
            else ((1 - x) / 2, F(0), (1 + z) / 2, (x - z) / 2)
        )
        local = np.array([float(x), 0.0, float(z)])
        world = fk.position[0] + fk.rotation[0] @ local
        normal = (node - world) / np.hypot.reduce(node - world)
        sample = patch_geometry(robot, np.zeros(1), c, node)
        assert sample.feature == expected_feature
        assert sample.exact_weights == weights
        np.testing.assert_array_equal(sample.weights, np.array(weights, dtype=float))
        np.testing.assert_array_equal(sample.local_point, local)
        np.testing.assert_array_equal(sample.world_point, world)
        np.testing.assert_array_equal(sample.normal, normal)


def test_zero_error_retains_exact_incidence_and_no_common_feature_rejects():
    from phymeo.geometry.robot_patch_sweep import patch_geometry

    # Identity FK/inverse has exactly E=0; even a one-ULP interior face remains
    # a face. The transformed tie rule is not a global boundary epsilon.
    node = np.array([np.nextafter(1.0, 0.0), 0.125, 0.25])
    sample = patch_geometry(model(), np.zeros(1), patch(), node)
    assert sample.feature == (1, 0)
    assert sample.local_point[0] == node[0]
    # At a scale where stored world corners lose the entire X extent, the
    # derived stage error covers opposite edges with no common feature.
    shifted = replace(model(), joint_position=np.array([[2.0**55, 0.0, 0.0]]))
    with pytest.raises(ph.SolverFailure, match="topology unresolved"):
        patch_geometry(shifted, np.zeros(1), patch(), np.array([2.0**55, 0.125, 0.0]))
