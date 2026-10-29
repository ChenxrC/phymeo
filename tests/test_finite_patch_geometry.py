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
