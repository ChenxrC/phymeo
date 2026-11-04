"""A moving OBB on a fixed OBB: real clipped areas, torque modes and lower-dimensional limits."""

from itertools import product

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.solvers.fixed_table_contact import FixedTableContacts
from phymeo.solvers.sphere_link_contact import LinkSolid

HALF = np.array([0.125, 0.0625, 0.03125])
DZ = 1 / 1024
LAW = RobotContactMaterial(4096.0, 8.0, 0.5, 0.0625)


def measured(
    *,
    half=HALF,
    xy=(0.0, 0.0),
    R=None,
    table_xy=(0.0, 0.0),
    table_half=(0.25, 0.125),
    omega=(0.0, 0.0, 0.0),
    law=LAW,
    z=None,
):
    R = np.eye(3) if R is None else np.asarray(R)
    vertices = np.array(list(product((-1.0, 1.0), repeat=3))) * half
    foot = LinkSolid("shoe", "moving_shoe", 0, "box", vertices, 0)
    table = FixedTableContacts(
        [foot], center_m=[*table_xy, -0.03125], half_extents_m=[*table_half, 0.03125], material=law
    )
    z = abs(R[2]) @ half - DZ if z is None else z
    c = np.array([[*xy, z]])
    return table.evaluate(c, R[None], c, np.array([[0.0, 0, 0, *omega]]))


def test_clipped_rotated_polygon_pressure_matches_independent_removed_triangle_centroid():
    radius = 0.125
    c = np.sqrt(0.5)
    R = np.array([[c, -c, 0], [c, c, 0], [0, 0, 1.0]])
    # A diamond |x|+|y|<=r with triangle x>r/2 removed. The vertex mean is incorrectly zero.
    out = measured(
        half=np.array([radius * c, radius * c, 0.03125]),
        R=R,
        table_xy=(-0.09375, 0),
        table_half=(0.15625, 0.25),
    )
    rows = out.contacts
    assert len(rows) >= 3 and all(h["patch_dimension"] == 2 for h in rows)
    areas = np.array([h["area_weight_m2"] for h in rows])
    weight = np.array([h["weight"] for h in rows])
    area = 7 * radius**2 / 4
    xc = -2 * radius / 21
    np.testing.assert_allclose(areas.sum(), area, atol=1e-14)
    np.testing.assert_allclose(weight, areas / area, atol=1e-13)
    points = np.array([h["table_surface_m"] for h in rows])
    np.testing.assert_allclose(weight @ points, [xc, 0, 0], atol=1e-13)
    np.testing.assert_allclose(out.link_wrenches[0], [0, 0, 4, 0, -4 * xc, 0], atol=1e-12)
    assert all(min(h["link_material_weights"], key=lambda w: w[1])[1] >= 0 for h in rows)


def test_flat_patch_transmits_pure_torsional_traction_and_tipping_damping():
    dry = RobotContactMaterial(4096.0, 8.0, 0.0, 0.0625)
    w = 0.125
    with_friction = measured(omega=(0, 0, w))
    normal_only = measured(omega=(0, 0, w), law=dry)
    delta = with_friction.link_wrenches[0] - normal_only.link_wrenches[0]
    exact = -0.5 * 4 * w / 0.0625 * (HALF[0] ** 2 + HALF[1] ** 2) / 3
    np.testing.assert_allclose(delta, [0, 0, 0, 0, 0, exact], atol=1e-13)
    assert sum(r["link_power_W"] for r in with_friction.contacts) == pytest.approx(w * exact)
    tip = measured(omega=(0, 0.5, 0), law=dry)
    np.testing.assert_allclose(
        tip.link_wrenches[0], [0, 0, 4, 0, -8 * 0.5 * HALF[0] ** 2 / 3, 0], atol=1e-12
    )
    assert all(r["table_power_W"] == 0 for r in tip.contacts)


def test_rotated_edge_and_vertex_limits_are_real_witnesses():
    rx = np.array([[1.0, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])
    ry = np.array([[0.8, 0, 0.6], [0, 1.0, 0], [-0.6, 0, 0.8]])
    edge = measured(R=rx).contacts
    assert len(edge) == 2 and all(r["weight"] == 0.5 and r["patch_dimension"] == 1 for r in edge)
    (tip,) = measured(R=rx @ ry).contacts
    assert tip["weight"] == 1 and tip["patch_dimension"] == 0
    assert tip["normal_load_N"] == pytest.approx(4.0)
    for r in [*edge, tip]:
        assert r["area_weight_m2"] == 0 and r["patch_area_m2"] == 0
    c = np.sqrt(0.5)
    R = np.array([[c, -c, 0], [c, c, 0], [0, 0, 1.0]])
    assert not measured(half=np.array([0.0625, 0.0625, 0.03125]), R=R, xy=(0.305, 0.18)).contacts


def test_mixed_round_and_box_pass_pair_load_once_and_ambiguous_axis_is_rejected():
    from itertools import product

    verts = np.array(list(product((-1.0, 1.0), repeat=3))) * HALF
    foot = LinkSolid("sole", "foot", 0, "box", verts, 0.0)
    roller = LinkSolid("roller", "other", 1, "sphere", [[0, 0, 0]], 0.03125)
    channel = FixedTableContacts(
        [roller, foot], center_m=[0, 0, -0.03125], half_extents_m=[0.5, 0.5, 0.03125], material=LAW
    )
    pos = np.array([[0.0, 0.0, HALF[2] - DZ], [0.24, 0.0, 0.03125 - 2 * DZ]])
    answer = channel.evaluate(pos, np.tile(np.eye(3), (2, 1, 1)), pos, np.zeros((2, 6)))
    np.testing.assert_allclose(answer.link_wrenches[:, 2], [4, 8], atol=1e-13)
    np.testing.assert_allclose(answer.table_wrench[:3], [0, 0, -12], atol=1e-13)
    same = FixedTableContacts([foot], center_m=[0, 0, 0], half_extents_m=HALF, material=LAW)
    with pytest.raises(ValueError, match="unique.*axis"):
        same.evaluate(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, 3)), np.zeros((1, 6)))


def test_same_face_patch_in_a_tilted_world_frame_keeps_its_material_area():
    from phymeo.dynamics.rotation import quaternion_matrix

    half = np.array([0.125, 0.0625, 0.03125])
    yaw = np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1.0]])
    q = np.array([0.17, 0.37, 0.29, 0.86])
    q /= np.linalg.norm(q)
    world = quaternion_matrix(q[None])[0]
    offset = np.array([2.0, -3.0, 0.5])
    c0 = np.array([0.0625, 0.015625, half[2] - DZ])
    t0 = np.array([0, 0, -0.03125])
    table_half = np.array([0.125, 0.125, 0.03125])
    original = measured(half=half, xy=c0[:2], R=yaw, table_half=table_half[:2])
    vertices = np.array(list(product((-1.0, 1.0), repeat=3))) * half
    sole = LinkSolid("shoe", "shoe", 0, "box", vertices, 0.0)
    table = FixedTableContacts(
        [sole],
        center_m=offset + world @ t0,
        half_extents_m=table_half,
        orientation_xyzw=q,
        material=LAW,
    )
    c = (offset + world @ c0)[None]
    moved = table.evaluate(c, (world @ yaw)[None], c, np.zeros((1, 6)))
    assert len(original.contacts) == len(moved.contacts) and all(
        r["patch_dimension"] == 2 for r in moved.contacts
    )
    assert moved.contacts[0]["patch_area_m2"] == pytest.approx(
        original.contacts[0]["patch_area_m2"], abs=3e-13
    )
    np.testing.assert_allclose(
        moved.table_wrench[:3], world @ original.table_wrench[:3], atol=2e-11
    )
    np.testing.assert_allclose(
        moved.table_wrench[3:], world @ original.table_wrench[3:], atol=2e-11
    )


def test_small_resolved_tilt_is_an_edge_not_a_full_area_patch():
    sine = 2.0**-25
    cosine = np.sqrt(1 - sine * sine)
    tilt = np.array([[1.0, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    rows = measured(R=tilt).contacts
    assert len(rows) == 2 and all(r["patch_dimension"] == 1 and r["weight"] == 0.5 for r in rows)
