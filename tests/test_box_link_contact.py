"""Actual finite capsule witnesses and weighted reciprocal loads on an oriented free box."""

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.solvers.sphere_link_contact import LinkSolid

LAW = RobotContactMaterial(2048.0, 8.0, 0.5, 0.04)
HALF = np.array([0.25, 0.125, 0.0625])
R = 0.03125
DEPTH = 1 / 1024


def measure(
    a,
    b,
    *,
    half=HALF,
    pose=None,
    center=(0, 0, 0),
    velocity=(0, 0, 0),
    omega=(0, 0, 0),
    rail_v=(0, 0, 0, 0, 0, 0),
    rail_origin=(0, 0, 0),
    law=LAW,
):
    from phymeo.solvers.box_link_contact import BoxLinkContacts

    solid = LinkSolid("fixture", "moving", 0, "capsule", [a, b], R)
    return BoxLinkContacts([solid]).evaluate(
        np.array([rail_origin], dtype=float),
        np.eye(3)[None],
        np.array([rail_origin], dtype=float),
        np.array([rail_v], dtype=float),
        np.array(center, dtype=float),
        half,
        np.array(velocity, dtype=float),
        np.array(omega, dtype=float),
        law,
        payload_rotation=np.eye(3) if pose is None else pose,
    )


def test_finite_parallel_face_keeps_both_material_ends_without_double_stiffness():
    z = HALF[2] + R - DEPTH
    out = measure([-0.5, 0.04, z], [0.5, 0.04, z], velocity=[0, 0, 0.125])
    assert len(out.contacts) == 2
    rows = sorted(out.contacts, key=lambda r: r["point_world_m"][0])
    np.testing.assert_allclose([r["axis_parameter"] for r in rows], [0.25, 0.75])
    np.testing.assert_allclose([r["weight"] for r in rows], [0.5, 0.5])
    np.testing.assert_allclose([r["normal_load_N"] for r in rows], [1.5, 1.5])
    for r, x in zip(rows, [-0.25, 0.25], strict=True):
        np.testing.assert_allclose(r["payload_surface_m"], [x, 0.04, 0.0625])
        np.testing.assert_allclose(r["link_surface_m"], [x, 0.04, 0.0625 - DEPTH])
        np.testing.assert_allclose(r["normal_world"], [0, 0, -1])
    np.testing.assert_allclose(out.payload_wrench, [0, 0, -3, -0.12, 0, 0], atol=2e-14)
    np.testing.assert_allclose(out.link_wrenches[0], [0, 0, 3, 0.12, 0, 0], atol=2e-14)
    assert sum(r["pair_power_W"] for r in rows) == pytest.approx(-0.375)


def test_skew_edge_one_point_and_finite_end_negative():
    # y-z normal (3,4)/5; the axis direction (0,4,-3) lies along the common tangent.
    n = np.array([0, 0.6, 0.8])
    t = np.array([0, 0.8, -0.6])
    c = np.array([0.10, 0.125, 0.0625])
    a = c + (R - DEPTH) * n - 0.01 * t
    b = c + (R - DEPTH) * n + 0.03 * t
    out = measure(a, b)
    (r,) = out.contacts
    assert r["axis_parameter"] == pytest.approx(0.25) and r["weight"] == 1
    np.testing.assert_allclose(r["payload_surface_m"], c, atol=1e-14)
    np.testing.assert_allclose(r["normal_world"], -n, atol=1e-14)
    out = measure(b, b + 0.02 * t)
    assert not out.contacts or all(r["axis_parameter"] == 0 for r in out.contacts)
    assert not measure(c + 0.06 * n, c + 0.1 * n).contacts


def test_almost_parallel_selects_one_end_and_segment_order_preserves_wrench():
    z = HALF[2] + R - DEPTH
    a = np.array([-0.1, 0.0625, z])
    b = np.array([0.2, 0.0625, z + 2**-35])
    (left,) = measure(a, b, velocity=[0.02, 0, 0]).contacts
    (right,) = measure(b, a, velocity=[0.02, 0, 0]).contacts
    assert left["axis_parameter"] == 0 and right["axis_parameter"] == 1
    for k in ("point_world_m", "force_on_payload_N", "torque_on_payload_Nm", "force_on_link_N"):
        np.testing.assert_allclose(left[k], right[k], atol=1e-14)
    np.testing.assert_allclose(left["force_on_payload_N"], [-0.5, 0, -2], atol=1e-14)


def test_common_rigid_transform_and_crossing_axis_is_rejected():
    A = np.array([[0.0, 0.0, 1.0], [1, 0, 0], [0, 1, 0]])
    c = np.array([3.0, -0.5, 2.0])
    z = HALF[2] + R - DEPTH
    a = np.array([-0.125, 0.0, z])
    b = np.array([0.125, 0.0, z])
    base = measure(a, b)
    moved = measure(A @ a, A @ b, center=c, pose=A, rail_origin=c)
    np.testing.assert_allclose(moved.payload_wrench[:3], A @ base.payload_wrench[:3])
    for r, s in zip(base.contacts, moved.contacts, strict=True):
        np.testing.assert_allclose(s["payload_surface_m"], c + A @ r["payload_surface_m"])
    with pytest.raises(ValueError, match="axis.*box"):
        measure([0, 0, -0.1], [0, 0, 0.1])


def test_swapping_participants_reverses_normal_and_swaps_original_surface_points():
    from phymeo.solvers.box_link_contact import box_capsule_witnesses

    z = HALF[2] + R - DEPTH
    path = np.array([[-0.5, 0.04, z], [0.5, 0.04, z]])

    def sample(order):
        return list(
            box_capsule_witnesses(np.zeros(3), np.eye(3), HALF, path, R, capsule_first=order)
        )

    for a, b in zip(sample(True), sample(False), strict=True):
        np.testing.assert_array_equal(a["normal"], -b["normal"])
        np.testing.assert_array_equal(a["first_surface"], b["second_surface"])
        np.testing.assert_array_equal(a["second_surface"], b["first_surface"])
        assert a["axis_parameter"] == b["axis_parameter"] and a["weight"] == b["weight"]


def test_panda_original_box_with_round_link_response_preserves_loaded_transfer(tmp_path):
    import json

    from phymeo.experiments.panda_cube import run_panda_cube

    result = run_panda_cube(
        box_size_m=(0.046, 0.058, 0.040),
        mass_kg=0.12,
        closed_width_m=0.024,
        initial_xy_offset_m=(0.001, 0.0),
        friction=0.65,
        structural_response=True,
        record=True,
    )
    (tmp_path / "panda-box.json").write_text(json.dumps(result), encoding="utf8")
    (tmp_path / "panda-box.inputs.json").write_text(json.dumps(result["inputs"]), encoding="utf8")
    assert (
        result["success"] and result["steps"] == 3600 and result["first_structural_contact"] is None
    )
    assert not any(r["structural_contacts"] for r in result["trace"])
    assert (
        result["first_unresolved_structure"] is None
        and result["other_robot_cube_axis_clearance_m"] > 0
    )
    assert result["carry_clearance_m"] > 0.035 and result["carry_displacement_m"] > 0.045
    assert result["peak_carry_table_N"] < 0.001 and result["final_target_error_m"] < 0.012


def test_weighted_face_uses_each_contact_velocity_to_resist_tipping():
    z = HALF[2] + R - DEPTH
    out = measure(
        [-0.5, 0, z], [0.5, 0, z], omega=(0, 1, 0), law=RobotContactMaterial(2048.0, 8.0, 0.0, 0.04)
    )
    np.testing.assert_allclose([r["normal_load_N"] for r in out.contacts], [2.0, 0.0], atol=1e-14)
    # A single midpoint would have the same force but would lose the entire restoring torque.
    np.testing.assert_allclose(out.payload_wrench, [0, 0, -2, 0, -0.5, 0], atol=1e-14)
    np.testing.assert_allclose(out.link_wrenches[0], -out.payload_wrench, atol=1e-14)


def test_round_endpoint_at_box_corner_preserves_its_finite_axis_material():
    corner = HALF.copy()
    n = np.array([1.0, 2.0, 2.0]) / 3
    start = corner + (R - DEPTH) * n
    finish = start + 0.15 * n
    for a, b, u in ((start, finish, 0.0), (finish, start, 1.0)):
        (row,) = measure(a, b).contacts
        assert row["weight"] == 1 and row["axis_parameter"] == u
        np.testing.assert_allclose(row["payload_surface_m"], corner, atol=1e-14)
        np.testing.assert_allclose(row["force_on_payload_N"], -2 * n, atol=1e-13)
        np.testing.assert_allclose(
            row["torque_on_payload_Nm"], np.cross(corner, -2 * n), atol=1e-13
        )
