"""Frozen fixed-table reactions use the robot's actual finite material points."""

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.solvers.sphere_link_contact import LinkSolid

R = 1 / 32
DEPTH = 1 / 1024
HALF = np.array([0.25, 0.125, 1 / 32])
TABLE_CENTER = np.array([0.5, -0.25, -1 / 32])


def table(ends, *, radius=R, material=RobotContactMaterial(4096.0, 8.0, 0.5, 0.04)):
    from phymeo.solvers.fixed_table_contact import FixedTableContacts

    solid = LinkSolid("skid", "carrier", 0, "capsule", ends, radius)
    return FixedTableContacts(
        [solid], center_m=TABLE_CENTER, half_extents_m=HALF, material=material
    )


def measure(contact, origin, velocity=(0, 0, 0), spin=(0, 0, 0)):
    origin = np.asarray(origin, dtype=float)
    return contact.evaluate(
        origin[None], np.eye(3)[None], origin[None], np.array([[*velocity, *spin]], dtype=float)
    )


def test_sliding_capsule_on_finite_table_has_one_weighted_pair_and_fixed_reaction():
    # A 2 kg carriage is balanced against g=2. Finite table trims the shaft to its middle half.
    contact = table([[-0.5, 0, 0], [0.5, 0, 0]])
    C = TABLE_CENTER + [0, 1 / 32, HALF[2] + R - DEPTH]
    result = measure(contact, C, velocity=(0.02, 0, 0))
    assert len(result.contacts) == 2
    np.testing.assert_allclose([r["weight"] for r in result.contacts], [0.5, 0.5])
    np.testing.assert_allclose([r["axis_parameter"] for r in result.contacts], [0.25, 0.75])
    np.testing.assert_allclose(result.table_wrench[:3], [1, 0, -4])
    np.testing.assert_allclose(result.link_wrenches[0, :3], [-1, 0, 4])
    p = np.array([0, 1 / 32, HALF[2] - 0.5 * DEPTH])
    np.testing.assert_allclose(result.table_wrench[3:], np.cross(p, [1, 0, -4]), atol=1e-14)
    for r in result.contacts:
        assert r["table_power_W"] == 0 and r["link_power_W"] == pytest.approx(-0.01)
        np.testing.assert_allclose(r["force_on_table_N"], -np.array(r["force_on_link_N"]))
        np.testing.assert_allclose(r["table_center_world_m"], TABLE_CENTER)
    J = np.zeros((1, 3, 1))
    J[0, 2, 0] = 1
    np.testing.assert_allclose(result.joint_load(J, np.zeros_like(J)), [4])


def test_true_round_end_contacts_finite_edge_and_near_miss_is_not_an_infinite_plane():
    near = TABLE_CENTER + [HALF[0], 0, HALF[2]]
    n = np.array([0.6, 0, 0.8])
    a = near + (R - DEPTH) * n
    b = a + 0.0625 * n
    result = measure(table([a, b]), [0, 0, 0])
    (row,) = result.contacts
    assert row["axis_parameter"] == 0 and row["feature"] == "end0"
    np.testing.assert_allclose(row["table_surface_m"], near, atol=1e-14)
    np.testing.assert_allclose(row["force_on_link_N"], 4 * n, atol=1e-13)
    np.testing.assert_allclose(row["force_on_table_N"], -4 * n, atol=1e-13)
    # The same low point still penetrates an infinite plane, but is beyond the finite edge.
    other = measure(table([a + [0.05, 0, 0], b + [0.05, 0, 0]]), [0, 0, 0])
    assert other.contacts == [] and not np.any(other.table_wrench)


def test_unloading_cannot_pull_and_spinning_link_uses_the_separate_point_speeds():
    C = TABLE_CENTER + [0, 0, HALF[2] + R - DEPTH]
    c = table([[-0.5, 0, 0], [0.5, 0, 0]], material=RobotContactMaterial(4096.0, 8.0, 0.0, 0.04))
    loaded = measure(c, C, spin=(0, 2, 0))
    np.testing.assert_allclose([r["normal_load_N"] for r in loaded.contacts], [0, 4], atol=1e-13)
    np.testing.assert_allclose(loaded.link_wrenches[0], [0, 0, 4, 0, -1, 0], atol=1e-13)
    assert sum(r["link_power_W"] for r in loaded.contacts) == pytest.approx(-2)
    release = measure(c, C, velocity=(0, 0, 1))
    assert not np.any(release.table_wrench)


def test_panda_box_table_link_response_is_separate_from_object_support(tmp_path):
    import json

    from phymeo.experiments.panda_cube import run_panda_cube

    result = run_panda_cube(
        mass_kg=0.12,
        box_size_m=(0.046, 0.058, 0.040),
        closed_width_m=0.024,
        initial_xy_offset_m=(0.001, 0),
        table_link_response=True,
        record=True,
    )
    (tmp_path / "panda.json").write_text(json.dumps(result), encoding="utf8")
    (tmp_path / "panda.inputs.json").write_text(json.dumps(result["inputs"]), encoding="utf8")
    assert (
        result["success"] and result["steps"] == 3600 and result["first_link_table_contact"] is None
    )
    assert all(
        not r["link_table_contacts"] and not np.any(r["robot_on_table_wrench"])
        for r in result["trace"]
    )
    assert result["first_unresolved_structure"] is None and result[
        "final_table_force_N"
    ] == pytest.approx(1.1772, abs=0.025)


def test_actual_vertical_joint_balances_its_capsule_then_lifts_clear():
    import phymeo as ph

    origin = TABLE_CENTER + [0, 1 / 32, HALF[2] + R - DEPTH]
    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        origin[None],
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.array([[0.02, 0.18, 0.18]]),
        np.array([0.0, 0.0, -2.0]),
    )
    contact = table([[-0.5, 0, 0], [0.5, 0, 0]])
    state = ph.SceneState({"robot": ph.RobotState(np.zeros(1), np.zeros(1))})
    h = 1 / 1024
    held_impulse = 0.0
    with ph.make_robot_runtime(model) as rt:
        for i in range(96):
            load, Q = contact.evaluate_robot(model, state.domains["robot"])
            drive = 0 if i < 32 else 6.0
            step = rt.step(
                state,
                ph.SceneInputs(
                    {"robot": ph.RobotInputs(Q + drive, np.zeros((1, 3)), np.zeros((1, 3)))}
                ),
                h,
            )
            assert step.report.accepted
            state = step.state
            q, v = state.domains["robot"].q[0], state.domains["robot"].v[0]
            if i < 32:
                assert abs(q) < 1e-14 and abs(v) < 1e-14 and len(load.contacts) == 2
                held_impulse += h * load.table_wrench[2]
            elif i == 32:
                assert v == pytest.approx(3 * h) and q == pytest.approx(3 * h * h)
        assert held_impulse == pytest.approx(-0.125)
        assert q > DEPTH and v > 0 and not load.contacts
