"""Independent sphere/link surface witnesses, two-body wrenches and coupled motion."""

from itertools import product

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial

LAW = RobotContactMaterial(2000.0, 10.0, 0.5, 0.04)


def binding(
    name="rail", kind="capsule", owner=0, vertices=((0.0, 0.0, -0.25), (0.0, 0.0, 0.25)), radius=0.1
):
    from phymeo.solvers.sphere_link_contact import LinkSolid

    return LinkSolid(name, name, owner, kind, vertices, radius)


def evaluate(
    shapes,
    center,
    speed=(0.0, 0.0, 0.0),
    omega=(0.0, 0.0, 0.0),
    law=LAW,
    com=(0.0, 0.0, 0.0),
    twist=(0.0,) * 6,
    rotation=None,
):
    from phymeo.solvers.sphere_link_contact import SphereLinkContacts

    c = np.array(center)
    rot = np.eye(3) if rotation is None else np.array(rotation)
    return SphereLinkContacts(shapes).evaluate(
        np.zeros((1, 3)),
        rot[None],
        np.array([com]),
        np.array([twist]),
        c,
        0.05,
        np.array(speed),
        np.array(omega),
        law,
    )


def test_finite_capsule_side_wrench_damping_power_and_com():
    result = evaluate([binding()], (0.149, 0.0, 0.075), (-0.1, 0.06, 0.0), com=(0, 0, 0.02))
    (row,) = result.contacts
    assert row["feature"] == "shaft" and row["gap_m"] == pytest.approx(-0.001)
    np.testing.assert_allclose(row["normal_world"], [1, 0, 0], atol=2e-15)
    np.testing.assert_allclose(row["sphere_surface_m"], [0.099, 0, 0.075])
    np.testing.assert_allclose(row["link_surface_m"], [0.1, 0, 0.075])
    np.testing.assert_allclose(row["point_world_m"], [0.0995, 0, 0.075])
    np.testing.assert_allclose(row["link_com_world_m"], [0, 0, 0.02])
    np.testing.assert_allclose(row["force_on_payload_N"], [3, -1.5, 0], atol=1e-12)
    np.testing.assert_allclose(row["force_on_link_N"], [-3, 1.5, 0], atol=1e-12)
    np.testing.assert_allclose(row["torque_on_payload_Nm"], [0, 0, 0.07425], atol=1e-12)
    np.testing.assert_allclose(row["torque_on_link_Nm"], [-0.0825, -0.165, 0.14925], atol=1e-12)
    assert row["pair_power_W"] == pytest.approx(-0.39)
    # Compression contributes -k*delta*delta_dot; damping and slip dissipate 0.19 W.
    assert row["pair_power_W"] + 0.2 == pytest.approx(-0.19)
    np.testing.assert_allclose(result.payload_wrench, [3, -1.5, 0, 0, 0, 0.07425], atol=1e-12)


def test_finite_capsule_end_and_separation_do_not_extend_its_shaft():
    cap = binding()
    n = np.array([0.6, 0, 0.8])
    tip = np.array([0, 0, 0.25])
    (row,) = evaluate([cap], tip + 0.149 * n).contacts
    assert row["feature"] == "end1"
    np.testing.assert_allclose(row["normal_world"], n)
    np.testing.assert_allclose(row["point_world_m"], tip + 0.0995 * n)
    assert not evaluate([cap], tip + [0, 0, 0.151]).contacts
    (release,) = evaluate([cap], tip + 0.149 * n, speed=2 * n).contacts
    assert release["normal_load_N"] == 0 and not np.any(release["force_on_payload_N"])


def test_spinning_structure_relative_velocity_and_registration_order():
    first = binding("left")
    second = binding("right", "sphere", vertices=((0.299, 0, 0.075),), radius=0.102)
    one = evaluate([first, second], [0.149, 0, 0.075], twist=[0, 0, 0, 0, 0, 2])
    other = evaluate([second, first], [0.149, 0, 0.075], twist=[0, 0, 0, 0, 0, 2])
    np.testing.assert_allclose(one.payload_wrench, other.payload_wrench, rtol=0, atol=0)
    assert [r["shape"] for r in one.contacts] == [r["shape"] for r in other.contacts]
    row = one.contacts[0]
    np.testing.assert_allclose(row["relative_velocity_m_s"], [0, -0.199, 0])
    assert row["pair_power_W"] < 0


def test_oriented_box_corner_and_ambiguous_geometry():
    corners = np.array(list(product((-1, 1), repeat=3))) * [0.1, 0.2, 0.3]
    box = binding("box", "box", vertices=corners, radius=0.0)
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    n = np.array([0.6, 0.8, 0])
    boundary = rot @ np.array([0.1, 0.2, 0])
    (row,) = evaluate([box], boundary + 0.049 * (rot @ n), rotation=rot).contacts
    np.testing.assert_allclose(row["force_on_payload_N"], 2 * (rot @ n), atol=1e-12)
    assert row["feature"] == "edge"
    assert not evaluate([box], boundary + 0.051 * (rot @ n), rotation=rot).contacts
    with pytest.raises(ValueError, match="normal"):
        evaluate([binding()], [0, 0, 0.1])
    with pytest.raises(ValueError, match="box"):
        binding("bad", "box", vertices=corners[[0, 1, 2, 3, 4, 6, 5, 7]], radius=0.0)


def test_link_and_free_sphere_advance_from_same_state_with_closed_impulse():
    import phymeo as ph
    from phymeo.dynamics.robot_dynamics import robot_dynamics_terms
    from phymeo.solvers.rigid_semi_implicit import RigidSemiImplicitSolver

    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.array([[0.1, 0.1, 0.1]]),
        np.zeros(3),
    )
    arm = ph.RobotState(np.zeros(1), np.zeros(1))
    _, J, A, _, _ = robot_dynamics_terms(model, arm)
    load = evaluate([binding()], [0.149, 0, 0.075])
    Q = load.joint_load(J, A)
    np.testing.assert_allclose(Q, [-2.0], atol=1e-13)
    dt = 1 / 64
    with ph.make_robot_runtime(model) as runtime:
        move = runtime.step(
            ph.SceneState({"robot": arm}),
            ph.SceneInputs({"robot": ph.RobotInputs(Q, np.zeros((1, 3)), np.zeros((1, 3)))}),
            dt,
        )
        assert move.report.accepted
        slide = move.state.domains["robot"]
    ball_model = ph.RigidBodyModel(np.ones(1), np.full((1, 3), 0.001), np.zeros(3))
    ball = ph.RigidBodyState(
        np.array([[0.149, 0, 0.075]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
    )
    out = RigidSemiImplicitSolver().step(
        ball_model,
        ball,
        ph.RigidBodyInputs(load.payload_wrench[None, :3], load.payload_wrench[None, 3:]),
        dt,
    )
    assert out.report.accepted
    np.testing.assert_allclose(slide.v, [-1 / 64], atol=1e-14)
    np.testing.assert_allclose(out.state.velocity, [[1 / 32, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(out.state.position[0, 0], 0.149 + dt / 32, atol=1e-14)
    np.testing.assert_allclose(2 * slide.v[0] + out.state.velocity[0, 0], 0, atol=1e-14)


def test_fixed_base_absorbs_reaction_without_a_generalized_force():
    load = evaluate([binding(owner=-1)], [0.149, 0.0, 0.075])
    (row,) = load.contacts
    assert row["owner_index"] == -1
    np.testing.assert_allclose(row["force_on_link_N"], [-2, 0, 0], atol=1e-14)
    np.testing.assert_array_equal(load.link_wrenches, np.zeros((1, 6)))


def test_original_50mm_ejection_now_exchanges_force_with_a_real_nonpad_link(tmp_path):
    import json

    from phymeo.experiments.panda_cube import run_panda_cube

    result = run_panda_cube(
        payload_geometry={"kind": "sphere", "radius_m": 0.025},
        mass_kg=0.08,
        grip_above_com_m=0.008,
        closed_width_m=0.024,
        friction=0.0,
        record=True,
    )
    (tmp_path / "active-phy.json").write_text(json.dumps(result), encoding="utf8")
    (tmp_path / "active.inputs.json").write_text(json.dumps(result["inputs"]), encoding="utf8")
    assert result["steps"] == 3600 and not result["carried_during_transfer"]
    first = result["first_structural_contact"]
    assert first["shape"] == "panda_link1_sc_0" and first["t_before_s"] == pytest.approx(0.836875)
    row = result["trace"][first["step_index"]]
    (contact,) = row["structural_contacts"]
    assert contact["gap_m"] == pytest.approx(-0.00047323797588965516, abs=1e-10)
    assert contact["normal_load_N"] > 0 and np.linalg.norm(row["structural_joint_load"]) > 0
    assert result["first_unresolved_structure"] is None
    before = result["trace"][first["step_index"] - 1]
    np.testing.assert_allclose(
        before["cube"],
        [0.08452481405519485, 0.0005738529561756424, 0.2500903823928133],
        atol=1e-11,
        rtol=0,
    )
    dt = result["inputs"]["dt_s"]
    p = 0.08 * (np.array(row["velocity"]) - before["velocity"] - dt * np.array([0.0, 0.0, -9.81]))
    np.testing.assert_allclose(p, dt * np.array(contact["force_on_payload_N"]), atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "kind,center,normal,gap,feature",
    [
        ("sphere", [0.0, 0.129, 0.0], [0, 1, 0], -0.001, "sphere"),
        ("box", [0.149, 0.0, 0.0], [1, 0, 0], -0.001, "face"),
        ("box", [0.099, 0.0, 0.0], [1, 0, 0], -0.051, "face"),
        (
            "box",
            np.array([0.1, 0.2, 0.3]) + 0.049 * np.array([1, 2, 2]) / 3,
            np.array([1, 2, 2]) / 3,
            -0.001,
            "corner",
        ),
    ],
)
def test_spherical_and_cuboid_supports_have_their_own_witness(kind, center, normal, gap, feature):
    corners = np.array(list(product((-1, 1), repeat=3))) * [0.1, 0.2, 0.3]
    solid = (
        binding("outer", kind, vertices=((0.0, 0.0, 0.0),), radius=0.08)
        if kind == "sphere"
        else binding("outer", kind, vertices=corners, radius=0.0)
    )
    (row,) = evaluate([solid], center).contacts
    assert row["feature"] == feature and row["gap_m"] == pytest.approx(gap, abs=1e-14)
    np.testing.assert_allclose(
        row["force_on_payload_N"], -2000 * gap * np.array(normal), atol=2e-13
    )
    np.testing.assert_allclose(
        row["link_surface_m"], np.array(center) - (0.05 + gap) * np.array(normal), atol=2e-15
    )
    np.testing.assert_allclose(row["torque_on_payload_Nm"], np.zeros(3), atol=2e-15)
