"""A structural link and a material face exchange one physical load and one joint effort."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.constraints.cloth_material import ClothMaterial
from phymeo.constraints.robot_contact import LinkCapsule
from phymeo.dynamics.robot_contact import RobotContactSystem


def make_scene(
    *, revolute=False, split=False, height=0.25, enabled=True, operator=None, friction=0.0
):
    x = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 2.0, 0.0]])
    faces = [(0, 1, 2)]
    if split:
        x = np.vstack([x, [0.0, -1.0, 0.0]])
        faces = [(0, 3, 2), (3, 1, 2)]
    cloth = ph.ParticleModel(np.ones(len(x)), np.zeros(3))
    axis = [0.0, 1.0, 0.0] if revolute else [0.0, 0.0, 1.0]
    robot = ph.RobotModel(
        np.array([-1]),
        ("revolute" if revolute else "prismatic",),
        np.array([[-0.5 if revolute else 0.0, 0.0, height]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([axis]),
        np.array([0.5]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    center = 0.5 if revolute else 0.0
    link = LinkCapsule(
        "beam",
        "arm",
        0,
        np.array([center - 0.25, 0.0, 0.0]),
        np.array([center + 0.25, 0.0, 0.0]),
        0.125,
    )
    links = RobotContactSystem({"arm": robot}, capsules=(link,))
    model = ph.SceneModel(
        (ph.DomainBinding("cloth", "particles", cloth), ph.DomainBinding("arm", "robot", robot))
    )
    surfaces = tuple(
        ph.TrianglePrismSurface(str(i), "cloth", ids, half_thickness_m=1 / 32)
        for i, ids in enumerate(faces)
    )
    edge = ph.DistanceConstraints(np.array([[0, 1]]), np.array([2.0]), np.array([1.0]))
    law = (
        ph.ClothLinkMaterial(
            width_m=1 / 8,
            stiffness_pa_per_m=64 / 3,
            damping_s_per_m=0.0,
            friction=friction,
            slip_speed_m_s=0.01,
        )
        if enabled
        else None
    )
    program = ph.CompliantClothContactProgram(
        model,
        "cloth",
        ClothMaterial(distance=edge),
        surfaces,
        x,
        (),
        robot_contact=links,
        max_penetration_m=0.01,
        cloth_link_material=law,
        cloth_link_operator=operator,
    )
    state = ph.SceneState(
        {
            "cloth": ph.ParticleState(x, np.zeros_like(x)),
            "arm": ph.RobotState(np.zeros(1), np.zeros(1)),
        }
    )
    inputs = ph.SceneInputs({"cloth": ph.StepInputs.zeros(len(x)), "arm": ph.RobotInputs.zeros(1)})
    return program, state, inputs


@pytest.mark.parametrize("split", [False, True])
def test_whole_shaft_load_refinement_and_common_point_moment(split):
    p, s, i = make_scene(split=split)
    accel, _, _, body = p.accelerations(s, i)
    rows = body["cloth_links"]
    assert len(rows) == (2 if split else 1)
    assert sum(row["normal_force_n"] for row in rows) == pytest.approx(2.0)
    np.testing.assert_allclose(accel["cloth"].sum(0), [0, 0, -2.0], atol=1e-13)
    np.testing.assert_allclose(body["cloth_link_efforts"]["arm"], [2.0], atol=1e-13)
    np.testing.assert_allclose(accel["arm"], [4.0], atol=1e-13)
    moment = np.cross(s.domains["cloth"].position, accel["cloth"]).sum(0)
    for row in rows:
        for site in row["sites"]:
            moment += np.cross(site["point_m"], site["force_on_link_n"])
            np.testing.assert_allclose(
                np.add(site["force_on_cloth_n"], site["force_on_link_n"]), 0, atol=1e-15
            )
    np.testing.assert_allclose(moment, 0, atol=1e-13)


def test_revolute_lever_arm_virtual_work_and_separation():
    p, s, i = make_scene(revolute=True)
    s.domains["arm"].v[0] = 0.25
    accel, _, _, body = p.accelerations(s, i)
    tau = body["cloth_link_efforts"]["arm"][0]
    assert tau == pytest.approx(-1.0)
    assert sum(row["power_w"] for row in body["cloth_links"]) == pytest.approx(tau * 0.25)
    assert accel["arm"][0] < 0.0
    p, s, i = make_scene(height=1.0)
    _, _, _, body = p.accelerations(s, i)
    assert body["cloth_links"] == []
    np.testing.assert_array_equal(body["cloth_link_efforts"]["arm"], [0.0])


def test_public_step_moves_both_domains_and_records_real_link_load():
    p, s, i = make_scene()
    with p.stepper(
        ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
    ) as stepper:
        result = stepper.step(s, i, 1 / 256)
    assert result.report.accepted
    diagnostic = result.report.substeps[0].diagnostics["robot_contact"]
    assert diagnostic["cloth_links"][0]["normal_force_n"] > 0
    qp = result.state.domains["arm"]
    assert qp.q[0] > 0 and qp.v[0] > 0
    assert result.state.domains["cloth"].velocity[:, 2].max() < 0
    assert qp.v[0] * 0.5 + result.state.domains["cloth"].velocity[:, 2].sum() == pytest.approx(
        0.0, abs=1e-12
    )


def test_inside_axis_and_hidden_axis_crossing_reject_without_publishing_state():
    p, s, i = make_scene(height=0.0)
    with p.stepper(
        ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
    ) as stepper:
        result = stepper.step(s, i, 1 / 256)
    assert not result.report.accepted and "axis" in result.report.reason
    np.testing.assert_array_equal(result.state.domains["arm"].q, s.domains["arm"].q)
    p, s, i = make_scene(height=0.25)
    s.domains["arm"].v[0] = -128.0
    with p.stepper(
        ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
    ) as stepper:
        result = stepper.step(s, i, 1 / 128)
    assert not result.report.accepted and "axis" in result.report.reason
    np.testing.assert_array_equal(
        result.state.domains["cloth"].position, s.domains["cloth"].position
    )


def test_default_program_does_not_publish_unconfigured_structural_loads():
    p, s, i = make_scene(enabled=False)
    accel, _, _, body = p.accelerations(s, i)
    assert "cloth_links" not in body
    np.testing.assert_array_equal(accel["cloth"], np.zeros((3, 3)))


def test_tangential_load_and_work_come_from_the_same_material_point():
    p, s, i = make_scene(friction=0.5)
    s.domains["cloth"].velocity[:, 0] = 0.5
    accel, _, _, body = p.accelerations(s, i)
    (row,) = body["cloth_links"]
    np.testing.assert_allclose(accel["cloth"].sum(0), [-1.0, 0.0, -2.0], atol=1e-13)
    assert row["power_w"] == pytest.approx(-0.5)
    for site in row["sites"]:
        np.testing.assert_allclose(site["force_on_cloth_n"], [-0.5, 0.0, -1.0], atol=1e-13)


def test_structural_law_binds_the_cpu_or_explicit_tyx_operator():
    with pytest.raises(TypeError, match="TyxNormalLayer"):
        make_scene(operator=object())
