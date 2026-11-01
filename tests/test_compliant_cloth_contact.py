"""Public compliant contact on a moving finite patch, independent of any demo trajectory."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.constraints.cloth_material import ClothMaterial


def triangle_case(*, speed=-0.1, shift=0.0, height=0.011, **settings):
    x = np.array([[0.0, 0.0, height], [1.0, 0.0, height], [0.0, 1.0, height]])
    x[:, 0] += shift
    cloth = ph.ParticleModel(np.full(3, 2.0), np.zeros(3))
    robot = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    scene = ph.SceneModel(
        (ph.DomainBinding("sheet", "particles", cloth), ph.DomainBinding("plate", "robot", robot))
    )
    face = ph.TrianglePrismSurface("tri", "sheet", (0, 1, 2), half_thickness_m=0.01)
    poly = ph.RobotContactPatch(
        "support",
        0,
        np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]]),
    )
    patch = ph.CompliantPatch(
        poly,
        robot_domain="plate",
        width_m=0.002,
        stiffness_pa_per_m=500.0,
        damping_s_per_m=10.0,
        friction=0.4,
        slip_speed_m_s=0.01,
    )
    program = ph.CompliantClothContactProgram(
        scene, "sheet", ClothMaterial(), (face,), x, (patch,), **settings
    )
    state = ph.SceneState(
        {
            "sheet": ph.ParticleState(x, np.tile([0.1, 0.0, speed], (3, 1))),
            "plate": ph.RobotState(np.zeros(1), np.zeros(1)),
        }
    )
    inputs = ph.SceneInputs({"sheet": ph.StepInputs.zeros(3), "plate": ph.RobotInputs.zeros(1)})
    return program, state, inputs


def step(program, state, inputs, dt=0.0005):
    with program.stepper(ph.StepSettings(substeps=1, record_substeps=True)) as runtime:
        return runtime.step(state, inputs, dt)


def test_pressure_friction_and_both_actual_domains_share_one_impulse():
    program, state, inputs = triangle_case()
    dt = 0.0005
    out = step(program, state, inputs, dt)
    assert out.report.accepted, out.report.reason
    assert len(out.substep_states) == 1
    d = out.report.substeps[0].diagnostics["patches"]["support"]
    f = np.asarray(d["force_n"])
    assert all(d["normal_force_n"] > 0)
    assert np.all(f[:, 0] < 0)
    assert np.all(abs(f[:, 0]) <= 0.4 * d["normal_force_n"] + 1e-14)
    a = out.state.domains["sheet"]
    np.testing.assert_allclose(
        0.5 * (a.velocity - state.domains["sheet"].velocity).sum(axis=0),
        dt * f.sum(axis=0),
        atol=1e-15,
    )
    assert 2 * out.state.domains["plate"].v[0] == pytest.approx(-dt * f[:, 2].sum(), abs=1e-15)
    assert out.state.domains["plate"].q[0] < 0
    assert np.array_equal(state.domains["plate"].q, np.zeros(1))


@pytest.mark.parametrize("options", [{"shift": 5.0}, {"height": 0.02}, {"speed": 0.2}])
def test_finite_patch_and_opening_layer_release(options):
    program, state, inputs = triangle_case(**options)
    out = step(program, state, inputs)
    assert out.report.accepted, out.report.reason
    row = out.report.substeps[0].diagnostics["patches"]["support"]
    assert not np.any(row["force_n"])
    assert not np.any(out.state.domains["plate"].v)


def test_degenerate_geometry_rolls_back_both_domains():
    program, state, inputs = triangle_case()
    bad = state.domains["sheet"].copy()
    bad.position[2] = bad.position[1]
    initial = ph.SceneState({"sheet": bad, "plate": state.domains["plate"]})
    out = step(program, initial, inputs)
    assert not out.report.accepted and not out.substep_states
    assert np.array_equal(out.state.domains["sheet"].position, bad.position)
    assert np.array_equal(out.state.domains["plate"].v, initial.domains["plate"].v)


def test_small_overlap_is_recorded_and_gross_overlap_rolls_back():
    loose, state, inputs = triangle_case(height=0.0098, speed=0.0, max_penetration_m=0.001)
    good = step(loose, state, inputs)
    assert good.report.accepted and good.report.substeps[0].position.penetration >= 0.00019
    strict, state, inputs = triangle_case(height=0.0098, speed=0.0, max_penetration_m=0.0001)
    bad = step(strict, state, inputs)
    assert not bad.report.accepted and "penetration limit" in bad.report.reason
    assert not bad.substep_states and np.array_equal(
        bad.state.domains["sheet"].position, state.domains["sheet"].position
    )


def test_rotating_patch_receives_the_world_point_moment():
    base, initial, inputs = triangle_case()
    axis = np.array([[0.0, 1.0, 0.0]])
    robot = ph.RobotModel(
        np.array([-1]),
        ("revolute",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        axis,
        np.ones(1),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    scene = ph.SceneModel(
        (
            ph.DomainBinding("sheet", "particles", base.cloth),
            ph.DomainBinding("plate", "robot", robot),
        )
    )
    program = ph.CompliantClothContactProgram(
        scene,
        "sheet",
        ClothMaterial(),
        base.surfaces,
        initial.domains["sheet"].position,
        base.patches,
    )
    h = 0.0005
    out = step(program, initial, inputs, h)
    assert out.report.accepted, out.report.reason
    row = out.report.substeps[0].diagnostics["patches"]["support"]
    torque = np.cross(row["points_m"], -row["force_n"]).sum(axis=0)[1]
    assert torque > 0
    assert out.state.domains["plate"].v[0] == pytest.approx(h * torque, abs=1e-14)
