"""Same-instant frame observations compared with frozen independent site geometry."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.solvers.driven_robot import DrivenRobotProgram


CASES = json.loads((Path(__file__).parent / "reference_newton" / "robot" / "site_frames.json").read_text())
FIELDS = ("position_m", "rotation", "linear_velocity_m_s", "angular_velocity_rad_s")


def api():
    from phymeo.sensors import RobotSite, measure_relative_site

    return RobotSite, measure_relative_site


def physical_model(fixture):
    count = len(fixture["parents"])
    return ph.RobotModel(
        parents=np.array(fixture["parents"]), joint_types=tuple(fixture["kinds"]),
        joint_position=np.array(fixture["anchors"], dtype=float),
        joint_orientation=np.tile([0., 0, 0, 1], (count, 1)),
        joint_axis=np.array(fixture["axes"], dtype=float), mass=np.array(fixture["mass_kg"], dtype=float),
        com=np.array(fixture["com_m"], dtype=float), inertia_diagonal=np.ones((count, 3)),
        gravity=np.zeros(3),
    )


def bindings(row):
    site, _ = api()
    return (physical_model(CASES["fixtures"][row["model"]]),
            ph.RobotState(np.array(row["q"], dtype=float), np.array(row["v"], dtype=float)),
            site(*row["reference"]), site(*row["target"]))


def compare(row, result, tol):
    for field in FIELDS:
        value = getattr(result, field)
        assert value.shape == np.shape(row[field])
        np.testing.assert_allclose(value, row[field], rtol=0, atol=tol, err_msg=f"{row['id']}/{field}")


@pytest.mark.parametrize("row", CASES["cases"], ids=lambda case: case["id"])
def test_independent_relative_measurements(row):
    _, measure = api()
    model, state, reference, target = bindings(row)
    result = measure(model, state, reference, target)
    compare(row, result, 2e-12)
    assert not result.position_m.flags.writeable
    assert not result.rotation.flags.writeable
    reversed_state = ph.RobotState(state.q, -state.v)
    reverse = measure(model, reversed_state, reference, target)
    for field in FIELDS:
        sign = -1 if "velocity" in field else 1
        np.testing.assert_allclose(getattr(reverse, field), sign * np.asarray(row[field]), atol=2e-12, rtol=0)
    np.testing.assert_array_equal(state.q, row["q"])
    np.testing.assert_array_equal(state.v, row["v"])


def test_oblique_sites_have_independently_known_world_point_velocities():
    model, state, reference, target = bindings(CASES["cases"][-1])
    for site, expected in ((reference, [-0.1, 0.075, 0]), (target, [0.125, -0.03125, 0])):
        port = ph.RobotPointPort(model, state, [site.link_index], np.array([site.local_position_m]))
        np.testing.assert_allclose(port.velocities()[0], expected, rtol=0, atol=2e-12)


def test_only_published_delayed_states_reach_the_sensor():
    site, measure = api()
    model = replace(physical_model(CASES["fixtures"]["hinge"]), joint_types=("prismatic",),
                    joint_axis=np.array([[1., 0, 0]]), mass=np.array([2.]))
    p = DrivenRobotProgram(model, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([2.]), delay_steps=2)
    start = p.initial_state(ph.RobotState(np.zeros(1), np.zeros(1)))
    command = p.inputs(np.array([1.]), feedforward=np.array([2.]))
    settings = ph.StepSettings(substeps=3, record_substeps=True)
    with p.stepper(settings) as rt:
        out = rt.step(start, command, 3/16)
    assert out.report.accepted
    world, tool = site(None), site(0)
    def observe(state):
        reading = measure(model, state.domains["robot"], world, tool)
        return reading.position_m[0], reading.linear_velocity_m_s[0]
    got = [observe(s) for s in out.substep_states]
    np.testing.assert_allclose(got, [(0,0), (0,0), (1/256,1/16)], atol=2e-12, rtol=0)

    class RefuseMotion:
        def problem_signature(self):
            return p.problem_signature()
        def verify(self, scene, before, inputs, h, candidate):
            return ph.VerificationReport(candidate.domains["robot"].v[0] == 0,
                                         reason="reject first moving candidate")
    with p.stepper(settings, verifier=RefuseMotion()) as rt:
        rejected = rt.step(start, command, 3/16)
    assert not rejected.report.accepted and rejected.report.completed_substeps == 2
    assert rejected.substep_states == ()
    assert observe(rejected.state) == (0., 0.)


def test_site_input_ownership_and_contract():
    site, measure = api()
    point = np.array([1., 2, 3])
    fixed = site(None, point)
    point[:] = 99
    np.testing.assert_array_equal(fixed.local_position_m, [1,2,3])
    model, state, _, target = bindings(CASES["cases"][1])
    with pytest.raises(ValueError, match="link"):
        measure(model, state, site(3), target)
    with pytest.raises(ValueError, match="orientation|quaternion"):
        site(0, [0,0,0], [0,0,0,2])
    with pytest.raises((TypeError, ValueError), match="link"):
        site(True)
    with pytest.raises(ValueError, match="position"):
        site(0, [0, np.nan, 0])
    with pytest.raises(ValueError, match="backend"):
        measure(model, state, fixed, target, backend="automatic")


@pytest.mark.tyx
def test_oblique_branch_device_readback(tyx_mode, tmp_path):
    _, measure = api()
    row = CASES["cases"][-1]
    result = measure(*bindings(row), backend=tyx_mode, cache_root=tmp_path)
    compare(row, result, 2e-5)
    assert result.kernel_stats["completed_launches"] >= 2
    assert result.kernel_stats["mode"] == tyx_mode
    if tyx_mode == "cuda":
        assert result.kernel_stats["uploads"] > 0 and result.kernel_stats["downloads"] > 0
    print(json.dumps(dict(case=row["id"], stats=dict(result.kernel_stats),
                          observed={f: getattr(result,f).tolist() for f in FIELDS})))
