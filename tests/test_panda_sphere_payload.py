"""Freeze an actual uniform ball, curved witnesses and a full controlled grasp/slip pair."""

import json

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.experiments.panda_cube import run_panda_cube

BALL = {"kind": "sphere", "radius_m": .025}
MASS = .080


def test_oblique_ball_sample_and_lateral_spinning_table_contact():
    from phymeo.experiments.panda_payload import SampledPayload

    shape = SampledPayload(payload_geometry=BALL)
    law = RobotContactMaterial(200, 0, .5, .04)
    # 3-4-5 radial normal; the second point lies in the enclosing cube but outside the ball.
    points = np.array([[.012, .016, 0], [.022, .022, 0]])
    w = np.array([0., 0, 2])
    velocity = np.cross(w, points) + [0, 0, .02]
    load = shape.pad_loads(points, velocity, np.zeros(3), np.eye(3), np.zeros(3), w, law)
    np.testing.assert_allclose(load.depths, [.005, 0], atol=1e-14)
    np.testing.assert_allclose(load.forces, [[.6, .8, -.25], [0, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(load.body_force, [-.6, -.8, .25], atol=1e-14)
    np.testing.assert_allclose(load.body_torque, [.004, -.003, 0], atol=1e-14)
    with pytest.raises(ValueError, match="center"):
        shape.pad_loads(np.zeros((1,3)), np.zeros((1,3)), np.zeros(3), np.eye(3), np.zeros(3), w, law)
    # The instantaneous sphere witness remains at the world bottom with a tilted material frame.
    rotation = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    table_center, half = np.array([0,0,-.05]), np.array([.1,.1,.05])
    floor_law = RobotContactMaterial(4000,0,.5,.04)
    forces, normal, arms = shape.table_loads(np.array([0.,0,.024]), rotation,
        np.array([.08,0,0]), np.array([0.,2,0]), table_center, half, floor_law)
    np.testing.assert_allclose(arms, [[0,0,-.025]], atol=1e-15)
    np.testing.assert_allclose(normal, [4], atol=1e-13)
    np.testing.assert_allclose(forces, [[-1.5,0,4]], atol=1e-13)
    np.testing.assert_allclose(np.cross(arms, forces), [[0,.0375,0]], atol=1e-14)
    # The finite tabletop edge has a 3-4-5 oblique normal, then no force beyond the sphere.
    for x, expected in ((.112, [12,0,16]), (.13,[0,0,0])):
        forces, _, _ = shape.table_loads(np.array([x,0,.016]), np.eye(3), np.zeros(3),
            np.zeros(3),table_center,half,floor_law)
        np.testing.assert_allclose(forces[0], expected, atol=1e-12)
    np.testing.assert_allclose(shape.inertia(MASS), [2e-5]*3, atol=1e-16)


@pytest.mark.parametrize("bad", [{"kind":"sphere","radius_m":0}, {"kind":"sphere", "radius_m":float("nan")},
    {"kind":"sphere","radius_m":.025, "half_height":.1}, {"kind":"cylinder","radius_m":.025}])
def test_invalid_shape_fails_before_advance(bad):
    with pytest.raises(ValueError):
        run_panda_cube(payload_geometry=bad)


@pytest.mark.parametrize("label,mu,carry",[("grip",.65,True),("slip",0.,False)])
def test_ordinary_panda_sphere_window(label,mu,carry,tmp_path):
    result = run_panda_cube(payload_geometry=BALL, mass_kg=MASS, grip_above_com_m=.008,
                           closed_width_m=.024, friction=mu, record=True)
    spec, trace = result["inputs"], result["trace"]
    (tmp_path / f"sphere-{label}.json").write_text(json.dumps(result),encoding="utf8")
    (tmp_path / f"sphere-{label}.inputs.json").write_text(json.dumps(spec),encoding="utf8")
    assert result["payload_geometry"] == spec["payload_geometry"] == BALL
    assert result["steps"] == len(trace) == 3600 and trace[-1]["t"] == 2.25
    np.testing.assert_allclose(spec["inertia_diagonal_kg_m2"], [2e-5]*3, atol=1e-18, rtol=0)
    assert result["cube_mass_kg"] == MASS and spec["full_size_m"] == [.05]*3
    initial = np.array(spec["body_position_m"])
    top = spec["table_center_m"][2]+spec["table_full_size_m"][2]/2
    assert initial[2] - BALL["radius_m"] == pytest.approx(top)
    moving=trace[2159]
    assert moving["t"] == 1.35
    assert result["carry_clearance_m"] == pytest.approx(moving["cube"][2]-BALL["radius_m"]-top)
    target=initial+[.075,0,0]
    assert result["final_target_error_m"] == pytest.approx(np.linalg.norm(trace[-1]["cube"]-target))
    assert result["max_cube_momentum_residual_Ns"] < 1e-12 and result["max_joint_limit_violation"] < 1e-4
    assert result["peak_bilateral_normal_N"] > 1.0
    assert result["other_robot_cube_axis_clearance_m"] > 0 and result["robot_table_axis_clearance_m"] > 0
    assert result["release_pad_force_N"] < .001 and abs(result["final_table_force_N"]-MASS*9.81) < .025
    assert result["success"] is carry and result["carried_during_transfer"] is carry
    if carry:
        assert result["carry_clearance_m"] > .035 and result["carry_displacement_m"] > .045
        assert result["min_carry_bilateral_normal_N"] > 0 and result["peak_carry_table_N"] < .001
        assert result["released_on_target"] and result["final_target_error_m"] < .012
    else:
        assert result["carry_clearance_m"] < .005 and abs(result["carry_displacement_m"]) < .015
        assert result["min_carry_bilateral_normal_N"] == 0 and result["peak_carry_table_N"] > .65
        assert not result["released_on_target"]
