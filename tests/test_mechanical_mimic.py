"""A passive mimic transmits loads through a holonomic constraint with only one motor."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from reference_newton.robot.mimic_oracle import H, physical

from phymeo.builders.urdf import load_urdf
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.dynamics.robot_kinematics import point_jacobians

ROOT = Path(__file__).parent / "reference_newton" / "robot"


@pytest.mark.parametrize("case", ["jaws", "elbow"])
def test_passive_gearing_matches_full_saddle_point_and_reciprocal_work(case):
    from phymeo.builders.mimic_urdf import load_mimic_urdf

    asset = load_mimic_urdf(ROOT / f"mimic_{case}.urdf", gravity=(0, 0, 0))
    ref = physical(case)
    ids = asset.independent_indices
    state = RobotState(ref["q"][list(ids)], ref["v"][list(ids)])
    full = asset.full_state(state)
    np.testing.assert_allclose(full.q, ref["q"], atol=2e-15)
    np.testing.assert_allclose(full.v, ref["v"], atol=2e-15)
    np.testing.assert_array_equal(asset.matrix, ref["T"])
    reduced_M, reduced_b = asset.mass_bias(state)
    np.testing.assert_allclose(reduced_M, ref["T"].T @ ref["mass"] @ ref["T"], atol=5e-14)
    np.testing.assert_allclose(reduced_b, ref["T"].T @ ref["bias"], atol=4e-14)
    name, point_local = (
        ("right", [[-0.003, 0.006, -0.004]]) if case == "jaws" else ("tool", [[0.22, 0.06, -0.03]])
    )
    tree = asset.tree
    i = tree.link_names.index(name)
    point = tree.authored_to_model_points(name, point_local)
    J, _ = point_jacobians(tree.model, full.q, [i], point)
    np.testing.assert_allclose(asset.project_jacobian(J)[0], ref["J"] @ ref["T"], atol=3e-14)
    full_load = RobotInputs.zeros(len(full.q))
    full_load.force[i] = ref["force"]
    full_load.torque[i] = ref["torque"]
    answer = asset.step(state, effort=ref["motor"][list(ids)], external=full_load, dt_s=H)
    np.testing.assert_allclose(answer.full_state.v, ref["v"] + H * ref["accel"], atol=2e-14)
    np.testing.assert_allclose(answer.full_state.q, ref["q"] + H * answer.full_state.v, atol=2e-14)
    loads = answer.loads
    np.testing.assert_allclose(loads.constraint_effort_full, ref["reaction"], atol=3e-13)
    np.testing.assert_allclose(loads.contact_effort_full, ref["external"], atol=3e-14)
    np.testing.assert_array_equal(loads.actuator_effort_full, ref["motor"])
    np.testing.assert_allclose(ref["T"].T @ loads.constraint_effort_full, 0, atol=3e-14)
    assert abs(loads.constraint_power_W) < 1e-14
    assert loads.multiplier_force == pytest.approx(ref["reaction"][asset.follower_index], abs=3e-13)
    assert loads.actuator_effort_full[asset.follower_index] == 0
    # The analytic gear reaction lies below the passive motor's rating, the rail
    # reaction exceeds it. Both transmissions obey the same unconstrained force balance.
    exceeds = abs(loads.multiplier_force) > tree.effort_limit[asset.follower_index]
    assert bool(exceeds) == (case == "jaws")
    print(
        case,
        "qdd",
        loads.acceleration_full.tolist(),
        "constraint",
        loads.constraint_effort_full.tolist(),
    )


def test_effective_follower_limits_and_failed_input_preserve_state():
    from phymeo.builders.mimic_urdf import load_mimic_urdf

    asset = load_mimic_urdf(ROOT / "mimic_jaws.urdf", gravity=(0, 0, 0))
    assert asset.joint_names == ("lift", "screw")
    np.testing.assert_allclose(asset.lower, [-0.01, -0.005])
    np.testing.assert_allclose(asset.upper, [0.09, 0.022])
    np.testing.assert_allclose(asset.velocity_limit, [0.45, 0.20])
    np.testing.assert_allclose(asset.effort_limit, [40, 9])
    state = RobotState(np.array([0.0, 0.02199]), np.array([0.0, 0.10]))
    with pytest.raises(ValueError, match="position|interval"):
        asset.step(state, effort=[0, 0], dt_s=0.001)
    np.testing.assert_array_equal(state.q, [0, 0.02199])
    with pytest.raises(ValueError, match="velocity"):
        asset.full_state(RobotState(np.zeros(2), np.array([0.0, 0.21])))
    with pytest.raises(ValueError, match="consistent|constraint"):
        asset.reduce_state(RobotState(np.array([0.0, 0.01, 0.01]), np.zeros(3)))
    bad = asset.full_state(RobotState(np.zeros(2), np.zeros(2)))
    bad.v[-1] = 0.1
    with pytest.raises(ValueError, match="consistent|constraint"):
        asset.reduce_state(bad)
    with pytest.raises(ValueError):
        asset.matrix.setflags(write=True)
    elbow = load_mimic_urdf(ROOT / "mimic_elbow.urdf")
    np.testing.assert_allclose(elbow.lower, [-0.25])
    np.testing.assert_allclose(elbow.upper, [5 / 12])


@pytest.mark.parametrize(
    "wrong",
    ["multi_cycle", "cycle", "unknown", "mixed", "zero", "empty", "unknown_control", "no_mimic"],
)
def test_rejects_ambiguous_and_out_of_domain_linkages(tmp_path, wrong):
    from phymeo.builders.mimic_urdf import load_mimic_urdf

    doc = ET.parse(ROOT / "mimic_jaws.urdf")
    follower = doc.find(".//joint[@name='idler']")
    entry = follower.find("mimic")
    if wrong == "multi_cycle":
        ET.SubElement(doc.find(".//joint[@name='screw']"), "mimic", joint="idler")
    elif wrong == "cycle":
        entry.set("joint", "idler")
    elif wrong == "unknown":
        entry.set("joint", "absent")
    elif wrong == "mixed":
        follower.set("type", "revolute")
    elif wrong == "zero":
        entry.set("multiplier", "0")
    elif wrong == "empty":
        entry.set("offset", "3")
    elif wrong == "unknown_control":
        ET.SubElement(follower, "dynamics", damping="2")
    else:
        follower.remove(entry)
    path = tmp_path / "invalid.urdf"
    doc.write(path)
    with pytest.raises(ValueError, match="mimic|intersection|dynamics|leader|follower"):
        load_mimic_urdf(path)
    if wrong not in ("no_mimic",):
        with pytest.raises(ValueError):
            load_urdf(path)


def test_rejected_scene_candidate_keeps_free_body_and_whole_mechanism_then_retries():
    from phymeo.builders.mimic_urdf import load_mimic_urdf
    from phymeo.core.rigid import RigidBodyState
    from phymeo.runtime.robot_payload_scene import RobotPayloadScene

    asset = load_mimic_urdf(ROOT / "mimic_jaws.urdf", gravity=(0, 0, 0))
    kwargs = dict(
        robot_state=RobotState(np.array([0.0, 0.021955]), np.array([0.0, 0.06])),
        payload_state=RigidBodyState(
            np.array([[0.1, 0.1, 0.6]]),
            np.array([[0.0, 0.0, 0.0, 1.0]]),
            np.array([[0.01, 0.0, 0.0]]),
            np.zeros((1, 3)),
        ),
        payload_geometry={"kind": "sphere", "radius_m": 0.02},
        mass_kg=0.1,
        dt_s=0.001,
        kp=(100, 3000),
        kd=(10, 35),
        table_center_m=(0, 0, -0.25),
        table_half_extents_m=(0.3, 0.3, 0.01),
    )
    scene = RobotPayloadScene(asset, **kwargs)
    other = RobotPayloadScene(asset, **kwargs)
    before = scene.observe()
    expected_geometry = scene.geometry()
    with pytest.raises(ValueError, match="position|interval"):
        scene.step([0, -0.004], effort_limit=[0, 0])
    assert scene.observe() is before and before.mimic_loads is None
    np.testing.assert_array_equal(
        scene.geometry().poses[expected_geometry.payload_key].start,
        expected_geometry.poses[expected_geometry.payload_key].start,
    )
    good = scene.step([0, -0.004])
    direct = other.step([0, -0.004])
    assert good.time_s == 0.001 and good.full_q.shape == (3,) and good.q.shape == (2,)
    for name in ("full_q", "full_v", "payload_position_m", "contact_effort"):
        np.testing.assert_array_equal(getattr(good, name), getattr(direct, name))
    np.testing.assert_array_equal(
        good.mimic_loads.constraint_effort_full, direct.mimic_loads.constraint_effort_full
    )
    with pytest.raises(ValueError):
        good.mimic_loads.constraint_effort_full.setflags(write=True)
