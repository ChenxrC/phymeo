"""An ordinary URDF can place inertial tooling and collision material on fixed branches."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from reference_newton.robot.fixed_tool_oracle import CASES, FORCE, MOTOR, H, reference

from phymeo.builders.urdf import load_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.rigid import RigidBodyState
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.core.scene import SceneInputs, SceneState
from phymeo.dynamics.robot_dynamics import forward_dynamics, robot_dynamics_terms
from phymeo.dynamics.robot_kinematics import forward_kinematics, point_jacobians
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.runtime.robot_payload_scene import RobotPayloadScene
from phymeo.solvers.robot import make_robot_runtime

ASSETS = Path(__file__).parent / "reference_newton" / "robot"
PROBE = ASSETS / "fixed_tool_probe.urdf"
PRESS = ASSETS / "fixed_tool_press.urdf"


@pytest.mark.parametrize("values", CASES)
def test_assembled_tool_keeps_authored_frames_mass_momentum_and_off_axis_force(values):
    asset = load_urdf(PROBE, gravity=(0, 0, 0))
    q, v = np.asarray(values[:2]), np.asarray(values[2:])
    expected = reference(q, v)
    assert asset.link_names == ("upper", "slide") and asset.joint_names == ("shoulder", "extension")
    np.testing.assert_array_equal(asset.model.parents, [-1, 0])
    assert asset.lower == (-2, -0.06) and asset.upper == (2, 0.12)
    assert asset.sites["pedestal"].link_index is None and asset.sites["elbow"].link_index == 0
    assert asset.sites["probe"].link_index == asset.sites["collet"].link_index == 1
    state = RobotState(q, v)
    geo, J, W, M, bias = robot_dynamics_terms(asset.model, state)
    for name, (p, R) in expected["frames"].items():
        actual_p, actual_R = asset.authored_link_pose(name, q)
        np.testing.assert_allclose(actual_p, p, atol=5e-14, rtol=0)
        np.testing.assert_allclose(actual_R, R, atol=5e-14, rtol=0)
    total = np.zeros(6)
    for i, (m, c, tensor) in enumerate(expected["combined"]):
        assert asset.model.mass[i] == m
        np.testing.assert_allclose(geo.com_position[i], c, atol=3e-14, rtol=0)
        Iw = geo.rotation[i] @ np.diag(asset.model.inertia_diagonal[i]) @ geo.rotation[i].T
        np.testing.assert_allclose(Iw, tensor, atol=6e-15, rtol=0)
        linear = m * (J[i] @ v)
        total += np.r_[linear, np.cross(geo.com_position[i], linear) + Iw @ (W[i] @ v)]
    np.testing.assert_allclose(total, expected["momentum"], atol=1e-13, rtol=0)
    np.testing.assert_allclose(M, expected["M"], atol=5e-14, rtol=0)
    np.testing.assert_allclose(bias, expected["bias"], atol=3e-14, rtol=0)
    assert 0.5 * v @ M @ v == pytest.approx(expected["energy"], abs=2e-14)
    solids = asset.collisions
    assert len({c.shape.key for c in solids}) == len(solids)
    tips = [c for c in solids if c.name == "tip"]
    assert len(tips) == 1 and tips[0].link_name == "probe"
    paths = asset.collision_motion(q)
    fixed = next(c for c in solids if c.name == "base_fiducial")
    np.testing.assert_allclose(paths[fixed.shape.key].start, [[0.25, -0.0625, 0.25]], atol=1e-16)
    tip = paths[tips[0].shape.key].start[0]
    np.testing.assert_allclose(tip, expected["tip"], atol=5e-14, rtol=0)
    native = asset.authored_to_model_points("probe", [[0.03125, 0, 0]])
    Jtip, Atip = point_jacobians(asset.model, q, [1], native)
    np.testing.assert_allclose(Jtip[0], expected["jacobian"], atol=5e-14, rtol=0)
    np.testing.assert_allclose(Jtip[0] @ v, expected["tip_velocity"], atol=5e-14, rtol=0)
    np.testing.assert_allclose(Atip[0] @ v, expected["omega"], atol=5e-14, rtol=0)
    force = np.zeros((2, 3))
    torque = force.copy()
    force[1] = FORCE
    torque[1] = np.cross(tip - geo.com_position[1], FORCE)
    loads = RobotInputs(MOTOR, force, torque)
    with make_robot_runtime(asset.model) as runtime:
        actual = runtime.step(SceneState({"robot": state}), SceneInputs({"robot": loads}), H)
    assert actual.report.accepted
    end = actual.state.domains["robot"]
    np.testing.assert_allclose(end.v, v + H * expected["acceleration"], rtol=0, atol=2e-14)
    np.testing.assert_allclose(end.q, q + H * end.v, rtol=0, atol=2e-14)
    print("Phy actual qdd", ((end.v - v) / H).tolist(), "M00", M[0, 0])
    np.testing.assert_allclose(
        forward_dynamics(asset.model, state, loads),
        expected["acceleration"],
        atol=2e-12,
        rtol=0,
    )


def test_two_files_share_the_same_live_scene_command_and_contact_path():
    for path, q in ((PROBE, np.array([0.37, 0.025])), (PRESS, np.array([0.025]))):
        asset = load_urdf(path, gravity=(0, 0, 0))
        tool = next(c for c in asset.collisions if c.name == "tip")
        center = asset.collision_motion(q)[tool.shape.key].start[0]
        direction = forward_kinematics(asset.model, q).axis[-1]
        ball = center + (tool.shape.radius + 0.020 - 0.001) * direction
        kwargs = dict(
            robot_state=RobotState(q, np.zeros_like(q)),
            payload_state=RigidBodyState(
                ball[None], np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, 3)), np.zeros((1, 3))
            ),
            payload_geometry={"kind": "sphere", "radius_m": 0.020},
            mass_kg=0.125,
            dt_s=0.001,
            kp=np.full(len(q), 400.0),
            kd=np.full(len(q), 6.0),
            structure_material=RobotContactMaterial(2000, 2, 0),
            table_center_m=(0, 0, -1),
            table_half_extents_m=(1, 1, 0.02),
        )
        ends = []
        for sign in (1, -1):
            scene = RobotPayloadScene.from_urdf(
                path, loader=lambda source: load_urdf(source, gravity=(0, 0, 0)), **kwargs
            )
            command = q.copy()
            command[-1] += 0.04 * sign
            first = scene.step(command)
            assert len(first.structural_contacts) == 1
            hit = first.structural_contacts[0]
            assert hit["shape"] == "tip" and hit["link"] == tool.link_name
            np.testing.assert_allclose(hit["force_on_payload_N"], 2 * direction, atol=1e-12)
            assert first.contact_effort[-1] == pytest.approx(-2, abs=1e-12)
            for _ in range(39):
                scene.step(command)
            ends.append(scene.observe())
        advance, back = ends
        delta = float(direction @ (advance.payload_position_m - back.payload_position_m))
        assert delta > 0.001
        assert advance.q[-1] > back.q[-1] + 0.003
        print(
            asset.name,
            advance.time_s,
            "actual command separation",
            advance.q[-1] - back.q[-1],
            "payload",
            delta,
        )


@pytest.mark.parametrize("failure", ["mimic", "inertia", "massless", "fixed_control"])
def test_fixed_subtrees_reject_unrepresented_controls_and_invalid_mass(tmp_path, failure):
    doc = ET.parse(PROBE)
    if failure == "mimic":
        ET.SubElement(doc.find(".//joint[@name='extension']"), "mimic", joint="shoulder")
        pattern = "mimic"
    elif failure == "inertia":
        doc.find(".//link[@name='probe']/inertial/inertia").set("ixy", "0.06")
        pattern = "probe.*inertia"
    elif failure == "massless":
        for name in ("slide", "collet", "probe"):
            link = doc.find(f".//link[@name='{name}']")
            link.remove(link.find("inertial"))
        pattern = "slide.*mass|slide.*inertia"
    else:
        ET.SubElement(doc.find(".//joint[@name='mount_collet']"), "limit", effort="200")
        pattern = "fixed.*limit|limit.*fixed"
    path = tmp_path / "bad-tool.urdf"
    doc.write(path)
    with pytest.raises(ValueError, match=pattern):
        load_urdf(path)


def test_strict_scalar_import_and_explicit_principal_axes_are_distinct(tmp_path):
    with pytest.raises(ValueError, match="fixed"):
        load_urdf(PROBE, collapse_fixed_joints=False)
    doc = ET.parse(ASSETS / "arm.urdf")
    doc.find(".//link[@name='boom']/inertial/inertia").set("ixy", "0.01")
    path = tmp_path / "full-tensor.urdf"
    doc.write(path)
    with pytest.raises(ValueError, match="non-diagonal"):
        load_urdf(path)
    asset = load_urdf(path, collapse_fixed_joints=True)
    q = np.zeros(2)
    geo = forward_kinematics(asset.model, q)
    tensor = geo.rotation[0] @ np.diag(asset.model.inertia_diagonal[0]) @ geo.rotation[0].T
    np.testing.assert_allclose(
        tensor, [[0.1, 0.01, 0], [0.01, 0.1, 0], [0, 0, 0.1]], atol=5e-14, rtol=0
    )
    np.testing.assert_allclose(asset.authored_link_pose("boom", q)[1], np.eye(3), atol=8e-15)


def test_curved_fixed_tool_preserves_authored_orientation_and_local_material_motion(tmp_path):
    doc = ET.parse(PROBE)
    geometry = doc.find(".//link[@name='probe']/collision/geometry")
    geometry.remove(geometry[0])
    ET.SubElement(geometry, "cylinder", radius="0.025", length="0.07")
    path = tmp_path / "barrel.urdf"
    doc.write(path)
    asset = load_urdf(path)
    q = np.array([0.37, 0.025])
    expected = reference(q, [0, 0])
    col = next(c for c in asset.collisions if c.name == "tip")
    motion = asset.collision_motion(q)[col.shape.key]
    np.testing.assert_allclose(motion.start[0], expected["tip"], atol=5e-14)
    np.testing.assert_allclose(
        quaternion_matrix(np.array([motion.body_orientation_xyzw]))[0],
        expected["frames"]["probe"][1],
        atol=5e-14,
    )
    arc = asset.collision_motion(q, np.array([q[0] + 0.19, q[1]]))[col.shape.key]
    assert arc.turn.angle == pytest.approx(0.19)


def test_massless_moving_frame_is_supported_by_its_actual_massive_fixed_tool(tmp_path):
    doc = ET.parse(PRESS)
    carriage = doc.find("./link[@name='ram']")
    carriage.remove(carriage.find("inertial"))
    path = tmp_path / "mass-in-tool.urdf"
    doc.write(path)
    asset = load_urdf(path, gravity=(0, 0, 0))
    assert asset.model.mass.tolist() == [0.25]
    state = RobotState(np.array([0.02]), np.array([0.0]))
    origin, R = asset.authored_link_pose("head", state.q)
    read = forward_kinematics(asset.model, state.q)
    np.testing.assert_allclose(
        read.com_position[0], origin + R @ [0.015625, 0, 0], atol=2e-14, rtol=0
    )
    loading = RobotInputs(np.zeros(1), np.array([[0.0, 0.0, 1.0]]), np.zeros((1, 3)))
    np.testing.assert_allclose(forward_dynamics(asset.model, state, loading), [4.0], atol=2e-14)
