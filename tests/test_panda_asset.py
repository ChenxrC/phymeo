"""Independent URDF transport, full inertial tensors and mirrored Panda hand surfaces."""

import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from phymeo.builders.panda import PANDA_URDF, load_panda
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.dynamics.rotation import quaternion_matrix

READY = np.array([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4])
HOLDOUT = np.array([0.43, 0.19, -0.35, -1.83, 0.72, 1.46, -0.6])


def euler(rpy):
    x, y, z = rpy
    cx, cy, cz = np.cos(rpy)
    sx, sy, sz = np.sin(rpy)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


def reference(q):
    # Independent homogeneous chain from the classic Panda drawing/description.
    origins = [(0, 0, .333), (0, 0, 0), (0, -.316, 0), (.0825, 0, 0),
               (-.0825, .384, 0), (0, 0, 0), (.088, 0, 0)]
    turns = [0, -1, 1, 1, -1, 1, 1]
    pos, rot = np.zeros(3), np.eye(3)
    frames = {}
    for i, (origin, turn, angle) in enumerate(zip(origins, turns, q[:7], strict=True)):
        pos = pos + rot @ origin
        rot = rot @ euler([turn * math.pi / 2, 0, 0]) @ euler([0, 0, angle])
        frames[f"panda_link{i + 1}"] = pos, rot
    flange = pos + rot @ [0, 0, .107]
    frames["panda_link8"] = flange, rot
    hand = rot @ euler([0, 0, -math.pi / 4])
    frames["panda_hand"] = flange, hand
    frames["panda_hand_tcp"] = flange + hand @ [0, 0, .1034], hand
    for sign, name, finger_q in zip((1, -1), ("left", "right"), q[7:], strict=True):
        finger = flange + hand @ [0, sign * finger_q, .0584]
        frames[f"panda_{name}finger"] = finger, hand
        frames[f"panda_{name}pad"] = finger + hand @ [0, -sign * .00002, .04525], hand
    return frames


@pytest.mark.parametrize("arm,width", [(READY, 0), (HOLDOUT, .027), (READY, .08)])
def test_loader_reproduces_all_authored_frames_and_rubber_pad_gap(arm, width):
    asset = load_panda()
    q = asset.configuration(arm, width)
    np.testing.assert_array_equal(q, np.r_[arm, width / 2, width / 2])
    assert asset.joint_names == tuple(f"panda_joint{i}" for i in range(1, 8)) + (
        "panda_finger_joint1", "panda_finger_joint2"
    )
    measured = {}
    for name, (p, r) in reference(q).items():
        x, m = asset.authored_link_pose(name, q)
        np.testing.assert_allclose(x, p, atol=3e-13, rtol=0)
        np.testing.assert_allclose(m, r, atol=3e-13, rtol=0)
        measured[name] = x
    normal = reference(q)["panda_hand"][1][:, 1]
    assert (measured["panda_leftpad"] - measured["panda_rightpad"]) @ normal == pytest.approx(
        width - 0.00004, abs=3e-13
    )
    paths = asset.collision_motion(q)
    for sign, side in ((1, "left"), (-1, "right")):
        body = f"panda_{side}finger"
        pad = next(c for c in asset.collisions if c.name == f"{body}_pad")
        verts = paths[pad.shape.key].start
        tip, hand = reference(q)[body]
        local = (verts - tip) @ hand
        np.testing.assert_allclose(np.ptp(local, axis=0), [.0175, .0152, .0185], atol=3e-13)
        assert (sign * local[:, 1]).min() == pytest.approx(-.00002, abs=3e-13)
        assert asset.sites[f"panda_{side}pad"].link_index == pad.link_index


def test_mass_properties_include_tensor_products_and_welded_hand():
    arm = load_panda()
    q = arm.configuration(HOLDOUT, .04)
    fk = forward_kinematics(arm.model, q)
    assert arm.model.mass[0] == pytest.approx(4.970684)
    assert arm.model.mass[6] == pytest.approx(.735522 + .73)
    np.testing.assert_allclose(arm.model.mass[-2:], [.015, .015])
    world_p, world_r = reference(q)["panda_link1"]
    inertia = np.array([[.70337, -.000139, .006772], [-.000139, .70661, .019169],
                        [.006772, .019169, .009117]])
    measured = fk.rotation[0] @ np.diag(arm.model.inertia_diagonal[0]) @ fk.rotation[0].T
    np.testing.assert_allclose(measured, world_r @ inertia @ world_r.T, atol=1e-13)
    np.testing.assert_allclose(fk.com_position[0], world_p + world_r @ [.003875, .002081, -.04762])


def test_offset_base_and_local_file_actually_change_fk(tmp_path):
    doc = ET.parse(PANDA_URDF)
    shoulder = doc.find("./joint[@name='panda_joint1']/origin")
    shoulder.set("xyz", "0 0 0.458")
    filename = tmp_path / "same-topology.urdf"
    doc.write(filename)
    quat = np.full(4, .5)
    base_xyz = np.array([.041, -.213, .377])
    asset = load_panda(filename, base_xyz=base_xyz, base_xyzw=quat)
    q = asset.configuration(HOLDOUT, .05)
    p0, r0 = reference(q)["panda_hand_tcp"]
    p, r = asset.authored_link_pose("panda_hand_tcp", q)
    base = quaternion_matrix(quat[None, :])[0]
    np.testing.assert_allclose(p, base_xyz + base @ (p0 + [0, 0, .125]), atol=3e-13)
    np.testing.assert_allclose(r, base @ r0, atol=3e-13)
    site = asset.sites["panda_hand_tcp"]
    fk = forward_kinematics(asset.model, q)
    tool_point = np.array([.01, -.015, .032])
    native = asset.authored_to_model_points("panda_hand_tcp", tool_point)
    i = site.link_index
    np.testing.assert_allclose(fk.position[i] + fk.rotation[i] @ native, p + r @ tool_point)


@pytest.mark.parametrize("width", [-.001, .081, np.nan])
def test_configuration_rejects_invalid_width(width):
    with pytest.raises(ValueError):
        load_panda().configuration(READY, width)


def test_invalid_mimic_is_not_silently_interpreted_as_symmetric(tmp_path):
    doc = ET.parse(PANDA_URDF)
    doc.find("./joint[@name='panda_finger_joint2']/mimic").set("multiplier", "-1")
    path = Path(tmp_path) / "different-mimic.urdf"
    doc.write(path)
    with pytest.raises(ValueError, match="mimic"):
        load_panda(path)
