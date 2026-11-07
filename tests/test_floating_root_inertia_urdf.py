"""URDF full root inertia: independent tensor and authored-frame acceptance."""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

import phymeo as ph

FIXTURES = Path(__file__).parent / "fixtures" / "urdf"
CENTER = np.array([1 / 8, -1 / 16, 1 / 32])
BODY_I = np.array([[3 / 8, -1 / 8, 0], [-1 / 8, 3 / 8, 0], [0, 0, 1 / 2]])
CYCLE = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])


def _rot(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-11)


def _tensor(model):
    r = _rot(model.root_inertia_frame_xyzw)
    return r @ np.diag(model.root.inertia_diagonal[0]) @ r.T


def _inertia(link, *, representation="tensor"):
    old = link.find("inertial")
    if old is not None:
        link.remove(old)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="2")
    ET.SubElement(
        inertial,
        "origin",
        xyz="0.125 -0.0625 0.03125",
        rpy="0 0 0" if representation == "tensor" else "0 0 0.7853981633974483",
    )
    if representation == "tensor":
        fields = dict(ixx=".375", iyy=".375", izz=".5", ixy="-.125", ixz="0", iyz="0")
    else:
        fields = dict(ixx=".25", iyy=".5", izz=".5", ixy="0", ixz="0", iyz="0")
    ET.SubElement(inertial, "inertia", **fields)


def _write(tmp_path, tree, name):
    path = tmp_path / (name + ".urdf")
    tree.write(path, encoding="unicode")
    return path


@pytest.mark.parametrize("representation", ["tensor", "rotated"])
@pytest.mark.parametrize("collapse", [False, True])
def test_full_root_urdf_keeps_authored_origin_collision_and_tensor(
    tmp_path, representation, collapse
):
    doc = ET.ElementTree(ET.Element("robot", name="full_root"))
    root = ET.SubElement(doc.getroot(), "link", name="base")
    _inertia(root, representation=representation)
    collision = ET.SubElement(root, "collision", name="authored_marker")
    ET.SubElement(collision, "origin", xyz=".25 -.125 .0625")
    ET.SubElement(ET.SubElement(collision, "geometry"), "sphere", radius=".0625")
    asset = ph.load_floating_urdf(
        _write(tmp_path, doc, "root"), gravity=(0, 0, 0), collapse_fixed_joints=collapse
    )
    model = asset.model
    _close(model.root.inverse_mass, [1 / 2])
    _close(model.root_com_m, CENTER)
    _close(_tensor(model), BODY_I)
    state = ph.FloatingRobotState(np.array([0.5, -0.25, 0.75, 0.5, 0.5, 0.5, 0.5]), np.zeros(6))
    geometry = ph.floating_kinematics(model, state)
    _close(geometry.position[0], state.q[:3])
    _close(geometry.rotation[0], CYCLE)
    col = asset.collisions[0]
    motion = asset.collision_motion(state.q)[col.shape.key]
    _close(motion.start, state.q[:3] + np.asarray(col.local_vertices) @ CYCLE.T)
    torque = (CYCLE @ [1.0, 0, 0])[None, :]
    terms = ph.floating_dynamics_terms(
        model, state, ph.FloatingRobotInputs(np.zeros(0), np.zeros((1, 3)), torque)
    )
    _close(terms.mass_matrix[3:, 3:], CYCLE @ BODY_I @ CYCLE.T)
    _close(terms.acceleration[3:], CYCLE @ [3.0, 1.0, 0.0])


@pytest.mark.parametrize("transformed", [False, True])
def test_offset_fixed_tool_products_keep_all_original_material_frames(tmp_path, transformed):
    doc = ET.parse(FIXTURES / "floating_fixed_composite.urdf")
    doc.find(".//joint/origin").set("xyz", ".25 .125 0")
    asset = ph.load_floating_urdf(
        _write(tmp_path, doc, "offset_tool"), gravity=(0, 0, 0), collapse_fixed_joints=True
    )
    offset = np.array([1 / 4, 1 / 8, 0])
    center = offset / 3
    # Independent two-body parallel-axis theorem, neither parser nor eigvectors.
    tensor = np.diag([0.3, 0.4, 0.5]) + (2 / 3) * (
        np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
    )
    _close(asset.model.root_com_m, center)
    _close(_tensor(asset.model), tensor)
    _close(asset.model.root.inverse_mass, [1 / 3])
    rotation = CYCLE if transformed else np.eye(3)
    quat = [0.5, 0.5, 0.5, 0.5] if transformed else [0, 0, 0, 1]
    shift = np.array([0.5, -0.25, 0.75]) if transformed else np.zeros(3)
    state = ph.FloatingRobotState(np.r_[shift, quat], np.zeros(6))
    assert asset.model.children is None and set(asset.sites) == {"base", "tool"}
    _close(asset.sites["base"].local_position_m, 0)
    _close(asset.authored_to_model_points("tool", [0, 0, 0]), offset)
    point, orient = asset.authored_link_pose("tool", state.q)
    _close(point, shift + rotation @ offset)
    _close(orient, rotation)
    for col in asset.collisions:
        local_offset = offset if col.link_name == "tool" else np.zeros(3)
        _close(
            asset.collision_motion(state.q)[col.shape.key].start,
            shift + (np.asarray(col.local_vertices) + local_offset) @ rotation.T,
        )
        assert col.link_index == -1 and col.shape.key.body == 0
    torque = rotation @ [0.125, 0.25, -0.0625]
    terms = ph.floating_dynamics_terms(
        asset.model, state, ph.FloatingRobotInputs(np.zeros(0), np.zeros((1, 3)), torque[None, :])
    )
    _close(terms.acceleration[3:], np.linalg.solve(rotation @ tensor @ rotation.T, torque))


@pytest.mark.parametrize("collapse", [False, True])
def test_affine_mimic_inherits_full_root_tensor_without_reframing_children(tmp_path, collapse):
    doc = ET.parse(FIXTURES / "floating_mimic_jaws.urdf")
    _inertia(doc.find(".//link[@name='base']"))
    asset = ph.load_floating_mimic_urdf(
        _write(tmp_path, doc, "mimic"), gravity=(0, 0, 0), collapse_fixed_joints=collapse
    )
    plain = ph.load_floating_mimic_urdf(
        FIXTURES / "floating_mimic_jaws.urdf", gravity=(0, 0, 0), collapse_fixed_joints=collapse
    )
    _close(_tensor(asset.tree.model), BODY_I)
    _close(asset.tree.model.root_com_m, CENTER)
    _close(asset.matrix, [[1], [-1]])
    _close(asset.offset_vector, [0, 0.004])
    state = ph.FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1, 0.001]), np.zeros(7))
    full = asset.full_state(state)
    _close(full.q[7:], [0.001, 0.003])
    _close(asset.velocity_projection[:6, :6], np.eye(6))
    left = asset.tree.collision_motion(full.q)
    right = plain.tree.collision_motion(plain.full_state(state).q)
    for key in left:
        _close(left[key].start, right[key].start)
    _close(
        ph.floating_kinematics(asset.tree.model, full).position,
        ph.floating_kinematics(plain.tree.model, plain.full_state(state)).position,
    )


@pytest.mark.parametrize("bad", ["negative", "triangle", "nan", "zero_mass", "missing"])
@pytest.mark.parametrize("collapse", [False, True])
def test_invalid_root_inertia_remains_rejected(tmp_path, bad, collapse):
    doc = ET.ElementTree(ET.Element("robot", name="invalid"))
    root = ET.SubElement(doc.getroot(), "link", name="base")
    _inertia(root)
    inertial = root.find("inertial")
    if bad == "missing":
        root.remove(inertial)
    elif bad == "zero_mass":
        inertial.find("mass").set("value", "0")
    else:
        inertial.find("inertia").set("ixx", {"negative": "-1", "triangle": "10", "nan": "nan"}[bad])
    with pytest.raises(ValueError, match="inertia|mass|finite"):
        ph.load_floating_urdf(
            _write(tmp_path, doc, bad), gravity=(0, 0, 0), collapse_fixed_joints=collapse
        )
