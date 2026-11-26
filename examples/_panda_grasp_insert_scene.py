"""Dual Panda cell for gripping, transporting and seating a free shouldered pin.

The left board is bolted to a lateral wrist bracket. On the right, two original
Panda finger bodies slide independently. Their two URDF rubber pad boxes are
the actual contact geoms. The unactuated workpiece starts on a fixed fork.
"""

from dataclasses import dataclass
import hashlib
from xml.etree import ElementTree as ET

import numpy as np

from _panda_assembly_scene import (
    AssemblyScene, HOME, _numbers, _origin, _pose_attributes, _quaternion, _source,
    build_scene as build_original_scene, forward_kinematics,
)


OPEN_JAW_M = .035   # Per finger, measured from the fully closed joint origin.
GRASP_HAND_Z = .10365
SHOULDER_FACE_Z = .019
TIP_Z = .029
PICKUP_OFFSET_WORLD = (0., -.145, .028)
PAD_NAMES = ("right_left_pad", "right_right_pad")
RAIL_NAMES = ("pickup_rail_left", "pickup_rail_right")

# Collision categories. Structural finger boxes can load the workpiece. Only
# rubber pads participate in finger contacts with the fixture and surroundings.
BOARD, PART, PAD, SUPPORT, FINGER_STRUCTURE = 1, 2, 4, 8, 16


@dataclass(frozen=True)
class GraspInsertScene(AssemblyScene):
    def initial_state(self):
        from phymeo.backends.mujoco_runtime import MuJoCoInitialState
        return MuJoCoInitialState(q=self.initial_qpos, v=(0.,)*22, time=0.)


def _contact(geom, category, partners):
    geom.set("contype", str(category))
    geom.set("conaffinity", str(partners))
    return geom


def _site(body, name, pos=(0., 0., 0.)):
    ET.SubElement(body, "site", name=name, pos=_numbers(pos), size=".001", rgba="0 0 0 0")


def _box(body, name, pos, half, mass, color, category, partners):
    geom = ET.SubElement(body, "geom", name=name, type="box", pos=_numbers(pos), size=_numbers(half),
                         mass=str(mass), rgba=color)
    return _contact(geom, category, partners)


def build_scene(*, urdf_path=None, timestep=.002):
    original = build_original_scene("pin", urdf_path=urdf_path, timestep=timestep)
    _, _, links, joints = _source(original.urdf_path)
    frames = forward_kinematics(original, original.initial_qpos)
    root = ET.fromstring(original.xml)
    root.set("model", "dual_panda_grasp_insert")
    world = root.find("worldbody")
    motors = root.find("actuator")

    hand = world.find(".//body[@name='right_panda_hand']")
    hand.remove(hand.find("body[@name='right_assembly_head']"))
    _site(hand, "right_grasp_center", (0., 0., GRASP_HAND_Z))
    finger_joints = []
    for index, (side, pad_name) in enumerate(zip(("left", "right"), PAD_NAMES, strict=True), start=1):
        urdf_joint = joints[f"panda_finger_joint{index}"]
        link = urdf_joint.find("child").get("link")
        finger = hand.find(f"body[@name='right_{link}']")
        # The original fixed-open scene bakes in a 30 mm offset. Put the body
        # back at its authored closed-jaw origin before inserting the slide.
        finger.attrib.update(_pose_attributes(_origin(urdf_joint)))
        limit = urdf_joint.find("limit")
        joint_name = "right_"+urdf_joint.get("name")
        finger_joints.append(joint_name)
        finger.insert(0, ET.Element("joint", name=joint_name, type="slide",
                      axis=urdf_joint.find("axis").get("xyz"), limited="true",
                      range=f"{limit.get('lower')} {limit.get('upper')}",
                      damping=urdf_joint.find("dynamics").get("damping")))
        limit_n = float(limit.get("effort"))
        ET.SubElement(motors, "motor", name=joint_name+"_motor", joint=joint_name,
                      gear="1", ctrllimited="true", ctrlrange=_numbers((-limit_n, limit_n)))
        for geom, src in zip(finger.findall("geom"), links[link].findall("collision"), strict=True):
            if src.get("name") == f"panda_{side}finger_pad":
                geom.set("name", pad_name)
                geom.set("rgba", ".12 .13 .14 1")
                geom.set("priority", "1")
                geom.set("friction", ".75 .005 .0001")
                _contact(geom, PAD, PART | BOARD | SUPPORT)
            else:
                _contact(geom, FINGER_STRUCTURE, PART)

    fixture = world.find(".//body[@name='left_board_fixture']")
    for geom in fixture.findall("geom"):
        _contact(geom, BOARD, PART | PAD)
    carrier = world.find(".//body[@name='left_board_carrier']")
    bar = carrier.find("geom[@name='board_carrier']")
    # End at the near edge of the 120 mm wide board, below its working face.
    # The old display-only bar ended at the center of the through hole.
    bar.set("fromto", "0 0 .080 .112 0 .132")
    bar.set("size", ".009")
    bar.set("mass", ".04")
    _contact(bar, BOARD, PART | PAD)
    _box(carrier, "board_edge_clamp", (.124, 0., .129), (.015, .020, .006), .020,
         ".33 .36 .39 1", BOARD, PART | PAD)

    part = world.find("body[@name='assembly_part']")
    for elem in list(part):
        if elem.tag != "freejoint":
            part.remove(elem)
    pickup_pose = frames["right_panda_hand"].copy()
    pickup_pose[:3, 3] = frames["assembly_hole"][:3, 3]+np.asarray(PICKUP_OFFSET_WORLD)
    if not np.allclose(pickup_pose[:3, 2], [0., 0., -1.], atol=1e-7):
        raise ValueError("the pickup fork requires a vertical neutral pin axis")
    part.attrib.update(_pose_attributes(pickup_pose))
    partners = BOARD | PAD | SUPPORT | FINGER_STRUCTURE
    # The three declared component masses sum to 120 g. Let MuJoCo compute
    # the composite COM/inertia from the actual primitive mass distribution.
    _box(part, "part_handle", (0., 0., 0.), (.010, .012, .013), .0786,
         ".47 .51 .55 1", PART, partners)
    _box(part, "part_shoulder", (0., 0., .016), (.015, .015, .003), .0372,
         ".72 .57 .28 1", PART, partners)
    shaft = ET.SubElement(part, "geom", type="capsule", name="assembly_pin",
                          fromto="0 0 .019 0 0 .025", size=".004", mass=".0042", rgba=".75 .78 .80 1")
    _contact(shaft, PART, partners)
    _site(part, "part_grasp_center")
    _site(part, "assembly_pin_tip", (0., 0., TIP_Z))
    _site(part, "assembly_pin_axis")
    _site(part, "part_shoulder_seat", (0., 0., SHOULDER_FACE_Z))

    cradle = ET.SubElement(world, "body", name="pickup_cradle", **_pose_attributes(pickup_pose))
    for sign, name in zip((-1., 1.), RAIL_NAMES, strict=True):
        _box(cradle, name, (sign*.011, 0., .029), (.0045, .020, .010), 0.,
             ".31 .37 .40 1", SUPPORT, PART | PAD)
    # Top of the pedestal is 10 mm below the shaft tip. Two rails spaced
    # 13 mm at the inside faces bear the shoulder across its center of mass.
    tabletop_z = -.005  # upper surface of assembly_table_visual in the original XML
    base_top = .039
    base_bottom = pickup_pose[2, 3]-tabletop_z
    if base_bottom <= base_top:
        raise ValueError("pickup pedestal cannot reach the work surface")
    _box(cradle, "pickup_base", (0., 0., (base_bottom+base_top)/2),
         (.025, .025, (base_bottom-base_top)/2), 0., ".26 .30 .33 1", SUPPORT, PART | PAD)
    _site(cradle, "pickup_grasp_center")
    _contact(world.find("geom[@name='assembly_table_visual']"), SUPPORT, PART | PAD)

    initial = (*HOME, OPEN_JAW_M, OPEN_JAW_M, *pickup_pose[:3, 3], *_quaternion(pickup_pose[:3, :3]))
    root.find("keyframe/key[@name='home']").set("qpos", _numbers(initial))
    xml = ET.tostring(root, encoding="unicode")
    provenance = dict(schema="dual-panda-grasp-insert-v1", source_scene_sha256=original.xml_sha256,
                      joint_order=[name for arm in original.arm_joint_names for name in arm]+finger_joints,
                      free_qpos_order="after 14 arm hinges and 2 finger slides: xyz,wxyz",
                      right_fingers="independently motorized original URDF slides and box pads",
                      left_fixture="bolted edge bracket; left fingers fixed open",
                      part="free shouldered pin; no attachment, spring coupling, thread, or pose servo",
                      part_mass_kg=.120, part_frame_to_tip_m=(0., 0., TIP_Z),
                      trial_friction=.75, shoulder_seated_tip_depth_m=TIP_Z-SHOULDER_FACE_Z,
                      pickup_offset_world_m=PICKUP_OFFSET_WORLD,
                      collision_scope="workpiece versus pads, right finger solids, board, bracket, fork and table; "
                                      "pads versus board, bracket, fork and table; arm envelopes visual only")
    return GraspInsertScene("grasp_insert", xml, hashlib.sha256(xml.encode()).hexdigest(), original.urdf_path,
                            original.urdf_sha256, tuple(map(float, initial)), original.arm_joint_names,
                            original.base_transforms, provenance, timestep)
