"""Source-bound dual Panda assembly fixture; no simulation occurs on import.

Both arms use the bundled Franka Panda URDF origins, axes, limits and inertias.
The board is a rigid wrist tool; the pin is an independent free rigid body.
The right wrist only supplies a driver site. A runner must explicitly apply
compliant, equal-and-opposite coupling loads; this module adds no weld or servo
that moves the free part. The square hole has four physical box walls.
The screw_proxy MJCF supplies the geometry. During each rollout the training
runner adds a compliant lead and rotational drag for an already-engaged part.
This reduced model has no individual thread flanks or final torque seating.

Only board/pin contacts are enabled. Robot collision envelopes are visual
proxies; self/arm/table collisions are excluded. This is a task-local assembly
model, not a collision-certified cell. A controller must drive the 14 motors;
FK and target poses do not move the simulator or certify successful insertion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
NEUTRAL = (0., -math.pi/4, 0., -3*math.pi/4, 0., math.pi/2, math.pi/4)
HOME = np.array(NEUTRAL * 2, dtype=float)
HOME.setflags(write=False)
# Tip is 4 mm past the mid-plane, i.e. 10 mm from the entrance.
NOMINAL_TASK_TARGET = (0., 0., .004, 0.)
DEFAULT_URDF = Path(__file__).resolve().parents[1] / "src/phymeo/builders/panda_assets/panda.urdf"
BOARD_HALF = .060
HOLE_HALF = .005
BOARD_THICKNESS = .012
PIN_RADIUS = .004
PIN_SEGMENT = .040
PIN_TIP_Z = PIN_SEGMENT + PIN_RADIUS
PART_FRAME_TO_TIP = np.eye(4)
PART_FRAME_TO_TIP[2, 3] = PIN_TIP_Z
PART_FRAME_TO_TIP.setflags(write=False)
PART_MASS = .12
PART_COM = (0., 0., -.006)
PART_INERTIA = (.00006, .00006, .00003)
INITIAL_GAP = .018
PITCH = .0015


def _numbers(a):
    return " ".join(format(float(x), ".17g") for x in np.asarray(a).ravel())


def _vector(text, default=(0., 0., 0.)):
    out = np.asarray(default if text is None else [float(x) for x in text.split()], dtype=float)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError("finite three-vector required")
    return out


def _rpy(rpy):
    x, y, z = rpy
    sx, sy, sz, cx, cy, cz = math.sin(x), math.sin(y), math.sin(z), math.cos(x), math.cos(y), math.cos(z)
    return np.array(((cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx),
                     (sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx),
                     (-sy, cy*sx, cy*cx)))


def _transform(xyz=(0., 0., 0.), rpy=(0., 0., 0.)):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = _rpy(rpy), xyz
    return t


def _origin(element):
    o = element.find("origin")
    return _transform() if o is None else _transform(_vector(o.get("xyz")), _vector(o.get("rpy")))


def _axis_rotation(axis, angle):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    k = np.array(((0., -z, y), (z, 0., -x), (-y, x, 0.)))
    t = np.eye(4)
    t[:3, :3] = np.eye(3) + math.sin(angle)*k + (1-math.cos(angle))*(k@k)
    return t


def _pose_attributes(t):
    # MuJoCo xyaxes is an unambiguous column-vector orientation, unlike Euler
    # conventions that differ between URDF and MJCF compiler settings.
    return dict(pos=_numbers(t[:3, 3]), xyaxes=_numbers(np.r_[t[:3, 0], t[:3, 1]]))


def _quaternion(rotation):
    """Matrix to unit wxyz without the trace=-1 singularity."""
    r = np.asarray(rotation, dtype=float)
    candidates = (1+r[0,0]+r[1,1]+r[2,2], 1+r[0,0]-r[1,1]-r[2,2],
                  1-r[0,0]+r[1,1]-r[2,2], 1-r[0,0]-r[1,1]+r[2,2])
    k = int(np.argmax(candidates))
    s = 2*math.sqrt(max(0., candidates[k]))
    if k == 0:
        q = np.array((s/4,(r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s))
    elif k == 1:
        q = np.array(((r[2,1]-r[1,2])/s,s/4,(r[0,1]+r[1,0])/s,(r[0,2]+r[2,0])/s))
    elif k == 2:
        q = np.array(((r[0,2]-r[2,0])/s,(r[0,1]+r[1,0])/s,s/4,(r[1,2]+r[2,1])/s))
    else:
        q = np.array(((r[1,0]-r[0,1])/s,(r[0,2]+r[2,0])/s,(r[1,2]+r[2,1])/s,s/4))
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _free_transform(pose):
    p = np.asarray(pose, dtype=float)
    if p.shape != (7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1) > 1e-5:
        raise ValueError("measured free pose must be xyz plus unit wxyz")
    w,x,y,z = p[3:] / np.linalg.norm(p[3:])
    t = np.eye(4)
    t[:3,3] = p[:3]
    t[:3,:3] = ((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
                (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
                (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
    return t


def _source(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    root = ET.fromstring(raw)
    links = {x.get("name"): x for x in root.findall("link")}
    joints = {x.get("name"): x for x in root.findall("joint")}
    if root.tag != "robot" or not all(n in joints for n in ARM_NAMES):
        raise ValueError("Panda URDF with seven named arm joints required")
    for i, name in enumerate(ARM_NAMES):
        j = joints[name]
        if (j.get("type") != "revolute" or j.find("parent").get("link") != f"panda_link{i}"
                or j.find("child").get("link") != f"panda_link{i+1}"):
            raise ValueError("Panda arm chain/order differs")
    return path, raw, links, joints


def _link_poses(joints, q):
    poses = {"panda_link0": np.eye(4)}
    values = dict(zip(ARM_NAMES, q))
    remaining = list(joints.values())
    while remaining:
        before = len(remaining)
        for j in tuple(remaining):
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            if parent not in poses:
                continue
            kind = j.get("type")
            t = _origin(j)
            if kind == "revolute":
                t = t @ _axis_rotation(_vector(j.find("axis").get("xyz")), values[j.get("name")])
            elif kind == "prismatic" and j.get("name") in ("panda_finger_joint1", "panda_finger_joint2"):
                t = t @ _transform(.03 * _vector(j.find("axis").get("xyz")))
            elif kind != "fixed":
                raise ValueError("unexpected non-Panda joint")
            poses[child] = poses[parent] @ t
            remaining.remove(j)
        if len(remaining) == before:
            raise ValueError("disconnected/cyclic URDF")
    return poses


@dataclass(frozen=True)
class AssemblyScene:
    task: str
    xml: str
    xml_sha256: str
    urdf_path: str
    urdf_sha256: str
    initial_qpos: tuple
    arm_joint_names: tuple
    base_transforms: tuple
    provenance: dict
    timestep: float = .002

    def initial_state(self):
        from phymeo.backends.mujoco_runtime import MuJoCoInitialState
        return MuJoCoInitialState(q=self.initial_qpos, v=(0.,)*20, time=0.)

    def compile(self):
        from phymeo.builders.mujoco_compiler import compile_mujoco_xml
        return compile_mujoco_xml(self.xml, require_asset=False)


def _emit_arm(world, actuator, links, joints, side, base):
    bodies = {"panda_link0": ET.SubElement(world, "body", name=f"{side}_panda_link0", **_pose_attributes(base))}
    children = {}
    for joint in joints.values():
        children.setdefault(joint.find("parent").get("link"), []).append(joint)

    def visit(name):
        body, link = bodies[name], links[name]
        inertial = link.find("inertial")
        if inertial is not None:
            inertia = inertial.find("inertia")
            a, b, c, d, e, f = [float(inertia.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            t = _origin(inertial)
            matrix = t[:3, :3] @ np.array(((a,d,e),(d,b,f),(e,f,c))) @ t[:3, :3].T
            ET.SubElement(body, "inertial", pos=_numbers(t[:3,3]), mass=inertial.find("mass").get("value"),
                          fullinertia=_numbers((matrix[0,0],matrix[1,1],matrix[2,2],matrix[0,1],matrix[0,2],matrix[1,2])))
        for index, collision in enumerate(link.findall("collision")):
            shape = list(collision.find("geometry"))[0]
            attrs = dict(name=f"{side}_{name}_proxy{index}", **_pose_attributes(_origin(collision)),
                         contype="0", conaffinity="0", mass="0", group="1", rgba=".83 .85 .87 1")
            if shape.tag == "box":
                attrs.update(type="box", size=_numbers(.5*_vector(shape.get("size"))))
            elif shape.tag == "sphere":
                attrs.update(type="sphere", size=shape.get("radius"))
            elif shape.tag == "cylinder":
                attrs.update(type="cylinder", size=_numbers((float(shape.get("radius")), .5*float(shape.get("length")))))
            else:
                raise ValueError("only declared primitive Panda collision proxies supported")
            ET.SubElement(body, "geom", **attrs)
        for j in children.get(name, ()):
            child, kind = j.find("child").get("link"), j.get("type")
            t = _origin(j)
            if kind == "prismatic":
                t = t @ _transform(.03*_vector(j.find("axis").get("xyz")))
            b = ET.SubElement(body, "body", name=f"{side}_{child}", **_pose_attributes(t))
            bodies[child] = b
            if kind == "revolute":
                limit = j.find("limit")
                joint_name = f"{side}_{j.get('name')}"
                ET.SubElement(b, "joint", name=joint_name, type="hinge", axis=j.find("axis").get("xyz"),
                              limited="true", range=f"{limit.get('lower')} {limit.get('upper')}",
                              damping=".3", armature=".01")
                effort = float(limit.get("effort"))
                ET.SubElement(actuator, "motor", name=joint_name+"_motor", joint=joint_name,
                              gear="1", ctrllimited="true", ctrlrange=_numbers((-effort,effort)))
            visit(child)
    visit("panda_link0")
    return bodies["panda_hand"]


def build_scene(task="pin", *, urdf_path=None, timestep=.002):
    """Build deterministic MJCF, fourteen motor coordinates and source binding.

    Initial pedestal translations are derived from neutral URDF FK, not an IK
    claim. Both bases remain upright, with opposite yaw. The tool axes initially
    align and the pin tip is 18 mm above the entry; physical success is untested.
    """
    if task not in ("pin", "screw_proxy"):
        raise ValueError("task must be pin or screw_proxy")
    if not math.isfinite(timestep) or timestep <= 0:
        raise ValueError("positive finite timestep required")
    path, raw, links, joints = _source(urdf_path or DEFAULT_URDF)
    local_hand = _link_poses(joints, NEUTRAL)["panda_hand"]
    left = _transform((-.40, 0., .12))
    board = left @ local_hand @ _transform((.18, 0., .12))
    right = _transform(rpy=(0., 0., math.pi))
    tip_local = local_hand @ _transform((0., 0., .12+PIN_TIP_Z))
    desired_tip = board[:3,3] - board[:3,2]*(BOARD_THICKNESS/2+INITIAL_GAP)
    right[:3,3] = desired_tip - right[:3,:3] @ tip_local[:3,3]
    if np.dot((right@local_hand)[:3,2], board[:3,2]) < 1-1e-9:
        raise ValueError("neutral opposing bases do not align pin/board axes")
    root = ET.Element("mujoco", model="dual_panda_"+task)
    ET.SubElement(root,"compiler",angle="radian",inertiafromgeom="auto",fusestatic="false")
    ET.SubElement(root,"option",timestep=str(timestep),gravity="0 0 -9.81",integrator="Euler",cone="pyramidal")
    default = ET.SubElement(root,"default")
    ET.SubElement(default,"geom",condim="3",friction=".35 .005 .0001",solref=".008 1",solimp=".9 .95 .001",margin="0")
    world, motors = ET.SubElement(root,"worldbody"), ET.SubElement(root,"actuator")
    hands = [_emit_arm(world,motors,links,joints,side,base) for side,base in (("left",left),("right",right))]
    fixture = ET.SubElement(hands[0],"body",name="left_board_fixture",pos=".18 0 .12")
    # Disjoint rectangles leave an actual 10x10 mm through opening, no central
    # invisible collider. No analytic cylinder-hole or threaded contact claimed.
    gap, outer, hz = HOLE_HALF, BOARD_HALF, BOARD_THICKNESS/2
    walls = [((-(outer+gap)/2,0,0),((outer-gap)/2,outer,hz)),
             (((outer+gap)/2,0,0),((outer-gap)/2,outer,hz)),
             ((0,-(outer+gap)/2,0),(gap,(outer-gap)/2,hz)),
             ((0,(outer+gap)/2,0),(gap,(outer-gap)/2,hz))]
    for i,(pos,half) in enumerate(walls):
        ET.SubElement(fixture,"geom",name=f"board_wall_{i}",type="box",pos=_numbers(pos),size=_numbers(half),
                      mass=".05",contype="1",conaffinity="2",rgba=".19 .48 .65 1")
    carrier = ET.SubElement(hands[0],"body",name="left_board_carrier")
    ET.SubElement(carrier,"geom",name="board_carrier",type="capsule",fromto="0 0 .08 .18 0 .12",size=".012",
                  mass=".06",contype="0",conaffinity="0",rgba=".30 .33 .36 1")
    ET.SubElement(fixture,"site",name="assembly_hole",size=".001",rgba="0 0 0 0")
    ET.SubElement(fixture,"site",name="assembly_entry",pos=_numbers((0,0,-hz)),size=".001",rgba="0 0 0 0")
    head = ET.SubElement(hands[1],"body",name="right_assembly_head",pos="0 0 .12")
    ET.SubElement(head,"site",name="assembly_driver",pos=_numbers((0,0,PIN_TIP_Z)),size=".001",rgba="0 0 0 0")
    part_home = right @ local_hand @ _transform((0,0,.12))
    part = ET.SubElement(world,"body",name="assembly_part",**_pose_attributes(part_home))
    ET.SubElement(part,"freejoint",name="assembly_part_free")
    # Declared engineered adapter inertia about its explicit COM, not an
    # inference from the drawing or a nearly massless pin. No hidden free body.
    ET.SubElement(part,"inertial",pos=_numbers(PART_COM),mass=str(PART_MASS),diaginertia=_numbers(PART_INERTIA))
    ET.SubElement(part,"geom",name="assembly_pin",type="capsule",fromto=_numbers((0,0,0,0,0,PIN_SEGMENT)),
                  size=str(PIN_RADIUS),contype="2",conaffinity="1",rgba=".72 .69 .42 1")
    ET.SubElement(part,"geom",name="assembly_pin_adapter",type="capsule",fromto="-.007 0 -.020 .007 0 -.020",size=".018",
                  contype="2",conaffinity="1",rgba=".35 .38 .42 1")
    ET.SubElement(part,"geom",name="spindle_orientation_marker",type="box",pos=".028 0 -.012",size=".003 .004 .004",
                  contype="0",conaffinity="0",rgba=".9 .15 .06 1")
    ET.SubElement(part,"site",name="assembly_pin_tip",pos=_numbers((0,0,PIN_TIP_Z)),size=".001",rgba="0 0 0 0")
    ET.SubElement(part,"site",name="assembly_pin_axis",size=".001",rgba="0 0 0 0")
    ET.SubElement(world,"geom",name="assembly_table_visual",type="box",pos="0 0 -.045",size="1.2 .7 .04",
                  contype="0",conaffinity="0",rgba=".23 .25 .27 1")
    q0 = tuple(NEUTRAL)*2 + tuple(part_home[:3,3]) + tuple(_quaternion(part_home[:3,:3]))
    ET.SubElement(ET.SubElement(root,"keyframe"),"key",name="home",qpos=_numbers(q0))
    xml = ET.tostring(root,encoding="unicode")
    names = tuple(tuple(f"{side}_{n}" for n in ARM_NAMES) for side in ("left","right"))
    provenance = dict(schema="dual-panda-free-assembly-v2",kinematics="bundled franka_ros 0.10.1 Panda URDF",
                      joint_order=[n for arm in names for n in arm],fingers="fixed at 30 mm per finger; not grasping",
                      tools="left board rigid; right driver site; independent free assembly part; no grasp claim",
                      collision_scope="board boxes versus free pin and rounded 50 mm adapter capsules",
                      part_mass_kg=PART_MASS,part_com_local_m=PART_COM,part_diaginertia_kg_m2=PART_INERTIA,
                      part_inertia_basis="declared equivalent 50 mm adapter; not visual-density inferred",
                      part_frame_to_tip=PART_FRAME_TO_TIP.tolist(),free_qpos_order="after 14 hinges: xyz,wxyz",
                      coupling="none in model; runner must apply explicit reciprocal compliant loads",
                      visual_scope="URDF structural collision envelopes, not manufacturer visual meshes",
                      dynamics="URDF inertias; example damping .3 and armature .01; added declared tool masses",
                      screw_model="pitch diagnostic only; no helical constraint or thread friction",
                      hole="square 10 mm through-hole",pin_radius_m=PIN_RADIUS,pitch_m_per_turn=PITCH,
                      runtime="explicit MuJoCoRuntime musa or cpu; no fallback; this assembly not yet device validated")
    return AssemblyScene(task,xml,hashlib.sha256(xml.encode()).hexdigest(),str(path),hashlib.sha256(raw).hexdigest(),
                         q0,names,tuple(tuple(tuple(float(x) for x in row) for row in t) for t in (left,right)),provenance,timestep)


def forward_kinematics(scene, qpos):
    """Arm links from q14 or q21; free part/tip only from measured q21."""
    q = np.asarray(qpos,dtype=float)
    if q.shape not in ((14,), (21,)) or not np.isfinite(q).all():
        raise ValueError("fourteen arm positions or full measured q21 required")
    _,raw,_,joints = _source(scene.urdf_path)
    if hashlib.sha256(raw).hexdigest() != scene.urdf_sha256:
        raise ValueError("source URDF changed")
    result = {}
    for i,side in enumerate(("left","right")):
        base = np.asarray(scene.base_transforms[i])
        result.update({f"{side}_{name}":base@t for name,t in _link_poses(joints,q[7*i:7*i+7]).items()})
    result["left_board_fixture"] = result["left_panda_hand"]@_transform((.18,0,.12))
    result["right_assembly_head"] = result["right_panda_hand"]@_transform((0,0,.12))
    result["assembly_hole"] = result["left_board_fixture"]
    result["assembly_entry"] = result["assembly_hole"]@_transform((0,0,-BOARD_THICKNESS/2))
    result["assembly_driver"] = result["right_assembly_head"]@PART_FRAME_TO_TIP
    if q.shape == (21,):
        result["assembly_part"] = _free_transform(q[14:])
        result["assembly_pin_tip"] = result["assembly_part"]@PART_FRAME_TO_TIP
        result["assembly_pin_axis"] = result["assembly_part"]
    return result


def free_part_pose(scene, q14):
    """INITIALIZATION ONLY: xyz,wxyz aligning the free part tip with driver.

    Never use this to overwrite accepted free-body state during a rollout.
    """
    q = np.asarray(q14,dtype=float)
    if q.shape != (14,):
        raise ValueError("free-part initialization requires arm q14")
    t = forward_kinematics(scene,q)["right_assembly_head"]
    return np.r_[t[:3,3],_quaternion(t[:3,:3])]


def initial_qpos_for_arms(scene,q14):
    """Return a new q21 declaration for an independently initialized episode."""
    return np.r_[np.asarray(q14,dtype=float),free_part_pose(scene,q14)]


def assembly_metrics(scene,qpos,*,unwrapped_turn_rad=None,engagement_depth_m=0.):
    """Pure endpoint geometry, never a force/contact/dwell success decision.

    For multi-turn pitch diagnostics caller must unwrap actual successive
    relative twists. Supplying time or desired command angle is not evidence.
    """
    if np.asarray(qpos).shape != (21,):
        raise ValueError("assembly metrics require actual free-body q21, not driver q14")
    fk = forward_kinematics(scene,qpos)
    board, tip = fk["assembly_hole"],fk["assembly_pin_tip"]
    relative = np.linalg.inv(board)@tip
    pos = relative[:3,3]
    depth = float(pos[2]+BOARD_THICKNESS/2)
    angle = math.atan2(float(relative[1,0]),float(relative[0,0]))
    tilt = math.acos(float(np.clip(relative[2,2],-1,1)))
    result = dict(lateral_error_m=float(np.linalg.norm(pos[:2])),axis_tilt_rad=tilt,
                  tip_in_board_frame_m=pos.tolist(),insertion_depth_m=depth,relative_twist_rad=angle,
                  square_tip_clearance_m=float(HOLE_HALF-PIN_RADIUS-max(abs(pos[0]),abs(pos[1]))),
                  geometric_only=True,physical_success=None,thread_contact_simulated=False)
    if unwrapped_turn_rad is not None:
        if not np.isfinite([unwrapped_turn_rad,engagement_depth_m]).all():
            raise ValueError("finite observed angle/depth required")
        result["pitch_residual_m"] = depth-float(engagement_depth_m)-PITCH*float(unwrapped_turn_rad)/(2*math.pi)
    return result



def task_vector(scene, q21, *, previous_twist=None):
    """Return [tip x,y,z in board frame, actual relative twist from HOME].

    The fourth value uses the actual free-part orientation versus board FK,
    NOT a q7 difference or the driver orientation. Full measured q21 required.
    Without previous_twist it is the principal angle [-pi,pi]. Supply the last
    returned twist at every accepted step to unwrap continuously, assuming less
    than pi actual rotation between samples. No state is silently retained.
    NOMINAL_TASK_TARGET is a geometric pin target, not a solved joint pose.
    """
    current = assembly_metrics(scene, q21)
    home_angle = assembly_metrics(scene, scene.initial_qpos)["relative_twist_rad"]
    delta = current["relative_twist_rad"] - home_angle
    twist = math.atan2(math.sin(delta), math.cos(delta))
    if previous_twist is not None:
        if not math.isfinite(previous_twist):
            raise ValueError("previous measured unwrapped twist must be finite")
        twist += 2*math.pi*round((float(previous_twist)-twist)/(2*math.pi))
    return np.array((*current["tip_in_board_frame_m"], twist), dtype=float)
