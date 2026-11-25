"""Two independently driven serial arms operating on one native free body.

The CPU and MUSA routes compile identical MJCF. Native contact impulses move
the workpiece. The only command written during an episode is motor control.
Lengths are metres, mass is kg, angles are radians, and torques are N m.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


SIDES = ("left", "right")
TASKS = ("carry", "push_t")
TIMESTEP = 0.002
ACTION_SIZE = 14
ACTION_INCREMENT = 0.045
PAD_RADIUS = 0.026
TWIN_PAD_HALF_SPAN = 0.032
TOOL_PROFILES = ("single-pad", "twin-pad")
GOAL_DWELL_S = 0.10


def _numbers(values):
    return " ".join(f"{float(value):.9g}" for value in values)


def _rotation(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


@dataclass(frozen=True)
class BimanualScene:
    task: str
    seed: int
    xml: str
    arm_dofs: tuple
    arm_joint_names: tuple
    start_xyz: tuple
    goal_xy: tuple
    goal_yaw: float
    mass: float
    # Each box is (offset in the payload body, half extents).
    boxes: tuple
    xml_sha256: str
    tool_profile: str = "single-pad"
    object_body: str = "payload"
    wrist_sites: tuple = ("left_wrist_camera", "right_wrist_camera")
    timestep: float = TIMESTEP


def _arm(world, motors, side, count, pad, focus, rng, task, tool_profile):
    sign = 1 if side == "left" else -1
    # Link vectors form a bent shoulder / elbow at zero joint coordinates.
    # The seed changes both major link lengths without moving either initial tool.
    upper = float(rng.uniform(.193, .213))
    lower = float(rng.uniform(.169, .189))
    parts = [
        ("0 0 1", (0, 0, .082), .75, 26, 1.8),
        ("0 1 0", (upper, -sign*.066, .138), .90, 34, 1.8),
        ("0 1 0", (lower, -sign*.053, -.253), .62, 12.5, 1.65),
        ("1 0 0", (.052, -sign*.008, -.063), .21, 8, 1.8),
        ("0 1 0", (.043, 0, -.027), .18, 6.5, 1.55),
        ("0 0 1", (.035, 0, -.012), .12, 5.0, 1.85),
    ]
    if count == 7:
        parts.insert(3, ("1 0 0", (.035, -sign*.008, -.024), .16, 8, 1.65))
    start = np.asarray(pad) - np.sum([part[1] for part in parts], axis=0)
    parent, origin = world, start
    joints = []
    color = ".24 .40 .61 1" if sign > 0 else ".63 .69 .73 1"
    for i, (axis, segment, mass, torque, bound) in enumerate(parts):
        body = SubElement(parent, "body", name=f"{side}_link{i}", pos=_numbers(origin))
        joint = f"{side}_j{i}"
        joints.append(joint)
        SubElement(body, "joint", name=joint, type="hinge", axis=axis, limited="true",
                   range=_numbers((-bound, bound)), damping=".32", armature=".045")
        SubElement(body, "geom", name=f"{side}_shell{i}", type="capsule",
                   fromto=_numbers((0, 0, 0, *segment)), size=".027" if i < 3 else ".017",
                   mass=str(mass), rgba=color, contype="0", conaffinity="0")
        SubElement(motors, "motor", name=f"{side}_m{i}", joint=joint, gear="1",
                   ctrllimited="true", ctrlrange=_numbers((-torque, torque)))
        parent, origin = body, segment
    offset = np.asarray(origin)
    if tool_profile == "single-pad":
        SubElement(parent, "geom", name=f"{side}_pad", type="sphere",
                   pos=_numbers(offset), size=str(PAD_RADIUS), mass=".095", rgba=".12 .17 .22 1",
                   contype="4", conaffinity="3", priority="1", friction="1.40 .005 .0001")
    else:
        # Rigid 95 g tool. Two 40 g spherical contact pads flank the old TCP.
        # The 15 g carrier is recessed relative to the 26 mm rubber spheres.
        span = TWIN_PAD_HALF_SPAN * np.array((1, 0, 0) if task == "carry" else (0, 1, 0))
        SubElement(parent, "geom", name=f"{side}_carrier", type="capsule",
                   fromto=_numbers((*(offset - span), *(offset + span))),
                   size=".009", mass=".015", rgba=".37 .42 .48 1", contype="0", conaffinity="0")
        for suffix, point in (("a", offset-span), ("b", offset+span)):
            SubElement(parent, "geom", name=f"{side}_pad_{suffix}", type="sphere",
                       pos=_numbers(point), size=str(PAD_RADIUS), mass=".040", rgba=".12 .17 .22 1",
                       contype="4", conaffinity="3", priority="1", friction="1.40 .005 .0001")
    SubElement(parent, "site", name=f"{side}_tool", pos=_numbers(offset), size=".003",
               rgba="0 0 0 0")
    camera_offset = np.array((-.065, sign*.06, .105))
    axis_z = np.asarray(pad) + camera_offset - np.asarray(focus)
    axis_z /= np.linalg.norm(axis_z)
    axis_x = np.cross((0, 0, 1), axis_z)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(axis_z, axis_x)
    SubElement(parent, "site", name=f"{side}_wrist_camera", size=".004",
               pos=_numbers(offset + camera_offset), xyaxes=_numbers((*axis_x, *axis_y)),
               rgba="0 0 0 0")
    return tuple(joints)


def _markers(world, goal_xy, yaw, boxes):
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array(((c, -s), (s, c)))
    SubElement(world, "site", name="task_goal", pos=_numbers((*goal_xy, .001)),
               euler=_numbers((0, 0, yaw)), size=".003", rgba="0 0 0 0")
    for index, (offset, half) in enumerate(boxes):
        center = np.asarray(offset[:2])
        corners = [center + (i*half[0], j*half[1]) for i, j in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        corners = [np.asarray(goal_xy) + rotation @ p for p in corners]
        for k, a in enumerate(corners):
            b = corners[(k+1) % len(corners)]
            SubElement(world, "geom", name=f"goal_marker_{index}_{k}", type="capsule",
                       fromto=_numbers((*a, .0005, *b, .0005)), size=".0025",
                       rgba=".04 .83 .60 1", contype="0", conaffinity="0")


def make_scene(task, seed=0, arm_dofs=None, *, tool_profile="single-pad"):
    """Build a source-bound scene. Its two wrist sites look along local -Z."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if tool_profile not in TOOL_PROFILES:
        raise ValueError(f"tool_profile must be one of {TOOL_PROFILES}")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    dofs = tuple(arm_dofs) if arm_dofs is not None else ((6, 7) if seed % 2 else (7, 6))
    if len(dofs) != 2 or any(d not in (6, 7) for d in dofs):
        raise ValueError("arm_dofs must contain a left and right count of six or seven")
    start_xy = rng.uniform(-.007, .007, size=2)
    goal = tuple(start_xy + (.12, rng.uniform(-.012, .012)))
    yaw = float(np.deg2rad(15) * (-1 if seed % 2 else 1)) if task == "push_t" else 0.
    if task == "carry":
        mass = 6.0
        boxes = (((0., 0., 0.), (.06, .11, .05)),)
    else:
        mass = 9.0
        boxes = (((0., 0., 0.), (.028, .13, .028)), ((.106, 0., 0.), (.078, .033, .028)))
    start_xyz = (*map(float, start_xy), boxes[0][1][2])
    root = Element("mujoco", model=f"bimanual_{task}_{seed}")
    SubElement(root, "compiler", angle="radian", autolimits="true")
    SubElement(root, "option", timestep=str(TIMESTEP), gravity="0 0 -9.81", integrator="Euler",
               cone="pyramidal", solver="Newton", iterations="64", tolerance="1e-8")
    defaults = SubElement(root, "default")
    SubElement(defaults, "geom", condim="3", solref=".012 1", solimp=".90 .97 .001")
    world = SubElement(root, "worldbody")
    SubElement(world, "geom", name="table", type="plane", size="1 1 .05", rgba=".28 .31 .34 1",
               contype="1", conaffinity="6", priority="1", friction=".52 .005 .0001")
    payload = SubElement(world, "body", name="payload", pos=_numbers(start_xyz))
    SubElement(payload, "freejoint", name="payload_free")
    volumes = np.array([8*np.prod(half) for _, half in boxes])
    for i, (offset, half) in enumerate(boxes):
        SubElement(payload, "geom", name=f"payload_{i}", type="box", pos=_numbers(offset),
                   size=_numbers(half), mass=str(mass*volumes[i]/volumes.sum()),
                   rgba=".85 .20 .055 1", contype="2", conaffinity="5",
                   friction=".52 .005 .0001")
    _markers(world, goal, yaw, boxes)
    motors = SubElement(root, "actuator")
    arms = []
    for side, count, sign in zip(SIDES, dofs, (1, -1), strict=True):
        if task == "carry":
            pad_offset = (0., sign*(.11 + PAD_RADIUS + .008), 0.)
        else:
            pad_offset = (-.028 - PAD_RADIUS - .008, sign*.082, .001)
        arms.append(_arm(world, motors, side, count, np.asarray(start_xyz) + pad_offset,
                         start_xyz, rng, task, tool_profile))
    xml = tostring(root, encoding="unicode")
    return BimanualScene(task, seed, xml, dofs, tuple(arms), start_xyz, goal, yaw, mass,
                         boxes, hashlib.sha256(xml.encode()).hexdigest(), tool_profile=tool_profile)


class BimanualTask:
    """Interpret one world's motor and contact channels in native model order."""

    def __init__(self, scene, model, active_arms="both"):
        if active_arms not in ("both", "left", "right"):
            raise ValueError("active_arms must be both, left, or right")
        self.scene, self.model, self.active_arms = scene, model, active_arms
        joints = [model.joint(name).id for names in scene.arm_joint_names for name in names]
        self.q_indices = model.jnt_qposadr[joints]
        self.v_indices = model.jnt_dofadr[joints]
        self.q_bounds = model.jnt_range[joints]
        self.slots = np.array([7*s+i for s, count in enumerate(scene.arm_dofs) for i in range(count)])
        self.enabled = np.array([active_arms in ("both", side) for side, count in zip(
            SIDES, scene.arm_dofs, strict=True) for _ in range(count)])
        self.source_bounds = model.actuator_ctrlrange.copy()
        self.target = model.qpos0[self.q_indices].copy()
        self.kp = np.concatenate([np.array([160., 180., 115.] + [42.]*(count-3))
                                  for count in scene.arm_dofs])
        self.kd = np.concatenate([np.array([9., 11., 6.] + [2.]*(count-3))
                                  for count in scene.arm_dofs])
        suffixes = ("pad",) if scene.tool_profile == "single-pad" else ("pad_a", "pad_b")
        self.pad_owner = {model.geom(f"{side}_{suffix}").id: i
                          for i, side in enumerate(SIDES) for suffix in suffixes}
        self.pad_geoms = tuple(self.pad_owner)
        self.pad_slot = {geom: i for i, geom in enumerate(self.pad_geoms)}
        self.floor = model.geom("table").id
        self.payload_geoms = {model.geom(f"payload_{i}").id for i in range(len(scene.boxes))}
        self.root_q = int(model.jnt_qposadr[model.joint("payload_free").id])
        self.root_v = int(model.jnt_dofadr[model.joint("payload_free").id])
        self.hold_ticks = 0
        self.flight_height = 0.
        self.flight_displacement = 0.
        self.flight_start_xy = None
        self.carried_airborne = False
        self.max_clearance = 0.
        self.both_loaded_steps = 0
        self.both_loaded_airborne_steps = 0
        self.all_pads_loaded_steps = 0
        self.all_pads_loaded_airborne_steps = 0
        self.max_pad_normal = np.zeros(2)
        self.max_pad_up = np.zeros(2)
        self.peak_pad_total_up = 0.
        self.success = False

    def torque_control(self, action):
        """14 padded motor fractions with zero effort on physically parked arms."""
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all() or np.any(abs(action) > 1.000001):
            raise ValueError("torque action must contain 14 finite normalized motor efforts in [-1, 1]")
        fraction = np.clip(action[self.slots], -1., 1.)
        limits = np.where(fraction >= 0., self.source_bounds[:, 1], -self.source_bounds[:, 0])
        return np.where(self.enabled, fraction*limits, 0.)

    def initial_state(self):
        # Park a disabled arm outside the reachable workpiece corridor; leave
        # its native passive dynamics enabled and all its motors unpowered.
        q = self.model.qpos0.copy()
        for s, side in enumerate(SIDES):
            if self.active_arms not in ("both", side):
                address = self.model.jnt_qposadr[self.model.joint(f"{side}_j0").id]
                q[address] = 1.15 if s == 0 else -1.15
        self.target = q[self.q_indices].copy()
        return MuJoCoInitialState(q)

    def set_action(self, action):
        values = np.asarray(action, dtype=float)
        if values.shape != (ACTION_SIZE,) or not np.isfinite(values).all() or np.any(abs(values) > 1.000001):
            raise ValueError("action must contain 14 finite normalized joint increments in [-1, 1]")
        self.target = np.clip(self.target + self.enabled*ACTION_INCREMENT*values[self.slots],
                              self.q_bounds[:, 0], self.q_bounds[:, 1])

    def control(self, q, v):
        demand = self.kp*(self.target-q[self.q_indices]) - self.kd*v[self.v_indices]
        return np.where(self.enabled, np.clip(demand, *self.source_bounds.T), 0.)

    def saturated_control(self, q, directions):
        """Independent strong control at the motor limits for each tool direction.

        Maximizes instantaneous motor power along the specified Jacobian-transpose
        command direction. This is a finite stress trial, not a global proof of
        maximum endpoint force at every posture. Only arm commands are written.
        """
        import mujoco as mj

        directions = np.asarray(directions, dtype=float)
        if directions.shape != (2, 3) or not np.isfinite(directions).all():
            raise ValueError("saturation probe requires two world-space 3D directions")
        if not hasattr(self, "_kinematics"):
            self._kinematics = mj.MjData(self.model)
        data = self._kinematics
        data.qpos[:] = q
        mj.mj_kinematics(self.model, data)
        mj.mj_comPos(self.model, data)
        ctrl = np.zeros(self.model.nu)
        jac = np.empty((3, self.model.nv))
        for s, side in enumerate(SIDES):
            if self.active_arms not in ("both", side):
                continue
            mj.mj_jacSite(self.model, data, jac, None, self.model.site(f"{side}_tool").id)
            columns = (self.slots // 7) == s
            direction = jac[:, self.v_indices[columns]].T @ directions[s]
            bounds = self.source_bounds[columns]
            ctrl[columns] = np.where(direction > 1e-9, bounds[:, 1],
                                     np.where(direction < -1e-9, bounds[:, 0], 0.))
        return ctrl

    def proprio(self, q, v):
        fields = np.zeros((3, ACTION_SIZE), dtype=np.float32)
        fields[0, self.slots] = q[self.q_indices]
        fields[1, self.slots] = v[self.v_indices]
        fields[2, self.slots] = self.enabled
        return fields.ravel()

    def _loads(self, contacts):
        normal, up = np.zeros(2), np.zeros(2)
        wrist_force = np.zeros((2, 3))
        payload_wrench = np.zeros((2, 6))
        pad_channel = np.zeros(len(self.pad_geoms))
        floor_up = 0.
        for contact in contacts:
            a, b = map(int, contact.geometry_ids)
            wrench = np.asarray(contact.wrench_contact_frame, dtype=float)
            if wrench[0] <= 0:
                continue
            # Native convention: geom2 receives +frame.T @ local force.
            force = np.asarray(contact.frame).reshape(3, 3).T @ wrench[:3]
            if a in self.pad_owner:
                wrist_force[self.pad_owner[a]] -= force
            if b in self.pad_owner:
                wrist_force[self.pad_owner[b]] += force
            if a not in self.payload_geoms and b not in self.payload_geoms:
                continue
            if a in self.payload_geoms:
                other, vertical = b, -force[2]
            else:
                other, vertical = a, force[2]
            if other == self.floor:
                floor_up += vertical
            elif other in self.pad_owner:
                i = self.pad_owner[other]
                sign = -1. if a in self.payload_geoms else 1.
                torque = np.asarray(contact.frame).reshape(3, 3).T @ wrench[3:]
                payload_wrench[i, :3] += sign*force
                payload_wrench[i, 3:] += sign*(np.cross(contact.position, force) + torque)
                normal[i] += wrench[0]
                up[i] += vertical
                pad_channel[self.pad_slot[other]] += wrench[0]
        return normal, up, float(floor_up), wrist_force, pad_channel, payload_wrench

    def evaluate(self, q, v, contacts, *, advance=True):
        sc = self.scene
        pos = q[self.root_q:self.root_q+3]
        rot = _rotation(q[self.root_q+3:self.root_q+7])
        clearance = min(pos[2] + rot[2] @ offset - abs(rot[2]) @ half for offset, half in sc.boxes)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        yaw_error = abs(np.arctan2(np.sin(yaw-sc.goal_yaw), np.cos(yaw-sc.goal_yaw)))
        tilt = np.arccos(np.clip(rot[2, 2], -1., 1.))
        orientation_error = np.arccos(np.clip((np.trace(rot)-1.)*.5, -1., 1.))
        goal_error = np.linalg.norm(pos[:2]-sc.goal_xy)
        normal, up, floor_up, wrist_force, pad_channel, payload_wrench = self._loads(contacts)
        weight = sc.mass*9.81
        iv = self.root_v
        omega_local = v[iv+3:iv+6]
        offset_world = rot @ self.model.body_ipos[self.model.body(sc.object_body).id]
        com_velocity = v[iv:iv+3] + np.cross(rot @ omega_local, offset_world)
        speed = np.linalg.norm(com_velocity)
        spin = np.linalg.norm(omega_local)
        airborne = floor_up < .05*weight and clearance >= .003
        supported = floor_up >= .8*weight
        slow = speed <= .035 and spin <= .25
        if advance:
            self.max_clearance = max(self.max_clearance, float(clearance))
            if airborne:
                if self.flight_start_xy is None:
                    self.flight_start_xy = pos[:2].copy()
                displacement = np.linalg.norm(pos[:2] - self.flight_start_xy)
                self.flight_height = max(float(clearance), self.flight_height)
                self.flight_displacement = max(float(displacement), self.flight_displacement)
                if self.flight_height >= .035 and self.flight_displacement >= .060:
                    self.carried_airborne = True
            else:
                self.flight_height = self.flight_displacement = 0.
                self.flight_start_xy = None
            self.both_loaded_steps += int(np.all(normal >= 1.))
            self.both_loaded_airborne_steps += int(airborne and np.all(normal >= 1.))
            all_pads_loaded = bool(np.all(pad_channel >= 1.))
            self.all_pads_loaded_steps += int(all_pads_loaded)
            self.all_pads_loaded_airborne_steps += int(airborne and all_pads_loaded)
            self.max_pad_normal = np.maximum(self.max_pad_normal, normal)
            self.max_pad_up = np.maximum(self.max_pad_up, up)
            self.peak_pad_total_up = max(self.peak_pad_total_up, float(up.sum()))
        if sc.task == "carry":
            at_goal = (self.carried_airborne and supported and normal.sum() <= 5.
                       and goal_error <= .025 and orientation_error <= np.deg2rad(12.) and slow)
        else:
            at_goal = (supported and goal_error <= .020 and yaw_error <= np.deg2rad(6.)
                       and tilt <= np.deg2rad(8.) and slow)
        if advance:
            self.hold_ticks = self.hold_ticks + 1 if at_goal else 0
            self.success |= self.hold_ticks*TIMESTEP + 1e-12 >= GOAL_DWELL_S
        return {
            "success": bool(self.success), "goal_dwell_s": self.hold_ticks*TIMESTEP,
            "goal_error_xy_m": float(goal_error), "goal_yaw_error_rad": float(yaw_error),
            "tilt_rad": float(tilt), "payload_xyz_m": pos.tolist(), "payload_yaw_rad": float(yaw),
            "payload_com_speed_m_s": float(speed), "payload_spin_rad_s": float(spin),
            "clearance_m": float(clearance), "max_clearance_m": self.max_clearance,
            "carried_airborne": self.carried_airborne, "floor_vertical_n": floor_up,
            "pad_normal_n": normal.tolist(), "pad_vertical_n": up.tolist(),
            "wrist_force_world_n": wrist_force.tolist(),
            "individual_pad_payload_normal_n": pad_channel.tolist(),
            "pad_payload_wrench_about_world_origin": payload_wrench.tolist(),
            "peak_pad_normal_n": self.max_pad_normal.tolist(),
            "peak_pad_vertical_n": self.max_pad_up.tolist(),
            "peak_combined_pad_vertical_n": self.peak_pad_total_up,
            "both_loaded_s": self.both_loaded_steps*TIMESTEP,
            "both_loaded_airborne_s": self.both_loaded_airborne_steps*TIMESTEP,
            "all_individual_pads_loaded_s": self.all_pads_loaded_steps*TIMESTEP,
            "all_individual_pads_loaded_airborne_s": self.all_pads_loaded_airborne_steps*TIMESTEP,
        }

    def potential(self, info):
        # Dense reward for learning; success always uses the physical dwell tests.
        value = -info["goal_error_xy_m"]/.12
        if self.scene.task == "push_t":
            return value - .35*info["goal_yaw_error_rad"]/np.deg2rad(15.)
        return (value + .55*min(info["max_clearance_m"]/.035, 1.)
                + .45*info["carried_airborne"])


class BimanualBatchEnv:
    """Same-model native worlds with sparse integer keys and independent episodes.

    A decision holds a target for action_repeat source steps; PD error and
    motor saturation are evaluated at every source step. qpos in info is for
    rendering and auditing. Only the 42-value proprio vector is policy input.
    """

    def __init__(self, task, seed=0, *, world_keys=(0,), arm_dofs=None, mode="cpu",
                 cache_root=None, action_repeat=5, horizon=200, tool_profile="single-pad"):
        if mode not in ("cpu", "musa", "cuda", "interpret"):
            raise ValueError("unsupported physics mode")
        self.world_keys = tuple(world_keys)
        if (not self.world_keys or len(set(self.world_keys)) != len(self.world_keys)
                or any(not isinstance(k, (int, np.integer)) or k < 0 for k in self.world_keys)):
            raise ValueError("world_keys must be unique nonnegative integers")
        if not isinstance(action_repeat, int) or not isinstance(horizon, int) or min(action_repeat, horizon) < 1:
            raise ValueError("action_repeat and horizon must be positive integers")
        self.task_name, self.seed, self.arm_dofs = task, seed, arm_dofs
        self.mode, self.cache_root = mode, cache_root
        self.action_repeat, self.horizon = action_repeat, horizon
        self.tool_profile = tool_profile
        self.scene = make_scene(task, seed, arm_dofs, tool_profile=tool_profile)
        self.compilation = compile_mujoco_xml(self.scene.xml, require_asset=False)
        self.model = self.compilation.mj_model
        self.runtime = None
        self._active = None

    def _states(self):
        if self.mode == "cpu":
            return self.runtime.snapshot()
        return self.runtime.integration_snapshot()

    def _info(self, key, metrics):
        state = self.states[key]
        receipt = state.last_step
        return dict(
            metrics, task=self.scene.task, seed=self.scene.seed, arm_dofs=self.scene.arm_dofs,
            active_arms=self._active[key], world=int(key), mode=self.mode, tool_profile=self.tool_profile,
            backend=self.runtime.capabilities["backend"], tick=int(state.tick), time_s=float(state.time),
            steps=self.decisions, xml_sha256=self.scene.xml_sha256, action_repeat=self.action_repeat,
            contact_phase=receipt.contact_phase if receipt is not None else "no-step-yet",
            contact_input_time_s=float(receipt.input_time) if receipt is not None else None,
            contact_output_time_s=float(receipt.output_time) if receipt is not None else None,
            qpos=np.array(state.q, dtype=float), qvel=np.array(state.v, dtype=float),
            applied_ctrl=np.array(state.ctrl, dtype=float),
        )

    def reset(self, seed=None, *, active_arms="both"):
        seed = self.seed if seed is None else seed
        selections = ({key: active_arms for key in self.world_keys}
                      if isinstance(active_arms, str) else dict(active_arms))
        if set(selections) != set(self.world_keys):
            raise ValueError("active_arms mapping must match world_keys")
        if seed != self.scene.seed:
            scene = make_scene(self.task_name, seed, self.arm_dofs, tool_profile=self.tool_profile)
            compilation = compile_mujoco_xml(scene.xml, require_asset=False)
            self.close()
            self.scene, self.compilation = scene, compilation
            self.model = compilation.mj_model
        self.seed = self.scene.seed
        self.worlds = {k: BimanualTask(self.scene, self.model, selections[k]) for k in self.world_keys}
        initial = {k: task.initial_state() for k, task in self.worlds.items()}
        if self.runtime is not None and self._active == selections:
            self.runtime.reset()
        else:
            self.close()
            options = {} if self.mode == "cpu" else dict(cache_root=self.cache_root,
                                                         contact_capacity=32, constraint_capacity=128)
            self.runtime = MuJoCoRuntime(self.compilation, mode=self.mode, world_keys=self.world_keys,
                                         initial_states=initial, **options)
        self._active = selections
        self.states = self._states()
        self.decisions = 0
        self.finished = {k: False for k in self.world_keys}
        self.baseline = {}
        self.terminated = {k: False for k in self.world_keys}
        result = {}
        for k, task in self.worlds.items():
            state = self.states[k]
            metrics = task.evaluate(state.q, state.v, (), advance=False)
            self.baseline[k] = task.potential(metrics)
            result[k] = task.proprio(state.q, state.v), self._info(k, metrics)
        return result

    def _step(self, commands, *, command_mode):
        if self.runtime is None:
            raise RuntimeError("reset the environment before stepping")
        if set(commands) != set(self.world_keys):
            raise ValueError("commands mapping must match world_keys")
        if all(self.finished.values()):
            raise RuntimeError("all episodes have ended; reset before the next decision")
        previously_finished = self.finished.copy()
        held_torques = {}
        if command_mode == "position":
            for key, task in self.worlds.items():
                if not self.finished[key]:
                    task.set_action(commands[key])
        elif command_mode == "torque":
            held_torques = {key: task.torque_control(commands[key]) for key, task in self.worlds.items()}
        self.decisions += 1
        metrics = {}
        for _ in range(self.action_repeat):
            controls = {}
            for key, task in self.worlds.items():
                state = self.states[key]
                if self.finished[key]:
                    control = np.zeros(self.model.nu)
                elif command_mode == "saturated":
                    control = task.saturated_control(state.q, commands[key])
                elif command_mode == "torque":
                    control = held_torques[key]
                else:
                    control = task.control(state.q, state.v)
                controls[key] = MuJoCoControl(control)
            self.runtime.set_controls(controls)
            self.runtime.step(TIMESTEP)
            self.states = self._states()
            for key, task in self.worlds.items():
                state = self.states[key]
                metrics[key] = task.evaluate(state.q, state.v, state.last_step.contacts,
                                             advance=not self.finished[key])
        out = {}
        for key, task in self.worlds.items():
            state, measurement = self.states[key], metrics[key]
            success = measurement["success"]
            self.terminated[key] |= success
            truncated = self.decisions >= self.horizon and not self.terminated[key]
            self.finished[key] |= self.terminated[key] or truncated
            potential = task.potential(measurement)
            reward = 0. if previously_finished[key] else potential-self.baseline[key]+float(success)
            self.baseline[key] = potential
            out[key] = (task.proprio(state.q, state.v), float(reward), self.terminated[key],
                        truncated, self._info(key, measurement))
        return out

    def step(self, actions):
        return self._step(actions, command_mode="position")

    def step_torque(self, actions):
        return self._step(actions, command_mode="torque")

    def step_saturated(self, directions):
        """Run a separate motor-limit control trial with a (2,3) direction per world."""
        return self._step(directions, command_mode="saturated")

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class BimanualEnv(BimanualBatchEnv):
    """One-world shorthand; shape and action conventions match BimanualBatchEnv."""

    @property
    def qpos(self):
        return self.states[0].q

    @property
    def time(self):
        return float(self.states[0].time)

    @property
    def tick(self):
        return int(self.states[0].tick)

    def reset(self, seed=None, *, active_arms="both"):
        return super().reset(seed, active_arms=active_arms)[0]

    def step(self, action):
        return super().step({0: action})[0]

    def step_torque(self, action):
        return super().step_torque({0: action})[0]

    def step_saturated(self, directions):
        return super().step_saturated({0: directions})[0]
