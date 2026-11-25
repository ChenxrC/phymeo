"""Transfer the registered dual-wrist PushT actor to two articulated force-limited tools.

Use --policy with the extracted push-t-policy.zip, or --prior for a development run.
The free payload, target pose, arm motors, camera mounting views and PushT dwell
criteria come from bimanual_musa_env. Each contact tip rides a real prismatic joint.
"""

import argparse
import gzip
import hashlib
import importlib
import json
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np

from bimanual_musa_env import (
    BimanualTask, GOAL_DWELL_S, MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime, SIDES, TIMESTEP,
    _numbers, _rotation, compile_mujoco_xml, make_scene,
)
from bimanual_rated_push import RATED_N
from bimanual_wrist_render import BimanualWristRenderer


STROKE = (-.16, .22)
RECESS = .065
GUIDE_STROKE = (-.040, .235)
GUIDE_RECESS = .115
GUIDE_PAD_RADIUS = .009
GUIDE_ELBOW = -.30


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _cover_housings(root):
    """Two conservative payload guards per finite capsule, with exact floor contacts.

    For endpoint distance L and capsule radius r, put ball centres at distance
    d from either end. Equate the end-cap bound r+d with the radius required at
    the centre seam, hypot(r, L/2-d). The whole capsule is then covered and
    the maximum overhang from its surface is d, plus the serialization guard.
    """
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("type") != "capsule" or geom.get("contype") != "8":
                continue
            a, b = np.array(geom.attrib["fromto"].split(), dtype=float).reshape(2, 3)
            radius = float(geom.attrib["size"])
            length = float(np.linalg.norm(b-a))
            if not (radius > 0. and length > 0.):
                raise ValueError("invalid rated-tool housing capsule")
            inset = length**2/(4.*(length+2.*radius))
            delta = inset*(b-a)/length
            geom.set("contype", "0")
            geom.set("conaffinity", "1")  # retain the exact visible capsule against the table
            for i, centre in enumerate((a+delta, b-delta)):
                ET.SubElement(body, "geom", name=f"{geom.attrib['name']}_guard{i}", type="sphere",
                              pos=_numbers(centre), size=str(radius+inset+2e-6),
                              density="0", rgba="0 0 0 0", contype="0", conaffinity="2")


def make_rated_scene(seed, guard_profile="capsule", controller="transfer"):
    if guard_profile not in ("capsule", "two-sphere-cover"):
        raise ValueError("unknown rated housing profile")
    source = make_scene("push_t", seed)
    root = ET.fromstring(source.xml)
    motors = root.find("actuator")
    guided = controller == "guided-wrist"
    shift = np.array((GUIDE_RECESS if guided else RECESS, 0., 0.))
    initial = [*source.start_xyz, 1., 0., 0., 0.]
    for side, count in zip(SIDES, source.arm_dofs, strict=True):
        base = root.find(f".//body[@name='{side}_link0']")
        wrist = root.find(f".//body[@name='{side}_link{count-1}']")
        base_xyz = np.fromstring(base.attrib["pos"], sep=" ") - shift
        if guided:
            angles = np.zeros(count)
            angles[2], angles[count-2] = GUIDE_ELBOW, -GUIDE_ELBOW
            initial.extend([*angles, 0.])  # native order includes this wrist's slide
            inside = sum(np.fromstring(root.find(f".//body[@name='{side}_link{i}']").attrib["pos"], sep=" ")
                         for i in range(3, count-1))
            x, y, z = inside
            c, s = np.cos(GUIDE_ELBOW), np.sin(GUIDE_ELBOW)
            base_xyz += inside - [c*x+s*z, y, -s*x+c*z]
        base.set("pos", _numbers(base_xyz))
        pad = wrist.find(f"geom[@name='{side}_pad']")
        stroke_range = GUIDE_STROKE if guided else STROKE
        reach = shift.copy()
        if guided:
            reach[0] += float(pad.attrib["size"])-GUIDE_PAD_RADIUS
            pad.set("size", str(GUIDE_PAD_RADIUS))
        flange = np.fromstring(pad.attrib["pos"], sep=" ")
        tip = wrist.find(f"site[@name='{side}_tool']")
        eye = wrist.find(f"site[@name='{side}_wrist_camera']")
        eye.set("pos", _numbers(np.fromstring(eye.attrib["pos"], sep=" ") + shift))
        for part in (pad, tip):
            wrist.remove(part)
        for i in range(count):
            shell = root.find(f".//geom[@name='{side}_shell{i}']")
            shell.set("contype", "8")
            shell.set("conaffinity", "3")  # actual housing contacts against payload and table are audited
        ET.SubElement(wrist, "geom", name=f"{side}_sleeve", type="capsule",
                      fromto=_numbers((*flange, *(flange + (.018, 0., 0.)))), size=".009",
                      mass=".045", rgba=".35 .41 .46 1", contype="8", conaffinity="3")
        if guided:
            ET.SubElement(wrist, "site", name=f"{side}_mount", pos=_numbers(flange), size=".002", rgba="0 0 0 0")
        slide = ET.SubElement(wrist, "body", name=f"{side}_plunger", pos=_numbers(flange))
        joint = ET.SubElement(slide, "joint", name=f"{side}_stroke", type="slide", axis="1 0 0",
                              limited="true", range=_numbers(stroke_range), damping="0", frictionloss="0")
        if guided:
            joint.set("armature", ".30")  # kg of reflected linear drive inertia
        ET.SubElement(slide, "geom", name=f"{side}_rod", type="capsule",
                      fromto=_numbers((0., 0., 0., reach[0]-.014, 0., 0.)), size=".004",
                      mass=".013", rgba=".66 .69 .71 1", contype="8", conaffinity="3")
        for part in (pad, tip):
            part.set("pos", _numbers(reach))
            slide.append(part)
        pad.set("condim", "1")  # the axial actuator cannot claim shear traction as rated pressure
        pad.set("priority", "2")
        ET.SubElement(motors, "motor", name=f"{side}_drive", joint=f"{side}_stroke",
                      gear="1", ctrllimited="true", forcelimited="true",
                      ctrlrange=_numbers((-RATED_N, RATED_N)), forcerange=_numbers((-RATED_N, RATED_N)))
    if guided:
        ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="rated-level-start", qpos=_numbers(initial))
    if guard_profile == "two-sphere-cover":
        _cover_housings(root)
    xml = ET.tostring(root, encoding="unicode")
    return source, replace(source, xml=xml, xml_sha256=digest(xml.encode()))


def guard_layout(model, profile):
    from phymeo.backends.mujoco_tyx._constraint_model import constraint_pairs

    pairs = constraint_pairs(model)
    pair_set = {frozenset((a, b)) for a, b in pairs}
    shape_counts = {}
    for a, b in pairs:
        label = "/".join(mj.mjtGeom(int(model.geom_type[g])).name for g in (a, b))
        shape_counts[label] = 1+shape_counts.get(label, 0)
    payload = [model.geom("payload_0").id, model.geom("payload_1").id]
    floor = model.geom("table").id
    capsules = [(g, model.geom(g).name) for g in range(model.ngeom)
                if model.geom_type[g] == mj.mjtGeom.mjGEOM_CAPSULE
                and model.geom_bodyid[g] != 0]
    housing = []
    for g, name in capsules:
        if frozenset((g, floor)) not in pair_set:
            raise ValueError(f"missing physical {name}/table pair")
        guards = [g] if profile == "capsule" else [model.geom(f"{name}_guard{i}").id for i in range(2)]
        if not all(frozenset((guard, box)) in pair_set for guard in guards for box in payload):
            raise ValueError(f"missing physical {name}/payload guard")
        overhang = 0.
        if profile == "two-sphere-cover":
            if any(frozenset((g, box)) in pair_set for box in payload):
                raise ValueError(f"{name} still has an unsupported capsule/box pair")
            half_length = float(model.geom_size[g, 1])
            axis = _rotation(model.geom_quat[g])[:, 2]
            center = model.geom_pos[g]
            # MuJoCo may reverse its local z for fromto. Sort in capsule coordinates.
            offsets = sorted([(float((model.geom_pos[j]-center) @ axis), j) for j in guards])
            for side, (axial, j) in zip((-1., 1.), offsets, strict=True):
                error = np.linalg.norm(model.geom_pos[j]-center-axial*axis)
                r = float(model.geom_size[g, 0])
                reach = float(model.geom_size[j, 0])
                end_distance = abs(side*half_length-axial)+r
                seam_distance = np.hypot(axial, r)
                if model.geom_bodyid[g] != model.geom_bodyid[j] or reach < max(end_distance, seam_distance)+error:
                    raise ValueError(f"{name}: compiled guard does not cover the capsule")
                overhang = max(overhang, reach-r+float(error))
        housing.append(dict(capsule=name, payload_geoms=[model.geom(j).name for j in guards],
                            max_surface_overhang_m=overhang))
    if len(capsules) != 17:
        raise ValueError("expected both wrist housings and rods, and all thirteen link shells")
    return dict(profile=profile, ngeom=int(model.ngeom), eligible_pairs=len(pairs), shape_counts=shape_counts,
                housing=housing, max_overhang_m=max(e["max_surface_overhang_m"] for e in housing))


def load_actor(package):
    if package is None:
        from _bimanual_visual_policy import PRIOR, VisualPolicy
        return VisualPolicy, PRIOR.copy(), 96, 5, {"kind": "untrained-registered-prior"}
    from bimanual_policy_replay import _load
    policy, contract, _, policy_hash, checkpoint_hash = _load(
        package / "frozen-policy.json", package / "checkpoint.json")
    if (policy["task"], policy["arms"], policy.get("variant")) != ("push_t", "both", "registered"):
        raise ValueError("this transfer requires the registered two-arm PushT weights")
    directory = (package / "sources").resolve()
    names = ("_bimanual_image_pose", "_bimanual_visual_policy")
    for name in names:
        path = directory / (name + ".py")
        if digest(path.read_bytes()) != contract["sources"][path.name] or name in sys.modules:
            raise ValueError(f"could not bind the frozen actor dependency {name}")
    sys.path.insert(0, str(directory))
    try:
        modules = [importlib.import_module(name) for name in names]
    finally:
        sys.path.remove(str(directory))
    if any(Path(mod.__file__).resolve().parent != directory for mod in modules):
        raise ValueError("frozen actor resolved outside its verified sources")
    if list(modules[1].PARAMETERS) != policy["parameters"]:
        raise ValueError("actor parameter order differs from checkpoint")
    return (modules[1].VisualPolicy, policy["weights"], contract["image_size"], contract["action_repeat"],
            {"kind": "frozen-registered-transfer", "policy_sha256": policy_hash,
             "contract_sha256": policy["contract_sha256"], "checkpoint_sha256": checkpoint_hash,
             "actor_sources": {name + ".py": contract["sources"][name + ".py"] for name in names}})


class Tools:
    def __init__(self, model, scene, active, actor_type, weights, repeat):
        self.task = BimanualTask(scene, model, active)
        self.enabled = np.array([active in ("both", s) for s in SIDES])
        self.model, self.data = model, mj.MjData(model)
        arm = [model.actuator(f"{s}_m{i}").id for s, n in zip(SIDES, scene.arm_dofs) for i in range(n)]
        self.arm = np.array(arm)
        self.task.source_bounds = model.actuator_ctrlrange[self.arm]
        self.stroke = np.array([model.joint(f"{s}_stroke").id for s in SIDES])
        self.sq = model.jnt_qposadr[self.stroke]
        self.sv = model.jnt_dofadr[self.stroke]
        self.drive = np.array([model.actuator(f"{s}_drive").id for s in SIDES])
        self.guard_owners = {i: model.geom(model.geom(i).name.rsplit("_guard", 1)[0]).id
                             for i in range(model.ngeom) if "_guard" in model.geom(i).name}
        self.tip = [model.site(f"{s}_tool").id for s in SIDES]
        self.actor = actor_type("push_t", model, weights, variant="registered",
                                decision_dt=repeat*TIMESTEP, servo_gains=(self.task.kp, self.task.kd))
        self.gain_force = np.zeros(2)
        self.prev_load = np.zeros(2)
        self.action = np.zeros(14)

    def decide(self, state, rgb):
        q, v = np.asarray(state.q), np.asarray(state.v)
        # The frozen actor updates its arm FK from proprio. Supply the two
        # additional actual tool encoder readings for the moved TCP sites.
        self.actor.data.qpos[self.sq] = q[self.sq]
        self.action = self.actor.act(self.task.proprio(q, v), rgb)
        self.task.set_action(self.action)
        features = self.actor.features
        obj, goal = features.get("object_xy"), features.get("goal_xy")
        self.gain_force[:] = 0.
        if obj is None or goal is None:
            return
        ex = float(goal[0]-obj[0])
        # 25 N average near sliding gives >46 N total on the nominal floor.
        # Differential effort proportional to the observed signed yaw error
        # loads the trailing side of the T. No payload or marker state enters.
        common = 25.5 * np.tanh(max(0., ex)/.014)
        turn = 19. * np.clip(features["image_yaw_error"], -.6, .6)
        self.gain_force = np.clip(common + np.array((-turn, turn)), 0., RATED_N)

    def controls(self, state):
        m, d, task = self.model, self.data, self.task
        q, v = np.asarray(state.q), np.asarray(state.v)
        d.qpos[task.q_indices], d.qpos[self.sq] = q[task.q_indices], q[self.sq]
        d.qvel[:] = 0.
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        mj.mj_comVel(m, d)
        gravity = np.empty(m.nv)
        mj.mj_rne(m, d, 0, gravity)
        ctrl = np.zeros(m.nu)
        arm_tau = task.kp*(task.target-q[task.q_indices])-task.kd*v[task.v_indices]
        axes = d.xaxis[self.stroke].copy()
        for i, (site, motor) in enumerate(zip(self.tip, self.drive, strict=True)):
            jac = np.zeros((3, m.nv))
            mj.mj_jacSite(m, d, jac, None, site)
            if self.enabled[i]:
                # Flow governor uses tip/arm encoders and the previous actual
                # pad load, with 20 mm/s free approach before sustained contact.
                velocity = float(axes[i] @ (jac[:, task.v_indices] @ v[task.v_indices]
                                            + jac[:, self.sv] @ v[self.sv]))
                ceiling = .020 + .065*np.clip((self.prev_load[i]-2.)/12., 0., 1.)
                force = min(self.gain_force[i], 2200.*(ceiling-velocity))
                if self.gain_force[i] < 1.:
                    force = min(force, -650.*max(0., q[self.sq[i]]))
                ctrl[motor] = np.clip(force, -RATED_N, RATED_N)
                # Compensate wrist reaction within the original rotary motor
                # limits. The registered actor already requests up to 19 N.
                extra = max(0., ctrl[motor]-19.)
                arm_tau += jac[:, task.v_indices].T @ (extra*axes[i])
            else:
                # There is one idle wrist in each unilateral transfer. Its
                # mount holds the common initial posture and its drive is off.
                arm_tau[~task.enabled] += gravity[task.v_indices[~task.enabled]]
        ctrl[self.arm] = np.clip(arm_tau, *task.source_bounds.T)
        return ctrl, axes


class GuidedWrist(Tools):
    """Registered RGB actor with a five-axis carriage servo and real force-driven slide."""

    def __init__(self, model, scene, active, actor_type, weights, repeat):
        super().__init__(model, scene, active, actor_type, weights, repeat)
        self.rest = model.key("rated-level-start").qpos.copy()
        self.task.target[:] = self.rest[self.task.q_indices]
        self.data.qpos[:] = self.rest
        mj.mj_kinematics(model, self.data)
        mj.mj_comPos(model, self.data)
        self.mounts = [model.site(s+"_mount").id for s in SIDES]
        self.mount0 = self.data.site_xpos[self.mounts].copy()
        self.mount_goal = self.mount0.copy()
        self.arm_slots = np.split(np.arange(len(self.arm)), [scene.arm_dofs[0]])
        self.reference_j = []
        for site, ix in zip(self.mounts, self.arm_slots, strict=True):
            jac = np.empty((3, model.nv))
            mj.mj_jacSite(model, self.data, jac, None, site)
            self.reference_j.append(jac[:, self.task.v_indices[ix]].copy())
        self.repeat = repeat
        self.obj0 = None
        self.missing_s = np.inf
        self.face_yaw = 0.
        self.force_ramp = np.zeros(2)
        self.wrist_reaction = np.zeros((2, 3))
        self.control_observation = {}

    def decide(self, state, rgb):
        super().decide(state, rgb)
        f = self.actor.features
        # The registered estimator holds its last pose when a view disappears.
        # Its per-decision fit error is present only on fresh object detections.
        if f.get("object_fit_error_px") is not None:
            self.missing_s = 0.
        else:
            self.missing_s += self.repeat*TIMESTEP
        self.gain_force[:] = 0.
        if self.missing_s > .16 or f.get("object_xy") is None or f.get("goal_xy") is None:
            return
        obj, goal = np.asarray(f["object_xy"]), np.asarray(f["goal_xy"])
        if self.obj0 is None:
            self.obj0 = obj.copy()
        yaw_error = float(f["image_yaw_error"])
        self.face_yaw = float(np.clip(f["goal_yaw"]-yaw_error, -.30, .30))
        motion = obj-self.obj0
        for i, ix in enumerate(self.arm_slots):
            if not self.enabled[i]:
                continue
            # Keep the installed viewing frame near the load. The learned
            # action influences actual carriage travel through its FK Jacobian.
            qa = self.task.q_indices[ix]
            learned = self.reference_j[i] @ (self.task.target[ix]-self.rest[qa])
            delta = np.clip(learned[:2], [-.012, -.015], [.012, .015])
            desired = self.mount0[i].copy()
            desired[:2] += delta + np.clip([.30*motion[0], .40*motion[1]], [-.008, -.015], [.040, .015])
            ds = desired-self.mount_goal[i]
            max_travel = .025*self.repeat*TIMESTEP
            self.mount_goal[i] += ds*min(1., max_travel/max(np.linalg.norm(ds), 1e-12))
        # At 50 N total the two 28 N drives still have 6 N of differential.
        # Perform the turn during translation, before static floor drag resumes.
        total = 50.0*np.tanh(max(0., (goal[0]-obj[0])*self.actor.gains[2])/.006)
        budget = min(total, 2.*RATED_N-total)
        difference = np.clip(46.*self.actor.gains[4]*yaw_error, -budget, budget)
        self.gain_force = .5*(total + np.array((-difference, difference)))

    def controls(self, state):
        m, d, task = self.model, self.data, self.task
        q, v = np.asarray(state.q), np.asarray(state.v)
        d.qpos[task.q_indices], d.qpos[self.sq] = q[task.q_indices], q[self.sq]
        d.qvel[task.v_indices], d.qvel[self.sv] = v[task.v_indices], v[self.sv]
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        mj.mj_comVel(m, d)
        bias = np.empty(m.nv)
        mj.mj_rne(m, d, 0, bias)
        axes = d.xaxis[self.stroke].copy()
        ctrl = np.zeros(m.nu)
        sensed = -np.sum(self.wrist_reaction*axes, axis=1)
        exposed = np.sum((d.site_xpos[self.tip]-d.site_xpos[self.mounts])*axes, axis=1)
        for i, ix in enumerate(self.arm_slots):
            qa, va = task.q_indices[ix], task.v_indices[ix]
            motor, axis = self.drive[i], axes[i]
            jv, jw = np.empty((3, m.nv)), np.empty((3, m.nv))
            mj.mj_jacSite(m, d, jv, jw, self.mounts[i])
            tip_j = np.empty((3, m.nv))
            mj.mj_jacSite(m, d, tip_j, None, self.tip[i])
            speed = float(axis @ (tip_j[:, va] @ v[va] + tip_j[:, self.sv[i]]*v[self.sv[i]]))
            target_force = self.gain_force[i]
            if self.enabled[i]:
                # In free space use a velocity servo with a 20 ms time scale
                # for the roughly 0.4 kg reflected moving mass. Ramp the true
                # force only after the prior pad wrench indicates axial loading.
                load = max(0., sensed[i])
                demand = target_force if load > .10 else 0.
                self.force_ramp[i] += np.clip(demand-self.force_ramp[i], -280.*TIMESTEP, 90.*TIMESTEP)
                speed_ref = (.014+.052*np.clip(load/12., 0., 1.)) if target_force > .5 else 0.
                # Retract gently on loss of vision/completion, using only the
                # measured wrist stroke. Keep both software and physical stops
                # behind the normal working range; all contacts are still audited.
                if target_force < .5 or q[self.sq[i]] > GUIDE_STROKE[1]-.024:
                    speed_ref = -.022*np.tanh(q[self.sq[i]]/.025)
                    self.force_ramp[i] = min(self.force_ramp[i], max(0., load-280.*TIMESTEP))
                if q[self.sq[i]] < GUIDE_STROKE[0]+.022:
                    speed_ref = max(speed_ref, .016)
                feed = min(self.force_ramp[i], load+90.*TIMESTEP)
                ctrl[motor] = np.clip(feed+20.*(speed_ref-speed), -RATED_N, RATED_N)
            desired_axis = np.array((np.cos(self.face_yaw), np.sin(self.face_yaw), 0.)) if self.enabled[i] else np.array((1., 0., 0.))
            rot_error = np.cross(axis, desired_axis)
            # Position plus the two rotational DOFs perpendicular to the tool.
            # The remaining joint-space freedom keeps the original bent mount.
            rotational = np.eye(3)-np.outer(axis, axis)
            j = np.vstack((jv[:, va], .14*rotational @ jw[:, va]))
            err = np.r_[self.mount_goal[i]-d.site_xpos[self.mounts[i]], .14*rot_error]
            pinv = np.linalg.solve(j @ j.T + .018**2*np.eye(6), j).T
            dq = pinv @ err + (np.eye(len(ix))-pinv @ j) @ (self.rest[qa]-q[qa])
            # Inverse dynamics uses robot q/v and gravity. Counter the actual
            # plunger reaction at its mount using the original rotary motors.
            tau = (bias[va] + jv[:, va].T @ (ctrl[motor]*axis)
                   + 210.*np.clip(dq, -.24, .24)-7.*v[va])
            ids = self.arm[ix]
            ctrl[ids] = np.clip(tau, *m.actuator_ctrlrange[ids].T)
        self.control_observation = dict(actual_mount_xyz_m=d.site_xpos[self.mounts].tolist(),
                                        requested_mount_xyz_m=self.mount_goal.tolist(), tip_xyz_m=d.site_xpos[self.tip].tolist(),
                                        indicated_exposed_m=exposed.tolist(), previous_axial_pad_n=sensed.tolist(),
                                        requested_tool_yaw_rad=self.face_yaw)
        return ctrl, axes


def measured_step(tool, before, after, axes, diagnostic):
    m, task = tool.model, tool.task
    diagnostic.qpos[:] = before.q
    mj.mj_kinematics(m, diagnostic)
    mj.mj_comPos(m, diagnostic)
    step = after.last_step
    if step.tick != after.tick or before.tick+1 != after.tick or abs(step.input_time-before.time) > 1e-7:
        raise RuntimeError("contact is not bound to the commanded input step")
    forces, approach = np.zeros((2, 3)), np.zeros(2)
    foreign, housing_gaps = {}, {}
    for contact in step.contacts:
        a, b = map(int, contact.geometry_ids)
        normal = float(contact.wrench_contact_frame[0])
        if normal <= 0.:
            continue
        other = b if a in task.payload_geoms else a if b in task.payload_geoms else None
        if other == task.floor:
            continue
        if other not in task.pad_owner:
            pair = "+".join(sorted((m.geom(a).name, m.geom(b).name)))
            foreign[pair] = foreign.get(pair, 0.) + normal
            if other in tool.guard_owners:
                box = b if other == a else a
                source = tool.guard_owners[other]
                housing_gaps[pair] = float(mj.mj_geomDistance(m, diagnostic, source, box, .5, None))
            continue
        i = task.pad_owner[other]
        frame = np.asarray(contact.frame).reshape(3, 3)
        forces[i] += (1. if other == a else -1.)*frame.T @ np.asarray(contact.wrench_contact_frame[:3])
        ja, jb = (np.empty((3, m.nv)) for _ in range(2))
        for geom, jac in ((a, ja), (b, jb)):
            mj.mj_jac(m, diagnostic, jac, None, contact.position, int(m.geom_bodyid[geom]))
        closing = float(frame[0] @ ((ja-jb) @ before.v))
        approach[i] = max(approach[i], closing)
    actual_drive = np.asarray(step.qfrc_actuator)[tool.sv]
    axial = np.sum(forces*axes, axis=1)
    return dict(actual_drive_n=actual_drive.tolist(), input_axes=axes.tolist(),
                tip_force_world_n=forces.tolist(), lateral_force_n=np.linalg.norm(forces-axial[:, None]*axes, axis=1).tolist(),
                approach_m_s=approach.tolist(), unexpected_pairs_n=foreign, exact_housing_gap_on_guard_contact_m=housing_gaps,
                stroke_m=np.asarray(after.q)[tool.sq].tolist())


def summarize(rows, active, limits=STROKE):
    def column(key):
        return np.asarray([r[key] for r in rows])
    magnitude = np.linalg.norm(column("tip_force_world_n"), axis=2)
    start = column("com_travel_m")
    angle = abs(column("yaw_rad"))
    loaded = magnitude >= 1.
    onset = loaded & ~np.vstack((np.zeros((1, 2), bool), loaded[:-1]))
    strikes = float(np.max(column("approach_m_s")[onset], initial=0.))
    stroke = column("stroke_m")
    enabled = np.array([active in ("both", s) for s in SIDES])
    loaded_span = best = 0
    for row in magnitude:
        loaded_span = loaded_span+1 if np.all(row[enabled] >= 18.) else 0
        best = max(best, loaded_span)
    window = min(len(rows), round(.020/TIMESTEP))
    cs = np.vstack((np.zeros((1, 2)), magnitude.cumsum(axis=0)))
    rolling_peak = np.max((cs[window:]-cs[:-window])/window, axis=0)
    offenders = {}
    for row in rows:
        for name, normal in row["unexpected_pairs_n"].items():
            offenders[name] = max(offenders.get(name, 0.), normal)
    reserve = float(np.min(np.minimum(stroke-limits[0], limits[1]-stroke)))
    raw_peak = np.max(abs(column("actual_drive_n")), axis=0)
    gates = dict(stroke_reserve=reserve >= .01, actuator_rating=bool(np.all(raw_peak <= RATED_N+.001)),
                 only_expected_contacts=max(offenders.values(), default=0.) < .01,
                 low_speed_engagement=strikes <= .025, meaningful_loading=best*TIMESTEP >= .20,
                 sustained_tip_force_within_10pct=bool(np.all(rolling_peak <= 1.10*RATED_N)))
    if active == "both":
        gates["original_pose_dwell"] = bool(rows[-1]["success"])
        gates["goal_held_at_window_end"] = rows[-1]["goal_dwell_s"]+1e-12 >= GOAL_DWELL_S
    else:
        gates["idle_tip_unloaded"] = bool(magnitude[:, ~enabled].max() < 1.)
        gates["single_does_not_move"] = bool(start.max() < .002 and angle.max() <= np.deg2rad(2.))
    return dict(passed=all(gates.values()), gates=gates, real_pose_success=bool(rows[-1]["success"]),
                max_com_displacement_m=float(start.max()), max_abs_yaw_rad=float(angle.max()),
                max_actuator_n=raw_peak.tolist(), peak_tip_resultant_n=magnitude.max(axis=0).tolist(),
                max_20ms_mean_tip_resultant_n=rolling_peak.tolist(), max_lateral_n=column("lateral_force_n").max(axis=0).tolist(),
                loaded_continuously_s=best*TIMESTEP, min_travel_reserve_m=reserve,
                peak_closing_speed_at_contact_m_s=strikes, unexpected_pair_peak_n=offenders,
                min_floor_vertical_n=float(column("floor_vertical_n").min()),
                final_goal_error_m=rows[-1]["goal_error_m"], final_yaw_error_rad=rows[-1]["yaw_error_rad"],
                final_goal_dwell_s=rows[-1]["goal_dwell_s"])


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, obj):
    path.write_text(json.dumps(obj, default=json_default, allow_nan=False, indent=2), encoding="utf-8")


def run(args):
    actor_type, weights, size, repeat, actor_info = load_actor(args.policy)
    source, scene = make_rated_scene(args.seed, args.guard_profile, args.controller)
    args.out.mkdir(parents=True, exist_ok=False)
    model_path = args.out / "rated-policy.xml"
    model_path.write_bytes(scene.xml.encode())
    comp = compile_mujoco_xml(scene.xml, require_asset=False)
    model = comp.mj_model
    layout = guard_layout(model, args.guard_profile)
    write_json(args.out/"guard-layout.json", layout)
    guided = args.controller == "guided-wrist"
    controller_type = GuidedWrist if guided else Tools
    actors = [controller_type(model, scene, name, actor_type, weights, repeat) for name in args.groups]
    keys = tuple(range(len(actors)))
    start_q = model.key("rated-level-start").qpos if guided else model.qpos0
    initial = {k: MuJoCoInitialState(start_q.copy()) for k in keys}
    options = dict(cache_root=args.cache_dir, contact_capacity=32, constraint_capacity=128) if args.mode == "musa" else {}
    rows = {k: [] for k in keys}
    traces = {k: dict(qpos=[], ticks=[], time_s=[], action=[], rewards=[]) for k in keys}
    diagnostics = {k: mj.MjData(model) for k in keys}
    pixels = {k: hashlib.sha256() for k in keys}
    metrics, potentials, initial_com = {}, {}, {}
    terminals, evidence = {}, {k: [] for k in keys}
    with ExitStack() as stack:
        runtime = stack.enter_context(MuJoCoRuntime(comp, mode=args.mode, world_keys=keys, initial_states=initial, **options))
        renderer = stack.enter_context(BimanualWristRenderer(model, args.cache_dir/"render",
                                       library_path=args.library, device_index=args.device_index, size=(size, size)))
        snapshot = runtime.snapshot if args.mode == "cpu" else runtime.integration_snapshot
        backend = runtime.capabilities["backend"]
        before_device = runtime.musa_device_report() if args.mode == "musa" else None
        visual_device = renderer.device_report
        states = snapshot()
        for k, tool in enumerate(actors):
            if not (np.array_equal(states[0].q, states[k].q) and np.array_equal(states[0].v, states[k].v)):
                raise RuntimeError("initial matched controls differ")
            q, v = np.asarray(states[k].q), np.asarray(states[k].v)
            metrics[k] = tool.task.evaluate(q, v, (), advance=False)
            addr = tool.task.root_q
            ipos = model.body_ipos[model.body("payload").id]
            initial_com[k] = q[addr:addr+3]+_rotation(q[addr+3:addr+7]) @ ipos
            potentials[k] = tool.task.potential(metrics[k])
            traces[k]["qpos"].append(q.copy())
            traces[k]["ticks"].append(int(states[k].tick))
            traces[k]["time_s"].append(float(states[k].time))
        for step in range(args.decisions):
            was_success = {k: metrics[k]["success"] for k in keys}
            for k, tool in enumerate(actors):
                st = states[k]
                rgb = renderer.observe(st.q, st.tick, st.time, epoch=k)
                pixels[k].update(rgb.tobytes())
                tool.decide(st, rgb)
                traces[k]["action"].append(tool.action.copy())
                features = tool.actor.features
                evidence[k].append(dict(input_tick=int(st.tick), input_time_s=float(st.time),
                                   wrist_rgb_sha256=[digest(image.tobytes()) for image in rgb],
                                   visual_object_xy=features.get("object_xy"), visual_goal_xy=features.get("goal_xy"),
                                   visual_yaw_error_rad=features.get("image_yaw_error"),
                                   drive_demand_n=tool.gain_force.tolist()))
            for _ in range(repeat):
                pending = {k: actors[k].controls(states[k]) for k in keys}
                runtime.set_controls({k: MuJoCoControl(pending[k][0]) for k in keys})
                previous = states
                runtime.step(TIMESTEP)
                states = snapshot()
                for k, tool in enumerate(actors):
                    st = states[k]
                    ctrl, axes = pending[k]
                    row = measured_step(tool, previous[k], st, axes, diagnostics[k])
                    q = np.asarray(st.q)
                    mt = tool.task.evaluate(q, np.asarray(st.v), st.last_step.contacts)
                    metrics[k] = mt
                    tool.prev_load = np.asarray(mt["pad_normal_n"])
                    if guided:
                        tool.wrist_reaction[:] = mt["wrist_force_world_n"]
                        row.update(tool.control_observation)
                    addr = tool.task.root_q
                    com = q[addr:addr+3]+_rotation(q[addr+3:addr+7]) @ ipos
                    row.update(time_s=float(st.time), tick=int(st.tick), input_time_s=float(st.last_step.input_time),
                               motor_ctrl=ctrl.tolist(), image_drive_demand_n=tool.gain_force.tolist(),
                               com_travel_m=float(np.linalg.norm((com-initial_com[k])[:2])),
                               floor_vertical_n=mt["floor_vertical_n"], yaw_rad=mt["payload_yaw_rad"],
                               goal_error_m=mt["goal_error_xy_m"], yaw_error_rad=mt["goal_yaw_error_rad"],
                               goal_dwell_s=mt["goal_dwell_s"], success=mt["success"])
                    rows[k].append(row)
            for k in keys:
                st, trace = states[k], traces[k]
                p = actors[k].task.potential(metrics[k])
                trace["rewards"].append(p-potentials[k]+float(metrics[k]["success"] and not was_success[k]))
                potentials[k] = p
                trace["qpos"].append(np.array(st.q, copy=True))
                trace["ticks"].append(int(st.tick))
                trace["time_s"].append(float(st.time))
                if metrics[k]["success"] and k not in terminals:
                    terminals[k] = (step+1, dict(metrics[k]), pixels[k].hexdigest())
            if (step+1) % 25 == 0:
                print(f"decision {step+1}/{args.decisions}", flush=True)
        after_device = runtime.musa_device_report() if args.mode == "musa" else None
        if after_device is not None and (after_device["backend"] != "musa" or not after_device["arch"]
                or after_device["completed_kernels"] <= before_device["completed_kernels"]
                or after_device["completed_kernels"] != after_device["launched_kernels"]):
            raise RuntimeError("device did not complete the paired physical trajectories")
    groups = {}
    for k, name in enumerate(args.groups):
        trace = traces[k]
        # Film the actual episode through its first terminal decision. The full
        # matched trial continues for drift, unloading and single-arm checks.
        count, final, rgb_hash = terminals.get(k, (args.decisions, metrics[k], pixels[k].hexdigest()))
        if count < args.decisions:
            np.savez_compressed(args.out/f"{name}-full.npz", **trace, model_sha256=scene.xml_sha256)
        trace = {field: values[:count+(field in ("qpos", "time_s", "ticks"))]
                 for field, values in trace.items()}
        path = args.out / f"{name}.npz"
        np.savez_compressed(path, **trace, model_sha256=scene.xml_sha256)
        path.with_suffix(".xml").write_bytes(scene.xml.encode())
        groups[name] = summarize(rows[k], name, model.jnt_range[actors[k].stroke[0]])
        with gzip.open(args.out/f"{name}-forces.jsonl.gz", "wt", encoding="utf-8") as out:
            for row in rows[k]:
                out.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
        mt = dict(final, task=scene.task, seed=args.seed, mode=args.mode, active_arms=name,
                  xml_sha256=scene.xml_sha256, tick=trace["ticks"][-1], time_s=trace["time_s"][-1])
        write_json(args.out/f"{name}.json", dict(task=scene.task, seed=args.seed, arm_dofs=scene.arm_dofs,
                   physics=args.mode, success=mt["success"], terminated=mt["success"], truncated=not mt["success"],
                   final_metrics=mt, physics_ticks=mt["tick"], decisions=count, rewards=trace["rewards"],
                   return_sum=sum(trace["rewards"]), scene_sha256=scene.xml_sha256, trajectory_sha256=digest(path.read_bytes()),
                   sensor_rgb_sha256=rgb_hash, job=dict(arms=name, black=False, label=actor_info["kind"]),
                   tool_profile="single-pad", wrist_drive="rated-axial", housing_profile=args.guard_profile,
                   controller=args.controller,
                   rated_control=groups[name], policy=actor_info))
        write_json(args.out/f"{name}-vision.json", evidence[k])
    result = dict(experiment="articulated rated visual PushT transfer", mode=args.mode, backend=backend,
                  controller=args.controller,
                  seed=args.seed, source_xml_sha256=source.xml_sha256, rated_xml_sha256=scene.xml_sha256,
                  housing_profile=args.guard_profile, housing_max_overhang_m=layout["max_overhang_m"],
                  source_script_sha256=digest(Path(__file__).read_bytes()), policy=actor_info,
                  goal_xy_m=scene.goal_xy, goal_yaw_rad=scene.goal_yaw, payload_mass_kg=scene.mass,
                  actuator_rating_n=RATED_N, decisions=args.decisions, action_repeat=repeat, timestep_s=TIMESTEP,
                  renderer=visual_device, device_before=before_device, device_after=after_device, groups=groups,
                  complete_comparison=set(args.groups) == {"both", "left", "right"},
                  passed=all(r["passed"] for r in groups.values()))
    write_json(args.out/"summary.json", result)
    print(json.dumps(groups, allow_nan=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    p = parser.add_mutually_exclusive_group(required=True)
    p.add_argument("--policy", type=Path, help="extracted trained policy directory including sources/")
    p.add_argument("--prior", action="store_true", help="develop using the untrained registered actor")
    parser.add_argument("--seed", type=int, default=55001)
    parser.add_argument("--mode", choices=("cpu", "musa"), default="cpu")
    parser.add_argument("--controller", choices=("transfer", "guided-wrist"), default="transfer",
                        help="original unrestricted actor transfer or level carriage with tactile force ramp")
    parser.add_argument("--guard-profile", choices=("capsule", "two-sphere-cover"), default="capsule",
                        help="original exact capsules or a conservative payload guard with two spheres per capsule")
    parser.add_argument("--groups", choices=("both", "left", "right"), nargs="+", default=["both", "left", "right"])
    parser.add_argument("--decisions", type=int, default=360)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="a fresh output directory")
    args = parser.parse_args()
    if args.decisions < 1 or len(set(args.groups)) != len(args.groups) or args.seed < 0:
        parser.error("use positive decisions, distinct groups and a nonnegative seed")
    if not run(args)["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
