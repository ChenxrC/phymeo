"""Run unchanged primitive G1 through public MuJoCo-TYX and independent CPU checks.

Only TYX advances the device trajectory. Public snapshot() uses an explicitly
detached CPU observer; its endpoint contact forces are never used as device
loads. Native same-input mj_forward and an independent feedback CPU trajectory
are separate oracles, never fallback state producers.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np
import onnxruntime as ort
import yaml

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco, compile_mujoco_xml

ASSET_NAME = "g1_primitive_actuated.xml"
ASSET_SHA = "f7e856e057a095a9f8708d302531d40ff2519440012c24488407ae8d0f0586aa"
FILES = (ASSET_NAME, "deploy.yaml", "dance1_subject2.npz", "policy.onnx", "policy.onnx.data")
SETTLING = 0.25
GATES = {
    "minimum_height_m": 0.35,
    "maximum_tilt_rad": 1.0,
    "mean_foot_load_ratio": (0.5, 1.5),
    "maximum_foot_load_ratio": 8.0,
    "maximum_joint_rmse_rad": 0.5,
    "maximum_reference_xy_error_m": 1.0,
    "settling_time_s": SETTLING,
}
TOL = {
    "body_position": (3e-5, 4e-5),
    "body_com": (3e-5, 4e-5),
    "body_rotation": (3e-5, 4e-5),
    "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (3e-4, 5e-5),
    "bias": (4e-4, 4e-4),
    "passive": (4e-4, 4e-4),
    "applied": (4e-4, 4e-4),
    "smooth_acceleration": (2e-3, 4e-3),
}


def json_value(value):
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def binding(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class Journal:
    def __init__(self, path):
        self.path = path.resolve()
        self.summary_path = self.path.with_suffix(self.path.suffix + ".summary.json")
        if self.path.exists() or self.summary_path.exists():
            raise FileExistsError("output and summary must both be fresh paths")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Large G1 receipts retain every field while streaming lossless storage.
        # Exclusive creation and normal finally-close preserve the original policy.
        self.stream = (
            gzip.open(self.path, "xt", encoding="utf-8")
            if self.path.suffix == ".gz"
            else self.path.open("x", encoding="utf-8")
        )
        self.sequence = 0

    def write(self, event, *, durable=False, **payload):
        self.stream.write(
            json.dumps(
                json_value({"sequence": self.sequence, "event": event, **payload}),
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        self.sequence += 1
        self.stream.flush()
        if durable:
            os.fsync(self.stream.fileno())

    def finish(self, summary):
        self.write("summary", durable=True, **summary)
        with self.summary_path.open("x", encoding="utf-8") as stream:
            json.dump(json_value(summary), stream, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def close(self):
        self.stream.close()


def rotation(quaternion):
    # Native FK normalizes a local observation copy; published q is untouched.
    q = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("invalid observed quaternion")
    result = np.zeros(9)
    mj.mju_quat2Mat(result, q / norm)
    return result.reshape(3, 3)


def axis_rotation(axis, angle):
    q = np.zeros(4)
    q[0] = np.cos(angle / 2)
    q[axis + 1] = np.sin(angle / 2)
    return rotation(q)


def torso(q, joints):
    return (
        rotation(q)
        @ axis_rotation(2, joints[12])
        @ axis_rotation(0, joints[13])
        @ axis_rotation(1, joints[14])
    )


def yaw(matrix):
    return axis_rotation(2, np.arctan2(matrix[1, 0], matrix[0, 0]))


def compile_source(assets, h):
    path = assets / ASSET_NAME
    if binding(path)["sha256"] != ASSET_SHA:
        raise ValueError("G1 asset differs from the predeclared primitive model")
    if h == 0.002:
        compiled = compile_mujoco(path, require_asset=False)
        derived = None
    else:
        xml = ET.fromstring(path.read_text(encoding="utf-8"))
        option = xml.find("option")
        if option is None:
            option = ET.SubElement(xml, "option")
        option.set("timestep", format(h, ".17g"))
        text = ET.tostring(xml, encoding="unicode")
        derived = {
            "change": "option.timestep only",
            "h": h,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        compiled = compile_mujoco_xml(text, require_asset=False)
    model = compiled.fresh_native_model()
    if (model.nq, model.nv, model.nu, model.nbody, model.ngeom) != (36, 35, 29, 31, 34):
        raise ValueError("unexpected unchanged G1 model dimensions")
    if int(model.opt.iterations) != 100 or float(model.opt.timestep) != h:
        raise ValueError("source iteration budget or timestep changed")
    if int(model.opt.integrator) != int(mj.mjtIntegrator.mjINT_EULER):
        raise ValueError("G1 source integrator was changed")
    if int(model.opt.cone) != int(mj.mjtCone.mjCONE_PYRAMIDAL):
        raise ValueError("G1 source cone was changed")
    if len(model.exclude_signature) != 4 or np.any(model.dof_frictionloss):
        raise ValueError("G1 exclusions or original dry friction changed")
    return compiled, model, derived


class Protocol:
    def __init__(self, model, assets):
        self.model = model
        self.config = yaml.safe_load((assets / "deploy.yaml").read_text(encoding="utf-8"))
        with np.load(assets / "dance1_subject2.npz", allow_pickle=False) as archive:
            self.motion = {key: archive[key].copy() for key in archive.files}
        self.session = ort.InferenceSession(
            str(assets / "policy.onnx"), providers=["CPUExecutionProvider"]
        )
        if self.session.get_inputs()[0].shape != [1, 154]:
            raise ValueError("policy input is not the original 154-vector")
        if self.session.get_outputs()[0].shape != [1, 29]:
            raise ValueError("policy output is not the original 29-vector")
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("this harness declares CPU ONNX inference only")
        self.joints = model.actuator_trnid[:, 0].astype(int)
        self.qa = model.jnt_qposadr[self.joints].astype(int)
        self.va = model.jnt_dofadr[self.joints].astype(int)
        if len(set(self.qa)) != 29 or self.config["joint_ids_map"] != list(range(29)):
            raise ValueError("unexpected native actuator or policy ordering")
        self.names = tuple(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(j)) for j in self.joints)
        self.root = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
        self.floor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_geoms = {}
        self.foot_bodies = []
        for side, prefix in enumerate(("left", "right")):
            self.foot_bodies.append(
                mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, prefix + "_ankle_roll_link")
            )
            for i in range(1, 8):
                geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"{prefix}_foot{i}_collision")
                if geom < 0:
                    raise ValueError("original foot geometry is absent")
                self.foot_geoms[geom] = side
        if self.root < 0 or self.floor < 0 or min(self.foot_bodies) < 0:
            raise ValueError("original root/floor/foot body mapping is absent")
        self.kp = np.asarray(self.config["stiffness"], float)
        self.kd = np.asarray(self.config["damping"], float)
        action = self.config["actions"]["JointPositionAction"]
        self.scale = np.asarray(action["scale"], float)
        self.offset = np.asarray(action["offset"], float)
        self.default = np.asarray(self.config["default_joint_pos"], float)
        if float(self.config["step_dt"]) != 0.02:
            raise ValueError("original policy period changed")
        if any(
            x.shape != (29,) or not np.isfinite(x).all()
            for x in (self.kp, self.kd, self.scale, self.offset, self.default)
        ):
            raise ValueError("invalid original controller vectors")
        self.inertial_from_body = rotation(model.body_iquat[self.root]).T
        self.mass = float(np.sum(model.body_mass))
        self.weight = self.mass * float(-model.opt.gravity[2])
        if not np.array_equal(model.opt.gravity, (0.0, 0.0, -9.81)):
            raise ValueError("the source gravity was changed")
        self.q0, self.v0 = self.initial(0)
        self.alignment = (
            yaw(torso(self.q0[3:7], self.q0[self.qa]))
            @ yaw(rotation(self.motion["body_quat_w"][0, 0])).T
        )

    def initial(self, frame=None):
        q, v = self.model.qpos0.copy(), np.zeros(self.model.nv)
        if frame is not None:
            q[:3] = self.motion["body_pos_w"][frame, 0]
            q[3:7] = self.motion["body_quat_w"][frame, 0]
            q[3:7] /= np.linalg.norm(q[3:7])
            q[self.qa] = self.motion["joint_pos"][frame]
            v[:3] = self.motion["body_lin_vel_w"][frame, 0]
            v[3:6] = rotation(q[3:7]).T @ self.motion["body_ang_vel_w"][frame, 0]
            v[self.va] = self.motion["joint_vel"][frame]
        return q, v

    def frame(self, time):
        return min(int(round(time * 50)), len(self.motion["joint_pos"]) - 1)

    def metrics(self, q, v, time, stand=False):
        frame = self.frame(time)
        reference = self.default if stand else self.motion["joint_pos"][frame]
        root_ref = (
            self.model.qpos0[:3]
            if stand
            else (
                self.q0[:3]
                + self.alignment
                @ (self.motion["body_pos_w"][frame, 0] - self.motion["body_pos_w"][0, 0])
            )
        )
        low, high = self.model.jnt_range[self.joints].T
        return {
            "finite": bool(np.isfinite(q).all() and np.isfinite(v).all()),
            "height": float(q[2]),
            "tilt": float(np.arccos(np.clip(rotation(q[3:7])[2, 2], -1, 1))),
            "joint_rmse": float(np.sqrt(np.mean((q[self.qa] - reference) ** 2))),
            "reference_xy_error": float(np.linalg.norm(q[:2] - root_ref[:2])),
            "limit_excess": np.maximum(0, np.maximum(low - q[self.qa], q[self.qa] - high)),
            "reference_frame": frame,
        }


class Controller:
    def __init__(self, protocol, q, h, stand=False):
        self.p, self.stand = protocol, stand
        self.decimation = int(round(0.02 / h))
        self.action = np.zeros(29)
        self.target = protocol.default.copy() if stand else q[protocol.qa].copy()

    def propose(self, q, v, time, tick):
        p = self.p
        action, target, observation = self.action.copy(), self.target.copy(), None
        if not self.stand and tick % self.decimation == 0:
            frame = p.frame(time)
            reference_q, reference_v = (p.motion["joint_pos"][frame], p.motion["joint_vel"][frame])
            relative = (
                torso(q[3:7], q[p.qa]).T
                @ p.alignment
                @ torso(p.motion["body_quat_w"][frame, 0], reference_q)
            )
            # Match native BODY local observation axes (body inertial frame),
            # evaluated from the current state rather than stale cached cvel.
            observation = np.concatenate(
                (
                    reference_q,
                    reference_v,
                    relative[:, :2].reshape(-1),
                    p.inertial_from_body @ v[3:6],
                    q[p.qa] - p.default,
                    v[p.va],
                    self.action,
                )
            ).astype(np.float32)[None]
            if observation.shape != (1, 154) or not np.isfinite(observation).all():
                raise ValueError("invalid policy observation")
            action = p.session.run(["actions"], {"obs": observation})[0][0]
            if action.shape != (29,) or not np.isfinite(action).all():
                raise ValueError("invalid policy action")
            target = p.offset + p.scale * action
        raw = p.kp * (target - q[p.qa]) - p.kd * v[p.va]
        ctrl = np.clip(raw, p.model.actuator_ctrlrange[:, 0], p.model.actuator_ctrlrange[:, 1])
        return {
            "action": action,
            "target": target,
            "observation": observation,
            "ctrl": ctrl,
            "unclamped_ctrl": raw,
            "saturated": raw != ctrl,
        }

    def commit(self, proposal):
        self.action = proposal["action"].copy()
        self.target = proposal["target"].copy()


def packet_for(protocol, proposal, grid_time, push_sign):
    wrench = np.zeros((protocol.model.nbody, 6))
    if 0.5 <= grid_time < 0.6:
        wrench[protocol.root, 0] = push_sign * 0.1 * protocol.weight
    return MuJoCoControl(
        proposal["ctrl"], qfrc_applied=np.zeros(protocol.model.nv), xfrc_applied=wrench
    )


def stored_packet(packet):
    return MuJoCoControl(
        np.asarray(packet.ctrl, np.float32),
        qfrc_applied=np.asarray(packet.qfrc_applied, np.float32),
        xfrc_applied=np.asarray(packet.xfrc_applied, np.float32),
    )


def set_native(data, packet):
    data.ctrl[:] = packet.ctrl
    data.qfrc_applied[:] = packet.qfrc_applied
    data.xfrc_applied[:] = packet.xfrc_applied


def native_fields(model, data, input_q, input_v, packet, *, full):
    if mj.mj_isSparse(model):
        raise ValueError("this fixed G1 oracle expects its native dense Jacobian")
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        frame = np.asarray(contact.frame).reshape(3, 3).copy()
        bodies = tuple(int(model.geom_bodyid[g]) for g in contact.geom)
        jp, jr = [], []
        for body in bodies:
            linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mj.mj_jac(model, data, linear, angular, contact.pos, body)
            jp.append(linear)
            jr.append(angular)
        jac = np.vstack((frame @ (jp[1] - jp[0]), frame @ (jr[1] - jr[0])))
        force = np.zeros(6)
        mj.mj_contactForce(model, data, index, force)
        contacts.append(
            {
                "index": index,
                "geometry_ids": tuple(map(int, contact.geom)),
                "body_ids": bodies,
                "dimension": int(contact.dim),
                "distance": float(contact.dist),
                "position": contact.pos.copy(),
                "frame": frame,
                "include_margin": float(contact.includemargin),
                "active": contact.efc_address >= 0,
                "wrench_contact_frame": force,
                "jacobian": jac,
                "efc_address": int(contact.efc_address),
            }
        )
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jaccom = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(model.nbody):
        mj.mj_jac(model, data, jaccom[body, :3], jaccom[body, 3:], data.xipos[body], body)
        applied += jaccom[body].T @ packet.xfrc_applied[body]
    energy = energy_fields(
        model,
        input_v,
        mass,
        data.xipos,
        data.qfrc_actuator,
        applied,
        data.qfrc_passive,
        data.qfrc_constraint,
    )
    result = {
        "contacts": contacts,
        "qacc": data.qacc.copy(),
        "qfrc_constraint": data.qfrc_constraint.copy(),
        "qfrc_actuator": data.qfrc_actuator.copy(),
        "energy": energy,
        "nefc": int(data.nefc),
        "body_com": data.xipos.copy(),
        "geom_position": data.geom_xpos.copy(),
    }
    if full:
        jac = np.asarray(data.efc_J).reshape(data.nefc, model.nv).copy()
        result.update(
            {
                "input_q": input_q.copy(),
                "input_v": input_v.copy(),
                "body_position": data.xpos.copy(),
                "body_rotation": data.xmat.reshape(model.nbody, 3, 3).copy(),
                "body_com": data.xipos.copy(),
                "jacobian_com": jaccom,
                "mass_matrix": mass,
                "bias": data.qfrc_bias.copy(),
                "passive": data.qfrc_passive.copy(),
                "applied": applied,
                "smooth_acceleration": data.qacc_smooth.copy(),
                "integration_acceleration": np.linalg.solve(
                    mass + model.opt.timestep * np.diag(model.dof_damping), mass @ data.qacc
                ),
                "efc_type": data.efc_type.copy(),
                "efc_id": data.efc_id.copy(),
                "efc_J": jac,
                "efc_pos": data.efc_pos.copy(),
                "efc_margin": data.efc_margin.copy(),
                "efc_vel": data.efc_vel.copy(),
                "efc_R": data.efc_R.copy(),
                "efc_D": data.efc_D.copy(),
                "efc_aref": data.efc_aref.copy(),
                "efc_force": data.efc_force.copy(),
                "efc_diagApprox": data.efc_diagApprox.copy(),
                "efc_imp": data.efc_KBIP.reshape(data.nefc, 4)[:, 2].copy(),
            }
        )
    return result


def native_oracle(model, q, v, packet, time, *, full):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = q, v, time
    set_native(data, packet)
    mj.mj_forward(model, data)
    return native_fields(model, data, q, v, packet, full=full), data


def energy_fields(model, v, mass, com, actuator, applied, passive, constraint):
    return {
        "kinetic": float(0.5 * v @ mass @ v),
        "gravity_potential": float(-np.sum(model.body_mass[:, None] * com * model.opt.gravity)),
        "actuator_power": float(actuator @ v),
        "external_power": float((applied - actuator) @ v),
        "passive_power": float(passive @ v),
        "viscous_power": float(-np.sum(model.dof_damping * v * v)),
        "constraint_power": float(constraint @ v),
        "phase": "actual_input_dynamics",
        "balance_scope": "kinetic+gravity; soft-contact potential not claimed",
    }


def field(item, name):
    return item[name] if isinstance(item, Mapping) else getattr(item, name)


def contact_rows(contacts):
    return [
        {
            name: field(c, name)
            for name in (
                "geometry_ids",
                "body_ids",
                "dimension",
                "distance",
                "position",
                "frame",
                "include_margin",
                "active",
                "wrench_contact_frame",
            )
        }
        for c in contacts
    ]


def support(protocol, contacts, v, body_com):
    foot_force, foot_torque = np.zeros((2, 3)), np.zeros((2, 3))
    slip, ground_other = np.zeros(2), np.zeros(3)
    foot_moment = np.zeros((2, 3))
    for contact in contacts:
        ga, gb = map(int, field(contact, "geometry_ids"))
        if protocol.floor not in (ga, gb):
            continue
        other, sign = (gb, 1.0) if ga == protocol.floor else (ga, -1.0)
        frame, wrench = field(contact, "frame"), field(contact, "wrench_contact_frame")
        force = sign * (np.asarray(frame).T @ np.asarray(wrench)[:3])
        torque = sign * (np.asarray(frame).T @ np.asarray(wrench)[3:])
        if other not in protocol.foot_geoms:
            ground_other += force
            continue
        side = protocol.foot_geoms[other]
        foot_force[side] += force
        foot_torque[side] += torque
        foot_moment[side] += torque + np.cross(
            np.asarray(field(contact, "position")) - body_com[protocol.foot_bodies[side]], force
        )
        velocity = np.asarray(field(contact, "jacobian")) @ v
        slip[side] = max(slip[side], float(np.linalg.norm(velocity[1:3])))
    return {
        "foot_force_world": foot_force,
        "foot_couple_world": foot_torque,
        "foot_moment_about_body_com_world": foot_moment,
        "foot_tangent_speed": slip,
        "loaded": foot_force[:, 2] > 1.0,
        "foot_vertical": float(foot_force[:, 2].sum()),
        "nonfoot_ground_force_world": ground_other,
    }


def compact_state(state):
    return {"q": state.q, "v": state.v, "time": state.time, "tick": state.tick}


class Stats:
    def __init__(self, protocol, h, stand=False):
        self.p, self.h, self.stand = protocol, h, stand
        self.steps = 0
        self.minimum_height = math.inf
        self.maximum_tilt = self.maximum_rmse = self.maximum_xy = self.peak_load = 0.0
        self.sum_load = self.settled_sum = 0.0
        self.settled_steps = self.both_loaded = 0
        self.slip = np.zeros(2)
        self.previous_loaded = (False, False)
        self.previous_energy = None
        self.work = {"actuator": 0.0, "external": 0.0, "passive": 0.0, "constraint": 0.0}
        self.max_energy_balance = 0.0
        self.max_nonfoot = 0.0
        self.foot_origin = None
        self.maximum_foot_xy = 0.0

    def update(self, q, v, output_time, input_time, loads, energy, foot_centers):
        metric = self.p.metrics(q, v, output_time, self.stand)
        if self.foot_origin is None:
            self.foot_origin = foot_centers[:, :2].copy()
        self.maximum_foot_xy = max(
            self.maximum_foot_xy,
            float(np.max(np.linalg.norm(foot_centers[:, :2] - self.foot_origin, axis=1))),
        )
        self.steps += 1
        self.minimum_height = min(self.minimum_height, metric["height"])
        self.maximum_tilt = max(self.maximum_tilt, metric["tilt"])
        self.maximum_rmse = max(self.maximum_rmse, metric["joint_rmse"])
        self.maximum_xy = max(self.maximum_xy, metric["reference_xy_error"])
        load = loads["foot_vertical"] / self.p.weight
        self.peak_load = max(self.peak_load, load)
        self.sum_load += load
        if input_time >= SETTLING:
            self.settled_sum += load
            self.settled_steps += 1
            self.both_loaded += int(np.all(loads["loaded"]))
        self.max_nonfoot = max(
            self.max_nonfoot, float(np.linalg.norm(loads["nonfoot_ground_force_world"]))
        )
        self.slip += self.h * loads["foot_tangent_speed"] * loads["loaded"]
        total = energy["kinetic"] + energy["gravity_potential"]
        balance = None
        if self.previous_energy is not None:
            old_total, old_power = self.previous_energy
            balance = total - old_total - self.h * old_power
            self.max_energy_balance = max(self.max_energy_balance, abs(balance))
        power = sum(energy[key + "_power"] for key in self.work)
        self.previous_energy = (total, power)
        for key in self.work:
            self.work[key] += self.h * energy[key + "_power"]
        loaded = tuple(map(bool, loads["loaded"]))
        events = []
        for side in range(2):
            if loaded[side] != self.previous_loaded[side]:
                events.append(
                    {
                        "foot": ("left", "right")[side],
                        "kind": "loaded" if loaded[side] else "unloaded",
                    }
                )
        self.previous_loaded = loaded
        failures = []
        if not metric["finite"]:
            failures.append("nonfinite_state")
        if metric["height"] < 0.35:
            failures.append("height")
        if metric["tilt"] > 1.0:
            failures.append("tilt")
        if load > 8.0:
            failures.append("foot_peak")
        if metric["joint_rmse"] > 0.5:
            failures.append("joint_rmse")
        if metric["reference_xy_error"] > 1.0:
            failures.append("reference_xy")
        return metric, events, failures, balance

    def summary(self, *, assessed):
        mean = self.settled_sum / self.settled_steps if self.settled_steps else None
        failures = []
        if assessed and (mean is None or not 0.5 <= mean <= 1.5):
            failures.append("settled_foot_mean")
        if assessed and self.stand:
            if self.maximum_tilt > 0.35:
                failures.append("stand_tilt")
            if self.maximum_xy > 0.1:
                failures.append("stand_root_xy")
            if self.maximum_foot_xy > 0.08:
                failures.append("stand_foot_xy")
            if mean is None or not 0.7 <= mean <= 1.3:
                failures.append("stand_foot_mean")
            if not self.settled_steps or self.both_loaded / self.settled_steps < 0.9:
                failures.append("stand_both_feet")
        return {
            "steps": self.steps,
            "minimum_height": self.minimum_height,
            "maximum_tilt": self.maximum_tilt,
            "maximum_joint_rmse": self.maximum_rmse,
            "maximum_reference_xy_error": self.maximum_xy,
            "peak_foot_load_ratio": self.peak_load,
            "maximum_foot_center_xy_drift": self.maximum_foot_xy,
            "foot_center_phase": "output_fk" if self.stand else "input_fk",
            "all_window_mean_foot_load_ratio": self.sum_load / max(1, self.steps),
            "settled_mean_foot_load_ratio": mean,
            "settled_steps": self.settled_steps,
            "settled_both_loaded": self.both_loaded,
            "accumulated_loaded_foot_slip_m": self.slip.copy(),
            "max_nonfoot_ground_force_n": self.max_nonfoot,
            "work_joules": dict(self.work),
            "max_input_energy_balance_j": self.max_energy_balance,
            "assessed": assessed,
            "checkpoint_failures": failures,
        }


def compare_array(checks, name, actual, expected, rtol, atol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    same_shape = actual.shape == expected.shape
    passed = (
        same_shape
        and bool(np.all(np.isfinite(actual)))
        and bool(np.all(np.isfinite(expected)))
        and bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
    )
    checks.append(
        {
            "field": name,
            "passed": passed,
            "rtol": rtol,
            "atol": atol,
            "actual_shape": actual.shape,
            "native_shape": expected.shape,
            "max_abs_error": float(np.max(np.abs(actual - expected)))
            if same_shape and actual.size
            else 0.0,
        }
    )


def match_contacts(actual, native):
    """Complete geometric bijection; force never participates in matching."""
    if len(actual) != len(native):
        raise ValueError(f"contact count {len(actual)} != native {len(native)}")
    choices = []
    for contact in actual:
        options = []
        for index, other in enumerate(native):
            if tuple(contact.geometry_ids) != tuple(other["geometry_ids"]):
                continue
            if contact.dimension != other["dimension"] or contact.active != other["active"]:
                continue
            if (
                np.allclose(contact.position, other["position"], rtol=3e-5, atol=4e-5)
                and np.isclose(contact.distance, other["distance"], rtol=3e-5, atol=4e-5)
                and np.allclose(contact.frame[0], other["frame"][0], rtol=0, atol=1e-4)
            ):
                options.append(index)
        choices.append(options)
    owner = {}

    def assign(index, visited):
        for target in choices[index]:
            if target in visited:
                continue
            visited.add(target)
            if target not in owner or assign(owner[target], visited):
                owner[target] = index
                return True
        return False

    for index in range(len(actual)):
        if not assign(index, set()):
            raise ValueError("no complete contact geometry bijection")
    return {index: target for target, index in owner.items()}


def compare_receipt(model, receipt, native):
    checks = []
    for name, tolerance in TOL.items():
        compare_array(checks, name, getattr(receipt.dynamics, name), native[name], *tolerance)
    for name in ("qacc", "integration_acceleration"):
        compare_array(checks, name, getattr(receipt, name), native[name], 3e-3, 1e-2)
    for name in ("qfrc_constraint", "qfrc_actuator"):
        compare_array(checks, name, getattr(receipt, name), native[name], 3e-3, 3e-3)
    mapping = match_contacts(receipt.contacts, native["contacts"])
    for index, contact in enumerate(receipt.contacts):
        other = native["contacts"][mapping[index]]
        for name, rtol, atol in (
            ("position", 3e-5, 4e-5),
            ("distance", 3e-5, 4e-5),
            ("frame", 0, 1e-4),
            ("jacobian", 2e-4, 5e-5),
            ("wrench_contact_frame", 3e-3, 3e-3),
        ):
            compare_array(
                checks, f"contact[{index}].{name}", getattr(contact, name), other[name], rtol, atol
            )
    native_keys = {}
    for index in range(native["nefc"]):
        kind, source = int(native["efc_type"][index]), int(native["efc_id"][index])
        if kind == 1:
            key = (kind, source, -1)
        elif kind == 3:
            dof = int(model.jnt_dofadr[source])
            key = (kind, source, -1 if native["efc_J"][index, dof] > 0 else 1)
        elif kind in (5, 6):
            key = (kind, source, index - native["contacts"][source]["efc_address"])
        else:
            raise ValueError(f"unexpected native G1 constraint type {kind}")
        if key in native_keys:
            raise ValueError("duplicate native constraint identity")
        native_keys[key] = index
    visited = set()
    columns = (
        ("position", "efc_pos"),
        ("margin", "efc_margin"),
        ("velocity", "efc_vel"),
        ("reference_acceleration", "efc_aref"),
        ("regularization", "efc_R"),
        ("inverse_regularization", "efc_D"),
        ("impedance", "efc_imp"),
        ("diag_approx", "efc_diagApprox"),
    )
    for index, row in enumerate(receipt.constraints):
        source = mapping[row.source_id] if row.constraint_type in (5, 6) else row.source_id
        key = (row.constraint_type, source, row.side_or_edge)
        if key not in native_keys or key in visited:
            raise ValueError("constraint owner/side/edge does not match native")
        visited.add(key)
        ni = native_keys[key]
        compare_array(checks, f"efc[{index}].J", row.jacobian, native["efc_J"][ni], 2e-4, 5e-5)
        for public, cpu in columns:
            compare_array(
                checks, f"efc[{index}].{public}", getattr(row, public), native[cpu][ni], 5e-4, 4e-4
            )
        compare_array(checks, f"efc[{index}].force", row.force, native["efc_force"][ni], 3e-3, 3e-3)
    if visited != set(native_keys):
        raise ValueError("native constraint rows were omitted")
    return {
        "checks": checks,
        "contact_mapping": mapping,
        "passed": all(check["passed"] for check in checks),
    }


def independent_kkt(receipt):
    rows = receipt.constraints
    if not rows:
        return {"normalized_kkt": 0.0, "bound_violation": 0.0, "passed": True}
    jac = np.asarray([row.jacobian for row in rows])
    regularization = np.asarray([row.regularization for row in rows])
    force = np.asarray([row.force for row in rows])
    lower = np.asarray([row.lower for row in rows])
    upper = np.asarray([np.inf if row.upper is None else row.upper for row in rows])
    mass = receipt.dynamics.mass_matrix
    hessian = jac @ np.linalg.solve(mass, jac.T) + np.diag(regularization)
    linear = jac @ receipt.dynamics.smooth_acceleration - np.asarray(
        [row.reference_acceleration for row in rows]
    )
    gradient = hessian @ force + linear
    violation = np.abs(gradient)
    violation[force <= lower] = np.maximum(0, -gradient[force <= lower])
    violation[force >= upper] = np.maximum(0, gradient[force >= upper])
    scale = max(1.0, float(np.max(np.abs(linear) + np.abs(hessian) @ np.abs(force))))
    residual = float(np.max(violation)) / scale
    bounds = float(max(0, np.max(lower - force), np.max(force - upper)))
    reconstruction = jac.T @ force
    force_ok = bool(np.allclose(reconstruction, receipt.qfrc_constraint, rtol=3e-3, atol=3e-3))
    return {
        "normalized_kkt": residual,
        "bound_violation": bounds,
        "generalized_force_max_abs_error": float(
            np.max(np.abs(reconstruction - receipt.qfrc_constraint))
        ),
        "passed": residual <= 5e-5 and bounds == 0 and force_ok,
    }


def receipt_energy(model, receipt):
    dyn = receipt.dynamics
    result = energy_fields(
        model,
        receipt.input_v,
        dyn.mass_matrix,
        dyn.body_com,
        receipt.qfrc_actuator,
        dyn.applied,
        dyn.passive,
        receipt.qfrc_constraint,
    )
    result["constraint_power_by_type"] = {
        str(kind): float(
            sum(
                row.force * row.velocity
                for row in receipt.constraints
                if row.constraint_type == kind
            )
        )
        for kind in (1, 3, 5, 6)
    }
    return result


def foot_centers(protocol, positions):
    return np.asarray(
        [
            np.mean(
                [positions[geom] for geom, side in protocol.foot_geoms.items() if side == index],
                axis=0,
            )
            for index in range(2)
        ]
    )


def input_checks(receipt, state, packet):
    for name, actual, expected in (
        ("q", receipt.input_q, state.q),
        ("v", receipt.input_v, state.v),
        ("ctrl", receipt.stored_control.ctrl, packet.ctrl),
        ("qfrc", receipt.stored_control.qfrc_applied, packet.qfrc_applied),
        ("xfrc", receipt.stored_control.xfrc_applied, packet.xfrc_applied),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(f"same-input oracle binding failed for {name}")
    if receipt.input_time != state.time:
        raise ValueError("same-input oracle time binding failed")


def actuator_check(protocol, receipt):
    actual = receipt.qfrc_actuator[protocol.va]
    requested = receipt.stored_control.ctrl
    valid = np.allclose(actual, requested, rtol=3e-5, atol=3e-5)
    low, high = protocol.model.actuator_ctrlrange.T
    allowance = 3e-5 * np.maximum(np.abs(low), np.abs(high)) + 3e-5
    valid = valid and np.all(actual >= low - allowance) and np.all(actual <= high + allowance)
    return {
        "passed": bool(valid),
        "actual_joint_torque": actual,
        "stored_control": requested,
        "minimum": low,
        "maximum": high,
    }


def inventory(protocol, compilation, derived, capabilities):
    model = protocol.model
    if (capabilities.get("possible_contact_pairs"), capabilities.get("candidate_slots")) != (
        502,
        942,
    ):
        raise ValueError("G1 candidate layout differs from its complete source inventory")
    source = compilation.source
    return {
        "compilation_source": {
            "origin": source.origin,
            "primary_sha256": source.primary_sha256,
            "archive_sha256": source.archive_sha256,
            "model_sha256": source.model_sha256,
            "mujoco_version": source.mujoco_version,
            "primary_xml_bytes": len(source.primary_xml),
            "archive_bytes": len(source.archive_bytes),
            "model_binary_bytes": len(source.model_binary),
        },
        "derived_timestep": derived,
        "mujoco_version": mj.__version__,
        "capabilities": capabilities,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "njnt": model.njnt,
        "source_options": {
            "h": model.opt.timestep,
            "iterations": model.opt.iterations,
            "tolerance": model.opt.tolerance,
            "cone": int(model.opt.cone),
            "impratio": model.opt.impratio,
            "integrator": int(model.opt.integrator),
            "gravity": model.opt.gravity.copy(),
        },
        "bodies": {
            "names": [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)],
            "parent": model.body_parentid.copy(),
            "mass": model.body_mass.copy(),
            "inertia": model.body_inertia.copy(),
            "inertial_position": model.body_ipos.copy(),
            "inertial_quaternion": model.body_iquat.copy(),
        },
        "joints": {
            "names": protocol.names,
            "type": model.jnt_type.copy(),
            "qpos_address": model.jnt_qposadr.copy(),
            "dof_address": model.jnt_dofadr.copy(),
            "limited": model.jnt_limited.copy(),
            "range": model.jnt_range.copy(),
            "margin": model.jnt_margin.copy(),
            "armature": model.dof_armature.copy(),
            "damping": model.dof_damping.copy(),
            "frictionloss": model.dof_frictionloss.copy(),
        },
        "actuators": {
            "joint": model.actuator_trnid.copy(),
            "gear": model.actuator_gear.copy(),
            "ctrl_limited": model.actuator_ctrllimited.copy(),
            "ctrl_range": model.actuator_ctrlrange.copy(),
            "force_limited": model.actuator_forcelimited.copy(),
            "force_range": model.actuator_forcerange.copy(),
            "gain": model.actuator_gainprm.copy(),
            "bias": model.actuator_biasprm.copy(),
        },
        "geometries": {
            "names": [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)],
            "body": model.geom_bodyid.copy(),
            "type": model.geom_type.copy(),
            "size": model.geom_size.copy(),
            "condim": model.geom_condim.copy(),
            "contype": model.geom_contype.copy(),
            "conaffinity": model.geom_conaffinity.copy(),
            "friction": model.geom_friction.copy(),
            "priority": model.geom_priority.copy(),
            "local_position": model.geom_pos.copy(),
            "local_quaternion": model.geom_quat.copy(),
            "solref": model.geom_solref.copy(),
            "solimp": model.geom_solimp.copy(),
            "margin": model.geom_margin.copy(),
            "gap": model.geom_gap.copy(),
        },
        "exclude_signature": model.exclude_signature.copy(),
        "foot_geometry_to_side": protocol.foot_geoms,
        "foot_geometry_count": len(protocol.foot_geoms),
        "policy_provider": protocol.session.get_providers(),
        "gates": GATES,
        "observation": "snapshot is a detached CPU FK observer; device loads come only from last_step",
        "native_oracles": "same-input forward and independent feedback CPU trajectory are distinct",
        "dance_scope": "at most 20 seconds, not the complete 131.48-second motion",
    }


def endpoint_foot_centers(protocol, q):
    """Detached native kinematics only; does not step or supply contact loads."""
    observed = mj.MjData(protocol.model)
    observed.qpos[:] = q
    mj.mj_kinematics(protocol.model, observed)
    return foot_centers(protocol, observed.geom_xpos)


def runtime_for(compilation, q, v, time, mode, cache_root):
    return MuJoCoRuntime(
        compilation,
        mode=mode,
        world_keys=(0,),
        initial_states={0: MuJoCoInitialState(q=q, v=v, time=time)},
        contact_capacity=128,
        constraint_capacity=512,
        cache_root=cache_root,
    )


def step_with_failure_capture(runtime, h, journal, **context):
    """Failure-only diagnostic read, explicitly authorized by the parent."""
    try:
        return runtime.step(h)
    except Exception:
        try:
            # No advance, retry, publication, reset, or CPU solve occurs here.
            # Read before context-manager close while the rejected bank exists.
            scratch = runtime._device.download()
        except Exception as download_error:
            scratch = {"download_failure": repr(download_error)}
        journal.write(
            "rejected_device_candidate",
            durable=True,
            full_failed_device_scratch=scratch,
            additional_step_calls=0,
            diagnostic_download_only=True,
            counters=runtime.counters,
            context=context,
            traceback=traceback.format_exc(),
        )
        raise


def snapshots(args, journal, protocol, compilation, derived):
    outcomes = []
    for label, frame in (("source-default", None), ("motion-0", 0), ("motion-200", 200)):
        q, v = protocol.initial(frame)
        time = 0.0 if frame is None else frame / 50
        with runtime_for(compilation, q, v, time, args.mode, args.cache_root) as runtime:
            state = runtime.snapshot()[0]
            journal.write(
                "snapshot_initial",
                label=label,
                state=compact_state(state),
                inventory=inventory(protocol, compilation, derived, runtime.capabilities),
            )
            controller = Controller(protocol, state.q, args.h, stand=frame is None)
            proposal = controller.propose(state.q, state.v, state.time, 0)
            packet = packet_for(protocol, proposal, 0.0, 0)
            stored = stored_packet(packet)
            native, native_data = native_oracle(
                protocol.model, state.q, state.v, stored, state.time, full=True
            )
            journal.write(
                "snapshot_attempt",
                durable=True,
                label=label,
                input=compact_state(state),
                control=stored,
                policy=proposal,
                same_input_native=native,
            )
            runtime.set_controls({0: packet})
            step_with_failure_capture(
                runtime,
                args.h,
                journal,
                phase="snapshots",
                label=label,
                input=compact_state(state),
                control=stored,
            )
            after = runtime.snapshot()[0]
            receipt = after.last_step
            input_checks(receipt, state, stored)
            journal.write(
                "snapshot_device_receipt",
                durable=True,
                label=label,
                state=compact_state(after),
                full_device_receipt=receipt,
            )
            comparison = compare_receipt(protocol.model, receipt, native)
            kkt = independent_kkt(receipt)
            actuator = actuator_check(protocol, receipt)
            # This is an independent native one-step oracle. It never supplies TYX state.
            mj.mj_step(protocol.model, native_data)
            endpoint = []
            compare_array(endpoint, "q", after.q, native_data.qpos, 0, 2e-4)
            compare_array(endpoint, "v", after.v, native_data.qvel, 0, 3e-3)
            compare_array(endpoint, "time", after.time, native_data.time, 0, 2e-6)
            passed = (
                comparison["passed"]
                and kkt["passed"]
                and actuator["passed"]
                and all(check["passed"] for check in endpoint)
            )
            journal.write(
                "snapshot_result",
                durable=True,
                label=label,
                passed=passed,
                state=compact_state(after),
                full_device_receipt=receipt,
                same_input_native=native,
                comparison=comparison,
                kkt=kkt,
                actuator=actuator,
                endpoint_checks=endpoint,
            )
            outcomes.append({"label": label, "passed": passed})
            if not passed:
                break
    return {
        "status": "PASS" if len(outcomes) == 3 and all(x["passed"] for x in outcomes) else "FAIL",
        "phase": "snapshots",
        "outcomes": outcomes,
    }


def trajectory(args, journal, protocol, compilation, derived):
    model, h = protocol.model, args.h
    stand = args.phase == "stand"
    q, v = protocol.initial(None if stand else 0)
    count = int(round(args.duration / h))
    device_stats, cpu_stats = Stats(protocol, h, stand), Stats(protocol, h, stand)
    checkpoints, failures = [], []
    same_input_forwards = cpu_steps = 0
    previous_constraint_identity = None
    with runtime_for(compilation, q, v, 0.0, args.mode, args.cache_root) as runtime:
        state = runtime.snapshot()[0]
        device_controller = Controller(protocol, state.q, h, stand)
        cpu_controller = Controller(protocol, state.q, h, stand)
        # One common published initial state. No later TYX endpoint is copied here.
        cpu = mj.MjData(model)
        cpu.qpos[:], cpu.qvel[:], cpu.time = state.q, state.v, state.time
        initial_feet = foot_centers(protocol, state.geometry_positions)
        device_stats.foot_origin = initial_feet[:, :2].copy()
        cpu_stats.foot_origin = initial_feet[:, :2].copy()
        journal.write(
            "trajectory_initial",
            durable=True,
            state=compact_state(state),
            independent_cpu_initial={"q": cpu.qpos.copy(), "v": cpu.qvel.copy(), "time": cpu.time},
            inventory=inventory(protocol, compilation, derived, runtime.capabilities),
            policy_history="zeros at initial state; commits only after successful step",
        )
        for tick in range(count):
            grid_time = tick * h
            output_grid_time = (tick + 1) * h
            at_two = tick + 1 == int(round(2.0 / h))
            selected = tick == 0 or at_two or tick + 1 == count
            proposal = packet = stored = native = native_data = receipt = None
            operation = "device_or_harness"
            try:
                proposal = device_controller.propose(state.q, state.v, state.time, tick)
                packet = packet_for(protocol, proposal, grid_time, args.push_sign)
                stored = stored_packet(packet)
                native, native_data = native_oracle(
                    model, state.q, state.v, stored, state.time, full=selected
                )
                same_input_forwards += 1
                journal.write(
                    "attempt",
                    durable=selected,
                    input=compact_state(state),
                    source_grid_time=grid_time,
                    requested_control=packet,
                    stored_oracle_control=stored,
                    policy=proposal,
                )
                runtime.set_controls({0: packet})
                step_with_failure_capture(
                    runtime,
                    h,
                    journal,
                    phase=args.phase,
                    input=compact_state(state),
                    control=stored,
                )
                after = runtime.snapshot()[0]
                receipt = after.last_step
                if after.tick != tick + 1 or abs(after.time - output_grid_time) > 2e-6:
                    raise RuntimeError("device output tick/source time binding failed")
                input_checks(receipt, state, stored)
                device_controller.commit(proposal)
                actuator = actuator_check(protocol, receipt)
                loads = support(
                    protocol, receipt.contacts, receipt.input_v, receipt.dynamics.body_com
                )
                energy = receipt_energy(model, receipt)
                metrics, events, device_failures, balance = device_stats.update(
                    after.q,
                    after.v,
                    after.time,
                    receipt.input_time,
                    loads,
                    energy,
                    foot_centers(protocol, after.geometry_positions)
                    if stand
                    else foot_centers(protocol, state.geometry_positions),
                )
                if not actuator["passed"]:
                    device_failures.append("actuator_limit")
                # Original native feedback control uses its own state/history,
                # independently of the same-input oracle and device trajectory.
                operation = "native_reference"
                cpu_q, cpu_v, cpu_time = cpu.qpos.copy(), cpu.qvel.copy(), float(cpu.time)
                cpu_proposal = cpu_controller.propose(cpu_q, cpu_v, cpu_time, tick)
                cpu_packet = packet_for(protocol, cpu_proposal, grid_time, args.push_sign)
                set_native(cpu, cpu_packet)
                warnings_before = [warning.number for warning in cpu.warning]
                mj.mj_step(model, cpu)
                cpu_steps += 1
                if abs(float(cpu.time) - (cpu_time + h)) > 2e-12:
                    raise RuntimeError(
                        "native reference time changed unexpectedly; possible auto-reset"
                    )
                for warning in (
                    mj.mjtWarning.mjWARN_BADQPOS,
                    mj.mjtWarning.mjWARN_BADQVEL,
                    mj.mjtWarning.mjWARN_BADQACC,
                ):
                    if cpu.warning[int(warning)].number != warnings_before[int(warning)]:
                        raise RuntimeError("native reference reported invalid state/acceleration")
                # Euler's cached M/FK/contact fields are the just-completed input
                # evaluation. qpos/qvel now hold its independent output state.
                cpu_fields = native_fields(model, cpu, cpu_q, cpu_v, cpu_packet, full=selected)
                cpu_controller.commit(cpu_proposal)
                cpu_loads = support(protocol, cpu_fields["contacts"], cpu_v, cpu_fields["body_com"])
                cpu_metrics, cpu_events, cpu_failures, cpu_balance = cpu_stats.update(
                    cpu.qpos,
                    cpu.qvel,
                    cpu.time,
                    cpu_time,
                    cpu_loads,
                    cpu_fields["energy"],
                    endpoint_foot_centers(protocol, cpu.qpos)
                    if stand
                    else foot_centers(protocol, cpu_fields["geom_position"]),
                )
                operation = "device_or_harness"
                aggregate_failure = [
                    {"trajectory": "device", "gate": item} for item in device_failures
                ] + [{"trajectory": "native_reference", "gate": item} for item in cpu_failures]
                if at_two or tick + 1 == count:
                    assessed = output_grid_time >= 2.0
                    device_summary = device_stats.summary(assessed=assessed)
                    cpu_summary = cpu_stats.summary(assessed=assessed)
                    aggregate_failure.extend(
                        {"trajectory": lane, "gate": gate}
                        for lane, summary in (
                            ("device", device_summary),
                            ("native_reference", cpu_summary),
                        )
                        for gate in summary["checkpoint_failures"]
                    )
                    checkpoint = {
                        "source_time": output_grid_time,
                        "device": device_summary,
                        "independent_cpu": cpu_summary,
                    }
                    checkpoints.append(checkpoint)
                    journal.write(
                        "checkpoint",
                        durable=True,
                        **checkpoint,
                        same_runtime_continues=at_two and count > tick + 1,
                    )
                constraint_identity = tuple(
                    (
                        row.constraint_type,
                        row.source_id,
                        row.side_or_edge,
                        row.geometry_slot,
                        row.dof,
                    )
                    for row in receipt.constraints
                )
                identity_changed = constraint_identity != previous_constraint_identity
                event_selected = bool(events or cpu_events or identity_changed)
                full_selected = selected or event_selected or bool(aggregate_failure)
                if full_selected and not selected:
                    # Expand fields from the SAME cached native evaluation;
                    # no second mj_forward or device step is performed.
                    native = native_fields(model, native_data, state.q, state.v, stored, full=True)
                    cpu_fields = native_fields(model, cpu, cpu_q, cpu_v, cpu_packet, full=True)
                journal.write(
                    "step",
                    durable=full_selected,
                    full_capture_reasons={
                        "fixed_checkpoint": selected,
                        "foot_event": bool(events or cpu_events),
                        "constraint_identity_changed": identity_changed,
                        "first_failure": bool(aggregate_failure),
                    },
                    constraint_identity=constraint_identity,
                    source_grid_time=grid_time,
                    source_output_time=output_grid_time,
                    device={
                        "state": compact_state(after),
                        "input_time": receipt.input_time,
                        "control": receipt.stored_control,
                        "policy": proposal,
                        "contacts": contact_rows(receipt.contacts),
                        "support": loads,
                        "metrics": metrics,
                        "events": events,
                        "energy": energy,
                        "input_energy_balance": balance,
                        "actuator": actuator,
                        "constraint_type_force": [
                            {
                                "type": row.constraint_type,
                                "source": row.source_id,
                                "side_or_edge": row.side_or_edge,
                                "force": row.force,
                                "velocity": row.velocity,
                                "position": row.position,
                            }
                            for row in receipt.constraints
                        ],
                        "qfrc_constraint": receipt.qfrc_constraint,
                        "full_receipt": receipt if full_selected else None,
                    },
                    same_input_native={
                        "contacts": contact_rows(native["contacts"]),
                        "qacc": native["qacc"],
                        "qfrc_constraint": native["qfrc_constraint"],
                        "energy": native["energy"],
                        "full_fields": native if full_selected else None,
                    },
                    independent_cpu={
                        "input_q": cpu_q,
                        "input_v": cpu_v,
                        "input_time": cpu_time,
                        "q": cpu.qpos.copy(),
                        "v": cpu.qvel.copy(),
                        "time": float(cpu.time),
                        "control": cpu_packet,
                        "policy": cpu_proposal,
                        "contacts": contact_rows(cpu_fields["contacts"]),
                        "support": cpu_loads,
                        "metrics": cpu_metrics,
                        "events": cpu_events,
                        "energy": cpu_fields["energy"],
                        "input_energy_balance": cpu_balance,
                        "qfrc_constraint": cpu_fields["qfrc_constraint"],
                        "full_fields": cpu_fields if full_selected else None,
                    },
                    failures=aggregate_failure,
                )
                state = after
                previous_constraint_identity = constraint_identity
                if aggregate_failure:
                    failures = aggregate_failure
                    break
            except Exception as error:
                failures = [
                    {"trajectory": operation, "gate": type(error).__name__, "message": str(error)}
                ]
                full_native = None
                if native_data is not None:
                    try:
                        full_native = native_fields(
                            model, native_data, state.q, state.v, stored, full=True
                        )
                    except Exception as diagnostic_error:
                        full_native = {"diagnostic_error": repr(diagnostic_error)}
                journal.write(
                    "first_failure",
                    durable=True,
                    source_grid_time=grid_time,
                    input=compact_state(state),
                    requested_control=packet,
                    policy=proposal,
                    full_device_receipt=receipt,
                    same_input_native=full_native,
                    independent_cpu_last={
                        "q": cpu.qpos.copy(),
                        "v": cpu.qvel.copy(),
                        "time": float(cpu.time),
                    },
                    failures=failures,
                    traceback=traceback.format_exc(),
                )
                break
        assessed = count * h >= 2.0
        completed = device_stats.steps == count and not failures
        return {
            "phase": args.phase,
            "status": "FAIL"
            if failures
            else "PASS"
            if completed and assessed
            else "DIAGNOSTIC_ONLY",
            "complete_requested_window": completed,
            "requested_duration": args.duration,
            "device": device_stats.summary(assessed=assessed and completed),
            "independent_cpu": cpu_stats.summary(assessed=assessed and completed),
            "checkpoints": checkpoints,
            "failures": failures,
            "same_input_native_forward_calls": same_input_forwards,
            "independent_native_steps": cpu_steps,
            "runtime_counters": runtime.counters,
            "continuous_runtime": True,
            "resets_after_initialization": 0,
            "stand_is_diagnostic": stand,
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument(
        "--phase", choices=("admission", "snapshots", "trajectory", "stand"), default="trajectory"
    )
    parser.add_argument("--h", type=float, choices=(0.001, 0.002), default=0.002)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--push-sign", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    args.assets = args.assets.resolve()
    if not 0 < args.duration <= 20 or not math.isfinite(args.duration):
        parser.error("duration must be positive and at most 20 seconds")
    if not np.isclose(args.duration / args.h, round(args.duration / args.h), rtol=0, atol=1e-9):
        parser.error("duration must contain an integer number of physics steps")
    if args.duration > 2 and (args.h != 0.002 or args.push_sign or args.phase != "trajectory"):
        parser.error("the predeclared 20-second extension uses h=.002, no push, original policy")
    if args.push_sign and (args.h != 0.002 or args.phase != "trajectory" or args.duration != 2):
        parser.error("the predeclared push cases use h=.002 and a two-second policy window")
    return args


def main():
    args = parse_args()
    journal = Journal(args.output)
    initial_bindings = None
    summary = None
    try:
        if mj.__version__ != "3.3.7":
            raise ValueError("this harness binds the predeclared MuJoCo 3.3.7 oracle")
        initial_bindings = {name: binding(args.assets / name) for name in FILES}
        journal.write(
            "source_binding",
            durable=True,
            files=initial_bindings,
            harness=binding(Path(__file__)),
            options=vars(args),
            execution_role="parent-owned serial queue; no CPU fallback",
            status="RUNNING",
        )
        compilation, model, derived = compile_source(args.assets, args.h)
        protocol = Protocol(model, args.assets)
        if args.phase == "admission":
            q, v = protocol.initial(0)
            with runtime_for(compilation, q, v, 0.0, args.mode, args.cache_root) as runtime:
                journal.write(
                    "admission",
                    durable=True,
                    inventory=inventory(protocol, compilation, derived, runtime.capabilities),
                    state=compact_state(runtime.snapshot()[0]),
                )
                summary = {
                    "status": "ADMITTED_NOT_PHYSICS",
                    "phase": "admission",
                    "runtime_counters": runtime.counters,
                }
        elif args.phase == "snapshots":
            summary = snapshots(args, journal, protocol, compilation, derived)
        else:
            summary = trajectory(args, journal, protocol, compilation, derived)
    except Exception as error:
        journal.write(
            "first_failure",
            durable=True,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
        )
        summary = {
            "status": "FAIL",
            "phase": args.phase,
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
    finally:
        try:
            final_bindings = {name: binding(args.assets / name) for name in FILES}
            unchanged = initial_bindings == final_bindings
            summary = {} if summary is None else summary
            summary["source_bindings_unchanged"] = unchanged
            summary["final_bindings"] = final_bindings
            if not unchanged:
                summary["status"] = "FAIL"
                summary["source_binding_failure"] = (
                    "start/end file bindings differ or start incomplete"
                )
        except Exception as error:
            summary = {} if summary is None else summary
            summary["status"] = "FAIL"
            summary["source_binding_failure"] = repr(error)
        try:
            journal.finish(summary)
        finally:
            journal.close()
    print(json.dumps(json_value(summary), allow_nan=False, indent=2))
    return 1 if summary["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
