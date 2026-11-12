"""Installed FREE+BALL pyramidal consumer; no checkout imports.

Two asymmetric models; two sparse worlds; 24-step windows and real lifecycle.
Copy outside the source tree and run against the freshly installed wheel.
"""

import argparse
import base64
import copy
import gzip
import hashlib
import json
import math
import struct
import sys
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def encode(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "object array in evidence")
        return dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            bytes_b64=base64.b64encode(value.tobytes()).decode(),
        )
    if isinstance(value, np.generic):
        return dict(dtype=value.dtype.str, scalar_b64=base64.b64encode(value.tobytes()).decode())
    if is_dataclass(value):
        return dict(
            type=type(value).__module__ + "." + type(value).__name__,
            fields={f.name: encode(getattr(value, f.name)) for f in fields(value)},
        )
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, tuple):
        return dict(tuple=[encode(v) for v in value])
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, bytes):
        return dict(bytes_b64=base64.b64encode(value).decode())
    if type(value) is float:
        return dict(float64_bits=struct.pack(">d", value).hex())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError("unhandled evidence value " + type(value).__name__)


def exact(left, right, label):
    # Checkpoint JSON may reorder mappings. Keep all typed keys, values and
    # sequence order; journal encoding still retains the original saved order.
    def canonical(value):
        if isinstance(value, dict):
            if set(value) == {"mapping_entries"}:
                entries = [[canonical(k), canonical(v)] for k, v in value["mapping_entries"]]
                entries.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
                return {"mapping_entries": entries}
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value

    require(canonical(encode(left)) == canonical(encode(right)), label)


class Journal:
    def __init__(self, path):
        self.path = path
        self.stream = gzip.open(path, "xb")
        self.sha = hashlib.sha256()
        self.size = self.events = 0

    def emit(self, event, **payload):
        line = (
            json.dumps(encode(dict(event=event, **payload)), allow_nan=False, separators=(",", ":"))
            + "\n"
        ).encode()
        self.stream.write(line)
        self.stream.flush()
        self.sha.update(line)
        self.size += len(line)
        self.events += 1

    def close(self):
        self.stream.close()
        return dict(
            path=str(self.path),
            events=self.events,
            logical_bytes=self.size,
            logical_sha256=self.sha.hexdigest(),
            compressed_bytes=self.path.stat().st_size,
            compressed_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
        )


def modules():
    return {
        name: str(Path(module.__file__).resolve())
        for name, module in sorted(sys.modules.items())
        if (name == "phymeo" or name.startswith("phymeo.")) and getattr(module, "__file__", None)
    }


def set_control(data, packet):
    for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
        getattr(data, name)[:] = np.asarray(getattr(packet, name), np.float32)


def provenance(installed_root, *, preflight=False):
    """Fail on checkout discovery without rewriting the caller's environment."""

    def is_source_entry(path):
        return ((path / "pyproject.toml").is_file() and (path / "src" / "phymeo").is_dir()) or (
            path.name.lower() == "src"
            and (path / "phymeo").is_dir()
            and (path.parent / "pyproject.toml").is_file()
        )

    require(
        (installed_root / "phymeo" / "__init__.py").is_file(), "actual installed package required"
    )
    resolved_paths = [Path(entry or Path.cwd()).resolve() for entry in sys.path]
    require(
        not any(is_source_entry(path) for path in resolved_paths),
        "checkout source entry in sys.path",
    )
    if preflight:
        require(
            not any(name == "phymeo" or name.startswith("phymeo.") for name in sys.modules),
            "product imported before installed-origin preflight",
        )
        for origin in (Path.cwd().resolve(), Path(__file__).resolve().parent, installed_root):
            require(
                not any(is_source_entry(parent) for parent in (origin, *origin.parents)),
                "consumer and cwd must be outside checkout sources",
            )
    import phymeo

    paths = modules()
    require(
        "phymeo" in paths and Path(phymeo.__file__).resolve().is_relative_to(installed_root),
        "installed phymeo provenance",
    )
    loaded = {}
    for name, filename in paths.items():
        actual = Path(filename).resolve(strict=True)
        require(actual.is_relative_to(installed_root), "noninstalled module " + name)
        loaded[name] = dict(
            path=str(actual), sha256=hashlib.sha256(actual.read_bytes()).hexdigest()
        )
    return dict(
        installed_root=str(installed_root),
        cwd=str(Path.cwd().resolve()),
        consumer=str(Path(__file__).resolve()),
        sys_path=[str(p) for p in resolved_paths],
        loaded=loaded,
    )


def control(model, tick, slot):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    command = 0.8 if tick < 8 else (-0.75 if tick < 16 else 0.0)
    u = np.full(model.nu, command + (0.015 * slot if command else 0.0), np.float32)
    sign = 1 if tick < 8 else -1
    force = np.array([sign * 0.03 * (i + 1) for i in range(model.nv)], np.float32)
    wrench = np.zeros((model.nbody, 6), np.float32)
    wrench[-1] = np.array([0.3, -0.2, 0.4, 0.01, -0.03, 0.02]) * (1 + 0.2 * slot)
    if tick >= 16:
        force.fill(0)
        wrench.fill(0)
    return MuJoCoControl(u, qfrc_applied=force, xfrc_applied=wrench)


def analytic(model, entry, packet):
    """Independent native JOINT address/moment and stateless force formulas."""
    offsets, columns, coefficients, ids = [0], [], [], []
    clamped, raw, scalar = [], [], []
    generalized = np.zeros(model.nv)
    dense = np.zeros((model.nu, model.nv))
    require(model.na == 0, "R3-P stateless domain")
    for actor in range(model.nu):
        joint = int(model.actuator_trnid[actor, 0])
        kind = int(model.jnt_type[joint])
        qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        width = 3 if kind == 1 else 1
        ids.append((actor, joint, kind, qa, dof, width))
        u = float(np.float32(packet.ctrl[actor]))
        if model.actuator_ctrllimited[actor]:
            u = float(np.clip(u, *model.actuator_ctrlrange[actor]))
        value = 0.0
        if not model.opt.disableactuator & (1 << int(model.actuator_group[actor])):
            value = float(model.actuator_gainprm[actor, 0]) * u
            if model.actuator_biastype[actor]:
                require(width == 1, "no quaternion-as-length oracle")
                bias = model.actuator_biasprm[actor]
                gear = model.actuator_gear[actor, 0]
                value += bias[0] + bias[1] * gear * entry.q[qa] + bias[2] * gear * entry.v[dof]
        force = (
            float(np.clip(value, *model.actuator_forcerange[actor]))
            if model.actuator_forcelimited[actor]
            else value
        )
        clamped.append(u)
        raw.append(value)
        scalar.append(force)
        for axis in range(width):
            col, coefficient = dof + axis, float(model.actuator_gear[actor, axis])
            columns.append(col)
            coefficients.append(coefficient)
            dense[actor, col] = coefficient
            generalized[col] += coefficient * force
        offsets.append(len(columns))
    return dict(
        ids=np.asarray(ids, np.int32),
        gear=model.actuator_gear.copy(),
        row_offsets=np.asarray(offsets, np.int32),
        dof_indices=np.asarray(columns, np.int32),
        moment=np.asarray(coefficients, np.float32),
        dense_moment=dense,
        clamped_control=np.asarray(clamped),
        raw_force=np.asarray(raw),
        scalar_force=np.asarray(scalar),
        generalized_force=generalized,
    )


def dense_native_moment(model, data):
    dense = np.zeros((model.nu, model.nv))
    for actor in range(model.nu):
        start, count = int(data.moment_rowadr[actor]), int(data.moment_rownnz[actor])
        dense[actor, data.moment_colind[start : start + count]] = data.actuator_moment[
            start : start + count
        ]
    return dense


def check_transmission(row, entry, native):
    tr, expected = row.transmission, native["analytic"]
    require(
        tr.schema == "native-free-ball-joint-motor-transmission-v1", "complete transmission schema"
    )
    for name in ("ids", "gear", "row_offsets", "dof_indices", "moment"):
        np.testing.assert_array_equal(getattr(tr, name), expected[name])
    for name in ("clamped_control", "raw_force", "scalar_force", "generalized_force"):
        np.testing.assert_allclose(getattr(tr, name), expected[name], **FORCE)
    np.testing.assert_allclose(tr.scalar_force, native["actuator_force"], **FORCE)
    require(tr.generalized_force.dtype == np.float32, "stored motor force FP32")
    require(row.qfrc_actuator.dtype == np.float64, "legacy receipt force FP64")
    exact(tr.generalized_force.astype(np.float64), row.qfrc_actuator, "same-stage actuator force")
    exact(tr.input_control, row.control.ctrl, "original control binding")
    np.testing.assert_allclose(expected["dense_moment"], native["dense_moment"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        expected["dense_moment"] @ entry.v, native["native_input"]["actuator_velocity"], **FORCE
    )
    np.testing.assert_allclose(
        tr.generalized_force @ entry.v,
        tr.scalar_force @ (expected["dense_moment"] @ entry.v),
        **FORCE,
    )
    for name in (
        "ids",
        "gear",
        "row_offsets",
        "dof_indices",
        "moment",
        "input_control",
        "clamped_control",
        "raw_force",
        "scalar_force",
        "generalized_force",
    ):
        value = getattr(tr, name)
        require(not value.flags.writeable, "immutable transmission " + name)
        try:
            value.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError("transmission permits mutable reenable " + name)


def native_state(data):
    return dict(
        q=data.qpos.copy(),
        v=data.qvel.copy(),
        act=data.act.copy(),
        time=float(data.time),
        ctrl=data.ctrl.copy(),
        qacc=data.qacc.copy(),
        qfrc_applied=data.qfrc_applied.copy(),
        xfrc_applied=data.xfrc_applied.copy(),
        qfrc_actuator=data.qfrc_actuator.copy(),
        qfrc_constraint=data.qfrc_constraint.copy(),
        actuator_force=data.actuator_force.copy(),
        act_dot=data.act_dot.copy(),
        actuator_length=data.actuator_length.copy(),
        actuator_velocity=data.actuator_velocity.copy(),
    )


H = 0.001
KEYS = (7, 41)
ACT = dict(rtol=3e-5, atol=3e-6)
FORCE = dict(rtol=3e-4, atol=2e-4)
CONTACT_FORCE = dict(rtol=3e-3, atol=3e-3)
ACC = dict(rtol=3e-3, atol=1e-2)
Q_ATOL, V_ATOL, TIME_ATOL = 1e-4, 1e-3, 2e-6
TOL = {
    "body_position": (3e-5, 4e-5),
    "body_rotation": (3e-5, 4e-5),
    "body_com": (3e-5, 4e-5),
    "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (3e-4, 5e-5),
    "bias": (4e-4, 4e-4),
    "passive": (4e-4, 4e-4),
    "applied": (4e-4, 4e-4),
    "smooth_acceleration": (2e-3, 4e-3),
}
ROOT_INERTIA = (
    '<inertial pos=".04 -.03 .02" quat=".9659258262890683 0 .2588190451025207 0" '
    'mass="2.6" diaginertia=".13 .19 .23"/>'
)
HEADER = """<mujoco model="free-ball-normal"><compiler angle="radian" fusestatic="false"/>
<option timestep=".001" gravity="0 0 -9.81" integrator="Euler" cone="pyramidal"
 iterations="128" tolerance="1e-10" jacobian="dense"/>
<default><geom condim="3" friction=".4 .01 .001"
 solref=".02 1" solimp=".8 .9 .002 .5 2"/></default>"""


def motor(name, target, gear, extra="", gain="1"):
    return (
        f'<general name="{name}" joint="{target}" gear="{gear}" '
        f'dyntype="none" gaintype="fixed" gainprm="{gain}" {extra}/>'
    )


def source(name):
    """Two prerecorded generic articulated systems; no result-dependent parameters."""
    if name == "feet":
        static = '<geom name="ground" type="plane" size="1 1 .1"/>'
        root_pos = ".2 -.1 .4"
        children = """
<body name="foot0" pos="-.18 0 -.301">
 <joint name="ball" type="ball" damping=".04" armature=".003"/>
 <inertial mass=".4" pos=".02 0 .01" diaginertia=".01 .012 .014"/>
 <geom name="toe0" type="sphere" size=".1" pos=".04 0 0"/>
</body>
<body name="foot1" pos=".18 0 -.301">
 <joint name="slide" type="slide" axis="0 0 1" damping=".04" armature=".003"/>
 <inertial mass=".5" pos="-.01 0 .01" diaginertia=".012 .014 .016"/>
 <geom name="toe1" type="sphere" size=".1"/>
</body>"""
        actors = motor("drive", "ball", "1 -2 .5", gain=".8") + motor("slide-drive", "slide", "-2")
        header = HEADER
    elif name == "siblings":
        static, root_pos = "", ".2 -.1 .7"
        children = """
<body name="left" pos="-.099 0 .2">
 <joint name="ball" type="ball" pos=".02 -.01 .03" damping=".04" armature=".003"/>
 <inertial mass="1" pos="0 .02 .01" diaginertia=".03 .04 .05"/>
 <geom name="left-shape" type="sphere" size=".1"/>
</body>
<body name="right" pos=".099 0 .2">
 <joint name="other" type="ball" pos=".02 -.01 .03" damping=".04" armature=".003"/>
 <inertial mass="1.4" pos="0 -.03 .015" diaginertia=".04 .05 .06"/>
 <geom name="right-shape" type="sphere" size=".1"/>
</body>"""
        actors = motor("drive", "ball", "1 -2 .5", gain=".8")
        actors += motor("counter", "ball", "-.25 .5 -.125", gain=".6")
        actors += motor("other-drive", "other", ".3 -.4 .2", gain=".7")
        header = HEADER.replace('gravity="0 0 -9.81"', 'gravity="0 0 0"')
    else:
        raise ValueError("only feet and siblings are declared")
    return (
        header
        + "<worldbody>"
        + static
        + '<body name="marker" pos="2 -1 .4"><geom size=".02" contype="0" conaffinity="0"/></body>'
        + f'<body name="root" pos="{root_pos}"><freejoint name="free"/>'
        + ROOT_INERTIA
        + '<geom size=".06" contype="0" conaffinity="0"/>'
        + children
        + "</body></worldbody><actuator>"
        + actors
        + "</actuator></mujoco>"
    )


def capacity_source(kind, overflow):
    """Exact normal pair and raw-slot boundaries; only static admission is tested."""
    count = (17 if overflow else 16) if kind == "pairs" else (5 if overflow else 4)
    geometry = 'type="sphere" size=".03"' if kind == "pairs" else 'type="box" size=".03 .04 .05"'
    sites = "".join(f'<geom name="site{i}" {geometry} pos="{i * 0.1} 0 0"/>' for i in range(count))
    return (
        HEADER
        + '<worldbody><geom name="ground" type="plane" size="1 1 .1"/>'
        + '<body name="root" pos="0 0 .6"><freejoint name="free"/>'
        + ROOT_INERTIA
        + '<body name="foot0" pos="0 0 -.3"><joint name="ball" type="ball"/>'
        + '<inertial mass=".4" pos=".02 0 .01" diaginertia=".01 .012 .014"/>'
        + sites
        + "</body></body></worldbody><actuator>"
        + motor("drive", "ball", "1 -2 .5")
        + "</actuator></mujoco>"
    )


def negative_source(name):
    xml = source("feet")
    if name == "activation":
        return xml.replace('dyntype="none"', 'dyntype="filterexact" dynprm=".02"', 1)
    if name in ("condim4", "condim6"):
        return xml.replace('condim="3"', f'condim="{name[-1]}"')
    if name == "elliptic":
        return xml.replace('cone="pyramidal"', 'cone="elliptic"')
    if name == "ball-limit":
        return xml.replace(
            'name="ball" type="ball"', 'name="ball" type="ball" limited="true" range="0 .8"'
        )
    if name == "scalar-limit":
        return xml.replace(
            'name="slide" type="slide"', 'name="slide" type="slide" limited="true" range="-.5 .5"'
        )
    if name == "dry":
        return xml.replace('name="ball" type="ball"', 'name="ball" type="ball" frictionloss=".1"')
    if name == "spring":
        return xml.replace('name="ball" type="ball"', 'name="ball" type="ball" stiffness=".1"')
    if name == "free-target":
        return xml.replace('joint="ball"', 'joint="free"')
    if name == "ball-affine":
        return xml.replace('name="drive"', 'name="drive" biastype="affine" biasprm="0 -.1 0"')
    if name == "parent-transmission":
        return xml.replace('joint="ball"', 'jointinparent="ball"')
    if name == "explicit-pair":
        return xml.replace(
            "</mujoco>", '<contact><pair geom1="ground" geom2="toe0"/></contact></mujoco>'
        )
    if name == "exclude":
        return xml.replace(
            "</mujoco>", '<contact><exclude body1="foot0" body2="foot1"/></contact></mujoco>'
        )
    if name == "cylinder":
        return xml.replace(
            'name="toe0" type="sphere" size=".1"', 'name="toe0" type="cylinder" size=".1 .1"'
        )
    if name in ("pair-overflow", "slot-overflow"):
        return capacity_source("pairs" if name == "pair-overflow" else "slots", True)
    if name == "geom-overflow":
        masked = "".join(
            f'<geom name="masked{i}" size=".01" contype="0" conaffinity="0"/>' for i in range(129)
        )
        return xml.replace("</worldbody>", masked + "</worldbody>")
    raise ValueError(name)


def initial(mj, model, name, slot):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    q, v = model.qpos0.copy(), np.zeros(model.nv)
    # Yaw changes no authored first-step foot heights or sibling separation.
    # Both sibling local rotations agree, so their equal anchor offset shifts
    # both centers equally; the 0.198 m center distance remains below 2r.
    q[:2] += np.array([0.02 * slot, -0.01 * slot])
    q[3:7] = (math.cos(0.07), 0, 0, math.sin(0.07))
    v[:6] = np.array([0.01, -0.02, -0.025, 0.01, -0.015, 0.02]) * (1 + 0.1 * slot)
    for joint in range(1, model.njnt):
        qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        if int(model.jnt_type[joint]) == int(mj.mjtJoint.mjJNT_BALL):
            q[qa : qa + 4] = (math.cos(0.03), 0, 0, math.sin(0.03))
            direction = -1 if name == "siblings" and joint == 2 else 1
            v[dof : dof + 3] = direction * np.array([0.02, -0.03, 0.015]) * (1 + 0.1 * slot)
        else:
            v[dof] = -0.02 * (1 + 0.1 * slot)
    return MuJoCoInitialState(q, v.astype(np.float32), act=np.zeros(0), time=0.125)


def fixture(name, mode, keys=KEYS):
    import mujoco as mj

    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    require(mj.__version__ == "3.3.7", "fixed native oracle")
    xml = source(name)
    model = mj.MjModel.from_xml_string(xml)
    compiled = compile_mujoco_xml(xml, require_asset=False)
    starts = {key: initial(mj, model, name, i) for i, key in enumerate(keys)}
    runtime = MuJoCoRuntime(
        compiled, mode=mode, implementation="tyx", world_keys=keys, initial_states=starts
    )
    return mj, model, compiled, starts, runtime


def references(mj, model, states):
    result = {}
    for key, state in states.items():
        data = mj.MjData(model)
        data.qpos[:], data.qvel[:], data.act[:], data.time = state.q, state.v, state.act, state.time
        result[key] = data
    return result


def native_fields(mj, model, entry, packet):
    data = references(mj, model, {0: entry})[0]
    set_control(data, packet)
    mj.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_applied.copy() + data.qfrc_actuator
    for body in range(model.nbody):
        mj.mj_jac(model, data, jac[body, :3], jac[body, 3:], data.xipos[body], body)
        applied += jac[body].T @ np.asarray(packet.xfrc_applied[body], np.float32)
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    contacts = []
    for i in range(data.ncon):
        native = data.contact[i]
        ids = (int(native.geom1), int(native.geom2))
        bodies = tuple(int(model.geom_bodyid[g]) for g in ids)
        frame = native.frame.reshape(3, 3).copy()
        cj = np.zeros((6, model.nv))
        for sign, body in zip((-1, 1), bodies):
            part = np.zeros((6, model.nv))
            mj.mj_jac(model, data, part[:3], part[3:], native.pos, body)
            cj[:3] += sign * (frame @ part[:3])
            cj[3:] += sign * (frame @ part[3:])
        wrench = np.zeros(6)
        mj.mj_contactForce(model, data, i, wrench)
        contacts.append(
            dict(
                native_index=i,
                geometry_ids=ids,
                body_ids=bodies,
                dimension=int(native.dim),
                position=native.pos.copy(),
                frame=frame,
                distance=float(native.dist),
                include_margin=float(native.includemargin),
                friction=native.friction.copy(),
                solref=native.solref.copy(),
                solimp=native.solimp.copy(),
                active=int(native.efc_address) >= 0,
                address=int(native.efc_address),
                jacobian=cj,
                wrench=wrench,
            )
        )
    expected = dict(
        input_q=entry.q.copy(),
        input_v=entry.v.copy(),
        time=entry.time,
        control=packet,
        native_input=native_state(data),
        body_position=data.xpos.copy(),
        body_rotation=data.xmat.reshape(model.nbody, 3, 3).copy(),
        body_com=data.xipos.copy(),
        jacobian_com=jac,
        mass_matrix=mass,
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=applied,
        smooth_acceleration=np.linalg.solve(mass, rhs),
        qacc=data.qacc.copy(),
        integration_acceleration=np.linalg.solve(
            mass + H * np.diag(model.dof_damping), rhs + data.qfrc_constraint
        ),
        qfrc_constraint=data.qfrc_constraint.copy(),
        qfrc_actuator=data.qfrc_actuator.copy(),
        actuator_force=data.actuator_force.copy(),
        act_dot=data.act_dot.copy(),
        contacts=contacts,
        efc_J=data.efc_J.reshape(data.nefc, model.nv).copy(),
        efc_R=data.efc_R.copy(),
        efc_aref=data.efc_aref.copy(),
        efc_force=data.efc_force.copy(),
        efc_type=data.efc_type.copy(),
        efc_id=data.efc_id.copy(),
        efc_pos=data.efc_pos.copy(),
        efc_margin=data.efc_margin.copy(),
        efc_vel=data.efc_vel.copy(),
        efc_D=data.efc_D.copy(),
        efc_impedance=data.efc_KBIP[:, 2].copy(),
        efc_diagApprox=data.efc_diagApprox.copy(),
        analytic=analytic(model, entry, packet),
        dense_moment=dense_native_moment(model, data),
    )
    mj.mj_step(model, data)
    expected["native_successor"] = native_state(data)
    return expected


def match_contacts(actual, native):
    require(len(actual) == len(native), "complete contact count")
    unused = set(range(len(native)))
    result = []
    for contact in actual:
        candidates = [
            i
            for i in unused
            if contact.geometry_ids == native[i]["geometry_ids"]
            and np.allclose(contact.position, native[i]["position"], rtol=3e-5, atol=4e-5)
            and np.allclose(contact.frame[0], native[i]["frame"][0], rtol=0, atol=1e-4)
        ]
        require(len(candidates) == 1, "unique geometry-only contact bijection")
        index = candidates[0]
        unused.remove(index)
        result.append(native[index])
    require(not unused, "all native contacts consumed")
    return result


def compare(row, before, after, packet, expected, model):
    require(
        row.world == after.world and row.tick == after.tick == before.tick + 1, "world/tick binding"
    )
    require(
        row.input_time == before.time and row.output_time == after.time and row.h == H,
        "clock binding",
    )
    exact(row.input_q, before.q, "input q")
    exact(row.input_v, before.v, "input v")
    exact(row.control, packet, "requested control")
    for field in ("ctrl", "qfrc_applied", "xfrc_applied"):
        np.testing.assert_array_equal(
            getattr(row.stored_control, field), np.asarray(getattr(packet, field), np.float32)
        )
    dyn = row.dynamics
    exact(dyn.input_q, row.input_q, "dynamics q")
    exact(dyn.input_v, row.input_v, "dynamics v")
    require(dyn.time == row.input_time, "dynamics time")
    for field, (rtol, atol) in TOL.items():
        value = getattr(dyn, field)
        require(
            value.shape == expected[field].shape and not value.flags.writeable,
            "complete immutable " + field,
        )
        np.testing.assert_allclose(value, expected[field], rtol=rtol, atol=atol, err_msg=field)
    for field in ("qacc", "integration_acceleration"):
        np.testing.assert_allclose(getattr(row, field), expected[field], **ACC)
    np.testing.assert_allclose(row.qfrc_actuator, expected["qfrc_actuator"], **FORCE)
    np.testing.assert_allclose(row.qfrc_constraint, expected["qfrc_constraint"], **CONTACT_FORCE)
    np.testing.assert_allclose(
        row.qfrc_actuator, expected["analytic"]["generalized_force"], **FORCE
    )
    require(model.na == 0 and after.act.shape == (0,), "stateless native integration")
    check_transmission(row, before, expected)

    native = match_contacts(row.contacts, expected["contacts"])
    generalized = np.zeros(model.nv)
    native_rows = []
    for contact, oracle in zip(row.contacts, native, strict=True):
        require(
            contact.body_ids == oracle["body_ids"] and contact.dimension == oracle["dimension"],
            "complete contact identity",
        )
        require(contact.active == oracle["active"], "active contact")
        for field in (
            "position",
            "frame",
            "distance",
            "include_margin",
            "friction",
            "solref",
            "solimp",
        ):
            np.testing.assert_allclose(getattr(contact, field), oracle[field], rtol=3e-5, atol=4e-5)
        cj = np.asarray(contact.jacobian)
        normal = cj if cj.ndim == 1 else cj[0]
        np.testing.assert_allclose(
            cj, oracle["jacobian"][0] if cj.ndim == 1 else oracle["jacobian"], rtol=2e-4, atol=5e-5
        )
        np.testing.assert_allclose(contact.wrench_contact_frame, oracle["wrench"], **CONTACT_FORCE)
        require(
            contact.normal_force >= 0 and contact.normal_force == contact.wrench_contact_frame[0],
            "normal force",
        )
        np.testing.assert_allclose(
            contact.normal_velocity, normal @ row.input_v, rtol=5e-4, atol=4e-4
        )
        force = contact.frame.T @ contact.wrench_contact_frame[:3]
        couple = contact.frame.T @ contact.wrench_contact_frame[3:]
        body_generalized = np.zeros(model.nv)
        for side, sign in ((1, -1), (2, 1)):
            body = contact.body_ids[side - 1]
            wrench = np.r_[
                sign * force,
                sign * (couple + np.cross(contact.position - dyn.body_com[body], force)),
            ]
            np.testing.assert_allclose(
                getattr(contact, f"wrench_on_body{side}_world"), wrench, rtol=4e-5, atol=4e-5
            )
            np.testing.assert_allclose(
                getattr(contact, f"force_on_geometry{side}_world"),
                sign * force,
                rtol=4e-5,
                atol=4e-5,
            )
            body_generalized += dyn.jacobian_com[body].T @ wrench
        from_contact = (
            normal * contact.normal_force if cj.ndim == 1 else cj.T @ contact.wrench_contact_frame
        )
        np.testing.assert_allclose(body_generalized, from_contact, rtol=4e-5, atol=4e-5)
        generalized += body_generalized
        if contact.active:
            count = 1 if contact.dimension == 1 else 2 * (contact.dimension - 1)
            native_rows.extend(range(oracle["address"], oracle["address"] + count))
            if cj.ndim == 1:
                adr = oracle["address"]
                np.testing.assert_allclose(
                    contact.regularization, expected["efc_R"][adr], rtol=5e-4, atol=4e-4
                )
                np.testing.assert_allclose(
                    contact.reference_acceleration, expected["efc_aref"][adr], rtol=5e-4, atol=4e-4
                )
        else:
            np.testing.assert_array_equal(contact.wrench_contact_frame, np.zeros(6))
    np.testing.assert_allclose(generalized, row.qfrc_constraint, rtol=5e-5, atol=5e-5)
    raw = row.raw_contact_fields
    # Native geometry-only bijection defines the EFC permutation. Never compare
    # a normal-only proxy or sort by force to obtain agreement.
    require(len(row.constraints) == len(native_rows) == len(expected["efc_R"]),
            "complete native pyramidal EFC count")
    ids, values, jac = (np.asarray(raw[k]) for k in ("efc_ids", "efc_data", "efc_J"))
    cmap = np.asarray(raw["contact_map"])
    n, nc = len(row.constraints), len(row.contacts)
    require(ids.shape[1] == 5 and values.shape[1] == 14 and jac.shape[1] >= model.nv,
            "complete EFC buffer layout")
    np.testing.assert_array_equal(raw["counts"], [nc, sum(c.active for c in row.contacts), n])
    scalar_fields = (
        ("position", "efc_pos"), ("margin", "efc_margin"), ("velocity", "efc_vel"),
        ("reference_acceleration", "efc_aref"), ("regularization", "efc_R"),
        ("inverse_regularization", "efc_D"), ("impedance", "efc_impedance"),
        ("diag_approx", "efc_diagApprox"),
    )
    reconstructed = np.zeros(model.nv)
    for i, (constraint, ni) in enumerate(zip(row.constraints, native_rows, strict=True)):
        require(constraint.schema == "mujoco-tyx-constraint-row-v1", "EFC schema")
        require(constraint.constraint_type == int(expected["efc_type"][ni]) == 6,
                "native pyramidal row type")
        require(native[constraint.source_id]["native_index"] == int(expected["efc_id"][ni]),
                "native contact owner")
        contact = row.contacts[constraint.source_id]
        edge = ni - native[constraint.source_id]["address"]
        require(constraint.side_or_edge == edge and edge in range(4) and constraint.dof == -1,
                "all four native edge identities")
        require(constraint.geometry_slot == cmap[constraint.source_id, 0], "geometry slot")
        np.testing.assert_array_equal(
            ids[i], [constraint.constraint_type, constraint.source_id, edge,
                     constraint.geometry_slot, constraint.dof])
        np.testing.assert_array_equal(jac[i, :model.nv], constraint.jacobian)
        np.testing.assert_allclose(constraint.jacobian, expected["efc_J"][ni],
                                   rtol=2e-4, atol=5e-5)
        axis, sign = edge // 2, (1 if edge % 2 == 0 else -1)
        independent_edge = contact.jacobian[0] + sign * contact.friction[axis] * contact.jacobian[1 + axis]
        np.testing.assert_allclose(constraint.jacobian, independent_edge, rtol=2e-4, atol=5e-5)
        for field, target in scalar_fields:
            np.testing.assert_allclose(getattr(constraint, field), expected[target][ni],
                                       rtol=5e-4, atol=4e-4, err_msg=field)
        np.testing.assert_allclose(constraint.force, expected["efc_force"][ni], **CONTACT_FORCE)
        require(constraint.lower == 0 and constraint.upper is None and constraint.force >= 0,
                "unchanged nonnegative pyramid edge force")
        np.testing.assert_array_equal(
            values[i, [0,1,2,3,4,5,6,7,8,9,10,12,13]],
            [constraint.position, constraint.margin, constraint.velocity,
             constraint.reference_acceleration, constraint.regularization,
             constraint.inverse_regularization, constraint.impedance, 0, 0,
             constraint.force, constraint.gradient, 0, constraint.diag_approx])
        require(values[i, 11] == 1, "active EFC raw flag")
        reconstructed += constraint.jacobian * constraint.force
    np.testing.assert_allclose(reconstructed, row.qfrc_constraint, rtol=5e-5, atol=5e-5)
    first = 0
    for ci, contact in enumerate(row.contacts):
        require(contact.dimension == 3 and contact.jacobian.shape == (6, model.nv),
                "full contact dimension and Jacobian")
        count = 4 if contact.active else 0
        np.testing.assert_array_equal(cmap[ci, 1:], [first if count else -1, count])
        np.testing.assert_array_equal(np.asarray(raw["contact_J"])[ci, :, :model.nv], contact.jacobian)
        np.testing.assert_array_equal(np.asarray(raw["contact_wrench"])[ci], contact.wrench_contact_frame)
        decoded = np.zeros(6)
        if count:
            forces = np.array([r.force for r in row.constraints[first:first + 4]])
            decoded[0] = np.sum(forces)
            decoded[1] = contact.friction[0] * (forces[0] - forces[1])
            decoded[2] = contact.friction[1] * (forces[2] - forces[3])
            require(all(r.source_id == ci for r in row.constraints[first:first + 4]),
                    "exact compact contact-to-edge ownership")
        np.testing.assert_allclose(decoded, contact.wrench_contact_frame, rtol=4e-5, atol=4e-5)
        # Contact velocity and both COM wrench powers use the actual input v.
        power = float(contact.wrench_contact_frame @ (contact.jacobian @ row.input_v))
        dual_power = sum(float(getattr(contact, f"wrench_on_body{side}_world") @
                               (dyn.jacobian_com[body] @ row.input_v))
                         for side, body in enumerate(contact.body_ids, 1))
        np.testing.assert_allclose(dual_power, power, rtol=5e-5, atol=5e-5)
        first += count
    require(first == n, "no omitted or extra EFC edge")
    # Independent physical-M QP residual, retaining the original mixed-solver
    # engineering audit gate. Euler M+hD is checked separately above.
    j = jac[:n, :model.nv].astype(float)
    force = np.array([r.force for r in row.constraints])
    reg = np.array([r.regularization for r in row.constraints])
    aref = np.array([r.reference_acceleration for r in row.constraints])
    hessian = j @ np.linalg.solve(dyn.mass_matrix.astype(float), j.T) + np.diag(reg)
    linear = j @ dyn.smooth_acceleration - aref
    native_j = expected["efc_J"][native_rows]
    native_h = native_j @ np.linalg.solve(expected["mass_matrix"], native_j.T) + np.diag(expected["efc_R"][native_rows])
    np.testing.assert_allclose(hessian, native_h, rtol=7e-4, atol=5e-4)
    gradient = hessian @ force + linear
    working = np.asarray(raw["working_set"])[:n]
    require(set(map(int, working)) <= {0, 1}, "no unsupported upper edge bound")
    np.testing.assert_array_equal(force[working == 0], 0)
    violation = np.where(working == 0, np.maximum(0, -gradient), np.abs(gradient))
    scale = max(1.0, np.max(np.abs(linear) + np.abs(hessian) @ np.abs(force), initial=0))
    require(np.max(violation, initial=0) / scale <= 5e-5, "independent physical-M KKT")
    for i, constraint in enumerate(row.constraints):
        np.testing.assert_allclose(constraint.gradient, gradient[i], rtol=3e-3, atol=5e-5 * scale)
    physical_rhs = dyn.applied + dyn.passive - dyn.bias + row.qfrc_constraint
    mass_error = dyn.mass_matrix.astype(float) @ row.qacc - physical_rhs
    mass_scale = max(1.0, np.max(np.abs(physical_rhs) + np.abs(dyn.mass_matrix) @ np.abs(row.qacc), initial=0))
    require(np.max(np.abs(mass_error), initial=0) / mass_scale <= 5e-5, "independent mass residual")
    diag = raw["solver_diagnostics"]
    require(diag["converged"] and diag["actual_efc"] == n and
            diag["normalized_kkt"] <= float(np.float32(max(model.opt.tolerance, 2e-5))),
            "original KKT gate")
    require(diag["normalized_mass_residual"] <= 5e-5 and diag["bound_violation"] <= 2e-7,
            "original mass/bound gates")
    require(diag["face_solves"] <= diag["budget"] == model.opt.iterations, "original source budget")
    np.testing.assert_array_equal(values[n:], 0)
    np.testing.assert_array_equal(jac[n:], 0)
    np.testing.assert_array_equal(np.asarray(raw["contact_wrench"])[nc:], 0)
    np.testing.assert_array_equal(np.asarray(raw["contact_J"])[nc:], 0)
    np.testing.assert_array_equal(cmap[nc:, :2], -1)
    np.testing.assert_array_equal(cmap[nc:, 2], 0)
    for target in (expected["native_successor"], expected["continuous"]):
        np.testing.assert_allclose(after.q, target["q"], rtol=0, atol=Q_ATOL)
        np.testing.assert_allclose(after.v, target["v"], rtol=0, atol=V_ATOL)
        np.testing.assert_allclose(after.act, target["act"], **ACT)
        require(abs(after.time - target["time"]) <= TIME_ATOL, "native clock")
    for joint, kind in enumerate(model.jnt_type):
        if int(kind) in (0, 1):
            qa = int(model.jnt_qposadr[joint]) + (3 if int(kind) == 0 else 0)
            require(
                abs(float(after.q[qa : qa + 4] @ after.q[qa : qa + 4]) - 1)
                <= 8 * np.finfo(np.float32).eps,
                "root/child quaternion gate",
            )
    np.testing.assert_allclose(
        after.v, before.v + float(np.float32(H)) * row.integration_acceleration, rtol=0, atol=3e-6
    )


def observations(runtime):
    full, integration = runtime.snapshot(), runtime.integration_snapshot()
    require(tuple(full) == tuple(integration) == tuple(runtime.world_keys), "world order")
    for key in runtime.world_keys:
        for name in (
            "world",
            "q",
            "v",
            "act",
            "ctrl",
            "time",
            "tick",
            "qfrc_applied",
            "xfrc_applied",
            "held_control",
            "last_step",
            "coordinate_system",
        ):
            exact(
                getattr(full[key], name),
                getattr(integration[key], name),
                "full/integration " + name,
            )
    return full, integration


def advance(runtime, mj, model, packets, refs, journal, label):
    full_entry, entry = observations(runtime)
    completed_before = runtime.counters["completed_steps"]
    accepted_before = runtime.counters["accepted_world_steps"]
    launches_before = runtime.counters["completed_kernels"]
    journal.emit(
        "input",
        label=label,
        entry=entry,
        full_entry=full_entry,
        requested=packets,
        counters=dict(runtime.counters),
    )
    expected = {k: native_fields(mj, model, s, packets[k]) for k, s in entry.items()}
    for key in runtime.world_keys:
        set_control(refs[key], packets[key])
        mj.mj_step(model, refs[key])
        expected[key]["continuous"] = native_state(refs[key])
    runtime.set_controls(packets)
    journal.emit(
        "attempt",
        label=label,
        entry=entry,
        full_entry=full_entry,
        requested=packets,
        native=expected,
        revision=runtime.checkpoint_revision,
        counters=dict(runtime.counters),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU forward/step inside TYX physics")

    try:
        with ExitStack() as guard:
            for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward"):
                guard.enter_context(patch.object(mj, name, forbidden))
            accepted = runtime.step(H)
        journal.emit(
            "physics-return",
            label=label,
            accepted=accepted,
            revision=runtime.checkpoint_revision,
            counters=dict(runtime.counters),
        )
        full, output = observations(runtime)
        journal.emit(
            "step",
            label=label,
            accepted=accepted,
            state=output,
            full=full,
            revision=runtime.checkpoint_revision,
            counters=dict(runtime.counters),
        )
        require(accepted == dict.fromkeys(runtime.world_keys, True), "all real worlds accepted")
        require(
            runtime.counters["completed_steps"] == completed_before + 1, "one actual batch step"
        )
        require(
            runtime.counters["accepted_world_steps"] == accepted_before + len(runtime.world_keys),
            "one accepted step per world",
        )
        stages = 6 + runtime.capabilities["normal_geometry_launches"]
        require(
            runtime.counters["completed_kernels"] == launches_before + stages,
            "unchanged contact pipeline launch count",
        )
        for key in runtime.world_keys:
            compare(
                output[key].last_step, entry[key], output[key], packets[key], expected[key], model
            )
        return output
    except BaseException as primary:
        try:
            journal.emit(
                "step-failure",
                label=label,
                error=repr(primary),
                traceback=traceback.format_exc(),
                published=runtime.integration_snapshot(),
                counters=dict(runtime.counters),
            )
        except BaseException as secondary:
            raise BaseExceptionGroup("original step and evidence failures", [primary, secondary])
        raise


def domain(name):
    require(name in ("feet", "siblings"), "declared pyramidal domain")
    return "free-ball-joint-motor-pyramidal-contact-v1"


def continuous(name, mode, journal):
    mj, model, compiled, starts, runtime = fixture(name, mode)
    try:
        require(runtime.capabilities["domain"] == domain(name), "precise composed domain")
        require(
            model.na == 0
            and tuple(runtime.capabilities["contact_dimensions"]) == (1, 3)
            and not runtime.capabilities["joint_limits_supported"]
            and not runtime.capabilities["dry_friction_supported"],
            "bounded stateless pyramidal domain",
        )
        require(
            (model.nq, model.nv) == ((12, 10) if name == "feet" else (15, 12)),
            "complete native FREE/BALL/scalar addresses",
        )
        state = runtime.integration_snapshot()
        start = state
        refs = references(mj, model, state)
        journal.emit(
            "declaration",
            name=name,
            mode=mode,
            source_xml=source(name),
            source_sha256=hashlib.sha256(source(name).encode()).hexdigest(),
            compilation=compiled.source,
            starts=starts,
            stored=state,
            capabilities=runtime.capabilities,
            joint_types=model.jnt_type.copy(),
            qa=model.jnt_qposadr.copy(),
            dof=model.jnt_dofadr.copy(),
            tolerances=TOL,
            force_gate=FORCE,
            contact_force_gate=CONTACT_FORCE,
            acceleration_gate=ACC,
            q_gate=Q_ATOL,
            v_gate=V_ATOL,
            time_gate=TIME_ATOL,
            steps=24,
            h=H,
        )
        active = {key: [] for key in KEYS}
        both_moments = {key: False for key in KEYS}
        tangent_response = {key: False for key in KEYS}
        for tick in range(24):
            packets = {key: control(model, tick, i) for i, key in enumerate(runtime.world_keys)}
            state = advance(runtime, mj, model, packets, refs, journal, f"{name}-{tick + 1}")
            for key in KEYS:
                row = state[key].last_step
                active[key].append(any(c.active and c.normal_force > 0 for c in row.contacts))
                tangent_response[key] = tangent_response[key] or any(
                    c.active and np.any(c.wrench_contact_frame[1:3] != 0) for c in row.contacts)
                require(np.any(row.qfrc_actuator != 0) or tick >= 16, "real JOINT drive retained")
                if tick >= 16:
                    np.testing.assert_array_equal(row.control.ctrl, np.zeros(model.nu))
                    np.testing.assert_array_equal(row.control.qfrc_applied, np.zeros(model.nv))
                    np.testing.assert_array_equal(
                        row.control.xfrc_applied, np.zeros((model.nbody, 6))
                    )
                if name == "siblings":
                    both_moments[key] = both_moments[key] or any(
                        np.linalg.norm(c.wrench_on_body1_world[3:]) > 0
                        and np.linalg.norm(c.wrench_on_body2_world[3:]) > 0
                        for c in row.contacts
                        if c.active and c.normal_force > 0
                    )
                    # Internal sibling contact has no resultant free-root force
                    # or torque, despite nonzero reaction in both child joints.
                    for c in row.contacts:
                        np.testing.assert_allclose(c.jacobian[:, :6], np.zeros((6, 6)), rtol=0, atol=5e-5)
                    np.testing.assert_allclose(
                        row.qfrc_constraint[:6], np.zeros(6), rtol=0, atol=5e-5
                    )
                if tick == 0:
                    live = [c for c in row.contacts if c.active and c.normal_force > 0]
                    require(
                        len(live) == (2 if name == "feet" else 1), "authored initial contact load"
                    )
                    if name == "feet":
                        require(row.qfrc_constraint[2] > 0, "actual free-root upward support")
                    else:
                        require(
                            np.linalg.norm(row.qfrc_constraint[6:9]) > 0
                            and np.linalg.norm(row.qfrc_constraint[9:12]) > 0,
                            "both BALL branches have contact response",
                        )
        require(all(any(values) for values in active.values()), "contact in both actual worlds")
        require(all(tangent_response.values()), "real tangential response in both worlds")
        for key in KEYS:
            require(np.any(state[key].q[:7] != start[key].q[:7]), "free root actually moves")
        if name == "siblings":
            require(all(both_moments.values()), "both moving bodies receive COM moment")
            for qa in (7, 11):
                require(
                    np.any(state[KEYS[0]].q[qa : qa + 4] != start[KEYS[0]].q[qa : qa + 4]),
                    "both BALL branches actually move",
                )
        require(runtime.counters["completed_steps"] == 24, "24 actual uninterrupted batches")
        require(runtime.counters["accepted_world_steps"] == 48, "48 actual world steps")
        require(runtime.counters["failed_batches"] == 0, "no hidden rejected step")
        journal.emit(
            "continuous-complete",
            name=name,
            active=active,
            state=state,
            counters=dict(runtime.counters),
            both_moments=both_moments,
            tangent_response=tangent_response,
        )
        return dict(name=name, accepted_world_steps=48, steps=24)
    finally:
        primary = sys.exc_info()[1]
        try:
            runtime.close()
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "R3-P continuous trajectory and cleanup failures",
                ([primary] if primary is not None else []) + [cleanup],
            ) from None


def lifecycle(name, mode, journal, keys=(7, 41, 103, 509), output_path=None):
    from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint

    mj, model, compiled, starts, runtime = fixture(name, mode, keys)
    clone = None
    calls = 0
    try:
        refs = references(mj, model, runtime.integration_snapshot())
        for tick in range(3):
            packets = {k: control(model, tick, i) for i, k in enumerate(keys)}
            advance(runtime, mj, model, packets, refs, journal, "lifecycle-warm")
            calls += 1
        held = {k: control(model, 8, i) for i, k in enumerate(keys)}
        runtime.set_controls(held)
        saved = runtime.integration_snapshot()
        raw = runtime.checkpoint().to_bytes()
        if output_path is not None:
            (Path(output_path) / "free-ball-pyramidal-checkpoint.bin").write_bytes(raw)
        capsule = MuJoCoTyxCheckpoint.from_bytes(raw)
        require(capsule.to_bytes() == raw, "portable capsule bytes")
        native_saved = {k: copy.copy(v) for k, v in refs.items()}
        journal.emit(
            "checkpoint", raw=raw, state=saved, native={k: native_state(v) for k, v in refs.items()}
        )
        clone = runtime.clone()
        exact(clone.integration_snapshot(), saved, "clone state")
        left = advance(runtime, mj, model, held, refs, journal, "A")
        calls += 1
        clone_refs = {k: copy.copy(v) for k, v in native_saved.items()}
        right = advance(clone, mj, model, held, clone_refs, journal, "clone-A")
        calls += 1
        exact(left, right, "clone real successor")
        revision = runtime.checkpoint_revision
        runtime.restore(capsule, expected_revision=revision)
        exact(runtime.integration_snapshot(), saved, "public exact restore")
        retained = runtime.checkpoint().to_bytes()
        try:
            runtime.restore(capsule, expected_revision=revision)
        except ValueError:
            pass
        else:
            raise AssertionError("stale CAS accepted")
        require(runtime.checkpoint().to_bytes() == retained, "stale CAS retained")
        refs = {k: copy.copy(v) for k, v in native_saved.items()}
        repeated = advance(runtime, mj, model, held, refs, journal, "B")
        calls += 1
        exact(left, repeated, "A/restore/B full receipt")
        checkpoint_before = runtime.checkpoint()
        runtime.reset((keys[-1],))
        after = runtime.integration_snapshot()
        for key in keys[:-1]:
            exact(after[key], repeated[key], "unselected complete publication")
            checkpoint_world_equal(checkpoint_before, runtime.checkpoint(), keys.index(key))
        require(after[keys[-1]].tick == 0 and after[keys[-1]].last_step is None, "selected reset")
        np.testing.assert_array_equal(
            after[keys[-1]].act, np.asarray(starts[keys[-1]].act, np.float32)
        )
        np.testing.assert_array_equal(after[keys[-1]].q, np.asarray(starts[keys[-1]].q, np.float32))
        np.testing.assert_array_equal(after[keys[-1]].v, np.asarray(starts[keys[-1]].v, np.float32))
        require(after[keys[-1]].time == starts[keys[-1]].time, "original reset time")
        refs[keys[-1]] = references(mj, model, {keys[-1]: after[keys[-1]]})[keys[-1]]
        journal.emit(
            "partial-reset",
            selected=keys[-1],
            state=after,
            checkpoint=runtime.checkpoint().to_bytes(),
        )
        for tick in (16, 17):
            packets = {k: control(model, tick, i) for i, k in enumerate(keys)}
            advance(runtime, mj, model, packets, refs, journal, "reset-successor")
            calls += 1
        return dict(name=name, accepted_world_steps=calls * len(keys), batch_calls=calls)
    finally:
        primary = sys.exc_info()[1]
        failures = []
        for owner in (clone, runtime):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    failures.append(error)
        if failures:
            raise BaseExceptionGroup(
                "R3-P lifecycle cleanup", ([primary] if primary else []) + failures
            )


def run(mode, output_path, journal, installed_root):
    feet = continuous("feet", mode, journal)
    siblings = continuous("siblings", mode, journal)
    life = lifecycle("siblings", mode, journal, KEYS, output_path)
    paths = provenance(installed_root)
    for module in (
        "phymeo.backends.mujoco_tyx._floating_model",
        "phymeo.backends.mujoco_tyx._floating_device",
        "phymeo.backends.mujoco_tyx._floating_kernels",
        "phymeo.backends.mujoco_tyx._constraint_device",
        "phymeo.backends.mujoco_tyx._constraint_solver",
    ):
        require(module in paths["loaded"], "actual backend provenance " + module)
    return dict(
        feet=feet,
        siblings=siblings,
        lifecycle=life,
        accepted_world_steps=feet["accepted_world_steps"]
        + siblings["accepted_world_steps"]
        + life["accepted_world_steps"],
        final_provenance=paths,
    )


def checkpoint_world_equal(left, right, index):
    """Read the public versioned bytes, never a private runtime bank."""

    def state(capsule):
        raw = capsule.to_bytes()
        header, digest, body = raw.split(b"\n", 2)
        require(
            header == b"PHYMEOTYXCP1" and digest == hashlib.sha256(body).hexdigest().encode(),
            "checkpoint encoding",
        )
        return json.loads(body)["state"]

    a, b = state(left), state(right)
    exact(a["worlds"][index], b["worlds"][index], "unselected complete mirror/receipt")
    for field in ("q", "v", "act", "clock"):
        if field not in a:
            require(field not in b, "checkpoint field")
            continue
        x, y = a[field], b[field]
        require(x["$array"] == y["$array"] and x["shape"] == y["shape"], "bank layout")
        shape = x["shape"]
        width = math.prod(shape[1:]) * np.dtype(x["$array"]).itemsize
        start = index * width
        require(
            base64.b64decode(x["data"])[start : start + width]
            == base64.b64decode(y["data"])[start : start + width],
            "unselected raw " + field,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-root", "--expected-install-root", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute(), "output must be absolute")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "free-ball-pyramidal.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="installed single FREE-root BALL+R/P; stateless direct JOINT motors and primitive condim3 pyramidal contacts",
        excluded="activation, condim4/6 and elliptic cone, ball limits/springs, dry friction, multiple FREE roots, heterogeneous grouping, training",
    )
    installed_root = None
    primary = None
    primary_traceback = None
    finalization_errors = []
    summary["finalization_errors"] = []

    def remember_finalization(stage, error):
        # Keep original exception objects for the final group, not just strings.
        finalization_errors.append(error)
        summary["status"] = "FAIL"
        summary["finalization_errors"].append(
            dict(
                stage=stage,
                type=type(error).__name__,
                message=str(error),
                traceback="".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                ),
            )
        )

    try:
        require(args.installed_root.is_absolute(), "installed-root must be absolute")
        installed_root = args.installed_root.resolve(strict=True)
        summary["initial_provenance"] = provenance(installed_root, preflight=True)
        journal.emit("provenance", details=summary["initial_provenance"])
        summary.update(run(args.mode, output, journal, installed_root))
        summary["status"] = "PASS"
    except BaseException as error:
        primary, primary_traceback = error, error.__traceback__
        summary["status"] = "FAIL"
        summary["exception"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
        try:
            (output / "failure.txt").write_text(summary["exception"]["traceback"], encoding="utf-8")
        except BaseException as failure:
            remember_finalization("primary-traceback-file", failure)
        try:
            journal.emit("failure", summary=summary)
        except BaseException as failure:
            remember_finalization("failure-journal", failure)
    finally:
        # Independent attempts: provenance/emit failure cannot bypass stream close
        # or the separate summary file, and none can replace the primary error.
        try:
            if installed_root is not None:
                summary["final_provenance"] = provenance(installed_root)
        except BaseException as failure:
            remember_finalization("final-provenance", failure)
        try:
            journal.emit("summary", summary=summary)
        except BaseException as failure:
            remember_finalization("summary-journal", failure)
        try:
            summary["trace"] = journal.close()
        except BaseException as failure:
            remember_finalization("journal-close-and-hash", failure)
        try:
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
            )
        except BaseException as failure:
            remember_finalization("summary-file", failure)
        try:
            print(json.dumps(summary, allow_nan=False))
        except BaseException as failure:
            remember_finalization("summary-stdout", failure)

    if finalization_errors:
        members = ([primary] if primary is not None else []) + finalization_errors
        raise BaseExceptionGroup("R3-P primary/finalization failures", members)
    if primary is not None:
        raise primary.with_traceback(primary_traceback)


if __name__ == "__main__":
    main()
