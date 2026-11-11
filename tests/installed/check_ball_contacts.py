"""Installed BALL-C contact composition consumer; no checkout imports.

Coupled hinge-ball-slide tree; two sparse worlds; 24 continuous steps and lifecycle.
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
    output = entry.act.copy()
    derivative = np.zeros(model.na)
    control = np.empty(model.nu)
    selected = np.empty(model.nu)
    raw = np.empty(model.nu)
    force = np.empty(model.nu)
    generalized = np.zeros(model.nv)
    for actor in range(model.nu):
        u = float(np.float32(packet.ctrl[actor]))
        if model.actuator_ctrllimited[actor]:
            u = float(np.clip(u, *model.actuator_ctrlrange[actor]))
        control[actor] = u
        adr = int(model.actuator_actadr[actor])
        disabled = bool(model.opt.disableactuator & (1 << int(model.actuator_group[actor])))
        value = u
        if adr >= 0:
            tau = max(1e-15, float(model.actuator_dynprm[actor, 0]))
            derivative[adr] = (u - entry.act[adr]) / tau
            value_next = (
                entry.act[adr]
                if disabled
                else entry.act[adr] + (u - entry.act[adr]) * -math.expm1(-H / tau)
            )
            if model.actuator_actlimited[actor]:
                value_next = float(np.clip(value_next, *model.actuator_actrange[actor]))
            output[adr] = value_next
            value = value_next if model.actuator_actearly[actor] else entry.act[adr]
        selected[actor] = value
        joint = int(model.actuator_trnid[actor, 0])
        qid, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        gear = float(model.actuator_gear[actor, 0])
        bias = (
            model.actuator_biasprm[actor, :3]
            if int(model.actuator_biastype[actor])
            else np.zeros(3)
        )
        raw[actor] = (
            0
            if disabled
            else float(model.actuator_gainprm[actor, 0]) * value
            + bias[0]
            + bias[1] * gear * entry.q[qid]
            + bias[2] * gear * entry.v[dof]
        )
        force[actor] = raw[actor]
        if model.actuator_forcelimited[actor]:
            force[actor] = np.clip(force[actor], *model.actuator_forcerange[actor])
        generalized[dof] += gear * force[actor]
    return dict(
        output_act=output,
        activation_derivative=derivative,
        clamped_control=control,
        force_input=selected,
        raw_force=raw,
        scalar_force=force,
        generalized_force=generalized,
    )


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
Q_ATOL, V_ATOL, TIME_ATOL = 2e-4, 3e-3, 2e-6
TOL = {
    "body_position": (2e-5, 3e-5),
    "body_rotation": (2e-5, 3e-5),
    "body_com": (2e-5, 3e-5),
    "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (2e-4, 3e-5),
    "bias": (3e-4, 2e-4),
    "passive": (3e-4, 2e-4),
    "applied": (3e-4, 2e-4),
    "smooth_acceleration": (5e-4, 8e-4),
}
INERTIA = '<inertial mass="1.3" pos=".04 -.03 .02" fullinertia=".04 .055 .065 .002 -.001 .003"/>'
JOINT = 'damping=".12" armature=".04"'
SCALAR = 'ref=".07" springref="-.03" stiffness=".2"'
HEADER = """<mujoco><compiler angle="radian" fusestatic="false"/>
<option timestep=".001" gravity="0 0 -9.81" integrator="Euler" cone="pyramidal"
 iterations="128" tolerance="1e-10" jacobian="dense"/>
<default><geom friction=".4 .01 .001" solref=".02 1" solimp=".8 .9 .002 .5 2"/></default>"""


def source(name):
    dim = 3 if name.startswith("c3") or name == "c4-two-ball" else 1
    if name.startswith("c1"):
        height = ".33577777777777778" if name == "c1-inactive" else ".13577777777777778"
        world = f"""<geom name="ground" type="plane" size="1 1 .1" condim="1"/>
        <body name="rotor" pos="0 0 {height}"><joint name="ball" type="ball" {JOINT}/>
        {INERTIA}<geom name="tip" type="sphere" pos=".08 .03 -.18" size=".1" condim="1"/></body>"""
        actors = ""
    elif name == "c4-two-ball":
        branches = []
        for label, x, mass, inertia in (
            ("a", "-.095", "1", ".04 .05 .06"),
            ("b", ".095", "1.7", ".06 .08 .09"),
        ):
            branches.append(f'''<body name="{label}" pos="{x} 0 .5">
            <joint name="ball-{label}" type="ball" {JOINT}/>
            <inertial mass="{mass}" pos=".02 -.01 .03" diaginertia="{inertia}"/>
            <body name="tip-{label}"><joint name="slide-{label}" type="slide" axis="1 0 0" {JOINT}/>
            <inertial mass="{mass}" pos=".01 .02 -.02" diaginertia="{inertia}"/>
            <geom name="sphere-{label}" type="sphere" pos="0 .06 -.1" size=".1" condim="3"/></body></body>''')
        world = "".join(branches)
        actors = '<motor joint="slide-a" gear="-.5"/><motor joint="slide-b" gear=".5"/>'
    else:
        world = f'''<geom name="ground" type="box" pos="0 0 -.05" size=".8 .8 .05" condim="{dim}"/>
        <body name="base" pos="0 0 .4"><joint name="hinge" type="hinge" axis="0 0 1" {JOINT} {SCALAR}/>
        {INERTIA}<body name="middle" pos="0 0 -.2"><joint name="ball" type="ball" {JOINT}/>
        {INERTIA}<body name="tip" pos="0 0 -.1"><joint name="slide" type="slide" axis="0 0 1" {JOINT} {SCALAR}/>
        {INERTIA}<geom name="tip-sphere" type="sphere" pos=".06 0 0" size=".1" condim="{dim}"/>
        </body></body></body>'''
        stateful = name.startswith("c3")
        dtype = "filterexact" if stateful else "none"
        early = "false" if name == "c3-late" else "true"
        limited = "true" if stateful else "false"
        force_range = ".1 .25" if name == "c3-disabled" else "-.7 .7"
        actors = f'''<motor name="direct" joint="hinge" gear="-.5"/>
        <general name="filter-slide" joint="slide" gear="-2" dyntype="{dtype}" dynprm=".004"
        gainprm="1.2" biastype="affine" biasprm=".01 -.3 -.04" actearly="{early}"
        ctrllimited="true" ctrlrange="-.4 .4" actlimited="{limited}" actrange="-.2 .3"
        forcelimited="true" forcerange="-.7 .7"/>
        <general name="filter-hinge" joint="hinge" gear="1.5" group="1" dyntype="{dtype}"
        dynprm=".002" gainprm=".8" forcelimited="true" forcerange="{force_range}"/>
        <general name="shared-hinge" joint="hinge" gear="-.25" dyntype="{dtype}" dynprm=".008" gainprm=".6"/>'''
    xml = HEADER + "<worldbody>" + world + "</worldbody>"
    if actors:
        xml += "<actuator>" + actors + "</actuator>"
    if name == "c3-disabled":
        xml = xml.replace('integrator="Euler"', 'integrator="Euler" actuatorgroupdisable="1"')
    return xml + "</mujoco>"


def capacity_source(family, overflow):
    count = (17 if overflow else 16) if family == "pairs" else (5 if overflow else 4)
    shape = "sphere" if family == "pairs" else "box"
    size = ".01" if shape == "sphere" else ".01 .01 .01"
    geoms = "".join(
        f'<geom name="site{i}" type="{shape}" pos="{i * 0.03} 0 0" size="{size}" condim="1"/>'
        for i in range(count)
    )
    return (
        HEADER
        + f"""<worldbody><geom name="ground" type="plane" size="1 1 .1" condim="1"/>
    <body pos="0 0 1"><joint name="ball" type="ball" {JOINT}/>{INERTIA}{geoms}</body></worldbody></mujoco>"""
    )


def negative_source(name):
    xml = source("c3-early")
    if name == "ball-spring":
        return xml.replace('name="ball" type="ball"', 'name="ball" type="ball" stiffness="1"')
    if name == "ball-limit":
        return xml.replace(
            'name="ball" type="ball"', 'name="ball" type="ball" limited="true" range="0 .5"'
        )
    if name == "scalar-limit":
        return xml.replace(
            'name="hinge" type="hinge"', 'name="hinge" type="hinge" limited="true" range="-1 1"'
        )
    if name == "dry":
        return xml.replace('name="ball" type="ball"', 'name="ball" type="ball" frictionloss=".1"')
    if name == "ball-motor":
        return xml.replace(
            '<motor name="direct" joint="hinge" gear="-.5"/>',
            '<motor name="direct" joint="ball" gear="1 2 3 0 0 0"/>',
        )
    if name == "free-ball":
        return (
            xml.replace(
                '<joint name="hinge" type="hinge" axis="0 0 1" ' + JOINT + " " + SCALAR + "/>",
                '<freejoint name="root"/>',
            )
            .replace("<actuator>", "<!--")
            .replace("</actuator>", "-->")
        )
    if name == "same-body":
        return xml.replace(
            '<joint name="ball" type="ball"',
            '<joint name="extra-slide" type="slide" axis="1 0 0"/><joint name="ball" type="ball"',
        )
    if name == "elliptic":
        return xml.replace('cone="pyramidal"', 'cone="elliptic"')
    if name in ("condim4", "condim6"):
        return xml.replace('condim="3"', 'condim="' + name[-1] + '"')
    if name in ("pair-overflow", "slot-overflow"):
        return capacity_source("pairs" if name == "pair-overflow" else "slots", True)
    if name == "explicit-pair":
        return xml.replace(
            "</mujoco>", '<contact><pair geom1="ground" geom2="tip-sphere"/></contact></mujoco>'
        )
    if name == "exclude":
        return xml.replace(
            "</mujoco>", '<contact><exclude body1="base" body2="tip"/></contact></mujoco>'
        )
    raise ValueError(name)


def initial(mj, model, name, slot):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    q, v = model.qpos0.copy(), np.zeros(model.nv)
    for joint in range(model.njnt):
        qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        if int(model.jnt_type[joint]) == 1:
            if name.startswith("c1"):
                q[qa : qa + 4] = (2 / 3, 1 / 3, 2 / 3, 0)
                v[dof : dof + 3] = np.array([0.02, -0.03, 0.01]) * (1 + 0.1 * slot)
            else:
                q[qa : qa + 4] = (
                    math.cos(0.125),
                    math.sin(0.125) if name == "c4-two-ball" else 0,
                    0,
                    0 if name == "c4-two-ball" else math.sin(0.125),
                )
                v[dof : dof + 3] = (
                    np.array([0.001, -0.001, 0.001])
                    if name == "c2-birth"
                    else np.array([0.02, -0.03, 0.04]) * (-1 if slot % 2 else 1)
                )
        elif int(model.jnt_type[joint]) == 3:
            q[qa] += 0.11
            v[dof] = 0.01 if name == "c2-birth" else (0.1 if slot % 2 else -0.1)
        elif name == "c4-two-ball":
            v[dof] = 0.02 if joint == 1 else -0.02
        else:
            q[qa] += 2**-16 if name == "c2-birth" else -0.001
            v[dof] = -1 if name == "c2-birth" else -0.02
    return MuJoCoInitialState(
        q, v.astype(np.float32), act=np.full(model.na, 0.34 + 0.01 * slot), time=0.125
    )


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
        analytic=analytic(model, entry, packet),
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
    if model.na:
        a = row.actuation
        exact(a.input_act.astype(np.float64), before.act, "raw entry activation")
        exact(a.output_act.astype(np.float64), after.act, "raw accepted activation")
        for field, target in expected["analytic"].items():
            np.testing.assert_allclose(
                getattr(a, field),
                target,
                **(ACT if field in ("output_act", "activation_derivative") else FORCE),
            )
        np.testing.assert_allclose(a.scalar_force, expected["actuator_force"], **FORCE)
        np.testing.assert_allclose(a.activation_derivative, expected["act_dot"], **ACT)
    for joint in range(model.njnt):
        if int(model.jnt_type[joint]) == 1:
            dof = int(model.jnt_dofadr[joint])
            np.testing.assert_array_equal(row.qfrc_actuator[dof : dof + 3], np.zeros(3))
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
    if row.constraints:
        require(len(row.constraints) == len(native_rows), "complete EFC count")
        ids, values, jac = (np.asarray(raw[k]) for k in ("efc_ids", "efc_data", "efc_J"))
        for i, (constraint, native_index) in enumerate(
            zip(row.constraints, native_rows, strict=True)
        ):
            np.testing.assert_allclose(
                constraint.jacobian, expected["efc_J"][native_index], rtol=2e-4, atol=5e-5
            )
            for field, target in (
                ("regularization", "efc_R"),
                ("reference_acceleration", "efc_aref"),
                ("force", "efc_force"),
            ):
                np.testing.assert_allclose(
                    getattr(constraint, field),
                    expected[target][native_index],
                    **(CONTACT_FORCE if field == "force" else dict(rtol=5e-4, atol=4e-4)),
                )
            require(
                constraint.constraint_type == int(expected["efc_type"][native_index]),
                "native row type",
            )
            require(
                native[constraint.source_id]["native_index"]
                == int(expected["efc_id"][native_index]),
                "native row owner",
            )
            np.testing.assert_array_equal(
                ids[i],
                [
                    constraint.constraint_type,
                    constraint.source_id,
                    constraint.side_or_edge,
                    constraint.geometry_slot,
                    constraint.dof,
                ],
            )
            np.testing.assert_array_equal(jac[i, : model.nv], constraint.jacobian)
            require(
                values[i, 9] == constraint.force
                and values[i, 3] == constraint.reference_acceleration
                and values[i, 4] == constraint.regularization,
                "raw EFC publication",
            )
        np.testing.assert_array_equal(
            raw["counts"],
            [len(row.contacts), sum(c.active for c in row.contacts), len(row.constraints)],
        )
        np.testing.assert_array_equal(
            np.asarray(raw["contact_wrench"])[: len(row.contacts)],
            np.asarray([c.wrench_contact_frame for c in row.contacts]),
        )
    elif "device_rows" in raw:
        rows = np.asarray(raw["device_rows"])
        require(np.count_nonzero(rows[:, 0]) == len(row.contacts), "all normal rows")
        for c in row.contacts:
            slots = np.asarray(raw["slot_ids"])
            found = np.flatnonzero(
                (slots[:, 1] == c.feature_index)
                & (slots[:, 2] == c.geometry_ids[0])
                & (slots[:, 3] == c.geometry_ids[1])
            )
            require(len(found) == 1, "normal slot")
            j = int(found[0])
            np.testing.assert_array_equal(np.asarray(raw["normal_jacobian"])[j], c.jacobian)
            require(
                rows[j, 3] == c.normal_force and bool(rows[j, 1]) == c.active, "raw normal force"
            )
    diag = raw["solver_diagnostics"]
    require(
        diag["converged"]
        and diag["normalized_kkt"] <= float(np.float32(max(model.opt.tolerance, 2e-5))),
        "original KKT gate",
    )
    require(
        diag.get("iterations", diag.get("face_solves")) <= diag["budget"] == model.opt.iterations,
        "original source budget",
    )
    for target in (expected["native_successor"], expected["continuous"]):
        np.testing.assert_allclose(after.q, target["q"], rtol=0, atol=Q_ATOL)
        np.testing.assert_allclose(after.v, target["v"], rtol=0, atol=V_ATOL)
        np.testing.assert_allclose(after.act, target["act"], **ACT)
        require(abs(after.time - target["time"]) <= TIME_ATOL, "native clock")
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


def domain(name, na):
    family = "pyramidal" if name.startswith("c3") or name == "c4-two-ball" else "normal"
    return "fixed-ball-scalar-" + family + "-contact" + ("-filterexact" if na else "") + "-v1"


def continuous(name, mode, journal):
    mj, model, compiled, starts, runtime = fixture(name, mode)
    try:
        require(runtime.capabilities["domain"] == domain(name, model.na), "precise domain")
        require(
            not runtime.capabilities["joint_limits_supported"]
            and not runtime.capabilities["dry_friction_supported"],
            "no ball limit/dry promise",
        )
        require(
            tuple(runtime.capabilities["contact_dimensions"])
            == ((1, 3) if name.startswith("c3") or name == "c4-two-ball" else (1,)),
            "dimensions",
        )
        if name == "c4-two-ball":
            require((model.nq, model.nv) == (10, 8), "native boundary")
        if name == "c2-birth" or name.startswith("c3"):
            require((model.nq, model.nv) == (6, 5), "native H/B/S addresses")
            require(
                int(model.jnt_qposadr[-1]) != int(model.jnt_dofadr[-1]),
                "actual scalar address mismatch",
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
            force_gate=CONTACT_FORCE,
            activation_gate=ACT,
            steps=24,
            h=H,
        )
        active = []
        first_contacts = None
        both_sides_loaded = False
        for tick in range(24):
            packets = {key: control(model, tick, i) for i, key in enumerate(runtime.world_keys)}
            state = advance(runtime, mj, model, packets, refs, journal, f"{name}-{tick + 1}")
            contacts = state[KEYS[0]].last_step.contacts
            active.append(any(c.active and c.normal_force > 0 for c in contacts))
            both_sides_loaded = both_sides_loaded or any(
                np.linalg.norm(c.wrench_on_body1_world) > 0
                and np.linalg.norm(c.wrench_on_body2_world) > 0
                for c in contacts
            )
            if tick == 0:
                first_contacts = contacts
                if name == "c2-birth":
                    require(len(contacts) == 0, "C2 first input remains separated")
                    require(2**-16 > 0, "predeclared positive gap")
                    mass = state[KEYS[0]].last_step.dynamics.mass_matrix
                    require(
                        np.max(np.abs(mass - np.diag(np.diag(mass)))) > 1e-4,
                        "full ball/scalar cross coupling",
                    )
                if name == "c1-loaded":
                    require(
                        any(c.active and c.normal_force > 0 for c in contacts), "C1 initial load"
                    )
                    require(
                        all(abs(x) > 1e-4 for x in contacts[0].jacobian), "all three ball J columns"
                    )
                    require(
                        np.linalg.norm(contacts[0].wrench_on_body2_world[3:]) > 0,
                        "nonzero body moment",
                    )
            if tick == 1 and name == "c2-birth":
                require(
                    any(c.active and c.normal_force > 0 for c in contacts),
                    "C2 actual later entry birth",
                )
        if name == "c1-inactive":
            require(not any(active), "inactive declaration zero force throughout")
        else:
            require(any(active), "real loaded contact required")
        if name == "c4-two-ball":
            for qa in (0, 5):
                require(
                    np.max(np.abs(state[KEYS[0]].q[qa : qa + 5] - start[KEYS[0]].q[qa : qa + 5]))
                    > 0,
                    "each dynamic branch moves",
                )
            require(both_sides_loaded, "both-side load recorded")
        journal.emit(
            "continuous-complete",
            name=name,
            active=active,
            first_contacts=first_contacts,
            state=state,
            counters=dict(runtime.counters),
        )
        return dict(name=name, accepted_world_steps=48, steps=24)
    finally:
        runtime.close()


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
            (Path(output_path) / "ball-contact-checkpoint.bin").write_bytes(raw)
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
                "BALL-C lifecycle cleanup", ([primary] if primary else []) + failures
            )


def run(mode, output_path, journal, installed_root):
    result = continuous("c3-early", mode, journal)
    life = lifecycle("c3-early", mode, journal, KEYS, output_path)
    paths = provenance(installed_root)
    require(
        "phymeo.backends.mujoco_tyx._constraint_device" in paths["loaded"],
        "actual contact backend provenance",
    )
    return dict(
        continuous=result,
        lifecycle=life,
        accepted_world_steps=result["accepted_world_steps"] + life["accepted_world_steps"],
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
    journal = Journal(output / "ball-contacts.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="installed fixed-root BALL+R/P; mixed NONE/FILTEREXACT scalar actors and primitive normal/pyramidal contacts; no limits",
        excluded="ball motors/limits/springs, free-root composition, G1, training",
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
        raise BaseExceptionGroup("BALL-C primary/finalization failures", members)
    if primary is not None:
        raise primary.with_traceback(primary_traceback)


if __name__ == "__main__":
    main()
