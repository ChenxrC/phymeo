"""Installed heterogeneous robot transaction consumer; no checkout imports.

Three qualified sources, six sparse worlds, full native field/continuous oracles.
This is a bounded functional consumer, not a fused-kernel or performance claim.
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
import xml.etree.ElementTree as ET
from collections.abc import Mapping
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
    """Independent MuJoCo 3.3.7 filter law and complete native JOINT moments."""
    offsets, columns, coefficients, ids = [0], [], [], []
    clamped, raw, scalar, force_inputs = [], [], [], []
    activation_ids, parameters = [], []
    generalized = np.zeros(model.nv)
    dense = np.zeros((model.nu, model.nv))
    output = np.asarray(entry.act, dtype=np.float64).copy()
    derivative = np.zeros(model.na)
    integrated = np.zeros(model.na)
    require(model.na > 0, "R1 stateful domain")
    for actor in range(model.nu):
        joint = int(model.actuator_trnid[actor, 0])
        kind = int(model.jnt_type[joint])
        qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        width = 3 if kind == 1 else 1
        ids.append((actor, joint, kind, qa, dof, width))
        adr = int(model.actuator_actadr[actor])
        early = bool(model.actuator_actearly[actor])
        disabled = bool(model.opt.disableactuator & (1 << int(model.actuator_group[actor])))
        masks = tuple(
            int(x[actor])
            for x in (
                model.actuator_ctrllimited,
                model.actuator_actlimited,
                model.actuator_forcelimited,
            )
        )
        activation_ids.append((qa, dof, joint, adr, int(early), int(disabled), *masks))
        bias = model.actuator_biasprm[actor, :3] if model.actuator_biastype[actor] else np.zeros(3)
        require(width == 1 or not model.actuator_biastype[actor], "BALL BIAS_NONE")
        tau = max(1e-15, float(model.actuator_dynprm[actor, 0])) if adr >= 0 else 0.0
        ctrlrange = model.actuator_ctrlrange[actor] if masks[0] else (0.0, 0.0)
        actrange = model.actuator_actrange[actor] if masks[1] else (0.0, 0.0)
        forcerange = model.actuator_forcerange[actor] if masks[2] else (0.0, 0.0)
        parameters.append(
            (
                model.actuator_gear[actor, 0],
                model.actuator_gainprm[actor, 0],
                *bias,
                *ctrlrange,
                *forcerange,
                tau,
                *actrange,
            )
        )
        u = float(np.float32(packet.ctrl[actor]))
        if masks[0]:
            u = float(np.clip(u, *ctrlrange))
        force_input = u
        if adr >= 0:
            x = float(entry.act[adr])
            derivative[adr] = (u - x) / tau
            integrated[adr] = 0.0 if disabled else derivative[adr]
            candidate = x if disabled else x - (u - x) * math.expm1(-H / tau)
            if masks[1]:
                candidate = float(np.clip(candidate, *actrange))
            output[adr] = candidate
            force_input = candidate if early else x
        value = 0.0
        if not disabled:
            value = float(model.actuator_gainprm[actor, 0]) * force_input
            if model.actuator_biastype[actor]:
                gear = model.actuator_gear[actor, 0]
                value += bias[0] + bias[1] * gear * entry.q[qa] + bias[2] * gear * entry.v[dof]
        force = float(np.clip(value, *forcerange)) if masks[2] else value
        clamped.append(u)
        force_inputs.append(force_input)
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
        activation_ids=np.asarray(activation_ids, np.int32),
        parameters=np.asarray(parameters, np.float32),
        input_act=np.asarray(entry.act, np.float32),
        output_act=output,
        activation_derivative=derivative,
        integration_derivative=integrated,
        force_input=np.asarray(force_inputs),
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
    require(tr.schema == "native-joint-motor-transmission-v1", "complete transmission schema")
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


def check_activation(row, entry, after, native):
    from phymeo.backends.mujoco_joint_transmission import MuJoCoTyxJointActivationStepReceipt

    require(type(row) is MuJoCoTyxJointActivationStepReceipt, "typed combined publication")
    act, expected = row.actuation, native["analytic"]
    require(act.schema == "native-joint-filterexact-actuation-v1", "combined activation schema")
    np.testing.assert_array_equal(act.ids, expected["activation_ids"])
    np.testing.assert_array_equal(act.parameters, expected["parameters"])
    exact(act.input_act.astype(np.float64), entry.act, "actual entry activation")
    exact(act.output_act.astype(np.float64), after.act, "actual accepted activation")
    for name in ("output_act", "activation_derivative", "integration_derivative", "force_input"):
        np.testing.assert_allclose(getattr(act, name), expected[name], **ACT)
    np.testing.assert_allclose(act.activation_derivative, native["act_dot"], **ACT)
    for name in ("clamped_control", "raw_force", "scalar_force", "generalized_force"):
        exact(getattr(act, name), getattr(row.transmission, name), "same device attempt " + name)
        np.testing.assert_allclose(getattr(act, name), expected[name], **FORCE)
    for field in fields(act):
        value = getattr(act, field.name)
        if isinstance(value, np.ndarray):
            require(not value.flags.writeable, "immutable activation " + field.name)
            try:
                value.setflags(write=True)
            except ValueError:
                pass
            else:
                raise AssertionError("activation permits mutable reenable " + field.name)


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


def motor(name, target, gear, extra="", gain="1"):
    return (
        f'<general name="{name}" joint="{target}" gear="{gear}" '
        f'dyntype="none" gaintype="fixed" gainprm="{gain}" {extra}/>'
    )


def _mechanical_source(name):
    """Predeclared original BALL-C geometry plus stateless direct JOINT motors."""
    require(
        name
        in (
            "mc1-normal",
            "mc2-friction",
            "mc3-mixed",
            "mc4-shared",
            "mc5-two-ball",
            "mc6-disabled",
        ),
        "declared case",
    )
    dim = 1 if name == "mc1-normal" else 3
    if name in ("mc1-normal", "mc2-friction", "mc4-shared"):
        world = f'''<geom name="ground" type="plane" size="1 1 .1" condim="{dim}"/>
        <body name="rotor" pos="0 0 .13577777777777778"><joint name="ball" type="ball" {JOINT}/>
        {INERTIA}<geom name="tip" type="sphere" pos=".08 .03 -.18" size=".1" condim="{dim}"/></body>'''
        actors = motor("ball-drive", "ball", "1 -2 .5")
        if name == "mc4-shared":
            actors += motor("counter", "ball", "-.25 .5 -.125", gain=".6")
            actors += motor("zero", "ball", "0 0 0", gain=".8")
    elif name == "mc5-two-ball":
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
        actors = motor("ball-a-drive", "ball-a", "1 -2 .5")
        actors += motor("ball-b-drive", "ball-b", "-.3 .2 .4")
        actors += motor("slide-a-drive", "slide-a", "-.5")
        actors += motor("slide-b-drive", "slide-b", ".5")
    else:
        world = f"""<geom name="ground" type="box" pos="0 0 -.05" size=".8 .8 .05" condim="3"/>
        <body name="base" pos="0 0 .4"><joint name="hinge" type="hinge" axis="0 0 1" {JOINT} {SCALAR}/>
        {INERTIA}<body name="middle" pos="0 0 -.2"><joint name="ball" type="ball" {JOINT}/>
        {INERTIA}<body name="tip" pos="0 0 -.1"><joint name="slide" type="slide" axis="0 0 1" {JOINT} {SCALAR}/>
        {INERTIA}<geom name="tip-sphere" type="sphere" pos=".06 0 0" size=".1" condim="3"/>
        </body></body></body>"""
        extra = (
            'group="1" forcelimited="true" forcerange=".1 .25"' if name == "mc6-disabled" else ""
        )
        actors = motor("ball-drive", "ball", "1 -2 .5", extra)
        actors += motor("hinge-drive", "hinge", "-.5")
        actors += motor(
            "slide-drive",
            "slide",
            "-2",
            'biastype="affine" biasprm=".01 -.3 -.04" ctrllimited="true" '
            'ctrlrange="-.4 .4" forcelimited="true" forcerange="-.7 .7"',
            gain="1.2",
        )
    xml = (
        HEADER + "<worldbody>" + world + "</worldbody><actuator>" + actors + "</actuator></mujoco>"
    )
    if name == "mc6-disabled":
        xml = xml.replace('integrator="Euler"', 'integrator="Euler" actuatorgroupdisable="1"')
    return xml


BASE_CASES = {
    "r1-late": "mc1-normal",
    "r2-early": "mc1-normal",
    "r3-normal": "mc1-normal",
    "r4-pyramid": "mc2-friction",
    "r5-mixed": "mc3-mixed",
    "r6-shared": "mc4-shared",
    "r7-two-ball": "mc5-two-ball",
    "r8-disabled": "mc6-disabled",
}


def source(name):
    """Only declared actuator laws change; the accepted mechanical fixtures remain."""
    root = ET.fromstring(_mechanical_source(BASE_CASES[name]))
    if name in ("r1-late", "r2-early"):
        world = root.find("worldbody")
        world.remove(world.find("geom"))
    selected = {
        "r1-late": {"ball-drive": False},
        "r2-early": {"ball-drive": True},
        "r3-normal": {"ball-drive": False},
        "r4-pyramid": {"ball-drive": True},
        "r5-mixed": {"ball-drive": True, "slide-drive": False},
        "r6-shared": {"ball-drive": False, "zero": True},
        "r7-two-ball": {"ball-a-drive": False, "ball-b-drive": True, "slide-b-drive": False},
        "r8-disabled": {"ball-drive": True, "slide-drive": False},
    }[name]
    for actor in root.find("actuator"):
        label = actor.get("name")
        if label not in selected:
            continue
        actor.set("dyntype", "filterexact")
        actor.set("dynprm", ".007" if label == "zero" else ".004")
        actor.set("actearly", "true" if selected[label] else "false")
        actor.set("actlimited", "true")
        actor.set("actrange", "-.3 .3")
        if actor.get("ctrllimited") is None:
            actor.set("ctrllimited", "true")
            actor.set("ctrlrange", "-.45 .45")
    return ET.tostring(root, encoding="unicode")


def initial(mj, model, name, slot):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    name = BASE_CASES[name]
    q, v = model.qpos0.copy(), np.zeros(model.nv)
    for joint in range(model.njnt):
        qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        if int(model.jnt_type[joint]) == 1:
            if name in ("mc1-normal", "mc2-friction", "mc4-shared"):
                q[qa : qa + 4] = (2 / 3, 1 / 3, 2 / 3, 0)
                v[dof : dof + 3] = np.array([0.02, -0.03, 0.01]) * (1 + 0.1 * slot)
            else:
                q[qa : qa + 4] = (
                    math.cos(0.125),
                    math.sin(0.125) if name == "mc5-two-ball" else 0,
                    0,
                    0 if name == "mc5-two-ball" else math.sin(0.125),
                )
                v[dof : dof + 3] = np.array([0.02, -0.03, 0.04]) * (-1 if slot % 2 else 1)
        elif int(model.jnt_type[joint]) == 3:
            q[qa] += 0.11
            v[dof] = 0.1 if slot % 2 else -0.1
        elif name == "mc5-two-ball":
            v[dof] = 0.02 if joint == 1 else -0.02
        else:
            q[qa] -= 0.001
            v[dof] = -0.02
    activation = np.asarray(
        [
            (-1 if address % 2 else 1) * (0.34 + 0.01 * slot + 0.005 * address)
            for address in range(model.na)
        ],
        np.float32,
    )
    return MuJoCoInitialState(q, v.astype(np.float32), act=activation, time=0.125)


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


def no_contact(row):
    return row.numerical_solver not in ("normal-dual-active-set-v1", "pyramidal-box-active-set-v1")


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
        np.testing.assert_allclose(
            getattr(row, field),
            expected[field],
            **(dict(rtol=5e-4, atol=8e-4) if no_contact(row) else ACC),
        )
    np.testing.assert_allclose(row.qfrc_actuator, expected["qfrc_actuator"], **FORCE)
    np.testing.assert_allclose(row.qfrc_constraint, expected["qfrc_constraint"], **CONTACT_FORCE)
    np.testing.assert_allclose(
        row.qfrc_actuator, expected["analytic"]["generalized_force"], **FORCE
    )
    require(model.na > 0 and after.act.shape == (model.na,), "complete activation state")
    check_activation(row, before, after, expected)
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
    if row.numerical_solver == "pyramidal-box-active-set-v1":
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
            np.asarray([c.wrench_contact_frame for c in row.contacts]).reshape(-1, 6),
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
    if not no_contact(row):
        diag = raw["solver_diagnostics"]
        require(
            diag["converged"]
            and diag["normalized_kkt"] <= float(np.float32(max(model.opt.tolerance, 2e-5))),
            "original KKT gate",
        )
        require(
            diag.get("iterations", diag.get("face_solves"))
            <= diag["budget"]
            == model.opt.iterations,
            "original source budget",
        )
    else:
        require(not row.contacts and not row.constraints, "no hidden contact rows")
    for target in (expected["native_successor"], expected["continuous"]):
        np.testing.assert_allclose(
            after.q, target["q"], rtol=0, atol=5e-5 if no_contact(row) else Q_ATOL
        )
        np.testing.assert_allclose(
            after.v, target["v"], rtol=0, atol=3e-4 if no_contact(row) else V_ATOL
        )
        np.testing.assert_allclose(after.act, target["act"], **ACT)
        require(abs(after.time - target["time"]) <= TIME_ATOL, "native clock")
    np.testing.assert_allclose(
        after.v, before.v + float(np.float32(H)) * row.integration_acceleration, rtol=0, atol=3e-6
    )


H = 0.001
free_KEYS = (3, 29)
free_Q_ATOL, free_V_ATOL, free_TIME_ATOL = 1e-4, 1e-3, 2e-6
free_FORCE = dict(rtol=3e-4, atol=2e-4)
free_TOL = {
    "body_position": (3e-5, 4e-5),
    "body_rotation": (3e-5, 4e-5),
    "body_com": (3e-5, 4e-5),
    "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (3e-4, 5e-5),
    "bias": (4e-4, 4e-4),
    "passive": (4e-4, 4e-4),
    "applied": (4e-4, 4e-4),
    "smooth_acceleration": (2e-3, 4e-3),
    "integration_acceleration": (2e-3, 4e-3),
}


def free_body(name, kind="ball", child="", pos=".12 -.04 .23"):
    joint = f'<joint name="{name}" type="{kind}" pos=".02 -.01 .03" damping=".04" armature=".003"/>'
    if kind == "slide":
        joint = joint.replace('type="slide"', 'type="slide" axis=".3 -.4 .5"')
    elif kind == "hinge":
        joint = joint.replace('type="hinge"', 'type="hinge" axis=".2 .7 -.3"')
    return (
        f'<body name="{name}-body" pos="{pos}" quat=".9238795325112867 0 .3826834323650898 0">'
        '<inertial pos=".03 -.02 .04" quat=".9659258262890683 .2588190451025207 0 0" '
        'mass=".7" diaginertia=".035 .045 .055"/>'
        + joint
        + '<geom type="sphere" size=".035" contype="0" conaffinity="0"/>'
        + child
        + "</body>"
    )


def free_motor(name, target, gear, extra="", gain="1"):
    return (
        f'<general name="{name}" joint="{target}" gear="{gear}" dyntype="none" '
        f'gaintype="fixed" gainprm="{gain}" {extra}/>'
    )


def free_source(case):
    actors = free_motor("drive", "ball", "1 -2 .5", gain=".8")
    if case == "mixed":
        child = free_body("ball", child=free_body("hinge", "hinge", free_body("slide", "slide")))
        actors += free_motor("hinge-drive", "hinge", "-.5")
        actors += free_motor(
            "slide-drive", "slide", "-2", 'biastype="affine" biasprm=".01 -.3 -.04"'
        )
    elif case == "branches":
        child = free_body("ball") + free_body("other", pos="-.18 .07 .16")
        actors += free_motor("other-drive", "other", "-.3 .4 .2")
    else:
        raise ValueError("only the two declared installed cases are admitted")
    return f"""<mujoco model="free-ball-{case}">
<compiler angle="radian"/>
<option timestep=".001" integrator="Euler" gravity="0 0 -1.2"/>
<worldbody>
<body name="marker" pos="2 -1 .4"><geom size=".02" contype="0" conaffinity="0"/></body>
<body name="root" pos=".2 -.1 .7" quat=".9238795325112867 .3826834323650898 0 0">
<freejoint name="free"/>
<inertial pos=".04 -.03 .02" quat=".9659258262890683 0 .2588190451025207 0"
 mass="2.6" diaginertia=".13 .19 .23"/>
<geom type="sphere" size=".06" contype="0" conaffinity="0"/>
{child}</body></worldbody><actuator>{actors}</actuator></mujoco>"""


def free_initial(mj, model, slot=0, sign=False):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    q = model.qpos0.copy()
    for joint, kind in enumerate(model.jnt_type):
        qa = int(model.jnt_qposadr[joint])
        if kind in (mj.mjtJoint.mjJNT_FREE, mj.mjtJoint.mjJNT_BALL):
            adr = qa + (3 if kind == mj.mjtJoint.mjJNT_FREE else 0)
            quat = np.array([1.0, 0.13 + 0.02 * joint, -0.09, 0.07])
            q[adr : adr + 4] = quat / np.linalg.norm(quat)
            if sign and slot:
                q[adr : adr + 4] *= -1
        else:
            q[qa] = 0.03 * (joint + 1)
    if not sign:
        q[:3] += slot * np.array([0.04, -0.02, 0.03])
    velocity = np.linspace(-0.13, 0.17, model.nv) * (1 if sign else 1 + 0.1 * slot)
    return MuJoCoInitialState(q, velocity, time=0.125)


def free_control(model, tick, slot=0):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    phase = 1 if tick < 8 else (-1 if tick < 16 else 0)
    ctrl = phase * np.array([(0.18 + 0.01 * a) * (-1 if a % 2 else 1) for a in range(model.nu)])
    qforce = np.zeros(model.nv)
    qforce[:6] = phase * np.array([0.3, -0.2, 0.15, 0.08, -0.06, 0.04])
    qforce[6:] = phase * np.linspace(0.025, 0.085, model.nv - 6)
    wrench = np.zeros((model.nbody, 6))
    wrench[-1] = phase * (1 + 0.1 * slot) * np.array([0.35, -0.25, 0.4, 0.025, -0.035, 0.015])
    return MuJoCoControl(ctrl, qfrc_applied=qforce, xfrc_applied=wrench)


def free_native_state(data):
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
    )


def free_native_fields(mj, model, entry, packet):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = entry.q, entry.v, entry.time
    set_control(data, packet)
    mj.mj_forward(model, data)
    require(data.ncon == 0 and data.nefc == 0, "FREE+BALL has no contact/constraint")
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_applied.copy() + data.qfrc_actuator
    for free_body in range(model.nbody):
        mj.mj_jac(
            model, data, jac[free_body, :3], jac[free_body, 3:], data.xipos[free_body], free_body
        )
        applied += jac[free_body].T @ packet.xfrc_applied[free_body]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    smooth = np.linalg.solve(mass, rhs)
    np.testing.assert_allclose(smooth, data.qacc_smooth, rtol=2e-10, atol=2e-9)
    expected = dict(
        input_q=np.array(entry.q),
        input_v=np.array(entry.v),
        time=entry.time,
        control=packet,
        native_input=free_native_state(data),
        body_position=data.xpos.copy(),
        body_rotation=data.xmat.reshape(model.nbody, 3, 3).copy(),
        body_com=data.xipos.copy(),
        jacobian_com=jac,
        mass_matrix=mass,
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=applied,
        smooth_acceleration=smooth,
        integration_acceleration=np.linalg.solve(mass + H * np.diag(model.dof_damping), rhs),
        qfrc_actuator=data.qfrc_actuator.copy(),
        qfrc_constraint=data.qfrc_constraint.copy(),
        scalar_force=data.actuator_force.copy(),
        actuator_velocity=data.actuator_velocity.copy(),
        dense_moment=free_dense_native_moment(mj, model, data),
        transmission=free_transmission_oracle(mj, model, entry, packet),
    )
    mj.mj_step(model, data)  # Independent same-input native successor, outside TYX guard.
    expected["native_successor"] = free_native_state(data)
    return expected


def free_transmission_oracle(mj, model, state, packet):
    """Independent native-address table and scalar-force law; no product helper."""
    offsets, columns, coefficients, ids = [0], [], [], []
    clamped, raw, scalar = [], [], []
    generalized = np.zeros(model.nv)
    moment = np.zeros((model.nu, model.nv))
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        kind = int(model.jnt_type[j])
        qa, dof = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        width = 3 if kind == int(mj.mjtJoint.mjJNT_BALL) else 1
        ids.append((a, j, kind, qa, dof, width))
        u = float(packet.ctrl[a])
        if model.actuator_ctrllimited[a]:
            u = float(np.clip(u, *model.actuator_ctrlrange[a]))
        value = 0.0
        if not model.opt.disableactuator & (1 << int(model.actuator_group[a])):
            value = float(model.actuator_gainprm[a, 0]) * u
            if model.actuator_biastype[a] == mj.mjtBias.mjBIAS_AFFINE:
                assert width == 1
                b = model.actuator_biasprm[a]
                gear = model.actuator_gear[a, 0]
                value += b[0] + b[1] * gear * state.q[qa] + b[2] * gear * state.v[dof]
        force = (
            float(np.clip(value, *model.actuator_forcerange[a]))
            if model.actuator_forcelimited[a]
            else value
        )
        clamped.append(u)
        raw.append(value)
        scalar.append(force)
        for axis in range(width):
            col, coefficient = dof + axis, float(model.actuator_gear[a, axis])
            columns.append(col)
            coefficients.append(coefficient)
            moment[a, col] = coefficient
            generalized[col] += coefficient * force
        offsets.append(len(columns))
    return dict(
        ids=np.asarray(ids, np.int32),
        gear=model.actuator_gear.copy(),
        row_offsets=np.asarray(offsets, np.int32),
        dof_indices=np.asarray(columns, np.int32),
        moment=np.asarray(coefficients, np.float32),
        dense_moment=moment,
        clamped_control=np.asarray(clamped),
        raw_force=np.asarray(raw),
        scalar_force=np.asarray(scalar),
        generalized_force=generalized,
    )


def free_dense_native_moment(mj, model, data):
    dense = np.zeros((model.nu, model.nv))
    for a in range(model.nu):
        start, count = int(data.moment_rowadr[a]), int(data.moment_rownnz[a])
        dense[a, data.moment_colind[start : start + count]] = data.actuator_moment[
            start : start + count
        ]
    return dense


def free_check_transmission(row, entry, native, analytic):
    tr = row.transmission
    require(tr.schema == "native-free-ball-joint-motor-transmission-v1", "transmission schema")
    for name in ("ids", "gear", "row_offsets", "dof_indices", "moment"):
        np.testing.assert_array_equal(getattr(tr, name), analytic[name])
    for name in ("clamped_control", "raw_force", "scalar_force", "generalized_force"):
        np.testing.assert_allclose(getattr(tr, name), analytic[name], **free_FORCE)
    np.testing.assert_allclose(tr.scalar_force, native["scalar_force"], **free_FORCE)
    # Device forces are FP32; the public step receipt is their exact FP64 lift.
    require(tr.generalized_force.dtype == np.dtype(np.float32), "device force dtype")
    require(row.qfrc_actuator.dtype == np.dtype(np.float64), "public force dtype")
    exact(
        tr.generalized_force.astype(np.float64),
        row.qfrc_actuator,
        "same-stage generalized actuator force",
    )
    exact(tr.input_control, row.control.ctrl, "original control binding")
    np.testing.assert_allclose(analytic["dense_moment"], native["dense_moment"], rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        analytic["dense_moment"] @ entry.v, native["actuator_velocity"], **free_FORCE
    )
    np.testing.assert_allclose(
        tr.generalized_force @ entry.v,
        tr.scalar_force @ (analytic["dense_moment"] @ entry.v),
        **free_FORCE,
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
        require(not value.flags.writeable, "immutable " + name)
        rejected = False
        try:
            value.setflags(write=True)
        except ValueError:
            rejected = True
        require(rejected, "bytes-backed " + name)


def scalar_source(case="hinge", *, early=False, tau=0.004, disabled=False):
    kind = "slide" if case == "slide" else "hinge"
    child = ""
    if case == "coupled":
        child = """<body name="tip" pos=".2 .03 .04" quat=".9238795325 0 0 .3826834324">
          <joint name="slide" type="slide" axis="1 0 0" armature=".05" damping=".11"/>
          <inertial pos=".03 -.02 .01" mass=".8" diaginertia=".02 .03 .04"/>
          <geom size=".03" contype="0" conaffinity="0"/></body>"""
    extra = ' actuatorgroupdisable="1"' if disabled else ""
    early_text = "true" if early else "false"
    if case == "coupled":
        actors = f'''<motor name="direct" joint="joint" gear="-.5"/>
          <general name="filter-slide" joint="slide" gear="-2" dyntype="filterexact"
           dynprm="{tau}" gainprm="1.2" biastype="affine" biasprm=".01 -.3 -.04"
           actearly="{early_text}"/>
          <general name="filter-hinge" joint="joint" gear="1.5" dyntype="filterexact"
           dynprm=".002" gainprm=".8"/>
          <general name="second-hinge" joint="joint" gear="-.25" dyntype="filterexact"
           dynprm=".008" gainprm=".6"/>'''
    else:
        force_range = ".1 .25" if disabled else "-.7 .7"
        gear = -2 if case == "slide" else 1.5
        actors = f'''<general name="filter" joint="joint" gear="{gear}" group="1"
         dyntype="filterexact" dynprm="{tau}" gainprm="1.4"
         biastype="affine" biasprm=".03 -.2 -.05" actearly="{early_text}"
         ctrllimited="true" ctrlrange="-.4 .4"
         actlimited="true" actrange="-.2 .3"
         forcelimited="true" forcerange="{force_range}"/>'''
    return f'''<mujoco model="rob2a-{case}"><compiler angle="radian"/>
      <option timestep="{H}" gravity=".1 -.2 -1" integrator="Euler"{extra}/>
      <worldbody><body name="base" pos=".04 .02 .3">
       <joint name="joint" type="{kind}" axis="0 1 0" armature=".04" damping=".12"/>
       <inertial pos=".03 .01 -.02" mass="1.1" diaginertia=".04 .05 .06"/>
       <geom size=".04" contype="0" conaffinity="0"/>{child}
      </body></worldbody><actuator>{actors}</actuator></mujoco>'''


def scalar_control(model, tick, slot=0):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    command = 0.8 if tick < 8 else (-0.75 if tick < 20 else 0.0)
    wrench = np.zeros((model.nbody, 6))
    wrench[-1] = [0.02, -0.015, 0.01, 0.005, -0.003, 0.004]
    return MuJoCoControl(
        np.full(model.nu, command + slot * 0.015),
        qfrc_applied=np.full(model.nv, 0.012),
        xfrc_applied=wrench,
    )


def scalar_analytic(model, state, packet):
    # Independent native field interpretation, not a call into product lowering.
    output = state.act.copy()
    derivative = np.zeros(model.na)
    used = np.empty(model.nu)
    clamped = np.empty(model.nu)
    raw = np.empty(model.nu)
    force = np.empty(model.nu)
    generalized = np.zeros(model.nv)
    for actor in range(model.nu):
        u = float(np.float32(packet.ctrl[actor]))
        if model.actuator_ctrllimited[actor]:
            u = float(np.clip(u, *model.actuator_ctrlrange[actor]))
        clamped[actor] = u
        adr = int(model.actuator_actadr[actor])
        disabled = bool(model.opt.disableactuator & (1 << int(model.actuator_group[actor])))
        value = u
        if adr >= 0:
            tau = max(1e-15, float(model.actuator_dynprm[actor, 0]))
            derivative[adr] = (u - state.act[adr]) / tau
            candidate = (
                state.act[adr]
                if disabled
                else state.act[adr] + (u - state.act[adr]) * -math.expm1(-H / tau)
            )
            if model.actuator_actlimited[actor]:
                candidate = float(np.clip(candidate, *model.actuator_actrange[actor]))
            output[adr] = candidate
            value = candidate if model.actuator_actearly[actor] else state.act[adr]
        used[actor] = value
        joint = int(model.actuator_trnid[actor, 0])
        qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        gear = float(model.actuator_gear[actor, 0])
        b = model.actuator_biasprm[actor, :3]
        raw[actor] = (
            0
            if disabled
            else float(model.actuator_gainprm[actor, 0]) * value
            + b[0]
            + b[1] * gear * state.q[qa]
            + b[2] * gear * state.v[dof]
        )
        force[actor] = raw[actor]
        if model.actuator_forcelimited[actor]:
            force[actor] = np.clip(force[actor], *model.actuator_forcerange[actor])
        generalized[dof] += gear * force[actor]
    return output, derivative, clamped, used, raw, force, generalized


# Original source-case declarations. The installed program imports no test helper.
WORLD_KEYS = (41, 7, 305, 11, 203, 59)
FAMILIES = {41: "scalar", 7: "ball", 305: "free", 11: "scalar", 203: "ball", 59: "free"}
SLOTS = {41: 0, 7: 0, 305: 0, 11: 1, 203: 1, 59: 1}
SCALAR_TOL = {
    "mass_matrix": (2e-4, 3e-5),
    "bias": (3e-4, 2e-4),
    "passive": (3e-4, 2e-4),
    "applied": (3e-4, 2e-4),
    "smooth_acceleration": (5e-4, 8e-4),
}


def scalar_compare(row, before, after, expected):
    for field, (rtol, atol) in SCALAR_TOL.items():
        np.testing.assert_allclose(
            getattr(row.dynamics, field), expected[field], rtol=rtol, atol=atol
        )
    np.testing.assert_allclose(
        row.dynamics.jacobian_com, expected["jacobian_com"], rtol=3e-5, atol=4e-5
    )
    act = row.actuation
    channels = (
        act.output_act,
        act.activation_derivative,
        act.clamped_control,
        act.force_input,
        act.raw_force,
        act.scalar_force,
        act.generalized_force,
    )
    for i, (got, want) in enumerate(zip(channels, expected["scalar_analytic"], strict=True)):
        np.testing.assert_allclose(got, want, **(ACT if i < 2 else FORCE))
        require(not got.flags.writeable, "complete immutable scalar activation")
    np.testing.assert_allclose(act.scalar_force, expected["actuator_force"], **FORCE)
    np.testing.assert_array_equal(act.generalized_force, row.qfrc_actuator)
    require(not row.contacts and not row.constraints, "scalar no-contact domain")
    for target in (expected["native_successor"], expected["continuous"]):
        np.testing.assert_allclose(after.q, target["q"], rtol=0, atol=5e-5)
        np.testing.assert_allclose(after.v, target["v"], rtol=0, atol=3e-4)
        np.testing.assert_allclose(after.act, target["act"], **ACT)
        require(abs(after.time - target["time"]) <= 2e-6, "scalar continuous time")


def free_compare(row, before, after, expected, model):
    for field, (rtol, atol) in free_TOL.items():
        got = (
            row.integration_acceleration
            if field == "integration_acceleration"
            else getattr(row.dynamics, field)
        )
        require(
            got.shape == expected[field].shape and not got.flags.writeable, "complete FREE dynamics"
        )
        np.testing.assert_allclose(got, expected[field], rtol=rtol, atol=atol)
    np.testing.assert_allclose(row.qacc, expected["smooth_acceleration"], rtol=2e-3, atol=4e-3)
    np.testing.assert_allclose(row.qfrc_actuator, expected["qfrc_actuator"], **free_FORCE)
    free_check_transmission(row, before, expected, expected["transmission"])
    np.testing.assert_array_equal(row.qfrc_constraint, expected["qfrc_constraint"])
    require(not row.contacts and not row.constraints, "FREE BALL no-contact scope")
    for target in (expected["native_successor"], expected["continuous"]):
        np.testing.assert_allclose(after.q, target["q"], rtol=0, atol=free_Q_ATOL)
        np.testing.assert_allclose(after.v, target["v"], rtol=0, atol=free_V_ATOL)
        require(abs(after.time - target["time"]) <= free_TIME_ATOL, "FREE continuous time")
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        if kind in (0, 1):
            qa = int(model.jnt_qposadr[joint]) + (3 if kind == 0 else 0)
            require(
                abs(np.dot(after.q[qa : qa + 4], after.q[qa : qa + 4]) - 1)
                <= 8 * np.finfo(np.float32).eps,
                "native root/child quaternion",
            )


def reference_states(mj, models, states):
    result = {}
    for key, state in states.items():
        native = mj.MjData(models[FAMILIES[key]])
        native.qpos[:], native.qvel[:], native.act[:], native.time = (
            state.q,
            state.v,
            state.act,
            state.time,
        )
        result[key] = native
    return result


def commands(models, tick):
    funcs = {"scalar": scalar_control, "ball": control, "free": free_control}
    return {
        key: funcs[FAMILIES[key]](models[FAMILIES[key]], tick, SLOTS[key]) for key in WORLD_KEYS
    }


def advance_heterogeneous(batch, mj, models, refs, packets, journal, progress, label):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    entry, full_entry = batch.integration_snapshot(), batch.snapshot()
    expected = {}
    for key in WORLD_KEYS:
        name, model = FAMILIES[key], models[FAMILIES[key]]
        packet = packets[key]
        actual = MuJoCoControl(
            np.asarray(packet.ctrl, np.float32),
            qfrc_applied=np.asarray(packet.qfrc_applied, np.float32),
            xfrc_applied=np.asarray(packet.xfrc_applied, np.float32),
        )
        expected[key] = (free_native_fields if name == "free" else native_fields)(
            mj, model, entry[key], actual
        )
        if name == "scalar":
            expected[key]["scalar_analytic"] = scalar_analytic(model, entry[key], actual)
        set_control(refs[key], actual)
        mj.mj_step(model, refs[key])
        expected[key]["continuous"] = native_state(refs[key])
    batch.set_controls(packets)
    journal.emit(
        "attempt",
        label=label,
        full_entry=full_entry,
        entry=entry,
        requested=packets,
        native=expected,
        h=H,
        revision=batch.checkpoint_revision,
    )
    progress["attempted_calls"] += 1

    def forbidden(*args, **kwargs):
        raise AssertionError("native physics called inside actual TYX step")

    with patch.object(mj, "mj_step", forbidden), patch.object(mj, "mj_forward", forbidden):
        accepted = batch.step(H)
        progress["physics_returned_calls"] += 1
        progress["accepted_world_steps"] += sum(bool(value) for value in accepted.values())
        journal.emit("physics-return", label=label, accepted=accepted, progress=dict(progress))
        output = batch.integration_snapshot()
        journal.emit("accepted-integration", label=label, output=output, receipt=batch.last_step)
    full = batch.snapshot()
    journal.emit(
        "step",
        label=label,
        full_output=full,
        output=output,
        receipt=batch.last_step,
        counters=batch.counters,
    )
    require(
        tuple(output) == WORLD_KEYS and dict(accepted) == dict.fromkeys(WORLD_KEYS, True),
        "all world identity/acceptance",
    )
    require(tuple(batch.last_step.receipts) == WORLD_KEYS, "complete aggregate receipt")
    for key in WORLD_KEYS:
        row, before, after = output[key].last_step, entry[key], output[key]
        require(row.world == key and row.tick == after.tick == before.tick + 1, "world/tick")
        require(
            row.input_time == before.time and row.output_time == after.time and row.h == H, "clock"
        )
        exact(row.input_q, before.q, "input q")
        exact(row.input_v, before.v, "input v")
        exact(row.control, packets[key], "requested controls")
        for field in ("ctrl", "qfrc_applied", "xfrc_applied"):
            np.testing.assert_array_equal(
                getattr(row.stored_control, field),
                np.asarray(getattr(packets[key], field), np.float32),
            )
        exact(batch.last_step.receipts[key], row, "full aggregate force/activation/contact receipt")
        exact(batch.last_step.states[key], after, "same force and motion publication")
        for field in (
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
        ):
            exact(getattr(full[key], field), getattr(after, field), "observer state " + field)
        if FAMILIES[key] == "ball":
            compare(row, before, after, packets[key], expected[key], models["ball"])
        elif FAMILIES[key] == "free":
            free_compare(row, before, after, expected[key], models["free"])
        else:
            scalar_compare(row, before, after, expected[key])
    progress["checked_calls"] += 1
    return output


def run(mode, output, installed_root, journal, progress):
    import mujoco as mj

    from phymeo.backends.mujoco_heterogeneous import (
        MuJoCoTyxHeterogeneousBatch,
        MuJoCoTyxModelSpec,
        MuJoCoTyxWorldSpec,
    )
    from phymeo.backends.mujoco_heterogeneous_checkpoint import MuJoCoTyxHeterogeneousCheckpoint
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    require(mj.__version__ == "3.3.7", "fixed independent native")
    xml = {
        "scalar": scalar_source("coupled", early=True),
        "ball": source("r4-pyramid"),
        "free": free_source("mixed"),
    }
    models = {name: mj.MjModel.from_xml_string(value) for name, value in xml.items()}
    specs = tuple(
        MuJoCoTyxModelSpec(name, compile_mujoco_xml(value, require_asset=False))
        for name, value in xml.items()
    )
    starts = {}
    for key in WORLD_KEYS:
        model, slot, family = models[FAMILIES[key]], SLOTS[key], FAMILIES[key]
        if family == "scalar":
            starts[key] = MuJoCoInitialState(
                model.qpos0 + 0.06,
                np.full(model.nv, -0.035),
                act=np.full(model.na, 0.34 + slot * 0.01),
                time=0.125,
            )
        elif family == "ball":
            starts[key] = initial(mj, model, "r4-pyramid", slot)
        else:
            starts[key] = free_initial(mj, model, slot)
    worlds = tuple(MuJoCoTyxWorldSpec(key, FAMILIES[key], starts[key]) for key in WORLD_KEYS)
    journal.emit(
        "declaration",
        xml=xml,
        source_sha256={
            key: hashlib.sha256(value.encode()).hexdigest() for key, value in xml.items()
        },
        worlds=worlds,
        mode=mode,
        h=H,
        progress=dict(progress),
        numerical_gates=dict(
            ball=TOL,
            free=free_TOL,
            scalar=SCALAR_TOL,
            ACT=ACT,
            FORCE=FORCE,
            CONTACT_FORCE=CONTACT_FORCE,
        ),
    )
    batch = None
    primary = None
    result = None
    try:
        batch = MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode)
        original = batch.integration_snapshot()
        refs = reference_states(mj, models, original)
        initial_refs = {key: copy.copy(value) for key, value in refs.items()}
        journal.emit(
            "binding",
            layout=batch.layout,
            capabilities=batch.capabilities,
            initial=original,
            provenance=provenance(installed_root),
        )
        require(batch.capabilities["bucket_count"] == 3, "three actual heterogeneous buckets")
        contact_seen, motor_seen = False, False
        for tick in range(24):
            current = advance_heterogeneous(
                batch,
                mj,
                models,
                refs,
                commands(models, tick),
                journal,
                progress,
                f"continuous-{tick}",
            )
            contact_seen |= bool(np.any(current[7].last_step.qfrc_constraint != 0))
            motor_seen |= bool(np.any(current[305].last_step.qfrc_actuator != 0))
        require(contact_seen and motor_seen, "actual contact load and direct motor load")
        require(batch.counters["boundary_clones"] == 0, "ordinary step does not clone")
        saved = batch.checkpoint()
        raw = saved.to_bytes()
        (output / "heterogeneous-checkpoint.bin").write_bytes(raw)
        saved_refs = {key: copy.copy(value) for key, value in refs.items()}
        successor = advance_heterogeneous(
            batch, mj, models, refs, commands(models, 24), journal, progress, "successor"
        )
        revision = batch.checkpoint_revision
        restored = batch.restore(
            MuJoCoTyxHeterogeneousCheckpoint.from_bytes(raw), expected_revision=revision
        )
        refs = {key: copy.copy(value) for key, value in saved_refs.items()}
        journal.emit(
            "restore",
            receipt=restored,
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            state=batch.integration_snapshot(),
        )
        repeated = advance_heterogeneous(
            batch, mj, models, refs, commands(models, 24), journal, progress, "restored-successor"
        )
        exact(repeated, successor, "exact restored successor")
        before_reset = batch.integration_snapshot()
        reset = batch.reset((41, 7))
        reset_state = batch.integration_snapshot()
        for key in WORLD_KEYS:
            if key in (41, 7):
                require(
                    reset_state[key].tick == 0 and reset_state[key].last_step is None,
                    "original selective reset",
                )
                for field in ("q", "v", "act", "time"):
                    exact(
                        getattr(reset_state[key], field),
                        getattr(original[key], field),
                        "original reset " + field,
                    )
                refs[key] = copy.copy(initial_refs[key])
            else:
                exact(reset_state[key], before_reset[key], "unselected full state/receipt")
        journal.emit("reset", receipt=reset, before=before_reset, after=reset_state)
        final = advance_heterogeneous(
            batch, mj, models, refs, commands(models, 25), journal, progress, "reset-successor"
        )
        held = batch.checkpoint().to_bytes()
        bad = dict(commands(models, 25))
        bad[999999] = bad.pop(59)
        try:
            batch.set_controls(bad)
        except ValueError as error:
            journal.emit("negative", name="wrong-world-control", error=repr(error))
        else:
            raise AssertionError("wrong world accepted")
        require(batch.checkpoint().to_bytes() == held, "bad control retains all worlds")
        try:
            batch.restore(saved, expected_revision=batch.checkpoint_revision - 1)
        except ValueError as error:
            journal.emit("negative", name="stale-restore", error=repr(error))
        else:
            raise AssertionError("stale restore accepted")
        require(batch.checkpoint().to_bytes() == held, "bad restore retains all worlds")
        require(
            progress
            == dict(
                attempted_calls=27,
                physics_returned_calls=27,
                checked_calls=27,
                accepted_world_steps=162,
            ),
            "bounded complete actual calls",
        )
        result = dict(
            mode=mode,
            progress=dict(progress),
            counters=batch.counters,
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            final=final,
            layout=batch.layout,
            capabilities=batch.capabilities,
            provenance=provenance(installed_root),
        )
        journal.emit("final", **result)
    except BaseException as error:
        primary = error
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        journal.emit("failure", error=repr(error), progress=dict(progress))
    finally:
        cleanup = []
        if batch is not None:
            try:
                batch.close()
            except BaseException as error:
                cleanup.append(error)
        if primary is not None:
            if cleanup:
                raise BaseExceptionGroup(
                    "heterogeneous consumer and cleanup failed", [primary, *cleanup]
                ) from None
            raise primary
        if cleanup:
            raise BaseExceptionGroup("heterogeneous consumer cleanup failed", cleanup)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-root", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and args.installed_root.is_absolute(),
        "absolute output/install paths",
    )
    output, root = args.output.resolve(), args.installed_root.resolve(strict=True)
    require(not output.exists(), "fresh output required")
    output.mkdir(parents=True)
    journal = Journal(output / "trajectory.jsonl.gz")
    summary, progress = (
        {"status": "FAIL", "mode": args.mode},
        dict(attempted_calls=0, physics_returned_calls=0, checked_calls=0, accepted_world_steps=0),
    )
    primary = None
    try:
        summary["preflight"] = provenance(root, preflight=True)
        result = run(args.mode, output, root, journal, progress)
        summary.update(
            status="PASS",
            progress=dict(progress),
            checkpoint_sha256=result["checkpoint_sha256"],
            counters=encode(result["counters"]),
            provenance=result["provenance"],
        )
    except BaseException as error:
        primary = error
        summary.update(error=repr(error), traceback=traceback.format_exc(), progress=dict(progress))
    finally:
        try:
            summary["trace"] = journal.close()
        except BaseException as cleanup:
            if primary is None:
                primary = cleanup
            else:
                primary = BaseExceptionGroup("consumer and trace close failed", [primary, cleanup])
            summary["trace_error"] = repr(cleanup)
        (output / "summary.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2), encoding="utf-8"
        )
    if primary is not None:
        raise primary
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
