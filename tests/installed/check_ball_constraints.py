"""Standalone BALL-S native scalar limits/dry + BALL activation qualification.

Inputs are frozen in p65-ball-scalar-constraints-cases-20261113.md.
Independent decoders are copied from the accepted T4/R3 consumers at c550f9c4;
this file has no checkout helper import and never advances TYX with native CPU.
"""
import argparse
import base64
import copy
import gzip
import hashlib
import itertools
import json
import math
import struct
import sys
import traceback
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
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


H = 0.001
KEYS = (3, 29)
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
ROW_TOL = dict(rtol=5e-4, atol=4e-4)
CASES = ("fixed-lower", "fixed-upper", "fixed-margin", "fixed-overlap", "free-filter-limits",
         "dry-positive", "dry-negative", "dry-stick", "dry-zero", "mixed-normal-none",
         "mixed-normal-filter", "mixed-pyramid-filter")
SHORT = {"fixed-margin", "fixed-overlap", "dry-stick", "dry-zero"}
RETAINED_OWNERS = []


@contextmanager
def expect_raises(types):
    try:
        yield
    except types:
        return
    raise AssertionError("expected explicit rejection " + str(types))


@contextmanager
def saved(path):
    journal = Journal(Path(path))
    try:
        yield journal
    finally:
        primary = sys.exc_info()[1]
        try:
            journal.close()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup("evidence and primary failure", [primary, cleanup])
            raise


@contextmanager
def owned(runtime, journal):
    try:
        yield runtime
    finally:
        primary = sys.exc_info()[1]
        failures = []
        try:
            runtime.close()
        except BaseException as cleanup:
            RETAINED_OWNERS.append(runtime)
            failures.append(cleanup)
            try:
                journal.emit("owner-close-failure", error=repr(cleanup), primary=repr(primary))
            except BaseException as evidence:
                failures.append(evidence)
        else:
            try:
                journal.emit("owner-closed")
            except BaseException as evidence:
                failures.append(evidence)
        if failures:
            raise BaseExceptionGroup("owner cleanup/evidence failure", ([primary] if primary else []) + failures)

INERTIA = '<inertial pos=".03 -.02 .04" mass=".7" diaginertia=".035 .045 .055" quat=".9659258262890683 .2588190451025207 0 0"/>'


def source(name):
    require(name in CASES, "fixed source identity")
    free = name == "free-filter-limits" or name.endswith("-filter")
    activation = free
    mixed = name.startswith("mixed-")
    limited = name.startswith("fixed-") or name == "free-filter-limits" or mixed
    dry = .2 if (name.startswith("dry-") and name != "dry-zero") or mixed else 0.
    dim = 3 if name == "mixed-pyramid-filter" else 1
    attrs = 'limited="true" range="-.25 .25" margin=".03125"' if limited else 'limited="false"'
    hinge_attrs = attrs
    if name == "fixed-overlap":
        attrs = hinge_attrs = 'limited="true" range="-.00390625 .00390625" margin=".0078125"'
    if mixed:
        hinge_attrs = 'limited="true" range="-.015625 .25" margin=".03125"'
    body = f'''<body name="ball-body" pos="{('0 0 0' if free else '0 0 .4')}">
      <joint name="ball" type="ball" limited="false" frictionloss="{dry}"/>{INERTIA}
      <geom type="sphere" size=".035" contype="0" conaffinity="0"/>
      <body name="hinge-body" pos=".12 0 -.1"><joint name="hinge" type="hinge" axis="0 1 0" {hinge_attrs} frictionloss="{dry}"/>{INERTIA}
      <geom type="sphere" size=".035" contype="0" conaffinity="0"/>
      <body name="tip" pos="0 0 -.201"><joint name="slide" type="slide" axis="0 0 1" {attrs} frictionloss="{dry}"/>
      <inertial pos="-.01 0 .01" mass=".5" diaginertia=".012 .014 .016"/>
      <geom name="tip-shape" type="sphere" size=".1" contype="{int(mixed)}" conaffinity="{int(mixed)}"/>
      </body></body></body>'''
    if free:
        body = '''<body name="free-body" pos="0 0 .4"><freejoint name="free"/>
        <inertial pos=".04 -.03 .02" mass="2.6" diaginertia=".13 .19 .23" quat=".9659258262890683 0 .2588190451025207 0"/>
        <geom type="sphere" size=".06" contype="0" conaffinity="0"/>''' + body + '</body>'
    stateful = ('dyntype="filterexact" dynprm=".004" actearly="false" ctrllimited="true" ctrlrange="-.45 .45" actlimited="true" actrange="-.3 .3"' if activation else 'dyntype="none"')
    slide_stateful = stateful.replace('.004', '.007').replace('actearly="false"', 'actearly="true"')
    floor = '<geom name="floor" type="plane" size="2 2 .1"/>' if mixed else ''
    return f'''<mujoco model="ball-s-{name}"><compiler angle="radian" autolimits="true" fusestatic="false"/>
    <option timestep=".001" gravity="0 0 0" integrator="Euler" solver="Newton" iterations="128" tolerance="1e-10" jacobian="dense" cone="pyramidal"/>
    <default><joint armature=".003" damping=".04" solreflimit=".02 1" solimplimit=".8 .9 .002 .5 2" solreffriction=".02 1" solimpfriction=".8 .9 .002 .5 2"/>
    <geom condim="{dim}" friction=".4 .01 .001" solref=".02 1" solimp=".8 .9 .002 .5 2"/></default>
    <worldbody>{floor}<body name="marker" pos="2 -1 .4"><geom type="sphere" size=".02" contype="0" conaffinity="0"/></body>{body}</worldbody>
    <actuator><general name="ball-drive" joint="ball" gear="1 -2 .5" gainprm=".8" {stateful}/>
    <general name="ball-counter" joint="ball" gear="-.25 .5 -.125" gainprm=".6" dyntype="none"/>
    <general name="hinge-drive" joint="hinge" gear="-.5" gainprm="1" dyntype="none"/>
    <general name="slide-drive" joint="slide" gear="-2" gainprm="1" biastype="affine" biasprm=".01 -.3 -.04" {slide_stateful}/></actuator></mujoco>'''


def negative_source(name):
    root = ET.fromstring(source("mixed-pyramid-filter"))
    ball = root.find(".//joint[@name='ball']")
    if name == "ball-limit": ball.set("limited", "true"); ball.set("range", "0 .2")
    elif name == "free-target": root.find("actuator/general").set("joint", "free")
    elif name == "spring": ball.set("stiffness", ".1")
    elif name == "elliptic": root.find("option").set("cone", "elliptic")
    elif name in ("condim4", "condim6"): root.find("default/geom").set("condim", name[-1])
    elif name == "cylinder": root.find(".//geom[@name='tip-shape']").set("type", "cylinder"); root.find(".//geom[@name='tip-shape']").set("size", ".1 .1")
    elif name == "exclude": ET.SubElement(ET.SubElement(root,"contact"), "exclude", body1="ball-body", body2="tip")
    elif name == "explicit-pair": ET.SubElement(ET.SubElement(root,"contact"), "pair", geom1="floor", geom2="tip-shape")
    elif name == "multiple-free":
        b = ET.SubElement(root.find("worldbody"),"body", name="another", pos="1 1 1")
        ET.SubElement(b,"freejoint"); ET.SubElement(b,"geom",type="sphere",size=".03",contype="0",conaffinity="0")
    else: raise ValueError(name)
    return ET.tostring(root,encoding="unicode")


def initial(mj, model, name, slot):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    q, v = model.qpos0.copy(), np.zeros(model.nv)
    sign = -.75 if slot else 1.
    for jid, kind in enumerate(model.jnt_type):
        qa, da = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        if kind == 0:
            quat = np.array((1.,0.,0.,-.125)); q[qa+3:qa+7] = quat / np.linalg.norm(quat)
            if slot: q[qa:qa+2] += (.03125,-.015625)
            v[da:da+6] = sign*np.array((.07,-.05,.02,.09,-.06,.04))
        elif kind == 1:
            quat = np.array((1.,0.,0.,.125)); q[qa:qa+4] = quat/np.linalg.norm(quat)
            v[da:da+3] = sign*np.array((.15,-.2,.1))
    ids = [mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,x) for x in ("hinge","slide")]
    qa, da = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    v[da] = sign*np.array((.125,-.1))
    if name in ("fixed-lower","free-filter-limits"): q[qa]=(-15/64,15/64); v[da]=(1.,-1.)
    elif name == "fixed-upper": q[qa]=(15/64,-15/64); v[da]=(-1.,1.)
    elif name == "fixed-margin": q[qa]=(-7/32,7/32); v[da]=(.125,-.125)
    elif name == "fixed-overlap": v[da]=(1/32,-1/32)
    elif name == "dry-negative": v *= -1
    elif name == "dry-stick": v[:] = 0
    act = np.resize(np.array((.12,-.08)), model.na)
    return MuJoCoInitialState(q,v,act=act)


def control(model, tick, slot=0, name=None):
    from phymeo.backends.mujoco_runtime import MuJoCoControl
    phase = 1 if tick<8 else -1 if tick<16 else 0
    ctrl = phase*(.08+.01*np.arange(model.nu)+.005*slot)
    qforce = phase*.01*(np.arange(model.nv)+1)
    wrench = np.zeros((model.nbody,6)); wrench[-1] = phase*np.array((.03,-.02,.04,.002,-.003,.001))
    if name == "dry-stick": ctrl[:] = .01; qforce[:] = .005; wrench[:] = 0
    if name is not None and name.startswith("mixed-") and tick>=16:
        ctrl[:] = 0; qforce[:] = 0; wrench[:] = 0; qforce[-1] = 64.
    # All native comparisons consume the exact FP32 storage values.
    return MuJoCoControl(ctrl.astype(np.float32).astype(float),
                         qfrc_applied=qforce.astype(np.float32).astype(float),
                         xfrc_applied=wrench.astype(np.float32).astype(float))


def fixture(name, mode, keys=KEYS, *, capacities=None, starts_override=None):
    import mujoco as mj
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    require(mj.__version__ == "3.3.7", "fixed CPU oracle")
    xml=source(name); model=mj.MjModel.from_xml_string(xml)
    compiled=compile_mujoco_xml(xml,require_asset=False)
    starts={key:initial(mj,model,name,i) for i,key in enumerate(keys)}
    if starts_override is not None: starts=starts_override(mj,model,starts)
    runtime=MuJoCoRuntime(compiled,mode=mode,implementation="tyx",world_keys=keys,
                         initial_states=dict(reversed(tuple(starts.items()))),**(capacities or {}))
    return mj,model,compiled,starts,runtime
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
    require(
        tr.schema == native["transmission_schema"], "complete transmission schema"
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


def check_activation(row, entry, after, native):
    from phymeo.backends.mujoco_joint_transmission import MuJoCoTyxJointActivationStepReceipt

    require(type(row) is MuJoCoTyxJointActivationStepReceipt, "typed combined publication")
    act, expected = row.actuation, native["analytic"]
    require(
        act.schema == native["actuation_schema"],
        "combined activation schema",
    )
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


def tiny_box_qp(matrix, linear, lower, upper):
    """Enumerate complete small box faces, independently of product active sets."""
    n = len(linear)
    assert n <= 6
    labels = [(0, 1, 2) if np.isfinite(upper[i]) else (0, 1) for i in range(n)]
    valid = []
    for flags in itertools.product(*labels):
        flags = np.array(flags)
        free = np.flatnonzero(flags == 1)
        fixed = np.flatnonzero(flags != 1)
        candidate = lower.copy()
        candidate[flags == 2] = upper[flags == 2]
        if len(free):
            rhs = -linear[free] - matrix[np.ix_(free, fixed)] @ candidate[fixed]
            candidate[free] = np.linalg.solve(matrix[np.ix_(free, free)], rhs)
        gradient = matrix @ candidate + linear
        if np.any(candidate < lower - 1e-8) or np.any(candidate > upper + 1e-8):
            continue
        if np.any(gradient[flags == 0] < -1e-7) or np.any(gradient[flags == 2] > 1e-7):
            continue
        if np.max(np.abs(gradient[free]), initial=0) > 1e-7:
            continue
        valid.append(candidate)
    assert valid, "independent strictly convex box problem has no feasible KKT face"
    for other in valid[1:]:
        np.testing.assert_allclose(other, valid[0], rtol=2e-6, atol=2e-6)
    return valid[0]


def native_rows(mj, model, q, v, act, control, time=0.0, *, tiny=False):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = q, v, time
    data.act[:] = act
    set_control(data, control)
    mj.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jc = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(model.nbody):
        mj.mj_jac(model, data, jc[body, :3], jc[body, 3:], data.xipos[body], body)
        applied += jc[body].T @ control.xfrc_applied[body]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    smooth = np.linalg.solve(mass, rhs)
    contacts = []
    for index in range(data.ncon):
        c = data.contact[index]
        geoms = tuple(map(int, c.geom))
        bodies = tuple(int(model.geom_bodyid[g]) for g in geoms)
        point_j = np.zeros((6, model.nv))
        frame = c.frame.reshape(3, 3).copy()
        for sign, body in ((-1, bodies[0]), (1, bodies[1])):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mj.mj_jac(model, data, jp, jr, c.pos, body)
            point_j[:3] += sign * frame @ jp
            point_j[3:] += sign * frame @ jr
        wrench = np.zeros(6)
        mj.mj_contactForce(model, data, index, wrench)
        dim = int(c.dim)
        contacts.append(
            dict(
                native_index=index,
                geometry_ids=geoms,
                body_ids=bodies,
                dimension=dim,
                distance=float(c.dist),
                position=c.pos.copy(),
                frame=frame,
                include_margin=float(c.includemargin),
                friction=c.friction.copy(),
                solref=c.solref.copy(),
                solimp=c.solimp.copy(),
                active=c.efc_address >= 0,
                efc_address=int(c.efc_address),
                jacobian=point_j,
                wrench_contact_frame=wrench,
                normal_force=float(wrench[0]),
            )
        )
    n = data.nefc
    j = data.efc_J.reshape(n, model.nv).copy()
    types, ids = data.efc_type.copy(), data.efc_id.copy()
    rows = []
    for i in range(n):
        kind, source = int(types[i]), int(ids[i])
        if kind == 1:
            side, dof, contact = -1, source, -1
        elif kind == 3:
            dof = int(model.jnt_dofadr[source])
            side, contact = (-1 if j[i, dof] > 0 else 1), -1
        else:
            assert kind in (5, 6)
            dof, contact = -1, source
            side = i - contacts[source]["efc_address"]
        lower = -float(data.efc_frictionloss[i]) if kind == 1 else 0.0
        upper = float(data.efc_frictionloss[i]) if kind == 1 else None
        rows.append(
            dict(
                constraint_type=kind,
                source_id=source,
                side_or_edge=side,
                dof=dof,
                contact=contact,
                jacobian=j[i],
                position=float(data.efc_pos[i]),
                margin=float(data.efc_margin[i]),
                velocity=float(data.efc_vel[i]),
                reference_acceleration=float(data.efc_aref[i]),
                regularization=float(data.efc_R[i]),
                inverse_regularization=float(data.efc_D[i]),
                impedance=float(data.efc_KBIP[i, 2]),
                diag_approx=float(data.efc_diagApprox[i]),
                lower=lower,
                upper=upper,
                force=float(data.efc_force[i]),
            )
        )
    regularization = np.array([r["regularization"] for r in rows])
    aref = np.array([r["reference_acceleration"] for r in rows])
    force = np.array([r["force"] for r in rows])
    matrix = j @ np.linalg.solve(mass, j.T) + np.diag(regularization)
    linear = j @ smooth - aref
    lower = np.array([r["lower"] for r in rows])
    upper = np.array([np.inf if r["upper"] is None else r["upper"] for r in rows])
    if tiny and n <= 6:
        np.testing.assert_allclose(
            force, tiny_box_qp(matrix, linear, lower, upper), rtol=2e-6, atol=2e-6
        )
    np.testing.assert_allclose(j.T @ force, data.qfrc_constraint, rtol=1e-9, atol=1e-8)
    return dict(
        native_input=native_state(data), act_dot=data.act_dot.copy(),
        actuator_force=data.actuator_force.copy(), dense_moment=dense_native_moment(model, data),
        body_position=data.xpos.copy(),
        body_rotation=data.xmat.reshape(model.nbody, 3, 3).copy(),
        body_com=data.xipos.copy(),
        jacobian_com=jc,
        mass_matrix=mass,
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=applied,
        smooth_acceleration=smooth,
        qacc=data.qacc.copy(),
        integration_acceleration=np.linalg.solve(
            mass + model.opt.timestep * np.diag(model.dof_damping), rhs + data.qfrc_constraint
        ),
        qfrc_constraint=data.qfrc_constraint.copy(),
        qfrc_actuator=data.qfrc_actuator.copy(),
        contacts=contacts,
        rows=rows,
        matrix=matrix,
        linear=linear,
        lower=lower,
        upper_bound=[r["upper"] for r in rows],
        force=force,
    )


def match_contacts(actual, expected):
    assert len(actual) == len(expected)
    candidates = [
        [
            j
            for j, oracle in enumerate(expected)
            if tuple(c.geometry_ids) == oracle["geometry_ids"]
            and np.allclose(c.position, oracle["position"], rtol=3e-5, atol=4e-5)
            and np.allclose(c.frame[0], oracle["frame"][0], rtol=0, atol=1e-4)
        ]
        for c in actual
    ]
    matched = {}

    def augment(i, seen):
        for j in candidates[i]:
            if j not in seen:
                seen.add(j)
                if j not in matched or augment(matched[j], seen):
                    matched[j] = i
                    return True
        return False

    for i in range(len(actual)):
        assert augment(i, set()), "complete native geometry bijection missing"
    result = [None] * len(actual)
    for j, i in matched.items():
        result[i] = expected[j]
    assert len({r["native_index"] for r in result}) == len(expected)
    return result


def assert_receipt(row, expected, model):
    """Compare all rows after geometry-only contact bijection, never force matching."""
    dyn = row.dynamics
    np.testing.assert_array_equal(dyn.input_q, row.input_q)
    np.testing.assert_array_equal(dyn.input_v, row.input_v)
    assert dyn.time == row.input_time
    for name, (rtol, atol) in TOL.items():
        value = getattr(dyn, name)
        assert value.shape == expected[name].shape
        np.testing.assert_allclose(value, expected[name], rtol=rtol, atol=atol, err_msg=name)
        with expect_raises(ValueError):
            value.setflags(write=True)
    for name in ("qacc", "integration_acceleration"):
        np.testing.assert_allclose(getattr(row, name), expected[name], **ACC)
    np.testing.assert_allclose(row.qfrc_constraint, expected["qfrc_constraint"], **CONTACT_FORCE)
    assert isinstance(row.constraints, tuple)
    contacts = match_contacts(row.contacts, expected["contacts"])
    n, nc, nv = len(row.constraints), len(contacts), model.nv
    assert n == len(expected["rows"])
    raw = row.raw_contact_fields
    ids, values, jac = (np.asarray(raw[key]) for key in ("efc_ids", "efc_data", "efc_J"))
    counts = np.asarray(raw["counts"])
    np.testing.assert_array_equal(counts, [nc, sum(c.active for c in row.contacts), n])
    cmap = np.asarray(raw["contact_map"])
    cj, cw = np.asarray(raw["contact_J"]), np.asarray(raw["contact_wrench"])
    assert ids.shape[1] == 5 and values.shape[1] == 14 and jac.shape[1] >= nv
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(jac))
    assert set(raw) >= {"candidate_rows", "candidate_jacobian", "working_set", "solver_diagnostics"}
    mapping = {i: c["native_index"] for i, c in enumerate(contacts)}
    expected_rows = {
        (r["constraint_type"], r["source_id"], r["side_or_edge"]): r for r in expected["rows"]
    }
    assert len(expected_rows) == n
    actual_keys, generalized = [], np.zeros(nv)
    force, lower, upper, aref, reg = [], [], [], [], []
    scalar_fields = (
        "position",
        "margin",
        "velocity",
        "reference_acceleration",
        "regularization",
        "inverse_regularization",
        "impedance",
        "diag_approx",
    )
    for i, r in enumerate(row.constraints):
        assert r.schema == "mujoco-tyx-constraint-row-v1"
        source = mapping[r.source_id] if r.constraint_type in (5, 6) else r.source_id
        key = (r.constraint_type, source, r.side_or_edge)
        actual_keys.append(key)
        oracle = expected_rows[key]
        assert r.dof == oracle["dof"]
        np.testing.assert_array_equal(
            ids[i], [r.constraint_type, r.source_id, r.side_or_edge, r.geometry_slot, r.dof]
        )
        if r.constraint_type in (5, 6):
            assert r.geometry_slot == cmap[r.source_id, 0]
        else:
            assert r.geometry_slot == -1
        for field in scalar_fields:
            np.testing.assert_allclose(getattr(r, field), oracle[field], **ROW_TOL, err_msg=field)
        np.testing.assert_allclose(r.jacobian, oracle["jacobian"], rtol=2e-4, atol=5e-5)
        np.testing.assert_array_equal(r.jacobian, jac[i, :nv])
        np.testing.assert_allclose(r.force, oracle["force"], **CONTACT_FORCE)
        np.testing.assert_allclose(r.lower, oracle["lower"], rtol=0, atol=2e-7)
        assert (r.upper is None) == (oracle["upper"] is None)
        if r.upper is not None:
            np.testing.assert_allclose(r.upper, oracle["upper"], rtol=0, atol=2e-7)
        assert values[i, 11] == 1
        np.testing.assert_array_equal(
            values[i, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]],
            [
                r.position,
                r.margin,
                r.velocity,
                r.reference_acceleration,
                r.regularization,
                r.inverse_regularization,
                r.impedance,
                r.lower,
                0 if r.upper is None else r.upper,
                r.force,
                r.gradient,
                int(r.upper is not None),
                r.diag_approx,
            ],
        )
        force.append(r.force)
        lower.append(r.lower)
        upper.append(np.inf if r.upper is None else r.upper)
        aref.append(r.reference_acceleration)
        reg.append(r.regularization)
        generalized += r.jacobian * r.force
        with expect_raises((AttributeError, TypeError)):
            r.force = 0
    assert len(set(actual_keys)) == n and set(actual_keys) == set(expected_rows)
    np.testing.assert_allclose(generalized, row.qfrc_constraint, rtol=5e-5, atol=5e-5)
    contact_generalized = np.zeros(nv)
    for i, (c, oracle) in enumerate(zip(row.contacts, contacts, strict=True)):
        assert c.schema == "mujoco-tyx-pyramidal-contact-v1"
        assert c.dimension == oracle["dimension"] and c.active == oracle["active"]
        assert tuple(c.body_ids) == oracle["body_ids"]
        np.testing.assert_allclose(c.distance, oracle["distance"], rtol=3e-5, atol=3e-5)
        for field in ("position", "frame", "friction", "solref", "solimp", "include_margin"):
            np.testing.assert_allclose(getattr(c, field), oracle[field], rtol=3e-5, atol=4e-5)
        np.testing.assert_allclose(c.jacobian, oracle["jacobian"], rtol=2e-4, atol=5e-5)
        np.testing.assert_array_equal(c.jacobian, cj[i, :, :nv])
        np.testing.assert_array_equal(c.wrench_contact_frame, cw[i])
        np.testing.assert_allclose(
            c.wrench_contact_frame, oracle["wrench_contact_frame"], **CONTACT_FORCE
        )
        assert c.normal_force == c.wrench_contact_frame[0]
        np.testing.assert_allclose(
            c.normal_velocity, c.jacobian[0] @ row.input_v, rtol=2e-5, atol=2e-5
        )
        first, count = map(int, cmap[i, 1:])
        expected_count = (1 if c.dimension == 1 else 2 * (c.dimension - 1)) if c.active else 0
        assert count == expected_count and (first >= 0 if c.active else first == -1)
        decoded = np.zeros(6)
        if c.active:
            edges = np.array([r.force for r in row.constraints[first : first + count]])
            assert all(r.source_id == i for r in row.constraints[first : first + count])
            decoded[0] = edges.sum()
            for k in range(c.dimension - 1):
                decoded[k + 1] = c.friction[k] * (edges[2 * k] - edges[2 * k + 1])
        np.testing.assert_allclose(c.wrench_contact_frame, decoded, rtol=3e-5, atol=3e-5)
        if c.dimension > 1:
            assert (
                np.sum(np.abs(decoded[1 : c.dimension]) / c.friction[: c.dimension - 1])
                <= decoded[0] + 5e-5
            )
        world_force = c.frame.T @ decoded[:3]
        world_couple = c.frame.T @ decoded[3:]
        expected_body_wrenches = []
        for sign, body in ((-1, c.body_ids[0]), (1, c.body_ids[1])):
            lever = c.position - dyn.body_com[body]
            expected_body_wrenches.append(
                np.r_[sign * world_force, sign * (world_couple + np.cross(lever, world_force))]
            )
        for actual, wanted in zip(
            (c.wrench_on_body1_world, c.wrench_on_body2_world), expected_body_wrenches, strict=True
        ):
            np.testing.assert_allclose(actual, wanted, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(c.force_on_geometry1_world, -world_force, rtol=4e-5, atol=4e-5)
        np.testing.assert_allclose(c.force_on_geometry2_world, world_force, rtol=4e-5, atol=4e-5)
        # Both sides' actual COM Jacobians include angular reaction from spin/rolling.
        from_bodies = sum(
            dyn.jacobian_com[b].T @ w
            for b, w in zip(c.body_ids, expected_body_wrenches, strict=True)
        )
        np.testing.assert_allclose(from_bodies, c.jacobian.T @ decoded, rtol=4e-5, atol=4e-5)
        contact_generalized += from_bodies
    noncontact = sum(
        (r.jacobian * r.force for r in row.constraints if r.constraint_type not in (5, 6)),
        np.zeros(nv),
    )
    np.testing.assert_allclose(
        contact_generalized + noncontact, row.qfrc_constraint, rtol=5e-5, atol=5e-5
    )
    j = jac[:n, :nv].astype(float)
    f, lo, hi = np.array(force), np.array(lower), np.array(upper)
    hessian = j @ np.linalg.solve(dyn.mass_matrix.astype(float), j.T) + np.diag(reg)
    linear = j @ dyn.smooth_acceleration - np.array(aref)
    native_order = [
        next(
            i
            for i, r in enumerate(expected["rows"])
            if (r["constraint_type"], r["source_id"], r["side_or_edge"]) == key
        )
        for key in actual_keys
    ]
    np.testing.assert_allclose(
        hessian, expected["matrix"][np.ix_(native_order, native_order)], rtol=7e-4, atol=5e-4
    )
    gradient = hessian @ f + linear
    feasibility = max(np.max(lo - f, initial=0), np.max(f - hi, initial=0))
    assert feasibility <= 2e-7
    working = np.asarray(raw["working_set"])[:n]
    assert set(map(int, working)) <= {0, 1, 2}
    violation = np.where(
        working == 0,
        np.maximum(0, -gradient),
        np.where(working == 2, np.maximum(0, gradient), np.abs(gradient)),
    )
    for i in range(n):
        if working[i] == 0:
            assert abs(f[i] - lo[i]) <= 2e-7
        if working[i] == 2:
            assert np.isfinite(hi[i]) and abs(f[i] - hi[i]) <= 2e-7
    independent_scale = max(1.0, np.max(np.abs(linear) + np.abs(hessian) @ np.abs(f), initial=0))
    assert np.max(violation, initial=0) / independent_scale <= 5e-5
    for i, r in enumerate(row.constraints):
        np.testing.assert_allclose(
            r.gradient, gradient[i], rtol=3e-3, atol=5e-5 * independent_scale
        )
    physical_rhs = dyn.applied + dyn.passive - dyn.bias + row.qfrc_constraint
    mass_error = dyn.mass_matrix.astype(float) @ row.qacc - physical_rhs
    mass_scale = max(
        1.0, np.max(np.abs(physical_rhs) + np.abs(dyn.mass_matrix) @ np.abs(row.qacc), initial=0)
    )
    mass_residual = np.max(np.abs(mass_error), initial=0) / mass_scale
    assert mass_residual <= 5e-5
    diag = raw["solver_diagnostics"]
    assert diag["converged"] == 1 and diag["actual_efc"] == n
    assert (
        (1 if np.max(np.abs(f), initial=0) > 0 else 0)
        <= diag["face_solves"]
        <= diag["budget"]
        <= model.opt.iterations
    )
    assert diag["bound_violation"] <= 2e-7
    assert diag["normalized_kkt"] <= float(np.float32(max(float(model.opt.tolerance), 2e-5)))
    assert diag["normalized_mass_residual"] <= 5e-5
    # No stale tail rows or stale compact identities survive an unload/reset attempt.
    np.testing.assert_array_equal(values[n:], 0)
    np.testing.assert_array_equal(jac[n:], 0)
    np.testing.assert_array_equal(cw[nc:], 0)
    np.testing.assert_array_equal(cj[nc:], 0)
    np.testing.assert_array_equal(cmap[nc:, :2], -1)
    np.testing.assert_array_equal(cmap[nc:, 2], 0)
    return dict(
        matrix=hessian,
        linear=linear,
        force=f,
        kkt=float(np.max(violation, initial=0) / independent_scale),
        normalized_mass_residual=float(mass_residual),
    )


def references(mj, model, states):
    result = {}
    for key, state in states.items():
        data = mj.MjData(model)
        data.qpos[:], data.qvel[:], data.act[:], data.time = state.q, state.v, state.act, state.time
        result[key] = data
    return result


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



def native_fields(mj, model, entry, packet, *, tiny=False):
    result = native_rows(mj, model, entry.q, entry.v, entry.act, packet, entry.time, tiny=tiny)
    result['analytic'] = analytic(model, entry, packet)
    free = bool(np.any(model.jnt_type == 0))
    result['transmission_schema'] = ('native-free-ball-joint-motor-transmission-v1' if free else 'native-joint-motor-transmission-v1')
    result['actuation_schema'] = ('native-free-ball-joint-filterexact-actuation-v1' if free else 'native-joint-filterexact-actuation-v1')
    successor = references(mj, model, {0:entry})[0]
    set_control(successor, packet)
    mj.mj_step(model, successor)
    result['native_successor'] = native_state(successor)
    return result


def compare(row, before, after, packet, expected, model):
    require(row.world == after.world and row.tick == after.tick == before.tick+1, 'world/tick')
    require(row.input_time == before.time and row.output_time == after.time and row.h == H, 'clock')
    exact(row.input_q, before.q, 'complete entry q'); exact(row.input_v, before.v, 'complete entry v')
    exact(row.control, packet, 'requested control')
    for field in ('ctrl','qfrc_applied','xfrc_applied'):
        np.testing.assert_array_equal(getattr(row.stored_control, field), np.asarray(getattr(packet,field),np.float32))
    if 'efc_ids' in row.raw_contact_fields:
        audit=assert_receipt(row,expected,model)
    else:
        require(not row.contacts and not row.constraints and not expected['rows'], 'unconstrained route stays empty')
        for field,(rtol,atol) in TOL.items():
            np.testing.assert_allclose(getattr(row.dynamics,field),expected[field],rtol=rtol,atol=atol,err_msg=field)
        for field in ('qacc','integration_acceleration'):
            np.testing.assert_allclose(getattr(row,field),expected[field],rtol=2e-3,atol=4e-3)
        np.testing.assert_array_equal(row.qfrc_constraint,0)
        audit={}
    np.testing.assert_allclose(row.qfrc_actuator,expected['qfrc_actuator'],**FORCE)
    np.testing.assert_allclose(row.qfrc_actuator,expected['analytic']['generalized_force'],**FORCE)
    transmission=getattr(row,'transmission',None)
    if transmission is not None: check_transmission(row,before,expected)
    if model.na:
        require(after.act.shape==(model.na,), 'complete native activation shape')
        if transmission is not None:
            check_activation(row,before,after,expected)
        else:
            activation=row.actuation
            exact(activation.input_act.astype(float),before.act,'scalar actual input activation')
            exact(activation.output_act.astype(float),after.act,'scalar actual accepted activation')
            for field in ('output_act','activation_derivative','integration_derivative','force_input'):
                np.testing.assert_allclose(getattr(activation,field),expected['analytic'][field],**ACT)
            np.testing.assert_allclose(activation.activation_derivative,expected['act_dot'],**ACT)
            for field in ('raw_force','scalar_force','generalized_force'):
                np.testing.assert_allclose(getattr(activation,field),expected['analytic'][field],**FORCE)
    else:
        require(after.act.shape==(0,), 'NONE activation empty')
    for target in (expected['native_successor'],expected['continuous']):
        np.testing.assert_allclose(after.q,target['q'],rtol=0,atol=Q_ATOL)
        np.testing.assert_allclose(after.v,target['v'],rtol=0,atol=V_ATOL)
        np.testing.assert_allclose(after.act,target['act'],**ACT)
        require(abs(after.time-target['time'])<=TIME_ATOL,'native time')
    for joint,kind in enumerate(model.jnt_type):
        if kind in (0,1):
            qa=int(model.jnt_qposadr[joint])+(3 if kind==0 else 0)
            require(abs(float(after.q[qa:qa+4] @ after.q[qa:qa+4])-1)<=8*np.finfo(np.float32).eps,'native quaternion gate')
    np.testing.assert_allclose(after.v,before.v+float(np.float32(H))*row.integration_acceleration,rtol=0,atol=3e-6)
    # Include ALL noncontact rows in power, not merely body contact forces.
    power=sum(r.force*float(r.jacobian@before.v) for r in row.constraints)
    np.testing.assert_allclose(power,float(row.qfrc_constraint@before.v),rtol=5e-5,atol=5e-5)
    for contact in row.contacts:
        side_power=sum(float(getattr(contact,f'wrench_on_body{side}_world') @ (row.dynamics.jacobian_com[body] @ before.v))
                       for side,body in enumerate(contact.body_ids,1))
        np.testing.assert_allclose(side_power,float(contact.wrench_contact_frame @ (contact.jacobian @ before.v)),rtol=5e-5,atol=5e-5)
    return audit


def advance(runtime, mj, model, packets, refs, journal, label, *, tiny=False):
    full_entry,entry=observations(runtime)
    before=dict(runtime.counters)
    journal.emit('input',label=label,full_entry=full_entry,entry=entry,requested=packets,
                 revision=runtime.checkpoint_revision,counters=before)
    expected={key:native_fields(mj,model,state,packets[key],tiny=tiny) for key,state in entry.items()}
    for key in runtime.world_keys:
        set_control(refs[key],packets[key]); mj.mj_step(model,refs[key])
        expected[key]['continuous']=native_state(refs[key])
    runtime.set_controls(packets)
    journal.emit('attempt',label=label,native=expected,h=H,revision=runtime.checkpoint_revision)
    def forbidden(*args,**kwargs): raise AssertionError('CPU advancement inside actual TYX step')
    try:
        with ExitStack() as guard:
            for symbol in ('mj_step','mj_step1','mj_step2','mj_forward'):
                guard.enter_context(patch.object(mj,symbol,forbidden))
            accepted=runtime.step(H)
        journal.emit('physics-return',label=label,accepted=accepted,counters=dict(runtime.counters))
        full,output=observations(runtime)
        journal.emit('step',label=label,full=full,state=output,revision=runtime.checkpoint_revision,counters=dict(runtime.counters))
        require(accepted==dict.fromkeys(runtime.world_keys,True),'all worlds accepted')
        require(runtime.counters['completed_steps']==before['completed_steps']+1,'one physical batch step')
        require(runtime.counters['accepted_world_steps']==before['accepted_world_steps']+len(runtime.world_keys),'one step per world')
        require(runtime.counters['completed_kernels']>before['completed_kernels'],'actual device launches')
        audits={key:compare(output[key].last_step,entry[key],output[key],packets[key],expected[key],model) for key in runtime.world_keys}
        journal.emit('independent-audit',label=label,audits=audits)
        return output
    except BaseException as primary:
        try:
            journal.emit('step-failure',label=label,error=repr(primary),traceback=traceback.format_exc(),
                         published=runtime.integration_snapshot(),counters=dict(runtime.counters))
        except BaseException as secondary:
            raise BaseExceptionGroup('original step and evidence failure',[primary,secondary])
        raise


def check_capabilities(runtime,model,name):
    caps=runtime.capabilities
    dry=tuple(map(int,np.flatnonzero(model.dof_frictionloss>0)))
    limited=tuple(map(int,np.flatnonzero(model.jnt_limited)))
    if name=='dry-zero':
        require('constraints' not in caps['domain'],'zero dry keeps old no-contact domain')
        return
    free=bool(np.any(model.jnt_type==0)); contact=name.startswith('mixed-')
    expected=('free' if free else 'fixed')+'-ball-'+('mixed-contact-joint-constraints' if contact else 'joint-constraints')+('-filterexact' if model.na else '')+'-v1'
    require(caps['domain']==expected,'explicit composed domain')
    require(caps['joint_limits_supported']==bool(limited) and caps['scalar_joint_limits_supported']==bool(limited),'scalar limits capability')
    require(not caps['ball_angular_limits_supported'] and not caps['g1_supported'],'no broadened BALL angular/G1 claim')
    require(caps['dry_friction_supported']==bool(dry) and caps['mixed_articulated_constraints_supported'],'dry and mixed capability')
    require(caps['floating_mixed_constraints_supported']==free,'FREE capability')
    require(tuple(caps['native_dry_dof_ids'])==dry,'complete native dry addresses')
    require(tuple(caps['native_scalar_limit_joint_ids'])==limited,'complete scalar joint addresses')
    require(caps['dry_constraint_rows']==len(dry) and caps['scalar_limited_joints']==len(limited),'static row census')
    dimensions=(3,) if name=='mixed-pyramid-filter' else (1,) if contact else ()
    require(tuple(caps['contact_dimensions'])==dimensions,'declared dimensions')
    slots=int(caps['candidate_slots']); require(slots==(1 if contact else 0),'complete static candidates')
    bound=len(dry)+2*len(limited)+slots*(4 if dimensions==(3,) else 1)
    require(caps['conservative_efc_bound']==bound,'diagnostic E bound')


def light(row):
    return dict(limits=tuple((r.source_id,r.side_or_edge) for r in row.constraints if r.constraint_type==3),
                dry=tuple((r.source_id,r.force,float(r.jacobian@row.input_v),r.lower,r.upper) for r in row.constraints if r.constraint_type==1),
                contacts=len(row.contacts),active_contacts=sum(c.active for c in row.contacts),
                force=float(sum(c.normal_force for c in row.contacts)),q=np.array(row.input_q),v=np.array(row.input_v))


def events(name, histories, model, mj):
    hinge=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'hinge')
    slide=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'slide')
    for key,history in histories.items():
        if name in ('fixed-lower','fixed-upper','free-filter-limits'):
            initial_sides=((hinge,1),(slide,-1)) if name=='fixed-upper' else ((hinge,-1),(slide,1))
            require(set(initial_sides)<=set(history[0]['limits']),f'{key}: initial two actual limit sides')
            for side in initial_sides: require(any(side not in point['limits'] for point in history[1:]),f'{key}: actual side release {side}')
        elif name=='fixed-margin': require(all(not point['limits'] for point in history),'exact boundary and inward successor have no limit rows')
        elif name=='fixed-overlap': require(set(history[0]['limits'])=={(hinge,-1),(hinge,1),(slide,-1),(slide,1)},'both real scalar sides')
        if name.startswith('dry-'):
            if name=='dry-zero': require(all(not point['dry'] for point in history),'no zero dry rows')
            else:
                require(all(len(point['dry'])==5 for point in history),'all three BALL and two scalar dry rows')
                work=sum(sum(force*velocity for _,force,velocity,_,_ in point['dry']) for point in history)
                require(work<=5e-5,'aggregate signed dry work')
                if name=='dry-stick':
                    require(any(abs(force)>1e-6 and lower+2e-7<force<upper-2e-7 for _,force,_,lower,upper in history[0]['dry']),'nonzero interior dry force')
                else: require(any(abs(force)>=.2-2e-7 for point in history for _,force,_,_,_ in point['dry']),'actual bounded slip')
        if name.startswith('mixed-'):
            require(history[0]['active_contacts']>0 and history[0]['limits'] and history[0]['dry'],'initial actual contact/limit/dry coexist')
            require(any(point['active_contacts']==0 for point in history[16:]),'same trajectory contact unload')


def continuous(name,mode,journal):
    journal.emit('fixture-attempt',name=name,mode=mode,source=source(name))
    mj,model,_,starts,runtime=fixture(name,mode)
    with owned(runtime,journal):
        journal.emit('fixture',name=name,starts=starts,capabilities=dict(runtime.capabilities),counters=dict(runtime.counters))
        check_capabilities(runtime,model,name)
        refs=references(mj,model,runtime.integration_snapshot())
        histories={k:[] for k in runtime.world_keys}
        native_histories={k:[] for k in runtime.world_keys}
        count=2 if name in SHORT else 24
        for tick in range(count):
            packets={k:control(model,tick,i,name) for i,k in reversed(tuple(enumerate(runtime.world_keys)))}
            # Native event records come from the independent continuous state at
            # its actual entry, not device fields or a native reset each step.
            for key in runtime.world_keys:
                state=type('Entry',(),dict(q=refs[key].qpos.copy(),v=refs[key].qvel.copy(),act=refs[key].act.copy(),time=float(refs[key].time)))()
                n=native_rows(mj,model,state.q,state.v,state.act,packets[key],state.time)
                native_histories[key].append(dict(limits=tuple((r['source_id'],r['side_or_edge']) for r in n['rows'] if r['constraint_type']==3),
                    dry=tuple((r['source_id'],r['force'],r['velocity'],r['lower'],r['upper']) for r in n['rows'] if r['constraint_type']==1),
                    active_contacts=sum(c['active'] for c in n['contacts'])))
            output=advance(runtime,mj,model,packets,refs,journal,f'{name}:{tick}',tiny=tick==0 and name in ('fixed-margin','fixed-overlap','dry-stick'))
            for key,state in output.items(): histories[key].append(light(state.last_step))
        native_error=None
        try: events(name,native_histories,model,mj)
        except AssertionError as error: native_error=repr(error)
        journal.emit('events',name=name,actual=histories,native=native_histories,native_event_failure=native_error)
        if native_error is not None: raise AssertionError('predeclared native event hypothesis failed: '+native_error)
        events(name,histories,model,mj)
        return dict(name=name,batch_steps=count,world_steps=count*len(runtime.world_keys))


def original(xml,mode,journal,*,label):
    import mujoco as mj
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime,MuJoCoControl
    require(mj.__version__=='3.3.7','native pin')
    model=mj.MjModel.from_xml_string(xml)
    # These exact original sources all declared h=.001; no source reserialization.
    require(model.opt.timestep==H,'original timestep unchanged')
    compiled=compile_mujoco_xml(xml,require_asset=False)
    journal.emit('original-source',label=label,source=xml,sha256=hashlib.sha256(xml.encode()).hexdigest())
    runtime=MuJoCoRuntime(compiled,mode=mode,implementation='tyx',world_keys=(3,))
    with owned(runtime,journal):
        refs=references(mj,model,runtime.integration_snapshot())
        packet=MuJoCoControl(np.zeros(model.nu),qfrc_applied=np.zeros(model.nv),xfrc_applied=np.zeros((model.nbody,6)))
        for tick in range(2): advance(runtime,mj,model,{3:packet},refs,journal,f'{label}:{tick}')


def lifecycle(name,mode,journal,keys=KEYS,output_path=None):
    from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
    mj,model,_,starts,runtime=fixture(name,mode,keys)
    with owned(runtime,journal):
        refs=references(mj,model,runtime.integration_snapshot())
        for tick in range(3): advance(runtime,mj,model,{k:control(model,tick,i,name) for i,k in enumerate(keys)},refs,journal,'warm')
        held={k:control(model,8,i,name) for i,k in enumerate(keys)}; runtime.set_controls(held)
        saved_state=runtime.integration_snapshot(); raw=runtime.checkpoint().to_bytes()
        capsule=MuJoCoTyxCheckpoint.from_bytes(raw)
        journal.emit('checkpoint',raw=raw,state=saved_state)
        if output_path is not None: (Path(output_path)/(name+'-checkpoint.bin')).write_bytes(raw)
        saved_refs={k:copy.copy(value) for k,value in refs.items()}
        with owned(runtime.clone(),journal) as clone:
            exact(clone.integration_snapshot(),saved_state,'clone complete state')
            a=advance(runtime,mj,model,held,refs,journal,'A')
            clone_a=advance(clone,mj,model,held,{k:copy.copy(value) for k,value in saved_refs.items()},journal,'clone-A')
            exact(a,clone_a,'clone actual successor')
        revision=runtime.checkpoint_revision
        runtime.restore(capsule,expected_revision=revision)
        exact(runtime.integration_snapshot(),saved_state,'complete restore')
        retained=runtime.checkpoint().to_bytes()
        with expect_raises(ValueError): runtime.restore(capsule,expected_revision=revision)
        require(runtime.checkpoint().to_bytes()==retained,'stale CAS leaves banks unchanged')
        refs={k:copy.copy(value) for k,value in saved_refs.items()}
        b=advance(runtime,mj,model,held,refs,journal,'restore-B'); exact(a,b,'A/B actual continuation')
        checkpoint_before=runtime.checkpoint(); runtime.reset((keys[-1],))
        after=runtime.integration_snapshot()
        for key in keys[:-1]: exact(after[key],b[key],'unselected history'); checkpoint_world_equal(checkpoint_before,runtime.checkpoint(),keys.index(key))
        require(after[keys[-1]].tick==0 and after[keys[-1]].last_step is None,'selected reset')
        for field in ('q','v','act'): np.testing.assert_array_equal(getattr(after[keys[-1]],field),np.asarray(getattr(starts[keys[-1]],field),np.float32))
        refs[keys[-1]]=references(mj,model,{keys[-1]:after[keys[-1]]})[keys[-1]]
        journal.emit('partial-reset',state=after,checkpoint=runtime.checkpoint().to_bytes())
        for tick in (16,17): advance(runtime,mj,model,{k:control(model,tick,i,name) for i,k in enumerate(keys)},refs,journal,'reset-successor')
        return dict(name=name,batch_steps=8,world_steps=8*len(keys))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('interpret','cuda'),required=True)
    parser.add_argument('--installed-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=CASES)
    args=parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(),'fresh absolute output')
    require(args.installed_root.is_absolute(),'absolute installed root')
    args.output.mkdir(parents=True)
    result=dict(schema='ball-s-installed-v1',status='FAILED',mode=args.mode,
                selected=[args.case] if args.case else list(CASES),closed=False)
    journal=Journal(args.output/'ball-constraints.jsonl.gz'); failures=[]
    try:
        result['provenance_before']=provenance(args.installed_root.resolve(),preflight=True)
        journal.emit('begin',selected=result['selected'],consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        result['windows']=[continuous(name,args.mode,journal) for name in result['selected']]
        result['lifecycle']=[] if args.case else [lifecycle(name,args.mode,journal,output_path=args.output) for name in ('fixed-lower','free-filter-limits','mixed-pyramid-filter')]
        result['provenance_after']=provenance(args.installed_root.resolve())
        result['status']='PASSED'
    except BaseException as error:
        failures.append(error); result['error']=repr(error); result['traceback']=traceback.format_exc()
        try: journal.emit('failed',error=repr(error),traceback=result['traceback'])
        except BaseException as evidence: failures.append(evidence); result['failure_journal_error']=repr(evidence)
    finally:
        result['closed']=not RETAINED_OWNERS
        try: result['journal']=journal.close()
        except BaseException as error:
            failures.append(error); result['journal_error']=repr(error)
        if failures: result['status']='FAILED'
        result['errors']=[repr(error) for error in failures]
        try: (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        except BaseException as error: failures.append(error)
    if failures: raise BaseExceptionGroup('BALL-S consumer/evidence failures',failures)


if __name__=='__main__': main()