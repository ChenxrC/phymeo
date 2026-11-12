"""Installed FREE+BALL composition; public APIs and independent native reference.

Two predeclared sources, sparse W2, 24 real steps each plus checkpoint/clone/reset
successors. No source tests/helpers, CPU dynamics fallback or changed old gates.
"""

import argparse
import base64
import copy
import gzip
import hashlib
import json
import struct
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

H = 0.001
KEYS = (3, 29)
Q_ATOL, V_ATOL, TIME_ATOL = 1e-4, 1e-3, 2e-6
FORCE = dict(rtol=3e-4, atol=2e-4)
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
    "integration_acceleration": (2e-3, 4e-3),
}

def body(name, kind="ball", child="", pos=".12 -.04 .23"):
    joint = (
        f'<joint name="{name}" type="{kind}" pos=".02 -.01 .03" '
        'damping=".04" armature=".003"/>'
    )
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


def motor(name, target, gear, extra="", gain="1"):
    return (
        f'<general name="{name}" joint="{target}" gear="{gear}" dyntype="none" '
        f'gaintype="fixed" gainprm="{gain}" {extra}/>'
    )


def source(case):
    actors = motor("drive", "ball", "1 -2 .5", gain=".8")
    if case == "mixed":
        child = body("ball", child=body("hinge", "hinge", body("slide", "slide")))
        actors += motor("hinge-drive", "hinge", "-.5")
        actors += motor("slide-drive", "slide", "-2", 'biastype="affine" biasprm=".01 -.3 -.04"')
    elif case == "branches":
        child = body("ball") + body("other", pos="-.18 .07 .16")
        actors += motor("other-drive", "other", "-.3 .4 .2")
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
        self.stream = gzip.open(path, "wb")
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
        for origin in (Path.cwd().resolve(), Path(__file__).resolve().parent):
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


def initial(mj, model, slot=0, sign=False):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    q = model.qpos0.copy()
    for joint, kind in enumerate(model.jnt_type):
        qa = int(model.jnt_qposadr[joint])
        if kind in (mj.mjtJoint.mjJNT_FREE, mj.mjtJoint.mjJNT_BALL):
            adr = qa + (3 if kind == mj.mjtJoint.mjJNT_FREE else 0)
            quat = np.array([1.0, .13 + .02 * joint, -.09, .07])
            q[adr:adr + 4] = quat / np.linalg.norm(quat)
            if sign and slot:
                q[adr:adr + 4] *= -1
        else:
            q[qa] = .03 * (joint + 1)
    if not sign:
        q[:3] += slot * np.array([.04, -.02, .03])
    velocity = np.linspace(-.13, .17, model.nv) * (1 if sign else 1 + .1 * slot)
    return MuJoCoInitialState(q, velocity, time=.125)


def control(model, tick, slot=0):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    phase = 1 if tick < 8 else (-1 if tick < 16 else 0)
    ctrl = phase * np.array([(.18 + .01 * a) * (-1 if a % 2 else 1) for a in range(model.nu)])
    qforce = np.zeros(model.nv)
    qforce[:6] = phase * np.array([.3, -.2, .15, .08, -.06, .04])
    qforce[6:] = phase * np.linspace(.025, .085, model.nv - 6)
    wrench = np.zeros((model.nbody, 6))
    wrench[-1] = phase * (1 + .1 * slot) * np.array([.35, -.25, .4, .025, -.035, .015])
    return MuJoCoControl(ctrl, qfrc_applied=qforce, xfrc_applied=wrench)



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
    )


def native_fields(mj, model, entry, packet):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = entry.q, entry.v, entry.time
    set_control(data, packet)
    mj.mj_forward(model, data)
    require(data.ncon == 0 and data.nefc == 0, "FREE+BALL has no contact/constraint")
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_applied.copy() + data.qfrc_actuator
    for body in range(model.nbody):
        mj.mj_jac(model, data, jac[body, :3], jac[body, 3:], data.xipos[body], body)
        applied += jac[body].T @ packet.xfrc_applied[body]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    smooth = np.linalg.solve(mass, rhs)
    np.testing.assert_allclose(smooth, data.qacc_smooth, rtol=2e-10, atol=2e-9)
    expected = dict(
        input_q=np.array(entry.q),
        input_v=np.array(entry.v),
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
        smooth_acceleration=smooth,
        integration_acceleration=np.linalg.solve(mass + H * np.diag(model.dof_damping), rhs),
        qfrc_actuator=data.qfrc_actuator.copy(),
        qfrc_constraint=data.qfrc_constraint.copy(),
        scalar_force=data.actuator_force.copy(),
        actuator_velocity=data.actuator_velocity.copy(),
        dense_moment=dense_native_moment(mj, model, data),
        transmission=transmission_oracle(mj, model, entry, packet),
    )
    mj.mj_step(model, data)  # Independent same-input native successor, outside TYX guard.
    expected["native_successor"] = native_state(data)
    return expected


def transmission_oracle(mj, model, state, packet):
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


def dense_native_moment(mj, model, data):
    dense = np.zeros((model.nu, model.nv))
    for a in range(model.nu):
        start, count = int(data.moment_rowadr[a]), int(data.moment_rownnz[a])
        dense[a, data.moment_colind[start : start + count]] = data.actuator_moment[
            start : start + count
        ]
    return dense


def check_transmission(row, entry, native, analytic):
    tr = row.transmission
    require(tr.schema == "native-free-ball-joint-motor-transmission-v1", "transmission schema")
    for name in ("ids", "gear", "row_offsets", "dof_indices", "moment"):
        np.testing.assert_array_equal(getattr(tr, name), analytic[name])
    for name in ("clamped_control", "raw_force", "scalar_force", "generalized_force"):
        np.testing.assert_allclose(getattr(tr, name), analytic[name], **FORCE)
    np.testing.assert_allclose(tr.scalar_force, native["scalar_force"], **FORCE)
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
        analytic["dense_moment"] @ entry.v, native["actuator_velocity"], **FORCE
    )
    np.testing.assert_allclose(
        tr.generalized_force @ entry.v,
        tr.scalar_force @ (analytic["dense_moment"] @ entry.v),
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
        require(not value.flags.writeable, "immutable " + name)
        rejected = False
        try:
            value.setflags(write=True)
        except ValueError:
            rejected = True
        require(rejected, "bytes-backed " + name)


def observations(runtime):
    full, integration = runtime.snapshot(), runtime.integration_snapshot()
    require(tuple(full) == tuple(integration) == KEYS, "ordered world identity")
    for key in KEYS:
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
            "coordinate_system",
        ):
            exact(
                getattr(full[key], field),
                getattr(integration[key], field),
                "full/integration " + field,
            )
        require(
            integration[key].observation_kind == "accepted-integration-state", "observation kind"
        )
    return full, integration


def advance(runtime, mj, model, packets, references, journal, label):
    full_entry, entry = observations(runtime)
    from phymeo.backends.mujoco_runtime import MuJoCoControl
    stored = {
        key: MuJoCoControl(
            np.asarray(value.ctrl, np.float32),
            qfrc_applied=np.asarray(value.qfrc_applied, np.float32),
            xfrc_applied=np.asarray(value.xfrc_applied, np.float32),
        )
        for key, value in packets.items()
    }
    journal.emit("request", label=label, full_entry=full_entry, entry=entry, requested=packets, stored=stored, h=H)
    expected = {key: native_fields(mj, model, entry[key], stored[key]) for key in KEYS}
    for key in KEYS:
        set_control(references[key], stored[key])
        mj.mj_step(model, references[key])  # Continuous CPU lane is not overwritten by TYX.
        expected[key]["continuous"] = native_state(references[key])
    runtime.set_controls(packets)
    journal.emit(
        "attempt",
        label=label,
        full_entry=full_entry,
        entry=entry,
        requested=packets,
        native=expected,
    )
    launches = runtime.counters["completed_kernels"]

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU forward/step used inside actual TYX step")

    with patch.object(mj, "mj_step", forbidden), patch.object(mj, "mj_forward", forbidden):
        accepted = runtime.step(H)
        journal.emit("physics-return", label=label, accepted=accepted, counters=dict(runtime.counters))
        integration_only = runtime.integration_snapshot()
        journal.emit("accepted-integration", label=label, state=integration_only)
    full_output, output = observations(runtime)
    exact(integration_only, output, "integration observation without CPU dynamics")
    journal.emit(
        "step",
        label=label,
        accepted=accepted,
        full_output=full_output,
        output=output,
        counters=dict(runtime.counters),
    )
    require(dict(accepted) == dict.fromkeys(KEYS, True), "whole-batch acceptance")
    require(runtime.counters["completed_kernels"] == launches + 3, "FREE+BALL original three stages")
    for key in KEYS:
        row, before, after, oracle = output[key].last_step, entry[key], output[key], expected[key]
        require(row.tick == after.tick == before.tick + 1, "one logical successor")
        require(row.input_time == before.time and row.output_time == after.time, "receipt time")
        require(row.h == H and row.contacts == () and row.constraints == (), "FREE+BALL phase/domain")
        exact(row.input_q, before.q, "actual input q")
        exact(row.input_v, before.v, "actual input v")
        exact(row.dynamics.input_q, row.input_q, "dynamics input q")
        exact(row.dynamics.input_v, row.input_v, "dynamics input v")
        require(row.dynamics.time == row.input_time, "dynamics input time")
        require(row.world == key, "receipt world identity")
        exact(row.control, packets[key], "requested control binding")
        for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
            np.testing.assert_array_equal(
                getattr(row.stored_control, name),
                np.asarray(getattr(packets[key], name), np.float32),
            )
        for name, (rtol, atol) in TOL.items():
            value = (
                row.integration_acceleration
                if name == "integration_acceleration"
                else getattr(row.dynamics, name)
            )
            require(value.shape == oracle[name].shape, "complete dynamics shape " + name)
            np.testing.assert_allclose(value, oracle[name], rtol=rtol, atol=atol, err_msg=name)
            require(not value.flags.writeable, "immutable dynamics " + name)
        np.testing.assert_allclose(row.qacc, oracle["smooth_acceleration"], rtol=2e-3, atol=4e-3)
        np.testing.assert_allclose(row.qfrc_actuator, oracle["qfrc_actuator"], **FORCE)
        check_transmission(row, before, oracle, oracle["transmission"])
        np.testing.assert_array_equal(row.qfrc_constraint, oracle["qfrc_constraint"])
        for native in (oracle["native_successor"], oracle["continuous"]):
            np.testing.assert_allclose(after.q, native["q"], rtol=0, atol=Q_ATOL)
            np.testing.assert_allclose(after.v, native["v"], rtol=0, atol=V_ATOL)
            require(abs(after.time - native["time"]) <= TIME_ATOL, "continuous/native clock")
        for joint in range(model.njnt):
            if model.jnt_type[joint] in (mj.mjtJoint.mjJNT_FREE, mj.mjtJoint.mjJNT_BALL):
                qa = int(model.jnt_qposadr[joint]) + (3 if model.jnt_type[joint] == mj.mjtJoint.mjJNT_FREE else 0)
                require(
                    abs(np.dot(after.q[qa : qa + 4], after.q[qa : qa + 4]) - 1)
                    <= 8 * np.finfo(np.float32).eps,
                    "published unit quaternion",
                )
    return output


def run(case, mode, output_path, journal, installed_root):
    import mujoco as mj

    from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    require(mj.__version__ == "3.3.7", "fixed native oracle version")
    source_xml = source(case)
    model = mj.MjModel.from_xml_string(source_xml)
    require(
        (model.nq, model.nv, model.nu, model.na) == {"mixed": (13, 11, 3, 0), "branches": (15, 12, 2, 0)}[case], "declared FREE/BALL/R/P dimensions"
    )
    initials = {key: initial(mj, model, slot) for slot, key in enumerate(KEYS)}
    compiled = compile_mujoco_xml(source_xml, require_asset=False)
    runtime = MuJoCoRuntime(
        compiled, mode=mode, implementation="tyx", world_keys=KEYS, initial_states=initials
    )
    clone = None
    main_calls = clone_calls = 0
    try:
        require(
            runtime.capabilities["domain"] == "free-ball-joint-motor-no-contact-v1",
            "explicit FREE+BALL domain",
        )
        start = runtime.integration_snapshot()
        references = {}
        for key in KEYS:
            value = mj.MjData(model)
            value.qpos[:], value.qvel[:], value.time = start[key].q, start[key].v, start[key].time
            references[key] = value
        journal.emit(
            "declaration",
            case=case,
            source_xml=source_xml,
            source_sha256=hashlib.sha256(source_xml.encode()).hexdigest(),
            mode=mode,
            keys=KEYS,
            initial=initials,
            stored_initial=start,
            capabilities=runtime.capabilities,
            tolerances=TOL,
            q_atol=Q_ATOL,
            v_atol=V_ATOL,
            time_atol=TIME_ATOL,
            native_joint_type=model.jnt_type.copy(),
            native_qadr=model.jnt_qposadr.copy(),
            native_dadr=model.jnt_dofadr.copy(),
            continuous_steps=24,
            force_gate=FORCE,
        )
        for tick in range(24):
            held = {key: control(model, tick, slot) for slot, key in enumerate(KEYS)}
            advance(runtime, mj, model, held, references, journal, f"{case}/continuous-{tick + 1}")
            main_calls += 1
        before_invalid = runtime.integration_snapshot()
        launches = runtime.counters["completed_kernels"]
        rejected = False
        try:
            runtime.set_controls(
                {KEYS[0]: control(model, 8, 0), KEYS[-1]: MuJoCoControl(np.zeros(model.nu + 1))}
            )
        except ValueError:
            rejected = True
        require(rejected, "wrong control capacity rejected")
        exact(
            before_invalid, runtime.integration_snapshot(), "invalid last-world control is atomic"
        )
        require(
            runtime.counters["completed_kernels"] == launches, "invalid control launched no kernel"
        )
        journal.emit(
            "invalid-control", before=before_invalid, retained=runtime.integration_snapshot()
        )
        capsule = runtime.checkpoint()
        raw = capsule.to_bytes()
        (output_path / f"{case}-checkpoint.bin").write_bytes(raw)
        capsule = MuJoCoTyxCheckpoint.from_bytes(raw)
        saved = runtime.integration_snapshot()
        saved_references = {key: copy.copy(value) for key, value in references.items()}
        journal.emit(
            "checkpoint",
            bytes=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            state=saved,
            native={key: native_state(value) for key, value in references.items()},
        )
        clone = runtime.clone()
        exact(clone.integration_snapshot(), saved, "clone accepted state")
        successor = advance(runtime, mj, model, held, references, journal, f"{case}/successor")
        main_calls += 1
        clone_references = {key: copy.copy(value) for key, value in saved_references.items()}
        clone_result = advance(clone, mj, model, held, clone_references, journal, f"{case}/clone-successor")
        clone_calls += 1
        exact(clone_result, successor, "clone real successor")
        runtime.restore(capsule, expected_revision=runtime.checkpoint_revision)
        exact(runtime.integration_snapshot(), saved, "restored accepted state")
        references = {key: copy.copy(value) for key, value in saved_references.items()}
        repeated = advance(runtime, mj, model, held, references, journal, f"{case}/restored-successor")
        main_calls += 1
        exact(repeated, successor, "restored real successor")
        before_reset = runtime.integration_snapshot()
        runtime.reset((KEYS[-1],))
        after_reset = runtime.integration_snapshot()
        exact(after_reset[KEYS[0]], before_reset[KEYS[0]], "unselected complete state retained")
        selected = after_reset[KEYS[-1]]
        require(
            selected.tick == 0
            and selected.last_step is None
            and selected.time == initials[KEYS[-1]].time,
            "selected original reset",
        )
        np.testing.assert_array_equal(selected.q, np.asarray(initials[KEYS[-1]].q, np.float32))
        np.testing.assert_array_equal(selected.v, np.asarray(initials[KEYS[-1]].v, np.float32))
        fresh = mj.MjData(model)
        fresh.qpos[:] = np.asarray(initials[KEYS[-1]].q, np.float32)
        fresh.qvel[:] = np.asarray(initials[KEYS[-1]].v, np.float32)
        fresh.time = initials[KEYS[-1]].time
        references[KEYS[-1]] = fresh
        journal.emit(
            "partial-original-reset", selected=(KEYS[-1],), before=before_reset, after=after_reset
        )
        final = advance(runtime, mj, model, held, references, journal, f"{case}/partial-reset-successor")
        main_calls += 1
        counts, clone_counts = dict(runtime.counters), dict(clone.counters)
        require(main_calls == 27 and clone_calls == 1, "actual calls not rewound ticks")
        require(
            counts["accepted_world_steps"] == 54 and clone_counts["accepted_world_steps"] == 2,
            "accepted world steps",
        )
        require(
            counts["failed_batches"] == clone_counts["failed_batches"] == 0, "no rejected physics"
        )
        paths = provenance(installed_root)
        for name in (
            "phymeo.backends.mujoco_tyx._floating_model",
            "phymeo.backends.mujoco_tyx._floating_device",
            "phymeo.backends.mujoco_tyx._floating_kernels",
            "phymeo.backends.mujoco_checkpoint",
            "phymeo.backends.mujoco_tyx._joint_transmission",
            "phymeo.backends.mujoco_joint_transmission",
        ):
            require(name in paths["loaded"], "used FREE+BALL module provenance " + name)
        journal.emit(
            "final",
            state=final,
            full=runtime.snapshot(),
            counters=counts,
            clone_counters=clone_counts,
        )
        return dict(
            main_calls=main_calls,
            clone_calls=clone_calls,
            accepted_world_steps=56,
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            main_counters=counts,
            clone_counters=clone_counts,
            final_provenance=paths,
        )
    except BaseException:
        # Keep the original failure even if its optional state capture fails.
        original_traceback = traceback.format_exc()
        (output_path / f"{case}-run-failure.txt").write_text(original_traceback, encoding="utf-8")
        try:
            journal.emit(
                "run-failure-state",
                state=runtime.integration_snapshot(),
                counters=dict(runtime.counters),
                main_calls=main_calls,
                clone_calls=clone_calls,
                traceback=original_traceback,
            )
        except BaseException:
            (output_path / f"{case}-failure-capture.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        raise
    finally:
        primary = sys.exception()
        cleanup = []
        for owner in (clone, runtime):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    cleanup.append(error)
        try:
            journal.emit("closed", case=case, main_calls=main_calls, clone_calls=clone_calls, cleanup_errors=[repr(error) for error in cleanup])
        except BaseException as error:
            cleanup.append(error)
        if cleanup:
            errors = ([primary] if primary is not None else []) + cleanup
            raise BaseExceptionGroup("FREE+BALL execution/cleanup failures", errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-root", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute(), "output must be absolute")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "free-ball.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="installed single FREE + BALL/R/P, direct JOINT NONE motors; na0",
        excluded="activation/contact/limits/dry, BALL affine bias/springs, multiple FREE roots, G1 and training",
    )
    try:
        require(args.installed_root.is_absolute(), "installed-root must be absolute")
        installed_root = args.installed_root.resolve(strict=True)
        summary["initial_provenance"] = provenance(installed_root, preflight=True)
        journal.emit("provenance", details=summary["initial_provenance"])
        summary["cases"] = {}
        for case in ("mixed", "branches"):
            summary["cases"][case] = run(case, args.mode, output, journal, installed_root)
        summary["accepted_world_steps"] = sum(value["accepted_world_steps"] for value in summary["cases"].values())
        require(summary["accepted_world_steps"] == 112, "two declared installed cases")
        summary["status"] = "PASS"
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["exception"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
        (output / "failure.txt").write_text(summary["exception"]["traceback"], encoding="utf-8")
        journal.emit("failure", summary=summary)
        raise
    finally:
        journal.emit("summary", summary=summary)
        summary["trace"] = journal.close()
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()

