"""Installed BALL-A mixed scalar activation consumer; no checkout imports.

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
from contextlib import ExitStack
import struct
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

H = 0.001
KEYS = (7, 91)
Q_ATOL, V_ATOL, TIME_ATOL = 5e-5, 3e-4, 2e-6
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
    "integration_acceleration": (5e-4, 8e-4),
}
# Explicit declaration of the original HINGE_BALL_SLIDE case. No test/helper
# import or fixture discovery is used by the installed consumer.
XML = """<mujoco model="hinge-ball-slide"><compiler angle="radian" fusestatic="false"/>
<option timestep=".001" gravity=".2 -.3 -9.81" integrator="Euler"/>
<worldbody>
<body name="base" pos=".1 -.03 .17" quat=".9238795325112867 0 .3826834323650898 0">
 <joint name="hinge" type="hinge" pos=".015 -.02 .01" damping=".12" armature=".04"
  axis="1 2 -1" ref=".07" springref="-.03" stiffness=".2"/>
 <inertial mass="1.3" pos=".04 -.03 .02" fullinertia=".04 .055 .065 .002 -.001 .003"/>
 <geom type="box" size=".04 .03 .05" contype="0" conaffinity="0"/>
 <body name="middle" pos=".1 -.03 .17" quat=".9238795325112867 0 .3826834323650898 0">
  <joint name="ball" type="ball" pos=".015 -.02 .01" damping=".12" armature=".04"/>
  <inertial mass="1.3" pos=".04 -.03 .02" fullinertia=".04 .055 .065 .002 -.001 .003"/>
  <geom type="box" size=".04 .03 .05" contype="0" conaffinity="0"/>
  <body name="tip" pos=".1 -.03 .17" quat=".9238795325112867 0 .3826834323650898 0">
   <joint name="slide" type="slide" pos=".015 -.02 .01" damping=".12" armature=".04"
    axis="1 2 -1" ref=".07" springref="-.03" stiffness=".2"/>
   <inertial mass="1.3" pos=".04 -.03 .02" fullinertia=".04 .055 .065 .002 -.001 .003"/>
   <geom type="box" size=".04 .03 .05" contype="0" conaffinity="0"/>
  </body>
 </body>
</body>
</worldbody></mujoco>"""


ACT = dict(rtol=3e-5, atol=3e-6)
FORCE = dict(rtol=3e-4, atol=2e-4)
# Original BALL-A filter-early declaration; only scalar JOINT transmissions.
XML = XML.replace("</mujoco>", """<actuator>
<motor name="direct" joint="hinge" gear="-.5"/>
<general name="filter-slide" joint="slide" gear="-2" dyntype="filterexact"
 dynprm=".004" gainprm="1.2" biastype="affine" biasprm=".01 -.3 -.04"
 actearly="true" ctrllimited="true" ctrlrange="-.4 .4"
 actlimited="true" actrange="-.2 .3" forcelimited="true" forcerange="-.7 .7"/>
<general name="filter-hinge" joint="hinge" gear="1.5" group="1"
 dyntype="filterexact" dynprm=".002" gainprm=".8"
 forcelimited="true" forcerange="-.7 .7"/>
<general name="shared-hinge" joint="hinge" gear="-.25"
 dyntype="filterexact" dynprm=".008" gainprm=".6"/>
</actuator></mujoco>""")


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
        require(not any(name == "phymeo" or name.startswith("phymeo.") for name in sys.modules),
                "product imported before installed-origin preflight")
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


def initial(mj, model, slot):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    q = model.qpos0.copy()
    for joint in range(model.njnt):
        qa = int(model.jnt_qposadr[joint])
        if model.jnt_type[joint] == mj.mjtJoint.mjJNT_BALL:
            axis = np.array([1.0 + 0.1 * slot, -2.0, 0.7 + 0.2 * joint])
            axis /= np.linalg.norm(axis)
            angle = 0.37 + 0.03 * slot + 0.05 * joint
            q[qa : qa + 4] = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]
        else:
            q[qa] += 0.11 + 0.013 * slot
    v = np.array(
        [(-1 if i % 2 else 1) * (0.2 + 0.07 * i + 0.01 * slot) for i in range(model.nv)], np.float32
    )
    return MuJoCoInitialState(q, v, act=np.full(model.na, .34 + .01 * slot), time=.125)


def control(model, tick, slot):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    command = .8 if tick < 8 else (-.75 if tick < 16 else 0.)
    u = np.full(model.nu, command + (.015 * slot if command else 0.), np.float32)
    sign = 1 if tick < 8 else -1
    force = np.array([sign * .03 * (i + 1) for i in range(model.nv)], np.float32)
    wrench = np.zeros((model.nbody, 6), np.float32)
    wrench[-1] = np.array([.3, -.2, .4, .01, -.03, .02]) * (1 + .2 * slot)
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


def native_fields(mj, model, entry, packet):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:], data.time = entry.q, entry.v, entry.act, entry.time
    set_control(data, packet)
    mj.mj_forward(model, data)
    require(data.ncon == 0 and data.nefc == 0, "BALL-A has no contact/constraint")
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_applied.copy() + data.qfrc_actuator
    for body in range(model.nbody):
        mj.mj_jac(model, data, jac[body, :3], jac[body, 3:], data.xipos[body], body)
        applied += jac[body].T @ packet.xfrc_applied[body]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    smooth = np.linalg.solve(mass, rhs)
    np.testing.assert_allclose(smooth, data.qacc_smooth, rtol=2e-11, atol=2e-10)
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
        actuator_force=data.actuator_force.copy(),
        act_dot=data.act_dot.copy(),
        actuator_length=data.actuator_length.copy(),
        actuator_velocity=data.actuator_velocity.copy(),
    )
    mj.mj_step(model, data)  # Independent same-input native successor, outside TYX guard.
    expected["native_successor"] = native_state(data)
    expected["analytic"] = analytic(model, entry, packet)
    return expected


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
    expected = {key: native_fields(mj, model, entry[key], packets[key]) for key in KEYS}
    for key in KEYS:
        set_control(references[key], packets[key])
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

    with ExitStack() as guard:
        for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward"):
            guard.enter_context(patch.object(mj, name, forbidden))
        accepted = runtime.step(H)
    full_output, output = observations(runtime)
    journal.emit(
        "step",
        label=label,
        accepted=accepted,
        full_output=full_output,
        output=output,
        counters=dict(runtime.counters),
    )
    require(dict(accepted) == dict.fromkeys(KEYS, True), "whole-batch acceptance")
    require(runtime.counters["completed_kernels"] == launches + 4, "BALL-A activation plus original tree stages")
    for key in KEYS:
        row, before, after, oracle = output[key].last_step, entry[key], output[key], expected[key]
        require(row.tick == after.tick == before.tick + 1, "one logical successor")
        require(row.input_time == before.time and row.output_time == after.time, "receipt time")
        require(row.h == H and row.contacts == () and row.constraints == (), "BALL-A phase/domain")
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
        np.testing.assert_allclose(row.qacc, oracle["smooth_acceleration"], rtol=5e-4, atol=8e-4)
        np.testing.assert_allclose(row.qfrc_actuator, oracle["qfrc_actuator"], **FORCE)
        np.testing.assert_array_equal(row.qfrc_constraint, oracle["qfrc_constraint"])
        a = row.actuation
        require(type(row).__name__ == "MuJoCoTyxActivationStepReceipt", "stateful receipt")
        require(a.schema == "native-scalar-filterexact-actuation-v1", "activation schema")
        require(a.phase == "tyx-euler-entry-actuation-and-next-activation", "actuation phase")
        exact(a.input_act, before.act, "entry activation")
        exact(a.output_act, after.act, "accepted activation")
        exact(a.generalized_force, row.qfrc_actuator, "joint force publication")
        for name, target in oracle["analytic"].items():
            actual = getattr(a, name)
            np.testing.assert_allclose(actual, target, **(ACT if name in ("output_act", "activation_derivative") else FORCE))
            require(not actual.flags.writeable, "immutable actuation " + name)
        np.testing.assert_allclose(a.scalar_force, oracle["actuator_force"], **FORCE)
        np.testing.assert_allclose(a.activation_derivative, oracle["act_dot"], **ACT)
        np.testing.assert_allclose(a.integration_derivative, a.activation_derivative, **ACT)
        for actor in range(model.nu):
            j = int(model.actuator_trnid[actor, 0])
            qa, dof, adr = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), int(model.actuator_actadr[actor])
            require(int(model.jnt_type[j]) in (2, 3), "scalar actor targets")
            np.testing.assert_array_equal(a.ids[actor], [qa, dof, j, adr,
                int(model.actuator_actearly[actor]), 0, int(model.actuator_ctrllimited[actor]),
                int(model.actuator_actlimited[actor]), int(model.actuator_forcelimited[actor])])
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == 1:
                dof = int(model.jnt_dofadr[j])
                np.testing.assert_array_equal(row.qfrc_actuator[dof:dof+3], np.zeros(3))
        for native in (oracle["native_successor"], oracle["continuous"]):
            np.testing.assert_allclose(after.q, native["q"], rtol=0, atol=Q_ATOL)
            np.testing.assert_allclose(after.v, native["v"], rtol=0, atol=V_ATOL)
            np.testing.assert_allclose(after.act, native["act"], **ACT)
            require(abs(after.time - native["time"] ) <= TIME_ATOL, "continuous/native clock")
        for joint in range(model.njnt):
            if model.jnt_type[joint] == mj.mjtJoint.mjJNT_BALL:
                qa = int(model.jnt_qposadr[joint])
                require(
                    abs(np.dot(after.q[qa : qa + 4], after.q[qa : qa + 4]) - 1)
                    <= 8 * np.finfo(np.float32).eps,
                    "published unit quaternion",
                )
    return output


def run(mode, output_path, journal, installed_root):
    import mujoco as mj

    from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    require(mj.__version__ == "3.3.7", "fixed native oracle version")
    model = mj.MjModel.from_xml_string(XML)
    require(
        (model.nq, model.nv, model.nu, model.na) == (6, 5, 4, 3), "declared ball/R/P dimensions"
    )
    require(any(int(model.jnt_qposadr[j]) != int(model.jnt_dofadr[j])
                for j in range(model.njnt) if int(model.jnt_type[j]) in (2, 3)),
            "real scalar qpos/DOF address mismatch")
    initials = {key: initial(mj, model, slot) for slot, key in enumerate(KEYS)}
    compiled = compile_mujoco_xml(XML, require_asset=False)
    runtime = MuJoCoRuntime(
        compiled, mode=mode, implementation="tyx", world_keys=KEYS, initial_states=initials
    )
    clone = None
    main_calls = clone_calls = 0
    try:
        require(
            runtime.capabilities["domain"] == "fixed-ball-scalar-filterexact-no-contact-v1",
            "explicit BALL-A domain",
        )
        start = runtime.integration_snapshot()
        references = {}
        for key in KEYS:
            value = mj.MjData(model)
            value.qpos[:], value.qvel[:], value.act[:], value.time = start[key].q, start[key].v, start[key].act, start[key].time
            references[key] = value
        journal.emit(
            "declaration",
            source_xml=XML,
            source_sha256=hashlib.sha256(XML.encode()).hexdigest(),
            mode=mode,
            keys=KEYS,
            initial=initials,
            stored_initial=start,
            capabilities=runtime.capabilities,
            tolerances=TOL, compilation_source=compiled.source,
            q_atol=Q_ATOL,
            v_atol=V_ATOL,
            time_atol=TIME_ATOL,
            native_joint_type=model.jnt_type.copy(),
            native_qadr=model.jnt_qposadr.copy(),
            native_dadr=model.jnt_dofadr.copy(),
            continuous_steps=24, activation_gate=ACT, force_gate=FORCE,
        )
        for tick in range(24):
            held = {key: control(model, tick, slot) for slot, key in enumerate(KEYS)}
            advance(runtime, mj, model, held, references, journal, f"continuous-{tick + 1}")
            main_calls += 1
        held = {key: control(model, 8, slot) for slot, key in enumerate(KEYS)}
        runtime.set_controls(held)
        capsule = runtime.checkpoint()
        raw = capsule.to_bytes()
        (output_path / "accepted-checkpoint.bin").write_bytes(raw)
        capsule = MuJoCoTyxCheckpoint.from_bytes(raw)
        require(capsule.binding["schema"] == "mujoco-tyx-exact-checkpoint-v2", "activation checkpoint schema")
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
        successor = advance(runtime, mj, model, held, references, journal, "successor")
        main_calls += 1
        clone_references = {key: copy.copy(value) for key, value in saved_references.items()}
        clone_result = advance(clone, mj, model, held, clone_references, journal, "clone-successor")
        clone_calls += 1
        exact(clone_result, successor, "clone real successor")
        restore_revision = runtime.checkpoint_revision
        runtime.restore(capsule, expected_revision=restore_revision)
        restored_bytes = runtime.checkpoint().to_bytes()
        try:
            runtime.restore(capsule, expected_revision=restore_revision)
        except ValueError as error:
            journal.emit("stale-restore-rejected", error=str(error), expected_revision=restore_revision,
                         current_revision=runtime.checkpoint_revision)
        else:
            raise AssertionError("stale restore accepted")
        require(runtime.checkpoint().to_bytes() == restored_bytes, "stale restore leaves all state untouched")
        exact(runtime.integration_snapshot(), saved, "restored accepted state")
        references = {key: copy.copy(value) for key, value in saved_references.items()}
        repeated = advance(runtime, mj, model, held, references, journal, "restored-successor")
        main_calls += 1
        exact(repeated, successor, "restored real successor")
        before_reset = runtime.integration_snapshot()
        runtime.reset((KEYS[-1],))
        after_reset = runtime.integration_snapshot()
        exact(after_reset[KEYS[0]], before_reset[KEYS[0]], "unselected complete state retained")
        selected = after_reset[KEYS[-1]]
        require(
            selected.tick == 0 and selected.last_step is None and selected.time == initials[KEYS[-1]].time,
            "selected original reset",
        )
        np.testing.assert_array_equal(selected.q, np.asarray(initials[KEYS[-1]].q, np.float32))
        np.testing.assert_array_equal(selected.v, np.asarray(initials[KEYS[-1]].v, np.float32))
        np.testing.assert_array_equal(selected.act, np.asarray(initials[KEYS[-1]].act, np.float32))
        fresh = mj.MjData(model)
        fresh.qpos[:] = np.asarray(initials[KEYS[-1]].q, np.float32)
        fresh.qvel[:] = np.asarray(initials[KEYS[-1]].v, np.float32)
        fresh.act[:] = selected.act
        fresh.time = selected.time
        references[KEYS[-1]] = fresh
        journal.emit(
            "partial-original-reset", selected=(KEYS[-1],), before=before_reset, after=after_reset
        )
        for retry in range(2):
            final = advance(runtime, mj, model, held, references, journal, f"partial-reset-successor-{retry}")
            main_calls += 1
        counts, clone_counts = dict(runtime.counters), dict(clone.counters)
        require(main_calls == 28 and clone_calls == 1, "actual calls not rewound ticks")
        require(
            counts["accepted_world_steps"] == 56 and clone_counts["accepted_world_steps"] == 2,
            "accepted world steps",
        )
        require(
            counts["failed_batches"] == clone_counts["failed_batches"] == 0, "no rejected physics"
        )
        paths = provenance(installed_root)
        for name in (
            "phymeo.backends.mujoco_tyx._tree_model",
            "phymeo.backends.mujoco_tyx._tree_device",
            "phymeo.backends.mujoco_tyx._tree_kernels",
            "phymeo.backends.mujoco_checkpoint",
            "phymeo.backends.mujoco_tyx._actuator_model",
            "phymeo.backends.mujoco_tyx._actuator_device",
            "phymeo.backends.mujoco_tyx._actuator_kernels",
        ):
            require(name in paths["loaded"], "used BALL-A module provenance " + name)
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
            accepted_world_steps=58,
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            main_counters=counts,
            clone_counters=clone_counts,
            final_provenance=paths,
        )
    except BaseException:
        # Keep the original failure even if its optional state capture fails.
        original_traceback = traceback.format_exc()
        (output_path / "run-failure.txt").write_text(original_traceback, encoding="utf-8")
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
            (output_path / "failure-capture.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        raise
    finally:
        # Preserve the actual primary failure and still attempt every owned runtime.
        primary = sys.exc_info()[1]
        failures = []
        for name, owner in (("clone", clone), ("runtime", runtime)):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    failures.append((name, error))
        close_record = dict(main_calls=main_calls, clone_calls=clone_calls,
                            failures=[dict(owner=name, type=type(error).__name__, message=str(error))
                                      for name, error in failures])
        try:
            journal.emit("closed", **close_record)
        except BaseException as error:
            failures.append(("close-evidence", error))
        if failures:
            (output_path / "cleanup-failure.json").write_text(
                json.dumps(close_record, indent=2, allow_nan=False), encoding="utf-8")
            if primary is None:
                raise BaseExceptionGroup("installed BALL-A cleanup failed", [error for _, error in failures])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-root", "--expected-install-root", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute(), "output must be absolute")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "ball-scalar-actuation.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="installed fixed-root BALL+R/P; mixed NONE/FILTEREXACT scalar actors, no contacts or limits",
        excluded="ball motors/limits/springs, free-root/contact composition, G1, training",
    )
    installed_root = None
    try:
        require(args.installed_root.is_absolute(), "installed-root must be absolute")
        installed_root = args.installed_root.resolve(strict=True)
        summary["initial_provenance"] = provenance(installed_root, preflight=True)
        journal.emit("provenance", details=summary["initial_provenance"])
        summary.update(run(args.mode, output, journal, installed_root))
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
        try:
            if installed_root is not None:
                summary["final_provenance"] = provenance(installed_root)
            journal.emit("summary", summary=summary)
        except BaseException:
            summary["status"] = "FAIL"
            summary["final_evidence_error"] = traceback.format_exc()
            raise
        finally:
            try:
                summary["trace"] = journal.close()
            except BaseException:
                summary["status"] = "FAIL"
                summary["trace_close_error"] = traceback.format_exc()
                raise
            finally:
                (output / "summary.json").write_text(
                    json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
