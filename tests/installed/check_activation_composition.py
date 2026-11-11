"""Source-external installed ROB2b mixed activation/contact consumer.

Copy outside the checkout; caller supplies the installed package and existing
optional dependencies. No sys.path edits, test imports, private device access,
CPU integration fallback or full source-matrix rerun occurs here.
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
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

H = 0.001
KEYS = (7, 91)
ACT_RTOL, ACT_ATOL = 3e-5, 3e-6
FORCE_RTOL, FORCE_ATOL = 3e-4, 2e-4
Q_ATOL, V_ATOL, TIME_ATOL = 2e-4, 3e-3, 2e-6
XML = """<mujoco model="installed-rob2b-mixed"><compiler angle="radian"/>
<option timestep=".001" gravity="0 0 -9.81" integrator="Euler" solver="Newton"
 iterations="128" tolerance="1e-10" jacobian="dense" cone="pyramidal"/>
<default><joint armature=".01" damping=".04"/>
 <geom condim="3" friction=".6 .02 .003" solref=".02 1" solimp=".8 .9 .002 .5 2"/></default>
<worldbody><geom name="floor" type="plane" size="2 2 .1"/>
 <body name="root" pos="0 0 .099"><freejoint name="rootjoint"/>
  <inertial pos=".012 -.006 .008" mass="2" diaginertia=".09 .11 .13"/>
  <geom name="round" type="sphere" size=".1" contype="1" conaffinity="1"/>
  <body name="link0" pos=".08 .015 .045">
   <joint name="j0" type="hinge" axis="0 1 0" limited="true" range="-.2 .2" margin=".01" frictionloss=".07"/>
   <inertial pos=".015 -.005 .01" mass=".3" diaginertia=".003 .004 .005"/>
   <geom size=".025" contype="0" conaffinity="0"/>
   <body name="link1" pos=".08 .015 .045"><joint name="j1" type="slide" axis="1 0 0"/>
    <inertial pos=".015 -.005 .01" mass=".3" diaginertia=".003 .004 .005"/>
    <geom size=".025" contype="0" conaffinity="0"/>
   </body></body></body></worldbody>
<actuator><motor joint="j0" gear="-.5"/>
 <general joint="j1" gear="-2" dyntype="filterexact" dynprm=".004" gainprm="1.2"
  biastype="affine" biasprm=".01 -.3 -.04" actearly="true" ctrllimited="true"
  ctrlrange="-.4 .4" actlimited="true" actrange="-.2 .3" forcelimited="true" forcerange="-.7 .7"/>
 <general joint="j0" gear="1.5" group="1" dyntype="filterexact" dynprm=".002"
  gainprm=".8" forcelimited="true" forcerange="-.7 .7"/>
 <general joint="j0" gear="-.25" dyntype="filterexact" dynprm=".008" gainprm=".6"/>
</actuator></mujoco>"""


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


def native_state(data, *, force_phase="last-step-input-dynamics"):
    return dict(
        force_phase=force_phase,
        q=data.qpos.copy(),
        v=data.qvel.copy(),
        act=data.act.copy(),
        time=float(data.time),
        ctrl=data.ctrl.copy(),
        qfrc_applied=data.qfrc_applied.copy(),
        xfrc_applied=data.xfrc_applied.copy(),
        qfrc_actuator=data.qfrc_actuator.copy(),
        qacc=data.qacc.copy(),
        actuator_force=data.actuator_force.copy(),
        act_dot=data.act_dot.copy(),
    )


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


def observations(runtime, mj):
    full = runtime.snapshot()

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU forward/step used for integration observation")

    with patch.object(mj, "mj_step", forbidden), patch.object(mj, "mj_forward", forbidden):
        integration = runtime.integration_snapshot()
    require(tuple(full) == tuple(integration), "observation world identity")
    for key in full:
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
        require(
            integration[key].observation_kind == "accepted-integration-state",
            "explicit observation kind",
        )
    return full, integration


def native_fields(mj, model, data):
    # All quantities refer to this mj_forward's input q/v/act/control, not its successor.
    mj.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for b in range(model.nbody):
        mj.mj_jac(model, data, jac[b, :3], jac[b, 3:], data.xipos[b], b)
        applied += jac[b].T @ data.xfrc_applied[b]
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    contacts = []
    for ci in range(data.ncon):
        c = data.contact[ci]
        frame = c.frame.reshape(3, 3).copy()
        ids = tuple(int(g) for g in c.geom)
        bodies = tuple(int(model.geom_bodyid[g]) for g in ids)
        cj = np.zeros((6, model.nv))
        for sign, b in ((-1, bodies[0]), (1, bodies[1])):
            jp = np.zeros((3, model.nv))
            jr = np.zeros_like(jp)
            mj.mj_jac(model, data, jp, jr, c.pos, b)
            cj[:3] += sign * frame @ jp
            cj[3:] += sign * frame @ jr
        force = np.zeros(6)
        mj.mj_contactForce(model, data, ci, force)
        contacts.append(
            dict(
                geometry_ids=ids,
                body_ids=bodies,
                dimension=int(c.dim),
                position=c.pos.copy(),
                frame=frame,
                distance=float(c.dist),
                include_margin=float(c.includemargin),
                active=c.efc_address >= 0,
                jacobian=cj,
                wrench_contact_frame=force,
            )
        )
    return dict(
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
        qfrc_constraint=data.qfrc_constraint.copy(),
        scalar_force=data.actuator_force.copy(),
        derivative=data.act_dot.copy(),
        qfrc_actuator=data.qfrc_actuator.copy(),
        contacts=contacts,
        efc_type=data.efc_type.copy(),
        efc_id=data.efc_id.copy(),
        efc_J=data.efc_J.reshape(data.nefc, model.nv).copy(),
        efc_R=data.efc_R.copy(),
        efc_aref=data.efc_aref.copy(),
        efc_force=data.efc_force.copy(),
    )


def provenance(installed_root, *, preflight=False):
    """Fail on checkout discovery without rewriting the caller's environment."""
    def is_source_entry(path):
        return (
            (path / "pyproject.toml").is_file() and (path / "src" / "phymeo").is_dir()
        ) or (
            path.name.lower() == "src" and (path / "phymeo").is_dir()
            and (path.parent / "pyproject.toml").is_file()
        )

    require((installed_root / "phymeo" / "__init__.py").is_file(), "actual installed package required")
    resolved_paths = [Path(entry or Path.cwd()).resolve() for entry in sys.path]
    require(not any(is_source_entry(path) for path in resolved_paths), "checkout source entry in sys.path")
    if preflight:
        for origin in (Path.cwd().resolve(), Path(__file__).resolve().parent):
            require(not any(is_source_entry(parent) for parent in (origin, *origin.parents)),
                    "consumer and cwd must be outside checkout sources")
    import phymeo

    paths = modules()
    require("phymeo" in paths and Path(phymeo.__file__).resolve().is_relative_to(installed_root), "installed phymeo provenance")
    loaded = {}
    for name, filename in paths.items():
        actual = Path(filename).resolve(strict=True)
        require(actual.is_relative_to(installed_root), "noninstalled module " + name)
        loaded[name] = dict(path=str(actual), sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
    return dict(installed_root=str(installed_root), cwd=str(Path.cwd().resolve()),
                consumer=str(Path(__file__).resolve()), sys_path=[str(p) for p in resolved_paths], loaded=loaded)


def compare_constraints(receipt, expected):
    require(len(receipt.contacts) == len(expected["contacts"]), "complete native contact membership")
    for contact, native in zip(receipt.contacts, expected["contacts"], strict=True):
        for name in ("geometry_ids", "body_ids", "dimension", "active"):
            require(getattr(contact, name) == native[name], "contact identity " + name)
        for name in ("position", "frame", "distance", "include_margin"):
            np.testing.assert_allclose(getattr(contact, name), native[name], rtol=3e-5, atol=4e-5)
        np.testing.assert_allclose(contact.jacobian, native["jacobian"], rtol=2e-4, atol=5e-5)
        np.testing.assert_allclose(contact.wrench_contact_frame, native["wrench_contact_frame"], rtol=3e-3, atol=3e-3)
        np.testing.assert_allclose(contact.force_on_geometry1_world + contact.force_on_geometry2_world, 0, rtol=0, atol=1e-12)
        for side, sign in ((1, -1), (2, 1)):
            body = contact.body_ids[side - 1]
            force = getattr(contact, f"force_on_geometry{side}_world")
            couple = sign * contact.frame.T @ contact.wrench_contact_frame[3:]
            target = np.r_[force, couple + np.cross(contact.position - receipt.dynamics.body_com[body], force)]
            np.testing.assert_allclose(getattr(contact, f"wrench_on_body{side}_world"), target, rtol=3e-5, atol=4e-5)
    require(len(receipt.constraints) == len(expected["efc_type"]), "complete native EFC membership")
    for i, row in enumerate(receipt.constraints):
        require(row.constraint_type == expected["efc_type"][i] and row.source_id == expected["efc_id"][i], "EFC identity")
        np.testing.assert_allclose(row.jacobian, expected["efc_J"][i], rtol=2e-4, atol=5e-5)
        for name, native in (("regularization", "efc_R"), ("reference_acceleration", "efc_aref")):
            np.testing.assert_allclose(getattr(row, name), expected[native][i], rtol=5e-4, atol=4e-4)
        np.testing.assert_allclose(row.force, expected["efc_force"][i], rtol=3e-3, atol=3e-3)

def advance(runtime, mj, model, packets, reference, journal, label):
    full_entry, entry = observations(runtime, mj)
    expected, analytic_expected = {}, {}
    for key in runtime.world_keys:
        native = mj.MjData(model)
        native.qpos[:], native.qvel[:], native.act[:] = entry[key].q, entry[key].v, entry[key].act
        native.time = entry[key].time
        set_control(native, packets[key])
        dynamics = native_fields(mj, model, native)
        native_input = native_state(native, force_phase="same-input-forward-dynamics")
        mj.mj_step(model, native)
        set_control(reference[key], packets[key])
        mj.mj_step(model, reference[key])
        expected[key] = dict(
            input=native_input,
            dynamics=dynamics,
            output=native_state(native),
            continuous=native_state(reference[key]),
        )
        analytic_expected[key] = analytic(model, entry[key], packets[key])
    journal.emit(
        "attempt",
        label=label,
        full_entry=full_entry,
        integration_entry=entry,
        requested_controls=packets,
        native=expected,
        analytic=analytic_expected,
    )
    runtime.set_controls(packets)
    counts_before = dict(runtime.counters)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU forward/step used for device physics/integration observation")

    with patch.object(mj, "mj_step", forbidden), patch.object(mj, "mj_forward", forbidden):
        accepted = runtime.step(H)
        immediate = runtime.integration_snapshot()
    full, current = observations(runtime, mj)
    counts_after = dict(runtime.counters)
    journal.emit(
        "step",
        label=label,
        accepted=accepted,
        immediate=immediate,
        full_output=full,
        integration_output=current,
        counters_before=counts_before,
        counters_after=counts_after,
    )
    exact(immediate, current, "observation did not alter accepted state")
    require(dict(accepted) == dict.fromkeys(runtime.world_keys, True), "complete accepted worlds")
    require(
        counts_after["completed_kernels"] - counts_before["completed_kernels"] == 8,
        "eight real composed activation launches",
    )
    for key, state in current.items():
        receipt = state.last_step
        a = receipt.actuation
        require(state.tick == entry[key].tick + 1, "continuous accepted tick")
        require(abs(state.time - entry[key].time - H) <= TIME_ATOL, "step clock")
        np.testing.assert_array_equal(a.input_act, entry[key].act)
        np.testing.assert_array_equal(a.output_act, state.act)
        require(type(receipt).__name__ == "MuJoCoTyxActivationStepReceipt", "complete stateful receipt schema")
        require(a.schema == "native-scalar-filterexact-actuation-v1", "actuation schema")
        np.testing.assert_array_equal(receipt.input_q, entry[key].q)
        np.testing.assert_array_equal(receipt.input_v, entry[key].v)
        for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
            np.testing.assert_array_equal(getattr(receipt.stored_control, name), np.asarray(getattr(packets[key], name), np.float32))
        for name, target in analytic_expected[key].items():
            act_channel = name in ("output_act", "activation_derivative")
            np.testing.assert_allclose(
                getattr(a, name),
                target,
                rtol=ACT_RTOL if act_channel else FORCE_RTOL,
                atol=ACT_ATOL if act_channel else FORCE_ATOL,
            )
        expected_entry = expected[key]["dynamics"]
        np.testing.assert_allclose(a.activation_derivative, expected_entry["derivative"], rtol=ACT_RTOL, atol=ACT_ATOL)
        np.testing.assert_allclose(
            a.scalar_force, expected_entry["scalar_force"], rtol=FORCE_RTOL, atol=FORCE_ATOL
        )
        np.testing.assert_allclose(
            a.generalized_force,
            expected_entry["qfrc_actuator"],
            rtol=FORCE_RTOL,
            atol=FORCE_ATOL,
        )
        np.testing.assert_array_equal(a.generalized_force, receipt.qfrc_actuator)
        for name, rtol, atol in (
            ("body_position", 3e-5, 4e-5), ("body_rotation", 3e-5, 4e-5),
            ("body_com", 3e-5, 4e-5), ("jacobian_com", 2e-4, 5e-5),
            ("mass_matrix", 3e-4, 5e-5), ("bias", 4e-4, 4e-4),
            ("passive", 4e-4, 4e-4), ("applied", 4e-4, 4e-4),
            ("smooth_acceleration", 2e-3, 4e-3),
        ):
            np.testing.assert_allclose(getattr(receipt.dynamics, name), expected_entry[name], rtol=rtol, atol=atol)
        np.testing.assert_allclose(receipt.qacc, expected_entry["qacc"], rtol=3e-3, atol=1e-2)
        np.testing.assert_allclose(receipt.qfrc_constraint, expected_entry["qfrc_constraint"], rtol=3e-3, atol=3e-3)
        np.testing.assert_array_equal(receipt.qfrc_actuator[:6], np.zeros(6))
        compare_constraints(receipt, expected_entry)
        for lane in ("output", "continuous"):
            native = expected[key][lane]
            np.testing.assert_allclose(state.q, native["q"], rtol=0, atol=Q_ATOL)
            np.testing.assert_allclose(state.v, native["v"], rtol=0, atol=V_ATOL)
            np.testing.assert_allclose(state.act, native["act"], rtol=ACT_RTOL, atol=ACT_ATOL)
            require(abs(state.time - native["time"]) <= TIME_ATOL, "native clock")
    return current


def run(mode, output, journal, installed_root):
    import mujoco as mj

    from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    provenance(installed_root)
    require(mj.__version__ == "3.3.7", "fixed independent MuJoCo oracle version")
    model = mj.MjModel.from_xml_string(XML)
    compiled = compile_mujoco_xml(XML, require_asset=False)
    require((model.nq, model.nv, model.nu, model.na) == (9, 8, 4, 3), "mixed coordinate dimensions")
    initial = {}
    for slot, key in enumerate(KEYS):
        q = model.qpos0.copy()
        q[7:9] = [.205, .02]
        initial[key] = MuJoCoInitialState(q, np.full(model.nv, -.015 - slot * .002),
            act=np.full(model.na, .34 + slot * .01), time=.125)
    def fresh_native(key):
        data = mj.MjData(model)
        for name, value in (
            ("qpos", initial[key].q),
            ("qvel", initial[key].v),
            ("act", initial[key].act),
        ):
            getattr(data, name)[:] = np.asarray(value, np.float32)
        data.time = initial[key].time
        return data

    reference = {key: fresh_native(key) for key in KEYS}

    def packet(tick, slot):
        command = 0.8 if tick < 4 else (-0.75 if tick < 8 else 0.0)
        wrench = np.zeros((model.nbody, 6))
        wrench[-1] = [0.02, -0.015, 0.01, 0.005, -0.003, 0.004]
        # No residual command offset during the predeclared unloading phase.
        ctrl = np.full(model.nu, command + (slot * .015 if command else 0.0))
        return MuJoCoControl(ctrl, qfrc_applied=np.full(model.nv, 0.012), xfrc_applied=wrench)

    def packets(tick):
        return {key: packet(tick, slot) for slot, key in enumerate(KEYS)}

    journal.emit(
        "declaration",
        source_xml=XML,
        initial=initial,
        worlds=KEYS,
        steps=16,
        gates=dict(
            act=(ACT_RTOL, ACT_ATOL),
            force=(FORCE_RTOL, FORCE_ATOL),
            q_atol=Q_ATOL,
            v_atol=V_ATOL,
            time_atol=TIME_ATOL,
        ),
        native_version=mj.__version__,
        source_sha256=hashlib.sha256(XML.encode()).hexdigest(),
        modules=provenance(installed_root),
    )
    runtime = MuJoCoRuntime(
        compiled, world_keys=KEYS, mode=mode, implementation="tyx", initial_states=initial
    )
    clone = None
    main_calls = clone_calls = 0
    try:
        require(runtime.capabilities["domain"] == "tree-pyramidal-mixed-filterexact-v1", "composed capability domain")
        require(runtime.capabilities["checkpoint"] == "exact-accepted-tyx-batch-v2", "stateful exact checkpoint capability")
        journal.emit("initial", capabilities=runtime.capabilities,
                     observations=observations(runtime, mj), modules=provenance(installed_root))
        seen = {key: set() for key in KEYS}
        for tick in range(16):
            current = advance(runtime, mj, model, packets(tick), reference, journal, "main-" + str(tick))
            for key, state in current.items():
                if any(c.normal_force > 0 for c in state.last_step.contacts): seen[key].add("contact")
                if any(r.constraint_type == 3 and r.force > 0 for r in state.last_step.constraints): seen[key].add("limit")
                if any(r.constraint_type == 1 and abs(r.force) > 0 for r in state.last_step.constraints): seen[key].add("dry")
            main_calls += 1
        require(all(value == {"contact", "limit", "dry"} for value in seen.values()), "each world exercised all three real constraint classes")
        held = packets(16)
        runtime.set_controls(held)
        checkpoint = runtime.checkpoint()
        raw = checkpoint.to_bytes()
        (output / "accepted.checkpoint").write_bytes(raw)
        capsule = MuJoCoTyxCheckpoint.from_bytes((output / "accepted.checkpoint").read_bytes())
        require(
            capsule.binding["schema"] == "mujoco-tyx-exact-checkpoint-v2",
            "stateful checkpoint version",
        )
        saved_state = runtime.integration_snapshot()
        saved_reference = {key: copy.copy(value) for key, value in reference.items()}
        journal.emit(
            "checkpoint",
            checkpoint_bytes=raw,
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            binding=capsule.binding,
            state=saved_state,
            native={key: native_state(value) for key, value in reference.items()},
        )
        clone = runtime.clone()
        exact(clone.integration_snapshot(), saved_state, "clone accepted activation/state")
        expected = advance(runtime, mj, model, held, reference, journal, "successor")
        main_calls += 1
        clone_reference = {key: copy.copy(value) for key, value in saved_reference.items()}
        clone_result = advance(clone, mj, model, held, clone_reference, journal, "clone-successor")
        clone_calls += 1
        exact(clone_result, expected, "clone real successor")
        runtime.restore(capsule, expected_revision=runtime.checkpoint_revision)
        exact(runtime.integration_snapshot(), saved_state, "restore exact accepted state")
        reference = {key: copy.copy(value) for key, value in saved_reference.items()}
        repeated = advance(runtime, mj, model, held, reference, journal, "restored-successor")
        main_calls += 1
        exact(repeated, expected, "restored real successor")
        before_reset = runtime.integration_snapshot()
        runtime.reset((KEYS[-1],))
        after_reset = runtime.integration_snapshot()
        exact(after_reset[KEYS[0]], before_reset[KEYS[0]], "unselected world retained")
        reset = after_reset[KEYS[-1]]
        np.testing.assert_array_equal(reset.act, np.asarray(initial[KEYS[-1]].act, np.float32))
        require(reset.tick == 0 and reset.last_step is None, "selected reset publication")
        reference[KEYS[-1]] = fresh_native(KEYS[-1])
        journal.emit("partial-reset", before=before_reset, after=after_reset, selected=(KEYS[-1],))
        final = advance(runtime, mj, model, held, reference, journal, "partial-reset-successor")
        main_calls += 1
        counts, clone_counts = dict(runtime.counters), dict(clone.counters)
        require(main_calls == 19 and clone_calls == 1, "actual calls independent of rewound ticks")
        require(
            counts["accepted_world_steps"] == 38 and clone_counts["accepted_world_steps"] == 2,
            "actual accepted world-step accounting",
        )
        require(
            counts["failed_batches"] == 0 and clone_counts["failed_batches"] == 0,
            "no physics refusal",
        )
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
            accepted_world_steps=40,
            main_counters=counts,
            clone_counters=clone_counts,
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            final_modules=provenance(installed_root),
            constraint_classes={str(key): sorted(value) for key, value in seen.items()},
        )
    finally:
        try:
            if clone is not None:
                clone.close()
        finally:
            runtime.close()
            journal.emit("closed", main_calls=main_calls, clone_calls=clone_calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-root", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute(), "output path must be absolute")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "activation-composition.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="installed free-root mixed activation/contact/limit/dry + exact continuation",
        excluded="ball/tendon/multiple-free-root/general MuJoCo qualification, training, G1",
    )
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
