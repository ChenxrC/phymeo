"""Collect real G2b-A heterogeneous states; this script does not render or replay physics.

Copy this file and heterogeneous_gallery_cases.py together for installed execution.
Native MuJoCo 3.3.7 supplies independent reference data only, outside real TYX steps.
"""
import argparse
import base64
import copy
import gzip
import hashlib
import importlib.util
import json
import math
import struct
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

_CASE_PATH = Path(__file__).with_name("heterogeneous_gallery_cases.py")
_SPEC = importlib.util.spec_from_file_location("g2ba_fixed_cases", _CASE_PATH)
cases = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cases
_SPEC.loader.exec_module(cases)
H, DEVICE_PAYLOAD_LIMIT = cases.H, cases.DEVICE_PAYLOAD_LIMIT
_RETAINED = []
FIXED = {
    "body_position": (2e-5, 3e-5), "body_rotation": (2e-5, 3e-5),
    "body_com": (2e-5, 3e-5), "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (2e-4, 3e-5), "bias": (3e-4, 2e-4),
    "passive": (3e-4, 2e-4), "applied": (3e-4, 2e-4),
    "smooth_acceleration": (5e-4, 8e-4), "integration_acceleration": (5e-4, 8e-4),
}
FREE = {
    "body_position": (3e-5, 4e-5), "body_rotation": (3e-5, 4e-5),
    "body_com": (3e-5, 4e-5), "jacobian_com": (2e-4, 5e-5),
    "mass_matrix": (3e-4, 5e-5), "bias": (4e-4, 4e-4),
    "passive": (4e-4, 4e-4), "applied": (4e-4, 4e-4),
    "smooth_acceleration": (2e-3, 4e-3), "integration_acceleration": (2e-3, 4e-3),
}
MEMORY = dict(host_peak_bytes="UNKNOWN", cuda_context_peak_bytes="UNKNOWN",
              jit_peak_bytes="UNKNOWN", native_arena_peak_bytes="UNKNOWN",
              json_peak_bytes="UNKNOWN", scope="owned CUDA array payload only")


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def encode(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "object array")
        return dict(dtype=value.dtype.str, shape=list(value.shape), bytes_b64=base64.b64encode(value.tobytes()).decode())
    if isinstance(value, np.generic):
        return encode(value.item())
    if is_dataclass(value):
        return dict(type=type(value).__module__ + "." + type(value).__name__, fields={f.name: encode(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, tuple):
        return dict(tuple=[encode(x) for x in value])
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, bytes):
        return dict(bytes_b64=base64.b64encode(value).decode())
    if type(value) is float:
        return dict(float64_bits=struct.pack(">d", value).hex())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError("unsupported evidence type " + type(value).__name__)


def exact(left, right, label):
    def canonical(value):
        if isinstance(value, dict):
            if set(value) == {"mapping_entries"}:
                rows = [[canonical(k), canonical(v)] for k, v in value["mapping_entries"]]
                return {"mapping_entries": sorted(rows, key=lambda row: json.dumps(row[0], sort_keys=True))}
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value
    require(canonical(encode(left)) == canonical(encode(right)), label)


def digest(path):
    hashed = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


class Journal:
    def __init__(self, path, typed=True):
        self.path, self.typed = Path(path), typed
        self.stream = gzip.open(self.path, "xb")
        self.sha = hashlib.sha256()
        self.records = self.bytes = 0

    def emit(self, **payload):
        line = (json.dumps(encode(payload) if self.typed else payload,
                           allow_nan=False, separators=(",", ":")) + "\n").encode()
        self.stream.write(line)
        self.stream.flush()
        self.sha.update(line)
        self.bytes += len(line)
        self.records += 1

    def close(self):
        self.stream.close()
        return dict(file=self.path.name, records=self.records, logical_bytes=self.bytes,
                    logical_sha256=self.sha.hexdigest(), bytes=self.path.stat().st_size,
                    sha256=digest(self.path))


def provenance(installed_root=None, preflight=False):
    root = None if installed_root is None else Path(installed_root).resolve(strict=True)
    if root is not None and preflight:
        require(not any(name == "phymeo" or name.startswith("phymeo.") for name in sys.modules), "product already imported before installed preflight")
        require((root / "phymeo" / "__init__.py").is_file(), "installed package absent")
        for origin in (Path.cwd().resolve(), Path(__file__).resolve().parent):
            require(not any((p / "pyproject.toml").is_file() and (p / "src" / "phymeo").is_dir() for p in (origin, *origin.parents)), "outside script/cwd required")
        for entry in sys.path:
            p = Path(entry or Path.cwd()).resolve()
            require(not ((p / "src" / "phymeo").is_dir() and (p / "pyproject.toml").is_file()) and not (p.name == "src" and (p / "phymeo").is_dir()), "checkout injected into sys.path")
    result = {}
    for name, module in sorted(sys.modules.items()):
        if (name == "phymeo" or name.startswith("phymeo.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve(strict=True)
            require(root is None or path.is_relative_to(root), "noninstalled module " + name)
            result[name] = dict(path=str(path), sha256=digest(path))
    return result


def _quaternion(angle, axis):
    axis = np.asarray(axis, np.float64)
    axis /= np.linalg.norm(axis)
    return np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis]


def initial(model, case):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    q = model.qpos0.copy()
    v = np.array([.035 * (-1 if i % 2 else 1) for i in range(model.nv)])
    for joint, kind in enumerate(model.jnt_type):
        address = int(model.jnt_qposadr[joint])
        if kind == 0:
            q[address+3:address+7] = _quaternion(.15, (1, 2, -1))
            v[:6] = (.04, -.025, .02, .08, -.06, .05)
        elif kind == 1:
            q[address:address+4] = _quaternion(.12, (.2, .7, -.3))
        else:
            q[address] += (.025 if kind == 2 else .08) * (-1 if joint % 2 else 1)
    return MuJoCoInitialState(q, v, act=np.zeros(model.na), time=0.0)


def physical_structure(model):
    names = ("body_parentid", "body_pos", "body_quat", "body_mass", "body_ipos", "body_iquat", "body_inertia", "jnt_type", "jnt_bodyid", "jnt_axis", "jnt_pos", "jnt_qposadr", "jnt_dofadr", "dof_damping", "dof_armature", "geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size", "actuator_trnid", "actuator_gear")
    record = {name: np.asarray(getattr(model, name)).copy() for name in names}
    record["gravity"] = np.asarray(model.opt.gravity).copy()
    return record


def declarations(indices=None):
    import mujoco as mj
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec
    require(mj.__version__ == "3.3.7", "pinned native oracle 3.3.7")
    selected = cases.CASES if indices is None else tuple(cases.CASES[i] for i in indices)
    models, specs, worlds, sources = {}, [], [], []
    for case in selected:
        xml = cases.xml(case)
        model = mj.MjModel.from_xml_string(xml)
        compiled = compile_mujoco_xml(xml, require_asset=False)
        structure = physical_structure(model)
        physical_sha = hashlib.sha256(json.dumps(encode(structure), sort_keys=True).encode()).hexdigest()
        require(float(model.opt.timestep) == H and model.na == 0 and model.nu > 0, "frozen native model dimensions/timestep")
        models[case.world] = model
        specs.append(MuJoCoTyxModelSpec(case.key, compiled))
        worlds.append(MuJoCoTyxWorldSpec(case.world, case.key, initial(model, case)))
        sources.append(dict(world=case.world, key=case.key, family=case.family, xml=xml,
            physical_sha256=physical_sha, physical_structure=structure,
            primary_sha256=hashlib.sha256(compiled.source.primary_xml).hexdigest(),
            archive_sha256=hashlib.sha256(compiled.source.archive_bytes).hexdigest(),
            model_sha256=hashlib.sha256(compiled.source.model_binary).hexdigest(),
            **{name: int(getattr(model, name)) for name in ("nq", "nv", "nu", "na", "nbody")}))
    require(len({item["physical_sha256"] for item in sources}) == len(selected), "genuinely different native physical structures")
    return dict(mj=mj, cases=selected, models=models, specs=tuple(specs), worlds=tuple(worlds), sources=sources)


def commands(model, ordinal):
    from phymeo.backends.mujoco_runtime import MuJoCoControl
    sign = 1 if (ordinal // 64) % 2 == 0 else -1
    ctrl = np.array([sign * (.04 + .01 * (i % 3)) for i in range(model.nu)], np.float32)
    applied = np.zeros(model.nv, np.float32)
    body = np.zeros((model.nbody, 6), np.float32)
    body[-1] = np.asarray((.01, -.015, .02, .001, -.002, .0015), np.float32) * sign
    return MuJoCoControl(ctrl, qfrc_applied=applied, xfrc_applied=body)


def set_native(data, packet):
    for field in ("ctrl", "qfrc_applied", "xfrc_applied"):
        getattr(data, field)[:] = np.asarray(getattr(packet, field), np.float32)


def native_state(data):
    return dict(q=data.qpos.copy(), v=data.qvel.copy(), act=data.act.copy(), time=float(data.time))


def reference(mj, model, entry, packet):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:], data.time = entry.q, entry.v, entry.act, entry.time
    set_native(data, packet)
    mj.mj_forward(model, data)
    require(data.ncon == data.nefc == 0, "no-contact/no-constraint premise")
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    jac = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(model.nbody):
        mj.mj_jac(model, data, jac[body, :3], jac[body, 3:], data.xipos[body], body)
        applied += jac[body].T @ np.asarray(packet.xfrc_applied[body], np.float32)
    rhs = applied + data.qfrc_passive - data.qfrc_bias
    result = dict(body_position=data.xpos.copy(), body_rotation=data.xmat.reshape(model.nbody, 3, 3).copy(),
        body_com=data.xipos.copy(), jacobian_com=jac, mass_matrix=mass, bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(), applied=applied,
        smooth_acceleration=np.linalg.solve(mass, rhs),
        integration_acceleration=np.linalg.solve(mass + H * np.diag(model.dof_damping), rhs),
        qfrc_actuator=data.qfrc_actuator.copy(), qfrc_constraint=data.qfrc_constraint.copy(),
        actuator_force=data.actuator_force.copy())
    moment = np.zeros((model.nu, model.nv))
    for actor in range(model.nu):
        start, count = int(data.moment_rowadr[actor]), int(data.moment_rownnz[actor])
        moment[actor, data.moment_colind[start:start+count]] = data.actuator_moment[start:start+count]
    result["moment"] = moment
    np.testing.assert_allclose(result["smooth_acceleration"], data.qacc, rtol=2e-10, atol=2e-9)
    mj.mj_step(model, data)
    result["successor"] = native_state(data)
    return result


def difference(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0.0))


def compare(case, model, before, after, packet, expected, continuous, *, long_window):
    row, gates = after.last_step, FREE if case.floating else FIXED
    require(row.world == before.world == after.world and row.tick == after.tick == before.tick+1, "row world/tick")
    require(row.input_time == before.time and row.output_time == after.time and row.h == H, "row time/h")
    exact(row.input_q, before.q, "input q binding")
    exact(row.input_v, before.v, "input v binding")
    exact(row.control, packet, "requested controls")
    exact(row.dynamics.input_q, row.input_q, "same-candidate dynamics q")
    exact(row.dynamics.input_v, row.input_v, "same-candidate dynamics v")
    require(row.dynamics.time == before.time, "same-candidate dynamics time")
    for field in ("ctrl", "qfrc_applied", "xfrc_applied"):
        np.testing.assert_array_equal(getattr(row.stored_control, field), np.asarray(getattr(packet, field), np.float32))
    require(all(np.all(np.isfinite(getattr(after, field))) for field in ("q", "v", "act")), "finite complete publication")
    require(len(row.contacts) == 0 and len(getattr(row, "constraints", ())) == 0, "no-contact publication")
    errors = {}
    for field, (rtol, atol) in gates.items():
        actual = row.integration_acceleration if field == "integration_acceleration" else getattr(row.dynamics, field)
        require(actual.shape == expected[field].shape and not actual.flags.writeable, "complete immutable " + field)
        errors[field] = difference(actual, expected[field])
        np.testing.assert_allclose(actual, expected[field], rtol=rtol, atol=atol, err_msg=field)
    force_gate = gates["applied"]
    np.testing.assert_allclose(row.qacc, expected["smooth_acceleration"], rtol=gates["smooth_acceleration"][0], atol=gates["smooth_acceleration"][1])
    np.testing.assert_allclose(row.qfrc_actuator, expected["qfrc_actuator"], rtol=force_gate[0], atol=force_gate[1])
    np.testing.assert_array_equal(row.qfrc_constraint, expected["qfrc_constraint"])
    if hasattr(row, "transmission"):
        tr = row.transmission
        dense = np.zeros((model.nu, model.nv))
        for actor in range(model.nu):
            lo, hi = int(tr.row_offsets[actor]), int(tr.row_offsets[actor+1])
            dense[actor, tr.dof_indices[lo:hi]] = tr.moment[lo:hi]
        np.testing.assert_allclose(dense, expected["moment"], rtol=0, atol=2e-7)
        np.testing.assert_allclose(tr.scalar_force, expected["actuator_force"], rtol=force_gate[0], atol=force_gate[1])
        exact(tr.generalized_force.astype(np.float64), row.qfrc_actuator, "transmission published force")
    qgate, vgate = (1e-4, 1e-3) if case.floating else (5e-5, 3e-4)
    for field, gate in (("q", qgate), ("v", vgate)):
        np.testing.assert_allclose(getattr(after, field), expected["successor"][field], rtol=0, atol=gate)
        errors["same-entry-"+field] = difference(getattr(after, field), expected["successor"][field])
        errors["continuous-"+field] = difference(getattr(after, field), continuous[field])
    np.testing.assert_array_equal(after.act, expected["successor"]["act"])
    require(abs(after.time - expected["successor"]["time"]) <= 2e-6 and abs(after.time-continuous["time"]) <= 2e-6, "clock reference")
    for joint, kind in enumerate(model.jnt_type):
        if kind in (0, 1):
            start = int(model.jnt_qposadr[joint]) + (3 if kind == 0 else 0)
            require(abs(np.linalg.norm(after.q[start:start+4])-1) <= 8*np.finfo(np.float32).eps, "native quaternion norm")
    strict = errors["continuous-q"] <= qgate and errors["continuous-v"] <= vgate
    if long_window:
        require(errors["continuous-q"] <= .01 and errors["continuous-v"] <= .05, "long functional continuous-state gate")
    else:
        require(strict, "short strict continuous-state gate")
    return dict(errors=errors, continuous_strict_pass=strict, q_strict_gate=qgate, v_strict_gate=vgate)


def gpu_identity(batch, mode):
    rows = []
    for index, owner in enumerate(batch._buckets):
        device = owner._device
        record = dict(bucket=index, runtime_type=type(device.runtime).__module__+"."+type(device.runtime).__name__,
                      runtime_device=int(device.runtime.device) if mode == "cuda" else None, mode=mode,
                      counts=dict(device.counts))
        if mode == "cuda":
            values = (*device.q, *device.v)
            require(all(isinstance(value, device.tx.CudaArray) and int(value.device_ptr) != 0 for value in values), "actual live CUDA arrays required")
            record["array_devices"] = [int(value.device_ordinal) for value in values]
            require(all(i == record["runtime_device"] for i in record["array_devices"]), "same runtime/device array ownership")
            require(device.counts["device_allocations"] > 0, "actual CUDA allocation count")
        rows.append(record)
    if mode == "cuda":
        require(len({row["runtime_device"] for row in rows}) == 1, "all resident buckets on one GPU")
    return rows


class Session:
    def __init__(self, mode, steps, output, *, indices=None, installed_root=None, resource_budget=None):
        require(mode in ("interpret", "cuda") and steps in (12, 256), "predeclared mode/window")
        require(steps != 256 or mode == "cuda", "formal full gallery is CUDA only")
        self.mode, self.steps, self.output = mode, steps, Path(output)
        require(self.output.is_absolute() and not self.output.exists(), "fresh absolute output required")
        self.indices, self.installed_root, self.supplied_budget = indices, installed_root, resource_budget
        self.batch = self.journal = self.states_log = None
        self.construction_attempted = False
        self.checked_steps = self.returned_steps = self.attempts = self.strict_failures = 0
        self.first_strict_failure, self.status, self.closed = None, "FAIL", False
        self.max_errors, self.result = {}, None
        self.timings = {name: 0.0 for name in ("compile", "construct", "controls", "step", "observe", "reference", "validation", "save", "cleanup")}
        self.manifest = {}

    @contextmanager
    def timed(self, phase):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[phase] += time.perf_counter()-start

    def emit(self, event, **payload):
        with self.timed("save"):
            self.journal.emit(event=event, **payload)

    def state_frame(self, states, event):
        with self.timed("save"):
            rows = [dict(world=key, model_key=self.by_world[key].key,
                q=states[key].q.tolist(), v=states[key].v.tolist(), act=states[key].act.tolist(),
                time=states[key].time, tick=states[key].tick) for key in self.keys]
            self.states_log.emit(schema="phymeo-g2ba-gallery-state-v1", event=event,
                ordinal=self.returned_steps, worlds=rows)

    def __enter__(self):
        self.output.mkdir(parents=True)
        self.journal = Journal(self.output / "trajectory.jsonl.gz")
        self.states_log = Journal(self.output / "states.jsonl.gz", typed=False)
        try:
            provenance(self.installed_root, preflight=True)
            self.emit("begin", mode=self.mode, steps=self.steps, h=H, memory=MEMORY,
                consumer_sha256=digest(__file__), cases_sha256=digest(_CASE_PATH),
                gates=dict(fixed=FIXED, free=FREE, continuous_functional_q=.01, continuous_functional_v=.05))
            with self.timed("compile"):
                self.declaration = declarations(self.indices)
            self.mj, self.models = self.declaration["mj"], self.declaration["models"]
            self.by_world = {case.world: case for case in self.declaration["cases"]}
            self.keys = tuple(self.by_world)
            source_dir = self.output / "sources"
            source_dir.mkdir()
            public = []
            for row in self.declaration["sources"]:
                path = source_dir / (row["key"] + ".xml")
                path.write_bytes(row["xml"].encode())
                public.append({**{k: v for k, v in row.items() if k not in ("xml", "physical_structure")},
                               "source_file": path.relative_to(self.output).as_posix(), "source_sha256": digest(path)})
            self.manifest = dict(schema="phymeo-g2ba-gallery-manifest-v1", status="RUNNING", qualification="PENDING",
                worlds=public, h=H, steps=self.steps, state_file="states.jsonl.gz", mode=self.mode)
            (self.output / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
            (self.output / "models.json").write_text(json.dumps(encode(self.declaration["sources"])), encoding="utf-8")
            self.emit("declarations", sources=self.declaration["sources"], worlds=self.declaration["worlds"])
            from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget
            from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
            self.budget = self.supplied_budget or MuJoCoTyxResourceBudget(DEVICE_PAYLOAD_LIMIT)
            self.construction_attempted = True
            with self.timed("construct"):
                self.batch = MuJoCoTyxHeterogeneousBatch(self.declaration["specs"], self.declaration["worlds"], mode=self.mode, resource_budget=self.budget)
            self.initial = self.batch.integration_snapshot()
            self.refs = {}
            for key, state in self.initial.items():
                data = self.mj.MjData(self.models[key])
                data.qpos[:], data.qvel[:], data.act[:], data.time = state.q, state.v, state.act, state.time
                self.refs[key] = data
            self.initial_refs = {key: copy.copy(data) for key, data in self.refs.items()}
            require(self.batch.world_keys == self.keys, "complete original world order")
            require(self.batch.capabilities["bucket_count"] == len(self.keys), "one distinct real source per bucket")
            for key, row in zip(self.keys, self.declaration["sources"], strict=True):
                for name in ("primary_sha256", "archive_sha256", "model_sha256"):
                    require(getattr(self.batch.layout[key], name) == row[name], "source/layout binding "+name)
            self.emit("initial", states=self.initial, layout=self.batch.layout, capabilities=self.batch.capabilities,
                resources=self.batch.resource_snapshot(), gpu=gpu_identity(self.batch, self.mode))
            self.state_frame(self.initial, "initial")
            return self
        except BaseException as error:
            self._finish(error)
            raise

    def commands(self, tick):
        return {key: commands(model, tick) for key, model in self.models.items()}

    @contextmanager
    def forbid_native(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("real gallery step called CPU/native/HOST dynamics")
        with ExitStack() as stack:
            for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_fullM", "mj_jac", "mj_rne"):
                stack.enter_context(patch.object(self.mj, name, forbidden))
            stack.enter_context(patch.object(np.linalg, "solve", forbidden))
            stack.enter_context(patch.object(np.linalg, "cholesky", forbidden))
            yield

    def advance(self, ordinal):
        with self.timed("observe"):
            before = self.batch.integration_snapshot()
        packets = self.commands(ordinal)
        with self.timed("controls"):
            self.batch.set_controls(packets)
        self.attempts += 1
        self.emit("attempt", ordinal=ordinal, entry=before, controls=packets, resources=self.batch.resource_snapshot())
        with self.timed("step"), self.forbid_native():
            accepted = self.batch.step(H)
        self.returned_steps += 1
        # Save the actual return and aggregate before any observation/reference/assertion.
        self.emit("returned", accepted=accepted, publication=self.batch.last_step, counters=self.batch.counters)
        with self.timed("observe"):
            after = self.batch.integration_snapshot()
        self.state_frame(after, "accepted")
        expected, continuous = {}, {}
        with self.timed("reference"):
            for key in self.keys:
                expected[key] = reference(self.mj, self.models[key], before[key], packets[key])
                set_native(self.refs[key], packets[key])
                self.mj.mj_step(self.models[key], self.refs[key])
                continuous[key] = native_state(self.refs[key])
        self.emit("raw-reference", ordinal=ordinal, states=after, native_same_entry=expected, native_continuous=continuous)
        audited = {}
        with self.timed("validation"):
            require(dict(accepted) == dict.fromkeys(self.keys, True), "every world atomically accepted")
            pub = self.batch.last_step
            require(pub.revision_after == pub.revision_before+1 and pub.world_keys == self.keys, "one aggregate publication")
            for key in self.keys:
                exact(after[key], pub.states[key], "snapshot equals actual aggregate state")
                audited[key] = compare(self.by_world[key], self.models[key], before[key], after[key], packets[key], expected[key], continuous[key], long_window=self.steps == 256)
                for field, value in audited[key]["errors"].items():
                    self.max_errors[field] = max(value, self.max_errors.get(field, 0.0))
                if not audited[key]["continuous_strict_pass"]:
                    self.strict_failures += 1
                    if self.first_strict_failure is None:
                        self.first_strict_failure = dict(ordinal=ordinal, world=key, **audited[key])
        self.checked_steps += 1
        self.emit("checked", ordinal=ordinal, audits=audited, resources=self.batch.resource_snapshot())
        return after

    def mark_pass(self):
        self.status = "PASS"

    def _finish(self, primary):
        errors = [] if primary is None else [primary]
        resources, counters, gpu = None, None, None
        if self.batch is not None:
            try:
                resources, counters = self.batch.resource_snapshot(), dict(self.batch.counters)
                gpu = gpu_identity(self.batch, self.mode)
            except BaseException as error:
                errors.append(error)
            try:
                with self.timed("cleanup"):
                    self.batch.close()
                self.closed = True
            except BaseException as error:
                _RETAINED.append(self.batch)
                errors.append(error)
        else:
            self.closed = None if self.construction_attempted else True  # Failed construction has no returned owner to prove closed.
        qualifier = "STRICT_FAIL_RECORDED" if self.strict_failures else "STRICT_PASS"
        result = dict(status=self.status if not errors else "FAIL", qualification=qualifier,
            functional_status="FUNCTIONAL_PASS" if self.status == "PASS" and not errors else "FAIL",
            mode=self.mode, h=H, expected_steps=self.steps, attempted_steps=self.attempts,
            returned_steps=self.returned_steps, checked_steps=self.checked_steps,
            accepted_world_steps=self.returned_steps * len(getattr(self, "keys", ())),
            model_count=len(getattr(self, "keys", ())), bucket_count=len(getattr(self, "keys", ())),
            closed=self.closed, strict_failure_count=self.strict_failures,
            first_strict_failure=self.first_strict_failure, maximum_errors=self.max_errors,
            resources=encode(resources), counters=counters, gpu=gpu, memory=MEMORY,
            timings=self.timings, errors=[repr(error) for error in errors], states_file="states.jsonl.gz")
        for name, journal in (("trajectory", self.journal), ("states", self.states_log)):
            if journal is not None:
                try:
                    if name == "trajectory":
                        journal.emit(event="closed", result=result, origins=provenance(self.installed_root))
                    result[name] = journal.close()
                except BaseException as error:
                    errors.append(error)
                    try:
                        journal.stream.close()
                    except BaseException as cleanup:
                        errors.append(cleanup)
        result["state_records"] = self.states_log.records if self.states_log is not None else 0
        result["status"] = self.status if not errors else "FAIL"
        result["errors"] = [repr(error) for error in errors]
        result["functional_status"] = "FUNCTIONAL_PASS" if result["status"] == "PASS" else "FAIL"
        self.manifest.update(status=result["status"], qualification=qualifier,
            functional_status=result["functional_status"], actual_state_records=result["state_records"],
            state_sha256=result.get("states", {}).get("sha256"), state_bytes=result.get("states", {}).get("bytes"))
        before_final_writes = len(errors)
        for name, data in (("summary.json", result), ("manifest.json", self.manifest)):
            try:
                (self.output / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
            except BaseException as error:
                errors.append(error)
        if len(errors) != before_final_writes:
            result.update(status="FAIL", functional_status="FAIL", errors=[repr(error) for error in errors])
            self.manifest.update(status="FAIL", functional_status="FAIL")
            # One bounded best-effort failure marker for each independent output;
            # never retry close, and never discard the original write exception.
            for name, data in (("summary.json", result), ("manifest.json", self.manifest)):
                try:
                    (self.output / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
                except BaseException as error:
                    errors.append(error)
        self.result = result
        if errors and (len(errors) > 1 or primary is None):
            raise BaseExceptionGroup("gallery primary/observation/cleanup/evidence failures", errors)

    def __exit__(self, exc_type, exc, traceback):
        self._finish(exc)
        return False


def run(mode, steps, output, installed_root=None):
    with Session(mode, steps, output, installed_root=installed_root) as session:
        for ordinal in range(steps):
            session.advance(ordinal)
        require(session.checked_steps == steps and session.returned_steps == steps, "full declared window")
        require(session.states_log.records == steps+1, "initial plus every actual returned state")
        session.mark_pass()
    return session.result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--steps", type=int, choices=(12, 256), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-root", type=Path)
    args = parser.parse_args()
    result = run(args.mode, args.steps, args.output, args.installed_root)
    print(json.dumps({key: result[key] for key in ("status", "qualification", "returned_steps", "checked_steps", "closed")}, sort_keys=True))


if __name__ == "__main__":
    main()
