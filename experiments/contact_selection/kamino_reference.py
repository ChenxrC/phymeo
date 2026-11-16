"""Pinned Newton 1.7.0 hard-contact dynamics; no Newton import until run.

This is an external experiment, not a Phymeo runtime or an area-pressure law.
Contact geometry and forces belong to the input of each accepted Newton step.
Only the commanded external wrench changes; the box pose is never prescribed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import numpy as np

SCHEMA = "phymeo-kamino-selection-v1"
CASES = (("padmm", 0.001), ("padmm", 0.0005), ("dvi", 0.001), ("dvi", 0.0005))


class UnsupportedReference(RuntimeError):
    """The pinned external environment cannot represent this requested case."""


def _finite(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != shape:
        raise ValueError(f"{name}: finite numeric shape {shape} required")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name}: nonfinite value")
    return result


def _rotation(pose):
    p = _finite(pose, (7,), "body pose xyz+xyzw")
    x, y, z, w = p[3:]
    if abs(float(np.dot(p[3:], p[3:])) - 1.0) > 5e-5:
        raise ValueError("body quaternion must be unit (FP32 tolerance); no normalization")
    return p[:3], np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def case_spec(backend, dt):
    """Return an owned fixed case specification; no environment access."""
    if backend not in ("padmm", "dvi") or type(dt) not in (int, float) or dt not in (0.001, 0.0005):
        raise ValueError("only the four preregistered backend/time-step combinations are supported")
    return {
        "id": f"{backend}-{int(round(dt * 1000000))}us",
        "backend": backend,
        "dt": dt,
        "duration_s": 0.6,
        "steps": 600 if dt == 0.001 else 1200,
        "mass_kg": 0.5,
        "box_half_extents_m": [0.02, 0.02, 0.01],
        "table_half_extents_m": [0.04, 0.04, 0.005],
        "table_position_m": [0, 0, -0.005],
        "initial_pose_xyzw": [0, 0, 0.011, 0, 0, 0, 1],
        "gravity_m_s2": [0, 0, -9.81],
        "friction": 0.5,
        "restitution": 0.0,
        "margin_m": 0.0,
        "gap_m": 0.0,
        "contact_capacity": 32,
        "force_world_n": [4, 0, 0],
        "application_local_m": [0, 0, 0.01],
        "force_interval_s": [0.2, 0.4],
        "integrator": "euler",
        "qualification": "EXTERNAL_HARD_CONTACT_DYNAMICS_NOT_AREA_PRESSURE",
        "pose_convention": "xyz+xyzw; body origin world pose, COM-local offset saved separately",
        "wrench_units": "world force N, moment N*m; public wrench acts on shape0 about input COM",
    }


def external_wrench(entry_time, body_pose_xyzw):
    """World force and COM torque from the same input pose, not endpoint pose."""
    if type(entry_time) not in (int, float) or not math.isfinite(entry_time) or entry_time < 0:
        raise ValueError("entry time must be nonnegative finite seconds")
    position, rotation = _rotation(body_pose_xyzw)
    arm = rotation @ np.array([0.0, 0.0, 0.01])
    force = np.array([4.0 if 0.2 <= entry_time < 0.4 else 0.0, 0.0, 0.0])
    return force.tolist(), np.cross(arm, force).tolist(), (position + arm).tolist()


def decode_contacts(entry_body_q, body_com, shape_body, raw):
    """Decode pinned public contacts without substituting the output pose.

    Public wrench0 is kept verbatim. Separate point-derived moments expose the
    static-body0 zero-torque export convention. Side 1 uses its own point1;
    separated contact anchors are not silently collapsed to a common point.
    """
    if not 1 <= len(entry_body_q) <= 128 or not 1 <= len(shape_body) <= 256:
        raise ValueError("bounded body/shape counts required")
    poses = _finite(entry_body_q, (len(entry_body_q), 7), "input body poses")
    coms = _finite(body_com, (len(poses), 3), "body-local COMs")
    if any(
        type(b) not in (int, np.int32, np.int64) or not -1 <= b < len(poses) for b in shape_body
    ):
        raise ValueError("shape ownership outside body array")
    count = raw["count"]
    if type(count) is not int or not 0 <= count <= 32 or type(raw["overflow"]) is not int:
        raise ValueError("invalid contact count/overflow type")
    if raw["overflow"] != 0:
        raise ValueError("native contact capacity overflow; truncated publication is rejected")
    capacity = len(raw["shape0"])
    if not count <= capacity <= 32:
        raise ValueError("contact capacity outside fixed bound")
    fields = {}
    for key, width in (("point0", 3), ("point1", 3), ("normal", 3), ("wrench0", 6)):
        fields[key] = _finite(raw[key], (capacity, width), key)
    for key in ("margin0", "margin1"):
        fields[key] = _finite(raw[key], (capacity,), key)
    for key in ("shape0", "shape1"):
        if len(raw[key]) != capacity or any(type(v) is not int for v in raw[key]):
            raise ValueError("contact shape IDs require matching integer arrays")
    transforms = [_rotation(p) for p in poses]

    def point_and_com(body, point):
        if body == -1:
            return point.copy(), np.zeros(3)
        pos, rot = transforms[body]
        return pos + rot @ point, pos + rot @ coms[body]

    decoded = []
    for i in range(count):
        s0, s1 = raw["shape0"][i], raw["shape1"][i]
        if not 0 <= s0 < len(shape_body) or not 0 <= s1 < len(shape_body) or s0 == s1:
            raise ValueError("active contact has invalid shape identity")
        b0, b1 = int(shape_body[s0]), int(shape_body[s1])
        if b0 == b1:
            raise ValueError("active contact must join different bodies")
        n = fields["normal"][i]
        if abs(float(np.dot(n, n)) - 1.0) > 2e-4:
            raise ValueError("active contact normal is not unit")
        p0, c0 = point_and_com(b0, fields["point0"][i])
        p1, c1 = point_and_com(b1, fields["point1"][i])
        f0 = fields["wrench0"][i, :3]
        f1 = -f0
        decoded.append(
            {
                "index": i,
                "shape0": s0,
                "shape1": s1,
                "body0": b0,
                "body1": b1,
                "point0_world": p0.tolist(),
                "point1_world": p1.tolist(),
                "com0_world": c0.tolist(),
                "com1_world": c1.tolist(),
                "normal_0_to_1_world": n.tolist(),
                "public_wrench0_world_com": fields["wrench0"][i].tolist(),
                "force0_world": f0.tolist(),
                "force1_world": f1.tolist(),
                "moment0_com_from_point": np.cross(p0 - c0, f0).tolist(),
                "moment1_com_from_point": np.cross(p1 - c1, f1).tolist(),
                "moment0_origin": np.cross(p0, f0).tolist(),
                "moment1_origin": np.cross(p1, f1).tolist(),
                "static_side_reference": "world-origin-for-reconstruction; public-static-torque-preserved",
            }
        )
    return decoded


def summarize_records(records, dt, expected_steps):
    """Physical gates and solver convergence remain separate, including prefixes."""
    case_spec("padmm", dt)
    if (
        type(expected_steps) is not int
        or not 1 <= expected_steps <= 1200
        or len(records) > expected_steps
    ):
        raise ValueError("bounded expected trajectory length required")
    for i, row in enumerate(records):
        if (
            row["phase"] != "step-input-contact"
            or type(row["input_tick"]) is not int
            or type(row["output_tick"]) is not int
            or row["input_tick"] != i
            or row["output_tick"] != i + 1
            or not math.isfinite(row["input_time"])
            or not math.isfinite(row["output_time"])
            or abs(row["input_time"] - i * dt) > 1e-12
            or abs(row["output_time"] - (i + 1) * dt) > 1e-12
        ):
            raise ValueError("non-contiguous input/output phase")
        for state in (row["entry"], row["exit"]):
            _rotation(state["body_q"][0])
            _finite(state["body_qd"], (1, 6), "body velocity")
        _finite(row["contact_force_on_box"], (3,), "contact force")
        if type(row["active_contacts"]) is not int or not 0 <= row["active_contacts"] <= 32:
            raise ValueError("invalid loaded contact count")
        if not row["solver_status"] or any(
            s["converged"] not in (False, True) for s in row["solver_status"]
        ):
            raise ValueError("missing or invalid convergence status")
    settled = [r for r in records if 0.15 <= r["input_time"] < 0.2]
    tail = [r for r in records if r["input_time"] >= 0.55]
    complete = len(records) == expected_steps
    first = records[0]["entry"]["body_q"][0] if records else None
    last = records[-1]["exit"]["body_q"][0] if records else None
    mean_force = (
        sum(r["contact_force_on_box"][2] for r in settled) / len(settled) if settled else None
    )
    zs = [r["entry"]["body_q"][0][2] for r in settled]
    z_range = max(zs) - min(zs) if zs else None
    end_vz = abs(settled[-1]["exit"]["body_qd"][0][2]) if settled else None
    tilt = max(
        (
            math.acos(float(np.clip(_rotation(r["exit"]["body_q"][0])[1][2, 2], -1, 1)))
            for r in records
        ),
        default=0.0,
    )
    loaded = [r["input_time"] for r in records if r["active_contacts"] > 0]
    gates = {
        "complete_finite": complete,
        "settle_support": mean_force is not None and abs(mean_force - 4.905) <= 0.4905,
        "settle_height": z_range is not None and z_range < 0.0001,
        "settle_velocity": end_vz is not None and end_vz < 0.001,
        "tilted": tilt > math.pi / 180,
        "translated": last is not None and last[0] - first[0] > 0.06,
        "unloaded_final_50ms": complete
        and len(tail) == int(round(0.05 / dt))
        and all(r["active_contacts"] == 0 for r in tail),
        "fallen": last is not None and first[2] - last[2] > 0.01,
    }
    return {
        "steps": len(records),
        "settle_force_mean_n": mean_force,
        "settle_com_z_range_m": z_range,
        "settle_end_abs_vz_m_s": end_vz,
        "max_tilt_rad": tilt,
        "displacement_x_m": last[0] - first[0] if last else None,
        "com_drop_m": first[2] - last[2] if last else None,
        "vertical_impulse_n_s": sum(r["contact_force_on_box"][2] * dt for r in records),
        "last_loaded_contact_time_s": loaded[-1] if loaded else None,
        "gates": gates,
        "functional_pass": all(gates.values()),
        "solver_converged": bool(records)
        and all(s["converged"] for r in records for s in r["solver_status"]),
        "nonconverged_steps": sum(
            not all(s["converged"] for s in r["solver_status"]) for r in records
        ),
        "loaded_contact_definition": "box-table point force norm greater than 1e-8 N",
    }


def validate_output(output):
    path = Path(output)
    if not path.is_absolute() or path.exists():
        raise ValueError("output must be an absolute, new directory")
    return path


def _json_value(value, diagnostic=False):
    if isinstance(value, np.void) and value.dtype.names:
        return {k: _json_value(value[k], diagnostic) for k in value.dtype.names}
    if isinstance(value, np.ndarray):
        return (
            [_json_value(v, diagnostic) for v in value]
            if value.ndim
            else _json_value(value.item(), diagnostic)
        )
    if isinstance(value, np.generic):
        return _json_value(value.item(), diagnostic)
    if isinstance(value, dict):
        return {str(k): _json_value(v, diagnostic) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v, diagnostic) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if diagnostic:
            return {"nonfinite_float": repr(value)}
        raise ValueError("nonfinite output cannot be published as a finite trajectory")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unserializable output type {type(value).__name__}")


def _write_json(path, value, diagnostic=False):
    path.write_text(
        json.dumps(_json_value(value, diagnostic), allow_nan=False, indent=2), encoding="utf-8"
    )


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment():
    versions, sources = {}, {}
    for name in ("newton", "warp-lang", "numpy"):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
            continue
        versions[name] = dist.version
        for item in dist.files or ():
            text = str(item).replace("\\", "/")
            if (
                text.endswith(".dist-info/METADATA")
                or (
                    name == "newton"
                    and text.startswith("newton/_src/solvers/kamino/")
                    and text.endswith(".py")
                )
                or text
                in (
                    "newton/_src/sim/builder.py",
                    "newton/_src/sim/state.py",
                    "newton/_src/sim/contacts.py",
                    "newton/_src/solvers/solver.py",
                    "warp/_src/context.py",
                )
            ):
                path = Path(dist.locate_file(item))
                sources[f"{name}:{text}"] = _hash(path)
    sources["driver"] = _hash(Path(__file__))
    result = {
        "versions": versions,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "source_sha256": sources,
        "source_scope": "metadata, complete Kamino Python subtree, named public interfaces and driver",
    }
    result["binding_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    return result


def _build(spec, device):
    if metadata.version("newton") != "1.7.0" or metadata.version("warp-lang") != "1.17.0":
        raise UnsupportedReference("requires pinned newton 1.7.0 and warp-lang 1.17.0")
    import newton
    import warp as wp
    from newton.solvers import SolverKamino

    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise UnsupportedReference("this preregistered suite requires real CUDA")
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0, 0, -9.81))
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world(label="finite-support")
    m, x, y, z = 0.5, 0.04, 0.04, 0.02
    diagonal = [m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12]
    inertia = wp.mat33(diagonal[0], 0, 0, 0, diagonal[1], 0, 0, 0, diagonal[2])
    box = builder.add_body(
        label="box",
        mass=m,
        inertia=inertia,
        lock_inertia=True,
        xform=wp.transform(wp.vec3(0, 0, 0.011), wp.quat_identity()),
    )
    shape_cfg = builder.ShapeConfig(
        mu=0.5, restitution=0, margin=0, gap=0, mu_torsional=0, mu_rolling=0
    )
    box_shape = builder.add_shape_box(box, hx=0.02, hy=0.02, hz=0.01, cfg=shape_cfg, label="box")
    table_shape = builder.add_shape_box(
        -1,
        hx=0.04,
        hy=0.04,
        hz=0.005,
        cfg=shape_cfg,
        label="table",
        xform=wp.transform(wp.vec3(0, 0, -0.005), wp.quat_identity()),
    )
    builder.end_world()
    model = builder.finalize(device=device)
    config = SolverKamino.Config(
        dynamics_solver=spec["backend"],
        use_collision_detector=True,
        use_fk_solver=False,
        integrator="euler",
    )
    config.collision_detector.pipeline = "primitive"
    config.collision_detector.max_contacts_per_world = 32
    if spec["backend"] == "padmm":
        config.padmm.primal_tolerance = 1e-6
        config.padmm.dual_tolerance = 1e-6
        config.padmm.compl_tolerance = 1e-6
        config.padmm.max_iterations = 200
        config.padmm.rho_0 = 0.1
        config.dynamics.preconditioning = True
    solver = SolverKamino(model, config)
    contacts = newton.CollisionPipeline(model, rigid_contact_max=32).contacts()
    observables = solver.observables({solver.ObservableKind.CONTACT_F}, contacts=contacts)
    wp.synchronize_device(device)
    return wp, model, solver, contacts, observables, config, shape_cfg, box, box_shape, table_shape


def _snapshot(state):
    result = {
        name: getattr(state, name).numpy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")
    }
    return _json_value(result)


def _raw_contacts(contacts, obs, solver):
    # Exactly one read of each array; retain full capacity, not only active rows.
    result = {
        "count": int(contacts.rigid_contact_count.numpy()[0]),
        "overflow": int(solver._contacts_kamino._data._contact_overflow_warning_emitted.numpy()[0]),
    }
    for short, full in (
        ("shape0", "shape0"),
        ("shape1", "shape1"),
        ("point0", "point0"),
        ("point1", "point1"),
        ("normal", "normal"),
        ("margin0", "margin0"),
        ("margin1", "margin1"),
    ):
        result[short] = getattr(contacts, "rigid_contact_" + full).numpy()
    result["wrench0"] = obs.contact_f.numpy()
    return result


def _frame(
    state, tick, dt, *, contacts=None, force=(0, 0, 0), torque=(0, 0, 0), total=(0, 0, 0), active=0
):
    return {
        "t": tick * dt,
        "tick": tick,
        "body_pose": state["body_q"][0],
        "linear_velocity": state["body_qd"][0][:3],
        "angular_velocity": state["body_qd"][0][3:],
        "contact_force_on_box": list(total),
        "input_force": list(force),
        "input_torque": list(torque),
        "active_contacts": int(active),
        "contacts": list(contacts) if contacts is not None else [],
        "contact_data_valid": contacts is not None,
        "phase": "input-pose-with-same-step-contact" if contacts is not None else "state-snapshot",
    }


def _run_case(directory, spec, device):
    directory.mkdir()
    records, frames, step_seconds, read_seconds, save_seconds = [], [], [], [], []
    result = {
        "id": spec["id"],
        "backend": spec["backend"],
        "dt": spec["dt"],
        "status": "FAILED",
        "file": f"{spec['id']}/records.jsonl.gz",
        "frames": f"{spec['id']}/frames.json",
    }
    phase, attempt, last_state = "construct", {}, None
    started = time.perf_counter()
    construction_s, preparation_s, last_state_tick = None, 0.0, 0
    try:
        wp, model, solver, contacts, obs, config, shape_cfg, box, box_shape, table_shape = _build(
            spec, device
        )
        state_in, state_out, control = model.state(), model.state(), model.control()
        wp.synchronize_device(device)
        construction_s = time.perf_counter() - started
        phase = "model-observation"
        if box != 0 or model.body_count != 1:
            raise ValueError("fixed model must have exactly one free dynamic box")
        static = {
            name: _json_value(getattr(model, name).numpy())
            for name in (
                "body_q",
                "body_mass",
                "body_inv_mass",
                "body_com",
                "body_inertia",
                "body_inv_inertia",
                "shape_body",
                "shape_transform",
                "shape_scale",
                "shape_type",
                "shape_flags",
            )
        }
        static.update(
            {
                "body_labels": list(model.body_label),
                "shape_labels": list(model.shape_label),
                "box_body": box,
                "box_shape": box_shape,
                "table_shape": table_shape,
                "shape_config": asdict(shape_cfg),
                "specification": spec,
            }
        )
        static["device"] = {
            "requested": device,
            "actual": str(model.device),
            "name": model.device.name,
            "arch": model.device.arch,
        }
        result["model"] = f"{spec['id']}/model.json"
        result["config"] = f"{spec['id']}/config.json"
        _write_json(directory / "model.json", static)
        _write_json(directory / "config.json", asdict(config))
        wp.synchronize_device(device)
        phase = "initial-observation"
        last_state = _snapshot(state_in)
        frames.append(_frame(last_state, 0, spec["dt"]))
        stride = int(round(0.01 / spec["dt"]))
        with gzip.open(directory / "records.jsonl.gz", "wt", encoding="utf-8") as stream:
            for i in range(spec["steps"]):
                phase, attempt = "input", {"input_tick": i, "output_tick": i + 1}
                prep_start = time.perf_counter()
                entry = _snapshot(state_in)
                attempt["entry"] = entry
                force, torque, point = external_wrench(i * spec["dt"], entry["body_q"][box])
                state_in.clear_forces()
                state_in.body_f.assign(np.asarray([[*force, *torque]], dtype=np.float32))
                external = state_in.body_f.numpy()
                attempt["external_body_wrench"] = external
                wp.synchronize_device(device)
                input_s = time.perf_counter() - prep_start
                preparation_s += input_s
                phase = "physics"
                begin = time.perf_counter()
                solver.step(state_in, state_out, control, contacts, spec["dt"], observables=obs)
                wp.synchronize_device(device)
                elapsed = time.perf_counter() - begin
                step_seconds.append(elapsed)
                phase = "observation"
                read_start = time.perf_counter()
                attempt["exit"] = {
                    name: getattr(state_out, name).numpy()
                    for name in ("body_q", "body_qd", "joint_q", "joint_qd")
                }
                attempt["raw_contacts"] = _raw_contacts(contacts, obs, solver)
                attempt["solver_status"] = solver.status.numpy()
                snapshot = _json_value(attempt)
                _rotation(snapshot["exit"]["body_q"][box])
                _finite(snapshot["exit"]["body_qd"], (1, 6), "output velocity")
                last_state, last_state_tick = snapshot["exit"], i + 1
                if not snapshot["solver_status"] or any(
                    s["converged"] not in (False, True) for s in snapshot["solver_status"]
                ):
                    raise ValueError("missing or invalid solver convergence status")
                decoded = decode_contacts(
                    entry["body_q"],
                    static["body_com"],
                    static["shape_body"],
                    snapshot["raw_contacts"],
                )
                total = np.zeros(3)
                active = 0
                for contact in decoded:
                    if {contact["shape0"], contact["shape1"]} != {box_shape, table_shape}:
                        raise ValueError("unexpected collision identity in the fixed model")
                    f = (
                        contact["force0_world"]
                        if contact["body0"] == box
                        else contact["force1_world"]
                    )
                    total += f
                    active += float(np.linalg.norm(f)) > 1e-8
                row = {
                    **snapshot,
                    "phase": "step-input-contact",
                    "input_time": i * spec["dt"],
                    "output_time": (i + 1) * spec["dt"],
                    "contacts": decoded,
                    "application_world_m": point,
                    "contact_force_on_box": total.tolist(),
                    "active_contacts": int(active),
                    "timing": {"step_s": elapsed, "input_preparation_s": input_s},
                }
                # All raw arrays and output state were checked before publication.
                observation_s = time.perf_counter() - read_start
                read_seconds.append(observation_s)
                row["timing"]["observation_s"] = observation_s
                phase = "save"
                save_start = time.perf_counter()
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                stream.flush()
                save_seconds.append(time.perf_counter() - save_start)
                records.append(row)
                if i % stride == 0:
                    frame = _frame(
                        entry,
                        i,
                        spec["dt"],
                        contacts=decoded,
                        force=external[box, :3].tolist(),
                        torque=external[box, 3:].tolist(),
                        total=total.tolist(),
                        active=active,
                    )
                    if i == 0:
                        frames[0] = frame
                    else:
                        frames.append(frame)
                state_in, state_out = state_out, state_in
        phase = "metrics"
        metrics = summarize_records(records, spec["dt"], spec["steps"])
        result["status"] = (
            "PASS"
            if metrics["functional_pass"] and metrics["solver_converged"]
            else ("NONCONVERGED" if not metrics["solver_converged"] else "FAIL_METRICS")
        )
    except Exception as exc:
        result["status"] = (
            "UNSUPPORTED"
            if phase == "construct"
            and isinstance(
                exc,
                (
                    UnsupportedReference,
                    ImportError,
                    metadata.PackageNotFoundError,
                    NotImplementedError,
                ),
            )
            else "FAILED"
        )
        result["failure"] = f"{spec['id']}/failure.json"
        _write_json(
            directory / "failure.json",
            {
                "phase": phase,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "attempt": attempt,
                "accepted_steps": len(records),
                "diagnostic_nonfinite_encoding": "tagged nonfinite_float",
            },
            diagnostic=True,
        )
    finally:
        if last_state is not None:
            # Terminal endpoint has no invented endpoint contact force. Its most
            # recent accepted contact frame remains separately timestamped.
            terminal_tick = last_state_tick
            try:
                terminal = _frame(last_state, terminal_tick, spec["dt"])
                terminal["phase"] = "terminal-endpoint-no-contact-query"
                terminal["publication_status"] = result["status"]
                frames.append(terminal)
            except (KeyError, ValueError, IndexError):
                pass
        result["metrics"] = summarize_records(records, spec["dt"], spec["steps"])
        result["timing"] = {
            "construction_and_initial_compilation_s": construction_s,
            "input_preparation_and_readback_s": preparation_s,
            "first_step_s": step_seconds[0] if step_seconds else None,
            "step_s": sum(step_seconds),
            "observation_s": sum(read_seconds),
            "save_s": sum(save_seconds),
            "step_plus_required_reads_s": sum(step_seconds) + sum(read_seconds) + preparation_s,
            "steps_after_100ms_mean_s": float(np.mean(step_seconds[int(round(0.1 / spec["dt"])) :]))
            if len(step_seconds) > int(round(0.1 / spec["dt"]))
            else None,
            "case_wall_s": time.perf_counter() - started,
            "method": "host perf_counter, device synchronized around every full step; no graph",
            "compilation_note": "first-step and all per-step costs retained; later lazy JIT is not subtracted",
        }
        result["accepted_steps"] = len(records)
        result["frames_count"] = len(frames)
        _write_json(directory / "frames.json", {"schema": SCHEMA, "case": spec, "frames": frames})
        if not (directory / "records.jsonl.gz").exists():
            with gzip.open(directory / "records.jsonl.gz", "wt", encoding="utf-8"):
                pass
        result["state_sha256"] = _hash(directory / "records.jsonl.gz")
        _write_json(directory / "summary.json", result)
    return result


def run_suite(output, device="cuda:0"):
    """Run four independent fixed CUDA cases, preserving each failed prefix."""
    output = validate_output(output)
    if device != "cuda:0":
        raise ValueError("the fixed selection experiment permits only cuda:0")
    output.mkdir(parents=True)
    environment = _environment()
    _write_json(output / "environment.json", environment)
    summary = {
        "schema": SCHEMA,
        "device": device,
        "phase": "ACTUAL_3D_DYNAMICS",
        "environment_sha256": environment["binding_sha256"],
        "cases": [],
        "step_size_comparisons": [],
    }
    for backend, dt in CASES:
        spec = case_spec(backend, dt)
        summary["cases"].append(_run_case(output / spec["id"], spec, device))
        _write_json(output / "summary.json", summary)
    for backend in ("padmm", "dvi"):
        coarse, fine = [c for c in summary["cases"] if c["backend"] == backend]
        a, b = coarse["metrics"], fine["metrics"]
        denominator = abs(b["vertical_impulse_n_s"])
        relative = (
            abs(a["vertical_impulse_n_s"] - b["vertical_impulse_n_s"]) / denominator
            if denominator > 0
            else None
        )
        ta, tb = a["last_loaded_contact_time_s"], b["last_loaded_contact_time_s"]
        difference = abs(ta - tb) if ta is not None and tb is not None else None
        qualified = coarse["status"] == fine["status"] == "PASS"
        summary["step_size_comparisons"].append(
            {
                "backend": backend,
                "vertical_impulse_relative_difference": relative,
                "last_contact_difference_s": difference,
                "eligible": qualified,
                "pass": qualified
                and relative is not None
                and relative <= 0.1
                and difference is not None
                and difference <= 0.01,
            }
        )
    _write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    args = parser.parse_args()
    summary = run_suite(args.output, args.device)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": [{"id": c["id"], "status": c["status"]} for c in summary["cases"]],
            },
            allow_nan=False,
        )
    )
    return (
        0
        if all(c["status"] == "PASS" for c in summary["cases"])
        and all(c["pass"] for c in summary["step_size_comparisons"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
