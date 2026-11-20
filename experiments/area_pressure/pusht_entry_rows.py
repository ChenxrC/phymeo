"""Exact archived RR input states and retained native unilateral rows.

No function advances a state. Extraction reads the accepted entry, not the
trace's output q/v or its preceding feedback_source. assemble_entry creates
fresh MjData and calls mj_forward once on the unmodified original model.

owners preserves original efc row order. Contact owners contain source_row,
constraint_type=6, source_id/contact_index (the original compact contact),
contact_efc_address, edge, dimension, geometry_ids and geometry_names in native
orientation. Limit owners contain source_row, constraint_type=3, source_id/
joint_id, joint_name, dof and side (-1 lower, +1 upper). Every retained scalar
force has the domain [0, infinity); equality, dry-friction, frictionless and
elliptic rows are rejected instead of being reinterpreted.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from phymeo.backends.mujoco_area_pressure import area_collision_ownership
from phymeo.training.manipulation_tasks import manipulation_scenes

SCHEMA = "phymeo-pusht-entry-inputs-v1"
PHASE = "accepted-step-input"
CONTROL_PHASE = "held-for-this-accepted-entry"
LABELS = ("initial", "first_load", "peak", "last_push_entry", "withdrawn")
DEFAULT_INPUTS = Path(__file__).with_name("pusht_entry_inputs.json")
DEFAULT_POSES = Path(__file__).with_name("pusht_reference_poses.json")


def _require(condition, message):
    if not condition:
        raise ValueError("PushT entry rows: " + message)


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def _read_json(path):
    return json.loads(Path(path).read_text("utf-8-sig"), parse_constant=_reject_constant)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _payload_digest(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _array(value, shape, name):
    raw = np.asarray(value)
    _require(raw.dtype.kind in "fiu" and raw.shape == shape, name + " shape/type")
    result = raw.astype(np.float64)
    _require(np.isfinite(result).all(), name + " finite")
    return result


def _context(poses_path=DEFAULT_POSES):
    poses = _read_json(poses_path)
    scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    _require(
        poses["schema"] == "phymeo-pusht-saved-poses-v1"
        and poses["scene"] == scene.key
        and poses["source_sha256"] == scene.source_sha256
        and poses["h"] == scene.timestep_s
        and set(poses["samples"]) == set(LABELS),
        "shipped pose/model binding",
    )
    model = mj.MjModel.from_xml_string(scene.mjcf)
    _require((model.nq, model.nv, model.nu, model.nbody) == (9, 8, 2, 5), "original RR layout")
    return poses, scene, model


def _check_pose(model, entry, expected):
    """Reproduce the shipped pose transform from saved FP32 body fields only."""
    positions = _array(entry["saved_body_position"], (model.nbody, 3), "saved body positions")
    rotations = _array(entry["saved_body_rotation"], (model.nbody, 3, 3), "saved body rotations")
    pusher = model.geom("pusher").id
    tool, body = int(model.geom_bodyid[pusher]), model.body("object").id
    matrix = np.empty(9)
    mj.mju_quat2Mat(matrix, model.geom_quat[pusher])
    axis = matrix.reshape(3, 3)[:, 2] * model.geom_size[pusher, 1]
    local_ends = model.geom_pos[pusher] + np.array([-axis, axis])
    ends = (local_ends @ rotations[tool].T + (positions[tool] - positions[body])) @ rotations[body]
    _require(
        entry["tick"] == expected["tick"]
        and np.array_equal(ends, np.asarray(expected["ends"]))
        and np.array_equal(rotations[body], np.asarray(expected["rotation"])),
        "entry does not match the exact shipped pose",
    )


def _validate_entry(entry, label, model, expected, h):
    _require(entry["label"] == label and type(entry["tick"]) is int, "label/input tick")
    _require(
        entry["phase"] == PHASE and entry["control_phase"] == CONTROL_PHASE, "input/control phase"
    )
    _require(entry["output_tick"] == entry["tick"] + 1, "accepted output tick")
    meta = entry["entry_meta"]
    _require(
        len(meta) == 6
        and all(type(x) is int for x in meta)
        and meta[:3] == [1, entry["tick"], entry["output_tick"]]
        and 0 <= meta[4] <= meta[3]
        and meta[5] == 0,
        "accepted entry metadata",
    )
    clock = _array(entry["entry_clock"], (4,), "entry clock")
    _require(
        entry["time"] == float(clock[0]) - float(clock[1])
        and entry["output_time"] == float(clock[2]) - float(clock[3])
        and abs(entry["time"] - entry["tick"] * h) <= 2e-6
        and abs(entry["output_time"] - entry["time"] - h) <= 2e-6,
        "compensated input/output time",
    )
    values = {}
    for name, shape in (
        ("q", (model.nq,)),
        ("v", (model.nv,)),
        ("ctrl", (model.nu,)),
        ("qfrc_applied", (model.nv,)),
        ("xfrc_applied", (model.nbody, 6)),
    ):
        values[name] = _array(entry[name], shape, name)
        _require(
            np.array_equal(values[name], values[name].astype(np.float32).astype(float)),
            name + " must preserve the exact archived FP32 values",
        )
    _require(
        type(entry["control_source_revision"]) is int
        and entry["control_revision"] == entry["control_source_revision"] + 1,
        "accepted control revision",
    )
    for name in ("trace_line_sha256", "archive_sha256", "model_sha256"):
        digest = entry[name]
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            name,
        )
    _check_pose(model, entry, expected)
    return values


def _archive_context(directory, poses_path=DEFAULT_POSES):
    """Bind both selected-state and continuous-control readers to one archive."""
    root = Path(directory).resolve()
    poses, scene, model = _context(poses_path)
    manifest = _read_json(root / "manifest.json")
    _require(
        manifest["schema"] == "phymeo-manipulation-control-v1"
        and manifest["mode"] == "cuda"
        and manifest["status"] == "COMPLETE"
        and manifest["record_entry"] is True
        and manifest["recorded_steps"] == manifest["steps"]
        and manifest["h"] == scene.timestep_s
        and len(manifest["worlds"]) == 1,
        "complete single-world original CUDA archive required",
    )
    world = manifest["worlds"][0]
    _require(
        world["key"] == scene.key and world["source_sha256"] == scene.source_sha256,
        "archive source",
    )
    source = (root / world["source_file"]).resolve()
    _require(source.is_relative_to(root), "archive source path escapes directory")
    trace = root / "trace.jsonl.gz"
    _require(_digest(source) == scene.source_sha256, "archived XML digest")
    trace_digest = _digest(trace)
    _require(
        trace_digest == manifest["trace_sha256"] == poses["saved_trace_sha256"],
        "original trace digest",
    )
    return poses, scene, model, manifest, world, trace, trace_digest


def _control_payload(row, world, scene, model):
    """Check actual native control channels against the accepted action receipt."""
    action = row["action_receipt"]
    _require(
        action["world_keys"] == [world["world"]] and len(action["receipts"]) == 1,
        "control world order",
    )
    receipt = action["receipts"][0]
    binding = receipt["binding"]
    _require(
        binding["source_sha256"] == scene.source_sha256
        and binding["world_keys"] == [world["world"]]
        and binding["native_layout"] == [model.nq, model.nv, model.nu, model.nbody],
        "native control source/layout",
    )
    _require(
        type(receipt["source_revision"]) is int
        and receipt["revision"] == receipt["source_revision"] + 1,
        "accepted control revision",
    )
    controls, channels = receipt["controls"][str(world["world"])], row["native_channels"]
    _require(len(channels) == 3, "three native control channels")
    fields = (
        ("ctrl", (model.nu,)),
        ("qfrc_applied", (model.nv,)),
        ("xfrc_applied", (model.nbody, 6)),
    )
    for channel, (name, shape) in zip(channels, fields, strict=True):
        _require(
            len(channel) == 1 and channel[0] == controls[name],
            "requested/applied control channel mismatch",
        )
        values = _array(controls[name], shape, name)
        _require(
            np.array_equal(values, values.astype(np.float32).astype(float)),
            name + " must preserve the exact archived FP32 values",
        )
    return controls, receipt, binding


def extract_entries(directory, *, poses_path=DEFAULT_POSES):
    """Return a JSON-ready five-frame packet; do not write or integrate anything."""
    poses, scene, model, _manifest, world, trace, trace_digest = _archive_context(
        directory, poses_path
    )
    key = str(world["world"])
    ticks = {poses["samples"][label]["tick"]: label for label in LABELS}
    _require(len(ticks) == len(LABELS), "distinct selected input ticks")
    samples = {}
    with gzip.open(trace, "rb") as stream:
        for tick, line in enumerate(stream):
            row = json.loads(line, parse_constant=_reject_constant)
            _require(
                row["step"] == tick + 1 and row["accepted"] == {key: True},
                "contiguous accepted archive",
            )
            if tick not in ticks:
                continue
            label, saved = ticks[tick], row["entry"]
            controls, receipt, binding = _control_payload(row, world, scene, model)
            _require(
                len(saved["entry_q"]) == len(saved["entry_v"]) == 1, "single-world saved input"
            )
            clock = saved["entry_clock"][0]
            sample = dict(
                label=label,
                tick=tick,
                output_tick=row["step"],
                phase=PHASE,
                control_phase=CONTROL_PHASE,
                time=float(clock[0]) - float(clock[1]),
                output_time=float(clock[2]) - float(clock[3]),
                q=saved["entry_q"][0],
                v=saved["entry_v"][0],
                ctrl=controls["ctrl"],
                qfrc_applied=controls["qfrc_applied"],
                xfrc_applied=controls["xfrc_applied"],
                entry_meta=saved["entry_meta"][0],
                entry_clock=clock,
                saved_body_position=saved["entry_body_position"][0],
                saved_body_rotation=saved["entry_body_rotation"][0],
                control_source_revision=receipt["source_revision"],
                control_revision=receipt["revision"],
                archive_sha256=binding["archive_sha256"],
                model_sha256=binding["model_sha256"],
                trace_line_sha256=hashlib.sha256(line).hexdigest(),
            )
            _validate_entry(sample, label, model, poses["samples"][label], scene.timestep_s)
            measurement = row["reference_feedback"][key]["accepted"]["measurement"]
            _require(
                measurement["input_tick"] == tick and measurement["output_tick"] == tick + 1,
                "accepted feedback entry phase",
            )
            samples[label] = sample
            if len(samples) == len(LABELS):
                break
    _require(set(samples) == set(LABELS), "missing selected exact states")
    samples = {label: samples[label] for label in LABELS}
    return dict(
        schema=SCHEMA,
        scene=scene.key,
        source_sha256=scene.source_sha256,
        saved_trace_sha256=trace_digest,
        poses_sha256=_digest(poses_path),
        h=scene.timestep_s,
        world=world["world"],
        layout=dict(nq=model.nq, nv=model.nv, nu=model.nu, nbody=model.nbody),
        samples_sha256=_payload_digest(samples),
        samples=samples,
    )


def iter_controls(directory, start_tick, stop_tick_exclusive):
    """Yield original controls for a complete half-open input-tick interval.

    The entire requested interval is validated before the first yield. Records
    contain controls and provenance only, never archived q/v. Initialize a
    separate rollout once with load_entries(); thereafter evolve its own state.
    """
    _require(
        type(start_tick) is int
        and type(stop_tick_exclusive) is int
        and 0 <= start_tick <= stop_tick_exclusive,
        "ordered nonnegative integer control ticks required",
    )
    _poses, scene, model, manifest, world, trace, trace_digest = _archive_context(directory)
    _require(
        stop_tick_exclusive <= manifest["recorded_steps"],
        "control tick interval exceeds complete archive",
    )
    key, records = str(world["world"]), []
    with gzip.open(trace, "rb") as stream:
        for tick, line in enumerate(stream):
            if tick >= stop_tick_exclusive:
                break
            row = json.loads(line, parse_constant=_reject_constant)
            _require(
                row["step"] == tick + 1 and row["accepted"] == {key: True},
                "contiguous accepted archive",
            )
            if tick < start_tick:
                continue
            controls, receipt, binding = _control_payload(row, world, scene, model)
            saved = row["entry"]
            _require(len(saved["entry_meta"]) == 1, "single-world entry metadata")
            meta = saved["entry_meta"][0]
            _require(
                len(meta) == 6
                and all(type(x) is int for x in meta)
                and meta[:3] == [1, tick, tick + 1]
                and 0 <= meta[4] <= meta[3]
                and meta[5] == 0,
                "accepted control input tick metadata",
            )
            clock = _array(saved["entry_clock"], (1, 4), "entry clock")[0]
            input_time, output_time = (
                float(clock[0]) - float(clock[1]),
                float(clock[2]) - float(clock[3]),
            )
            _require(
                abs(input_time - tick * scene.timestep_s) <= 2e-6
                and abs(output_time - input_time - scene.timestep_s) <= 2e-6,
                "control input/output clock phase",
            )
            measurement = row["reference_feedback"][key]["accepted"]["measurement"]
            _require(
                measurement["input_tick"] == tick and measurement["output_tick"] == tick + 1,
                "accepted feedback entry phase",
            )
            records.append(
                dict(
                    tick=tick,
                    output_tick=tick + 1,
                    time=input_time,
                    output_time=output_time,
                    h=scene.timestep_s,
                    phase=PHASE,
                    control_phase=CONTROL_PHASE,
                    ctrl=_array(controls["ctrl"], (model.nu,), "ctrl"),
                    qfrc_applied=_array(controls["qfrc_applied"], (model.nv,), "qfrc_applied"),
                    xfrc_applied=_array(controls["xfrc_applied"], (model.nbody, 6), "xfrc_applied"),
                    world=world["world"],
                    source_sha256=scene.source_sha256,
                    saved_trace_sha256=trace_digest,
                    trace_line_sha256=hashlib.sha256(line).hexdigest(),
                    archive_sha256=binding["archive_sha256"],
                    model_sha256=binding["model_sha256"],
                    control_source_revision=receipt["source_revision"],
                    control_revision=receipt["revision"],
                )
            )
    _require(
        len(records) == stop_tick_exclusive - start_tick,
        "incomplete requested control tick interval",
    )
    yield from records


def load_entries(path=DEFAULT_INPUTS):
    """Load portable exact states keyed by label; numerical vectors are FP64.

    Provenance binds the selected archive and shipped poses. The samples hash
    detects changed packet content; offline loading does not claim to re-read
    or authenticate the absent complete original trace.
    """
    report = _read_json(path)
    poses, scene, model = _context()
    _require(
        report["schema"] == SCHEMA
        and report["scene"] == scene.key
        and report["source_sha256"] == scene.source_sha256
        and report["saved_trace_sha256"] == poses["saved_trace_sha256"]
        and report["poses_sha256"] == _digest(DEFAULT_POSES)
        and report["h"] == scene.timestep_s
        and report["layout"] == dict(nq=model.nq, nv=model.nv, nu=model.nu, nbody=model.nbody)
        and set(report["samples"]) == set(LABELS),
        "portable original-source/pose/layout binding",
    )
    _require(
        _payload_digest(report["samples"]) == report["samples_sha256"], "portable samples digest"
    )
    entries = {}
    for label in LABELS:
        entry = dict(report["samples"][label])
        values = _validate_entry(entry, label, model, poses["samples"][label], report["h"])
        entry.update(values)
        entry.update(
            h=report["h"],
            source_sha256=report["source_sha256"],
            saved_trace_sha256=report["saved_trace_sha256"],
            poses_sha256=report["poses_sha256"],
            world=report["world"],
        )
        entries[label] = entry
    return entries


def assemble_entry(model, entry):
    """Return FP64 M, base load and every retained native unilateral source row.

    base_load = actuator + generalized/body applied + passive - bias. It never
    includes qfrc_constraint, efc_force or a saved native contact solution.
    data is fresh original-model MjData for geometry; its solved native forces
    are not an oracle for the replacement contact problem.
    """
    _require(int(model.opt.cone) == int(mj.mjtCone.mjCONE_PYRAMIDAL), "pyramidal cone required")
    _require(model.na == 0, "stateful actuators require an explicit activation entry")
    data = mj.MjData(model)
    data.qpos[:] = _array(entry["q"], (model.nq,), "q")
    data.qvel[:] = _array(entry["v"], (model.nv,), "v")
    data.ctrl[:] = _array(entry["ctrl"], (model.nu,), "ctrl")
    data.qfrc_applied[:] = _array(
        entry.get("qfrc_applied", np.zeros(model.nv)), (model.nv,), "qfrc_applied"
    )
    data.xfrc_applied[:] = _array(
        entry.get("xfrc_applied", np.zeros((model.nbody, 6))), (model.nbody, 6), "xfrc_applied"
    )
    data.time = float(entry.get("time", 0.0))
    _require(np.isfinite(data.time), "entry time")
    mj.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv), dtype=np.float64)
    mj.mj_fullM(model, mass, data.qM)
    external = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(1, model.nbody):
        wrench = data.xfrc_applied[body]
        if np.any(wrench):
            mj.mj_applyFT(model, data, wrench[:3], wrench[3:], data.xipos[body], body, external)
    base_load = external + data.qfrc_passive - data.qfrc_bias
    _require(np.isfinite(mass).all() and np.isfinite(base_load).all(), "finite mass/base load")
    _require(data.efc_J.size == data.nefc * model.nv, "dense original efc Jacobian required")
    full_j = data.efc_J.reshape(data.nefc, model.nv)
    limit_type = int(mj.mjtConstraint.mjCNSTR_LIMIT_JOINT)
    contact_type = int(mj.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)
    _require(
        np.isin(data.efc_type, (limit_type, contact_type)).all(),
        "only nonnegative pyramidal/scalar-limit rows supported",
    )
    ownership = area_collision_ownership(mj, model, (("tool", "object"),))
    excluded_pairs = {frozenset(pair) for pair in ownership.replaced_geometry_pairs[0]}
    pusher = model.geom("pusher").id
    wanted = {frozenset((pusher, model.geom(name).id)) for name in ("object-0", "object-1")}
    _require(
        excluded_pairs == wanted and len(excluded_pairs) == 2,
        "exact two original tool-object pairs",
    )
    owner_by_row, excluded_rows = {}, set()
    contacts = []
    for ci, contact in enumerate(data.contact):
        pair = tuple(map(int, contact.geom))
        first, dim = int(contact.efc_address), int(contact.dim)
        excluded = frozenset(pair) in excluded_pairs
        indices = ()
        if first >= 0:
            _require(dim in (3, 4, 6), "active contact must have pyramidal friction")
            indices = tuple(range(first, first + 2 * (dim - 1)))
            _require(indices[-1] < data.nefc, "contact row span")
            for edge, row in enumerate(indices):
                _require(
                    row not in owner_by_row
                    and int(data.efc_type[row]) == contact_type
                    and int(data.efc_id[row]) == ci,
                    "contact address/type/identity coverage",
                )
                owner_by_row[row] = dict(
                    kind="contact",
                    source_row=row,
                    constraint_type=contact_type,
                    source_id=ci,
                    contact_index=ci,
                    contact_efc_address=first,
                    edge=edge,
                    dimension=dim,
                    geometry_ids=pair,
                    geometry_names=tuple(model.geom(g).name for g in pair),
                )
                if excluded:
                    excluded_rows.add(row)
        contacts.append(
            dict(
                contact_index=ci,
                geometry_ids=pair,
                geometry_names=tuple(model.geom(g).name for g in pair),
                efc_address=first,
                source_rows=indices,
                retained=not excluded,
                position=contact.pos.copy(),
                frame=contact.frame.reshape(3, 3).copy(),
                friction=contact.friction.copy(),
            )
        )
    for row, kind in enumerate(data.efc_type):
        if int(kind) != limit_type:
            continue
        joint = int(data.efc_id[row])
        _require(0 <= joint < model.njnt, "limit joint identity")
        _require(
            int(model.jnt_type[joint])
            in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)),
            "scalar joint limit required",
        )
        dof, qa = int(model.jnt_dofadr[joint]), int(model.jnt_qposadr[joint])
        side = -1 if full_j[row, dof] > 0 else 1
        expected = np.zeros(model.nv)
        expected[dof] = -side
        distance = (
            data.qpos[qa] - model.jnt_range[joint, 0]
            if side < 0
            else model.jnt_range[joint, 1] - data.qpos[qa]
        )
        _require(
            row not in owner_by_row
            and bool(model.jnt_limited[joint])
            and np.array_equal(full_j[row], expected)
            and distance < model.jnt_margin[joint],
            "source scalar limit side/Jacobian",
        )
        owner_by_row[row] = dict(
            kind="limit",
            source_row=row,
            constraint_type=limit_type,
            source_id=joint,
            joint_id=joint,
            joint_name=model.joint(joint).name,
            dof=dof,
            side=side,
        )
    _require(set(owner_by_row) == set(range(data.nefc)), "complete original row ownership")
    kept = np.array([row for row in range(data.nefc) if row not in excluded_rows], dtype=np.int64)
    jacobian = full_j[kept].copy()
    regularization, reference = data.efc_R[kept].copy(), data.efc_aref[kept].copy()
    _require(
        np.isfinite(jacobian).all()
        and np.isfinite(regularization).all()
        and np.isfinite(reference).all()
        and np.all(regularization > 0),
        "finite retained unilateral coefficients with positive R",
    )
    return dict(
        mass=mass,
        base_load=base_load,
        jacobian=jacobian,
        regularization=regularization,
        reference_acceleration=reference,
        owners=tuple(owner_by_row[int(row)] for row in kept),
        data=data,
        contacts=tuple(contacts),
        retained_indices=kept,
        excluded_indices=np.array(sorted(excluded_rows), dtype=np.int64),
        original_row_count=int(data.nefc),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, required=True, metavar="ARCHIVE_DIRECTORY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = extract_entries(args.extract)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
