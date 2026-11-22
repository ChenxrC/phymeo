"""Verify the saved 250-step withdrawal fork, then restore twice without stepping.

python -m experiments.area_pressure.pusht_lookahead_fork --source SOURCE.json \
    --fork fork.json --output NEW.json [--cache-root CACHE]

Every candidate/callback is compared in full except candidate.step_wall_s.
Release-file locators are bound separately to complete decoded file contents.
No tolerance, quaternion normalization, controller/planner call or physics step.
Public Scene.restore is the actual checkpoint API; endpoint snapshot() is not
used to substitute newly reconstructed forces for the original receipt.
"""

import argparse
import base64
import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path

from .pusht_lookahead_source import (
    EVENT,
    REPORT,
    _canonical_sha,
    _constant,
    _digest,
    _read,
    load_lookahead_source,
)

SCHEMA = "phymeo-pusht-lookahead-fork-audit-v1"
WITHDRAWAL_STEPS = 250
CHANNELS = ("ctrl", "qfrc_applied", "xfrc_applied")


def _require(ok, message):
    if not ok:
        raise ValueError("lookahead fork: " + message)


def _equal(actual, expected, label):
    """Exact JSON values, including numeric type and signed zero."""
    if _canonical_sha(actual) != _canonical_sha(expected):

        def first(a, b, path):
            if type(a) is not type(b):
                return path + " (type)"
            if isinstance(a, dict):
                if set(a) != set(b):
                    return path + " (keys)"
                for key in a:
                    if _canonical_sha(a[key]) != _canonical_sha(b[key]):
                        return first(a[key], b[key], path + "." + key)
            elif isinstance(a, list):
                if len(a) != len(b):
                    return path + " (length)"
                for i, (x, y) in enumerate(zip(a, b, strict=True)):
                    if _canonical_sha(x) != _canonical_sha(y):
                        return first(x, y, path + "." + str(i))
            return path

        raise ValueError("lookahead fork: unequal " + first(actual, expected, label))


def _bound(path, sha256, label):
    path = Path(path).resolve()
    _require(_digest(path) == sha256, label + " SHA256")
    return path


def _file(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _digest(path)}


def _journal(path, report):
    """Verify the full stream, retaining only the pre-ordinary withdrawal."""
    digest = hashlib.sha256()
    rows, prefix_hashes = [], {}
    retained = report["initial_state"]
    pending = None
    planned = ended = False
    records = 0
    init = probe = release = None
    with path.open("rb") as stream:
        for raw in stream:
            _require(raw.endswith(b"\n") and not ended, "complete ordered journal")
            digest.update(raw)
            records += 1
            event = json.loads(raw, parse_constant=_constant)
            _require(event["schema"] == EVENT, "journal schema")
            kind, row = event["event"], event["payload"]
            if kind == "initialization":
                _require(records == 1 and init is None, "single initialization")
                init = row
                for key in (
                    "initial_state",
                    "controller_initial",
                    "goal_history_initial",
                    "provenance",
                    "goal",
                    "step_budget",
                ):
                    _equal(row[key], report[key], "initialization." + key)
            elif kind == "planner_probe" and not planned:
                _require(probe is None and pending is None, "single entry gate")
                probe = row
                _equal(row, report["entry_probe"], "entry gate report")
            elif kind == "release_reference" and not planned:
                _require(release is None and pending is None, "single release request")
                release = row
                _equal(row["reference"], report["release_reference"], "release reference")
                _equal(row["seed"], report["seed_evidence"], "release seed")
            elif kind == "planner":
                _require(not planned and pending is None and rows, "ordinary plan boundary")
                _equal(row, report["ordinary_route"], "ordinary plan report")
                _equal(row["source_q"], retained["q"], "ordinary plan input q")
                _equal(row["source_v"], retained["v"], "ordinary plan input v")
                planned = True
            elif kind == "candidate" and not planned:
                index = event.get("trial_index")
                _require(
                    init is not None and release is not None and pending is None,
                    "withdrawal candidate ordering",
                )
                _require(
                    index == row["index"] == row["trial_index"] == len(rows)
                    and type(index) is int
                    and row["phase"] == "withdraw"
                    and row["selected"] is True
                    and row["restored"] is False
                    and row["rollback_verification"] is None,
                    "one accepted withdrawal candidate per original tick",
                )
                _equal(row["input_state"], retained, "withdrawal continuous input")
                after = row["output_state"]
                _require(
                    after["local_tick"] == retained["local_tick"] + 1
                    and after["time"] > retained["time"]
                    and row["original_input_tick"]
                    == report["source_reference_tick_origin"] + retained["local_tick"]
                    and row["original_output_tick"] == row["original_input_tick"] + 1,
                    "withdrawal adjacent clock/tick",
                )
                pending = row
            elif kind == "accepted_callback" and not planned:
                _require(
                    pending is not None and event.get("trial_index") == len(rows),
                    "withdrawal callback pairing",
                )
                _equal(
                    row["feedback_accepted"]["measurement"],
                    pending["measurement"],
                    "callback actual measurement",
                )
                _require(
                    row["driver_state_after"]["tick"] == pending["original_output_tick"]
                    and row["goal_result"]["history"]["tick"]
                    == pending["input_state"]["local_tick"]
                    and row["goal_result"]["history"]["time_s"] == pending["input_state"]["time"],
                    "callback state phase",
                )
                rows.append((pending, row))
                retained, pending = pending["output_state"], None
                prefix_hashes[records] = digest.hexdigest()
            elif kind == "final":
                _require(pending is None, "no incomplete final callback")
                _equal(row, {k: v for k, v in report.items() if k != "journal"}, "final event")
                ended = True
            elif planned and kind in ("candidate", "accepted_callback"):
                # The full original future is bound by load_lookahead_source;
                # hash it here, but never use it as fork state or control.
                pass
            else:
                raise ValueError("lookahead fork: unsupported event " + kind)
    _require(
        ended
        and records == report["journal"]["records"]
        and digest.hexdigest() == report["journal"]["sha256"],
        "complete journal digest",
    )
    _require(len(rows) == WITHDRAWAL_STEPS and pending is None, "exactly 250 withdrawal receipts")
    return dict(
        rows=rows,
        init=init,
        probe=probe,
        release=release,
        planned=planned,
        endpoint=retained,
        prefix_hashes=prefix_hashes,
        records=records,
    )


def _bind_records(source_path, fork_path):
    source, fork = _read(source_path), _read(fork_path)
    _require(source["schema"] == "phymeo-pusht-lookahead-source-v1", "source schema")
    original_path = _bound(**source["sources"]["report"], label="original report")
    audit_path = _bound(**source["sources"]["audit"], label="original audit")
    rebuilt = load_lookahead_source(
        original_path, audit_path, route_path=source["baseline"]["route_path"]
    )
    _equal(source, rebuilt, "complete original entry/source/baseline binding")
    original, entry = _read(original_path), source["entry"]
    _require(
        fork["schema"] == REPORT
        and fork["status"] == "step_budget"
        and fork["step_budget"] == WITHDRAWAL_STEPS
        and all(
            fork[k] == WITHDRAWAL_STEPS
            for k in (
                "accepted_steps",
                "recorded_steps",
                "physical_attempts",
                "physical_trial_steps",
            )
        )
        and fork["rejected_trials"] == 0
        and fork["planner_calls"] == 0
        and fork["release_planner_calls"] == 1
        and fork["regular_plan_ready"] is False
        and not any(
            fork.get(k)
            for k in (
                "ordinary_route",
                "active_route",
                "unrecorded_accepted_step",
                "failed_candidate",
                "checkpoint_save_error",
                "errors",
            )
        ),
        "completed withdrawal-only fork, no new strategy or incomplete publication",
    )
    for key in (
        "world",
        "side_coverage",
        "publication_frame",
        "control_variant",
        "selection_policy",
        "material_admission",
        "force_phase",
        "goal_tick_origin",
        "start_tick",
        "local_tick_origin",
        "source_reference_tick_origin",
        "provenance",
        "initial_state",
        "controller_initial",
        "goal_history_initial",
        "goal",
        "previous_route",
        "original_step_budget",
        "parent_physical_attempts",
        "checkpoint_restore",
        "inherited_point_loads",
        "entry_probe",
        "seed_evidence",
    ):
        _equal(fork[key], original[key], "same original starting source." + key)
    _require(
        fork["world"] == 6 and entry["history_source"]["callbacks_before_plan"] == 250,
        "physical world and original withdrawal length",
    )
    original_journal = _bound(**source["sources"]["journal"], label="original journal")
    fork_journal = _bound(
        fork_path.parent / fork["journal"]["file"], fork["journal"]["sha256"], "fork journal"
    )
    old, new = _journal(original_journal, original), _journal(fork_journal, fork)
    _require(old["planned"] and not new["planned"], "stop before original ordinary planning")
    _equal(old["probe"], new["probe"], "original entry gate")
    _equal(old["release"]["seed"], new["release"]["seed"], "original release seed")
    release_paths = []
    for report, base in ((original, original_path.parent), (fork, fork_path.parent)):
        ref = report["release_reference"]
        release_paths.append(_bound(base / ref["file"], ref["sha256"], "release route"))
    _equal(_read(release_paths[0]), _read(release_paths[1]), "complete release request contents")
    refs = [deepcopy(report["release_reference"]) for report in (original, fork)]
    for ref in refs:
        ref.pop("file")
        ref.pop("sha256")
    _equal(refs[0], refs[1], "release reference excluding bound file locator/hash")
    candidate_digest, callback_digest = hashlib.sha256(), hashlib.sha256()
    for i, ((a, ac), (b, bc)) in enumerate(zip(old["rows"], new["rows"], strict=True)):
        a, b = deepcopy(a), deepcopy(b)
        for row in (a, b):
            wall = row.pop("step_wall_s")
            _require(
                type(wall) in (float, int) and math.isfinite(wall) and wall >= 0,
                "finite nonnegative excluded wall time",
            )
        _equal(a, b, f"withdrawal[{i}].candidate")
        _equal(ac, bc, f"withdrawal[{i}].callback")
        candidate_digest.update(bytes.fromhex(_canonical_sha(a)))
        callback_digest.update(bytes.fromhex(_canonical_sha(ac)))
    last, callback = new["rows"][-1]
    for value in (old["endpoint"], new["endpoint"], fork["final_state"]):
        _equal(value, entry["state"], "exact source entry q/v/clock at fork endpoint")
    _equal(callback["driver_state_after"], entry["controller_boundary"], "controller boundary")
    _equal(callback["goal_result"]["history"], entry["goal_history"], "Goal history at entry")
    _equal(fork["goal_history_final"], entry["goal_history"], "saved terminal Goal")
    _equal(fork["last_goal"], callback["goal_result"], "saved terminal Goal result")
    expected_controller = deepcopy(original["controller_initial"])
    expected_controller.update(entry["controller_boundary"])
    expected_controller.update(entry["planner_counts_before"])
    _equal(fork["controller_final"], expected_controller, "full original controller budget/history")
    _equal(last["area"], entry["previous_area"], "previous actual area receipt source")
    _equal(last["measurement"], entry["previous_measurement"], "previous actual measurement")
    _require(
        callback["feedback_accepted"]["segment_impulse_ns"] == entry["segment_impulse_ns"]
        and callback["feedback_accepted"]["total_impulse_ns"] == entry["total_impulse_ns"]
        and fork["release_status"] == "released",
        "original impulses and release completion",
    )
    _equal(fork["separation_proof"], original["separation_proof"], "final original release proof")
    remaining = fork["original_step_budget"] - fork["parent_physical_attempts"] - 250
    _require(remaining == source["budget"]["remaining_steps"], "unchanged remaining budget")
    return (
        source,
        fork,
        last,
        callback,
        new,
        dict(
            original_journal=_file(original_journal),
            fork_journal=_file(fork_journal),
            receipts=250,
            callbacks=250,
            physics_reexecuted_by_verifier=0,
            first_input_tick=new["rows"][0][0]["original_input_tick"],
            final_output_tick=last["original_output_tick"],
            candidate_content_sha256=candidate_digest.hexdigest(),
            callback_content_sha256=callback_digest.hexdigest(),
            excluded_candidate_fields={
                "step_wall_s": "host elapsed wall time only; finite/nonnegative"
            },
            file_locator_differences={
                "release_reference.file/sha256": "each file hash checked; decoded request contents equal"
            },
            release_routes=[_file(p) for p in release_paths],
            candidate_fields_checked=sorted(set(last) - {"step_wall_s"}),
            callback_fields_checked=sorted(callback),
            remaining_steps=remaining,
        ),
    )


def _checkpoint_binding(fork_path, fork, entry, journal):
    saved = fork["final_checkpoint"]
    metadata_path = _bound(
        fork_path.parent / saved["metadata_file"], saved["metadata_sha256"], "terminal metadata"
    )
    meta = _read(metadata_path)
    _require(
        meta["schema"] == saved["schema"] == "phymeo-pusht-active-terminal-checkpoint-v1"
        and (metadata_path.parent / meta["source_report"]).resolve() == fork_path,
        "terminal metadata source/schema",
    )
    capsule_path = _bound(
        metadata_path.parent / meta["checkpoint_file"],
        meta["checkpoint_sha256"],
        "terminal capsule",
    )
    _require(
        capsule_path == (fork_path.parent / saved["file"]).resolve()
        and saved["sha256"] == meta["checkpoint_sha256"],
        "capsule report binding",
    )
    for key in (
        "source_tick",
        "local_tick",
        "time",
        "world",
        "controller_at_physical_endpoint",
        "physical_state_matches_report",
    ):
        _equal(saved[key], meta[key], "terminal metadata summary." + key)
    _require(
        meta["source_tick"] == entry["tick"]
        and meta["world"] == 6
        and meta["controller_at_physical_endpoint"] is True
        and meta["physical_state_matches_report"] is True
        and meta["asset_sha256"] == entry["scene_asset_sha256"]
        and meta["local_tick"] == entry["state"]["local_tick"]
        and meta["time"] == entry["state"]["time"],
        "exact terminal source/asset/clock",
    )
    for key, expected in (
        ("state", entry["state"]),
        ("q", entry["state"]["q"]),
        ("v", entry["state"]["v"]),
        ("target", entry["goal"]),
        ("controller", fork["controller_final"]),
        ("goal_history", entry["goal_history"]),
        ("provenance", fork["provenance"]),
        ("active_route", None),
        ("control_variant", entry["control_variant"]),
        ("accepted_steps", 250),
        ("recorded_steps", 250),
    ):
        _equal(meta[key], expected, "terminal metadata." + key)
    prefix = meta["journal"]
    _require(
        prefix["file"] == fork["journal"]["file"]
        and prefix["records"] == journal["records"] - 1
        and journal["prefix_hashes"].get(prefix["records"]) == prefix["sha256"],
        "metadata binds all 250 callbacks, before exactly the final event",
    )
    return capsule_path, dict(
        metadata=_file(metadata_path),
        capsule=_file(capsule_path),
        metadata_journal_prefix=prefix,
        source_tick=meta["source_tick"],
        local_tick=meta["local_tick"],
        asset_sha256=meta["asset_sha256"],
    )


def _payload(raw, headers):
    """Inspect only checksum-checked public capsule bytes, not runtime internals."""
    header, digest, body = raw.split(b"\n", 2)
    _require(
        header in headers and hashlib.sha256(body).hexdigest().encode("ascii") == digest,
        "public capsule envelope checksum",
    )
    return json.loads(body, parse_constant=_constant)


def _capsule_content(capsule):
    batch = capsule.batch_checkpoint
    envelope = _payload(batch.to_bytes(), (b"PHYMEOTYXHET1", b"PHYMEOTYXHET2"))
    leaves = []
    for leaf in envelope["leaves"]:
        raw = base64.b64decode(leaf["bytes"], validate=True)
        _require(hashlib.sha256(raw).hexdigest() == leaf["sha256"], "leaf checksum")
        leaves.append(_payload(raw, (b"PHYMEOTYXCP1",)))
    _require(
        envelope["binding"]["world_keys"] == [6] and len(leaves) == 1,
        "single original world/bucket capsule",
    )
    return envelope, leaves


def _readback_form(value):
    """Owned typed JSON tree retaining every numerical byte of public snapshots."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return {
            "array_dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        }
    if is_dataclass(value):
        return {
            "record": type(value).__qualname__,
            "fields": {f.name: _readback_form(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, Mapping):
        return {str(k): _readback_form(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [_readback_form(v) for v in value]}
    if isinstance(value, np.generic):
        return _readback_form(np.asarray(value))
    _require(value is None or type(value) in (bool, int, float, str), "supported public value")
    return value


def _receipt_binding(snapshot, last, entry, model, asset):
    import numpy as np

    from scripts.run_manipulation_control import plain

    from .pusht_active_checkpoint import require_saved_state
    from .pusht_lateral_replan import point_loads

    require_saved_state(snapshot, entry["state"])
    receipt = snapshot.last_step
    _require(
        receipt is not None
        and receipt.world == snapshot.world == 6
        and receipt.tick == snapshot.tick
        and receipt.output_time == snapshot.time
        and receipt.input_time == last["input_state"]["time"]
        and receipt.h == 0.001,
        "original preceding receipt clock/phase",
    )
    for key in ("q", "v"):
        _equal(
            plain(getattr(receipt, "input_" + key)),
            last["input_state"][key],
            "restored receipt input " + key,
        )
    for channel in CHANNELS:
        expected = last["control"][channel]
        for owner in (snapshot, snapshot.held_control, receipt.control, receipt.stored_control):
            _equal(plain(getattr(owner, channel)), expected, "restored " + channel)
    (pair,) = receipt.raw_contact_fields["area_friction"]
    _equal(plain(pair["identity"]), entry["material_identity"], "restored full material identity")
    _equal(plain(asset.identity), entry["material_identity"], "rebuilt asset identity")
    _require(
        tuple(pair["body_ids"]) == (model.body("tool").id, model.body("object").id),
        "restored ordered physical bodies",
    )
    point = point_loads(pair)
    _equal(
        point,
        {k: last["area"][k] for k in point},
        "restored source/sample IDs and every actual bilateral point force",
    )
    normal, tangent = np.asarray(pair["normal_force"]), np.asarray(pair["tangential_force"])
    area = dict(
        samples=len(normal),
        active_samples=int(np.count_nonzero(normal > 0)),
        normal_load_n=float(normal.sum()),
        zero_normal_load=bool(np.all(normal == 0)),
        pusher_body="tool",
        normal_force_on_pusher_world_n=(normal[:, None] * pair["normals"]).sum(axis=0).tolist(),
        tangential_force_on_pusher_world_n=tangent.sum(axis=0).tolist(),
        tangential_net_norm_n=float(np.linalg.norm(tangent.sum(axis=0))),
        tangential_point_norm_sum_n=float(np.linalg.norm(tangent, axis=1).sum()),
        total_force_on_pusher_world_n=np.asarray(pair["force_on_a"]).sum(axis=0).tolist(),
        **point,
    )
    _equal(area, last["area"], "complete archived area observation")
    _require(
        area["normal_load_n"] == 0
        and area["samples"] == 0
        and np.all(np.asarray(pair["force_on_a"]) == 0)
        and np.all(np.asarray(pair["force_on_b"]) == 0),
        "original last receipt is empty/zero-load; not an endpoint measurement",
    )
    supports = []
    for index, contact in enumerate(receipt.contacts):
        names = tuple(model.geom(int(i)).name for i in contact.geometry_ids)
        if "support" not in names:
            continue
        other = 1 if names[0] == "support" else 0
        force = contact.force_on_geometry2_world if other == 1 else contact.force_on_geometry1_world
        supports.append(
            dict(
                contact_index=index,
                geometry_names=list(names),
                active=contact.active,
                normal_n=float(contact.normal_force),
                force_on_supported_world_n=force.tolist(),
            )
        )
    native = dict(
        detected_contacts=len(receipt.contacts),
        active_contacts=sum(c.active for c in receipt.contacts),
        rows=len(receipt.constraints),
        limit_rows=sum(r.constraint_type == 3 for r in receipt.constraints),
        support_normal_n=sum(s["normal_n"] for s in supports),
        object_support_normal_n=sum(
            s["normal_n"]
            for s in supports
            if any(n in ("object-0", "object-1") for n in s["geometry_names"])
        ),
        supports=supports,
    )
    _equal(native, last["native"], "all archived native rows and support forces")
    _require(
        native["object_support_normal_n"] > 0 and native["rows"] > 0,
        "real retained table support in preceding receipt",
    )
    solver = receipt.raw_contact_fields["area_friction_solver"]
    diagnostics = solver["raw_diagnostics"]
    meta = np.asarray(diagnostics["meta"])
    calls = int(meta[2])
    observed = dict(
        schema=solver["schema"],
        outer_iterations=int(meta[1]),
        native_solve_calls=calls,
        normalized_residual=float(solver["normalized_residual"]),
        output_velocity=np.asarray(solver["output_velocity"]).tolist(),
        meta=meta.tolist(),
        trace_meta=np.asarray(diagnostics["trace_meta"])[:calls].tolist(),
        trace=np.asarray(diagnostics["trace"])[:calls].tolist(),
    )
    _equal(observed, last["solver"], "all archived solver diagnostics")
    _equal(observed["output_velocity"], entry["state"]["v"], "solver output velocity")
    return dict(
        input_tick=last["original_input_tick"],
        output_tick=last["original_output_tick"],
        receipt_local_tick=receipt.tick,
        input_time=receipt.input_time,
        output_time=receipt.output_time,
        area=area,
        native=native,
        solver=observed,
        phase="preceding accepted input-geometry receipt",
    )


def _restore_twice(capsule_path, entry, last, cache_root):
    import numpy as np

    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
        MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
    )
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.training.manipulation_tasks import manipulation_scenes

    from .pusht_public_scene import build_scene_spec, build_side_asset

    source = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    _require(source.source_sha256 == entry["source_model_sha256"], "actual model source SHA256")
    capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(capsule_path.read_bytes())
    saved, leaves = _capsule_content(capsule)
    _require(capsule.asset_sha256 == entry["scene_asset_sha256"], "capsule material SHA256")
    previous = None
    results = []
    for index in range(2):
        # Fresh public declarations/owners. q0 is only the compatible restore
        # destination; it never normalizes or substitutes the saved endpoint.
        spec = build_scene_spec(source.mjcf, side_coverage=entry["side_coverage"])
        model = spec.compilation.mj_model
        asset = build_side_asset(model, side_coverage=entry["side_coverage"])
        initial = MuJoCoInitialState(q=model.qpos0.copy(), v=np.zeros(model.nv))
        with MuJoCoTyxHeterogeneousBatch(
            (spec,), (MuJoCoTyxWorldSpec(6, spec.key, initial),), mode="cuda", cache_root=cache_root
        ) as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(
                batch, {spec.key: (asset,)}, publication_frame=entry["publication_frame"]
            )
            _require(
                scene.asset_sha256 == entry["scene_asset_sha256"]
                and scene.publication_frame == "rigid-origin",
                "actual source asset/frame",
            )
            before_revision = batch.checkpoint_revision
            boundary = scene.restore(capsule, expected_revision=before_revision)
            _require(
                batch.checkpoint_revision == before_revision + 1
                and boundary.operation == "restore"
                and not boundary.cleanup_errors,
                "actual successful public restore with local CAS +1",
            )
            snapshots = batch.integration_snapshot()
            _require(
                tuple(snapshots) == (6,) and not scene.last_geometry,
                "world6 readback and documented cleared geometry cache",
            )
            physical = _receipt_binding(snapshots[6], last, entry, model, asset)
            typed = _readback_form(snapshots)
            if previous is not None:
                _equal(typed, previous, "two fresh owners: entire public integration snapshot")
            previous = typed
            recaptured = scene.checkpoint()
            current, current_leaves = _capsule_content(recaptured)
            _equal(current["binding"], saved["binding"], "recapture model/world/layout binding")
            _require(len(current_leaves) == len(leaves), "unchanged captured leaf count")
            for old, new in zip(leaves, current_leaves, strict=True):
                _equal(new["binding"], old["binding"], "leaf source/capacity binding")
                _equal(
                    new["state"],
                    old["state"],
                    "all captured bytes: device banks, clock, mirrors, receipt and accepted entry",
                )
            publication = deepcopy(saved["publication"])
            _require(publication is not None, "historical accepted aggregate publication")
            publication.setdefault(
                "source_checkpoint",
                {
                    "revision": capsule.batch_checkpoint.source_revision,
                    "sha256": capsule.batch_checkpoint.sha256,
                },
            )
            _equal(
                current["publication"], publication, "historical publication exact source lineage"
            )
            _require(
                current["revision"] == before_revision + 1
                and batch.checkpoint_revision == before_revision + 1,
                "readback/capture does not advance integration or CAS",
            )
            results.append(
                dict(
                    restore_index=index,
                    physics_steps=0,
                    world=6,
                    local_revision_before=before_revision,
                    local_revision_after=batch.checkpoint_revision,
                    absolute_tick=entry["tick"],
                    local_tick=snapshots[6].tick,
                    time=snapshots[6].time,
                    q=snapshots[6].q.tolist(),
                    v=snapshots[6].v.tolist(),
                    integration_fields=[f.name for f in fields(snapshots[6])],
                    integration_snapshot_sha256=_canonical_sha(typed),
                    integration_field_sha256={
                        f.name: _canonical_sha(_readback_form(getattr(snapshots[6], f.name)))
                        for f in fields(snapshots[6])
                    },
                    complete_leaf_state_sha256=[_canonical_sha(x["state"]) for x in current_leaves],
                    complete_leaf_state_matches_source=True,
                    captured_state_sections=sorted(current_leaves[0]["state"]),
                    solver_history=current_leaves[0]["binding"]["solver_history"],
                    historical_publication=current["publication"],
                    geometry_cache_empty=True,
                    publication_frame=scene.publication_frame,
                    last_receipt=physical,
                )
            )
    return results


def verify_fork(source_path, fork_path, *, cache_root=None):
    """Bind saved evidence and perform two new-owner public restores, zero steps."""
    source_path, fork_path = Path(source_path).resolve(), Path(fork_path).resolve()
    source, fork, last, callback, journal, comparison = _bind_records(source_path, fork_path)
    capsule, checkpoint = _checkpoint_binding(fork_path, fork, source["entry"], journal)
    restores = _restore_twice(capsule, source["entry"], last, cache_root)
    return dict(
        schema=SCHEMA,
        status="verified",
        source_binding=dict(
            source=_file(source_path),
            fork=_file(fork_path),
            original_report=source["sources"]["report"],
            original_audit=source["sources"]["audit"],
            model_sha256=source["entry"]["model_sha256"],
            source_model_sha256=source["entry"]["source_model_sha256"],
            query_sha256=source["entry"]["query_sha256"],
            asset_sha256=source["entry"]["scene_asset_sha256"],
            publication_frame=source["entry"]["publication_frame"],
        ),
        withdrawal_comparison=comparison,
        terminal_binding=dict(
            state=source["entry"]["state"],
            tick=source["entry"]["tick"],
            goal=source["entry"]["goal"],
            controller=fork["controller_final"],
            goal_history=fork["goal_history_final"],
            previous_feedback=callback["feedback_accepted"],
            planner_counts_before=source["entry"]["planner_counts_before"],
            remaining_steps=source["budget"]["remaining_steps"],
        ),
        checkpoint_binding=checkpoint,
        restores=restores,
        physics_steps=0,
        independent_restore_count=2,
        limitations=[
            "No new control, planner call or physics step; equality does not establish future trajectory equality.",
            "The capsule owns physics state/receipt/accepted-entry cache, not HOST controller or Goal objects. Their saved summaries and all 250 callbacks were bound, not reconstructed here.",
            "Previous forces and support belong to the last accepted input geometry, not a new endpoint force measurement. No endpoint snapshot() or forward solve was requested by this verifier.",
            "Scene.last_geometry is deliberately empty after restore; publication_frame is supplied explicitly and is not encoded in the asset digest.",
            "Full recaptured leaf state is exact; operational resource counters and new local revisions are not simulated state. Historical aggregate source_checkpoint is checked separately.",
            "SHA256 detects mismatch/corruption, not malicious replacement of an entire evidence chain; no new physical calibration or geometric-accuracy claim.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fork", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    # Reserve before allocating owners; failures retain an explicit failed
    # verification record and propagate the original exception/nonzero exit.
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            result = verify_fork(args.source, args.fork, cache_root=args.cache_root)
        except Exception as error:
            json.dump(
                dict(
                    schema=SCHEMA,
                    status="verification_failed",
                    physics_steps=0,
                    source=str(args.source.resolve()),
                    fork=str(args.fork.resolve()),
                    error={"type": type(error).__name__, "message": str(error)},
                ),
                stream,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            raise
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
