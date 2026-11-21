"""Restore a bound active-route terminal capsule and saved controller callbacks.

Only accepted callbacks are replayed. No dynamics, reference command, PD,
planner, or q/v injection is used to reconstruct the controller/Goal boundary.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

SCHEMA = "phymeo-pusht-active-terminal-checkpoint-v1"
AUDITS = ("phymeo-pusht-active-comparison-v1", "phymeo-pusht-resume-audit-v1")


def _require(ok, reason):
    if not ok:
        raise ValueError("active resume: " + reason)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bad_constant(value):
    raise ValueError("active resume: nonfinite JSON " + value)


def _read(path):
    return json.loads(Path(path).read_text("utf-8-sig"), parse_constant=_bad_constant)


def load_resume_source(report_path, audit_path, metadata_path, *, control_variant):
    """Bind a fully recorded endpoint, its audit, terminal metadata and capsule."""
    report_path, audit_path, metadata_path = map(
        lambda p: Path(p).resolve(), (report_path, audit_path, metadata_path)
    )
    report, audit, metadata = map(_read, (report_path, audit_path, metadata_path))
    count = report["recorded_steps"]
    _require(
        report["schema"] == "phymeo-pusht-active-route-v1"
        and report["world"] == 6
        and report.get("control_variant", "position") == control_variant
        and control_variant in ("position", "load-compensation-half")
        and type(count) is int
        and count > 0
        and count == report["accepted_steps"]
        and report.get("unrecorded_accepted_step") is None,
        "same control variant and nonempty complete accepted prefix required",
    )
    _require(
        audit["schema"] in AUDITS
        and audit["status"] == "verified-saved-prefix"
        and audit["source_endpoint_bound"] is True
        and audit["source"]["summary_sha256"] == _digest(report_path)
        and audit["source"]["journal_sha256"] == report["journal"]["sha256"]
        and audit["verified_steps"] == audit["accepted_steps"] == count
        and audit["verified_final_state"] == report["final_state"]
        and audit.get("unrecorded_accepted_step") is None
        and audit["control"]["variant"] == control_variant
        and (
            control_variant == "position"
            or audit["control"]["load_compensation_verified_steps"] == count
        ),
        "parent audit does not bind the complete saved endpoint",
    )
    saved, state = report["final_checkpoint"], report["final_state"]
    _require(
        metadata["schema"] == saved["schema"] == SCHEMA
        and (report_path.parent / saved["metadata_file"]).resolve() == metadata_path
        and _digest(metadata_path) == saved["metadata_sha256"]
        and (metadata_path.parent / metadata["source_report"]).resolve() == report_path
        and metadata["world"] == 6
        and metadata["state"] == state
        and metadata["q"] == state["q"]
        and metadata["v"] == state["v"]
        and metadata["local_tick"] == state["local_tick"]
        and metadata["time"] == state["time"]
        and metadata["source_tick"] == report["start_tick"] + count
        and metadata["controller"] == report["controller_final"]
        and metadata["controller"]["tick"] == metadata["source_tick"]
        and metadata["controller"]["status"] == "continue"
        and metadata["controller"]["partial_route"] is not None
        and metadata["goal_history"] == report["goal_history_final"]
        and metadata["goal_history"]["tick"] == metadata["source_tick"] - 1
        and metadata["active_route"] == report["active_route"]
        and metadata["target"] == report["goal"]
        and metadata["asset_sha256"] == report["provenance"]["scene_asset_sha256"]
        and metadata["journal"] == report["journal"]
        and metadata["accepted_steps"] == metadata["recorded_steps"] == count,
        "terminal metadata/source state/controller/Goal binding",
    )
    for key in (
        "source_tick",
        "local_tick",
        "time",
        "world",
        "controller_at_physical_endpoint",
        "physical_state_matches_report",
    ):
        _require(saved[key] == metadata[key], "terminal checkpoint summary " + key)
    _require(
        metadata["controller_at_physical_endpoint"] is True
        and metadata["physical_state_matches_report"] is True
        and metadata["source_tick"] < report["active_route"]["end_tick"],
        "terminal capsule lacks a continuing exact controller boundary",
    )
    leaf = Path(metadata["checkpoint_file"])
    _require(
        leaf.name == metadata["checkpoint_file"] and not leaf.is_absolute(),
        "sibling capsule required",
    )
    capsule_path = metadata_path.parent / leaf
    _require(
        capsule_path == (report_path.parent / saved["file"]).resolve()
        and _digest(capsule_path) == saved["sha256"] == metadata["checkpoint_sha256"],
        "terminal capsule bytes",
    )
    journal_path = (report_path.parent / report["journal"]["file"]).resolve()
    _require(
        report["journal"]["records"] == count
        and _digest(journal_path) == report["journal"]["sha256"],
        "parent accepted journal bytes/count",
    )
    failed = report.get("failed_candidate")
    if failed is not None:
        _require(
            failed["restored"] is True
            and failed["rollback_verification"]["passed"] is True
            and failed["input_states"]["6"] == state,
            "failed candidate was not restored to the terminal input",
        )
    return dict(
        report=report,
        metadata=metadata,
        audit=audit,
        report_path=report_path,
        audit_path=audit_path,
        metadata_path=metadata_path,
        capsule_path=capsule_path,
        journal_path=journal_path,
        retry_command=None if failed is None else failed.get("submitted_control"),
    )


def prepare_resume_entry(prior, release, source, audit, metadata, route_output, *, control_variant):
    """Rebuild controller/Goal from the release seed and each bound accepted suffix."""
    from scripts.run_manipulation_control import plain

    from .pusht_active_entry import _accepted, _state, prepare_active_entry

    latest = load_resume_source(source, audit, metadata, control_variant=control_variant)
    chain, seen, current = [], set(), latest
    while True:
        _require(
            current["report_path"] not in seen and len(chain) < 32,
            "cyclic or excessive resume chain",
        )
        seen.add(current["report_path"])
        chain.append(current)
        provenance = current["report"]["provenance"]
        for path, name in ((prior, "prior_summary"), (release, "release_report")):
            _require(_digest(path) == provenance[name + "_sha256"], "original " + name + " source")
        if "parent_report_path" not in provenance:
            break
        base = current["report_path"].parent
        paths = [
            (base / provenance[name + "_path"]).resolve()
            for name in ("parent_report", "parent_audit", "checkpoint_metadata")
        ]
        for path, name in zip(paths, ("parent_report", "parent_audit", "checkpoint_metadata")):
            _require(_digest(path) == provenance[name + "_sha256"], "resume chain " + name)
        current = load_resume_source(*paths, control_variant=control_variant)
    entry = prepare_active_entry(prior, release, route_output)
    for item in reversed(chain):
        report, meta, driver = item["report"], item["metadata"], entry.driver
        _require(
            report["initial_state"] == entry.state
            and report["start_tick"] == entry.tick
            and report["controller_initial"] == plain(driver.summary())
            and driver.active is not None
            and driver.active[0] == report["active_route"]["start_tick"]
            and plain(driver.active[1].nodes) == report["active_route"]["nodes"]
            and driver.routes[-1]["sha256"] == report["active_route"]["sha256"],
            "accepted suffix does not continue the same active controller/route",
        )
        state, history, last, count = entry.state, entry.history, entry.previous, 0
        with item["journal_path"].open("rt", encoding="utf-8-sig") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line, parse_constant=_bad_constant)
                tick = entry.tick + index
                _require(
                    row["schema"] == "phymeo-pusht-active-route-step-v1"
                    and row["index"] == index
                    and row["original_input_tick"] == tick
                    and row["original_output_tick"] == tick + 1
                    and row["input_state"] == state
                    and row["output_state"]["local_tick"] == state["local_tick"] + 1,
                    "accepted suffix publication chain",
                )
                _state(entry.model, row["input_state"])
                _state(entry.model, row["output_state"])
                history = _accepted(entry, row, row, history)
                state, last, count = row["output_state"], row, index + 1
        _require(
            count == report["recorded_steps"]
            and state == meta["state"]
            and plain(history) == meta["goal_history"]
            and plain(driver.summary()) == meta["controller"],
            "replayed terminal controller/Goal endpoint",
        )
        entry = replace(
            entry, history=history, state=deepcopy(state), previous=deepcopy(last), tick=driver.tick
        )
    entry.provenance = dict(
        entry.provenance,
        parent_report_path=str(latest["report_path"]),
        parent_report_sha256=_digest(latest["report_path"]),
        parent_audit_path=str(latest["audit_path"]),
        parent_audit_sha256=_digest(latest["audit_path"]),
        source_tick=entry.tick,
        source_local_tick=entry.state["local_tick"],
        source_time=entry.state["time"],
        source_controller="bound accepted callbacks through terminal capsule; no prior physics replay",
        resume_chain_reports=len(chain),
    )
    return entry, latest


def restore_resume_entry(scene, batch, entry, bound):
    """Use the public capsule restore and bind its retained preceding area receipt."""
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
    )

    from .pusht_active_checkpoint import require_saved_state

    meta = bound["metadata"]
    _require(
        scene.asset_sha256 == meta["asset_sha256"] == entry.provenance["scene_asset_sha256"]
        and meta["state"] == entry.state
        and meta["source_tick"] == entry.tick
        and _digest(bound["capsule_path"]) == meta["checkpoint_sha256"],
        "restore capsule/asset/entry binding",
    )
    capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
        bound["capsule_path"].read_bytes()
    )
    scene.restore(capsule, expected_revision=batch.checkpoint_revision)
    snapshot = batch.integration_snapshot()[6]
    require_saved_state(snapshot, entry.state)
    receipt, previous = snapshot.last_step, entry.previous
    _require(
        receipt is not None
        and receipt.tick == snapshot.tick
        and receipt.world == 6
        and receipt.output_time == snapshot.time
        and receipt.input_time == previous["input_state"]["time"]
        and np.array_equal(receipt.input_q, previous["input_state"]["q"])
        and np.array_equal(receipt.input_v, previous["input_state"]["v"]),
        "restored preceding accepted receipt phase",
    )
    pairs = receipt.raw_contact_fields["area_friction"]
    _require(len(pairs) == 1, "unique preceding material pair")
    force = np.asarray(pairs[0]["force_on_a"]).sum(axis=0)
    _require(
        np.isfinite(force).all()
        and np.array_equal(force, previous["area"]["total_force_on_pusher_world_n"]),
        "restored previous published tool force",
    )
    return dict(
        checkpoint_sha256=meta["checkpoint_sha256"],
        restored_local_tick=snapshot.tick,
        absolute_input_tick=entry.tick,
        time=snapshot.time,
        parent_report_sha256=_digest(bound["report_path"]),
        previous_tool_force_world_n=force.tolist(),
        previous_receipt_bound=True,
    )


def bind_retry_command(bound, tick, command):
    """A restored rejected attempt must receive the exact same causal command."""
    saved = bound["retry_command"]
    if saved is None:
        return None
    _require(saved["tick"] == tick and saved["output_tick"] == tick + 1, "retry command tick")
    for field in ("ctrl", "qfrc_applied", "xfrc_applied"):
        actual, expected = np.asarray(getattr(command, field)), np.asarray(saved[field])
        _require(
            actual.shape == expected.shape
            and np.array_equal(actual, expected)
            and actual.astype(np.float32).tobytes() == expected.astype(np.float32).tobytes(),
            "first retry differs from the original failed command: " + field,
        )
    return True
