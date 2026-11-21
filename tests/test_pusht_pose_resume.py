"""Portable B continuation bindings; opaque capsules here are not physics evidence."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def _write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved(tmp_path):
    states = [dict(local_tick=i, time=i * 0.001, q=[0.0], v=[0.0]) for i in range(3)]
    route_file = tmp_path / "route-01.json.gz"
    route_file.write_bytes(b"opaque original route for loader binding")
    route = dict(
        start_tick=1,
        end_tick=5,
        file=str(route_file),
        sha256=hashlib.sha256(route_file.read_bytes()).hexdigest(),
        nodes=[],
        completed_routes_before=0,
    )
    controller = dict(tick=2, status="continue", partial_route=dict(start_tick=1))
    history = dict(tick=1, time_s=0.001)
    events = []
    for i in range(2):
        row = dict(
            index=i,
            selected=True,
            restored=False,
            original_input_tick=i,
            original_output_tick=i + 1,
            input_state=states[i],
            output_state=states[i + 1],
            area=dict(total_force_on_pusher_world_n=[0.0, -2.0, 0.0]),
        )
        events.append(
            dict(
                schema="phymeo-pusht-pose-replan-event-v1",
                event="candidate",
                trial_index=i,
                payload=row,
            )
        )
        events.append(
            dict(
                schema="phymeo-pusht-pose-replan-event-v1",
                event="accepted_callback",
                trial_index=i,
                payload=dict(feedback_accepted={"measurement": {"input_tick": i}}),
            )
        )
    journal_file = tmp_path / "run-B.trials.jsonl"
    journal_file.write_text("".join(json.dumps(x) + "\n" for x in events), encoding="utf-8")
    journal = dict(
        file=journal_file.name,
        records=len(events),
        sha256=hashlib.sha256(journal_file.read_bytes()).hexdigest(),
    )
    capsule = tmp_path / "run-B.final.scene"
    capsule.write_bytes(b"opaque portable capsule; never passed to physical restore")
    source, audit_path, meta_path = (
        tmp_path / name
        for name in ("run-B.json", "comparison.json", "run-B.final.scene.metadata.json")
    )
    meta = dict(
        schema="phymeo-pusht-active-terminal-checkpoint-v1",
        source_report=source.name,
        checkpoint_file=capsule.name,
        checkpoint_sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),
        source_tick=2,
        local_tick=2,
        time=0.002,
        world=6,
        state=states[-1],
        q=states[-1]["q"],
        v=states[-1]["v"],
        controller=controller,
        goal_history=history,
        target={"target_yaw": 0.2},
        active_route=route,
        asset_sha256="a" * 64,
        control_variant="load-compensation-half",
        journal=journal,
        accepted_steps=2,
        recorded_steps=2,
        controller_at_physical_endpoint=True,
        physical_state_matches_report=True,
    )
    meta_sha = _write(meta_path, meta)
    saved = dict(
        schema=meta["schema"],
        file=capsule.name,
        sha256=meta["checkpoint_sha256"],
        metadata_file=meta_path.name,
        metadata_sha256=meta_sha,
        **{
            k: meta[k]
            for k in (
                "source_tick",
                "local_tick",
                "time",
                "world",
                "controller_at_physical_endpoint",
                "physical_state_matches_report",
            )
        },
    )
    command = dict(
        tick=2, output_tick=3, ctrl=[0.25, -0.5], qfrc_applied=[0.0], xfrc_applied=[[0.0] * 6]
    )
    report = dict(
        schema="phymeo-pusht-pose-replan-case-v1",
        key="B",
        world=6,
        initialization_mode="independent-pose-reset",
        start_tick=0,
        local_tick_origin=0,
        status="partial_failure",
        control_variant="load-compensation-half",
        initial_state=states[0],
        final_state=states[-1],
        accepted_steps=2,
        recorded_steps=2,
        controller_final=controller,
        goal_history_final=history,
        goal=meta["target"],
        active_route=route,
        provenance=dict(scene_asset_sha256="a" * 64),
        journal=journal,
        final_checkpoint=saved,
        failed_candidate=dict(
            restored=True,
            rollback_verification={"passed": True},
            input_states={"6": states[-1]},
            control=command,
        ),
    )
    report_sha = _write(source, report)
    audit = dict(
        schema="phymeo-pusht-pose-replan-comparison-v1",
        status="verified-saved-prefixes",
        cases={
            "B": dict(
                schema="phymeo-pusht-pose-replan-case-audit-v1",
                status="verified-saved-prefix",
                accepted_steps=2,
                recorded_steps=2,
                unrecorded_accepted_step=None,
                reported_final_state=states[-1],
                source=dict(file=str(source), sha256=report_sha, journal_sha256=journal["sha256"]),
            )
        },
    )
    _write(audit_path, audit)
    return source, audit_path, meta_path, report, audit, meta


def test_pose_resume_binds_exact_endpoint_and_original_failed_command(tmp_path):
    from experiments.area_pressure.pusht_active_resume import bind_retry_command
    from experiments.area_pressure.pusht_pose_resume import load_pose_source

    source, audit, meta, report, *_ = _saved(tmp_path)
    bound = load_pose_source(source, audit, meta)
    assert bound["metadata"]["state"] == report["final_state"]
    assert bound["previous"]["original_output_tick"] == 2
    assert bound["previous"]["feedback_accepted"]["measurement"]["input_tick"] == 1
    assert bound["retry_command"] == report["failed_candidate"]["control"]
    command = SimpleNamespace(
        **{
            k: np.asarray(v)
            for k, v in bound["retry_command"].items()
            if k in ("ctrl", "qfrc_applied", "xfrc_applied")
        }
    )
    assert bind_retry_command(bound, 2, command) is True
    command.ctrl[0] += 0.125
    with pytest.raises(ValueError, match="first retry"):
        bind_retry_command(bound, 2, command)


@pytest.mark.parametrize(
    "fault", ["audit", "goal", "clock", "incomplete", "retry", "route", "capsule"]
)
def test_pose_resume_rejects_unbound_source_before_physics(tmp_path, fault):
    from experiments.area_pressure.pusht_pose_resume import load_pose_source

    source, audit_path, meta_path, report, audit, meta = _saved(tmp_path)
    if fault == "audit":
        audit["cases"]["B"]["source"]["sha256"] = "0" * 64
    elif fault == "goal":
        meta["goal_history"] = dict(meta["goal_history"], tick=0)
    elif fault == "clock":
        meta["time"] += 0.0001
    elif fault == "incomplete":
        report["accepted_steps"] += 1
    elif fault == "retry":
        report["failed_candidate"]["control"]["tick"] -= 1
    elif fault == "route":
        (tmp_path / "route-01.json.gz").write_bytes(b"different route")
    else:
        (tmp_path / "run-B.final.scene").write_bytes(b"different capsule")
    report["final_checkpoint"]["metadata_sha256"] = _write(meta_path, meta)
    digest = _write(source, report)
    if fault != "audit":
        audit["cases"]["B"]["source"]["sha256"] = digest
    _write(audit_path, audit)
    with pytest.raises(ValueError):
        load_pose_source(source, audit_path, meta_path)


def test_pose_resume_dispatches_existing_loop_with_bound_prepared_entry(monkeypatch, tmp_path):
    from experiments.area_pressure import pusht_active_route, pusht_pose_resume

    entry, bound = object(), {"report": {"key": "B"}}
    seen = {}
    monkeypatch.setattr(pusht_pose_resume, "prepare_pose_resume_entry", lambda *a: (entry, bound))

    def run(*args, **kwargs):
        seen.update(kwargs)
        return {"status": "route_complete"}

    monkeypatch.setattr(pusht_active_route, "run", run)
    result = pusht_pose_resume.run(
        "prior.json", "run-B.json", "comparison.json", "meta.json", tmp_path / "continued.json"
    )
    assert result["status"] == "route_complete"
    assert seen["prepared_resume"] == (entry, bound)
    assert seen["control_variant"] == "load-compensation-half"


def _with_terminal_tail(tmp_path):
    from copy import deepcopy

    source, audit_path, meta_path, report, audit, meta = _saved(tmp_path)
    report["failure_stage"], report["errors"] = "scene_step", ["saved physical rejection"]
    journal_path = tmp_path / report["journal"]["file"]
    prefix = journal_path.read_bytes()
    tail = [
        dict(
            schema="phymeo-pusht-pose-replan-event-v1",
            event="failure",
            trial_index=2,
            payload=deepcopy(
                {
                    k: report.get(k)
                    for k in (
                        "failure_stage",
                        "errors",
                        "failed_candidate",
                        "unrecorded_accepted_step",
                    )
                }
            ),
        ),
        dict(
            schema="phymeo-pusht-pose-replan-event-v1",
            event="final",
            payload=deepcopy({k: v for k, v in report.items() if k != "journal"}),
        ),
    ]
    journal_path.write_bytes(prefix + b"".join((json.dumps(row) + "\n").encode() for row in tail))
    report["journal"] = dict(
        file=journal_path.name,
        records=meta["journal"]["records"] + 2,
        sha256=hashlib.sha256(journal_path.read_bytes()).hexdigest(),
    )
    audit["cases"]["B"]["source"].update(
        sha256=_write(source, report), journal_sha256=report["journal"]["sha256"]
    )
    _write(audit_path, audit)
    return source, audit_path, meta_path, report, audit, meta, prefix, tail


def test_pose_capsule_binds_exact_journal_prefix_before_failure_and_final_events(tmp_path):
    from experiments.area_pressure.pusht_pose_resume import load_pose_source

    source, audit, meta, report, _, metadata, *_ = _with_terminal_tail(tmp_path)
    assert metadata["journal"]["records"] + 2 == report["journal"]["records"]
    bound = load_pose_source(source, audit, meta)
    proof = bound["checkpoint_journal_binding"]
    assert proof["checkpoint_records"] == 4 and proof["terminal_events"] == ["failure", "final"]
    assert proof["checkpoint_sha256"] == metadata["journal"]["sha256"]
    assert bound["previous"]["original_output_tick"] == 2


@pytest.mark.parametrize(
    "fault", ["prefix-hash", "additional-physics", "wrong-failure", "wrong-final"]
)
def test_pose_capsule_journal_prefix_cannot_hide_new_physics_or_rewrite_terminal_evidence(
    tmp_path, fault
):
    from copy import deepcopy

    from experiments.area_pressure.pusht_pose_resume import load_pose_source

    source, audit_path, meta_path, report, audit, meta, prefix, tail = _with_terminal_tail(tmp_path)
    if fault == "prefix-hash":
        meta["journal"]["sha256"] = "0" * 64
        report["final_checkpoint"]["metadata_sha256"] = _write(meta_path, meta)
        tail[-1]["payload"]["final_checkpoint"] = deepcopy(report["final_checkpoint"])
    elif fault == "additional-physics":
        tail.insert(
            0,
            dict(
                schema="phymeo-pusht-pose-replan-event-v1",
                event="candidate",
                trial_index=2,
                payload={},
            ),
        )
    elif fault == "wrong-failure":
        tail[0]["payload"]["failed_candidate"]["control"]["tick"] -= 1
    else:
        tail[-1]["payload"]["final_state"]["time"] += 0.001
    journal = tmp_path / report["journal"]["file"]
    journal.write_bytes(prefix + b"".join((json.dumps(row) + "\n").encode() for row in tail))
    report["journal"].update(
        records=meta["journal"]["records"] + len(tail),
        sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
    )
    audit["cases"]["B"]["source"].update(
        sha256=_write(source, report), journal_sha256=report["journal"]["sha256"]
    )
    _write(audit_path, audit)
    with pytest.raises(ValueError, match="checkpoint journal"):
        load_pose_source(source, audit_path, meta_path)
