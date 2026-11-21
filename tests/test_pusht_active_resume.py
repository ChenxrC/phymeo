"""Portable terminal-resume bindings; saved fixtures are not physics evidence."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest


def _json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved(tmp_path):
    state = dict(local_tick=2019, time=3.7190001766430214, q=[0.0, 0.0], v=[0.0, 0.0])
    before = dict(local_tick=2018, time=3.718000176595524, q=[0.0, 0.0], v=[0.0, 0.0])
    route = dict(start_tick=1950, end_tick=4200, file="route.json", sha256="r" * 64)
    controller = dict(tick=3719, status="continue", partial_route=dict(start_tick=1950))
    history = dict(tick=3718, time_s=before["time"])
    row = dict(
        schema="phymeo-pusht-active-route-step-v1",
        index=0,
        original_input_tick=3718,
        original_output_tick=3719,
        input_state=before,
        output_state=state,
        area=dict(total_force_on_pusher_world_n=[0.0, -2.0, 0.0]),
    )
    journal_path = tmp_path / "parent.steps.jsonl"
    journal_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    journal = dict(
        file=journal_path.name,
        records=1,
        sha256=hashlib.sha256(journal_path.read_bytes()).hexdigest(),
    )
    capsule = tmp_path / "parent.final.scene"
    capsule.write_bytes(b"portable opaque capsule fixture; not a physical state")
    capsule_sha = hashlib.sha256(capsule.read_bytes()).hexdigest()
    source_path = tmp_path / "parent.json"
    metadata_path = tmp_path / "parent.final.scene.metadata.json"
    metadata = dict(
        schema="phymeo-pusht-active-terminal-checkpoint-v1",
        checkpoint_file=capsule.name,
        checkpoint_sha256=capsule_sha,
        source_report=source_path.name,
        source_tick=3719,
        local_tick=2019,
        world=6,
        time=state["time"],
        state=state,
        q=state["q"],
        v=state["v"],
        asset_sha256="a" * 64,
        active_route=route,
        target={"target_xy": [0.5, 0.1]},
        controller=controller,
        goal_history=history,
        journal=journal,
        accepted_steps=1,
        recorded_steps=1,
        controller_at_physical_endpoint=True,
        physical_state_matches_report=True,
    )
    meta_sha = _json(metadata_path, metadata)
    report = dict(
        schema="phymeo-pusht-active-route-v1",
        status="partial_failure",
        world=6,
        control_variant="load-compensation-half",
        start_tick=3718,
        initial_state=before,
        final_state=state,
        accepted_steps=1,
        recorded_steps=1,
        local_tick_origin=2018,
        controller_initial={"tick": 3718},
        controller_final=controller,
        goal_history_final=history,
        active_route=route,
        goal=metadata["target"],
        provenance=dict(scene_asset_sha256="a" * 64),
        journal=journal,
        final_checkpoint=dict(
            schema=metadata["schema"],
            file=capsule.name,
            sha256=capsule_sha,
            metadata_file=metadata_path.name,
            metadata_sha256=meta_sha,
            source_tick=3719,
            local_tick=2019,
            time=state["time"],
            world=6,
            controller_at_physical_endpoint=True,
            physical_state_matches_report=True,
        ),
    )
    report_sha = _json(source_path, report)
    audit_path = tmp_path / "parent-audit.json"
    audit = dict(
        schema="phymeo-pusht-active-comparison-v1",
        status="verified-saved-prefix",
        source_endpoint_bound=True,
        verified_steps=1,
        accepted_steps=1,
        verified_final_state=state,
        unrecorded_accepted_step=None,
        source=dict(summary_sha256=report_sha, journal_sha256=journal["sha256"]),
        control=dict(variant="load-compensation-half", load_compensation_verified_steps=1),
    )
    _json(audit_path, audit)
    return source_path, audit_path, metadata_path, report, audit, metadata, row


def test_terminal_resume_loads_exact_clock_without_replacing_local_origin(tmp_path):
    from experiments.area_pressure.pusht_active_resume import load_resume_source

    source, audit, metadata, report, *_ = _saved(tmp_path)
    bound = load_resume_source(source, audit, metadata, control_variant="load-compensation-half")
    assert bound["metadata"]["source_tick"] == 3719
    assert bound["metadata"]["state"] == report["final_state"]
    assert bound["metadata"]["local_tick"] == 2019
    assert bound["metadata"]["time"] == 3.7190001766430214
    assert bound["journal_path"] == tmp_path / "parent.steps.jsonl"


@pytest.mark.parametrize(
    "fault", ["audit-hash", "variant", "tail", "controller", "clock", "capsule"]
)
def test_terminal_resume_rejects_unbound_or_noncausal_sources_before_restore(tmp_path, fault):
    from experiments.area_pressure.pusht_active_resume import load_resume_source

    source, audit_path, meta_path, report, audit, metadata, _ = _saved(tmp_path)
    if fault == "audit-hash":
        audit["source"]["summary_sha256"] = "0" * 64
    elif fault == "variant":
        report["control_variant"] = "position"
    elif fault == "tail":
        report["accepted_steps"] += 1
    elif fault == "controller":
        metadata["controller_at_physical_endpoint"] = False
    elif fault == "clock":
        metadata["time"] = 3.719  # Decimal display is not the accepted scalar clock.
    else:
        (tmp_path / "parent.final.scene").write_bytes(b"changed")
    report["final_checkpoint"]["metadata_sha256"] = _json(meta_path, metadata)
    report_sha = _json(source, report)
    if fault != "audit-hash":
        audit["source"]["summary_sha256"] = report_sha
    _json(audit_path, audit)
    with pytest.raises(ValueError):
        load_resume_source(source, audit_path, meta_path, control_variant="load-compensation-half")


def test_terminal_resume_keeps_first_nonzero_compensated_command_binding():
    import mujoco as mj

    from experiments.area_pressure.pusht_active_route import _first_binding
    from phymeo.backends.mujoco_runtime import MuJoCoControl
    from phymeo.control.manipulation_diagnostic import NativeJointPD

    model = mj.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><default><joint limited="true" range="-3 3"/>'
        '<motor ctrllimited="true" ctrlrange="-20 20"/></default><worldbody>'
        '<body><joint name="j0" axis="0 0 1"/><geom type="sphere" size=".1"/>'
        '<body pos=".4 0 0"><joint name="j1" axis="0 0 1"/>'
        '<geom name="pusher" type="sphere" size=".1" pos=".3 0 0"/>'
        '</body></body></worldbody><actuator><motor joint="j0"/><motor joint="j1"/>'
        "</actuator></mujoco>"
    )
    q, v = np.array([0.3, -0.4]), np.zeros(2)
    target = q + np.array([0.01, 0.02])
    pd = NativeJointPD(model, ("j0", "j1"), 25.0, 3.0)
    _, record = pd.command(q, v, target, np.zeros(2))
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    jacobian = np.zeros((3, model.nv))
    mj.mj_jacGeom(model, data, jacobian, None, model.geom("pusher").id)
    direction, force = np.array([0.0, 1.0, 0.0]), np.array([0.0, -2.0, 0.0])
    correction = jacobian.T @ direction
    combined = np.clip(record.unclipped_effort + correction, -19.9, 19.9)
    direct = combined.astype(np.float32)
    route = SimpleNamespace(
        nodes=((0.0, "push", target), (2.25, "push", target)),
        proposal=SimpleNamespace(direction_world=direction),
    )
    entry = SimpleNamespace(model=model, pd=pd, driver=SimpleNamespace(active=(1950, route)))
    receipt = SimpleNamespace(
        tick=2019,
        output_time=3.7190001766430214,
        raw_contact_fields={"area_friction": [{"force_on_a": force[None, :]}]},
    )
    snapshot = SimpleNamespace(q=q, v=v, tick=2019, time=receipt.output_time, last_step=receipt)
    diagnostic = dict(
        active_phase=dict(absolute_tick=3719, route_start_tick=1950, elapsed_s=1.769),
        target_q=target.tolist(),
        target_v=[0.0, 0.0],
        stage="push",
        transport=dict(direct_pd_ctrl=direct.tolist()),
        load_compensation=dict(
            source_local_tick=2019,
            source_output_time=snapshot.time,
            previous_force_on_tool_world_n=force.tolist(),
            gain=0.5,
            enabled=True,
            correction_nm=correction.tolist(),
            combined_effort_nm=combined.tolist(),
        ),
    )
    binding = _first_binding(entry, snapshot, MuJoCoControl(direct), diagnostic)
    assert binding["exact_matches"] is True
    assert binding["independently_recomputed_direct_ctrl"] == direct.tolist()
    diagnostic["load_compensation"]["previous_force_on_tool_world_n"][1] = -3.0
    with pytest.raises(ValueError):
        _first_binding(entry, snapshot, MuJoCoControl(direct), diagnostic)


def test_terminal_restore_uses_public_capsule_and_retains_previous_force(tmp_path, monkeypatch):
    from experiments.area_pressure.pusht_active_resume import (
        load_resume_source,
        restore_resume_entry,
    )
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
    )

    source, audit, metadata, report, _, _, previous = _saved(tmp_path)
    bound = load_resume_source(source, audit, metadata, control_variant="load-compensation-half")
    receipt = SimpleNamespace(
        tick=2019,
        world=6,
        output_time=report["final_state"]["time"],
        input_time=previous["input_state"]["time"],
        input_q=np.asarray(previous["input_state"]["q"]),
        input_v=np.asarray(previous["input_state"]["v"]),
        raw_contact_fields={"area_friction": [{"force_on_a": np.array([[0.0, -2.0, 0.0]])}]},
    )
    snapshot = SimpleNamespace(
        tick=2019,
        time=report["final_state"]["time"],
        q=np.zeros(2),
        v=np.zeros(2),
        last_step=receipt,
    )
    entry = SimpleNamespace(
        state=report["final_state"],
        tick=3719,
        previous=previous,
        provenance={"scene_asset_sha256": "a" * 64},
    )
    sentinel, calls = object(), []
    monkeypatch.setattr(
        MuJoCoTyxHeterogeneousAreaSceneCheckpoint, "from_bytes", staticmethod(lambda raw: sentinel)
    )

    def restore(capsule, *, expected_revision):
        assert capsule is sentinel and expected_revision == 17
        calls.append("public restore")

    scene = SimpleNamespace(asset_sha256="a" * 64, restore=restore)
    batch = SimpleNamespace(checkpoint_revision=17, integration_snapshot=lambda: {6: snapshot})
    proof = restore_resume_entry(scene, batch, entry, bound)
    assert calls == ["public restore"]
    assert proof["restored_local_tick"] == 2019 and proof["absolute_input_tick"] == 3719
    assert proof["time"] == 3.7190001766430214
    assert proof["previous_tool_force_world_n"] == [0.0, -2.0, 0.0]
    receipt.raw_contact_fields["area_friction"][0]["force_on_a"][0, 1] = -3.0
    with pytest.raises(ValueError, match="previous published tool force"):
        restore_resume_entry(scene, batch, entry, bound)


def test_terminal_retry_binds_all_channels_to_the_original_rejected_attempt():
    from experiments.area_pressure.pusht_active_resume import bind_retry_command
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    saved = dict(
        tick=3719,
        output_tick=3720,
        ctrl=[0.5, 0.25],
        qfrc_applied=[0.0] * 8,
        xfrc_applied=[[0.0] * 6 for _ in range(5)],
    )
    command = MuJoCoControl(
        saved["ctrl"], qfrc_applied=saved["qfrc_applied"], xfrc_applied=saved["xfrc_applied"]
    )
    assert bind_retry_command({"retry_command": saved}, 3719, command) is True
    saved["ctrl"][0] = 0.75
    with pytest.raises(ValueError, match="original failed command"):
        bind_retry_command({"retry_command": saved}, 3719, command)


def test_terminal_resume_rejects_a_bound_zero_commit_parent(tmp_path):
    from experiments.area_pressure.pusht_active_resume import load_resume_source

    source, audit_path, meta_path, report, audit, metadata, _ = _saved(tmp_path)
    # Keep every endpoint, clock, count and digest consistent. The sole missing
    # prerequisite is a newly committed row in this proposed parent segment.
    journal_path = tmp_path / report["journal"]["file"]
    journal_path.write_bytes(b"")
    journal = dict(
        file=journal_path.name,
        records=0,
        sha256=hashlib.sha256(journal_path.read_bytes()).hexdigest(),
    )
    report.update(
        start_tick=3719,
        initial_state=deepcopy(report["final_state"]),
        local_tick_origin=2019,
        controller_initial=deepcopy(report["controller_final"]),
        accepted_steps=0,
        recorded_steps=0,
        journal=deepcopy(journal),
    )
    metadata.update(accepted_steps=0, recorded_steps=0, journal=deepcopy(journal))
    report["final_checkpoint"]["metadata_sha256"] = _json(meta_path, metadata)
    report_sha = _json(source, report)
    audit.update(verified_steps=0, accepted_steps=0)
    audit["control"]["load_compensation_verified_steps"] = 0
    audit["source"].update(
        summary_sha256=report_sha,
        journal_sha256=journal["sha256"],
    )
    _json(audit_path, audit)
    with pytest.raises(ValueError, match="nonempty complete accepted prefix"):
        load_resume_source(source, audit_path, meta_path, control_variant="load-compensation-half")
