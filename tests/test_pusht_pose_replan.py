"""Portable contracts for completed-route reuse and independent pose episodes."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


def _completed_source(tmp_path):
    from test_pusht_active_resume import _json, _saved

    source, audit_path, meta_path, report, audit, metadata, row = _saved(tmp_path)
    report["status"] = "route_complete"
    report["active_route"]["end_tick"] = metadata["source_tick"]
    report["active_route"]["completed_routes_before"] = 1
    report["controller_final"].update(partial_route=None, completed_routes=2)
    metadata["active_route"] = deepcopy(report["active_route"])
    metadata["controller"] = deepcopy(report["controller_final"])
    report["final_checkpoint"]["metadata_sha256"] = _json(meta_path, metadata)
    audit["source"]["summary_sha256"] = _json(source, report)
    _json(audit_path, audit)
    return source, audit_path, meta_path, report, audit, metadata


def test_completed_capsule_requires_explicit_endpoint_mode(tmp_path):
    from experiments.area_pressure.pusht_active_resume import load_resume_source

    source, audit, metadata, *_ = _completed_source(tmp_path)
    with pytest.raises(ValueError):
        load_resume_source(source, audit, metadata, control_variant="load-compensation-half")
    bound = load_resume_source(
        source, audit, metadata, control_variant="load-compensation-half", allow_completed=True
    )
    assert bound["metadata"]["controller"]["partial_route"] is None
    assert bound["metadata"]["source_tick"] == bound["report"]["active_route"]["end_tick"]


def test_completed_capsule_mode_still_rejects_unrecorded_accepted_tail(tmp_path):
    from test_pusht_active_resume import _json

    from experiments.area_pressure.pusht_active_resume import load_resume_source

    source, audit_path, meta_path, report, audit, metadata = _completed_source(tmp_path)
    report["accepted_steps"] += 1
    metadata["accepted_steps"] += 1
    report["final_checkpoint"]["metadata_sha256"] = _json(meta_path, metadata)
    audit["source"]["summary_sha256"] = _json(source, report)
    _json(audit_path, audit)
    with pytest.raises(ValueError):
        load_resume_source(
            source,
            audit_path,
            meta_path,
            control_variant="load-compensation-half",
            allow_completed=True,
        )


def test_independent_pose_reset_uses_registered_yaw_default_robot_and_zero_velocity():
    from experiments.area_pressure.pusht_pose_replan import reset_initial

    model = SimpleNamespace(
        nq=9,
        nv=8,
        qpos0=np.array([0.0, 0.0, 0.52, 0.0, 0.016, 1.0, 0.0, 0.0, 0.0]),
        jnt_qposadr=np.array([0, 1, 2]),
        joint=lambda name: SimpleNamespace(id=2),
    )
    original = model.qpos0.copy()
    terminal = dict(q=[-0.6, 0.94, 0.53, 0.02, 0.01499, 1.0, 0.0, 0.0, 0.0], v=[0.2] * 8)
    initial = reset_initial(model, terminal)
    np.testing.assert_array_equal(initial.q[:2], model.qpos0[:2])
    np.testing.assert_array_equal(initial.q[2:4], terminal["q"][2:4])
    assert initial.q[4] == model.qpos0[4] and initial.time == 0.0
    np.testing.assert_array_equal(initial.v, np.zeros(8))
    np.testing.assert_allclose(
        initial.q[5:9], [np.cos(0.25), 0.0, 0.0, np.sin(0.25)], atol=0, rtol=0
    )
    np.testing.assert_array_equal(model.qpos0, original)


def test_replanning_requires_zero_load_and_strict_pairwise_separation():
    from experiments.area_pressure.pusht_pose_replan import require_separated
    from phymeo.control.manipulation_release_monitor import ReleasePairEvidence, ReleasePairKey

    material = ReleasePairKey("area-material", ("material", "whole"), (1, 2))
    table = ReleasePairKey("native-geometry", ("support", "object-0"), (0, 2))
    clear = ReleasePairEvidence(
        material, 0.0, 0.0, 0.0, None, None, False, False, True, (0, 0, 0), (0, 0, 0)
    )
    support = replace(
        clear, key=table, normal_load_n=1.5, output_contact=True, output_strictly_separated=False
    )
    evidence = SimpleNamespace(pairs=(clear, support))
    assert require_separated(evidence, (table,))["strictly_separated"] is True
    for changed in (
        replace(clear, normal_load_n=1e-12),
        replace(clear, normal_load_n=None),
        replace(clear, output_contact=True),
        replace(clear, output_strictly_separated=False),
        replace(clear, output_depth_m=1e-12),
    ):
        with pytest.raises(ValueError):
            require_separated(SimpleNamespace(pairs=(changed, support)), (table,))


def test_pose_cases_are_independent_even_if_first_case_is_blocked():
    from experiments.area_pressure.pusht_pose_replan import run_independent_cases

    calls = []

    def blocked():
        calls.append("A")
        raise ValueError("strict release rejected")

    def second():
        calls.append("B")
        return {"status": "route_complete", "key": "B"}

    cases = run_independent_cases({"A": blocked, "B": second})
    assert calls == ["A", "B"]
    assert cases["A"]["status"] == "partial_failure"
    assert cases["B"]["status"] == "route_complete"


def _active_seed_record():
    from test_pusht_release_material import _after, _fixture, _previous

    from experiments.area_pressure.pusht_release_entry import ReleaseEntry

    model, asset, before = _fixture()
    after = _after(before)
    before["local_tick"], after["local_tick"] = 2499, 2500
    previous = _previous(before, after)
    measurement = previous.pop("measurement")
    measurement.update(
        world=6,
        input_tick=4199,
        output_tick=4200,
        area_samples=3,
        group_moment_com_world=[0.125, -0.25, 0.375],
    )
    previous.update(
        schema="phymeo-pusht-active-route-step-v1",
        index=480,
        original_input_tick=4199,
        original_output_tick=4200,
        feedback_accepted={"measurement": measurement},
    )
    previous["area"]["samples"] = 3
    entry = ReleaseEntry(None, model, None, None, None, after, previous, 4200, None, {})
    return entry, asset, before, after


def test_active_callback_measurement_builds_unchanged_release_seed_view():
    from experiments.area_pressure.pusht_pose_replan import release_seed_view
    from experiments.area_pressure.pusht_release_material import material_evidence

    entry, asset, before, after = _active_seed_record()
    callback = deepcopy(entry.previous["feedback_accepted"]["measurement"])
    view, provenance = release_seed_view(
        entry, source_report_sha256="a" * 64, source_journal_sha256="b" * 64
    )
    assert view is not entry and view.previous is not entry.previous
    assert "measurement" not in entry.previous
    assert entry.previous["feedback_accepted"]["measurement"] == callback
    assert view.previous["measurement"] == callback
    assert provenance["source_field"] == "feedback_accepted.measurement"
    assert provenance["source_report_sha256"] == "a" * 64
    assert provenance["source_journal_sha256"] == "b" * 64
    assert provenance["input_tick"] == 4199 and provenance["output_tick"] == 4200
    rows = material_evidence(entry.model, asset, before, after, previous=view.previous)
    assert rows[0].normal_load_n == 17.0
    assert rows[0].force_on_a_world == (1.0, 2.0, 3.0)
    assert rows[0].force_on_b_world == (-1.0, -2.0, -3.0)
    assert all(row.normal_load_n is None for row in rows[1:])


@pytest.mark.parametrize("fault", ["input-phase", "opposite-force", "conflicting-flat-copy"])
def test_release_seed_view_rejects_misphased_or_conflicting_callback_force(fault):
    from experiments.area_pressure.pusht_pose_replan import release_seed_view

    entry, *_ = _active_seed_record()
    measurement = entry.previous["feedback_accepted"]["measurement"]
    if fault == "input-phase":
        measurement["input_tick"] -= 1
    elif fault == "opposite-force":
        measurement["group_force_world"][1] += 0.25
    else:
        entry.previous["measurement"] = deepcopy(measurement)
        entry.previous["measurement"]["time_s"] += 0.001
    with pytest.raises(ValueError):
        release_seed_view(entry, source_report_sha256="a" * 64, source_journal_sha256="b" * 64)
