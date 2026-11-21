"""Release monitor contracts from explicit per-pair evidence, without physics execution."""

from dataclasses import FrozenInstanceError, replace

import pytest

from phymeo.control.manipulation_release_monitor import (
    ReleaseMonitor,
    ReleasePairEvidence,
    ReleasePairKey,
    ReleaseStep,
)

AREA = ReleasePairKey("area-material", ("tool-pressure", "object-side", "whole"), (1, 2))
SOURCE = ReleasePairKey("native-geometry", ("source-capsule", "object-box"), (1, 2))
TABLE = ReleasePairKey("native-geometry", ("table-plane", "object-box"), (0, 2))
OTHER = ReleasePairKey("native-geometry", ("robot-link", "object-box"), (3, 2))


def evidence(
    key,
    load,
    before,
    after,
    *,
    speed=0.01,
    output_speed=None,
    input_contact=True,
    output_contact=True,
    separated=False,
):
    return ReleasePairEvidence(
        key=key,
        input_depth_m=before,
        output_depth_m=after,
        normal_load_n=load,
        input_normal_speed_m_s=speed,
        output_normal_speed_m_s=speed if output_speed is None else output_speed,
        input_contact=input_contact,
        output_contact=output_contact,
        output_strictly_separated=separated,
        force_on_a_world=(0.0, -load, 0.0),
        force_on_b_world=(0.0, load, 0.0),
    )


def seed():
    return ReleaseStep(
        world=7,
        input_tick=10,
        output_tick=11,
        input_time=0.010,
        output_time=0.011,
        input_q=(0.0,),
        input_v=(0.01,),
        output_q=(0.001,),
        output_v=(0.01,),
        pairs=(
            evidence(AREA, 10.0, 0.002, 0.002),
            evidence(SOURCE, 0.0, 0.001, 0.001),
            evidence(TABLE, 5.0, 0.00001, 0.00001),
            evidence(
                OTHER, 0.0, 0.0, 0.0, input_contact=False, output_contact=False, separated=True
            ),
        ),
    )


def next_step(previous, pairs=None, *, publication="accepted"):
    if pairs is None:
        pairs = (
            evidence(AREA, 9.0, 0.002, 0.001),
            evidence(SOURCE, 0.0, 0.001, 0.0005),
            # Original table support can carry more load; it is not a release pair.
            evidence(TABLE, 7.0, 0.00001, 0.00002, speed=-0.01),
            evidence(
                OTHER, 0.0, 0.0, 0.0, input_contact=False, output_contact=False, separated=True
            ),
        )
    return ReleaseStep(
        world=previous.world,
        input_tick=previous.output_tick,
        output_tick=previous.output_tick + 1,
        input_time=previous.output_time,
        output_time=previous.output_time + 0.001,
        input_q=previous.output_q,
        input_v=previous.output_v,
        output_q=tuple(q + 0.001 for q in previous.output_q),
        output_v=previous.output_v,
        pairs=pairs,
        publication=publication,
    )


def started(**kwargs):
    monitor = ReleaseMonitor(support_keys=(TABLE,), **kwargs)
    initial = seed()
    assert monitor.start(initial) is None
    return monitor, initial


def changed(step, key, **values):
    return replace(
        step, pairs=tuple(replace(row, **values) if row.key == key else row for row in step.pairs)
    )


def codes(decision):
    return {(item.code, item.key) for item in decision.violations}


def snapshot(monitor):
    return monitor.status, monitor.history, monitor.last_step, monitor.terminal


def test_release_is_per_pair_and_keeps_original_table_support_without_authorizing_resume():
    monitor, initial = started()
    first = next_step(initial)
    decision = monitor.observe(first)
    assert decision.status == "continue" and decision.admissible and decision.committed
    assert not decision.release_complete and not decision.regular_control_authorized
    assert decision.violations == () and decision.newly_bound_keys == ()
    second = next_step(
        first,
        tuple(
            evidence(
                row.key,
                0.0,
                row.output_depth_m,
                0.0,
                input_contact=row.output_contact,
                output_contact=False,
                separated=True,
            )
            if row.key in (AREA, SOURCE)
            else replace(row, input_depth_m=row.output_depth_m)
            for row in first.pairs
        ),
    )
    released = monitor.observe(second)
    assert released.status == "released" and released.release_complete
    assert released.admissible and released.committed and not released.regular_control_authorized
    assert monitor.last_step == second and monitor.history == (decision, released)
    assert not monitor.terminal  # Keep guarding the remainder of the same withdrawal.
    third = next_step(
        second,
        tuple(
            replace(row, input_depth_m=row.output_depth_m, input_contact=row.output_contact)
            for row in second.pairs
        ),
    )
    assert monitor.observe(third).status == "released"
    recontact = changed(
        next_step(third, third.pairs),
        AREA,
        output_contact=True,
        output_strictly_separated=False,
        output_depth_m=0.0001,
    )
    rejected = monitor.observe(recontact)
    assert ("contact_reappeared", AREA) in codes(rejected)
    assert rejected.status == "unsafe" and monitor.terminal


@pytest.mark.parametrize(
    "fault, code",
    [
        ("load", "normal_load_increased"),
        ("deeper", "penetration_increased"),
        ("inward-entry", "inward_motion"),
        ("inward-output", "inward_motion"),
    ],
)
def test_real_initial_inertia_or_growth_is_rejected_and_the_accepted_step_is_retained(fault, code):
    monitor, initial = started()
    frame = next_step(initial)
    if fault == "load":
        frame = changed(
            frame,
            AREA,
            normal_load_n=10.000000000001,
            force_on_a_world=(0.0, -10.000000000001, 0.0),
            force_on_b_world=(0.0, 10.000000000001, 0.0),
        )
    elif fault == "deeper":
        frame = changed(frame, AREA, output_depth_m=0.0021)
    elif fault == "inward-entry":
        frame = changed(frame, AREA, input_normal_speed_m_s=-0.000001)
    else:
        frame = changed(frame, AREA, output_normal_speed_m_s=-0.000001)
    result = monitor.observe(frame)
    assert (code, AREA) in codes(result)
    assert result.status == "unsafe" and not result.admissible
    assert result.committed and monitor.last_step == frame and monitor.history == (result,)
    assert monitor.terminal  # The monitor cannot pretend the already accepted physics rolled back.


def test_decreasing_total_load_cannot_hide_another_pair_or_deeper_source_geometry():
    monitor, initial = started()
    frame = changed(
        next_step(initial),
        AREA,
        normal_load_n=1.0,
        force_on_a_world=(0.0, -1.0, 0.0),
        force_on_b_world=(0.0, 1.0, 0.0),
    )
    frame = changed(
        frame,
        SOURCE,
        normal_load_n=0.5,
        output_depth_m=0.0015,
        force_on_a_world=(0.0, -0.5, 0.0),
        force_on_b_world=(0.0, 0.5, 0.0),
    )
    result = monitor.observe(frame)
    assert ("normal_load_increased", SOURCE) in codes(result)
    assert ("penetration_increased", SOURCE) in codes(result)
    assert result.status == "unsafe"


def test_initially_clear_geometry_pair_becoming_contact_is_not_a_new_allowed_support():
    monitor, initial = started()
    frame = changed(
        next_step(initial),
        OTHER,
        output_depth_m=0.0001,
        output_contact=True,
        output_strictly_separated=False,
    )
    result = monitor.observe(frame)
    assert ("new_contact", OTHER) in codes(result)
    assert result.status == "unsafe"


def test_zero_force_and_empty_contact_do_not_certify_strict_separation():
    monitor, initial = started()
    frame = next_step(
        initial,
        tuple(
            evidence(row.key, 0.0, row.output_depth_m, 0.0, output_contact=False)
            if row.key in (AREA, SOURCE)
            else row
            for row in initial.pairs
        ),
    )
    result = monitor.observe(frame)
    assert result.admissible and result.status == "continue" and not result.release_complete


def test_trials_never_advance_history_and_only_the_selected_continuous_step_can_commit():
    monitor, initial = started()
    before = snapshot(monitor)
    trial = changed(next_step(initial, publication="trial"), AREA, input_normal_speed_m_s=-0.1)
    rejected = monitor.assess(trial)
    assert not rejected.admissible and not rejected.committed and snapshot(monitor) == before
    alternative = next_step(initial, publication="trial")
    assert monitor.assess(alternative).admissible and snapshot(monitor) == before
    with pytest.raises(ValueError, match="publication"):
        monitor.observe(alternative)
    selected = replace(alternative, publication="accepted")
    assert monitor.observe(selected).committed
    assert len(monitor.history) == 1 and monitor.last_step == selected
    with pytest.raises(ValueError, match="continuous"):
        monitor.assess(alternative)


def test_missing_face_history_is_bound_on_first_real_step_and_never_invented_as_zero_load():
    face = ReleasePairKey("area-material", ("tool-pressure", "object-side", "face-0"), (1, 2))
    unknown = ReleasePairEvidence(face, None, None, None, None, None, None, None, None, None, None)
    initial = replace(seed(), pairs=seed().pairs + (unknown,))
    monitor = ReleaseMonitor(support_keys=(TABLE,))
    monitor.start(initial)
    first = next_step(initial, next_step(seed()).pairs + (evidence(face, 2.0, 0.0002, 0.0001),))
    trial = monitor.assess(replace(first, publication="trial"))
    assert trial.newly_bound_keys == (face,) and monitor.history == ()
    result = monitor.observe(first)
    assert result.admissible and result.newly_bound_keys == (face,)
    second = next_step(
        first, tuple(replace(row, input_depth_m=row.output_depth_m) for row in first.pairs)
    )
    second = changed(
        second,
        face,
        normal_load_n=3.0,
        force_on_a_world=(0.0, -3.0, 0.0),
        force_on_b_world=(0.0, 3.0, 0.0),
    )
    result = monitor.observe(second)
    assert ("normal_load_increased", face) in codes(result)


@pytest.mark.parametrize("fault", ["world", "tick", "q", "v", "catalog", "phase", "missing"])
def test_missing_or_misbound_evidence_fails_without_changing_monitor(fault):
    monitor, initial = started()
    frame = next_step(initial)
    if fault == "world":
        frame = replace(frame, world=8)
    elif fault == "tick":
        frame = replace(frame, input_tick=20, output_tick=21)
    elif fault == "q":
        frame = replace(frame, input_q=(123.0,))
    elif fault == "v":
        frame = replace(frame, input_v=(-0.01,))
    elif fault == "catalog":
        frame = replace(frame, pairs=frame.pairs[:-1])
    elif fault == "phase":
        frame = replace(frame, force_phase="output-configuration")
    else:
        frame = changed(frame, AREA, normal_load_n=None)
    before = snapshot(monitor)
    with pytest.raises(ValueError):
        monitor.observe(frame)
    assert snapshot(monitor) == before


def test_invalid_action_reaction_or_unknown_support_never_gets_a_release_receipt():
    initial = seed()
    with pytest.raises(ValueError, match="support"):
        ReleaseMonitor(support_keys=(OTHER,)).start(initial)
    monitor, initial = started()
    frame = changed(next_step(initial), AREA, force_on_b_world=(0.0, 8.0, 0.0))
    before = snapshot(monitor)
    with pytest.raises(ValueError, match="reaction"):
        monitor.observe(frame)
    assert snapshot(monitor) == before


def test_budget_and_frozen_history_do_not_turn_a_stationary_loaded_step_into_release():
    monitor, initial = started(max_steps=1)
    result = monitor.observe(next_step(initial))
    assert result.status == "step_budget" and not result.release_complete
    assert not result.regular_control_authorized and monitor.terminal
    with pytest.raises((AttributeError, FrozenInstanceError)):
        result.status = "released"
    with pytest.raises(AttributeError):
        monitor.status = "continue"
