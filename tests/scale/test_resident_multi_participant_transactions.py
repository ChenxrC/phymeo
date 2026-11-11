"""ROB1b world-complete failure, retry, reset and restore; authoring only."""

from dataclasses import replace

import pytest
from test_resident_multi_participant import (
    PARTICIPANTS,
    H,
    encode,
    inputs,
    physical_state,
    record,
    save,
)

import phymeo as ph
from phymeo.backends import _resident_batch_transaction as transaction


def prepare(scene, data, **kwargs):
    options = dict(
        candidate_capacity=len(data["worlds"]), site_capacity=128, narrow_phase="host_patch"
    )
    options.update(kwargs)
    return scene.prepare_contacts(data["model"], data["bindings"], data["laws"], **options)


def snapshot(scene, plan):
    return dict(
        states=scene.snapshot(),
        receipts=plan.last_accepted,
        status=plan.last_status,
        epoch=scene.state_epoch,
        bank=scene.stats()["current_buffer"],
    )


def assert_retained(scene, plan, previous):
    assert scene.snapshot() == previous["states"]
    assert plan.last_accepted is previous["receipts"]
    assert plan.last_status is previous["status"]
    assert scene.state_epoch == previous["epoch"]
    assert scene.stats()["current_buffer"] == previous["bank"]


@pytest.mark.tyx
@pytest.mark.parametrize("last_world", (False, True))
def test_last_participant_rejection_retains_entire_world_then_real_retry(
    last_world, tyx_mode, tmp_path, monkeypatch
):
    data = inputs(2, reverse=True, external=True)
    target_world = data["worlds"][-1 if last_world else 0]
    target = ph.ResidentParticipantKey(target_world, PARTICIPANTS[-1])
    rows = []
    with ph.ResidentSceneBatch(data["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        scene.set_controls(data["controls"])
        with prepare(scene, data) as plan:
            assert scene.step_contacts(H, plan) == dict.fromkeys(data["worlds"], True)
            before = snapshot(scene, plan)
            original = transaction.ResidentBatchTransaction._before_finalize
            seen = []

            def reject_last(service, attempt):
                assert original(service, attempt) is None
                assert target in attempt["rows"]
                seen.append(dict(attempt["rows"]))
                return {target: 6}

            with monkeypatch.context() as patch:
                patch.setattr(transaction.ResidentBatchTransaction, "_before_finalize", reject_last)
                accepted = scene.step_contacts(H, plan)
            assert len(seen) == 1
            assert accepted == {w: w != target_world for w in data["worlds"]}
            retained = scene.snapshot()
            assert retained[target_world] == before["states"][target_world]
            assert plan.last_accepted[target_world] is before["receipts"][target_world]
            for world in data["worlds"]:
                expected_tick = 1 if world == target_world else 2
                assert retained[world].tick == expected_tick
                assert all(p.tick == expected_tick for p in retained[world].participants.values())
            rows.append(record(scene, plan, before["states"]))
            assert scene.step_contacts(H, plan) == dict.fromkeys(data["worlds"], True)
            after = scene.snapshot()
            assert after[target_world].tick == 2
            assert all(p.tick == 2 for p in after[target_world].participants.values())
            rows.append(record(scene, plan, retained))
            save(
                tmp_path,
                "last-participant-reject-retry",
                dict(inputs=data, target=target, injected_rows=seen, before=before, rows=rows),
            )


@pytest.mark.tyx
def test_late_receipt_exception_keeps_every_world_and_committed_controls(
    tyx_mode, tmp_path, monkeypatch
):
    data = inputs(2)
    with ph.ResidentSceneBatch(data["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        scene.set_controls(data["controls"])
        with prepare(scene, data) as plan:
            assert scene.step_contacts(H, plan) == dict.fromkeys(data["worlds"], True)
            changed = inputs(2, external=True)["controls"]
            scene.set_controls(changed)
            before = snapshot(scene, plan)
            original = transaction.ResidentBatchTransaction._prepare_publication
            called = []

            def fail_after_full_receipt(service, *args):
                result = original(service, *args)
                called.append(result)
                raise RuntimeError("ROB1b injected final receipt failure")

            with monkeypatch.context() as patch:
                patch.setattr(
                    transaction.ResidentBatchTransaction,
                    "_prepare_publication",
                    fail_after_full_receipt,
                )
                with pytest.raises(RuntimeError, match="injected final receipt"):
                    scene.step_contacts(H, plan)
            assert len(called) == 1
            assert_retained(scene, plan, before)
            assert scene.step_contacts(H, plan) == dict.fromkeys(data["worlds"], True)
            after = scene.snapshot()
            for world in data["worlds"]:
                for key, state in after[world].participants.items():
                    assert state.control == before["states"][world].participants[key].control
            save(
                tmp_path,
                "late-receipt-retain-retry",
                dict(
                    inputs=data,
                    submitted_controls=changed,
                    before=before,
                    prepared_unpublished=called,
                    retry=record(scene, plan, before["states"]),
                ),
            )


@pytest.mark.tyx
def test_world_site_capacity_is_shared_and_candidate_overflow_is_global(tyx_mode, tmp_path):
    data = inputs(2)
    with ph.ResidentSceneBatch(data["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        scene.set_controls(data["controls"])
        initial = scene.snapshot()
        before_stats = dict(scene.stats())
        with pytest.raises(ValueError, match="candidate|capacity"):
            prepare(scene, data, candidate_capacity=1)
        assert scene.snapshot() == initial
        assert scene.stats()["device_owned_bytes"] == before_stats["device_owned_bytes"]
        bad = data["worlds"][-1]
        caps = {world: (0 if world == bad else 128) for world in data["worlds"]}
        with prepare(scene, data, site_capacity=caps) as limited:
            assert scene.step_contacts(H, limited) == {w: w != bad for w in data["worlds"]}
            after = scene.snapshot()
            assert after[bad] == initial[bad]
            assert bad not in limited.last_accepted
            rejected = record(scene, limited, initial)
        with prepare(scene, data) as full:
            assert scene.step_contacts(H, full) == dict.fromkeys(data["worlds"], True)
            assert scene.snapshot()[bad].tick == 1
            save(
                tmp_path,
                "site-capacity-world-scope",
                dict(
                    inputs=data,
                    capacities=caps,
                    rejected=rejected,
                    retry=record(scene, full, after),
                ),
            )


@pytest.mark.tyx
def test_complete_world_restore_cas_replay_and_partial_reset(tyx_mode, tmp_path):
    data = inputs(2, reverse=True, transformed=True, external=True)
    with ph.ResidentSceneBatch(data["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        initial = scene.snapshot()
        scene.set_controls(data["controls"])
        with prepare(scene, data) as plan:
            for _ in range(3):
                assert scene.step_contacts(H, plan) == dict.fromkeys(data["worlds"], True)
            saved = scene.snapshot()
            continuation = []
            for _ in range(3):
                before = scene.snapshot()
                assert scene.step_contacts(H, plan) == dict.fromkeys(data["worlds"], True)
                continuation.append(record(scene, plan, before))
            old = scene.state_epoch
            current = snapshot(scene, plan)
            with pytest.raises(ValueError, match="epoch"):
                scene.restore(saved, expected_epoch=old - 1)
            assert_retained(scene, plan, current)
            world = data["worlds"][-1]
            with pytest.raises(ValueError, match="participant|complete|membership"):
                incomplete = replace(
                    saved[world],
                    participants={
                        next(iter(saved[world].participants)): next(
                            iter(saved[world].participants.values())
                        )
                    },
                )
                scene.restore({world: incomplete}, expected_epoch=scene.state_epoch)
            assert_retained(scene, plan, current)
            old_receipt = plan.last_accepted[world]
            scene.restore(saved, expected_epoch=scene.state_epoch)
            assert scene.snapshot() == saved
            with pytest.raises(ValueError, match="stale"):
                plan.require_current(old_receipt)
            replay = []
            for index in range(3):
                before = scene.snapshot()
                assert scene.step_contacts(H, plan) == dict.fromkeys(data["worlds"], True)
                row = record(scene, plan, before)
                replay.append(row)
                assert physical_state(row["accepted"]) == physical_state(
                    continuation[index]["accepted"]
                )
                for world in data["worlds"]:
                    actual, expected = (
                        row["receipts"][world],
                        continuation[index]["receipts"][world],
                    )
                    assert encode(actual.sites) == encode(expected.sites)
                    assert encode(actual.participants) == encode(expected.participants)
            before_reset = scene.snapshot()
            reset_world, kept_world = data["worlds"]
            scene.reset((reset_world,))
            reset = scene.snapshot()
            assert reset[reset_world] == initial[reset_world]
            assert reset[kept_world] == before_reset[kept_world]
            assert scene.step_contacts(H, plan) == dict.fromkeys(data["worlds"], True)
            save(
                tmp_path,
                "restore-replay-and-partial-reset",
                dict(
                    inputs=data,
                    saved=saved,
                    continuation=continuation,
                    replay=replay,
                    before_reset=before_reset,
                    reset=reset,
                    next=record(scene, plan, reset),
                ),
            )


@pytest.mark.tyx
def test_bad_last_control_and_foreign_world_checkpoint_are_atomic(tyx_mode, tmp_path):
    data = inputs(2)
    with ph.ResidentSceneBatch(
        data["participants"], mode=tyx_mode, cache_root=tmp_path / "a"
    ) as scene:
        with ph.ResidentSceneBatch(
            data["participants"], mode=tyx_mode, cache_root=tmp_path / "b"
        ) as other:
            before = scene.snapshot()
            controls = dict(data["controls"])
            controls[max(controls)] = object()
            with pytest.raises((ValueError, TypeError)):
                scene.set_controls(controls)
            assert scene.snapshot() == before
            with pytest.raises(ValueError, match="owner|source|another"):
                scene.restore(other.snapshot(), expected_epoch=scene.state_epoch)
            assert scene.snapshot() == before
            with prepare(scene, data) as plan:
                with pytest.raises(ValueError, match="owner|another"):
                    other.step_contacts(H, plan)
                assert other.snapshot() != {}
                assert scene.snapshot() == before
                save(
                    tmp_path,
                    "bad-source-and-control",
                    dict(
                        inputs=data,
                        initial=before,
                        retained=scene.snapshot(),
                        other=other.snapshot(),
                        stats=scene.stats(),
                    ),
                )


@pytest.mark.tyx
def test_invalid_source_still_allows_plan_and_scene_resource_release(
    tyx_mode, tmp_path, monkeypatch
):
    data = inputs(1)
    scene = ph.ResidentSceneBatch(data["participants"], mode=tyx_mode, cache_root=tmp_path)
    plan = prepare(scene, data)

    def invalid_source(layout):
        raise ValueError("injected stale participant source")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(type(scene.layout), "require_current", invalid_source)
            with pytest.raises(ValueError, match="stale participant source"):
                scene.step_contacts(H, plan)
            plan.close()
            scene.close()
            assert plan.stats()["closed"]
            assert scene.stats()["scene_closed"]
            assert scene.stats()["device_owned_bytes"] == 0
    finally:
        scene.close()
