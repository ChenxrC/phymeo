"""ROB1c mixed-storage world commit and real late-restore fault coverage."""

from dataclasses import replace

import pytest
from test_resident_multi_participant import save
from test_resident_multi_participant_robots import (
    FAMILIES,
    WORLDS,
    H,
    audit_receipt,
    build,
    command,
    drives,
    free_box,
    prepare,
    r04,
)

import phymeo as ph
from phymeo.backends import _resident_batch_transaction as transaction


def mixed_case():
    specs, pairs = {}, []
    for pid, family in zip((7, 17, 27, 37), FAMILIES, strict=True):
        specs[pid] = r04(family)["specs"][7]
        pairs.extend(((pid, 0, 59, 0), (pid, 1, 59, 0)))
    box = free_box([0, 0, 0], eccentric=False, mass=4.0)
    box["v"][:] = 0
    specs[59] = box
    return dict(
        name="R05-R06-four-storage",
        specs=specs,
        pairs=tuple(pairs),
        stiffness=64.0,
        friction=0.0,
        slip_speed=1 / 16,
        steps=40,
        switch=None,
    )


def capture(scene, plan):
    return dict(
        states=scene.snapshot(),
        receipts=plan.last_accepted,
        status=plan.last_status,
        epoch=scene.state_epoch,
        bank=scene.stats()["current_buffer"],
        stats=scene.stats(),
    )


def retained(scene, plan, previous):
    assert scene.snapshot() == previous["states"]
    assert plan.last_accepted is previous["receipts"]
    assert plan.last_status is previous["status"]
    assert scene.state_epoch == previous["epoch"]
    assert scene.stats()["current_buffer"] == previous["bank"]


def step(scene, plan, data, built, history):
    before = scene.snapshot()
    assert scene.step_contacts(H, plan) == dict.fromkeys(built["worlds"], True)
    after = scene.snapshot()
    checks = {
        w: audit_receipt(
            data, built, w, before[w], after[w], plan.last_accepted[w], drives(data, before[w].tick)
        )
        for w in built["worlds"]
    }
    history.append(
        dict(
            entry=before,
            accepted=after,
            receipts=plan.last_accepted,
            status=plan.last_status,
            checks=checks,
            stats=scene.stats(),
        )
    )


@pytest.mark.tyx
@pytest.mark.parametrize("last_world", (False, True))
def test_mixed_storage_last_member_rejects_whole_world_then_retry(
    last_world, tyx_mode, tmp_path, monkeypatch
):
    data, history = mixed_case(), []
    built = build(data, tmp_path, worlds=WORLDS[:2], reverse=True)
    target_world = built["worlds"][-1 if last_world else 0]
    target = ph.ResidentParticipantKey(target_world, 37)  # final floating-affine storage
    with ph.ResidentSceneBatch(built["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        scene.set_controls(built["controls"])
        with prepare(scene, built) as plan:
            step(scene, plan, data, built, history)
            previous = capture(scene, plan)
            seen = []

            def reject(service, attempt):
                seen.append(tuple(attempt["rows"]))
                assert set(attempt["rows"]) == set(built["controls"])
                return {target: 6}

            with monkeypatch.context() as patch:
                patch.setattr(transaction.ResidentBatchTransaction, "_before_finalize", reject)
                result = scene.step_contacts(H, plan)
            assert len(seen) == 1
            assert result == {w: w != target_world for w in built["worlds"]}
            now = capture(scene, plan)
            assert now["states"][target_world] == previous["states"][target_world]
            assert now["receipts"][target_world] is previous["receipts"][target_world]
            for w in built["worlds"]:
                expected = 1 if w == target_world else 2
                assert now["states"][w].tick == expected
                assert {p.tick for p in now["states"][w].participants.values()} == {expected}
            history.append(dict(injected=target, before=previous, observed=now, accepted=result))
            step(scene, plan, data, built, history)
            assert scene.snapshot()[target_world].tick == 2
            save(
                tmp_path,
                "mixed-reject-retry",
                dict(
                    inputs=data,
                    target=target,
                    seen=seen,
                    history=history,
                    final=capture(scene, plan),
                ),
            )


@pytest.mark.tyx
def test_mixed_storage_final_publication_exception_has_no_partial_commit(
    tyx_mode, tmp_path, monkeypatch
):
    data, history = mixed_case(), []
    built = build(data, tmp_path, worlds=WORLDS[:2])
    with ph.ResidentSceneBatch(built["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        scene.set_controls(built["controls"])
        with prepare(scene, built) as plan:
            step(scene, plan, data, built, history)
            changed = dict(built["controls"])
            key = ph.ResidentParticipantKey(built["worlds"][-1], 7)
            drive = data["specs"][7]["drive"].copy()
            drive[:, 2] += 1 / 64
            changed[key] = command(data["specs"][7], built["controllers"][key], drive)
            scene.set_controls(changed)
            previous = capture(scene, plan)
            original = transaction.ResidentBatchTransaction._prepare_publication
            seen = []

            def fail(service, *args):
                result = original(service, *args)
                assert len(result.receipts) == 2
                assert all(len(r.participants) == 5 for r in result.receipts.values())
                seen.append(result)
                raise RuntimeError("ROB1c after complete four-storage receipts")

            with monkeypatch.context() as patch:
                patch.setattr(transaction.ResidentBatchTransaction, "_prepare_publication", fail)
                with pytest.raises(RuntimeError, match="after complete"):
                    scene.step_contacts(H, plan)
            assert len(seen) == 1
            retained(scene, plan, previous)
            assert scene.step_contacts(H, plan) == dict.fromkeys(built["worlds"], True)
            assert (
                scene.snapshot()[key.world].participants[key].control
                == previous["states"][key.world].participants[key].control
            )
            save(
                tmp_path,
                "mixed-late-publication",
                dict(
                    inputs=data,
                    previous=previous,
                    failed_publications=seen,
                    after=capture(scene, plan),
                    history=history,
                ),
            )


@pytest.mark.tyx
def test_r06_tick19_checkpoint_replays21_and_partial_world_reset(tyx_mode, tmp_path):
    data, history = mixed_case(), []
    built = build(data, tmp_path, worlds=WORLDS[:2], reverse=True)
    with ph.ResidentSceneBatch(built["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        scene.set_controls(built["controls"])
        with prepare(scene, built) as plan:
            for _ in range(19):
                step(scene, plan, data, built, history)
            checkpoint = scene.snapshot()
            for _ in range(21):
                step(scene, plan, data, built, history)
            expected = scene.snapshot()
            world = built["worlds"][0]
            untouched = expected[built["worlds"][1]]
            scene.restore({world: checkpoint[world]}, expected_epoch=scene.state_epoch)
            assert scene.snapshot()[world] == checkpoint[world]
            assert scene.snapshot()[built["worlds"][1]] == untouched
            for _ in range(21):
                step(scene, plan, data, built, history)
            assert scene.snapshot()[world] == expected[world]
            before_reset = scene.snapshot()
            scene.reset((world,))
            assert scene.snapshot()[world].tick == 0
            assert scene.snapshot()[built["worlds"][1]] == before_reset[built["worlds"][1]]
            scene.set_controls(
                {key: value for key, value in built["controls"].items() if key.world == world}
            )
            step(scene, plan, data, built, history)
            assert scene.snapshot()[world].tick == 1
            save(
                tmp_path,
                "tick19-restore-reset",
                dict(
                    inputs=data,
                    checkpoint=checkpoint,
                    expected=expected,
                    before_reset=before_reset,
                    history=history,
                    final=capture(scene, plan),
                ),
            )


@pytest.mark.tyx
def test_late_restore_after_all_upload_checks_keeps_banks_and_real_retry(
    tyx_mode, tmp_path, monkeypatch
):
    data, history = mixed_case(), []
    built = build(data, tmp_path, worlds=WORLDS[:2])
    with ph.ResidentSceneBatch(built["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        scene.set_controls(built["controls"])
        with prepare(scene, built) as plan:
            step(scene, plan, data, built, history)
            checkpoint = scene.snapshot()
            for _ in range(2):
                step(scene, plan, data, built, history)
            previous = capture(scene, plan)
            calls = []

            def fail(service):
                calls.append(service.state_epoch)
                raise RuntimeError("ROB1c after actual restore stage upload and check")

            with monkeypatch.context() as patch:
                patch.setattr(transaction.ResidentBatchTransaction, "_before_restore_commit", fail)
                with pytest.raises(RuntimeError, match="actual restore"):
                    scene.restore(checkpoint, expected_epoch=scene.state_epoch)
            assert calls == [previous["epoch"]]
            retained(scene, plan, previous)
            scene.restore(checkpoint, expected_epoch=scene.state_epoch)
            assert scene.snapshot() == checkpoint
            step(scene, plan, data, built, history)
            save(
                tmp_path,
                "late-restore",
                dict(
                    inputs=data,
                    checkpoint=checkpoint,
                    previous=previous,
                    failure_calls=calls,
                    history=history,
                    final=capture(scene, plan),
                ),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("missing_participant", "wrong_world", "stale_epoch"))
def test_mixed_checkpoint_rejects_incomplete_identity_before_upload(fault, tyx_mode, tmp_path):
    data = mixed_case()
    built = build(data, tmp_path, worlds=WORLDS[:2])
    with ph.ResidentSceneBatch(built["participants"], mode=tyx_mode, cache_root=tmp_path) as scene:
        scene.set_controls(built["controls"])
        with prepare(scene, built) as plan:
            assert scene.step_contacts(H, plan) == dict.fromkeys(built["worlds"], True)
            previous = capture(scene, plan)
            world = built["worlds"][-1]
            checkpoint = previous["states"][world]
            epoch = scene.state_epoch
            if fault == "missing_participant":
                subset = dict(checkpoint.participants)
                subset.pop(next(reversed(subset)))
                checkpoint = replace(checkpoint, participants=subset)
            elif fault == "wrong_world":
                checkpoint = replace(checkpoint, world=built["worlds"][0])
            else:
                epoch -= 1
            with pytest.raises((ValueError, RuntimeError)):
                scene.restore({world: checkpoint}, expected_epoch=epoch)
            retained(scene, plan, previous)
            save(
                tmp_path,
                "identity-rejection",
                dict(
                    inputs=data,
                    fault=fault,
                    previous=previous,
                    attempted=checkpoint,
                    expected_epoch=epoch,
                    after=capture(scene, plan),
                ),
            )
