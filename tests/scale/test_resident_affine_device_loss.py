"""A permanent restore/recovery sync loss requires a new affine batch owner.

Adjacent C5 coverage only: the original mixed A03/FreeSphere input and gates are
reused. Removing the injected callback never clears the old batch's poison.
UNEXECUTED by the author; interpret/CUDA execution belongs to the parent.
"""

import pytest
from test_resident_affine_step import (
    H,
    _action,
    _check_readback,
    _close,
    _json,
    _load,
    _ref,
    _save,
    _world,
)

from phymeo.backends import tyx_world_batch as resident


@pytest.mark.tyx
@pytest.mark.parametrize("warm_steps", (0, 1))
def test_permanent_restore_sync_loss_requires_checkpoint_rebuild(
    warm_steps, tyx_mode, tmp_path, monkeypatch
):
    data = _ref().case("A03")
    asset = _load(tmp_path, data)
    world = _world(asset, data, delay=1)
    worlds = [(7, world), (23, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0)))]
    actions = {7: _action(world, asset, data), 23: (1.0, 0.0, 0.0)}
    record = dict(input=data, h_s=H, warm_steps=warm_steps, mode=tyx_mode, commands=actions)
    try:
        with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "control") as control:
            with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "lost") as lost:
                lost.set_controls(actions)
                control.set_controls(actions)
                for _ in range(warm_steps):
                    assert lost.step(H) == control.step(H) == {7: True, 23: True}
                # Restore would move both clocks and the sphere back one actual
                # step, and restore the complete K/N states and delayed command.
                restore_to = lost.snapshot()
                assert lost.step(H) == control.step(H) == {7: True, 23: True}
                before = lost.snapshot()
                before_fk = lost.tree_kinematics([7])
                before_scalar = lost.kinematics([23])
                before_stats = lost.stats()
                assert before == control.snapshot()
                assert before_stats["current_buffer"] == (warm_steps + 1) % 2
                assert before_stats["state_epoch"] == warm_steps + 1
                assert all(point.tick == warm_steps + 1 for point in before.values())
                assert all(point.tick == warm_steps for point in restore_to.values())
                assert before[23].position != restore_to[23].position
                record.update(
                    restore_target=restore_to,
                    before_checkpoint=before,
                    before_tree_kinematics=before_fk,
                    before_scalar_kinematics=before_scalar,
                    before_stats=before_stats,
                )
                _save(tmp_path, "permanent-device-loss-input", record)

                calls = []
                original_sync = lost._sync

                def permanent_sync_loss():
                    # Drain real queued work safely, then fail the API at both
                    # the restore barrier and the attempted recovery barrier.
                    original_sync()
                    calls.append("restore" if not calls else "recovery")
                    raise RuntimeError("C5 permanent device synchronization loss")

                with monkeypatch.context() as patch:
                    patch.setattr(lost, "_sync", permanent_sync_loss)
                    with pytest.raises(RuntimeError, match="C5 permanent device") as failure:
                        lost.restore(restore_to)
                record["restore_failure"] = dict(
                    type=type(failure.value).__name__, message=str(failure.value)
                )
                assert calls == ["restore", "recovery"]
                assert lost._unavailable is True
                failed_stats = lost.stats()
                assert failed_stats["current_buffer"] == before_stats["current_buffer"]
                assert failed_stats["state_epoch"] == before_stats["state_epoch"]
                assert (
                    failed_stats["recovery_sync_attempts"]
                    == before_stats["recovery_sync_attempts"] + 1
                )
                record.update(sync_calls=calls.copy(), failed_stats=failed_stats)

                # The callback has already been restored. Every public data or
                # mutation path must still reject before touching either bank;
                # no failed-owner readback is passed off as an accepted snapshot.
                blocked = {}
                operations = (
                    ("snapshot", lost.snapshot),
                    ("tree_kinematics", lost.tree_kinematics),
                    ("tree_diagnostics", lost.tree_diagnostics),
                    ("scalar_kinematics", lambda: lost.kinematics([23])),
                    ("summary", lost.summary),
                    ("step", lambda: lost.step(H)),
                    ("restore", lambda: lost.restore(before)),
                    ("set_controls", lambda: lost.set_controls(actions)),
                    ("reset", lost.reset),
                )
                for name, operation in operations:
                    with pytest.raises(
                        RuntimeError, match="rebuild from a saved checkpoint"
                    ) as error:
                        operation()
                    blocked[name] = dict(type=type(error.value).__name__, message=str(error.value))
                assert calls == ["restore", "recovery"]
                assert lost._unavailable is True
                after_blocked = lost.stats()
                for name in (
                    "current_buffer",
                    "state_epoch",
                    "launch_attempts",
                    "synchronizations",
                    "recovery_sync_attempts",
                ):
                    assert after_blocked[name] == failed_stats[name]
                record.update(blocked_operations=blocked, after_blocked_stats=after_blocked)

            # Close the poisoned owner first. Rebuild from the saved successful
            # checkpoint, not from the unverified restore destination or scratch.
            assert lost.stats()["device_owned_bytes"] == 0
            record["closed_stats"] = lost.stats()
            with resident.TyxWorldBatch(
                worlds, tyx_mode, cache_root=tmp_path / "rebuilt"
            ) as rebuilt:
                rebuilt.restore(before)
                restored = rebuilt.snapshot()
                restored_fk = rebuilt.tree_kinematics([7])
                restored_scalar = rebuilt.kinematics([23])
                assert restored == before == control.snapshot()
                assert _json(restored_fk) == _json(before_fk)
                assert _json(restored_scalar) == _json(before_scalar)
                assert rebuilt.tree_diagnostics() == {7: None}
                record.update(
                    restored_checkpoint=restored,
                    restored_tree_kinematics=restored_fk,
                    restored_scalar_kinematics=restored_scalar,
                    restored_stats=rebuilt.stats(),
                    continuation=[],
                )
                # No set_controls call here: complete saved held controls and the
                # delay queue themselves must drive both subsequent real steps.
                for index in range(2):
                    entry = rebuilt.snapshot()
                    assert entry == control.snapshot()
                    assert rebuilt.step(H) == control.step(H) == {7: True, 23: True}
                    after, expected = rebuilt.snapshot(), control.snapshot()
                    assert after == expected
                    diagnostic, fk, oracle = _check_readback(
                        rebuilt, 7, asset, data, entry[7], after[7]
                    )
                    assert _json(fk) == _json(control.tree_kinematics([7])[7])
                    tick = warm_steps + 2 + index
                    assert after[7].tick == after[23].tick == tick
                    # Same predeclared 1 N on 2 kg; no free-body state injection.
                    _close(after[23].velocity, [tick * H / 2, 0, 0])
                    _close(after[23].position, [H * H * tick * (tick + 1) / 4, 0, 0])
                    record["continuation"].append(
                        dict(
                            before=entry,
                            rebuilt=after,
                            independent_control=expected,
                            diagnostic=diagnostic,
                            kinematics=fk,
                            oracle=oracle,
                        )
                    )
                record.update(rebuilt_stats=rebuilt.stats(), control_stats=control.stats())
    finally:
        _save(tmp_path, "permanent-device-loss-result", record)


@pytest.mark.tyx
@pytest.mark.parametrize("warm_steps", (0, 1))
def test_ordinary_affine_and_scalar_share_control_restore_and_retry_transaction(
    warm_steps, tyx_mode, tmp_path, monkeypatch
):
    from dataclasses import replace

    import numpy as np

    import phymeo as ph

    data = _ref().case("A03")
    asset = _load(tmp_path, data)
    affine = _world(asset, data, delay=1)
    program = ph.DrivenRobotProgram(
        asset.tree.model,
        kp=np.zeros(3),
        kd=np.zeros(3),
        effort_limit=np.full(3, 16.0),
        delay_steps=1,
    )
    ordinary = resident.FixedRobotTree(
        program, program.initial_state(ph.RobotState(np.zeros(3), np.zeros(3)))
    )
    worlds = [(101, ordinary), (7, affine), (23, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0)))]
    actions = {
        101: program.inputs(np.zeros(3), feedforward=np.array([1.0, -2.0, 3.0])),
        7: _action(affine, asset, data),
        23: (1.0, 0.0, 0.0),
    }
    accepted = {101: True, 7: True, 23: True}
    with (
        resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "trial") as trial,
        resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "control") as control,
    ):
        for batch in (trial, control):
            batch.set_controls(actions)
        for _ in range(warm_steps):
            assert trial.step(H) == control.step(H) == accepted
        before, stats = trial.snapshot(), trial.stats()
        changed = {
            101: program.inputs(np.zeros(3), feedforward=np.array([-1.0, 2.0, -3.0])),
            23: (0.0, 1.0, 0.0),
            7: _action(affine, asset, data, ff=[-3]),
        }
        bad = changed.copy()  # ordinary/scalar inputs precede the invalid affine input.
        bad[7] = replace(changed[7], graph_token=("foreign graph",))
        with pytest.raises(ValueError, match="graph"):
            trial.set_controls(bad)
        after_validation = trial.stats()
        assert trial.snapshot() == before
        assert trial.stats()["current_buffer"] == stats["current_buffer"]
        assert trial.stats()["state_epoch"] == stats["state_epoch"]
        assert after_validation["launch_attempts"] == stats["launch_attempts"]
        for batch in (trial, control):
            batch.set_controls(changed)
        changed_snapshot = trial.snapshot()
        original_upload, calls = trial._upload, []

        def partial_bank_upload(buffer, values, category):
            original_upload(buffer, values, category)
            calls.append(category)
            # Ordinary controls have uploaded; fail at the subsequent affine command bank.
            if len(calls) == 3:
                raise RuntimeError("mixed ordinary/affine upload failure")

        with monkeypatch.context() as patch:
            patch.setattr(trial, "_upload", partial_bank_upload)
            with pytest.raises(RuntimeError, match="mixed ordinary/affine"):
                trial.step(H)
        assert len(calls) == 3
        assert trial.snapshot() == changed_snapshot
        assert trial.stats()["current_buffer"] == stats["current_buffer"]
        assert trial.stats()["state_epoch"] == stats["state_epoch"]
        assert trial.tree_diagnostics() == {101: None, 7: None}
        assert trial.step(H) == control.step(H) == accepted
        assert trial.snapshot() == control.snapshot()
        successful = trial.snapshot()
        after_stats = trial.stats()
        bad_restore = dict(before)
        point = bad_restore[7]
        bad_restore[7] = replace(
            point, full_position=(point.full_position[0] + 1 / 4, *point.full_position[1:])
        )
        with pytest.raises(ValueError, match="affine|full|graph|state|constraint"):
            trial.restore(bad_restore)
        assert trial.snapshot() == successful
        assert trial.stats()["current_buffer"] == after_stats["current_buffer"]
        assert trial.stats()["state_epoch"] == after_stats["state_epoch"]
        assert trial.stats()["restore_uploaded_bytes"] == after_stats["restore_uploaded_bytes"]
        for batch in (trial, control):
            batch.restore(before)
        assert trial.snapshot() == control.snapshot() == before
        trace = []
        for _ in range(2):
            assert trial.step(H) == control.step(H) == accepted
            got, expected = trial.snapshot(), control.snapshot()
            assert got == expected
            assert _json(trial.tree_kinematics()) == _json(control.tree_kinematics())
            assert tuple(got) == (101, 7, 23)
            assert tuple(trial.tree_kinematics()) == (101, 7)
            trace.append(dict(actual=got, control=expected, diagnostics=trial.tree_diagnostics()))
        _save(
            tmp_path,
            "mixed-tree-transaction",
            dict(
                input=data,
                ordinary_initial=ordinary.state,
                commands=actions,
                changed=changed,
                before=before,
                partial_upload_calls=calls,
                accepted_after_retry=successful,
                invalid_restore=bad_restore,
                trace=trace,
                stats=trial.stats(),
            ),
        )
