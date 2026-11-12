"""Whole heterogeneous publication, boundary restore and late-failure tests."""

from dataclasses import replace

import numpy as np
import pytest
from test_mujoco_tyx_heterogeneous import (
    H, MODES, ORDER, MuJoCoControl, MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec, MuJoCoTyxWorldSpec, ball, declarations, packets, scalar,
)

from phymeo.backends.mujoco_heterogeneous_checkpoint import MuJoCoTyxHeterogeneousCheckpoint
from phymeo.training.resident_stepper import PhysicsStepFailure


def make(mode):
    mj, xml, models, specs, worlds = declarations()
    return models, MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode)


def step(batch, models, tick=0):
    batch.set_controls(packets(models, tick))
    accepted = batch.step(H)
    assert dict(accepted) == dict.fromkeys(ORDER, True)
    return batch.integration_snapshot()


@pytest.mark.parametrize("mode", MODES)
def test_checkpoint_bytes_restore_and_partial_original_reset(mode, tmp_path):
    models, batch = make(mode)
    history = {}
    try:
        start = batch.integration_snapshot()
        step(batch, models)
        capsule = batch.checkpoint()
        (tmp_path / "heterogeneous.checkpoint").write_bytes(capsule.to_bytes())
        restored = MuJoCoTyxHeterogeneousCheckpoint.from_bytes(capsule.to_bytes())
        successor = step(batch, models, 1)
        history["successor"] = successor
        revision = batch.checkpoint_revision
        batch.restore(restored, expected_revision=revision)
        assert batch.checkpoint_revision == revision + 1
        repeated = step(batch, models, 1)
        history["repeated"] = repeated
        ball.exact(successor, repeated, "checkpoint exact actual successor")
        before = batch.integration_snapshot()
        old_owners = tuple(batch._buckets)
        result = batch.reset((41, 7))
        after = batch.integration_snapshot()
        history.update(before_reset=before, after_reset=after, reset=result)
        assert batch._buckets[2] is old_owners[2]
        for key in ORDER:
            if key in (41, 7):
                assert after[key].tick == 0 and after[key].last_step is None
                for name in ("q", "v", "act", "time"):
                    ball.exact(getattr(after[key], name), getattr(start[key], name), name)
            else:
                ball.exact(after[key], before[key], "unselected full accepted record")
        history["after_reset_successor"] = step(batch, models, 2)
        assert history["after_reset_successor"][41].tick == 1
        assert history["after_reset_successor"][11].tick == before[11].tick + 1
        assert batch.counters["boundary_clones"] == 5
    finally:
        batch.close()
        scalar.save(tmp_path / "heterogeneous-lifecycle.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ("last-status", "last-download", "late-aggregate"))
def test_last_bucket_failure_retains_every_world_and_retry(mode, fault, monkeypatch, tmp_path):
    models, batch = make(mode)
    _, reference = make(mode)
    history = {}
    try:
        command = packets(models, 0)
        batch.set_controls(command)
        reference.set_controls(command)
        before = batch.checkpoint().to_bytes()
        old_state, old_receipt = batch.integration_snapshot(), batch.last_step
        target = batch._buckets[-1]
        if fault == "late-aggregate":
            def fail(operation, pending):
                if operation == "step":
                    raise RuntimeError("last aggregate preparation fault")
            monkeypatch.setattr(batch, "_before_commit", fail)
        else:
            original = target._device.download
            def broken():
                actual = original()
                if fault == "last-download":
                    raise RuntimeError("last real download complete then fail")
                actual["status"][-1] = 2
                return actual
            monkeypatch.setattr(target._device, "download", broken)
        with pytest.raises(PhysicsStepFailure) as caught:
            batch.step(H)
        assert dict(caught.value.accepted_worlds) == dict.fromkeys(ORDER, False)
        assert batch.checkpoint().to_bytes() == before
        assert batch.last_step is old_receipt
        ball.exact(batch.integration_snapshot(), old_state, "all buckets retain")
        history.update(fault=fault, before=old_state, failure=str(caught.value), counters=batch.counters)
        assert batch.counters["completed_steps"] == 0
        assert batch.counters["accepted_world_steps"] == 0
        assert batch.counters["bucket_physics_prepares"] == 3
        monkeypatch.undo()
        assert batch.step(H) == reference.step(H)
        history["retry"] = batch.integration_snapshot()
        ball.exact(history["retry"], reference.integration_snapshot(), "real retry")
    finally:
        batch.close()
        reference.close()
        scalar.save(tmp_path / "heterogeneous-late-step.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
def test_invalid_last_control_has_no_partial_publication(mode, tmp_path):
    models, batch = make(mode)
    try:
        good = packets(models, 0)
        batch.set_controls(good)
        before = batch.checkpoint().to_bytes()
        launches = batch.counters["completed_kernels"]
        bad = dict(good)
        bad[59] = MuJoCoControl(np.zeros(models["free"].nu + 1))
        with pytest.raises((ValueError, TypeError)):
            batch.set_controls(bad)
        assert batch.checkpoint().to_bytes() == before
        assert batch.counters["completed_kernels"] == launches
        batch.step(H)
        scalar.save(tmp_path / "heterogeneous-control.json.gz", dict(output=batch.integration_snapshot(), receipt=batch.last_step))
    finally:
        batch.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("operation", ("reset", "restore"))
def test_boundary_last_prepublication_failure_retains_and_retries(mode, operation, monkeypatch, tmp_path):
    models, batch = make(mode)
    history = {}
    try:
        step(batch, models)
        saved = batch.checkpoint()
        step(batch, models, 1)
        before = batch.checkpoint().to_bytes()
        old = tuple(batch._buckets)
        def fail(name, pending):
            if name == operation:
                raise RuntimeError("all replacement buckets prepared before refusal")
        monkeypatch.setattr(batch, "_before_commit", fail)
        with pytest.raises(RuntimeError, match="replacement"):
            if operation == "reset":
                batch.reset((41, 7, 305))
            else:
                batch.restore(saved, expected_revision=batch.checkpoint_revision)
        assert tuple(batch._buckets) == old
        assert batch.checkpoint().to_bytes() == before
        history["retained"] = batch.integration_snapshot()
        monkeypatch.undo()
        if operation == "reset":
            batch.reset((41, 7, 305))
        else:
            batch.restore(saved, expected_revision=batch.checkpoint_revision)
        history["retry"] = step(batch, models, 2)
    finally:
        batch.close()
        scalar.save(tmp_path / "heterogeneous-boundary-failure.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
def test_binding_and_stale_cas_rejected_before_replacement(mode, monkeypatch, tmp_path):
    models, batch = make(mode)
    mj, xml, native, specs, worlds = declarations(tuple(reversed(ORDER)))
    other = MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode)
    try:
        saved = batch.checkpoint()
        step(batch, models)
        before = batch.checkpoint().to_bytes()
        calls = []
        def forbidden(*args, **kwargs):
            calls.append(True)
            raise AssertionError("clone called before binding validation")
        monkeypatch.setattr(batch, "_clone_bucket", forbidden)
        with pytest.raises(ValueError, match="revision"):
            batch.restore(saved, expected_revision=batch.checkpoint_revision - 1)
        with pytest.raises(ValueError, match="binding"):
            batch.restore(other.checkpoint(), expected_revision=batch.checkpoint_revision)
        assert not calls and batch.checkpoint().to_bytes() == before
        scalar.save(tmp_path / "heterogeneous-binding.json.gz", dict(binding=saved.binding, retained=batch.integration_snapshot()))
    finally:
        batch.close()
        other.close()


@pytest.mark.parametrize("mode", MODES)
def test_reentrant_writer_rejected_without_dropping_original_draft(mode, monkeypatch):
    models, batch = make(mode)
    try:
        batch.set_controls(packets(models, 0))
        calls = []
        def nested(operation, pending):
            if operation == "step":
                for function in (lambda: batch.step(H), lambda: batch.reset(), lambda: batch.set_controls(packets(models, 1)), batch.close):
                    with pytest.raises(RuntimeError, match="transaction"):
                        function()
                    calls.append(True)
        monkeypatch.setattr(batch, "_before_commit", nested)
        batch.step(H)
        assert len(calls) == 4 and batch.counters["completed_steps"] == 1
    finally:
        monkeypatch.undo()
        batch.close()


@pytest.mark.parametrize("mode", MODES)
def test_postpublication_cleanup_failure_keeps_owner_and_accepted_reset(mode, monkeypatch):
    models, batch = make(mode)
    try:
        step(batch, models)
        target = batch._buckets[0]
        original = target.close
        calls = []
        def fail_once():
            calls.append(True)
            if len(calls) == 1:
                raise RuntimeError("owner close refused before actual free")
            return original()
        monkeypatch.setattr(target, "close", fail_once)
        receipt = batch.reset((41,))
        assert batch.integration_snapshot()[41].tick == 0
        assert receipt.cleanup_errors and batch.cleanup_errors
        assert target in batch._retired
        batch.close()
        assert len(calls) == 2
    finally:
        batch.close()
