"""Parent-owned mixed-bank transaction tests; C4 physics oracles stay separate."""
import numpy as np
import pytest

from phymeo.backends import tyx_world_batch as resident
from test_resident_tree_step import _action, _program, _rr_model, _save, _tree

H = 1 / 1024


def _worlds():
    program, initial = _program(_rr_model(), delay=1)
    worlds = [(7, _tree(program, initial)),
              (23, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0)))]
    return program, worlds


def _controls(program, alternate):
    return {7: _action(program, ff=(.25, -.125) if alternate else (0, .125),
                       effort=(1, 1 / 6) if alternate else (0, 1 / 4)),
            23: (2, 0, 0) if alternate else (0, 1, 0)}


@pytest.mark.tyx
@pytest.mark.parametrize("warm_steps", (0, 1))
def test_mixed_control_upload_failure_retransmits_both_banks(
        warm_steps, tyx_mode, tmp_path, monkeypatch):
    program, worlds = _worlds()
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path/"failed") as batch, \
            resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path/"fresh") as fresh:
        first, next_controls = _controls(program, False), _controls(program, True)
        batch.set_controls(first)
        fresh.set_controls(first)
        for _ in range(warm_steps):
            assert batch.step(H) == fresh.step(H) == {7: True, 23: True}
        batch.set_controls(next_controls)
        fresh.set_controls(next_controls)
        before, epoch, direction = batch.snapshot(), batch.stats()["state_epoch"], batch.stats()["current_buffer"]
        seen, original = [], batch._upload

        def upload_then_fail(buffer, data, category):
            original(buffer, data, category)
            seen.append(category)
            if category == "tree_control":
                raise RuntimeError("injected after tree command upload")

        with monkeypatch.context() as patch:
            patch.setattr(batch, "_upload", upload_then_fail)
            with pytest.raises(RuntimeError, match="after tree command upload"):
                batch.step(H)
        assert seen == ["control", "tree_control"]
        assert batch.snapshot() == before
        assert batch.stats()["state_epoch"] == epoch
        assert batch.stats()["current_buffer"] == direction
        assert batch.tree_diagnostics() == {7: None}
        retry_uploads = []

        def record_upload(buffer, data, category):
            retry_uploads.append(category)
            return original(buffer, data, category)

        with monkeypatch.context() as patch:
            patch.setattr(batch, "_upload", record_upload)
            assert batch.step(H) == fresh.step(H) == {7: True, 23: True}
        assert retry_uploads == ["control", "tree_control"]
        assert batch.snapshot() == fresh.snapshot()
        _save(tmp_path, f"mixed-upload-{warm_steps}-{tyx_mode}", dict(
            input=next_controls, before=before, after=batch.snapshot(), attempted_uploads=seen,
            retry_uploads=retry_uploads, stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("warm_steps", (0, 1))
def test_mixed_restore_late_tree_check_never_partially_publishes(
        warm_steps, tyx_mode, tmp_path, monkeypatch):
    program, worlds = _worlds()
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls(_controls(program, False))
        for _ in range(warm_steps):
            assert batch.step(H) == {7: True, 23: True}
        restore_to = batch.snapshot()
        batch.set_controls(_controls(program, True))
        assert batch.step(H) == {7: True, 23: True}
        before, fk = batch.snapshot(), batch.tree_kinematics()[7]
        epoch, direction = batch.stats()["state_epoch"], batch.stats()["current_buffer"]
        original = batch._trees.check_restore

        def check_then_fail():
            original()
            raise RuntimeError("injected after both restore banks")

        with monkeypatch.context() as patch:
            patch.setattr(batch._trees, "check_restore", check_then_fail)
            with pytest.raises(RuntimeError, match="both restore banks"):
                batch.restore(restore_to)
        assert batch.snapshot() == before
        assert batch.stats()["state_epoch"] == epoch
        assert batch.stats()["current_buffer"] == direction
        np.testing.assert_array_equal(batch.tree_kinematics()[7].link_position_m, fk.link_position_m)
        assert batch.tree_diagnostics() == {7: None}
        batch.restore(restore_to)
        assert batch.snapshot() == restore_to
        assert batch.stats()["state_epoch"] == epoch + 1
        assert batch.tree_diagnostics() == {7: None}
        batch.set_controls(_controls(program, True))
        assert batch.step(H) == {7: True, 23: True}
        assert batch.snapshot() == before
        _save(tmp_path, f"mixed-restore-{warm_steps}-{tyx_mode}", dict(
            restore_to=restore_to, before=before, after=batch.snapshot(), stats=batch.stats()))
