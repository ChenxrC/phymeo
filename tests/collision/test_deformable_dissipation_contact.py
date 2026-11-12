"""D6-A real D3 candidate/transaction qualification; authored, not executed here."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

pytestmark = pytest.mark.tyx


def consumer():
    path = Path(__file__).parents[1] / "installed" / "check_deformable_dissipation.py"
    spec = importlib.util.spec_from_file_location("d6_installed_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure(ph)
    return module


@pytest.mark.parametrize("case", ("E", "C"))
def test_dissipation_contact_real_ordered_phase_and_lifecycle(case, tyx_mode, tmp_path):
    c = consumer()
    journal = c.Journal(tmp_path / (case + "-complete.jsonl.gz"))
    progress = c.new_progress()
    try:
        items = tuple(c.contact_fixture(case, world=w) for w in (7, 31))
        c.run_contact(items, tyx_mode, journal, progress, lifecycle=True)
        assert progress["steps_attempted"] == progress["steps_returned"] == progress["steps_checked"] == 5
        assert progress["worldsteps"] == 10
        assert progress["restores"] == progress["resets"] == progress["stale_rejections"] == 1
    finally:
        journal.close()


@pytest.mark.parametrize("case", ("E", "C"))
def test_dissipation_contact_actual_late_candidate_failure_and_retry(case, tyx_mode, tmp_path, monkeypatch):
    c = consumer()
    items = tuple(c.contact_fixture(case, world=w) for w in (7, 31))
    journal = c.Journal(tmp_path / (case + "-late.jsonl.gz"))
    progress = c.new_progress()
    owner = None
    try:
        owner = c.create(items, tyx_mode, contact_capacity=4)
        owner.set_controls({i["participant"].key: i["command"] for i in items})
        references = {i["world"]: c.reference_initial(i) for i in items}
        _, references = c.advance_contact(owner, items, references, journal, progress, "initial")
        checkpoint, before = owner.checkpoint(), owner.snapshot()
        revision, publication = owner.revision, owner.last_step
        original = owner._prepare_response_publication
        completed = []

        def fail_after_real_candidate(*args, **kwargs):
            result = original(*args, **kwargs)
            completed.append(result)
            journal.emit("actual-candidate-before-fault", candidate=result)
            raise RuntimeError("D6 actual complete candidate fault")

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_prepare_response_publication", fail_after_real_candidate)
            with pytest.raises(RuntimeError, match="D6 actual complete candidate fault"):
                c.advance_contact(owner, items, references, journal, progress, "late-fault")
        assert len(completed) == 1
        assert c.encode(owner.checkpoint()) == c.encode(checkpoint)
        assert owner.revision == revision and owner.last_step is publication
        c.state_equal(before, owner.snapshot(), items)
        for item in items:
            phase = completed[0].node_receipt.worlds[item["world"]].dissipation
            assert phase.trace.shape == (5, 32)
            assert np.any(phase.node_velocity_impulse != 0)
        retried, _ = c.advance_contact(owner, items, c.copy_references(references), journal, progress, "retry")
        owner.restore(checkpoint, expected_revision=owner.revision)
        replay, _ = c.advance_contact(owner, items, c.copy_references(references), journal, progress, "restored-successor")
        c.state_equal(retried, replay, items)
        journal.emit("retained-and-replayed", before=before, checkpoint=checkpoint, retry=retried, replay=replay, stats=owner.stats())
    finally:
        try:
            if owner is not None:
                owner.close()
        finally:
            journal.close()
