"""BALL-C C5/C6 exact accepted-state lifecycle, tests first."""
from contextlib import ExitStack
import pytest
from test_mujoco_tyx_ball_contacts import MODES, case

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ("c3-early", "c3-late"))
def test_ball_contact_checkpoint_clone_partial_reset(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "lifecycle.jsonl.gz")
    try:
        case.lifecycle(name, mode, journal, (7, 41, 103, 509), tmp_path)
    finally:
        journal.close()

FAULTS = (
    ("c1-loaded", "status"), ("c1-loaded", "readback"), ("c1-loaded", "restore"),
    ("c3-early", "status"), ("c3-early", "readback"), ("c3-early", "restore"),
    ("c3-early", "activation"),
)

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,fault", FAULTS)
def test_ball_contact_late_failure_exact_retention(name, fault, mode, monkeypatch, tmp_path):
    from phymeo.training.resident_stepper import PhysicsStepFailure
    import phymeo.backends.mujoco_checkpoint as checkpoint_module
    mj, model, compiled, starts, runtime = case.fixture(name, mode, (7, 41))
    journal = case.Journal(tmp_path / "failure.jsonl.gz")
    try:
        refs = case.references(mj, model, runtime.integration_snapshot())
        packets = {k: case.control(model, 0, i) for i, k in enumerate(runtime.world_keys)}
        case.advance(runtime, mj, model, packets, refs, journal, "warm")
        runtime.set_controls({k: case.control(model, 8, i) for i, k in enumerate(runtime.world_keys)})
        before = runtime.integration_snapshot()
        capsule = runtime.checkpoint()
        raw = capsule.to_bytes()
        revision = runtime.checkpoint_revision
        journal.emit("pre-fault", name=name, fault=fault, state=before, checkpoint=raw, revision=revision)
        with runtime.clone() as clone:
            with monkeypatch.context() as patch:
                if fault == "restore":
                    original = checkpoint_module._verify_bank
                    def refuse(*args, **kwargs):
                        original(*args, **kwargs)
                        raise RuntimeError("BALL-C real candidate verified then rejected")
                    patch.setattr(checkpoint_module, "_verify_bank", refuse)
                    with pytest.raises(RuntimeError, match="real candidate verified"):
                        runtime.restore(capsule, expected_revision=revision)
                else:
                    original = runtime._device.download
                    def refuse():
                        result = original()
                        journal.emit("actual-failed-download", raw=result)
                        if fault == "readback":
                            raise RuntimeError("BALL-C complete real download refused")
                        if fault == "status":
                            result["status"][-1] = 2
                        else:
                            result["input_act"][-1, 0] += .25
                        return result
                    patch.setattr(runtime._device, "download", refuse)
                    with pytest.raises(PhysicsStepFailure) as caught:
                        runtime.step(case.H)
                    assert caught.value.acceptance_known and not caught.value.physics_accepted
                    assert caught.value.accepted_worlds == dict.fromkeys(runtime.world_keys, False)
                case.exact(runtime.integration_snapshot(), before, "whole publication retained")
                assert runtime.checkpoint().to_bytes() == raw
                assert runtime.checkpoint_revision == revision
            held = {k: s.held_control for k, s in before.items()}
            left = case.advance(runtime, mj, model, held, case.references(mj, model, before), journal, "retry")
            right = case.advance(clone, mj, model, held, case.references(mj, model, before), journal, "uninterrupted")
            case.exact(left, right, "retry equals real independent clone")
    finally:
        runtime.close()
        journal.close()

@pytest.mark.parametrize("mode", MODES)
def test_ball_contact_wrong_checkpoint_binding(mode, tmp_path):
    journal = case.Journal(tmp_path / "wrong-binding.jsonl.gz")
    try:
        with ExitStack() as owners:
            mj, model, compiled, starts, runtime = case.fixture("c3-early", mode, (7, 41))
            owners.callback(runtime.close)
            _, _, _, _, other = case.fixture("c3-late", mode, (7, 41))
            owners.callback(other.close)
            before = runtime.checkpoint().to_bytes()
            revision = runtime.checkpoint_revision
            foreign = other.checkpoint()
            with pytest.raises(ValueError):
                runtime.restore(foreign, expected_revision=revision)
            assert runtime.checkpoint().to_bytes() == before
            assert runtime.checkpoint_revision == revision
            journal.emit("rejected", before=before, foreign=foreign.to_bytes(), retained=runtime.integration_snapshot())
    finally:
        journal.close()