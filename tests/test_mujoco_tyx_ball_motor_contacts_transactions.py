"""BALL-MC whole-batch force/motion publication and lifecycle, tests first."""

from contextlib import ExitStack

import pytest
from test_mujoco_tyx_ball_motor_contacts import MODES, case

PATHS = ("mc1-normal", "mc3-mixed")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", PATHS)
def test_ball_motor_contact_checkpoint_clone_partial_reset(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "lifecycle.jsonl.gz")
    try:
        case.lifecycle(name, mode, journal, case.KEYS, tmp_path)
    finally:
        journal.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", PATHS)
@pytest.mark.parametrize("fault", ("status", "readback", "publication", "restore"))
def test_ball_motor_contact_late_failure_retains_whole_batch(
    name, fault, mode, monkeypatch, tmp_path
):
    import phymeo.backends.mujoco_checkpoint as checkpoint_module
    import phymeo.backends.mujoco_joint_transmission as publication
    from phymeo.training.resident_stepper import PhysicsStepFailure

    mj, model, compiled, starts, runtime = case.fixture(name, mode)
    journal = case.Journal(tmp_path / "failure.jsonl.gz")
    try:
        refs = case.references(mj, model, runtime.integration_snapshot())
        packets = {k: case.control(model, 0, i) for i, k in enumerate(runtime.world_keys)}
        case.advance(runtime, mj, model, packets, refs, journal, "pre-fault-step")
        runtime.set_controls(
            {k: case.control(model, 8, i) for i, k in enumerate(runtime.world_keys)}
        )
        before = runtime.integration_snapshot()
        capsule = runtime.checkpoint()
        raw = capsule.to_bytes()
        revision = runtime.checkpoint_revision
        journal.emit(
            "pre-fault",
            name=name,
            fault=fault,
            state=before,
            checkpoint=raw,
            revision=revision,
            counters=dict(runtime.counters),
        )
        with runtime.clone() as clone:
            with monkeypatch.context() as patch:
                if fault == "restore":
                    original = checkpoint_module._verify_bank

                    def refuse(*args, **kwargs):
                        original(*args, **kwargs)
                        raise RuntimeError("BALL-MC real candidate verified then rejected")

                    patch.setattr(checkpoint_module, "_verify_bank", refuse)
                    with pytest.raises(RuntimeError, match="real candidate verified"):
                        runtime.restore(capsule, expected_revision=revision)
                else:
                    if fault == "publication":
                        original = publication.publish_joint_transmission

                        def refuse(plan, result, index, input_control):
                            value = original(plan, result, index, input_control)
                            journal.emit(
                                "actual-publication-prefix", index=index, transmission=value
                            )
                            if index == len(runtime.world_keys) - 1:
                                raise RuntimeError("BALL-MC final-world publication refused")
                            return value

                        patch.setattr(publication, "publish_joint_transmission", refuse)
                    else:
                        original = runtime._device.download

                        def refuse():
                            result = original()
                            journal.emit("actual-failed-download", raw=result)
                            if fault == "readback":
                                raise RuntimeError("BALL-MC complete real download refused")
                            result["status"][-1] = 2
                            return result

                        patch.setattr(runtime._device, "download", refuse)
                    with pytest.raises(PhysicsStepFailure) as caught:
                        runtime.step(case.H)
                    assert caught.value.acceptance_known and not caught.value.physics_accepted
                    assert caught.value.accepted_worlds == dict.fromkeys(runtime.world_keys, False)
                case.exact(runtime.integration_snapshot(), before, "whole publication retained")
                assert runtime.checkpoint().to_bytes() == raw
                assert runtime.checkpoint_revision == revision
                journal.emit(
                    "rejected-retained",
                    state=runtime.integration_snapshot(),
                    checkpoint=runtime.checkpoint().to_bytes(),
                    revision=runtime.checkpoint_revision,
                    counters=dict(runtime.counters),
                )
            held = {k: state.held_control for k, state in before.items()}
            left = case.advance(
                runtime, mj, model, held, case.references(mj, model, before), journal, "retry"
            )
            right = case.advance(
                clone, mj, model, held, case.references(mj, model, before), journal, "uninterrupted"
            )
            case.exact(left, right, "retry equals actual uninterrupted clone")
    finally:
        runtime.close()
        journal.close()


@pytest.mark.parametrize("mode", MODES)
def test_ball_motor_contact_foreign_checkpoint_retains_source(mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "wrong-binding.jsonl.gz")
    try:
        with ExitStack() as owners:
            _, _, _, _, runtime = case.fixture("mc3-mixed", mode)
            owners.callback(runtime.close)
            _, _, _, _, other = case.fixture("mc6-disabled", mode)
            owners.callback(other.close)
            before = runtime.checkpoint().to_bytes()
            revision = runtime.checkpoint_revision
            foreign = other.checkpoint()
            with pytest.raises(ValueError):
                runtime.restore(foreign, expected_revision=revision)
            assert runtime.checkpoint().to_bytes() == before
            assert runtime.checkpoint_revision == revision
            journal.emit(
                "rejected",
                before=before,
                foreign=foreign.to_bytes(),
                retained=runtime.integration_snapshot(),
            )
    finally:
        journal.close()
