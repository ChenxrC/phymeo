"""R3-N whole-batch normal/CSR force publication and actual successor recovery."""
import sys
from contextlib import ExitStack

import numpy as np
import pytest
from test_mujoco_tyx_free_ball_normal import CASES, MODES, case


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CASES)
def test_free_ball_normal_checkpoint_clone_partial_reset(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "lifecycle.jsonl.gz")
    try:
        case.lifecycle(name, mode, journal, case.KEYS, tmp_path)
    finally:
        journal.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("fault", ("status", "readback", "publication", "restore"))
def test_free_ball_normal_late_failure_retains_and_retries(
    name, fault, mode, monkeypatch, tmp_path
):
    import phymeo.backends.mujoco_checkpoint as checkpoint_module
    import phymeo.backends.mujoco_joint_transmission as publication
    from phymeo.training.resident_stepper import PhysicsStepFailure

    mj, model, compiled, starts, runtime = case.fixture(name, mode)
    journal = case.Journal(tmp_path / "late-failure.jsonl.gz")
    try:
        refs = case.references(mj, model, runtime.integration_snapshot())
        packets = {k: case.control(model, 0, i) for i, k in enumerate(runtime.world_keys)}
        case.advance(runtime, mj, model, packets, refs, journal, "pre-fault")
        runtime.set_controls({k: case.control(model, 8, i) for i, k in enumerate(case.KEYS)})
        before = runtime.integration_snapshot()
        capsule = runtime.checkpoint()
        raw, revision = capsule.to_bytes(), runtime.checkpoint_revision
        journal.emit(
            "fault-attempt", name=name, fault=fault, source=case.source(name),
            starts=starts, state=before, capsule=raw, revision=revision,
            counters=dict(runtime.counters),
        )
        with runtime.clone() as clone:
            with monkeypatch.context() as patch:
                if fault == "restore":
                    original = checkpoint_module._verify_bank

                    def refuse(*args, **kwargs):
                        original(*args, **kwargs)
                        raise RuntimeError("R3-N verified candidate restore refused")

                    patch.setattr(checkpoint_module, "_verify_bank", refuse)
                    with pytest.raises(RuntimeError, match="verified candidate"):
                        runtime.restore(capsule, expected_revision=revision)
                else:
                    if fault == "publication":
                        original = publication.publish_joint_transmission

                        def refuse(plan, result, index, input_control):
                            value = original(plan, result, index, input_control)
                            journal.emit("actual-publication-prefix", index=index, transmission=value)
                            if index == len(case.KEYS) - 1:
                                raise RuntimeError("R3-N final world publication refused")
                            return value

                        patch.setattr(publication, "publish_joint_transmission", refuse)
                    else:
                        original = runtime._device.download

                        def refuse():
                            result = original()
                            journal.emit("actual-failed-download", raw=result)
                            if fault == "readback":
                                raise RuntimeError("R3-N complete actual download refused")
                            result["status"][-1] = 2
                            return result

                        patch.setattr(runtime._device, "download", refuse)
                    with pytest.raises(PhysicsStepFailure) as caught:
                        runtime.step(case.H)
                    assert caught.value.acceptance_known and not caught.value.physics_accepted
                    assert caught.value.accepted_worlds == dict.fromkeys(case.KEYS, False)
                case.exact(runtime.integration_snapshot(), before, "complete retained publication")
                assert runtime.checkpoint().to_bytes() == raw
                assert runtime.checkpoint_revision == revision
                for key in case.KEYS:
                    assert runtime.integration_snapshot()[key].last_step is before[key].last_step
                journal.emit(
                    "rejected-retained", state=runtime.integration_snapshot(),
                    capsule=runtime.checkpoint().to_bytes(), revision=runtime.checkpoint_revision,
                    counters=dict(runtime.counters),
                )
            held = {k: s.held_control for k, s in before.items()}
            actual = case.advance(
                runtime, mj, model, held, case.references(mj, model, before), journal, "retry"
            )
            other = case.advance(
                clone, mj, model, held, case.references(mj, model, before), journal, "uninterrupted"
            )
            case.exact(actual, other, "retry full force/motion successor")
    finally:
        primary = sys.exc_info()[1]
        errors = []
        for close in (runtime.close, journal.close):
            try:
                close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("R3-N failure evidence cleanup", ([primary] if primary else []) + errors)


@pytest.mark.parametrize("mode", MODES)
def test_free_ball_normal_bad_control_and_foreign_checkpoint_are_atomic(mode, tmp_path):
    from phymeo.backends.mujoco_runtime import MuJoCoControl

    journal = case.Journal(tmp_path / "binding-control.jsonl.gz")
    try:
        with ExitStack() as owners:
            mj, model, _, _, runtime = case.fixture("feet", mode)
            owners.callback(runtime.close)
            _, _, _, _, other = case.fixture("siblings", mode)
            owners.callback(other.close)
            packets = {k: case.control(model, 0, i) for i, k in enumerate(case.KEYS)}
            case.advance(runtime, mj, model, packets, case.references(
                mj, model, runtime.integration_snapshot()), journal, "before")
            before, raw = runtime.integration_snapshot(), runtime.checkpoint().to_bytes()
            revision = runtime.checkpoint_revision
            with pytest.raises(ValueError):
                runtime.set_controls({
                    case.KEYS[0]: case.control(model, 8, 0),
                    case.KEYS[-1]: MuJoCoControl(np.zeros(model.nu + 1)),
                })
            with pytest.raises(ValueError):
                runtime.restore(other.checkpoint(), expected_revision=revision)
            case.exact(runtime.integration_snapshot(), before, "bad binding/control retain all")
            assert runtime.checkpoint().to_bytes() == raw
            assert runtime.checkpoint_revision == revision
            journal.emit("rejected", before=before, raw=raw, foreign=other.checkpoint().to_bytes())
            case.advance(runtime, mj, model, packets, case.references(mj, model, before), journal, "successor")
    finally:
        journal.close()
