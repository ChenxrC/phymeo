"""R1 whole-batch force/motion publication and lifecycle, tests first."""

import base64
import hashlib
import json
from contextlib import ExitStack

import numpy as np
import pytest
from test_mujoco_tyx_ball_activation import MODES, case

PATHS = ("r3-normal", "r5-mixed")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ("r1-late", *PATHS))
def test_ball_activation_checkpoint_clone_partial_reset(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "lifecycle.jsonl.gz")
    try:
        case.lifecycle(name, mode, journal, case.KEYS, tmp_path)
    finally:
        journal.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", PATHS)
@pytest.mark.parametrize("fault", ("status", "readback", "actuation", "transmission", "restore"))
def test_ball_activation_late_failure_retains_whole_batch(name, fault, mode, monkeypatch, tmp_path):
    import phymeo.backends.mujoco_actuation as activation_publication
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
                        raise RuntimeError("R1 real candidate verified then rejected")

                    patch.setattr(checkpoint_module, "_verify_bank", refuse)
                    with pytest.raises(RuntimeError, match="real candidate verified"):
                        runtime.restore(capsule, expected_revision=revision)
                else:
                    if fault == "actuation":
                        original = activation_publication.publish_actuation

                        def refuse(plan, result, index):
                            value = original(plan, result, index)
                            journal.emit("actual-actuation-prefix", index=index, actuation=value)
                            if index == len(runtime.world_keys) - 1:
                                raise RuntimeError("R1 final-world activation refused")
                            return value

                        patch.setattr(activation_publication, "publish_actuation", refuse)
                    elif fault == "transmission":
                        original = publication.publish_joint_transmission

                        def refuse(plan, result, index, input_control):
                            value = original(plan, result, index, input_control)
                            journal.emit(
                                "actual-publication-prefix", index=index, transmission=value
                            )
                            if index == len(runtime.world_keys) - 1:
                                raise RuntimeError("R1 final-world publication refused")
                            return value

                        patch.setattr(publication, "publish_joint_transmission", refuse)
                    else:
                        original = runtime._device.download

                        def refuse():
                            result = original()
                            journal.emit("actual-failed-download", raw=result)
                            if fault == "readback":
                                raise RuntimeError("R1 complete real download refused")
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
def test_ball_activation_foreign_checkpoint_retains_source(mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "wrong-binding.jsonl.gz")
    try:
        with ExitStack() as owners:
            _, _, _, _, runtime = case.fixture("r5-mixed", mode)
            owners.callback(runtime.close)
            _, _, _, _, other = case.fixture("r8-disabled", mode)
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


def _tamper_one_record(raw, record_name):
    header, _, encoded = raw.split(b"\n", 2)
    value = json.loads(encoded)
    changed = []

    def visit(node):
        if isinstance(node, dict):
            if node.get("$record") == record_name and not changed:
                array = node["fields"]["scalar_force"]
                data = np.frombuffer(base64.b64decode(array["data"]), array["$array"]).copy()
                data[0] += np.float32(0.125)
                array["data"] = base64.b64encode(data.tobytes()).decode("ascii")
                changed.append(record_name)
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    assert changed == [record_name]
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return header + b"\n" + hashlib.sha256(body).hexdigest().encode() + b"\n" + body


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", PATHS)
@pytest.mark.parametrize("record", ("MuJoCoTyxActuation", "MuJoCoTyxJointTransmission"))
def test_ball_activation_checkpoint_single_record_tamper_retains_successor(
    name, record, mode, tmp_path
):
    from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint

    mj, model, _, _, runtime = case.fixture(name, mode)
    journal = case.Journal(tmp_path / "tampered-combined-record.jsonl.gz")
    try:
        state = runtime.integration_snapshot()
        packets = {k: case.control(model, 0, i) for i, k in enumerate(runtime.world_keys)}
        case.advance(
            runtime, mj, model, packets, case.references(mj, model, state), journal, "entry"
        )
        before = runtime.integration_snapshot()
        capsule = runtime.checkpoint()
        raw, revision = capsule.to_bytes(), runtime.checkpoint_revision
        forged = _tamper_one_record(raw, record)
        journal.emit("tamper-attempt", record=record, original=raw, forged=forged, state=before)
        with runtime.clone() as clone:
            with pytest.raises(ValueError, match="same-attempt"):
                bad = MuJoCoTyxCheckpoint.from_bytes(forged)
                runtime.restore(bad, expected_revision=revision)
            case.exact(runtime.integration_snapshot(), before, "tamper whole publication retained")
            assert runtime.checkpoint().to_bytes() == raw
            assert runtime.checkpoint_revision == revision
            held = {k: s.held_control for k, s in before.items()}
            actual = case.advance(
                runtime, mj, model, held, case.references(mj, model, before), journal, "retained"
            )
            control = case.advance(
                clone, mj, model, held, case.references(mj, model, before), journal, "uninterrupted"
            )
            case.exact(actual, control, "tamper rejected before changing successor")
    finally:
        runtime.close()
        journal.close()
