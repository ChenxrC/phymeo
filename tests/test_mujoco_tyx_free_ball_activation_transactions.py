"""R3-A whole-batch FREE/BALL activation publication; authoring-only matrix.

Three unchanged source routes x two modes: 30 late faults, six lifecycle nodes,
and six checkpoint nodes (seven independent checksum-valid corruptions each).
The case module owns the independent native references and frozen source inputs.
Unknown opaque release remains covered by the original ownership tests; this
suite does not invent a safe retry after an unknown release.
"""

import base64
import hashlib
import json
import sys

import numpy as np
import pytest
from test_mujoco_tyx_free_ball_activation import MODES, case

ROUTES = ("no-contact-late", "feet-normal", "siblings-pyramidal")
FAULTS = ("status", "readback", "actuation", "transmission", "restore")
CORRUPTIONS = (
    "actuation-force",
    "transmission-force",
    "fixed-actuation-schema",
    "fixed-transmission-schema",
    "fixed-schema-pair",
    "activation-address",
    "activation-size",
)
ACTUATION_SCHEMA = "native-free-ball-joint-filterexact-actuation-v1"
TRANSMISSION_SCHEMA = "native-free-ball-joint-motor-transmission-v1"


def _close(runtime, journal):
    primary = sys.exc_info()[1]
    errors = []
    operations = (runtime.close, journal.close) if runtime is not None else (journal.close,)
    for operation in operations:
        try:
            operation()
        except BaseException as error:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup(
            "R3-A transaction evidence cleanup", ([primary] if primary else []) + errors
        )


def _assert_free_pair(states):
    for state in states.values():
        row = state.last_step
        assert row is not None
        assert row.actuation.schema == ACTUATION_SCHEMA
        assert row.transmission.schema == TRANSMISSION_SCHEMA
        for field in ("clamped_control", "raw_force", "scalar_force", "generalized_force"):
            left, right = getattr(row.actuation, field), getattr(row.transmission, field)
            assert left.dtype == right.dtype and left.shape == right.shape
            assert left.tobytes() == right.tobytes()
        assert row.transmission.input_control.tobytes() == row.control.ctrl.tobytes()
        np.testing.assert_array_equal(
            row.actuation.generalized_force.astype(np.float64), row.qfrc_actuator
        )


def _retained(runtime, before, raw, revision, journal, label):
    actual = runtime.integration_snapshot()
    accepted_raw = runtime.checkpoint().to_bytes()
    journal.emit(
        "rejected-retained",
        label=label,
        state=actual,
        checkpoint=accepted_raw,
        revision=runtime.checkpoint_revision,
        counters=dict(runtime.counters),
    )
    case.exact(actual, before, label + ": complete publication")
    # checkpoint() reads every accepted q/v/compensated-clock/act device bank;
    # this is not merely a host snapshot comparison. It also retains controls,
    # full force/constraint receipts, ticks and original reset inputs.
    assert accepted_raw == raw
    assert runtime.checkpoint_revision == revision
    for key in runtime.world_keys:
        assert actual[key].last_step is before[key].last_step
    _assert_free_pair(actual)


def _retry(runtime, clone, mj, model, before, journal):
    held = {key: state.held_control for key, state in before.items()}
    actual = case.advance(
        runtime, mj, model, held, case.references(mj, model, before), journal, "retry"
    )
    uninterrupted = case.advance(
        clone, mj, model, held, case.references(mj, model, before), journal, "uninterrupted"
    )
    case.exact(actual, uninterrupted, "real retry full force/activation/motion successor")
    _assert_free_pair(actual)
    _assert_free_pair(uninterrupted)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ROUTES)
def test_free_ball_activation_checkpoint_clone_partial_reset(name, mode, tmp_path):
    pytest.importorskip("mujoco")
    journal = case.Journal(tmp_path / "lifecycle.jsonl.gz")
    try:
        # Includes an accepted checkpoint, actual A/clone-A, restore then actual B,
        # stale revision rejection, partial original reset and real successors.
        # The unselected raw clock/act/control/receipt is checked without deleting
        # any history or numeric field from the comparison.
        case.lifecycle(name, mode, journal, case.KEYS, tmp_path)
    finally:
        _close(None, journal)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ROUTES)
@pytest.mark.parametrize("fault", FAULTS)
def test_free_ball_activation_late_failure_retains_whole_batch(
    name, fault, mode, monkeypatch, tmp_path
):
    import phymeo.backends.mujoco_actuation as activation_publication
    import phymeo.backends.mujoco_checkpoint as checkpoint_module
    import phymeo.backends.mujoco_joint_transmission as transmission_publication
    from phymeo.training.resident_stepper import PhysicsStepFailure

    journal = case.Journal(tmp_path / "late-failure.jsonl.gz")
    runtime = None
    try:
        journal.emit("fixture-attempt", name=name, mode=mode, source=case.source(name))
        mj, model, _, starts, runtime = case.fixture(name, mode)
        refs = case.references(mj, model, runtime.integration_snapshot())
        packets = {key: case.control(model, 0, i) for i, key in enumerate(runtime.world_keys)}
        case.advance(runtime, mj, model, packets, refs, journal, "pre-fault")
        runtime.set_controls(
            {key: case.control(model, 8, i) for i, key in enumerate(runtime.world_keys)}
        )
        before, capsule = runtime.integration_snapshot(), runtime.checkpoint()
        raw, revision = capsule.to_bytes(), runtime.checkpoint_revision
        _assert_free_pair(before)
        journal.emit(
            "fault-attempt",
            name=name,
            fault=fault,
            starts=starts,
            state=before,
            checkpoint=raw,
            revision=revision,
            h=case.H,
            counters=dict(runtime.counters),
        )
        reached = []
        with runtime.clone() as clone:
            with monkeypatch.context() as patch:
                if fault == "restore":
                    original = checkpoint_module._verify_bank

                    def refuse(*args, **kwargs):
                        original(*args, **kwargs)
                        reached.append("verified-replacement")
                        journal.emit("actual-replacement-bank-verified", expected=args[1])
                        raise RuntimeError("R3-A real verified candidate restore refused")

                    patch.setattr(checkpoint_module, "_verify_bank", refuse)
                    with pytest.raises(RuntimeError, match="real verified candidate") as caught:
                        runtime.restore(capsule, expected_revision=revision)
                    assert reached == ["verified-replacement"]
                else:
                    if fault == "actuation":
                        original = activation_publication.publish_actuation

                        def refuse(plan, result, index):
                            value = original(plan, result, index)
                            reached.append(index)
                            journal.emit("actual-actuation-prefix", index=index, actuation=value)
                            if index == len(runtime.world_keys) - 1:
                                raise RuntimeError("R3-A final-world activation refused")
                            return value

                        patch.setattr(activation_publication, "publish_actuation", refuse)
                    elif fault == "transmission":
                        original = transmission_publication.publish_joint_transmission

                        def refuse(plan, result, index, input_control):
                            value = original(plan, result, index, input_control)
                            reached.append(index)
                            journal.emit(
                                "actual-transmission-prefix", index=index, transmission=value
                            )
                            if index == len(runtime.world_keys) - 1:
                                raise RuntimeError("R3-A final-world transmission refused")
                            return value

                        patch.setattr(transmission_publication, "publish_joint_transmission", refuse)
                    else:
                        original = runtime._device.download

                        def refuse():
                            result = original()
                            reached.append("complete-real-download")
                            journal.emit("actual-failed-download", raw=result)
                            if fault == "readback":
                                raise RuntimeError("R3-A complete real download refused")
                            result["status"][-1] = 2
                            journal.emit("injected-final-world-status", raw=result)
                            return result

                        patch.setattr(runtime._device, "download", refuse)
                    with pytest.raises(PhysicsStepFailure) as caught:
                        runtime.step(case.H)
                    assert caught.value.acceptance_known and not caught.value.physics_accepted
                    assert caught.value.accepted_worlds == dict.fromkeys(runtime.world_keys, False)
                    expected = (
                        list(range(len(runtime.world_keys)))
                        if fault in ("actuation", "transmission")
                        else ["complete-real-download"]
                    )
                    assert reached == expected
                journal.emit(
                    "fault-return", fault=fault, reached=reached, error=repr(caught.value)
                )
                _retained(runtime, before, raw, revision, journal, fault)
            _retry(runtime, clone, mj, model, before, journal)
    except BaseException as error:
        journal.emit("test-failed", name=name, fault=fault, error=repr(error))
        raise
    finally:
        _close(runtime, journal)


def _array(record):
    return np.frombuffer(
        base64.b64decode(record["data"], validate=True), dtype=record["$array"]
    ).reshape(tuple(record["shape"])).copy()


def _replace_array(record, array):
    assert record["$array"] == array.dtype.str
    record["shape"] = list(array.shape)
    record["data"] = base64.b64encode(array.tobytes()).decode("ascii")


def _forge_last_world(raw, corruption):
    header, digest, encoded = raw.split(b"\n", 2)
    assert digest == hashlib.sha256(encoded).hexdigest().encode("ascii")
    value = json.loads(encoded)
    worlds = value["state"]["worlds"]
    assert tuple(item["key"] for item in worlds) == case.KEYS
    receipt = worlds[-1]["receipt"]
    assert receipt["$record"] == "MuJoCoTyxJointActivationStepReceipt"
    fields = receipt["fields"]
    assert fields["actuation"]["$record"] == "MuJoCoTyxActuation"
    assert fields["transmission"]["$record"] == "MuJoCoTyxJointTransmission"
    act, transmission = fields["actuation"]["fields"], fields["transmission"]["fields"]
    assert act["schema"] == ACTUATION_SCHEMA
    assert transmission["schema"] == TRANSMISSION_SCHEMA
    if corruption in ("actuation-force", "transmission-force"):
        target = act if corruption == "actuation-force" else transmission
        array = _array(target["scalar_force"])
        array[0] += np.float32(0.125)
        _replace_array(target["scalar_force"], array)
    elif corruption in (
        "fixed-actuation-schema", "fixed-transmission-schema", "fixed-schema-pair"
    ):
        if corruption != "fixed-transmission-schema":
            act["schema"] = "native-joint-filterexact-actuation-v1"
        if corruption != "fixed-actuation-schema":
            transmission["schema"] = "native-joint-motor-transmission-v1"
    elif corruption == "activation-address":
        array = _array(act["ids"])
        owned = np.flatnonzero(array[:, 3] >= 0)
        assert owned.size
        array[owned[-1], 3] = -1
        _replace_array(act["ids"], array)
    elif corruption == "activation-size":
        array = _array(act["output_act"])
        assert array.size
        _replace_array(act["output_act"], array[:-1].copy())
    else:
        raise AssertionError(corruption)
    # Preserve all binding fields, earlier worlds and source revision. These are
    # deliberately incoherent payloads with a VALID outer corruption checksum,
    # not signatures, and never represent a legitimate new physical fixture.
    original = json.loads(encoded)
    assert value["binding"] == original["binding"]
    assert value["revision"] == original["revision"]
    assert worlds[:-1] == original["state"]["worlds"][:-1]
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert body != encoded
    return header + b"\n" + hashlib.sha256(body).hexdigest().encode("ascii") + b"\n" + body


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ROUTES)
def test_free_ball_activation_checkpoint_records_cannot_cross_domain(name, mode, tmp_path):
    from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint

    journal = case.Journal(tmp_path / "checkpoint-corruptions.jsonl.gz")
    runtime = None
    try:
        journal.emit("fixture-attempt", name=name, mode=mode, source=case.source(name))
        mj, model, _, _, runtime = case.fixture(name, mode)
        state = runtime.integration_snapshot()
        packets = {key: case.control(model, 0, i) for i, key in enumerate(runtime.world_keys)}
        case.advance(
            runtime, mj, model, packets, case.references(mj, model, state), journal, "before"
        )
        before, capsule = runtime.integration_snapshot(), runtime.checkpoint()
        raw, revision = capsule.to_bytes(), runtime.checkpoint_revision
        _assert_free_pair(before)
        with runtime.clone() as clone:
            for corruption in CORRUPTIONS:
                forged = _forge_last_world(raw, corruption)
                # The outer encoding must be accepted before the record/source
                # validation under test; a checksum error is not this negative.
                bad = MuJoCoTyxCheckpoint.from_bytes(forged)
                assert bad.to_bytes() == forged
                journal.emit(
                    "corrupt-restore-attempt",
                    corruption=corruption,
                    original=raw,
                    forged=forged,
                    revision=revision,
                    state=before,
                )
                with pytest.raises(ValueError) as caught:
                    runtime.restore(bad, expected_revision=revision)
                journal.emit("corrupt-restore-return", corruption=corruption, error=str(caught.value))
                _retained(runtime, before, raw, revision, journal, corruption)
            _retry(runtime, clone, mj, model, before, journal)
    except BaseException as error:
        journal.emit("test-failed", name=name, error=repr(error))
        raise
    finally:
        _close(runtime, journal)
