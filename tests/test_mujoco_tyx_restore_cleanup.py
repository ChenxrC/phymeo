"""G2-R0: failed real restore candidates retain cleanup owners and both errors.

The two pre-ledger cleanup faults are known Python failures with no native
release side effect. They are not UNKNOWN_RELEASE and do not authorize retrying
an ambiguous device callback. This test also safely owns leaked candidates when
run against the pre-fix implementation to establish the expected RED.
"""

import traceback

import pytest
from test_mujoco_tyx_activation import encode, save
from test_mujoco_tyx_clone_cleanup import MODES, XML

from phymeo.backends import mujoco_checkpoint as checkpoint_implementation
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

KEYS = (7, 91)
H = 0.001


def exception_record(error):
    if error is None:
        return None
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "children": (
            tuple(exception_record(child) for child in error.exceptions)
            if isinstance(error, BaseExceptionGroup)
            else ()
        ),
    }


def mirror_record(mirror):
    return {
        "time": float(mirror.time),
        "cpu_state": mirror.cpu_state.copy(),
        "fields": {
            name: getattr(mirror, name).copy() for name in checkpoint_implementation.MIRROR_FIELDS
        },
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ("verified_bank", "last_observer"))
@pytest.mark.parametrize("cleanup", ("success", "preledger_exception", "preledger_baseexception"))
def test_failed_restore_retains_candidate_cleanup_owner(
    mode, fault, cleanup, monkeypatch, tmp_path
):
    record = {
        "mode": mode,
        "fault": fault,
        "cleanup": cleanup,
        "xml": XML,
        "world_keys": KEYS,
        "h": H,
        "cleanup_fault_domain": "known-before-ledger; no UNKNOWN_RELEASE injection",
        "steps": [],
        "banks": [],
        "observers": [],
        "fallback_cleanup": [],
    }
    runtime = reference = None
    tracked = []
    primary = MemoryError("authored failure after complete " + fault)
    cleanup_error = (
        KeyboardInterrupt("authored cleanup BaseException before ledger")
        if cleanup == "preledger_baseexception"
        else RuntimeError("authored cleanup Exception before ledger")
    )
    inject_cleanup = False
    cleanup_injected = False
    failure = None
    finalization_errors = []

    def track(bank, label):
        item = {
            "label": label,
            "owner_id": id(bank),
            "close_calls": 0,
            "actual_release_count": 0,
            "events": [],
        }
        tracked.append((bank, item))
        record["banks"].append(item)
        original_close = bank.close

        def close():
            nonlocal cleanup_injected
            item["close_calls"] += 1
            before = bank.resources.state
            event = {
                "before": before,
                "buffers_before": bank.resources.pending_buffer_count,
                "runtime_before": bank.resources.runtime_pending,
            }
            item["events"].append(event)
            if (
                label == "failed-candidate"
                and inject_cleanup
                and cleanup != "success"
                and not cleanup_injected
            ):
                cleanup_injected = True
                event["preledger_failure"] = exception_record(cleanup_error)
                assert before == "OPEN"
                raise cleanup_error
            try:
                original_close()
            except BaseException as error:
                event["release_error"] = exception_record(error)
                raise
            finally:
                event["after"] = bank.resources.state
                event["buffers_after"] = bank.resources.pending_buffer_count
                event["runtime_after"] = bank.resources.runtime_pending
            if before == "OPEN" and bank.resources.state == "CLOSED":
                item["actual_release_count"] += 1

        monkeypatch.setattr(bank, "close", close)
        return item

    def accepted_step(owner, label):
        result = owner.step(H)
        evidence = {"label": label, "accepted": dict(result)}
        record["steps"].append(evidence)
        # Save the real return before any further observation or assertion.
        evidence["integration"] = owner.integration_snapshot()
        evidence["snapshot"] = owner.snapshot()
        assert dict(result) == dict.fromkeys(KEYS, True)
        return evidence["snapshot"]

    try:
        compilation = compile_mujoco_xml(XML, require_asset=False)
        runtime = MuJoCoRuntime(compilation, mode=mode, world_keys=KEYS)
        controls = {7: MuJoCoControl([0.6]), 91: MuJoCoControl([-0.2])}
        record["requested_controls"] = controls
        runtime.set_controls(controls)
        target_snapshot = accepted_step(runtime, "checkpoint-entry")
        target = runtime.checkpoint()
        record.update(target=target_snapshot, target_checkpoint=target.to_bytes().hex())
        assert all(state.act.size == 1 and state.act[0] != 0 for state in target_snapshot.values())
        reference = runtime.clone()
        assert encode(reference.snapshot()) == encode(target_snapshot)
        track(reference._device, "reference-accepted")
        accepted_step(runtime, "advance-before-failed-restore")
        old = runtime._device
        old_bank = old.active
        old_revision = runtime.checkpoint_revision
        old_fields = {
            name: getattr(runtime, name)
            for name in ("_data", "_controls", "_ticks", "_receipts", "_initial")
        }
        track(old, "original-accepted")
        before = runtime.snapshot()
        before_integration = runtime.integration_snapshot()
        before_checkpoint = runtime.checkpoint().to_bytes()
        record.update(
            before=before,
            before_integration=before_integration,
            before_checkpoint=before_checkpoint.hex(),
            before_revision=old_revision,
            before_active_bank=old_bank,
            before_owner_id=id(old),
        )
        assert before_checkpoint != target.to_bytes()
        assert not runtime._checkpoint_retired

        original_build = runtime._build_device
        original_verify = checkpoint_implementation._verify_bank
        original_observe = runtime._observe_native
        candidates = []

        def build(data, *, packed=None):
            bank = original_build(data, packed=packed)
            candidates.append(bank)
            label = "failed-candidate" if len(candidates) == 1 else "retry-candidate"
            track(bank, label)
            record.setdefault("candidate_builds", []).append(
                {
                    "label": label,
                    "owner_id": id(bank),
                    "active_bank": bank.active,
                    "packed_input": tuple(value.copy() for value in packed),
                    "mirrors": {key: mirror_record(value) for key, value in data.items()},
                }
            )
            return bank

        def verified(bank, packed):
            original_verify(bank, packed)
            record["verified_candidate_bank"] = tuple(
                value.copy() for value in checkpoint_implementation._accepted_bank(bank)
            )
            record["verified_candidate_id"] = id(bank)
            if fault == "verified_bank":
                raise primary

        def observed(mirror):
            value = original_observe(mirror)
            ordinal = len(record["observers"])
            record["observers"].append(
                {
                    "world": KEYS[ordinal],
                    "mirror": mirror_record(mirror),
                    "native_time": float(value.time),
                    "native_fields": {
                        name: getattr(value, name).copy()
                        for name in checkpoint_implementation.MIRROR_FIELDS
                    },
                }
            )
            if ordinal == len(KEYS) - 1:
                raise primary
            return value

        # Build instrumentation remains for the genuine successful retry.
        monkeypatch.setattr(runtime, "_build_device", build)
        caught = None
        try:
            with monkeypatch.context() as patch:
                patch.setattr(checkpoint_implementation, "_verify_bank", verified)
                if fault == "last_observer":
                    patch.setattr(runtime, "_observe_native", observed)
                inject_cleanup = True
                try:
                    runtime.restore(target, expected_revision=old_revision)
                except BaseException as error:
                    caught = error
        finally:
            inject_cleanup = False

        # These observations are after all failure injection has been removed.
        # Record first, so RED preserves both the exception and lost-owner fact.
        record["restore_exception"] = exception_record(caught)
        record["retained_owner_ids"] = tuple(id(owner) for owner in runtime._checkpoint_retired)
        record["after_failure_revision"] = runtime.checkpoint_revision
        record["after_failure_owner_id"] = id(runtime._device)
        record["after_failure_active_bank"] = runtime._device.active
        record["after_failure"] = runtime.snapshot()
        record["after_failure_integration"] = runtime.integration_snapshot()
        record["after_failure_checkpoint"] = runtime.checkpoint().to_bytes().hex()
        assert len(candidates) == 1
        failed = candidates[0]
        assert record["verified_candidate_id"] == id(failed)
        assert len(record["observers"]) == (len(KEYS) if fault == "last_observer" else 0)
        assert runtime._device is old and old.active == old_bank
        assert runtime.checkpoint_revision == old_revision
        assert all(getattr(runtime, name) is value for name, value in old_fields.items())
        assert encode(record["after_failure"]) == encode(before)
        assert encode(record["after_failure_integration"]) == encode(before_integration)
        assert bytes.fromhex(record["after_failure_checkpoint"]) == before_checkpoint
        if cleanup == "success":
            assert caught is primary
            assert not runtime._checkpoint_retired
            assert failed.resources.state == "CLOSED"
        else:
            assert isinstance(caught, BaseExceptionGroup)
            assert caught.exceptions == (primary, cleanup_error)
            assert runtime._checkpoint_retired == [failed]
            assert runtime._checkpoint_retired[0] is failed
            assert failed.resources.state == "OPEN"
            assert not failed.resources.failures
        failed_item = next(item for bank, item in tracked if bank is failed)
        assert failed_item["close_calls"] == 1
        assert failed_item["actual_release_count"] == (1 if cleanup == "success" else 0)

        runtime.restore(target, expected_revision=old_revision)
        record["retry"] = {
            "returned": True,
            "revision": runtime.checkpoint_revision,
            "owner_id": id(runtime._device),
            "active_bank": runtime._device.active,
            "retained_owner_ids": tuple(id(owner) for owner in runtime._checkpoint_retired),
        }
        record["retry"]["snapshot"] = runtime.snapshot()
        record["retry"]["integration"] = runtime.integration_snapshot()
        record["retry"]["checkpoint"] = runtime.checkpoint().to_bytes().hex()
        assert len(candidates) == 2 and runtime._device is candidates[1]
        assert runtime.checkpoint_revision == old_revision + 1
        assert old.resources.state == "CLOSED"
        assert encode(record["retry"]["snapshot"]) == encode(target_snapshot)
        assert encode(record["retry"]["integration"]) == encode(reference.integration_snapshot())
        for index in range(2):
            actual = accepted_step(runtime, "retry-successor-" + str(index))
            expected = accepted_step(reference, "reference-successor-" + str(index))
            assert encode(actual) == encode(expected)
            assert encode(runtime.integration_snapshot()) == encode(
                reference.integration_snapshot()
            )
        # GREEN proves the production runtime itself releases the retained owner.
        # Test fallback is solely for safe teardown of the expected old-source RED.
        runtime.close()
        reference.close()
        record["production_closed"] = {
            "retained_owner_ids": tuple(id(owner) for owner in runtime._checkpoint_retired),
            "bank_states": tuple(bank.resources.state for bank, _ in tracked),
        }
        assert not runtime._checkpoint_retired
        assert all(bank.resources.state == "CLOSED" for bank, _ in tracked)
        assert all(item["actual_release_count"] == 1 for _, item in tracked)
        assert not record["fallback_cleanup"]
        record["status"] = "PASS"
    except BaseException as error:
        failure = error
        record["status"] = "FAIL"
        record["failure"] = exception_record(error)
    finally:
        inject_cleanup = False
        # No injected method can reach this path: only the permanent transparent
        # counter wrapper remains. UNKNOWN ledgers are never retried here.
        for label, owner in (("runtime", runtime), ("reference", reference)):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as error:
                    finalization_errors.append(error)
                    record.setdefault("finalization_errors", []).append(
                        {"owner": label, "error": exception_record(error)}
                    )
        for bank, item in tracked:
            if bank.resources.state == "OPEN":
                record["fallback_cleanup"].append(item["label"])
                try:
                    bank.close()
                except BaseException as error:
                    finalization_errors.append(error)
                    record.setdefault("finalization_errors", []).append(
                        {"owner": item["label"], "error": exception_record(error)}
                    )
        record["final_bank_states"] = tuple(
            {
                "label": item["label"],
                "state": bank.resources.state,
                "actual_release_count": item["actual_release_count"],
            }
            for bank, item in tracked
        )
        if finalization_errors:
            record["status"] = "FAIL"
        try:
            save(tmp_path / "restore-cleanup.json.gz", record)
        except BaseException as error:
            finalization_errors.append(error)
    errors = ([] if failure is None else [failure]) + finalization_errors
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("restore cleanup regression and finalization failed", errors)
