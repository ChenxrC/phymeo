"""D1 accepted/candidate ownership, scoped loans and failure boundaries."""
from dataclasses import replace

import numpy as np
import pytest

from phymeo.backends.tyx_deformable_batch import TyxDeformableBatch
from test_tyx_deformable_batch import H, SETTINGS, arrays_equal, forces, record, world

pytestmark = pytest.mark.tyx


def test_checkpoint_partial_reset_and_real_successor(tyx_mode, tmp_path):
    specs = (world(), world(31, square=True, fixed=True))
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as batch:
        for t in range(3):
            batch.step(forces(specs,t), H, expected_revision=batch.revision)
            record(tmp_path,"warm",input=forces(specs,t),receipt=batch.last_step,state=batch.snapshot())
        saved = batch.snapshot()
        capsule = batch.checkpoint()
        batch.step(forces(specs,4), H, expected_revision=batch.revision)
        successor = batch.snapshot()
        record(tmp_path,"branch-a",checkpoint=capsule,receipt=batch.last_step,state=successor)
        old_revision = batch.revision
        batch.restore(capsule, expected_revision=old_revision)
        arrays_equal(batch.snapshot(), saved)
        with pytest.raises(ValueError):
            batch.restore(capsule, expected_revision=old_revision)
        batch.step(forces(specs,4), H, expected_revision=batch.revision)
        record(tmp_path,"branch-b",receipt=batch.last_step,state=batch.snapshot())
        for key in batch.world_keys:
            np.testing.assert_array_equal(batch.snapshot()[key].position, successor[key].position)
            np.testing.assert_array_equal(batch.snapshot()[key].velocity, successor[key].velocity)
        untouched = batch.snapshot()[7]
        before = batch.revision
        batch.reset((), expected_revision=before)
        assert batch.revision == before
        batch.reset((31,), expected_revision=before)
        arrays_equal({7:batch.snapshot()[7]}, {7:untouched})
        assert batch.snapshot()[31].tick == 0 and batch.snapshot()[31].last_step is None
        np.testing.assert_array_equal(batch.snapshot()[31].position, specs[1].initial_state.position)
        record(tmp_path,"partial-reset",state=batch.snapshot(),checkpoint=batch.checkpoint())
        for _ in range(2):
            assert batch.step(forces(specs), H, expected_revision=batch.revision).accepted
            record(tmp_path,"post-reset",receipt=batch.last_step,state=batch.snapshot())
        with pytest.raises(ValueError):
            batch.restore(replace(capsule,binding_sha256="0"*64), expected_revision=batch.revision)


@pytest.mark.parametrize("operation", ["step", "reset", "restore", "close", "nested"])
def test_loan_prevents_reentry_and_expires(operation, tyx_mode):
    specs = (world(),)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as batch:
        checkpoint = batch.checkpoint()
        with batch.borrow_vertices(expected_revision=0) as loan:
            assert loan.world_keys == (7,) and loan.revision == 0 and loan.active_bank == 0
            assert loan.position(7).shape == (3,3)
            with pytest.raises(RuntimeError):
                if operation == "step":
                    batch.step(forces(specs), H, expected_revision=0)
                elif operation == "reset":
                    batch.reset((7,), expected_revision=0)
                elif operation == "restore":
                    batch.restore(checkpoint, expected_revision=0)
                elif operation == "close":
                    batch.close()
                else:
                    with batch.borrow_vertices(expected_revision=0):
                        pass
        with pytest.raises(RuntimeError):
            loan.position(7)
        assert batch.step(forces(specs), H, expected_revision=0).accepted
        with pytest.raises(ValueError):
            with batch.borrow_vertices(expected_revision=0):
                pass


@pytest.mark.parametrize("fault", ["readback", "last-world", "receipt", "restore", "reset"])
def test_late_failure_keeps_whole_accepted_bank(fault, tyx_mode, monkeypatch, tmp_path):
    specs = (world(), world(31, square=True, fixed=True))
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as batch:
        batch.step(forces(specs), H, expected_revision=0)
        saved, checkpoint, revision = batch.snapshot(), batch.checkpoint(), batch.revision
        record(tmp_path,"before-fault",fault=fault,input=forces(specs),checkpoint=checkpoint,state=saved)
        with monkeypatch.context() as patch:
            if fault in ("restore", "reset"):
                original = batch._verify_replacement
                def refuse(*args, **kwargs):
                    original(*args, **kwargs)
                    record(tmp_path,"verified-replacement",expected=args[0],candidate_bank=args[1])
                    raise RuntimeError("real prepared state refused")
                patch.setattr(batch, "_verify_replacement", refuse)
                with pytest.raises(RuntimeError, match="real prepared"):
                    if fault == "restore":
                        batch.restore(checkpoint, expected_revision=revision)
                    else:
                        batch.reset((31,), expected_revision=revision)
            elif fault == "receipt":
                original = batch._make_receipt
                def refuse(*args, **kwargs):
                    candidate = original(*args, **kwargs)
                    record(tmp_path,"complete-candidate-receipt",receipt=candidate)
                    raise RuntimeError("receipt after actual data refused")
                patch.setattr(batch, "_make_receipt", refuse)
                with pytest.raises(RuntimeError, match="receipt after"):
                    batch.step(forces(specs), H, expected_revision=revision)
            else:
                original = batch._read_candidate
                def refuse(key, *args, **kwargs):
                    result = original(key, *args, **kwargs)
                    record(tmp_path,"actual-candidate-read",world=key,arrays=result)
                    if fault == "readback" or key == 31:
                        raise RuntimeError("complete actual candidate read refused")
                    return result
                patch.setattr(batch, "_read_candidate", refuse)
                with pytest.raises(RuntimeError, match="complete actual"):
                    batch.step(forces(specs), H, expected_revision=revision)
        record(tmp_path,"after-fault",state=batch.snapshot(),checkpoint=batch.checkpoint())
        arrays_equal(batch.snapshot(), saved)
        assert batch.revision == revision
        with batch.borrow_vertices(expected_revision=revision) as loan:
            assert loan.active_bank == checkpoint.active_bank
        assert batch.step(forces(specs), H, expected_revision=revision).accepted
        after = batch.snapshot()
        record(tmp_path,"successful-retry",receipt=batch.last_step,state=after)
        batch.restore(checkpoint, expected_revision=batch.revision)
        assert batch.step(forces(specs), H, expected_revision=batch.revision).accepted
        for key in batch.world_keys:
            np.testing.assert_array_equal(batch.snapshot()[key].position, after[key].position)
            np.testing.assert_array_equal(batch.snapshot()[key].velocity, after[key].velocity)


def test_nonfinite_force_stale_cas_and_duplicate_reset_refused(tyx_mode):
    specs=(world(),)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as batch:
        saved=batch.snapshot()
        before_kernels = batch.stats()["kernels"]
        for bad in (float("nan"), float("inf")):
            packet = forces(specs)
            packet[7].force[0,0] = bad
            with pytest.raises(ValueError):
                batch.step(packet,H,expected_revision=0)
            arrays_equal(batch.snapshot(),saved)
            assert batch.stats()["kernels"] == before_kernels
        packet = forces(specs)
        packet[7].force = np.zeros((2,3),np.float32)
        with pytest.raises(ValueError):
            batch.step(packet,H,expected_revision=0)
        assert batch.stats()["kernels"] == before_kernels
        with pytest.raises(ValueError):
            batch.step(forces(specs),H,expected_revision=1)
        with pytest.raises(ValueError):
            batch.reset((7,7),expected_revision=0)
        arrays_equal(batch.snapshot(),saved)


def test_cleanup_pending_is_close_only_and_retry_does_not_republish(tyx_mode, monkeypatch):
    specs=(world(),)
    batch=TyxDeformableBatch(specs,mode=tyx_mode,settings=SETTINGS)
    original=batch._buffers.close
    calls=[]
    def pending():
        calls.append(True)
        raise RuntimeError("before ledger entry")
    monkeypatch.setattr(batch._buffers,"close",pending)
    with pytest.raises(RuntimeError,match="before ledger"):
        batch.close()
    assert batch.resource_state=="PENDING_CLOSE"
    with pytest.raises(RuntimeError):
        batch.step(forces(specs),H,expected_revision=0)
    monkeypatch.setattr(batch._buffers,"close",original)
    batch.close()
    assert batch.resource_state=="CLOSED" and len(calls)==1


def test_unknown_release_never_retries_opaque_callback(tyx_mode):
    specs=(world(),)
    batch=TyxDeformableBatch(specs,mode=tyx_mode,settings=SETTINGS)
    class Unknown:
        def __init__(self): self.calls=0
        def free(self):
            self.calls+=1
            raise RuntimeError("opaque release outcome unknown")
    owner=Unknown()
    batch._buffers.resources.add_buffer(owner)
    with pytest.raises(BaseExceptionGroup):
        batch.close()
    assert batch.resource_state=="UNKNOWN_RELEASE"
    with pytest.raises(BaseExceptionGroup):
        batch.close()
    assert owner.calls==1
    with pytest.raises(RuntimeError):
        batch.snapshot()


@pytest.mark.parametrize("where", ["first-before", "last-after"])
def test_real_kernel_failure_keeps_bank_and_allows_known_python_retry(where, tyx_mode, monkeypatch, tmp_path):
    specs = (world(), world(31, square=True, fixed=True))
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as batch, \
         TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as twin:
        for owner in (batch, twin):
            owner.step(forces(specs), H, expected_revision=0)
        saved = batch.snapshot()
        record(tmp_path,"before-launch-fault",where=where,state=saved,input=forces(specs,1))
        revision = batch.revision
        bank = batch.checkpoint().active_bank
        original = batch._buffers.call
        calls = []
        clocks = []
        def fail(kernel, *args):
            name = getattr(kernel, "__name__", None)
            # TYX Kernel wrappers expose the original function through .fn in some pins;
            # function identity is authoritative and needs no private execution state.
            from phymeo.backends._tyx_kernels import integrate_particles
            from phymeo.backends._tyx_deformable_state import advance_clock
            calls.append(name)
            if where == "first-before" and kernel is integrate_particles:
                raise RuntimeError("known Python fault before first physical kernel")
            result = original(kernel, *args)
            if kernel is advance_clock:
                clocks.append(True)
                if where == "last-after" and len(clocks) == 2:
                    batch._buffers.sync()
                    raise RuntimeError("known Python fault after actual last clock kernel")
            return result
        with monkeypatch.context() as patch:
            patch.setattr(batch._buffers, "call", fail)
            with pytest.raises(RuntimeError, match="known Python fault"):
                batch.step(forces(specs, 1), H, expected_revision=revision)
        arrays_equal(batch.snapshot(), saved)
        assert batch.revision == revision and batch.checkpoint().active_bank == bank
        record(tmp_path,"after-launch-fault",state=batch.snapshot(),checkpoint=batch.checkpoint(),calls=calls)
        assert calls and (where != "last-after" or len(clocks) == 2)
        assert batch.step(forces(specs, 1), H, expected_revision=revision).accepted
        assert twin.step(forces(specs, 1), H, expected_revision=twin.revision).accepted
        record(tmp_path,"kernel-fault-retry",actual=batch.last_step,independent=twin.last_step,
               actual_state=batch.snapshot(),independent_state=twin.snapshot())
        for key in batch.world_keys:
            for field in ("position", "velocity", "clock"):
                assert getattr(batch.snapshot()[key], field).tobytes() == getattr(twin.snapshot()[key], field).tobytes()


@pytest.mark.parametrize("bad", ["owner", "world", "topology", "hash"])
def test_checkpoint_binding_and_content_reject_before_launch(bad, tyx_mode):
    from types import MappingProxyType
    from dataclasses import replace
    specs = (world(),)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as batch, \
         TyxDeformableBatch((world(square=True),), mode=tyx_mode, settings=SETTINGS) as other:
        capsule = batch.checkpoint()
        if bad == "owner":
            capsule = replace(capsule, owner_id=other.checkpoint().owner_id)
        elif bad == "world":
            capsule = replace(capsule, worlds=MappingProxyType({31:capsule.worlds[7]}))
        elif bad == "topology":
            capsule = replace(capsule, binding_sha256=other.binding_sha256)
        else:
            capsule = replace(capsule, content_sha256="0"*64)
        saved, count = batch.snapshot(), batch.stats()["kernels"]
        with pytest.raises(ValueError):
            batch.restore(capsule, expected_revision=0)
        arrays_equal(batch.snapshot(), saved)
        assert batch.stats()["kernels"] == count and batch.revision == 0
