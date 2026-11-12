"""D3 atomic common candidates and bounded failure protocols; unexecuted authoring."""

from dataclasses import replace

import numpy as np
import pytest

import phymeo as ph
from test_deformable_contact_response import (
    H, attempt, close, create, encode, fixture, forces, manifest, root_state, saved,
)

pytestmark = pytest.mark.tyx


def retained(before, after, items):
    assert set(before) == set(after)
    for item in items:
        a,b = before[item["world"]],after[item["world"]]
        assert a.node.tick == b.node.tick and a.node.time == b.node.time
        for field in ("position","velocity","clock"):
            av,bv = getattr(a.node,field),getattr(b.node,field)
            assert av.dtype == bv.dtype and av.shape == bv.shape and av.tobytes() == bv.tobytes()
        assert a.rigid.tick == b.rigid.tick
        assert root_state(a,item) == root_state(b,item)


def rejected_step(batch, items, evidence, *, match):
    before,revision,old = batch.snapshot(),batch.revision,batch.last_step
    checkpoint_before = batch.checkpoint()
    banks_before = (batch._nodes._active,batch._scene._service._batch._active)
    row = dict(entry=before,revision=revision,h=H,inputs=forces(items),expected_rejection=match,
               checkpoint_before=checkpoint_before)
    evidence["attempts"].append(row)
    with pytest.raises((ValueError,RuntimeError,ph.SolverFailure),match=match) as failure:
        batch.step(row["inputs"],H,expected_revision=revision)
    row.update(error=str(failure.value),last_attempt=batch.last_attempt,state=batch.snapshot(),stats=batch.stats())
    retained(before,batch.snapshot(),items)
    checkpoint_after = batch.checkpoint()
    row["checkpoint_after"] = checkpoint_after
    assert encode(checkpoint_before) == encode(checkpoint_after)
    banks_after = (batch._nodes._active,batch._scene._service._batch._active)
    row["banks_before"],row["banks_after"] = banks_before,banks_after
    assert banks_after == banks_before
    assert batch.revision == revision and batch.last_step is old
    return row


def test_real_parallel_conditioning_rejects_the_whole_response(tyx_mode,tmp_path):
    item = fixture("P",variant="conditioning")
    with saved(tmp_path,"conditioning",tyx_mode,case=manifest(item)) as evidence, create((item,),tyx_mode) as batch:
        row = rejected_step(batch,(item,),evidence,match="CONDITIONING|conditioning|normal|resolved")
        assert row["last_attempt"]["query"] is not None
        # Diagnostic evidence remains the actual query, not a fabricated force.
        assert row["last_attempt"]["node_receipt"] is None
        assert row["last_attempt"]["rigid_publication"] is None


def test_sparse_world_order_and_repeated_local_ids(tyx_mode,tmp_path):
    first = (fixture("P",world=31),fixture("E",world=7))
    second = (fixture("E",world=7,reverse=True),fixture("P",world=31,reverse=True))
    records = []
    with saved(tmp_path,"world-order",tyx_mode,cases=[manifest(i) for i in first]) as evidence:
        for items in (first,second):
            with create(items,tyx_mode) as batch:
                run = []
                for _ in range(3):
                    _,record = attempt(batch,items,H,evidence)
                    run.append(record["state"])
                records.append(run)
        for a,b in zip(*records,strict=True):
            for item in first:
                w = item["world"]
                close(a[w].node.position,b[w].node.position)
                close(a[w].node.velocity,b[w].node.velocity)
                close(root_state(a[w],item).position,root_state(b[w],item).position)
                close(root_state(a[w],item).velocity,root_state(b[w],item).velocity)


@pytest.mark.parametrize("seam",("_query_entry","_prepare_response_publication"))
def test_late_real_attempt_failure_retains_both_banks_and_retry(seam,tyx_mode,tmp_path,monkeypatch):
    items = (fixture("P",world=7),fixture("E",world=31))
    with saved(tmp_path,"late-"+seam,tyx_mode,cases=[manifest(i) for i in items]) as evidence, create(items,tyx_mode) as batch:
        attempt(batch,items,H,evidence)
        original = getattr(batch,seam)
        completed = []
        def after_actual(*args,**kwargs):
            value = original(*args,**kwargs)
            completed.append(value)
            raise RuntimeError("D3 late real seam")
        with monkeypatch.context() as patch:
            patch.setattr(batch,seam,after_actual)
            rejected_step(batch,items,evidence,match="D3 late real seam")
        assert len(completed) == 1
        evidence["completed_before_failure"] = completed
        _,retry = attempt(batch,items,H,evidence)
        with create(items,tyx_mode) as fresh:
            attempt(fresh,items,H,evidence)
            _,reference = attempt(fresh,items,H,evidence)
            for item in items:
                w = item["world"]
                close(retry["state"][w].node.position,reference["state"][w].node.position,atol=0,rtol=0)
                close(retry["state"][w].node.velocity,reference["state"][w].node.velocity,atol=0,rtol=0)
                close(root_state(retry["state"][w],item).position,
                      root_state(reference["state"][w],item).position,atol=0,rtol=0)
                close(root_state(retry["state"][w],item).velocity,
                      root_state(reference["state"][w],item).velocity,atol=0,rtol=0)


def test_active_capacity_exact_and_one_less_are_atomic(tyx_mode,tmp_path):
    items = (fixture("P",world=7),fixture("E",world=31))
    with saved(tmp_path,"active-capacity",tyx_mode,cases=[manifest(i) for i in items]) as evidence:
        with create(items,tyx_mode,contact_capacity=2) as exact:
            result,_ = attempt(exact,items,H,evidence)
            assert sum(len(row["sites"]) for row in result.worlds.values()) == 2
        with create(items,tyx_mode,contact_capacity=1) as short:
            row = rejected_step(short,items,evidence,match="capacity")
            assert row["last_attempt"]["node_receipt"] is None
            assert row["last_attempt"]["rigid_publication"] is None


def test_checkpoint_restore_partial_reset_then_real_successors(tyx_mode,tmp_path):
    items = (fixture("P",world=7),fixture("E",world=31))
    with saved(tmp_path,"restore-reset",tyx_mode,cases=[manifest(i) for i in items]) as evidence, create(items,tyx_mode) as batch:
        initial = batch.snapshot()
        attempt(batch,items,H,evidence)
        checkpoint = batch.checkpoint()
        evidence["checkpoint"] = checkpoint
        _,run_a = attempt(batch,items,H,evidence)
        request = dict(checkpoint=checkpoint,expected_revision=batch.revision)
        evidence["restore"] = dict(request=request)
        batch.restore(checkpoint,expected_revision=request["expected_revision"])
        evidence["restore"]["after"] = batch.snapshot()
        _,run_b = attempt(batch,items,H,evidence)
        for item in items:
            w = item["world"]
            close(run_a["state"][w].node.position,run_b["state"][w].node.position,atol=0,rtol=0)
            close(run_a["state"][w].node.velocity,run_b["state"][w].node.velocity,atol=0,rtol=0)
            close(root_state(run_a["state"][w],item).position,
                  root_state(run_b["state"][w],item).position,atol=0,rtol=0)
        before = batch.snapshot()
        revision = batch.revision
        evidence["reset"] = dict(worlds=(7,),expected_revision=revision,before=before)
        batch.reset((7,),expected_revision=revision)
        reset = batch.snapshot()
        evidence["reset"]["after"] = reset
        retained({31:before[31]},{31:reset[31]},(items[1],))
        close(reset[7].node.position,initial[7].node.position,atol=0,rtol=0)
        assert reset[7].node.tick == reset[7].rigid.tick == 0
        result,_ = attempt(batch,items,H,evidence)
        assert result.states[7].node.tick == 1 and result.states[31].node.tick == before[31].node.tick+1
        old = batch.snapshot()
        with pytest.raises((ValueError,RuntimeError),match="revision|stale"):
            batch.step(forces(items),H,expected_revision=revision)
        retained(old,batch.snapshot(),items)


@pytest.mark.parametrize("bad",("missing-law","foreign-world","unknown-material-point","memory"))
def test_source_and_budget_fail_before_any_step(bad,tyx_mode,tmp_path):
    item = fixture("P")
    if bad == "missing-law":
        item["materials"] = {}
    elif bad == "foreign-world":
        item["binding"] = ph.ResidentParticipantBinding(ph.ResidentParticipantKey(7,999),item["binding"].binding)
    elif bad == "unknown-material-point":
        surface = item["node"].surfaces[0]
        bad_faces = surface.oriented_point_ids.copy()
        bad_faces[1,2] = 999
        item["node"] = replace(item["node"],surfaces=(replace(surface,oriented_point_ids=bad_faces),))
    options = dict(memory_limit_bytes=1) if bad == "memory" else {}
    with saved(tmp_path,"admission-"+bad,tyx_mode,case=manifest(item)) as evidence:
        with pytest.raises((ValueError,TypeError,RuntimeError,MemoryError)) as error:
            with create((item,),tyx_mode,**options):
                raise AssertionError("invalid D3 admission unexpectedly accepted")
        evidence["rejection"] = dict(type=type(error.value).__name__,message=str(error.value))


def test_device_path_does_not_call_host_force_geometry_or_material_solver(tyx_mode,tmp_path,monkeypatch):
    import phymeo.constraints.distance as material
    import phymeo.solvers.collision_loads as load_module
    import phymeo.dynamics.floating_robot as floating
    item = fixture("P")
    def forbidden(*args,**kwargs):
        raise AssertionError("D3 attempted HOST current-state mechanics")
    with saved(tmp_path,"no-host-mechanics",tyx_mode,case=manifest(item)) as evidence, create((item,),tyx_mode) as batch:
        with monkeypatch.context() as patch:
            patch.setattr(material,"project_distances",forbidden)
            patch.setattr(load_module,"_host_law",forbidden)
            patch.setattr(floating,"floating_kinematics",forbidden)
            attempt(batch,(item,),H,evidence)


def test_query_capacity_does_not_partially_apply_first_world(tyx_mode,tmp_path):
    items = (fixture("P",world=7),fixture("E",world=31))
    limits = ph.BoundaryDistanceLimits(candidates=1)
    with saved(tmp_path,"query-capacity",tyx_mode,cases=[manifest(i) for i in items],limits=limits) as evidence, create(items,tyx_mode,limits=limits) as batch:
        rejected_step(batch,items,evidence,match="capacity")


@pytest.mark.parametrize("side_effect",(False,True))
def test_unknown_release_retains_dependencies_and_never_retries(side_effect,tyx_mode,tmp_path):
    from phymeo.backends.mujoco_tyx._resources import quarantined_resources
    item = fixture("P")
    calls = []
    class Ambiguous:
        def free(self):
            calls.append("entered")
            if side_effect:
                calls.append("side-effect")
            raise RuntimeError("D3 release outcome unknown")
    with saved(tmp_path,"unknown-release-"+str(side_effect),tyx_mode,case=manifest(item)) as evidence:
        batch = create((item,),tyx_mode)
        attempt(batch,(item,),H,evidence)
        ledger = batch._engine.resources
        owner = Ambiguous()
        ledger.add_buffer(owner)
        for _ in range(2):
            with pytest.raises(BaseExceptionGroup,match="UNKNOWN_RELEASE|cleanup|release") as error:
                batch.close()
            evidence.setdefault("close_failures",[]).append(str(error.value))
        assert calls.count("entered") == 1
        assert ledger.state == "UNKNOWN_RELEASE" and ledger.runtime_pending
        assert ledger.pending_buffer_count == 0
        assert ledger in quarantined_resources()
        assert any(f.owner is owner for f in ledger.failures)
        with pytest.raises((ValueError,RuntimeError),match="closed|release|poison|pending|UNKNOWN"):
            batch.step(forces((item,)),H,expected_revision=batch.revision)
        evidence["release"] = dict(calls=calls,state=ledger.state,pending=ledger.pending_buffer_count,
            runtime_pending=ledger.runtime_pending,owners=[type(f.owner).__name__ for f in ledger.failures])


@pytest.mark.parametrize("tyx_mode",("cuda",),indirect=True)
def test_cuda_successful_free_then_throw_is_still_unknown(tyx_mode,tmp_path):
    from phymeo.backends.mujoco_tyx._resources import quarantined_resources
    item = fixture("P")
    with saved(tmp_path,"cuda-free-unknown",tyx_mode,case=manifest(item)) as evidence:
        batch = create((item,),tyx_mode)
        attempt(batch,(item,),H,evidence)
        ledger = batch._engine.resources
        original = ledger._buffers[-1]
        entered = []
        def actual_then_throw():
            original.callback()
            entered.append(True)
            raise RuntimeError("D3 after actual CUDA free")
        ledger._buffers[-1] = replace(original,callback=actual_then_throw)
        for _ in range(2):
            with pytest.raises(BaseExceptionGroup,match="UNKNOWN_RELEASE|cleanup|release") as error:
                batch.close()
            evidence.setdefault("close_failures",[]).append(str(error.value))
        assert entered == [True]
        assert ledger.state == "UNKNOWN_RELEASE" and ledger.runtime_pending
        assert ledger in quarantined_resources() and ledger.pending_buffer_count == 0
        assert any(f.owner is original.owner for f in ledger.failures)
        evidence["release"] = dict(actual_completed=True,state=ledger.state,runtime_pending=ledger.runtime_pending)
