"""D6-A ownership, late failures, model binding and exact resource boundaries."""

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
from deformable_dissipation_cases import (
    H_VALUES, SETTINGS, make_world, packets, saved, step_record,
)

import phymeo as ph
from phymeo.backends.deformable_dissipation import DistanceDissipation
from phymeo.backends.tyx_deformable_batch import TyxDeformableBatch
from phymeo.constraints.distance import DistanceConstraints
from phymeo.core.state import ParticleState

pytestmark = pytest.mark.tyx


def same_state(actual, expected, *, same_history=True):
    assert tuple(actual) == tuple(expected)
    for key in actual:
        left, right = actual[key], expected[key]
        assert (left.world, left.tick, left.time) == (right.world, right.tick, right.time)
        for name in ("position", "velocity", "clock"):
            a, b = getattr(left, name), getattr(right, name)
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        if same_history:
            assert left.last_step is right.last_step


def query_plan(batch, specs):
    shapes, bindings = [], []
    for spec in specs:
        surface, sphere = ph.CollisionKey(spec.key, 101, 0), ph.CollisionKey(spec.key, 202, 0)
        shapes.extend((ph.DeformableBoundarySurface(surface, 23),
                       ph.CollisionShape(sphere, "sphere", (17,), radius=0.125)))
        bindings.append(ph.ResidentStaticBinding(
            sphere, ph.CollisionMotion(np.array(((0.25, 0.25, 0.5),), np.float64))))
    return batch.prepare_boundary_query(ph.DeformableBoundaryModel(tuple(shapes)), tuple(bindings))


@pytest.mark.parametrize("fault", ("last-world", "receipt"))
def test_late_failure_keeps_all_banks_queries_and_history_then_retries(fault, tyx_mode, monkeypatch, tmp_path):
    specs = (make_world("fixed_chain"), make_world("chain", key=31))
    with saved(tmp_path, "atomic-" + fault, cases=specs, mode=tyx_mode) as emit:
        with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS, trace=True) as batch:
            step_record(batch, specs, H_VALUES[0], emit)
            plan = query_plan(batch, specs)
            old_query = plan.query(margin=1.0)
            before, checkpoint, revision = batch.snapshot(), batch.checkpoint(), batch.revision
            old_receipt, bank = batch.last_step, checkpoint.active_bank
            emit("before-fault", checkpoint=checkpoint, query=old_query, stats=batch.stats())
            with monkeypatch.context() as patch:
                if fault == "last-world":
                    original = batch._read_candidate
                    def fail(key, candidate_bank):
                        result = original(key, candidate_bank)
                        emit("actual-candidate", world=key, bank=candidate_bank, arrays=result)
                        if key == 31:
                            raise RuntimeError("known late second-world read refusal")
                        return result
                    patch.setattr(batch, "_read_candidate", fail)
                else:
                    original = batch._make_receipt
                    def fail(*args):
                        result = original(*args)
                        emit("actual-candidate-receipt", receipt=result)
                        raise RuntimeError("known late complete-receipt refusal")
                    patch.setattr(batch, "_make_receipt", fail)
                with pytest.raises(RuntimeError, match="known late"):
                    batch.step(packets(specs), H_VALUES[0], expected_revision=revision)
            same_state(batch.snapshot(), before)
            assert batch.revision == revision and batch.last_step is old_receipt
            assert batch._active == bank and plan.last_result is old_query
            plan.require_current(old_query)
            with batch.borrow_vertices(expected_revision=revision) as loan:
                assert loan.active_bank == bank
                for key in batch.world_keys:
                    np.testing.assert_array_equal(batch._buffers.read(loan.position(key)),
                                                  before[key].position)
                    native_v = batch._buffers.worlds[key][f"v{bank}"]
                    np.testing.assert_array_equal(batch._buffers.read(native_v), before[key].velocity)
            retry = step_record(batch, specs, H_VALUES[0], emit)
            with pytest.raises(ValueError):
                plan.require_current(old_query)
            after = batch.snapshot()
            batch.restore(checkpoint, expected_revision=batch.revision)
            repeated = step_record(batch, specs, H_VALUES[0], emit)
            same_state(repeated.states, after, same_history=False)
            for key in batch.world_keys:
                np.testing.assert_array_equal(repeated.worlds[key].dissipation.trace,
                                              retry.worlds[key].dissipation.trace)
            emit("retry-and-restored-successor", retry=retry, repeated=repeated,
                 query=plan.query(margin=1.0))


def test_checkpoint_partial_reset_preserves_unselected_phase_and_real_successors(tyx_mode, tmp_path):
    specs = (make_world("chain"), make_world("pair", key=31))
    with saved(tmp_path, "checkpoint-reset", cases=specs, mode=tyx_mode) as emit:
        with TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS, trace=True) as batch:
            for _ in range(3):
                step_record(batch, specs, H_VALUES[0], emit)
            checkpoint, original = batch.checkpoint(), batch.snapshot()
            first = step_record(batch, specs, H_VALUES[0], emit)
            batch.restore(checkpoint, expected_revision=batch.revision)
            same_state(batch.snapshot(), original)
            second = step_record(batch, specs, H_VALUES[0], emit)
            same_state(first.states, second.states, same_history=False)
            for key in batch.world_keys:
                np.testing.assert_array_equal(first.worlds[key].dissipation.trace,
                                              second.worlds[key].dissipation.trace)
            untouched = batch.snapshot()[7]
            revision = batch.revision
            batch.reset((), expected_revision=revision)
            assert batch.revision == revision
            batch.reset((31,), expected_revision=revision)
            assert batch.snapshot()[7] is untouched
            reset = batch.snapshot()[31]
            np.testing.assert_array_equal(reset.position, specs[1].initial_state.position)
            np.testing.assert_array_equal(reset.velocity, specs[1].initial_state.velocity)
            assert reset.last_step is None and reset.tick == 0
            emit("partial-reset", checkpoint=batch.checkpoint())
            next_step = step_record(batch, specs, H_VALUES[0], emit)
            assert next_step.states[7].tick == untouched.tick + 1
            assert next_step.states[31].tick == 1
            np.testing.assert_array_equal(next_step.worlds[31].input_velocity,
                                          specs[1].initial_state.velocity)
            with pytest.raises(ValueError):
                batch.reset((7,), expected_revision=revision)


@pytest.mark.parametrize("field", ("trace", "node_impulse", "kinetic_change", "reconstructed_velocity"))
def test_checkpoint_hash_binds_every_new_phase_payload(field, tyx_mode, tmp_path):
    spec = make_world()
    with saved(tmp_path, "phase-binding-" + field, case=spec, mode=tyx_mode) as emit:
        with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS) as batch:
            step_record(batch, (spec,), H_VALUES[0], emit)
            checkpoint = batch.checkpoint()
            state, row = checkpoint.worlds[7], checkpoint.worlds[7].last_step
            changed = np.array(getattr(row.dissipation, field), copy=True)
            changed.flat[0] += 0.125
            frozen = np.frombuffer(changed.tobytes(), dtype=changed.dtype).reshape(changed.shape)
            phase = replace(row.dissipation, **{field: frozen})
            forged_row = replace(row, dissipation=phase)
            forged_state = replace(state, last_step=forged_row)
            states = MappingProxyType({7: forged_state})
            receipt = replace(checkpoint.last_step, states=states,
                              worlds=MappingProxyType({7: forged_row}))
            forged = replace(checkpoint, worlds=states, last_step=receipt)
            before, revision = batch.snapshot(), batch.revision
            emit("forged-checkpoint", original=checkpoint, forged=forged)
            with pytest.raises((ValueError, TypeError)):
                batch.restore(forged, expected_revision=revision)
            same_state(batch.snapshot(), before)
            assert batch.revision == revision
            step_record(batch, (spec,), H_VALUES[0], emit)


@pytest.mark.parametrize("bad", ("reordered", "reversed", "missing", "unknown-id", "wrong-type", "fixed-velocity"))
def test_declaration_binding_rejects_before_any_runtime_allocation(bad, monkeypatch):
    import phymeo.backends._tyx_deformable_buffers as buffers

    spec = make_world("fixed_chain" if bad == "fixed-velocity" else "chain")
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid declaration reached native runtime")
    monkeypatch.setattr(buffers, "create_runtime", forbidden)
    ids = spec.point_ids[spec.constraints.indices].copy()
    c = spec.dissipation.coefficients_ns_per_m.copy()
    if bad == "reordered":
        ids, c = ids[::-1], c[::-1]
    elif bad == "reversed":
        ids = ids[:, ::-1]
    elif bad == "missing":
        ids, c = ids[:1], c[:1]
    elif bad == "unknown-id":
        ids[0, 0] = 99991
    if bad == "wrong-type":
        spec = replace(spec, dissipation=object())
    elif bad == "fixed-velocity":
        v = spec.initial_state.velocity.copy()
        v[0, 0] = 1 / 8
        spec = replace(spec, initial_state=ParticleState(spec.initial_state.position, v))
    else:
        spec = replace(spec, dissipation=DistanceDissipation(ids, c))
    with pytest.raises((ValueError, TypeError)):
        TyxDeformableBatch((spec,), mode="interpret", settings=SETTINGS)
    assert calls == []


@pytest.mark.parametrize("trace", (False, True))
def test_active_memory_formula_exact_budget_and_one_less(trace, tyx_mode, monkeypatch, tmp_path):
    import phymeo.backends._tyx_deformable_buffers as buffers

    spec, absent = make_world("chain"), make_world("chain", damping=None)
    need = TyxDeformableBatch.requirements((spec,), settings=SETTINGS, trace=trace)
    old = TyxDeformableBatch.requirements((absent,), settings=SETTINGS, trace=trace)
    n, e, p = spec.model.particle_count, len(spec.constraints.indices), len(spec.constraints.indices)
    assert need.device_bytes - old.device_bytes == 12 * n + 132 * p
    # Full parameter and receipt ownership, plus original two-device reservation.
    assert need.host_peak_bytes - old.host_peak_bytes >= (
        24 * e + 4 * p + 3 * (72 * n + 176 * e) + 2 * (12 * n + 132 * p))
    assert need.total_bytes == need.device_bytes + need.host_peak_bytes
    with saved(tmp_path, "memory-boundary", case=spec, requirements=need, mode=tyx_mode) as emit:
        with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS, trace=trace,
                               memory_limit_bytes=need.total_bytes) as batch:
            step_record(batch, (spec,), H_VALUES[0], emit)
            assert batch.stats()["native_buffer_bytes"] == need.device_bytes
        calls = []
        def forbidden(*args, **kwargs):
            calls.append(True)
            raise AssertionError("short budget allocated a runtime")
        monkeypatch.setattr(buffers, "create_runtime", forbidden)
        with pytest.raises(MemoryError):
            TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS, trace=trace,
                               memory_limit_bytes=need.total_bytes - 1)
        assert calls == []


def test_new_trace_allocation_failure_cleans_actual_prior_resources(tyx_mode, monkeypatch, tmp_path):
    from phymeo.backends._tyx_deformable_buffers import DeformableBuffers

    spec, observed = make_world("chain"), []
    original = DeformableBuffers.allocate
    def fail(self, value):
        result = original(self, value)
        if np.asarray(value).shape == (2, 32):
            observed.append(self)
            raise RuntimeError("known failure after actual new trace allocation")
        return result
    with saved(tmp_path, "late-allocation", case=spec, mode=tyx_mode) as emit:
        with monkeypatch.context() as patch:
            patch.setattr(DeformableBuffers, "allocate", fail)
            with pytest.raises(RuntimeError, match="actual new trace allocation"):
                TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS)
        assert len(observed) == 1 and observed[0].resources.state == "CLOSED"
        assert observed[0].resources.pending_buffer_count == 0
        assert not observed[0].resources.runtime_pending
        emit("cleanup", state=observed[0].resources.state,
             actual_allocations=observed[0].allocations, native_bytes=observed[0].native_bytes)
        with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS) as retry:
            step_record(retry, (spec,), H_VALUES[0], emit)


@pytest.mark.parametrize("damping", ("positive", "zero"))
def test_zero_length_positive_damping_refuses_without_axis_fallback(damping, tyx_mode, tmp_path):
    spec = make_world(damping=damping)
    x = np.zeros((2, 3), np.float32)
    spec = replace(spec, initial_state=ParticleState(x, x),
                   constraints=DistanceConstraints(spec.constraints.indices, np.zeros(1),
                                                   np.full(1, 1 / 64)))
    with saved(tmp_path, "zero-length-" + damping, case=spec, mode=tyx_mode) as emit:
        with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS) as batch:
            before = batch.snapshot()
            emit("before", state=before, stats=batch.stats())
            if damping == "positive":
                with pytest.raises((ValueError, RuntimeError)):
                    batch.step(packets((spec,)), H_VALUES[0], expected_revision=0)
                same_state(batch.snapshot(), before)
                assert batch.revision == 0 and batch.last_step is None and batch._active == 0
            else:
                result = step_record(batch, (spec,), H_VALUES[0], emit)
                assert not hasattr(result.worlds[7], "dissipation")
                np.testing.assert_array_equal(result.states[7].velocity, 0)
            emit("after", state=batch.snapshot(), stats=batch.stats())


@pytest.mark.parametrize("tyx_mode", ("cuda",), indirect=True)
def test_new_damping_trace_unknown_release_keeps_runtime_without_second_free(tyx_mode, tmp_path):
    spec = make_world("chain")
    with saved(tmp_path, "unknown-new-trace", case=spec, mode=tyx_mode) as emit:
        batch = TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS)
        step_record(batch, (spec,), H_VALUES[0], emit)
        before, receipt = batch.snapshot(), batch.last_step
        ledger, calls = batch._buffers.resources, []
        matches = [(index, release) for index, release in enumerate(ledger._buffers)
                   if tuple(getattr(release.owner, "shape", ())) == (2, 32)]
        assert len(matches) == 1
        index, release = matches[0]
        def free_then_fail():
            release.callback()
            calls.append(True)
            raise RuntimeError("new damping trace free returned then outcome made unknown")
        ledger._buffers[index] = replace(release, callback=free_then_fail)
        with pytest.raises(BaseExceptionGroup):
            batch.close()
        assert calls == [True] and batch.resource_state == "UNKNOWN_RELEASE"
        assert ledger.pending_buffer_count == 0 and ledger.runtime_pending
        assert ledger.failures[0].owner is release.owner
        assert batch.last_step is receipt and batch._states is before
        emit("unknown", resource_state=batch.resource_state, retained_receipt=receipt,
             runtime_retained=ledger.runtime_pending, sibling_pending=ledger.pending_buffer_count)
        with pytest.raises(BaseExceptionGroup):
            batch.close()
        assert calls == [True]
        with pytest.raises(RuntimeError):
            batch.step(packets((spec,)), H_VALUES[0], expected_revision=batch.revision)


def test_original_coefficient_binds_even_when_native_word_is_the_same(tyx_mode, tmp_path):
    original = make_world()
    changed = replace(original, dissipation=DistanceDissipation(
        original.dissipation.ordered_point_ids, np.array((2 + 2**-30,), np.float64)))
    np.testing.assert_array_equal(original.dissipation.coefficients_ns_per_m.astype(np.float32),
                                  changed.dissipation.coefficients_ns_per_m.astype(np.float32))
    with saved(tmp_path, "original-parameter-binding", cases=(original, changed), mode=tyx_mode) as emit:
        with (
            TyxDeformableBatch((original,), mode=tyx_mode, settings=SETTINGS) as a,
            TyxDeformableBatch((changed,), mode=tyx_mode, settings=SETTINGS) as b,
        ):
            assert a.binding_sha256 != b.binding_sha256
            ar = step_record(a, (original,), H_VALUES[0], emit)
            br = step_record(b, (changed,), H_VALUES[0], emit)
            np.testing.assert_array_equal(ar.states[7].velocity, br.states[7].velocity)
            np.testing.assert_array_equal(ar.worlds[7].dissipation.trace,
                                          br.worlds[7].dissipation.trace)
            assert ar.worlds[7].dissipation.coefficients_ns_per_m[0] == 2
            assert br.worlds[7].dissipation.coefficients_ns_per_m[0] == 2 + 2**-30
            with pytest.raises(ValueError):
                b.restore(a.checkpoint(), expected_revision=b.revision)