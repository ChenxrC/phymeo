"""D2 admission, full owned-native budget and late-publication fault contracts."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
import test_deformable_boundary_query as q

import phymeo as ph
from phymeo.backends._deformable_boundary_source import (
    DeformableBoundaryModel,
    DeformableBoundarySurface,
)
from phymeo.backends.tyx_deformable_batch import TyxDeformableBatch

pytestmark = pytest.mark.tyx


def test_complete_native_peak_cap_and_cap_minus_one(tyx_mode, tmp_path):
    specs = (q.world(square=True),)
    model, bindings, rounds = q.scene(specs)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=q.SETTINGS) as batch:
        with closing(batch.prepare_boundary_query(model, bindings)) as first:
            with q.saved(tmp_path, specs, model, bindings) as events:
                value = q.check(batch, first, specs, model, rounds, events)
                s, v, w = 4, 8, 1  # Four material points plus 1+2+1 static axis points.
                persistent = 32 * s + 40 * v
                gather = 36 * v + 16 * s + 4 * w
                group = 40 * 2 + 48 * (2 * 2 - 1) + (52 + 4) * 3
                expected = persistent + gather + group + 1060 * value.required_candidates
                assert first.stats()["peak_required_bytes"] == expected
                assert first.stats()["persistent_bytes"] == persistent
                assert first.stats()["actual_bank_host_readback_bytes"] == 48
        with closing(
            batch.prepare_boundary_query(model, bindings, memory_limit_bytes=expected)
        ) as exact:
            assert exact.query(margin=q.MARGIN).required_candidates == value.required_candidates
        with closing(
            batch.prepare_boundary_query(model, bindings, memory_limit_bytes=expected - 1)
        ) as low:
            with pytest.raises(MemoryError):
                low.query(margin=q.MARGIN)
            assert low.last_result is None
            assert low.last_attempt is not None
            assert low.stats()["peak_required_bytes"] == expected


def test_candidate_cap_prepass_is_complete(tyx_mode, tmp_path):
    specs = (q.world(square=True),)
    model, bindings, _ = q.scene(specs)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=q.SETTINGS) as batch:
        with closing(batch.prepare_boundary_query(model, bindings)) as full:
            count = full.query(margin=q.MARGIN).required_candidates
        assert count > 1
        with closing(
            batch.prepare_boundary_query(
                model, bindings, limits=ph.BoundaryDistanceLimits(candidates=count)
            )
        ) as exact:
            assert exact.query(margin=q.MARGIN).required_candidates == count
        with closing(
            batch.prepare_boundary_query(
                model, bindings, limits=ph.BoundaryDistanceLimits(candidates=count - 1)
            )
        ) as low:
            with pytest.raises(ph.BoundaryDistanceCapacityError):
                low.query(margin=q.MARGIN)
            assert low.last_result is None
            assert low.last_attempt is not None


@pytest.mark.parametrize("fault", ["first_launch", "last_launch", "last_download", "packet"])
def test_known_late_fault_preserves_publication(fault, tyx_mode, tmp_path, monkeypatch):
    specs = (q.world(),)
    model, bindings, rounds = q.scene(specs)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=q.SETTINGS) as batch:
        with closing(batch.prepare_boundary_query(model, bindings)) as plan:
            with q.saved(tmp_path, specs, model, bindings) as events:
                old = q.check(batch, plan, specs, model, rounds, events)
                layouts, sequence, epoch, active = (
                    plan._layouts,
                    plan._sequence,
                    batch.revision,
                    batch._active,
                )
                before = batch.snapshot()
                original_launch, original_download = plan._launch, plan._download
                calls = []

                def launch(label, *args):
                    if fault == "first_launch" and not calls:
                        raise RuntimeError("authored first launch fault")
                    result = original_launch(label, *args)
                    calls.append(label)
                    if fault == "last_launch" and "narrow" in label:
                        raise RuntimeError("authored completed narrow fault")
                    return result

                def download(array, label):
                    result = original_download(array, label)
                    if fault == "last_download" and label.endswith("meta"):
                        raise RuntimeError("authored completed readback fault")
                    return result

                monkeypatch.setattr(plan, "_launch", launch)
                monkeypatch.setattr(plan, "_download", download)
                if fault == "packet":
                    original_packet = plan._build_packet

                    def packet(*args, **kwargs):
                        original_packet(*args, **kwargs)
                        raise RuntimeError("authored completed packet fault")

                    monkeypatch.setattr(plan, "_build_packet", packet)
                try:
                    with pytest.raises(RuntimeError, match="authored"):
                        plan.query(margin=q.MARGIN)
                finally:
                    events.append(
                        dict(attempt=plan.last_attempt, state=batch.snapshot(), stats=plan.stats())
                    )
                assert plan.last_result is old and plan._layouts is layouts
                assert (plan._sequence, batch.revision, batch._active) == (sequence, epoch, active)
                assert all(batch.snapshot()[w] is state for w, state in before.items())
                monkeypatch.undo()
                q.check(batch, plan, specs, model, rounds, events)
                receipt = batch.step(q.forces(specs), q.H, expected_revision=batch.revision)
                events.append(dict(receipt=receipt))
                assert receipt.accepted
                q.check(batch, plan, specs, model, rounds, events)


@pytest.mark.parametrize("body_fault", [False, True])
def test_loan_exit_sync_failure_never_publishes(body_fault, tyx_mode, tmp_path, monkeypatch):
    specs = (q.world(),)
    model, bindings, _ = q.scene(specs)
    batch = TyxDeformableBatch(specs, mode=tyx_mode, settings=q.SETTINGS)
    plan = batch.prepare_boundary_query(model, bindings)
    try:
        old = plan.query(margin=q.MARGIN)
        layouts, sequence = plan._layouts, plan._sequence
        real_sync = batch._buffers.sync
        primary, secondary = RuntimeError("body failure"), RuntimeError("exit failure")
        calls = []

        def sync():
            real_sync()
            assert batch._loan is not None
            calls.append(len(calls))
            if len(calls) == 2:
                raise secondary

        monkeypatch.setattr(batch._buffers, "sync", sync)
        if body_fault:

            def fail(*args, **kwargs):
                raise primary

            monkeypatch.setattr(plan, "_source_record", fail)
        with pytest.raises(BaseException) as caught:
            plan.query(margin=q.MARGIN)
        assert calls == [0, 1]
        assert plan.last_result is old and plan._layouts is layouts and plan._sequence == sequence
        assert batch._loan is None and plan.last_attempt is not None
        if body_fault:
            assert isinstance(caught.value, BaseExceptionGroup)
            assert caught.value.exceptions == (primary, secondary)
        else:
            assert caught.value is secondary
        with pytest.raises(RuntimeError):
            plan.query(margin=q.MARGIN)
        with pytest.raises(RuntimeError):
            batch.step(q.forces(specs), q.H, expected_revision=batch.revision)
        with q.saved(tmp_path, specs, model, bindings) as events:
            # Poison intentionally denies snapshot as well; retain the prior complete state.
            events.append(
                dict(
                    old=old,
                    attempt=plan.last_attempt,
                    stats=plan.stats(),
                    publication_blocked=True,
                    producer_revision=batch.revision,
                )
            )
    finally:
        monkeypatch.undo()
        plan.close()
        batch.close()


def test_no_current_position_upload_and_scoped_source(tyx_mode, tmp_path, monkeypatch):
    specs = (q.world(),)
    model, bindings, rounds = q.scene(specs)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=q.SETTINGS) as batch:
        with closing(batch.prepare_boundary_query(model, bindings)) as plan:
            original_upload = plan._upload

            def upload(target, value):
                # Query uploads only integer selectors/BVH topology, never current points.
                assert np.asarray(value).dtype.kind in "iub"
                return original_upload(target, value)

            monkeypatch.setattr(plan, "_upload", upload)
            with plan._borrow_source() as view:
                plan._validate_source(view)
                with pytest.raises(RuntimeError):
                    batch.step(q.forces(specs), q.H, expected_revision=batch.revision)
            with pytest.raises(RuntimeError):
                plan._validate_source(view)
            with q.saved(tmp_path, specs, model, bindings) as events:
                q.check(batch, plan, specs, model, rounds, events)


@pytest.mark.parametrize("corruption", ["ticks", "points"])
def test_actual_source_readback_corruption_retains_publication(
    corruption, tyx_mode, tmp_path, monkeypatch
):
    specs = (q.world(),)
    model, bindings, rounds = q.scene(specs)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=q.SETTINGS) as batch:
        with closing(batch.prepare_boundary_query(model, bindings)) as plan:
            with q.saved(tmp_path, specs, model, bindings) as events:
                old = q.check(batch, plan, specs, model, rounds, events)
                original = plan._download
                changed = []

                def download(array, label):
                    value = original(array, label)
                    if label == "source/" + corruption:
                        value.flat[0] += 1
                        changed.append(value.copy())
                    return value

                monkeypatch.setattr(plan, "_download", download)
                with pytest.raises(RuntimeError):
                    plan.query(margin=q.MARGIN)
                events.append(
                    dict(
                        corruption=corruption,
                        mutated_readback=changed,
                        attempt=plan.last_attempt,
                        publication=plan.last_result,
                    )
                )
                assert changed and plan.last_result is old and batch.revision == 0
                monkeypatch.setattr(plan, "_download", original)
                q.check(batch, plan, specs, model, rounds, events)


@pytest.mark.parametrize(
    "case",
    [
        "missing_surface",
        "duplicate_surface",
        "wrong_world",
        "moving_static",
        "missing_binding",
        "oversize",
    ],
)
def test_admission_before_query_allocation(case, tyx_mode):
    specs = (q.world(),)
    model, bindings, _ = q.scene(specs)
    with TyxDeformableBatch(specs, mode=tyx_mode, settings=q.SETTINGS) as batch:
        with pytest.raises((ValueError, TypeError)):
            if case == "missing_surface":
                model = DeformableBoundaryModel(
                    (replace(model.shapes[0], surface_key=999), *model.shapes[1:])
                )
            elif case == "duplicate_surface":
                model = DeformableBoundaryModel(
                    (*model.shapes, DeformableBoundarySurface(ph.CollisionKey(7, 303, 0), 23))
                )
            elif case == "wrong_world":
                model = DeformableBoundaryModel(
                    (DeformableBoundarySurface(ph.CollisionKey(999, 101, 0), 23),)
                )
                bindings = ()
            elif case == "moving_static":
                first = bindings[0]
                bindings = (
                    replace(
                        first, motion=ph.CollisionMotion(first.motion.start, first.motion.start + 1)
                    ),
                    *bindings[1:],
                )
            elif case == "missing_binding":
                bindings = bindings[1:]
            else:
                model = DeformableBoundaryModel(
                    tuple(
                        DeformableBoundarySurface(ph.CollisionKey(7, 101, i), 23)
                        for i in range(257)
                    )
                )
            batch.prepare_boundary_query(model, bindings)
