"""M1 source/capacity/publication/lease tests; all initially UNEXECUTED."""

from contextlib import closing, contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import test_device_convex_family_sources as fsrc
import test_device_convex_query as k0
import test_resident_boundary_query as m1

import phymeo as ph


@contextmanager
def scene(mode, path, *, limits=None, budget=m1.BUDGET, two=False):
    worlds = (7, 4096) if two else (7,)
    trees = {w: fsrc.declaration("ordinary", 0) for w in worlds}
    with ph.TyxWorldBatch(list(trees.items()), mode, cache_root=path) as batch:
        shapes, bindings, expected = [], [], {}
        for w, tree in trees.items():
            triangle, points, _ = m1.reference().plain_flat("front", "sphere")
            model, bs, truth = m1.static_scene(
                batch,
                tree,
                world=w,
                triangle=triangle,
                round_points=points,
                translation=(0, 0, 0) if w == 7 else (2**20, -(2**20), 1),
            )
            shapes.extend(model.shapes)
            bindings.extend(bs)
            expected.update(truth)
        model = ph.CollisionModel(tuple(shapes))
        with closing(
            batch.prepare_boundary_query(
                model,
                tuple(bindings),
                limits=limits or ph.BoundaryDistanceLimits(),
                memory_limit_bytes=budget,
            )
        ) as plan:
            yield batch, plan, model, tuple(bindings), expected


def assert_rejected(plan, batch, old, before):
    assert plan.last_result is old
    assert k0.canonical(batch.snapshot()) == k0.canonical(before)
    assert plan.last_attempt.status == "FAILED"
    assert plan.stats()["live_bytes"] == plan.stats()["persistent_bytes"]


@pytest.mark.tyx
@pytest.mark.parametrize(
    "fault",
    (
        "owner",
        "world",
        "epoch",
        "active",
        "layout",
        "registration",
        "topology",
        "point-map",
        "expired",
        "standby",
    ),
)
def test_source_fault_rejects_before_dependent_launch_and_retries(
    fault, tyx_mode, tmp_path, monkeypatch
):
    with scene(tyx_mode, tmp_path) as (batch, plan, model, bindings, expected):
        with m1.saved(
            tmp_path,
            "M10-" + fault,
            tyx_mode,
            inputs=dict(model=model, bindings=bindings),
            plan=plan,
        ) as record:
            old = m1.query(batch, plan, record, 0.5)
            before = batch.snapshot()
            borrow = plan._borrow_source

            @contextmanager
            def corrupted():
                with borrow() as view:
                    accepted = view.accepted
                    if fault == "owner":
                        view = replace(view, owner=object())
                    elif fault == "world":
                        view = replace(view, world_keys=(999,))
                    elif fault == "epoch":
                        view = replace(
                            view, accepted=replace(accepted, state_epoch=accepted.state_epoch + 1)
                        )
                    elif fault == "active":
                        view = replace(view, accepted=replace(accepted, active=1 - accepted.active))
                    elif fault == "layout":
                        view = replace(view, world_layouts=(3,))
                    elif fault == "registration":
                        view = replace(view, registration=object())
                    elif fault == "topology":
                        wrong = list(view.bindings)
                        wrong[-1] = replace(wrong[-1], topology=("unrelated-physical-model",))
                        view = replace(view, bindings=tuple(wrong))
                    elif fault == "point-map":
                        shapes = list(view.model.shapes)
                        shapes[0] = replace(shapes[0], vertex_ids=(13, 37, 91))
                        view = replace(view, model=ph.CollisionModel(tuple(shapes)))
                    elif fault == "expired":
                        view._lease.active = False
                    else:
                        sources = list(view.sources)
                        sources[0] = replace(sources[0], active=1 - sources[0].active)
                        view = replace(view, sources=tuple(sources))
                    yield view

            monkeypatch.setattr(plan, "_borrow_source", corrupted)
            launches = plan.stats()["launches"]
            with pytest.raises((ValueError, RuntimeError)):
                m1.query(batch, plan, record, 0.5)
            assert plan.stats()["launches"] == launches
            assert_rejected(plan, batch, old, before)
            monkeypatch.setattr(plan, "_borrow_source", borrow)
            retried = m1.query(batch, plan, record, 0.5)
            m1.assert_packet(batch, plan, retried, model, expected)


@pytest.mark.tyx
def test_selected_reset_restore_and_next_step_stale_binding(tyx_mode, tmp_path):
    trees = {
        w: fsrc.declaration(layout, i)
        for i, (w, layout) in enumerate(zip(m1.WORLDS, m1.LAYOUTS, strict=True))
    }
    layouts = dict(zip(m1.WORLDS, m1.LAYOUTS, strict=True))
    with ph.TyxWorldBatch(list(trees.items()), tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, data = m1.moving_scene(batch, trees, layouts)
        with closing(
            batch.prepare_boundary_query(
                model, bindings, limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=m1.BUDGET
            )
        ) as plan:
            with m1.saved(
                tmp_path, "M11-reset-restore", tyx_mode, inputs=data, plan=plan
            ) as record:
                assert all(batch.step(m1.H).values())
                checkpoint = batch.snapshot()
                old = m1.query(batch, plan, record, 0.5)
                batch.reset([17])
                changed = batch.snapshot()
                for w in set(trees) - {17}:
                    assert k0.canonical(changed[w]) == k0.canonical(checkpoint[w])
                with pytest.raises(ValueError):
                    plan.require_current(old)
                newer = m1.query(batch, plan, record, 0.5)
                m1.assert_packet(batch, plan, newer, model, m1.expected_scene(changed, data))
                batch.restore(checkpoint)
                with pytest.raises(ValueError):
                    plan.require_current(newer)
                restored = m1.query(batch, plan, record, 0.5)
                m1.assert_packet(
                    batch, plan, restored, model, m1.expected_scene(batch.snapshot(), data)
                )
                assert all(batch.step(m1.H).values())
                with pytest.raises(ValueError):
                    plan.require_current(restored)
                final = m1.query(batch, plan, record, 0.5)
                m1.assert_packet(
                    batch, plan, final, model, m1.expected_scene(batch.snapshot(), data)
                )


@pytest.mark.tyx
@pytest.mark.parametrize(
    "resource", ("groups", "candidates", "vertices", "faces", "segments", "memory")
)
def test_exact_and_one_less_capacity(resource, tyx_mode, tmp_path):
    expected = m1.reference().memory_layout(4, 8, ((1, 1), (1, 1)), (1, 1))
    exact = ph.BoundaryDistanceLimits(groups=2, candidates=2, vertices=3, faces=1, segments=1)
    budget = expected["peak_required_bytes"]
    with scene(tyx_mode, tmp_path, limits=exact, budget=budget, two=True) as (
        batch,
        plan,
        model,
        bindings,
        truth,
    ):
        with m1.saved(
            tmp_path,
            "M12-" + resource,
            tyx_mode,
            inputs=dict(limits=exact, formula=expected),
            plan=plan,
        ) as record:
            old = m1.query(batch, plan, record, 0.5)
            m1.assert_packet(batch, plan, old, model, truth)
            for key in ("persistent_bytes", "prepass_required_bytes", "peak_required_bytes"):
                assert plan.stats()[key] == expected[key]
            assert plan.stats()["peak_owned_bytes"] == budget
            smaller = (
                replace(exact, **{resource: getattr(exact, resource) - 1})
                if resource != "memory"
                else exact
            )
            small_budget = budget - 1 if resource == "memory" else budget
            with closing(
                batch.prepare_boundary_query(
                    model, bindings, limits=smaller, memory_limit_bytes=small_budget
                )
            ) as small:
                before = batch.snapshot()
                with pytest.raises((ph.BoundaryDistanceCapacityError, MemoryError)):
                    m1.query(batch, small, record, 0.5)
                assert_rejected(small, batch, None, before)
                assert small.stats()["peak_owned_bytes"] <= expected["prepass_required_bytes"]
                assert plan.last_result is old
                assert small.last_attempt.required_capacities
                assert all(
                    not label.endswith("/narrow")
                    for label in small.last_attempt.counters["launch_labels"]
                )
                record["smaller"] = dict(attempt=small.last_attempt, stats=small.stats())


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("bvh-read", "narrow-read", "source-stamp", "packet"))
def test_last_group_failure_has_no_partial_publication(fault, tyx_mode, tmp_path, monkeypatch):
    with scene(tyx_mode, tmp_path, two=True) as (batch, plan, model, bindings, expected):
        with m1.saved(
            tmp_path,
            "M13-" + fault,
            tyx_mode,
            inputs=dict(model=model, bindings=bindings),
            plan=plan,
        ) as record:
            old = m1.query(batch, plan, record, 0.5)
            before = batch.snapshot()
            real_read, real_build = plan._download, plan._build_packet
            fired = []

            def read(array, label):
                result = real_read(array, label)
                if label == "group:1/bvh" and fault == "bvh-read":
                    fired.append(label)
                    raise RuntimeError("last BVH read")
                if label == "group:1/meta" and fault == "narrow-read":
                    fired.append(label)
                    raise RuntimeError("last narrow read")
                if label == "source/stamps" and fault == "source-stamp":
                    fired.append(label)
                    result[-1, 1] += 1
                return result

            def build(*args, **kwargs):
                real_build(*args, **kwargs)
                fired.append("packet")
                raise RuntimeError("immutable packet construction")

            if fault == "packet":
                monkeypatch.setattr(plan, "_build_packet", build)
            else:
                monkeypatch.setattr(plan, "_download", read)
            with pytest.raises((ValueError, RuntimeError)):
                m1.query(batch, plan, record, 0.5)
            assert fired
            assert_rejected(plan, batch, old, before)
            assert plan.last_attempt.raw_prefix
            monkeypatch.setattr(plan, "_download", real_read)
            monkeypatch.setattr(plan, "_build_packet", real_build)
            new = m1.query(batch, plan, record, 0.5)
            assert new.sequence == old.sequence + 1
            m1.assert_packet(batch, plan, new, model, expected)


@pytest.mark.tyx
def test_empty_filtered_and_zero_candidate_have_honest_bytes(tyx_mode, tmp_path):
    with ph.TyxWorldBatch((), tyx_mode, cache_root=tmp_path) as batch:
        with pytest.raises(ValueError):
            batch.prepare_boundary_query(
                ph.CollisionModel(()), (), limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=0
            )
        with closing(
            batch.prepare_boundary_query(
                ph.CollisionModel(()), (), limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=1
            )
        ) as plan:
            with m1.saved(tmp_path, "empty-model", tyx_mode, plan=plan) as record:
                value = m1.query(batch, plan, record, 0.0)
                assert value.groups == value.pairs == ()
                assert value.required_candidates == 0
                assert plan.stats()["live_bytes"] == plan.stats()["peak_owned_bytes"] == 0
                assert plan.stats()["allocations"] == plan.stats()["launches"] == 0
                assert not batch._boundary_runtime_leases
    with scene(tyx_mode, tmp_path / "nonempty") as (batch, plan, model, bindings, _):
        excluded = ph.CollisionModel(
            model.shapes, excluded=((model.shapes[0].key, model.shapes[1].key),)
        )
        required = m1.reference().memory_layout(2, 4, (), ())
        with closing(
            batch.prepare_boundary_query(
                excluded,
                bindings,
                limits=ph.BoundaryDistanceLimits(),
                memory_limit_bytes=required["peak_required_bytes"],
            )
        ) as p:
            value = p.query(margin=0.5)
            assert not value.groups and value.required_candidates == 0
            assert value.pairs[0].reason == "excluded"
            assert p.stats()["peak_owned_bytes"] == required["peak_required_bytes"]
        value = plan.query(margin=0.0)
        assert len(value.groups) == 1 and value.required_candidates == 0
        assert len(value.groups[0].packet.rows) == 0


@pytest.mark.tyx
@pytest.mark.parametrize("operation", ("step", "restore", "controls", "prepare", "close"))
def test_actual_source_loan_blocks_callback_mutation(operation, tyx_mode, tmp_path, monkeypatch):
    with scene(tyx_mode, tmp_path) as (batch, plan, model, bindings, expected):
        old = plan.query(margin=0.5)
        before = batch.snapshot()
        actions = dict(
            step=lambda: batch.step(m1.H),
            restore=lambda: batch.restore(before),
            controls=lambda: batch.set_controls({}),
            prepare=lambda: batch.prepare_boundary_query(
                model, bindings, limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=m1.BUDGET
            ),
            close=batch.close,
        )
        original = plan._download
        called = []

        def callback(array, label):
            if not called:
                called.append(label)
                with pytest.raises(RuntimeError, match="boundary source loan"):
                    actions[operation]()
                if operation == "close":
                    with pytest.raises(RuntimeError, match="boundary source loan"):
                        plan.close()
            return original(array, label)

        monkeypatch.setattr(plan, "_download", callback)
        with m1.saved(tmp_path, "reentry-" + operation, tyx_mode, plan=plan) as record:
            new = m1.query(batch, plan, record, 0.5)
            assert called and new is not old
            m1.assert_packet(batch, plan, new, model, expected)
            assert k0.canonical(before) == k0.canonical(batch.snapshot())


@pytest.mark.tyx
def test_complete_filters_and_validated_rigid_installation(tyx_mode, tmp_path):
    tree = fsrc.declaration("ordinary", 0)
    ref = m1.reference()
    tri, point, _ = ref.plain_flat("front", "sphere")
    keys = [
        ph.CollisionKey(7, 10, 0),
        ph.CollisionKey(7, 20, 0),
        ph.CollisionKey(7, 20, 1),
        ph.CollisionKey(7, 30, 0),
        ph.CollisionKey(7, 40, 0),
    ]
    shapes = (
        ph.CollisionShape(keys[0], "triangle", ref.IDS),
        ph.CollisionShape(keys[1], "sphere", (11,), radius=0.125),
        ph.CollisionShape(keys[2], "sphere", (11,), radius=0.125),
        ph.CollisionShape(keys[3], "sphere", (11,), radius=0.125, mask=0),
        ph.CollisionShape(keys[4], "capsule", (11, 37), radius=0.125),
    )
    model = ph.CollisionModel(shapes, excluded=((keys[0], keys[1]),))
    motions = [tri, point, point, point, np.vstack((point, point + (0, 0, 0.25)))]
    with ph.TyxWorldBatch([(7, tree)], tyx_mode, cache_root=tmp_path) as batch:
        state = batch.snapshot()[7]
        pos, R, group = ref.pose(state, "ordinary", "follower")
        bindings = tuple(
            m1.binding(tree, state, "ordinary", key, group, (points - pos) @ R)
            if key.body == 20
            else ph.ResidentStaticBinding(key, ph.CollisionMotion(points))
            for key, points in zip(keys, motions, strict=True)
        )
        with closing(
            batch.prepare_boundary_query(
                model, bindings, limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=m1.BUDGET
            )
        ) as plan:
            with m1.saved(
                tmp_path,
                "M08-filters",
                tyx_mode,
                inputs=dict(model=model, bindings=bindings),
                plan=plan,
            ) as record:
                result = m1.query(batch, plan, record, 0.5)
                m1.assert_packet(batch, plan, result, model, dict(zip(keys, motions, strict=True)))
                assert len(result.pairs) == 10 and result.required_candidates == 2
                reasons = {pair.reason for pair in result.pairs}
                assert {"excluded", "same_rigid_body", "mask", "same_family"} <= reasons
                # Legal mesh + round on one declared rigid body; no fake one-face mesh.
                vertices, faces, center, _ = ref.mesh("shared_pruned")
                ids = tuple(11 + 13 * i for i in range(len(vertices)))
                mesh = ph.CollisionShape(
                    ph.CollisionKey(7, 20, 2),
                    "mesh",
                    ids,
                    triangles=tuple(tuple(ids[i] for i in f) for f in faces),
                )
                same = ph.CollisionModel((shapes[1], mesh))
                bs = (
                    bindings[1],
                    m1.binding(tree, state, "ordinary", mesh.key, group, (vertices - pos) @ R),
                )
                with closing(
                    batch.prepare_boundary_query(
                        same, bs, limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=m1.BUDGET
                    )
                ) as inner:
                    value = m1.query(batch, inner, record, 0.5)
                    assert not value.groups and value.required_candidates == 0
                    assert value.pairs[0].reason == "same_rigid_body"
                with pytest.raises(ValueError, match="rigid"):
                    ph.CollisionModel(
                        (shapes[1], replace(shapes[0], key=ph.CollisionKey(7, 20, 3)))
                    )
                with pytest.raises(ValueError):
                    ph.CollisionShape(
                        ph.CollisionKey(7, 99, 0), "mesh", ref.IDS, triangles=(ref.IDS,)
                    )


@pytest.mark.tyx
@pytest.mark.parametrize(
    "fault", ("wrong-layout", "body-alias", "missing", "foreign-world", "topology")
)
def test_registration_cannot_forge_rigid_owner(fault, tyx_mode, tmp_path):
    with scene(tyx_mode, tmp_path) as (batch, plan, model, bindings, _):
        old = plan.query(margin=0.5)
        changes = list(bindings)
        dynamic = changes[1]
        if fault == "wrong-layout":
            changes[1] = ph.ResidentAffineTreeBinding(
                dynamic.key,
                7,
                dynamic.group_index,
                dynamic.local_points_m,
                topology=dynamic.topology,
            )
        elif fault == "body-alias":
            other = ph.CollisionShape(ph.CollisionKey(7, 303, 0), "sphere", (11,), radius=0.125)
            model = ph.CollisionModel(model.shapes + (other,))
            changes.append(replace(dynamic, key=other.key))
        elif fault == "missing":
            changes = changes[:1]
        elif fault == "foreign-world":
            changes[1] = replace(dynamic, key=ph.CollisionKey(999, 202, 0), world=999)
        else:
            changes[1] = replace(dynamic, topology=("unrelated-physical-model",))
        before = batch.snapshot()
        with pytest.raises((ValueError, TypeError)):
            batch.prepare_boundary_query(
                model,
                tuple(changes),
                limits=ph.BoundaryDistanceLimits(),
                memory_limit_bytes=m1.BUDGET,
            )
        assert plan.last_result is old
        assert k0.canonical(before) == k0.canonical(batch.snapshot())


@pytest.mark.tyx
def test_latest_publication_full_source_binding_and_owner(tyx_mode, tmp_path):
    with scene(tyx_mode, tmp_path / "a") as (batch, plan, model, bindings, _):
        old = plan.query(margin=0.5)
        with closing(
            batch.prepare_boundary_query(
                model, bindings, limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=m1.BUDGET
            )
        ) as other:
            with pytest.raises(ValueError):
                other.require_current(old)
        for field in ("shape_revisions", "point_ids", "world_anchors", "gathered_points"):
            changed = dict(old.source)
            changed[field] = ("unrelated-source",)
            with pytest.raises(ValueError):
                plan.require_current(replace(old, source=changed))
        current = plan.query(margin=0.25)
        with pytest.raises(ValueError):
            plan.require_current(old)
        assert plan.require_current(current) is current
        with pytest.raises(ValueError):
            current.source["stamps"]["poses"].setflags(write=True)


class FakeRuntime:
    def __init__(self, events):
        self.events, self.calls = events, 0

    def close(self):
        self.calls += 1
        self.events.append("runtime")


class FakeArray:
    def __init__(self, name, events, phase=None):
        self.name, self.events, self.phase = name, events, phase
        self.calls, self.released = 0, False

    def free(self):
        self.calls += 1
        self.events.append(self.name)
        if self.phase == "before":
            raise RuntimeError("authored before free")
        assert not self.released
        self.released = True
        if self.phase == "after":
            raise RuntimeError("authored after free")


@pytest.mark.parametrize("phase", ("before", "after"))
@pytest.mark.parametrize("first", ("query", "batch-close"))
def test_host_unknown_retains_real_plan_lease_and_batch(phase, first, monkeypatch, tmp_path):
    """Synthetic buffers, actual M1 ledger/parent close path; not device physics."""
    import gc
    import json
    import weakref

    import tyx as tx

    from phymeo.backends.mujoco_tyx._resources import quarantined_resources

    def exercise():
        with monkeypatch.context() as patch:
            events, buffers = [], []
            batch = ph.TyxWorldBatch((), mode="interpret")
            plan = batch.prepare_boundary_query(
                ph.CollisionModel(()),
                (),
                limits=ph.BoundaryDistanceLimits(),
                memory_limit_bytes=1024,
            )
            batch._runtime = FakeRuntime(events)
            batch._resources.callback(batch._runtime.close)
            batch._resources.callback(events.append, "parent-owned")
            plan._mode = "cuda"
            plan._runtime = batch._runtime
            plan._tx = tx

            def allocate(shape, dtype):
                del shape, dtype
                buffer = FakeArray(
                    "buffer-" + str(len(buffers)), events, phase if len(buffers) == 1 else None
                )
                buffers.append(buffer)
                return buffer

            patch.setattr(tx, "cuda_empty", allocate)
            previous = SimpleNamespace(schema="HOST-protocol-sentinel")
            plan._last_result = previous
            resources = plan._new_query_resources()
            for _ in range(3):
                plan._allocate(resources, shape=(2, 3), dtype=np.float32)
            assert plan.stats()["live_bytes"] == 72
            with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
                plan._close_query_resources()
            assert resources.state == "UNKNOWN_RELEASE" and resources.runtime_pending
            assert plan.stats()["live_bytes"] == 24
            assert resources.failures[0].owner.array is buffers[1]
            assert buffers[1].released is (phase == "after")
            assert all(b.calls == 1 for b in buffers)
            assert buffers[0].released and buffers[2].released
            if first == "query":
                with pytest.raises((RuntimeError, BaseExceptionGroup)):
                    plan.query(margin=0.0)
            else:
                with pytest.raises((RuntimeError, BaseExceptionGroup)):
                    batch.close()
            for operation in (plan.close, batch.close, batch.close):
                with pytest.raises((RuntimeError, BaseExceptionGroup)):
                    operation()
            assert batch._runtime.calls == 0 and "parent-owned" not in events
            assert plan.last_result is previous
            assert resources in quarantined_resources()
            assert resources._runtime.owner.batch is batch
            assert resources._runtime.owner.runtime is batch._runtime
            assert all(b.calls == 1 for b in buffers)
            refs = tuple(weakref.ref(x) for x in (batch, batch._runtime, buffers[1], resources))
            record = dict(
                phase=phase,
                first=first,
                events=list(events),
                unknown_bytes=24,
                buffer_calls=[b.calls for b in buffers],
                ledger=resources.state,
                runtime_calls=batch._runtime.calls,
                old_publication_retained=True,
            )
            return refs, record

    refs, record = exercise()
    gc.collect()
    assert all(ref() is not None for ref in refs)
    (tmp_path / "M15-unknown.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.cuda
def test_real_cuda_free_then_error_quarantines_shared_runtime(monkeypatch, tmp_path):
    from phymeo.backends.mujoco_tyx._resources import DeviceResources, quarantined_resources

    # Do not use a context manager that would replace the original release error.
    tree = fsrc.declaration("ordinary", 0)
    batch = ph.TyxWorldBatch([(7, tree)], "cuda", cache_root=tmp_path)
    tri, points, _ = m1.reference().plain_flat("front", "sphere")
    model, bindings, _ = m1.static_scene(batch, tree, triangle=tri, round_points=points)
    plan = batch.prepare_boundary_query(
        model, bindings, limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=m1.BUDGET
    )
    runtime = batch._runtime
    captured, selected = [], []
    state = dict(calls=0, completed=False)
    real_add = DeviceResources.add_buffer
    old = plan.query(margin=0.5)

    def capture(resources, owner):
        if resources is plan._query_resources:
            captured.append(owner)
            if len(captured) == 4 and not selected:
                selected.append(owner)
                real_free = owner.array.free

                def late():
                    state["calls"] += 1
                    real_free()
                    state["completed"] = True
                    raise RuntimeError("authored after actual M1 CUDA free")

                monkeypatch.setattr(owner.array, "free", late)
        return real_add(resources, owner)

    monkeypatch.setattr(DeviceResources, "add_buffer", capture)
    try:
        with m1.saved(
            tmp_path,
            "M16-cuda-unknown",
            "cuda",
            inputs=dict(model=model, bindings=bindings),
            plan=plan,
        ) as record:
            with pytest.raises(BaseExceptionGroup, match="UNKNOWN_RELEASE"):
                plan.query(margin=0.5)
            ledger = plan._query_resources
            assert ledger in quarantined_resources() and ledger.state == "UNKNOWN_RELEASE"
            assert state == dict(calls=1, completed=True)
            assert plan.last_result is old
            assert ledger.failures[0].owner is selected[0]
            assert all(x.released for x in captured if x is not selected[0])
            assert not selected[0].released
            assert plan.stats()["live_bytes"] == plan.stats()["persistent_bytes"] + selected[0].size
            assert not runtime.closed
            for operation in (lambda: plan.query(margin=0.5), plan.close, batch.close):
                with pytest.raises((RuntimeError, BaseExceptionGroup)):
                    operation()
            assert state["calls"] == 1 and not runtime.closed
            record["release"] = dict(
                state=state,
                ledger=ledger.state,
                unknown_size=selected[0].size,
                siblings=len(captured) - 1,
                actual_live=plan.stats()["live_bytes"],
            )
    finally:
        # No direct runtime close: other real resident buffers still depend on it.
        # The intentionally unknown source remains strongly quarantined until process exit.
        try:
            batch.close()
        except (RuntimeError, BaseExceptionGroup):
            pass


@pytest.mark.tyx
@pytest.mark.parametrize("stage", ("prepass", "narrow"))
def test_allocation_failure_peak_is_actual_prefix_and_retry(stage, tyx_mode, tmp_path, monkeypatch):
    with scene(tyx_mode, tmp_path, two=True) as (batch, plan, model, bindings, expected):
        with m1.saved(
            tmp_path,
            "allocation-prefix-" + stage,
            tyx_mode,
            inputs=dict(model=model, bindings=bindings),
            plan=plan,
        ) as record:
            old = m1.query(batch, plan, record, 0.5)
            original = plan._allocate
            before = batch.snapshot()
            live_prefix = []
            failed = []

            def allocate(resources, *, shape, dtype):
                if resources is plan._query_resources:
                    live_prefix.append(plan.stats()["live_bytes"])
                    is_narrow = tuple(shape) == (1, 5, 3)
                    if (stage == "prepass" and len(live_prefix) == 2) or (
                        stage == "narrow" and is_narrow
                    ):
                        failed.append((tuple(shape), np.dtype(dtype).str))
                        raise MemoryError("authored allocation prefix")
                return original(resources, shape=shape, dtype=dtype)

            monkeypatch.setattr(plan, "_allocate", allocate)
            with pytest.raises(MemoryError):
                m1.query(batch, plan, record, 0.5)
            assert failed and len(failed) == 1
            assert_rejected(plan, batch, old, before)
            assert max(live_prefix) <= plan.stats()["prepass_required_bytes"]
            assert all(
                not x.endswith("/narrow") for x in plan.last_attempt.counters["launch_labels"]
            )
            record["actual_prefix"] = dict(
                live_bytes=live_prefix, failed=failed, attempt=plan.last_attempt
            )
            monkeypatch.setattr(plan, "_allocate", original)
            new = m1.query(batch, plan, record, 0.5)
            m1.assert_packet(batch, plan, new, model, expected)


@pytest.mark.tyx
@pytest.mark.parametrize("entry", ("query-cleanup", "direct-close"))
def test_pending_cleanup_retains_handle_and_blocks_reuse(entry, tyx_mode, tmp_path, monkeypatch):
    """A failure before ledger.close retains a retryable owner, not an unknown free."""
    with scene(tyx_mode, tmp_path) as (batch, plan, model, bindings, expected):
        with m1.saved(
            tmp_path,
            "pending-" + entry,
            tyx_mode,
            inputs=dict(model=model, bindings=bindings),
            plan=plan,
        ) as record:
            old = m1.query(batch, plan, record, 0.5)
            before = batch.snapshot()
            identity_before = (batch._state_epoch, batch._active)
            record["entry_before_pending"] = before
            calls = []

            def pending():
                calls.append("before-ledger-entry")
                raise OSError("authored pending cleanup")

            if entry == "query-cleanup":
                monkeypatch.setattr(plan, "_close_query_resources", pending)
                with pytest.raises(OSError, match="pending cleanup"):
                    plan.query(margin=0.5)
                retained = plan._query_resources
            else:
                retained = plan._persistent_resources
                original_close = retained.close
                monkeypatch.setattr(retained, "close", pending)
                with pytest.raises(BaseExceptionGroup, match="pending"):
                    plan.close()
            assert retained.state == "OPEN" and retained.runtime_pending
            assert plan.last_result is old
            launches = plan.stats()["launches"]
            for operation in (lambda: plan.query(margin=0.5), lambda: batch.step(m1.H)):
                with pytest.raises(RuntimeError, match="pending"):
                    operation()
            assert plan.stats()["launches"] == launches
            assert (batch._state_epoch, batch._active) == identity_before
            with pytest.raises(RuntimeError, match="pending"):
                batch.snapshot()
            assert calls == ["before-ledger-entry"]
            if entry == "direct-close":
                monkeypatch.setattr(retained, "close", original_close)
            plan.close()
            assert retained.state == "CLOSED" and not retained.runtime_pending
            assert not batch._boundary_runtime_leases
            assert plan.stats()["live_bytes"] == 0
            # A pending parent remains close-only until its complete stack closes.
            with pytest.raises(RuntimeError, match="pending"):
                batch.step(m1.H)
            batch.close()
            assert batch._closed
            record["pending_cleanup"] = dict(
                entry=entry,
                calls=calls,
                ledger=retained.state,
                runtime_leases=0,
                live_bytes=0,
                old_publication_retained=plan.last_result is old,
            )
