"""World-local pair allocation, independent publications and failure isolation."""

from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest
from cases import BASE_X_M, HOLDOUT_X_M, RADIUS_M, SCALES, PhymeoGroup, permuted, world_ids

import phymeo as ph


def workspace(worlds, backend="host", **options):
    assert hasattr(ph, "WorldCollisionWorkspace"), "public partitioned collision workspace is missing"
    return ph.WorldCollisionWorkspace(worlds, backend, **options)


def one_world(world, xs=BASE_X_M):
    shapes, paths = [], {}
    for local in reversed(range(len(xs))):
        key = ph.CollisionKey(world, local, 0)
        shapes.append(ph.CollisionShape(key, "sphere", (0,), RADIUS_M))
        paths[key] = ph.CollisionMotion([[xs[local], 0, 0]])
    return ph.CollisionModel(shapes), paths


def register(groups):
    return ([(w, m) for w, (m, p) in groups.items()], {w: p for w, (m, p) in groups.items()},
            {w: ph.CollisionPolicy() for w in groups})


def run_g1(count, backend, tmp_path, monkeypatch):
    from phymeo.geometry import collision_pipeline as pipeline

    ids = world_ids(count)
    models = {w: one_world(w) for w in permuted(ids)}
    bindings, motion, policies = register(models)
    original = pipeline.combinations

    def guarded(indices, size):
        assert len(indices) <= 4, "constructed a temporary all-world pair enumeration"
        return original(indices, size)

    monkeypatch.setattr(pipeline, "combinations", guarded)
    with ExitStack() as stack:
        single = {w: PhymeoGroup((w,), "host", tmp_path) for w in ids}
        for api in single.values():
            stack.callback(api.close)
        batch = stack.enter_context(workspace(bindings, backend, cache_root=tmp_path / "cache"))
        moved = ids[count // 2]
        base_ref = {}
        for changed in (None, moved, None):
            paths = dict(motion)
            if changed is not None:
                paths[changed] = {**paths[changed], ph.CollisionKey(changed, 1, 0):
                                  ph.CollisionMotion([[HOLDOUT_X_M, 0, 0]])}
            before = batch.stats()
            out = batch.detect(paths, policies)
            assert tuple(out) == tuple(models)
            for w in ids:
                single[w].select(changed)
                ref = single[w].query()
                single[w].inspect(ref, changed)
                actual = out[w]
                assert replace(actual, backend="host") == ref
                assert batch.last_result(w) is actual
                if changed is None and w in base_ref:
                    assert actual == base_ref[w]
                elif not base_ref.get(w):
                    base_ref[w] = actual
            after = batch.stats()
            if backend == "cuda":
                assert after["completed_launches"] - before["completed_launches"] == 2
                assert after["synchronizations"] - before["synchronizations"] == 1
                assert after["downloaded_bytes"] - before.get("downloaded_bytes", 0) == 24*4*count + 4*6*count
        # Check actual buffers as well as the public diagnostic.
        pairs = batch._work._pairs
        assert len(pairs) == after["host_pair_table_slots"] == 6 * count
        assert pairs.nbytes == after["host_pair_table_bytes"] == 12 * 6 * count
        shapes = batch._work._model.shapes
        assert all(shapes[i].key.world == shapes[j].key.world for i, j, _ in pairs)
        if backend == "cuda":
            assert batch._work._engine._pairs.shape == (6*count, 3)
            assert batch._work._engine._flags.shape == (6*count,)
            assert after["device_pair_table_slots"] == 6*count


@pytest.mark.parametrize("count", SCALES)
def test_host_g1_same_public_rows_without_cross_world_slots(count, tmp_path, monkeypatch):
    run_g1(count, "host", tmp_path, monkeypatch)


@pytest.mark.tyx
@pytest.mark.parametrize("count", [2, 64])
def test_device_g1_queries_share_one_engine(count, tyx_mode, tmp_path, monkeypatch):
    run_g1(count, tyx_mode, tmp_path, monkeypatch)


def test_world_capacity_commits_neighbor_sweep_and_allows_retry():
    a, b = 14, 3
    scenes = {a: one_world(a, BASE_X_M[:2]), b: one_world(b, BASE_X_M[:2])}
    bindings, motion, policies = register(scenes)
    ball = ph.CollisionKey(b, 1, 0)
    travel = {**motion[b], ball: ph.CollisionMotion([[3/16, 0, 0]], [[5/16, 0, 0]])}
    trial = {a: motion[a], b: travel}
    with workspace(bindings) as batch, ph.CollisionWorkspace(scenes[b][0]) as standalone:
        reference = ph.detect(scenes[b][0], travel, policies[b], standalone)
        assert reference.pairs[0].sweep.first_t == (0.0, 0.0)
        assert reference.pairs[0].witness.gap_m == pytest.approx(1/16)
        first = batch.detect(motion, policies)
        capacity = {**policies, a: ph.CollisionPolicy(candidate_capacity=0)}
        attempts = batch.detect(trial, capacity)
        assert attempts[a].status == "capacity_exceeded" and attempts[a].required_candidates == 1
        assert not attempts[a].pairs and not attempts[a].bounds
        assert batch.last_result(a) is first[a]
        assert batch.last_result(b) is attempts[b] and attempts[b] == reference
        restored = batch.detect(trial, policies)
        assert restored[a] == first[a] and restored[b] == reference


def test_per_world_margin_masks_and_empty_and_single_shape_models():
    with workspace([]) as empty_batch:
        assert empty_batch.detect({}, {}) == {} and empty_batch.stats()["host_pair_table_slots"] == 0
    ids = (70, 3, 103, 8)
    empty, single, separated, filtered = ids
    groups = {empty: one_world(empty, ()), single: one_world(single, (1,)),
              separated: one_world(separated, (0, HOLDOUT_X_M)), filtered: one_world(filtered, BASE_X_M[:2])}
    model, paths = groups[filtered]
    shapes = (replace(model.shapes[0], mask=0), model.shapes[1])
    groups[filtered] = (ph.CollisionModel(shapes), paths)
    bindings, motion, policies = register(groups)
    policies[separated] = ph.CollisionPolicy(margin=0.125)
    with workspace(bindings) as batch:
        out = batch.detect(motion, policies)
        assert list(out) == list(ids)
        for w, (model, paths) in groups.items():
            with ph.CollisionWorkspace(model) as prev:
                assert out[w] == ph.detect(model, paths, policies[w], prev)
        assert out[separated].pairs[0].witness.gap_m == pytest.approx(1/16)
        assert out[filtered].pairs[0].reason == "mask"
        assert len(out[single].bounds) == 1 and not out[empty].bounds


def test_invalid_registration_and_requests_are_rejected_before_device_or_publication(monkeypatch):
    ids = (3, 8)
    bindings, motion, policies = register({w: one_world(w) for w in ids})
    for bad in ((bindings[0], bindings[0]), ((7, bindings[0][1]),), ((True, ph.CollisionModel(())),)):
        with pytest.raises((TypeError, ValueError)):
            workspace(bad)
    with workspace(bindings) as batch:
        def forbidden(*args, **kwargs):
            pytest.fail("invalid input opened or dispatched a query engine")
        monkeypatch.setattr(ph.CollisionWorkspace, "_from_world_pairs", forbidden)
        for paths, rules in (({3: motion[3]}, policies), (motion, {3: policies[3]}),
                             ({3: motion[8], 8: motion[3]}, policies),
                             (motion, {3: None, 8: policies[8]})):
            with pytest.raises((TypeError, ValueError)):
                batch.detect(paths, rules)
        assert batch.last_result(3) is None and batch.last_result(8) is None


@pytest.mark.tyx
@pytest.mark.parametrize("failure", ["launch", "download"])
def test_batch_fault_keeps_old_observations_and_retries(tyx_mode, tmp_path, monkeypatch, failure):
    bindings, motion, policies = register({w: one_world(w, BASE_X_M[:2]) for w in (3, 8)})
    with workspace(bindings, tyx_mode, cache_root=tmp_path / "cache") as batch:
        old = batch.detect(motion, policies)
        engine = batch._work._engine
        where = "_classify" if failure == "launch" else "_read"

        def broken(*args, **kwargs):
            raise RuntimeError(f"injected {failure}")
        monkeypatch.setattr(engine, where, broken)
        changed = {3: motion[3], 8: one_world(8, (0, HOLDOUT_X_M))[1]}
        with pytest.raises(RuntimeError, match=f"injected {failure}"):
            batch.detect(changed, policies)
        assert batch.last_result(3) is old[3] and batch.last_result(8) is old[8]
        resumed = batch.detect(changed, policies)
        assert batch._work._engine is not engine
        assert resumed[3] == old[3] and resumed[8].required_candidates == 0
