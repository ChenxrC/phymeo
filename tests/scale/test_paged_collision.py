"""Whole-world page packing, same public geometry and cross-page transactions."""

import json
from importlib import util

import numpy as np
import pytest
from g4_geometry_cases import WORLD_IDS, check_geometry, copies, digest, frozen

import phymeo as ph


def api():
    assert util.find_spec("phymeo.geometry.paged_collision") is not None, (
        "missing paged collision workspace"
    )
    from phymeo.geometry import paged_collision

    return paged_collision


def four(w):
    shapes = []
    paths = {}
    for i, x in enumerate((0.0, 3 / 16, 2.0, 4.0)):
        key = ph.CollisionKey(w, i, 0)
        shapes.append(ph.CollisionShape(key, "sphere", (0,), 1 / 8))
        paths[key] = ph.CollisionMotion([[x, 0.0, 0.0]])
    return ph.CollisionModel(shapes), paths


@pytest.mark.parametrize("backend", ["host", "cuda"])
@pytest.mark.parametrize(
    "name,expected,split", [("sparse65", 0, (195, 65)), ("sheet72", 4, (219, 73))]
)
def test_crossing_old_shape_cap_returns_every_g4_world(name, expected, split, backend, tmp_path):
    m = api()
    model, old = frozen(name)
    worlds, frames = copies(model, old, WORLD_IDS)
    rule = dict.fromkeys(WORLD_IDS, ph.CollisionPolicy())
    with pytest.raises(ValueError, match="aggregate"):
        ph.WorldCollisionWorkspace(worlds, backend)
    with (
        m.PagedWorldCollisionWorkspace(worlds, backend, cache_root=tmp_path) as batch,
        ph.WorldCollisionWorkspace(worlds[:1], backend, cache_root=tmp_path) as single,
    ):
        assert tuple(p.shapes for p in batch.layout) == split
        assert tuple(p.world_keys for p in batch.layout) == (WORLD_IDS[:3], WORLD_IDS[3:])
        for frame in frames:
            ref = single.detect(
                {WORLD_IDS[0]: frame[WORLD_IDS[0]]}, {WORLD_IDS[0]: rule[WORLD_IDS[0]]}
            )[WORLD_IDS[0]]
            fingerprint = digest(ref, WORLD_IDS[0])
            found = batch.detect(frame, rule)
            assert tuple(found) == WORLD_IDS
            for (key, model), out in zip(worlds, found.values(), strict=True):
                assert batch.last_result(key) is out
                assert digest(out, key) == fingerprint
                audit = check_geometry(name, model, frame[key], out, expected)
                assert audit["hits"] == (0 if name == "sparse65" else 3)
        s = batch.stats()
        assert [p["dispatches"] for p in s["pages"]] == [len(frames)] * 2
        assert s["peak_resident_pages"] == 2
        for p in s["pages"]:
            assert p["actual_broad_device_bytes"] == (
                p["planned_broad_bytes"] if backend == "cuda" else 0
            )
            assert p["device_pair_table_slots"] == (p["pair_slots"] if backend == "cuda" else 0)


def cube_count(keys):
    groups = {k: four(k) for k in keys}
    return [(k, m) for k, (m, _) in groups.items()], {k: p for k, (_, p) in groups.items()}


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_128_overlaid_worlds_keep_local_ids_and_negative_control(backend, tmp_path):
    m = api()
    keys = tuple(1000 + 7 * i for i in range(128))
    worlds, frames = cube_count(keys)
    worlds.insert(1, (1, ph.CollisionModel(())))
    worlds.append((4, ph.CollisionModel(())))
    frames[1] = {}
    frames[4] = {}
    rules = dict.fromkeys(frames, ph.CollisionPolicy())
    changed = keys[76]
    frames[changed] = dict(frames[changed])
    frames[changed][ph.CollisionKey(changed, 1, 0)] = ph.CollisionMotion([[5 / 16, 0, 0]])
    with m.PagedWorldCollisionWorkspace(worlds, backend, cache_root=tmp_path) as batch:
        assert [p.shapes for p in batch.layout] == [256, 256]
        out = batch.detect(frames, rules)
        assert tuple(out) == tuple(k for k, _ in worlds)
        assert sum(len(x.pairs) for x in out.values()) == 768
        assert sum(sum(v.status == "overlap" for v in x.pairs) for x in out.values()) == 127
        assert not out[changed].required_candidates and not out[1].pairs and not out[4].bounds
        for key in keys:
            if key != changed:
                (hit,) = [p for p in out[key].pairs if p.status == "overlap"]
                assert hit.first.world == hit.second.world == key
                assert (hit.first.body, hit.second.body) == (0, 1)
                assert hit.witness.gap_m == -1 / 16
                np.testing.assert_array_equal(hit.witness.first_point, (1 / 8, 0, 0))
                np.testing.assert_array_equal(hit.witness.second_point, (1 / 16, 0, 0))
        assert batch.stats()["resident_device_pair_slots"] == (768 if backend == "cuda" else 0)


def test_shape_vertex_pair_and_publication_budgets_before_any_page_allocates(monkeypatch):
    m = api()

    def forbidden(*args, **kwargs):
        raise AssertionError("page constructor ran during packing or budget refusal")

    monkeypatch.setattr(m, "WorldCollisionWorkspace", forbidden)
    worlds, _ = cube_count((18, 3, 41))
    # A four-sphere world needs six local slots. Facet count and vertex budgets
    # also act independently when their public values are set below shape bounds.
    for limits, resource, required, cap in (
        (m.CollisionPageLimits(shapes=3), "shapes", 4, 3),
        (m.CollisionPageLimits(vertices=3), "vertices", 4, 3),
        (m.CollisionPageLimits(pairs=5), "pairs", 6, 5),
        (m.CollisionPageLimits(result_pairs=17), "result_pairs", 18, 17),
        (m.CollisionPageLimits(resident_broad_bytes=403), "resident_broad_bytes", 404, 403),
    ):
        with pytest.raises(m.CollisionPageCapacityError) as caught:
            m.PagedWorldCollisionWorkspace(worlds, limits=limits)
        assert (caught.value.resource, caught.value.required, caught.value.limit) == (
            resource,
            required,
            cap,
        )

    def too_many():
        yield worlds[0]
        yield worlds[1]
        raise AssertionError("enumerated worlds after the complete-result budget was exceeded")

    with pytest.raises(m.CollisionPageCapacityError) as early:
        m.PagedWorldCollisionWorkspace(too_many(), limits=m.CollisionPageLimits(result_pairs=6))
    assert early.value.required == 12 and early.value.limit == 6
    for limits in (
        m.CollisionPageLimits(pairs=7),
        m.CollisionPageLimits(vertices=7),
        m.CollisionPageLimits(resident_broad_bytes=500),
    ):
        with m.PagedWorldCollisionWorkspace(worlds, limits=limits) as work:
            assert [p.world_keys for p in work.layout] == [(18,), (3,), (41,)]
    with pytest.raises(ValueError):
        m.CollisionPageLimits(shapes=257)
    with pytest.raises(ValueError):
        m.CollisionPageLimits(vertices=8193)
    with pytest.raises(ValueError):
        m.CollisionPageLimits(resident_pages=0)
    with pytest.raises(ValueError):
        m.PagedWorldCollisionWorkspace([worlds[0], worlds[0]])


@pytest.mark.parametrize("backend", ["host", "cuda"])
@pytest.mark.parametrize("by_bytes", [False, True])
def test_streaming_page_eviction_preserves_accepted_worlds_and_early_validation(
    backend, by_bytes, tmp_path
):
    m = api()
    keys = (27, 3, 118, 5)
    worlds, motion = cube_count(keys)
    limits = m.CollisionPageLimits(
        shapes=8,
        resident_pages=2 if by_bytes else 1,
        resident_broad_bytes=900 if by_bytes else None,
    )
    rules = dict.fromkeys(keys, ph.CollisionPolicy())
    with m.PagedWorldCollisionWorkspace(
        worlds, backend, limits=limits, cache_root=tmp_path
    ) as batch:
        assert [p.world_keys for p in batch.layout] == [(27, 3), (118, 5)]
        good = batch.detect(motion, rules)
        before = batch.stats()
        assert before["peak_resident_pages"] == 1 and before["evictions"] == 1
        assert [p["resident"] for p in before["pages"]] == [False, True]
        with pytest.raises(ValueError):
            batch.detect({**motion, 5: {}}, rules)
        assert batch.stats()["page_dispatches"] == before["page_dispatches"]
        # Reject the first world after its inner page was evicted. The neighbor
        # in the other page uses an actual held-out no-contact configuration.
        trial = dict(motion)
        trial[5] = dict(motion[5])
        kk = ph.CollisionKey(5, 1, 0)
        trial[5][kk] = ph.CollisionMotion([[5 / 16, 0, 0]])
        limited = dict(rules)
        limited[27] = ph.CollisionPolicy(candidate_capacity=0)
        result = batch.detect(trial, limited)
        assert result[27].status == "capacity_exceeded" and result[27].required_candidates == 1
        assert batch.last_result(27) is good[27] and not result[27].pairs
        assert result[5].required_candidates == 0 and batch.last_result(5) is result[5]
        assert result[3] == good[3] and result[118] == good[118]
        again = batch.detect(trial, rules)
        assert again[27] == good[27] and again[5] == result[5]
        stats = batch.stats()
        assert (
            stats["page_builds"] == 6
            and stats["evictions"] == 5
            and stats["peak_resident_pages"] == 1
        )
        assert stats["peak_broad_device_bytes"] == (
            (4 + 52 * 8 + 24 * 8 + 16 * 12) if backend == "cuda" else 0
        )
        assert [p["dispatches"] for p in stats["pages"]] == [3, 3]
        (tmp_path / "page-eviction.json").write_text(
            json.dumps(
                dict(
                    by_bytes=by_bytes,
                    stats=stats,
                    held=27,
                    required=result[27].required_candidates,
                    neighbor=5,
                    neighbor_candidates=result[5].required_candidates,
                ),
                indent=2,
            ),
            encoding="utf8",
        )


def test_real_cuda_fault_on_second_page_rolls_back_first_and_rebuilds(tmp_path, monkeypatch):
    m = api()
    keys = (7, 19, 5, 48)
    worlds, paths = cube_count(keys)
    rules = dict.fromkeys(keys, ph.CollisionPolicy())
    with m.PagedWorldCollisionWorkspace(
        worlds,
        "cuda",
        limits=m.CollisionPageLimits(shapes=8, resident_pages=2),
        cache_root=tmp_path,
    ) as batch:
        old = batch.detect(paths, rules)
        changed = dict(paths)
        changed[7] = dict(paths[7])
        changed[7][ph.CollisionKey(7, 1, 0)] = ph.CollisionMotion([[5 / 16, 0, 0]])
        engine = batch._resident[1]._work._engine

        def failed(*args):
            raise RuntimeError("injected second-page device download")

        monkeypatch.setattr(engine, "_read", failed)
        with pytest.raises(RuntimeError, match="second-page device"):
            batch.detect(changed, rules)
        assert all(batch.last_result(k) is v for k, v in old.items())
        counts = batch.stats()
        assert counts["resident_pages"] == 0 and counts["failed_queries"] == 1
        assert counts["last_failure"]["page"] == 1 and counts["page_dispatches"] == 4
        retry = batch.detect(changed, rules)
        assert retry[7].required_candidates == 0 and retry[19] == old[19] and retry[5] == old[5]
        assert batch.stats()["page_builds"] == 4 and batch._resident[1]._work._engine is not engine
        (tmp_path / "page-fault-retry.json").write_text(
            json.dumps(
                dict(
                    after_failure=counts,
                    after_retry=batch.stats(),
                    previous={k: digest(v, k) for k, v in old.items()},
                    retried={k: digest(v, k) for k, v in retry.items()},
                ),
                indent=2,
            ),
            encoding="utf8",
        )


def test_empty_registry_and_zero_pair_page_do_not_launch(tmp_path):
    m = api()
    with m.PagedWorldCollisionWorkspace([], "cuda", cache_root=tmp_path) as empty:
        assert empty.detect({}, {}) == {} and empty.layout == ()
        assert empty.stats()["page_dispatches"] == 0
    key = ph.CollisionKey(9, 0, 0)
    mod = ph.CollisionModel((ph.CollisionShape(key, "sphere", (1,), 0.125),))
    with m.PagedWorldCollisionWorkspace(
        [(1, ph.CollisionModel(())), (9, mod)], "cuda", cache_root=tmp_path
    ) as batch:
        results = batch.detect(
            {1: {}, 9: {key: ph.CollisionMotion([[1, 0, 0]])}},
            {1: ph.CollisionPolicy(), 9: ph.CollisionPolicy()},
        )
        assert not results[1].bounds and len(results[9].bounds) == 1
        assert batch.layout[0].planned_broad_bytes == batch.stats()["peak_broad_device_bytes"] == 0
        assert batch.stats()["completed_launches"] == 0
