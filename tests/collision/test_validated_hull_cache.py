"""Exact stored-pose hull reuse: public witnesses, ownership and failed-input controls."""

from dataclasses import fields, is_dataclass, replace
from fractions import Fraction
from itertools import product
from struct import pack

import numpy as np
import pytest

import phymeo as ph
from phymeo.core.capabilities import SolverFailure
from phymeo.geometry import _convex_hull as hull_module
from phymeo.geometry import collision_pipeline as pipeline


def _bits(value):
    """Every public dataclass field, including non-comparing registration fields."""
    if is_dataclass(value):
        return (type(value).__qualname__, tuple((f.name, _bits(getattr(value, f.name))) for f in fields(value)))
    if isinstance(value, np.ndarray):
        data = tuple(_bits(v) for v in value.flat) if value.dtype.hasobject else value.tobytes()
        return ("array", value.dtype.str, value.shape, data)
    if isinstance(value, (float, np.floating)):
        return ("float", pack("<d", float(value)))
    if isinstance(value, Fraction):
        return ("fraction", value.numerator, value.denominator)
    if isinstance(value, dict):
        return ("dict", tuple((_bits(k), _bits(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_bits(v) for v in value))
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    raise TypeError(type(value))


def _hull_bits(hull):
    return _bits((hull.points, hull.step, hull.vertices, hull.edges, hull.edge_rays, tuple(
        (f.normal, f.plane, f.cols, f.ids, f.chart.ids, f.chart.ring) for f in hull.faces
    )))


def _scene(center=(0, 0, 37 / 32), *, world=0, swapped=False, transform=False):
    kh, ks = ph.CollisionKey(world, 1, 0), ph.CollisionKey(world, 2, 0)
    if swapped:
        kh, ks = ks, kh
    hull = ph.CollisionShape(kh, "hull", tuple(range(40, 48)))
    ball = ph.CollisionShape(ks, "sphere", (99,), 1 / 8)
    points = np.array(tuple(product((-1.0, 1.0), repeat=3)))
    center = np.array(center, dtype=float)
    if transform:
        # Exact signed permutation and dyadic offset; actual stored coordinates are used.
        points = points[:, [2, 0, 1]] * [-1, 1, -1] + [3.5, -2.25, 0.75]
        center = center[[2, 0, 1]] * [-1, 1, -1] + [3.5, -2.25, 0.75]
    return ph.CollisionModel((ball, hull)), {
        kh: ph.CollisionMotion(points), ks: ph.CollisionMotion([center])
    }, hull, ball


def _material_points(row, model, paths, time=0.0, witness=None):
    witness = row.witness if witness is None else witness
    by_key = {s.key: s for s in model.shapes}
    for key, weights, offset, point in (
        (row.first, witness.first_weights, witness.first_offset, witness.first_point),
        (row.second, witness.second_weights, witness.second_offset, witness.second_point),
    ):
        shape, path = by_key[key], paths[key]
        assert path.turn is None
        points = path.start if path.end is None else path.start + time * (path.end - path.start)
        assert sum(w for _, w in weights) == pytest.approx(1.0, abs=1e-15)
        assert all(node in shape.vertex_ids for node, _ in weights)
        rebuilt = sum(w * points[shape.vertex_ids.index(node)] for node, w in weights)
        if offset is not None:
            rebuilt = rebuilt + offset
        np.testing.assert_allclose(rebuilt, point, atol=2e-14, rtol=0)


def _cache(shapes):
    from phymeo.geometry._validated_hull_cache import ValidatedHullCache

    return ValidatedHullCache(shapes)


def _uncached_get(self, shape, path, builder=None):
    return (hull_module.HullShape if builder is None else builder)(shape, path)


def test_public_repeated_fixed_hull_reuses_validation_after_real_witness(monkeypatch):
    """Primary RED: old public query is correct, but repeats the actual constructor."""
    model, paths, hull, ball = _scene()
    original, calls = hull_module.HullShape.__init__, []

    def counted(self, shape, path):
        calls.append(shape.key)
        original(self, shape, path)

    monkeypatch.setattr(hull_module.HullShape, "__init__", counted)
    with ph.CollisionWorkspace(model) as work:
        first = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), work)
        (row,) = first.pairs
        assert first.status == "ok" and row.candidate and row.status == "separated"
        assert row.reason == "sphere_hull_projection"
        assert row.witness.gap_m == 1 / 32
        assert row.witness.normal == (0.0, 0.0, 1.0)
        assert row.witness.first_point == (0.0, 0.0, 1.0)
        assert row.witness.second_point == (0.0, 0.0, 33 / 32)
        _material_points(row, model, paths)
        saved = _bits(first)
        assert calls == [hull.key]
        changed = {**paths, ball.key: ph.CollisionMotion([[1 / 8, 0, 37 / 32]])}
        second = ph.detect(model, changed, ph.CollisionPolicy(margin=0.25), work)
        assert second.pairs[0].witness.first_point == (1 / 8, 0.0, 1.0)
        assert second.pairs[0].witness.gap_m == 1 / 32
        _material_points(second.pairs[0], model, changed)
        assert _bits(first) == saved
        assert calls == [hull.key], "same stored hull was rebuilt for a new sphere query"


@pytest.mark.parametrize("center", [(0, 0, 37 / 32), (35 / 32, 36 / 32, 0), (33 / 32, 34 / 32, 34 / 32), (0, 0, 9 / 8), (0, 0, 0)])
@pytest.mark.parametrize("swapped,transform", [(False, False), (True, True)])
def test_complete_public_payload_matches_original_constructor(center, swapped, transform, monkeypatch):
    model, paths, _, _ = _scene(center, swapped=swapped, transform=transform)
    from phymeo.geometry._validated_hull_cache import ValidatedHullCache

    with monkeypatch.context() as local:
        local.setattr(ValidatedHullCache, "get", _uncached_get)
        with ph.CollisionWorkspace(model) as fresh:
            expected = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), fresh)
    with ph.CollisionWorkspace(model) as work:
        for _ in range(2):
            actual = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), work)
            assert _bits(actual) == _bits(expected)
            if actual.pairs[0].witness is not None:
                _material_points(actual.pairs[0], model, paths)


def test_byte_binding_representation_and_moving_paths_use_real_constructor():
    model, paths, hull, _ = _scene()
    cache, calls = _cache(model.shapes), []
    start = paths[hull.key].start

    def build(shape, path):
        calls.append(path)
        return hull_module.HullShape(shape, path)

    baseline = _hull_bits(cache.get(hull, ph.CollisionMotion(start), build))
    assert _hull_bits(cache.get(hull, ph.CollisionMotion(start.copy()), build)) == baseline
    assert len(calls) == 1
    changed = start.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], -np.inf)
    cache.get(hull, ph.CollisionMotion(changed), build)
    cache.get(hull, ph.CollisionMotion(start[[1, 0, 2, 3, 4, 5, 6, 7]]), build)
    cache.get(hull, ph.CollisionMotion(start, start), build)
    cache.get(hull, ph.CollisionMotion(start, turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), 0)), build)
    cache.get(hull, ph.CollisionMotion(start, body_orientation_xyzw=(0, 0, 0, 1)), build)
    assert len(calls) == 6
    for moving in (
        ph.CollisionMotion(start, start + [0.125, 0, 0]),
        ph.CollisionMotion(start, turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), 0.125)),
    ):
        cache.get(hull, moving, build)
        cache.get(hull, moving, build)
    assert len(calls) == 10 and len(cache) == 0


def test_copies_and_builder_identity_cannot_poison_cached_material():
    model, paths, hull, _ = _scene()
    cache, originals = _cache(model.shapes), []

    def build(shape, path):
        value = hull_module.HullShape(shape, path)
        originals.append(value)
        return value

    first = cache.get(hull, paths[hull.key], build)
    expected = _hull_bits(first)
    first.points[40][0] = Fraction(777)
    first.faces[0].normal[0] = Fraction(999)
    first.faces[0].chart.ids.clear()
    originals[0].points[41][0] = Fraction(-888)
    second = cache.get(hull, paths[hull.key], build)
    assert second is not first and _hull_bits(second) == expected and len(originals) == 1

    def different_builder(shape, path):
        raise SolverFailure("different builder must be called")

    with pytest.raises(SolverFailure, match="different builder"):
        cache.get(hull, paths[hull.key], different_builder)
    assert len(cache) == 0
    assert _hull_bits(cache.get(hull, paths[hull.key], build)) == expected
    assert len(originals) == 2
    cache.clear()
    assert len(cache) == 0
    cache.get(hull, paths[hull.key], build)
    assert len(originals) == 3
    stats = cache.stats()
    stats["validated_hull_builds"] = -1
    assert cache.stats()["validated_hull_builds"] == 3


def test_default_builder_is_resolved_at_each_call(monkeypatch):
    model, paths, hull, _ = _scene()
    cache = _cache(model.shapes)
    cache.get(hull, paths[hull.key])
    original, calls = hull_module.HullShape, []

    def changed(shape, path):
        calls.append(shape.key)
        return original(shape, path)

    monkeypatch.setattr(hull_module, "HullShape", changed)
    cache.get(hull, paths[hull.key])
    cache.get(hull, paths[hull.key])
    assert calls == [hull.key]


@pytest.mark.parametrize("invalid", ["duplicate", "coplanar", "count"])
def test_invalid_replacements_are_not_hidden_or_negative_cached(invalid):
    model, paths, hull, _ = _scene()
    cache, calls = _cache(model.shapes), []

    def build(shape, path):
        calls.append(shape.key)
        return hull_module.HullShape(shape, path)

    good = cache.get(hull, paths[hull.key], build)
    points = paths[hull.key].start.copy()
    if invalid == "duplicate":
        points[0] = points[1]
    elif invalid == "coplanar":
        points[:, 2] = 0
    else:
        points = points[:-1]
    for _ in range(2):
        with pytest.raises(ValueError):
            cache.get(hull, ph.CollisionMotion(points), build)
        assert len(cache) == 0
    assert _hull_bits(cache.get(hull, paths[hull.key], build)) == _hull_bits(good)
    assert len(calls) == 4


def test_registry_identity_full_registration_capacity_and_world_isolation():
    model, paths, hull, ball = _scene()
    cache = _cache(model.shapes)
    cache.get(hull, paths[hull.key])
    for foreign in (replace(hull), replace(hull, revision=1), replace(hull, key=replace(hull.key, world=3))):
        with pytest.raises(ValueError, match="registered hull"):
            cache.get(foreign, paths[hull.key])
    with pytest.raises(ValueError, match="registered hull"):
        cache.get(ball, paths[ball.key])
    object.__setattr__(hull, "revision", 4)
    try:
        with pytest.raises(ValueError, match="registration"):
            cache.get(hull, paths[hull.key])
    finally:
        object.__setattr__(hull, "revision", 0)
    with pytest.raises(ValueError):
        _cache((hull, hull))
    with pytest.raises(ValueError):
        _cache(tuple(replace(ball, key=ph.CollisionKey(i, 2, 0)) for i in range(257)))
    many = ph.CollisionShape(hull.key, "hull", tuple(range(65)))
    limited = _cache((many,))
    points = ph.CollisionMotion([[i, i % 3, i % 5] for i in range(65)])
    for _ in range(2):
        with pytest.raises(SolverFailure, match="convex_vertex_capacity"):
            limited.get(many, points)
        assert len(limited) == 0
    assert limited.stats()["validated_hull_attempts"] == 2


def test_public_multiworld_failure_capacity_and_retry_preserve_output(monkeypatch):
    m0, p0, h0, _ = _scene()
    m1, p1, h1, _ = _scene(world=1, transform=True)
    model, paths = ph.CollisionModel(m0.shapes + m1.shapes), p0 | p1
    original, calls = hull_module.HullShape.__init__, []

    def counted(self, shape, path):
        calls.append(shape.key)
        original(self, shape, path)

    monkeypatch.setattr(hull_module.HullShape, "__init__", counted)
    with ph.CollisionWorkspace(model) as work:
        first = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), work)
        saved = _bits(first)
        assert len(first.pairs) == 6
        assert sum(row.reason == "different_world" for row in first.pairs) == 4
        assert calls == [h0.key, h1.key]
        failed = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25, candidate_capacity=0), work)
        assert failed.status == "capacity_exceeded" and work.last_result is first
        assert calls == [h0.key, h1.key]
        with pytest.raises(ValueError, match="registered vertices"):
            ph.detect(model, paths | {h0.key: ph.CollisionMotion(p0[h0.key].start[:-1])}, ph.CollisionPolicy(), work)
        assert work.last_result is first and _bits(first) == saved
        assert _bits(ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), work)) == saved
        assert calls == [h0.key, h1.key]
        shifted = paths | {h1.key: ph.CollisionMotion(p1[h1.key].start + [0, 0, 1 / 64])}
        ph.detect(model, shifted, ph.CollisionPolicy(margin=0.25), work)
        assert calls == [h0.key, h1.key, h1.key]


def test_world_resolution_check_and_affine_window_are_not_cached(monkeypatch):
    from phymeo.geometry._validated_hull_cache import ValidatedHullCache

    model, paths, hull, ball = _scene()
    scenarios = [
        {k: ph.CollisionMotion(p.start + [2**40, 0, 0]) for k, p in paths.items()},
        paths | {ball.key: ph.CollisionMotion([[0, 0, 1.5]], [[0, 0, 1.0]])},
    ]
    for index, scene in enumerate(scenarios):
        with monkeypatch.context() as local:
            local.setattr(ValidatedHullCache, "get", _uncached_get)
            with ph.CollisionWorkspace(model) as fresh:
                expected = ph.detect(model, scene, ph.CollisionPolicy(margin=0.25), fresh)
        with ph.CollisionWorkspace(model) as work:
            for _ in range(2):
                actual = ph.detect(model, scene, ph.CollisionPolicy(margin=0.25), work)
                assert _bits(actual) == _bits(expected)
            if index == 0:
                assert actual.pairs[0].status == "unresolved"
                assert actual.pairs[0].reason == "support_world_witness_resolution"
            else:
                row = actual.pairs[0]
                assert row.status == "hit" and row.sweep.first_t[0] <= 0.75 <= row.sweep.first_t[1]
                _material_points(row, model, scene, row.sweep.witness_time, row.sweep.witness)
            assert work.stats()["validated_hull_builds"] == 1
            assert work.stats()["validated_hull_hits"] > 0


@pytest.mark.tyx
def test_device_still_classifies_each_query_and_preserves_failure(tyx_mode, tmp_path, monkeypatch):
    model, paths, hull, _ = _scene()
    with ph.CollisionWorkspace(model) as host:
        expected = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), host)

    def forbidden(*args, **kwargs):
        raise AssertionError("device query silently used HOST broad phase")

    monkeypatch.setattr(pipeline, "_host_query", forbidden)
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path) as work:
        first = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), work)
        second = ph.detect(model, paths, ph.CollisionPolicy(margin=0.25), work)
        assert _bits(first.pairs) == _bits(expected.pairs) == _bits(second.pairs)
        assert _bits(first.bounds) == _bits(expected.bounds)
        stats = work.stats()
        assert stats["completed_launches"] == 4
        assert stats["validated_hull_builds"] == 1 and stats["validated_hull_hits"] == 1
        assert ph.detect(model, paths, ph.CollisionPolicy(margin=0.25, candidate_capacity=0), work).status == "capacity_exceeded"
        assert work.last_result is second

        def fault(*args, **kwargs):
            raise RuntimeError("injected classify failure")

        monkeypatch.setattr(work._engine, "query", fault)
        with pytest.raises(RuntimeError, match="injected classify"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert work.last_result is second
        with pytest.raises(RuntimeError, match="closed"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
