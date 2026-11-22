"""Bounded TYX clipping contracts with exact and independent analytic oracles.

These static queries do not step physics or claim an entirely device-resident
geometry pipeline: device descriptors still require exact HOST certification.
Fault injection checks publication only and never supplies a numerical oracle.
"""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from test_area_pressure_intersection import (
    CASE_NAMES,
    _actual,
    _case,
    _double_input,
    _full_input,
    _horizontal_vertices,
    _identity,
    _rigid_input,
    _stamp,
    _unit_input,
)

from phymeo.backends.tyx_area_pressure_intersection import TyxAreaPressureIntersector
from phymeo.geometry import _area_pressure_clipping as exact
from phymeo.geometry.area_pressure import quadrature_area_pressure
from phymeo.geometry.area_pressure_intersection import (
    AreaPressureIntersectionError,
    RigidPressureMesh,
    TetraPressureMesh,
    intersect_area_pressure,
)

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
COUNTERS = (
    "attempts",
    "launches",
    "device_polygons",
    "fallback_pairs",
    "downloaded_bytes",
    "uploaded_bytes",
    "live_bytes",
    "peak_bytes",
)
TIMINGS = ("device_seconds", "refine_seconds", "fallback_seconds", "total_seconds")


def _meshes(pressure, rigid, domain="local"):
    return TetraPressureMesh(**pressure, pressure_domain=domain), RigidPressureMesh(**rigid)


def _input_bytes(pressure, rigid):
    return tuple(
        (value.shape, value.dtype.str, value.tobytes())
        for value in (
            pressure.world_vertices,
            pressure.cells,
            pressure.nodal_pressure,
            rigid.world_vertices,
            rigid.faces,
        )
    )


def _run(intersector, meshes, *, stamp=None):
    return intersector.intersect(
        *meshes, identity=_identity(), stamp=_stamp() if stamp is None else stamp
    )


def _reference(meshes, *, stamp=None):
    return intersect_area_pressure(
        *meshes, identity=_identity(), stamp=_stamp() if stamp is None else stamp
    )


def _bytes_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def _same_result(actual, expected):
    assert actual.pressure_domain == expected.pressure_domain
    assert actual.candidate_pairs == expected.candidate_pairs
    assert actual.polygon_count == expected.polygon_count
    assert actual.shared_face_duplicates == expected.shared_face_duplicates
    assert actual.surface.identity == expected.surface.identity
    assert actual.surface.stamp == expected.surface.stamp
    assert len(actual.surface.triangles) == len(expected.surface.triangles)
    for left, right in zip(actual.surface.triangles, expected.surface.triangles, strict=True):
        for field in ("triangle_id", "compliant_cell_id", "rigid_face_id"):
            assert getattr(left, field) == getattr(right, field)
        for field in ("vertices", "vertex_pressure", "pressure_gradient"):
            _bytes_equal(getattr(left, field), getattr(right, field))
    left = quadrature_area_pressure(actual.surface)
    right = quadrature_area_pressure(expected.surface)
    assert left.identity == right.identity and left.stamp == right.stamp
    assert left.sample_ids == right.sample_ids and left.source_ids == right.source_ids
    assert left.total_area_m2 == right.total_area_m2
    for field in ("points", "normals", "dA", "pressure", "gradient"):
        _bytes_equal(getattr(left, field), getattr(right, field))
    return left


def _stats(intersector):
    stats = intersector.stats
    assert set(COUNTERS + TIMINGS) <= set(stats)
    for key in COUNTERS + TIMINGS:
        assert np.isfinite(stats[key]) and stats[key] >= 0, key
    for key in COUNTERS:
        assert isinstance(stats[key], int) and not isinstance(stats[key], bool), key
    assert stats["peak_bytes"] >= stats["live_bytes"]
    with pytest.raises(TypeError):
        stats["attempts"] = -1
    return dict(stats)


def _immutable(result):
    with pytest.raises(FrozenInstanceError):
        result.polygon_count = 99
    with pytest.raises(FrozenInstanceError):
        result.surface.stamp = _stamp()
    for triangle in result.surface.triangles:
        for field in ("vertices", "vertex_pressure", "pressure_gradient"):
            value = getattr(triangle, field)
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.setflags(write=True)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", CASE_NAMES)
def test_fixed_cases_match_exact_canonical_output_and_analytic_wrench(mode, name, monkeypatch):
    pressure, rigid, domain, area, wrench, polygons, duplicates, sources = _case(name)
    meshes = _meshes(pressure, rigid, domain)
    before = _input_bytes(*meshes)
    expected = _reference(meshes)
    # The simple horizontal section is exactly representable in FP32 and is a
    # positive device-route control. Obtain the independent HOST result first.
    fallback = "reject" if name == "horizontal" else "exact_host"
    with TyxAreaPressureIntersector(mode=mode, capacity=8, fallback=fallback) as intersector:
        assert intersector.last_result is None
        with monkeypatch.context() as patch:
            if name == "horizontal":

                def forbidden_clip(*args, **kwargs):
                    raise AssertionError("resolved device polygon called HOST clipping")

                patch.setattr(exact, "clip_triangle", forbidden_clip)
            actual = _run(intersector, meshes)
        assert intersector.last_result is actual
        quad = _same_result(actual, expected)
        integrated = _actual(quad)
        assert integrated["area"] == pytest.approx(area, rel=1e-10, abs=1e-10)
        np.testing.assert_allclose(integrated["wrench"], wrench, rtol=1e-10, atol=1e-10)
        assert actual.polygon_count == polygons
        assert actual.shared_face_duplicates == duplicates
        assert {(t.compliant_cell_id, t.rigid_face_id) for t in actual.surface.triangles} == sources
        assert actual.candidate_pairs == len(meshes[0].cells) * len(meshes[1].faces)
        routes = intersector.last_routes
        assert isinstance(routes, tuple) and len(routes) == actual.candidate_pairs
        assert all(
            route == "device" or route.startswith(("exact_host:", "exact_empty:"))
            for route in routes
        )
        stats = _stats(intersector)
        assert stats["attempts"] == 1 and stats["launches"] >= 1
        if mode == "cuda":
            assert stats["uploaded_bytes"] > 0 and stats["downloaded_bytes"] > 0
        else:
            assert stats["uploaded_bytes"] == stats["downloaded_bytes"] == 0
        assert stats["fallback_pairs"] == sum(route.startswith("exact_host:") for route in routes)
        if name == "horizontal":
            assert routes == ("device",)
            assert stats["device_polygons"] == 1 and stats["fallback_pairs"] == 0
        _immutable(actual)
        assert _input_bytes(*meshes) == before
    assert _stats(intersector)["live_bytes"] == 0
    _same_result(actual, expected)  # Closing scratch must not invalidate owned output.


@pytest.mark.parametrize("mode", MODES)
def test_rigid_motion_and_input_order_preserve_geometry_identity_and_wrench(mode):
    pressure, rigid, domain, area, wrench, _, _, _ = _case("oblique")
    rotation = np.array(
        ((12 / 13, 0, 5 / 13), (4 / 13, 3 / 5, -48 / 65), (-3 / 13, 4 / 5, 36 / 65))
    )
    translation = np.array((0.7, -1.1, 0.4))
    moved = _meshes(
        dict(
            pressure, world_vertices=np.array(pressure["world_vertices"]) @ rotation.T + translation
        ),
        dict(rigid, world_vertices=np.array(rigid["world_vertices"]) @ rotation.T + translation),
        domain,
    )
    expected = _reference(moved)
    with TyxAreaPressureIntersector(mode=mode, capacity=8) as intersector:
        actual = _run(intersector, moved)
        integrated = _actual(_same_result(actual, expected))
        force = rotation @ np.array(wrench[:3])
        moment = rotation @ np.array(wrench[3:]) + np.cross(translation, force)
        assert integrated["area"] == pytest.approx(area, rel=1e-10, abs=1e-10)
        np.testing.assert_allclose(
            integrated["wrench"], np.r_[force, moment], rtol=1e-10, atol=1e-10
        )

        pressure = _double_input()
        rigid = _rigid_input(
            _horizontal_vertices(1 / 4) + _horizontal_vertices(1 / 2),
            faces=((0, 1, 2), (3, 4, 5)),
            face_ids=("low", "high"),
        )
        original_meshes = _meshes(pressure, rigid)
        original = _run(intersector, original_meshes)
        original_routes = intersector.last_routes
        _same_result(original, _reference(original_meshes))
        order = (4, 2, 0, 3, 1)
        old_to_new = {old: new for new, old in enumerate(order)}
        pressure_reordered = dict(
            node_ids=tuple(pressure["node_ids"][i] for i in order),
            world_vertices=tuple(pressure["world_vertices"][i] for i in order),
            nodal_pressure=tuple(pressure["nodal_pressure"][i] for i in order),
            cell_ids=tuple(reversed(pressure["cell_ids"])),
            cells=tuple(
                tuple(old_to_new[cell[i]] for i in (3, 1, 0, 2))
                for cell in reversed(pressure["cells"])
            ),
        )
        rigid_reordered = dict(rigid, faces=((4, 5, 3), (1, 2, 0)), face_ids=("high", "low"))
        reordered_meshes = _meshes(pressure_reordered, rigid_reordered)
        changed = _run(intersector, reordered_meshes)
        _same_result(changed, _reference(reordered_meshes))
        _same_result(changed, original)
        assert intersector.last_routes == original_routes


@pytest.mark.parametrize("mode", MODES)
def test_empty_degenerate_fallback_reject_and_unsupported_queries_preserve_publication(mode):
    empty_vertices = (
        _horizontal_vertices(2),
        _horizontal_vertices(1),
        ((-1, -1, 2), (2, -1, -1), (-1, 2, 2)),
    )
    with TyxAreaPressureIntersector(mode=mode, capacity=8) as intersector:
        for index, vertices in enumerate(empty_vertices):
            meshes = _meshes(_unit_input(), _rigid_input(vertices))
            before = _stats(intersector)
            actual = _run(intersector, meshes)
            quad = _same_result(actual, _reference(meshes))
            assert actual.surface.triangles == ()
            assert _actual(quad) == dict(area=0.0, wrench=[0.0] * 6)
            assert len(intersector.last_routes) == 1
            expected_route = "exact_empty:aabb" if index == 0 else "exact_host:"
            assert intersector.last_routes[0].startswith(expected_route)
            assert _stats(intersector)["fallback_pairs"] == before["fallback_pairs"] + (index != 0)

        accepted = intersector.last_result
        shared_rigid = _rigid_input(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        unsupported = (
            (
                _unit_input(),
                _rigid_input(_horizontal_vertices(1 / 4), faces=((0, 2, 1),)),
                "local",
                "unsupported_gradient",
            ),
            (
                _unit_input(),
                _rigid_input(((1 / 4, -1, -1), (1 / 4, 3, -1), (1 / 4, -1, 3))),
                "local",
                "unsupported_gradient",
            ),
            (_double_input(3), shared_rigid, "local", "unsupported_shared_face_gradient"),
            (_double_input(0), shared_rigid, "local", "unsupported_gradient"),
            (
                _full_input(),
                _rigid_input(_horizontal_vertices(1 / 8)),
                "full",
                "unsupported_gradient",
            ),
        )
        for pressure, rigid, domain, category in unsupported:
            meshes = _meshes(pressure, rigid, domain)
            before = _input_bytes(*meshes)
            with pytest.raises(AreaPressureIntersectionError) as reference_error:
                _reference(meshes)
            assert reference_error.value.category == category
            with pytest.raises(AreaPressureIntersectionError) as error:
                _run(intersector, meshes)
            assert error.value.category == category
            assert intersector.last_result is accepted
            assert _input_bytes(*meshes) == before

    with TyxAreaPressureIntersector(mode=mode, capacity=8, fallback="reject") as intersector:
        positive = _meshes(_unit_input(), _rigid_input(_horizontal_vertices(1 / 4)))
        accepted = _run(intersector, positive)
        for index, vertices in enumerate(empty_vertices):
            meshes = _meshes(_unit_input(), _rigid_input(vertices))
            if index == 0:
                accepted = _run(intersector, meshes)
                _same_result(accepted, _reference(meshes))
                assert intersector.last_routes == ("exact_empty:aabb",)
                continue
            with pytest.raises(AreaPressureIntersectionError) as error:
                _run(intersector, meshes)
            assert error.value.category == "device_unresolved"
            assert intersector.last_result is accepted
            assert _stats(intersector)["fallback_pairs"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_capacity_and_metadata_preflight_reject_without_partial_launch_or_result(mode, monkeypatch):
    for capacity in (0, 2049, True):
        with pytest.raises(ValueError):
            TyxAreaPressureIntersector(mode=mode, capacity=capacity)
    positive = _meshes(_unit_input(), _rigid_input(_horizontal_vertices(1 / 4)))
    two_candidates = _meshes(_double_input(), _rigid_input(_horizontal_vertices(1 / 4)))
    with TyxAreaPressureIntersector(mode=mode, capacity=1) as intersector:
        accepted = _run(intersector, positive)
        before = _stats(intersector)
        inputs_before = _input_bytes(*two_candidates)
        with monkeypatch.context() as patch:

            def forbidden_device(*args, **kwargs):
                raise AssertionError("capacity/metadata preflight launched the device")

            patch.setattr(intersector, "_device_clip", forbidden_device)
            with pytest.raises(AreaPressureIntersectionError) as error:
                _run(intersector, two_candidates)
            assert error.value.category == "resource_limit"
            for identity, stamp in ((None, _stamp()), (_identity(), None)):
                with pytest.raises(ValueError):
                    intersector.intersect(*positive, identity=identity, stamp=stamp)
            assert intersector.last_result is accepted
        after = _stats(intersector)
        for field in ("launches", "fallback_pairs", "uploaded_bytes", "downloaded_bytes"):
            assert after[field] == before[field]
        assert _input_bytes(*two_candidates) == inputs_before
        _same_result(_run(intersector, positive), accepted)
    intersector.close()  # Idempotent cleanup must not free accepted output.
    assert _stats(intersector)["live_bytes"] == 0
    with pytest.raises(RuntimeError):
        _run(intersector, positive)


@pytest.mark.parametrize("mode", MODES)
def test_late_device_readback_failure_retains_last_result_and_retry_is_exact(mode, monkeypatch):
    meshes = _meshes(_unit_input(), _rigid_input(_horizontal_vertices(1 / 4)))
    before_inputs = _input_bytes(*meshes)
    newer_stamp = replace(
        _stamp(), step_index=12, sample_time_s=0.126, configuration_id="new-query"
    )
    expected = _reference(meshes, stamp=newer_stamp)
    with TyxAreaPressureIntersector(mode=mode, capacity=4) as intersector:
        accepted = _run(intersector, meshes)
        original_clip = intersector._device_clip
        observed = []

        def fail_after_device_readback(coefficients, count):
            assert coefficients.dtype == np.dtype(np.float32)
            assert coefficients.shape == (4, 4, 3) and count == 1
            meta, active = original_clip(coefficients, count)
            assert meta.shape == (4, 2) and meta.dtype == np.dtype(np.int32)
            assert active.shape == (4, 8, 7) and active.dtype == np.dtype(np.int32)
            observed.append(True)
            raise RuntimeError("injected after actual device clipping and readback")

        with monkeypatch.context() as patch:
            patch.setattr(intersector, "_device_clip", fail_after_device_readback)
            with pytest.raises(RuntimeError, match="after actual device clipping"):
                _run(intersector, meshes, stamp=newer_stamp)
        assert observed == [True]
        assert intersector.last_result is accepted
        assert accepted.surface.stamp == _stamp()
        _same_result(accepted, _reference(meshes))
        retried = _run(intersector, meshes, stamp=newer_stamp)
        assert intersector.last_result is retried and retried is not accepted
        _same_result(retried, expected)
        assert _input_bytes(*meshes) == before_inputs
        assert _stats(intersector)["attempts"] == 3
    assert _stats(intersector)["live_bytes"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_impossible_device_metadata_cannot_be_hidden_by_exact_fallback(mode, monkeypatch):
    meshes = _meshes(_unit_input(), _rigid_input(_horizontal_vertices(1 / 4)))
    with TyxAreaPressureIntersector(mode=mode, capacity=1) as intersector:
        accepted = _run(intersector, meshes)
        original = intersector._device_clip
        for forged in ((1, 9), (1, 2), (9, 3)):

            def corrupt(coefficients, count):
                meta, active = original(coefficients, count)
                meta[0] = forged
                return meta, active

            fallback_before = intersector.stats["fallback_pairs"]
            with monkeypatch.context() as patch:
                patch.setattr(intersector, "_device_clip", corrupt)
                with pytest.raises(RuntimeError, match="invalid device clipping"):
                    _run(intersector, meshes)
            assert intersector.last_result is accepted
            assert intersector.stats["fallback_pairs"] == fallback_before


@pytest.mark.parametrize("mode", MODES)
def test_per_query_publication_origin_keeps_world_quadrature(mode):
    from phymeo.backends.tyx_area_pressure_intersection import AreaPressureIntersectionRequest

    pressure, rigid, domain, *_ = _case("horizontal")
    meshes = _meshes(pressure, rigid, domain)
    origins = (np.array([1.5, -2.0, 0.125]), np.array([-0.25, 3.0, 1.0]))
    requests = tuple(
        AreaPressureIntersectionRequest(*meshes, _identity(), _stamp(), origin_world=origin)
        for origin in origins
    )
    expected = _reference(meshes)
    with TyxAreaPressureIntersector(mode=mode, capacity=8, fallback="reject") as intersector:
        left, right = intersector.intersect_many(requests)
        for result, origin in zip((left, right), origins, strict=True):
            assert result.surface.triangles
            for item in result.surface.triangles:
                np.testing.assert_array_equal(item.origin_world, origin)
            q = quadrature_area_pressure(result.surface)
            base = quadrature_area_pressure(expected.surface)
            np.testing.assert_allclose(q.points, base.points, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(q.dA, base.dA)
            np.testing.assert_array_equal(q.pressure, base.pressure)
        with pytest.raises(ValueError, match="origin_world"):
            intersector.intersect(
                *meshes, identity=_identity(), stamp=_stamp(), origin_world=[np.nan, 0, 0]
            )
