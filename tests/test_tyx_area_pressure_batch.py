"""Atomic bounded multi-query clipping; HOST exact and analytic static oracles.

All numerical results come from real intersector calls. Wrappers only observe
chunk boundaries or inject an exception after real device work and readback.
No scene stepping, cost benchmark, or whole-device geometry claim is involved.
"""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from test_area_pressure_intersection import (
    _actual,
    _case,
    _horizontal_vertices,
    _identity,
    _rigid_input,
    _stamp,
    _unit_input,
)
from test_tyx_area_pressure_intersection import (
    MODES,
    _immutable,
    _input_bytes,
    _meshes,
    _same_result,
)

from phymeo.backends.tyx_area_pressure_intersection import (
    AreaPressureIntersectionRequest,
    TyxAreaPressureIntersector,
)
from phymeo.geometry.area_pressure_intersection import (
    AreaPressureIntersectionError,
    intersect_area_pressure,
)


def _request(name, ordinal):
    pressure, rigid, domain, *_ = _case(name)
    return _request_from_meshes(_meshes(pressure, rigid, domain), ordinal)


def _request_from_meshes(meshes, ordinal):
    return AreaPressureIntersectionRequest(
        pressure_mesh=meshes[0],
        rigid_mesh=meshes[1],
        identity=replace(
            _identity(),
            compliant_body=f"body-a-{ordinal}",
            rigid_body=f"body-b-{ordinal}",
            compliant_material=f"material-{ordinal}",
        ),
        stamp=replace(
            _stamp(),
            world_id=f"world-{ordinal}",
            step_index=ordinal,
            sample_time_s=ordinal / 100,
            configuration_id=f"configuration-{ordinal}",
        ),
    )


def _reference(request):
    return intersect_area_pressure(
        request.pressure_mesh,
        request.rigid_mesh,
        identity=request.identity,
        stamp=request.stamp,
    )


def _requests():
    # Request order deliberately differs from lexical identity order. All five
    # candidate pairs have positive area, so capacity two requires 2+2+1 lanes.
    return (
        _request("cross_cells", 8),
        _request("multiple_faces", 2),
        _request("horizontal", 5),
    )


def _publication(intersector):
    return (
        intersector.last_results,
        intersector.last_batch_routes,
        intersector.last_result,
        intersector.last_routes,
    )


def _unchanged_publication(intersector, previous):
    assert intersector.last_results is previous[0]
    assert intersector.last_batch_routes == previous[1]
    assert intersector.last_result is previous[2]
    assert intersector.last_routes == previous[3]


def _assert_batch(intersector, actual, requests, expected):
    assert isinstance(actual, tuple) and len(actual) == len(requests)
    assert intersector.last_results is actual
    assert isinstance(intersector.last_batch_routes, tuple)
    assert len(intersector.last_batch_routes) == len(requests)
    for result, request, reference, routes in zip(
        actual, requests, expected, intersector.last_batch_routes, strict=True
    ):
        _same_result(result, reference)
        assert result.surface.identity == request.identity
        assert result.surface.stamp == request.stamp
        assert isinstance(routes, tuple) and len(routes) == result.candidate_pairs
        assert all(
            route == "device"
            or route.startswith("exact_host:")
            or route in ("exact_empty:aabb", "exact_empty:plane")
            for route in routes
        )
    assert intersector.last_result is actual[-1]
    assert intersector.last_routes == intersector.last_batch_routes[-1]


def _counter_snapshot(intersector):
    stats = intersector.stats
    for field in ("batch_calls", "attempts", "candidate_pairs", "culled_pairs", "launches"):
        assert type(stats[field]) is int and stats[field] >= 0
    with pytest.raises(TypeError):
        stats["batch_calls"] = -1
    return dict(stats)


@pytest.mark.parametrize("mode", MODES)
def test_many_preserves_request_order_and_exact_outputs_across_chunks(mode, monkeypatch):
    requests = _requests()
    expected = tuple(_reference(request) for request in requests)
    before = tuple(_input_bytes(request.pressure_mesh, request.rigid_mesh) for request in requests)
    with TyxAreaPressureIntersector(mode=mode, capacity=2) as intersector:
        original = intersector._device_clip
        counts = []

        def record_chunk(coefficients, count):
            counts.append(count)
            assert coefficients.shape == (2, 4, 3)
            assert coefficients.dtype == np.dtype(np.float32)
            return original(coefficients, count)

        with monkeypatch.context() as patch:
            patch.setattr(intersector, "_device_clip", record_chunk)
            actual = intersector.intersect_many(requests)
        assert counts == [2, 2, 1]
        _assert_batch(intersector, actual, requests, expected)
        for name, result, reference in zip(
            ("cross_cells", "multiple_faces", "horizontal"), actual, expected, strict=True
        ):
            integrated = _actual(_same_result(result, reference))
            _, _, _, area, wrench, *_ = _case(name)
            assert integrated["area"] == pytest.approx(area, rel=1e-10, abs=1e-10)
            np.testing.assert_allclose(integrated["wrench"], wrench, rtol=1e-10, atol=1e-10)
            _immutable(result)
        stats = _counter_snapshot(intersector)
        assert stats["batch_calls"] == 1 and stats["attempts"] == 3
        assert stats["candidate_pairs"] == 5 and stats["culled_pairs"] == 0
        assert stats["launches"] == 3
        assert (
            tuple(_input_bytes(request.pressure_mesh, request.rigid_mesh) for request in requests)
            == before
        )
    for result, reference in zip(actual, expected, strict=True):
        _same_result(result, reference)


@pytest.mark.parametrize("mode", MODES)
def test_strict_empty_certificates_skip_launch_but_touch_keeps_fallback(mode, monkeypatch):
    aabb = _request_from_meshes(_meshes(_unit_input(), _rigid_input(_horizontal_vertices(2))), 1)
    # Both AABBs overlap: every point lies inside [0,1]^3. The complete finite
    # face nevertheless has x+y+z>1 and is strictly outside the tetrahedron.
    plane = _request_from_meshes(
        _meshes(
            _unit_input(),
            _rigid_input(((3 / 4, 3 / 4, 1 / 4), (7 / 8, 3 / 4, 1 / 4), (3 / 4, 7 / 8, 1 / 4))),
        ),
        2,
    )
    requests = (aabb, plane)
    expected = tuple(_reference(request) for request in requests)
    boundary = _request_from_meshes(
        _meshes(_unit_input(), _rigid_input(_horizontal_vertices(1))), 3
    )
    boundary_expected = _reference(boundary)
    with TyxAreaPressureIntersector(mode=mode, capacity=1) as intersector:
        with monkeypatch.context() as patch:

            def forbidden_device(*args, **kwargs):
                raise AssertionError("strictly empty batch launched device clipping")

            patch.setattr(intersector, "_device_clip", forbidden_device)
            actual = intersector.intersect_many(requests)
        _assert_batch(intersector, actual, requests, expected)
        assert intersector.last_batch_routes == (("exact_empty:aabb",), ("exact_empty:plane",))
        for result in actual:
            assert result.surface.triangles == ()
        stats = _counter_snapshot(intersector)
        assert stats["candidate_pairs"] == stats["culled_pairs"] == 2
        for field in ("launches", "fallback_pairs", "uploaded_bytes", "downloaded_bytes"):
            assert stats[field] == 0
        boundary_result = intersector.intersect_many((boundary,))
        _assert_batch(intersector, boundary_result, (boundary,), (boundary_expected,))
        assert boundary_result[0].surface.triangles == ()
        assert intersector.last_batch_routes[0][0].startswith("exact_host:")
        after = _counter_snapshot(intersector)
        assert after["culled_pairs"] == 2
        assert after["launches"] == after["fallback_pairs"] == 1

    with TyxAreaPressureIntersector(mode=mode, capacity=1, fallback="reject") as intersector:
        actual = intersector.intersect_many(requests)
        _assert_batch(intersector, actual, requests, expected)
        previous = _publication(intersector)
        with pytest.raises(AreaPressureIntersectionError) as error:
            intersector.intersect_many((boundary,))
        assert error.value.category == "device_unresolved"
        _unchanged_publication(intersector, previous)
        assert intersector.stats["fallback_pairs"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_last_request_preflight_rejects_before_any_launch_or_publication(mode, monkeypatch):
    valid = _request("horizontal", 4)
    empty = _request_from_meshes(_meshes(_unit_input(), _rigid_input(_horizontal_vertices(2))), 9)
    invalid_empty_mesh = _request_from_meshes(
        _meshes(_unit_input(), _rigid_input(_horizontal_vertices(2))), 10
    )
    # A frozen object's public fields are not a trust certificate. The private
    # exact cache and separated geometry still describe the old valid asset.
    object.__setattr__(
        invalid_empty_mesh.pressure_mesh, "nodal_pressure", np.array((-1.0, 0.0, 0.0, 1.0))
    )
    with TyxAreaPressureIntersector(mode=mode, capacity=2) as intersector:
        intersector.intersect_many((valid,))
        previous = _publication(intersector)
        before = _counter_snapshot(intersector)
        bad_last = (
            replace(valid, identity=None),
            replace(empty, stamp=None),  # Geometric emptiness must not excuse invalid metadata.
            replace(valid, pressure_mesh=None),
            invalid_empty_mesh,
        )
        with monkeypatch.context() as patch:

            def forbidden_device(*args, **kwargs):
                raise AssertionError("last-query preflight was deferred until after a launch")

            patch.setattr(intersector, "_device_clip", forbidden_device)
            for invalid in bad_last:
                with pytest.raises(ValueError):
                    intersector.intersect_many((valid, invalid))
                _unchanged_publication(intersector, previous)
        after = _counter_snapshot(intersector)
        for field in ("launches", "fallback_pairs", "uploaded_bytes", "downloaded_bytes"):
            assert after[field] == before[field]
        np.testing.assert_array_equal(
            invalid_empty_mesh.pressure_mesh.nodal_pressure, (-1.0, 0.0, 0.0, 1.0)
        )
        _assert_batch(
            intersector, intersector.intersect_many((valid,)), (valid,), (_reference(valid),)
        )


@pytest.mark.parametrize("mode", MODES)
def test_second_chunk_failure_retains_entire_previous_batch_and_retry(mode, monkeypatch):
    requests = _requests()
    expected = tuple(_reference(request) for request in requests)
    seed = (_request("horizontal", 12), _request("horizontal", 3))
    with TyxAreaPressureIntersector(mode=mode, capacity=2) as intersector:
        accepted = intersector.intersect_many(seed)
        previous = _publication(intersector)
        before = _counter_snapshot(intersector)
        original = intersector._device_clip
        counts = []

        def fail_after_second_readback(coefficients, count):
            result = original(coefficients, count)
            counts.append(count)
            if len(counts) == 2:
                raise RuntimeError("injected after second real chunk readback")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(intersector, "_device_clip", fail_after_second_readback)
            with pytest.raises(RuntimeError, match="second real chunk readback"):
                intersector.intersect_many(requests)
        assert counts == [2, 2]
        _unchanged_publication(intersector, previous)
        assert intersector.stats["launches"] == before["launches"] + 2
        for result, request in zip(accepted, seed, strict=True):
            _same_result(result, _reference(request))
        retried = intersector.intersect_many(requests)
        _assert_batch(intersector, retried, requests, expected)
        assert intersector.stats["launches"] == before["launches"] + 5

        previous = _publication(intersector)
        publish = intersector._publish_query
        published_candidates = []

        def fail_after_second_publication(*args, **kwargs):
            candidate = publish(*args, **kwargs)
            published_candidates.append(candidate)
            if len(published_candidates) == 2:
                raise RuntimeError("injected after second query publication preparation")
            return candidate

        with monkeypatch.context() as patch:
            patch.setattr(intersector, "_publish_query", fail_after_second_publication)
            with pytest.raises(RuntimeError, match="second query publication preparation"):
                intersector.intersect_many(seed)
        assert len(published_candidates) == 2
        _unchanged_publication(intersector, previous)
        _assert_batch(
            intersector,
            intersector.intersect_many(seed),
            seed,
            tuple(_reference(request) for request in seed),
        )


@pytest.mark.parametrize("mode", MODES)
def test_query_and_batch_pair_capacity_reject_without_truncation_or_fallback(mode, monkeypatch):
    one = _request("horizontal", 1)
    two = _request("cross_cells", 2)
    three = _request_from_meshes(
        _meshes(
            _unit_input(),
            _rigid_input(
                _horizontal_vertices(1 / 4)
                + _horizontal_vertices(1 / 2)
                + _horizontal_vertices(3 / 4),
                faces=((0, 1, 2), (3, 4, 5), (6, 7, 8)),
                face_ids=("low", "middle", "high"),
            ),
        ),
        3,
    )
    with TyxAreaPressureIntersector(mode=mode, capacity=2) as intersector:
        intersector.intersect_many((one,))
        previous = _publication(intersector)
        before = _counter_snapshot(intersector)
        # Each individual two-pair query fits capacity; 1025*2 exceeds the
        # separate global budget while remaining below the request-count cap.
        invalid_batches = ((one, three), (two,) * 1025, (one,) * 2049)
        with monkeypatch.context() as patch:

            def forbidden_device(*args, **kwargs):
                raise AssertionError("over-capacity batch launched or truncated work")

            patch.setattr(intersector, "_device_clip", forbidden_device)
            for requests in invalid_batches:
                with pytest.raises(AreaPressureIntersectionError) as error:
                    intersector.intersect_many(requests)
                assert error.value.category == "resource_limit"
                _unchanged_publication(intersector, previous)
        after = _counter_snapshot(intersector)
        for field in ("launches", "fallback_pairs", "uploaded_bytes", "downloaded_bytes"):
            assert after[field] == before[field]
        valid = (one, two)
        _assert_batch(
            intersector,
            intersector.intersect_many(valid),
            valid,
            tuple(_reference(request) for request in valid),
        )


@pytest.mark.parametrize("mode", MODES)
def test_request_immutability_empty_batch_legacy_and_closed_contract(mode):
    request = _request("horizontal", 1)
    with pytest.raises(FrozenInstanceError):
        request.stamp = _stamp()
    expected = _reference(request)
    with TyxAreaPressureIntersector(mode=mode, capacity=2) as intersector:
        accepted = intersector.intersect_many((request,))
        previous = _publication(intersector)
        for invalid in ([request], iter((request,)), (request, object())):
            with pytest.raises(ValueError):
                intersector.intersect_many(invalid)
            _unchanged_publication(intersector, previous)
        before = _counter_snapshot(intersector)
        assert intersector.intersect_many(()) == ()
        assert intersector.last_results == intersector.last_batch_routes == ()
        assert intersector.last_result is None and intersector.last_routes == ()
        after = _counter_snapshot(intersector)
        assert after["batch_calls"] == before["batch_calls"] + 1
        for field in ("attempts", "candidate_pairs", "culled_pairs", "launches"):
            assert after[field] == before[field]
        legacy = intersector.intersect(
            request.pressure_mesh,
            request.rigid_mesh,
            identity=request.identity,
            stamp=request.stamp,
        )
        _same_result(legacy, expected)
        assert intersector.last_results == (legacy,)
        assert intersector.last_result is legacy
        assert intersector.last_batch_routes == (intersector.last_routes,)
    intersector.close()
    assert intersector.stats["live_bytes"] == 0
    _same_result(accepted[0], expected)
    _same_result(legacy, expected)
    with pytest.raises(RuntimeError):
        intersector.intersect_many((request,))
    with pytest.raises(RuntimeError):
        intersector.intersect_many(())
