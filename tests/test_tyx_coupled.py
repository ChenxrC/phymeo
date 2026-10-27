"""Real device contact primitives, followed by joint-driver tests as M6 progresses."""

import importlib
from contextlib import ExitStack

import numpy as np
import pytest

pytestmark = pytest.mark.tyx


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("coupled-tyx")


def execute(name, scalars, arrays, mode, cache, *, repeats=1):
    kernel = getattr(importlib.import_module("phymeo.backends._tyx_coupled_kernels"), name)
    import tyx as tx

    with ExitStack() as resources:
        runtime = tx.Runtime(backend="cuda" if mode == "cuda" else "cpu", cache_root=cache)
        resources.callback(runtime.close)
        resources.enter_context(tx.use_runtime(runtime))
        values = []
        for value in arrays:
            if mode == "cuda":
                value = tx.cuda_array(value)
                resources.callback(value.free)
            else:
                value = np.array(value, order="C", copy=True)
            values.append(value)
        resources.callback(runtime.sync)
        with np.errstate(all="ignore"):
            for _ in range(repeats):
                kernel(*scalars, *values, mode=mode)
            runtime.sync()
        return [value.to_numpy() if mode == "cuda" else value.copy() for value in values]


def position_arrays(particle=(0.8, 0, 0), body=(0, 0, 0), wp=1.0, wb=0.5):
    return [
        np.array([particle], np.float32),
        np.array([body], np.float32),
        np.array([wp], np.float32),
        np.array([wb], np.float32),
        np.ones(1, np.float32),
        np.zeros(1, np.float32),
        np.zeros((1, 3), np.float32),
        np.zeros(1, np.int32),
    ]


def test_device_position_projection_splits_response_by_mass(tyx_mode, cache):
    values = execute("project_contacts", [1, 1], position_arrays(), tyx_mode, cache)
    p, b, _, _, _, multipliers, loads, failed = values
    np.testing.assert_array_equal(failed, 0)
    np.testing.assert_allclose(p, [[14 / 15, 0, 0]], atol=2e-7)
    np.testing.assert_allclose(b, [[-1 / 15, 0, 0]], atol=2e-7)
    np.testing.assert_allclose(multipliers, [2 / 15], atol=2e-7)
    np.testing.assert_allclose(loads, [[2 / 15, 0, 0]], atol=2e-7)
    np.testing.assert_allclose(p + 2 * b, [[0.8, 0, 0]], atol=2e-7)


def test_multiple_spheres_use_stable_particle_major_pairs(tyx_mode, cache):
    data = [
        np.array([[0.8, 0, 0], [10.8, 0, 0]], np.float32),
        np.array([[0, 0, 0], [10, 0, 0]], np.float32),
        np.ones(2, np.float32),
        np.array([0.5, 2], np.float32),
        np.ones(2, np.float32),
        np.zeros(4, np.float32),
        np.zeros((4, 3), np.float32),
        np.zeros(4, np.int32),
    ]
    p, b, _, _, _, multipliers, loads, failed = execute(
        "project_contacts", [2, 2], data, tyx_mode, cache
    )
    np.testing.assert_allclose(p[:, 0], [14 / 15, 10 + 13 / 15], atol=1e-6)
    np.testing.assert_allclose(b[:, 0], [-1 / 15, 10 - 2 / 15], atol=1e-6)
    np.testing.assert_allclose(multipliers, [2 / 15, 0, 0, 1 / 15], atol=1e-6)
    np.testing.assert_array_equal(loads[[1, 2]], 0)
    np.testing.assert_array_equal(failed, 0)


def test_device_position_multiplier_can_release_without_attractive_final_force(tyx_mode, cache):
    values = execute("project_contacts", [1, 1], position_arrays(), tyx_mode, cache)
    values[0][0, 0] += 1
    released = execute("project_contacts", [1, 1], values, tyx_mode, cache)
    np.testing.assert_array_equal(released[5], 0)
    np.testing.assert_array_equal(released[-1], 0)
    assert released[6][0, 0] < 0  # Unwinds the previously accumulated multiplier.
    np.testing.assert_allclose(released[0], [[1.8, 0, 0]], atol=3e-7)
    np.testing.assert_allclose(released[1], 0, atol=3e-7)


@pytest.mark.parametrize("fixed", [False, True])
def test_center_coincidence_and_fixed_pairs_are_deterministic(tyx_mode, cache, fixed):
    initial = position_arrays(particle=(0, 0, 0), wp=0 if fixed else 1, wb=0 if fixed else 0.5)
    a = execute("project_contacts", [1, 1], initial, tyx_mode, cache)
    b = execute("project_contacts", [1, 1], initial, tyx_mode, cache)
    for left, right in zip(a, b, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(a[-1], 0)
    if fixed:
        np.testing.assert_array_equal(a[0], 0)
        np.testing.assert_array_equal(a[1], 0)
        np.testing.assert_array_equal(a[5], 0)
    else:
        np.testing.assert_allclose(a[0] - a[1], [[1, 0, 0]], atol=2e-7)


def test_final_geometry_is_recomputed_after_shared_body_moves(tyx_mode, cache):
    p = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    b = np.array([[0.1, 0, 0]], np.float32)
    data = [
        p,
        b,
        np.ones(1, np.float32),
        np.empty((2, 3), np.float32),
        np.empty(2, np.float32),
        np.zeros(2, np.int32),
    ]
    _, _, _, normals, gaps, failed = execute("contact_geometry", [2, 1], data, tyx_mode, cache)
    delta = p.astype(np.float64) - b.astype(np.float64)
    lengths = np.linalg.norm(delta, axis=1)
    np.testing.assert_allclose(normals, delta / lengths[:, None], atol=2e-7)
    np.testing.assert_allclose(gaps, lengths - 1, atol=2e-7)
    np.testing.assert_array_equal(failed, 0)


def test_shared_body_velocity_sweeps_release_old_impulses_and_conserve_momentum(tyx_mode, cache):
    arrays = [
        np.array([[-1, 0, 0], [-4, 0, 0]], np.float32),
        np.zeros((1, 3), np.float32),
        np.ones(2, np.float32),
        np.ones(1, np.float32),
        np.array([[1, 0, 0], [1, 0, 0]], np.float32),
        np.zeros(2, np.float32),
        np.zeros(2, np.float32),
        np.zeros((2, 3), np.float32),
        np.zeros(2, np.int32),
    ]
    p, b, _, _, _, _, impulses, _, failed = execute(
        "project_contact_velocities", [2, 1, np.float32(1e-6)], arrays, tyx_mode, cache, repeats=5
    )
    np.testing.assert_allclose(p[:, 0], [-1, -2], atol=2e-6)
    np.testing.assert_allclose(b[:, 0], [-2], atol=2e-6)
    np.testing.assert_allclose(impulses, [0, 2], atol=2e-6)
    np.testing.assert_allclose(p.sum(axis=0) + b.sum(axis=0), [-5, 0, 0], atol=2e-6)
    np.testing.assert_array_equal(failed, 0)


def test_device_overflow_is_flagged_instead_of_hidden_by_projection(tyx_mode, cache):
    arrays = position_arrays(particle=(3e38, 0, 0), body=(-3e38, 0, 0))
    values = execute("project_contacts", [1, 1], arrays, tyx_mode, cache)
    assert np.any(values[-1])


@pytest.mark.parametrize("stage", ["position", "velocity", "position_delta", "geometry_delta"])
def test_checked_arithmetic_failure_cannot_be_hidden_by_max_or_clamping(tyx_mode, cache, stage):
    if stage == "velocity":
        arrays = [
            np.array([[-1e-20, 0, 0]], np.float32),
            np.zeros((1, 3), np.float32),
            np.array([1e20], np.float32),
            np.array([1e20], np.float32),
            np.array([[1, 0, 0]], np.float32),
            np.zeros(1, np.float32),
            np.zeros(1, np.float32),
            np.zeros((1, 3), np.float32),
            np.zeros(1, np.int32),
        ]
        values = execute(
            "project_contact_velocities", [1, 1, np.float32(1e-6)], arrays, tyx_mode, cache
        )
    else:
        arrays = position_arrays()
        if stage == "position":
            arrays[0][0] = [0.5e-20, 0, 0]
            arrays[2][:] = arrays[3][:] = 1e20
            arrays[4][:] = 1e-20
        else:
            arrays[0][0] = [np.finfo(np.float32).tiny, 0, 0]
            arrays[1][0] = [
                np.nextafter(np.float32(np.finfo(np.float32).tiny), np.float32(np.inf)),
                0,
                0,
            ]
        if stage == "geometry_delta":
            values = execute(
                "contact_geometry",
                [1, 1],
                [
                    arrays[0],
                    arrays[1],
                    arrays[4],
                    np.empty((1, 3), np.float32),
                    np.empty(1, np.float32),
                    np.zeros(1, np.int32),
                ],
                tyx_mode,
                cache,
            )
        else:
            values = execute("project_contacts", [1, 1], arrays, tyx_mode, cache)
    assert np.any(values[-1])


def test_reconstruction_preserves_free_velocity_and_updates_only_corrected_rows(tyx_mode, cache):
    arrays = [
        np.ones(2, np.float32),
        np.array([[1e20, 0, 0], [0, 0, 0]], np.float32),
        np.array([[1e20, 0, 0], [0.1, 0, 0]], np.float32),
        np.array([[1e20, 0, 0], [0.2, 0, 0]], np.float32),
        np.array([[1, 0, 0], [1, 0, 0]], np.float32),
        np.zeros(2, np.int32),
    ]
    values = execute("reconstruct_velocity", [2, np.float32(0.1)], arrays, tyx_mode, cache)
    np.testing.assert_allclose(values[-2], [[1, 0, 0], [2, 0, 0]], atol=2e-7)
    np.testing.assert_array_equal(values[-1], 0)
