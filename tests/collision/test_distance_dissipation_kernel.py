"""D6-A fixed-candidate tests-first; parent alone executes interpret/CUDA."""

from dataclasses import fields

import numpy as np
import pytest

from phymeo.backends.deformable_dissipation import DistanceDissipation, make_dissipation_step

ATOL, RTOL = 2e-6, 2e-5
MODES = pytest.mark.tyx


@pytest.fixture
def run_kernel(tyx_mode, tmp_path):
    import tyx as tx

    from phymeo.backends._tyx_distance_dissipation import (
        copy_dissipation_velocity,
        dissipate_distances,
    )
    from phymeo.backends.mujoco_tyx._resources import DeviceResources

    resources = DeviceResources()
    runtime = tx.Runtime(
        backend="cuda" if tyx_mode == "cuda" else "cpu", cache_root=tmp_path / "cache"
    )
    resources.set_runtime(runtime)
    calls = []

    def run(x, v, edges, weights, coefficients, h):
        x, v = np.asarray(x, np.float32), np.asarray(v, np.float32)
        edges = np.asarray(edges, np.int32).reshape(-1, 2)
        w, c = np.asarray(weights, np.float32), np.asarray(coefficients, np.float32)
        count, p = len(edges), max(1, len(edges))
        native_edges = np.zeros((p, 2), np.int32)
        native_edges[:count] = edges
        native_c = np.zeros(p, np.float32)
        native_c[:count] = c
        arrays = dict(
            x=x,
            v=v,
            edges=native_edges,
            w=w,
            c=native_c,
            saved=np.zeros_like(v),
            trace=np.zeros((p, 32), np.float32),
            status=np.zeros(1, np.int32),
        )
        device = {}
        with tx.use_runtime(runtime):
            for name, a in arrays.items():
                value = tx.cuda_array(a) if tyx_mode == "cuda" else a.copy()
                device[name] = value
                if tyx_mode == "cuda":
                    resources.add_buffer(value)
            copy_dissipation_velocity(len(v), device["v"], device["saved"], mode=tyx_mode)
            dissipate_distances(
                count,
                np.float32(h),
                device["edges"],
                device["c"],
                device["w"],
                device["x"],
                device["v"],
                device["trace"],
                device["status"],
                mode=tyx_mode,
            )
            runtime.sync()
            out = {
                name: value.to_numpy() if tyx_mode == "cuda" else value.copy()
                for name, value in device.items()
            }
        calls.append(out)
        # Save actual operands and all raw results before any oracle/assertion.
        np.savez(
            tmp_path / f"kernel-{len(calls)}.npz",
            h=np.float32(h),
            count=count,
            input_x=x,
            input_v=v,
            input_weights=w,
            input_coefficients=c,
            **out,
        )
        np.testing.assert_array_equal(out["saved"].view(np.uint32), v.view(np.uint32))
        np.testing.assert_array_equal(out["x"].view(np.uint32), x.view(np.uint32))
        return out["v"], out["trace"][:count], int(out["status"][0])

    try:
        yield run
    finally:
        resources.close()


def oracle(x, v, edges, weights, c, h):
    """Independent FP64 ordered implicit subflows on the fixed candidate."""
    x, current = np.asarray(x, float), np.asarray(v, float).copy()
    w, c = np.asarray(weights, float), np.asarray(c, float)
    impulses, energy = np.zeros_like(current), []
    for (i, j), coefficient in zip(edges, c, strict=True):
        before = current[[i, j]].copy()
        if coefficient:
            d = x[i] - x[j]
            n = d / np.linalg.norm(d)
            rate = n @ (current[i] - current[j])
            impulse = -h * coefficient * rate / (1 + h * coefficient * (w[i] + w[j]))
            impulses[i] += impulse * n
            impulses[j] -= impulse * n
            current[i] += w[i] * impulse * n
            current[j] -= w[j] * impulse * n
        energy.append(
            sum(
                (current[k] @ current[k] - before[a] @ before[a]) / (2 * w[k])
                for a, k in enumerate((i, j))
                if w[k] > 0
            )
        )
    return current, impulses, np.asarray(energy)


def phase(edges, w, c, before, after, trace):
    ids = (101 + 13 * np.arange(len(w), dtype=np.int64))[np.asarray(edges)]
    parameters = DistanceDissipation(ids, np.asarray(c, np.float64))
    return make_dissipation_step(
        parameters,
        np.asarray(edges, np.int32),
        np.asarray(w, np.float32),
        np.asarray(before, np.float32),
        after,
        trace,
    )


def primitive_checks(row):
    """Check stored operands independently, including the retained nonunit g2."""
    f = np.float32

    def add(a, b):
        return f(f(a) + f(b))

    def mul(a, b):
        return f(f(a) * f(b))

    def dot(a, b):
        return add(add(mul(a[0], b[0]), mul(a[1], b[1])), mul(a[2], b[2]))

    n, vi, vj = row[:3], row[6:9], row[9:12]
    g2 = dot(n, n)
    rate = dot(n, np.array([add(a, -b) for a, b in zip(vi, vj, strict=True)]))
    b, a = mul(row[18], row[19]), add(row[20], row[21])
    denominator = add(1, mul(mul(b, a), g2))
    impulse = f(-mul(b, rate) / denominator)
    np.testing.assert_array_equal(row[[3, 4, 23, 27]], [rate, impulse, denominator, g2])
    np.testing.assert_array_equal(row[24:27], [mul(impulse, z) for z in n])
    np.testing.assert_array_equal(row[28:], 0)
    for before, after, weight, sign in (
        (vi, row[12:15], row[20], 1),
        (vj, row[15:18], row[21], -1),
    ):
        expected = [
            add(value, sign * mul(mul(weight, impulse), direction)) if weight else value
            for value, direction in zip(before, n, strict=True)
        ]
        np.testing.assert_array_equal(after, expected)
    actual_rate = dot(
        n, np.array([add(a, -b) for a, b in zip(row[12:15], row[15:18], strict=True)])
    )
    assert row[5] == actual_rate


def half_ulp(value):
    """Bound one RN primitive; floor includes the subnormal rounding region."""
    value = np.asarray(value, np.float32)
    up = np.nextafter(value, np.float32(np.inf)).astype(float) - value.astype(float)
    down = value.astype(float) - np.nextafter(value, np.float32(-np.inf)).astype(float)
    return 0.5 * np.maximum(np.maximum(up, down), float.fromhex("0x1p-149"))


def energy_rounding_bound(row):
    """Each stored endpoint update, plus stored dot-product roundoff; no edge cancellation."""
    n, impulse = row[:3].astype(float), float(row[4])
    s, g2, wi, wj = map(float, row[[3, 27, 20, 21]])
    vi, vj = row[6:9].astype(float), row[9:12].astype(float)
    kinetic_error = 0.0
    for before, after, weight, sign in ((vi, row[12:15], wi, 1), (vj, row[15:18], wj, -1)):
        if weight:
            first = np.float32(weight * impulse)
            second = np.float32(first * n)
            # Propagate each actual primitive's half-ulp, not an arbitrary energy credit.
            err = abs(n) * half_ulp(first) + half_ulp(second) + half_ulp(after)
            ideal = before + sign * weight * impulse * n
            kinetic_error += float(np.sum((abs(ideal) * err + 0.5 * err * err) / weight))
    scalar_error = abs(impulse * (s - n @ (vi - vj)))
    scalar_error += 0.5 * (wi + wj) * impulse * impulse * abs(g2 - n @ n)
    # FP64 audit arithmetic is itself rounded. Bound those sums/products by
    # gamma_32 times their absolute operand scale; this is not an energy credit.
    audit_scale = abs(impulse * s) + 0.5 * (wi + wj) * abs(g2) * impulse * impulse
    for before, after, weight in (
        (vi, row[12:15].astype(float), wi),
        (vj, row[15:18].astype(float), wj),
    ):
        if weight:
            audit_scale += float(np.sum(before * before + after * after) / (2 * weight))
    u = np.finfo(np.float64).eps / 2
    return kinetic_error + scalar_error + (32 * u / (1 - 32 * u)) * audit_scale


@MODES
@pytest.mark.parametrize("h", (1 / 128, 1 / 256))
@pytest.mark.parametrize("sign", (-1, 1))
def test_two_free_masses_once_per_h(run_kernel, h, sign):
    x = np.array([[-0.5, 0, 0], [0.5, 0, 0]], np.float32)
    v = np.array([[0.25, 0, 0], [-0.125, 0, 0]], np.float32) * sign
    edges, w, c = np.array([[0, 1]]), [1, 0.5], [2]
    after, trace, status = run_kernel(x, v, edges, w, c, h)
    assert status == 0
    expected, impulses, _ = oracle(x, v, edges, w, c, h)
    item = phase(edges, w, c, v, after, trace)
    np.testing.assert_allclose(after, expected, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(item.node_impulse, impulses, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(item.node_velocity_impulse.sum(axis=0), 0, atol=ATOL, rtol=0)
    primitive_checks(trace[0])
    assert item.ideal_work[0] <= 0
    assert item.kinetic_change[0] <= energy_rounding_bound(trace[0])
    assert abs(item.rounding_work[0]) <= energy_rounding_bound(trace[0])


@MODES
@pytest.mark.parametrize("reverse", (False, True))
def test_noncollinear_ordered_chain(run_kernel, reverse):
    x = np.array([[0, 0, 0], [1, 1, 1], [-0.5, 1, 1]], np.float32)
    v = np.array([[0.25, -0.125, 0.0625], [-0.25, 0.5, -0.125], [0.125, 0.25, -0.5]], np.float32)
    edges = np.array([[0, 1], [1, 2]], np.int32)[:: (-1 if reverse else 1)]
    w, c, h = [1, 0.5, 0.25], [2, 2], 1 / 128
    after, trace, status = run_kernel(x, v, edges, w, c, h)
    assert status == 0
    item = phase(edges, w, c, v, after, trace)
    expected, impulse, _ = oracle(x, v, edges, w, c, h)
    np.testing.assert_allclose(after, expected, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(item.node_impulse, impulse, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(item.node_impulse.sum(axis=0), 0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(np.cross(x, item.node_impulse).sum(axis=0), 0, atol=ATOL, rtol=0)
    for k, row in enumerate(trace):
        primitive_checks(row)
        assert item.kinetic_change[k] <= energy_rounding_bound(row)
        assert abs(item.rounding_work[k]) <= energy_rounding_bound(row)
    assert np.any(trace[:, 27] != 1), "the frozen nonaxis geometry exercises actual stored g2"
    other, _, _ = oracle(x, v, edges[::-1], w, c, h)
    assert np.max(np.abs(expected - other)) > 1e-8, "ordered pass is not a simultaneous solve"


@MODES
def test_fixed_support_and_both_fixed(run_kernel):
    x = np.array([[0, 0, 0], [1, 0.5, 0]], np.float32)
    v = np.array([[0, 0, 0], [0.25, -0.125, 0]], np.float32)
    after, trace, status = run_kernel(x, v, [[0, 1]], [0, 0.5], [2], 1 / 128)
    assert status == 0
    item = phase([[0, 1]], [0, 0.5], [2], v, after, trace)
    np.testing.assert_array_equal(after[0], 0)
    assert np.any(item.node_impulse[0])
    np.testing.assert_array_equal(item.node_velocity_impulse[0], 0)
    np.testing.assert_array_equal(item.node_impulse[0], -item.node_impulse[1])
    after, trace, status = run_kernel(x, np.zeros_like(v), [[0, 1]], [0, 0], [2], 1 / 128)
    assert status == 0 and trace[0, 4] == 0
    np.testing.assert_array_equal(after, 0)


@MODES
def test_instantaneous_rigid_zero_mode_and_covariance(run_kernel):
    x = np.array([[0, 0, 0], [1, 0.5, 0], [0, 1, 0.5], [0.5, 0, 1]], np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], np.int32)
    w, c = np.array([1, 0.5, 0.25, 2], np.float32), np.full(4, 2)
    rotation = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], np.float32)
    translation, boost = np.array([3, -2, 5]), np.array([0.125, 0.25, -0.125])
    omega = np.array([0.125, -0.25, 0.5])
    v = (np.cross(omega, x) + boost).astype(np.float32)
    out, trace, status = run_kernel(x, v, edges, w, c, 1 / 128)
    assert status == 0
    np.testing.assert_allclose(out, v, atol=ATOL, rtol=0)
    np.testing.assert_allclose(trace[:, [3, 4, 5]], 0, atol=ATOL, rtol=0)
    strained = v.copy()
    strained[2] += np.array([0.125, -0.25, 0.0625])
    base, t0, status = run_kernel(x, strained, edges, w, c, 1 / 128)
    assert status == 0
    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    xp = (x @ rotation.T + translation)[permutation]
    vp = (strained @ rotation.T + boost)[permutation]
    out, t1, status = run_kernel(xp, vp, inverse[edges], w[permutation], c, 1 / 128)
    assert status == 0
    np.testing.assert_allclose(out[inverse], base @ rotation.T + boost, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(t1[:, 24:27], t0[:, 24:27] @ rotation.T, atol=ATOL, rtol=RTOL)


@MODES
@pytest.mark.parametrize("z", (0, 1 / 8))
def test_membrane_axial_null_mode_not_global_drag(run_kernel, z):
    x = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, z]], np.float32)
    v = np.zeros_like(x)
    v[4, 2] = 0.25
    edges = np.array([[4, i] for i in range(4)], np.int32)
    w, c, h = [0, 0, 0, 0, 1], np.full(4, 2), 1 / 128
    out, trace, status = run_kernel(x, v, edges, w, c, h)
    assert status == 0
    expected, _, _ = oracle(x, v, edges, w, c, h)
    np.testing.assert_allclose(out, expected, atol=ATOL, rtol=RTOL)
    coefficient = sum(
        float(ci) * ((x[4, 2] - x[i, 2]) ** 2) / float((x[4] - x[i]) @ (x[4] - x[i]))
        for i, ci in enumerate(c)
    )
    assert coefficient == pytest.approx(4 * 2 * z * z / (2 + z * z), abs=1e-15)
    if z == 0:
        np.testing.assert_array_equal(out, v)
        np.testing.assert_array_equal(trace[:, [3, 4, 5]], 0)


@MODES
@pytest.mark.parametrize(
    "failure",
    (
        "zero-length",
        "nan-x",
        "inf-v",
        "nan-c",
        "negative-c",
        "negative-weight",
        "fixed-moving",
        "bad-h",
        "overflow",
    ),
)
def test_actual_kernel_status_not_silent_zero(failure, run_kernel):
    x = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
    v = np.array([[0.25, 0, 0], [0, 0, 0]], np.float32)
    w, c, h = np.array([1, 0.5], np.float32), np.array([2], np.float32), 1 / 128
    if failure == "zero-length":
        x[1] = x[0]
    elif failure == "nan-x":
        x[0, 0] = np.nan
    elif failure == "inf-v":
        v[0, 0] = np.inf
    elif failure == "nan-c":
        c[0] = np.nan
    elif failure == "negative-c":
        c[0] = -1
    elif failure == "negative-weight":
        w[0] = -1
    elif failure == "fixed-moving":
        w[0] = 0
    elif failure == "bad-h":
        h = 0
    elif failure == "overflow":
        c[0], h = np.finfo(np.float32).max, 2
    _, _, status = run_kernel(x, v, [[0, 1]], w, c, h)
    assert status != 0


@MODES
def test_zero_coefficient_has_no_invented_direction(run_kernel):
    x = np.zeros((2, 3), np.float32)
    v = np.array([[0.25, -0.0, 0], [-0.125, 0, 0]], np.float32)
    after, trace, status = run_kernel(x, v, [[0, 1]], [1, 0.5], [0], 1 / 128)
    assert status == 0
    np.testing.assert_array_equal(after.view(np.uint32), v.view(np.uint32))
    np.testing.assert_array_equal(trace[0, :6], 0)
    np.testing.assert_array_equal(trace[0, 22:], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    item = phase([[0, 1]], [1, 0.5], [0], v, after, trace)
    np.testing.assert_array_equal(item.node_impulse, 0)


@pytest.mark.parametrize(
    "invalid",
    (
        "negative",
        "nan",
        "inf",
        "overflow",
        "underflow",
        "subnormal",
        "shape",
        "ids",
        "self",
        "capacity",
    ),
)
def test_parameter_validation_and_owned_bytes(invalid):
    ids = np.array([[101, 113]], np.int64)
    c = np.array([2.0])
    if invalid == "negative":
        c[0] = -1
    elif invalid == "nan":
        c[0] = np.nan
    elif invalid == "inf":
        c[0] = np.inf
    elif invalid == "overflow":
        c[0] = 1e300
    elif invalid == "underflow":
        c[0] = 1e-300
    elif invalid == "subnormal":
        c[0] = float.fromhex("0x1p-140")
    elif invalid == "shape":
        c = np.array([1, 2.0])
    elif invalid == "ids":
        ids = ids.astype(float)
    elif invalid == "self":
        ids[0, 1] = ids[0, 0]
    elif invalid == "capacity":
        ids, c = np.tile(ids, (1537, 1)), np.ones(1537)
    with pytest.raises((ValueError, TypeError)):
        DistanceDissipation(ids, c)


def test_parameters_own_original_double_and_zero_edge():
    ids = np.array([[101, 113]], np.int64)
    c = np.array([1.0 + 2**-30], np.float64)
    item = DistanceDissipation(ids, c)
    ids[:] = 7
    c[:] = 0
    assert item.coefficients_ns_per_m[0] == 1 + 2**-30
    np.testing.assert_array_equal(item.ordered_point_ids, [[101, 113]])
    for a in (item.ordered_point_ids, item.coefficients_ns_per_m):
        with pytest.raises(ValueError):
            a.setflags(write=True)
    empty = DistanceDissipation(np.empty((0, 2), np.int64), np.empty(0, np.float64))
    assert empty.ordered_point_ids.shape == (0, 2)


@MODES
def test_observer_complete_chain_is_read_only_and_rejects_corruption(run_kernel):
    x = np.array([[0, 0, 0], [1, 2, 0.5], [-0.5, 1, 1]], np.float32)
    v = np.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]], np.float32)
    edges, w, c = np.array([[0, 1], [1, 2]], np.int32), np.array([1, 0.5, 0.25], np.float32), [2, 2]
    after, trace, status = run_kernel(x, v, edges, w, c, 1 / 128)
    assert status == 0
    before_copy, after_copy, trace_copy = v.copy(), after.copy(), trace.copy()
    item = phase(edges, w, c, v, after, trace)
    assert item.schema == "distance-dissipation-step-v1"
    np.testing.assert_array_equal(item.kinetic_change - item.ideal_work, item.rounding_work)
    for field in fields(item):
        value = getattr(item, field.name)
        if isinstance(value, np.ndarray):
            with pytest.raises(ValueError):
                value.setflags(write=True)
    np.testing.assert_array_equal(v, before_copy)
    np.testing.assert_array_equal(after, after_copy)
    np.testing.assert_array_equal(trace, trace_copy)
    for location in ((1, 6), (0, 19), (0, 20), (0, 24), (0, 27), (0, 28), (0, 23)):
        broken = trace.copy()
        broken[location] += np.float32(0.125)
        with pytest.raises(ValueError):
            phase(edges, w, c, v, after, broken)
    broken = after.copy()
    broken[2, 0] += 0.125
    with pytest.raises(ValueError):
        phase(edges, w, c, v, broken, trace)


@MODES
def test_empty_edge_padding_is_not_an_extra_material_step(run_kernel):
    x = np.array([[1, 2, 3]], np.float32)
    v = np.array([[0.25, -0.0, -0.125]], np.float32)
    edges = np.empty((0, 2), np.int32)
    after, trace, status = run_kernel(x, v, edges, [1], [], 1 / 128)
    assert status == 0 and trace.shape == (0, 32)
    np.testing.assert_array_equal(after.view(np.uint32), v.view(np.uint32))
    item = phase(edges, [1], [], v, after, trace)
    assert item.kinetic_change.shape == (0,)
    np.testing.assert_array_equal(item.node_impulse, 0)
    np.testing.assert_array_equal(item.node_velocity_impulse, 0)
