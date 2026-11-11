"""Independent affine-simplex arithmetic and unchanged query witness gates.

Tests first; the six original Y/HN nodes remain in their original files.
"""

from contextlib import ExitStack
from fractions import Fraction

import numpy as np
import pytest
import test_device_convex_query as k0
import test_device_convex_support_family as family
import tyx as tx

import phymeo as ph
from phymeo.backends._tyx_convex_query_kernels import affine_simplex_coordinate

# Each complete literal carries positive weights; no production output is an oracle.
LITERALS = (
    ("point", ((1 / 4, -1 / 8, 3 / 16),), (1,)),
    ("edge", ((-1 / 2, 1 / 4, -1 / 32), (1 / 2, 1 / 4, -1 / 32)), (3 / 8, 5 / 8)),
    (
        "triangle",
        ((-1 / 2, -1 / 4, -1 / 32), (1 / 2, -1 / 4, -1 / 32), (0, 1 / 2, -1 / 32)),
        (1 / 4, 1 / 4, 1 / 2),
    ),
    (
        "tetrahedron",
        (
            (-1 / 4, -1 / 4, -1 / 4),
            (3 / 4, -1 / 4, -1 / 4),
            (-1 / 4, 3 / 4, -1 / 4),
            (-1 / 4, -1 / 4, 3 / 4),
        ),
        (1 / 4, 1 / 4, 1 / 4, 1 / 4),
    ),
    # Input vertices and pre-normalization weights from captured HN mask13.
    # Expected q is independently evaluated with exact rational normalization.
    (
        "captured-positive-triangle",
        (
            (-0.41770511865615845, -0.20885252952575684, -0.03125),
            (0.07033774256706238, 0.4293552041053772, -0.03125),
            (0.548933207988739, -0.07002487778663635, -0.03125),
        ),
        (0.43814510107040405, 0.26202794909477234, 0.2998269200325012),
    ),
)
FRAMES = {
    "identity": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "cycle": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "oblique": ((3 / 5, 0, 4 / 5), (0, 1, 0), (-4 / 5, 0, 3 / 5)),
}


def exact_convex_point(vertices, weights):
    """Fraction-weighted source vertices; never calls the production helper."""
    w = [Fraction.from_float(float(value)) for value in weights]
    total = sum(w, Fraction())
    assert total > 0 and all(value > 0 for value in w)
    return np.array(
        [
            float(
                sum(
                    (
                        weight * Fraction.from_float(float(vertex[axis]))
                        for weight, vertex in zip(w, vertices, strict=True)
                    ),
                    Fraction(),
                )
                / total
            )
            for axis in range(3)
        ]
    )


def inputs(frame):
    matrix = np.asarray(FRAMES[frame], dtype=float)
    records = []
    for name, points, weights in LITERALS:
        points = np.asarray(points, dtype=float) @ matrix.T
        for reverse in (False, True):
            p = points[::-1] if reverse else points
            w = weights[::-1] if reverse else weights
            records.append((name + ("-reverse" if reverse else ""), p, w))
    # Deliberately poison all unused vertices/weights. The device sample must not
    # read them and the scalar helper must not evaluate unused 0*NaN terms.
    vertices = np.full((len(records), 4, 3), np.nan, dtype=np.float32)
    weights = np.full((len(records), 4), np.nan, dtype=np.float32)
    counts = np.empty(len(records), np.int32)
    for row, (_, point, weight) in enumerate(records):
        n = len(weight)
        counts[row] = n
        vertices[row, :n] = point
        weights[row, :n] = weight
    return records, vertices, weights, counts


def _samples(
    count: tx.i32,
    sizes: tx.ndarray(tx.i32, 1),
    points: tx.ndarray(tx.f32, 3),
    raw_weights: tx.ndarray(tx.f32, 2),
    poison: tx.ndarray(tx.f32, 1),
    output: tx.ndarray(tx.f32, 2),
    normalized: tx.ndarray(tx.f32, 2),
):
    for row in tx.range(count):
        n = sizes[row]
        total = raw_weights[row, 0]
        if n > 1:
            total = total + raw_weights[row, 1]
        if n > 2:
            total = total + raw_weights[row, 2]
        if n > 3:
            total = total + raw_weights[row, 3]
        w1 = poison[row]
        w2 = poison[row]
        w3 = poison[row]
        for k in tx.static_range(4):
            normalized[row, k] = 0.0
            if k < n:
                normalized[row, k] = raw_weights[row, k] / total
        if n > 1:
            w1 = raw_weights[row, 1] / total
        if n > 2:
            w2 = raw_weights[row, 2] / total
        if n > 3:
            w3 = raw_weights[row, 3] / total
        for axis in tx.static_range(3):
            pivot = points[row, 0, axis]
            e1 = poison[row]
            e2 = poison[row]
            e3 = poison[row]
            if n > 1:
                e1 = points[row, 1, axis] - pivot
            if n > 2:
                e2 = points[row, 2, axis] - pivot
            if n > 3:
                e3 = points[row, 3, axis] - pivot
            output[row, axis] = affine_simplex_coordinate(n, pivot, e1, e2, e3, w1, w2, w3)


@pytest.mark.tyx
@pytest.mark.parametrize("frame", tuple(FRAMES))
def test_four_dimensions_positive_weights_and_affine_planes(frame, tyx_mode, tmp_path):
    declarations, vertices, weights, counts = inputs(frame)
    expected = np.stack(
        [exact_convex_point(vertices[row, :n], weights[row, :n]) for row, n in enumerate(counts)]
    )
    with family.saved(
        tmp_path,
        "affine-simplex-" + frame,
        tyx_mode,
        dict(
            declarations=declarations,
            vertices=vertices,
            weights=weights,
            counts=counts,
            expected=expected,
        ),
    ) as record:
        with ph.TyxWorldBatch([(17, k0.ordinary_forest())], tyx_mode, cache_root=tmp_path) as batch:
            with ExitStack() as resources:

                def upload(value):
                    if tyx_mode == "cuda":
                        result = batch._tx.cuda_array(value)
                        resources.callback(result.free)
                        return result
                    return value.copy()

                buffers = (
                    upload(counts),
                    upload(vertices),
                    upload(weights),
                    upload(np.full(len(counts), np.nan, np.float32)),
                    upload(np.zeros((len(counts), 3), np.float32)),
                    upload(np.zeros((len(counts), 4), np.float32)),
                )
                batch._launch(tx.kernel(_samples), len(counts), *buffers)
                batch._sync()
                actual = buffers[-2].to_numpy() if tyx_mode == "cuda" else buffers[-2].copy()
                stored = buffers[-1].to_numpy() if tyx_mode == "cuda" else buffers[-1].copy()
        record.update(actual=actual, stored_weights=stored)
        assert np.isfinite(actual).all() and np.isfinite(stored).all()
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
        for row, n in enumerate(counts):
            assert np.all(stored[row, :n] >= 0) and np.all(stored[row, n:] == 0)
            assert abs(float(stored[row].sum(dtype=np.float64)) - 1) <= 64 * 2**-23
            # Constant affine coordinates are preserved exactly, even for the
            # recorded non-dyadic weights; this is a geometric invariant.
            for axis in range(3):
                if np.all(vertices[row, :n, axis] == vertices[row, 0, axis]):
                    assert actual[row, axis] == vertices[row, 0, axis]
            # Original real-weight route remains an independent witness check.
            material = stored[row, :n].astype(float) @ vertices[row, :n].astype(float)
            assert np.linalg.norm(actual[row] - material) <= 2e-4
        # Reordering vertices/weights together represents the same material point.
        np.testing.assert_allclose(actual[::2], actual[1::2], atol=2e-6, rtol=0)
        # The first three cases are points/edges/triangles in declared planes.
        normal = np.asarray(FRAMES[frame], dtype=float)[:, 2]
        for row in (2, 3, 4, 5, 8, 9):
            assert abs(float((actual[row] - expected[row]) @ normal)) <= 2e-6


@pytest.mark.tyx
@pytest.mark.parametrize(
    "pair,cycle,oblique",
    (("BB", False, False), ("BB", True, False), ("BE", False, True)),
    ids=("face-tie", "rotated-reversed-face-tie", "oblique-curved-witness"),
)
def test_query_material_weights_and_bilateral_witness_are_preserved(
    pair, cycle, oblique, tyx_mode, tmp_path
):
    data = family.reference().case(pair, "sep", cycle=cycle, oblique=oblique)
    with family.saved(
        tmp_path, "affine-query-" + pair + "-" + str(cycle), tyx_mode, data
    ) as record:
        with family.prepared(data, tyx_mode, tmp_path) as (_, _, plan):
            trace = plan.query_with_diagnostics(
                pairs=(tuple(ph.CollisionKey(*key) for key in data["keys"]),)
            )
            record["trace"] = trace
            result = trace.query
            record["checks"] = family.check_result(result, data)
            plan.require_current_diagnostics(trace)
        row = result.rows[0]
        steps = row.gjk_used
        assert steps > 0 and trace.integers[0, steps - 1, 9] in (2, 3)
        q = trace.floats[0, steps - 1, 4:7].astype(float)
        a = np.asarray(row.first.surface_relative_m)
        b = np.asarray(row.second.surface_relative_m)
        assert np.linalg.norm((a - b) - row.scale_m * q) <= 2e-4 * row.scale_m
        # Validate every recorded usable candidate's constant affine coordinates,
        # not just the final result. Never substitute HOST q into the device.
        for iteration in range(steps):
            integers, floats = trace.integers[0, iteration], trace.floats[0, iteration]
            n = int(integers[4])
            vertices = list(floats[32:68].reshape(4, 9)[:n, :3])
            if integers[20]:
                vertices.append(floats[18:21])
            for mask in range(1, 16):
                flags = integers[48 + 3 * (mask - 1) : 51 + 3 * (mask - 1)]
                if not flags[0] or not flags[1]:
                    continue
                chosen = np.array([vertices[k] for k in range(len(vertices)) if mask & (1 << k)])
                candidate = floats[123 + 20 * (mask - 1) : 126 + 20 * (mask - 1)]
                for axis in range(3):
                    if np.all(chosen[:, axis] == chosen[0, axis]):
                        assert candidate[axis] == chosen[0, axis]
