"""Retained-candidate stagnation publication; tests first, UNEXECUTED.

The nine original inputs and independent query/patch gates are reused verbatim.
Synthetic predicate inputs below are branch tests, not physical trajectories.
"""

import importlib.util
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import test_device_convex_patch_qualification as qualification
import test_device_convex_query as k0
import test_device_convex_query_trace as trace_protocol
import test_device_convex_support_family as family
import tyx as tx

import phymeo as ph
from phymeo.backends._tyx_convex_query_kernels import retained_candidate_fixed

IDENTITIES = (
    "K0-Q01-CB-sep",
    "K0-Q01-CB-overlap",
    "K0-Q01-BH-sep",
    "K0-Q03-CB-reverse",
    "K0-Q03-CB-cycle",
    "K0-Q03-CB-tilt",
    "K0-Q03-CB-permute",
    "K0-Q03-CB-cycle_reverse_ids",
    "cb_cycle_reverse_ids_segment",
)


@lru_cache(maxsize=1)
def frozen_cases():
    path = (
        Path(__file__).resolve().parents[1]
        / "reference_newton/rr/device_convex_stagnation_trace.py"
    )
    spec = importlib.util.spec_from_file_location("col_frozen_stagnation_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.IDENTITIES == IDENTITIES
    return module


def predicate_inputs():
    # i: duplicate, appended, entry_count, reduced_count, selected[4].
    # f: entry(q,square), reduced(q,square), entry weights[4], reduced weights[4].
    names, integer_rows, float_rows, expected = [], [], [], []

    def add(name, integers, floats, answer):
        names.append(name)
        integer_rows.append(np.asarray(integers, dtype=np.int32).copy())
        float_rows.append(np.asarray(floats, dtype=np.float32).copy())
        expected.append(answer)

    for count, weights in (
        (1, (1,)),
        (2, (1 / 2, 1 / 2)),
        (3, (1 / 4, 1 / 4, 1 / 2)),
        (4, (1 / 4, 1 / 4, 1 / 4, 1 / 4)),
    ):
        integers = np.array([1, 0, count, count, -777, -777, -777, -777], np.int32)
        floats = np.full(16, np.nan, np.float32)
        integers[4 : 4 + count] = np.arange(count)
        floats[:4] = floats[4:8] = (1 / 4, -1 / 8, 1 / 16, 21 / 256)
        floats[8 : 8 + count] = weights
        floats[12 : 12 + count] = weights
        add("fixed-count-" + str(count), integers, floats, 1)
    base_i, base_f = integer_rows[2].copy(), float_rows[2].copy()
    zero = base_f.copy()
    zero[8:11] = zero[12:15] = (1 / 2, 1 / 2, 0)
    add("fixed-clamped-zero-weight", base_i, zero, 1)

    for name in (
        "not-duplicate", "appended", "count-dropped", "count-added",
        "empty", "count-overflow", "same-square-different-q", "different-square",
        "different-normalized-weights", "different-order", "active-nan-q",
        "active-nan-weight", "infinite-q", "negative-weight",
    ):
        integers, floats = base_i.copy(), base_f.copy()
        if name == "not-duplicate":
            integers[0] = 0
        elif name == "appended":
            integers[1] = 1
        elif name == "count-dropped":
            integers[3] = 2
        elif name == "count-added":
            integers[3] = 4
        elif name == "empty":
            integers[2:4] = 0
        elif name == "count-overflow":
            integers[2:4] = 5
        elif name == "same-square-different-q":
            floats[4] = -floats[0]
        elif name == "different-square":
            floats[7] = 22 / 256
        elif name == "different-normalized-weights":
            floats[12:15] = (1 / 8, 3 / 8, 1 / 2)
        elif name == "different-order":
            integers[4:6] = (1, 0)
        elif name == "active-nan-q":
            floats[0] = floats[4] = np.nan
        elif name == "active-nan-weight":
            floats[8] = floats[12] = np.nan
        elif name == "infinite-q":
            floats[0] = floats[4] = np.inf
        else:
            floats[8:11] = floats[12:15] = (-1 / 4, 1 / 4, 1)
        add(name, integers, floats, 0)
    return names, np.stack(integer_rows), np.stack(float_rows), np.asarray(expected, np.int32)


def _predicate_samples(
    count: tx.i32,
    integers: tx.ndarray(tx.i32, 2),
    floats: tx.ndarray(tx.f32, 2),
    actual: tx.ndarray(tx.i32, 1),
):
    for row in tx.range(count):
        actual[row] = retained_candidate_fixed(
            integers[row, 0], integers[row, 1], integers[row, 2], integers[row, 3],
            floats[row, 0], floats[row, 1], floats[row, 2], floats[row, 3],
            floats[row, 4], floats[row, 5], floats[row, 6], floats[row, 7],
            floats[row, 8], floats[row, 9], floats[row, 10], floats[row, 11],
            floats[row, 12], floats[row, 13], floats[row, 14], floats[row, 15],
            integers[row, 4], integers[row, 5], integers[row, 6], integers[row, 7],
        )


@pytest.mark.tyx
def test_actual_shared_fixed_candidate_predicate(tyx_mode, tmp_path):
    names, integers, floats, expected = predicate_inputs()
    with family.saved(
        tmp_path, "stagnation-shared-predicate", tyx_mode,
        dict(names=names, integers=integers, floats=floats, expected=expected),
    ) as record:
        with ph.TyxWorldBatch([(17, k0.ordinary_forest())], tyx_mode, cache_root=tmp_path) as batch:
            with ExitStack() as resources:
                def upload(value):
                    if tyx_mode == "cuda":
                        buffer = batch._tx.cuda_array(value)
                        resources.callback(buffer.free)
                        return buffer
                    return value.copy()

                buffers = (
                    upload(integers), upload(floats),
                    upload(np.full(len(names), -1, np.int32)),
                )
                batch._launch(tx.kernel(_predicate_samples), len(names), *buffers)
                batch._sync()
                actual = buffers[-1].to_numpy() if tyx_mode == "cuda" else buffers[-1].copy()
        record["actual"] = actual
        np.testing.assert_array_equal(actual, expected)


def assert_fixed_publication(trace):
    row = trace.query.rows[0]
    assert row.status == "RESOLVED" and row.epa_used == 0
    assert row.gjk_used <= trace.policy["gjk_iterations"]
    assert row.support_calls == row.gjk_used
    i = trace.integers[0, row.gjk_used - 1]
    f = trace.floats[0, row.gjk_used - 1]
    n = int(i[4])
    assert 1 <= n <= 4 and tuple(i[[5, 7, 21]]) == (n, n, n)
    assert i[6] == 1 and i[20] == 0 and i[9] == 3
    assert i[8] == (1 << n) - 1  # A real unchanged full-subset reduction occurred.
    np.testing.assert_array_equal(f[:4], f[4:8])
    before_vertices, after_vertices = f[32:68].reshape(4, 9), f[72:108].reshape(4, 9)
    np.testing.assert_array_equal(before_vertices[:n], after_vertices[:n])
    np.testing.assert_array_equal(f[68 : 68 + n], f[108 : 108 + n])
    np.testing.assert_array_equal(i[24 : 24 + 2 * n], i[36 : 36 + 2 * n])
    np.testing.assert_array_equal(i[32 : 32 + n], np.arange(n))
    np.testing.assert_array_equal(i[44 : 44 + n], np.arange(n))
    assert f[21] > 0 and f[22] == np.float32(f[21] - f[11])
    # This is the original engineering stagnation route, NOT a plane certificate.
    # In all nine frozen captures the real width misses the unchanged plane gate.
    assert f[22] > np.float32(8e-7)
    assert row.gap_lower_m < row.gap_upper_m and row.gap_m == row.gap_upper_m

    weights = f[68 : 68 + n].astype(float)
    assert np.all(weights >= 0) and abs(float(weights.sum()) - 1) <= 64 * 2**-23
    q = f[:3].astype(float)
    normal = -q / np.linalg.norm(q)
    np.testing.assert_allclose(row.normal, normal, atol=2e-4, rtol=0)
    source = trace.query.source
    slots = i[24:32].reshape(4, 2)[:n]
    radii = source["uploaded_radius_m"][trace.shape_rows[0]].astype(float)
    material = []
    for side, witness in enumerate((row.first, row.second)):
        expected_weights = {}
        for weight, slot in zip(weights, slots[:, side], strict=True):
            if weight > 0:
                key = int(source["vertex_ids"][int(slot)])
                expected_weights[key] = expected_weights.get(key, 0.0) + weight
        assert tuple(witness.vertex_ids) == tuple(sorted(expected_weights))
        np.testing.assert_allclose(
            witness.weights, [expected_weights[key] for key in witness.vertex_ids],
            atol=64 * 2**-23, rtol=0,
        )
        core = weights @ before_vertices[:n, 3 + 3 * side : 6 + 3 * side].astype(float)
        skin = (1 if side == 0 else -1) * radii[side] * normal
        np.testing.assert_allclose(witness.offset_m, skin, atol=2e-4 * row.scale_m, rtol=0)
        point = core + skin
        error = float(np.linalg.norm(point - witness.surface_relative_m))
        assert error <= 2e-4 * row.scale_m
        material.append(dict(expected_weights=expected_weights, core=core, skin=skin, point=point, error_m=error))
    a = np.asarray(row.first.surface_relative_m) - row.first.offset_m
    b = np.asarray(row.second.surface_relative_m) - row.second.offset_m
    core_q_error = float(np.linalg.norm(a - b - row.scale_m * q))
    assert core_q_error <= 2e-4 * row.scale_m

    # Check the honest stored bracket against its current q/support operands.
    # This rounding allowance checks the FP32 wire arithmetic, not geometry.
    upper = float(f[21]) * row.scale_m - float(radii.sum())
    lower = min(upper, float(f[11]) * row.scale_m - float(radii.sum()))
    arithmetic_roundoff = 4 * 2**-23 * (
        row.scale_m * (abs(float(f[21])) + abs(float(f[11]))) + 2 * float(radii.sum())
    )
    np.testing.assert_allclose(
        (row.gap_lower_m, row.gap_upper_m), (lower, upper),
        atol=arithmetic_roundoff, rtol=0,
    )
    return dict(
        entry_q=f[:4], retained_q=f[4:8], weights=weights, selected_mask=int(i[8]),
        decision=int(i[9]), width=float(f[22]), current_bound=float(f[11]),
        published_bracket=(row.gap_lower_m, row.gap_upper_m), expected_bracket=(lower, upper),
        arithmetic_roundoff_m=arithmetic_roundoff, material=material, core_q_error_m=core_q_error,
    )


@pytest.mark.tyx
@pytest.mark.parametrize("identity", IDENTITIES)
def test_nine_original_regressions_keep_one_complete_candidate(identity, tyx_mode, tmp_path):
    frozen = frozen_cases()
    case, expected_patch, original_node = frozen.declaration(identity, k0, qualification)
    with k0.saved(tmp_path, "fixed-" + identity, tyx_mode, case) as record:
        record.update(original_node=original_node, expected_patch=expected_patch)
        record["independent_literal_oracle"] = k0.reference().literal_parallel_precondition(case)
        with frozen.prepared(case, expected_patch, tyx_mode, tmp_path, k0, qualification) as (
            batch, broad, plan, before,
        ):
            record["before"] = before
            keys = tuple(ph.CollisionKey(*key) for key in case["keys"])
            model, bindings = k0.registrations(case, before[keys[0].world].topology)
            record.update(model=model, bindings=bindings)
            plain = plan.query()
            record["plain"] = plain
            trace = plan.query_with_diagnostics(pairs=(keys,))
            record["trace"] = trace
            trace_protocol.same_geometry(plain, trace.query)
            trace_protocol.assert_trace(trace, plan, 1)
            k0.check_raw(trace.query)
            k0.check_source(trace.query, model, bindings, before, case=case)
            if expected_patch is None:
                record["geometry"] = k0.check_case_row(trace.query.rows[0], case, trace.query.source)
            else:
                qualification._affected_patch_geometry(trace.query.rows[0], expected_patch)
                normal = np.asarray(trace.query.rows[0].normal)
                assert abs(np.linalg.norm(normal) - 1) <= 2e-4
                assert np.linalg.norm(normal - expected_patch["normal"]) <= 2e-3
                record["geometry"] = dict(original_complete_patch_oracle=True)
            record["candidate"] = assert_fixed_publication(trace)
            record["after"] = batch.snapshot()
            assert k0.canonical(before) == k0.canonical(record["after"])
            record["stats"] = dict(plan.stats)
            record["broad_stats"] = broad.stats()
