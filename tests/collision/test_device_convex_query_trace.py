"""Bounded opt-in GJK trace protocol; no new geometry acceptance threshold."""

import copy
import json
from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest
import test_device_convex_query as k0
import test_device_convex_support_family as family

import phymeo as ph
from phymeo.backends._device_convex_query_trace import (
    FLOAT_WIDTH,
    INT_WIDTH,
    SCHEMA,
    _TraceCapture,
    normalize_trace_pairs,
)


def selected(plan):
    return tuple(row.pair for row in plan.last_result.rows)


def same_geometry(a, b):
    assert a.shape_keys == b.shape_keys and a.shape_revisions == b.shape_revisions
    assert a.rows == b.rows
    for name in a.raw:
        left, right = np.array(a.raw[name]), np.array(b.raw[name])
        if name == "meta":
            left[:, 12] = 0
            right[:, 12] = 0
        np.testing.assert_array_equal(left, right)


def assert_trace(trace, plan, count):
    assert trace.schema == SCHEMA
    assert plan.require_current_diagnostics(trace) is trace
    assert plan.last_result is trace.query and plan.last_diagnostics is trace
    assert trace.floats.shape == (count, plan._gjk, FLOAT_WIDTH)
    assert trace.integers.shape == (count, plan._gjk, INT_WIDTH)
    assert trace.floats.dtype == np.float32 and trace.integers.dtype == np.int32
    assert trace.shape_rows.shape == (count, 2)
    assert trace.stats["owned_bytes"] == count * plan._gjk * 512 * 4 + count * 2 * 4
    assert trace.stats["readback_bytes"] == trace.stats["owned_bytes"]
    assert trace.stats["clear_launches"] == 1
    assert trace.stats["allocations"] == 3
    assert trace.stats["host_upload_bytes"] == count * 8
    assert trace.stats["uploaded_bytes"] == (count * 8 if plan._batch._mode == "cuda" else 0)

    assert trace.stats["host_read_bytes"] == trace.stats["readback_bytes"]
    assert trace.stats["downloads"] == (3 if plan._batch._mode == "cuda" else 0)
    for array in (trace.floats, trace.integers, trace.shape_rows, trace.pair_rows):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    for j, row_index in enumerate(trace.pair_rows):
        row = trace.query.rows[int(row_index)]
        assert row.pair == trace.pairs[j]
        np.testing.assert_array_equal(
            trace.shape_rows[j], [trace.query.shape_keys.index(key) for key in row.pair]
        )
        used = int(trace.query.raw["meta"][int(row_index), 4])
        assert np.all(trace.integers[j, :used, 0] == 1)
        assert np.all(trace.integers[j, used:] == 0)
        assert np.all(trace.floats[j, used:] == 0)
        if used:
            np.testing.assert_array_equal(trace.integers[j, :used, 3], np.arange(used))
            assert np.all(trace.integers[j, :used, 1] == trace.query.attempt_tag)
            assert np.all(trace.integers[j, :used, 2] == row_index)
            assert trace.integers[j, used - 1, 9] in range(1, 9)
            assert np.all(trace.integers[j, :used, 4:6] >= 0)
            assert np.all(trace.integers[j, :used, 4:6] <= 4)


@pytest.mark.tyx
@pytest.mark.parametrize("identity", ("BE-sep", "SE-overlap"))
def test_trace_off_on_default_geometry_and_accounting(identity, tyx_mode, tmp_path):
    name, phase = identity.split("-")
    data = family.reference().case(name, phase)
    with family.prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
        plain = plan.query()
        baseline = dict(plan.stats)
        trace = plan.query_with_diagnostics(pairs=selected(plan))
        assert_trace(trace, plan, 1)
        same_geometry(plain, trace.query)
        assert plan.stats["live_bytes"] == baseline["live_bytes"]
        assert trace.stats["released"] is True
        off_before = dict(plan.stats)
        again = plan.query()
        same_geometry(trace.query, again)
        assert plan.last_diagnostics is trace
        assert plan.stats["allocations"] == off_before["allocations"]
        assert (
            again.stats["attempt_delta"]["query_launches"]
            == plain.stats["attempt_delta"]["query_launches"]
        )


@pytest.mark.tyx
def test_eight_selected_pairs_are_bounded_and_query_order_bound(tyx_mode, tmp_path):
    declarations = []
    for world in (101, 7, 43, 29, 211, 59, 3, 83):
        data = copy.deepcopy(family.reference().case("BE", "sep"))
        data["world"] = world
        data["keys"] = tuple((world, key[1], key[2]) for key in data["keys"])
        declarations.append(data)
    trees = [
        (d["world"], k0.ordinary_forest(anchor=d["centers"][0], orientation=d["body_quaternion"]))
        for d in declarations
    ]
    with ph.TyxWorldBatch(trees, tyx_mode, cache_root=tmp_path) as batch:
        shapes, bindings, pairs = [], [], []
        states = batch.snapshot()
        for data in declarations:
            a, b = family.shape_for(data, 0), family.shape_for(data, 1)
            shapes.extend((a, b))
            bindings.extend(
                (
                    ph.ResidentTreeBinding(
                        a.key,
                        data["world"],
                        0,
                        data["local"][0],
                        topology=states[data["world"]].topology,
                    ),
                    ph.ResidentStaticBinding(b.key, ph.CollisionMotion(data["points"][1])),
                )
            )
            pairs.append((a.key, b.key))
        with batch.prepare_collision(ph.CollisionModel(tuple(shapes)), tuple(bindings)) as broad:
            with ExitStack() as stack:
                plan = batch.prepare_convex_query(broad, pairs=tuple(pairs))
                stack.callback(plan.close)
                trace = plan.query_with_diagnostics(pairs=tuple(reversed(pairs)))
                assert_trace(trace, plan, 8)
                assert trace.pairs == tuple(tuple(sorted(pair)) for pair in reversed(pairs))
                assert tuple(trace.pair_rows) != tuple(range(8))


@pytest.mark.parametrize("bad", ("empty", "duplicate", "nine", "malformed"))
def test_trace_selection_host_validation(bad):
    a, b = ph.CollisionKey(1, 2, 3), ph.CollisionKey(1, 2, 4)
    values = dict(
        empty=(),
        duplicate=((a, b), (b, a)),
        nine=tuple((ph.CollisionKey(i, 2, 3), ph.CollisionKey(i, 2, 4)) for i in range(9)),
        malformed=((a,),),
    )
    with pytest.raises((ValueError, TypeError)):
        normalize_trace_pairs(values[bad])
    if bad == "malformed":
        raw = np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)
        encoded = k0.encode(raw)
        json.dumps(encoded, allow_nan=False)
        assert encoded["bytes_hex"] == raw.tobytes().hex()
        assert encoded["values"] == [
            {"nonfinite": "nan"},
            {"nonfinite": "inf"},
            {"nonfinite": "-inf"},
        ]


@pytest.mark.tyx
def test_bad_or_nonmember_pair_preserves_both_results(tyx_mode, tmp_path):
    data = family.reference().case("BE", "sep")
    with family.prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
        first = plan.query_with_diagnostics(
            pairs=((ph.CollisionKey(*data["keys"][0]), ph.CollisionKey(*data["keys"][1])),)
        )
        old_broad = broad.last_result
        p = first.pairs[0]
        for bad in ((), (p, p), ((p[0], ph.CollisionKey(p[0].world, 999, 999)),)):
            with pytest.raises((ValueError, TypeError)):
                plan.query_with_diagnostics(pairs=bad)
            assert plan.last_result is first.query
            assert plan.last_diagnostics is first and broad.last_result is old_broad
        # A legal registered pair can still be absent from this query membership.
        with ExitStack() as stack:
            empty = batch.prepare_convex_query(broad, pairs=())
            stack.callback(empty.close)
            no_rows = empty.query()
            with pytest.raises(ValueError, match="not present"):
                empty.query_with_diagnostics(pairs=(p,))
            assert empty.last_result is no_rows and empty.last_diagnostics is None


@pytest.mark.tyx
@pytest.mark.parametrize("short", (False, True))
def test_temporary_owned_capacity_is_exact_and_freed(short, tyx_mode, tmp_path):
    data = family.reference().case("BE", "sep")
    with family.prepared(data, tyx_mode, tmp_path / "measure") as (_, _, initial):
        resident = initial.stats["live_bytes"]
    needed = 64 * 512 * 4 + 8
    with family.prepared(
        data, tyx_mode, tmp_path / "bounded", memory_limit_bytes=resident + needed - int(short)
    ) as (_, _, plan):
        plain = plan.query()
        before = dict(plan.stats)
        if short:
            with pytest.raises(MemoryError):
                plan.query_with_diagnostics(pairs=selected(plan))
            assert plan.last_result is plain and plan.last_diagnostics is None
            assert plan.stats["allocations"] == before["allocations"]
        else:
            trace = plan.query_with_diagnostics(pairs=selected(plan))
            assert_trace(trace, plan, 1)
        assert plan.stats["live_bytes"] == resident


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("read", "construct", "allocation"))
def test_late_trace_failure_is_atomic_and_retryable(fault, tyx_mode, tmp_path, monkeypatch):
    data = family.reference().case("BE", "sep")
    with family.prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
        first = plan.query_with_diagnostics(
            pairs=((ph.CollisionKey(*data["keys"][0]), ph.CollisionKey(*data["keys"][1])),)
        )
        old_broad, live = broad.last_result, plan.stats["live_bytes"]
        with monkeypatch.context() as patch:
            if fault == "allocation":
                original, calls = plan._allocate, [0]

                def fail_allocate(*args, **kwargs):
                    calls[0] += 1
                    if calls[0] == 2:
                        raise RuntimeError("injected trace allocation")
                    return original(*args, **kwargs)

                patch.setattr(plan, "_allocate", fail_allocate)
            else:
                name = "readback" if fault == "read" else "build"
                original = getattr(_TraceCapture, name)

                def fail_after(self, *args, **kwargs):
                    original(self, *args, **kwargs)
                    raise RuntimeError("injected late trace " + fault)

                patch.setattr(_TraceCapture, name, fail_after)
            with pytest.raises(RuntimeError, match="injected"):
                plan.query_with_diagnostics(pairs=first.pairs)
        assert plan.last_result is first.query
        assert plan.last_diagnostics is first and broad.last_result is old_broad
        assert plan.stats["live_bytes"] == live
        retry = plan.query_with_diagnostics(pairs=first.pairs)
        assert_trace(retry, plan, 1)
        same_geometry(first.query, retry.query)


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("owner", "stale", "binding"))
def test_trace_source_owner_epoch_and_policy_binding(fault, tyx_mode, tmp_path):
    data = family.reference().case("BE", "sep")
    with family.prepared(data, tyx_mode, tmp_path) as (batch, _, plan):
        trace = plan.query_with_diagnostics(
            pairs=((ph.CollisionKey(*data["keys"][0]), ph.CollisionKey(*data["keys"][1])),)
        )
        if fault == "stale":
            assert all(batch.step(1 / 1024).values())
            with pytest.raises(ValueError, match="stale"):
                plan.require_current_diagnostics(trace)
        elif fault == "owner":
            with family.prepared(data, tyx_mode, tmp_path / "other") as (_, _, other):
                with pytest.raises(ValueError, match="owner"):
                    other.require_current_diagnostics(trace)
        else:
            bad_frame = trace.floats.copy()
            bad_frame[0, 0, 412] += np.float32(1)
            forged = (
                replace(trace, floats=bad_frame),
                replace(trace, pairs=tuple((b, a) for a, b in trace.pairs)),
                replace(trace, policy=dict(trace.policy, gjk_iterations=1)),
                replace(trace, pair_rows=np.full_like(trace.pair_rows, -1)),
            )
            for item in forged:
                with pytest.raises(ValueError):
                    plan.require_current_diagnostics(item)


@pytest.mark.tyx
def test_analytic_round_query_has_zero_history_and_bound_terminal(tyx_mode, tmp_path):
    # Equal-radius ellipsoids still run GJK; use the original SS literal instead.
    data = family.reference().case("SE", "sep")
    data = copy.deepcopy(data)
    data["kinds"] = ("S", "S")
    data["radii"] = (0.125, 0.125)
    with family.prepared(data, tyx_mode, tmp_path) as (_, _, plan):
        plain = plan.query()
        trace = plan.query_with_diagnostics(pairs=selected(plan))
        same_geometry(plain, trace.query)
        assert_trace(trace, plan, 1)
        assert not np.any(trace.integers) and not np.any(trace.floats)
        assert int(trace.query.raw["meta"][0, 4]) == 0


@pytest.mark.tyx
def test_closed_plan_rejects_trace_without_allocation(tyx_mode, tmp_path):
    data = family.reference().case("BE", "sep")
    with family.prepared(data, tyx_mode, tmp_path) as (_, _, plan):
        plain = plan.query()
        pairs = selected(plan)
        plan.close()
        with pytest.raises(RuntimeError, match="closed"):
            plan.query_with_diagnostics(pairs=pairs)
        assert plan.last_result is plain and plan.last_diagnostics is None
