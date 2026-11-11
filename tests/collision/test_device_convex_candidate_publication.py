"""Same-candidate separation publication; independent geometry gates stay fixed."""

from dataclasses import replace

import numpy as np
import pytest
import test_device_convex_query as k0
import test_device_convex_query_trace as trace_protocol
import test_device_convex_support_family as family

import phymeo as ph


def selected(data):
    return (tuple(ph.CollisionKey(*key) for key in data["keys"]),)


def assert_same_candidate(trace, data):
    """Read actual trace state and independently check its published witnesses."""
    result = trace.query
    checks = family.check_result(result, data)
    row = result.rows[0]
    used = row.gjk_used
    assert used > 1 and row.epa_used == 0
    ints = trace.integers[0, used - 1]
    floats = trace.floats[0, used - 1]
    n = int(ints[4])
    assert 1 <= n <= 4 and ints[5] == n and ints[7] == n
    assert ints[9] == 2 and ints[20] == 0 and ints[21] == n
    # Zero mask is deliberate: this accepting iteration performs no reduction.
    assert ints[8] == 0
    np.testing.assert_array_equal(ints[48:93], 0)
    np.testing.assert_array_equal(floats[112:412], 0)
    np.testing.assert_array_equal(floats[:4], floats[4:8])
    np.testing.assert_array_equal(floats[32:68], floats[72:108])
    np.testing.assert_array_equal(floats[68:72], floats[108:112])
    np.testing.assert_array_equal(ints[24:24 + 2*n], ints[36:36 + 2*n])
    np.testing.assert_array_equal(ints[32:32 + n], ints[44:44 + n])
    assert floats[21] > 0 and floats[22] <= np.float32(8e-7)
    assert floats[22] == np.float32(floats[21] - floats[11])
    # duplicate is diagnostic only; it cannot authorize a wider plane gap.
    assert ints[6] in (0, 1)
    assert np.all(floats[68:68 + n] >= 0)
    assert abs(float(floats[68:68 + n].sum(dtype=np.float64)) - 1) <= 64*2**-23

    q = floats[:3].astype(float)
    normal = -q / np.linalg.norm(q)
    np.testing.assert_allclose(row.normal, normal, atol=2e-4, rtol=0)
    vertices = floats[32:68].reshape(4, 9)[:n].astype(float)
    weights = floats[68:68 + n].astype(float)
    source_ids = result.source["vertex_ids"]
    slots = ints[24:32].reshape(4, 2)[:n]
    material_checks = []
    for side, witness in enumerate((row.first, row.second)):
        expected_weights = {}
        for weight, slot in zip(weights, slots[:, side], strict=True):
            if weight > 0:
                original_id = int(source_ids[int(slot)])
                expected_weights[original_id] = expected_weights.get(original_id, 0.0) + weight
        assert tuple(witness.vertex_ids) == tuple(sorted(expected_weights))
        np.testing.assert_allclose(
            witness.weights, [expected_weights[key] for key in witness.vertex_ids],
            atol=64*2**-23, rtol=0,
        )
        # For the chosen BB/BE cases the trace support representation is the
        # published witness frame (zero old-core skin, or full-surface support).
        expected_point = weights @ vertices[:, 3 + 3*side:6 + 3*side]
        actual_point = np.asarray(witness.surface_relative_m)
        error = float(np.linalg.norm(expected_point - actual_point))
        assert error <= 2e-4*row.scale_m
        material_checks.append(dict(expected_weights=expected_weights,
                                    expected_point=expected_point, error_m=error))
    difference = np.asarray(row.first.surface_relative_m) - row.second.surface_relative_m
    q_witness_error = float(np.linalg.norm(difference - row.scale_m*q))
    assert q_witness_error <= 2e-4*row.scale_m
    return dict(geometry=checks, retained_material=material_checks,
                q_witness_error_m=q_witness_error)


@pytest.mark.tyx
@pytest.mark.parametrize(
    "pair,cycle,oblique",
    (("BB", False, False), ("BB", True, False), ("BE", False, True)),
    ids=("finite-face", "rotated-reversed-face", "oblique-curved"),
)
def test_same_candidate_normal_and_bilateral_material_publish_together(
    pair, cycle, oblique, tyx_mode, tmp_path
):
    data = family.reference().case(pair, "sep", cycle=cycle, oblique=oblique)
    with family.saved(tmp_path, "candidate-" + pair + "-" + str(cycle), tyx_mode, data) as record:
        with family.prepared(data, tyx_mode, tmp_path) as (batch, _, plan):
            before = batch.snapshot()
            plain = plan.query()
            record["plain"] = plain
            trace = plan.query_with_diagnostics(pairs=selected(data))
            record["trace"] = trace
            trace_protocol.same_geometry(plain, trace.query)
            trace_protocol.assert_trace(trace, plan, 1)
            record["checks"] = assert_same_candidate(trace, data)
            assert k0.canonical(before) == k0.canonical(batch.snapshot())


@pytest.mark.tyx
@pytest.mark.parametrize("budget", (1, 2))
def test_exact_duplicate_requires_same_candidate_plane_and_existing_budget(
    budget, tyx_mode, tmp_path
):
    # Axis-aligned identical finite boxes: the first support gives the exact
    # separating face vector. A second call repeats it and certifies that SAME q.
    # One call has no pre-existing candidate to certify; no extra call is hidden.
    data = family.reference().case("BB", "sep")
    with family.saved(tmp_path, "candidate-budget-" + str(budget), tyx_mode, data) as record:
        with family.prepared(data, tyx_mode, tmp_path, gjk_iterations=budget) as (batch, _, plan):
            before = batch.snapshot()
            trace = plan.query_with_diagnostics(pairs=selected(data))
            record["trace"] = trace
            trace_protocol.assert_trace(trace, plan, 1)
            row = trace.query.rows[0]
            assert row.gjk_used == budget and row.support_calls == budget
            assert row.epa_used == 0 and k0.canonical(before) == k0.canonical(batch.snapshot())
            if budget == 1:
                family.check_raw(trace.query)
                assert row.status == "RESOURCE_LIMIT" and row.reason == "GJK_LIMIT"
                assert not row.normal_valid and not row.witness_valid
                assert trace.integers[0, 0, 9] == 8
                assert trace.integers[0, 0, 4] == 0
                assert trace.integers[0, 0, 20] == 1
            else:
                record["checks"] = assert_same_candidate(trace, data)
                assert trace.integers[0, 1, 6] == 1
                assert trace.integers[0, 1, 20] == 0


@pytest.mark.tyx
def test_independent_validator_rejects_mixed_candidate_and_duplicate_excuse(tyx_mode, tmp_path):
    # These are evidence-validator forgeries, NOT claimed physical trajectories
    # and not mocks of the product support function or solver.
    data = family.reference().case("BB", "sep")
    with family.saved(tmp_path, "candidate-validator-negative", tyx_mode, data) as record:
        with family.prepared(data, tyx_mode, tmp_path) as (_, _, plan):
            trace = plan.query_with_diagnostics(pairs=selected(data))
            record["trace"] = trace
            record["checks"] = assert_same_candidate(trace, data)
        last = trace.query.rows[0].gjk_used - 1
        record["validator_forgeries"] = {}
        for name in ("outgoing-q", "different-weights", "duplicate-wide-plane"):
            floats, integers = trace.floats.copy(), trace.integers.copy()
            if name == "outgoing-q":
                floats[0, last, 4] += np.float32(1/64)
            elif name == "different-weights":
                floats[0, last, 108] *= np.float32(1/2)
            else:
                integers[0, last, 6] = 1
                floats[0, last, 11] = 0
                floats[0, last, 22] = floats[0, last, 21]
            forged = replace(trace, floats=floats, integers=integers)
            record["validator_forgeries"][name] = dict(floats=floats, integers=integers)
            with pytest.raises(AssertionError):
                assert_same_candidate(forged, data)
        assert_same_candidate(trace, data)
