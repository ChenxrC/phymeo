"""True rigid arcs, continuous first events and world/material witnesses."""

from math import asin, pi

import numpy as np
import pytest
from rotating_round_cases import authored, isolated, scene

import phymeo as ph


def query(model, paths):
    with ph.CollisionWorkspace(model) as work:
        return ph.detect(model, paths, ph.CollisionPolicy(), work).pairs[0]


@pytest.mark.parametrize("name", ["base", "clear", "holdout"])
@pytest.mark.parametrize("reverse", [False, True])
def test_authored_arc(name, reverse):
    model, paths, oracle = authored(name, reverse)
    row = query(model, paths)
    assert row.candidate and row.family == "RR"
    if oracle.time is None:
        assert row.status == "separated" and row.sweep.first_t is None
        assert row.sweep.start_gap_m > 0 and row.sweep.end_gap_m > 0
        return
    assert row.status == "hit"
    lo, hi = row.sweep.first_t
    assert lo <= oracle.time <= hi and hi - lo < 1e-8
    assert row.sweep.start_gap_m > 0 and row.sweep.end_gap_m > 0
    witness = row.sweep.witness
    np.testing.assert_allclose(witness.first_point, oracle.point, atol=1e-9, rtol=0)
    np.testing.assert_allclose(witness.second_point, oracle.point, atol=1e-9, rtol=0)
    ball_weights, cap_weights = (
        (witness.first_weights, witness.second_weights)
        if reverse
        else (witness.second_weights, witness.first_weights)
    )
    assert ball_weights == ((19, 1.0),)
    assert {i for i, _ in cap_weights} == {11, 37} and sum(w for _, w in cap_weights) == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_isolated_endcap_tangency_and_one_ulp_miss(reverse):
    hit = query(*isolated(reverse))
    assert hit.status == "hit"
    lo, hi = hit.sweep.first_t
    assert lo <= 0.25 <= hi and hi - lo < 1e-8
    weights = hit.sweep.witness.second_weights if reverse else hit.sweep.witness.first_weights
    assert dict(weights) == {11: 1.0, 37: 0.0}
    miss = query(*isolated(reverse, miss=True))
    assert miss.candidate and miss.status == "separated" and miss.sweep.first_t is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_oblique_axis_signed_multiple_turns_and_moving_pivot(sign):
    # Rotate about (3/5, 4/5, 0) in the plane spanned by z and (4/5,-3/5,0).
    pivot = np.array((4.0, -2.0, 3.0))
    perpendicular = np.array((0.8, -0.6, 0))
    initial = np.array((0, 0, 1.0))
    key_args = dict(axis=(0.6, 0.8, 0), pivot=pivot, travel=np.array((1, -0.5, 0.25)))
    model, paths = scene(
        [pivot - initial, pivot + initial],
        pivot + sign * 0.75 * perpendicular,
        sign * 5 * pi,
        **key_args,
    )
    row = query(model, paths)
    assert row.status == "hit"
    expected = (pi / 2 - asin(0.1 / 0.75)) / (5 * pi)
    assert row.sweep.first_t[0] <= expected <= row.sweep.first_t[1]
    assert row.sweep.first_t[1] - row.sweep.first_t[0] < 1e-8


def test_negative_half_turn_with_off_center_pivot_enters_in_second_chart():
    # Entire capsule is on one side of its pivot, so its first encounter is after pi.
    model, paths = scene([[0.5, 0, 0], [1, 0, 0]], [0, 0.75, 0], -2 * pi)
    row = query(model, paths)
    expected = (3 * pi / 2 - asin(0.1 / 0.75)) / (2 * pi)
    assert row.status == "hit" and row.sweep.first_t[0] <= expected <= row.sweep.first_t[1]


def test_turn_stops_before_contact_and_initial_overlap_is_preserved():
    segment = [[0.5, 0, 0], [1, 0, 0]]
    short = query(*scene(segment, [0, 0.75, 0], 0.75))
    assert short.status == "separated" and short.sweep.first_t is None
    initial = query(*scene(segment, [0.8, 0, 0], 2 * pi))
    assert initial.status == "hit" and initial.sweep.first_t == (0, 0)
    assert initial.sweep.witness.gap_m == pytest.approx(-0.1)


def test_degenerate_segment_still_has_material_address_at_a_cap_event():
    model, paths = scene([[1, 0, 0], [1, 0, 0]], [0, 1.125, 0], pi, radii=(0.0625, 0.0625))
    row = query(model, paths)
    assert row.status == "hit" and row.sweep.first_t[0] <= 0.5 <= row.sweep.first_t[1]
    assert dict(row.sweep.witness.first_weights) == {37: 0.5, 11: 0.5}


@pytest.mark.parametrize("clear", [False, True])
def test_sphere_moves_independently_across_a_turning_shaft(clear):
    start = [0, 0.75, 0.25]
    drift = [0, 0, -0.125 if clear else -0.25]
    if clear:
        start[2] = 0.265625
    model, paths = scene(
        [[0.5, 0, 0], [1, 0, 0]], start, pi, ball_travel=drift, radii=(0.0625, 0.0625)
    )
    row = query(model, paths)
    if clear:
        assert row.candidate and row.status == "separated" and row.sweep.first_t is None
    else:
        # Before 1/2 the vertical separation alone exceeds the two radii.
        # At 1/2 the ball projection meets the shaft and the normal is vertical.
        assert row.status == "hit"
        lo, hi = row.sweep.first_t
        assert lo <= 0.5 <= hi and hi - lo < 1e-8
        np.testing.assert_allclose(row.sweep.witness.normal, [0, 0, 1], atol=1e-7)


def test_subdivided_turn_restores_absolute_first_time_and_endpoint_order():
    model, paths, oracle = authored("base")
    cap, orb = (s.key for s in model.shapes)
    cut = 0.25
    turn = paths[cap].turn
    angle = cut * turn.angle
    c, s = np.cos(angle), np.sin(angle)
    mid = paths[cap].start @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    def arc(start, fraction):
        return ph.CollisionMotion(
            start,
            turn=ph.RigidTurn(turn.pivot_start, turn.pivot_end, turn.axis, fraction * turn.angle),
        )

    before = query(model, {cap: arc(paths[cap].start, cut), orb: paths[orb]})
    after = query(model, {cap: arc(mid, 1 - cut), orb: paths[orb]})
    assert before.status == "separated" and after.status == "hit"
    lo, hi = after.sweep.first_t
    assert cut + (1 - cut) * lo <= oracle.time <= cut + (1 - cut) * hi
    reversed_model = ph.CollisionModel(
        [
            ph.CollisionShape(cap, "capsule", (11, 37), 0.05),
            model.shapes[1],
        ]
    )
    reordered = query(reversed_model, {cap: arc(mid[::-1], 1 - cut), orb: paths[orb]})
    assert reordered.sweep == after.sweep
