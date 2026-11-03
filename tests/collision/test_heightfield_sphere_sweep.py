"""Whole-step contact extrema for a translating sphere and fixed triangle height graph."""

from fractions import Fraction

import numpy as np
import pytest
from test_heightfield_sphere import detect, scene

import phymeo as ph


def assert_event(event, t, land, radius, start, end, reverse, point, normal):
    assert isinstance(event, ph.HeightfieldEvent)
    lo, hi = event.interval
    assert Fraction(lo) <= t <= Fraction(hi) and 0 <= hi - lo <= 2 ** -40
    assert lo <= event.witness_time <= hi
    face, w = event.face, event.witness
    assert isinstance(face, ph.HeightfieldFace)
    positions = dict(zip(land.vertex_ids, land.heightfield.world_vertices(), strict=True))
    tri = np.array([positions[i] for i in face.vertex_ids])
    p, q = (w.second_point, w.first_point) if reverse else (w.first_point, w.second_point)
    n = np.asarray(normal)
    center = np.asarray(start) + event.witness_time * (np.asarray(end) - np.asarray(start))
    np.testing.assert_allclose(face.barycentric @ tri, p, rtol=0, atol=1e-11)
    np.testing.assert_allclose(p, point, rtol=0, atol=1e-10)
    np.testing.assert_allclose(q, center - radius*n, rtol=0, atol=1e-10)
    np.testing.assert_allclose(w.normal, -n if reverse else n, rtol=0, atol=1e-10)
    assert abs(w.gap_m) <= 1e-10


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("backwards", [False, True])
def test_roof_first_and_last_match_two_physical_witnesses(reverse, backwards):
    grid = ph.Heightfield([[0, 1, 0]]*3, (1, 1), origin=(-1, -1, 0))
    sign = -1 if backwards else 1
    start, end = (-sign * 0.75, 0.125, 1.25), (sign * 0.75, 0.125, 1.25)
    radius = 5/16
    model, paths, land = scene(grid, start, radius, reverse=reverse, end=end)
    row = detect(model, paths)
    assert row.status == "hit"
    first, last = row.heightfield.first_contact, row.heightfield.last_contact
    for e, t, normal in ((first, Fraction(3, 8), [-0.6*sign, 0, 0.8]),
                         (last, Fraction(5, 8), [0.6*sign, 0, 0.8])):
        assert_event(e, t, land, radius, start, end, reverse, [0, 0.125, 1], normal)
    assert (row.sweep.first_t, row.sweep.witness_time, row.sweep.witness) == (
        first.interval, first.witness_time, first.witness
    )
    assert row.sweep.start_gap_m == pytest.approx(1/np.sqrt(2) - radius)
    assert row.witness.gap_m == pytest.approx(row.sweep.start_gap_m)
    ids = dict(zip(land.vertex_ids, grid.world_vertices(), strict=True))
    final_tri = np.array([ids[i] for i in row.heightfield.vertex_ids])
    p = row.witness.second_point if reverse else row.witness.first_point
    np.testing.assert_allclose(row.heightfield.barycentric @ final_tri, p, rtol=0, atol=1e-11)
    assert tuple(p) != (0, 0.125, 1)


def test_subdivision_and_overlapping_endpoints():
    grid = ph.Heightfield([[0, 1, 0]]*3, (1, 1), origin=(-1, -1, 0))
    for a, b, expected in [(-0.75, 0, (0.75, 1.0)), (0, 0.75, (0.0, 0.25))]:
        model, paths, _ = scene(grid, [a, 0.125, 1.25], 5/16, end=[b, 0.125, 1.25])
        row = detect(model, paths)
        assert row.status == "hit"
        assert row.heightfield.first_contact.interval == (expected[0],)*2
        assert row.heightfield.last_contact.interval == (expected[1],)*2


def test_multiple_windows_leave_a_separated_midpoint():
    grid = ph.Heightfield([[0, 1, 0, 1, 0]]*3, (1, 1), origin=(-2, -1, 0))
    start, end = [-1.75, 0.125, 1.25], [1.75, 0.125, 1.25]
    model, paths, land = scene(grid, start, 5/16, end=end)
    row = detect(model, paths)
    assert row.status == "hit"
    assert_event(row.heightfield.first_contact, Fraction(9, 56), land, 5/16, start, end,
                 False, [-1, 0.125, 1], [-0.6, 0, 0.8])
    assert_event(row.heightfield.last_contact, Fraction(47, 56), land, 5/16, start, end,
                 False, [1, 0.125, 1], [0.6, 0, 0.8])
    at_mid, midpaths, _ = scene(grid, [0, 0.125, 1.25], 5/16)
    assert detect(at_mid, midpaths).status == "separated"


def test_nondyadic_isolated_corner_graze_and_one_ulp_miss():
    grid = ph.Heightfield([[0, 0], [0, 0]], (1, 1))
    for x, hit in ((1.1875, True), (np.nextafter(1.1875, np.inf), False)):
        start, end = [x, 1.25, -0.25], [x, 1.25, 0.5]
        model, paths, land = scene(grid, start, 5/16, end=end)
        row = detect(model, paths)
        if hit:
            assert row.status == "hit"
            for e in (row.heightfield.first_contact, row.heightfield.last_contact):
                assert_event(e, Fraction(1, 3), land, 5/16, start, end, False, [1, 1, 0],
                             [0.6, 0.8, 0])
        else:
            assert row.status == "separated" and row.sweep.first_t is None
            assert row.heightfield.first_contact is row.heightfield.last_contact is None


def test_ball_arc_remains_outside_linear_sweep_contract():
    grid = ph.Heightfield([[0, 0], [0, 0]], (1, 1))
    model, paths, land = scene(grid, [0.5, 0.5, 0], 0.125)
    ball_key = next(k for k in paths if k != land.key)
    paths[ball_key] = ph.CollisionMotion(
        [[0.5, 0.5, 0]], turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 1, 0), 1)
    )
    row = detect(model, paths)
    assert row.status == "unsupported" and row.sweep is None


def test_rotated_grid_and_zero_angle_translating_pivot():
    rotation = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    offset = np.array([2, -3, 4])
    grid = ph.Heightfield([[0, 1, 0]]*3, (1, 1), origin=offset + rotation @ [-1, -1, 0],
                          rotation=rotation)
    start = offset + rotation @ [-0.75, 0.125, 1.25]
    end = offset + rotation @ [0.75, 0.125, 1.25]
    model, paths, land = scene(grid, start, 5/16, reverse=True)
    ball_key = next(k for k in paths if k != land.key)
    paths[ball_key] = ph.CollisionMotion(
        [start], turn=ph.RigidTurn((0, 0, 0), end-start, (0, 1, 0), 0)
    )
    row = detect(model, paths)
    assert row.status == "hit"
    p = offset + rotation @ [0, 0.125, 1]
    for event, t, n in ((row.heightfield.first_contact, Fraction(3, 8), [-0.6, 0, 0.8]),
                        (row.heightfield.last_contact, Fraction(5, 8), [0.6, 0, 0.8])):
        assert_event(event, t, land, 5/16, start, end, True, p, rotation @ n)
