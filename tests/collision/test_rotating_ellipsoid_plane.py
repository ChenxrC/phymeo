"""Complete finite-angle support contact sets of a centered rotating ellipsoid."""

import json
from dataclasses import asdict
from decimal import Decimal as D
from math import pi

import numpy as np
import pytest
from rotating_ellipsoid_oracles import analytic_extrema, physical_points
from test_curved_solids import CYCLIC, IDENTITY, scene, solid

import phymeo as ph

AXES = (0.5, 0.25, 0.25)
IDENTITY_FRAME = np.eye(3)
CYCLE = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def build(
    angle=pi,
    height=0.375,
    dims=AXES,
    axis=(0, 1, 0),
    *,
    reverse=False,
    transformed=False,
    orientation=None,
):
    rotation = CYCLE if transformed else IDENTITY_FRAME
    anchor = np.array([2, -1, 0.5]) if transformed else np.zeros(3)
    center = anchor + rotation @ [0, 0, height]
    n = rotation @ [0, 0, 3]
    quat = orientation or (CYCLIC if transformed else IDENTITY)
    model, paths, key, plane = scene(
        solid("ellipsoid", dims, quat), center, "plane", anchor, reverse=reverse, normal=n
    )
    turn = ph.RigidTurn(center, center, rotation @ axis, angle)
    paths[key] = ph.CollisionMotion([center], turn=turn)
    return model, paths, key, plane, turn, rotation


def detect(model, paths):
    with ph.CollisionWorkspace(model) as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
        return row


def check_material(w, key, plane, curved_first, paths, dims, initial, turn, t):
    center, anchor = paths[key].start[0], paths[plane].start[0]
    # These tests bind the world plane by rotating the initial local +Z together
    # with the whole fixture. Independent-initial-pose tests set the plane separately.
    normal = initial @ [0, 0, 1]
    gap, ellipse, on_plane = physical_points(
        center, anchor, normal, dims, initial, turn.axis, t * turn.angle
    )
    expect_normal = -normal if curved_first else normal
    np.testing.assert_allclose(w.normal, expect_normal, rtol=0, atol=1e-5)
    assert w.gap_m == pytest.approx(gap, rel=0, abs=1e-6)
    for point, offset, weights, ref, origin, node in zip(
        (w.first_point, w.second_point),
        (w.first_offset, w.second_offset),
        (w.first_weights, w.second_weights),
        (ellipse, on_plane) if curved_first else (on_plane, ellipse),
        (center, anchor) if curved_first else (anchor, center),
        (47, 91) if curved_first else (91, 47),
        strict=True,
    ):
        assert weights == ((node, 1.0),)
        np.testing.assert_allclose(point, ref, rtol=0, atol=1e-6)
        np.testing.assert_allclose(origin + offset, point, rtol=0, atol=1e-10)
    return ellipse


@pytest.mark.parametrize(
    "angle,height,reverse,transformed,triaxial",
    [
        (pi, 0.375, False, False, False),
        (-pi, 0.375, False, False, False),
        (pi, 0.375, True, False, False),
        (pi, 0.5, False, False, False),
        (pi, 9 / 16, False, False, False),
        (3.0, 0.5 - 1 / 16384, False, False, False),
        (2.0, 0.375, False, False, False),
        (pi, 0.375, False, False, True),
        (-pi, 0.375, True, True, True),
    ],
)
def test_analytic_contact_extrema_and_physical_material(
    angle, height, reverse, transformed, triaxial, tmp_path
):
    dims = (0.5, 0.375, 0.25) if triaxial else AXES
    axis = (0.6, 0.8, 0) if triaxial else (0, 1, 0)
    model, paths, key, plane, turn, rotation = build(
        angle, height, dims, axis, reverse=reverse, transformed=transformed
    )
    expected = analytic_extrema(dims, height, rotation.T @ turn.axis, turn.angle)
    row = detect(model, paths)
    assert row.status == ("hit" if expected else "separated")
    assert row.sweep is not None and row.curved.owner == key
    first = row.first == key
    assert row.sweep.start_gap_m == pytest.approx(height - dims[2], abs=1e-10)
    assert row.sweep.end_gap_m == row.witness.gap_m
    data = dict(
        angle_rad=angle,
        stored_axis=turn.axis,
        height_m=height,
        reverse=reverse,
        transformed=transformed,
        triaxial=triaxial,
        oracle=list(map(str, expected)) if expected else None,
        start_gap_m=row.sweep.start_gap_m,
        end_gap_m=row.sweep.end_gap_m,
        first=asdict(row.curved.first_contact) if row.curved.first_contact else None,
        last=asdict(row.curved.last_contact) if row.curved.last_contact else None,
    )
    (tmp_path / "ellipsoid-arc.json").write_text(
        json.dumps(data, allow_nan=False), encoding="utf-8"
    )
    point = check_material(row.witness, key, plane, first, paths, dims, rotation, turn, 1.0)
    np.testing.assert_allclose(row.curved.points_m[0], point, rtol=0, atol=1e-6)
    if expected is None:
        assert (
            row.sweep.first_t is None
            and row.curved.first_contact is row.curved.last_contact is None
        )
        return
    events = (row.curved.first_contact, row.curved.last_contact)
    assert row.sweep.first_t == events[0].interval and row.sweep.witness == events[0].witness
    assert row.sweep.witness_time == events[0].witness_time
    for target, event in zip(expected, events, strict=True):
        assert isinstance(event, ph.CurvedEvent)
        lo, hi = event.interval
        assert 0 <= lo <= event.witness_time <= hi <= 1
        assert D.from_float(lo) <= target <= D.from_float(hi) and hi - lo <= 1e-6
        check_material(
            event.witness, key, plane, first, paths, dims, rotation, turn, event.witness_time
        )
    if height == 0.5:
        assert expected[0] == expected[1] > D("0.5") and events[0].interval == events[1].interval
    if angle == 3.0:
        assert events[0].interval[1] < events[1].interval[0]
        assert all(
            physical_points(
                paths[key].start[0],
                paths[plane].start[0],
                [0, 0, 1],
                dims,
                IDENTITY_FRAME,
                turn.axis,
                angle * j / 64,
            )[0]
            > 0
            for j in range(65)
        )


def test_contacts_at_both_ends_can_have_clear_interior():
    q = (0, np.sqrt(0.5), 0, np.sqrt(0.5))
    model, paths, key, plane, turn, _ = build(orientation=q)
    row = detect(model, paths)
    assert row.status == "hit" and row.curved.first_contact.interval == (0, 0)
    assert row.curved.last_contact.interval == (1, 1)
    assert row.sweep.start_gap_m == pytest.approx(-0.125)
    # At quarter of an ideal revolution, its long axis is horizontal again.
    initial = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert (
        physical_points(
            paths[key].start[0],
            paths[plane].start[0],
            [0, 0, 1],
            AXES,
            initial,
            turn.axis,
            turn.angle / 2,
        )[0]
        > 0.12
    )


@pytest.mark.parametrize(
    "kind", ["off-center", "moving-pivot", "moving-plane", "long-arc", "translation"]
)
def test_outside_centered_static_plane_arc_contract(kind):
    model, paths, key, plane, turn, _ = build()
    center = paths[key].start[0]
    start = center + [0.25, 0, 0] if kind == "off-center" else center
    end = start + [0, 0.25, 0] if kind == "moving-pivot" else start
    angle = np.nextafter(pi, np.inf) if kind == "long-arc" else pi
    paths[key] = ph.CollisionMotion([center], turn=ph.RigidTurn(start, end, turn.axis, angle))
    if kind == "moving-plane":
        paths[plane] = ph.CollisionMotion([[0, 0, 0]], [[0, 0, 0.125]])
    if kind == "translation":
        paths[key] = ph.CollisionMotion([center], [center + [0, 0, 0.125]])
    row = detect(model, paths)
    if kind == "translation":
        assert row.status == "separated" and row.sweep.first_t is None
        assert row.sweep.start_gap_m == pytest.approx(1/8) and row.sweep.end_gap_m == pytest.approx(1/4)
    else:
        assert row.status == "unsupported" and row.sweep is None and row.curved is None


def test_root_isolation_failure_is_explicit(monkeypatch):
    from phymeo.geometry import _rotating_ellipsoid_plane as arc

    model, paths, *_ = build(angle=3.0, height=0.5 - 1 / 16384)
    monkeypatch.setattr(arc, "_MAX_REFINEMENTS", 0)
    row = detect(model, paths)
    assert row.status == "unresolved" and row.sweep is None and row.curved is None


def test_skew_axis_against_independent_initial_orientation():
    # In y=tan(theta/2) the support residual has coefficients of signs
    # (-, 0, +, -, -), from the constant term up. There are at most two
    # positive roots. Opposite signs on the two half-arcs locate both.
    dims = (0.5, 0.375, 0.25)
    z, w = 1 / np.sqrt(10), 3 / np.sqrt(10)
    model, paths, key, plane, turn, _ = build(
        dims=dims, axis=(2 / 3, 2 / 3, 1 / 3), orientation=(0, 0, z, w)
    )
    norm2 = z * z + w * w
    cosine, sine = (w * w - z * z) / norm2, 2 * z * w / norm2
    initial = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    n = np.array([0, 0, 1])

    def material(t):
        return physical_points(
            paths[key].start[0], paths[plane].start[0], n, dims, initial, turn.axis, turn.angle * t
        )

    assert material(0)[0] > 0 and material(0.5)[0] < 0 and material(1)[0] > 0
    row = detect(model, paths)
    assert row.status == "hit"
    for event, (left, right) in zip(
        (row.curved.first_contact, row.curved.last_contact), ((0, 0.5), (0.5, 1)), strict=True
    ):
        positive = material(left)[0] > 0
        for _ in range(50):
            mid = (left + right) / 2
            if (material(mid)[0] > 0) == positive:
                left = mid
            else:
                right = mid
        root = (left + right) / 2
        assert event.interval[0] <= root <= event.interval[1]
        gap, ellipsoid, on_plane = material(event.witness_time)
        assert event.witness.gap_m == pytest.approx(gap, abs=1e-6)
        points = (ellipsoid, on_plane) if row.first == key else (on_plane, ellipsoid)
        np.testing.assert_allclose(
            [event.witness.first_point, event.witness.second_point], points, atol=1e-6, rtol=0
        )
