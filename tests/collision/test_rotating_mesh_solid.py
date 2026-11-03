"""Public closed-solid CCD under coaxial rotations and a shared linear pivot drift."""

import json
import math
from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np
import pytest
from rotating_mesh_oracles import harmonics, position
from test_mesh_mesh_contact import box
from test_rotating_mesh_candidates import double_u, mesh_pair, orbital, set_turn
from test_rotating_mesh_surface import bar_and_wall

import phymeo as ph


def detect(shapes, paths, *, swap=False, excluded=False):
    if swap:
        shapes = [replace(shapes[0], key=shapes[1].key), replace(shapes[1], key=shapes[0].key)]
    mapping = {s.key: p for s, p in zip(shapes, paths, strict=True)}
    excluded_pairs = [(shapes[0].key, shapes[1].key)] if excluded else []
    model = ph.CollisionModel(shapes, excluded=excluded_pairs)
    with ph.CollisionWorkspace(model) as work:
        (row,) = ph.detect(model, mapping, ph.CollisionPolicy(margin=0.125), work).pairs
    return row, model.shapes, [mapping[s.key] for s in model.shapes]


def materials(w, shapes, paths, t):
    for shape, path, weights, point in zip(
        shapes,
        paths,
        (w.first_weights, w.second_weights),
        (w.first_point, w.second_point),
        strict=True,
    ):
        assert abs(sum(x for _, x in weights) - 1) < 1e-10 and min(x for _, x in weights) >= 0
        nodes = dict(zip(shape.vertex_ids, position(harmonics(path), t), strict=True))
        np.testing.assert_allclose(sum(x * nodes[k] for k, x in weights), point, atol=1e-6, rtol=0)
    if w.normal is not None:
        assert np.dot(np.subtract(w.second_point, w.first_point), w.normal) == pytest.approx(
            w.gap_m, abs=1e-6
        )


def bracket(interval, oracle):
    assert 0 <= interval[0] <= oracle <= interval[1] <= 1 and interval[1] - interval[0] <= 1e-6


@pytest.mark.parametrize("mode", ["notch", "right", "coaxial", "partial", "tangent", "miss"])
@pytest.mark.parametrize("swap", [False, True])
def test_ordinary_detect_rotating_closed_solids(mode, swap, tmp_path):
    if mode in ("tangent", "miss"):
        shapes, paths = bar_and_wall(0.625 if mode == "tangent" else 0.625 + 1 / 1024)
        expected = math.atan2(0.5, 0.375) / 3 if mode == "tangent" else None
    else:
        shapes, paths = double_u("notch" if mode == "notch" else "right")
        pivot = (0.375, 0, 0)
        shift = (0.125, 0.0625, 0)
        angle = math.pi / 4 if mode == "partial" else math.pi / 2
        if mode == "coaxial":
            paths[0] = set_turn(paths[0], np.add(pivot, [0, 0, 0.5]), shift, (0, 0, 1), 0.25)
            paths[1] = set_turn(paths[1], pivot, shift, (0, 0, -1), -(0.25 + angle))
        elif mode == "partial":
            paths[1] = set_turn(paths[1], pivot, shift, (0, 0, 1), angle)
        expected = (
            (math.pi / 4 - math.acos(1.25 / math.sqrt(2))) / angle if mode != "notch" else None
        )
    row, ordered, ordered_paths = detect(shapes, paths, swap=swap)
    assert row.status == ("hit" if expected is not None else "separated"), row.reason
    trace = row.mesh_mesh
    assert trace.start.relation == "separated" and trace.start.witness.gap_m > 0
    assert 0 < trace.end_error_m <= 1e-6
    assert trace.end_certificate == (
        "strict_transverse_intersection" if mode == "partial" else "positive_boundary_clearance"
    )
    assert trace.end.relation == ("boundary_overlap" if mode == "partial" else "separated")
    assert trace.end.witness == row.witness
    assert (
        row.sweep.end_gap_m == row.witness.gap_m
        and row.sweep.start_gap_m == trace.start.witness.gap_m
    )
    materials(row.witness, ordered, ordered_paths, 1)
    for support in trace.end.boundary:
        materials(support.witness, ordered, ordered_paths, 1)
    if mode == "partial":
        assert trace.end.boundary_gap_m == 0 and trace.end.boundary
    if mode in ("right", "coaxial"):
        assert trace.end.boundary_gap_m == pytest.approx(0.125, abs=1e-6)
    if expected is None:
        assert row.sweep.first_t is trace.first_boundary is None
    else:
        bracket(row.sweep.first_t, expected)
        event = trace.first_boundary
        assert row.sweep.first_t == event.interval and row.sweep.witness_time == event.witness_time
        assert trace.first_ratio is None
        materials(row.sweep.witness, ordered, ordered_paths, event.witness_time)
        for s in event.supports:
            assert s.faces.first in ordered[0].triangles and s.faces.second in ordered[1].triangles
            materials(s.witness, ordered, ordered_paths, event.witness_time)
    (tmp_path / "solid-arc.json").write_text(
        json.dumps(
            dict(
                mode=mode,
                swap=swap,
                first=row.sweep.first_t,
                witness_time=row.sweep.witness_time,
                start=trace.start.relation,
                end=trace.end.relation,
                end_gap_m=trace.end.boundary_gap_m,
                end_certificate=trace.end_certificate,
                end_error_m=trace.end_error_m,
                boundary=asdict(trace.first_boundary) if trace.first_boundary else None,
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("leaves", [False, True])
def test_contained_start_is_time_zero_even_when_first_surface_is_later_or_absent(leaves, tmp_path):
    eps = 0.0625
    center = np.array([0.75, 0, 0]) if leaves else np.zeros(3)
    shapes, paths = mesh_pair((box((-1,) * 3, (1,) * 3), box(center - eps, center + eps)))
    angle = math.pi / 2 if leaves else 0.7
    pivot = (0.75, 1, 0) if leaves else (0, 0, 0)
    paths[1] = set_turn(paths[1], pivot, (0, 0, 0), (0, 0, 1), angle)
    row, ordered, ordered_paths = detect(shapes, paths)
    assert row.status == "hit", row.reason
    trace = row.mesh_mesh
    assert trace.start.relation == "first_contains_second" and trace.start.boundary_gap_m > 0
    assert not trace.start.boundary and trace.entry is trace.start
    assert (
        row.sweep.first_t == (0.0, 0.0)
        and row.sweep.witness_time == 0
        and row.sweep.witness == trace.start.witness
    )
    assert trace.first_ratio == (0, 1)
    materials(row.sweep.witness, ordered, ordered_paths, 0)
    materials(row.witness, ordered, ordered_paths, 1)
    if leaves:
        expected = (math.asin(0.25 / math.hypot(1 + eps, eps)) - math.atan2(eps, 1 + eps)) / angle
        assert trace.end.relation == "separated"
        bracket(trace.first_boundary.interval, expected)
        assert trace.first_boundary.interval[0] > 0
        assert trace.end.boundary_gap_m == pytest.approx(0.75 - eps, abs=1e-6)
    else:
        assert trace.first_boundary is None and trace.end.relation == "first_contains_second"
        assert trace.end.boundary_gap_m == pytest.approx(
            1 - eps * (math.sin(angle) + math.cos(angle)), abs=1e-6
        )
    (tmp_path / "contained-solid-arc.json").write_text(
        json.dumps(
            dict(
                leaves=leaves,
                initial=trace.start.relation,
                initial_boundary_gap_m=trace.start.boundary_gap_m,
                first_solid=row.sweep.first_t,
                first_boundary=asdict(trace.first_boundary) if trace.first_boundary else None,
                end=trace.end.relation,
                end_boundary_gap_m=trace.end.boundary_gap_m,
                end_certificate=trace.end_certificate,
                end_error_m=trace.end_error_m,
                endpoint_witness=asdict(row.witness),
            ),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def test_unsupported_skew_and_drift_filtered_and_adjacent_endpoint_control():
    shapes, paths, _ = orbital(dual=True)
    row, *_ = detect(shapes, paths)
    assert row.status == "unsupported" and row.sweep is row.mesh_mesh is None
    row, *_ = detect(shapes, paths, excluded=True)
    assert row.status == "filtered" and row.sweep is None
    shapes, paths = bar_and_wall(0.625)
    paths[1] = ph.CollisionMotion(paths[1].start, paths[1].start + [0, 0.125, 0])
    row, *_ = detect(shapes, paths)
    assert row.status == "unsupported"
    # This stored angle terminates within a floating-point step of the exact
    # peak. A root ordering or endpoint relation that cannot be certified is explicit.
    shapes, paths = bar_and_wall(0.625, angle=math.atan2(0.5, 0.375))
    row, *_ = detect(shapes, paths)
    assert row.status == "unresolved" and row.sweep is None


def test_original_linear_mesh_ccd_still_publishes_exact_entry_ratio():
    shapes, paths = mesh_pair(
        (box((0, 0, 0), (1, 1, 1)), box((-0.75, 0.25, 0.25), (-0.5, 0.75, 0.75)))
    )
    paths[1] = ph.CollisionMotion(paths[1].start, paths[1].start + [2.5, 0, 0])
    row, *_ = detect(shapes, paths)
    assert row.status == "hit" and row.mesh_mesh.first_ratio == (1, 5)
    bracket(row.sweep.first_t, Fraction(1, 5))


def test_persistently_touching_solids_publish_the_world_endpoint_under_common_motion():
    shapes, paths = mesh_pair((box((0, 0, 0), (1, 1, 1)), box((1, 0.25, 0.25), (1.25, 0.75, 0.75))))
    angle = 0.7
    paths = [
        set_turn(path, (0.5, 0, i / 4), (0.25, -0.125, 0.5), (0, 0, 1), angle)
        for i, path in enumerate(paths)
    ]
    row, ordered, ordered_paths = detect(shapes, paths)
    assert row.status == "hit", row.reason
    trace = row.mesh_mesh
    assert trace.start.relation == trace.end.relation == "touching"
    assert (
        trace.start.boundary_facet_pairs_tested > 0 and trace.end.boundary_facet_pairs_tested == 0
    )
    assert trace.end_certificate == "fixed_relative_pose" and trace.first_boundary.interval == (
        0,
        0,
    )
    assert row.sweep.first_t == (0, 0) and trace.end.boundary_gap_m == 0
    for support in trace.end.boundary:
        materials(support.witness, ordered, ordered_paths, 1)
    np.testing.assert_allclose(
        row.witness.normal, (math.cos(angle), math.sin(angle), 0), atol=1e-9, rtol=0
    )


def test_chords_of_the_inner_u_vertices_miss_the_wall_reached_by_the_true_arc():
    shapes, paths = double_u("right")
    terms = [harmonics(path) for path in paths]
    drift = paths[0].end[0] - paths[0].start[0]
    ends = np.concatenate((position(terms[1], 0), position(terms[1], 1) - drift))
    # Every chord, and every triangular convex combination of those chords,
    # stays strictly in the opening throughout a linearly interpolated step.
    assert np.min(ends[:, 0]) > -1 and np.max(ends[:, 0]) < 1 and np.min(ends[:, 1]) > -1
    row, *_ = detect(shapes, paths)
    assert row.status == "hit" and row.mesh_mesh.end.relation == "separated"


@pytest.mark.parametrize("exact_frame", [True, False], ids=["integer-similarity", "binary-rounded"])
def test_whole_fixture_oblique_frame_preserves_or_diagnoses_the_material_event(exact_frame):
    shapes, local = double_u("right")
    if exact_frame:
        # Columns are orthogonal, length 3. Integer arithmetic preserves the
        # original simultaneous material events; the stored (2,2,1) direction
        # is exactly proportional to the third column even after normalization.
        transform = np.array([[-2, 1, 2], [1, -2, 2], [2, 2, 1]])
        scale = 3
    else:
        # Binary approximations to the intended 3/5,4/5 rotation split tied
        # edges into different roots inside the same adjudication cell.
        transform = np.array([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]])
        scale = 1
    rotation = transform / scale
    origin = np.array([0.25, -0.5, 0.125])
    motions = []
    for path in local:
        start = path.start @ transform.T + origin
        if path.turn:
            pivot = transform @ path.turn.pivot_start + origin
            motions.append(
                ph.CollisionMotion(
                    start,
                    turn=ph.RigidTurn(pivot, pivot, rotation @ path.turn.axis, path.turn.angle),
                )
            )
        else:
            motions.append(ph.CollisionMotion(start))
    row, ordered, paths = detect(shapes, motions, swap=True)
    if not exact_frame:
        assert row.status == "unresolved" and row.reason == "mesh_arc_competing_first_roots"
        assert row.sweep is row.mesh_mesh is None
        return
    assert row.status == "hit", row.reason
    assert row.mesh_mesh.start.relation == row.mesh_mesh.end.relation == "separated"
    expected = (math.pi / 4 - math.acos(1.25 / math.sqrt(2))) / (math.pi / 2)
    bracket(row.sweep.first_t, expected)
    for support in row.mesh_mesh.first_boundary.supports:
        w = support.witness
        materials(w, ordered, paths, row.sweep.witness_time)
        for point in (w.first_point, w.second_point):
            local_point = rotation.T @ (point - origin) / scale
            assert local_point[0] == pytest.approx(1, abs=1e-6)
