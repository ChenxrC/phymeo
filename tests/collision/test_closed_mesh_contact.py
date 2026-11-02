"""Closed surface occupancy, swept boundary contact and stable material addresses."""

from dataclasses import replace

import numpy as np
import pytest
from case_catalog import geometry_for
from closed_mesh_cases import NAMES, authored, containing, shapes
from test_solid_boundaries import triangulate_u

import phymeo as ph


def query(model, paths):
    with ph.CollisionWorkspace(model) as work:
        return ph.detect(model, paths, ph.CollisionPolicy(), work).pairs[0]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("reverse", [False, True])
def test_rr05_notch_crossing_full_containment(name, reverse):
    model, paths, oracle = authored(name, reverse)
    row = query(model, paths)
    assert row.candidate and row.family == "RR"
    trace = row.mesh
    assert trace.start.relation == "separated" and trace.start.boundary_gap_m > 0
    if oracle["first_t"] is None:
        assert row.status == "separated" and row.sweep.first_t is None
        assert trace.end.relation == "separated" and trace.end.boundary_gap_m > 0
        return
    assert row.status == "hit" and row.sweep.first_t[0] <= oracle["first_t"] <= row.sweep.first_t[1]
    assert trace.end.relation == ("second_contains_first" if reverse else "first_contains_second")
    assert trace.end.boundary_gap_m > 0 and not trace.end.contacts
    assert trace.entry.relation == "touching" and trace.entry.boundary_gap_m == 0
    assert trace.entry.contacts and trace.entry.facets
    for pt in trace.entry.contacts:
        assert pt.point[0] == oracle["wall_x"]
        assert abs(sum(x for _, x in pt.first_weights) - 1) < 1e-12
        assert abs(sum(x for _, x in pt.second_weights) - 1) < 1e-12
    t = next(phase["t"] for phase in oracle["samples"] if phase["phase"] == "cut")
    static = {}
    for key, path in paths.items():
        delta = np.zeros(3) if path.turn is None else np.asarray(path.turn.pivot_end)
        static[key] = ph.CollisionMotion(path.start + t * delta)
    cut = query(model, static)
    assert cut.status == "overlap" and cut.mesh.start.relation == "overlap"
    assert cut.mesh.start.contacts and cut.mesh.start.boundary_gap_m == 0


def test_mesh_strictly_inside_hull_and_nested_initial_hit():
    model, paths = containing()
    row = query(model, paths)
    assert row.status == "overlap" and row.mesh.start.relation == "second_contains_first"
    assert row.mesh.start.boundary_gap_m == pytest.approx(1)
    mesh, hull = model.shapes
    paths[hull.key] = ph.CollisionMotion(
        paths[hull.key].start, turn=ph.RigidTurn((0, 0, 0), (10, 0, 0), (0, 0, 1), 0)
    )
    leaving = query(model, paths)
    assert leaving.sweep.first_t == (0, 0) and leaving.mesh.end.relation == "separated"


def test_open_duplicate_inward_and_self_intersecting_mesh_do_not_report_clear():
    _, _, g = geometry_for("RR05", "base")
    vertices, faces = triangulate_u(g["A"], False)
    for bad in (faces[:-1], np.concatenate([faces, faces[:1]]), faces[:, ::-1]):
        model, paths = shapes(vertices, bad, g["B"]["vertices"])
        row = query(model, paths)
        assert row.status == "invalid_geometry" and row.mesh is None
    folded = vertices.copy()
    folded[10] = [-1.75, 1.5, 0.25]
    model, paths = shapes(folded, faces, g["B"]["vertices"])
    assert query(model, paths).status == "invalid_geometry"


def test_mesh_connectivity_is_owned_and_bound_to_material_ids():
    model, paths, _ = authored("base")
    mesh = model.shapes[0]
    external = list(map(list, mesh.triangles))
    copy = replace(mesh, triangles=external)
    external[0][0] = -8
    assert copy.triangles == mesh.triangles
    with pytest.raises(ValueError):
        replace(mesh, triangles=[(100, 103, 999)])


def test_common_translation_keeps_notch_event_and_both_world_points():
    model, paths, oracle = authored("left_arm_holdout")
    mesh, hull = model.shapes
    travel = np.array((0, 2, -1))
    moved = {
        mesh.key: ph.CollisionMotion(
            paths[mesh.key].start, turn=ph.RigidTurn((0, 0, 0), travel, (1, 0, 0), 0)
        ),
        hull.key: ph.CollisionMotion(
            paths[hull.key].start,
            turn=ph.RigidTurn(
                (0, 0, 0), np.asarray(paths[hull.key].turn.pivot_end) + travel, (1, 0, 0), 0
            ),
        ),
    }
    original = query(model, paths)
    row = query(model, moved)
    assert row.mesh.first_ratio == original.mesh.first_ratio
    assert row.mesh.entry.facets == original.mesh.entry.facets
    assert len(row.mesh.entry.contacts) == len(original.mesh.entry.contacts)
    for first, second in zip(original.mesh.entry.contacts, row.mesh.entry.contacts, strict=True):
        assert (first.first_weights, first.second_weights) == (
            second.first_weights,
            second.second_weights,
        )
        np.testing.assert_allclose(
            second.point, np.asarray(first.point) + travel * oracle["first_t"], atol=1e-13
        )


def test_start_inside_counts_even_when_the_whole_sweep_never_reaches_a_boundary():
    model, paths, _ = authored("base")
    mesh, hull = model.shapes
    paths[hull.key] = ph.CollisionMotion(
        paths[hull.key].start + [1.25, 0, 0],
        turn=ph.RigidTurn((0, 0, 0), (0.125, 0, 0), (1, 0, 0), 0),
    )
    row = query(model, paths)
    assert row.status == "hit" and row.sweep.first_t == (0, 0)
    assert row.mesh.entry.relation == "first_contains_second" and not row.mesh.entry.contacts
    assert row.mesh.entry.boundary_gap_m > 0 and row.sweep.witness.gap_m < 0


def test_mesh_capacity_is_explicit_and_rejected_query_preserves_history():
    model, paths, _ = authored("base")
    with ph.CollisionWorkspace(model) as work:
        first = ph.detect(model, paths, ph.CollisionPolicy(), work)
        failed = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert failed.status == "capacity_exceeded" and work.last_result is first
        assert ph.detect(model, paths, ph.CollisionPolicy(), work) == first
