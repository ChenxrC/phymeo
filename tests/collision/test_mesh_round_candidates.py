"""B3 public mechanism RED and complete-result controls for finite face selection."""

from dataclasses import asdict, replace
from fractions import Fraction as F
from math import sqrt

import numpy as np
import pytest
from mesh_round_candidate_cases import (
    CLEAR_SWEEP, CONTAINS, CROSSING, FACE, INSIDE, SHAFT, STATIC_CASES, SWEEP_CASES,
    BoxOracle, Case, build, check_materials, face_distances, five_fields,
    prepare_selector, record_facets, required_faces,
)

import phymeo as ph


def test_public_bvh_uses_mesh_round_faces_after_correct_static_witness(monkeypatch):
    """Removing the round-face midphase must fail after the real physics is checked."""
    model, paths, mesh, round_shape = build(FACE)
    assert min(face_distances(mesh, round_shape, paths)) == F(9, 1024)
    with ph.CollisionWorkspace(model) as baseline:
        expected = ph.detect(model, paths, ph.CollisionPolicy(), baseline)
    (prior,) = expected.pairs
    assert prior.status == "overlap" and prior.mesh_round.end.relation == "boundary_overlap"
    assert prior.mesh_round.end.axis_gap_m == -1 / 32
    check_materials(prior, prior.mesh_round.end, mesh, round_shape, paths)
    calls = []
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        original = work._trees.facets

        def observed(pending, key, leaves, queries, driver):
            result = original(pending, key, leaves, queries, driver)
            calls.append((len(leaves), tuple(tuple(row) for row in result), driver))
            return result

        monkeypatch.setattr(work._trees, "facets", observed)
        actual = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert actual == expected
        assert five_fields(actual.pairs[0]) == five_fields(prior)
        check_materials(actual.pairs[0], actual.pairs[0].mesh_round.end, mesh, round_shape, paths)
        assert calls, "public mesh_midphase='bvh' still never dispatches mesh-round face candidates"
        assert all(driver is None for _, _, driver in calls)
        assert any(0 < len(row) < total for total, rows, _ in calls for row in rows)
        assert work.stats()["mesh_bvh_refits"] == len(calls)


@pytest.mark.parametrize("case", STATIC_CASES, ids=lambda case: case.name)
def test_static_candidates_preserve_all_public_fields_and_independent_materials(case, monkeypatch):
    model, paths, mesh, rounded = build(case)
    distances = face_distances(mesh, rounded, paths)
    policy = ph.CollisionPolicy(margin=4)
    with ph.CollisionWorkspace(model) as full, ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        expected = ph.detect(model, paths, policy, full)
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert actual == expected
        row = actual.pairs[0]
        assert five_fields(row) == five_fields(expected.pairs[0])
        pose = row.mesh_round.end
        assert pose.relation == case.relation
        assert pose.axis_gap_m == pytest.approx(sqrt(float(min(distances))) - case.radius, abs=2e-12)
        if case.axis_gap_m is not None:
            assert pose.axis_gap_m == pytest.approx(case.axis_gap_m, abs=2e-12)
        assert calls and len(calls) == 1
        assert set(required_faces(mesh, rounded, paths)) <= set(calls[0]["rows"][0])
        assert calls[0]["key"][1].faces == mesh.triangles
        check_materials(row, pose, mesh, rounded, paths)
        if case is INSIDE:
            assert pose.axis_location == "inside" and not pose.boundary
            assert pose.boundary_gap_m == 7 / 8
            assert {s.face for s in pose.supports} == set(mesh.triangles)
            assert pose.witness.normal is None
        elif case is CONTAINS:
            assert pose.boundary_gap_m == pytest.approx(2 - sqrt(3), abs=2e-12)
            # Every original corner is furthest. The candidate subset must not
            # replace the whole-solid support enumeration: each face/node stays.
            addresses = {(s.face, s.witness.first_weights) for s in pose.supports}
            assert addresses == {(face, ((node, 1.0),)) for face in mesh.triangles for node in face}
            assert not pose.boundary and pose.witness.normal is None
        elif case is CROSSING:
            assert min(distances) == 0 and pose.axis_location == "on_boundary"
            assert pose.boundary and pose.boundary_gap_m == 0
        if min(distances) > 0 and case is not CONTAINS:
            nearest = {mesh.triangles[i] for i, d2 in enumerate(distances) if d2 == min(distances)}
            assert {s.face for s in pose.supports} == nearest


@pytest.mark.parametrize("case", (FACE, SHAFT), ids=lambda case: case.name)
def test_108_facet_box_really_prunes_without_losing_full_face_oracle(case, monkeypatch):
    model, paths, mesh, rounded = build(case, dense=True)
    assert len(mesh.triangles) == 108
    required = set(required_faces(mesh, rounded, paths))
    with ph.CollisionWorkspace(model) as full, ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        expected = ph.detect(model, paths, ph.CollisionPolicy(), full)
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert actual == expected
        assert actual.pairs[0].mesh_round.end.axis_gap_m == -1 / 32
        assert len(calls) == 1
        selected = set(calls[0]["rows"][0])
        assert required <= selected and 0 < len(selected) < 108
        check_materials(actual.pairs[0], actual.pairs[0].mesh_round.end, mesh, rounded, paths)


@pytest.mark.parametrize("case", SWEEP_CASES, ids=lambda case: case.name)
def test_translation_retains_earliest_event_and_empty_boundary_candidate_semantics(case, monkeypatch):
    model, paths, mesh, rounded = build(case)
    policy = ph.CollisionPolicy(margin=1 / 16)
    with ph.CollisionWorkspace(model) as full, ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        expected = ph.detect(model, paths, policy, full)
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert actual == expected
        row = actual.pairs[0]
        assert five_fields(row) == five_fields(expected.pairs[0])
        assert row.mesh_round.start.relation == row.mesh_round.end.relation == "separated"
        check_materials(row, row.mesh_round.start, mesh, rounded, paths)
        check_materials(row, row.mesh_round.end, mesh, rounded, paths, 1)
        assert len(calls) >= 3 and all(len(call["rows"]) == 1 for call in calls)
        if case.first_t is not None:
            assert row.status == "hit"
            lo, hi = row.sweep.first_t
            assert F(lo) <= case.first_t <= F(hi) and hi - lo <= 2**-38
            event = row.mesh_round.first_contact
            assert event.interval == row.sweep.first_t and event.witness_time == row.sweep.witness_time
            check_materials(row, event.pose, mesh, rounded, paths, event.witness_time)
            # Both bodies move, but the relative X speed is exactly four metres.
            assert np.any(paths[mesh.key].end != paths[mesh.key].start)
            assert min(face_distances(mesh, rounded, paths, case.first_t)) == F(case.radius) ** 2
        else:
            assert row.status == "separated" and row.sweep.first_t is None
            assert row.mesh_round.first_contact is None
            assert case is CLEAR_SWEEP
            # The entire swept sphere is above y=1 by 1/64 m. An empty
            # boundary candidate list must not erase start/end nearest witnesses.
            assert any(not call["rows"][0] for call in calls)
            assert calls[0]["rows"][0] and calls[1]["rows"][0]


@pytest.mark.parametrize("variant,reverse,reorder", [
    ("identity", True, False), ("identity", False, True),
    ("rigid", True, True), ("mirror", False, True),
])
def test_material_permutation_order_and_rigid_covariance(variant, reverse, reorder):
    model, paths, mesh, rounded = build(SHAFT, variant=variant, reverse=reverse, reorder=reorder)
    assert min(face_distances(mesh, rounded, paths)) == F(9, 1024)
    with ph.CollisionWorkspace(model) as full, ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        expected = ph.detect(model, paths, ph.CollisionPolicy(), full)
        actual = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert actual == expected and actual.pairs[0].mesh_round.end.axis_gap_m == -1 / 32
        row = actual.pairs[0]
        check_materials(row, row.mesh_round.end, mesh, rounded, paths)
        parameters = [p.axis_parameter for support in row.mesh_round.end.supports
                      for p in support.witness.capsule_axis_span]
        assert min(parameters) == 0 and max(parameters) == 1


def test_same_key_actual_coordinate_refit_and_transaction_capacity_rollback(monkeypatch):
    model, paths, mesh, rounded = build(FACE)
    changed = {**paths, mesh.key: ph.CollisionMotion(paths[mesh.key].start - (1 / 2, 0, 0))}
    policy = ph.CollisionPolicy(margin=1)
    with ph.CollisionWorkspace(model) as full, ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        first = ph.detect(model, paths, policy, work)
        before = asdict(first)
        layouts = work._trees.layouts.copy()
        count = len(calls)
        rejected = ph.detect(model, changed, replace(policy, candidate_capacity=0), work)
        assert rejected.status == "capacity_exceeded" and not rejected.pairs
        assert work.last_result is first and work._trees.layouts == layouts and len(calls) == count
        with pytest.raises(RuntimeError, match="caller abort"):
            with work.transaction():
                staged = ph.detect(model, changed, policy, work)
                assert staged.pairs[0].status == "separated"
                assert work.last_result is first
                raise RuntimeError("caller abort")
        assert work.last_result is first and work._trees.layouts == layouts
        assert asdict(first) == before
        recovered = ph.detect(model, changed, policy, work)
        assert recovered == ph.detect(model, changed, policy, full)
        assert recovered.pairs[0].mesh_round.end.axis_gap_m == 15 / 32
        assert calls[0]["key"] == calls[-1]["key"]
        assert not np.array_equal(calls[0]["leaves"], calls[-1]["leaves"])
        assert work.stats()["mesh_bvh_builds"] == 1
        assert ph.detect(model, paths, policy, work) == first
        newer = ph.CollisionModel(tuple(replace(shape, revision=1) for shape in model.shapes))
        with pytest.raises(ValueError, match="different collision model"):
            ph.detect(newer, paths, policy, work)
        assert work.last_result == first
    with ph.CollisionWorkspace(newer, mesh_midphase="bvh") as fresh:
        result = ph.detect(newer, paths, policy, fresh)
        assert result.pairs[0].mesh_round == first.pairs[0].mesh_round
        assert all(bound.revision == 1 for bound in result.bounds)


@pytest.mark.parametrize("kind", ("different_world", "same_rigid_body", "mask", "excluded"))
def test_topological_filters_do_not_launch_face_search(kind, monkeypatch):
    model, paths, mesh, rounded = build(FACE)
    old_key = rounded.key
    if kind == "different_world":
        rounded = replace(rounded, key=ph.CollisionKey(1, rounded.key.body, rounded.key.shape))
    elif kind == "same_rigid_body":
        rounded = replace(rounded, key=ph.CollisionKey(0, mesh.key.body, 1))
    elif kind == "mask":
        rounded = replace(rounded, mask=0)
    paths = {mesh.key: paths[mesh.key], rounded.key: paths[old_key]}
    exclusions = ((mesh.key, rounded.key),) if kind == "excluded" else ()
    model = ph.CollisionModel((mesh, rounded), exclusions)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        def forbidden(*args):
            raise AssertionError("filtered material reached face traversal")
        monkeypatch.setattr(work._trees, "facets", forbidden)
        row = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs[0]
        assert row.status == "filtered" and row.reason == kind and row.mesh_round is None


def test_aabb_rejection_still_validates_entire_closed_topology(monkeypatch):
    _, paths, mesh, rounded = build(FACE)
    mesh = replace(mesh, triangles=mesh.triangles[:-1])
    paths[rounded.key] = ph.CollisionMotion([[20, 0, 0]])
    model = ph.CollisionModel((mesh, rounded))
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        def forbidden(*args):
            raise AssertionError("code5 must not run a face query")
        monkeypatch.setattr(work._trees, "facets", forbidden)
        row = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work).pairs[0]
        assert not row.candidate and row.status == "invalid_geometry"
        assert row.reason == "mesh_open_or_misoriented_edge" and row.mesh_round is None


def test_double_turn_uses_existing_full_path_without_this_static_selector(monkeypatch):
    from dual_turn_mesh_round_cases import B01, public_case

    model, paths, _, _ = public_case(B01)
    with ph.CollisionWorkspace(model) as full, ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        expected = ph.detect(model, paths, ph.CollisionPolicy(), full)
        def forbidden(*args):
            raise AssertionError("dual-turn path entered static face selector")
        monkeypatch.setattr(work._trees, "facets", forbidden)
        actual = ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert actual == expected and actual.pairs[0].status == "hit"
        assert work.stats()["mesh_bvh_refits"] == 0


@pytest.mark.parametrize("delta", (0.0, -(2**-26), 2**-26))
def test_fp32_boundary_rounding_preserves_fp64_gap_and_all_ties(delta, monkeypatch):
    case = Case("boundary", ((9 / 8 + delta, 0, 0),), 1 / 8,
                "touching" if not delta else "boundary_overlap" if delta < 0 else "separated", delta)
    model, paths, mesh, rounded = build(case)
    policy = ph.CollisionPolicy(margin=2**-20)
    with ph.CollisionWorkspace(model) as full, ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        expected = ph.detect(model, paths, policy, full)
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert actual == expected and actual.pairs[0].mesh_round.end.axis_gap_m == delta
        assert actual.pairs[0].mesh_round.end.relation == case.relation
        assert set(required_faces(mesh, rounded, paths)) <= set(calls[0]["rows"][0])


@pytest.mark.parametrize("case", (FACE, INSIDE, CROSSING))
def test_selector_callback_binding_and_exact_nearest_upper_bound(case):
    callback = BoxOracle(scrambled=True)
    _, paths, mesh, rounded, prepared, selector, distances = prepare_selector(case, callback)
    axis = paths[rounded.key].start
    selected = selector.sample(axis[0], axis[-1], radius=F(rounded.radius), vertex_distance2=distances)
    assert selected == tuple(sorted(set(selected)))
    assert set(required_faces(mesh, rounded, paths)) <= set(selected)
    binding, leaves, queries, raw = callback.calls[0]
    assert selected == raw
    assert (binding.key, binding.revision, binding.vertex_ids, binding.faces) == (
        mesh.key, mesh.revision, mesh.vertex_ids, mesh.triangles)
    assert hash(binding) == hash(binding)
    for index, face in enumerate(mesh.triangles):
        for node in face:
            for k, value in enumerate(prepared.points[node]):
                assert F(float(leaves[index, k])) <= value <= F(float(leaves[index, 3 + k]))
    # At least sqrt(U²), not an ordinary inward-rounded float sqrt. This
    # bound covers closest faces even when a small sphere lies deep inside.
    upper2 = min(distances.values())
    for k in range(3):
        low, high = min(F(axis[0, k]), F(axis[-1, k])), max(F(axis[0, k]), F(axis[-1, k]))
        lower_pad = low - F(float(queries[0, k]))
        upper_pad = F(float(queries[0, k + 3])) - high
        assert lower_pad >= rounded.radius and upper_pad >= rounded.radius
        assert lower_pad**2 >= upper2 and upper_pad**2 >= upper2


@pytest.mark.parametrize("reply", (None, [], [[], []], [[-1]], [[12]], [[True]], [[1.0]], [[]]))
def test_selector_bad_or_empty_sample_fails_without_full_face_fallback(reply):
    from phymeo.core.capabilities import SolverFailure

    calls = []
    def bad(*args):
        calls.append(args)
        return reply
    _, paths, _, rounded, _, selector, distances = prepare_selector(FACE, bad)
    axis = paths[rounded.key].start
    with pytest.raises(SolverFailure):
        selector.sample(axis[0], axis[-1], radius=rounded.radius, vertex_distance2=distances)
    assert len(calls) == 1


def test_selector_complete_vertex_binding_and_empty_translation_are_distinct():
    callback = BoxOracle()
    _, paths, _, rounded, _, selector, distances = prepare_selector(CLEAR_SWEEP, callback)
    axis = paths[rounded.key].start
    assert selector.translation(axis[0], axis[-1], (4, 0, 0), radius=rounded.radius) == ()
    assert selector.sample(axis[0], axis[-1], radius=rounded.radius, vertex_distance2=distances)
    calls = len(callback.calls)
    for invalid in ({}, {**distances, 9999: F(0)}, {key: F(-1) for key in distances}):
        with pytest.raises(ValueError):
            selector.sample(axis[0], axis[-1], radius=rounded.radius, vertex_distance2=invalid)
    assert len(callback.calls) == calls


def test_selector_rejects_mismatched_material_rows_and_finite_storage_overflow():
    from phymeo.geometry._closed_mesh import ClosedMesh
    from phymeo.geometry._mesh_round_candidates import MeshRoundFaceSelector

    _, paths, mesh, rounded = build(FACE)
    callback = BoxOracle()
    prepared = ClosedMesh(mesh, paths[mesh.key])
    changed = replace(mesh, triangles=tuple((a, c, b) for a, b, c in mesh.triangles))
    with pytest.raises(ValueError):
        MeshRoundFaceSelector(changed, prepared, callback)
    selector = MeshRoundFaceSelector(mesh, prepared, callback)
    with pytest.raises(ValueError, match="finite FP32"):
        selector.translation((1e40, 0, 0), (1e40, 1, 0), (0, 0, 0), radius=rounded.radius)
    assert not callback.calls


def test_public_empty_sample_is_unresolved_not_clear_or_full_enumeration(monkeypatch):
    model, paths, _, _ = build(FACE)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = []
        def empty(*args):
            calls.append(args)
            return [[]]
        monkeypatch.setattr(work._trees, "facets", empty)
        row = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs[0]
        assert calls and row.status == "unresolved" and row.witness is None
        assert row.mesh_round is None and "empty" in row.reason
