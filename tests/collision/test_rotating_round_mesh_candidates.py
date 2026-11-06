"""Rotating mesh-round face selection, with existing independent path oracles."""

from dataclasses import asdict

import pytest
from dual_turn_mesh_round_cases import B01, B02, B03, B04, public_case, root_interval, static_paths
from mesh_round_candidate_cases import record_facets
from test_dual_turn_mesh_round import check_entry, check_materials

import phymeo as ph


def assert_face_calls(calls, mesh, driver):
    assert calls, "rotating mesh-round branch never queried per-face candidates"
    for call in calls:
        binding = call["key"][1]
        assert binding.key == mesh.key
        assert binding.faces == mesh.triangles and binding.vertex_ids == mesh.vertex_ids
        assert call["driver"] is driver
        assert len(call["rows"]) == 1 and call["rows"][0]
        assert call["rows"][0] == tuple(sorted(set(call["rows"][0])))
        assert all(0 <= index < len(mesh.triangles) for index in call["rows"][0])


def test_turning_pair_queries_real_face_candidates_after_independent_hit(monkeypatch):
    model, paths, mesh, capsule = public_case(B01)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as full:
        static = ph.detect(model, static_paths(paths, 3 / 8), policy, full).pairs[0]
        assert static.status == "overlap" and static.mesh_round.start.axis_gap_m < 0
        expected = ph.detect(model, paths, policy, full)
    reference = root_interval(B01)
    check_entry(expected.pairs[0], reference, mesh, capsule, paths)
    assert expected.pairs[0].mesh_round.start.relation == "separated"
    assert expected.pairs[0].mesh_round.end.relation == "separated"
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert asdict(actual) == asdict(expected)
        check_entry(actual.pairs[0], reference, mesh, capsule, paths)
        assert_face_calls(calls, mesh, None)


@pytest.mark.parametrize("case", (B02, B03, B04), ids=lambda case: case.name)
def test_dual_turn_sphere_pivot_and_shaft_use_original_material_faces(case, monkeypatch):
    model, paths, mesh, rounded = public_case(case)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as full:
        expected = ph.detect(model, paths, policy, full)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert asdict(actual) == asdict(expected)
        event = check_entry(actual.pairs[0], root_interval(case), mesh, rounded, paths)
        if case is B04:
            spans = check_materials(actual.pairs[0], event.pose, mesh, rounded, paths, event.witness_time)
            assert spans and all(0 < low <= high < 1 for low, high in spans)
        assert_face_calls(calls, mesh, None)


def single_case(kind, *, dense=False, world=0):
    """Two previously accepted single-turn paths, without production pose helpers."""
    from dataclasses import replace
    from math import pi

    import numpy as np
    from mesh_round_candidate_cases import subdivided_box
    from rotating_capsule_mesh_cases import B01 as SINGLE, box, public_case as single_public
    from test_rotating_mesh_sphere import scene as mesh_sphere_scene

    if kind == "capsule":
        return single_public(SINGLE, world=world)
    if dense:
        points, faces = subdivided_box()
        vertices = np.asarray(points) * (0.5, 0.125, 0.25) + (1.5, 0, 0)
    else:
        vertices, faces = box((1, -1 / 8, -1 / 4), (2, 1 / 8, 1 / 4))
    model, paths, mesh, rounded = mesh_sphere_scene(
        vertices, faces, (0, 1.5, 1 / 16), 1 / 8, pi, drift=(1 / 4, -3 / 8, 1 / 8),
    )
    if world:
        replacements = {s.key: replace(s, key=replace(s.key, world=world)) for s in model.shapes}
        paths = {replacements[key].key: path for key, path in paths.items()}
        mesh, rounded = replacements[mesh.key], replacements[rounded.key]
        model = ph.CollisionModel(tuple(replacements.values()))
    return model, paths, mesh, rounded


def assert_trace_materials(row, mesh, rounded, paths):
    trace = row.mesh_round
    assert trace is not None
    for pose, time in ((trace.start, 0), (trace.end, 1)):
        check_materials(row, pose, mesh, rounded, paths, time)
    if trace.first_contact is not None:
        event = trace.first_contact
        check_materials(row, event.pose, mesh, rounded, paths, event.witness_time)


@pytest.mark.parametrize("kind", ("capsule", "mesh_sphere"))
def test_single_turn_enters_between_clear_endpoints_and_uses_faces(kind, monkeypatch):
    from math import acos, pi

    model, paths, mesh, rounded = single_case(kind)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as full:
        expected = ph.detect(model, paths, policy, full)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert actual == expected
        row = actual.pairs[0]
        assert row.status == "hit"
        trace = row.mesh_round
        assert trace.start.relation == trace.end.relation == "separated"
        low, high = row.sweep.first_t
        assert low <= acos(1 / 6) / pi <= high and high - low <= 2**-27
        assert_trace_materials(row, mesh, rounded, paths)
        if kind == "capsule":
            # The whole axis has endpoint z=+-1, the finite box has |z|<=1/4.
            # Its endpoint balls therefore stay at least 5/8 m clear throughout.
            assert 1 - 1 / 4 - rounded.radius == 5 / 8
            spans = [point.axis_parameter for support in trace.first_contact.pose.supports
                     for point in support.witness.capsule_axis_span]
            assert min(spans) > 0 and max(spans) < 1
        assert_face_calls(calls, mesh, None)


def test_dense_rotating_mesh_prunes_actual_nearest_evaluations(monkeypatch):
    from phymeo.geometry import _rotating_mesh_sphere

    model, paths, mesh, rounded = single_case("mesh_sphere", dense=True)
    assert len(mesh.triangles) == 108
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as full:
        expected = ph.detect(model, paths, policy, full)
    visited = []
    original = _rotating_mesh_sphere._point_nearest

    def observed(facets, point):
        facets = tuple(facets)
        visited.append(tuple(f.ids for f in facets))
        return original(facets, point)

    monkeypatch.setattr(_rotating_mesh_sphere, "_point_nearest", observed)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert actual == expected
        assert actual.pairs[0].status == "hit"
        assert_face_calls(calls, mesh, None)
        assert any(0 < len(call["rows"][0]) < 108 for call in calls)
        assert visited and any(0 < len(faces) < 108 for faces in visited)
        original_faces = set(mesh.triangles)
        assert all(set(faces) <= original_faces for faces in visited)
        assert_trace_materials(actual.pairs[0], mesh, rounded, paths)


@pytest.mark.parametrize("kind", ("clear_height", "concave", "inside", "contains", "grazing"))
def test_solid_relations_and_ambiguous_grazing_are_not_replaced_by_candidate_boxes(kind, monkeypatch):
    from dataclasses import replace
    from fractions import Fraction as F
    from math import sqrt

    from dual_turn_mesh_round_cases import B05, B06, B07, B08

    case = {"clear_height": B05, "concave": B06, "inside": B07,
            "contains": replace(B07, points=((0.0, 0.0, 0.0),), radius=4.0),
            "grazing": B08}[kind]
    model, paths, mesh, rounded = public_case(case)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as full:
        expected = ph.detect(model, paths, policy, full)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert actual == expected
        assert_face_calls(calls, mesh, None)
        row = actual.pairs[0]
        if kind == "grazing":
            assert row.status in ("hit", "unresolved")
            if row.status == "unresolved":
                assert row.witness is row.mesh_round is row.sweep is None
            return
        assert_trace_materials(row, mesh, rounded, paths)
        if kind in ("clear_height", "concave"):
            clearance = 7 / 8 - 4 / (3 * sqrt(3)) if kind == "clear_height" else 5 / 8
            assert clearance > 0 and row.status == "separated" and row.sweep.first_t is None
            assert row.mesh_round.start.axis_gap_m >= clearance - 2e-12
        elif kind == "inside":
            check_entry(row, (F(0), F(0)), mesh, rounded, paths)
            assert row.mesh_round.start.relation == row.mesh_round.end.relation == "mesh_contains_round"
        else:
            assert row.sweep.first_t == (0, 0)
            assert row.mesh_round.start.relation == row.mesh_round.end.relation == "round_contains_mesh"
            # All eight cube corners are equally far at entry; preserve all face/node supports.
            start = row.mesh_round.start
            assert len(start.supports) == 36
            assert {s.face for s in start.supports} == set(mesh.triangles)
            assert start.boundary_gap_m == pytest.approx(4 - sqrt(12), abs=2e-12)


@pytest.mark.parametrize("variant", ("reorder", "cycle", "mirror"))
def test_material_order_and_rigid_change_of_frame_preserve_first_and_bilateral_witnesses(variant, monkeypatch):
    import numpy as np
    from dual_turn_mesh_round_cases import transform

    case = B03
    if variant != "reorder":
        matrix = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0))) if variant == "cycle" else np.diag((-1, 1, 1))
        case = transform(case, matrix, (9 / 4, -3 / 2, 3 / 8))
    model, paths, mesh, rounded = public_case(case, reverse=True, reorder=True)
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as full:
        expected = ph.detect(model, paths, policy, full)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert actual == expected
        check_entry(actual.pairs[0], root_interval(B03), mesh, rounded, paths)
        assert_face_calls(calls, mesh, None)


def test_default_query_does_not_use_the_new_sample_helper(monkeypatch):
    from phymeo.geometry._rotating_mesh_candidates import MeshRoundArcCandidates

    def forbidden(*args, **kwargs):
        raise AssertionError("default all_facets entered the opt-in face selector")

    monkeypatch.setattr(MeshRoundArcCandidates, "__init__", forbidden)
    model, paths, mesh, rounded = public_case(B02)
    with ph.CollisionWorkspace(model) as work:
        result = ph.detect(model, paths, ph.CollisionPolicy(), work)
        check_entry(result.pairs[0], root_interval(B02), mesh, rounded, paths)
        assert work.stats()["mesh_bvh_refits"] == 0


@pytest.mark.parametrize("kind", ("angle", "samples", "invalid_mesh", "different_world"))
def test_existing_limits_and_invalid_geometry_stay_explicit(kind, monkeypatch):
    from dataclasses import replace
    from phymeo.geometry import _dual_turn_mesh_round

    model, paths, mesh, rounded = public_case(B01, round_world=9 if kind == "different_world" else None)
    if kind == "angle":
        paths[rounded.key] = replace(paths[rounded.key], turn=replace(paths[rounded.key].turn, angle=33.0))
    elif kind == "samples":
        monkeypatch.setattr(_dual_turn_mesh_round, "_MAX_SAMPLES", 1)
    elif kind == "invalid_mesh":
        mesh = replace(mesh, triangles=mesh.triangles[:-1])
        model = ph.CollisionModel((mesh, rounded))
    policy = ph.CollisionPolicy()
    with ph.CollisionWorkspace(model) as full:
        expected = ph.detect(model, paths, policy, full)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        actual = ph.detect(model, paths, policy, work)
        assert actual == expected
        row = actual.pairs[0]
        assert row.status == {"angle": "unresolved", "samples": "unresolved",
                              "invalid_mesh": "invalid_geometry", "different_world": "filtered"}[kind]
        assert row.witness is row.mesh_round is row.sweep is None
        if kind in ("angle", "invalid_mesh", "different_world"):
            assert not calls
        else:
            assert calls


def arc_helper(case, callback, *, dense=False, world=0):
    from mesh_round_candidate_cases import build
    from phymeo.geometry._closed_mesh import ClosedMesh
    from phymeo.geometry._mesh_round_candidates import MeshRoundFaceSelector
    from phymeo.geometry._rotating_mesh_candidates import MeshRoundArcCandidates

    _, paths, mesh, rounded = build(case, dense=dense, world=world)
    prepared = ClosedMesh(mesh, paths[mesh.key])
    selector = MeshRoundFaceSelector(mesh, prepared, callback)
    return MeshRoundArcCandidates(prepared, selector), prepared, mesh, rounded, paths


@pytest.mark.parametrize("kind", ("nearest_ties", "finite_axis", "axis_crossing", "dense_near"))
def test_sample_helper_preserves_independent_nearest_ties_and_radius_boundary(kind):
    from fractions import Fraction as F
    from mesh_round_candidate_cases import BoxOracle, CROSSING, FACE, INSIDE, SHAFT, required_faces

    case = {"nearest_ties": INSIDE, "finite_axis": SHAFT,
            "axis_crossing": CROSSING, "dense_near": FACE}[kind]
    callback = BoxOracle(scrambled=True)
    helper, prepared, mesh, rounded, paths = arc_helper(case, callback, dense=kind == "dense_near")
    ends = paths[rounded.key].start
    selected = helper.sample_axis(ends, radius=F(rounded.radius))
    assert selected == tuple(sorted(set(selected)))
    assert set(required_faces(mesh, rounded, paths)) <= set(selected)
    assert callback.calls and len(callback.calls) == 1
    binding, leaves, queries, _ = callback.calls[0]
    assert binding.vertex_ids == tuple(prepared.points)
    assert binding.faces == mesh.triangles
    if kind == "nearest_ties":
        assert selected == tuple(range(12))  # all six faces tie at distance 1 > radius
    if kind == "dense_near":
        assert 0 < len(selected) < 108
    for face_index, face in enumerate(mesh.triangles):
        for node in face:
            assert all(F(float(leaves[face_index, k])) <= value <= F(float(leaves[face_index, k + 3]))
                       for k, value in enumerate(prepared.points[node]))
    assert queries.shape == (1, 6)


def test_helper_requeries_each_actual_axis_instead_of_reusing_previous_indices():
    import numpy as np
    from mesh_round_candidate_cases import BoxOracle, FACE
    from rotating_capsule_mesh_cases import segment_triangle_reference

    callback = BoxOracle()
    helper, prepared, mesh, rounded, paths = arc_helper(FACE, callback, dense=True)
    ends = paths[rounded.key].start.copy()
    first = helper.sample_axis(ends, radius=rounded.radius)
    ends[:, 0] *= -1
    second = helper.sample_axis(ends, radius=rounded.radius)
    assert first != second and len(callback.calls) == 2
    squares = [segment_triangle_reference((ends[0], ends[-1]), [prepared.points[i] for i in face])[0]
               for face in mesh.triangles]
    required = {i for i, d2 in enumerate(squares) if d2 == min(squares) or d2 <= rounded.radius**2}
    assert required <= set(second)
    np.testing.assert_array_equal(paths[rounded.key].start, [[35 / 32, 0, 0]])


@pytest.mark.parametrize("kind", ("world", "coordinates", "faces"))
def test_helper_rejects_selector_from_another_material_geometry(kind):
    from dataclasses import replace
    from mesh_round_candidate_cases import BoxOracle, FACE, build
    from phymeo.geometry._closed_mesh import ClosedMesh
    from phymeo.geometry._mesh_round_candidates import MeshRoundFaceSelector
    from phymeo.geometry._rotating_mesh_candidates import MeshRoundArcCandidates

    _, paths, mesh, _ = build(FACE)
    prepared = ClosedMesh(mesh, paths[mesh.key])
    callback = BoxOracle()
    selector = MeshRoundFaceSelector(mesh, prepared, callback)
    other, path = mesh, paths[mesh.key]
    if kind == "world":
        other = replace(mesh, key=replace(mesh.key, world=8))
    elif kind == "coordinates":
        path = ph.CollisionMotion(path.start + (1 / 64, 0, 0))
    else:
        # The x-min quad originally uses diagonal a--d. Flip it to b--c;
        # merely reversing row order is erased by CollisionShape canonicalization.
        a, b, c, d = mesh.vertex_ids[:4]
        quad = {a, b, c, d}
        faces = tuple(face for face in mesh.triangles if not set(face) <= quad)
        other = replace(mesh, triangles=(*faces, (a, b, c), (b, d, c)))
        assert other.triangles != mesh.triangles
    other_prepared = ClosedMesh(other, path)
    with pytest.raises(ValueError, match="binding"):
        MeshRoundArcCandidates(other_prepared, selector)
    assert not callback.calls


@pytest.mark.parametrize("ends,radius", (
    ((), 1 / 8),
    (((0, 0, 0),) * 3, 1 / 8),
    (((0, 0),), 1 / 8),
    (((float("nan"), 0, 0),), 1 / 8),
    (((True, 0, 0),), 1 / 8),
    (((0, 0, 0),), 0),
    (((0, 0, 0),), -1),
))
def test_invalid_axis_is_rejected_before_callback(ends, radius):
    from mesh_round_candidate_cases import BoxOracle, FACE

    callback = BoxOracle()
    helper, _, _, _, _ = arc_helper(FACE, callback)
    with pytest.raises((TypeError, ValueError)):
        helper.sample_axis(ends, radius=radius)
    assert not callback.calls


@pytest.mark.parametrize("reply", ([[]], [[-1]], [[True]]))
def test_helper_never_turns_empty_or_invalid_sample_into_full_faces(reply):
    from mesh_round_candidate_cases import FACE
    from phymeo.core.capabilities import SolverFailure

    helper, _, _, rounded, paths = arc_helper(FACE, lambda *args: reply)
    with pytest.raises(SolverFailure):
        helper.sample_axis(paths[rounded.key].start, radius=rounded.radius)


def test_capacity_abort_refit_and_retry_keep_prior_publication(monkeypatch):
    from dataclasses import replace
    import numpy as np

    model, paths, mesh, rounded = public_case(B03)
    policy = ph.CollisionPolicy()
    shift = np.array((1 / 32, 0, 0))
    turn = paths[mesh.key].turn
    changed = paths | {mesh.key: ph.CollisionMotion(
        paths[mesh.key].start + shift,
        turn=replace(turn, pivot_start=tuple(np.asarray(turn.pivot_start) + shift),
                     pivot_end=tuple(np.asarray(turn.pivot_end) + shift)),
    )}
    with ph.CollisionWorkspace(model) as full:
        changed_expected = ph.detect(model, changed, policy, full)
    with ph.CollisionWorkspace(model, mesh_midphase="bvh") as work:
        calls = record_facets(work, monkeypatch)
        previous = ph.detect(model, paths, policy, work)
        before, layouts, count = asdict(previous), work._trees.layouts.copy(), len(calls)
        rejected = ph.detect(model, changed, ph.CollisionPolicy(candidate_capacity=0), work)
        assert rejected.status == "capacity_exceeded" and not rejected.pairs
        assert work.last_result is previous and work._trees.layouts == layouts and len(calls) == count
        with pytest.raises(RuntimeError, match="caller abort"):
            with work.transaction():
                staged = ph.detect(model, changed, policy, work)
                assert staged == changed_expected and work.last_result is previous
                raise RuntimeError("caller abort")
        assert work.last_result is previous and work._trees.layouts == layouts
        actual = ph.detect(model, changed, policy, work)
        assert actual == changed_expected and asdict(previous) == before
        assert work.stats()["mesh_bvh_builds"] == 1
        assert calls[0]["key"] == calls[-1]["key"]
        assert not np.array_equal(calls[0]["leaves"], calls[-1]["leaves"])
        assert ph.detect(model, paths, policy, work) == previous
