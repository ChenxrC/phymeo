"""Public finite-heightfield witnesses for a rounded material node, including world offsets."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

CATALOG = Path(__file__).parents[1] / "reference_newton" / "rd" / "cases.json"
FROZEN = next(
    row
    for row in json.loads(CATALOG.read_text(encoding="utf-8"))["cases"]
    if row["id"] == "RD-HEIGHTFIELD"
)
RADIUS = 1 / 16


def flat():
    return ph.Heightfield([[0.25] * 3] * 3, (1, 1), origin=(-1, -1, 0))


def scene(grid, start, radius, *, end=None, reverse=False):
    field_key = ph.CollisionKey(0, 2 if reverse else 1, 0)
    node_key = ph.CollisionKey(0, 1 if reverse else 2, 0)
    ids = tuple(17 + 3 * i for i in range(grid.vertex_count))
    field = ph.CollisionShape(field_key, "heightfield", ids, heightfield=grid)
    node = ph.CollisionShape(node_key, "rounded_particle", (73,), radius)
    model = ph.CollisionModel((node, field))
    paths = {
        field_key: ph.CollisionMotion(grid.world_vertices()),
        node_key: ph.CollisionMotion([start], None if end is None else [end]),
    }
    return model, paths, field, node


def query(model, paths, margin=0.0):
    with ph.CollisionWorkspace(model) as work:
        result = ph.detect(model, paths, ph.CollisionPolicy(margin=margin), work)
    assert result.status == "ok"
    (row,) = result.pairs
    return row


def check(row, witness, face, field, node, center, ground_point, normal, gap):
    assert row.family == "RD" and witness.source == "host-heightfield"
    terrain_first = row.first == field.key
    assert {row.first, row.second} == {field.key, node.key}
    p, n = np.asarray(ground_point), np.asarray(normal)
    if terrain_first:
        ground, skin = witness.first_point, witness.second_point
        terrain_weights, node_weights = witness.first_weights, witness.second_weights
        ground_offset, node_offset = witness.first_offset, witness.second_offset
    else:
        skin, ground = witness.first_point, witness.second_point
        node_weights, terrain_weights = witness.first_weights, witness.second_weights
        node_offset, ground_offset = witness.first_offset, witness.second_offset
    assert ground_offset is None
    assert node_weights == ((node.vertex_ids[0], 1.0),)
    assert len(terrain_weights) == 3 and {i for i, _ in terrain_weights} == set(face.vertex_ids)
    vertices = dict(zip(field.vertex_ids, field.heightfield.world_vertices(), strict=True))
    weighted_ground = sum(w * vertices[i] for i, w in terrain_weights)
    tri = np.array([vertices[key] for key in face.vertex_ids])
    assert np.all(np.asarray(face.barycentric) >= 0)
    assert sum(face.barycentric) == pytest.approx(1.0, abs=2e-12)
    np.testing.assert_allclose(face.barycentric @ tri, p, rtol=0, atol=2e-12)
    np.testing.assert_allclose(weighted_ground, p, rtol=0, atol=2e-12)
    np.testing.assert_allclose(ground, p, rtol=0, atol=2e-12)
    np.testing.assert_allclose(node_offset, -node.radius * n, rtol=0, atol=2e-12)
    np.testing.assert_allclose(skin, np.asarray(center) + node_offset, rtol=0, atol=2e-12)
    np.testing.assert_allclose(witness.normal, n if terrain_first else -n, rtol=0, atol=2e-12)
    assert witness.gap_m == pytest.approx(gap, abs=2e-12)
    assert np.dot(
        np.subtract(witness.second_point, witness.first_point), witness.normal
    ) == pytest.approx(gap, abs=2e-12)


def test_rounded_particle_registration_is_opt_in():
    key = ph.CollisionKey(0, 2, 0)
    node = ph.CollisionShape(key, "rounded_particle", (73,), RADIUS)
    assert not node.rigid and node.radius == RADIUS
    assert not ph.CollisionShape(key, "particle", (73,)).rigid
    with pytest.raises(ValueError):
        ph.CollisionShape(key, "particle", (73,), RADIUS)
    with pytest.raises(ValueError):
        ph.CollisionShape(key, "rounded_particle", (73,), 0)
    with pytest.raises(ValueError):
        ph.CollisionShape(key, "rounded_particle", (73, 88), RADIUS)


@pytest.mark.parametrize("reverse", [False, True])
def test_frozen_nodes_and_finite_boundary_distances(reverse):
    vertices = np.asarray(FROZEN["inputs"]["vertices_m"])
    radius = FROZEN["inputs"]["particle_radius_m"]
    outcomes = []
    for i, (center, foot, n, gap) in enumerate(
        zip(
            vertices,
            [
                [-0.125, 0.125, 0.25],
                [1, 0.5, 0.25],
                [-0.5, 0.5, 0.25],
            ],
            [
                [0, 0, 1],
                np.array([3, 0, -1]) / np.sqrt(10),
                [0, 0, 1],
            ],
            [-1 / 32, (np.sqrt(10) - 2) / 32, 3 / 16],
            strict=True,
        )
    ):
        model, paths, field, node = scene(flat(), center, radius, reverse=reverse)
        raw = query(model, paths)
        outcomes.append(raw.status)
        # A separate enlarged candidate margin asks for positive nearest distances.
        diagnostic = raw if i == 0 else query(model, paths, margin=0.25)
        check(
            diagnostic,
            diagnostic.witness,
            diagnostic.heightfield,
            field,
            node,
            center,
            foot,
            n,
            gap,
        )
        if i == 0:
            assert diagnostic.heightfield.cell == (1, 0) and diagnostic.heightfield.triangle == 0
            np.testing.assert_allclose(diagnostic.heightfield.barycentric, [1 / 8, 3 / 4, 1 / 8])
    assert outcomes == ["overlap", "separated", "separated"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "center,radius,foot,normal,gap,status",
    [
        ([-0.125, 0.125, 7 / 32], 1 / 16, [-0.125, 0.125, 1 / 4], [0, 0, -1], -1 / 32, "overlap"),
        (
            [1 + 3 / 64, 3 / 8, 1 / 4 - 4 / 64],
            5 / 64,
            [1, 3 / 8, 1 / 4],
            [0.6, 0, -0.8],
            0,
            "touching",
        ),
    ],
)
def test_back_face_and_lateral_edge(reverse, center, radius, foot, normal, gap, status):
    model, paths, field, node = scene(flat(), center, radius, reverse=reverse)
    row = query(model, paths)
    assert row.status == status
    check(row, row.witness, row.heightfield, field, node, center, foot, normal, gap)


@pytest.mark.parametrize("reverse", [False, True])
def test_rotated_nonuniform_slope_keeps_body_vertex_and_skin_coordinates(reverse):
    rotation = np.array([[0.0, 0, 1], [1.0, 0, 0], [0, 1.0, 0]])
    origin = np.array([2.0, -3, 0.5])
    grid = ph.Heightfield([[0, 3 / 8, 3 / 4]] * 2, (0.5, 0.75), origin=origin, rotation=rotation)
    # dx=1/2, dy=3/4. Triangle SW/SE/NE at (row 0, col 1) with weights 1/4,1/2,1/4.
    foot = np.array([2.65625, -2.125, 0.6875])
    n = np.array([0.8, -0.6, 0])
    center = np.array([2.71875, -2.171875, 0.6875])
    model, paths, field, node = scene(grid, center, 1 / 8, reverse=reverse)
    row = query(model, paths)
    assert row.status == "overlap" and row.heightfield.cell == (0, 1)
    assert row.heightfield.triangle == 0 and row.heightfield.vertex_ids == (20, 23, 32)
    np.testing.assert_allclose(row.heightfield.barycentric, [0.25, 0.5, 0.25], rtol=0, atol=2e-12)
    check(row, row.witness, row.heightfield, field, node, center, foot, n, -3 / 64)


def check_sweep(row, field, node, start, end):
    assert row.status == "hit" and row.family == "RD"
    foot = np.array([-0.125, 0.125, 0.25])
    n = np.array([0, 0, 1 if start[2] > end[2] else -1])
    first, last = row.heightfield.first_contact, row.heightfield.last_contact
    assert row.sweep.first_t == first.interval and row.sweep.witness == first.witness
    for event, time, normal in ((first, Fraction(3, 8), n), (last, Fraction(5, 8), -n)):
        lo, hi = event.interval
        assert Fraction(lo) <= time <= Fraction(hi) and hi - lo <= 2**-40
        center = start + event.witness_time * (end - start)
        assert lo <= event.witness_time <= hi
        check(row, event.witness, event.face, field, node, center, foot, normal, 0)
    assert row.sweep.start_gap_m == pytest.approx(3 / 16)
    assert row.sweep.end_gap_m == pytest.approx(3 / 16)
    check(row, row.witness, row.heightfield, field, node, end, foot, -n, 3 / 16)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("upward", [False, True])
def test_affine_node_crosses_a_finite_heightfield_with_separated_endpoints(reverse, upward):
    bottom, top = np.array([-0.125, 0.125, 0]), np.array([-0.125, 0.125, 0.5])
    start, end = (bottom, top) if upward else (top, bottom)
    model, paths, field, node = scene(flat(), start, RADIUS, reverse=reverse, end=end)
    check_sweep(query(model, paths), field, node, start, end)


def test_oblique_crossing_with_tangential_drift_publishes_two_distinct_ground_points():
    # The sloping patch is vertical in world Z, with world normal (4/5,-3/5,0).
    rot = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    grid = ph.Heightfield([[0, 3 / 8, 3 / 4]] * 2, (0.5, 0.75), origin=(2, -3, 0.5), rotation=rot)
    start = np.array([2.90625, -2.3125, 0.59375])
    end = np.array([2.40625, -1.9375, 0.78125])
    radius = 5 / 64
    model, paths, field, node = scene(grid, start, radius, end=end)
    row = query(model, paths)
    assert row.status == "hit"
    assert row.sweep.start_gap_m == pytest.approx(15 / 64)
    assert row.sweep.end_gap_m == pytest.approx(15 / 64)
    n = np.array([0.8, -0.6, 0.0])
    for e, t, foot, normal in (
        (row.heightfield.first_contact, 3 / 8, [2.65625, -2.125, 0.6640625], n),
        (row.heightfield.last_contact, 5 / 8, [2.65625, -2.125, 0.7109375], -n),
    ):
        lo, hi = e.interval
        assert lo <= t <= hi and hi - lo <= 2**-40
        center = start + e.witness_time * (end - start)
        check(row, e.witness, e.face, field, node, center, foot, normal, 0)


def test_static_terrain_contract_and_material_turn_boundary():
    start = np.array([-0.125, 0.125, 0.28125])
    model, paths, field, node = scene(flat(), start, RADIUS)
    ground = field.heightfield.world_vertices()
    moving_field = {**paths, field.key: ph.CollisionMotion(ground, ground + [0, 0, 1 / 8])}
    row = query(model, moving_field)
    assert row.candidate and row.status == "unsupported"
    arc = ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), 0.25)
    with pytest.raises(ValueError, match="rigid turn requires a rigid shape"):
        query(model, {**paths, node.key: ph.CollisionMotion([start], turn=arc)})


def test_shared_node_center_paths_are_validated_before_publication():
    center = np.array([-0.125, 0.125, 0.28125])
    model, paths, field, node = scene(flat(), center, RADIUS)
    # The same material vertex is present in a triangle and a rounded single-node registration.
    tri = ph.CollisionShape(
        ph.CollisionKey(node.key.world, node.key.body, 1), "triangle", (73, 74, 75)
    )
    points = np.array([center, center + [0.25, 0, 0], center + [0, 0.25, 0]])
    model = ph.CollisionModel((*model.shapes, tri))
    paths[tri.key] = ph.CollisionMotion(points)
    with ph.CollisionWorkspace(model) as work:
        previous = ph.detect(model, paths, ph.CollisionPolicy(), work)
        row = next(p for p in previous.pairs if {p.first, p.second} == {field.key, node.key})
        assert row.status == "overlap"
        for start, end in (
            ([center + [0, 0, 1 / 64]], None),
            ([center], [center + [0, 0, 1 / 64]]),
        ):
            with pytest.raises(ValueError, match="shared material vertex"):
                ph.detect(
                    model,
                    {**paths, node.key: ph.CollisionMotion(start, end)},
                    ph.CollisionPolicy(),
                    work,
                )
            assert work.last_result is previous
        assert ph.detect(model, paths, ph.CollisionPolicy(), work) == previous


def test_full_triangle_and_other_rounded_primitive_pairs_have_their_own_dispatch():
    field = flat()
    fk, tk = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    land = ph.CollisionShape(fk, "heightfield", tuple(range(field.vertex_count)), heightfield=field)
    tri = ph.CollisionShape(tk, "triangle", (0, 1, 2))
    points = [[-0.25, -0.25, 0.125], [0.25, -0.25, 0.375], [0, 0.25, 0.375]]
    row = query(
        ph.CollisionModel((land, tri)),
        {fk: ph.CollisionMotion(field.world_vertices()), tk: ph.CollisionMotion(points)},
    )
    assert row.family == "RD" and row.candidate and row.status == "unsupported"
    sphere = ph.CollisionShape(fk, "sphere", (0,), 1 / 8)
    node = ph.CollisionShape(tk, "rounded_particle", (8,), 1 / 8)
    paths = {fk: ph.CollisionMotion([[0, 0, 0]]), tk: ph.CollisionMotion([[0.125, 0, 0]])}
    other = query(ph.CollisionModel((sphere, node)), paths)
    assert other.candidate and other.status == "unsupported"


@pytest.mark.tyx
def test_real_terrain_face_mask_precedes_host_node_events(tyx_mode, tmp_path, monkeypatch):
    start, end = np.array([-0.125, 0.125, 0.5]), np.array([-0.125, 0.125, 0])
    model, paths, field, node = scene(flat(), start, RADIUS, reverse=True, end=end)
    observed = []
    with ph.CollisionWorkspace(model, tyx_mode, cache_root=tmp_path) as work:
        select = work._engine.heightfield_face_candidates

        def record_selected(*args):
            selected = select(*args)
            observed.append(tuple(selected))
            return selected

        monkeypatch.setattr(work._engine, "heightfield_face_candidates", record_selected)
        result = ph.detect(model, paths, ph.CollisionPolicy(), work)
        (row,) = result.pairs
        check_sweep(row, field, node, start, end)
        stats = work.stats()
    assert observed == [(4, 5)]
    assert stats["heightfield_completed_launches"] == 1 and stats["heightfield_cells_tested"] == 4
    assert stats["heightfield_faces_kept"] == 2
    if tyx_mode == "cuda":
        assert stats["heightfield_downloads"] == stats["heightfield_synchronizations"] == 1
    print(
        json.dumps(
            dict(
                backend=tyx_mode,
                selected_faces=observed[0],
                first=row.sweep.first_t,
                last=row.heightfield.last_contact.interval,
                source=row.witness.source,
                stats=stats,
            ),
            allow_nan=False,
        )
    )
