"""Public rigid sphere contacts with a finite, two-sided triangle height graph."""

import json
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

CATALOG = json.loads(
    (Path(__file__).parents[1] / "reference_newton" / "rr" / "cases.json").read_text("utf-8")
)
FROZEN = ("RR-RIDGE-POSES", "RR-RIDGE-MISS", "RR-SLOPE-FACE")


def scene(grid, center, radius, *, reverse=False, end=None, terrain_motion=None, exclude=False):
    terrain_key = ph.CollisionKey(0, 2 if reverse else 1, 0)
    ball_key = ph.CollisionKey(0, 1 if reverse else 2, 0)
    ids = tuple(17 + 3 * i for i in range(grid.vertex_count))
    terrain = ph.CollisionShape(terrain_key, "heightfield", ids, heightfield=grid)
    ball = ph.CollisionShape(ball_key, "sphere", (11,), radius)
    excluded = [(terrain_key, ball_key)] if exclude else []
    model = ph.CollisionModel([terrain, ball], excluded)
    paths = {
        terrain_key: terrain_motion or ph.CollisionMotion(grid.world_vertices()),
        ball_key: ph.CollisionMotion([center], None if end is None else [end]),
    }
    return model, paths, terrain


def detect(model, paths, *, margin=0.0):
    with ph.CollisionWorkspace(model) as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(margin=margin), work).pairs
    return row


def check_witness(row, terrain, center, radius, point, normal, gap, tol=1e-11):
    witness, chart = row.witness, row.heightfield
    assert row.family == "RR" and row.sweep is None
    assert isinstance(chart, ph.HeightfieldTrace)
    terrain_first = row.first == terrain.key
    ground_point, sphere_point = witness.first_point, witness.second_point
    ground_weights, ball_weights = witness.first_weights, witness.second_weights
    if not terrain_first:
        ground_point, sphere_point = sphere_point, ground_point
        ground_weights, ball_weights = ball_weights, ground_weights
    direction = np.asarray(normal)
    np.testing.assert_allclose(witness.normal, direction if terrain_first else -direction, atol=tol)
    np.testing.assert_allclose(ground_point, point, atol=tol)
    np.testing.assert_allclose(sphere_point, center - radius * direction, atol=tol)
    assert witness.gap_m == pytest.approx(gap, abs=tol)
    assert np.dot(np.subtract(witness.second_point, witness.first_point), witness.normal) == (
        pytest.approx(gap, abs=tol)
    )
    assert ball_weights == ((11, 1.0),)
    positions = dict(zip(terrain.vertex_ids, terrain.heightfield.world_vertices(), strict=True))
    assert {i for i, _ in ground_weights} <= set(chart.vertex_ids)
    tri = np.asarray([positions[i] for i in chart.vertex_ids])
    assert np.all(np.asarray(chart.barycentric) >= 0)
    assert sum(chart.barycentric) == pytest.approx(1.0)
    np.testing.assert_allclose(chart.barycentric @ tri, ground_point, atol=tol)
    np.testing.assert_allclose(
        sum(positions[i] * w for i, w in ground_weights), ground_point, atol=tol
    )


@pytest.mark.parametrize("case_id", FROZEN)
@pytest.mark.parametrize("reverse", [False, True])
def test_frozen_ridge_samples(case_id, reverse):
    case = next(c for c in CATALOG["cases"] if c["id"] == case_id)
    inputs = case["inputs"]
    field, sphere = (CATALOG["assets"][key] for key in inputs["assets"])
    grid = ph.Heightfield(field["heights_yx_m"], (1, 1), origin=(-1, -1, 0))
    start, end = np.asarray(inputs["b_path_m"])
    for t, expected in zip(inputs["sample_fractions"], case["expected"]["samples"], strict=True):
        center = start + t * (end - start)
        model, paths, terrain = scene(grid, center, sphere["radius_m"], reverse=reverse)
        row = detect(model, paths)
        if expected["contact"]:
            assert row.status == "overlap"
            check_witness(
                row,
                terrain,
                center,
                sphere["radius_m"],
                expected["a_point_world_m"],
                expected["normal_a_to_b"],
                expected["gap_m"],
            )
        else:
            assert row.status == "separated"


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("face", [0, 1])
def test_rotated_non_square_grid_material_addresses(face, reverse):
    # This is a folded quadrilateral. Its two true triangles have different normals.
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    origin = np.array([2, 3, 4])
    grid = ph.Heightfield([[0, 0, 0], [0, 0.5, 0]], (0.5, 0.75), origin=origin, rotation=rotation)
    if face == 0:
        foot, normal = (0.875, 0.1875, 0), (0, 0, 1)
        bary, indices = (0.25, 0.5, 0.25), (1, 2, 5)
    else:
        foot, normal = (0.625, 0.5625, 0.25), np.array([3, -2, 3]) / np.sqrt(22)
        bary, indices = (0.25, 0.25, 0.5), (1, 5, 4)
    point = origin + rotation @ foot
    direction = rotation @ normal
    center = point + direction / 32
    model, paths, terrain = scene(grid, center, 1 / 16, reverse=reverse)
    row = detect(model, paths)
    assert row.status == "overlap"
    check_witness(row, terrain, center, 1 / 16, point, direction, -1 / 32)
    assert row.heightfield.cell == (0, 1) and row.heightfield.triangle == face
    assert row.heightfield.vertex_ids == tuple(terrain.vertex_ids[i] for i in indices)
    np.testing.assert_allclose(row.heightfield.barycentric, bary, atol=1e-12)


@pytest.mark.parametrize(
    "center,radius,foot,normal,gap,status",
    [
        ((0.25, 0.5, 0.125), 0.125, (0.25, 0.5, 0), (0, 0, 1), 0, "touching"),
        ((0.25, 0.5, -0.375), 0.125, (0.25, 0.5, 0), (0, 0, -1), 0.25, "separated"),
        ((1.09375, 1.125, 0), 0.1875, (1, 1, 0), (0.6, 0.8, 0), -0.03125, "overlap"),
        ((1.09375, 0.5, -0.125), 0.125, (1, 0.5, 0), (0.6, 0, -0.8), 0.03125, "separated"),
        ((0.5, 0.5, 0), 0.125, (0.5, 0.5, 0), (0, 0, 1), -0.125, "overlap"),
    ],
)
def test_finite_edges_corners_and_both_sides(center, radius, foot, normal, gap, status):
    grid = ph.Heightfield([[0, 0], [0, 0]], (1, 1))
    model, paths, terrain = scene(grid, center, radius)
    # An extra broad-phase margin lets the narrow phase publish positive gaps.
    row = detect(model, paths, margin=1)
    assert row.status == status
    check_witness(row, terrain, np.asarray(center), radius, foot, normal, gap)


def test_swept_candidate_filter_capacity_and_stale_pose():
    grid = ph.Heightfield([[0, 1, 0]] * 3, (1, 1), origin=(-1, -1, 0))
    start, end = (-0.75, 0.125, 1.25), (0.75, 0.125, 1.25)
    moving_land = ph.CollisionMotion(grid.world_vertices(), grid.world_vertices() + [0, 0, 0.125])
    model, paths, terrain = scene(grid, start, 5 / 16, end=end, terrain_motion=moving_land)
    with ph.CollisionWorkspace(model) as work:
        result = ph.detect(model, paths, ph.CollisionPolicy(), work)
        (row,) = result.pairs
        assert (row.status, row.reason) == ("unsupported", "heightfield_swept_pair")
        assert row.witness is None and row.sweep is None and row.heightfield is None
        full = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
        assert full.status == "capacity_exceeded" and work.last_result is result
        paths[terrain.key] = ph.CollisionMotion(grid.world_vertices() + [0, 0, 10])
        with pytest.raises(ValueError, match="heightfield"):
            ph.detect(model, paths, ph.CollisionPolicy(), work)
        assert work.last_result is result
    model, paths, _ = scene(grid, start, 5 / 16, end=end, exclude=True)
    row = detect(model, paths)
    assert (row.status, row.reason) == ("filtered", "excluded")
    model, paths, _ = scene(grid, [0, 0, -10], 0.125)
    assert detect(model, paths).reason == "aabb"


def test_invalid_sizes_pose_and_resource_limit():
    for heights, spacing in [
        ([[0, 1]], (1, 1)),
        ([[0, 1], [1, np.nan]], (1, 1)),
        ([[0, 0], [0, 0]], (0, 1)),
        (np.zeros((2, 4097)), (1, 1)),
    ]:
        with pytest.raises(ValueError):
            ph.Heightfield(heights, spacing)
    with pytest.raises(ValueError):
        ph.Heightfield([[0, 0], [0, 0]], (1, 1), rotation=np.diag([1, 1, -1]))
    grid = ph.Heightfield([[0, 0], [0, 0]], (1, 1))
    key = ph.CollisionKey(0, 1, 0)
    with pytest.raises(ValueError):
        ph.CollisionShape(key, "heightfield", (1, 2, 3), heightfield=grid)
    with pytest.raises(ValueError):
        ph.CollisionShape(key, "sphere", (1,), 0.125, heightfield=grid)
