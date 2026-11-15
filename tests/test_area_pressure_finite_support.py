"""Independent V05 finite-layer oracles; prescribed pose, never free dynamics.

These checks use rectangle, affine-load, triangle and rotated-square formulas.
They do not reproduce the product's polygon clipping or triangulation algorithm.
"""

import builtins
import json
from dataclasses import FrozenInstanceError
from math import cos, pi, sin, sqrt

import numpy as np
import pytest

from experiments.area_pressure.finite_support_reference import build_finite_patch
from experiments.area_pressure.friction_reference import integrate_friction
from experiments.area_pressure.reference import SurfaceQuadrature, integrate_pressure


STIFFNESS = 1e7
DEPTH = .002
P0 = .5 * 9.81 / .04**2
DELTA = P0 / STIFFNESS
LEVELS = (1, 4, 8)
RTOL = 1e-8
ATOL = 1e-10


def _rectangle(cx=0., cy=0., width=.04, height=.04):
    return ((cx - width / 2, cy - height / 2), (cx + width / 2, cy - height / 2),
            (cx + width / 2, cy + height / 2), (cx - width / 2, cy + height / 2))


def _regular_polygon(count, radius):
    return tuple((radius * cos(2 * pi * i / count), radius * sin(2 * pi * i / count))
                 for i in range(count))


def _friction(patch, velocity=(0., 0., 0.)):
    count = len(patch.pressure_pa)
    return integrate_friction(
        patch.quadrature, patch.pressure_pa,
        velocity_a_world=np.tile(velocity, (count, 1)), velocity_b_world=np.zeros((count, 3)),
        mu=.5, law="coulomb", slip_speed_m_s=.001,
    )


def _assert_patch_domain(patch):
    assert patch.qualification == "NUMPY_FINITE_AFFINE_LAYER_REFERENCE"
    assert isinstance(patch.quadrature, SurfaceQuadrature)
    assert 0 <= patch.active_area_m2 <= patch.overlap_area_m2 + ATOL
    assert len(patch.overlap_polygon_xy) <= 40 and len(patch.active_polygon_xy) <= 40
    quad = patch.quadrature
    assert len(quad.triangles) <= 8192 and len(quad.vertices_world) <= 8193
    assert len(quad.points_world) <= 24576
    assert len(quad.points_world) == len(patch.pressure_pa) == 3 * len(quad.triangles)
    points = np.asarray(quad.points_world).reshape(-1, 3)
    vertices = np.asarray(quad.vertices_world).reshape(-1, 3)
    normals = np.asarray(quad.normals_world).reshape(-1, 3)
    weights = np.asarray(quad.area_weights_m2)
    assert np.all(weights > 0)
    assert float(weights.sum()) == pytest.approx(patch.active_area_m2, rel=RTOL, abs=ATOL)
    np.testing.assert_array_equal(points[:, 2], np.zeros(len(points)))
    np.testing.assert_allclose(normals, np.tile((0., 0., 1.), (len(points), 1)), rtol=0, atol=1e-14)
    for samples in (points, vertices):
        for polygon in (patch.footprint_xy, patch.support_xy):
            polygon = np.asarray(polygon)
            edge = np.roll(polygon, -1, axis=0) - polygon
            delta = samples[:, None, :2] - polygon[None, :, :]
            # Signed distance to each input edge, not an axis-aligned membership test.
            distance = (edge[None, :, 0] * delta[:, :, 1]
                        - edge[None, :, 1] * delta[:, :, 0]) / np.linalg.norm(edge, axis=1)
            assert np.all(distance >= -1e-12)
        gap = samples[:, :2] @ np.asarray(patch.gap_coefficients[:2]) + patch.gap_coefficients[2]
        assert np.all(gap <= 1e-12)
        assert np.all(gap >= -patch.layer_depth_m - 1e-12)
    gap = points[:, :2] @ np.asarray(patch.gap_coefficients[:2]) + patch.gap_coefficients[2]
    expected_pressure = patch.stiffness_pa_per_m * np.maximum(-gap, 0.)
    np.testing.assert_allclose(patch.pressure_pa, expected_pressure, rtol=RTOL, atol=ATOL)
    if patch.active_area_m2 == 0:
        assert quad.points_world == () and patch.pressure_pa == ()
    json.dumps(patch.to_dict(), allow_nan=False)


def _assert_loads(result, normal_force, normal_moment, friction_force=(0., 0., 0.),
                  friction_moment=(0., 0., 0.), cop=None):
    normal_force, normal_moment = np.asarray(normal_force), np.asarray(normal_moment)
    friction_force, friction_moment = np.asarray(friction_force), np.asarray(friction_moment)
    for actual, expected in (
        (result.normal.first_force_n, normal_force),
        (result.normal.first_moment_nm, normal_moment),
        (result.first_friction_force_n, friction_force),
        (result.first_friction_moment_nm, friction_moment),
        (result.first_total_force_n, normal_force + friction_force),
        (result.first_total_moment_nm, normal_moment + friction_moment),
    ):
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    if cop is None:
        assert result.normal.cop_world is None
        assert result.normal.pressure_centroid_world is None
    else:
        np.testing.assert_allclose(result.normal.cop_world, cop, rtol=0, atol=1e-9)
        np.testing.assert_allclose(result.normal.pressure_centroid_world, cop, rtol=0, atol=1e-9)
    for owner, suffixes in (
        (result.normal, ("force_per_point_n", "force_n", "moment_nm")),
        (result, ("friction_force_per_point_n", "friction_force_n", "friction_moment_nm",
                  "total_force_n", "total_moment_nm")),
    ):
        for suffix in suffixes:
            first = np.asarray(getattr(owner, "first_" + suffix))
            second = np.asarray(getattr(owner, "second_" + suffix))
            np.testing.assert_allclose(first + second, np.zeros_like(first), rtol=0, atol=1e-12)
    assert result.friction_power_w <= 1e-12


def test_all_93_prescribed_displacement_cells_and_com_friction_height_arm():
    support = _rectangle(width=.08, height=.08)
    for j in range(31):
        x = .002 * j
        # The integer index keeps the declared terminal width exactly zero.
        width = .04 if j <= 10 else .002 * (30 - j)
        area, force = .04 * width, P0 * .04 * width
        cop_x = x if x <= .02 else (x + .02) / 2
        cop = (cop_x, 0., 0.) if area > 0 else None
        velocity = (0., 0., 0.) if j == 0 else (.02, 0., 0.)
        fx = 0. if j == 0 else -.5 * force
        com = np.array((x, 0., .01 - DELTA))
        results = []
        for subdivisions in LEVELS:
            patch = build_finite_patch(_rectangle(cx=x), support,
                                       gap_coefficients=(0., 0., -DELTA),
                                       subdivisions=subdivisions)
            _assert_patch_domain(patch)
            assert patch.overlap_area_m2 == pytest.approx(area, rel=RTOL, abs=ATOL)
            assert patch.active_area_m2 == pytest.approx(area, rel=RTOL, abs=ATOL)
            result = _friction(patch, velocity)
            results.append(result)
            _assert_loads(result, (0., 0., force), (0., -force * cop_x, 0.),
                          friction_force=(fx, 0., 0.), cop=cop)
            normal_com = np.asarray(result.normal.first_moment_nm) - np.cross(
                com, result.normal.first_force_n
            )
            friction_com = np.asarray(result.first_friction_moment_nm) - np.cross(
                com, result.first_friction_force_n
            )
            total_com = np.asarray(result.first_total_moment_nm) - np.cross(
                com, result.first_total_force_n
            )
            expected_normal_com = (0., force * (x - cop_x), 0.)
            expected_friction_com = (0., -com[2] * fx, 0.)
            np.testing.assert_allclose(normal_com, expected_normal_com, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(friction_com, expected_friction_com, rtol=RTOL, atol=ATOL)
            expected_total = np.cross((cop_x - x, 0., -com[2]), (fx, 0., force))
            np.testing.assert_allclose(total_com, expected_total, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(total_com, normal_com + friction_com, rtol=0, atol=1e-12)
            assert result.friction_power_w == pytest.approx(fx * velocity[0], rel=RTOL, abs=ATOL)
            if j == 0:
                assert force == pytest.approx(.5 * 9.81, rel=RTOL, abs=ATOL)
            elif j < 30:
                assert friction_com[1] > 0  # Kept distinct from the normal pitching moment.
            else:
                assert patch.overlap_area_m2 == patch.active_area_m2 == 0.
                np.testing.assert_array_equal(result.first_total_force_n, (0., 0., 0.))
                np.testing.assert_array_equal(result.first_total_moment_nm, (0., 0., 0.))
            if 10 < j < 30:
                assert result.normal.pressure_load_n < .5 * 9.81  # Never renormalize to mg.
        # Explicit last-two refinement agreement, in addition to each analytic check.
        for field in ("first_total_force_n", "first_total_moment_nm"):
            np.testing.assert_allclose(getattr(results[-1], field), getattr(results[-2], field),
                                       rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", ("outside", "lifted", "tilted"))
def test_all_nine_frozen_holdouts(case):
    support = _rectangle(width=.08, height=.08)
    if case == "outside":
        footprint, gap, overlap, area, force, cop = _rectangle(cx=.061), (0., 0., -DELTA), 0., 0., 0., None
    elif case == "lifted":
        footprint, gap, overlap, area, force, cop = _rectangle(), (0., 0., .0001), .0016, 0., 0., None
    else:
        footprint, gap, overlap, area, force, cop = (
            _rectangle(), (.02, 0., -.0001), .0016, .001, 2.5, (-.035 / 3, 0., 0.)
        )
    for subdivisions in LEVELS:
        patch = build_finite_patch(footprint, support, gap_coefficients=gap, subdivisions=subdivisions)
        _assert_patch_domain(patch)
        assert patch.overlap_area_m2 == pytest.approx(overlap, rel=RTOL, abs=ATOL)
        assert patch.active_area_m2 == pytest.approx(area, rel=RTOL, abs=ATOL)
        result = _friction(patch, (.02, 0., 0.) if case == "outside" else (0., 0., 0.))
        _assert_loads(result, (0., 0., force), (0., 7 / 240 if case == "tilted" else 0., 0.), cop=cop)
        if case != "tilted":
            assert patch.active_area_m2 == 0. and patch.active_polygon_xy == ()
        else:
            x = np.asarray(patch.active_polygon_xy)[:, 0]
            assert x.min() == pytest.approx(-.02, abs=1e-12)
            assert x.max() == pytest.approx(.005, abs=1e-12)


def test_rotated_square_intersection_is_the_analytic_octagon_not_an_aabb():
    radius = .02
    rotation = np.array(((1., -1.), (1., 1.))) / sqrt(2)
    square = np.asarray(_rectangle(width=2 * radius, height=2 * radius))
    footprint = square @ rotation.T
    corner = radius * (sqrt(2) - 1)
    expected_vertices = np.array(((radius, corner), (corner, radius), (-corner, radius),
                                  (-radius, corner), (-radius, -corner), (-corner, -radius),
                                  (corner, -radius), (radius, -corner)))
    expected_area = 8 * (sqrt(2) - 1) * radius**2
    for subdivisions in LEVELS:
        patch = build_finite_patch(footprint, square, gap_coefficients=(0., 0., -.0001),
                                   subdivisions=subdivisions)
        _assert_patch_domain(patch)
        assert len(patch.overlap_polygon_xy) == len(patch.active_polygon_xy) == 8
        actual = np.asarray(patch.active_polygon_xy)
        distances = np.linalg.norm(actual[:, None, :] - expected_vertices[None, :, :], axis=2)
        assert np.max(np.min(distances, axis=0)) <= 1e-12
        assert np.max(np.min(distances, axis=1)) <= 1e-12
        assert patch.active_area_m2 == pytest.approx(expected_area, rel=RTOL, abs=ATOL)
        _assert_loads(_friction(patch), (0., 0., 1000 * expected_area), (0., 0., 0.), cop=(0., 0., 0.))


@pytest.mark.parametrize("polygon,area", (
    (((-.01, -.01), (.02, -.01), (-.01, .02)), .00045),
    (_regular_polygon(6, .01), 3 * sqrt(3) / 2 * .01**2),
), ids=("triangle", "hexagon"))
def test_containment_for_either_input_polygon_has_known_area_centroid_and_load(polygon, area):
    enclosing = _rectangle(width=.08, height=.08)
    for footprint, support in ((polygon, enclosing), (enclosing, polygon)):
        patch = build_finite_patch(footprint, support, gap_coefficients=(0., 0., -.00025), subdivisions=4)
        _assert_patch_domain(patch)
        assert patch.active_area_m2 == pytest.approx(area, rel=RTOL, abs=ATOL)
        _assert_loads(_friction(patch), (0., 0., 2500 * area), (0., 0., 0.), cop=(0., 0., 0.))


def test_yaw_translation_and_gap_constant_transform_together_with_world_and_com_moments():
    footprint = np.asarray(_rectangle())
    support = np.asarray(_rectangle(width=.08, height=.08))
    slope = np.array((.02, 0.))
    constant = -.0001
    rotation = np.array(((.6, -.8), (.8, .6)))
    rotation3 = np.array(((.6, -.8, 0.), (.8, .6, 0.), (0., 0., 1.)))
    translation = np.array((.13, -.07))
    transformed_slope = rotation @ slope
    transformed_constant = constant - transformed_slope @ translation
    expected_cop = rotation @ np.array((-.035 / 3, 0.)) + translation
    patch = build_finite_patch(footprint @ rotation.T + translation,
                               support @ rotation.T + translation,
                               gap_coefficients=(*transformed_slope, transformed_constant),
                               subdivisions=4)
    _assert_patch_domain(patch)
    assert patch.active_area_m2 == pytest.approx(.001, rel=RTOL, abs=ATOL)
    velocity = rotation3 @ np.array((.02, 0., 0.))
    normal_force = np.array((0., 0., 2.5))
    friction_force = rotation3 @ np.array((-1.25, 0., 0.))
    cop = np.array((*expected_cop, 0.))
    result = _friction(patch, velocity)
    _assert_loads(result, normal_force, np.cross(cop, normal_force), friction_force,
                  np.cross(cop, friction_force), cop=cop)
    com = np.array((*translation, .01))
    total_com = np.asarray(result.first_total_moment_nm) - np.cross(com, result.first_total_force_n)
    np.testing.assert_allclose(total_com, np.cross(cop - com, normal_force + friction_force),
                               rtol=RTOL, atol=ATOL)
    assert result.friction_power_w == pytest.approx(-1.25 * .02, rel=RTOL, abs=ATOL)
    # Circular shifts change storage only, even when the first vertex changes.
    for shift in (1, 2, 3):
        changed = build_finite_patch(np.roll(footprint, shift, axis=0) @ rotation.T + translation,
                                     np.roll(support, -shift, axis=0) @ rotation.T + translation,
                                     gap_coefficients=(*transformed_slope, transformed_constant),
                                     subdivisions=4)
        normal = integrate_pressure(changed.quadrature, changed.pressure_pa)
        assert changed.active_area_m2 == pytest.approx(patch.active_area_m2, rel=RTOL, abs=ATOL)
        np.testing.assert_allclose(normal.cop_world, cop, rtol=0, atol=1e-9)
        np.testing.assert_allclose(normal.first_moment_nm, result.normal.first_moment_nm,
                                   rtol=RTOL, atol=ATOL)


def test_exact_edge_vertex_separation_and_zero_gap_have_no_active_load():
    support = _rectangle()
    for footprint, gap, overlap in (
        (_rectangle(cx=.04), (0., 0., -DELTA), 0.),
        (_rectangle(cx=.04, cy=.04), (0., 0., -DELTA), 0.),
        (_rectangle(cx=.040000001), (0., 0., -DELTA), 0.),
        (_rectangle(), (0., 0., 0.), .0016),
    ):
        patch = build_finite_patch(footprint, support, gap_coefficients=gap, subdivisions=8)
        _assert_patch_domain(patch)
        assert patch.overlap_area_m2 == pytest.approx(overlap, rel=RTOL, abs=ATOL)
        assert patch.active_area_m2 == 0. and patch.active_polygon_xy == ()
        result = _friction(patch, (.02, 0., 0.))
        _assert_loads(result, (0., 0., 0.), (0., 0., 0.))
        np.testing.assert_array_equal(result.first_total_force_n, (0., 0., 0.))
        np.testing.assert_array_equal(result.first_total_moment_nm, (0., 0., 0.))


def test_layer_depth_checks_overlap_vertices_including_before_quadrature_sampling():
    square = _rectangle()
    boundary = build_finite_patch(square, square, gap_coefficients=(0., 0., -DEPTH))
    assert integrate_pressure(boundary.quadrature, boundary.pressure_pa).pressure_load_n == pytest.approx(32.)
    for gap in ((0., 0., -DEPTH - 1e-9), (.1, 0., -.00011)):
        # The tilted corner is deeper than H although coarse interior samples can be shallower.
        with pytest.raises(ValueError):
            build_finite_patch(square, square, gap_coefficients=gap, subdivisions=1)
    # The left part of this footprint is too deep but lies outside the finite support.
    patch = build_finite_patch(_rectangle(width=.2), _rectangle(cx=.055, width=.01, height=.02),
                               gap_coefficients=(.1, 0., -.006), subdivisions=4)
    _assert_patch_domain(patch)
    assert patch.active_area_m2 == pytest.approx(.0002, rel=RTOL, abs=ATOL)
    _assert_loads(_friction(patch), (0., 0., 1.), (0., -(.05 + .01 / 3), 0.),
                  cop=(.05 + .01 / 3, 0., 0.))
    empty = build_finite_patch(_rectangle(cx=.1), square, gap_coefficients=(0., 0., -.003))
    assert empty.overlap_area_m2 == empty.active_area_m2 == 0.
    assert empty.pressure_pa == ()


def test_invalid_polygon_shape_orientation_convexity_and_dtype_are_rejected():
    square = _rectangle()
    pentagon = _regular_polygon(5, .02)
    star = tuple(pentagon[i] for i in (0, 2, 4, 1, 3))
    invalid = (
        (), square[:2], tuple(reversed(square)), (*square, square[0]),
        ((0., 0.), (.02, 0.), (.01, .01), (.02, .02), (0., .02)),
        ((0., 0.), (.01, 0.), (.02, 0.), (.02, .02), (0., .02)),
        (square[0], square[2], square[1], square[3]), star,
        ((np.nan, 0.), (.02, 0.), (0., .02)),
        ((np.inf, 0.), (.02, 0.), (0., .02)),
        _rectangle(cx=1.01), tuple((*point, 0.) for point in square),
        np.asarray(square, dtype=object), np.asarray(square, dtype=complex),
        np.asarray(square, dtype=str), np.asarray(((False, False), (True, False), (False, True))),
        _regular_polygon(17, .02),
    )
    for polygon in invalid:
        before = repr(polygon)
        for footprint, support in ((polygon, square), (square, polygon)):
            with pytest.raises(ValueError):
                build_finite_patch(footprint, support, gap_coefficients=(0., 0., -DELTA))
        assert repr(polygon) == before


def test_invalid_material_gap_subdivision_and_finite_arithmetic_are_rejected():
    square = _rectangle()
    for options in (
        {"gap_coefficients": (0., 0.)}, {"gap_coefficients": (0., 0., np.nan)},
        {"gap_coefficients": (True, 0., -DELTA)}, {"gap_coefficients": (.1, .1, -DELTA)},
        {"stiffness_pa_per_m": 0.}, {"stiffness_pa_per_m": -1.},
        {"stiffness_pa_per_m": np.inf}, {"stiffness_pa_per_m": True},
        {"layer_depth_m": 0.}, {"layer_depth_m": -1.},
        {"layer_depth_m": np.nan}, {"layer_depth_m": False},
        {"subdivisions": 0}, {"subdivisions": 9}, {"subdivisions": True},
        {"subdivisions": 1.0}, {"subdivisions": 10**100},
        {"gap_coefficients": (0., 0., -2.), "layer_depth_m": 3., "stiffness_pa_per_m": 1e308},
    ):
        with pytest.raises(ValueError):
            build_finite_patch(square, square, **{"gap_coefficients": (0., 0., -DELTA), **options})


def test_maximum_input_polygon_and_subdivision_are_bounded_without_truncation():
    footprint = _regular_polygon(16, .02)
    angle = pi / 16
    rotation = np.array(((cos(angle), -sin(angle)), (sin(angle), cos(angle))))
    support = np.asarray(footprint) @ rotation.T
    patch = build_finite_patch(footprint, support, gap_coefficients=(0., 0., -DELTA), subdivisions=8)
    _assert_patch_domain(patch)
    assert len(patch.overlap_polygon_xy) == len(patch.active_polygon_xy) == 32
    assert len(patch.footprint_xy) == len(patch.support_xy) == 16
    assert patch.subdivisions == 8
    assert patch.active_area_m2 > 0 and len(patch.quadrature.points_world) > 0


def test_owned_frozen_json_and_readonly_inputs_without_engine_calls(monkeypatch):
    footprint = np.asarray(_rectangle())
    support = np.asarray(_rectangle(width=.08, height=.08))
    gap = np.array((.02, 0., -.0001))
    arrays = (footprint, support, gap)
    snapshots = tuple(array.tobytes() for array in arrays)
    for array in arrays:
        array.setflags(write=False)
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in ("phymeo", "newton", "warp", "pydrake", "tyx", "mujoco"):
            raise AssertionError("finite prescribed-pose reference must not invoke a physics engine")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(builtins, "__import__", guard)
        patch = build_finite_patch(footprint, support, gap_coefficients=gap, subdivisions=4)
    assert tuple(array.tobytes() for array in arrays) == snapshots
    for array in arrays:
        array.setflags(write=True)
        array[:] = 99.
    assert patch.footprint_xy == _rectangle()
    assert patch.support_xy == _rectangle(width=.08, height=.08)
    assert patch.gap_coefficients == (.02, 0., -.0001)
    with pytest.raises(FrozenInstanceError):
        patch.active_area_m2 = 1.
    with pytest.raises(TypeError):
        patch.active_polygon_xy[0][0] = 0.
    with pytest.raises(TypeError):
        patch.quadrature.points_world[0][0] = 0.
    payload = patch.to_dict()
    assert set(payload) == {
        "qualification", "footprint_xy", "support_xy", "gap_coefficients", "stiffness_pa_per_m",
        "layer_depth_m", "subdivisions", "overlap_polygon_xy", "active_polygon_xy",
        "overlap_area_m2", "active_area_m2", "quadrature", "pressure_pa",
    }
    decoded = json.loads(json.dumps(payload, allow_nan=False))
    assert decoded["qualification"] == "NUMPY_FINITE_AFFINE_LAYER_REFERENCE"
    assert decoded["gap_coefficients"] == [.02, 0., -.0001]
    payload["active_area_m2"] = 99.
    payload["quadrature"]["points_world"] = ()
    assert patch.active_area_m2 == pytest.approx(.001, rel=RTOL, abs=ATOL)
    assert len(patch.quadrature.points_world) > 0
