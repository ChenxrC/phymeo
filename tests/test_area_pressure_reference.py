"""Independent analytic V00/V01/V02 oracles for the standalone static experiment."""

import builtins
import json
from dataclasses import FrozenInstanceError, replace
from math import pi

import numpy as np
import pytest

from experiments.area_pressure.reference import (
    MAX_TRIANGLES,
    integrate_pressure,
    rectangle_case,
    sphere_plane_case,
    triangle_quadrature,
)


def test_triangle_rule_integrates_degree_two_and_linear_pressure_moment():
    quad = triangle_quadrature(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), ((0, 1, 2),))
    points, weights = np.asarray(quad.points_world), np.asarray(quad.area_weights_m2)
    assert len(points) == 3 and sum(weights) == pytest.approx(.5, abs=1e-15)
    # Integrals over x>=0, y>=0, x+y<=1: x^2=1/12, xy=1/24.
    assert weights @ points[:, 0] ** 2 == pytest.approx(1 / 12, abs=1e-15)
    assert weights @ (points[:, 0] * points[:, 1]) == pytest.approx(1 / 24, abs=1e-15)
    result = integrate_pressure(quad, points[:, 0])
    np.testing.assert_allclose(result.first_force_n, (0., 0., 1 / 6), rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.first_moment_nm, (1 / 24, -1 / 12, 0.), rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.cop_world, (.5, .25, 0.), rtol=0, atol=1e-15)


@pytest.mark.parametrize("width,height,area,force", (
    (.02, .02, .0004, .8), (.04, .02, .0008, 1.6), (.04, .04, .0016, 3.2),
))
def test_v00_area_scaling_and_one_four_sixteen_cell_invariance(width, height, area, force):
    for count in (1, 2, 4):
        result = rectangle_case(width, height, subdivisions=(count, count))
        assert len(result.quadrature.triangles) == 2 * count * count
        assert len(result.pressure_pa) == 6 * count * count
        assert result.area_m2 == pytest.approx(area, rel=1e-12)
        assert result.pressure_load_n == pytest.approx(force, rel=1e-12)
        np.testing.assert_allclose(result.first_force_n, (0., 0., force), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(result.first_moment_nm, (0., 0., 0.), rtol=0, atol=1e-15)
        np.testing.assert_allclose(result.cop_world, (0., 0., 0.), rtol=0, atol=1e-15)
        assert all(value == 2000. for value in result.pressure_pa)
        np.testing.assert_array_equal(np.asarray(result.first_force_per_point_n) + result.second_force_per_point_n, 0.)
        np.testing.assert_array_equal(np.asarray(result.first_force_n) + result.second_force_n, 0.)
        np.testing.assert_array_equal(np.asarray(result.first_moment_nm) + result.second_moment_nm, 0.)


def test_v01_linear_pressure_has_analytic_cop_and_moment_for_every_refinement():
    for count in (1, 2, 4):
        result = rectangle_case(subdivisions=(count, count), pressure_gradient_pa_per_m=(1e5, 0.))
        assert result.area_m2 == pytest.approx(.0004, rel=1e-12)
        np.testing.assert_allclose(result.first_force_n, (0., 0., .8), rtol=0, atol=1e-12)
        np.testing.assert_allclose(result.cop_world, (1 / 600, 0., 0.), rtol=0, atol=1e-12)
        np.testing.assert_allclose(result.first_moment_nm, (0., -1 / 750, 0.), rtol=0, atol=1e-12)
        assert result.pressure_centroid_world == result.cop_world
        for point, pressure in zip(result.quadrature.points_world, result.pressure_pa, strict=True):
            assert pressure == pytest.approx(2000. + 1e5 * point[0], abs=1e-10)


def test_rigid_transform_common_origin_and_bilateral_loads_are_covariant():
    rotation = ((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))
    result = rectangle_case(pressure_gradient_pa_per_m=(1e5, 0.), position_world=(1., -2., 3.),
                            rotation_world=rotation)
    np.testing.assert_allclose(result.first_force_n, (.8, 0., 0.), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.cop_world, (1., -2 + 1 / 600, 3.), rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.first_moment_nm, (0., 2.4, 1.6 - 1 / 750), rtol=0, atol=1e-12)
    shifted = rectangle_case(pressure_gradient_pa_per_m=(1e5, 0.), position_world=(1., -2., 3.),
                             rotation_world=rotation, origin_world=(1., -2., 3.))
    np.testing.assert_allclose(shifted.first_moment_nm, (0., 0., -1 / 750), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(result.first_moment_nm) + result.second_moment_nm, 0.)
    # Reverse winding exchanges the traction direction at the same geometric domain.
    quad = triangle_quadrature(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), ((0, 2, 1),))
    reverse = integrate_pressure(quad, (2., 2., 2.), origin_world=(.1, .2, .3))
    np.testing.assert_allclose(reverse.first_force_n, (0., 0., -1.), rtol=0, atol=1e-15)
    np.testing.assert_allclose(reverse.second_force_n, (0., 0., 1.), rtol=0, atol=1e-15)


@pytest.mark.parametrize("depth", (.0001, .0002, .0004))
def test_v02_disk_matches_independent_radial_field_area_and_integral(depth):
    radius, modulus = .02, 1e5
    # These closed-form oracles are independent of tessellation and its quadrature.
    expected_area = pi * (2 * radius * depth - depth * depth)
    expected_force = pi * modulus * depth * depth * (1 - 2 * depth / (3 * radius))
    results = []
    for resolution in (.002, .001, .0005):
        result = sphere_plane_case(radius, depth, modulus, resolution)
        results.append(result)
        assert result.qualification == "NUMPY_STATIC_REFERENCE"
        assert 0 < result.area_m2 <= expected_area * (1 + 1e-12)
        assert 0 < result.pressure_load_n and result.cop_world is not None
        np.testing.assert_allclose(result.first_force_n[:2], (0., 0.), rtol=0, atol=1e-15)
        np.testing.assert_allclose(result.cop_world, (0., 0., 0.), rtol=0, atol=.0002)
        points = np.asarray(result.quadrature.points_world)
        expected_pressure = modulus * (1 - np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + (radius - depth) ** 2) / radius)
        np.testing.assert_allclose(result.pressure_pa, expected_pressure, rtol=1e-10, atol=1e-9)
        assert all(0 <= value <= modulus * depth / radius * (1 + 1e-12) for value in result.pressure_pa)
        vertices = np.asarray(result.quadrature.vertices_world)
        for triangle in result.quadrature.triangles:
            a, b, c = vertices[list(triangle)]
            assert max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)) <= resolution * (1 + 1e-12)
        json.dumps(result.to_dict(), allow_nan=False)
    previous, finest = results[-2:]
    assert abs(finest.area_m2 / expected_area - 1) <= .05
    assert abs(finest.first_force_n[2] / expected_force - 1) <= .05
    assert abs(finest.area_m2 - previous.area_m2) / finest.area_m2 <= .05
    assert abs(finest.first_force_n[2] - previous.first_force_n[2]) / finest.first_force_n[2] <= .05


def test_zero_disk_and_zero_pressure_have_no_load_and_no_defined_cop():
    for resolution in (.002, .001, .0005):
        result = sphere_plane_case(indentation_m=0., resolution_m=resolution)
        assert result.area_m2 == result.pressure_load_n == 0.
        assert result.quadrature.points_world == result.pressure_pa == ()
        assert result.first_force_n == result.second_force_n == (0., 0., 0.)
        assert result.first_moment_nm == result.second_moment_nm == (0., 0., 0.)
        assert result.cop_world is result.pressure_centroid_world is None
        json.dumps(result.to_dict(), allow_nan=False)
    no_pressure = rectangle_case(pressure_pa=0.)
    assert no_pressure.area_m2 > 0 and no_pressure.pressure_load_n == 0.
    assert no_pressure.cop_world is None
    assert integrate_pressure(triangle_quadrature((), ()), ()).area_m2 == 0.


def test_nonplanar_pressure_centroid_is_not_presented_as_a_cop():
    quad = triangle_quadrature(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                               ((0, 1, 2), (0, 2, 3)))
    result = integrate_pressure(quad, (1.,) * 6)
    np.testing.assert_allclose(result.first_force_n, (.5, 0., .5), rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.pressure_centroid_world, (1 / 6, 1 / 3, 1 / 6), rtol=0, atol=1e-15)
    assert result.cop_world is None


def test_negative_area_pressure_invalid_mesh_and_nonfinite_data_reject():
    vertices = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    quad = triangle_quadrature(vertices, ((0, 1, 2),))
    for weights in ((-.1, .1, .1), (0., .1, .1), (np.nan, .1, .1), (.1, .1)):
        with pytest.raises(ValueError):
            integrate_pressure(replace(quad, area_weights_m2=weights), (1., 1., 1.))
    for pressure in ((-1., 1., 1.), (np.inf, 1., 1.), (np.nan, 1., 1.), (1.,)):
        with pytest.raises(ValueError):
            integrate_pressure(quad, pressure)
    for bad in (((0, 1, 3),), ((0, 1, 1),), ((0., 1., 2.),), ((True, 1, 2),)):
        with pytest.raises(ValueError):
            triangle_quadrature(vertices, bad)
    with pytest.raises(ValueError):
        triangle_quadrature(((0., 0., 0.), (1., 0., 0.), (2., 0., 0.)), ((0, 1, 2),))
    with pytest.raises(ValueError):
        triangle_quadrature(((0., 0., 0.), (1., np.nan, 0.), (0., 1., 0.)), ((0, 1, 2),))
    with pytest.raises(ValueError):
        integrate_pressure(quad, (1., 1., 1.), origin_world=(0., np.inf, 0.))
    with pytest.raises(ValueError):
        replace(rectangle_case(), area_m2=float("inf")).to_dict()


def test_case_parameter_validation_and_capacity_reject_before_large_allocation():
    for options in ({"width_m": 0.}, {"height_m": -1.}, {"width_m": np.inf},
                    {"pressure_pa": -1.}, {"pressure_gradient_pa_per_m": (1e9, 0.)},
                    {"subdivisions": (0, 1)}, {"subdivisions": (True, 1)},
                    {"subdivisions": (10000, 10000)}, {"position_world": (0., 0.)},
                    {"rotation_world": np.diag((1., 1., -1.))}):
        with pytest.raises(ValueError):
            rectangle_case(**options)
    for options in ({"radius_m": 0.}, {"radius_m": np.nan}, {"indentation_m": -.001},
                    {"indentation_m": .021}, {"modulus_pa": 0.}, {"modulus_pa": np.inf},
                    {"resolution_m": 0.}, {"resolution_m": 1e-12}, {"resolution_m": True},
                    {"plane_position_world": (0., 0., np.nan)}):
        with pytest.raises(ValueError):
            sphere_plane_case(**options)
    with pytest.raises(ValueError):
        triangle_quadrature(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)),
                            np.zeros((MAX_TRIANGLES + 1, 3), dtype=np.int64))


def test_frozen_owned_serializable_results_do_not_call_phymeo_or_native(monkeypatch):
    vertices = np.array(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    triangles = np.array(((0, 1, 2),), dtype=np.int64)
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in ("phymeo", "mujoco", "tyx"):
            raise AssertionError("the experiment must be independent of production physics")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guard)
        quad = triangle_quadrature(vertices, triangles)
        result = integrate_pressure(quad, (2., 2., 2.))
        sphere_plane_case()
    vertices[:] = 99
    triangles[:] = 0
    assert quad.vertices_world[0] == (0., 0., 0.) and quad.triangles == ((0, 1, 2),)
    assert result.first_force_n == pytest.approx((0., 0., 1.))
    with pytest.raises(FrozenInstanceError):
        result.area_m2 = 1.
    with pytest.raises(TypeError):
        result.quadrature.points_world[0][0] = 1.
    payload = result.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False))["area_m2"] == .5
    payload["area_m2"] = 9.
    assert result.area_m2 == .5
