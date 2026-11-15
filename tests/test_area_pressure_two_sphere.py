"""Independent pressure-parameter oracles for the V03 static curved interface."""

import builtins
import json
from dataclasses import FrozenInstanceError, replace
from math import pi

import numpy as np
import pytest

from experiments.area_pressure.two_sphere_reference import two_sphere_case


def _pressure_parameter_oracle(ra, rb, ea, eb, overlap):
    """No product root or mesh: common pressure parameterizes both sphere radii."""
    distance = ra + rb - overlap
    axis_pressure = overlap / (ra / ea + rb / eb)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    pressure = (nodes + 1) * axis_pressure / 2
    weights = weights * axis_pressure / 2
    a = ra * (1 - pressure / ea)
    b = rb * (1 - pressure / eb)
    da, db = -ra / ea, -rb / eb
    z = (distance**2 + a**2 - b**2) / (2 * distance)
    dz = (a * da - b * db) / distance
    rho_squared = a**2 - z**2
    derivative = 2 * a * da - 2 * z * dz
    # Integration by parts: 2*pi*integral(p*rho*d_rho) = pi*integral(rho^2*d_p).
    force = pi * (weights @ rho_squared)
    area = pi * (weights @ np.sqrt(derivative**2 + 4 * rho_squared * dz**2))
    edge_z = (distance**2 + ra**2 - rb**2) / (2 * distance)
    projected = pi * (ra**2 - edge_z**2)
    return force, area, projected


@pytest.mark.parametrize("ea,eb,axis_z", ((1e5, 3e5, .0197), (3e5, 1e5, .0199)))
def test_v03_curved_interface_fixed_refinements_and_independent_integral(ea, eb, axis_z):
    force, surface_area, projected_area = _pressure_parameter_oracle(.02, .02, ea, eb, .0004)
    results = []
    for resolution in (.002, .001, .0005):
        result = two_sphere_case(modulus_a_pa=ea, modulus_b_pa=eb, resolution_m=resolution)
        results.append(result)
        np.testing.assert_allclose(result.centers_world, ((0., 0., 0.), (0., 0., .0396)), rtol=0, atol=1e-15)
        assert result.axis_height_m == pytest.approx(axis_z, rel=0, abs=1e-12)
        assert result.axis_pressure_pa == pytest.approx(1500., rel=0, abs=1e-7)
        assert result.circle_height_m == pytest.approx(.0198, rel=0, abs=1e-15)
        assert result.circle_radius_m**2 == pytest.approx(.02**2 - .0198**2, rel=0, abs=1e-18)
        assert result.max_pressure_residual_pa <= 1e-7
        assert max(map(abs, result.pressure_residual_pa)) <= 1e-7
        assert max(map(abs, result.vertex_pressure_residual_pa)) <= 1e-7
        integral = result.integral
        quad = integral.quadrature
        points = np.asarray(quad.points_world)
        normals = np.asarray(quad.normals_world)
        radii_a = np.linalg.norm(points, axis=1)
        radii_b = np.linalg.norm(points - (0., 0., .0396), axis=1)
        # Independent direct field evaluations, not the product's bisection residual.
        pa = ea * (1 - radii_a / .02)
        pb = eb * (1 - radii_b / .02)
        np.testing.assert_allclose(result.pressure_a_pa, pa, rtol=0, atol=1e-7)
        np.testing.assert_allclose(result.pressure_b_pa, pb, rtol=0, atol=1e-7)
        np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-7)
        gradient = -ea / .02 * points / radii_a[:, None]
        gradient += eb / .02 * (points - (0., 0., .0396)) / radii_b[:, None]
        expected_normals = gradient / np.linalg.norm(gradient, axis=1)[:, None]
        np.testing.assert_allclose(normals, expected_normals, rtol=0, atol=1e-12)
        assert np.all(normals[:, 2] < 0)  # integral.first is force ON sphere A.
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(quad.area_weights_m2) * -normals[:, 2],
                                   result.projected_area_weights_m2, rtol=1e-12, atol=1e-20)
        assert integral.area_m2 > result.area_projected_m2 > 0
        assert integral.pressure_load_n > -integral.first_force_n[2] > 0
        assert integral.cop_world is None  # Curved pressure centroid is not a planar COP.
        np.testing.assert_allclose(integral.first_force_n[:2], (0., 0.), rtol=0, atol=1e-14)
        np.testing.assert_allclose(integral.first_moment_nm, (0., 0., 0.), rtol=0, atol=1e-14)
        np.testing.assert_array_equal(np.asarray(integral.first_force_n) + integral.second_force_n, 0.)
        np.testing.assert_array_equal(np.asarray(integral.first_moment_nm) + integral.second_moment_nm, 0.)
        vertices = np.asarray(quad.vertices_world)
        boundary = np.isclose(np.linalg.norm(vertices[:, :2], axis=1), result.circle_radius_m,
                              rtol=1e-12, atol=0)
        np.testing.assert_allclose(vertices[boundary, 2], .0198, rtol=0, atol=1e-15)
        # resolution_m bounds projected mesh edges; curved dA uses the separate Jacobian.
        for triangle in quad.triangles:
            a, b, c = vertices[list(triangle), :2]
            assert max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)) <= resolution * (1 + 1e-12)
        assert np.linalg.norm(integral.pressure_centroid_world[:2]) <= .0002
        json.dumps(result.to_dict(), allow_nan=False)
    previous, finest = results[-2:]
    assert abs(finest.integral.area_m2 / surface_area - 1) <= .05
    assert abs(finest.area_projected_m2 / projected_area - 1) <= .05
    assert abs(-finest.integral.first_force_n[2] / force - 1) <= .05
    assert abs(finest.integral.area_m2 - previous.integral.area_m2) / finest.integral.area_m2 <= .05
    assert abs(finest.area_projected_m2 - previous.area_projected_m2) / finest.area_projected_m2 <= .05
    assert abs(finest.integral.first_force_n[2] - previous.integral.first_force_n[2]) / abs(finest.integral.first_force_n[2]) <= .05


def test_modulus_exchange_reflects_the_surface_without_replacing_it_by_midplane():
    original = two_sphere_case(resolution_m=.001)
    exchanged = two_sphere_case(modulus_a_pa=3e5, modulus_b_pa=1e5, resolution_m=.001)
    first = np.asarray(original.integral.quadrature.points_world)
    second = np.asarray(exchanged.integral.quadrature.points_world)
    np.testing.assert_allclose(first[:, :2], second[:, :2], rtol=0, atol=0)
    np.testing.assert_allclose(first[:, 2] + second[:, 2], .0396, rtol=0, atol=1e-12)
    assert np.ptp(first[:, 2]) > 1e-5
    assert np.max(np.abs(first[:, 2] - .0198)) > 5e-5
    np.testing.assert_allclose(original.integral.pressure_pa, exchanged.integral.pressure_pa, rtol=0, atol=1e-7)
    np.testing.assert_allclose(original.radial_slopes, -np.asarray(exchanged.radial_slopes), rtol=0, atol=1e-12)
    np.testing.assert_allclose(original.integral.first_force_n, exchanged.integral.first_force_n, rtol=0, atol=1e-12)


def test_equal_radius_and_modulus_is_a_plane_with_independent_radial_force():
    result = two_sphere_case(modulus_b_pa=1e5)
    integral = result.integral
    np.testing.assert_allclose(np.asarray(integral.quadrature.points_world)[:, 2], .0198, rtol=0, atol=1e-12)
    np.testing.assert_allclose(integral.quadrature.normals_world, np.tile((0., 0., -1.), (len(integral.pressure_pa), 1)), rtol=0, atol=1e-12)
    assert integral.area_m2 == pytest.approx(result.area_projected_m2, rel=0, abs=1e-18)
    np.testing.assert_allclose(integral.cop_world, (0., 0., .0198), rtol=0, atol=1e-12)
    assert result.axis_pressure_pa == pytest.approx(1000., rel=0, abs=1e-7)
    # Each equal sphere contributes half the overlap; independent closed-form radial law.
    force = pi * 1e5 * .0002**2 * (1 - 2 * .0002 / (3 * .02))
    assert abs(-integral.first_force_n[2] / force - 1) <= .05


def test_unequal_radii_and_moduli_preserve_shallow_overlap_contract():
    ra, rb, ea, eb, overlap = .015, .025, 2e5, 1e5, .0003
    result = two_sphere_case(ra, rb, ea, eb, overlap, .0005)
    force, area, projected = _pressure_parameter_oracle(ra, rb, ea, eb, overlap)
    axis_pressure = overlap / (ra / ea + rb / eb)
    assert result.axis_height_m == pytest.approx(ra * (1 - axis_pressure / ea), rel=0, abs=1e-12)
    assert result.axis_pressure_pa == pytest.approx(axis_pressure, rel=0, abs=1e-7)
    assert result.max_pressure_residual_pa <= 1e-7
    assert abs(-result.integral.first_force_n[2] / force - 1) <= .05
    assert abs(result.integral.area_m2 / area - 1) <= .05
    assert abs(result.area_projected_m2 / projected - 1) <= .05


def test_bilateral_moments_share_one_origin_and_pointwise_forces_are_opposites():
    result = two_sphere_case(origin_world=(.01, -.02, .03))
    integral = result.integral
    np.testing.assert_array_equal(np.asarray(integral.first_force_per_point_n) + integral.second_force_per_point_n, 0.)
    force = -integral.first_force_n[2]
    np.testing.assert_allclose(integral.first_moment_nm, (-.02 * force, -.01 * force, 0.), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(integral.first_moment_nm) + integral.second_moment_nm, 0.)
    assert integral.origin_world == (.01, -.02, .03)


def test_zero_overlap_is_an_empty_domain_not_a_relative_error_case():
    result = two_sphere_case(overlap_m=0., resolution_m=.0005)
    assert result.circle_radius_m == result.area_projected_m2 == 0.
    assert result.axis_height_m is result.axis_pressure_pa is None
    assert result.pressure_a_pa == result.pressure_b_pa == result.pressure_residual_pa == ()
    assert result.vertex_pressure_residual_pa == result.projected_area_weights_m2 == result.radial_slopes == ()
    assert result.max_pressure_residual_pa == 0.
    assert result.integral.area_m2 == result.integral.pressure_load_n == 0.
    assert result.integral.first_force_n == result.integral.first_moment_nm == (0., 0., 0.)
    assert result.integral.cop_world is result.integral.pressure_centroid_world is None
    assert result.integral.quadrature.vertices_world == result.integral.quadrature.points_world == ()
    json.dumps(result.to_dict(), allow_nan=False)


def test_bad_parameters_nonfinite_values_and_unbounded_mesh_are_rejected():
    for options in ({"radius_a_m": 0.}, {"radius_b_m": -.02}, {"radius_a_m": np.nan},
                    {"modulus_a_pa": 0.}, {"modulus_b_pa": np.inf}, {"modulus_a_pa": True},
                    {"overlap_m": -.001}, {"overlap_m": .02}, {"overlap_m": .03},
                    {"overlap_m": np.nan}, {"resolution_m": 0.}, {"resolution_m": 1e-12},
                    {"resolution_m": np.inf}, {"radius_a_m": 1e308, "radius_b_m": 1e308},
                    {"origin_world": (0., 0.)}, {"origin_world": (0., 0., np.inf)},
                    {"origin_world": (0., True, 0.)}):
        with pytest.raises(ValueError):
            two_sphere_case(**options)
    with pytest.raises(ValueError):
        two_sphere_case(overlap_m=0., origin_world=(0., 0., np.nan))
    with pytest.raises(TypeError):
        two_sphere_case(center_b_world=(.01, 0., .0396))  # Not an arbitrary-pose contact finder.


def test_owned_frozen_json_payload_and_no_production_physics_import(monkeypatch):
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in ("phymeo", "mujoco", "tyx", "scipy"):
            raise AssertionError("V03 must remain an independent NumPy static reference")
        return original_import(name, *args, **kwargs)

    origin = [0., 0., .01]
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guard)
        result = two_sphere_case(origin_world=origin, resolution_m=.002)
    origin[2] = 99.
    assert result.integral.origin_world == (0., 0., .01)
    with pytest.raises(FrozenInstanceError):
        result.circle_radius_m = 0.
    with pytest.raises(TypeError):
        result.integral.quadrature.points_world[0][0] = 0.
    payload = result.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False))["integral"]["qualification"] == "NUMPY_STATIC_REFERENCE"
    payload["integral"]["area_m2"] = 99.
    assert result.integral.area_m2 != 99.
    with pytest.raises(ValueError):
        replace(result, max_pressure_residual_pa=float("nan")).to_dict()
