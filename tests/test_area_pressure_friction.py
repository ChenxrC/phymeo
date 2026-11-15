"""Independent V04 prescribed-slip oracles; no dynamics or static-friction claim.

The continuous references below use polar Gauss/midpoint integration, not the
product's triangles, quadrature weights, or implementation helpers.
"""

import builtins
import json
from dataclasses import FrozenInstanceError, replace
from math import pi, sin, sqrt

import numpy as np
import pytest

from experiments.area_pressure.friction_reference import disk_quadrature, integrate_friction
from experiments.area_pressure.reference import PressureIntegral, SurfaceQuadrature, triangle_quadrature


RADIUS = 0.01
PRESSURE = 2000.0
MU = 0.5
SLIP_SPEED = 0.001
REFINEMENTS = ((2, 16), (4, 32), (8, 64))
CENTERS = (((0.0, 0.0),), ((-0.015, 0.0), (0.015, 0.0)))


def _single_torque(law, omega):
    if law == "coulomb":
        return -np.sign(omega) * (2 / 3) * MU * PRESSURE * pi * RADIUS**3
    nodes, weights = np.polynomial.legendre.leggauss(64)
    radius = (nodes + 1) * RADIUS / 2
    integral = (weights * RADIUS / 2) @ (
        radius**3 / np.sqrt(omega**2 * radius**2 + SLIP_SPEED**2)
    )
    return -2 * pi * MU * PRESSURE * omega * integral


def _polar_disk_wrenches(centers, law, omega):
    """Independent 64 radial Gauss x 256 midpoint angles about each disk center.

    Velocity and torque use the GLOBAL x,y after adding the disk center. In
    particular this is not two copies of a disk spinning about its own center.
    """
    nodes, weights = np.polynomial.legendre.leggauss(64)
    radius = (nodes + 1) * RADIUS / 2
    theta = 2 * pi * (np.arange(256) + 0.5) / 256
    area = (weights * RADIUS / 2 * radius)[:, None] * (2 * pi / 256)
    result = []
    for cx, cy in centers:
        x = cx + radius[:, None] * np.cos(theta)
        y = cy + radius[:, None] * np.sin(theta)
        speed = abs(omega) * np.hypot(x, y)
        denominator = speed if law == "coulomb" else np.sqrt(speed**2 + SLIP_SPEED**2)
        fx = MU * PRESSURE * omega * y / denominator
        fy = -MU * PRESSURE * omega * x / denominator
        result.append((float(np.sum(area * fx)), float(np.sum(area * fy)),
                       float(np.sum(area * (x * fy - y * fx)))))
    return np.asarray(result)


def _rotating_result(centers, law, omega, rings, sectors):
    quad = disk_quadrature(RADIUS, centers, rings=rings, sectors=sectors)
    points = np.asarray(quad.points_world)
    velocity = np.column_stack((-omega * points[:, 1], omega * points[:, 0],
                                np.zeros(len(points))))
    return integrate_friction(
        quad, np.full(len(points), PRESSURE), velocity_a_world=velocity,
        velocity_b_world=np.zeros_like(velocity), mu=MU, law=law,
        slip_speed_m_s=SLIP_SPEED,
    )


def _assert_point_laws_and_balances(result):
    quad = result.normal.quadrature
    normal = np.asarray(quad.normals_world).reshape(-1, 3)
    tangent = np.asarray(result.tangential_velocity_world).reshape(-1, 3)
    traction = np.asarray(result.tangential_traction_pa).reshape(-1, 3)
    force = np.asarray(result.first_friction_force_per_point_n).reshape(-1, 3)
    pressure = np.asarray(result.normal.pressure_pa)
    np.testing.assert_allclose(np.sum(tangent * normal, axis=1), 0, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.sum(traction * normal, axis=1), 0, rtol=0, atol=1e-10)
    assert np.all(np.linalg.norm(traction, axis=1) <= result.mu * pressure + 1e-10)
    np.testing.assert_allclose(force, traction * np.asarray(quad.area_weights_m2)[:, None],
                               rtol=0, atol=1e-10)
    relative = np.asarray(result.velocity_a_world).reshape(-1, 3)
    relative = relative - np.asarray(result.velocity_b_world).reshape(-1, 3)
    powers = np.asarray(result.friction_power_per_point_w)
    np.testing.assert_allclose(powers, np.sum(force * relative, axis=1), rtol=0, atol=1e-12)
    assert np.all(powers <= 1e-12) and result.friction_power_w <= 1e-12
    assert result.friction_power_w == pytest.approx(float(powers.sum()), rel=0, abs=1e-12)
    for owner, fields in (
        (result, ("friction_force_per_point_n", "friction_force_n", "friction_moment_nm",
                  "total_force_n", "total_moment_nm")),
        (result.normal, ("force_per_point_n", "force_n", "moment_nm")),
    ):
        for field in fields:
            first = np.asarray(getattr(owner, "first_" + field))
            second = np.asarray(getattr(owner, "second_" + field))
            np.testing.assert_allclose(first + second, np.zeros_like(first), rtol=0, atol=1e-12)
    for quantity in ("force_n", "moment_nm"):
        np.testing.assert_allclose(
            getattr(result, "first_total_" + quantity),
            np.asarray(getattr(result.normal, "first_" + quantity))
            + getattr(result, "first_friction_" + quantity), rtol=0, atol=1e-12,
        )


@pytest.mark.parametrize("centers", CENTERS, ids=("single", "double-common-axis"))
@pytest.mark.parametrize("law", ("coulomb", "regularized"))
@pytest.mark.parametrize("omega", (1.0, -1.0))
def test_fixed_three_meshes_against_independent_continuous_torque(centers, law, omega):
    continuous = _polar_disk_wrenches(centers, law, omega)
    expected_torque = _single_torque(law, omega) if len(centers) == 1 else continuous[:, 2].sum()
    expected_normal = len(centers) * PRESSURE * pi * RADIUS**2
    results = []
    for rings, sectors in REFINEMENTS:
        result = _rotating_result(centers, law, omega, rings, sectors)
        results.append(result)
        assert isinstance(result.normal, PressureIntegral)
        assert result.qualification == "NUMPY_PRESCRIBED_SLIP_REFERENCE"
        assert result.normal.qualification == "NUMPY_STATIC_REFERENCE"
        quad = result.normal.quadrature
        assert len(quad.triangles) == len(centers) * sectors * (2 * rings - 1)
        assert len(quad.vertices_world) == len(centers) * (1 + rings * sectors)
        assert len(quad.points_world) == 3 * len(quad.triangles)
        # Actual inscribed polygon area, independent of radial subdivision.
        polygon_area = len(centers) * sectors * RADIUS**2 * sin(2 * pi / sectors) / 2
        assert result.normal.area_m2 == pytest.approx(polygon_area, rel=1e-12)
        assert result.normal.area_m2 < len(centers) * pi * RADIUS**2
        assert result.normal.pressure_load_n == pytest.approx(PRESSURE * polygon_area, rel=1e-12)
        np.testing.assert_allclose(result.first_friction_force_n, (0, 0, 0), rtol=0, atol=1e-10)
        np.testing.assert_allclose(result.first_friction_moment_nm[:2], (0, 0), rtol=0, atol=1e-12)
        assert omega * result.first_friction_moment_nm[2] < 0
        assert result.friction_power_w == pytest.approx(
            omega * result.first_friction_moment_nm[2], rel=0, abs=1e-12
        )
        _assert_point_laws_and_balances(result)
    previous, finest = results[-2:]
    assert abs(finest.normal.first_force_n[2] / expected_normal - 1) <= 0.05
    assert abs(finest.first_friction_moment_nm[2] / expected_torque - 1) <= 0.05
    assert abs(finest.first_friction_moment_nm[2] - previous.first_friction_moment_nm[2]) / abs(
        finest.first_friction_moment_nm[2]
    ) <= 0.05
    if len(centers) == 2:
        assert abs(finest.first_friction_moment_nm[2]) > 2 * abs(_single_torque(law, omega))
        points = np.asarray(finest.normal.quadrature.points_world)
        forces = np.asarray(finest.first_friction_force_per_point_n)
        for mask, oracle in zip((points[:, 0] < 0, points[:, 0] > 0), continuous):
            force = forces[mask].sum(axis=0)
            moment = np.cross(points[mask], forces[mask]).sum(axis=0)
            assert abs(force[0]) <= 1e-10
            assert force[1] == pytest.approx(oracle[1], rel=0.05, abs=1e-10)
            assert moment[2] == pytest.approx(oracle[2], rel=0.05)
        # The two nonzero lateral forces cancel globally; neither disk has zero force.
        assert continuous[0, 1] * continuous[1, 1] < 0


def _point_quad():
    return SurfaceQuadrature(
        ((0., 0., 0.), (4., 0., 0.), (0., 4., 0.)), ((0, 1, 2),),
        ((1., 2., 0.),), ((0., 0., 1.),), (2.,), (0,),
    )


@pytest.mark.parametrize("law", ("coulomb", "regularized"))
def test_hand_point_relative_velocity_normal_projection_force_moment_and_power(law):
    # Relative v=(3,4,7); only (3,4,0) drives traction. B need not be stationary.
    result = integrate_friction(
        _point_quad(), (3.,), velocity_a_world=((13., 2., 8.),),
        velocity_b_world=((10., -2., 1.),), mu=.5, law=law, slip_speed_m_s=.001,
    )
    factor = 1.0 if law == "coulomb" else 5 / sqrt(25 + .001**2)
    force = np.array((-1.8, -2.4, 0.)) * factor
    np.testing.assert_allclose(result.tangential_velocity_world, ((3., 4., 0.),), rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.first_friction_force_n, force, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.first_friction_moment_nm, (0., 0., 1.2 * factor),
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.normal.first_force_n, (0., 0., 6.), rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.normal.first_moment_nm, (12., -6., 0.), rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.first_total_force_n, force + (0., 0., 6.), rtol=0, atol=1e-14)
    assert result.friction_power_w == pytest.approx(-15 * factor, rel=0, abs=1e-13)
    assert result.normal.area_m2 == 2.0  # Explicit area is not renormalized to mesh area 8.
    _assert_point_laws_and_balances(result)
    alternate_speed = integrate_friction(
        _point_quad(), (3.,), velocity_a_world=((13., 2., 8.),),
        velocity_b_world=((10., -2., 1.),), law=law, slip_speed_m_s=2.,
    )
    if law == "coulomb":
        assert alternate_speed.first_friction_force_n == result.first_friction_force_n
    else:
        assert np.linalg.norm(alternate_speed.first_friction_force_n) < np.linalg.norm(force)


@pytest.mark.parametrize("centers", CENTERS, ids=("single", "double"))
def test_fixed_finest_zero_rotation_keeps_normal_load_without_static_friction(centers):
    result = _rotating_result(centers, "regularized", 0., 8, 64)
    assert result.normal.pressure_load_n > 0
    np.testing.assert_array_equal(result.first_friction_force_n, (0., 0., 0.))
    np.testing.assert_array_equal(result.first_friction_moment_nm, (0., 0., 0.))
    assert result.friction_power_w == 0.
    _assert_point_laws_and_balances(result)


def test_zero_mu_pressure_and_tangential_slip_each_remove_only_friction():
    for law in ("coulomb", "regularized"):
        for mu, pressure, velocity in ((0., 3., (1., 2., 3.)),
                                      (.5, 0., (1., 2., 3.)),
                                      (.5, 3., (0., 0., 7.))):
            result = integrate_friction(
                _point_quad(), (pressure,), velocity_a_world=(velocity,),
                velocity_b_world=((0., 0., 0.),), mu=mu, law=law,
            )
            np.testing.assert_array_equal(result.first_friction_force_n, (0., 0., 0.))
            np.testing.assert_array_equal(result.first_friction_moment_nm, (0., 0., 0.))
            assert result.friction_power_w == 0.
            assert result.normal.pressure_load_n == 2 * pressure
            _assert_point_laws_and_balances(result)


def _bent_quad():
    # Two oriented non-coplanar triangles, area 1/2 each: normals +z and +x.
    return triangle_quadrature(
        ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.),
         (2., 0., 0.), (2., 1., 0.), (2., 0., 1.)), ((0, 1, 2), (3, 4, 5)),
    )


def test_generic_bent_surface_uses_each_local_normal_and_its_own_moment_arm():
    quad = _bent_quad()
    result = integrate_friction(
        quad, (2.,) * 6, velocity_a_world=((1., 2., 3.),) * 6,
        velocity_b_world=((0., 0., 0.),) * 6, law="coulomb",
    )
    first = -.5 * np.array((1., 2., 0.)) / sqrt(5)
    second = -.5 * np.array((0., 2., 3.)) / sqrt(13)
    moment = np.cross((1 / 3, 1 / 3, 0.), first) + np.cross((2., 1 / 3, 1 / 3), second)
    np.testing.assert_allclose(result.first_friction_force_n, first + second, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.first_friction_moment_nm, moment, rtol=0, atol=1e-14)
    np.testing.assert_allclose(result.normal.first_force_n, (1., 0., 1.), rtol=0, atol=1e-14)
    assert result.normal.cop_world is None
    _assert_point_laws_and_balances(result)


def test_shared_origin_shift_rigid_rotation_common_boost_and_slip_reversal():
    quad = _bent_quad()
    va = np.tile((1., 2., 3.), (6, 1))
    vb = np.tile((.2, -.1, .3), (6, 1))
    origin = np.array((.1, -.2, .3))
    options = dict(velocity_a_world=va, velocity_b_world=vb, origin_world=origin)
    base = integrate_friction(quad, (2.,) * 6, **options)
    shift = np.array((.4, .2, -.1))
    moved_origin = integrate_friction(quad, (2.,) * 6, **{**options, "origin_world": origin + shift})
    rotation = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
    offset = np.array((-.3, .5, .7))
    rotated_quad = replace(
        quad, vertices_world=np.asarray(quad.vertices_world) @ rotation.T + offset,
        points_world=np.asarray(quad.points_world) @ rotation.T + offset,
        normals_world=np.asarray(quad.normals_world) @ rotation.T,
    )
    transformed = integrate_friction(
        rotated_quad, (2.,) * 6, velocity_a_world=va @ rotation.T + (4., -2., 7.),
        velocity_b_world=vb @ rotation.T + (4., -2., 7.), origin_world=rotation @ origin + offset,
    )
    reverse = integrate_friction(
        quad, (2.,) * 6, velocity_a_world=vb, velocity_b_world=va, origin_world=origin,
    )
    for owner, moved, rotated, quantities in (
        (base, moved_origin, transformed, ("friction", "total")),
        (base.normal, moved_origin.normal, transformed.normal, ("",)),
    ):
        for prefix in quantities:
            prefix = prefix + "_" if prefix else ""
            force = np.asarray(getattr(owner, "first_" + prefix + "force_n"))
            moment = np.asarray(getattr(owner, "first_" + prefix + "moment_nm"))
            np.testing.assert_allclose(getattr(moved, "first_" + prefix + "moment_nm"),
                                       moment - np.cross(shift, force), rtol=0, atol=1e-12)
            np.testing.assert_allclose(getattr(rotated, "first_" + prefix + "force_n"),
                                       rotation @ force, rtol=0, atol=1e-12)
            np.testing.assert_allclose(getattr(rotated, "first_" + prefix + "moment_nm"),
                                       rotation @ moment, rtol=0, atol=1e-12)
    for field in ("first_friction_force_n", "first_friction_moment_nm"):
        np.testing.assert_allclose(getattr(reverse, field), -np.asarray(getattr(base, field)),
                                   rtol=0, atol=1e-12)
    assert reverse.friction_power_w == pytest.approx(base.friction_power_w, rel=0, abs=1e-12)
    assert transformed.friction_power_w == pytest.approx(base.friction_power_w, rel=0, abs=1e-12)
    for result in (base, moved_origin, transformed, reverse):
        _assert_point_laws_and_balances(result)


def test_disk_geometry_admits_tangency_and_preserves_distinct_domains():
    quad = disk_quadrature(.01, ((0., 0.), (.02, 0.)), rings=1, sectors=8)
    assert len(quad.vertices_world) == 18 and len(quad.triangles) == 16
    vertices = np.asarray(quad.vertices_world)
    left_faces = [face for face in quad.triangles if vertices[list(face), 0].mean() < .01]
    right_faces = [face for face in quad.triangles if vertices[list(face), 0].mean() > .01]
    assert len(left_faces) == len(right_faces) == 8
    first = {i for face in left_faces for i in face}
    second = {i for face in right_faces for i in face}
    assert first.isdisjoint(second)
    assert sum(quad.area_weights_m2) == pytest.approx(8 * .01**2 * sin(pi / 4), rel=1e-12)
    for options in (
        {"radius_m": 0.}, {"radius_m": True}, {"radius_m": np.nan},
        {"rings": 0}, {"rings": True}, {"rings": 1.5},
        {"sectors": 7}, {"sectors": False}, {"sectors": np.inf},
        {"centers_xy": ()}, {"centers_xy": ((0., 0.),) * 3},
        {"centers_xy": ((0., 0.), (.019, 0.))},
        {"centers_xy": ((0., 0., 0.),)}, {"centers_xy": ((np.inf, 0.),)},
    ):
        with pytest.raises(ValueError):
            disk_quadrature(**options)


@pytest.mark.parametrize("law", ("coulomb", "regularized"))
def test_bad_inputs_are_rejected_without_mutating_inputs(law):
    quad = _point_quad()
    before = repr(quad)
    defaults = dict(velocity_a_world=((1., 2., 3.),), velocity_b_world=((0., 0., 0.),), law=law)
    for options in (
        {"mu": -1.}, {"mu": True}, {"mu": np.nan},
        {"slip_speed_m_s": 0.}, {"slip_speed_m_s": -1.},
        {"slip_speed_m_s": True}, {"slip_speed_m_s": np.inf},
        {"law": "static"}, {"velocity_a_world": ((1., 2.),)},
        {"velocity_b_world": ()}, {"velocity_b_world": ((0., 0., np.nan),)},
        {"velocity_a_world": ((True, False, True),)},
        {"origin_world": (0., 0.)}, {"origin_world": (0., 0., np.inf)},
    ):
        with pytest.raises(ValueError):
            integrate_friction(quad, (3.,), **{**defaults, **options})
    for pressure in ((-1.,), (np.inf,), (True,), (), (1., 2.), (1e308,)):
        with pytest.raises(ValueError):
            integrate_friction(quad, pressure, **defaults)
    for bad_quad in (
        replace(quad, area_weights_m2=(0.,)),
        replace(quad, normals_world=((0., 0., 2.),)),
        replace(quad, points_world=((np.nan, 2., 0.),)),
        replace(quad, triangle_indices=(1,)),
    ):
        with pytest.raises(ValueError):
            integrate_friction(bad_quad, (3.,), **defaults)
    assert repr(quad) == before


def test_exact_mesh_capacity_and_first_over_budget_domains():
    # One ring at this boundary reaches all three inherited capacities exactly.
    quad = disk_quadrature(rings=1, sectors=8192)
    assert len(quad.vertices_world) == 8193
    assert len(quad.triangles) == 8192
    assert len(quad.points_world) == 24576
    for options in (
        {"rings": 1, "sectors": 8193},
        {"rings": 2, "sectors": 4096},
        # 8192 triangles fit, but two independent centers require 8194 vertices.
        {"rings": 1, "sectors": 4096, "centers_xy": ((-.02, 0.), (.02, 0.))},
        {"rings": 10**100, "sectors": 8},
    ):
        with pytest.raises(ValueError):
            disk_quadrature(**options)
    oversized = replace(_point_quad(), points_world=((1., 2., 0.),) * 24577,
                        normals_world=((0., 0., 1.),) * 24577,
                        area_weights_m2=(1.,) * 24577, triangle_indices=(0,) * 24577)
    with pytest.raises(ValueError):
        integrate_friction(oversized, (1.,) * 24577,
                           velocity_a_world=((0., 0., 0.),) * 24577,
                           velocity_b_world=((0., 0., 0.),) * 24577)


def test_empty_surface_is_a_finite_zero_integral_for_each_law():
    quad = triangle_quadrature((), ())
    for law in ("coulomb", "regularized"):
        result = integrate_friction(quad, (), velocity_a_world=np.empty((0, 3)),
                                    velocity_b_world=np.empty((0, 3)), law=law)
        assert result.normal.area_m2 == 0. and result.normal.pressure_load_n == 0.
        assert result.friction_power_w == 0.
        assert result.first_friction_force_per_point_n == ()
        np.testing.assert_array_equal(result.first_total_force_n, (0., 0., 0.))
        np.testing.assert_array_equal(result.first_total_moment_nm, (0., 0., 0.))
        json.dumps(result.to_dict(), allow_nan=False)


def test_owned_frozen_finite_json_and_no_engine_import_during_calls(monkeypatch):
    quad = _point_quad()
    va = np.array(((3., 4., 7.),))
    vb = np.zeros((1, 3))
    pressure = np.array((3.,))
    mutable_points = np.asarray(quad.points_world).copy()
    caller_quad = replace(quad, points_world=mutable_points)
    arrays = (va, vb, pressure, mutable_points)
    snapshots = tuple(array.tobytes() for array in arrays)
    for array in arrays:
        array.setflags(write=False)
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in ("phymeo", "newton", "warp", "pydrake", "tyx", "mujoco"):
            raise AssertionError("prescribed-slip quadrature must not invoke a physics engine")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guard)
        result = integrate_friction(caller_quad, pressure, velocity_a_world=va, velocity_b_world=vb)
        disk_quadrature(rings=1, sectors=8)
    assert tuple(array.tobytes() for array in arrays) == snapshots
    for array in arrays:
        array.setflags(write=True)
        array[:] = 99.
    assert result.velocity_a_world == ((3., 4., 7.),)
    assert result.normal.quadrature.points_world == ((1., 2., 0.),)
    assert result.normal.pressure_pa == (3.,)
    with pytest.raises(FrozenInstanceError):
        result.mu = .7
    with pytest.raises(TypeError):
        result.first_friction_force_per_point_n[0][0] = 0.
    with pytest.raises(TypeError):
        result.normal.quadrature.points_world[0][0] = 0.
    payload = result.to_dict()
    assert set(payload) == {
        "qualification", "law", "mu", "slip_speed_m_s", "normal", "velocity_a_world",
        "velocity_b_world", "tangential_velocity_world", "tangential_traction_pa",
        "first_friction_force_per_point_n", "second_friction_force_per_point_n",
        "first_friction_force_n", "second_friction_force_n", "first_friction_moment_nm",
        "second_friction_moment_nm", "first_total_force_n", "second_total_force_n",
        "first_total_moment_nm", "second_total_moment_nm", "friction_power_per_point_w",
        "friction_power_w",
    }
    decoded = json.loads(json.dumps(payload, allow_nan=False))
    assert decoded["qualification"] == "NUMPY_PRESCRIBED_SLIP_REFERENCE"
    assert decoded["normal"]["pressure_pa"] == [3.]
    payload["mu"] = 99.
    payload["normal"]["pressure_pa"] = (99.,)
    assert result.mu == .5 and result.normal.pressure_pa == (3.,)
