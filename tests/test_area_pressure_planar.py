"""Independent short tests for the frozen planar pressure-body reference.

Closed-form rectangles and a tie-edge triangle anchor the geometry. Potential
finite differences are test oracles only; they never drive the implicit kick.
No complete 1800-step engineering window is duplicated here.
"""

import builtins
import json
from dataclasses import FrozenInstanceError, asdict, replace
from math import cos, pi, sin, sqrt

import numpy as np
import pytest

from experiments.area_pressure.planar_reference import (
    PlanarConfig, PlanarState, contact_wrench, mechanical_energy_j, sample_contact, step_body,
)


def _state(*, x=0., z=.0097, angle=0., vx=0., vz=0., omega=0., time=.01):
    return PlanarState(time, x, z, angle, vx, vz, omega)


def _q(state):
    return np.array((state.x_m, state.z_m, state.angle_rad))


def _v(state):
    return np.array((state.vx_m_s, state.vz_m_s, state.omega_rad_s))


def _rotation(angle):
    return np.array(((cos(angle), sin(angle)), (-sin(angle), cos(angle))))


def _rectangle(hx=.02, hz=.01):
    return ((-hx, -hz), (hx, -hz), (hx, hz), (-hx, hz))


def _polygon(count, radius):
    return tuple((radius * cos(2 * pi * i / count), radius * sin(2 * pi * i / count))
                 for i in range(count))


def _flat_oracle(config, depth):
    length = .04
    scale = config.width_m * config.stiffness_pa_per_m
    return (scale * (length * depth - depth**2),
            scale * (length * depth**2 / 2 - depth**3 / 3))


def _check_contact(config, sample):
    points = np.asarray(sample.points_xz).reshape(-1, 2)
    normals = np.asarray(sample.normals_xz).reshape(-1, 2)
    pressure, weights = np.asarray(sample.pressure_pa), np.asarray(sample.area_weights_m2)
    sources = np.asarray(sample.source_edges).reshape(-1, 2)
    count = len(points)
    assert count <= 512 and count == len(pressure) == len(weights) == len(sources)
    assert count % 2 == 0  # Two Gauss points per retained positive-pressure segment.
    assert np.all(weights > 0) and np.all(pressure > 0)
    assert sample.area_m2 == pytest.approx(float(weights.sum()), rel=1e-12, abs=1e-14)
    assert 0 <= sample.max_depth_m <= config.max_depth_m + 1e-12
    assert sample.potential_j >= 0
    body = np.asarray(config.body_polygon_xz) @ _rotation(sample.state.angle_rad).T
    body += (sample.state.x_m, sample.state.z_m)
    np.testing.assert_allclose(sample.body_world_xz, body, rtol=0, atol=1e-13)
    if count:
        assert sources.dtype.kind in "iu"
        for edge_id in range(len(config.support_polygon_xz)):
            assert np.count_nonzero(sources[:, 0] == edge_id) <= 58
        edge = np.roll(body, -1, axis=0) - body
        delta = points[:, None, :] - body[None, :, :]
        distances = (edge[None, :, 0] * delta[:, :, 1]
                     - edge[None, :, 1] * delta[:, :, 0]) / np.linalg.norm(edge, axis=1)
        depths = np.min(distances, axis=1)
        assert np.all(depths >= -1e-12)
        np.testing.assert_allclose(pressure, config.stiffness_pa_per_m * depths,
                                   rtol=1e-10, atol=1e-8)
        assert float(depths.max()) <= sample.max_depth_m + 1e-12
        support = np.asarray(config.support_polygon_xz)
        for point, normal, (support_id, body_id), ds in zip(points, normals, sources, distances):
            assert 0 <= support_id < len(support) and 0 <= body_id < len(body)
            a, b = support[support_id], support[(support_id + 1) % len(support)]
            direction = b - a
            length = np.linalg.norm(direction)
            expected_normal = np.array((direction[1], -direction[0])) / length
            np.testing.assert_allclose(normal, expected_normal, rtol=0, atol=1e-12)
            assert abs(np.dot(point - a, expected_normal)) <= 1e-12
            parameter = np.dot(point - a, direction) / length**2
            assert -1e-10 <= parameter <= 1 + 1e-10
            assert ds[body_id] == pytest.approx(float(ds.min()), rel=0, abs=1e-12)
    else:
        assert sample.points_xz == sample.pressure_pa == sample.source_edges == ()
    force_per_point = pressure[:, None] * normals * weights[:, None]
    relative = points - (sample.state.x_m, sample.state.z_m)
    moment = np.sum(relative[:, 1] * force_per_point[:, 0] - relative[:, 0] * force_per_point[:, 1])
    force = force_per_point.sum(axis=0)
    np.testing.assert_allclose(sample.elastic_generalized_force, (*force, moment), rtol=1e-11, atol=1e-12)


def _manual_wrench(config, sample, velocity):
    """Direct per-point law using the declared world tangent and COM arm."""
    points = np.asarray(sample.points_xz).reshape(-1, 2)
    normals = np.asarray(sample.normals_xz).reshape(-1, 2)
    tangent = np.column_stack((-normals[:, 1], normals[:, 0]))
    relative = points - (sample.state.x_m, sample.state.z_m)
    speed = np.asarray(velocity[:2]) + velocity[2] * np.column_stack((relative[:, 1], -relative[:, 0]))
    vn, vt = np.sum(normals * speed, axis=1), np.sum(tangent * speed, axis=1)
    pe = np.asarray(sample.pressure_pa)
    pn = pe * np.maximum(0., 1 - config.dissipation_s_per_m * vn)
    pt = -config.friction * pn * vt / np.sqrt(vt**2 + config.slip_speed_m_s**2)
    weights = np.asarray(sample.area_weights_m2)
    point_force = (pn[:, None] * normals + pt[:, None] * tangent) * weights[:, None]
    force = point_force.sum(axis=0)
    moment = np.sum(relative[:, 1] * point_force[:, 0] - relative[:, 0] * point_force[:, 1])
    support_moment = -np.sum(points[:, 1] * point_force[:, 0] - points[:, 0] * point_force[:, 1])
    dissipation = -float(np.sum(((pn - pe) * vn + pt * vt) * weights))
    return np.array((*force, moment)), support_moment, pn, pt, vn, vt, dissipation


def _check_wrench(config, sample, wrench, velocity):
    expected, moment, pn, pt, vn, vt, dissipation = _manual_wrench(config, sample, velocity)
    np.testing.assert_allclose(wrench.generalized_force, expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(wrench.support_force_xz, -expected[:2], rtol=0, atol=1e-11)
    assert wrench.support_moment_origin_nm == pytest.approx(moment, rel=1e-11, abs=1e-12)
    for field, values in (("normal_pressure_pa", pn), ("tangent_pressure_pa", pt),
                          ("normal_velocity_m_s", vn), ("tangent_velocity_m_s", vt)):
        np.testing.assert_allclose(getattr(wrench, field), values, rtol=1e-11, atol=1e-10)
    assert wrench.dissipation_power_w == pytest.approx(dissipation, rel=1e-11, abs=1e-12)
    assert wrench.dissipation_power_w >= -1e-10
    fx, fz, my = wrench.generalized_force
    assert my + sample.state.z_m * fx - sample.state.x_m * fz + wrench.support_moment_origin_nm == pytest.approx(
        0., rel=0, abs=1e-10
    )


@pytest.mark.parametrize("depth", (.0003065625, .001))
def test_full_top_face_exact_pressure_force_energy_and_width_scaling(depth):
    baseline = None
    for width in (.04, .08):
        config = PlanarConfig(width_m=width, dissipation_s_per_m=0., friction=0.)
        state = _state(z=.01 - depth)
        sample = sample_contact(config, state)
        _check_contact(config, sample)
        force, potential = _flat_oracle(config, depth)
        np.testing.assert_allclose(sample.elastic_generalized_force, (0., force, 0.), rtol=1e-10, atol=1e-10)
        assert sample.potential_j == pytest.approx(potential, rel=1e-10, abs=1e-13)
        assert sample.max_depth_m == pytest.approx(depth, rel=0, abs=1e-12)
        assert sample.area_m2 == pytest.approx(width * .04, rel=1e-11, abs=1e-14)
        assert set(pair[0] for pair in sample.source_edges) == {2}
        assert mechanical_energy_j(config, state) == pytest.approx(
            potential + config.mass_kg * config.gravity_m_s2 * state.z_m, rel=1e-11, abs=1e-13
        )
        if baseline is None:
            baseline = sample
        else:
            assert sample.potential_j == pytest.approx(2 * baseline.potential_j, rel=1e-11)
            np.testing.assert_allclose(sample.elastic_generalized_force,
                                       2 * np.asarray(baseline.elastic_generalized_force), rtol=1e-11, atol=1e-12)
    if depth == .0003065625:
        assert baseline.elastic_generalized_force[1] < .5 * 9.81
        assert baseline.elastic_generalized_force[1] / (.5 * 9.81) == pytest.approx(1 - depth / .04, rel=1e-11)


def test_bottom_face_has_reversed_outward_normal_and_same_potential():
    config = PlanarConfig(dissipation_s_per_m=0., friction=0.)
    depth = .0004
    sample = sample_contact(config, _state(z=-.02 + depth))
    _check_contact(config, sample)
    force, potential = _flat_oracle(config, depth)
    np.testing.assert_allclose(sample.elastic_generalized_force, (0., -force, 0.), rtol=1e-10, atol=1e-10)
    assert sample.potential_j == pytest.approx(potential, rel=1e-10, abs=1e-13)
    assert set(pair[0] for pair in sample.source_edges) == {0}


def test_side_face_and_both_corner_faces_have_independent_closed_form_loads():
    config = PlanarConfig(dissipation_s_per_m=0., friction=0.)
    depth, thickness = .0005, .01
    sample = sample_contact(config, _state(x=.06 - depth, z=-.005))
    _check_contact(config, sample)
    scale = config.width_m * config.stiffness_pa_per_m
    force = scale * thickness * depth
    np.testing.assert_allclose(sample.elastic_generalized_force, (force, 0., 0.), rtol=1e-10, atol=1e-10)
    assert sample.potential_j == pytest.approx(scale * thickness * depth**2 / 2, rel=1e-10, abs=1e-13)
    assert sample.area_m2 == pytest.approx(config.width_m * (thickness + 2 * depth), rel=1e-11, abs=1e-14)
    assert set(pair[0] for pair in sample.source_edges) == {0, 1, 2}
    wrench = contact_wrench(config, sample)
    _check_wrench(config, sample, wrench, (0., 0., 0.))
    assert wrench.support_moment_origin_nm == pytest.approx(.005 * force, rel=1e-10, abs=1e-12)


def test_segment_on_nearest_edge_tie_is_integrated_once_with_analytic_moment():
    # In local coordinates s=x+a, t=z+b the overlap is 0<=t<=s<=delta.
    # One target edge is the ENTIRE pressure tie s=t; p=kappa*t in the volume.
    a, b, depth = .02, .01, .001
    support = ((-a, -b), (-a + depth, -b - depth), (-a + depth, -b + depth))
    config = PlanarConfig(support_polygon_xz=support, dissipation_s_per_m=0., friction=0.)
    state = _state(z=0.)
    sample = sample_contact(config, state)
    _check_contact(config, sample)
    scale = config.width_m * config.stiffness_pa_per_m
    expected = (0., scale * depth**2 / 2, scale * (a * depth**2 / 2 - depth**3 / 3))
    np.testing.assert_allclose(sample.elastic_generalized_force, expected, rtol=1e-10, atol=1e-11)
    assert sample.potential_j == pytest.approx(scale * depth**3 / 6, rel=1e-10, abs=1e-13)
    assert sample.area_m2 == pytest.approx(config.width_m * depth * (1 + sqrt(2)), rel=1e-11, abs=1e-14)
    assert set(pair[0] for pair in sample.source_edges) == {1, 2}
    assert all(pair[1] == 0 for pair in sample.source_edges)  # bottom=0 wins the bottom/left=3 tie.
    cycled = replace(config, body_polygon_xz=np.roll(config.body_polygon_xz, 1, axis=0),
                      support_polygon_xz=np.roll(support, 2, axis=0))
    changed = sample_contact(cycled, state)
    _check_contact(cycled, changed)
    np.testing.assert_allclose(changed.elastic_generalized_force, expected, rtol=1e-10, atol=1e-11)
    assert changed.potential_j == pytest.approx(sample.potential_j, rel=1e-10, abs=1e-13)
    assert changed.area_m2 == pytest.approx(sample.area_m2, rel=1e-11, abs=1e-14)


@pytest.mark.parametrize("angle", (pi / 2, .73))
def test_rigid_rotation_and_common_translation_transform_geometry_and_wrench(angle):
    config = PlanarConfig()
    state = _state(vx=.03, vz=-.007, omega=.1)
    original = sample_contact(config, state)
    wrench = contact_wrench(config, original)
    rotation, offset = _rotation(angle), np.array((.1, -.05))
    position = rotation @ _q(state)[:2] + offset
    velocity = rotation @ _v(state)[:2]
    transformed_config = replace(config, support_polygon_xz=np.asarray(config.support_polygon_xz) @ rotation.T + offset)
    transformed_state = replace(state, x_m=position[0], z_m=position[1], angle_rad=state.angle_rad + angle,
                                vx_m_s=velocity[0], vz_m_s=velocity[1])
    transformed = sample_contact(transformed_config, transformed_state)
    _check_contact(transformed_config, transformed)
    after = contact_wrench(transformed_config, transformed)
    _check_wrench(transformed_config, transformed, after, _v(transformed_state))
    assert transformed.area_m2 == pytest.approx(original.area_m2, rel=1e-10, abs=1e-13)
    assert transformed.potential_j == pytest.approx(original.potential_j, rel=1e-10, abs=1e-13)
    assert transformed.max_depth_m == pytest.approx(original.max_depth_m, rel=0, abs=1e-12)
    np.testing.assert_allclose(after.generalized_force[:2], rotation @ np.asarray(wrench.generalized_force[:2]),
                               rtol=1e-10, atol=1e-10)
    assert after.generalized_force[2] == pytest.approx(wrench.generalized_force[2], rel=1e-10, abs=1e-12)
    assert after.dissipation_power_w == pytest.approx(wrench.dissipation_power_w, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("body", (None, ((-.02, -.01), (.02, -.01), (.005, .015))),
                         ids=("rotated-rectangle", "asymmetric-triangle"))
def test_elastic_wrench_is_negative_potential_gradient_in_all_three_coordinates(body):
    config = PlanarConfig(dissipation_s_per_m=0., friction=0., gravity_m_s2=0.)
    if body is not None:
        config = replace(config, body_polygon_xz=body)
    state = _state(x=.035 if body is None else .01, z=.0098, angle=.025)
    sample = sample_contact(config, state)
    _check_contact(config, sample)
    assert sample.area_m2 > 0 and sample.potential_j > 0
    for field, epsilon, index, atol in (("x_m", 1e-7, 0, 2e-6), ("z_m", 1e-7, 1, 2e-6),
                                        ("angle_rad", 1e-6, 2, 2e-8)):
        for h in (epsilon, epsilon / 2):
            plus = sample_contact(config, replace(state, **{field: getattr(state, field) + h})).potential_j
            minus = sample_contact(config, replace(state, **{field: getattr(state, field) - h})).potential_j
            expected = -(plus - minus) / (2 * h)
            assert sample.elastic_generalized_force[index] == pytest.approx(expected, rel=2e-5, abs=atol)


def test_complete_overlap_cells_detect_depth_hidden_from_boundary_vertices():
    cases = (
        _rectangle(.1, .1),  # All body vertices have p=0; the body interior is deep.
        ((0., -.009), (.019, 0.), (0., .009), (-.019, 0.)),  # Support vertices are only 1mm deep.
    )
    for support in cases:
        config = PlanarConfig(support_polygon_xz=support)
        with pytest.raises(ValueError):
            sample_contact(config, _state(z=0.))


def test_zero_pressure_touch_and_true_empty_domain_have_no_contact_area():
    config = PlanarConfig()
    for state in (_state(z=.01), _state(z=.010000001), _state(z=.1),
                  _state(x=.06, z=-.005), _state(x=.06, z=.01)):
        sample = sample_contact(config, state)
        _check_contact(config, sample)
        assert sample.area_m2 == sample.potential_j == sample.max_depth_m == 0.
        assert sample.points_xz == ()
        wrench = contact_wrench(config, sample)
        _check_wrench(config, sample, wrench, _v(state))
        np.testing.assert_array_equal(wrench.generalized_force, (0., 0., 0.))


def test_contact_law_projection_rotation_clipping_and_shared_origin_reaction():
    config = PlanarConfig()
    state = _state(x=.035, z=.0098, angle=.025)
    sample = sample_contact(config, state)
    _check_contact(config, sample)
    velocity = (.08, .01, 4.)
    wrench = contact_wrench(config, sample, velocity=velocity)
    _check_wrench(config, sample, wrench, velocity)
    # Zero actual load does not erase a geometrically positive elastic patch.
    flat = sample_contact(config, _state())
    clipped = contact_wrench(config, flat, velocity=(0., .1, 0.))
    _check_wrench(config, flat, clipped, (0., .1, 0.))
    assert flat.area_m2 > 0 and flat.potential_j > 0
    np.testing.assert_array_equal(clipped.generalized_force, (0., 0., 0.))
    assert clipped.dissipation_power_w > 0
    frictionless = replace(config, friction=0., dissipation_s_per_m=0.)
    conservative = contact_wrench(frictionless, flat, velocity=(.1, -.1, 2.))
    np.testing.assert_allclose(conservative.generalized_force, flat.elastic_generalized_force,
                               rtol=1e-11, atol=1e-12)
    assert conservative.dissipation_power_w == 0.


def _check_kick(config, kick, before, external, duration):
    np.testing.assert_allclose(kick.velocity_before, before, rtol=0, atol=1e-13)
    mid, after = np.asarray(kick.mid_velocity), np.asarray(kick.velocity_after)
    mass = np.array((config.mass_kg, config.mass_kg, config.inertia_kg_m2))
    np.testing.assert_allclose(mid, (np.asarray(before) + after) / 2, rtol=0, atol=1e-12)
    _check_contact(config, kick.sample)
    _check_wrench(config, kick.sample, kick.contact, mid)
    force = np.asarray(kick.contact.generalized_force)
    residual = mid - before - duration / 2 * (force + external) / mass
    scaled = max(abs(residual[0]), abs(residual[1]), .02 * abs(residual[2]))
    assert scaled <= 1e-12
    assert 0 <= kick.residual_m_s <= 1e-12
    assert 0 <= kick.iterations <= 24
    assert kick.duration_s == duration
    np.testing.assert_allclose(kick.external_generalized_force, external, rtol=0, atol=1e-13)
    np.testing.assert_allclose(kick.contact_impulse, duration * force, rtol=0, atol=1e-12)
    np.testing.assert_allclose(kick.external_impulse, duration * external, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mass * (after - before),
                               np.asarray(kick.contact_impulse) + kick.external_impulse, rtol=0, atol=1e-9)
    assert kick.dissipated_j == pytest.approx(duration * kick.contact.dissipation_power_w, rel=1e-11, abs=1e-12)
    assert kick.dissipated_j >= -1e-10
    assert kick.external_work_j == pytest.approx(duration * np.dot(external, mid), rel=1e-11, abs=1e-12)
    kinetic_delta = np.sum(mass * (after**2 - np.asarray(before)**2)) / 2
    assert kinetic_delta == pytest.approx(duration * np.dot(force + external, mid), rel=1e-10, abs=1e-11)


def _check_step(config, entry, record, dt, user_force, user_moment):
    external = np.array((user_force[0], user_force[1] - config.mass_kg * config.gravity_m_s2, user_moment))
    assert record.entry == entry and record.first.sample.state == entry
    assert record.second.sample.state == record.drifted
    _check_kick(config, record.first, _v(entry), external, dt / 2)
    _check_kick(config, record.second, record.first.velocity_after, external, dt / 2)
    np.testing.assert_allclose(_q(record.drifted), _q(entry) + dt * np.asarray(record.first.velocity_after), rtol=0, atol=1e-12)
    np.testing.assert_allclose(_v(record.drifted), record.first.velocity_after, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_q(record.exit), _q(record.drifted), rtol=0, atol=1e-12)
    np.testing.assert_allclose(_v(record.exit), record.second.velocity_after, rtol=0, atol=1e-12)
    assert record.drifted.time_s == record.exit.time_s == entry.time_s + dt
    expected = np.asarray(record.first.contact_impulse) + record.second.contact_impulse
    np.testing.assert_allclose(record.contact_impulse, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(record.average_contact_generalized_force, expected / dt, rtol=0, atol=1e-12)
    assert record.dissipated_j == pytest.approx(record.first.dissipated_j + record.second.dissipated_j, abs=1e-12)


def test_normal_only_implicit_kicks_match_independent_closed_form_midpoint_equations():
    config = PlanarConfig(friction=0.)
    entry, dt = _state(vz=-.01), .001
    user_force, moment = (0., 1.), 0.
    mass, tau = config.mass_kg, dt / 2
    external_z = 1 - mass * config.gravity_m_s2
    depth = .01 - entry.z_m
    first_force, _ = _flat_oracle(config, depth)
    first_mid = (entry.vz_m_s + tau / (2 * mass) * (external_z + first_force)) / (
        1 + tau * first_force * config.dissipation_s_per_m / (2 * mass)
    )
    first_after = 2 * first_mid - entry.vz_m_s
    drift_z = entry.z_m + dt * first_after
    second_force, _ = _flat_oracle(config, .01 - drift_z)
    second_mid = (first_after + tau / (2 * mass) * (external_z + second_force)) / (
        1 + tau * second_force * config.dissipation_s_per_m / (2 * mass)
    )
    record = step_body(config, entry, dt, external_force_xz_n=user_force)
    _check_step(config, entry, record, dt, user_force, moment)
    assert record.first.mid_velocity[1] == pytest.approx(first_mid, rel=0, abs=1e-12)
    assert record.second.mid_velocity[1] == pytest.approx(second_mid, rel=0, abs=1e-12)
    assert record.exit.vz_m_s == pytest.approx(2 * second_mid - first_after, rel=0, abs=1e-12)
    np.testing.assert_allclose((record.exit.x_m, record.exit.angle_rad, record.exit.vx_m_s, record.exit.omega_rad_s),
                               (0., 0., 0., 0.), rtol=0, atol=1e-12)


def test_coupled_rotation_and_friction_kick_uses_one_midpoint_for_law_and_motion():
    config = PlanarConfig()
    entry = _state(x=.01, z=.0095, angle=.01, vx=.03, vz=-.005, omega=.2)
    before = asdict(entry)
    user_force, moment, dt = (2., .3), .001, .0005
    record = step_body(config, entry, dt, external_force_xz_n=user_force, external_moment_nm=moment)
    _check_step(config, entry, record, dt, user_force, moment)
    assert record.dissipated_j > 0
    assert asdict(entry) == before


@pytest.mark.parametrize("gravity", (0., 9.81))
def test_empty_domain_constant_force_kdk_and_separate_gravity_work(gravity):
    config = PlanarConfig(gravity_m_s2=gravity, friction=0., dissipation_s_per_m=0.)
    entry = _state(z=.2, angle=.3, vx=.2, vz=-.3, omega=.4)
    user, moment, dt = (2., 1.), .002, .001
    external = np.array((2., 1 - config.mass_kg * gravity, moment))
    mass = np.array((config.mass_kg, config.mass_kg, config.inertia_kg_m2))
    acceleration = external / mass
    record = step_body(config, entry, dt, external_force_xz_n=user, external_moment_nm=moment)
    _check_step(config, entry, record, dt, user, moment)
    np.testing.assert_allclose(_v(record.exit), _v(entry) + dt * acceleration, rtol=0, atol=1e-12)
    np.testing.assert_allclose(_q(record.exit), _q(entry) + dt * _v(entry) + .5 * dt**2 * acceleration,
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(record.contact_impulse, (0., 0., 0.))
    assert record.dissipated_j == 0.
    user_generalized = np.array((*user, moment))
    user_work = dt / 2 * np.dot(user_generalized, np.asarray(record.first.mid_velocity) + record.second.mid_velocity)
    energy_delta = mechanical_energy_j(config, record.exit) - mechanical_energy_j(config, entry)
    assert energy_delta == pytest.approx(user_work, rel=1e-10, abs=1e-12)
    external_work = record.first.external_work_j + record.second.external_work_j
    kinetic_delta = np.sum(mass * (_v(record.exit)**2 - _v(entry)**2)) / 2
    assert external_work == pytest.approx(kinetic_delta, rel=1e-10, abs=1e-12)
    if gravity:
        assert abs(external_work - user_work) > 1e-6


def test_second_geometry_depth_failure_preserves_the_whole_entry_transaction():
    config = PlanarConfig(gravity_m_s2=0., friction=0., dissipation_s_per_m=0.)
    entry = _state(z=.0081, vz=-5.)
    assert sample_contact(config, entry).max_depth_m < config.max_depth_m
    before_state, before_config = asdict(entry), asdict(config)
    with pytest.raises(ValueError):
        step_body(config, entry, .001)
    assert asdict(entry) == before_state and asdict(config) == before_config


def test_eight_edge_inputs_and_declared_output_capacity_are_preserved():
    body = _polygon(8, .01)
    support = np.asarray(_polygon(8, .02)) + (0., -.0295)
    config = PlanarConfig(body_polygon_xz=body, support_polygon_xz=support)
    sample = sample_contact(config, _state(z=0.))
    _check_contact(config, sample)
    assert len(config.body_polygon_xz) == len(config.support_polygon_xz) == 8
    assert 0 < len(sample.points_xz) <= 512
    assert len(sample.points_xz) // 2 <= 256


def test_invalid_polygons_parameters_states_and_inputs_reject_without_mutation():
    pentagon = _polygon(5, .02)
    invalid_polygons = ((), _rectangle()[:2], tuple(reversed(_rectangle())),
                        (*_rectangle(), _rectangle()[0]), _polygon(9, .02), _rectangle(1.01, .01),
                        ((0., 0.), (1e-13, 0.), (.01, .01), (0., .01)),
                        tuple(pentagon[i] for i in (0, 2, 4, 1, 3)),
                        ((0., 0.), (.02, 0.), (.01, .005), (.02, .02), (0., .02)),
                        np.asarray(_rectangle(), dtype=object), np.asarray(_rectangle(), dtype=complex),
                        ((True, 0.), (.02, 0.), (0., .02)))
    for polygon in invalid_polygons:
        for field in ("body_polygon_xz", "support_polygon_xz"):
            with pytest.raises(ValueError):
                PlanarConfig(**{field: polygon})
    for options in ({"mass_kg": 0.}, {"inertia_kg_m2": -1.}, {"width_m": True},
                    {"stiffness_pa_per_m": np.inf}, {"stiffness_pa_per_m": 0.}, {"width_m": 0.}, {"max_depth_m": 0.}, {"dissipation_s_per_m": -1.},
                    {"friction": -1.}, {"slip_speed_m_s": 0.}, {"gravity_m_s2": np.nan}):
        with pytest.raises(ValueError):
            PlanarConfig(**options)
    state = _state()
    for field in asdict(state):
        for value in (True, np.nan, np.inf):
            with pytest.raises(ValueError):
                replace(state, **{field: value})
    config, before = PlanarConfig(), asdict(state)
    with pytest.raises(ValueError):
        sample_contact(config, replace(state, x_m=1.01))
    for dt in (0., -.001, .001001, True, np.nan):
        with pytest.raises(ValueError):
            step_body(config, state, dt)
    for kwargs in ({"external_force_xz_n": (1.,)}, {"external_force_xz_n": (True, 0.)},
                   {"external_force_xz_n": (np.inf, 0.)}, {"external_moment_nm": True}):
        with pytest.raises(ValueError):
            step_body(config, state, .001, **kwargs)
    sample = sample_contact(config, state)
    for velocity in ((1., 2.), (0., 0., np.nan), (True, 0., 0.)):
        with pytest.raises(ValueError):
            contact_wrench(config, sample, velocity=velocity)
    assert asdict(state) == before


def test_owned_frozen_complete_records_and_no_physics_engine_import(monkeypatch):
    body, support = np.asarray(_rectangle()), np.asarray(((-.04, -.01), (.04, -.01), (.04, 0.), (-.04, 0.)))
    for array in (body, support):
        array.setflags(write=False)
    config = PlanarConfig(body_polygon_xz=body, support_polygon_xz=support)
    state = _state(vx=.01)
    for array in (body, support):
        array.setflags(write=True)
        array[:] = 99.
    assert config.body_polygon_xz == _rectangle()
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in ("phymeo", "newton", "warp", "pydrake", "tyx", "mujoco"):
            raise AssertionError("planar NumPy reference must not call a physics engine")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guard)
        record = step_body(config, state, .0005)
    for owner, field in ((config, "mass_kg"), (state, "x_m"), (record, "dissipated_j"),
                          (record.first, "duration_s"), (record.first.sample, "area_m2"),
                          (record.first.contact, "dissipation_power_w")):
        with pytest.raises(FrozenInstanceError):
            setattr(owner, field, 99.)
    with pytest.raises(TypeError):
        record.first.sample.points_xz[0][0] = 99.
    payload = record.to_dict()
    assert set(payload) == {"entry", "drifted", "exit", "first", "second", "contact_impulse",
                            "average_contact_generalized_force", "dissipated_j"}
    for name in ("first", "second"):
        assert set(payload[name]) == {"sample", "duration_s", "velocity_before", "velocity_after", "mid_velocity",
                                     "contact", "external_generalized_force", "contact_impulse", "external_impulse",
                                     "dissipated_j", "external_work_j", "iterations", "residual_m_s"}
        assert set(payload[name]["contact"]) == {"generalized_force", "support_force_xz", "support_moment_origin_nm",
                                                "normal_pressure_pa", "tangent_pressure_pa", "normal_velocity_m_s",
                                                "tangent_velocity_m_s", "dissipation_power_w"}
        assert set(payload[name]["sample"]) == {"state", "body_world_xz", "points_xz", "normals_xz",
                                               "area_weights_m2", "pressure_pa", "source_edges", "area_m2",
                                               "potential_j", "max_depth_m", "elastic_generalized_force"}
    json.loads(json.dumps(payload, allow_nan=False))
    payload["entry"]["x_m"] = 99.
    payload["first"]["sample"]["pressure_pa"] = ()
    assert record.entry == state and len(record.first.sample.pressure_pa) > 0
