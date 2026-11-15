"""Independent V06 single-step and short-window tests, not full trajectory gates.

The existing sphere_plane_case supplies the frozen mesh pressure integral.
Kick/drift arithmetic and continuous energy/force formulas below are independent
of the new impact implementation. No product solver or external engine is used.
"""

import builtins
import json
from dataclasses import FrozenInstanceError, asdict, replace
from math import pi

import numpy as np
import pytest

from experiments.area_pressure.impact_reference import (
    AxialImpactConfig,
    AxialPairState,
    mechanical_energy_j,
    sample_contact,
    step_pair,
)
from experiments.area_pressure.reference import PressureIntegral, sphere_plane_case


STATE_FIELDS = ("time_s", "position_a_m", "position_b_m", "velocity_a_m_s", "velocity_b_m_s")


def _state(config, depth=.001, *, time=.02, position_b=.003, va=-.2, vb=.03):
    position_a = config.radius_m + position_b + config.target_thickness_m / 2 - depth
    return AxialPairState(time, position_a, position_b, va, vb)


def _depth(config, state):
    return config.radius_m + state.position_b_m + config.target_thickness_m / 2 - state.position_a_m


def _energy(config, state):
    depth = max(0., _depth(config, state))
    kinetic = config.mass_a_kg * state.velocity_a_m_s**2 / 2
    kinetic += config.mass_b_kg * state.velocity_b_m_s**2 / 2
    potential = pi * config.modulus_pa * (depth**3 / 3 - depth**4 / (6 * config.radius_m))
    return kinetic + potential


def _static_integral(config, state):
    return sphere_plane_case(
        radius_m=config.radius_m, indentation_m=max(0., _depth(config, state)),
        modulus_pa=config.modulus_pa, resolution_m=config.resolution_m,
        plane_position_world=(0., 0., state.position_b_m + config.target_thickness_m / 2),
        origin_world=(0., 0., 0.),
    )


def _manual_kick(config, state, duration):
    """Only the old pressure mesh is reused; all new dynamics are hand arithmetic."""
    elastic = _static_integral(config, state)
    force = elastic.first_force_n[2]
    u = state.velocity_a_m_s - state.velocity_b_m_s
    inverse_mass = 1 / config.mass_a_kg + 1 / config.mass_b_kg
    d = config.dissipation_s_per_m
    impulse = duration * force * max(0., 1 - d * u) / (1 + duration * force * d * inverse_mass / 2)
    mid_velocity = u + inverse_mass * impulse / 2
    factor = max(0., 1 - d * mid_velocity)
    return {
        "elastic": elastic, "impulse": impulse, "mid_velocity": mid_velocity, "factor": factor,
        "va": state.velocity_a_m_s + impulse / config.mass_a_kg,
        "vb": state.velocity_b_m_s - impulse / config.mass_b_kg,
        "dissipation": (duration * force - impulse) * mid_velocity,
    }


def _assert_state(actual, expected):
    for field in STATE_FIELDS:
        assert getattr(actual, field) == pytest.approx(getattr(expected, field), rel=0, abs=1e-12)


def _assert_sample(config, state, sample, elastic):
    assert isinstance(sample.elastic, PressureIntegral)
    assert sample.indentation_m == pytest.approx(_depth(config, state), rel=0, abs=1e-14)
    u = state.velocity_a_m_s - state.velocity_b_m_s
    assert sample.relative_velocity_m_s == pytest.approx(u, rel=0, abs=1e-14)
    factor = max(0., 1 - config.dissipation_s_per_m * u)
    assert sample.instantaneous_factor == pytest.approx(factor, rel=1e-12, abs=1e-14)
    assert sample.instantaneous_force_n == pytest.approx(elastic.first_force_n[2] * factor,
                                                        rel=1e-12, abs=1e-12)
    assert sample.elastic.origin_world == (0., 0., 0.)
    assert sample.elastic.area_m2 == pytest.approx(elastic.area_m2, rel=1e-12, abs=1e-15)
    assert sample.elastic.pressure_load_n == pytest.approx(elastic.pressure_load_n, rel=1e-12, abs=1e-12)
    assert len(sample.elastic.pressure_pa) == len(elastic.pressure_pa)
    np.testing.assert_allclose(sample.elastic.pressure_pa, elastic.pressure_pa, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(sample.elastic.quadrature.points_world,
                               elastic.quadrature.points_world, rtol=0, atol=1e-14)
    np.testing.assert_allclose(sample.elastic.first_force_n, elastic.first_force_n, rtol=1e-12, atol=1e-12)
    for suffix in ("force_per_point_n", "force_n", "moment_nm"):
        first = np.asarray(getattr(sample.elastic, "first_" + suffix))
        second = np.asarray(getattr(sample.elastic, "second_" + suffix))
        np.testing.assert_allclose(first + second, np.zeros_like(first), rtol=0, atol=1e-12)
    np.testing.assert_allclose(sample.elastic.first_moment_nm, (0., 0., 0.), rtol=0, atol=1e-12)
    if sample.indentation_m <= 0:
        assert sample.elastic.area_m2 == sample.elastic.pressure_load_n == 0.
        assert sample.elastic.pressure_pa == () and sample.elastic.cop_world is None
    else:
        points = np.asarray(sample.elastic.quadrature.points_world)
        np.testing.assert_allclose(points[:, 2], state.position_b_m + config.target_thickness_m / 2,
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(sample.elastic.cop_world,
                                   (0., 0., state.position_b_m + config.target_thickness_m / 2),
                                   rtol=0, atol=1e-12)


def _assert_kick(config, before, kick, duration, expected):
    _assert_sample(config, before, kick.sample, expected["elastic"])
    assert kick.duration_s == pytest.approx(duration, rel=0, abs=1e-15)
    assert kick.velocity_a_before == before.velocity_a_m_s
    assert kick.velocity_b_before == before.velocity_b_m_s
    for field, key in (("velocity_a_after", "va"), ("velocity_b_after", "vb"),
                       ("relative_mid_velocity_m_s", "mid_velocity"), ("factor", "factor"),
                       ("impulse_n_s", "impulse"), ("dissipated_j", "dissipation")):
        assert getattr(kick, field) == pytest.approx(expected[key], rel=1e-12, abs=1e-12)
    assert kick.impulse_n_s >= 0 and kick.dissipated_j >= -1e-12
    assert config.mass_a_kg * (kick.velocity_a_after - kick.velocity_a_before) == pytest.approx(
        kick.impulse_n_s, rel=0, abs=1e-12
    )
    assert config.mass_b_kg * (kick.velocity_b_after - kick.velocity_b_before) == pytest.approx(
        -kick.impulse_n_s, rel=0, abs=1e-12
    )
    # An equation check using the saved MIDPOINT factor, not its incoming value.
    assert kick.impulse_n_s == pytest.approx(
        duration * kick.sample.elastic.first_force_n[2] * kick.factor, rel=1e-12, abs=1e-12
    )
    assert kick.relative_mid_velocity_m_s == pytest.approx(
        (kick.velocity_a_before - kick.velocity_b_before
         + kick.velocity_a_after - kick.velocity_b_after) / 2, rel=0, abs=1e-12
    )


@pytest.mark.parametrize("masses", ((.5, 1.), (1., .5)))
@pytest.mark.parametrize("dissipation", (0., 20.))
def test_two_finite_masses_match_manual_mesh_driven_kick_drift_kick(masses, dissipation):
    config = AxialImpactConfig(mass_a_kg=masses[0], mass_b_kg=masses[1], dissipation_s_per_m=dissipation)
    entry = _state(config)
    initial = asdict(entry)
    dt = .001
    first = _manual_kick(config, entry, dt / 2)
    drifted = AxialPairState(entry.time_s + dt,
                             entry.position_a_m + dt * first["va"],
                             entry.position_b_m + dt * first["vb"], first["va"], first["vb"])
    second = _manual_kick(config, drifted, dt / 2)
    expected_exit = replace(drifted, velocity_a_m_s=second["va"], velocity_b_m_s=second["vb"])
    record = step_pair(config, entry, dt)
    _assert_state(record.entry, entry)
    _assert_state(record.drifted, drifted)
    _assert_state(record.exit, expected_exit)
    _assert_kick(config, entry, record.first, dt / 2, first)
    _assert_kick(config, drifted, record.second, dt / 2, second)
    assert record.exit.position_a_m == record.drifted.position_a_m
    assert record.exit.position_b_m == record.drifted.position_b_m
    assert record.impulse_n_s == pytest.approx(first["impulse"] + second["impulse"], rel=0, abs=1e-12)
    assert record.average_force_n == pytest.approx(record.impulse_n_s / dt, rel=1e-12, abs=1e-12)
    assert record.dissipated_j == pytest.approx(first["dissipation"] + second["dissipation"], rel=0, abs=1e-12)
    assert config.mass_a_kg * record.exit.velocity_a_m_s + config.mass_b_kg * record.exit.velocity_b_m_s == pytest.approx(
        config.mass_a_kg * entry.velocity_a_m_s + config.mass_b_kg * entry.velocity_b_m_s,
        rel=0, abs=1e-12,
    )
    assert record.exit.velocity_b_m_s < entry.velocity_b_m_s  # Target is not fixed.
    assert mechanical_energy_j(config, entry) == pytest.approx(_energy(config, entry), rel=1e-12, abs=1e-14)
    assert mechanical_energy_j(config, record.exit) == pytest.approx(_energy(config, expected_exit), rel=1e-12, abs=1e-14)
    if dissipation == 0:
        assert record.dissipated_j == 0.
    else:
        assert record.dissipated_j > 0.
        for kick in (record.first, record.second):
            assert kick.factor < kick.sample.instantaneous_factor
            assert kick.impulse_n_s < kick.duration_s * kick.sample.instantaneous_force_n
    assert asdict(entry) == initial


def test_mesh_pressure_integral_not_continuous_force_formula_drives_impulse():
    config = AxialImpactConfig()
    entry = _state(config, depth=.002, va=0., vb=0.)
    elastic = _static_integral(config, entry)
    depth = _depth(config, entry)
    analytic_force = pi * config.modulus_pa * depth**2 * (1 - 2 * depth / (3 * config.radius_m))
    mesh_force = elastic.first_force_n[2]
    # The inscribed polygon/triangle quadrature is not the exact continuous disk.
    assert not np.isclose(mesh_force, analytic_force, rtol=1e-10, atol=1e-12)
    record = step_pair(config, entry, .001)
    assert record.first.impulse_n_s == pytest.approx(.0005 * mesh_force, rel=0, abs=1e-12)
    assert abs(record.first.impulse_n_s - .0005 * analytic_force) > 1e-12
    assert mechanical_energy_j(config, entry) == pytest.approx(
        pi * config.modulus_pa * (depth**3 / 3 - depth**4 / (6 * config.radius_m)),
        rel=1e-12, abs=1e-14,
    )


def test_separated_pair_drifts_freely_with_signed_gap_and_constant_kinetic_energy():
    config = AxialImpactConfig(dissipation_s_per_m=20.)
    entry = AxialPairState(0., .027, 0., -.2, 0.)
    sample = sample_contact(config, entry)
    _assert_sample(config, entry, sample, _static_integral(config, entry))
    assert sample.indentation_m == pytest.approx(-.002, rel=0, abs=1e-15)
    assert sample.instantaneous_factor == pytest.approx(5.)
    record = step_pair(config, entry, .001)
    expected = AxialPairState(.001, .0268, 0., -.2, 0.)
    _assert_state(record.drifted, expected)
    _assert_state(record.exit, expected)
    assert record.impulse_n_s == record.average_force_n == record.dissipated_j == 0.
    assert mechanical_energy_j(config, entry) == pytest.approx(.01, rel=0, abs=1e-15)
    assert mechanical_energy_j(config, record.exit) == pytest.approx(.01, rel=0, abs=1e-15)
    touching = _state(config, depth=0., position_b=0., va=0., vb=0.)
    assert sample_contact(config, touching).elastic.area_m2 == 0.


def test_first_entry_has_only_second_kick_and_next_drift_moves_the_target():
    config = AxialImpactConfig()
    entry = _state(config, depth=-.00005, position_b=0., va=-.2, vb=0.)
    record = step_pair(config, entry, .001)
    assert record.first.sample.indentation_m < 0
    assert record.first.impulse_n_s == 0.
    assert record.second.sample.indentation_m == pytest.approx(.00015, rel=0, abs=1e-14)
    assert record.second.impulse_n_s > 0.
    assert record.drifted.position_a_m == pytest.approx(entry.position_a_m - .0002, abs=1e-14)
    assert record.drifted.position_b_m == record.exit.position_b_m == 0.
    assert record.exit.velocity_b_m_s < 0.
    assert record.impulse_n_s == record.second.impulse_n_s
    assert record.average_force_n == pytest.approx(record.second.impulse_n_s / .001, rel=1e-12)
    following = step_pair(config, record.exit, .001)
    assert following.drifted.position_b_m < 0.
    assert following.drifted.position_b_m == pytest.approx(
        record.exit.position_b_m + .001 * following.first.velocity_b_after, rel=0, abs=1e-12
    )


def test_departure_keeps_first_impulse_when_second_sample_is_empty():
    config = AxialImpactConfig()
    entry = _state(config, depth=.00005, position_b=0., va=.2, vb=0.)
    record = step_pair(config, entry, .001)
    assert record.first.impulse_n_s > 0.
    assert record.second.sample.indentation_m < 0.
    assert record.second.sample.instantaneous_force_n == record.second.impulse_n_s == 0.
    assert record.average_force_n > 0.
    assert record.average_force_n == pytest.approx(record.first.impulse_n_s / .001, rel=1e-12)
    assert record.exit.velocity_a_m_s == record.first.velocity_a_after
    assert record.exit.velocity_b_m_s == record.first.velocity_b_after


def test_hunt_crossley_clip_has_no_attraction_but_retains_elastic_load_and_kick_ledger():
    config = AxialImpactConfig(dissipation_s_per_m=20.)
    entry = _state(config, depth=.002, va=.1, vb=0.)
    record = step_pair(config, entry, .001)
    assert record.impulse_n_s == record.average_force_n == 0.
    assert record.exit.velocity_a_m_s == .1 and record.exit.velocity_b_m_s == 0.
    for kick in (record.first, record.second):
        assert kick.sample.elastic.pressure_load_n > 0.
        assert kick.sample.instantaneous_factor == kick.sample.instantaneous_force_n == 0.
        assert kick.factor == kick.impulse_n_s == 0.
        assert kick.relative_mid_velocity_m_s == .1
        assert kick.dissipated_j == pytest.approx(
            kick.duration_s * kick.sample.elastic.first_force_n[2] * .1, rel=1e-12, abs=1e-14
        )
    assert record.dissipated_j > 0.
    at_threshold = _state(config, depth=.002, va=.05, vb=0.)
    assert sample_contact(config, at_threshold).instantaneous_force_n == 0.


def test_unclipped_unloading_kick_reduces_repulsion_and_has_positive_dissipation():
    config = AxialImpactConfig(dissipation_s_per_m=20.)
    entry = _state(config, depth=.002, va=.02, vb=0.)
    expected = _manual_kick(config, entry, .0005)
    record = step_pair(config, entry, .001)
    _assert_kick(config, entry, record.first, .0005, expected)
    assert 0 < record.first.factor < 1
    assert 0 < record.first.impulse_n_s < .0005 * record.first.sample.elastic.first_force_n[2]
    assert record.first.dissipated_j > 0

def test_elastic_equal_mass_symmetry_and_single_step_time_reversal():
    config = AxialImpactConfig(mass_a_kg=.5, mass_b_kg=.5)
    entry = _state(config, depth=.0015, va=-.1, vb=.1)
    record = step_pair(config, entry, .0005)
    assert record.dissipated_j == 0.
    assert record.exit.velocity_a_m_s + record.exit.velocity_b_m_s == pytest.approx(0., abs=1e-14)
    assert record.exit.position_a_m + record.exit.position_b_m == pytest.approx(
        entry.position_a_m + entry.position_b_m, rel=0, abs=1e-14
    )
    backwards = replace(record.exit, velocity_a_m_s=-record.exit.velocity_a_m_s,
                        velocity_b_m_s=-record.exit.velocity_b_m_s)
    reversed_record = step_pair(config, backwards, .0005)
    expected = replace(entry, time_s=entry.time_s + .001,
                       velocity_a_m_s=-entry.velocity_a_m_s, velocity_b_m_s=-entry.velocity_b_m_s)
    _assert_state(reversed_record.exit, expected)
    assert mechanical_energy_j(config, reversed_record.exit) == pytest.approx(
        _energy(config, entry), rel=1e-12, abs=1e-14
    )


def test_common_position_translation_and_velocity_boost_preserve_relative_dynamics():
    config = AxialImpactConfig(dissipation_s_per_m=20.)
    entry = _state(config, depth=.0012)
    shift, boost, dt = .123, .37, .001
    changed = replace(entry, position_a_m=entry.position_a_m + shift,
                       position_b_m=entry.position_b_m + shift,
                       velocity_a_m_s=entry.velocity_a_m_s + boost,
                       velocity_b_m_s=entry.velocity_b_m_s + boost)
    original, transformed = step_pair(config, entry, dt), step_pair(config, changed, dt)
    assert transformed.impulse_n_s == pytest.approx(original.impulse_n_s, rel=1e-10, abs=1e-12)
    assert transformed.dissipated_j == pytest.approx(original.dissipated_j, rel=1e-10, abs=1e-12)
    for field in ("position_a_m", "position_b_m"):
        assert getattr(transformed.exit, field) == pytest.approx(
            getattr(original.exit, field) + shift + boost * dt, rel=0, abs=1e-12
        )
    for field in ("velocity_a_m_s", "velocity_b_m_s"):
        assert getattr(transformed.exit, field) == pytest.approx(getattr(original.exit, field) + boost,
                                                                rel=0, abs=1e-12)
    for before, after, plane_shift in ((original.first, transformed.first, shift),
                                        (original.second, transformed.second, shift + boost * dt)):
        assert after.sample.indentation_m == pytest.approx(before.sample.indentation_m, rel=0, abs=1e-14)
        np.testing.assert_allclose(np.asarray(after.sample.elastic.quadrature.points_world),
                                   np.asarray(before.sample.elastic.quadrature.points_world) + (0., 0., plane_shift),
                                   rtol=0, atol=1e-12)
    momentum = config.mass_a_kg * entry.velocity_a_m_s + config.mass_b_kg * entry.velocity_b_m_s
    energy_shift = boost * momentum + (config.mass_a_kg + config.mass_b_kg) * boost**2 / 2
    assert mechanical_energy_j(config, changed) - mechanical_energy_j(config, entry) == pytest.approx(
        energy_shift, rel=1e-12, abs=1e-14
    )


def test_second_kick_domain_failure_does_not_publish_or_mutate_entry():
    config = AxialImpactConfig()
    entry = _state(config, depth=.0059, va=-20., vb=0.)
    before_config, before_state = asdict(config), asdict(entry)
    # Entry itself is supported and below 6mm; the deliberately fast drift is not.
    assert sample_contact(config, entry).elastic.pressure_load_n > 0.
    with pytest.raises(ValueError):
        step_pair(config, entry, .001)
    assert asdict(config) == before_config and asdict(entry) == before_state


def test_positive_contact_requires_full_finite_support_depth_and_mesh_budget():
    narrow = AxialImpactConfig(target_half_width_m=.001)
    separated = _state(narrow, depth=-.002)
    assert sample_contact(narrow, separated).elastic.area_m2 == 0.
    with pytest.raises(ValueError):
        sample_contact(narrow, _state(narrow, depth=.001))
    config = AxialImpactConfig()
    for depth in (.006001, .011):
        with pytest.raises(ValueError):
            sample_contact(config, _state(config, depth=depth))
    # This is a resource-rejection test, not an additional spatial convergence run.
    with pytest.raises(ValueError):
        tiny_cells = AxialImpactConfig(resolution_m=1e-9)
        sample_contact(tiny_cells, _state(tiny_cells, depth=.001))


def test_invalid_constructor_inputs_and_step_arguments_are_rejected():
    for options in (
        {"mass_a_kg": 0.}, {"mass_b_kg": -1.}, {"mass_a_kg": np.inf}, {"mass_b_kg": True},
        {"radius_m": 0.}, {"target_half_width_m": 0.}, {"target_thickness_m": -1.},
        {"modulus_pa": 0.}, {"modulus_pa": True}, {"resolution_m": np.nan},
        {"resolution_m": 0.}, {"dissipation_s_per_m": -1.}, {"dissipation_s_per_m": False},
        {"max_indentation_m": 0.}, {"max_indentation_m": .011},
        {"radius_m": .005, "max_indentation_m": .006},
    ):
        with pytest.raises(ValueError):
            AxialImpactConfig(**options)
    valid_state = dict(time_s=0., position_a_m=.027, position_b_m=0., velocity_a_m_s=-.2, velocity_b_m_s=0.)
    for field in STATE_FIELDS:
        for invalid in (True, np.nan, np.inf):
            with pytest.raises(ValueError):
                AxialPairState(**{**valid_state, field: invalid})
    with pytest.raises(ValueError):
        AxialPairState(**{**valid_state, "time_s": -1.})
    config, entry = AxialImpactConfig(), AxialPairState(**valid_state)
    before = asdict(entry)
    for dt in (0., -.001, .010001, np.inf, np.nan, True):
        with pytest.raises(ValueError):
            step_pair(config, entry, dt)
    assert asdict(entry) == before
    # A finite timestamp whose next increment rounds away must not publish a step.
    stalled_clock = replace(entry, time_s=1e308)
    with pytest.raises(ValueError):
        step_pair(config, stalled_clock, .001)
    assert stalled_clock.time_s == 1e308


def test_finite_inputs_with_nonfinite_derived_state_load_or_energy_are_rejected():
    config = AxialImpactConfig()
    huge_relative = AxialPairState(0., .027, 0., 1e308, -1e308)
    with pytest.raises(ValueError):
        sample_contact(config, huge_relative)
    huge_gap = AxialPairState(0., -1e308, 1e308, 0., 0.)
    with pytest.raises(ValueError):
        sample_contact(config, huge_gap)
    huge_energy = AxialPairState(0., .027, 0., 1e308, 0.)
    with pytest.raises(ValueError):
        mechanical_energy_j(config, huge_energy)


def test_frozen_complete_json_owned_results_and_no_engine_import(monkeypatch):
    config, entry = AxialImpactConfig(dissipation_s_per_m=20.), _state(AxialImpactConfig())
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in ("phymeo", "newton", "warp", "pydrake", "tyx", "mujoco"):
            raise AssertionError("axial NumPy reference must not import a product physics engine")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", guard)
        record = step_pair(config, entry, .001)
    for owner, field in ((config, "mass_a_kg"), (entry, "time_s"), (record, "impulse_n_s"),
                          (record.first, "factor"), (record.first.sample, "indentation_m")):
        with pytest.raises(FrozenInstanceError):
            setattr(owner, field, 99.)
    with pytest.raises(TypeError):
        record.first.sample.elastic.quadrature.points_world[0][0] = 99.
    payload = record.to_dict()
    assert set(payload) == {"entry", "drifted", "exit", "first", "second", "impulse_n_s",
                            "average_force_n", "dissipated_j"}
    for name in ("entry", "drifted", "exit"):
        assert set(payload[name]) == set(STATE_FIELDS)
    for name in ("first", "second"):
        assert set(payload[name]) == {"sample", "duration_s", "velocity_a_before", "velocity_b_before",
                                     "velocity_a_after", "velocity_b_after", "relative_mid_velocity_m_s",
                                     "factor", "impulse_n_s", "dissipated_j"}
        assert set(payload[name]["sample"]) == {"indentation_m", "relative_velocity_m_s", "elastic",
                                                "instantaneous_factor", "instantaneous_force_n"}
        assert payload[name]["sample"]["elastic"]["pressure_pa"]
    decoded = json.loads(json.dumps(payload, allow_nan=False))
    assert decoded["entry"]["time_s"] == entry.time_s
    assert decoded["first"]["sample"]["elastic"]["qualification"] == "NUMPY_STATIC_REFERENCE"
    payload["entry"]["velocity_a_m_s"] = 99.
    payload["first"]["sample"]["elastic"]["pressure_pa"] = ()
    assert record.entry == entry
    assert len(record.first.sample.elastic.pressure_pa) > 0
