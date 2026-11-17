"""Independent scalar, rational and manufactured oracles for frozen-J steps."""

import json
import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import numpy as np
import pytest

from experiments.area_pressure.coupled_friction_reference import (
    CoupledFrictionFailure,
    CoupledFrictionProblem,
    CoupledFrictionSample,
    fixed_problems,
    solve,
    write_reference,
)


def _problems():
    return dict(fixed_problems())


def _bisect(function, lo, hi):
    # Independent 100-step scalar oracle, not the production Newton iteration.
    assert function(lo) <= 0 <= function(hi)
    for _ in range(100):
        mid = (lo + hi) / 2
        if function(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _residual(problem, velocity):
    a = problem.mass_matrix + problem.h_s * np.diag(problem.damping)
    rhs = problem.mass_matrix @ problem.input_velocity + problem.h_s * problem.applied_force
    force = np.zeros(len(velocity))
    for sample in problem.samples:
        g = sample.jacobian_a - sample.jacobian_b
        n = sample.normal_world
        relative = g @ velocity
        normal_speed = n @ relative
        tangent = relative - n * normal_speed
        normal_force = max(0.0, -sample.stiffness_n_m * (sample.phi_m + problem.h_s * normal_speed))
        vector = normal_force * n - sample.mu * normal_force * tangent / math.sqrt(
            float(tangent @ tangent) + sample.slip_speed_m_s**2
        )
        force += g.T @ vector
    return a @ velocity - rhs - problem.h_s * force


def test_support_and_release_match_closed_normal_and_independent_scalar_root():
    for name in ("support", "release"):
        result = solve(_problems()[name])
        if name == "support":
            expected_tangent = _bisect(
                lambda v: v - 0.02 + 0.004 * v / math.sqrt(v * v + 1e-6), 0.016, 0.02
            )
            expected = (expected_tangent, 0.0)
            expected_force = 4.0
        else:
            expected = (0.02, 0.3)
            expected_force = 0.0
        np.testing.assert_allclose(result.velocity, expected, rtol=0, atol=2e-9)
        assert result.samples[0].normal_force_n == pytest.approx(expected_force, rel=0, abs=2e-8)
        if name == "release":
            np.testing.assert_array_equal(result.samples[0].tangent_force_world, np.zeros(3))


def test_shared_scalar_normal_tangent_coupling_matches_bracket_oracle():
    problem = _problems()["shared-one-dof"]

    def equation(v):
        fn = 4 - 4 * v
        return 0.5 * (v - 0.02) - 0.001 * fn * (1 - 0.5 * v / math.sqrt(v * v + 1e-6))

    expected = _bisect(equation, 0.02, 0.028)
    result = solve(problem)
    assert result.velocity[0] == pytest.approx(expected, rel=0, abs=2e-9)
    assert result.samples[0].normal_force_n == pytest.approx(4 - 4 * expected, rel=0, abs=2e-8)
    # Separately solving the pure normal coordinate and then subtracting a
    # friction impulse cannot satisfy this actual shared scalar equation.
    normal_only = (0.5 * 0.02 + 0.001 * 4) / (0.5 + 0.004)
    split = normal_only - 0.001 * 0.5 * (4 - 4 * normal_only) * normal_only / (
        0.5 * math.sqrt(normal_only**2 + 1e-6)
    )
    assert abs(equation(split)) > 1e-6


def test_manufactured_solution_has_final_normal_loads_and_nonsymmetric_jacobian():
    problem = _problems()["two-point-manufactured"]
    # Hand-built force: vt=.012, vn=(.004,-.008), fn=(1.992,2.424).
    qx = -float(F(270144, 10000)) / math.sqrt(145)
    expected_force = np.array((-2.485 - qx, -0.42, 0.04894 + 0.01 * qx))
    np.testing.assert_allclose(problem.applied_force, expected_force, rtol=0, atol=2e-15)
    result = solve(problem)
    np.testing.assert_allclose(result.velocity, (0.015, -0.002, 0.3), rtol=0, atol=2e-8)
    np.testing.assert_allclose(
        [s.normal_force_n for s in result.samples], (1.992, 2.424), rtol=0, atol=2e-7
    )
    np.testing.assert_allclose(
        [s.normal_velocity_m_s for s in result.samples], (0.004, -0.008), rtol=0, atol=2e-9
    )
    for sample in result.samples:
        np.testing.assert_allclose(sample.tangent_velocity_world, (0.012, 0, 0), rtol=0, atol=2e-9)
    expected_jacobian = np.empty((3, 3))
    for column in range(3):
        delta = np.zeros(3)
        delta[column] = 2e-7
        expected_jacobian[:, column] = (
            _residual(problem, result.velocity + delta)
            - _residual(problem, result.velocity - delta)
        ) / (4e-7)
    np.testing.assert_allclose(result.residual_jacobian, expected_jacobian, rtol=1e-6, atol=1e-8)
    assert np.max(np.abs(result.residual_jacobian - result.residual_jacobian.T)) > 1e-6


def test_frictionless_two_point_matches_rational_two_by_two_system_and_empty_is_free():
    # Jn rows=(0,1,+.02),(0,1,-.02), M^-1=diag(2,2,2500).
    a, b, c = F(1, 2000) + F(3, 10**6), F(1, 10**6), F(1, 3000) + F(3, 10**6)
    r0, r1 = F(1096, 10**6), F(904, 10**6)
    force0 = (c * r0 - b * r1) / (a * c - b * b)
    force1 = (a * r1 - b * r0) / (a * c - b * b)
    assert force0 > 0 and force1 > 0
    result = solve(_problems()["two-point-frictionless"])
    np.testing.assert_allclose(
        [s.normal_force_n for s in result.samples],
        (float(force0), float(force1)),
        rtol=0,
        atol=2e-8,
    )
    expected = (0, -0.1 + 0.002 * float(force0 + force1), 0.2 + 0.05 * float(force0 - force1))
    np.testing.assert_allclose(result.velocity, expected, rtol=0, atol=2e-9)
    empty = solve(_problems()["empty"])
    np.testing.assert_array_equal(empty.velocity, empty.free_velocity)
    np.testing.assert_allclose(empty.velocity, (0.02, -0.008), rtol=0, atol=1e-17)
    assert empty.samples == () and empty.iterations == 0


def test_all_six_cases_obey_full_momentum_final_coulomb_power_and_energy():
    for _, problem in fixed_problems():
        result = solve(problem)
        h, v = problem.h_s, result.velocity
        contact = np.zeros(len(v))
        normal_work = tangent_work = 0.0
        spring0 = spring1 = 0.0
        for source, actual in zip(problem.samples, result.samples, strict=True):
            g = source.jacobian_a - source.jacobian_b
            contact += g.T @ actual.force_on_a_world
            np.testing.assert_array_equal(actual.force_on_b_world, -actual.force_on_a_world)
            np.testing.assert_allclose(
                actual.moment_on_a_world,
                np.cross(source.point_world, actual.force_on_a_world),
                rtol=0,
                atol=1e-14,
            )
            np.testing.assert_array_equal(actual.moment_on_b_world, -actual.moment_on_a_world)
            assert (
                np.linalg.norm(actual.tangent_force_world)
                <= source.mu * actual.normal_force_n + 1e-12
            )
            assert actual.tangent_power_w <= 1e-14
            assert abs(source.normal_world @ actual.tangent_force_world) <= 1e-13
            np.testing.assert_allclose(
                actual.force_on_a_world @ (g @ v),
                actual.normal_power_w + actual.tangent_power_w,
                rtol=0,
                atol=1e-13,
            )
            normal_work += h * actual.normal_power_w
            tangent_work += h * actual.tangent_power_w
            spring0 += 0.5 * source.stiffness_n_m * source.phi_m**2
            spring1 += 0.5 * source.stiffness_n_m * min(actual.linear_gap_m, 0) ** 2
        np.testing.assert_allclose(result.generalized_contact_force, contact, rtol=0, atol=1e-13)
        residual = (
            (problem.mass_matrix + h * np.diag(problem.damping)) @ v
            - problem.mass_matrix @ problem.input_velocity
            - h * (problem.applied_force + contact)
        )
        np.testing.assert_allclose(result.residual, residual, rtol=0, atol=1e-15)
        assert result.residual_norm <= result.tolerance
        energy = result.energy
        kinetic0 = 0.5 * problem.input_velocity @ problem.mass_matrix @ problem.input_velocity
        kinetic1 = 0.5 * v @ problem.mass_matrix @ v
        external_work = h * problem.applied_force @ v
        damping_loss = h * np.dot(problem.damping * v, v)
        dv = v - problem.input_velocity
        numerical_loss = 0.5 * dv @ problem.mass_matrix @ dv
        assert energy.kinetic_entry_j == pytest.approx(kinetic0, rel=0, abs=1e-14)
        assert energy.kinetic_exit_j == pytest.approx(kinetic1, rel=0, abs=1e-14)
        assert energy.spring_entry_j == pytest.approx(spring0, rel=0, abs=1e-14)
        assert energy.spring_exit_j == pytest.approx(spring1, rel=0, abs=1e-14)
        assert energy.external_work_j == pytest.approx(external_work, rel=0, abs=1e-14)
        assert energy.damping_loss_j == pytest.approx(damping_loss, rel=0, abs=1e-14)
        assert energy.normal_work_j == pytest.approx(normal_work, rel=0, abs=1e-14)
        assert energy.tangent_work_j == pytest.approx(tangent_work, rel=0, abs=1e-14)
        assert energy.kinetic_numerical_loss_j == pytest.approx(numerical_loss, rel=0, abs=1e-14)
        expected_balance = (
            kinetic1
            - kinetic0
            - external_work
            + damping_loss
            - normal_work
            - tangent_work
            + numerical_loss
        )
        assert energy.kinetic_balance_residual_j == pytest.approx(
            expected_balance, rel=0, abs=1e-14
        )
        assert abs(expected_balance) <= 2e-9
        assert energy.normal_spring_balance_j <= 1e-12


def test_permutation_spatial_covariance_and_both_moving_sides():
    problem = _problems()["two-point-manufactured"]
    base = solve(problem)
    reverse = solve(replace(problem, samples=problem.samples[::-1]))
    np.testing.assert_allclose(reverse.velocity, base.velocity, rtol=0, atol=2e-9)
    rotation = np.array(((0, 0, 1), (0, 1, 0), (-1, 0, 0)), dtype=float)
    translation = np.array((0.03, -0.02, 0.01))
    moved = tuple(
        replace(
            s,
            point_world=rotation @ s.point_world + translation,
            normal_world=rotation @ s.normal_world,
            jacobian_a=rotation @ s.jacobian_a,
            jacobian_b=rotation @ s.jacobian_b,
        )
        for s in problem.samples
    )
    transformed = solve(replace(problem, samples=moved))
    np.testing.assert_allclose(transformed.velocity, base.velocity, rtol=0, atol=2e-9)
    for a, b in zip(base.samples, transformed.samples, strict=True):
        np.testing.assert_allclose(
            b.force_on_a_world, rotation @ a.force_on_a_world, rtol=0, atol=2e-8
        )
        np.testing.assert_allclose(
            b.moment_on_a_world,
            rotation @ a.moment_on_a_world + np.cross(translation, rotation @ a.force_on_a_world),
            rtol=0,
            atol=2e-9,
        )
    # Both side Jacobians are nonzero, with exactly the same physical difference.
    moving = tuple(
        replace(s, jacobian_a=2 * s.jacobian_a, jacobian_b=s.jacobian_a) for s in problem.samples
    )
    both = solve(replace(problem, samples=moving))
    np.testing.assert_allclose(both.velocity, base.velocity, rtol=0, atol=2e-9)
    # A change of generalized coordinates produces a genuinely full SPD mass
    # matrix without introducing unsupported off-diagonal joint damping.
    frictionless = _problems()["two-point-frictionless"]
    basis = np.array(((1, 1, 0), (0, 1, 0), (0, 0, 1)), dtype=float)
    changed = replace(
        frictionless,
        mass_matrix=basis.T @ frictionless.mass_matrix @ basis,
        input_velocity=np.linalg.solve(basis, frictionless.input_velocity),
        applied_force=basis.T @ frictionless.applied_force,
        samples=tuple(
            replace(s, jacobian_a=s.jacobian_a @ basis, jacobian_b=s.jacobian_b @ basis)
            for s in frictionless.samples
        ),
    )
    np.testing.assert_allclose(
        basis @ solve(changed).velocity, solve(frictionless).velocity, rtol=0, atol=2e-9
    )


def test_near_unit_normal_and_pure_normal_motion_do_not_create_friction():
    sample = CoupledFrictionSample(
        (0, 0, 0),
        (0, 0, 1 + 5e-13),
        ((0,), (0,), (1,)),
        ((0,), (0,), (0,)),
        4000,
        -0.001,
        0.5,
        0.001,
    )
    problem = CoupledFrictionProblem(((0.5,),), (0,), (-0.1,), (0,), 0.001, (sample,))
    result = solve(problem)
    np.testing.assert_array_equal(problem.samples[0].normal_world, (0, 0, 1))
    np.testing.assert_array_equal(result.samples[0].tangent_velocity_world, (0, 0, 0))
    np.testing.assert_array_equal(result.samples[0].tangent_force_world, (0, 0, 0))


def test_owned_shapes_capacity_finite_and_positive_domain():
    original = _problems()["support"]
    mass = original.mass_matrix.copy()
    problem = replace(original, mass_matrix=mass)
    mass[0, 0] = 9
    assert problem.mass_matrix[0, 0] == 0.5
    with pytest.raises(ValueError):
        problem.mass_matrix.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        problem.h_s = 0.1
    bad = (
        dict(mass_matrix=((0.5, 0.1), (0, 0.5))),
        dict(mass_matrix=((0.5, 0), (0, 0))),
        dict(mass_matrix=((float("nan"), 0), (0, 0.5))),
        dict(damping=(-1, 0)),
        dict(input_velocity=(True, 0)),
        dict(h_s=0),
        dict(h_s=float("inf")),
        dict(samples=original.samples * 9),
        dict(mass_matrix=np.eye(7)),
    )
    for kwargs in bad:
        with pytest.raises(ValueError):
            replace(original, **kwargs)
    for kwargs in (
        dict(normal_world=(0, 0, 1.01)),
        dict(stiffness_n_m=0),
        dict(phi_m=0.1),
        dict(mu=-0.1),
        dict(slip_speed_m_s=0),
        dict(jacobian_b=((0,), (0,), (0,))),
        dict(point_world=(0, float("inf"), 0)),
    ):
        with pytest.raises(ValueError):
            replace(original.samples[0], **kwargs)
    damaged = replace(original)
    object.__setattr__(damaged, "mass_matrix", np.zeros((2, 2)))
    with pytest.raises(ValueError):
        solve(damaged)


def test_budget_failure_has_owned_attempt_diagnostics_and_no_partial_result():
    problem = _problems()["support"]
    before = problem.input_velocity.tobytes()
    with pytest.raises(CoupledFrictionFailure) as info:
        solve(problem, max_iterations=0)
    failure = info.value
    assert failure.attempts and failure.attempts[-1].residual_norm > failure.tolerance
    assert not hasattr(failure, "result")
    assert problem.input_velocity.tobytes() == before
    with pytest.raises(CoupledFrictionFailure):
        solve(problem, max_backtracks=0)
    for kwargs in (dict(max_iterations=25), dict(max_iterations=True), dict(max_backtracks=13)):
        with pytest.raises(ValueError):
            solve(problem, **kwargs)


def test_writer_saves_all_six_frozen_problems_and_actual_attempts(tmp_path):
    output = tmp_path / "coupled"
    write_reference(output)
    payload = json.loads((output / "reference.json").read_text(encoding="utf-8"))
    assert payload["status"] == "complete" and len(payload["records"]) == 6
    assert payload["scope"] == "frozen-jacobian-shared-dof-host-reference"
    assert payload["is_tyx_runtime"] is False
    assert [r["case"] for r in payload["records"]] == [name for name, _ in fixed_problems()]
    for record in payload["records"]:
        assert record["result"]["attempts"]
        assert record["result"]["residual_norm"] <= record["result"]["tolerance"]
        assert "mass_matrix" in record["result"]["problem"]
        assert "energy" in record["result"]
    with pytest.raises(ValueError):
        write_reference(output)
