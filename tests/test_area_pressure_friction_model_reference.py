"""Independent closed cases for a fixed, decoupled two-DOF model comparison."""

import json
import math
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

from experiments.area_pressure.friction_model_reference import (
    FrictionInput,
    compare_models,
    solve_friction_case,
    write_comparison,
)

LAWS = ("coulomb-final-normal", "smooth-final-normal", "smooth-lagged-input-normal")
CASES = ("support-slide", "support-start", "impact", "release")
STEPS = (0.001, 0.0005)


def _record(case, h, law):
    return solve_friction_case(FrictionInput(case, h), law)


@pytest.mark.parametrize(
    "h,impact_force", [(0.001, Fraction(50, 21)), (0.0005, Fraction(700, 501))]
)
def test_exact_coulomb_matches_independent_closed_vectors(h, impact_force):
    # These forces are hand-reduced from the declared normal scalar system.
    expected = {
        "support-slide": (4.0, -2.0, 0.0, 0.02 - 4 * h),
        "support-start": (4.0, -1.0, 0.0, 0.0),
        "impact": (
            float(impact_force),
            -float(impact_force) / 2,
            -0.5 + 2 * h * float(impact_force),
            0.02 - h * float(impact_force),
        ),
        "release": (0.0, 0.0, 0.3, 0.02),
    }
    for case, (normal, tangent, vn, vt) in expected.items():
        row = _record(case, h, LAWS[0])
        assert row.normal_force_n == pytest.approx(normal, rel=0.0, abs=2e-14)
        assert row.tangent_force_n == pytest.approx(tangent, rel=0.0, abs=2e-14)
        assert row.normal_velocity_m_s == pytest.approx(vn, rel=0.0, abs=2e-15)
        assert row.tangent_velocity_m_s == pytest.approx(vt, rel=0.0, abs=2e-15)
        assert row.iterations == 0
        assert row.instantaneous_coulomb_excess_n <= 1e-14


@pytest.mark.parametrize("h", STEPS)
def test_smooth_roots_satisfy_independent_equation_and_hand_intervals(h):
    for case in CASES:
        for law in LAWS[1:]:
            row = _record(case, h, law)
            sample = row.input
            u = sample.tangent_velocity_m_s + h * sample.external_tangent_force_n / 0.5
            normal = row.normal_force_n if law == LAWS[1] else max(0.0, -4000 * sample.gap_m)
            speed = row.tangent_velocity_m_s
            force = -0.5 * normal * speed / math.sqrt(speed * speed + 0.001**2)
            # Substitute into the equation, never call a second producer as oracle.
            residual = speed - u - h * force / 0.5
            assert abs(residual) <= 1e-13
            assert row.solver_residual_m_s == pytest.approx(abs(residual), rel=0.0, abs=1e-17)
            assert row.tangent_force_n == pytest.approx(force, rel=0.0, abs=1e-14)
            assert max(0.0, u - h * 0.5 * normal / 0.5) <= speed <= u
            assert 0 <= row.iterations <= 80
            lo, hi = row.final_bracket_m_s
            assert 0 <= lo <= speed <= hi <= u
            assert row.tangent_power_w <= 1e-14


@pytest.mark.parametrize("h", STEPS)
def test_model_distinctions_are_exposed_without_calling_them_product_failures(h):
    exact = _record("support-start", h, LAWS[0])
    smooth = _record("support-start", h, LAWS[1])
    lagged = _record("support-start", h, LAWS[2])
    assert exact.tangent_velocity_m_s == 0.0
    assert exact.tangent_force_n == -1.0
    assert 0 < smooth.tangent_velocity_m_s < 2 * h
    assert smooth.tangent_velocity_m_s == pytest.approx(
        lagged.tangent_velocity_m_s, rel=0.0, abs=1e-13
    )
    assert smooth.tangent_force_n == pytest.approx(lagged.tangent_force_n, rel=0.0, abs=1e-12)

    exact = _record("release", h, LAWS[0])
    smooth = _record("release", h, LAWS[1])
    lagged = _record("release", h, LAWS[2])
    assert exact.normal_force_n == smooth.normal_force_n == lagged.normal_force_n == 0.0
    assert exact.tangent_force_n == smooth.tangent_force_n == 0.0
    assert lagged.normal_input_force_n == pytest.approx(0.4, rel=0.0, abs=1e-15)
    assert abs(lagged.tangent_force_n) > 0.19
    assert lagged.instantaneous_coulomb_excess_n > 0.19
    assert lagged.tangent_power_w < 0.0
    assert lagged.kinetic_tangent_exit_j < exact.kinetic_tangent_exit_j
    assert lagged.gap_m > 0.0
    assert lagged.spring_exit_j == 0.0

    exact = _record("impact", h, LAWS[0])
    lagged = _record("impact", h, LAWS[2])
    assert exact.normal_force_n > 1.0
    assert abs(exact.tangent_force_n) > 0.5
    assert abs(lagged.tangent_force_n) <= 0.2 + 1e-15
    assert lagged.tangent_velocity_m_s > exact.tangent_velocity_m_s


def test_all_24_rows_have_same_step_momentum_and_backward_euler_work():
    rows = compare_models()
    assert isinstance(rows, tuple) and len(rows) == 24
    assert [(r.input.case, r.input.h_s, r.law) for r in rows] == [
        (case, h, law) for case in CASES for h in STEPS for law in LAWS
    ]
    for row in rows:
        a = row.input
        h, mass = a.h_s, 0.5
        vn, vt = row.normal_velocity_m_s, row.tangent_velocity_m_s
        dn, dt = vn - a.normal_velocity_m_s, vt - a.tangent_velocity_m_s
        assert mass * dn == pytest.approx(
            h * (a.external_normal_force_n + row.normal_force_n), rel=0.0, abs=1e-13
        )
        assert mass * dt == pytest.approx(
            h * (a.external_tangent_force_n + row.tangent_force_n), rel=0.0, abs=1e-13
        )
        assert row.gap_m == pytest.approx(a.gap_m + h * vn, rel=0.0, abs=1e-17)
        assert row.tangent_displacement_m == pytest.approx(h * vt, rel=0.0, abs=1e-17)
        assert row.normal_impulse_ns == h * row.normal_force_n
        assert row.tangent_impulse_ns == h * row.tangent_force_n
        assert row.external_work_j == pytest.approx(
            h * (a.external_normal_force_n * vn + a.external_tangent_force_n * vt),
            rel=0.0,
            abs=1e-17,
        )
        assert row.normal_work_j == pytest.approx(h * row.normal_force_n * vn, rel=0.0, abs=1e-17)
        assert row.tangent_work_j == pytest.approx(h * row.tangent_force_n * vt, rel=0.0, abs=1e-17)
        k0 = 0.5 * mass * (a.normal_velocity_m_s**2 + a.tangent_velocity_m_s**2)
        k1 = 0.5 * mass * (vn**2 + vt**2)
        u0 = 2000 * min(a.gap_m, 0.0) ** 2
        u1 = 2000 * min(row.gap_m, 0.0) ** 2
        loss = 0.5 * mass * (dn**2 + dt**2)
        assert row.kinetic_entry_j == pytest.approx(k0, rel=0.0, abs=1e-17)
        assert row.kinetic_exit_j == pytest.approx(k1, rel=0.0, abs=1e-17)
        assert row.spring_entry_j == pytest.approx(u0, rel=0.0, abs=1e-17)
        assert row.spring_exit_j == pytest.approx(u1, rel=0.0, abs=1e-17)
        assert row.be_kinetic_numerical_loss_j == pytest.approx(loss, rel=0.0, abs=1e-17)
        assert k1 - k0 + loss == pytest.approx(
            row.external_work_j + row.normal_work_j + row.tangent_work_j, rel=0.0, abs=2e-14
        )
        assert row.normal_spring_balance_j <= 1e-14
        assert row.mechanical_balance_j <= 1e-14
        assert row.tangent_force_n * vt <= 1e-14
        assert abs(row.tangent_force_n) <= 0.5 * row.friction_scale_force_n + 1e-14
        assert row.normal_law_residual_n <= 1e-12


def test_inputs_results_are_frozen_and_bad_or_mutated_inputs_are_rejected():
    sample = FrictionInput("support-slide", 0.001)
    result = solve_friction_case(sample, LAWS[0])
    assert result.input == sample and result.input is not sample
    with pytest.raises(FrozenInstanceError):
        sample.h_s = 0.0005
    with pytest.raises(FrozenInstanceError):
        result.normal_force_n = 0.0
    for case, h in [
        ("unknown", 0.001),
        (True, 0.001),
        ("impact", True),
        ("impact", float("nan")),
        ("impact", float("inf")),
        ("impact", 0.002),
    ]:
        with pytest.raises(ValueError):
            FrictionInput(case, h)
    for law in ("coulomb", None, True, 3):
        with pytest.raises(ValueError):
            solve_friction_case(sample, law)
    damaged = replace(sample)
    object.__setattr__(damaged, "mu", float("nan"))
    with pytest.raises(ValueError):
        solve_friction_case(damaged, LAWS[0])
    with pytest.raises(ValueError):
        solve_friction_case({"case": "impact", "h_s": 0.001}, LAWS[0])


def test_writer_preserves_all_rows_scope_sources_and_refuses_overwrite(tmp_path):
    output = tmp_path / "comparison"
    write_comparison(output)
    payload = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert payload["schema"] == "phymeo-friction-model-reference-v1"
    assert len(payload["records"]) == 24
    assert payload["scope"] == "decoupled-2dof-independent-reference"
    assert payload["is_tyx_runtime"] is False
    assert payload["is_general_contact_solver"] is False
    assert len(payload["source_sha256"]) == 64
    assert "https://arxiv.org/html/2312.03908v3" in payload["research_sources"]
    text = (output / "comparison.md").read_text(encoding="utf-8")
    assert "Lagged" in text and "Coulomb" in text and "24" in text
    assert "NOT TYX" in text and "分离" in text
    with pytest.raises(ValueError):
        write_comparison(output)
    with pytest.raises(ValueError):
        write_comparison("relative-output")


def test_unconverged_bisection_stops_and_writer_keeps_the_accepted_prefix(monkeypatch, tmp_path):
    import experiments.area_pressure.friction_model_reference as module

    monkeypatch.setattr(module, "MAX_BISECTION_STEPS", 0)
    with pytest.raises(RuntimeError, match="residual|bisection"):
        solve_friction_case(FrictionInput("impact", 0.001), LAWS[1])
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        write_comparison(output)
    payload = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert len(payload["records"]) == 1  # exact first row; next smooth solve must stop
    assert payload["records"][0]["law"] == LAWS[0]
    assert payload["failure"]["case"] == "support-slide"
    assert payload["failure"]["law"] == LAWS[1]
    assert not (output / "comparison.md").exists()
