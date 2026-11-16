"""HOST-only oracles and export checks; these tests never solve an FFT contact."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.contact_selection.fft_reference import (
    DELTAS_M,
    FORCETOL_N,
    GRIDS,
    MAXITER,
    PENTOL_M,
    THERMOTOL,
    assess_fields,
    extract_fields,
    frozen_config,
    hertz_fields,
    hertz_reference,
    physical_grid,
    preview_fields,
)


@pytest.mark.parametrize(
    "delta,a,force",
    [
        (0.00005, 0.001, 1 / 15),
        (0.0001, np.sqrt(2) / 1000, 2 * np.sqrt(2) / 15),
        (0.0002, 0.002, 8 / 15),
    ],
)
def test_hertz_fixed_cases_have_independent_si_values(delta, a, force):
    actual = hertz_reference(delta)
    assert actual["radius_m"] == pytest.approx(a, rel=1e-14)
    assert actual["force_n"] == pytest.approx(force, rel=1e-14)
    assert actual["area_m2"] == pytest.approx(np.pi * a * a, rel=1e-14)
    assert actual["peak_pressure_pa"] == pytest.approx(1e8 * a / np.pi)


def test_hertz_pressure_and_displacement_include_exterior_tail():
    x = np.array([-0.002, -0.001, 0.0, 0.001, 0.002])
    pressure, displacement = hertz_fields(x, 0.00005)
    assert pressure[2, 2] == pytest.approx(100000 / np.pi)
    assert pressure[0, 2] == 0
    assert pressure[1, 2] == pytest.approx(0, abs=1e-3)
    assert displacement[2, 2] == pytest.approx(0.00005)
    assert displacement[1, 2] == pytest.approx(0.000025)
    expected_tail = (-2e-6 * np.pi / 6 + np.sqrt(3) * 1e-6) / (np.pi * 0.02)
    assert displacement[0, 2] == pytest.approx(expected_tail)
    assert displacement[0, 2] > 0


@pytest.mark.parametrize("n", GRIDS)
def test_physical_grid_is_full_unmasked_paraboloid_with_origin_node(n):
    x, height, pixel_area = physical_grid(n)
    assert x.shape == (n,)
    assert height.shape == (n, n)
    assert x[n // 2] == 0
    assert x[0] == -0.08
    assert x[-1] == pytest.approx(0.08 - 0.16 / n)
    assert pixel_area == pytest.approx((0.16 / n) ** 2)
    assert height[0, 0] == pytest.approx(-0.32)
    assert height[n // 2, n // 2] == 0
    assert np.isfinite(height).all()


def _manufactured_fields(n=128, delta=0.00005):
    """KKT-compatible values, deliberately not an elastic-solve oracle."""
    _, height, area = physical_grid(n)
    pressure = np.zeros((n, n))
    pressure[n // 2, n // 2] = 2.0
    gap = np.maximum(-height - delta, 0)
    displacement = height + delta + gap
    full = np.zeros((2 * n, 2 * n))
    full[:n, :n] = displacement
    force = pressure * area
    result = SimpleNamespace(x=full, jac=force, offset=delta)
    substrate = SimpleNamespace(
        area_per_pt=area,
        local_topography_subdomain_slices=(slice(0, n), slice(0, n)),
    )
    system = SimpleNamespace(compute_gap=lambda u, offset: u[:n, :n] - height - offset)
    fields = extract_fields(result, substrate, system, n, delta)
    reaction = np.zeros_like(full)
    reaction[:n, :n] = force
    return fields, reaction, result, substrate, system


def test_extraction_keeps_positive_pixel_force_distinct_from_pressure_and_owns_arrays():
    fields, _, result, _, _ = _manufactured_fields()
    middle = 64
    assert fields["p"][middle, middle] == 2.0
    assert fields["f"][middle, middle] == pytest.approx(2 * 0.00125**2)
    assert fields["gap"][middle, middle] == 0
    assert fields["u"].shape == (128, 128)
    assert fields["u_full"].shape == (256, 256)
    original = fields["f"].copy()
    result.jac[:] = 9
    np.testing.assert_array_equal(fields["f"], original)


def test_kkt_and_hertz_accuracy_are_separate_and_area_uses_strict_positive_force():
    fields, reaction, *_ = _manufactured_fields()
    report = assess_fields(fields, reaction, solver_success=True)
    assert report["kkt"]["passed"]
    assert not report["hertz"]["target_passed"]
    assert report["contact_pixels"] == 1
    assert report["area_m2"] == pytest.approx(0.00125**2)
    assert report["force_n"] == pytest.approx(2 * 0.00125**2)
    expected_force_error = abs(2 * 0.00125**2 - 1 / 15) / (1 / 15)
    assert report["hertz"]["force_relative_error"] == pytest.approx(expected_force_error)
    fields["f"][63, 64] = 1e-30
    fields["p"][63, 64] = 1e-30 / 0.00125**2
    reaction[63, 64] = 1e-30
    assert assess_fields(fields, reaction, solver_success=True)["contact_pixels"] == 2


@pytest.mark.parametrize(
    "fault",
    ["penetration", "tension", "complementarity", "padding", "elastic", "boundary", "solver"],
)
def test_independent_kkt_failures_remain_visible(fault):
    fields, reaction, *_ = _manufactured_fields()
    if fault == "penetration":
        fields["gap"][0, 0] = -2 * PENTOL_M
    elif fault == "tension":
        fields["f"][0, 0] = -2 * FORCETOL_N
        fields["p"][0, 0] = -2 * FORCETOL_N / (0.16 / 128) ** 2
    elif fault == "complementarity":
        fields["gap"][64, 64] = 1e-3
    elif fault == "padding":
        reaction[128, 128] = 20 * FORCETOL_N
    elif fault == "elastic":
        reaction[64, 64] += 20 * FORCETOL_N
    elif fault == "boundary":
        fields["f"][0, 0] = 1e-20
        fields["p"][0, 0] = 1e-20 / (0.16 / 128) ** 2
        fields["gap"][0, 0] = 0
        reaction[0, 0] = 1e-20
    report = assess_fields(fields, reaction, solver_success=fault != "solver")
    assert not report["kkt"]["passed"]


def test_preview_preserves_full_center_line_and_native_local_crop():
    fields, *_ = _manufactured_fields(512, 0.0002)
    fields["p"][:] = np.arange(512 * 512).reshape(512, 512)
    view = preview_fields(fields)
    assert len(view["full_domain"]["x_m"]) == 128
    assert len(view["center_line"]["x_m"]) == 512
    assert view["center_line"]["pressure_pa"] == fields["p"][:, 256].tolist()
    crop = view["contact_zoom"]
    indexes = np.flatnonzero(abs(fields["x"]) <= 0.004)
    assert crop["x_m"] == fields["x"][indexes].tolist()
    np.testing.assert_array_equal(crop["pressure_pa"], fields["p"][np.ix_(indexes, indexes)])
    assert crop["interpolated"] is False


def test_frozen_tolerances_have_explicit_si_units_and_no_parameter_overrides():
    config = frozen_config(128, DELTAS_M[0])
    assert config["solver"] == {
        "pentol_m": 1e-10,
        "forcetol_n": 1e-10,
        "thermotol": 1e-8,
        "maxiter": 10000,
    }
    assert (PENTOL_M, FORCETOL_N, THERMOTOL, MAXITER) == (1e-10, 1e-10, 1e-8, 10000)
    config["solver"]["maxiter"] = 1
    assert frozen_config(128, DELTAS_M[0])["solver"]["maxiter"] == 10000


@pytest.mark.parametrize(
    "grid,delta",
    [
        (True, 0.00005),
        (127, 0.00005),
        (1024, 0.00005),
        (128, True),
        (128, float("nan")),
        (128, 0),
        (128, 0.0003),
    ],
)
def test_unregistered_configuration_is_rejected(grid, delta):
    with pytest.raises(ValueError):
        frozen_config(grid, delta)


@pytest.mark.parametrize("fault", ["shape", "offset", "area", "slice", "nan", "gap", "masked"])
def test_bad_external_export_is_rejected_without_mutating_input(fault):
    _, _, result, substrate, system = _manufactured_fields()
    if fault == "shape":
        result.jac = result.jac[:-1]
    elif fault == "offset":
        result.offset += 1e-6
    elif fault == "area":
        substrate.area_per_pt *= 2
    elif fault == "slice":
        substrate.local_topography_subdomain_slices = (slice(1, 129), slice(0, 128))
    elif fault == "nan":
        result.x[0, 0] = np.nan
    elif fault == "gap":
        system.compute_gap = lambda *_: np.ones((128, 128))
    else:
        result.jac = np.ma.array(result.jac, mask=False)
        result.jac.mask[0, 0] = True
    before = result.x.copy()
    result.x.flags.writeable = False
    with pytest.raises(ValueError):
        extract_fields(result, substrate, system, 128, 0.00005)
    np.testing.assert_array_equal(result.x, before)
