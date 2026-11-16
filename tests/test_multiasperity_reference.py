"""Geometry/integration/qualification and HOST adapter fakes, never an FFT oracle."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.contact_selection import multiasperity_reference as reference


@pytest.mark.parametrize("grid", [256, 512])
def test_fixed_geometry_peaks_profile_baseline_and_disjoint_support(grid):
    data = reference.geometry(grid)
    scale = grid // 256
    first = (grid // 2 - 32 * scale, grid // 2 - 4 * scale)
    second = (grid // 2 + 32 * scale, grid // 2 + 4 * scale)
    quarter_radius = (first[0] - 16 * scale, first[1])
    assert data["x_m"][first[0]] == pytest.approx(-0.003)
    assert data["y_m"][first[1]] == pytest.approx(-0.000375)
    assert data["heights"][first] == pytest.approx(0, abs=1e-18)
    assert data["heights"][second] == pytest.approx(-40e-6, abs=1e-18)
    assert data["heights"][quarter_radius] == pytest.approx(200e-6 * 27 / 64 - 200e-6)
    assert data["heights"][0, 0] == -200e-6
    assert not np.any(data["support1"] & data["support2"])
    assert np.max(data["heights"]) == pytest.approx(0, abs=1e-18)
    assert data["x_m"][0] == -0.012
    assert data["x_m"][-1] == pytest.approx(0.012 - 0.024 / grid)


def test_isolated_modes_remove_only_other_height_term_not_the_baseline():
    joint = reference.geometry(512)
    first = reference.geometry(512, "isolated_1")
    second = reference.geometry(512, "isolated_2")
    assert np.max(first["heights"]) == pytest.approx(0, abs=1e-18)
    assert np.max(second["heights"]) == pytest.approx(-40e-6)
    assert np.count_nonzero(first["peak2_term"]) == 0
    assert np.count_nonzero(second["peak1_term"]) == 0
    np.testing.assert_allclose(
        joint["heights"], first["heights"] + second["heights"] + 200e-6, rtol=0, atol=1e-18
    )


def test_budget_has_exactly_twelve_frozen_cold_start_cases():
    specs = reference.case_specs()
    assert len(specs) == len(set(specs)) == 12
    assert specs[-2:] == ((512, 100e-6, "isolated_1"), (512, 100e-6, "isolated_2"))
    assert {d for _, d, mode in specs if mode == "coupled"} == {20e-6, 40e-6, 60e-6, 80e-6, 100e-6}
    config = reference.configuration(256, 20e-6)
    assert config["physical_sizes_m"] == [0.024, 0.024]
    assert config["contact_modulus_pa"] == 1e6
    assert config["solver"] == {
        "pentol_m": 1e-10,
        "forcetol_n": 1e-10,
        "thermotol": 1e-8,
        "maxiter": 10000,
        "external_force": None,
        "hardness": None,
        "initial_displacements": None,
        "initial_forces": None,
    }
    assert reference.case_id(512, 20e-6) == "c-n512-d020um"
    assert reference.case_id(512, 100e-6, "isolated_2") == "i2-n512-d100um"


@pytest.mark.parametrize(
    "grid,delta,mode",
    [
        (True, 20e-6, "coupled"),
        (128, 20e-6, "coupled"),
        (256, 20, "coupled"),
        (512, float("nan"), "coupled"),
        (256, 100e-6, "isolated_1"),
        (512, 80e-6, "isolated_2"),
        (512, 100e-6, "unknown"),
    ],
)
def test_unregistered_units_grid_or_mode_are_rejected(grid, delta, mode):
    with pytest.raises(ValueError):
        reference.configuration(grid, delta, mode)


def _load_oracle():
    force = np.array([[1.0, 2.0], [3.0, 4.0]])
    x, y = np.array([-0.001, 0.002]), np.array([-0.002, 0.004])
    first = np.array([[True, False], [False, False]])
    second = np.array([[False, False], [False, True]])
    return force, x, y, 18e-6, first, second


def test_independent_pixel_force_wrench_cop_and_per_peak_loads():
    args = _load_oracle()
    args[0].flags.writeable = False
    result = reference.integrate_loads(*args)
    assert result["force_n"] == 10
    assert result["area_m2"] == pytest.approx(72e-6)
    assert result["peak_pressure_pa"] == pytest.approx(4 / 18e-6)
    assert result["wrench_world_origin"] == pytest.approx([0, 0, 10, 0.016, -0.011, 0])
    assert result["cop_xy_m"] == pytest.approx([0.0011, 0.0016])
    assert [p["force_n"] for p in result["per_peak"]] == [1, 4]
    assert [p["area_m2"] for p in result["per_peak"]] == pytest.approx([18e-6, 18e-6])
    assert result["outside_support"]["force_n"] == 5
    assert result["outside_support"]["contact_pixels"] == 2
    np.testing.assert_array_equal(args[0], [[1, 2], [3, 4]])
    empty = list(args)
    empty[0] = np.zeros((2, 2))
    assert reference.integrate_loads(*empty)["cop_xy_m"] is None


@pytest.mark.parametrize("fault", ["area_units", "force_shape", "nonfinite", "overlap"])
def test_load_integration_rejects_wrong_units_arrays_or_identity(fault):
    args = list(_load_oracle())
    if fault == "area_units":
        args[3] = 18.0
    elif fault == "force_shape":
        args[0] = np.zeros((2, 3))
    elif fault == "nonfinite":
        args[1][0] = np.nan
    else:
        args[5] = args[4].copy()
    with pytest.raises(ValueError):
        reference.integrate_loads(*args)


def _manufactured_fields():
    """KKT bookkeeping fixture only, not a half-space solution."""
    n, delta = 256, 20e-6
    fields = reference.geometry(n)
    gap = np.maximum(-(fields["heights"] + delta), 0)
    full = np.zeros((2 * n, 2 * n))
    full[:n, :n] = fields["heights"] + delta + gap
    force = np.zeros((n, n))
    force[96, 124] = 2e-4
    reaction = np.zeros_like(full)
    reaction[:n, :n] = force
    fields.update(
        grid=n,
        delta_m=delta,
        mode="coupled",
        u_full=full,
        u=full[:n, :n].copy(),
        f=force,
        p=force / (0.024 / n) ** 2,
        gap=gap,
        public_gap=gap.copy(),
        contact_mask=force > 0,
    )
    return fields, reaction


def test_complementarity_uses_actual_total_force_times_indentation():
    fields, reaction = _manufactured_fields()
    clean = reference.assess_fields(fields, reaction, solver_success=True)
    assert clean["kkt_pass"]
    assert clean["per_peak"][1]["force_n"] == 0
    fields["gap"][96, 124] = 0.5e-9
    result = reference.assess_fields(fields, reaction, solver_success=True)
    assert result["kkt"]["complementarity_normalized"] == pytest.approx(0.5e-9 / 20e-6)
    assert result["kkt"]["checks"]["active_gap"]
    assert not result["kkt"]["checks"]["complementarity"]
    assert not result["kkt_pass"]


@pytest.mark.parametrize(
    "fault",
    ["negative_pressure", "penetration", "padding", "elastic", "boundary", "zero_force", "solver"],
)
def test_kkt_failure_does_not_repair_original_fields(fault):
    fields, reaction = _manufactured_fields()
    if fault == "negative_pressure":
        fields["f"][1, 1] = -2e-10
    elif fault == "penetration":
        fields["gap"][1, 1] = -2e-10
    elif fault == "padding":
        reaction[300, 300] = 2e-9
    elif fault == "elastic":
        reaction[96, 124] += 2e-9
    elif fault == "boundary":
        fields["f"][0, 0] = 1e-18
        fields["gap"][0, 0] = 0
        reaction[0, 0] = 1e-18
    elif fault == "zero_force":
        fields["f"][:] = 0
        reaction[:] = 0
    fields["p"] = fields["f"] / (0.024 / 256) ** 2
    before = fields["f"].copy()
    fields["f"].flags.writeable = False
    result = reference.assess_fields(fields, reaction, solver_success=fault != "solver")
    assert not result["kkt_pass"]
    np.testing.assert_array_equal(fields["f"], before)
    if fault == "zero_force":
        assert result["cop_xy_m"] is None
        assert result["kkt"]["complementarity_normalized"] is None


@pytest.mark.parametrize("fault", ["domain_units", "domain_shape", "force_shape", "offset", "gap"])
def test_external_export_rejects_wrong_domain_and_state_binding(fault):
    fields, _ = _manufactured_fields()
    substrate = SimpleNamespace(
        nb_grid_pts=(256, 256),
        nb_domain_grid_pts=(512, 512),
        physical_sizes=(0.024, 0.024),
        area_per_pt=(0.024 / 256) ** 2,
        local_topography_subdomain_slices=(slice(0, 256), slice(0, 256)),
    )
    raw = SimpleNamespace(x=fields["u_full"], jac=fields["f"], offset=20e-6)
    system = SimpleNamespace(
        compute_gap=lambda value, delta: value[:256, :256] - (fields["heights"] + delta)
    )
    if fault == "domain_units":
        substrate.physical_sizes = (24, 24)
    elif fault == "domain_shape":
        substrate.physical_sizes = 0.024
    elif fault == "force_shape":
        raw.jac = raw.jac[:-1]
    elif fault == "offset":
        raw.offset = 40e-6
    else:
        system.compute_gap = lambda *_: np.ones((256, 256))
    with pytest.raises(ValueError):
        reference.extract_fields(raw, substrate, system, 256, 20e-6)


def _case(grid, delta=100e-6, mode="coupled", force=10.0, area=1e-6, peak=1e5, cop=(0.0, 0.0)):
    return {
        "id": reference.case_id(grid, delta, mode),
        "grid": grid,
        "delta_m": delta,
        "mode": mode,
        "config": reference.configuration(grid, delta, mode),
        "status": "KKT_PASS",
        "return_code": 0,
        "usable_discrete_solution": True,
        "metrics": {
            "force_n": force,
            "area_m2": area,
            "peak_pressure_pa": peak,
            "cop_xy_m": list(cop),
            "kkt_pass": True,
            "wrench_world_origin": [0, 0, force, cop[1] * force, -cop[0] * force, 0],
        },
        "counters": {"solve_calls": 1},
        "files": {},
        "timing": {},
    }


@pytest.mark.parametrize(
    "output", ["force_relative", "area_relative", "peak_relative", "cop_distance_m"]
)
def test_grid_comparison_uses_512_denominator_and_independent_output_gates(output):
    fine, coarse = _case(512), _case(256)
    replacements = {
        "force_relative": ("force_n", 10.201),
        "area_relative": ("area_m2", 1.081e-6),
        "peak_relative": ("peak_pressure_pa", 105100.0),
        "cop_distance_m": ("cop_xy_m", [51e-6, 0.0]),
    }
    key, value = replacements[output]
    coarse["metrics"][key] = value
    result = reference.compare_grids(coarse, fine)
    assert result["status"] == "GRID_LIMIT_FAIL"
    assert result["checks"][output] is False
    assert all(v for k, v in result["checks"].items() if k != output)
    assert not result["all_pass"]
    if output == "force_relative":
        assert result["errors"][output] == pytest.approx(0.201 / 10)


def test_grid_comparison_rejects_other_domain_and_withholds_failed_kkt():
    coarse, fine = _case(256), _case(512)
    fine["config"]["physical_sizes_m"] = [0.16, 0.16]
    with pytest.raises(ValueError, match="bindings"):
        reference.compare_grids(coarse, fine)
    fine = _case(512)
    fine["usable_discrete_solution"] = False
    result = reference.compare_grids(coarse, fine)
    assert result["status"] == "UNAVAILABLE"
    assert not result["all_pass"]


def test_coupling_adds_wrenches_and_force_weights_cop_without_pass_gate():
    joint = _case(512, force=4.0, cop=(0.001, 0.002))
    first = _case(512, mode="isolated_1", force=1.0, cop=(-0.002, 0.001))
    second = _case(512, mode="isolated_2", force=3.0, cop=(0.003, 0.003))
    result = reference.compare_coupling(joint, first, second)
    assert result["isolated_sum_cop_xy_m"] == pytest.approx([0.00175, 0.0025])
    assert result["force_relative_error"] == 0
    assert result["cop_distance_m"] == pytest.approx(0.00025 * math.sqrt(13))
    assert result["moment_absolute_error_nm"] == pytest.approx([0.002, 0.003, 0])
    assert result["moment_relative_components"][:2] == pytest.approx([0.25, 0.75])
    assert result["moment_relative_components"][2] is None
    assert result["is_accuracy_gate"] is False
    zero_moment = _case(512, force=4.0)
    assert reference.compare_coupling(zero_moment, first, second)["moment_relative_norm"] is None


def test_late_failure_preserves_raw_arrays_and_metadata_without_second_solve(tmp_path, monkeypatch):
    fields, _ = _manufactured_fields()
    calls = []

    class Result(dict):
        __getattr__ = dict.__getitem__

    class Surface:
        def __init__(self, heights, sizes, **kwargs):
            np.testing.assert_array_equal(heights, fields["heights"])
            assert sizes == (0.024, 0.024)
            assert kwargs == {"periodic": False, "unit": "m"}

    class Substrate:
        def __init__(self, grid, **kwargs):
            assert grid == (256, 256)
            assert kwargs == {
                "young": 1e6,
                "physical_sizes": (0.024, 0.024),
                "fft": "serial",
                "check_boundaries": True,
            }
            self.nb_grid_pts, self.nb_domain_grid_pts = grid, (512, 512)
            self.physical_sizes, self.area_per_pt = (0.024, 0.024), (0.024 / 256) ** 2
            self.local_topography_subdomain_slices = (slice(0, 256), slice(0, 256))

        def evaluate_force(self, value):
            raise AssertionError("no extra elastic call after failed public solve")

    class System:
        def __init__(self, substrate, surface):
            self.substrate, self.surface = substrate, surface

        def minimize_proxy(self, *, solver, **kwargs):
            solver(self.substrate, self.surface, **kwargs)
            raise RuntimeError("synthetic post-solve boundary failure")

        def compute_gap(self, value, delta):
            return value[:256, :256] - (fields["heights"] + delta)

    def solver(*_, **kwargs):
        calls.append(kwargs)
        assert kwargs == {
            "offset": 20e-6,
            "external_force": None,
            "hardness": None,
            "initial_displacements": None,
            "initial_forces": None,
            "pentol": 1e-10,
            "forcetol": 1e-10,
            "thermotol": 1e-8,
            "maxiter": 10000,
        }
        return Result(
            maxcv={"max_pres": np.array(0.0)},
            x=fields["u_full"],
            jac=fields["f"],
            offset=20e-6,
            success=True,
            nit=2,
            nfev=5,
        )

    monkeypatch.setattr(reference, "_process_memory", lambda: {"test_fake": True})
    case = reference._run_case(
        tmp_path, 256, 20e-6, "coupled", (Surface, Substrate, System, solver), {"test_fake": True}
    )
    assert len(calls) == 1
    assert case["status"] == "ERROR"
    assert case["metrics"] is None and not case["usable_discrete_solution"]
    assert case["counters"]["extra_elastic_force_evaluations"] == 0
    saved_json = json.loads((tmp_path / case["files"]["case"]).read_text(encoding="utf-8"))
    assert saved_json["optimizer"]["maxcv"]["max_pres"]["shape"] == []
    with np.load(tmp_path / case["files"]["arrays"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["result_x"], fields["u_full"])
        np.testing.assert_array_equal(saved["result_jac"], fields["f"])
        assert saved["p"].shape == (256, 256)


@pytest.mark.parametrize("grid_failure", [False, True])
def test_complete_schedule_reports_grid_limit_failure_as_data_not_crash(
    tmp_path, monkeypatch, grid_failure
):
    # All numbers here are orchestration fixtures, not physical FFT outputs.
    calls = []

    def fake_case(directory, grid, delta, mode, api, binding, dependency_failure):
        calls.append((grid, delta, mode))
        return _case(grid, delta, mode, force=10.3 if grid == 256 and grid_failure else 10.0)

    monkeypatch.setattr(reference, "_load_external", lambda: (None, {"test_fake": True}))
    monkeypatch.setattr(reference, "_run_case", fake_case)
    directory = tmp_path / "new-output"
    summary = reference.run(directory)
    assert calls == list(reference.case_specs())
    assert summary["actual_solve_calls"] == 12
    assert len(summary["grid_comparisons"]) == 5
    assert summary["return_code"] == int(grid_failure)
    assert summary["status"] == ("FAIL_RECORDED" if grid_failure else "PASS")
    assert summary["coupling_comparison"]["force_relative_error"] == 1.0
    assert summary["coupling_comparison"]["is_accuracy_gate"] is False
    assert (directory / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        reference.run(directory)
    assert len(calls) == 12
    with pytest.raises(ValueError, match="absolute"):
        reference.run("relative-output-not-created")
